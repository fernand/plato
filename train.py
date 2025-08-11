import math
import os
from dataclasses import dataclass

import comet_ml
import datasets
import torch
import torch.nn.functional as F
import transformers
from torch import nn, Tensor

@torch.compile
def zeropower_via_newtonschulz5(G: Tensor, steps: int) -> Tensor:
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G
    if G.size(-2) > G.size(-1):
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, weight_decay=0.01, momentum=0.95):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            momentum = group['momentum']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state['momentum_buffer'] = torch.zeros_like(grad)
                momentum_buffer = state['momentum_buffer']
                p.mul_(1 - group['weight_decay'])
                momentum_buffer.lerp_(grad, 1 - momentum)
                grad = grad.lerp_(momentum_buffer, momentum)
                v = zeropower_via_newtonschulz5(grad.bfloat16(), 5)
                p.add_(other=v, alpha=-group['lr'])

class Rotary(nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x):
        seq_len = x.shape[1]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq).to(x.device)
            self.cos_cached = freqs.cos()
            self.sin_cached = freqs.sin()
        return self.cos_cached[None, :, None, :], self.sin_cached[None, :, None, :] # type: ignore

def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4 # multihead attention
    d = x.shape[3]//2
    x1 = x[..., :d]
    x2 = x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)

def rmsnorm(x0, eps:float=1e-6):
    x = x0.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return x.type_as(x0)

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        self.c_attn = nn.Linear(self.n_embd, 3 * self.n_embd, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.rotary = Rotary(self.head_dim)

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, self.head_dim)
        q = q.view(B, T, self.n_head, self.head_dim)
        v = v.view(B, T, self.n_head, self.head_dim)
        cos, sin = self.rotary(q)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.wup = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.wdown = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.wup(x)
        x = F.gelu(x)
        x = self.wdown(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)
        self.attn_scale = (1 / math.sqrt(2 * config.n_layer))

    def forward(self, x):
        x = x + self.attn_scale * self.attn(rmsnorm(x))
        x = x + self.mlp(rmsnorm(x))
        return x

@dataclass
class GPTConfig:
    vocab_size: int
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.transformer = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.wte.weight = self.lm_head.weight

    def forward(self, idx, targets=None, return_logits=True):
        x = self.wte(idx)
        for block in self.transformer:
            x = block(x)
        x = rmsnorm(x)
        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        # there are performance reasons why not returning logits is prudent, if not needed
        if not return_logits:
            logits = None
        return logits, loss

    def configure_optimizers(self, adam_wd, adam_lr, adam_betas):
        return [
            Muon(self.transformer.parameters(), lr=10*adam_lr, weight_decay=0, momentum=0.95),
            torch.optim.AdamW(self.lm_head.parameters(), lr=adam_lr, weight_decay=adam_wd, betas=adam_betas)
        ]

def log_gradient_stats(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad = param.grad.data
            print(f'\n{name}:')
            print(f'  Shape: {list(grad.shape)}')
            print(f'  Mean: {grad.mean():.6f}, Std: {grad.std():.6f}')
            print(f'  Min: {grad.min():.6f}, Max: {grad.max():.6f}')
            print(f'  Norm: {grad.norm(2):.6f}')

if __name__ == '__main__':
    import time
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True, help='path to tokenized HF dataset')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size, in units of #batch dimensions')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='max LR value')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay')
    args = parser.parse_args()

    assert torch.cuda.is_available(), 'CUDA not available'

    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_file="plato_tokenizer.json",
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
        unk_token="<unk>",
        mask_token="<mask>"
    )
    tokenizer.padding_side = 'right'
    # Round to 16.
    vocab_size = ((len(tokenizer) + 15) // 16) * 16

    ds = datasets.load_from_disk(args.dataset_path)
    train_dataset = ds['train']
    num_iters = len(train_dataset) // args.batch_size
    val_dataset = ds.get('validation', ds.get('test', None))

    target_batch_size = 512
    batch_ratio = target_batch_size // args.batch_size
    step_scale = batch_ratio
    val_loss_every = batch_ratio * 128
    warmup_iters = int(0.028 * num_iters)
    warmdown_iters = int(0.24 * num_iters)
    val_max_steps = 20
    save_every = batch_ratio * 1000

    sequence_length = 512

    def collate_fn(examples):
        # Always pad to sequence_length + 1 (for shifted input/target pairs)
        padded_length = sequence_length + 1
        # Stack all input_ids with padding to the fixed length
        input_ids = torch.stack([
            torch.tensor(ex['input_ids'][:padded_length] +
                        [tokenizer.pad_token_id] * max(0, padded_length - len(ex['input_ids'])))
            for ex in examples
        ])
        x = input_ids[:, :-1]
        y = input_ids[:, 1:]
        # Replace padding tokens with -1 in targets for ignore_index
        y = torch.where(y == tokenizer.pad_token_id, -1, y)
        return x, y

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=1,
        pin_memory=True
    ) if val_dataset is not None else None
    train_iter = iter(train_loader)

    def get_batch(loader_iter, loader):
        '''Get next batch, reset iterator if needed'''
        try:
            x, y = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            x, y = next(loader_iter)
        return x.cuda(), y.cuda(), loader_iter

    model = GPT(GPTConfig(vocab_size=vocab_size))
    model = model.train().cuda()
    torch.set_float32_matmul_precision('high')
    model = torch.compile(model, dynamic=False)
    ctx = torch.autocast(device_type='cuda', dtype=torch.bfloat16)
    print('Model # params', sum(p.numel() for p in model.parameters()))

    # See https://arxiv.org/abs/2507.07101 for beta2 scaling.
    optimizers = model.configure_optimizers(adam_wd=args.weight_decay,
                                               adam_lr=args.learning_rate,
                                               adam_betas=(0.9, (0.95)**(1.0/step_scale)))

    def get_lr(it):
        assert it <= num_iters
        if it < warmup_iters:
            return args.learning_rate * (it+1) / warmup_iters
        elif it < num_iters - warmdown_iters:
            return args.learning_rate
        else:
            decay_ratio = (num_iters - it) / warmdown_iters
            return args.learning_rate * decay_ratio

    experiment = comet_ml.Experiment(
        api_key=os.environ['COMET_API_KEY'],
        project_name='ts-pt',
        workspace='fernand',
        auto_metric_logging=False,
        log_env_gpu=False,
        log_env_cpu=False,
        log_env_host=False,
        log_env_details=False,
        # disabled=True,
    )

    lossf = 0.0
    x, y, train_iter = get_batch(train_iter, train_loader)
    for step in range(num_iters + 1):
        if step % step_scale == 0:
            t0 = time.perf_counter()
        last_step = (step == num_iters)

        if (val_loss_every > 0 \
            and (step % val_loss_every == 0 or last_step)) \
            and (val_loader is not None):
            model.eval()
            val_iter = iter(val_loader)  # Reset validation iterator
            with torch.no_grad():
                val_loss = 0.0
                for i in range(min(val_max_steps, len(val_loader))):
                    x_val, y_val, val_iter = get_batch(val_iter, val_loader)
                    _, loss = model(x_val, y_val, return_logits=False)
                    val_loss += loss.item()
                val_loss /= min(val_max_steps, len(val_loader))
            print(f'val loss {val_loss}')
            experiment.log_metric('val_loss', val_loss, step)

        if last_step:
            break

        model.train()
        with ctx:
            _, loss = model(x, y, return_logits=False)
        x, y, train_iter = get_batch(train_iter, train_loader)
        loss.backward()

        lr = get_lr(step)
        lr_scale = [32, 1]
        for opt_idx, optimizer in enumerate(optimizers):
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_scale[opt_idx] * lr
            optimizer.step()
            # if (step > 0 or step_scale == 1) and step % step_scale == 0:
            #     log_gradient_stats(model)
            optimizer.zero_grad(set_to_none=True)
        experiment.log_metric('lr', lr, step)

        torch.cuda.synchronize()
        lossf += loss.item() / step_scale
        if (step > 0 or step_scale == 1) and step % step_scale == 0:
            t1 = time.perf_counter()
            print(f'step {1+(step//step_scale):4d}/{num_iters//step_scale} | train loss {lossf:.6f} | lr {lr:.2e} | {(t1-t0)*1000:.0f} ms')
            experiment.log_metric('train_loss', lossf, step)
            lossf = 0.0

        if (step + 1) % save_every == 0:
            os.makedirs('logs/%s' % experiment.id, exist_ok=True)
            torch.save(model.state_dict(), 'logs/%s/model_step%06d.pt' % (experiment.id, step))

    os.makedirs('logs/%s' % experiment.id, exist_ok=True)
    torch.save(model.state_dict(), 'logs/%s/final.pt' % experiment.id)
