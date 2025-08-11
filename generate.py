import transformers
import torch
import torch.nn.functional as F

from train import GPT, GPTConfig

def generate_text(
    model,
    tokenizer,
    prompt,
    max_new_tokens=512,
    temperature=1.0,
    top_k=None,
    top_p=None,
    device='cuda'
):
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    eos_token_id = tokenizer.eos_token_id
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits, _ = model(input_ids, targets=None, return_logits=True)
            # Get logits for the last position (next token prediction)
            next_token_logits = logits[0, -1, :]
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            if top_k is not None:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = -float('inf')
            # Apply top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep also the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[indices_to_remove] = -float('inf')
            # Sample from the distribution
            probs = F.softmax(next_token_logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1)
            # For greedy decoding (temperature=0), use argmax instead
            if temperature == 0:
                next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token_id.unsqueeze(0)], dim=1)
            if next_token_id.item() == eos_token_id:
                break

    generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=False)
    return generated_text

if __name__ == '__main__':
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_file="plato_tokenizer.json",
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
        unk_token="<unk>",
        mask_token="<mask>"
    )
    vocab_size = len(tokenizer)

    torch.set_float32_matmul_precision('high')

    model_dict = torch.load(
        'logs/5c2acc6cb90841488d483ea340d26fbb/final.pt',
        weights_only=True,
        map_location=torch.device('cuda'))['model']
    model = GPT(GPTConfig(vocab_size=vocab_size)).eval().cuda()
    state_dict = {}
    # The compiled torch state dict prepends parameter names with '_orig_mod.'
    for k, v in model_dict.items():
        state_dict[k.removeprefix('_orig_mod.')] = v
    model.load_state_dict(state_dict)

    # prompt = 'Diane stepped into the forest.'
    prompt = 'What is in the forest?'

    sampled_output = generate_text(
        model,
        tokenizer,
        prompt,
        temperature=0.8,
        top_k=50,
        top_p=0.95,
        device='cuda'
    )
    print(sampled_output)
