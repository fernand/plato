import math
from pathlib import Path

import numpy as np
from datasets import Dataset, Features, Sequence, Value
from transformers import PreTrainedTokenizerFast

tokenizer = PreTrainedTokenizerFast(
    tokenizer_file='plato_tokenizer.json',
    bos_token='<s>',
    eos_token='</s>',
    pad_token='<pad>',
    unk_token='<unk>',
    mask_token='<mask>'
)

def tokenize_text(text):
    # Keep counts exact; we pad only the final chunk per file.
    return tokenizer(text, add_special_tokens=False, return_attention_mask=False)['input_ids']

def chunk_per_file(token_ids, chunk_size, pad_id):
    L = len(token_ids)
    num_chunks = max(1, math.ceil(L / chunk_size))
    for i in range(num_chunks):
        start = i * chunk_size
        sl = token_ids[start:start + chunk_size]
        if len(sl) < chunk_size:
            pad_len = chunk_size - len(sl)
            sl = sl + [pad_id] * pad_len
            attn = [1] * (chunk_size - pad_len) + [0] * pad_len
        else:
            attn = [1] * len(sl)
        labels = np.array(sl, dtype=np.int64)
        labels = np.where(np.array(attn, dtype=np.int64) == 1, labels, -1)
        yield {
            'input_ids': np.array(sl, dtype=np.int64),
            'attention_mask': np.array(attn, dtype=np.int64),
            'labels': labels
        }

def _process_batch(batch, chunk_size):
    pad_id = tokenizer.pad_token_id
    out_file, out_ids, out_attn, out_labels = [], [], [], []
    for fp in batch['file']:
        text = Path(fp).read_text(encoding='utf-8')
        token_ids = tokenize_text(text)
        for ex in chunk_per_file(token_ids, chunk_size, pad_id):
            out_file.append(fp)
            out_ids.append(ex['input_ids'])
            out_attn.append(ex['attention_mask'])
            out_labels.append(ex['labels'])
    return {
        'file': out_file,
        'input_ids': out_ids,
        'attention_mask': out_attn,
        'labels': out_labels,
    }

def build_dataset(filepaths, chunk_size, num_workers=32):
    features = Features({
        'file': Value('string'),
        'input_ids': Sequence(Value('int64')),
        'attention_mask': Sequence(Value('int64')),
        'labels': Sequence(Value('int64')),
    })
    files_ds = Dataset.from_dict({'file': list(map(str, filepaths))}, features=Features({'file': Value('string')}))
    ds = files_ds.map(
        lambda batch: _process_batch(batch, chunk_size),
        batched=True,
        num_proc=num_workers,  # 32-way parallel
        desc=f'Tokenize+chunk (N={chunk_size})'
    )
    # Materialize with the desired features
    return ds.cast(features)

if __name__ == '__main__':
    files = sorted(Path('./plato_works').glob('*.txt'))
    ds = build_dataset(files, chunk_size=1024 + 1, num_workers=32)
    ds.save_to_disk('./tokenized_plato_works')