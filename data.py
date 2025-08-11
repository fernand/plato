from datasets import load_dataset
from transformers import GPTNeoXTokenizerFast

dataset = load_dataset('roneneldan/TinyStories')
tokenizer = GPTNeoXTokenizerFast.from_pretrained('EleutherAI/gpt-neox-20b')

def tokenize_with_eos(batch):
    return tokenizer(
        [text + tokenizer.eos_token for text in batch['text']],
        truncation=False,
        padding=False
    )

tokenized_dataset = dataset.map(
    tokenize_with_eos,
    batched=True,
    remove_columns=dataset['train'].column_names,
    num_proc=32,
)

tokenized_dataset.save_to_disk('tokenized_tinystories_neox')