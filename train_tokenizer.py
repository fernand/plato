import tokenizers
from pathlib import Path

def train(files: list[str]):
    tokenizer = tokenizers.ByteLevelBPETokenizer()
    tokenizer.train(files=files, vocab_size=2**14, min_frequency=2, special_tokens=[
        "<s>",
        "<pad>",
        "</s>",
        "<unk>",
        "<mask>",
    ])
    tokenizer.save('tokenizer.json')

def tokenize(files: list[str], tokenizer_path: str = 'tokenizer.json'):
    tokenizer = tokenizers.Tokenizer.from_file(tokenizer_path)
    texts = []
    for file_path in files:
        with open(file_path, 'r', encoding='utf-8') as f:
            texts.append(f.read())
    encodings = tokenizer.encode_batch(texts)
    total_tokens = sum(len(encoding) for encoding in encodings)
    print(f'Total tokens: {total_tokens}')

if __name__ == '__main__':
    files = [str(p) for p in Path('text').glob('*.txt')]
    train(files)
    tokenize(files)
