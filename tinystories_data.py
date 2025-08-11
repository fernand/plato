import datasets
import transformers

if __name__ == '__main__':
    dataset = datasets.load_dataset('roneneldan/TinyStories')
    # tokenizer = transformers.GPTNeoXTokenizerFast.from_pretrained('EleutherAI/gpt-neox-20b')
    tokenizer = transformers.PreTrainedTokenizerFast(
            tokenizer_file="plato_tokenizer.json",
            bos_token="<s>",
            eos_token="</s>",
            pad_token="<pad>",
            unk_token="<unk>",
            mask_token="<mask>"
        )

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

    tokenized_dataset.save_to_disk('tokenized_tinystories_plato')
