import os
import json
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, decoders, trainers

def iterate_texts_from_dataset(dataset_dir, lowercase=True, max_samples=100000):
    """
    Generator that yields text from each JSON file in the dataset directory.
    Processes at most max_samples files to limit memory usage.
    Handles both dicts (using the "text" key) and lists.
    """
    count = 0
    for filename in os.listdir(dataset_dir):
        if filename.lower().endswith('.json'):
            path = os.path.join(dataset_dir, filename)
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        text = data.get("text", str(data))
                    elif isinstance(data, list):
                        text = " ".join(str(item) for item in data)
                    else:
                        text = str(data)
                    if lowercase:
                        text = text.lower()
                    yield text
                    count += 1
                    if count >= max_samples:
                        break
            except Exception as e:
                print(f"Error reading {path}: {e}")

def build_tokenizer_from_dataset(dataset_dir, output_path, vocab_size=200000, use_bytelevel=True, lowercase=True, max_samples=100000):
    """
    Build and train a tokenizer from a dataset of JSON files using a streaming approach.
    
    Args:
        dataset_dir (str): Path to the directory containing JSON files.
        output_path (str): Where to save the trained tokenizer (JSON file).
        vocab_size (int): Desired vocabulary size.
        use_bytelevel (bool): If True, builds a ByteLevel BPE tokenizer; if False, builds a WordLevel tokenizer.
        lowercase (bool): If True, lowercases texts during training.
        max_samples (int): Maximum number of files/texts to process.
    
    Returns:
        tokenizer (tokenizers.Tokenizer): The trained tokenizer.
    """
    texts_iterator = iterate_texts_from_dataset(dataset_dir, lowercase=lowercase, max_samples=max_samples)
    
    if use_bytelevel:
        # Create a ByteLevel BPE tokenizer.
        tokenizer = Tokenizer(models.BPE(unk_token="<UNK>"))
        tokenizer.normalizer = normalizers.Sequence([normalizers.NFKC()])
        # Custom pre-tokenizer for numbers and punctuation
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Whitespace(),
            pre_tokenizers.Punctuation(),
            pre_tokenizers.Split(r"(\d)", behavior='isolated')  # Split each digit as its own token
        ])
        tokenizer.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size, 
            special_tokens=["<PAD>", "<UNK>", "<BOS>", "<EOS>"],
            # Avoid merging sequences of digits
            continuing_subword_prefix="##",
            end_of_word_suffix="",
            initial_alphabet=list("0123456789")  # Lock digits as atomic tokens
        )
    else:
        # Create a WordLevel tokenizer.
        tokenizer = Tokenizer(models.WordLevel(unk_token="<UNK>"))
        tokenizer.normalizer = normalizers.Sequence([normalizers.NFKC()])
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Whitespace(),
            pre_tokenizers.Punctuation(),
            pre_tokenizers.Split(r"(\d)", behavior='isolated')  # Split each digit as its own token
        ])
        tokenizer.decoder = decoders.WordLevel()
        trainer = trainers.WordLevelTrainer(
            vocab_size=vocab_size, 
            special_tokens=["<PAD>", "<UNK>", "<BOS>", "<EOS>"],
            initial_alphabet=list("0123456789")  # Lock digits as atomic tokens
        )
    
    # Train the tokenizer using the iterator (this avoids loading everything into memory)
    tokenizer.train_from_iterator(texts_iterator, trainer=trainer)
    
    tokenizer.save(output_path)
    print(f"Tokenizer saved to {output_path}")
    
    return tokenizer

# Example usage:
dataset_dir = r"/home/austin/Magpie-Reasoning-V2-250K-CoT-Deepseek-R1-Llama-70B/data"
output_path = "tokenizer_from_dataset_bytelevel.json"
tokenizer = build_tokenizer_from_dataset(dataset_dir, output_path, vocab_size=300000, use_bytelevel=True, max_samples=400000)
