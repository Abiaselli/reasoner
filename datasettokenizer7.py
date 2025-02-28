import os
import json
import re
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, decoders, trainers
from collections import Counter

def clean_text(text):
    """
    Clean text by removing non-English characters.
    Keeps only letters, digits, standard punctuation, and whitespace.
    """
    # Regular expression to keep only English characters, digits, and standard punctuation
    text = re.sub(r"[^a-zA-Z0-9\s!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~]", "", text)
    return text

def split_repeated_substrings(text):
    """
    Split repeated substrings if they repeat more than twice.
    Example: 'nnvyije0nnvyije0' -> 'nnvyije', '0', 'nnvyije', '0'
    """
    # Regex to find repeated substrings of 3 or more characters
    pattern = re.compile(r"(.{3,}?)\1{1,}")
    return pattern.sub(lambda m: " ".join(m.group(0)), text)

def iterate_texts_from_dataset(dataset_dir, lowercase=True, max_samples=100000):
    """
    Generator that yields cleaned text from each JSON file in the dataset directory.
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
                    # Clean the text to keep only English characters
                    text = clean_text(text)
                    # Split repeated substrings
                    text = split_repeated_substrings(text)
                    yield text
                    count += 1
                    if count >= max_samples:
                        break
            except Exception as e:
                print(f"Error reading {path}: {e}")

def build_tokenizer_from_dataset(dataset_dir, output_path, vocab_size=200000, use_bytelevel=True, lowercase=True, max_samples=100000, frequency_cutoff=50):
    """
    Build and train a tokenizer from a dataset of JSON files using a streaming approach.
    
    Args:
        dataset_dir (str): Path to the directory containing JSON files.
        output_path (str): Where to save the trained tokenizer (JSON file).
        vocab_size (int): Desired vocabulary size.
        use_bytelevel (bool): If True, builds a ByteLevel BPE tokenizer; if False, builds a WordLevel tokenizer.
        lowercase (bool): If True, lowercases texts during training.
        max_samples (int): Maximum number of files/texts to process.
        frequency_cutoff (int): Minimum frequency for tokens to be kept in the vocabulary.
    
    Returns:
        tokenizer (tokenizers.Tokenizer): The trained tokenizer.
    """
    texts_iterator = iterate_texts_from_dataset(dataset_dir, lowercase=lowercase, max_samples=max_samples)
    
    if use_bytelevel:
        # Create a ByteLevel BPE tokenizer.
        tokenizer = Tokenizer(models.BPE(unk_token="<UNK>"))
        tokenizer.normalizer = normalizers.Sequence([normalizers.NFKC()])
        # Custom pre-tokenizer for English characters, numbers, and punctuation
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Whitespace(),
            pre_tokenizers.Punctuation(),
            # Split each digit as its own token
            pre_tokenizers.Split(r"(\d)", behavior='isolated'),
            # Split repeated substrings
            pre_tokenizers.Split(r"(.{3,}?)\1{1,}", behavior='isolated')
        ])
        tokenizer.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size, 
            special_tokens=["<PAD>", "<UNK>", "<BOS>", "<EOS>"],
            continuing_subword_prefix="##",
            end_of_word_suffix="",
            # Lock characters as atomic tokens to prevent merges
            initial_alphabet=list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")
        )
    else:
        # Create a WordLevel tokenizer.
        tokenizer = Tokenizer(models.WordLevel(unk_token="<UNK>"))
        tokenizer.normalizer = normalizers.Sequence([normalizers.NFKC()])
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Whitespace(),
            pre_tokenizers.Punctuation(),
            pre_tokenizers.Split(r"(\d)", behavior='isolated'),
            pre_tokenizers.Split(r"(.{3,}?)\1{1,}", behavior='isolated')
        ])
        tokenizer.decoder = decoders.WordLevel()
        trainer = trainers.WordLevelTrainer(
            vocab_size=vocab_size, 
            special_tokens=["<PAD>", "<UNK>", "<BOS>", "<EOS>"],
            initial_alphabet=list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")
        )
    
    # Train the tokenizer using the iterator (this avoids loading everything into memory)
    tokenizer.train_from_iterator(texts_iterator, trainer=trainer)
    
    # Get the vocabulary and count frequencies
    vocab = tokenizer.get_vocab()
    token_counts = Counter(vocab)
    
    # Sort tokens by frequency in descending order
    sorted_tokens = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Filter tokens by frequency cutoff
    filtered_tokens = {token: count for token, count in sorted_tokens if count >= frequency_cutoff}
    
    # Save the filtered vocabulary
    filtered_vocab_path = output_path.replace(".json", "_filtered.json")
    with open(filtered_vocab_path, "w") as f:
        json.dump(filtered_tokens, f, indent=4)
    print(f"Filtered vocabulary saved to {filtered_vocab_path}")
    
    # Save the tokenizer
    tokenizer.save(output_path)
    print(f"Tokenizer saved to {output_path}")
    
    return tokenizer

# Example usage:
dataset_dir = r"/home/austin/Magpie-Reasoning-V2-250K-CoT-Deepseek-R1-Llama-70B/data"
output_path = "tokenizer_from_dataset_wordlevel.json"
tokenizer = build_tokenizer_from_dataset(dataset_dir, output_path, vocab_size=100000, use_bytelevel=True, max_samples=400000, frequency_cutoff=50)
