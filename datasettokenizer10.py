import os
import json
import re
import time
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, decoders, trainers
from collections import Counter

def clean_text(text):
    """ Clean text by removing non-English characters. """
    return re.sub(r"[^a-zA-Z0-9\s!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~]", "", text)

def split_repeated_substrings(text):
    """ Split repeated substrings if they repeat more than twice. """
    pattern = re.compile(r"(.{3,}?)\1{1,}")
    return pattern.sub(lambda m: " ".join(m.group(0)), text)

def iterate_texts_from_dataset(dataset_dir, lowercase=True, max_samples=100000):
    """
    Yields cleaned text from JSON files.
    `max_samples` limits **number of text sequences**, NOT number of files.
    """
    count = 0
    total_files = len([f for f in os.listdir(dataset_dir) if f.lower().endswith('.json')])
    print(f"Found {total_files} JSON files in {dataset_dir}. Processing a maximum of {max_samples} text samples.")

    for filename in os.listdir(dataset_dir):
        if filename.lower().endswith('.json'):
            path = os.path.join(dataset_dir, filename)
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    # Handle different JSON structures
                    if isinstance(data, dict):
                        texts = [data.get("text", str(data))]
                    elif isinstance(data, list):
                        texts = [str(item) for item in data]
                    else:
                        texts = [str(data)]

                    for text in texts:
                        if lowercase:
                            text = text.lower()
                        text = clean_text(text)
                        text = split_repeated_substrings(text)

                        yield text
                        count += 1

                        # Print a progress update every 5000 samples
                        if count % 5000 == 0:
                            print(f"Processed {count}/{max_samples} text samples...")

                        if count >= max_samples:
                            print(f"Reached max_samples limit ({max_samples}). Stopping dataset iteration.")
                            return
            except Exception as e:
                print(f"Error reading {path}: {e}")

    print(f"Completed processing {count} text samples.")

def build_tokenizer_from_dataset(dataset_dir, output_path, vocab_size=200000, use_bytelevel=True, lowercase=True, max_samples=100000, frequency_cutoff=50):
    """ Build and train a tokenizer from a dataset of JSON files with live status updates. """
    texts_iterator = iterate_texts_from_dataset(dataset_dir, lowercase=lowercase, max_samples=max_samples)

    print("\nInitializing tokenizer...")
    if use_bytelevel:
        tokenizer = Tokenizer(models.BPE(unk_token="<UNK>"))
        tokenizer.normalizer = normalizers.Sequence([normalizers.NFKC()])
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Whitespace(),
            pre_tokenizers.Punctuation(),
            pre_tokenizers.Split(r"(\d)", behavior='isolated'),
            pre_tokenizers.Split(r"(.{3,}?)\1{1,}", behavior='isolated')
        ])
        tokenizer.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size, 
            min_frequency=frequency_cutoff,  # ✅ This ensures low-frequency tokens are never included
            special_tokens=["<PAD>", "<UNK>", "<BOS>", "<EOS>"],
            continuing_subword_prefix="##",
            initial_alphabet=list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")
        )
    else:
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
            min_frequency=frequency_cutoff,  # ✅ This ensures low-frequency tokens are never included
            special_tokens=["<PAD>", "<UNK>", "<BOS>", "<EOS>"],
            initial_alphabet=list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")
        )

    print("\nStarting tokenizer training...\n")
    
    # Timer to measure training duration
    start_time = time.time()
    
    tokenizer.train_from_iterator(texts_iterator, trainer=trainer)
    
    print("Tokenizer training completed in {:.2f} seconds.".format(time.time() - start_time))

    tokenizer.save(output_path)
    print(f"\nTokenizer saved to {output_path}")

    return tokenizer

# Example usage:
dataset_dir = r"C:\Users\abias\Magpie-Reasoning-V2-250K-CoT-Deepseek-R1-Llama-70B\data"
output_path = "tokenizer_from_dataset_bytelevel.json"
tokenizer = build_tokenizer_from_dataset(dataset_dir, output_path, vocab_size=100000, use_bytelevel=True, max_samples=30000, frequency_cutoff=25)
