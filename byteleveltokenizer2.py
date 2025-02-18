import os 
import json
import re
import string
from collections import Counter

# ---------------------------
# Parameters & Reserved Tokens
# ---------------------------
JSON_FOLDER = r"C:\Users\abias\Magpie-Reasoning-V2-250K-CoT-Deepseek-R1-Llama-70B\data"
VOCAB_SIZE = 50000
CHAR_LIMIT = 5

# Reserved tokens: special tokens always added at the end.
RESERVED_SPECIAL = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
# Digits should be prioritized first.
PRIORITY_DIGITS = list("0123456789")
# The alphabet tokens (lowercase a-z) must appear if missing.
ALPHABET = list(string.ascii_lowercase)

# ---------------------------
# Helper functions
# ---------------------------
def process_token(token):
    """
    Given a token (from splitting on whitespace), further break it down:
      - If the token is purely numeric, break into individual digits.
      - Otherwise, split into word parts and punctuation (each punctuation symbol separate).
      - For any segment longer than CHAR_LIMIT, break it into chunks of max length CHAR_LIMIT.
    """
    result_tokens = []
    # if token is purely numeric, split into digits:
    if token.isdigit():
        result_tokens.extend(list(token))
        return result_tokens

    # Otherwise, use regex to split into word segments and punctuation.
    # This will match sequences of alphanumerics (the \w+ part) or individual non-alphanumeric non-space characters.
    sub_tokens = re.findall(r'\w+|[^\w\s]', token, flags=re.UNICODE)

    for sub in sub_tokens:
        # If the subtoken is numeric, break it into digits.
        if sub.isdigit():
            result_tokens.extend(list(sub))
        else:
            # For segments longer than the limit, break them into chunks.
            if len(sub) > CHAR_LIMIT:
                # break the subtoken into consecutive chunks
                for i in range(0, len(sub), CHAR_LIMIT):
                    result_tokens.append(sub[i:i+CHAR_LIMIT])
            else:
                result_tokens.append(sub)
    return result_tokens

def load_texts_from_json(folder):
    """
    Iterate over all JSON files in the given folder.
    Each JSON file is assumed to have a "text" field.
    Returns a list of text strings.
    """
    texts = []
    for filename in os.listdir(folder):
        if filename.lower().endswith('.json'):
            path = os.path.join(folder, filename)
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # If the JSON has a key "text", use it.
                    if "text" in data:
                        texts.append(data["text"])
                    else:
                        # Alternatively, if the JSON is just a string or has other structure, adjust here.
                        texts.append(str(data))
            except Exception as e:
                print(f"Error reading {path}: {e}")
    return texts

# ---------------------------
# Build the frequency counter
# ---------------------------
freq_counter = Counter()

texts = load_texts_from_json(JSON_FOLDER)
for text in texts:
    # split on whitespace first (this gives us “word” candidates)
    tokens = text.split()
    for token in tokens:
        sub_tokens = process_token(token)
        # For a byte-level approach we treat the token string as its UTF-8 bytes,
        # but here we count based on the human-readable form.
        for sub in sub_tokens:
            freq_counter[sub] += 1

# ---------------------------
# Enforce priority for digits
# ---------------------------
# Remove any counts for our priority digits (they will be forced at the beginning)
for d in PRIORITY_DIGITS:
    if d in freq_counter:
        del freq_counter[d]

# ---------------------------
# Sort tokens by frequency (descending)
# ---------------------------
sorted_tokens = [token for token, _ in freq_counter.most_common()]

# ---------------------------
# Assemble the final vocabulary list
# ---------------------------
# We must reserve slots for:
#   - Priority digits (always first)
#   - Special tokens (always at the end)
# Also, ensure that every letter in ALPHABET appears before special tokens.
# First, start with the digits:
final_vocab = PRIORITY_DIGITS.copy()

# Next, add the frequency-sorted tokens (excluding any that are already added)
for token in sorted_tokens:
    if token not in final_vocab:
        final_vocab.append(token)

# Now, check that every letter of the alphabet is present.
# If any letter is missing, add it.
for letter in ALPHABET:
    if letter not in final_vocab:
        final_vocab.append(letter)

# Now, we must reserve space at the end for the special tokens.
# That is, our non-special part should have VOCAB_SIZE - len(RESERVED_SPECIAL) tokens.
max_main_tokens = VOCAB_SIZE - len(RESERVED_SPECIAL)
if len(final_vocab) > max_main_tokens:
    final_vocab = final_vocab[:max_main_tokens]
else:
    # If there is still room, you might choose to leave it as is or fill with additional tokens.
    pass

# Finally, append the special tokens at the end.
final_vocab.extend(RESERVED_SPECIAL)

# A sanity check: make sure we have exactly VOCAB_SIZE tokens.
if len(final_vocab) != VOCAB_SIZE:
    print(f"Warning: final vocabulary size is {len(final_vocab)} (expected {VOCAB_SIZE})")

# Create a token-to-index mapping.
token_to_id = {token: idx for idx, token in enumerate(final_vocab)}

# ---------------------------
# Save the vocabulary/tokenizer to JSON
# ---------------------------
output = {
    "vocab_size": len(final_vocab),
    "token_to_id": token_to_id,
    "id_to_token": final_vocab  # Optional: list by index.
}

with open("tokenizer.json", "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print("Tokenizer saved to tokenizer.json")
