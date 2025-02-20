import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import json
import threading
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
from transformers import PreTrainedTokenizerFast, AddedToken
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, decoders
import copy
import random
import math
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
import re
import pandas as pd

# Debug for CUDA
print(f"CUDA Available: {torch.cuda.is_available()}")
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

os.environ["TORCH_USE_CUDA_DSA"] = "1"
tokenizer = None

training_goal = 0.5
attempt_limit = 10
seq_len = 512




def save_checkpoint(model, optimizer, epoch, phase, path):
    checkpoint = {
        'epoch': epoch,
        'phase': phase,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(checkpoint, path)

def load_checkpoint(path, model, optimizer=None):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['phase']


def init_tokenizer(tokenizer_path):
    global tokenizer
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
    logging.info(f"Tokenizer pad_token set to: {tokenizer.pad_token}, ID: {tokenizer.pad_token_id}")


@staticmethod
def tokenize_chunk(chunk):
    # Tokenizer is now the global variable initialized in each process
    encoded = tokenizer(chunk, return_attention_mask=False, truncation=True, max_length=seq_len)
    return encoded['input_ids']

# Collate function
def collate_fn(batch):
    if len(batch[0]) == 3:
        # Dataset returns: input_ids, labels, seq_lengths
        input_ids, labels, seq_lengths = zip(*batch)
        input_ids = torch.stack(input_ids)
        labels = torch.stack(labels)
        seq_lengths = torch.tensor(seq_lengths, dtype=torch.long)
        return input_ids, labels, seq_lengths
    elif len(batch[0]) == 4:
        # Dataset returns: input_ids, labels, label_tots seq_lengths
        input_ids, labels, labels_tot, seq_lengths = zip(*batch)
        input_ids = torch.stack(input_ids)
        labels = torch.stack(labels)
        seq_lengths = torch.tensor(seq_lengths, dtype=torch.long)
        labels_tot = torch.stack(labels_tot)
    elif len(batch[0]) == 5:
        # Dataset returns: input_ids, labels, label_tots seq_lengths
        input_ids, labels, labels_tot, seq_lengths, attention_masks = zip(*batch)
        input_ids = torch.stack(input_ids)
        labels = torch.stack(labels)
        seq_lengths = torch.tensor(seq_lengths, dtype=torch.long)
        labels_tot = torch.stack(labels_tot)
        attention_masks = torch.stack(attention_masks)
        return input_ids, labels, labels_tot, seq_lengths, attention_masks
    else:
        raise ValueError("Unexpected number of elements in dataset sample.")

def quaternion_mse_loss(output, target):
    """
    Quaternion Mean Squared Error (QMSE) Loss.
    Computes the squared difference between quaternion components.
    Modify Output projection to use Quaternion linear for this.
    
    Args:
        output: (batch, seq_len, features, 4) - Quaternion predictions
        target: (batch, seq_len, features, 4) - Quaternion targets

    Returns:
        QMSE Loss (scalar)
    """
    return torch.mean((output - target) ** 2)

def quaternion_cross_entropy(output, target):
    """
    Quaternion Cross-Entropy Loss.
    Measures divergence between quaternion softmax outputs and targets.
    Modify Output projection to use Quaternion linear for this.

    Args:
        output: (batch, seq_len, features, 4) - Quaternion predictions
        target: (batch, seq_len, features, 4) - Quaternion targets

    Returns:
        Scalar loss
    """
    # Compute quaternion dot product as a similarity measure
    similarity = torch.sum(output * target, dim=-1)
    loss = -torch.mean(torch.log(torch.exp(similarity) / torch.sum(torch.exp(similarity), dim=-1, keepdim=True)))
    return loss

def quaternion_angular_loss(output, target):
    """
    Quaternion Angular Loss.
    Encourages quaternion outputs to maintain rotational consistency with targets.
    Modify Output projection to use Quaternion linear for this.

    Args:
        output: (batch, seq_len, features, 4) - Quaternion predictions
        target: (batch, seq_len, features, 4) - Quaternion targets

    Returns:
        Scalar loss
    """
    # Normalize quaternions to avoid scale issues
    output = output / torch.norm(output, dim=-1, keepdim=True)
    target = target / torch.norm(target, dim=-1, keepdim=True)
    
    # Compute cosine similarity loss (similar to quaternion dot product)
    angular_distance = torch.acos(torch.clamp(torch.sum(output * target, dim=-1), -1, 1))
    return torch.mean(angular_distance ** 2)

class ChunkedDataset(torch.utils.data.Dataset):
    def __init__(self, tokenized_data_path, tokenizer, max_length=seq_len):
        self.tokenized_data_path = tokenized_data_path
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Get a list of chunk files
        self.chunk_files = [os.path.join(self.tokenized_data_path, f) 
                            for f in os.listdir(self.tokenized_data_path) 
                            if f.startswith('chunk_') and f.endswith('.jsonl')]
        self.chunk_files.sort()  # Ensure the chunks are in order

        # Build an index mapping from global indices to (chunk_idx, sample_idx)
        self.index_mapping = []
        for chunk_idx, chunk_file in enumerate(self.chunk_files):
            with open(chunk_file, 'r', encoding='utf-8') as f:
                num_lines = sum(1 for _ in f)
            self.index_mapping.extend([(chunk_idx, i) for i in range(num_lines)])

        # Initialize current chunk data
        self.current_chunk_idx = -1  # Indicates no chunk is currently loaded
        self.current_chunk_data = []  # Will hold the data from the current chunk

    def __len__(self):
        return len(self.index_mapping)

    def __getitem__(self, idx):

        if idx < 0 or idx >= len(self.index_mapping):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.index_mapping)}")

        chunk_idx, sample_idx = self.index_mapping[idx]

        # Load the appropriate chunk if not already loaded
        if self.current_chunk_idx != chunk_idx:
            self.load_chunk(chunk_idx)

        record = self.current_chunk_data[sample_idx]
        input_ids = record['input_ids']
        labels = record['labels']

        # Calculate original sequence length before padding
        original_seq_length = min(len(input_ids), self.max_length)
        logging.debug(f"original sequence length = {original_seq_length}")
        # Pad sequences to max_length
        input_ids = input_ids[:self.max_length] + [self.tokenizer.pad_token_id] * max(0, self.max_length - len(input_ids))
        labels = labels[:self.max_length] + [self.tokenizer.pad_token_id] * max(0, self.max_length - len(labels))

        assert isinstance(input_ids, list), "input_ids should be a list"
        assert isinstance(labels, list), "labels should be a list"
        assert all(isinstance(id, int) for id in input_ids), "All input_ids should be integers"
        assert all(isinstance(id, int) for id in labels), "All labels should be integers"
        assert len(input_ids) == self.max_length, "input_ids should be padded to max_length"
        assert len(labels) == self.max_length, "labels should be padded to max_length"
        
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        seq_lengths = torch.tensor(original_seq_length, dtype=torch.long)

        # Check for empty sequences
        if len(input_ids) == 0:
            logging.error(f"Empty input_ids at index {idx}.")
            raise ValueError(f"Empty input_ids at index {idx}.")
        if len(labels) == 0:
            logging.error(f"Empty labels at index {idx}.")
            raise ValueError(f"Empty labels at index {idx}.")
    
        return input_ids, attention_mask, labels, seq_lengths

    def load_chunk(self, idx):
        chunk_file = self.chunk_files[idx]
        with open(chunk_file, 'r', encoding='utf-8') as f:
            self.current_chunk_data = [json.loads(line.strip()) for line in f]
        self.current_chunk_idx = idx


# Generate src mask function
def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def create_combined_mask(batch_input_ids, pad_token_id):
    """
    Create a combined attention mask that incorporates both the causal (subsequent) mask
    and the padding mask. This function ensures that each row has at least one valid token.
    """
    batch_size, seq_length = batch_input_ids.size()
    device = batch_input_ids.device
    
    # Generate causal (subsequent) mask: shape (seq_len, seq_len)
    causal_mask = generate_square_subsequent_mask(seq_len).to(device)
    logging.debug(f"Shape of causal_mask before expand: {causal_mask.shape}")

    # Expand to batch dimension: (batch_size, seq_len, seq_len)
    causal_mask = causal_mask.unsqueeze(0).expand(batch_size, -1, -1)
    logging.debug(f"Shape of causal_mask after expansion: {causal_mask.shape}")
    # Create padding mask: valid tokens are True, padded tokens are False.
    # Shape: (batch_size, seq_len)
    padding_mask = (batch_input_ids != pad_token_id)
    # Expand padding mask to match the shape (batch_size, seq_len, seq_len)
    # Here we broadcast along one dimension so that we mask out positions in each row.
    padding_mask_expanded = padding_mask.unsqueeze(1).expand(batch_size, seq_len, seq_len)
    logging.debug(f"Shape of padding_mask after expansion: {padding_mask_expanded.shape}")

    # Combine masks: where padding_mask is False, set to -inf.
    # This keeps the causal structure while ensuring that padded positions are fully masked.
    combined_mask = causal_mask.masked_fill(~padding_mask_expanded, float('-inf'))
    logging.debug(f"Shape of combined_mask after fill: {combined_mask.shape}")

    # Check each row: if an entire row is -inf, force the first token (or a designated position) to be valid.
    for i in range(batch_size):
        for j in range(seq_len):
            if torch.all(combined_mask[i, j] == float('-inf')):
                combined_mask[i, j, 0] = 0.0  # Force at least one valid position
    
    return combined_mask


class LogarithmicNumberSystem:
    """
    CUDA-Optimized Logarithmic Number System (LNS) for efficient GPU computation.
    """
    def __init__(self, epsilon=1e-6, use_cuda=True):
        self.epsilon = epsilon
        self.device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")

    def to_log(self, x):
        logging.debug(f"shape of x to log {x.shape}")

        """ Convert tensor to log-space using CUDA acceleration. """
        return torch.log(torch.clamp(x.to(self.device), min=self.epsilon))

    def from_log(self, log_x):
        logging.debug(f"shape of log_x to to convert back {log_x.shape}")

        """ Convert back from log-space. """
        return torch.exp(log_x)

    def log_add(self, log_x, log_y):
        """ Logarithmic addition using CUDA-accelerated Log-Sum-Exp trick. """
        logging.debug(f"shape of log_x for add {log_x.shape}")
        logging.debug(f"shape of log_y for add {log_y.shape}")

        max_val, _ = torch.max(torch.stack([log_x, log_y], dim=-1), dim=-1)
        return max_val + torch.log(torch.exp(log_x - max_val) + torch.exp(log_y - max_val))

    def log_sub(self, log_x, log_y):
        """ Logarithmic subtraction with CUDA support. """
        logging.debug(f"shape of log_x for add {log_x.shape}")
        logging.debug(f"shape of log_y for add {log_y.shape}")
        max_val, _ = torch.max(torch.stack([log_x, log_y], dim=-1), dim=-1)
        logging.debug(f"shape of max_val for sub {log_x.shape}")
        sub_result = torch.exp(log_x - max_val) - torch.exp(log_y - max_val)
        logging.debug(f"shape of sub_result for sub {log_x.shape}")
        
        return max_val + torch.log(torch.clamp(sub_result, min=self.epsilon))

    def log_mul(self, log_x, log_y):
        """ Logarithmic multiplication using CUDA (log-space addition). """
        logging.debug(f"shape of log_x for mul {log_x.shape}")
        logging.debug(f"shape of log_y for mul {log_y.shape}")
        return log_x + log_y

    def log_div(self, log_x, log_y):
        """ Logarithmic division using CUDA (log-space subtraction). """
        logging.debug(f"shape of log_x for div {log_x.shape}")
        logging.debug(f"shape of log_y for div {log_y.shape}")
        return log_x - log_y
    
    def log_add_einsum(self, equation, log_x, log_y):
        """
        Implements log-space einsum operation by applying log-sum-exp trick.
        """
        # Ensure tensors have same shape
        assert log_x.shape == log_y.shape, f"Shape mismatch: {log_x.shape} vs {log_y.shape}"

        max_val = torch.max(log_x, log_y)
        logging.debug(f"shape of max_val for einsum {max_val.shape}")
        logging.debug(f"shape of log_x for einsum {log_x.shape}")
        logging.debug(f"shape of log_y for einsum {log_y.shape}")
        log_x_adj = log_x - max_val
        log_y_adj = log_y - max_val
        logging.debug(f"Einsum equation: {equation}")
        logging.debug(f"log_x_adj shape: {log_x_adj.shape}")
        logging.debug(f"log_y_adj shape: {log_y_adj.shape}")
        log_x_adj = log_sum_exp(log_x_adj, dim=-1)
        #log_x_adj = log_x_adj.expand(-1,-1,128, -1)
        log_y_adj = log_sum_exp(log_y_adj, dim=-1)
        #log_y_adj = log_y_adj.expand(-1,-1,128, -1)
        logging.debug(f"log_x_adj shape after log_sum_exp: {log_x_adj.shape}")
        logging.debug(f"log_y_adj shape after log_sum_exp: {log_y_adj.shape}")
        einsum_tensor = torch.einsum(equation, [log_x_adj, log_y_adj])
        logging.debug(f"einsum_tenspr shape: {einsum_tensor.shape}")
        einsum_tensor = einsum_tensor.unsqueeze(-1)
        # ✅ Ensure max_val reduces along the last dim before logsumexp
        max_val, _ = torch.max(einsum_tensor, dim=-1, keepdim=True)  
        logging.debug(f"Shape of max_val: {max_val.shape}")  # Should be [batch, seq_len, seq_len, 1]
        einsum_tensor_adj = einsum_tensor - max_val

        logging.debug(f"Shape of einsum t after max subtraction: {einsum_tensor_adj.shape}")
        einsum_tensor_adj = torch.logsumexp(einsum_tensor_adj, dim=-1)
        logging.debug(f"Shape einsum t before sum: {einsum_tensor_adj.shape}")
        # ✅ Apply logsumexp only across the correct dimension
        output = torch.einsum('bkd,bkdt->bkd', einsum_tensor_adj, max_val)
        logging.debug(f"Shape einsum output: {output.shape}")

        return  output


def log_sum_exp(tensor, dim=-1, keepdim=True):
    """
    Optimized Log-Sum-Exp function for stable summation in log-space.
    Prevents overflow and underflow issues by normalizing.
    """
    logging.debug(f"shape of tensor for log_sum_exp {tensor.shape}")
    
    max_val, _ = torch.max(tensor, dim=dim, keepdim=True)  # Find max value
    return max_val + torch.log(torch.sum(torch.exp(tensor - max_val), dim=dim, keepdim=keepdim))


class Transformer_Model_LNS(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, seq_length, num_heads):
        super(Transformer_Model, self).__init__()
        self.embedding = Transformer_Token_Embedding(vocab_size, embedding_dim)
        self.attention = Attention(embedding_dim)

        self.examination = nn.ModuleList([Examination(vocab_size, embedding_dim, hidden_dim, seq_length, num_layers) for _ in range(num_layers)])
        self.reasoning = nn.ModuleList([Reasoning(vocab_size, embedding_dim, hidden_dim, seq_length, num_layers) for _ in range(num_layers)])
        self.doubt = nn.ModuleList([Validate(vocab_size, embedding_dim, hidden_dim, seq_length, num_layers) for _ in range(num_layers)])
        self.imagine = nn.ModuleList([Imagination(vocab_size, embedding_dim, hidden_dim, seq_length, num_layers) for _ in range(num_layers)])
        self.reward = Reinforcement(vocab_size, embedding_dim, hidden_dim, seq_length, num_layers)

        self.output_projection = nn.Linear(embedding_dim, vocab_size)
        self.pos_encoder = nn.Embedding(seq_length, embedding_dim)
        self.vocab_size=vocab_size
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads, dim_feedforward=hidden_dim, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.cross_node_attention = nn.MultiheadAttention(embedding_dim, num_heads, batch_first=True)

    def forward(self, input_ids, targets, reason_targets, mask=None):
        logging.debug(f"Input IDs Shape: {input_ids.shape}")  # Log input shape
        torch.autograd.set_detect_anomaly(True)

        thought = self.embedding(input_ids)
        logging.debug(f"Embedding Output Shape: {thought.shape}")  # Log embedding output

        batch_size, seq_length = thought.size(0), thought.size(1)
        positions = torch.arange(seq_length, device=input_ids.device).unsqueeze(0)
        pos_encodings = self.pos_encoder(positions)
        pos_encodings = pos_encodings.expand(batch_size, seq_length, -1)

        # Add positional encodings to embeddings
        src = thought + pos_encodings

        # Forward through transformer encoder (self-attention)
        encode_thought = self.transformer_encoder(src, mask)

        attention_weights = None

        logging.debug(f"Attention Output Shape: {src.shape}")  # Log attention output

        confidence = False
        attempt = 0
        max_attempts = attempt_limit

        while not confidence and attempt < max_attempts:
            thought = self.examination[attempt % len(self.examination)](encode_thought.to(dtype=torch.float32), reason_targets)
            logging.debug(f"Examination Output Shape: {thought.shape}")  # Log examination output
            thought, attention_weights = self.cross_node_attention(encode_thought, thought, thought, attn_mask=mask)

            thought, confidence = self.reasoning[attempt % len(self.reasoning)](thought, reason_targets)
            logging.debug(f"Reasoning Output Shape: {thought.shape}")  # Log reasoning output
            train_of_thought = thought.clone()
            thought, attention_weights = self.cross_node_attention(encode_thought, thought, thought, attn_mask=mask)

            thought, confidence = self.doubt[attempt % len(self.doubt)](thought, targets)
            logging.debug(f"Doubt Output Shape: {thought.shape}, Confidence: {confidence}")  # Log doubt
            thought, attention_weights = self.cross_node_attention(encode_thought, thought, thought, attn_mask=mask)

            attempt += 1

            if not confidence:
                thought = self.imagine[attempt % len(self.imagine)](thought, input_ids)
                logging.debug(f"Imagination Output Shape: {thought.shape}")  # Log imagination
        thought, reward = self.reward(thought, targets, attempt)
        logging.debug(f"Reward Value: {reward}")  # Log final reward

        return thought, reward, train_of_thought

    
### SIMPLE ATTENTION MECHANISM ###
class Attention(nn.Module):
    def __init__(self, embedding_dim):
        super(Attention, self).__init__()
        self.query_weight = nn.Linear(embedding_dim, embedding_dim)
        self.key_weight = nn.Linear(embedding_dim, embedding_dim)
        self.value_weight = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x, mask=None):
        """
        Computes self-attention with optional masking.
        Args:
            x (torch.Tensor): Shape [batch_size, seq_len, embedding_dim]
            mask (torch.Tensor, optional): Shape [batch_size, seq_len, seq_len]
        Returns:
            attended_output (torch.Tensor): Shape [batch_size, seq_len, embedding_dim]
        """
        Q = self.query_weight(x)
        K = self.key_weight(x)
        V = self.value_weight(x)

        # Scaled Dot-Product Attention
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(Q.size(-1))

        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))

        attention_weights = F.softmax(attention_scores, dim=-1)
        attended_output = torch.matmul(attention_weights, V)

        return attended_output

def causal_mask(seq_len):
    """
    Creates a mask to prevent attending to future tokens.
    Args:
        seq_len (int): Length of the sequence
    Returns:
        mask (torch.Tensor): Shape [seq_len, seq_len], lower triangular matrix
    """
    return torch.tril(torch.ones((seq_len, seq_len))).unsqueeze(0)  # Add batch dimension

def padding_mask(input_ids, pad_token_id=0):
    """
    Creates a mask for padded tokens in a batch.
    Args:
        input_ids (torch.Tensor): Shape [batch_size, seq_len]
        pad_token_id (int): Token ID representing padding (default 0)
    Returns:
        mask (torch.Tensor): Shape [batch_size, seq_len, seq_len]
    """
    mask = (input_ids != pad_token_id).unsqueeze(1).expand(-1, input_ids.size(1), -1)
    return mask


### TOKEN EMBEDDING ###
class Transformer_Token_Embedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(Transformer_Token_Embedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, x):
        return self.embedding(x)


### TRANSFORMER LAYER ###
class Transformer_Layer(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super(Transformer_Layer, self).__init__()
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embedding_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


### REASONING PIPELINE ###
class Transformer_Model(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, seq_length, num_heads):
        super(Transformer_Model, self).__init__()
        self.embedding = Transformer_Token_Embedding(vocab_size, embedding_dim)

        self.examination = Examination(vocab_size, embedding_dim, hidden_dim) 
        self.reasoning = Reasoning(vocab_size, embedding_dim, hidden_dim)
        self.doubt = Validate(vocab_size, embedding_dim, hidden_dim) 
        self.imagine = Imagination(vocab_size, embedding_dim, hidden_dim)
        self.reinforcement = Reinforcement(vocab_size, embedding_dim, hidden_dim, num_layers)

        self.output_projection = nn.Linear(embedding_dim, vocab_size)
        self.pos_encoder = nn.Embedding(seq_length, embedding_dim)
        self.vocab_size=vocab_size
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads, dim_feedforward=hidden_dim, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.cross_node_attention = nn.MultiheadAttention(embedding_dim, num_heads, batch_first=True)

    def forward(self, input_ids, mask):
        logging.debug(f"Input IDs Shape: {input_ids.shape}")  # Log input shape
        torch.autograd.set_detect_anomaly(True)

        thought = self.embedding(input_ids)
        logging.debug(f"Embedding Output Shape: {thought.shape}")  # Log embedding output

        batch_size, seq_length = thought.size(0), thought.size(1)
        positions = torch.arange(seq_length, device=input_ids.device).unsqueeze(0)
        pos_encodings = self.pos_encoder(positions)
        pos_encodings = pos_encodings.expand(batch_size, seq_length, -1)

        # Add positional encodings to embeddings
        src = thought + pos_encodings

        # Forward through transformer encoder (self-attention)
        encode_thought = self.transformer_encoder(src, mask)

        attention_weights = None

        logging.debug(f"Attention Output Shape: {src.shape}")  # Log attention output

        thought = self.examination(encode_thought.to(dtype=torch.float32))
        logging.debug(f"Examination Output Shape: {thought.shape}")  # Log examination output
        
        thought, attention_weights = self.cross_node_attention(encode_thought, thought, thought, attn_mask=mask)

        thought = self.reasoning(thought)
        logging.debug(f"Reasoning Output Shape: {thought.shape}")  # Log reasoning output
        train_of_thought = thought.clone()

        thought, attention_weights = self.cross_node_attention(encode_thought, thought, thought, attn_mask=mask)

        thought = self.doubt(thought)
        logging.debug(f"Doubt Output Shape: {thought.shape}")  # Log doubt

        thought, attention_weights = self.cross_node_attention(encode_thought, thought, thought, attn_mask=mask)

        thought = self.imagine(thought, input_ids)
        logging.debug(f"Imagination Output Shape: {thought.shape}")  # Log imagination
        
        thought = self.reinforcement(thought)
        logging.debug(f"Reinforcement Output Shape: {thought.shape}")  # Log imagination
        # Project back to vocab space so that output has shape [batch, seq_len, vocab_size]
        thought = self.output_projection(thought)
        logging.debug(f"Logits Output Shape: {thought.shape}")  # Log imagination
        
        return thought, train_of_thought

### EXAMINATION MODULE ###
class Examination(nn.Module):
    """
    Module to examine inputs to determine issue and path forward
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(Examination, self).__init__()
        self.layer = Transformer_Layer(embedding_dim, hidden_dim)
        self.output_projection = nn.Linear(embedding_dim, vocab_size)
        self.vocab_size = vocab_size
        self.embedding = Transformer_Token_Embedding(vocab_size, embedding_dim)  # Embedding layer    

    def forward(self, x):
        logging.debug(f"Exam input Shape: {x.shape}")  
        thought = x.clone()

        thought = self.layer(thought)
        logging.debug(f"Exam thought Shape 2: {thought.shape}")  
        thought = self.output_projection(thought).to(dtype=torch.float32)
        logging.debug(f"Exam thought Shape 3: {thought.shape}")  
        temperature = 0.7  # Lower temperature → more one-hot-like outputs
        thought = gumbel_softmax(thought.clone(), temperature=temperature, hard=True)

        logging.debug(f"Gumnel thought Shape exam output: {thought.shape}") 
        emb_weight = self.embedding.embedding.weight.clone()

        thought = torch.matmul(thought, emb_weight)         
        logging.debug(f"Exam output: {thought.shape}")  # Log output shape
        return thought.to(dtype=torch.float32)


### IMAGINATION (GENETIC ALGORITHM) ###
class Imagination(nn.Module):
    """
    Module for thought mutation
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(Imagination, self).__init__()
        self.embedding = Transformer_Token_Embedding(vocab_size, embedding_dim)  # Embedding layer    
        self.layer = Transformer_Layer(embedding_dim, hidden_dim)
        self.output_projection = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x, input_ids):
        last_thought = x.clone()
        logging.debug(f"Imagine input: {x.shape}")  # Log output shape

        thought = self.layer(last_thought)
        logging.debug(f"Imagine thought Shape 2: {thought.shape}")  # Log output shape
        thought = self.output_projection(thought).to(dtype=torch.float32)
        logging.debug(f"Imagine thought Shape 4: {thought.shape}")  # Log output shape
        temperature = 1  # Lower temperature → more one-hot-like outputs
        thought = gumbel_softmax(thought, temperature=temperature, hard=True)

        logging.debug(f"Imagine thought Shape 5: {thought.shape}")  # Log output shape
        emb_weight = self.embedding.embedding.weight.clone()

        thought = torch.matmul(thought, emb_weight)         
        logging.debug(f"Imagine output: {thought.shape}")  # Log output shape
        return thought.to(dtype=torch.float32)  # Crossover with a transformed copy


### REASONING MODULE ###
class Reasoning(nn.Module):
    """
    Module for train of thought reasoning
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(Reasoning, self).__init__()
        self.embedding = Transformer_Token_Embedding(vocab_size, embedding_dim)  # Embedding layer    
        self.layer = Transformer_Layer(embedding_dim, hidden_dim)
        self.output_projection = nn.Linear(embedding_dim, vocab_size)
        self.vocab_size = vocab_size
        
    def forward(self, x):
        last_thought = x.clone()
        logging.debug(f"Reasoning input shape: {x.shape}")  # Log output shape

        thought = self.layer(last_thought)
        logging.debug(f"Reasoning thought Shape 1: {thought.shape}") 
        thought = self.output_projection(thought).to(dtype=torch.float32)
        logging.debug(f"Reasoning thought Shape 2: {thought.shape}") 
        test_thought = thought.clone()
        logging.debug(f"Reasoning thought Shape 3: {test_thought.shape}") 

        thought = self.layer(last_thought)
        logging.debug(f"Reasoning thought Shape 4: {thought.shape}") 
        thought = self.output_projection(thought).to(dtype=torch.float32)
        logging.debug(f"Reasoning thought Shape 5: {thought.shape}") 
        temperature = 1  # Lower temperature → more one-hot-like outputs
        thought = gumbel_softmax(thought.clone(), temperature=temperature, hard=True)
        logging.debug(f"Reasoning thought Shape 7: {thought.shape}") 
        emb_weight = self.embedding.embedding.weight.clone()

        thought = torch.matmul(thought, emb_weight)         
        logging.debug(f"Reasoning output: {thought.shape}")  # Log output shape
        return thought
    
### VALIDATION MODULE ###
class Validate(nn.Module):
    """
    Transformer module to second guess itself
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(Validate, self).__init__()
        self.embedding = Transformer_Token_Embedding(vocab_size, embedding_dim)  # Embedding layer    
        self.layer = Transformer_Layer(embedding_dim, hidden_dim)
        self.output_projection = nn.Linear(embedding_dim, vocab_size)
        self.vocab_size = vocab_size
        
    def forward(self, x):
        last_thought = x.clone()
        thought = self.layer(last_thought)
        thought = self.output_projection(thought).to(dtype=torch.float32)

        temperature = 0.7  # Lower temperature → more one-hot-like outputs
        thought = gumbel_softmax(thought.clone(), temperature=temperature, hard=True)

        logging.debug(f"Gumnel thought Shape: {thought.shape}") 
        logging.debug(f"Validation input: {x.shape}")  # Log output shape
        emb_weight = self.embedding.embedding.weight.clone()

        thought = torch.matmul(thought, emb_weight)         
        logging.debug(f"Validate output: {thought.shape}")  # Log output shape
        return thought

     
### REINFORCEMENT MODULE ###
class Reinforcement(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers):
        super(Reinforcement, self).__init__()
        self.layers = nn.ModuleList([Transformer_Layer(embedding_dim, hidden_dim) for _ in range(num_layers)])

    def forward(self, x):
        last_thought = x.clone()
        for layer in self.layers:
            last_thought = layer(last_thought.clone())

            logging.debug(f"Reinforcement thought Shape: {last_thought.shape}") 

        return last_thought

def mutate(x):
    mutation_rate = 0.1
    mutation_mask = (torch.rand_like(x) < mutation_rate).to(dtype=torch.float32)
    return x + mutation_mask.to(dtype=torch.float32) * torch.randn_like(x) * 0.05


def crossover(thought1, thought2):
    mask = (torch.rand_like(thought1) > 0.5).to(dtype=torch.float32)
    return thought1 * mask + thought2 * (1 - mask)

def mutate_parameters(module):
    """ Applies mutation to module parameters in a small range to optimize performance. """
    mutation_rate = 0.05  # Adjust this as needed
    with torch.no_grad():
        for param in module.parameters():
            if torch.rand(1) < mutation_rate:
                param.add_(torch.randn_like(param) * 0.1)


def sample_gumbel(shape, eps=1e-20, device='cpu'):
    """Sample from Gumbel(0, 1)"""
    U = torch.rand(shape, device=device)
    return -torch.log(-torch.log(U + eps) + eps)

def gumbel_softmax_sample(logits, temperature):
    """ Draw a sample from the Gumbel-Softmax distribution """
    gumbel_noise = sample_gumbel(logits.shape, device=logits.device)
    y = logits + gumbel_noise
    return F.softmax(y / temperature, dim=-1)

def gumbel_softmax(logits, temperature=1.0, hard=False):
    """
    Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
        logits: [*, num_classes] unnormalized log-probs
        temperature: non-negative scalar
        hard: if True, take argmax, but still use softmax gradients
    Returns:
        [*, num_classes] sample from the Gumbel-Softmax distribution.
    """
    y = gumbel_softmax_sample(logits, temperature)
    if hard:
        # Straight-through trick: make hard one-hot output, but keep soft gradients
        y_hard = torch.zeros_like(y)
        y_hard.scatter_(-1, y.argmax(dim=-1, keepdim=True), 1.0)
        # Set gradients of y_hard equal to those of y
        y = (y_hard - y).detach() + y
    logging.debug(f"Gumbel shape: {y.shape}") 

    return y

class LogarithmicNumberSystem:
    """
    CUDA-Optimized Logarithmic Number System (LNS) for efficient GPU computation.
    """
    def __init__(self, epsilon=1e-6, use_cuda=True):
        self.epsilon = epsilon
        self.device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")

    def to_log(self, x):
        logging.debug(f"shape of x to log {x.shape}")

        """ Convert tensor to log-space using CUDA acceleration. """
        return torch.log(torch.clamp(x.to(self.device), min=self.epsilon))

    def from_log(self, log_x):
        logging.debug(f"shape of log_x to to convert back {log_x.shape}")

        """ Convert back from log-space. """
        return torch.exp(log_x)

    def log_add(self, log_x, log_y):
        """ Logarithmic addition using CUDA-accelerated Log-Sum-Exp trick. """
        logging.debug(f"shape of log_x for add {log_x.shape}")
        logging.debug(f"shape of log_y for add {log_y.shape}")

        max_val, _ = torch.max(torch.stack([log_x, log_y], dim=-1), dim=-1)
        return max_val + torch.log(torch.exp(log_x - max_val) + torch.exp(log_y - max_val))

    def log_sub(self, log_x, log_y):
        """ Logarithmic subtraction with CUDA support. """
        logging.debug(f"shape of log_x for add {log_x.shape}")
        logging.debug(f"shape of log_y for add {log_y.shape}")
        max_val, _ = torch.max(torch.stack([log_x, log_y], dim=-1), dim=-1)
        logging.debug(f"shape of max_val for sub {log_x.shape}")
        sub_result = torch.exp(log_x - max_val) - torch.exp(log_y - max_val)
        logging.debug(f"shape of sub_result for sub {log_x.shape}")
        
        return max_val + torch.log(torch.clamp(sub_result, min=self.epsilon))

    def log_mul(self, log_x, log_y):
        """ Logarithmic multiplication using CUDA (log-space addition). """
        logging.debug(f"shape of log_x for mul {log_x.shape}")
        logging.debug(f"shape of log_y for mul {log_y.shape}")
        return log_x + log_y

    def log_div(self, log_x, log_y):
        """ Logarithmic division using CUDA (log-space subtraction). """
        logging.debug(f"shape of log_x for div {log_x.shape}")
        logging.debug(f"shape of log_y for div {log_y.shape}")
        return log_x - log_y
    
    def log_add_einsum(self, equation, log_x, log_y):
        """
        Implements log-space einsum operation by applying log-sum-exp trick.
        """
        # Ensure tensors have same shape
        assert log_x.shape == log_y.shape, f"Shape mismatch: {log_x.shape} vs {log_y.shape}"

        max_val = torch.max(log_x, log_y)
        logging.debug(f"shape of max_val for einsum {max_val.shape}")
        logging.debug(f"shape of log_x for einsum {log_x.shape}")
        logging.debug(f"shape of log_y for einsum {log_y.shape}")
        log_x_adj = log_x - max_val
        log_y_adj = log_y - max_val
        logging.debug(f"Einsum equation: {equation}")
        logging.debug(f"log_x_adj shape: {log_x_adj.shape}")
        logging.debug(f"log_y_adj shape: {log_y_adj.shape}")
        log_x_adj = log_sum_exp(log_x_adj, dim=-1)
        #log_x_adj = log_x_adj.expand(-1,-1,128, -1)
        log_y_adj = log_sum_exp(log_y_adj, dim=-1)
        #log_y_adj = log_y_adj.expand(-1,-1,128, -1)
        logging.debug(f"log_x_adj shape after log_sum_exp: {log_x_adj.shape}")
        logging.debug(f"log_y_adj shape after log_sum_exp: {log_y_adj.shape}")
        einsum_tensor = torch.einsum(equation, [log_x_adj, log_y_adj])
        logging.debug(f"einsum_tenspr shape: {einsum_tensor.shape}")
        einsum_tensor = einsum_tensor.unsqueeze(-1)
        # ✅ Ensure max_val reduces along the last dim before logsumexp
        max_val, _ = torch.max(einsum_tensor, dim=-1, keepdim=True)  
        logging.debug(f"Shape of max_val: {max_val.shape}")  # Should be [batch, seq_len, seq_len, 1]
        einsum_tensor_adj = einsum_tensor - max_val

        logging.debug(f"Shape of einsum t after max subtraction: {einsum_tensor_adj.shape}")
        einsum_tensor_adj = torch.logsumexp(einsum_tensor_adj, dim=-1)
        logging.debug(f"Shape einsum t before sum: {einsum_tensor_adj.shape}")
        # ✅ Apply logsumexp only across the correct dimension
        output = torch.einsum('bkd,bkdt->bkd', einsum_tensor_adj, max_val)
        logging.debug(f"Shape einsum output: {output.shape}")

        return  output


def log_sum_exp(tensor, dim=-1, keepdim=True):
    """
    Optimized Log-Sum-Exp function for stable summation in log-space.
    Prevents overflow and underflow issues by normalizing.
    """
    logging.debug(f"shape of tensor for log_sum_exp {tensor.shape}")
    
    max_val, _ = torch.max(tensor, dim=dim, keepdim=True)  # Find max value
    return max_val + torch.log(torch.sum(torch.exp(tensor - max_val), dim=dim, keepdim=keepdim))

class QuaternionGeneticAlgorithm:
    def __init__(self, model, mutation_rate, population_size=5):
        self.model = model
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.population = [self._randomize_weights() for _ in range(population_size)]

    def _randomize_weights(self):
        new_model = copy.deepcopy(self.model)
        for param in new_model.parameters():
            param.data += torch.randn_like(param) * self.mutation_rate  # Mutate weights
        return new_model

    def select_best(self, loss_fn, inputs, targets, tot_targets, mask):
        best_model = None
        best_loss = float('inf')
        n=0
        for model in self.population:
            loss = 0
            logits, reward, train_of_thought_logits = model(inputs, mask)
            # Flatten logits and targets:
            tot_logits_flat = train_of_thought_logits.reshape(-1, train_of_thought_logits.size(-1))
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                logging.error("Logits contain NaN or Inf values!")
                raise ValueError("Logits contain NaN or Inf values!")
            if torch.isnan(targets).any() or torch.isinf(targets).any():
                logging.error("Logits contain NaN or Inf values!")
                raise ValueError("Logits contain NaN or Inf values!")
            if torch.isnan(tot_logits_flat).any() or torch.isinf(tot_logits_flat).any():
                logging.error("Logits contain NaN or Inf values!")
                raise ValueError("Logits contain NaN or Inf values!")

            logits_flat = logits.reshape(-1, logits.size(-1))
            loss += loss_fn(logits_flat, tot_targets)
            logging.debug(f"Logits_flat shape: {logits_flat.shape}")
            logging.debug(f"targets_flat shape: {targets.shape}")  
        

            if loss < best_loss:
                    best_loss = loss
                    n=n+1
                    print(f"Best model iteration {n}, Loss: {loss.item()}")
                    best_model = model
            
            else:
                loss = 0
                logits, reward, train_of_thought_logits = model(inputs, mask)
                # Flatten logits and targets:
                tot_logits_flat = train_of_thought_logits.reshape(-1, train_of_thought_logits.size(-1))
                if torch.isnan(logits).any() or torch.isinf(logits).any():
                    logging.error("Logits contain NaN or Inf values!")
                    raise ValueError("Logits contain NaN or Inf values!")
                if torch.isnan(targets).any() or torch.isinf(targets).any():
                    logging.error("Logits contain NaN or Inf values!")
                    raise ValueError("Logits contain NaN or Inf values!")
                if torch.isnan(tot_logits_flat).any() or torch.isinf(tot_logits_flat).any():
                    logging.error("Logits contain NaN or Inf values!")
                    raise ValueError("Logits contain NaN or Inf values!")

                logits_flat = logits.reshape(-1, logits.size(-1))
                loss += loss_fn(logits_flat, tot_targets)
                n=n+1
                print(f"Iteration {n}, Loss: {loss}")
                if loss < best_loss:
                        best_loss = loss
                        n=n+1
                        print(f"Best model iteration {n}, Loss: {loss.item()}")
                        best_model = model
                logging.debug(f"Logits_flat shape: {logits_flat.shape}")
                logging.debug(f"targets_flat shape: {targets.shape}")    
        return best_model

    def evolve(self, loss_fn, inputs, targets, tot_targets, mask):
        best_model = self.select_best(loss_fn, inputs, targets, tot_targets, mask)
        self.population = [copy.deepcopy(best_model) for _ in range(self.population_size)]
        for model in self.population:
            for param in model.parameters():
                param.data += torch.randn_like(param) * self.mutation_rate  # Apply mutation
        # Return the best model from the new population.
        return self.select_best(loss_fn, inputs, targets, tot_targets, mask)
    
class QuaternionFireflyOptimizer:
    def __init__(self, model, num_fireflies=5, alpha=0.1, beta=0.5):
        self.population = [copy.deepcopy(model) for _ in range(num_fireflies)]
        self.alpha = alpha
        self.beta = beta

    def move_towards(self, firefly1, firefly2):
        for p1, p2 in zip(firefly1.parameters(), firefly2.parameters()):
            p1.data += self.beta * (p2.data - p1.data) + self.alpha * torch.randn_like(p1)

    def optimize(self, loss_fn, data_loader):
        fitness = [sum(loss_fn(m(batch[0]), batch[1]).item() for batch in data_loader) for m in self.population]
        best_idx = torch.argmin(torch.tensor(fitness))
        best_firefly = self.population[best_idx]
        n=0
        for i in range(len(self.population)):
            if i != best_idx:
                self.move_towards(self.population[i], best_firefly)
                n=n+1
                print(f"Iteration {n}, Loss: {fitness.item()}")
        return best_firefly

class QuaternionNEAT:
    def __init__(self, model, population_size=5):
        self.population = [copy.deepcopy(model) for _ in range(population_size)]

    def mutate_topology(self, model):
        new_model = copy.deepcopy(model)
        if random.random() < 0.5:
            # Add a new quaternion neuron
            new_layer = nn.Linear(new_model.layers[0].in_features, new_model.layers[0].out_features)
            new_model.layers.insert(random.randint(0, len(new_model.layers)), new_layer)
        return new_model

    def evolve(self, loss_fn, data_loader):
        n=0
        best_model = min(self.population, key=lambda m: sum(loss_fn(m(batch[0]), batch[1]).item() for batch in data_loader))

        self.population = [self.mutate_topology(best_model) for _ in range(len(self.population))]
        n=n+1
        print(f"Iteration {n}, Loss: {self.population.item()}")
        return best_model

class ReasoningModelGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Reasoning Model GUI")

        # Transformer Parameters
        self.layers = []
        # Model Configuration Variables
        self.model_name = tk.StringVar(value="Reasoning Model")
        self.num_parameters = tk.IntVar(value=1024)
        self.num_heads = tk.IntVar(value=8)
        self.vocab_size = tk.IntVar(value=10000)
        self.hidden_size = tk.IntVar(value=8)
        self.num_layers = tk.IntVar(value=2)

        self.pad_token_id = 0  # Default value, adjust based on your tokenizer setup

        # Device Selection Variable
        self.device_option = tk.StringVar(value="GPU" if torch.cuda.is_available() else "CPU")

        # Dynamically calculate parameters based on other inputs
        self.vocab_size.trace_add("write", lambda *args: self.update_num_parameters())
        self.hidden_size.trace_add("write", lambda *args: self.update_num_parameters())
        self.num_layers.trace_add("write", lambda *args: self.update_num_parameters())

        # Set initial calculated value
        self.update_num_parameters()

        # Training Parameters
        self.dataset_path = ""
        self.vocab_path = ""
        self.tokenizer_path = ""
        self.batch_size = tk.IntVar(value=1)
        self.learning_rate = tk.DoubleVar(value=0.01)
        self.epochs = tk.IntVar(value=1)

        # Training Variables
        self.loss_history = []
        self.accuracy_history = []
        self.current_epoch = 0
        self.stop_training = threading.Event()

        # Model and Data Variables
        self.model = None
        self.tokenizer = None
        self.dataset_path = None
        self.vocab_path = None
        self.tokenizer_path = None
        self.model_path = None
        self.train_data = None  # To store the dataset
        self.tokenized_data_path = None  # To store the tokenized data file path
        self.use_genetic_algo = "Genetic Algorithm"  # default to optim
        self.validation_loader = None
        
        # Device (CPU or GPU) - Initially set based on device_option
        self.device = torch.device(self.map_device(self.device_option.get()))

        # Select log file path
        self.select_log_file()

        # Setup logging
        logging.basicConfig(filename=self.log_file_path, level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')

        logging.info(f"Using device: {self.device}")

        self.create_widgets()

    def map_device(self, selected_device):
        device_mapping = {
            "CPU": "cpu",
            "GPU": "cuda"
        }
        return device_mapping.get(selected_device, "cpu")

    def create_widgets(self):
        # Transformer Construction Frame
        transformer_frame = ttk.LabelFrame(self.root, text="Transformer Construction", padding=(10, 10))
        transformer_frame.pack(fill="x", padx=10, pady=5)

        ttk.Label(transformer_frame, text="Number of Parameters:").grid(row=0, column=0, sticky="w")
        ttk.Entry(transformer_frame, textvariable=self.num_parameters, state="readonly").grid(row=0, column=1)

        ttk.Label(transformer_frame, text="Number of Heads:").grid(row=1, column=0, sticky="w")
        ttk.Entry(transformer_frame, textvariable=self.num_heads).grid(row=1, column=1)
        
        ttk.Label(transformer_frame, text="Vocabulary Size:").grid(row=2, column=0, sticky="w")
        ttk.Entry(transformer_frame, textvariable=self.vocab_size).grid(row=2, column=1)

        ttk.Label(transformer_frame, text="Hidden Size:").grid(row=3, column=0, sticky="w")
        ttk.Entry(transformer_frame, textvariable=self.hidden_size).grid(row=3, column=1)

        ttk.Label(transformer_frame, text="Number of Layers:").grid(row=2, column=4, sticky="w")
        ttk.Entry(transformer_frame, textvariable=self.num_layers).grid(row=2, column=5)

        # Device Selection
        ttk.Label(transformer_frame, text="Select Device:").grid(row=4, column=0, sticky="w", pady=(10, 0))
        device_options = ["CPU"]
        if torch.cuda.is_available():
            device_options.append("GPU")
        device_combo = ttk.Combobox(transformer_frame, textvariable=self.device_option, values=device_options, state="readonly")
        device_combo.grid(row=4, column=1, sticky="w", pady=(10, 0))
        device_combo.bind("<<ComboboxSelected>>", self.on_device_change)

        # Attach parameter calculation to variable updates
        self.vocab_size.trace_add("write", lambda *args: self.update_num_parameters())
        self.hidden_size.trace_add("write", lambda *args: self.update_num_parameters())
        self.num_layers.trace_add("write", lambda *args: self.update_num_parameters())

        # For resuming training
        ttk.Button(transformer_frame, text="Select Model File", command=self.select_model_file).grid(row=3, column=2, pady=5)

        # Architecture selection
        self.architecture = tk.StringVar(value="Reasoning Model")
        ttk.Label(transformer_frame, text="Select Architecture:").grid(row=0, column=2, sticky="w")
        ttk.Combobox(transformer_frame, textvariable=self.architecture, values=["Reasoning Model", "Reasoning Model LNS"], state="readonly").grid(row=0, column=3)

        ttk.Button(transformer_frame, text="Add Layer", command=self.add_layer).grid(row=4, column=0, pady=5)
        ttk.Button(transformer_frame, text="Save Transformer and Model", command=self.save_transformer_and_model).grid(row=1, column=3, pady=5)
        ttk.Button(transformer_frame, text="Load Transformer", command=self.load_transformer).grid(row=1, column=2, pady=5)
        ttk.Button(transformer_frame, text="Initialize/Load Model", command=self.load_model).grid(row=2, column=3, pady=5)
        self.genetic_algo_var = tk.StringVar(value="GHR Optim")
        ttk.Label(transformer_frame, text="Algo:").grid(row=0, column=2, sticky="w")
        ttk.Combobox(transformer_frame, textvariable=self.genetic_algo_var, values=["GHR Optim", "Genetic Algorithm", "Firefly", "NEAT"], state="readonly").grid(row=0, column=4)

        # Data Selection Frame
        data_frame = ttk.LabelFrame(self.root, text="Data Selection", padding=(10, 10))
        data_frame.pack(fill="x", padx=10, pady=5)
        self.use_chunked_dataset = tk.BooleanVar(value=False)
        self.test_bool = tk.BooleanVar(value=False)
        
        ttk.Checkbutton(data_frame, text="Use Chunked Dataset", variable=self.use_chunked_dataset).pack(pady=5)
        ttk.Checkbutton(data_frame, text="Use Std/bert Model", variable=self.test_bool).pack(pady=5)
        ttk.Button(data_frame, text="Select Dataset Directory", command=self.select_dataset).pack(pady=5)
        ttk.Button(data_frame, text="Load Dataset", command=self.load_dataset).pack(pady=5)
        ttk.Button(data_frame, text="Save Dataset as Text File", command=self.save_dataset_as_text).pack(pady=5)
        ttk.Button(data_frame, text="Select Vocabulary File", command=self.select_vocab).pack(pady=5)
        ttk.Button(data_frame, text="Create Tokenizer from Vocab", command=self.create_tokenizer_from_vocab).pack(pady=5)
        ttk.Button(data_frame, text="Load Tokenizer", command=self.load_tokenizer).pack(pady=5)
        ttk.Button(data_frame, text="Test Tokenizer", command=self.test_tokenizer).pack(pady=5)
        ttk.Button(data_frame, text="Select Validation Dataset", command=self.select_validation_dataset).pack(pady=5)

        # New buttons for tokenized data
        ttk.Button(data_frame, text="Select/Create Tokenized Data", command=self.select_or_create_tokenized_data).pack(pady=5)
        ttk.Button(data_frame, text="Tokenize Data", command=self.tokenize_data).pack(pady=5)

        # Training Configuration Frame
        train_frame = ttk.LabelFrame(self.root, text="Training Configuration", padding=(10, 10))
        train_frame.pack(fill="x", padx=10, pady=5)

        ttk.Label(train_frame, text="Batch Size:").grid(row=0, column=0, sticky="w")
        ttk.Entry(train_frame, textvariable=self.batch_size).grid(row=0, column=1)

        ttk.Label(train_frame, text="Learning Rate:").grid(row=1, column=0, sticky="w")
        ttk.Entry(train_frame, textvariable=self.learning_rate).grid(row=1, column=1)

        ttk.Label(train_frame, text="Epochs:").grid(row=2, column=0, sticky="w")
        ttk.Entry(train_frame, textvariable=self.epochs).grid(row=2, column=1)

        ttk.Button(train_frame, text="Start Training", command=self.start_training).grid(row=3, column=0, pady=5)
        ttk.Button(train_frame, text="Save Model", command=self.save_model).grid(row=3, column=1, pady=5)
        ttk.Button(train_frame, text="Stop Training", command=self.stop_training_command).grid(row=4, column=0, pady=5)
        self.training_mode = tk.StringVar(value="response")  # Default
        training_modes = ["imitation", "completion", "response"]
        ttk.Combobox(data_frame, textvariable=self.training_mode, values=training_modes, state="readonly").pack(pady=5)
        ttk.Button(train_frame, text="Run Validation", command=self.run_validation_button).grid(row=5, column=0, pady=5)
        ttk.Button(train_frame, text="Test Inference", command=self.test_inference).grid(row=5, column=1, pady=5)

        # Progress Bar
        self.progress_bar = ttk.Progressbar(self.root, orient='horizontal', length=400, mode='determinate')
        self.progress_bar.pack(pady=10)
        self.status_label = ttk.Label(self.root, text="Status: Ready")
        self.status_label.pack(pady=5)

    def select_log_file(self):
        self.log_file_path = filedialog.asksaveasfilename(
            title="Select Log File Location",
            defaultextension=".log",
            filetypes=[("Log files", "*.log"), ("All files", "*.*")]
        )
        if self.log_file_path:
            print(f"Log file will be saved to: {self.log_file_path}")
        else:
            self.log_file_path = 'training_debug.log'  # Default log file
            print(f"No log file selected. Using default: {self.log_file_path}")

    def calculate_parameters(self, vocab_size, embedding_dim, hidden_dim, num_layers):
        embedding_params = vocab_size * embedding_dim * 4  # Quaternion embeddings (4x normal embedding size)
        transformer_params = num_layers * (
            (embedding_dim * hidden_dim * 4) +  # Attention layers
            (hidden_dim * hidden_dim * 4) +  # Feed-forward layers
            (hidden_dim * 4 * embedding_dim * 4)  # Output layers
        )
        output_projection_params = embedding_dim * 4 * vocab_size  # Final projection
        return embedding_params + transformer_params + output_projection_params

    def test_inference(self):
        if not self.model:
            messagebox.showerror("Error", "Model not loaded.")
            return

        # Set the model to evaluation mode
        self.model.eval()

        # Prompt the user for input text
        sample_input = simpledialog.askstring("Test Inference", "Enter input text:")
        if sample_input:
            try:
                # Tokenize the input (ensure you use return_tensors="pt" for a proper tensor)
                input_ids = self.tokenizer.encode(sample_input, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    # Forward pass through the model; if your model's forward method requires targets, pass None or adjust as needed
                    outputs, _= self.model(input_ids, mask=None)
                # Use argmax over the vocabulary dimension to pick predicted tokens
                predicted_ids = outputs.argmax(dim=-1)
                # Decode the predicted token IDs (skip special tokens if desired)
                predicted_text = self.tokenizer.decode(predicted_ids[0], skip_special_tokens=True)
                messagebox.showinfo("Inference Output", predicted_text)
                logging.info(f"Inference Output, {predicted_text}")

            except Exception as e:
                messagebox.showerror("Error", f"Inference failed: {str(e)}")
        # Optionally, return to train mode if further training is planned
        self.model.train()

    def run_validation(self, validation_loader, loss_fn):
        self.model.eval()
        total_val_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch_input_ids, batch_labels, batch_labels_tot, seq_lengths in validation_loader:
                batch_input_ids = batch_input_ids.to(self.device)
                batch_labels = batch_labels.to(self.device)
                # Here, if your model requires three inputs, you may also send batch_labels_tot
                outputs, _, _ = self.model(batch_input_ids, batch_labels.reshape(-1), batch_labels_tot.reshape(-1))
                # Flatten outputs and targets for loss calculation
                logits = outputs.reshape(-1, outputs.size(-1))
                targets = batch_labels.reshape(-1)
                loss = loss_fn(logits, targets)
                total_val_loss += loss.item()
                num_batches += 1

        avg_val_loss = total_val_loss / num_batches if num_batches > 0 else float('inf')
        logging.info(f"Validation Loss: {avg_val_loss:.4f}")
        self.model.train()
        return avg_val_loss

    def update_num_parameters(self):
        vocab_size = self.vocab_size.get()
        embed_size = self.hidden_size.get()
        num_layers = self.num_layers.get()
        hidden_size = self.hidden_size.get()

        total_params = self.calculate_parameters(vocab_size, embed_size, num_layers, hidden_size)
        self.num_parameters.set(total_params)

    def on_device_change(self, event):
        selected_device = self.device_option.get()
        if selected_device == "GPU" and not torch.cuda.is_available():
            messagebox.showerror("Error", "GPU selected but CUDA is not available on this system.")
            self.device_option.set("CPU")
            selected_device = "CPU"
        device_str = self.map_device(selected_device)
        self.device = torch.device(device_str)
        logging.info(f"Device changed to: {self.device}")
        messagebox.showinfo("Device Selection", f"Computation device set to: {selected_device}")

    def resize_checkpoint_weights(self, state_dict, new_vocab_size, embed_size):
        """
        Resize checkpoint weights to match the current model's dimensions.
        """
        # This method may need to be updated depending on the model's state_dict keys
        return state_dict

    def select_model_file(self):
        self.model_path = filedialog.askopenfilename(
            title="Select Model File",
            filetypes=[("Model Files", "*.pth;*.json"), ("All files", "*.*")]
        )
        if self.model_path:
            if self.model_path.endswith('.json'):
                # Load model configuration
                with open(self.model_path, 'r') as f:
                    config = json.load(f)
                # Update GUI parameters
                self.vocab_size.set(config.get("vocab_size", self.vocab_size.get()))
                self.hidden_size.set(config.get("embed_size", self.hidden_size.get()))
                self.num_layers.set(config.get("num_layers", self.num_layers.get()))
                self.num_heads.set(config.get("num_heads", self.num_heads.get()))
                self.architecture.set(config.get("architecture", self.architecture.get()))
                messagebox.showinfo("Success", f"Model configuration loaded from: {self.model_path}")
            elif self.model_path.endswith('.pth'):
                # Load model weights
                config_directory = os.path.dirname(self.model_path)
                config_path = os.path.join(config_directory, 'model_config.json')
                if os.path.exists(config_path):
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                    # Update GUI parameters
                    self.vocab_size.set(config.get("vocab_size", self.vocab_size.get()))
                    self.hidden_size.set(config.get("embed_size", self.hidden_size.get()))
                    self.num_layers.set(config.get("num_layers", self.num_layers.get()))
                    self.num_heads.set(config.get("num_heads", self.num_heads.get()))

                    self.architecture.set(config.get("architecture", self.architecture.get()))
                    # Load the model
                    self.load_model()
                    # Load model state
                    state_dict = torch.load(self.model_path, map_location=self.device)
                    self.model.load_state_dict(state_dict)
                    messagebox.showinfo("Success", f"Model weights and configuration loaded from: {self.model_path}")
                else:
                    messagebox.showwarning("Warning", "Model configuration file not found. Please ensure the configuration is set correctly.")
            else:
                messagebox.showerror("Error", "Unsupported file format selected.")

    def save_transformer_and_model(self):
        if not self.model:
            messagebox.showerror("Error", "Model has not been initialized. Please initialize the model first.")
            return
        if not self.tokenizer:
            messagebox.showerror("Error", "Tokenizer has not been initialized. Please load a tokenizer first.")
            return

        transformer_data = {
            "vocab_size": self.vocab_size.get(),
            "embed_size": self.hidden_size.get(),
            "hidden_size": self.hidden_size.get(),
            "num_layers": self.num_layers.get(),
            "architecture": self.architecture.get(),
            "num_heads": self.num_heads.get(),
            "num_parameters": self.num_parameters.get(),
            "layers": self.layers
        }

        directory = filedialog.askdirectory(title="Select Save Directory")
        if directory:
            # Save configuration
            config_path = os.path.join(directory, "model_config.json")
            with open(config_path, "w") as file:
                json.dump(transformer_data, file, indent=4)

            # Save weights
            if self.architecture.get() == "Reasoning Model":
                model_file_name = 'reasoning_model.pth'
            elif self.architecture.get() == "Reasoning Model LNS":
                model_file_name = 'reasoning_model_lns.pth'
            else:
                messagebox.showerror("Error", f"Unsupported architecture: {self.architecture.get()}")
                return

            model_path = os.path.join(directory, model_file_name)
            torch.save(self.model.state_dict(), model_path)

            # Save tokenizer
            self.tokenizer.save_pretrained(directory)

            messagebox.showinfo("Success", "Model, tokenizer, and configuration saved successfully!")
            logging.info("Model, tokenizer, and configuration saved successfully.")

    def select_dataset(self):
        self.dataset_path = filedialog.askdirectory(title="Select Dataset Directory")
        if self.dataset_path:
            messagebox.showinfo("Success", f"Dataset directory selected: {self.dataset_path}")

    def select_vocab(self):
        self.vocab_path = filedialog.askopenfilename(
            title="Select Vocabulary File",
            filetypes=[("JSON files", "*.json"), ("Text files", "*.txt"), ("All files", "*.*")]
        )
        if self.vocab_path:
            messagebox.showinfo("Success", f"Vocabulary file selected: {self.vocab_path}")

    def select_tokenizer(self):
        self.tokenizer_path = filedialog.askopenfilename(
            title="Select Tokenizer File",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if self.tokenizer_path:
            messagebox.showinfo("Success", f"Tokenizer file selected: {self.tokenizer_path}")

    def test_tokenizer(self):
        if not self.tokenizer:
            messagebox.showerror("Error", "Tokenizer not loaded.")
            return
        sample_text = simpledialog.askstring("Test Tokenizer", "Enter a sample text to tokenize:")
        if sample_text:
            tokens = self.tokenizer.tokenize(sample_text)
            token_ids = self.tokenizer.encode(sample_text)
            logging.info(f"Sample Text: {sample_text}")
            logging.info(f"Tokens: {tokens}")
            logging.info(f"Token IDs: {token_ids}")
            messagebox.showinfo("Tokenizer Test", f"Tokens: {tokens}\nToken IDs: {token_ids}")

    def save_dataset_as_text(self):
        if not hasattr(self, 'text_data') or not self.text_data:
            messagebox.showerror("Error", "No dataset loaded or processed to save.")
            return

        save_path = filedialog.asksaveasfilename(
            title="Save Dataset as Text File",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if save_path:
            try:
                with open(save_path, 'w', encoding='utf-8') as f:
                    for line in self.text_data:
                        f.write(line + '\n')
                messagebox.showinfo("Success", f"Dataset saved to {save_path}")
                logging.info(f"Dataset saved to {save_path}")
            except Exception as e:
                logging.error(f"Failed to save dataset: {e}")
                messagebox.showerror("Error", f"Failed to save dataset: {e}")
                
    def create_tokenizer_from_vocab(self):
        try:
            # Ask the user to select the vocabulary file (our generated tokenizer.json)
            vocab_path = filedialog.askopenfilename(
                title="Select Vocabulary File",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )
            if not vocab_path:
                messagebox.showerror("Error", "No vocabulary file selected.")
                return

            # Load the vocab from the JSON.
            with open(vocab_path, 'r', encoding='utf-8') as f:
                vocab_data = json.load(f)

            if "token_to_id" not in vocab_data:
                raise ValueError("The JSON file does not contain a 'token_to_id' key.")

            vocab = vocab_data["token_to_id"]

            # Check if merges exist in the file.
            if "merges" in vocab_data:
                merges = vocab_data["merges"]
                # Create a BPE model if merges are available.
                model = models.BPE(vocab=vocab, merges=merges, unk_token="<UNK>")
            else:
                # Fallback: use a WordLevel model if no merges are found.
                model = models.WordLevel(vocab=vocab, unk_token="<UNK>")

            # Create the tokenizer with the selected model.
            tokenizer = Tokenizer(model)

            # Set normalizer to NFKC for Unicode normalization.
            tokenizer.normalizer = normalizers.NFKC()
            # Use ByteLevel pre-tokenizer for byte-level tokenization.
            tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
            # Use ByteLevel decoder for correct reconstruction of text.
            tokenizer.decoder = decoders.ByteLevel()

            # Wrap with PreTrainedTokenizerFast for HF integration.
            self.tokenizer = PreTrainedTokenizerFast(
                tokenizer_object=tokenizer,
                unk_token="<UNK>",
                pad_token="<PAD>",
                bos_token="<BOS>",
                eos_token="<EOS>",
                model_max_length=seq_len  # Ensure seq_len is defined in your context.
            )

            # Ensure special tokens are added.
            self.tokenizer.add_special_tokens({
                'unk_token': '<UNK>',
                'pad_token': '<PAD>',
                'bos_token': '<BOS>',
                'eos_token': '<EOS>'
            })

            # Save the tokenizer.
            save_directory = filedialog.askdirectory(title="Select Directory to Save Tokenizer")
            if save_directory:
                os.makedirs(save_directory, exist_ok=True)
                self.tokenizer.save_pretrained(save_directory)
                self.tokenizer_path = os.path.join(save_directory, 'tokenizer.json')
                messagebox.showinfo("Success", f"Tokenizer saved to {self.tokenizer_path}")
                logging.info(f"Tokenizer saved to {self.tokenizer_path}")
            else:
                messagebox.showerror("Error", "No save directory selected for tokenizer.")
                return

            # Test the tokenizer.
            test_text = "Hello World!\nThis is a test.\tLet's remove line breaks and tabs."
            tokens = self.tokenizer.tokenize(test_text)
            logging.info(f"Test tokenization of '{test_text}': {tokens}")

            tokenizer_vocab = self.tokenizer.get_vocab()
            sorted_vocab = dict(sorted(tokenizer_vocab.items(), key=lambda item: item[1]))
            logging.info(f"Sorted Tokenizer Vocabulary: {sorted_vocab}")

            logging.info("Tokenizer created and saved successfully")
        except Exception as e:
            logging.error(f"Failed to create tokenizer: {str(e)}")
            messagebox.showerror("Error", f"Failed to create tokenizer: {str(e)}")
            raise


    def load_tokenizer(self):
        try:
            self.tokenizer_path = filedialog.askopenfilename(
                title="Select Tokenizer File",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )
            if not self.tokenizer_path or not os.path.exists(self.tokenizer_path):
                raise FileNotFoundError("Tokenizer file not selected or does not exist.")

            # Load the PreTrainedTokenizerFast from file.
            self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=self.tokenizer_path)
            logging.info(f"Tokenizer loaded from {self.tokenizer_path}")

            # If a special tokens map exists, load and add them.
            special_tokens_path = os.path.join(os.path.dirname(self.tokenizer_path), "special_tokens_map.json")
            if os.path.exists(special_tokens_path):
                with open(special_tokens_path, "r", encoding="utf-8") as file:
                    special_tokens = json.load(file)
                # Convert nested dicts to AddedToken if needed.
                for key, value in special_tokens.items():
                    if isinstance(value, dict):
                        special_tokens[key] = AddedToken(value["content"],
                                                        lstrip=value.get("lstrip", False),
                                                        rstrip=value.get("rstrip", False))
                    elif not isinstance(value, (str, AddedToken)):
                        raise ValueError(f"Invalid token format for key {key}: {value}")
                self.tokenizer.add_special_tokens(special_tokens)
                logging.info(f"Special tokens added: {special_tokens}")

            # Load tokenizer configuration if available.
            tokenizer_config_path = os.path.join(os.path.dirname(self.tokenizer_path), "tokenizer_config.json")
            if os.path.exists(tokenizer_config_path):
                with open(tokenizer_config_path, "r", encoding="utf-8") as file:
                    tokenizer_config = json.load(file)
                    self.tokenizer.init_kwargs.update(tokenizer_config)
                    if "model_max_length" in tokenizer_config:
                        self.tokenizer.model_max_length = tokenizer_config["model_max_length"]
                    logging.info(f"Tokenizer configuration loaded: {tokenizer_config}")

            # Ensure a reasonable model_max_length is set.
            if not hasattr(self.tokenizer, "model_max_length") or self.tokenizer.model_max_length > 1024 * 1024:
                self.tokenizer.model_max_length = seq_len  # Default value; ensure seq_len is defined
            logging.info(f"Model max length set to: {self.tokenizer.model_max_length}")

            # Log the vocabulary size.
            tokenizer_vocab_size = len(self.tokenizer)
            logging.info(f"Loaded tokenizer vocabulary size: {tokenizer_vocab_size}")
            self.vocab_size.set(tokenizer_vocab_size)

            # Ensure special tokens are correctly set.
            if not self.tokenizer.pad_token:
                self.tokenizer.pad_token = "<PAD>"
                self.tokenizer.pad_token_id = self.tokenizer.convert_tokens_to_ids("<PAD>")
                logging.warning("Pad token was not set. Defaulting to <PAD>.")
            if not self.tokenizer.unk_token:
                self.tokenizer.unk_token = "<UNK>"
                self.tokenizer.unk_token_id = self.tokenizer.convert_tokens_to_ids("<UNK>")
                logging.warning("UNK token was not set. Defaulting to <UNK>.")
            if not self.tokenizer.bos_token:
                self.tokenizer.bos_token = "<BOS>"
                self.tokenizer.bos_token_id = self.tokenizer.convert_tokens_to_ids("<BOS>")
                logging.warning("BOS token was not set. Defaulting to <BOS>.")
            if not self.tokenizer.eos_token:
                self.tokenizer.eos_token = "<EOS>"
                self.tokenizer.eos_token_id = self.tokenizer.convert_tokens_to_ids("<EOS>")
                logging.warning("EOS token was not set. Defaulting to <EOS>.")

            print("Special tokens map:", self.tokenizer.special_tokens_map)
            print("Pad token ID:", self.tokenizer.pad_token_id)
            print("Model max length:", self.tokenizer.model_max_length)

        except Exception as e:
            logging.error(f"Failed to load tokenizer: {str(e)}")
            messagebox.showerror("Error", f"Failed to load tokenizer: {str(e)}")


    def select_or_create_tokenized_data(self):
        use_chunked = self.use_chunked_dataset.get()
        answer = messagebox.askyesno("Select or Create Tokenized Data", "Do you want to use existing tokenized data?")
        
        if answer:
            if use_chunked:
                # User wants to use existing chunked tokenized data, select a directory
                self.tokenized_data_path = filedialog.askdirectory(
                    title="Select Tokenized Data Directory",
                    mustexist=True
                )
                if self.tokenized_data_path:
                    messagebox.showinfo("Success", f"Tokenized data directory selected: {self.tokenized_data_path}")
            else:
                # User wants to use existing single tokenized data file, select a file
                self.tokenized_data_path = filedialog.askopenfilename(
                    title="Select Tokenized Data File",
                    filetypes=[("JSON Lines files", "*.jsonl"), ("All files", "*.*")]
                )
                if self.tokenized_data_path:
                    # Attempt to load the file to validate its content
                    try:
                        with open(self.tokenized_data_path, 'r', encoding='utf-8') as f:
                            self.input_ids, self.labels, self.labels_tot = [], [], []
                            for line in f:
                                record = json.loads(line)
                                self.input_ids.append(record['input_ids'])
                                self.labels.append(record['labels'])
                                self.labels_tot.append(record['labels_tot'])
                        messagebox.showinfo("Success", f"Tokenized data file loaded: {self.tokenized_data_path}")
                        logging.info(f"Tokenized data file loaded successfully with {len(self.input_ids)} entries.")
                    except Exception as e:
                        messagebox.showerror("Error", f"Failed to load tokenized data file: {str(e)}")
        else:
            if use_chunked:
                # User wants to create new chunked tokenized data, select a directory to save
                self.tokenized_data_path = filedialog.askdirectory(
                    title="Select Directory to Save Tokenized Data"
                )
                if self.tokenized_data_path:
                    os.makedirs(self.tokenized_data_path, exist_ok=True)  # Ensure directory is created
                    messagebox.showinfo("Success", f"Tokenized data will be saved to directory: {self.tokenized_data_path}")
            else:
                # User wants to create new single tokenized data file, select a file path
                self.tokenized_data_path = filedialog.asksaveasfilename(
                    title="Save Tokenized Data As",
                    defaultextension=".jsonl",
                    filetypes=[("JSON Lines files", "*.jsonl"), ("All files", "*.*")]
                )
                if self.tokenized_data_path:
                    messagebox.showinfo("Success", f"Tokenized data will be saved to file: {self.tokenized_data_path}")


            
    def tokenize_data(self):
        if not self.tokenizer:
            messagebox.showerror("Error", "Tokenizer not loaded.")
            return
        if not hasattr(self, 'query_target_pairs') or not self.query_target_pairs:
            messagebox.showerror("Error", "No query-target pairs loaded. Please load the dataset first.")
            return
        if not self.tokenized_data_path:
            messagebox.showerror("Error", "Tokenized data path not set. Please select or create tokenized data.")
            return

        # Select training mode
        training_mode = self.training_mode.get()  # "imitation", "completion", "response"
        self.input_ids = []  # Initialize for unchunked dataset
        self.labels = []  # Initialize for unchunked dataset
        self.labels_tot = [] #for train of thought labels
        
        try:
            use_chunked = self.use_chunked_dataset.get()
            if use_chunked:
                #create path if none
                os.makedirs(self.tokenized_data_path, exist_ok=True)
                chunk_size = 32
                num_chunks = (len(self.query_target_pairs) + chunk_size - 1) // chunk_size

                for chunk_idx in range(num_chunks):
                    chunk_pairs = self.query_target_pairs[chunk_idx * chunk_size: (chunk_idx + 1) * chunk_size]
                    chunk_file_path = os.path.join(self.tokenized_data_path, f'chunk_{chunk_idx}.jsonl')

                    with open(chunk_file_path, 'w', encoding='utf-8') as f:
                        for query, target in chunk_pairs:
                            input_ids, labels, labels_tot = self._generate_training_pairs(query, target, tot_target, training_mode)
                            if input_ids and labels and labels_tot:
                                record = {'input_ids': input_ids, 'labels': labels, 'labels_tot': labels_tot}
                                f.write(json.dumps(record) + '\n')
                logging.info(f"Chunk {chunk_idx} tokenized and saved to {chunk_file_path}")

                messagebox.showinfo("Success", f"Data tokenized into {num_chunks} chunks and saved successfully to {self.tokenized_data_path}.")
                logging.info(f"Data tokenized into {num_chunks} chunks and saved successfully to {self.tokenized_data_path}.")
            else:
                with open(self.tokenized_data_path, 'w', encoding='utf-8') as f:
                    for query, target, tot_target in self.query_target_pairs:
                        input_ids, labels, labels_tot = self._generate_training_pairs(query, target, tot_target, training_mode)

                        if input_ids and labels and labels_tot:
                            self.input_ids.append(input_ids)  # Store for training
                            self.labels.append(labels)  # Store for training
                            self.labels_tot.append(labels_tot)  # Store for training
                            record = {'input_ids': input_ids, 'labels': labels, 'labels_tot': labels_tot}


                            f.write(json.dumps(record) + '\n')
                logging.info(f"Input IDs: {len(self.input_ids)} sequences loaded.")
                logging.info(f"Labels: {len(self.labels)} sequences loaded.")
                logging.info(f"Labels_tot: {len(self.labels_tot)} sequences loaded.")
                logging.info(f"Input_ids sample: {self.input_ids[0][:10]}...")  # Shows only first 10 tokens
                logging.info(f"Labels sample: {self.labels[0][:10]}...")  # Shows only first 10 tokens
                logging.info(f"Labels_tot sample: {self.labels_tot[0][:10]}...")  # First 10 tokens for ToT

                messagebox.showinfo("Success", f"Data tokenized and saved successfully to {self.tokenized_data_path}.")
                logging.info(f"Data tokenized and saved successfully to {self.tokenized_data_path}.")
        except Exception as e:
            logging.error(f"Tokenization failed: {str(e)}")
            messagebox.showerror("Error", f"Tokenization failed: {str(e)}")

    def _generate_training_pairs(self, query, target, tot_target, training_mode):
        # Debugging logs
        logging.debug(f"Generating Training Pairs - Query: {query}")
        logging.debug(f"Generating Training Pairs - Target: {target}")
        logging.debug(f"Generating Training Pairs - ToT: {tot_target}")

        # Ensure inputs are valid strings before tokenization
        query_ids = self.tokenizer.encode(str(query) if query else "", truncation=True, max_length=seq_len)
        target_ids = self.tokenizer.encode(str(target) if target else "", truncation=True, max_length=seq_len)
        tot_target_ids = self.tokenizer.encode(str(tot_target) if tot_target else "", truncation=True, max_length=seq_len)

        if training_mode == "imitation":
            input_ids = query_ids + [self.tokenizer.eos_token_id] 
            labels = query_ids + [self.tokenizer.eos_token_id] 
        elif training_mode == "completion":
            partial_length = len(query_ids) // 2
            partial_input = query_ids[:partial_length]
            input_ids = partial_input + [self.tokenizer.eos_token_id]
            labels = query_ids + [self.tokenizer.eos_token_id]  
        else:  # response mode
            input_ids = query_ids + [self.tokenizer.eos_token_id]
            labels = target_ids + [self.tokenizer.eos_token_id]
            labels_tot = tot_target_ids + [self.tokenizer.eos_token_id]

        return input_ids, labels, labels_tot


    def add_layer(self):
        layer_type = simpledialog.askstring("Layer Type", "Enter layer type (e.g., attention, feed_forward)")
        if layer_type:
            layer_config = {
                "type": layer_type,
                "parameters": {}  # Placeholder for future parameter configuration
            }
            self.layers.append(layer_config)
            messagebox.showinfo("Layer Added", f"Layer of type '{layer_type}' added.")

    def save_transformer(self):
        transformer_data = {
            "num_parameters": self.num_parameters.get(),
            "layers": self.layers
        }

        file_path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON files", "*.json")])
        if file_path:
            with open(file_path, "w") as file:
                json.dump(transformer_data, file, indent=4)
            messagebox.showinfo("Save", "Transformer saved successfully!")
            logging.info(f"Number of layers in the model: {len(self.model.transformer_encoder.layers)}")

    def load_transformer(self):
        file_path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
        if file_path:
            with open(file_path, "r") as file:
                transformer_data = json.load(file)
            self.num_parameters.set(transformer_data["num_parameters"])
            self.num_heads.set(transformer_data["num_heads"])
            self.layers = transformer_data["layers"]
            messagebox.showinfo("Success", "Transformer loaded successfully")

    def load_model(self):
        try:
            if not self.tokenizer:
                vocab_size = self.vocab_size.get()
            else:
                vocab_size = len(self.tokenizer)

            # Log and validate vocab size
            logging.info(f"Tokenizer vocabulary size: {vocab_size}")
            self.vocab_size.set(vocab_size)

            # Initialize the model based on architecture
            if self.architecture.get() == "Reasoning Model":
                self.model = Transformer_Model(
                    vocab_size=vocab_size,
                    embedding_dim=self.hidden_size.get(),
                    hidden_dim=self.hidden_size.get(),
                    num_layers=self.num_layers.get(),
                    num_heads=self.num_heads.get(),

                    seq_length=seq_len
                )
            elif self.architecture.get() == "Reasoning Model LNS":
                self.model = Transformer_Model_LNS(
                    vocab_size=vocab_size,
                    embedding_dim=self.hidden_size.get(),
                    hidden_dim=self.hidden_size.get(),
                    num_layers=self.num_layers.get(),
                    num_heads=self.num_heads.get(),
                    seq_length=seq_len
                )
            else:
                messagebox.showerror("Error", f"Unsupported architecture: {self.architecture.get()}")
                return

            # Move the entire model to the selected device
            self.model.to(self.device)
            logging.info(f"Model moved to device: {self.device}")

            # Load checkpoint if a model file is selected
            if self.model_path and self.model_path.endswith('.pth'):
                state_dict = torch.load(self.model_path, map_location=self.device)
                self.model.load_state_dict(state_dict, strict=True)
                logging.info("Model weights loaded and resized successfully.")

            logging.info(f"Model initialized on device: {self.device}")
            messagebox.showinfo("Success", "Model initialized successfully.")

        except Exception as e:
            logging.error(f"Failed to initialize model: {str(e)}")
            messagebox.showerror("Error", f"Failed to initialize model: {str(e)}")


    def calculate_learning_rate(self, total_params):
        # Calculate learning rate based on total parameters using the derived formula
        # LR = 17.38 * (Model Size)^-0.424
        lr = 17.38 * (total_params ** -0.424)
        return lr

    def start_training(self):
        # Start training in a separate thread to keep the GUI responsive
        self.stop_training.clear()
        training_thread = threading.Thread(target=self.training_loop)
        training_thread.start()

    def update_progress(self, progress_value):
        self.progress_bar['value'] = progress_value

    def update_status(self, message):
        self.status_label.config(text=f"Status: {message}")

    def save_checkpoint(self, model, optimizer, epoch, path):
        if not isinstance(path, (str, os.PathLike)):
            raise TypeError(f"Expected path to be str or os.PathLike, got {type(path).__name__}")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, path)
        

    def validate_training_parameters(self):
        # Validate batch size
        try:
            batch_size = int(self.batch_size.get())
            if batch_size <= 0:
                raise ValueError
        except (TypeError, ValueError):
            logging.error(f"Invalid batch size: {self.batch_size.get()}")
            messagebox.showerror("Error", "Batch size must be a positive integer.")
            return False

        # Validate epochs
        try:
            epochs = int(self.epochs.get())
            if epochs <= 0:
                raise ValueError
        except (TypeError, ValueError):
            logging.error(f"Invalid epochs value: {self.epochs.get()}")
            messagebox.showerror("Error", "Epochs must be a positive integer.")
            return False

        if not self.tokenized_data_path or not os.path.exists(self.tokenized_data_path):
            logging.error("Tokenized data path is invalid or does not exist.")
            messagebox.showerror("Error", "Tokenized data is not selected or does not exist.")
            return False

        if not hasattr(self.tokenizer, 'pad_token_id') or self.tokenizer.pad_token_id is None:
            logging.error("Tokenizer pad_token_id is not set.")
            messagebox.showerror("Error", "Tokenizer is missing pad_token_id.")
            return False

        return True

    def training_loop(self):
        if not self.validate_training_parameters():
            return

        logging.info("All training parameters and data are properly initialized.")
        if not self.model:
            logging.error("Model not initialized before training")
            return
        self.use_genetic_algo = self.genetic_algo_var.get()

        try:
            if self.use_chunked_dataset.get():
                # Initialize the ChunkedDataset
                dataset = ChunkedDataset(
                    tokenized_data_path=self.tokenized_data_path,
                    tokenizer=self.tokenizer,
                    max_length=seq_len
                )
                dataloader = DataLoader(
                    dataset,
                    batch_size=self.batch_size.get(),
                    shuffle=True,
                    num_workers=0,
                    pin_memory=False,
                    collate_fn=collate_fn
                )
            else:
                # Initialize the standard dataset and dataloader

                # Ensure the tokenizer is loaded and has a valid pad_token_id
                pad_token_id = tokenizer.pad_token_id if tokenizer else 0  # Default to 1 if tokenizer isn't set      
                max_length = seq_len  # Adjust as needed
                logging.info("max_length set")
                # Convert lists of token IDs to tensors and calculate original sequence lengths
                input_ids, seq_lengths = zip(*[
                    (
                        torch.tensor(tokens + [pad_token_id] * (max_length - len(tokens)), dtype=torch.int64, device=self.device)[:max_length],
                        min(len(tokens), max_length)
                    )
                    for tokens in self.input_ids
                ])
                logging.info("input ids torched to tensor")

                labels = [
                    torch.tensor(tokens + [pad_token_id] * (max_length - len(tokens)), dtype=torch.int64, device=self.device)[:max_length]
                    for tokens in self.labels
                ]
                logging.info("labels torched to tensor")

                labels_tot = [
                    torch.tensor(tokens + [pad_token_id] * (max_length - len(tokens)), dtype=torch.int64, device=self.device)[:max_length]
                    for tokens in self.labels_tot
                ]
                logging.info("labels_tot torched to tensor")


                attention_masks = [(ids != self.tokenizer.pad_token_id).long() for ids in input_ids]
                logging.info("attention masks set for pad tokens")

                attention_masks = torch.stack(attention_masks)
                # Stack tensors
                input_ids = torch.stack(input_ids)
                labels = torch.stack(labels)

                labels_tot = torch.stack(labels_tot)
                seq_lengths = torch.tensor(seq_lengths, dtype=torch.long)
                logging.info("datas stacked and seq lengths torched")

                # Perform assertions to validate tensors
                assert isinstance(input_ids, torch.Tensor), "input_ids should be a tensor"
                assert isinstance(labels, torch.Tensor), "labels should be a tensor"
                assert input_ids.dtype == torch.long, "input_ids should be of type torch.long"
                assert labels.dtype == torch.long, "labels should be of type torch.long"
                assert input_ids.size(1) == max_length, "input_ids should be padded to max_length"
                assert labels.size(1) == max_length, "labels should be padded to max_length"

                dataset = torch.utils.data.TensorDataset(input_ids, labels, labels_tot, seq_lengths, attention_masks)
                logging.info("dataset torched")
                dataloader = DataLoader(
                    dataset,
                    batch_size=int(self.batch_size.get()),
                    shuffle=True,
                    num_workers=0,  # Set to 0 to prevent multiple workers from loading chunks simultaneously
                    pin_memory=False,
                    collate_fn=collate_fn
                )
                logging.info("dataloader defined")
            ##chunked vs. standard else complete
            # Log dataset samples


            # Adjust learning rate based on architecture
            total_params = self.num_parameters.get()
            lr = self.learning_rate.get()
            logging.info(f"Learning Rate: {lr} for total parameters: {total_params}")

            # Learning rate scheduler
            total_steps = self.epochs.get() * len(dataloader)
            logging.info(f"Total training steps: {total_steps}")
            # Separate parameters based on their shape.

            # Create two optimizers:
            #Enable for standard optimizer/scheduler
            #num_warmup_steps = total_steps // 10  # Warmup for 10% of training
            #scheduler = self.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, total_steps)
            optimizer_std = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
            optimizer_tot = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)

            logging.info("Scheduler defined")
            loss_fn = nn.CrossEntropyLoss(ignore_index=pad_token_id)

            self.model.train()
            logging.info("Model set to training mode")
            progress_step = 0
            n = 0
            
            for epoch in range(self.epochs.get()):
                if self.stop_training.is_set():
                    logging.info("Training stopped by user.")
                    messagebox.showinfo("Info", "Training stopped by user.")
                    break

                epoch_loss = 0
                logging.info(f"Epoch {epoch+1} started")
                previous_loss = float('inf')
                max_inner_updates = 3  # limit of repitions
          
                # Training loop
                for batch_idx, (batch_input_ids, batch_labels, batch_labels_tot, seq_lengths, batch_attention_masks) in enumerate(dataloader):
                    if self.stop_training.is_set():
                        logging.info("Training stopped by user.")
                        messagebox.showinfo("Info", "Training stopped by user.")
                        return
                    inner_updates = 0
                    current_loss = float('inf')
                    while inner_updates < max_inner_updates and current_loss >= previous_loss:
                                        
                        optimizer_tot.zero_grad(set_to_none=True)
                        optimizer_std.zero_grad(set_to_none=True)
                        logging.debug("Optimizer gradients zeroed")

                        # Move batches and targets to the correct device 
                        batch_input_ids = batch_input_ids.to(self.device)
                        batch_labels = batch_labels.to(self.device)
                        batch_labels_tot = batch_labels_tot.to(self.device)
                        batch_attention_masks = batch_attention_masks.to(self.device)

                        seq_lengths = seq_lengths.to(self.device)
                        logging.debug("Batch moved to device")

                        # Logging epoch and batch info
                        logging.debug(f'Epoch: {epoch + 1}, Batch: {batch_idx + 1}')
                        logging.debug(f'Batch input_ids shape: {batch_input_ids.shape}')  # (batch_size, 1024)
                        logging.debug(f'Using device: {self.device}')


                        logging.debug(f"Shape of batch_input_ids before creat_combined_mask: {batch_input_ids.shape}")

                        batch_size, seq_length = batch_input_ids.size()
                        num_heads = self.num_heads.get()
                        # Assuming batch_input_ids and pad_token_id are defined.
                        combined_mask = create_combined_mask(batch_input_ids, self.tokenizer.pad_token_id)
                        logging.debug(f"Shape of batch_labels: {batch_labels.shape}")
                        combined_mask = combined_mask.unsqueeze(1)  # Shape (batch_size, 1, seq_len, seq_len)
                        combined_mask = combined_mask.expand(batch_size, num_heads, seq_len, seq_len)  # Shape (batch_size, num_heads, seq_len, seq_len)
                        combined_mask = combined_mask.reshape(batch_size * num_heads, seq_len, seq_len)  # Flatten batch_size * num_heads

                        # Log the shape of the combined mask
                        logging.debug(f"Shape of combined_mask: {combined_mask.shape}")
                        logging.debug(f"Shape of batch_input_ids being passed to model: {batch_input_ids.shape}")

                        targets_reshaped = batch_labels.reshape(-1)              

                        tot_targets_reshaped = batch_labels_tot.reshape(-1)             

                        # Check the flag and run evolution once per epoch if requested:
                        if self.use_genetic_algo == "Genetic Algorithm":
                            logging.info("Applying genetic algorithm evolution step...")
                            qga = QuaternionGeneticAlgorithm(self.model, lr)
                            # Evolve using the same loss function and dataloader (or a validation subset)
                            self.model = qga.evolve(loss_fn, batch_input_ids, targets_reshaped, tot_targets_reshaped, mask=combined_mask)
                            #Remove optimizer steps and gradient code enable this for Quaternion NeuroEvolution of Augmenting Topologies (NEAT)
                        elif self.use_genetic_algo == "NEAT":
                            neat = QuaternionNEAT(self.model)
                            self.model = neat.evolve(F.cross_entropy, dataloader)
                        elif self.use_genetic_algo == "Firefly":
                            #Remove optimizer steps and gradient lines to enable this for Quaternion Firefly Algo
                            firefly_optimizer = QuaternionFireflyOptimizer(self.model)
                            self.model = firefly_optimizer.optimize(F.cross_entropy, dataloader)
                        else:
                        # Forward pass
                            try:

                                outputs, train_of_thought_logits = self.model(batch_input_ids, combined_mask)
                                if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                                        logging.error("Model outputs contain NaN or Inf values.")


                            except Exception as e:
                                raise ValueError(f"forward pass failed for {str(e)}")


                            logging.debug(f"Shape of outputs: {outputs.shape}")
                            # Flatten the logits and targets:
                            tot_logits_flat = train_of_thought_logits.reshape(-1, train_of_thought_logits.size(-1))
                            logits_reshaped = outputs.reshape(-1, outputs.size(-1))   
                            loss = loss_fn(logits_reshaped, tot_targets_reshaped)
                            

                            logging.info(f"Loss computed: {loss.item()}")


                            # Backward pass and optimization
                            loss.backward(retain_graph=True)
                            logging.info("Loss backward computed")

                            for param_group in optimizer_std.param_groups:
                                logging.debug(f"Learning rate: {param_group['lr']}")
                            
                            # Check for NaN or Inf in gradients
                            for name, param in self.model.named_parameters():
                                if param.grad is not None:
                                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                                        logging.error(f"Gradient for {name} contains NaN or Inf.")
                                        continue
                                    
                            for name, param in self.model.named_parameters():
                                if param.grad is not None:
                                    logging.debug(f"Gradient for {name}: mean={param.grad.mean().item():.4f}, max={param.grad.max().item():.4f}, min={param.grad.min().item():.4f}")
                                else:
                                    logging.debug(f"Gradient for {name} is None")


                            total_norm = 0.0
                            for p in self.model.parameters():
                                if p.grad is not None:
                                    total_norm += p.grad.data.norm(2).item() ** 2
                            total_norm = total_norm ** 0.5
                            logging.info(f"Gradient norm: {total_norm}")

                            ###Uncomment these for gradient clipping
                            #torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                            

                            # Log gradients for debugging
                            for name, param in self.model.named_parameters():
                                if param.grad is not None:
                                    logging.debug(f"Gradients for {name}: {param.grad}")
                                else:
                                    logging.debug(f"No gradients found for {name}.")
                                                        
                            optimizer_tot.step()
                            optimizer_std.step()

                            n+=1
                            print(f"Iteration {n}, Loss: {loss.item()}")

                                                    
                            # Before optimizer step
                            for name, param in self.model.named_parameters():
                                if param.requires_grad:
                                    logging.debug(f"Before step - {name}: mean={param.data.mean().item():.4f}, std={param.data.std().item():.4f}")


                            # After optimizer step
                            for name, param in self.model.named_parameters():
                                if param.requires_grad:
                                    logging.debug(f"After step - {name}: mean={param.data.mean().item():.4f}, std={param.data.std().item():.4f}")


                            logging.info("Optimizer step update completed")
                            with torch.no_grad():
                                for param in self.model.parameters():
                                    if param.grad is not None:
                                        param.grad = param.grad.detach()

                            
                            current_loss = loss.item()
                            
                        previous_loss = current_loss
                        inner_updates += 1
                        logging.info(f"Reptition {inner_updates} completed")

                        
                    progress_step += 1
                    progress_value = (progress_step / total_steps) * 100
                    self.root.after(0, self.update_progress, progress_value)

                    # Save checkpoint at specified intervals
                    save_interval = 5  # Save every 25%
                    progress_percentage = (batch_idx + 1) / len(dataloader) * 100
                    if abs(progress_percentage % save_interval) < 1e-6:  # Avoid floating-point issues
                        checkpoint_path = f"checkpoints/epoch_{epoch}_batch_{batch_idx}.pth"
                        self.save_checkpoint(self.model, optimizer_std, epoch, checkpoint_path)
                        logging.info(f"Checkpoint saved at epoch {epoch}, batch {batch_idx}, progress: {progress_percentage:.2f}%")
                    
                    # perform validation after specified progress steps
                    validation_interval = 5  # Save every 25%
                    progress_percentage = (batch_idx + 1) / len(dataloader) * 100
                    if abs(progress_percentage % validation_interval) < 1e-6:  # Avoid floating-point issues

                        if self.validation_loader is not None:  
                            val_loss = self.run_validation(self.validation_loader, loss_fn)
                            print(f"Validation Loss after epoch {epoch+1}: {val_loss:.4f}")
                            logging.info(f"Validation Loss after epoch {epoch+1}: {val_loss:.4f}")

                # Log epoch loss
                logging.info(f"Epoch {epoch + 1}/{self.epochs.get()} completed")
                self.root.after(0, self.update_status, f"Epoch {epoch + 1}/{self.epochs.get()} completed. Current LR")

        except Exception as e:
            logging.error(f"An error occurred during training: {str(e)}")
            messagebox.showerror("Error", f"An error occurred during training: {str(e)}")

    def improved_collate_fn(self, batch):
        input_ids, attention_masks, labels, seq_lengths = zip(*batch)
        
        # Convert sequences to tensors if they aren't already
        input_ids = [x if isinstance(x, torch.Tensor) else torch.tensor(x) for x in input_ids]
        attention_masks = [x if isinstance(x, torch.Tensor) else torch.tensor(x) for x in attention_masks]
        labels = [x if isinstance(x, torch.Tensor) else torch.tensor(x) for x in labels]
        
        # Find max length in batch
        max_len = seq_len
        
        # Pad sequences using torch operations
        def pad_sequence(sequences, max_len, pad_value):
            return torch.stack([
                torch.cat([
                    seq,
                    torch.full((max_len - len(seq),), pad_value, dtype=seq.dtype, device=seq.device)
                ]) if len(seq) < max_len else seq[:max_len]
                for seq in sequences
            ])
        
        # Pad all sequences
        padded_input_ids = pad_sequence(input_ids, max_len, self.tokenizer.pad_token_id)
        padded_attention_masks = pad_sequence(attention_masks, max_len, 0)
        padded_labels = pad_sequence(labels, max_len, self.tokenizer.pad_token_id)
        
        # Convert sequence lengths to tensor
        seq_lengths = torch.tensor(seq_lengths, dtype=torch.long)
        
        return padded_input_ids, padded_attention_masks, padded_labels, seq_lengths

    def get_cosine_schedule_with_warmup(self, optimizer, num_warmup_steps, num_training_steps):
        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
            return max(0.0, 0.5 * (1.0 + torch.cos(torch.pi * progress)))
            
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    def save_model(self):
        if not self.model:
            messagebox.showerror("Error", "Model has not been initialized. Cannot save.")
            logging.error("Attempted to save model but model was not initialized.")
            return
        if not self.tokenizer:
            messagebox.showerror("Error", "Tokenizer has not been initialized. Cannot save.")
            logging.error("Attempted to save model but tokenizer was not initialized.")
            return

        save_directory = filedialog.askdirectory(title="Select Save Directory")
        if save_directory:
            config = {
            "vocab_size": self.vocab_size.get(),
            "embed_size": self.hidden_size.get(),
            "hidden_size": self.hidden_size.get(),
            "num_layers": self.num_layers.get(),
            "architecture": self.architecture.get(),
            "num_parameters": self.num_parameters.get(),
            "num_heads": self.num_heads.get(),
            "layers": self.layers
        }
            config_path = os.path.join(save_directory, 'model_config.json')
            with open(config_path, 'w') as f:
                json.dump(config, f)

            # Ensure embeddings match tokenizer
            tokenizer_vocab_size = len(self.tokenizer)

            # Save the model state dictionary
            if self.architecture.get() == "Reasoning Model":
                model_file_name = 'reasoning_model.pth'
            elif self.architecture.get() == "Reasoning Model LNS":
                model_file_name = 'reasoning_model_lns.pth'
            else:
                messagebox.showerror("Error", f"Unsupported architecture: {self.architecture.get()}")
                return

            model_path = os.path.join(save_directory, model_file_name)
            torch.save(self.model.state_dict(), model_path)

            # Save the tokenizer
            self.tokenizer.save_pretrained(save_directory)

            messagebox.showinfo("Success", "Model, tokenizer, and config saved successfully.")
            logging.info("Model, tokenizer, and config saved successfully.")

    def stop_training_command(self):
        self.stop_training.set()
        messagebox.showinfo("Stop Training", "Training stopped.")
        logging.info("Training stopped by user.")

    def expand_transformer(self):
        # Placeholder method; not used in current implementation
        pass

    
    def load_dataset(self):
        if self.use_chunked_dataset.get():
            # Load data from chunked files
            self.tokenized_data_path = filedialog.askdirectory(
                title="Select Tokenized Data Directory"
            )
            if not self.tokenized_data_path:
                messagebox.showerror("Error", "No tokenized data directory selected.")
                return

            # Check if directory contains chunked data files
            chunk_files = [f for f in os.listdir(self.tokenized_data_path) if f.startswith('chunk_') and f.endswith('.jsonl')]
            if not chunk_files:
                messagebox.showerror("Error", "No chunked data files found in the selected directory.")
                return

            self.chunked_files = [os.path.join(self.tokenized_data_path, f) for f in chunk_files]
            messagebox.showinfo("Success", f"Loaded chunked dataset with {len(self.chunked_files)} files.")
            logging.info(f"Loaded chunked dataset with {len(self.chunked_files)} files.")
        else:
            # Load standard dataset
            if not self.dataset_path:
                messagebox.showerror("Error", "No dataset directory selected.")
                return

            dataset_files = os.listdir(self.dataset_path)
            self.query_target_pairs = []

            for file in dataset_files:
                file_path = os.path.join(self.dataset_path, file)
                if file.endswith('.json') or file.endswith('.jsonl'):
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            if file.endswith('.jsonl'):
                                for line in f:
                                    conversation = json.loads(line.strip())
                                    self.query_target_pairs.extend(self.extract_query_target_pairs([conversation]))

                                # After loading query_target_pairs
                                for i in range(min(5, len(self.query_target_pairs))):
                                    query, target, tot_target = self.query_target_pairs[i]


                            else:
                                data = json.load(f)
                                self.query_target_pairs.extend(self.extract_query_target_pairs(data)) 
                                # After loading query_target_pairs
                                for i in range(min(5, len(self.query_target_pairs))):
                                    query, target, tot_target = self.query_target_pairs[i]
                               
                    except Exception as e:
                        messagebox.showerror("Error", f"Failed to read JSON file '{file}': {str(e)}")
                else:
                    messagebox.showwarning("Warning", f"Unsupported file format: '{file}'")

            if not self.query_target_pairs:
                messagebox.showerror("Error", "No valid query/target pairs found in the dataset.")
                return

            # Store text data for saving as a text file
            self.text_data = []
            for query, target, tot_target in self.query_target_pairs:
                self.text_data.append(f"User: {query}\nAssistant: {target}\nReasoning: {tot_target}")

            messagebox.showinfo("Success", f"Loaded dataset with {len(self.query_target_pairs)} query/target pairs.")
            logging.info(f"Loaded dataset with {len(self.query_target_pairs)} query/target pairs.")

    def extract_query_target_pairs(self, data):
        query_target_pairs = []

        for conversation in data:
            if conversation.get("messages"):
                messages = conversation.get("messages", [])
                for i in range(len(messages) - 1):
                    if messages[i].get("role") == "user" and messages[i + 1].get("role") == "assistant":
                        query = messages[i].get("content") or messages[i].get("value", "")
                        target = messages[i + 1].get("content") or messages[i + 1].get("value", "")
                        tot_target = self.extract_thinking(messages[i + 1])  # Extract reasoning
                        query_target_pairs.append((query.strip(), target.strip(), tot_target))

                    elif messages[i].get("from") == "user" and messages[i + 1].get("from") == "assistant":
                        query = messages[i].get("value", "")
                        target = messages[i + 1].get("value", "")
                        tot_target = self.extract_thinking(messages[i + 1])
                        query_target_pairs.append((query.strip(), target.strip(), tot_target))

            elif conversation.get("conversations"):
                messages = conversation.get("conversations", [])
                for i in range(len(messages) - 1):
                    if messages[i].get("from") == "user" and messages[i + 1].get("from") == "assistant":
                        query = messages[i].get("value", "")
                        target = messages[i + 1].get("value", "")
                        tot_target = self.extract_thinking(messages[i + 1])
                        query_target_pairs.append((query.strip(), target.strip(), tot_target))
                    elif messages[i].get("from") == "human" and messages[i + 1].get("from") == "gpt":
                        query = messages[i].get("value", "")
                        target = messages[i + 1].get("value", "")
                        tot_target = self.extract_thinking(messages[i + 1])
                        query_target_pairs.append((query.strip(), target.strip(), tot_target))
            else:
                user_messages = conversation.get("user", [])
                assistant_messages = conversation.get("assistant", [])
                for i in range(min(len(user_messages), len(assistant_messages))):
                    query = user_messages[i].replace('\n', ' ').strip()
                    target = assistant_messages[i].replace('\n', ' ').strip()
                    tot_target = self.extract_thinking(assistant_messages[i])
                    query_target_pairs.append((query, target, tot_target))

        return query_target_pairs

    def extract_query_target_pairs_json(self, file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
                f.seek(0)  # Reset file pointer
                
                if first_line.startswith('['):
                    data = json.load(f)  # JSON array format
                else:
                    data = [json.loads(line.strip()) for line in f]  # JSONL format

            return self.extract_query_target_pairs(data)

        except Exception as e:
            logging.error(f"Failed to load JSON file: {e}")
            return []

    def extract_thinking(self, assistant_response):
        """
        Extracts reasoning or 'train of thought' from the assistant's response if present.
        """
        tot_target = None
        if isinstance(assistant_response, dict):
            response_text = assistant_response.get("value", "") or assistant_response.get("content", "")
        else:
            response_text = assistant_response  # If it's already a string

        if "<think>" in response_text and "</think>" in response_text:
            tot_target = re.search(r"<think>(.*?)</think>", response_text, re.DOTALL)
        elif "<thinking>" in response_text and "</thinking>" in response_text:
            tot_target = re.search(r"<thinking>(.*?)</thinking>", response_text, re.DOTALL)
        elif "<|begin_of_thought|>" in response_text and "<|end_of_thought|>" in response_text:
            tot_target = re.search(r"<\|begin_of_thought\|>(.*?)<\|end_of_thought\|>", response_text, re.DOTALL)
        elif "longCOT" in assistant_response:
            tot_target = assistant_response["longCOT"]
        elif assistant_response.get("role") == "reasoning":
            tot_target = assistant_response.get("content")
        elif "thinking" in assistant_response:
            tot_target = assistant_response["thinking"]

        return tot_target.group(1).strip() if tot_target else ""  # 🔹 Ensure no `None` values


    def extract_query_target_pairs_parquet(self, file_path):
        try:
            df = pd.read_parquet(file_path)
            query_target_pairs = []

            for _, row in df.iterrows():
                user_query = row.get("question") or row.get("input")
                assistant_response = row.get("answer") or row.get("response")
                tot_target = self.extract_thinking(row)

                if user_query and assistant_response:
                    query_target_pairs.append((user_query.strip(), assistant_response.strip(), tot_target.strip() if tot_target else None))

            return query_target_pairs

        except Exception as e:
            logging.error(f"Failed to load Parquet file: {e}")
            return []
        
    def create_validation_loader(self):
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer else 0
        # If using a chunked dataset for validation
        if self.use_chunked_dataset.get():
            if hasattr(self, 'validation_tokenized_data_path') and self.validation_tokenized_data_path:
                dataset = ChunkedDataset(
                    tokenized_data_path=self.validation_tokenized_data_path,
                    tokenizer=self.tokenizer,
                    max_length=seq_len
                )
            else:
                messagebox.showerror("Error", "No chunked validation data directory selected.")
                return None
        else:
            # Using an unchunked (in-memory) validation dataset
            if not (hasattr(self, 'validation_input_ids') and hasattr(self, 'validation_labels') and hasattr(self, 'validation_labels_tot')):
                messagebox.showerror("Error", "Validation data not loaded. Please load validation data first.")
                return None

            # Convert the lists of token IDs to tensors with proper padding
            input_ids = torch.stack([
                torch.tensor(tokens + [pad_token_id] * (seq_len - len(tokens)), dtype=torch.long, device=self.device)[:seq_len]
                for tokens in self.validation_input_ids
            ])
            labels = torch.stack([
                torch.tensor(tokens + [pad_token_id] * (seq_len - len(tokens)), dtype=torch.long, device=self.device)[:seq_len]
                for tokens in self.validation_labels
            ])
            labels_tot = torch.stack([
                torch.tensor(tokens + [pad_token_id] * (seq_len - len(tokens)), dtype=torch.long, device=self.device)[:seq_len]
                for tokens in self.validation_labels_tot
            ])
            seq_lengths = torch.tensor(
                [min(len(tokens), seq_len) for tokens in self.validation_input_ids],
                dtype=torch.long, device=self.device
            )
            dataset = torch.utils.data.TensorDataset(input_ids, labels, labels_tot, seq_lengths)

        loader = DataLoader(dataset, batch_size=self.batch_size.get(), shuffle=False, collate_fn=collate_fn)
        return loader

        
    def select_validation_dataset(self):
        # Ask the user to select the validation dataset directory
        self.validation_dataset_path = filedialog.askdirectory(title="Select Validation Dataset Directory")
        if not self.validation_dataset_path:
            messagebox.showerror("Error", "No validation dataset directory selected.")
            return

        # Load validation query/target pairs similar to your load_dataset implementation
        dataset_files = os.listdir(self.validation_dataset_path)
        self.validation_query_target_pairs = []
        
        for file in dataset_files:
            file_path = os.path.join(self.validation_dataset_path, file)
            if file.endswith('.json') or file.endswith('.jsonl'):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        if file.endswith('.jsonl'):
                            for line in f:
                                conversation = json.loads(line.strip())
                                self.validation_query_target_pairs.extend(self.extract_query_target_pairs([conversation]))
                        else:
                            data = json.load(f)
                            self.validation_query_target_pairs.extend(self.extract_query_target_pairs(data))
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to load {file}: {str(e)}")
        
        if not self.validation_query_target_pairs:
            messagebox.showerror("Error", "No valid query/target pairs found in the validation dataset.")
            return

        # Tokenize validation data similar to your _generate_training_pairs method
        self.validation_input_ids = []
        self.validation_labels = []
        self.validation_labels_tot = []
        
        # Use the same training mode as set in the GUI (imitation, completion, or response)
        training_mode = self.training_mode.get()
        
        for query, target, tot_target in self.validation_query_target_pairs:
            input_ids, labels, labels_tot = self._generate_training_pairs(query, target, tot_target, training_mode)
            self.validation_input_ids.append(input_ids)
            self.validation_labels.append(labels)
            self.validation_labels_tot.append(labels_tot)
        
        # Create tensors with proper padding
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer else 0
        input_ids_tensor = torch.stack([
            torch.tensor(ids + [pad_token_id] * (seq_len - len(ids)), dtype=torch.long, device=self.device)[:seq_len]
            for ids in self.validation_input_ids
        ])
        labels_tensor = torch.stack([
            torch.tensor(ids + [pad_token_id] * (seq_len - len(ids)), dtype=torch.long, device=self.device)[:seq_len]
            for ids in self.validation_labels
        ])
        labels_tot_tensor = torch.stack([
            torch.tensor(ids + [pad_token_id] * (seq_len - len(ids)), dtype=torch.long, device=self.device)[:seq_len]
            for ids in self.validation_labels_tot
        ])
        seq_lengths_tensor = torch.tensor(
            [min(len(ids), seq_len) for ids in self.validation_input_ids],
            dtype=torch.long, device=self.device
        )
        
        # Create a TensorDataset and DataLoader for validation
        validation_dataset = torch.utils.data.TensorDataset(input_ids_tensor, labels_tensor, labels_tot_tensor, seq_lengths_tensor)
        self.validation_loader = DataLoader(validation_dataset, batch_size=self.batch_size.get(), shuffle=False, collate_fn=collate_fn)
        
        messagebox.showinfo("Success", f"Validation dataset loaded with {len(validation_dataset)} samples.")


    def run_validation(self, validation_loader, loss_fn):
        self.model.eval()
        total_val_loss = 0.0
        num_batches = 0
        with torch.no_grad():
            for batch_input_ids, batch_labels, batch_labels_tot, seq_lengths in validation_loader:
                batch_input_ids = batch_input_ids.to(self.device)
                batch_labels = batch_labels.to(self.device)
                # Adjust the forward call as needed if your model requires three inputs.
                outputs, _, _ = self.model(batch_input_ids, batch_labels.reshape(-1), batch_labels_tot.reshape(-1))
                # Flatten outputs and targets for loss calculation
                logits = outputs.reshape(-1, outputs.size(-1))
                targets = batch_labels.reshape(-1)
                loss = loss_fn(logits, targets)
                total_val_loss += loss.item()
                num_batches += 1
        avg_val_loss = total_val_loss / num_batches if num_batches > 0 else float('inf')
        logging.info(f"Validation Loss: {avg_val_loss:.4f}")
        self.model.train()
        return avg_val_loss

    def run_validation_button(self):
        if not self.model:
            messagebox.showerror("Error", "Model not loaded.")
            return
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer else 0
        loss_fn = nn.CrossEntropyLoss(ignore_index=pad_token_id)
        validation_loader = self.create_validation_loader()
        if validation_loader is None:
            return
        val_loss = self.run_validation(validation_loader, loss_fn)
        messagebox.showinfo("Validation", f"Validation Loss: {val_loss:.4f}")


# Main application entry point
if __name__ == "__main__":
    root = tk.Tk()


    app = ReasoningModelGUI(root)
    root.mainloop()
