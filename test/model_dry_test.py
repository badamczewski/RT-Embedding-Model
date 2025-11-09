"""
Example usage of the EmbeddingModel with attention masking
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from model import EmbeddingModel

# Model parameters
vocab_size = 10000  # Size of your vocabulary
embedding_dim = 256  # Output embedding dimension
context_window = 16  # Number of tokens in context window (matches model default)
padding_token_id = 0  # Token ID used for padding

# Create the model
model = EmbeddingModel(
    vocab_size=vocab_size,
    embedding_dim=embedding_dim,
    context_window=context_window
)

# Example 1: Full-length sequences (no padding needed)
print("Example 1: Full-length sequences")
batch_size = 4
token_ids = torch.randint(1, vocab_size, (batch_size, context_window))  # Start from 1 to avoid padding token

with torch.no_grad():
    embeddings = model(token_ids)

print(f"Input shape: {token_ids.shape}")
print(f"Output embedding shape: {embeddings.shape}")
print()

# Example 2: Variable length sequences with automatic padding
print("Example 2: Variable length sequences (auto-padding)")
# Create sequences of different lengths
seq_lengths = [8, 12, 5, 10]
token_sequences = []
for length in seq_lengths:
    seq = torch.randint(1, vocab_size, (length,))  # Start from 1 to avoid padding token
    token_sequences.append(seq)

# Pad to same length manually for batching (or let model handle it)
max_len = max(seq_lengths)
padded_sequences = []
for seq in token_sequences:
    if len(seq) < max_len:
        padding = torch.full((max_len - len(seq),), padding_token_id, dtype=seq.dtype)
        padded_seq = torch.cat([seq, padding])
    else:
        padded_seq = seq
    padded_sequences.append(padded_seq)

token_ids_batch = torch.stack(padded_sequences)

with torch.no_grad():
    embeddings = model(token_ids_batch, padding_token_id=padding_token_id)

print(f"Input shape: {token_ids_batch.shape}")
print(f"Sequence lengths: {seq_lengths}")
print(f"Output embedding shape: {embeddings.shape}")
print()

# Example 3: Manual attention mask
print("Example 3: Manual attention mask")
token_ids = torch.randint(1, vocab_size, (batch_size, context_window))
# Create custom mask (e.g., mask out some positions)
attention_mask = torch.zeros(batch_size, context_window, dtype=torch.bool)
attention_mask[:, -3:] = True  # Mask last 3 tokens

with torch.no_grad():
    embeddings = model(token_ids, attention_mask=attention_mask)

print(f"Input shape: {token_ids.shape}")
print(f"Attention mask shape: {attention_mask.shape}")
print(f"Output embedding shape: {embeddings.shape}")
print()

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print()

# Create a masked language model (MLM) task
print("Example 4: Masked language model (MLM)")
mlm_model = EmbeddingModel(
    vocab_size=vocab_size,
    embedding_dim=embedding_dim,
    context_window=context_window,
    is_masked_language=True  # Enable MLM mode
)
token_ids = torch.randint(1, vocab_size, (batch_size, context_window))
attention_mask = torch.zeros(batch_size, context_window, dtype=torch.bool)
attention_mask[:, -3:] = True  # Mask last 3 tokens

with torch.no_grad():
    logits = mlm_model(token_ids, attention_mask=attention_mask)

print(f"Input shape: {token_ids.shape}")
print(f"Attention mask shape: {attention_mask.shape}")
print(f"Output logits shape: {logits.shape}")

