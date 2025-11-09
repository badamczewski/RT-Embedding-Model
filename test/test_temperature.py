"""
Test script to demonstrate temperature scaling functionality
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from model import EmbeddingModel

# Model parameters
vocab_size = 10000
embedding_dim = 256
context_window = 16

print("=" * 60)
print("Test 1: Model with temperature scaling (default)")
print("=" * 60)
model_with_temp = EmbeddingModel(
    vocab_size=vocab_size,
    embedding_dim=embedding_dim,
    context_window=context_window,
    use_temperature=True,
    temperature_init=1.0
)

print(f"Initial temperature: {model_with_temp.temperature.item():.4f}")
print(f"Temperature is learnable: {model_with_temp.log_temperature.requires_grad}")

# Create test input
batch_size = 2
token_ids = torch.randint(1, vocab_size, (batch_size, 10))

with torch.no_grad():
    embeddings = model_with_temp(token_ids)

print(f"Output embeddings shape: {embeddings.shape}")
print(f"Embeddings L2 norm (should be ~1.0): {embeddings.norm(dim=1)}")
print()

print("=" * 60)
print("Test 2: Model WITHOUT temperature scaling")
print("=" * 60)
model_no_temp = EmbeddingModel(
    vocab_size=vocab_size,
    embedding_dim=embedding_dim,
    context_window=context_window,
    use_temperature=False
)

with torch.no_grad():
    embeddings_no_temp = model_no_temp(token_ids)

print(f"Output embeddings shape: {embeddings_no_temp.shape}")
print(f"Embeddings L2 norm (should be ~1.0): {embeddings_no_temp.norm(dim=1)}")
print()

print("=" * 60)
print("Test 3: Compare embeddings with and without temperature")
print("=" * 60)
# Compute cosine similarity between the two models' outputs
cosine_sim = torch.nn.functional.cosine_similarity(embeddings, embeddings_no_temp, dim=1)
print(f"Cosine similarity between temp and no-temp embeddings: {cosine_sim}")
print()

print("=" * 60)
print("Test 4: Temperature with different initial values")
print("=" * 60)
for init_temp in [0.1, 0.5, 1.0, 2.0]:
    model = EmbeddingModel(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        context_window=context_window,
        use_temperature=True,
        temperature_init=init_temp
    )
    print(f"Initial temperature: {init_temp:.1f} -> Actual temperature: {model.temperature.item():.4f}")

print()
print("=" * 60)
print("Test 5: Temperature scaling effect on similarity")
print("=" * 60)
# Create two similar sequences
token_ids1 = torch.randint(1, vocab_size, (1, 10))
token_ids2 = token_ids1.clone()  # Same sequence

model_temp = EmbeddingModel(
    vocab_size=vocab_size,
    embedding_dim=embedding_dim,
    context_window=context_window,
    use_temperature=True,
    temperature_init=0.1  # Low temperature = sharper
)

model_no_temp = EmbeddingModel(
    vocab_size=vocab_size,
    embedding_dim=embedding_dim,
    context_window=context_window,
    use_temperature=False
)

with torch.no_grad():
    emb1_temp = model_temp(token_ids1)
    emb2_temp = model_temp(token_ids2)
    emb1_no_temp = model_no_temp(token_ids1)
    emb2_no_temp = model_no_temp(token_ids2)

sim_temp = torch.nn.functional.cosine_similarity(emb1_temp, emb2_temp, dim=1).item()
sim_no_temp = torch.nn.functional.cosine_similarity(emb1_no_temp, emb2_no_temp, dim=1).item()

print(f"Same sequence similarity (with temp=0.1): {sim_temp:.4f}")
print(f"Same sequence similarity (no temp): {sim_no_temp:.4f}")
print(f"Temperature makes similarity: {'sharper' if sim_temp > sim_no_temp else 'smoother'}")

print()
print("All temperature tests completed successfully!")

