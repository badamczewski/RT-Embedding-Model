"""
Test script to demonstrate the logging functionality
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

# Create the model
model = EmbeddingModel(
    vocab_size=vocab_size,
    embedding_dim=embedding_dim,
    context_window=context_window
)

# Create test input
batch_size = 2
token_ids = torch.randint(1, vocab_size, (batch_size, 10))  # Shorter than context_window

print("=" * 60)
print("Test 1: Logging DISABLED (default)")
print("=" * 60)
with torch.no_grad():
    embeddings = model(token_ids)
print(f"Output shape: {embeddings.shape}\n")

print("=" * 60)
print("Test 2: Logging ENABLED - Shapes only")
print("=" * 60)
model.logger.enable(categories={'shapes'})
with torch.no_grad():
    embeddings = model(token_ids)
print(f"Output shape: {embeddings.shape}\n")

print("=" * 60)
print("Test 3: Logging ENABLED - Statistics only")
print("=" * 60)
model.logger.enable(categories={'statistics'})
with torch.no_grad():
    embeddings = model(token_ids)
print(f"Output shape: {embeddings.shape}\n")

print("=" * 60)
print("Test 4: Logging ENABLED - Shapes + Statistics")
print("=" * 60)
model.logger.enable(categories={'shapes', 'statistics'})
with torch.no_grad():
    embeddings = model(token_ids)
print(f"Output shape: {embeddings.shape}\n")

print("=" * 60)
print("Test 5: Logging ENABLED - All categories")
print("=" * 60)
model.logger.enable(categories={'all'})
with torch.no_grad():
    embeddings = model(token_ids)
print(f"Output shape: {embeddings.shape}\n")

print("=" * 60)
print("Test 6: Context manager (temporary logging)")
print("=" * 60)
model.logger.disable()  # Disable first
with model.logger.enabled(categories={'shapes'}):
    with torch.no_grad():
        embeddings = model(token_ids)
print(f"Output shape: {embeddings.shape}\n")

print("=" * 60)
print("Test 7: Logging DISABLED again")
print("=" * 60)
with torch.no_grad():
    embeddings = model(token_ids)
print(f"Output shape: {embeddings.shape}\n")

print("All tests completed successfully!")

