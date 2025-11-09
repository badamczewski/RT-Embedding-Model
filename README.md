# RT Embedding Model

A PyTorch implementation of an embedding model with RoPE (Rotary Position Embedding), temperature scaling, and Masked Language Modeling (MLM) pre-training support. This is a research and learning project to test out various ideas about embedding models and how a good Machine Learning project should look like. 

## Features

- **RoPE-based Transformer**: Uses Rotary Position Embedding instead of learned positional embeddings
- **Temperature Scaling**: Learnable temperature parameter for better similarity search
- **MLM Pre-training**: Support for Masked Language Modeling pre-training
- **SentencePiece Tokenization**: Subword tokenization with SentencePiece
- **Selective Logging**: Configurable logging system using Loguru
- **Attention Masking**: Proper handling of variable-length sequences

## Project Structure

```
ML_Embeddings/
├── model/                 # Model implementation
│   ├── embedding.py      # Main EmbeddingModel class
│   ├── rope.py           # RoPE implementation
│   ├── tokenizer.py      # SentencePiece tokenizer wrapper
│   ├── logger.py         # ModelLogger for debugging
│   └── training_logger.py # TrainingLogger for training loop
├── scripts/              # Training scripts
│   ├── train_tokenizer.py # Train SentencePiece tokenizer
│   └── train_mlm.py      # MLM pre-training script
├── test/                 # Tests
│   ├── model_dry_test.py # Model functionality tests
│   ├── test_logger.py    # Logger tests
│   ├── test_tokenizer.py # Tokenizer tests
│   └── test_mlm_training.py # MLM training tests
├── data/                 # Training data (not in git)
├── models/               # Trained models and checkpoints (not in git)
├── logs/                 # Training logs (not in git)
└── requirements.txt     # Python dependencies
```

## Installation

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Train a Tokenizer

```bash
python scripts/train_tokenizer.py \
    --input data/corpus.txt \
    --output models/tokenizer \
    --vocab_size 20000
```

This will create:
- `models/tokenizer.model` - The tokenizer model
- `models/tokenizer.vocab` - Vocabulary file

**Note**: The tokenizer will automatically include a `[MASK]` token for MLM training.

### 2. Pre-train the Model (MLM)

```bash
python scripts/train_mlm.py \
    --tokenizer models/tokenizer.model \
    --corpus data/corpus.txt \
    --output_dir models/checkpoints \
    --batch_size 32 \
    --learning_rate 5e-4 \
    --epochs 5 \
    --gradient_accumulation_steps 1
```

### 4. Contrastive Learning

TODO - Not yet implemented

### 5. Use the Model

```python
from model import EmbeddingModel, SentencePieceTokenizer
import torch

# Load tokenizer
tokenizer = SentencePieceTokenizer(model_path="models/tokenizer.model")

# Create model (for embeddings, not MLM)
model = EmbeddingModel(
    vocab_size=len(tokenizer),
    embedding_dim=256,
    context_window=16,
    is_masked_language=False  # Set to False for embedding mode
)

# Encode text
text = "Your text here"
token_ids = tokenizer.encode(text)

# Pad/truncate to context_window
# ... (handle padding)

# Get embeddings
with torch.no_grad():
    embeddings = model(token_ids)
```

## Model Architecture

- **Embedding Dimension**: 256 (configurable)
- **Context Window**: 16 tokens (configurable)
- **Hidden Dimension**: 512 (configurable)
- **Transformer Layers**: 6 encoder layers
- **Attention Heads**: 8
- **Positional Encoding**: RoPE (Rotary Position Embedding)
- **Pooling**: Masked mean pooling
- **Normalization**: L2 normalization + optional temperature scaling

## Configuration

### Model Parameters

- `vocab_size`: Vocabulary size (must match tokenizer)
- `embedding_dim`: Output embedding dimension (default: 256)
- `context_window`: Maximum sequence length (default: 16)
- `hidden_dim`: Hidden dimension (default: 512)
- `is_masked_language`: Enable MLM head for pre-training (default: False)
- `use_temperature`: Enable temperature scaling (default: True)
- `temperature_init`: Initial temperature value (default: 1.0)

### Training Parameters

- `batch_size`: Batch size (default: 32)
- `learning_rate`: Learning rate (default: 5e-4)
- `weight_decay`: Weight decay for AdamW (default: 0.01)
- `epochs`: Number of training epochs (default: 5)
- `gradient_accumulation_steps`: Gradient accumulation (default: 1)
- `validation_split`: Validation split ratio (default: 0.1)
- `mask_prob`: MLM masking probability (default: 0.15)

## MLM Pre-training

The model supports BERT-style Masked Language Modeling:

- **Masking Strategy**: 15% of tokens are selected
  - 80% → Replaced with `[MASK]` token
  - 10% → Replaced with random token
  - 10% → Kept unchanged

## Logging

### Model Logger (for debugging)

```python
# Enable selective logging
model.logger.enable(categories={'shapes', 'statistics'})

# Or use context manager
with model.logger.enabled(categories={'shapes'}):
    output = model(input)

# Disable logging
model.logger.disable()
```

Available categories: `shapes`, `statistics`, `values`, `info`, `all`

### Training Logger

Training logs are automatically saved to `logs/` directory with timestamps.

## Special Tokens

- `PAD`: ID 0 (padding)
- `UNK`: ID 1 (unknown)
- `BOS`: ID 2 (beginning of sentence)
- `EOS`: ID 3 (end of sentence)
- `MASK`: ID varies but should be 4 by default (added during tokenizer training)

## Requirements

- Python 3.8+
- PyTorch >= 2.0.0
- NumPy >= 1.24.0
- Loguru >= 0.7.0
- SentencePiece >= 0.1.99

## Training

The model was trained using wikipedia corpus where each sentence is a seperate line (~7 milion sentences).

## License

Apache 2.0

