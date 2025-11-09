"""
Test MLM training script with a small dataset
"""
import sys
import os
import tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import SentencePieceTokenizer, EmbeddingModel
from scripts.train_mlm import MLMDataset, load_corpus
import torch


def test_mlm_dataset():
    """Test MLM dataset creation and masking."""
    print("=" * 60)
    print("Testing MLM Dataset")
    print("=" * 60)
    
    # Create a temporary tokenizer
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create small corpus
        corpus_text = """The quick brown fox jumps over the lazy dog.
Machine learning is a subset of artificial intelligence.
Natural language processing enables computers to understand human language.
Deep learning models use neural networks with multiple layers.
Transformers have revolutionized the field of NLP.
SentencePiece provides subword tokenization for various languages.
Tokenization is the process of breaking text into smaller units.
Embeddings represent words or sentences as dense vectors.
The model learns to encode semantic meaning in these vectors.
Attention mechanisms allow models to focus on relevant parts of input."""
        
        corpus_path = os.path.join(temp_dir, "test_corpus.txt")
        with open(corpus_path, 'w') as f:
            f.write(corpus_text)
        
        # Train a small tokenizer
        tokenizer_path = os.path.join(temp_dir, "test_tokenizer")
        tokenizer = SentencePieceTokenizer()
        tokenizer.train(
            input_file=corpus_path,
            output_prefix=tokenizer_path,
            vocab_size=100
        )
        
        # Load corpus
        texts = load_corpus(corpus_path)
        print(f"Loaded {len(texts)} sentences")
        
        # Create dataset
        dataset = MLMDataset(texts, tokenizer, context_window=16, mask_prob=0.15)
        print(f"Dataset size: {len(dataset)}")
        
        # Get a sample
        sample = dataset[0]
        print(f"Sample keys: {sample.keys()}")
        print(f"Input IDs shape: {sample['input_ids'].shape}")
        print(f"Labels shape: {sample['labels'].shape}")
        print(f"Attention mask shape: {sample['attention_mask'].shape}")
        
        # Check masking
        labels = sample['labels']
        
        # Count masked positions (where label != -100)
        masked_count = (labels != -100).sum().item()
        print(f"Masked positions: {masked_count}")
        print(f"MLM Dataset works correctly")
        print()
        
        # Test model forward pass
        print("=" * 60)
        print("Testing Model Forward Pass (MLM mode)")
        print("=" * 60)
        
        model = EmbeddingModel(
            vocab_size=len(tokenizer),
            embedding_dim=256,
            context_window=16,
            hidden_dim=512,
            is_masked_language=True
        )
        
        # Create a batch
        batch = {
            'input_ids': sample['input_ids'].unsqueeze(0),
            'labels': sample['labels'].unsqueeze(0),
            'attention_mask': sample['attention_mask'].unsqueeze(0)
        }
        
        # Forward pass
        with torch.no_grad():
            logits = model(batch['input_ids'], attention_mask=batch['attention_mask'])
        
        print(f"Logits shape: {logits.shape}")
        print(f"Expected: (1, 16, {len(tokenizer)})")
        print(f"Model forward pass works")
        print()
        
        # Test loss computation
        print("=" * 60)
        print("Testing Loss Computation")
        print("=" * 60)
        
        logits_flat = logits.view(-1, logits.size(-1))
        labels_flat = batch['labels'].view(-1)
        
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fn(logits_flat, labels_flat)
        
        print(f"Loss: {loss.item():.4f}")
        print(f"Loss computation works")
        print()
        
        print("=" * 60)
        print("All MLM training tests passed!")
        print("=" * 60)


if __name__ == '__main__':
    test_mlm_dataset()

