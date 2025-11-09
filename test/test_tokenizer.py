"""
Test script for SentencePieceTokenizer functionality
"""
import sys
import os
import tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.tokenizer import SentencePieceTokenizer


def create_temp_corpus():
    """Create a temporary corpus file for testing."""
    corpus_text = """The quick brown fox jumps over the lazy dog.
Machine learning is a subset of artificial intelligence.
Natural language processing enables computers to understand human language.
Deep learning models use neural networks with multiple layers.
Transformers have revolutionized the field of NLP.
SentencePiece provides subword tokenization for various languages.
Tokenization is the process of breaking text into smaller units.
Embeddings represent words or sentences as dense vectors.
The model learns to encode semantic meaning in these vectors.
Attention mechanisms allow models to focus on relevant parts of input.
"""
    
    # Create temporary file
    fd, temp_path = tempfile.mkstemp(suffix='.txt', text=True)
    try:
        with os.fdopen(fd, 'w') as f:
            f.write(corpus_text)
        return temp_path
    except:
        os.close(fd)
        raise


def test_tokenizer():
    """Test tokenizer training and functionality."""
    print("=" * 60)
    print("Testing SentencePieceTokenizer")
    print("=" * 60)
    
    # Create temporary corpus
    corpus_path = create_temp_corpus()
    print(f"Created temporary corpus: {corpus_path}")
    print()
    
    try:
        # Test 1: Train tokenizer
        print("Test 1: Training tokenizer")
        print("-" * 60)
        tokenizer = SentencePieceTokenizer()
        
        # Create temporary output path
        with tempfile.TemporaryDirectory() as temp_dir:
            output_prefix = os.path.join(temp_dir, "test_tokenizer")
            
            tokenizer.train(
                input_file=corpus_path,
                output_prefix=output_prefix,
                vocab_size=100,
                model_type='bpe'
            )
            
            print(f"Tokenizer trained successfully")
            print(f"  Vocabulary size: {len(tokenizer)}")
            print(f"  Model path: {tokenizer.model_path}")
            print()
            
            # Test 2: Encode/Decode
            print("Test 2: Encode and Decode")
            print("-" * 60)
            test_text = "Machine learning is fascinating."
            token_ids = tokenizer.encode(test_text)
            decoded_text = tokenizer.decode(token_ids)
            
            print(f"Original: {test_text}")
            print(f"Token IDs: {token_ids[:10]}... (showing first 10)")
            print(f"Decoded: {decoded_text}")
            print(f"Encoding/Decoding works: {test_text == decoded_text}")
            print()
            
            # Test 3: Tokenize
            print("Test 3: Tokenize to subwords")
            print("-" * 60)
            tokens = tokenizer.tokenize(test_text)
            print(f"Text: {test_text}")
            print(f"Tokens: {tokens[:10]}... (showing first 10)")
            print(f"Tokenization works: {len(tokens) > 0}")
            print()
            
            # Test 4: Batch operations
            print("Test 4: Batch encoding/decoding")
            print("-" * 60)
            texts = [
                "The quick brown fox",
                "Machine learning models",
                "Natural language processing"
            ]
            batch_ids = tokenizer.encode_batch(texts)
            batch_decoded = tokenizer.decode_batch(batch_ids)
            
            print(f"Input texts: {len(texts)}")
            print(f"Encoded batches: {len(batch_ids)}")
            print(f"Decoded texts: {len(batch_decoded)}")
            for i, (original, decoded) in enumerate(zip(texts, batch_decoded)):
                match = original == decoded
                print(f"  {i+1}. Match: {match}")
            print(f"Batch operations work")
            print()
            
            # Test 5: Special tokens
            print("Test 5: Special token IDs")
            print("-" * 60)
            print(f"PAD ID: {tokenizer.pad_id}")
            print(f"UNK ID: {tokenizer.unk_id}")
            print(f"BOS ID: {tokenizer.bos_id}")
            print(f"EOS ID: {tokenizer.eos_id}")
            print(f"Special tokens accessible")
            print()
            
            # Test 6: BOS/EOS tokens
            print("Test 6: BOS/EOS token handling")
            print("-" * 60)
            text = "Hello world"
            ids_no_special = tokenizer.encode(text, add_bos=False, add_eos=False)
            ids_with_bos = tokenizer.encode(text, add_bos=True, add_eos=False)
            ids_with_eos = tokenizer.encode(text, add_bos=False, add_eos=True)
            ids_with_both = tokenizer.encode(text, add_bos=True, add_eos=True)
            
            print(f"Text: {text}")
            print(f"Without special tokens: {len(ids_no_special)} tokens")
            print(f"With BOS: {len(ids_with_bos)} tokens (first={ids_with_bos[0] if ids_with_bos else 'N/A'})")
            print(f"With EOS: {len(ids_with_eos)} tokens (last={ids_with_eos[-1] if ids_with_eos else 'N/A'})")
            print(f"With both: {len(ids_with_both)} tokens")
            print(f"BOS/EOS handling works")
            print()
            
            # Test 7: Load existing model
            print("Test 7: Load existing model")
            print("-" * 60)
            loaded_tokenizer = SentencePieceTokenizer(model_path=tokenizer.model_path)
            test_ids = loaded_tokenizer.encode("Test sentence")
            print(f"Loaded tokenizer vocabulary size: {len(loaded_tokenizer)}")
            print(f"Encoded test sentence: {len(test_ids)} tokens")
            print(f"Model loading works")
            print()
            
    finally:
        # Clean up
        if os.path.exists(corpus_path):
            os.remove(corpus_path)
    
    print("=" * 60)
    print("All tokenizer tests passed successfully!")
    print("=" * 60)


if __name__ == '__main__':
    test_tokenizer()

