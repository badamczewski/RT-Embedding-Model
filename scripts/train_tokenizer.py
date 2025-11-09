"""
Script to train a SentencePiece tokenizer from a text corpus.

Usage:
    python scripts/train_tokenizer.py --input data/corpus.txt --output models/tokenizer --vocab_size 10000
"""

import argparse
import sys
import os

# Add parent directory to path to import model
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.tokenizer import SentencePieceTokenizer


def main():
    parser = argparse.ArgumentParser(
        description='Train a SentencePiece tokenizer from a text corpus',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to input text file (one sentence per line)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output prefix for model files (will create .model and .vocab files)'
    )
    
    parser.add_argument(
        '--vocab_size',
        type=int,
        default=10000,
        help='Vocabulary size'
    )
    
    parser.add_argument(
        '--model_type',
        type=str,
        default='bpe',
        choices=['bpe', 'unigram'],
        help='SentencePiece model type: bpe (Byte Pair Encoding) or unigram'
    )
    
    parser.add_argument(
        '--character_coverage',
        type=float,
        default=0.9995,
        help='Character coverage (0.9995 for most languages, 0.98 for CJK)'
    )
    
    args = parser.parse_args()
    
    # Validate input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output) if os.path.dirname(args.output) else '.'
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created output directory: {output_dir}")
    
    print(f"Training SentencePiece tokenizer...")
    print(f"  Input file: {args.input}")
    print(f"  Output prefix: {args.output}")
    print(f"  Vocabulary size: {args.vocab_size}")
    print(f"  Model type: {args.model_type}")
    print(f"  Character coverage: {args.character_coverage}")
    print()
    
    # Initialize tokenizer and train
    tokenizer = SentencePieceTokenizer()
    
    try:
        tokenizer.train(
            input_file=args.input,
            output_prefix=args.output,
            vocab_size=args.vocab_size,
            model_type=args.model_type,
            character_coverage=args.character_coverage
        )
        
        print(f"Tokenizer training completed successfully!")
        print(f"  Model saved to: {args.output}.model")
        print(f"  Vocabulary saved to: {args.output}.vocab")
        print(f"  Actual vocabulary size: {len(tokenizer)}")
        print()
        print("You can now load the tokenizer with:")
        print(f"  tokenizer = SentencePieceTokenizer(model_path='{args.output}.model')")
        
    except Exception as e:
        print(f"Error during training: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()

