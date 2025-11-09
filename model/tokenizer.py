import sentencepiece as spm
from typing import List, Optional, Union
import os


class SentencePieceTokenizer:
    """
    SentencePiece tokenizer wrapper for the embedding model.
    
    Supports training new tokenizers and loading existing models.
    Uses BPE (Byte Pair Encoding) by default for subword tokenization.
    """
    
    def __init__(self, model_path: Optional[str] = None, vocab_size: int = 10000):
        """
        Initialize tokenizer.
        
        Args:
            model_path: Path to trained SentencePiece model (.model file)
            vocab_size: Vocabulary size (used when training new model, ignored if model_path provided)
        """
        self.model_path = model_path
        self.vocab_size = vocab_size
        self.sp = None
        
        if model_path and os.path.exists(model_path):
            self.sp = spm.SentencePieceProcessor()
            self.sp.load(model_path)
            self.vocab_size = len(self.sp)
        elif model_path:
            raise ValueError(f"SentencePiece model not found at {model_path}")
    
    def train(self, input_file: str, output_prefix: str, 
              vocab_size: int = 10000, 
              model_type: str = 'bpe',
              character_coverage: float = 0.9995,
              **kwargs):
        """
        Train a new SentencePiece model from a text corpus.
        
        Args:
            input_file: Path to training text file (one sentence per line)
            output_prefix: Output prefix for .model and .vocab files
            vocab_size: Size of vocabulary to learn
            model_type: 'bpe' (Byte Pair Encoding) or 'unigram'
            character_coverage: Character coverage (0.9995 for most languages, 0.98 for CJK)
            **kwargs: Additional SentencePiece training parameters
        """
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Training file not found: {input_file}")
        
        # Ensure output directory exists
        output_dir = os.path.dirname(output_prefix) if os.path.dirname(output_prefix) else '.'
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        # Default training parameters
        # Add MASK token as user-defined symbol (will be assigned ID 4)
        train_params = {
            'input': input_file,
            'model_prefix': output_prefix,
            'vocab_size': vocab_size,
            'model_type': model_type,
            'character_coverage': character_coverage,
            'user_defined_symbols': ['[MASK]'],  # Add MASK token to vocabulary
            'pad_id': 0,  # Padding token
            'unk_id': 1,  # Unknown token
            'bos_id': 2,  # Beginning of sentence
            'eos_id': 3,  # End of sentence
        }
        
        # Override with any additional kwargs
        train_params.update(kwargs)
        
        # Train the model
        spm.SentencePieceTrainer.train(**train_params)
        
        # Load the trained model
        self.model_path = f"{output_prefix}.model"
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(self.model_path)
        self.vocab_size = len(self.sp)
    
    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        """
        Encode text to token IDs.
        
        Args:
            text: Input text string
            add_bos: Add beginning-of-sentence token
            add_eos: Add end-of-sentence token
        
        Returns:
            List of token IDs
        """
        if self.sp is None:
            raise ValueError("Tokenizer not initialized. Load a model or train one first.")
        return self.sp.encode(text, out_type=int, add_bos=add_bos, add_eos=add_eos)
    
    def encode_batch(self, texts: List[str], add_bos: bool = False, add_eos: bool = False) -> List[List[int]]:
        """
        Encode batch of texts to token IDs.
        
        Args:
            texts: List of input text strings
            add_bos: Add beginning-of-sentence token
            add_eos: Add end-of-sentence token
        
        Returns:
            List of lists of token IDs
        """
        if self.sp is None:
            raise ValueError("Tokenizer not initialized. Load a model or train one first.")
        return self.sp.encode(texts, out_type=int, add_bos=add_bos, add_eos=add_eos)
    
    def decode(self, ids: Union[List[int], List[List[int]]]) -> str:
        """
        Decode token IDs to text.
        
        Args:
            ids: Token IDs (list of ints or list of lists)
        
        Returns:
            Decoded text string
        """
        if self.sp is None:
            raise ValueError("Tokenizer not initialized. Load a model or train one first.")
        return self.sp.decode(ids)
    
    def decode_batch(self, ids_list: List[List[int]]) -> List[str]:
        """
        Decode batch of token IDs to texts.
        
        Args:
            ids_list: List of token ID sequences
        
        Returns:
            List of decoded text strings
        """
        if self.sp is None:
            raise ValueError("Tokenizer not initialized. Load a model or train one first.")
        return self.sp.decode(ids_list)
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into subword tokens.
        
        Args:
            text: Input text string
        
        Returns:
            List of subword token strings
        """
        if self.sp is None:
            raise ValueError("Tokenizer not initialized. Load a model or train one first.")
        return self.sp.encode(text, out_type=str)
    
    def __len__(self):
        """Return vocabulary size."""
        return len(self.sp) if self.sp else self.vocab_size
    
    @property
    def pad_id(self) -> int:
        """Get padding token ID."""
        return self.sp.pad_id() if self.sp else 0
    
    @property
    def unk_id(self) -> int:
        """Get unknown token ID."""
        return self.sp.unk_id() if self.sp else 1
    
    @property
    def bos_id(self) -> int:
        """Get beginning of sentence token ID."""
        return self.sp.bos_id() if self.sp else 2
    
    @property
    def eos_id(self) -> int:
        """Get end of sentence token ID."""
        return self.sp.eos_id() if self.sp else 3
    
    @property
    def mask_id(self) -> int:
        """
        Get mask token ID. 
        
        Returns the ID of [MASK] token if it exists in the vocabulary.
        If the tokenizer was trained without [MASK], this will raise a warning
        and use a fallback approach.
        """
        if self.sp is None:
            return 4  # Default reserved ID for MASK
        
        # Try to get MASK token ID
        try:
            mask_token_id = self.sp.piece_to_id('[MASK]')
            unk_id = self.sp.unk_id()
            
            # Check if [MASK] actually exists (not falling back to UNK)
            if mask_token_id != unk_id:
                return mask_token_id
            else:
                # [MASK] doesn't exist, it's falling back to UNK
                # This means the tokenizer was trained without [MASK]
                # We'll use a reserved token ID approach
                # Note: This is not ideal - the tokenizer should be retrained with [MASK]
                import warnings
                warnings.warn(
                    "[MASK] token not found in vocabulary. "
                    "The tokenizer should be retrained with [MASK] token for proper MLM. "
                    "Using reserved token ID 4 as fallback.",
                    UserWarning
                )
                # Use ID 4 as fallback (assuming it's not a common token)
                # This is a workaround - ideally retrain tokenizer with [MASK]
                return 4
        except Exception:
            # If piece_to_id fails, use fallback
            return 4

