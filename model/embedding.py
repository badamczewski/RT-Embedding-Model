import torch
import torch.nn as nn
from .rope import RoPE
from .logger import ModelLogger


class EmbeddingModel(nn.Module):
    """
    Encoder-only transformer model for text embeddings with external non attention based RoPE and temperature scaling, 
    with optional masked language modeling (MLM) head for pre-training.
    RoPE is applied outside of the attention block (for science ofc :D).
    
    Args:
        vocab_size: Size of the vocabulary
        embedding_dim: Dimension of the output embeddings (default: 256).
        context_window: Number of tokens in the context window (default: 16).
        hidden_dim: Hidden dimension for the model (default: 512).
        is_masked_language: Whether to include a masked language modeling (MLM) head for pre-training (default: False).
        use_temperature: Whether to apply learnable temperature scaling to embeddings (default: True).
        temperature_init: Initial value for temperature parameter (default: 1.0). Temperature is learned during training.
    """
    
    def __init__(self, vocab_size, embedding_dim=256, context_window=16, hidden_dim=512, is_masked_language=False, use_temperature=True, temperature_init=1.0):
        super(EmbeddingModel, self).__init__()
        
        self.is_masked_language = is_masked_language
        self.embedding_dim = embedding_dim
        self.context_window = context_window
        self.vocab_size = vocab_size
        self.use_temperature = use_temperature
        
        # Initialize logger
        self.logger = ModelLogger(model_name="EmbeddingModel")
        
        # Token embedding layer
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        
        # Rotary Position Embedding (RoPE)
        self.rope = RoPE(dim=hidden_dim, max_seq_len=context_window)
        
        # Transformer encoder for processing the sequence
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 2,
            dropout=0.1,
            batch_first=True
        )
        # Add layer normalization to the transformer encoder (to improve stability)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6, norm=nn.LayerNorm(hidden_dim))
        
        # Projection layer to output embedding dimension
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )

        # Language model head for masked language modeling (MLM) tasks.
        # This should be only used in pre-training.
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        
        # Temperature scaling parameter (learnable)
        # Temperature controls the sharpness of similarity distributions
        # Lower temperature (< 1.0) = sharper, emphasizes high similarities
        # Higher temperature (> 1.0) = smoother, reduces differences
        if use_temperature:
            # Use log-space parameter to ensure temperature is always positive
            # temperature = exp(log_temperature), so we learn log_temperature
            self.log_temperature = nn.Parameter(torch.log(torch.tensor(temperature_init, dtype=torch.float32)))
        else:
            self.register_buffer('log_temperature', torch.tensor(0.0))  # log(1.0) = 0.0 (no scaling)
    
    @property
    def temperature(self):
        """Get the current temperature value (always positive)."""
        return torch.exp(self.log_temperature)
        
    def create_attention_mask(self, token_ids, padding_token_id=0):
        """
        Create attention mask for variable length sequences.
        
        Args:
            token_ids: Tensor of shape (batch_size, seq_len) containing token IDs
            padding_token_id: Token ID used for padding (default: 0)
        
        Returns:
            attention_mask: Boolean tensor of shape (batch_size, seq_len) where
                          True means the position should be masked (ignored)
        """
        # Create mask: True for padding tokens (to be masked), False for real tokens
        attention_mask = (token_ids == padding_token_id)
        return attention_mask
    
    def forward(self, token_ids, attention_mask=None, padding_token_id=0):
        """
        Forward pass of the embedding model.
        
        Args:
            token_ids: Tensor of shape (batch_size, seq_len) containing token IDs.
                      Sequences can be shorter than context_window (will be padded).
                      Maximum length is context_window.
            attention_mask: Optional boolean tensor of shape (batch_size, seq_len) where
                          True means the position should be masked (ignored).
                          If None, will be created from padding_token_id.
            padding_token_id: Token ID used for padding (default: 0)
        
        Returns:
            embeddings: Tensor of shape (batch_size, embedding_dim) containing embeddings
        """
        batch_size, seq_len = token_ids.shape
        
        self.logger.log_shapes("token_ids (input)", token_ids)
        self.logger.log_info(f"Processing batch: {batch_size} sequences, initial length: {seq_len}")
        
        # Ensure sequence length doesn't exceed context window
        if seq_len > self.context_window:
            error_msg = f"Sequence length {seq_len} exceeds context window {self.context_window}"
            self.logger.log_error(error_msg)
            raise ValueError(error_msg)
        
        # Pad sequences to context_window if needed
        if seq_len < self.context_window:
            padding = torch.full(
                (batch_size, self.context_window - seq_len),
                padding_token_id,
                dtype=token_ids.dtype,
                device=token_ids.device
            )
            token_ids = torch.cat([token_ids, padding], dim=1)
            seq_len = self.context_window
            self.logger.log_info(f"Padded sequences to context_window: {self.context_window}")
        
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = self.create_attention_mask(token_ids, padding_token_id)
            self.logger.log_info("Created attention mask from padding tokens")
        else:
            # Pad attention mask if sequence was padded
            if attention_mask.shape[1] < self.context_window:
                padding_mask = torch.ones(
                    (batch_size, self.context_window - attention_mask.shape[1]),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device
                )
                attention_mask = torch.cat([attention_mask, padding_mask], dim=1)
                self.logger.log_info("Padded attention mask to match context_window")
        
        # Get token embeddings
        token_embeds = self.token_embedding(token_ids)  # (batch_size, context_window, hidden_dim)
        self.logger.log_shapes("token_embeds", token_embeds)
        self.logger.log_statistics("token_embeds", token_embeds)
        
        # Apply Rotary Position Embedding (RoPE)
        #
        # This is added outside of the attention block since the built-in class does not support RoPE.
        # Instead we will treat is like a positional encoding.
        #
        x = self.rope(token_embeds)  # (batch_size, context_window, hidden_dim)
        self.logger.log_shapes("after_rope", x)
        self.logger.log_statistics("after_rope", x)
        
        # Process through transformer encoder with attention mask
        # src_key_padding_mask: True means positions to ignore
        x = self.transformer_encoder(x, src_key_padding_mask=attention_mask)  # (batch_size, context_window, hidden_dim)
        self.logger.log_shapes("after_transformer", x)
        self.logger.log_statistics("after_transformer", x)
        
        # If masked language modeling is enabled, return the logits.
        # This predicts the masked tokens.
        if self.is_masked_language:
            logits = self.lm_head(x)  # (batch_size, context_window, vocab_size)
            self.logger.log_shapes("logits (MLM output)", logits)
            return logits  # (batch_size, context_window, vocab_size)

        # Masked mean pooling (only average over non-padded tokens)
        # Convert mask: True (padded) -> 0, False (real token) -> 1
        mask_expanded = (~attention_mask).float().unsqueeze(-1)  # (batch_size, context_window, 1)
        
        # Sum over sequence, dividing by number of real tokens
        x_masked = x * mask_expanded  # Zero out padded positions
        seq_lengths = mask_expanded.sum(dim=1)  # (batch_size, 1) - number of real tokens per sequence
        
        # Avoid division by zero
        seq_lengths = torch.clamp(seq_lengths, min=1.0)
        x = x_masked.sum(dim=1) / seq_lengths  # (batch_size, hidden_dim)
        self.logger.log_shapes("after_pooling", x)
        self.logger.log_statistics("after_pooling", x)
        
        # Project to output embedding dimension
        embeddings = self.projection(x)  # (batch_size, embedding_dim)
        self.logger.log_shapes("after_projection", embeddings)
        self.logger.log_statistics("after_projection", embeddings)

        # L2 normalize - this improves IR tasks like cosine similarity search.
        embeddings = nn.functional.normalize(embeddings, p=2, dim=1)
        
        # Apply temperature scaling if enabled
        # Temperature scaling adjusts the spread of embeddings in similarity space
        # After scaling, we re-normalize to keep embeddings on the unit sphere
        if self.use_temperature:
            temp = self.temperature
            embeddings = embeddings / temp
            embeddings = nn.functional.normalize(embeddings, p=2, dim=1)
            self.logger.log_value("temperature", temp.item(), category="values")
        
        self.logger.log_shapes("embeddings (final output)", embeddings)
        self.logger.log_statistics("embeddings (final output)", embeddings)
        
        return embeddings
    
    def get_embedding(self, token_ids, attention_mask=None, padding_token_id=0):
        """
        Get embeddings for input tokens (alias for forward method).
        
        Args:
            token_ids: Tensor of shape (batch_size, seq_len) containing token IDs
            attention_mask: Optional boolean tensor of shape (batch_size, seq_len) where
                          True means the position should be masked (ignored)
            padding_token_id: Token ID used for padding (default: 0)
        
        Returns:
            embeddings: Tensor of shape (batch_size, embedding_dim) containing embeddings
        """
        return self.forward(token_ids, attention_mask, padding_token_id)

