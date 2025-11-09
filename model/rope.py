import torch
import torch.nn as nn


class RoPE(nn.Module):
    """
    Rotary Position Embedding (RoPE) implementation.
    Applies rotary embeddings to input tensors based on position.
    """
    
    def __init__(self, dim, max_seq_len=512, base=10000.0):
        super(RoPE, self).__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Precompute frequency matrix
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
    def forward(self, x, seq_len=None):
        """
        Apply rotary position embedding to input tensor.
        
        Args:
            x: Tensor of shape (batch_size, seq_len, dim)
            seq_len: Optional sequence length (defaults to x.shape[1])
        
        Returns:
            Tensor with RoPE applied
        """
        if seq_len is None:
            seq_len = x.shape[1]
        
        batch_size, _, dim = x.shape
        
        # Create position indices
        t = torch.arange(seq_len, device=x.device, dtype=x.dtype)
        
        # Compute frequencies for each position
        freqs = torch.outer(t, self.inv_freq)  # (seq_len, dim // 2)
        
        # Create cos and sin embeddings
        cos = freqs.cos()  # (seq_len, dim // 2)
        sin = freqs.sin()  # (seq_len, dim // 2)
        
        # Reshape for broadcasting: (1, seq_len, 1, dim // 2)
        cos = cos.unsqueeze(0).unsqueeze(2)  # (1, seq_len, 1, dim // 2)
        sin = sin.unsqueeze(0).unsqueeze(2)  # (1, seq_len, 1, dim // 2)
        
        # Split x into pairs of dimensions for rotation
        # Reshape to (batch_size, seq_len, dim // 2, 2)
        x_reshaped = x.view(batch_size, seq_len, dim // 2, 2)
        
        # Extract x1 and x2 (the two components of each pair)
        x1 = x_reshaped[..., 0]  # (batch_size, seq_len, dim // 2)
        x2 = x_reshaped[..., 1]  # (batch_size, seq_len, dim // 2)
        
        # Apply rotation: [x1*cos - x2*sin, x1*sin + x2*cos]
        rotated_x1 = x1 * cos.squeeze(2) - x2 * sin.squeeze(2)
        rotated_x2 = x1 * sin.squeeze(2) + x2 * cos.squeeze(2)
        
        # Stack and reshape back
        rotated = torch.stack([rotated_x1, rotated_x2], dim=-1)  # (batch_size, seq_len, dim // 2, 2)
        rotated = rotated.view(batch_size, seq_len, dim)  # (batch_size, seq_len, dim)
        
        return rotated

