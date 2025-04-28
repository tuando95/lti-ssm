import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple


class RandomSSM(nn.Module):
    """
    Baseline: An SSM with fixed random matrices, initialized once and not trained.
    Accepts dimensions directly instead of a config dict for simpler instantiation.
    """
    def __init__(self, n_dim: int, m_dim: int, p_dim: int):
        super().__init__()
        self.n_dim = n_dim
        self.m_dim = m_dim
        self.p_dim = p_dim

        # Fixed random matrices (initialized once)
        # Scale A like in Glorot initialization for stability
        A = torch.randn(self.n_dim, self.n_dim) / np.sqrt(self.n_dim)
        B = torch.randn(self.n_dim, self.m_dim)
        C = torch.randn(self.p_dim, self.n_dim)
        D = torch.randn(self.p_dim, self.m_dim)

        # Register as buffers (fixed, not trained)
        self.register_buffer('A', A)
        self.register_buffer('B', B)
        self.register_buffer('C', C)
        self.register_buffer('D', D)

        print(f"Initialized RandomSSM baseline (n={n_dim}, m={m_dim}, p={p_dim})")

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """
        Simulates the system for the given input sequence u.

        Args:
            u (torch.Tensor): Input sequence (batch_size, T, m_dim).

        Returns:
            torch.Tensor: Output sequence y_hat (batch_size, T, p_dim).
        """
        batch_size, T, _ = u.shape
        # Initialize hidden state (batch_size, n_dim)
        x = torch.zeros(batch_size, self.n_dim, device=u.device, dtype=u.dtype)
        
        y_hat_sequence = []
        for k in range(T):
            u_k = u[:, k, :] # (batch_size, m_dim)
            # State update: x[k+1] = x[k] @ A.T + u[k] @ B.T
            # Note: using einsum for clarity with batch dimension
            x = torch.einsum('bn,nn->bn', x, self.A.T) + torch.einsum('bm,mn->bn', u_k, self.B.T)
            # Output: y[k] = x[k] @ C.T + u[k] @ D.T
            y_k = torch.einsum('bn,np->bp', x, self.C.T) + torch.einsum('bm,mp->bp', u_k, self.D.T)
            y_hat_sequence.append(y_k.unsqueeze(1)) # Append (batch_size, 1, p_dim)
        
        # Concatenate outputs along the time dimension
        y_hat_sequence = torch.cat(y_hat_sequence, dim=1) # (batch_size, T, p_dim)
        
        # Return y_hat and None to match the expected signature (y_pred_seq, x_sequence)
        return y_hat_sequence, None


# Simple Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x: Tensor, shape [seq_len, batch_size, embedding_dim]"""
        # Transformer layers expect (seq_len, batch, dim)
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class SimpleTransformer(nn.Module):
    """
    Baseline: A simple Transformer encoder model for sequence-to-sequence mapping.
    Maps input sequence u to output sequence y_hat.
    Accepts dimensions and hyperparams directly.
    """
    def __init__(self, m_dim: int, p_dim: int, d_model: int, nhead: int, num_encoder_layers: int, 
                 dim_feedforward: Optional[int] = None, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        if dim_feedforward is None:
            dim_feedforward = d_model * 4 # Common default

        self.input_embed = nn.Linear(m_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout, 
            batch_first=True # Expect (batch, seq, feature)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.output_linear = nn.Linear(d_model, p_dim)

        print(f"Initialized SimpleTransformer baseline (layers={num_encoder_layers}, d_model={d_model}, nhead={nhead})")

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """
        Args:
            u (torch.Tensor): Input sequence (batch_size, T, m_dim).

        Returns:
            torch.Tensor: Output sequence y_hat (batch_size, T, p_dim).
        """
        # Embed input
        embedded_u = self.input_embed(u) * np.sqrt(self.d_model) # Scale embedding
        
        # Add positional encoding - needs (T, batch, dim)
        # embedded_u = embedded_u.permute(1, 0, 2) # Transpose to (T, batch, dim)
        # pos_encoded_u = self.pos_encoder(embedded_u)
        # pos_encoded_u = pos_encoded_u.permute(1, 0, 2) # Transpose back to (batch, T, dim)
        # NOTE: The above is correct if batch_first=False. If batch_first=True, 
        #       PositionalEncoding needs adjustment or input needs different handling. 
        #       Let's try adapting PositionalEncoding for batch_first.
        #       Simpler alternative: Apply positional encoding directly if shape matches. Need PE (batch, T, dim)
        #       Current PE is (max_len, 1, dim). Let's use it directly on embedded_u (batch, T, dim)
        T = u.size(1)
        pos_encoded_u = embedded_u + self.pos_encoder.pe[:T, 0, :].unsqueeze(0) # Add PE (1, T, dim) - broadcasting
        pos_encoded_u = self.pos_encoder.dropout(pos_encoded_u)

        # Pass through transformer encoder (expects batch_first=True)
        transformer_output = self.transformer_encoder(pos_encoded_u) # (batch, T, d_model)

        # Final linear layer
        y_hat = self.output_linear(transformer_output) # (batch, T, p_dim)
        
        # Return y_hat and None to match the expected signature (y_pred_seq, x_sequence)
        return y_hat, None
