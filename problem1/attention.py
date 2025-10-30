"""
Attention mechanisms for sequence-to-sequence modeling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.functional import softmax


# def scaled_dot_product_attention(Q, K, V, mask=None):
#     """
#     Compute scaled dot-product attention.

#     Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

#     Args:
#         Q: Query tensor [batch, ..., seq_len_q, d_k]
#         K: Key tensor [batch, ..., seq_len_k, d_k]
#         V: Value tensor [batch, ..., seq_len_v, d_k]
#         mask: Optional mask [batch, ..., seq_len_q, seq_len_k]
#               Values: 1 for positions to attend, 0 for positions to mask

#     Returns:
#         output: Attention output [batch, ..., seq_len_q, d_k]
#         attention_weights: Attention weights [batch, ..., seq_len_q, seq_len_k]
#     """
#     d_k = Q.size(-1)

#     # Compute raw attention scores: QK^T
#     scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))

#     # Apply mask if provided
#     if mask is not None:
#         scores = scores.masked_fill(mask == 0, float('-inf'))

#     # Compute attention weights
#     attention_weights = F.softmax(scores, dim=-1)

#     # Compute output as weighted sum of values
#     output = torch.matmul(attention_weights, V)
#     return output, attention_weights


def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.size(-1)

    # Use a float (or device-aware tensor) for scaling to avoid CPU/CUDA mismatch
    scale = math.sqrt(d_k)  

    scores = torch.matmul(Q, K.transpose(-2, -1)) / scale

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    attention_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attention_weights, V)
    return output, attention_weights



class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism.

    Splits d_model into num_heads, applies attention in parallel,
    then concatenates and projects the results.
    """

    def __init__(self, d_model, num_heads):
        """
        Initialize multi-head attention.

        Args:
            d_model: Model dimension (must be divisible by num_heads)
            num_heads: Number of attention heads
        """
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Linear projections for Q, K, V
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)

        # Output projection
        self.W_O = nn.Linear(d_model, d_model)



    def split_heads(self, x):
        """
        Split tensor into multiple heads.

        Args:
            x: Input tensor [batch, seq_len, d_model]

        Returns:
            Tensor with shape [batch, num_heads, seq_len, d_k]
        """
        batch_size, seq_len, _ = x.size()

        x = x.view(batch_size, seq_len, self.num_heads, self.d_k)
        x = x.transpose(1, 2)  # [batch, num_heads, seq_len, d_k]
        return x

    def combine_heads(self, x):
        """
        Combine multiple heads back into single tensor.

        Args:
            x: Input tensor [batch, num_heads, seq_len, d_k]

        Returns:
            Tensor with shape [batch, seq_len, d_model]
        """
        batch_size, num_heads, seq_len, d_k = x.size()
        x = x.transpose(1, 2).contiguous().view(batch_size, seq_len, num_heads * d_k)
        return x

    # def forward(self, query, key, value, mask=None):
    #     """
    #     Forward pass of multi-head attention.

    #     Args:
    #         query: Query tensor [batch, seq_len_q, d_model]
    #         key: Key tensor [batch, seq_len_k, d_model]
    #         value: Value tensor [batch, seq_len_v, d_model]
    #         mask: Optional attention mask

    #     Returns:
    #         output: Attention output [batch, seq_len_q, d_model]
    #         attention_weights: Attention weights [batch, num_heads, seq_len_q, seq_len_k]
    #     """

    #     batch_size = query.size(0)

    #     # Linear projections
    #     Q = self.W_Q(query)
    #     K = self.W_K(key)
    #     V = self.W_V(value)

    #     # Split into heads
    #     Q = self.split_heads(Q)
    #     K = self.split_heads(K)
    #     V = self.split_heads(V)

    #     # Scaled dot-product attention
    #     d_k = self.d_k
    #     scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))

    #     if mask is not None:
    #         scores = scores.masked_fill(mask == 0, float('-inf'))

    #     attention_weights = F.softmax(scores, dim=-1)
    #     attention_output = torch.matmul(attention_weights, V)

    #     # Combine heads
    #     combined_output = self.combine_heads(attention_output)

    #     # Final linear projection
    #     output = self.W_O(combined_output)

    #     return output, attention_weights

    def forward(self, query, key, value, mask=None, return_attention=False):
        """
        Forward pass of multi-head attention.

        Args:
            query: Query tensor [batch, seq_len_q, d_model]
            key: Key tensor [batch, seq_len_k, d_model]
            value: Value tensor [batch, seq_len_v, d_model]
            mask: Optional attention mask in one of these shapes:
                - causal mask: [1, 1, L, L]
                - padding mask: [batch, L]  (1 for keep, 0 for pad)
                - expanded padding mask: [batch, 1, 1, L]
                - full mask: [batch, 1, seq_q, seq_k] or [batch, seq_q, seq_k]

        Returns:
            output: Attention output [batch, seq_len_q, d_model]
            attention_weights: Attention weights [batch, num_heads, seq_len_q, seq_len_k]
        """
        batch_size = query.size(0)

        # Linear projections
        Q = self.W_Q(query)  # [batch, seq_q, d_model]
        K = self.W_K(key)    # [batch, seq_k, d_model]
        V = self.W_V(value)  # [batch, seq_v, d_model]

        # Split into heads: [batch, num_heads, seq_len, d_k]
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        # Compute scores with a Python float scale to avoid creating CPU tensors
        scale = math.sqrt(self.d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / scale
        # scores shape: [batch, num_heads, seq_q, seq_k]

        # ---- Robust mask handling ----
        # if mask is not None:
        #     # Convert mask to boolean
        #     m = mask

        #     # If mask is padding mask [batch, seq_k] -> [batch, 1, 1, seq_k]
        #     if m.dim() == 2:
        #         # assume 1 means keep, 0 means mask
        #         m = m.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, seq_k]

        #     # If mask is [batch, seq_q, seq_k] -> make it [batch, 1, seq_q, seq_k]
        #     elif m.dim() == 3:
        #         m = m.unsqueeze(1)  # [batch, 1, seq_q, seq_k]

        #     # If mask is [1, 1, L, L] (causal), keep as-is but ensure device/dtype
        #     # Now expand to [batch, num_heads, seq_q, seq_k] for broadcasting
        #     # We also ensure mask is boolean and on same device as scores
        #     m = m.to(device=scores.device)
        #     m = m.to(dtype=torch.bool)

        #     # If mask has leading singleton batch/head dims (e.g., [1,1,L,L])
        #     # expand to [batch_size, num_heads, seq_q, seq_k]
        #     # Only expand singleton dimensions
        #     m = mask.to(device=scores.device, dtype=torch.bool)

        #     # Ensure 4D: [batch, 1, seq_q, seq_k] or [1, 1, seq_len, seq_len]
        #     while m.dim() < 4:
        #         m = m.unsqueeze(0)

        #     # Repeat to match batch_size and num_heads only if singleton
        #     m = m.repeat(batch_size // m.size(0), self.num_heads // m.size(1), 1, 1)

        #     # Apply mask: positions with False (0) are masked.
        #     scores = scores.masked_fill(~m, float('-inf'))

        # m: input mask (None, [batch, seq], or [seq, seq])
        m = mask
        if m is not None:
            if m.dim() == 2:  # padding mask [batch, seq]
                m = m[:, None, None, :]  # [batch, 1, 1, seq]
            elif m.dim() == 3:  # [batch, seq_q, seq_k]
                m = m[:, None, :, :]  # [batch, 1, seq_q, seq_k]
            elif m.dim() == 4:
                pass  # already [batch, heads, seq_q, seq_k]
            
            # expand singleton dims
            batch_size, seq_q, seq_k = scores.shape[0], scores.shape[2], scores.shape[3]
            if m.size(0) == 1:
                m = m.expand(batch_size, -1, -1, -1)
            if m.size(1) == 1:
                m = m.expand(-1, scores.size(1), -1, -1)
            
            # ensure mask matches seq_q, seq_k
            if m.size(2) != seq_q or m.size(3) != seq_k:
                m = m.expand(-1, -1, seq_q, seq_k)
            
            m = m.to(dtype=torch.bool, device=scores.device)
            
            scores = scores.masked_fill(~m, float('-inf'))


        # Compute attention weights
        attention_weights = F.softmax(scores, dim=-1)

        # Optional: numerical safety check (uncomment for debugging)
        # if torch.isnan(attention_weights).any():
        #     print("NaN in attention_weights; scores stats:", torch.isnan(scores).any(), scores.min(), scores.max())

        attention_output = torch.matmul(attention_weights, V)  # [batch, num_heads, seq_q, d_k]

        # Combine heads
        combined_output = self.combine_heads(attention_output)  # [batch, seq_q, d_model]

        # Final linear projection
        output = self.W_O(combined_output)

        if return_attention:
            return output, attention_weights
        else:
            return output, None



def create_causal_mask(seq_len, device=None):
    """
    Create causal mask to prevent attending to future positions.

    Args:
        seq_len: Sequence length
        device: Device to create tensor on

    Returns:
        Mask tensor [1, 1, seq_len, seq_len] lower triangular matrix
    """
    # Lower triangular matrix
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    return mask.unsqueeze(0).unsqueeze(0)
