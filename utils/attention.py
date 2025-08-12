import math

import torch
from torch import nn
import torch.nn.functional as F

def masked_softmax(X, valid_lens):
    """Perform softmax by masking elements on last axis.
    
    Args:
        X (torch.Tensor): (B, N, M) Input tensor to be masked, where
            B is the batch size, N is the number of queries, and M is 
            the number of keys
        valid_len (torch.Tensor): (B,) or (B, N) Tensor of type int
            describing how many elements in the sequence to keep

    Returns: 
        torch.Tensor: (B, N, M) Masked input array 
    """
    def _sequence_mask(X, valid_len, value=0.0):
        """Sets to value all but the first valid_len elements of a sequence
        
        Expects X to be 2D Tensor (B, M), valid_lens 1D tensor (B)"""
        mask = torch.arange(X.shape[1], dtype=torch.float32, 
                            device=X.device)[None, :] < valid_len[:, None]
        X[~mask] = value
        return X

    B, N, M = X.shape
    if valid_lens is None:
        return F.softmax(X, dim=-1)
    else:
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, N)
        else:
            valid_lens = valid_lens.reshape(-1)
        X = _sequence_mask(X.reshape(-1, M), valid_lens, value=-1e6)
        return F.softmax(X.reshape((B, N, M)), dim=-1)

class DotProductAttention(nn.Module):
    """Scaled dot product attention.

    Can be written as
        Dropout(Softmax(Mask(QK^T/sqrt(D)))) V"""
    def __init__(self, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, queries, keys, values, valid_lens=None):
        """Computes masked dot product attention w/ Q, K, V

        B: batch size
        N: number of queries
        M: number of keys/values
        D: dimension of queries/keys
        V: dimension of values
        
        Args:
            queries (Tensor): (B, N, D)
            keys (Tensor): (B, M, D)
            values (Tensor): (B, M, V)
            valid_lens (Tensor): (B) or (B, N)
            
        Returns:
            Tensor: (B, N, V)"""
        D = queries.shape[-1]
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(D)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)

class AdditiveAttention(nn.Module):
    """Additive attention layer
    
    Addititive attention scoring: 
        a(q, k) = w_v^T tanh(W_q q + W_k k)
    Attention weights: 
        \alpha = Softmax(Mask(a(Q, K)))
    Attention output:
        \alpha @ V"""
    def __init__(self, num_hiddens, dropout):
        super().__init__()
        self.W_k = nn.LazyLinear(num_hiddens, bias=False)
        self.W_q = nn.LazyLinear(num_hiddens, bias=False)
        self.w_v = nn.LazyLinear(1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens=None):
        """Computes additive attention w/ Q, K, V

        B: batch size
        N: number of queries
        M: numbers of keys/values
        q, k, v: query, key, value dimension
        H: num_hiddens (dimension of W_q @ Q, W_k @ K)
        
        Args:
            queries (Tensor): (B, N, q)
            keys (Tensor): (B, M, k)
            values (Tensor): (B, M, v)
            valid_lens (Tensor): (B) or (B, N)
            
        Returns:
            Tensor: (B, N, v)"""
        queries = self.W_q(queries)
        keys = self.W_k(keys)
        # (B, N, M, H)
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        features = torch.tanh(features)
        # (B, N, M)
        scores = self.w_v(features).squeeze(-1)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)

class MultiHeadAttention(nn.Module):
    """Multi-headed attention (MHA)"""
    def __init__(self, num_hiddens, num_heads, dropout=0, bias=False):
        """Initializes MHA
        
        Args:
            num_hiddens (int): total number of hidden layers
            num_heads (int): how many attention heads
            dropout (float): applied to attention weights"""
        super().__init__()
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)
        self.W_q = nn.LazyLinear(num_hiddens, bias=bias)
        self.W_k = nn.LazyLinear(num_hiddens, bias=bias)
        self.W_v = nn.LazyLinear(num_hiddens, bias=bias)
        self.W_o = nn.LazyLinear(num_hiddens, bias=bias)
    
    def forward(self, queries, keys, values, valid_lens):
        """Computes masked MHA using Q, K, V

        Computes linear transformation to Q, K, V so that
        each has dimension H. Then reshapes Q, K, V
        into h separate heads each of dimension H/h. Then
        applies DotProductAttention each of the h heads.
        This allows for each head to have a different
        attention matrix.

        B: batch size
        H: num_hiddens
        N: number of queries
        M: numbers of keys/values
        q, k, v: dimensions of queries, keys, values
        
        Args:
            queries (torch.Tensor): (B, N, q)
            keys (torch.Tensor): (B, M, k)
            values (torch.Tensor): (B, M, v)
        
        Returns:
            torch.Tensor: (B, N, H)"""
        queries = self.W_q(queries)
        keys = self.W_k(keys)
        values = self.W_k(values)

        # (B, N or M, H) -> (B * h, N or M, H/h)
        queries = self.transpose_qkv(queries)
        keys = self.transpose_qkv(keys)
        values = self.transpose_qkv(values)

        # batch dimension changes, so valid_lens must change
        if valid_lens is not None:
            valid_lens = torch.repeat_interleave(
                valid_lens, repeats=self.num_heads, dim=0
            )

        output = self.attention(queries, keys, values, valid_lens)
        # (B * h, N, H/h) -> (B, N, H)
        output_concat = self.transpose_out(output)
        # (B, N, H) -> (B, N, H)
        return self.W_o(output_concat)

    def transpose_qkv(self, X):
        X = X.reshape(X.shape[0], X.shape[1], self.num_heads, -1)
        X = X.permute(0, 2, 1, 3)
        return X.reshape(-1, X.shape[2], X.shape[3])

    def transpose_out(self, X):
        X = X.reshape(-1, self.num_heads, X.shape[1], X.shape[2])
        X = X.permute(0, 2, 1, 3)
        return X.reshape(X.shape[0], X.shape[1], -1)

class PositionalEncoding(nn.Module):
    """Adds a positional encoding to an embedding vector"""
    def __init__(self, num_hiddens, dropout, max_len=1000):
        """Initializes positional encoding module.

        P_{i,2j} = sin(i / 10000**(2j/d))
        P_{i,2j+1} = cos(i / 10000**(2j/d))
        
        Args:
            num_hiddens (int): dimensionality of embedding
            dropout (float): applied after sum
            max_len (int): largest possible num_steps"""
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
            0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)
    
    def forward(self, X):
        """Forward pass, adds positional encoding to X
        
        Args: 
            X (torch.Tensor): (B, N, d)
        
        Returns:
            torch.Tensor: X + P, (B, N, d)"""
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)

class PositionWiseFFN(nn.Module):
    """2-layer feed-forward network used within transformer block"""
    def __init__(self, ffn_num_hiddens, ffn_num_outputs):
        super().__init__()
        self.dense1 = nn.LazyLinear(ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.LazyLinear(ffn_num_outputs)

    def forward(self, X):
        """Forward pass of FFN
        
        Linear layers perform identical operations on every token,
        only modify hidden dimension
        """
        return self.dense2(self.relu(self.dense1(X)))

class AddNorm(nn.Module):
    """Compute residual block with post-norm
    
    LayerNorm(X + Dropout(Y))"""
    def __init__(self, norm_shape, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(norm_shape)
    
    def forward(self, X, Y):
        """Forward pass of AddNorm"""
        return self.ln(X + self.dropout(Y))
    
class TransformerEncoderBlock(nn.Module):
    """Encoder block for encoder-decoder transformer"""
    def __init__(self, num_hiddens, ffn_num_hiddens, num_heads, dropout, use_bias=False):
        super().__init__()
        self.attention = MultiHeadAttention(num_hiddens, num_heads, dropout, use_bias)
        self.addnorm1 = AddNorm(num_hiddens, dropout)
        self.ffn = PositionWiseFFN(ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(num_hiddens, dropout)

    def forward(self, X, valid_lens):
        """Forward pass of transformer encoder block

        Y_n = LayerNorm(X_n + Dropout(MSHA(Mask(X_n))))
        X_{n+1} = LayerNorm(Y_n + Dropout(FFN(Y_n)))
        
        Args:
            X (torch.Tensor): (B, N, H) Input tensor
            valid_lens (torch.Tensor): (B,) or (B, N) Tensor of type int
                describing how many tokens to attend to in each sequence

        Returns:
            torch.Tensor: (B, N, H) Output tensor after attention and FFN 
        
        """
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens))
        return self.addnorm2(Y, self.ffn(Y))

class TransformerDecoderBlock(nn.Module):
    """Decoder block for encoder-decoder transformer
    
    For details, see:
    https://d2l.ai/chapter_attention-mechanisms-and-transformers/transformer.html

    In the first sublayer, a MHSA sublayer with a causal mask is 
    applied to the target sequence. During training, the mask is 
    a 2D tensor of shape (B, N) where the 1st token has length 1,
    the 2nd token has length 2, etc. During prediction, the input X
    is the concatenation of all previous predictions, and the 
    goal is to predict the next token, and dec_valid_lens is None
    (no mask).

    In the second sublayer, the output of the encoder MHSA is used
    as keys and values for mult-headed attention. The queries are 
    given by the output of the first sublayer. 
    
    In the third sublayer, a position-wise feed-forward network
    is applied to the output of the second sublayer.
    
    Between each sublayer a normed residual connection is applied:
        LayerNorm(X + Dropout(sublayer(X)))"""
    def __init__(self, num_hiddens, ffn_num_hiddens, num_heads, dropout, i):
        super().__init__()
        self.i = i
        self.attention1 = MultiHeadAttention(num_hiddens, 
                                             num_heads, dropout)
        self.addnorm1 = AddNorm(num_hiddens, dropout)
        self.attention2 = MultiHeadAttention(num_hiddens, 
                                             num_heads, dropout)
        self.addnorm2 = AddNorm(num_hiddens, dropout)
        self.ffn = PositionWiseFFN(ffn_num_hiddens, num_hiddens)
        self.addnorm3 = AddNorm(num_hiddens, dropout)
    
    def forward(self, X, state):
        enc_outputs, enc_valid_lens, pred_state = state
        if pred_state[self.i] is None: # training or first prediction
            key_values = X
        else: # prediction mode, not first prediction
            key_values = torch.cat((pred_state[self.i], X), dim=1)
        pred_state[self.i] = key_values
        if self.training:
            batch_size, num_steps, _ = X.shape
            dec_valid_lens = torch.arange(
                1, num_steps+1, device=X.device).repeat(batch_size, 1)
        else:
            dec_valid_lens = None # use all predictions seen so far
        X2 = self.attention1(X, key_values, key_values, dec_valid_lens)
        Y = self.addnorm1(X, X2)
        Y2 = self.attention2(Y, enc_outputs, enc_outputs, enc_valid_lens)
        Z = self.addnorm2(Y, Y2)
        Z2 = self.ffn(Z)
        return self.addnorm3(Z, Z2), state
    
class PatchEmbedding(nn.Module):
    """Patch embedding module for vision transformers
    
    Extracts P patches, applies identical linear transformation
    to each patch. Each patch is transformed from dimension 
    c*(p**2) to dimension H (num_hiddens) """
    def __init__(self, img_size, patch_size, num_hiddens):
        super().__init__()
        def _make_tuple(x):
            if not isinstance(x, (list, tuple)):
                return (x, x)
            else:
                return x
        img_size, patch_size = _make_tuple(img_size), _make_tuple(patch_size)
        self.num_patches = (img_size[0] // patch_size[0]) * (
            img_size[1] // patch_size[1])
        self.conv = nn.LazyConv2d(num_hiddens, kernel_size=patch_size, 
                                  stride=patch_size)
        
    def forward(self, X):
        """Forward pass of patch embedding module
        
        B: batch_size
        c, h, w: channels, height, width
        H: num_hiddens
        P: num_patches

        Args:
            X (torch.Tensor): (B, c, h, w)
        
        Returns:
            torch.Tensor: (B, P, H)
        """
        return self.conv(X).flatten(2).transpose(1, 2)

class ViTMLP(nn.Module):
    """2-layer position-wise MLP w/ GELU activation and dropout"""
    def __init__(self, mlp_num_hiddens, mlp_num_outputs, dropout=0.5):
        super().__init__()
        self.dense1 = nn.LazyLinear(mlp_num_hiddens)
        self.gelu = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.dense2 = nn.LazyLinear(mlp_num_outputs)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.dropout1(self.gelu(self.dense1(x)))
        return self.dropout2(self.dense2(x))

class ViTBlock(nn.Module):
    """Vision transformer block module
    
    y^n = x^n + MHSA(layernorm(x^n))
    x^{n+1} = y^n + MLP(layernorm(y^n))"""
    def __init__(self, num_hiddens, mlp_num_hiddens,
                 num_heads, dropout, use_bias=False):
        """Initializes module
        
        Args:
            num_hiddens (int): size of hidden layer
            mlp_num_hiddens (int): hidden layer in MLP
            num_heads (int): number of heads in MHSA
            dropout (float): applied in MHSA, MLP
            use_bias (bool): whether to use bias in MHSA"""
        super().__init__()
        self.ln1 = nn.LayerNorm(num_hiddens)
        self.attention = MultiHeadAttention(num_hiddens, num_heads, 
                                            dropout, use_bias)
        self.ln2 = nn.LayerNorm(num_hiddens)
        self.mlp = ViTMLP(mlp_num_hiddens, num_hiddens, dropout)
    
    def forward(self, X, valid_lens=None):
        """Forward pass of transformer block
        
        Args:
            X (torch.Tensor): (B, N, H)
            valid_lens (torch.Tensor): masks MHSA
                since ViT is encoder-only, no masking
        
        Returns:
            torch.Tensor: (B, N, H)"""
        X_n = self.ln1(X)
        Y = X + self.attention(X_n, X_n, X_n, valid_lens)
        Y_n = self.ln2(Y)
        return Y + self.mlp(Y_n)