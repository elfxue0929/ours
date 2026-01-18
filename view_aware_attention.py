"""
View-aware Attention Module for SparseViT
Step 4: View-aware sparse attention mechanism
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ViewAwareAttention(nn.Module):
    """
    Enhanced attention with view-aware importance weighting
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, 
                 attn_drop=0., proj_drop=0., use_view_aware=True, 
                 view_alpha=1.0, sparse_topk=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.use_view_aware = use_view_aware
        self.view_alpha = view_alpha
        self.sparse_topk = sparse_topk  # If set, use top-k sparse attention
        
        # Standard attention components
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # View-aware importance MLP
        if use_view_aware:
            hidden_dim = max(dim // 4, 32)
            self.view_mlp = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 3),  # Output: [RGB, Noise, Freq] weights
            )
    
    def forward(self, x):
        B, N, C = x.shape
        
        # 1. Compute view importance weights if enabled
        if self.use_view_aware and hasattr(self, 'view_mlp'):
            # Get view weights for each token
            view_scores = self.view_mlp(x)  # (B, N, 3)
            view_weights = F.softmax(view_scores, dim=-1)  # (B, N, 3)
            
            # Extract individual view weights
            w_rgb = view_weights[..., 0]    # (B, N)
            w_noise = view_weights[..., 1]  # (B, N)
            w_freq = view_weights[..., 2]   # (B, N)
            
            # Non-semantic importance: noise + freq
            w_nonsemantic = w_noise + w_freq  # (B, N)
        else:
            w_nonsemantic = None
        
        # 2. Compute Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # 3. Compute attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, num_heads, N, N)
        
        # 4. Apply view-aware weighting if enabled
        if self.use_view_aware and w_nonsemantic is not None:
            # Expand non-semantic weights for broadcasting
            # w_nonsemantic: (B, N) -> (B, 1, 1, N)
            w_nonsemantic_expand = w_nonsemantic.unsqueeze(1).unsqueeze(1)
            
            # Bias attention scores towards non-semantic tokens
            # Higher w_nonsemantic[j] makes token j more likely to be attended to
            attn_biased = attn + self.view_alpha * w_nonsemantic_expand
        else:
            attn_biased = attn
        
        # 5. Apply sparse attention if specified
        if self.sparse_topk is not None and self.sparse_topk < N:
            # Keep only top-k attention scores per query
            topk_values, topk_indices = torch.topk(attn_biased, k=min(self.sparse_topk, N), dim=-1)
            
            # Create sparse attention mask with correct dtype
            attn_sparse = torch.full_like(attn, float('-inf'), dtype=attn.dtype)
            # Ensure topk_values has the same dtype as attn_sparse
            attn_sparse.scatter_(-1, topk_indices, topk_values.to(attn_sparse.dtype))
            
            # Apply softmax on sparse attention
            attn = F.softmax(attn_sparse, dim=-1)
        else:
            # Standard dense attention
            attn = F.softmax(attn_biased, dim=-1)
        
        attn = self.attn_drop(attn)
        
        # 6. Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        # Optionally return view weights for analysis
        if self.use_view_aware and hasattr(self, 'view_mlp'):
            self.last_view_weights = view_weights
        
        return x


class ViewAwareSABlock(nn.Module):
    """
    Sparse Attention Block with view-aware mechanism
    """
    def __init__(self, dim, num_heads, sparse_size=0, mlp_ratio=4., 
                 qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 use_view_aware=True, view_alpha=1.0, sparse_topk=None):
        super().__init__()
        self.pos_embed = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = norm_layer(dim)
        
        # Use ViewAwareAttention instead of standard Attention
        self.attn = ViewAwareAttention(
            dim,
            num_heads=num_heads, 
            qkv_bias=qkv_bias, 
            qk_scale=qk_scale,
            attn_drop=attn_drop, 
            proj_drop=drop,
            use_view_aware=use_view_aware,
            view_alpha=view_alpha,
            sparse_topk=sparse_topk
        )
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        
        from SparseViT import Mlp  # Import from main file
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, 
                       act_layer=act_layer, drop=drop)
        
        # Layer scale parameters
        self.sparse_size = sparse_size
        global layer_scale, init_value
        self.ls = layer_scale if 'layer_scale' in globals() else True
        
        if self.ls:
            init_val = init_value if 'init_value' in globals() else 1e-6
            self.gamma_1 = nn.Parameter(init_val * torch.ones((dim)), requires_grad=True)
            self.gamma_2 = nn.Parameter(init_val * torch.ones((dim)), requires_grad=True)
    
    def forward(self, x):
        from SparseViT import alter_sparse, alter_unsparse  # Import helper functions
        
        x_before = x.flatten(2).transpose(1, 2)
        B, N, H, W = x.shape
        
        if self.ls:
            # Apply spatial sparsity via blocking
            x, Ho, Hp, C = alter_sparse(x, self.sparse_size)
            Bf, Nf, Hf, Wf = x.shape
            x = x.flatten(2).transpose(1, 2)
            
            # Apply view-aware attention
            x = self.attn(self.norm1(x))
            
            # Reshape back
            x = x.transpose(1, 2).reshape(Bf, Nf, Hf, Wf)
            x = alter_unsparse(x, Ho, Hp, C, self.sparse_size)
            x = x.flatten(2).transpose(1, 2)
            
            # Residual connections with layer scale
            x = x_before + self.drop_path(self.gamma_1 * x)
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x), H, W))
        else:
            # Same as above but without layer scale
            x, Ho, Hp, C = alter_sparse(x, self.sparse_size)
            Bf, Nf, Hf, Wf = x.shape
            x = x.flatten(2).transpose(1, 2)
            x = self.attn(self.norm1(x))
            x = x.transpose(1, 2).reshape(Bf, Nf, Hf, Wf)
            x = alter_unsparse(x, Ho, Hp, C, self.sparse_size)
            x = x.flatten(2).transpose(1, 2)
            x = x_before + self.drop_path(x)
            x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        
        x = x.transpose(1, 2).reshape(B, N, H, W)
        return x


def create_view_aware_blocks(num_blocks, dim, num_heads, sparse_size,
                             mlp_ratio=4., qkv_bias=False, qk_scale=None,
                             drop=0., attn_drop=0., drop_path=0., 
                             norm_layer=nn.LayerNorm, use_view_aware=True,
                             view_alpha=1.0, sparse_topk=None):
    """
    Helper function to create multiple view-aware blocks
    """
    blocks = []
    for i in range(num_blocks):
        block = ViewAwareSABlock(
            dim=dim,
            num_heads=num_heads,
            sparse_size=sparse_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
            act_layer=nn.GELU,
            norm_layer=norm_layer,
            use_view_aware=use_view_aware,
            view_alpha=view_alpha,
            sparse_topk=sparse_topk
        )
        blocks.append(block)
    return nn.ModuleList(blocks)
