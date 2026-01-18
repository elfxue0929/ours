"""
Multi-View Feature Fusion Module for SparseViT
Step 3: Fusion of RGB, Noise, and Freq features
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiViewFusion(nn.Module):
    """
    Fuse multi-view features (RGB, Noise, Freq) into a unified representation
    """
    def __init__(self, c_rgb=64, c_noise=32, c_freq=32, c_out=64, fusion_type='concat_proj'):
        super().__init__()
        self.fusion_type = fusion_type
        
        if fusion_type == 'concat_proj':
            # Concatenate and project
            total_channels = c_rgb + c_noise + c_freq
            self.proj = nn.Sequential(
                nn.Conv2d(total_channels, c_out, kernel_size=1),
                nn.BatchNorm2d(c_out),
                nn.GELU()
            )
            
        elif fusion_type == 'weighted_sum':
            # Learned weighted sum with channel alignment
            self.align_noise = nn.Conv2d(c_noise, c_rgb, kernel_size=1)
            self.align_freq = nn.Conv2d(c_freq, c_rgb, kernel_size=1)
            self.weights = nn.Parameter(torch.ones(3) / 3.0)
            self.proj = nn.Conv2d(c_rgb, c_out, kernel_size=1) if c_rgb != c_out else nn.Identity()
            
        elif fusion_type == 'attention':
            # Cross-attention based fusion
            self.align_noise = nn.Conv2d(c_noise, c_rgb, kernel_size=1)
            self.align_freq = nn.Conv2d(c_freq, c_rgb, kernel_size=1)
            
            # Self-attention to fuse features
            self.norm = nn.LayerNorm(c_rgb)
            self.attn = nn.MultiheadAttention(c_rgb, num_heads=4)
            self.proj = nn.Conv2d(c_rgb, c_out, kernel_size=1) if c_rgb != c_out else nn.Identity()
            
            # ========== 新增：View Gate Head ==========
            self.view_gate = nn.Sequential(
                nn.Conv2d(c_rgb * 3, c_rgb, kernel_size=1, bias=False),
                nn.GELU(),
                nn.Conv2d(c_rgb, 3, kernel_size=1, bias=True)
            )
            # Gate 初始化成"接近平均"
            nn.init.zeros_(self.view_gate[-1].weight)
            nn.init.zeros_(self.view_gate[-1].bias)
            
            # View Dropout 参数
            self.view_dropout = 0.2
            
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")
    
    def forward(self, F_rgb, F_noise, F_freq, return_gate: bool = False):
        """
        Args:
            F_rgb: (B, c_rgb, H, W)
            F_noise: (B, c_noise, H, W)
            F_freq: (B, c_freq, H, W)
        Returns:
            Fused features: (B, c_out, H, W)
        """
        B, C_rgb, H, W = F_rgb.shape
        
        if self.fusion_type == 'concat_proj':
            # Simple concatenation and projection
            x = torch.cat([F_rgb, F_noise, F_freq], dim=1)  # (B, C_total, H, W)
            x = self.proj(x)  # (B, c_out, H, W)
            
        elif self.fusion_type == 'weighted_sum':
            # Align channels
            F_noise_aligned = self.align_noise(F_noise)  # (B, c_rgb, H, W)
            F_freq_aligned = self.align_freq(F_freq)      # (B, c_rgb, H, W)
            
            # Weighted sum
            weights = torch.softmax(self.weights, dim=0)
            x = weights[0] * F_rgb + weights[1] * F_noise_aligned + weights[2] * F_freq_aligned
            x = self.proj(x)
            
        elif self.fusion_type == 'attention':
            # Align channels
            F_noise_aligned = self.align_noise(F_noise)
            F_freq_aligned = self.align_freq(F_freq)
            
            # Stack features: (B, 3, C, H, W)
            x_views = torch.stack([F_rgb, F_noise_aligned, F_freq_aligned], dim=1)
            
            # ========== View Dropout（训练时随机 drop 一个 view）==========
            if self.training and self.view_dropout > 0:
                if torch.rand(1).item() < self.view_dropout:
                    drop_id = torch.randint(0, 3, (1,)).item()
                    x_views[:, drop_id] = 0
            
            # Reshape for attention: (B, 3, C, H, W) -> (3, B*H*W, C)
            features = x_views.permute(1, 0, 3, 4, 2).reshape(3, B * H * W, C_rgb)
            
            # Apply self-attention
            features_norm = self.norm(features)
            attended, _ = self.attn(features_norm, features_norm, features_norm)
            attended = attended + features  # Residual connection
            
            # Reshape back: (3, B*H*W, C) -> (B, 3, C, H, W)
            x_attn = attended.reshape(3, B, H, W, C_rgb).permute(1, 0, 4, 2, 3).contiguous()
            
            # ========== Gated Aggregation（替代 mean）==========
            # step 1: concat views
            x_cat = x_attn.view(B, 3 * C_rgb, H, W)  # (B, 3C, H, W)
            
            # step 2: predict gate
            gate_logits = self.view_gate(x_cat)      # (B, 3, H, W)
            gate = torch.softmax(gate_logits, dim=1) # normalize over views
            
            # step 3: weighted sum
            gate_weight = gate.unsqueeze(2)          # (B, 3, 1, H, W)
            x = (x_attn * gate_weight).sum(dim=1)    # (B, C, H, W)
            
            x = self.proj(x)
            if return_gate:
                return x, gate
        
        return x


class AdaptiveMultiViewFusion(nn.Module):
    """
    Adaptive fusion with spatial and channel attention
    """
    def __init__(self, c_rgb=64, c_noise=32, c_freq=32, c_out=64):
        super().__init__()
        
        # Channel alignment
        self.align_noise = nn.Conv2d(c_noise, c_rgb, kernel_size=1)
        self.align_freq = nn.Conv2d(c_freq, c_rgb, kernel_size=1)
        
        # Spatial attention for each branch
        self.spatial_attn_rgb = nn.Sequential(
            nn.Conv2d(c_rgb, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
        self.spatial_attn_noise = nn.Sequential(
            nn.Conv2d(c_rgb, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
        self.spatial_attn_freq = nn.Sequential(
            nn.Conv2d(c_rgb, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
        
        # Channel attention
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c_rgb * 3, c_rgb * 3 // 4, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(c_rgb * 3 // 4, c_rgb * 3, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Final projection
        self.proj = nn.Sequential(
            nn.Conv2d(c_rgb * 3, c_out, kernel_size=1),
            nn.BatchNorm2d(c_out),
            nn.GELU()
        )
    
    def forward(self, F_rgb, F_noise, F_freq):
        # Align channels
        F_noise = self.align_noise(F_noise)
        F_freq = self.align_freq(F_freq)
        
        # Apply spatial attention
        F_rgb = F_rgb * self.spatial_attn_rgb(F_rgb)
        F_noise = F_noise * self.spatial_attn_noise(F_noise)
        F_freq = F_freq * self.spatial_attn_freq(F_freq)
        
        # Concatenate
        x = torch.cat([F_rgb, F_noise, F_freq], dim=1)
        
        # Apply channel attention
        channel_weights = self.channel_attn(x)
        x = x * channel_weights
        
        # Final projection
        x = self.proj(x)
        
        return x

