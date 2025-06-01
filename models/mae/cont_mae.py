import torch
import torch.nn as nn
from functools import partial
from models.mae.model_mae import MaskedAutoencoderViT
import models.mae.video_vit as video_vit

class ContrastiveMAE(nn.Module):
    """
    Uses MAE ViT encoder for contrastive learning instead of reconstruction.
    Extracts features for dual contrastive loss (spatial + temporal).
    """
    def __init__(self, 
                 img_size=224,
                 patch_size=16, 
                 in_chans=3,
                 embed_dim=1024,
                 depth=24,
                 num_heads=16,
                 mlp_ratio=4.0,
                 norm_layer=nn.LayerNorm,
                 num_frames=16,
                 t_patch_size=2,
                 projection_dim=256,
                 temporal_projection_dim=128,
                 use_cls_token=True,
                 **kwargs):
        super().__init__()
        
        self.use_cls_token = use_cls_token
        self.embed_dim = embed_dim
        
        # MAE encoder (without decoder)
        self.patch_embed = video_vit.PatchEmbed(
            img_size, patch_size, in_chans, embed_dim, num_frames, t_patch_size
        )
        
        num_patches = self.patch_embed.num_patches
        input_size = self.patch_embed.input_size
        self.input_size = input_size
        
        # Position embeddings
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        else:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            
        # Transformer blocks
        self.blocks = nn.ModuleList([
            video_vit.Block(
                embed_dim, num_heads, mlp_ratio,
                qkv_bias=True, qk_scale=None, norm_layer=norm_layer
            ) for _ in range(depth)
        ])
        
        self.norm = norm_layer(embed_dim)
        
        # Projection heads for contrastive learning
        self.spatial_projector = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, projection_dim)
        )
        
        self.temporal_projector = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, temporal_projection_dim)
        )
        
        self.initialize_weights()
        
    def initialize_weights(self):
        # Initialize position embeddings
        torch.nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.use_cls_token:
            torch.nn.init.trunc_normal_(self.cls_token, std=0.02)
            
        # Initialize patch embedding
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        
        # Initialize other layers
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward_encoder(self, x):
        """Extract features from video input"""
        # Patch embedding
        x = self.patch_embed(x)  # [N, T, L, C]
        N, T, L, C = x.shape
        x = x.reshape(N, T * L, C)  # [N, T*L, C]
        
        # Add cls token if used
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(N, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
        
        # Add position embeddings
        x = x + self.pos_embed
        
        # Apply transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        
        return x
    
    def extract_spatial_features(self, x):
        """Extract features for spatial contrastive learning"""
        features = self.forward_encoder(x)
        
        if self.use_cls_token:
            # Use CLS token for global representation
            global_features = features[:, 0]  # [N, embed_dim]
        else:
            # Use mean pooling
            global_features = features.mean(dim=1)  # [N, embed_dim]
            
        return global_features
    
    def extract_temporal_features(self, x):
        """Extract features for temporal contrastive learning"""
        features = self.forward_encoder(x)
        
        if self.use_cls_token:
            # Use CLS token
            temporal_features = features[:, 0]  # [N, embed_dim]
        else:
            # Use mean pooling
            temporal_features = features.mean(dim=1)  # [N, embed_dim]
            
        return temporal_features
    
    def forward(self, x_i, x_j=None):
        """
        Forward pass for contrastive learning
        Args:
            x_i: First view [N, C, T, H, W]
            x_j: Second view [N, C, T, H, W] (optional, for dual input)
        Returns:
            Dictionary containing spatial and temporal features
        """
        if x_j is None:
            # Single input mode
            h_i = self.extract_spatial_features(x_i)
            z_i = self.spatial_projector(h_i)
            
            # For temporal features, use the same features but different projector
            temporal_h_i = self.extract_temporal_features(x_i)
            temporal_z_i = self.temporal_projector(temporal_h_i)
            
            return {
                'spatial_features': h_i,
                'spatial_projections': z_i,
                'temporal_features': temporal_h_i,
                'temporal_projections': temporal_z_i
            }
        else:
            # Dual input mode (SimCLR style)
            h_i = self.extract_spatial_features(x_i)
            h_j = self.extract_spatial_features(x_j)
            z_i = self.spatial_projector(h_i)
            z_j = self.spatial_projector(h_j)
            
            # Temporal features
            temporal_h_i = self.extract_temporal_features(x_i)
            temporal_h_j = self.extract_temporal_features(x_j)
            temporal_z_i = self.temporal_projector(temporal_h_i)
            temporal_z_j = self.temporal_projector(temporal_h_j)
            
            return {
                'h_i': h_i, 'h_j': h_j,
                'z_i': z_i, 'z_j': z_j,
                'temporal_h_i': temporal_h_i, 'temporal_h_j': temporal_h_j,
                'temporal_z_i': temporal_z_i, 'temporal_z_j': temporal_z_j
            }