import torch
import torch.nn as nn
import numpy as np
from argparse import Namespace
from data.utils import video_transform
from tools.nt_xnet import NT_Xent
from tools.tc_loss import temporal_contrastive_loss

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MAEContrastiveTrainer(object):
    def __init__(self, model, optimizer, config):
        self.model = model
        self.optimizer = optimizer
        self.config = config
        
        # Initialize spatial contrastive loss (NT-Xent)
        self.spatial_criterion = NT_Xent(
            batch_size=config['batch_size'], 
            temperature=config['temperature_spatial'], 
            world_size=1
        )
        
        # Prepare temporal loss arguments
        self.temporal_args = Namespace(
            clusters=config['tc_clusters'],
            num_iters=config['tc_num_iters'],
            do_entro=config['tc_do_entro'],
            default_device=DEVICE
        )
        
        print(f"Initialized MAEContrastiveTrainer:")
        print(f"  - Model: ContrastiveMAE with {config['mae_contrastive']['embed_dim']} dim")
        print(f"  - Spatial loss weight: {config['spatial_loss_weight']}")
        print(f"  - Temporal loss weight: {config['temporal_loss_weight']}")
        print(f"  - Dual loss mode: {config['dual_loss_mode']}")
    
    def training_step(self, batched_data):
        """
        Training step with dual contrastive losses using MAE encoder
        """
        # Prepare data
        vid_i = video_transform(batched_data['vid_i']).to(DEVICE, non_blocking=True)
        vid_j = video_transform(batched_data['vid_j']).to(DEVICE, non_blocking=True)
        
        self.model.zero_grad()
        self.optimizer.zero_grad()
        
        total_loss = 0.0
        loss_components = {}
        
        # Forward pass through ContrastiveMAE
        outputs = self.model.forward(vid_i, vid_j)
        
        # --- Spatial Contrastive Loss (NT-Xent) ---
        if self.config['dual_loss_mode'] in ['spatial_only', 'both']:
            z_i = outputs['z_i']  # Spatial projections
            z_j = outputs['z_j']
            spatial_loss = self.spatial_criterion(z_i, z_j)
            weighted_spatial_loss = self.config['spatial_loss_weight'] * spatial_loss
            total_loss += weighted_spatial_loss
            loss_components['spatial_loss'] = spatial_loss.item()
        
        # --- Temporal Contrastive Loss ---
        if (self.config['enable_temporal_loss'] and 
            self.config['dual_loss_mode'] in ['temporal_only', 'both']):
            try:
                # Use temporal features (encoder features)
                temporal_features1 = outputs['temporal_h_i']  # [N, embed_dim]
                temporal_features2 = outputs['temporal_h_j']  # [N, embed_dim]
                
                # Ensure sufficient samples for clustering
                if (temporal_features1.shape[0] >= self.config['tc_clusters'] and 
                    temporal_features2.shape[0] >= self.config['tc_clusters']):
                    
                    temporal_loss = temporal_contrastive_loss(
                        temporal_features1,
                        temporal_features2,
                        self.config['temperature_temporal'],
                        self.temporal_args
                    )
                    
                    if not torch.isnan(temporal_loss) and not torch.isinf(temporal_loss):
                        weighted_temporal_loss = self.config['temporal_loss_weight'] * temporal_loss
                        total_loss += weighted_temporal_loss
                        loss_components['temporal_loss'] = temporal_loss.item()
                    else:
                        print(f"Warning: Temporal loss was NaN/Inf, skipping")
                        loss_components['temporal_loss'] = 0.0
                else:
                    loss_components['temporal_loss'] = 0.0
                    
            except Exception as e:
                print(f"Error computing temporal loss: {e}")
                loss_components['temporal_loss'] = 0.0
        
        # Backward pass
        if total_loss > 0:
            total_loss.backward()
            self.optimizer.step()
        else:
            print("Warning: Total loss is 0, skipping optimization step")
        
        loss_components['total_loss'] = total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss
        return loss_components

class MAEContrastiveEvaluator(object):
    def __init__(self, model, return_features=False):
        self.model = model
        self.return_features = return_features

    def eval_step(self, batched_data):
        """Extract features for evaluation"""
        vid = video_transform(batched_data['vid']).to(DEVICE)
        lbl_batch = batched_data['lbl']
        
        self.model.eval()
        with torch.no_grad():
            # Get spatial features for downstream tasks
            outputs = self.model.forward(vid)
            features = outputs['spatial_features']  # Use spatial features for evaluation
            
        return features.cpu().detach().numpy(), lbl_batch