from configs.config import config
import cv2, os, random
from tqdm import tqdm
import numpy as np
from termcolor import cprint
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from models.utils.loss import get_loss
from data.utils import video_transform
from tools.nt_xnet import NT_Xent
from tools.tc_loss import temporal_contrastive_loss
from argparse import Namespace

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def dprint(input, debug=config['DEBUG']):
    if debug:
        print(input)


class DualContrastiveTrainer(object):
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
        
        print(f"Initialized DualContrastiveTrainer:")
        print(f"  - Spatial loss weight: {config['spatial_loss_weight']}")
        print(f"  - Temporal loss weight: {config['temporal_loss_weight']}")
        print(f"  - Temporal loss enabled: {config['enable_temporal_loss']}")
        print(f"  - Dual loss mode: {config['dual_loss_mode']}")
    
    def extract_temporal_features(self, vid_i, vid_j):
        """
        Extract temporal features for contrastive learning.
        This method should be adapted based on your model architecture.
        """
        # Option 1: Use encoder features directly
        with torch.no_grad():
            # Get encoder representations (before projection head)
            h_i, _, _, _ = self.model.forward(vid_i, vid_j)
            h_j, _, _, _ = self.model.forward(vid_j, vid_i)
            
        # Use the encoder features as temporal features
        temporal_features1 = h_i.detach()  # Detach to prevent gradient flow
        temporal_features2 = h_j.detach()
        
        return temporal_features1, temporal_features2
    
    def training_step(self, batched_data):
        """
        Performs one training step with dual contrastive losses.
        """
        # Prepare data
        vid_i = video_transform(batched_data['vid_i']).to(DEVICE, non_blocking=True)
        vid_j = video_transform(batched_data['vid_j']).to(DEVICE, non_blocking=True)
        
        self.model.zero_grad()
        self.optimizer.zero_grad()
        
        total_loss = 0.0
        loss_components = {}
        
        # Forward pass for spatial contrastive loss
        h_i, h_j, z_i, z_j = self.model.forward(vid_i, vid_j)
        
        # --- Spatial Contrastive Loss (NT-Xent) ---
        if self.config['dual_loss_mode'] in ['spatial_only', 'both']:
            spatial_loss = self.spatial_criterion(z_i, z_j)
            weighted_spatial_loss = self.config['spatial_loss_weight'] * spatial_loss
            total_loss += weighted_spatial_loss
            loss_components['spatial_loss'] = spatial_loss.item()
        
        # --- Temporal Contrastive Loss ---
        if self.config['enable_temporal_loss'] and self.config['dual_loss_mode'] in ['temporal_only', 'both']:
            try:
                # Extract temporal features (using encoder outputs)
                temporal_features1, temporal_features2 = self.extract_temporal_features(vid_i, vid_j)
                
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
                        print(f"Warning: Temporal loss was NaN/Inf, skipping this component")
                        loss_components['temporal_loss'] = 0.0
                else:
                    # Not enough samples for clustering
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
        
        # Return loss info
        loss_components['total_loss'] = total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss
        return loss_components

class SimCLR_Trainer(object):
    """Keep original SimCLR trainer for backward compatibility"""
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.criterion = NT_Xent(batch_size=config['batch_size'], temperature=0.5, world_size=1)
        
    def training_step(self, batched_data):
        vid_i = video_transform(batched_data['vid_i']).to(DEVICE)
        vid_j = video_transform(batched_data['vid_j']).to(DEVICE)
        
        self.model.zero_grad()
        self.optimizer.zero_grad()

        vid_i = vid_i.cuda(non_blocking=True)
        vid_j = vid_j.cuda(non_blocking=True)

        h_i, h_j, z_i, z_j = self.model.forward(vid_i, vid_j)
        loss = self.criterion(z_i, z_j)
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

class SimCLR_Evaluator(object):
    def __init__(self, model, return_features=False):
        self.model = model
        self.return_features = return_features

    def eval_step(self, batched_data):
        vid = video_transform(batched_data['vid']).to(DEVICE)
        lbl_batch = batched_data['lbl']
        
        self.model.eval()
        with torch.no_grad():
            h_i, _, _, _ = self.model.forward(vid, vid) 
            
        lbls = lbl_batch
        return h_i.cpu().detach().numpy(), lbls