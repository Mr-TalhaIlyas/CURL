import argparse
import sys
import os
sys.path.append(os.path.dirname(__file__))

from configs.config import config
from main_simclr import *  # Import everything from main_simclr

def parse_args():
    parser = argparse.ArgumentParser(description='Dual Contrastive Learning for Fetal Movement Detection')
    
    # Dual loss configuration
    parser.add_argument('--enable_temporal_loss', action='store_true', 
                       help='Enable temporal contrastive loss')
    parser.add_argument('--spatial_loss_weight', type=float, default=1.0,
                       help='Weight for spatial NT-Xent loss')
    parser.add_argument('--temporal_loss_weight', type=float, default=0.5,
                       help='Weight for temporal contrastive loss')
    parser.add_argument('--temperature_spatial', type=float, default=0.5,
                       help='Temperature for spatial loss')
    parser.add_argument('--temperature_temporal', type=float, default=0.1,
                       help='Temperature for temporal loss')
    parser.add_argument('--tc_clusters', type=int, default=8,
                       help='Number of clusters for temporal loss')
    parser.add_argument('--dual_loss_mode', type=str, default='both',
                       choices=['spatial_only', 'temporal_only', 'both'],
                       help='Which losses to use')
    
    # Training configuration  
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=None,
                       help='Learning rate')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Update config with command line arguments
    if args.epochs is not None:
        config['epochs'] = args.epochs
    if args.batch_size is not None:
        config['batch_size'] = args.batch_size
    if args.lr is not None:
        config['learning_rate'] = args.lr
        
    # Update dual loss configuration
    config['enable_temporal_loss'] = args.enable_temporal_loss
    config['spatial_loss_weight'] = args.spatial_loss_weight
    config['temporal_loss_weight'] = args.temporal_loss_weight
    config['temperature_spatial'] = args.temperature_spatial
    config['temperature_temporal'] = args.temperature_temporal
    config['tc_clusters'] = args.tc_clusters
    config['dual_loss_mode'] = args.dual_loss_mode
    
    print("Updated configuration:")
    print(f"  Enable temporal loss: {config['enable_temporal_loss']}")
    print(f"  Dual loss mode: {config['dual_loss_mode']}")
    print(f"  Spatial weight: {config['spatial_loss_weight']}")
    print(f"  Temporal weight: {config['temporal_loss_weight']}")
    
    # Run training (the main training loop from main_simclr.py will execute)