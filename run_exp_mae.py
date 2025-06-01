import argparse
import sys
import os

def parse_args():
    parser = argparse.ArgumentParser(description='MAE Contrastive Learning for Fetal Videos')
    
    # Model parameters
    parser.add_argument('--embed_dim', type=int, default=1024, help='MAE encoder embedding dimension')
    parser.add_argument('--depth', type=int, default=24, help='Number of transformer layers')
    parser.add_argument('--num_heads', type=int, default=16, help='Number of attention heads')
    
    # Contrastive loss parameters
    parser.add_argument('--enable_temporal_loss', action='store_true', help='Enable temporal contrastive loss')
    parser.add_argument('--spatial_loss_weight', type=float, default=1.0, help='Weight for spatial loss')
    parser.add_argument('--temporal_loss_weight', type=float, default=0.5, help='Weight for temporal loss')
    parser.add_argument('--temperature_spatial', type=float, default=0.5, help='Temperature for spatial loss')
    parser.add_argument('--temperature_temporal', type=float, default=0.1, help='Temperature for temporal loss')
    parser.add_argument('--dual_loss_mode', type=str, default='both', 
                       choices=['spatial_only', 'temporal_only', 'both'])
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Update config
    sys.path.append(os.path.dirname(__file__))
    from configs.config import config
    
    # Update configuration with command line arguments
    if hasattr(args, 'embed_dim'):
        config['mae_contrastive']['embed_dim'] = args.embed_dim
    if hasattr(args, 'depth'):
        config['mae_contrastive']['depth'] = args.depth
    
    config['enable_temporal_loss'] = args.enable_temporal_loss
    config['spatial_loss_weight'] = args.spatial_loss_weight
    config['temporal_loss_weight'] = args.temporal_loss_weight
    config['dual_loss_mode'] = args.dual_loss_mode
    
    if args.epochs:
        config['epochs'] = args.epochs
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.lr:
        config['learning_rate'] = args.lr
    
    print("Running MAE Contrastive Learning with configuration:")
    print(f"  - Dual loss mode: {config['dual_loss_mode']}")
    print(f"  - Spatial weight: {config['spatial_loss_weight']}")
    print(f"  - Temporal weight: {config['temporal_loss_weight']}")
    
    # Import and run main script
    exec(open('main_mae_cont.py').read())