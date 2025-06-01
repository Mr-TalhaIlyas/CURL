import os, psutil, math, gc
import numpy as np
from collections import Counter
from functools import partial

# Set working directory and configuration
os.chdir('/home/user01/Data/fetal/new_scripts/')
from configs.config import config

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = config['gpus_to_use']

# Wandb logging setup
if config['LOG_WANDB']:
    import wandb
    wandb.init(
        dir=config['log_directory'],
        project=config['project_name'], 
        name=f"{config['experiment_name']}_finetune",
        config_include_keys=config.keys(), 
        config=config
    )

import pprint
print(f'Printing Configuration File:\n{30*"="}\n')
pprint.pprint(config)

# Import libraries
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchmetrics import Accuracy
import matplotlib.pyplot as plt
from termcolor import cprint
from tqdm import tqdm

# Project imports
from data.dataloader import GEN_DATA_LISTS, FetalMovementDataset
from data.utils import print_formatted_table
from models.mae.model_vit import VisionTransformer
from models.contrastive_mae import ContrastiveMAE
from models.utils.lr_scheduler import LR_Scheduler, CyclicLR
from models.utils.tools import (save_chkpt, load_pretrained_MAE_chkpt, 
                               param_groups_lrd, load_chkpt)
from models.utils.loss import FocalLoss
from tools.training import Trainer, Evaluator
from tools.inference import print_classification_report
from models.utils.tools import NativeScalerWithGradNormCount as NativeScaler
from data.utils import video_transform
from timm.loss import LabelSmoothingCrossEntropy

print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ContrastiveFineTuner:
    """Enhanced fine-tuning class that supports both standard MAE and contrastive pre-trained models"""
    
    def __init__(self, config):
        self.config = config
        self.device = DEVICE
        self.setup_data()
        self.setup_model()
        self.setup_training()
        
    def setup_data(self):
        """Setup data loaders with class balancing"""
        print("Setting up data loaders...")
        
        # Load data splits
        data = GEN_DATA_LISTS(self.config)
        train_data, test_data = data.get_folds(self.config['num_fold'])
        
        if self.config['sanity_check']:
            data.chk_paths(train_data)
            data.chk_paths(test_data)
        
        # Create datasets
        train_dataset = FetalMovementDataset(train_data, self.config, augment=True)
        test_dataset = FetalMovementDataset(test_data, self.config, augment=False)
        
        # Setup class balancing
        labels = train_dataset.get_labels()
        class_counts = Counter(labels)
        total_samples = len(labels)
        num_classes = len(class_counts)
        
        # Calculate class weights
        class_weights = {}
        for cls in class_counts:
            class_weights[cls] = total_samples / (num_classes * class_counts[cls])
        
        print(f'Class distribution: {dict(class_counts)}')
        print(f'Class weights: {class_weights}')
        
        # Create weighted sampler
        weights = [class_weights[label] for label in labels]
        weights = torch.DoubleTensor(weights)
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,  # Use sampler instead
            num_workers=self.config['num_workers'],
            drop_last=True,
            pin_memory=self.config['pin_memory'],
            prefetch_factor=self.config['prefetch_factor'],
            persistent_workers=self.config['persistent_workers'],
            sampler=sampler
        )
        
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config['num_workers'],
            drop_last=True,
            pin_memory=self.config['pin_memory'],
            prefetch_factor=self.config['prefetch_factor'],
            persistent_workers=self.config['persistent_workers']
        )
        
        print(f"Train Loader: {len(self.train_loader)} iterations")
        print(f"Test Loader: {len(self.test_loader)} iterations")
        
    def setup_model(self):
        """Setup model with pre-trained weights"""
        print("Setting up model...")
        
        # Determine model type based on config
        model_type = self.config.get('finetune_model_type', 'standard_mae')  # 'standard_mae' or 'contrastive_mae'
        
        if model_type == 'contrastive_mae':
            print("Loading ContrastiveMAE for fine-tuning...")
            # Create the contrastive model first
            base_model = ContrastiveMAE(**self.config['mae_contrastive'])
            
            # Extract the encoder and add classification head
            self.model = ContrastiveMAEClassifier(
                encoder=base_model,
                num_classes=self.config['num_classes'],
                dropout=self.config['mae_finetune']['dropout']
            )
            
            # Load contrastive pre-trained weights
            checkpoint_path = self.config.get('contrastive_checkpoint_path', None)
            if checkpoint_path and os.path.exists(checkpoint_path):
                print(f"Loading contrastive pre-trained weights from: {checkpoint_path}")
                self.load_contrastive_checkpoint(checkpoint_path)
            else:
                print("No contrastive checkpoint specified or file not found")
                
        else:
            print("Loading standard MAE ViT for fine-tuning...")
            # Standard MAE ViT model
            self.model = VisionTransformer(
                img_size=224,
                patch_size=16,
                in_chans=3,
                num_frames=16,
                t_patch_size=2,
                embed_dim=1024,
                depth=24,
                num_heads=16,
                mlp_ratio=4,
                drop_path_rate=self.config['mae_finetune']['drop_path_rate'],
                dropout=self.config['mae_finetune']['dropout'],
                num_classes=self.config['num_classes'],
                norm_layer=partial(nn.LayerNorm, eps=1e-6)
            )
            
            # Load standard MAE pre-trained weights
            mae_checkpoint_path = self.config.get('mae_checkpoint_path', '/home/user01/Data/fetal/chkpts/MAE_ViT_2.pth')
            if os.path.exists(mae_checkpoint_path):
                print(f"Loading MAE pre-trained weights from: {mae_checkpoint_path}")
                load_info = load_pretrained_MAE_chkpt(self.model, mae_checkpoint_path)
                matched, unmatched, extra_pretrained, extra_model = load_info
                print(f"Matched layers: {len(matched)}, Unmatched: {len(unmatched)}")
            else:
                print("MAE checkpoint not found, training from scratch")
        
        self.model.to(self.device)
        print(f"Model moved to device: {self.device}")
        
    def load_contrastive_checkpoint(self, checkpoint_path):
        """Load checkpoint from contrastive pre-training"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # Load encoder weights (excluding classification head)
            model_dict = self.model.state_dict()
            pretrained_dict = {}
            
            for k, v in state_dict.items():
                # Skip classification head and projection heads
                if any(skip in k for skip in ['head', 'spatial_projector', 'temporal_projector']):
                    continue
                    
                # Handle potential key mismatches
                if k in model_dict:
                    pretrained_dict[k] = v
                elif k.replace('encoder.', '') in model_dict:
                    pretrained_dict[k.replace('encoder.', '')] = v
                elif 'encoder.' + k in model_dict:
                    pretrained_dict['encoder.' + k] = v
            
            model_dict.update(pretrained_dict)
            self.model.load_state_dict(model_dict, strict=False)
            
            print(f"Loaded {len(pretrained_dict)} layers from contrastive checkpoint")
            
        except Exception as e:
            print(f"Error loading contrastive checkpoint: {e}")
            print("Proceeding with random initialization...")
            
    def setup_training(self):
        """Setup optimizer, scheduler, and loss function"""
        print("Setting up training components...")
        
        # Layer-wise learning rate decay
        param_groups = param_groups_lrd(
            self.model, 
            self.config['mae_finetune']['weight_decay'],
            no_weight_decay_list=getattr(self.model, 'no_weight_decay', lambda: [])(),
            layer_decay=self.config['mae_finetune']['layer_decay']
        )
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            param_groups, 
            lr=self.config['mae_finetune']['blr']
        )
        
        # Mixed precision scaler
        self.loss_scaler = NativeScaler(fp32=self.config['mae_finetune']['fp32'])
        
        # Learning rate scheduler
        scheduler_type = self.config.get('scheduler_type', 'cyclic')
        
        if scheduler_type == 'cyclic':
            self.scheduler = CyclicLR(
                self.optimizer,
                step_size=len(self.train_loader) * 5,
                base_lr=1e-6,
                max_lr=self.config['mae_finetune']['blr'],
                mode='triangular'
            )
        else:
            self.scheduler = LR_Scheduler(
                self.config['lr_schedule'], 
                self.config['mae_finetune']['blr'],
                self.config['mae_finetune']['epochs'],
                iters_per_epoch=len(self.train_loader),
                warmup_epochs=self.config['mae_finetune']['warmup_epochs']
            )
        
        # Loss function
        loss_type = self.config.get('loss_type', 'focal')
        
        if loss_type == 'focal':
            self.criterion = FocalLoss(
                alpha=torch.tensor([0.6, 3.0]).to(self.device),
                gamma=2, 
                reduction='mean'
            )
        elif loss_type == 'cross_entropy':
            self.criterion = nn.CrossEntropyLoss(
                weight=torch.tensor([0.6, 3.0]).to(self.device),
                label_smoothing=self.config['mae_finetune']['smoothing']
            )
        else:
            self.criterion = LabelSmoothingCrossEntropy(
                smoothing=self.config['mae_finetune']['smoothing']
            )
        
        # Metrics
        self.accuracy = Accuracy(task="multiclass", num_classes=self.config['num_classes'])
        self.evaluator = Evaluator(self.model)
        
        print(f"Using {loss_type} loss and {scheduler_type} scheduler")
        
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        epoch_losses = []
        epoch_accs = []
        
        accum_iter = self.config['mae_finetune']['accum_iter']
        iters_per_epoch = len(self.train_loader)
        
        self.optimizer.zero_grad()
        
        for step, data_batch in enumerate(self.train_loader):
            # Update learning rate
            if hasattr(self.scheduler, 'step'):
                self.scheduler.step()
            
            # Prepare data
            vid = video_transform(data_batch['vid']).to(self.device)
            lbl_batch = data_batch['lbl'].type(torch.long).to(self.device)
            
            # Forward pass with mixed precision
            with torch.cuda.amp.autocast(enabled=not self.config['mae_finetune']['fp32']):
                preds = self.model(vid)
                loss = self.criterion(preds, lbl_batch)
            
            # Gradient accumulation
            loss = loss / accum_iter
            update_grad = ((step + 1) % accum_iter == 0) or ((step + 1) == iters_per_epoch)
            
            # Backward pass
            self.loss_scaler(
                loss,
                self.optimizer,
                clip_grad=self.config['mae_finetune']['clip_grad'],
                parameters=self.model.parameters(),
                create_graph=False,
                update_grad=update_grad
            )
            
            if update_grad:
                self.optimizer.zero_grad()
            
            # Calculate metrics
            loss_value = loss.item() * accum_iter
            if not math.isfinite(loss_value):
                raise Exception(f"Loss is {loss_value}, stopping training")
            
            acc = self.accuracy(
                preds.softmax(dim=-1).cpu().detach(),
                lbl_batch.cpu().detach()
            ).item()
            
            epoch_losses.append(loss_value)
            epoch_accs.append(acc)
        
        return np.nanmean(epoch_losses), np.nanmean(epoch_accs)
    
    def evaluate(self):
        """Evaluate on test set"""
        test_accs = []
        all_preds = []
        all_lbls = []
        
        for data_batch in self.test_loader:
            acc, preds, lbls = self.evaluator.eval_step(data_batch)
            test_accs.append(acc)
            all_preds.append(preds.argmax(1))
            all_lbls.append(lbls)
        
        all_preds = np.concatenate(all_preds, axis=0)
        all_lbls = np.concatenate(all_lbls, axis=0)
        
        # Calculate AUC and other metrics
        auc = print_classification_report(all_preds, all_lbls)
        avg_acc = np.nanmean(test_accs)
        
        return avg_acc, auc, all_preds, all_lbls
    
    def train(self):
        """Main training loop"""
        print("Starting training...")
        
        # Initialize wandb
        if self.config['LOG_WANDB']:
            wandb.watch(self.model, log='parameters', log_freq=100)
            wandb.log({"Acc": 0, "total_loss": 10, "Test Acc": 0, "learning_rate": 0}, step=0)
        
        best_auc = 0
        start_epoch = 0
        
        for epoch in range(start_epoch, self.config['mae_finetune']['epochs']):
            # Training
            train_loss, train_acc = self.train_epoch(epoch)
            
            print(f'Epoch: {epoch+1}/{self.config["mae_finetune"]["epochs"]} => '
                  f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
            
            # Evaluation (every 2 epochs)
            if (epoch + 1) % 2 == 0:
                print('Evaluating...')
                test_acc, auc, all_preds, all_lbls = self.evaluate()
                
                # Save best model
                if auc > best_auc and epoch > 0:
                    best_auc = auc
                    save_chkpt(self.model, self.optimizer, epoch, 
                             loss=train_loss, acc=auc, return_chkpt=True)
                    print(40*'$' + ' New best model saved!')
                
                # Wandb logging
                if self.config['LOG_WANDB']:
                    wandb.log({
                        "Test Acc": test_acc,
                        "AUC": auc if auc is not None else 0
                    }, step=epoch+1)
            
            # Regular checkpoint saving
            if (epoch + 1) % 5 == 0:
                save_chkpt(self.model, self.optimizer, epoch, 
                         loss=train_loss, acc=train_acc, return_chkpt=True)
            
            # Wandb logging
            if self.config['LOG_WANDB']:
                wandb.log({
                    "total_loss": train_loss,
                    "Acc": train_acc,
                    "learning_rate": self.optimizer.param_groups[0]['lr']
                }, step=epoch+1)
            
            print(f'Learning rate: {self.optimizer.param_groups[0]["lr"]:.6f}')
        
        if self.config['LOG_WANDB']:
            wandb.run.finish()
            
        print("Training completed!")

class ContrastiveMAEClassifier(nn.Module):
    """Classifier built on top of ContrastiveMAE encoder"""
    
    def __init__(self, encoder, num_classes, dropout=0.1):
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(encoder.embed_dim, num_classes)
        
        # Initialize classification head
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.constant_(self.head.bias, 0)
    
    def no_weight_decay(self):
        """Return parameters that should not have weight decay"""
        return getattr(self.encoder, 'no_weight_decay', lambda: [])()
    
    def forward(self, x):
        # Extract spatial features from encoder
        spatial_features = self.encoder.extract_spatial_features(x)
        features = self.dropout(spatial_features)
        return self.head(features)

if __name__ == "__main__":
    # override config here for specific experiments
    # config['finetune_model_type'] = 'contrastive_mae'  # or 'standard_mae'
    # config['contrastive_checkpoint_path'] = '/path/to/contrastive/checkpoint.pth'
    
    trainer = ContrastiveFineTuner(config)
    trainer.train()