import os, psutil, time, datetime, math
os.chdir('/home/user01/Data/fetal/new_scripts/')
from configs.config import config

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = config['gpus_to_use']

if config['LOG_WANDB']:
    import wandb
    wandb.init(dir=config['log_directory'],
               project=config['project_name'], 
               name=f"{config['experiment_name']}_mae_contrastive",
               config_include_keys=config.keys(), config=config)

import pprint
print(f'Printing Configuration File:\n{30*"="}\n')
pprint.pprint(config)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from functools import partial
import numpy as np
from tqdm import tqdm

from data.simclr_loader import GEN_DATA_LISTS, SimCLR_FetalDataset
from data.utils import custom_collate_fn
from models.mae.cont_mae import ContrastiveMAE
from tools.mae_cont_trainer import MAEContrastiveTrainer
from models.utils.lr_scheduler import LR_Scheduler
from models.utils.tools import save_chkpt
from models.utils.tools import NativeScalerWithGradNormCount as NativeScaler

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data loading
data = GEN_DATA_LISTS(config)
train_data, test_data = data.get_folds(config['num_fold'])
full_data = data.merge_paths(train_data, test_data)

if config['sanity_check']:
    data.chk_paths(train_data)

train_dataset = SimCLR_FetalDataset(full_data, config, augment=True)
train_loader = DataLoader(train_dataset,
                        batch_size=config['batch_size'], 
                        shuffle=True,
                        num_workers=config['num_workers'], 
                        drop_last=True,
                        collate_fn=custom_collate_fn, 
                        pin_memory=config['pin_memory'],
                        prefetch_factor=2, 
                        persistent_workers=True)

print(f"Train Loader: {len(train_loader)} iterations")

# Model initialization
model = ContrastiveMAE(**config['mae_contrastive'])
model.to(DEVICE)

# Optimizer and scheduler
param_groups = [{'params': model.parameters(), 'lr': config['learning_rate']}]
optimizer = torch.optim.AdamW(param_groups, 
                             lr=config['learning_rate'],
                             weight_decay=config['WEIGHT_DECAY'])

scheduler = LR_Scheduler(config['lr_schedule'], 
                        config['learning_rate'], 
                        config['epochs'],
                        iters_per_epoch=len(train_loader), 
                        warmup_epochs=config['warmup_epochs'])

# Loss scaler for mixed precision
loss_scaler = NativeScaler(fp32=True)

# Initialize trainer
trainer = MAEContrastiveTrainer(model, optimizer, config)

# Logging
if config['LOG_WANDB']:
    wandb.watch(model, log='parameters', log_freq=100)
    wandb.log({"total_loss": 1, "learning_rate": 0}, step=0)

# Training loop
start_epoch = 0
best_loss = float('inf')

for epoch in range(start_epoch, config['epochs']):
    start_time = time.time()
    model.train()
    
    epoch_losses = []
    spatial_losses = []
    temporal_losses = []
    
    for step, data_batch in enumerate(train_loader):
        # Update learning rate
        scheduler(optimizer, step, epoch)
        
        # Training step
        loss_components = trainer.training_step(data_batch)
        
        # Collect losses
        epoch_losses.append(loss_components['total_loss'])
        if 'spatial_loss' in loss_components:
            spatial_losses.append(loss_components['spatial_loss'])
        if 'temporal_loss' in loss_components:
            temporal_losses.append(loss_components['temporal_loss'])
    
    # Calculate averages
    avg_total_loss = np.nanmean(epoch_losses)
    log_message = f'Epoch: {epoch+1}/{config["epochs"]} => Total Loss: {avg_total_loss:.4f}'
    
    if spatial_losses:
        avg_spatial_loss = np.nanmean(spatial_losses)
        log_message += f', Spatial Loss: {avg_spatial_loss:.4f}'
        
    if temporal_losses:
        avg_temporal_loss = np.nanmean(temporal_losses)
        log_message += f', Temporal Loss: {avg_temporal_loss:.4f}'
    
    print(log_message)
    
    # Save best model
    if avg_total_loss < best_loss and epoch > 0:
        best_loss = avg_total_loss
        chkpt = save_chkpt(model, optimizer, epoch, 
                          loss=avg_total_loss, acc=0, 
                          return_chkpt=True)
        print(40*'$')
    
    # Wandb logging
    if config['LOG_WANDB']:
        log_dict = {
            "total_loss": avg_total_loss,
            "learning_rate": optimizer.param_groups[0]['lr']
        }
        if spatial_losses:
            log_dict["spatial_loss"] = avg_spatial_loss
        if temporal_losses:
            log_dict["temporal_loss"] = avg_temporal_loss
            
        wandb.log(log_dict, step=epoch+1)
    
    # Time tracking
    end_time = time.time()
    seconds = end_time - start_time
    print(f'Time per epoch: {str(datetime.timedelta(seconds=seconds))}')
    print(f'Learning rate: {optimizer.param_groups[0]["lr"]:.6f}')

if config['LOG_WANDB']:
    wandb.run.finish()

print("Training completed!")