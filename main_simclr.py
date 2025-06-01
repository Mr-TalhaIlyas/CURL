
#%%
import os, psutil
# Setting the path for the updated libstdc++
# os.environ['LD_LIBRARY_PATH'] = '/usr/lib/x86_64-linux-gnu:' + os.environ.get('LD_LIBRARY_PATH', '')
# import ctypes
# ctypes.CDLL('libstdc++.so.6', mode=ctypes.RTLD_GLOBAL)
# os.chdir(os.path.dirname(__file__))
os.chdir('/home/user01/Data/fetal/new_scripts/')
from configs.config import config

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"] = config['gpus_to_use'];

# Use an environment variable to override the fold number, if provided
# fold_number = os.getenv('FOLD_NUMBER', config['num_fold'])
# config['num_fold'] = int(fold_number)
# config['experiment_name'] = f"Exp1_fold{config['num_fold']}"

if config['LOG_WANDB']:
    import wandb
    # from datetime import datetime
    # my_id = datetime.now().strftime("%Y%m%d%H%M")
    wandb.init(dir=config['log_directory'],
               project=config['project_name'], name=config['experiment_name'],
            #    resume='allow', id=my_id, # this one introduces werid behaviour in the app
               config_include_keys=config.keys(), config=config)
    # print(f'WANDB config ID : {my_id}')

import pprint
print(f'Printing Configuration File:\n{30*"="}\n')
pprint.pprint(config)


import torch
print(f"PyTorch Version: {torch.__version__}")
if torch.cuda.is_available():
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("CUDA not available")
print(f"Number of CPUs: {os.cpu_count()}")
memory = psutil.virtual_memory()
print(f"Total Memory: {memory.total / (1024**3):.2f} GB") 
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from fmutils import fmutils as fmu

import cv2, random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 150
from termcolor import cprint
from tqdm import tqdm

from data.simclr_loader import GEN_DATA_LISTS, SimCLR_FetalDataset
from data.utils import custom_collate_fn, values_fromreport, print_formatted_table

from models.simclr_slowfast import SimCLR, SlowFast_Encoder
from models.utils.lr_scheduler import LR_Scheduler
from models.utils.tools import save_chkpt
from tools.simclr_training import SimCLR_Trainer, DualContrastiveTrainer


from sklearn.metrics import confusion_matrix, classification_report

from models.utils.visualization import display_video, display_video_lbl, show_batch_frames
from tsaug.visualization import plot
from IPython.display import HTML
from tools.inference import print_classification_report

#%%
num_classes = config['num_classes']
sub_classes = 1
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data = GEN_DATA_LISTS(config)

train_data, test_data = data.get_folds(config['num_fold'])
full_data = data.merge_paths(train_data, test_data)

if config['sanity_check']:
    data.chk_paths(train_data)
    data.chk_paths(test_data)

train_dataset = SimCLR_FetalDataset(full_data, config, augment=True)

train_loader = DataLoader(train_dataset,
                        batch_size=config['batch_size'], shuffle=True,
                        num_workers=config['num_workers'], drop_last=True,
                        collate_fn=custom_collate_fn, pin_memory=config['pin_memory'],
                        prefetch_factor=2, persistent_workers=True,
                        # sampler=BiasedSampler(train_dataset)
                        )


#%%
if config['sanity_check']:
    # DataLoader Sanity Checks
    batch = next(iter(train_loader))
    show_batch_frames(batch, config['batch_size'])
    # x=batch['vid'].reshape(-1,224,224,3)
    # HTML(display_video(x).to_html5_video())
    # HTML(display_video(batch['vid_i'][0]).to_html5_video())
    # x = batch['lbl'][0]
    # print('Classes', x)
    # print('Visualizing a optical flow...')
    # plt.imshow(batch['frames'][0][0,...])

#%%
# NEW
# batch = next(iter(train_loader))
# x=batch['vid'].reshape(-1,224,224,3)
# l=batch['lbl'].numpy()
# HTML(display_video_lbl(x,l).to_html5_video())

# %%

encoder = SlowFast_Encoder(config['model'])
model = SimCLR(encoder)
model.to(DEVICE)

optimizer = torch.optim.AdamW([{'params': model.parameters(),
                                'lr':config['learning_rate']}],
                                weight_decay=config['WEIGHT_DECAY'])

scheduler = LR_Scheduler(config['lr_schedule'], config['learning_rate'], config['epochs'],
                         iters_per_epoch=len(train_loader), warmup_epochs=config['warmup_epochs'])

# Choose trainer based on configuration
if config.get('enable_temporal_loss', False) and config.get('dual_loss_mode', 'both') in ['temporal_only', 'both']:
    print("Using DualContrastiveTrainer")
    trainer = DualContrastiveTrainer(model, optimizer, config)
else:
    print("Using standard SimCLR_Trainer")  
    trainer = SimCLR_Trainer(model, optimizer)
#%%
# Initializing plots
if config['LOG_WANDB']:
    wandb.watch(model, log='parameters', log_freq=100)
    wandb.log({ "total_loss": 10, "learning_rate": 0}, step=0)
    
#%%
# num_repeats_per_epoch = config['num_repeats_per_epoch']
 
start_epoch = 0
epoch, best_loss = 0, 5
total_avg_acc = []

for epoch in range(start_epoch, config['epochs']):
    model.train()
    trloss = []
    spatial_losses = []
    temporal_losses = []
    
    for step, data_batch in enumerate(train_loader):
        scheduler(optimizer, step, epoch)
        
        # Training step - returns different formats based on trainer type
        if isinstance(trainer, DualContrastiveTrainer):
            loss_components = trainer.training_step(data_batch)
            total_loss = loss_components['total_loss']
            trloss.append(total_loss)
            
            # Track individual loss components
            if 'spatial_loss' in loss_components:
                spatial_losses.append(loss_components['spatial_loss'])
            if 'temporal_loss' in loss_components:
                temporal_losses.append(loss_components['temporal_loss'])
                
        else:
            # Standard SimCLR trainer
            loss = trainer.training_step(data_batch)
            trloss.append(loss)
    
    # Logging
    avg_total_loss = np.nanmean(trloss)
    log_message = f'Epoch: {epoch+1}/{config["epochs"]} => Total Loss: {avg_total_loss:.4f}'
    
    if spatial_losses:
        avg_spatial_loss = np.nanmean(spatial_losses)
        log_message += f', Spatial Loss: {avg_spatial_loss:.4f}'
        
    if temporal_losses:
        avg_temporal_loss = np.nanmean(temporal_losses)
        log_message += f', Temporal Loss: {avg_temporal_loss:.4f}'
    
    print(log_message)
    
    # Save best model
    current_loss = avg_total_loss
    if current_loss < best_loss and epoch != 0:
        best_loss = current_loss
        chkpt = save_chkpt(model, optimizer, epoch, loss=avg_total_loss,
                            acc=0, return_chkpt=True)
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

if config['LOG_WANDB']:
    wandb.run.finish()
#%%





































