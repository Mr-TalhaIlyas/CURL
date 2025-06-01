
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
from functools import partial
import cv2, random, math, datetime, time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 150
from termcolor import cprint
from tqdm import tqdm

from data.simclr_loader import GEN_DATA_LISTS, SimCLR_FetalDataset
from data.utils import custom_collate_fn, values_fromreport, print_formatted_table

# from models.simclr_slowfast import SimCLR, SlowFast_Encoder
from models.mae.model_mae import MaskedAutoencoderViT
from models.utils.lr_scheduler import LR_Scheduler
from models.utils.tools import save_chkpt, add_weight_decay

from sklearn.metrics import confusion_matrix, classification_report

from models.utils.visualization import display_video, display_video_lbl, show_batch_frames
from tsaug.visualization import plot
from IPython.display import HTML
from tools.inference import print_classification_report
from models.utils.tools import NativeScalerWithGradNormCount as NativeScaler
from data.utils import video_transform
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

model = MaskedAutoencoderViT(img_size=224,
                                patch_size=16,
                                in_chans=3,
                                num_frames=16,
                                t_patch_size=2,
                                embed_dim=1024,
                                depth=24,
                                num_heads=16,
                                mlp_ratio=4,
                                norm_layer=partial(nn.LayerNorm, eps=1e-6))
model.to(DEVICE)

param_groups = add_weight_decay(model,
                                config['mae']['weight_decay'],
                                bias_wd=config['mae']['bias_wd'])

optimizer = optimizer =  torch.optim._multi_tensor.AdamW(
                                param_groups,
                                lr=config['mae']['blr'],  # No global weight decay; use parameter-specific decay
                                weight_decay=0.0,
                                betas=(0.9, 0.95))

scheduler = LR_Scheduler(config['lr_schedule'], config['mae']['blr'], config['epochs'],
                         iters_per_epoch=len(train_loader), warmup_epochs=config['mae']['warmup_epochs'])

# accuracy = Accuracy(task="multiclass", num_classes=num_classes)
loss_scaler = NativeScaler(fp32=config['mae']['fp32'])

# trainer = MAEViT_Trainer(model, optimizer)

#%%
# Initializing plots
if config['LOG_WANDB']:
    wandb.watch(model, log='parameters', log_freq=100)
    wandb.log({ "total_loss": 1, "learning_rate": 0}, step=0)
    
#%%
# num_repeats_per_epoch = config['num_repeats_per_epoch']
 
start_epoch = 0
epoch, best_loss = 0, 5
total_avg_acc = []

# get time for each epoch

for epoch in range(start_epoch, config['epochs']):
    start_time = time.time()
    model.train() # <-set mode important
    trloss = []
    
    for step, data_batch in enumerate(train_loader):

        optimizer.zero_grad()
        scheduler(optimizer, step, epoch)

        vid_i = video_transform(data_batch['vid_i']).to(DEVICE)
        vid_j = video_transform(data_batch['vid_j']).to(DEVICE)
        vid = torch.cat([vid_i, vid_j], dim=0)

        with torch.cuda.amp.autocast(enabled=not config['mae']['fp32']):
            loss, _, _ = model(vid, config['mae']['mask_ratio'])
        
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            raise Exception("Loss is {}, stopping training".format(loss_value))

        loss_scaler(loss,
                    optimizer, 
                    clip_grad=config['mae']['clip_grad'],
                    parameters=model.parameters(),
                    update_grad=True # (data_iter_step + 1) % accum_iter == 0 -> for DDP
                    )

        # for printing
        trloss.append(loss_value)
        # print(loss)
    print(f'Epoch: {epoch+1}/{config["epochs"]}=> Average loss: {np.nanmean(trloss):.4f}')
    
    mean_loss = np.nanmean(trloss)
    current_loss = mean_loss  # np.nanmean(test_acc)
    if current_loss < best_loss and epoch != 0:
        best_loss = current_loss
        chkpt = save_chkpt(model, optimizer, epoch, loss=np.nanmean(trloss),
                            acc=0, return_chkpt=True)
        print(40*'$')
        
    if config['LOG_WANDB']:
        wandb.log({"total_loss": np.nanmean(trloss),
                   "learning_rate": optimizer.param_groups[0]['lr']}, step=epoch+1)
    print(f'learning rate: {optimizer.param_groups[0]["lr"]}')

    end_time = time.time()
    seconds = end_time - start_time
    print(f'Time taken for epoch: {str(datetime.timedelta(seconds=seconds))}')

if config['LOG_WANDB']:
    wandb.run.finish()
#%%





































