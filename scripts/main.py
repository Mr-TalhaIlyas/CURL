
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

from data.dataloader import GEN_DATA_LISTS, FetalMovementDataset
from data.utils import collate, values_fromreport, print_formatted_table

from models.slowfast import SlowFast
from models.vidmae import VidMAE
from models.lstm import CNN_LSTM
from models.utils.lr_scheduler import LR_Scheduler
from models.utils.tools import save_chkpt, load_pretrained_MAE_chkpt
from tools.training import Trainer, Evaluator

from sklearn.metrics import confusion_matrix, classification_report

from models.utils.visualization import display_video, display_video_lbl
from tsaug.visualization import plot
from IPython.display import HTML
from tools.inference import print_classification_report
#%%
num_classes = config['num_classes']
sub_classes = 1
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data = GEN_DATA_LISTS(config)

train_data, test_data = data.get_folds(config['num_fold'])
if config['sanity_check']:
    data.chk_paths(train_data)
    data.chk_paths(test_data)

train_dataset = FetalMovementDataset(train_data, config, augment=True)

train_loader = DataLoader(train_dataset,
                        batch_size=config['batch_size'], shuffle=True,
                        num_workers=config['num_workers'], drop_last=True,
                        collate_fn=None, pin_memory=config['pin_memory'],
                        prefetch_factor=2, persistent_workers=True,
                        # sampler=BiasedSampler(train_dataset)
                        )

test_dataset = FetalMovementDataset(test_data, config, augment=False)
test_loader = DataLoader(test_dataset,
                        batch_size=config['batch_size'], shuffle=False,
                        num_workers=config['num_workers'], drop_last=True,
                        collate_fn=None, pin_memory=config['pin_memory'],
                        prefetch_factor=2, persistent_workers=True,
                        # sampler=BiasedSampler(train_dataset)
                        )


#%%
if config['sanity_check']:
    # DataLoader Sanity Checks
    batch = next(iter(train_loader))
    # x=batch['vid'].reshape(-1,224,224,3)
    # HTML(display_video(x).to_html5_video())
    HTML(display_video(batch['vid'][0]).to_html5_video())
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
model = SlowFast(config['model'])
# model = CNN_LSTM(num_classes)
# model = VidMAE(config['model'])
model.to(DEVICE)

# ema_model = SlowFast(config['model'])
# ema_model.to(DEVICE)    

optimizer = torch.optim.AdamW([{'params': model.parameters(),
                                'lr':config['learning_rate']}],
                                weight_decay=config['WEIGHT_DECAY'])

scheduler = LR_Scheduler(config['lr_schedule'], config['learning_rate'], config['epochs'],
                         iters_per_epoch=len(train_loader), warmup_epochs=config['warmup_epochs'])

# accuracy = Accuracy(task="multiclass", num_classes=num_classes)

trainer = Trainer(model, optimizer)
evaluator = Evaluator(model)
#%%
# Initializing plots
if config['LOG_WANDB']:
    wandb.watch(model, log='parameters', log_freq=100)
    wandb.log({ "Acc": 0, "total_loss": 10,
                "Test Acc": 0,"learning_rate": 0}, step=0)
    
#%%
# num_repeats_per_epoch = config['num_repeats_per_epoch']
 
start_epoch = 0
epoch, best_auc = 0, 0
total_avg_acc = []

for epoch in range(start_epoch, config['epochs']):
     
    model.train() # <-set mode important
    trloss = []
    # Repeat the training data num_repeats_per_epoch times before moving to the next epoch
    tracc = []
    
    for step, data_batch in enumerate(train_loader):
        scheduler(optimizer, step, epoch)
        loss, acc = trainer.training_step(data_batch)
        # for printing
        trloss.append(loss)
        tracc.append(acc)

    print(f'Epoch: {epoch+1}/{config["epochs"]}=> Average loss: {np.nanmean(trloss):.4f}, Average Acc: {np.nanmean(tracc):.4f}')
    # fix following to infer only every 2nd epoch
    test_acc, all_preds, all_lbls = [], [], [] 
    if (epoch + 1) % 2 == 0:  # eval every N epoch
        print('Validating...')
        
        for _, data_batch in enumerate(test_loader):
            acc, preds, lbls = evaluator.eval_step(data_batch)
            test_acc.append(acc)
            all_preds.append(preds)
            all_lbls.append(lbls)

        if config['LOG_WANDB']:
            wandb.log({ "Test Acc": np.nanmean(test_acc),
                      }, step=epoch+1)
        
        all_preds = np.concatenate(all_preds, axis=0)
        all_lbls = np.concatenate(all_lbls, axis=0)
        
        auc = print_classification_report(all_preds, all_lbls)

        current_auc = auc  # np.nanmean(test_acc)
        if current_auc > best_auc and epoch != 0:
                best_auc = current_auc
                chkpt = save_chkpt(model, optimizer, epoch, loss=np.nanmean(trloss),
                                    acc=current_auc, return_chkpt=True)
                print(40*'$')
        
    if config['LOG_WANDB']:
        wandb.log({"total_loss": np.nanmean(trloss), 
                   "Acc": np.nanmean(tracc),
                   "learning_rate": optimizer.param_groups[0]['lr']}, step=epoch+1)
            

if config['LOG_WANDB']:
    wandb.run.finish()
#%%










































#%%
# from tqdm import trange
# train_dataset = Fetal_Loader(train_data, config, augment=False)

# train_loader = DataLoader(train_dataset,
#                           batch_size=1, shuffle=False,
#                         num_workers=config['num_workers'], drop_last=True,
#                         collate_fn=None, pin_memory=config['pin_memory'],
#                         prefetch_factor=2, persistent_workers=True,
#                         # sampler=BiasedSampler(train_dataset)
#                         )
# for i in range(500):
#     for step, data_batch in enumerate(train_loader):
#         vid = data_batch['vid']
#         x = vid.shape
#         print(data_batch['filename'], data_batch['lbl'])
# # %%
# # clear torch cuda memory
# torch.cuda.empty_cache()
# %%
