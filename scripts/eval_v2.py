#%%
import os, psutil
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
from models.utils.tools import save_chkpt, load_chkpt
from tools.training import Trainer, Evaluator

from sklearn.metrics import confusion_matrix, classification_report

from models.utils.visualization import display_video, display_video_lbl
from tsaug.visualization import plot
from IPython.display import HTML
from tools.inference import print_classification_report
import decord
from data.utils import video_transform
from tqdm import trange
from sklearn.metrics import confusion_matrix
#%%

num_classes = config['num_classes']
sub_classes = 1
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

clip_dir = '/home/user01/Data/fetal/clips/'
file_type = 'test'
folds = 5

for f in range(folds):
    num_fold = f+1

    path = f"{config['folds']}/{file_type}_fold_{num_fold}.txt"
    with open(path, 'r') as file:
        samples = [line.strip() for line in file.readlines()]
        
    all_clips = os.listdir(clip_dir)
    clips = [clip for clip in all_clips if clip.split('_')[1] in samples]

    print(f"Number of clips: {len(clips)}")
    # read clips from clips directory

    vr = decord.VideoReader(f"{clip_dir}/{clips[0]}")
    frames = vr.get_batch(list(np.arange(0,48))).asnumpy()
    print(f'{num_fold}: {frames.shape}')

    model = SlowFast(config['model'])
    # model = CNN_LSTM(num_classes)
    # model = VidMAE(config['model'])
    model.to(DEVICE)
    optimizer = torch.optim.AdamW([{'params': model.parameters(),
                                    'lr':config['learning_rate']}],
                                    weight_decay=config['WEIGHT_DECAY'])

    scheduler = LR_Scheduler(config['lr_schedule'], config['learning_rate'], config['epochs'],
                            iters_per_epoch=len(all_clips), warmup_epochs=config['warmup_epochs'])

    # accuracy = Accuracy(task="multiclass", num_classes=num_classes)

    load_chkpt(model, optimizer,
            chkpt_path=f'/home/user01/Data/fetal/chkpts/latest_f{num_fold}_soft.pth')

    preds, lbls = [], []

    for c in trange(len(clips)):
        vr = decord.VideoReader(f"{clip_dir}/{clips[c]}")
        frames = vr.get_batch(list(np.arange(0,48))).asnumpy()
        frames = frames[np.newaxis,...]
        vid = video_transform(torch.from_numpy(frames)).to(DEVICE)

        model.eval()
        with torch.no_grad():
            pred = model.forward(vid) 
        pred = pred.softmax(dim=-1).cpu().detach().argmax(1)

        lbl = int(clips[c].split('_')[0])
        if lbl <= 1:
            lbl = 0
        elif lbl > 1:
            lbl = 1

        preds.append(pred.item())
        lbls.append(lbl)

    tn, fp, fn, tp = confusion_matrix(lbls, preds).ravel()

    print(f'confused samples {num_fold}: {fp+fn}')
# %%
