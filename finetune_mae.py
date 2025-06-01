
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


import torch, math, gc
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
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchmetrics import Accuracy
from fmutils import fmutils as fmu

import cv2, random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 150
from termcolor import cprint
from tqdm import tqdm
from collections import Counter

from data.dataloader import GEN_DATA_LISTS, FetalMovementDataset
from data.utils import collate, values_fromreport, print_formatted_table

from models.slowfast import SlowFast
from models.vidmae import VidMAE
from models.lstm import CNN_LSTM
from models.mae.model_vit import VisionTransformer
from models.utils.lr_scheduler import LR_Scheduler, CyclicLR
from models.utils.tools import save_chkpt, load_pretrained_MAE_chkpt, param_groups_lrd, load_chkpt
from models.utils.loss import FocalLoss
from tools.training import Trainer, Evaluator

from sklearn.metrics import confusion_matrix, classification_report

from models.utils.visualization import display_video, display_video_lbl
from tsaug.visualization import plot
from IPython.display import HTML
from tools.inference import print_classification_report
from models.utils.tools import NativeScalerWithGradNormCount as NativeScaler
from torchmetrics import Accuracy
from functools import partial
from data.utils import video_transform
from timm.loss import LabelSmoothingCrossEntropy
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
# start a biased sampler
labels = train_dataset.get_labels()
class_counts = Counter(labels)
total_samples = len(labels)
num_classes = len(class_counts)
class_weights = {}
for cls in class_counts:
    class_weights[cls] = total_samples / (num_classes * class_counts[cls])
print(f'Class Weights: {class_weights}')
# weight for each sample
weights = [class_weights[label] for label in labels]
weights = torch.DoubleTensor(weights)

sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

# sub_train_dataset = torch.utils.data.Subset(train_dataset, list(range(0, 128)))
train_loader = DataLoader(train_dataset,
                        batch_size=config['batch_size'], shuffle=False,
                        num_workers=config['num_workers'], drop_last=True,
                        collate_fn=None, pin_memory=config['pin_memory'],
                        prefetch_factor=config['prefetch_factor'],
                        persistent_workers=config['persistent_workers'],
                        sampler=sampler
                        )

test_dataset = FetalMovementDataset(test_data, config, augment=False)
# sub_test_dataset = torch.utils.data.Subset(test_dataset, list(range(0, 64)))
test_loader = DataLoader(test_dataset,
                        batch_size=config['batch_size'], shuffle=False,
                        num_workers=config['num_workers'], drop_last=True,
                        collate_fn=None, pin_memory=config['pin_memory'],
                        prefetch_factor=config['prefetch_factor'],
                        persistent_workers=config['persistent_workers'],
                        # sampler=BiasedSampler(train_dataset)
                        )

print(f"Train Loader: {len(train_loader)} iter.")
print(f"Test Loader: {len(test_loader)} iter.")
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
model = VisionTransformer(img_size=224,
                        patch_size=16,
                        in_chans=3,
                        num_frames=16,
                        t_patch_size=2,
                        embed_dim=1024,
                        depth=24,
                        num_heads=16,
                        mlp_ratio=4,
                        drop_path_rate=config['mae_finetune']['drop_path_rate'],
                        dropout=config['mae_finetune']['dropout'],
                        num_classes=config['num_classes'],
                        norm_layer=partial(nn.LayerNorm, eps=1e-6))

laod_info = load_pretrained_MAE_chkpt(model, '/home/user01/Data/fetal/chkpts/MAE_ViT_2.pth')
matched, unmatched, extra_pretrained_layers, extra_model_layers = laod_info
model.to(DEVICE)

# ema_model = SlowFast(config['model'])
# ema_model.to(DEVICE)    

param_groups = param_groups_lrd(model, config['mae_finetune']['weight_decay'],
                                no_weight_decay_list=model.no_weight_decay(),
                                layer_decay=config['mae_finetune']['layer_decay'])

optimizer = torch.optim.AdamW(param_groups, lr=config['mae_finetune']['blr'])

loss_scaler = NativeScaler(fp32=config['mae_finetune']['fp32'])

# scheduler = LR_Scheduler(config['lr_schedule'], config['mae_finetune']['blr'],
#                          config['mae_finetune']['epochs'],
#                          iters_per_epoch=len(train_loader),
#                          warmup_epochs=config['mae_finetune']['warmup_epochs'])

scheduler = CyclicLR(optimizer,
                    step_size=len(train_loader)*5, # setting 5 the model will take 3 cycles in 30 epocehs
                    base_lr=1e-6,                  #  5 up + 5 down = 10 epochs so 3 cycles = 30 epochs
                    max_lr=config['mae_finetune']['blr'],
                    mode='triangular')
# old weights [0.66850875,1.98360248] # [0.59918644, 3.02050581]
# criterion = nn.CrossEntropyLoss(
#                                 weight=torch.tensor([0.59918644, 3.02050581]).to(DEVICE),
#                                 label_smoothing=config['mae_finetune']['smoothing'])
# criterion = LabelSmoothingCrossEntropy(smoothing=config['mae_finetune']['smoothing'])

criterion = FocalLoss(alpha=torch.tensor([0.6, 3.0]).to(DEVICE),
                      gamma=2, reduction='mean')

accuracy = Accuracy(task="multiclass", num_classes=2)
# accuracy = Accuracy(task="multiclass", num_classes=num_classes)

# trainer = Trainer(model, optimizer)
evaluator = Evaluator(model)
#%%
# Initializing plots
if config['LOG_WANDB']:
    wandb.watch(model, log='parameters', log_freq=100)
    wandb.log({ "Acc": 0, "total_loss": 10,
                "Test Acc": 0,"learning_rate": 0}, step=0)
    
#%%
# num_repeats_per_epoch = config['num_repeats_per_epoch']
accum_iter = config['mae_finetune']['accum_iter']
iters_per_epoch = len(train_loader)
effective_iters_per_epoch = math.ceil(iters_per_epoch / accum_iter)

start_epoch = 0
current_iter = 0
epoch, best_auc = 0, 0
total_avg_acc = []

for epoch in range(start_epoch, config['mae_finetune']['epochs']):
    model.train() # <-set mode important
    trloss = []
    tracc = []
    
    optimizer.zero_grad()
    for step, data_batch in enumerate(train_loader):

        # Update learning rate per iteration
        # scheduler(optimizer, current_iter)
        # current_iter += 1
        scheduler.step()

        vid = video_transform(data_batch['vid']).to(DEVICE)
        lbl_batch = data_batch['lbl'].type(torch.long).to(DEVICE) # Bx1

        with torch.cuda.amp.autocast(enabled=not config['mae_finetune']['fp32']):
            preds = model(vid)
            loss = criterion(preds, lbl_batch)

         # normalize loss for gradient accumulation
        loss = loss / accum_iter
        # Update gradient only at accumulation steps
        update_grad = ((step + 1) % accum_iter == 0) or ((step + 1) == iters_per_epoch)

        loss_scaler(loss,
                    optimizer, 
                    clip_grad=config['mae_finetune']['clip_grad'],
                    parameters=model.parameters(),
                    create_graph=False,
                    update_grad=update_grad 
                    )

        if update_grad:
            optimizer.zero_grad()

        loss_value = loss.item() * accum_iter
        if not math.isfinite(loss_value):
            raise Exception(f"Loss is {loss_value}, stopping training")
        
        acc = accuracy(preds.softmax(dim=-1).cpu().detach(),
                            lbl_batch.cpu().detach())#.argmax(1)
        acc = acc.item()

        # for printing
        trloss.append(loss_value)
        tracc.append(acc)
        # if step == 5: # only for testing
        #     break
    print(f'Epoch: {epoch+1}/{config["mae_finetune"]["epochs"]}=> Average loss: {np.nanmean(trloss):.4f}, Average Acc: {np.nanmean(tracc):.4f}')
    # fix following to infer only every 2nd epoch
    test_acc, all_preds, all_lbls = [], [], [] 
    if (epoch + 1) % 2 == 0:  # eval every N epoch
        print('Validating...')
        
        for vstep, data_batch in enumerate(test_loader):
            acc, preds, lbls = evaluator.eval_step(data_batch)
            test_acc.append(acc)
            all_preds.append(preds.argmax(1))
            all_lbls.append(lbls)
            # if vstep == 5: # only for testing
            #     break
        if config['LOG_WANDB']:
            wandb.log({ "Test Acc": np.nanmean(test_acc),
                      }, step=epoch+1)
        
        all_preds = np.concatenate(all_preds, axis=0)
        all_lbls = np.concatenate(all_lbls, axis=0)
        
        auc = print_classification_report(all_preds, all_lbls)

        current_auc = auc  # np.nanmean(test_acc)
        try:
            if current_auc > best_auc and epoch != 0:
                    best_auc = current_auc
                    chkpt = save_chkpt(model, optimizer, epoch, loss=np.nanmean(trloss),
                                        acc=current_auc, return_chkpt=True)
                    print(40*'$')
        except TypeError:
            print('Error in saving checkpoint')
    try:
        # save chkpt every 5 epochs
        if (epoch + 1) % 5 == 0:
            chkpt = save_chkpt(model, optimizer, epoch, loss=np.nanmean(trloss),
                                acc=current_auc, return_chkpt=True)
            print(40*'$')
    except TypeError:
        print('Error in saving checkpoint')
        
    if config['LOG_WANDB']:
        wandb.log({"total_loss": np.nanmean(trloss), 
                   "Acc": np.nanmean(tracc),
                   "learning_rate": optimizer.param_groups[0]['lr']}, step=epoch+1)
    print(f'learning rate: {optimizer.param_groups[0]["lr"]}')

if config['LOG_WANDB']:
    wandb.run.finish()
#%%
