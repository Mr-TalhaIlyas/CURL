
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

def label_fixing(y_true, y_prob, conf=0.8):
    # because of high  noise in labels.
    tt = (y_prob > conf).astype(np.uint8)  # only keep high conf. predictions
    a = (y_true + tt) # add the high conf. predictions to the true labels then 0's will be 1's
    y_true_f = np.clip(a, 0, 1) # clip the values to 0 and 1

    return y_true_f
#%%
num_classes = config['num_classes']
sub_classes = 1
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data = GEN_DATA_LISTS(config)

train_data, test_data = data.get_folds(config['num_fold'])
if config['sanity_check']:
    data.chk_paths(train_data)
    data.chk_paths(test_data)

#%%
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
#%%
model.eval()
evaluator = Evaluator(model)


# load_chkpt(model, optimizer, chkpt)
load_chkpt(model, optimizer, f"{config['checkpoint_path']}/MAE_finetune_f1_v5-cycle-2xlr.pth")

path = "/home/user01/Data/fetal/data/folds/PatientIDs.txt"
with open(path, 'r') as file:
    pat_ids = [line.strip() for line in file.readlines()]


#%%
config['y_seconds'] = 30
config['sample_duration'] = 3
config['downsampled_frame_rate'] = 16
config['noise_threshold'] = 0
config['batch_size'] = 32
config['num_workers'] = 2
config['num_fold'] = 1
config['group_labels'] = True
test_dataset = FetalMovementDataset(train_data, config, augment=False)
test_loader = DataLoader(test_dataset,
                        batch_size=config['batch_size'], shuffle=False,
                        num_workers=config['num_workers'], drop_last=True,
                        collate_fn=None, pin_memory=config['pin_memory'],
                        prefetch_factor=1, persistent_workers=True,
                        # sampler=BiasedSampler(train_dataset)
                        )
labels = test_dataset.get_labels()
print(np.unique(labels))
#%%
from confidenceinterval import roc_auc_score as roc_auc_score_ci, classification_report_with_ci
from sklearn.metrics import confusion_matrix, classification_report, roc_curve
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay
from data.utils import convert_labels

test_acc, all_preds, all_lbls, all_feats = [], [], [], [] 
preds_probs = []
removed = []

for data_batch in tqdm(test_loader):
    # if data_batch['filename'][0] in pat_ids:
    acc, preds, lbls = evaluator.eval_step(data_batch)
    test_acc.append(acc)
    all_preds.append(preds)
    all_lbls.append(lbls)
    # else:
    #     removed.append(data_batch['filename'][0])

print("Test Acc", np.nanmean(test_acc))

print(np.unique(all_lbls))
all_preds = np.concatenate(all_preds, axis=0)
all_lbls = np.concatenate(all_lbls, axis=0)

labels = convert_labels(all_lbls, config['grouped_labels'])
#%%
# convert softmax all_preds output to prob
y_prob  = np.argmax(all_preds, axis=1)
# y_prob = all_preds[:,1]
y_true =  np.clip(all_lbls, 0, 1)
# y_true_f = label_fixing(y_true, y_prob, conf=0.8)

# y_true = y_true_f

fpr, tpr, thresholds = roc_curve(y_true, y_prob) 
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]

y_pred = (all_preds[:,1]>0.6).astype(np.uint8) # all_preds.argmax(1),

auc = print_classification_report(y_pred, y_true)
auc, ci = roc_auc_score_ci(y_true, y_prob,
                            confidence_level=0.95)

print(f"AUC: {auc:.4f} with CI: {ci}")

print(classification_report_with_ci(y_true, y_pred))

RocCurveDisplay.from_predictions(y_true, y_prob, plot_chance_level=True)
PrecisionRecallDisplay.from_predictions(y_true, y_prob, plot_chance_level=True)
# %%
import numpy as np
import pandas as pd

# Example data:
# y_true contains actual labels (0 for non-movement, 3, 4, or 5 for different movement types)
# y_prob contains the probabilities from the binary classifier (for class 1 = movement)
y_true = labels
# convert softmax output to prob
y_prob =  np.argmax(all_preds, axis=1)

# Step 1: Threshold probabilities to get binary predictions
threshold = 0.5
y_pred = (y_prob >= threshold).astype(int)

# Step 2: Create a DataFrame and build a confusion matrix using crosstab
df = pd.DataFrame({'Actual': y_true, 'Predicted': y_pred})
conf_matrix = pd.crosstab(df['Actual'], df['Predicted'], rownames=['Actual'], colnames=['Predicted'])
print("Confusion Matrix:")
print(conf_matrix)

# Step 3: Calculate the percentage of each movement type predicted as non-movement
movement_types = [3, 4, 5]
for movement in movement_types:
    if movement in conf_matrix.index:
        total_count = conf_matrix.loc[movement].sum()
        false_negatives = conf_matrix.loc[movement].get(0, 0)  # predicted as 0 when movement is expected
        percentage = (false_negatives / total_count) * 100
        print(f"Movement {movement}: {percentage:.2f}% predicted as non-movement")
#%%
grouped_labels = {
                    0: 0,  # None-Movement -> No Movements
                    1: 2, # 5  # Head Motion -> Head and Facial Movements
                    2: 1,  # Twitch -> Small and Quick Movements
                    3: 1,  # Startle -> Small and Quick Movements
                    4: 2,  # Wave -> Limb Movements
                    5: 2,  # Kick -> Limb Movements
                    6: 3,  # Pant -> Respiratory Movements
                    7: 4,  # Hiccups -> Respiratory Movements
                    8: 5,  # Trunk -> Trunk and Large Body Movements
                    9: 6, # Precthl -> Trunk and Large Body Movements
                    10: 2  # Limb -> Limb Movements
                                }

sub_lbls = convert_labels(all_lbls, grouped_labels)
print(np.unique(sub_lbls, return_counts=True))
# %%

from sklearn.metrics import recall_score, f1_score
sub_probs = np.zeros((len(all_preds), 7)) # 6 + 1 non-movement

for i in range(len(all_preds)):
    sub_probs[i, sub_lbls[i]] = y_prob[i]

sub_probs = (sub_probs > 0.5).astype(np.uint8)
sub_preds = sub_probs.argmax(1)


sen = recall_score(sub_lbls, sub_preds, labels = [0, 1, 2, 3, 4, 5, 6],
                   average=None, zero_division=0.0)[1:]
f1 = f1_score(sub_lbls, sub_preds, labels = [0, 1, 2, 3, 4, 5, 6],
              average=None, zero_division=0.0)[1:]
print("Recall", sen, "F1", f1)
# %%

def plot_metrics_radar(metrics_dict, class_names):
    """
    Plots multiple metrics on a radar chart.

    Parameters:
    - metrics_dict: dictionary where keys are metric names and values are arrays of metric values per class.
    - class_names: list of class names.
    """
    num_vars = len(class_names)

    # Compute the angle each axis is at
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    # The plot is a circle, so we need to "complete the loop"
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 6), subplot_kw=dict(polar=True))

    for metric_name, values in metrics_dict.items():
        # Append the first value to the end to close the circle
        values = np.concatenate((values, [values[0]]))
        ax.plot(angles, values, label=metric_name, linewidth=2)
        ax.fill(angles, values, alpha=0.25)

    # Fix axis to go in the right order and start at the top
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Draw axis lines for each angle and label
    ax.set_thetagrids(np.degrees(angles[:-1]), class_names)

    # Set the range for the radial axis
    ax.set_ylim(0, 1)

    # Add labels for radial axis
    ax.set_rlabel_position(0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])

    # Add title and legend
    ax.set_title('Sensitivity and F1-Score per Movement Class', size=14, position=(0.5, 1.1), ha='center')
    ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))

    plt.show()
# %%
