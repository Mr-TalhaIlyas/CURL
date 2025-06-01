#%%
import os
import os, psutil
# os.chdir(os.path.dirname(__file__))
# os.chdir('/home/talha/Data/fetal/scripts/')
from configs.config import config
import gc, time
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
# # The GPU id to use, usually either "0" or "1";
# os.environ["CUDA_VISIBLE_DEVICES"] = config['gpus_to_use'];
from data.dataloader import GEN_DATA_LISTS
from configs.config import config
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from data.infer_loader import get_infer_sample
from torchmetrics import Accuracy
from data.utils import video_transform
from models import slowfast
from tqdm import tqdm
from data.utils import values_fromreport
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from tabulate import tabulate

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def print_classification_report(all_preds, all_lbls):
    # if the values in all_lbls are not binary, then clip the values to 0 and 1
    # all_lbls = np.clip(all_lbls, a_min=0, a_max=1)
    # all_preds = np.clip(all_preds, a_min=0, a_max=1)
    # Calculate confusion matrix
    matrix = confusion_matrix(all_lbls, all_preds, labels=[0, 1])
    no_mov, mov = matrix.diagonal()/matrix.sum(axis=1)
    tn, fp, fn, tp = matrix.ravel()

    # Sensitivity and Specificity
    sensitivity_no_mov = tn / (tn + fp) if (tn + fp) > 0 else 0
    specificity_no_mov = tp / (tp + fn) if (tp + fn) > 0 else 0
    sensitivity_mov = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity_mov = tn / (tn + fp) if (tn + fp) > 0 else 0

    # Calculate AUC
    if len(np.unique(all_lbls)) == 2:
        auc = roc_auc_score(all_lbls, all_preds)
    else:
        auc = 'N/A'  # If there are not exactly two classes, AUC is not applicable

    # Calculate classification report
    report = classification_report(all_lbls, all_preds, labels=[0,1],
                                   output_dict=True, zero_division=0)

    # Extract accuracy, precision, and recall scores
    try:
        accuracy = report['accuracy']
    except KeyError:
        accuracy = 0.0 # If there are no samples, accuracy is 0
    precision1 = report['1']['precision']
    recall1 = report['1']['recall']
    f1_1 = report['1']['f1-score']

    precision0 = report['0']['precision']
    recall0 = report['0']['recall']
    f1_0 = report['0']['f1-score']

    precision = report['weighted avg']['precision']
    recall = report['weighted avg']['recall']
    f1 = report['weighted avg']['f1-score']

    # Create table
    table = [['Class', 'Acc', 'Pre', 'Rec/Sen', 'F1', 'Spe', 'AUC'],
             ['No Movement', f'{no_mov:.4f}', f'{precision0:.4f}', f'{recall0:.4f}', f'{f1_0:.4f}', f'{specificity_no_mov:.4f}', auc if isinstance(auc, str) else f'{auc:.4f}'],
             ['Movement', f'{mov:.4f}', f'{precision1:.4f}', f'{recall1:.4f}', f'{f1_1:.4f}', f'{specificity_mov:.4f}', auc if isinstance(auc, str) else f'{auc:.4f}'],
             ['Avg', f'{accuracy:.4f}', f'{precision:.4f}', f'{recall:.4f}', f'{f1:.4f}', '-', auc if isinstance(auc, str) else f'{auc:.4f}']
            ]

    # Print table
    print(tabulate(table, headers='firstrow', tablefmt='grid'))

    return auc

# Example usage
# all_preds = [list of predictions]
# all_lbls = [list of true labels]
# print_classification_report(all_preds, all_lbls)

# def print_classification_report(all_preds, all_lbls):

#     # Calculate confusion matrix
#     matrix = confusion_matrix(all_lbls, all_preds, labels=[0, 1], normalize='true')
#     no_mov, mov = matrix.diagonal()/matrix.sum(axis=1)
#     # Calculate classification report
#     report = classification_report(all_lbls, all_preds, output_dict=True, zero_division=0)

#     # Extract accuracy, precision, and recall scores
#     accuracy = report['accuracy']
#     precision1 = report['1']['precision']
#     recall1 = report['1']['recall']
#     f1_1 = report['1']['f1-score']

#     precision0 = report['0']['precision']
#     recall0 = report['0']['recall']
#     f1_0 = report['0']['f1-score']

#     precision = report['weighted avg']['precision']
#     recall = report['weighted avg']['recall']
#     f1 = report['weighted avg']['f1-score']

#     # Create table
#     table = [['Class', 'Accuracy', 'Precision', 'Recall', 'F1'],
#             ['No Movement', f'{no_mov:.4f}', f'{precision0:.4f}', f'{recall0:.4f}', f'{f1_0:.4f}'],
#             ['Movement', f'{mov:.4f}', f'{precision1:.4f}', f'{recall1:.4f}', f'{f1_1:.4f}'],
#             ['Avg', f'{accuracy:.4f}', f'{precision:.4f}', f'{recall:.4f}', f'{f1:.4f}']
#             ]

#     # Print table
#     print(tabulate(table, headers='firstrow', tablefmt='grid'))

# def get_accuracy(preds, lbls):
    
#     # lbls = np.clip(lbls, a_min=0, a_max=1)
#     # preds = np.clip(preds, a_min=0, a_max=1)
    
#     matrix = confusion_matrix(lbls, preds,
#                             labels=[0,1], normalize='true')
#     no_mov, mov = matrix.diagonal()/matrix.sum(axis=1)

#     return no_mov, mov, ((no_mov+mov)/2)

# def get_precision_recall(preds, lbls):
    
#     # lbls = np.clip(lbls, a_min=0, a_max=1)
#     # preds = np.clip(preds, a_min=0, a_max=1)
    
#     report = classification_report(lbls, preds,
#                                     output_dict=True,
#                                     zero_division=0)
#     precision, recall, f1 = values_fromreport(report)
#     return precision, recall, f1

# class Inference(object):
#     def __init__(self, dataset_dict, model):

#         self.dataset_dict = dataset_dict
#         self.model = model
#         self.sup_accuracy = Accuracy(task="multiclass", num_classes=3)
    
#     def calculate_accuracy(self, preds, labels):
#         all_accuracy = {}
#         for key in preds:
#             aux_acc = self.sup_accuracy(preds[key].softmax(dim=-1).cpu().detach().squeeze(-1), # squeeze lass dim as batch==1
#                                         labels.argmax(1).cpu().detach())
#             all_accuracy[key] = aux_acc.item()
#         return all_accuracy
    
#     def get_sample(self, sample_idx, overlap_sec=3):
#         data_sample, sliding_windows = get_infer_sample(self.dataset_dict,
#                                                         sample_idx,
#                                                         overlap_sec)
#         return data_sample, sliding_windows

#     def get_preds(self, data_sample, sliding_windows, batch_size=1, return_feats=False):
        
#         #from numpy to torch and device
#         for key in data_sample:
#             if key != 'filename':
#                 data_sample[key] = torch.tensor(data_sample[key],
#                                                 dtype=torch.float32, device='cpu')
#         all_preds = []
#         all_lbls = []
#         if return_feats:
#             all_feats = []
#         # print('Running inference...')
#         # predict on batches
#         # for i in tqdm(range(0, len(data_sample['lbl']), batch_size), desc='Infer'):
#         for i in range(0, len(data_sample['lbl']), batch_size):
            
#             # flow = video_transform(data_sample['flow'][i:i+batch_size]).to(DEVICE)
#             vid = video_transform(data_sample['vid'][i:i+batch_size]).to(DEVICE)

#             lbl_batch = torch.argmax(data_sample['lbl'][i:i+batch_size], dim=1) # Bx3 -> Bx1
            
#             self.model.eval()
#             with torch.no_grad():
#                 preds = self.model.forward(vid)
#                 if return_feats:
#                     all_feats.append(self.model.out_feats.detach().cpu().numpy().squeeze())
#             # print(preds.shape, lbl_batch.shape)
#             all_preds.append(preds.cpu().detach().numpy()) # Bx3

#             all_lbls.append(lbl_batch.cpu().detach().numpy())

#         all_preds = np.concatenate(all_preds, axis=0)
#         all_lbls = np.concatenate(all_lbls, axis=0)
#         # print('====>', all_preds.shape, all_lbls.shape)
#         filename = data_sample['filename']
        
#         if return_feats:
#             all_feats = np.asarray(all_feats)

#             return (all_preds.argmax(1), all_lbls, all_feats, filename)
#         else:
#             return (all_preds.argmax(1), all_lbls, filename)

# #%%

# # data = GEN_DATA_LISTS(config)

# # train_data, test_data = data.get_folds(config['num_fold'])

# def run_inference(infer_data, infer_model, overlap_sec=3, return_feats=False):
#     infer_model.eval()
#     infer = Inference(infer_data, infer_model)

#     files = []
#     all_preds, all_lbls = [], []
#     if return_feats:
#         all_feats = []
#     for idx in range(len(infer_data['flow_paths'])):
#     # for idx in range(1):
#         if (idx+1) % 4 == 0:
#             print(f"Test Samples: {idx+1}/{len(infer_data['flow_paths'])}")
#         data_sample, sliding_window = infer.get_sample(sample_idx=idx, overlap_sec=overlap_sec)
#         # print("Data sampel reveived")
#         if return_feats:
#             preds, lbls, feats, nfile = infer.get_preds(data_sample, sliding_window,
#                                                         batch_size=1, return_feats=return_feats)
#         else:
#             preds, lbls, nfile = infer.get_preds(data_sample, sliding_window, 
#                                                  batch_size=1, return_feats=return_feats)
#             del data_sample
#             del sliding_window
#             gc.collect()
#             time.sleep(10)
            
#         all_preds.append(preds)
#         all_lbls.append(lbls)
#         files.append(nfile)
#         if return_feats:
#             all_feats.append(feats)

#     all_preds = np.concatenate(all_preds, axis=0)
#     all_lbls = np.concatenate(all_lbls, axis=0)
#     # files = np.concatenate(files, axis=0)

#     if return_feats:
#         all_feats = np.concatenate(all_feats, axis=0)

#     # loop over all the preds and lbls to get accuracy and precision recall
#     noise, no_mov, mov = get_accuracy(all_preds, all_lbls)

#     precision, recall, f1 = get_precision_recall(all_preds, all_lbls)

#     if return_feats:
#         # then return the feats as well along with all labels for tSNE plotting
#         return {'filenames': files,
#                 'metrics': [noise, no_mov, mov, precision, recall, f1],
#                 'all_lbls': all_lbls,
#                 'all_preds': all_preds,
#                 'all_feats': all_feats
#                 }
#     else:
#         return {
#                 # 'filenames': files,
#                 'metrics': [noise, no_mov, mov, precision, recall, f1],
#                 # 'all_lbls': all_lbls,
#                 # 'all_preds': all_preds,
#                 }

# #%%
# # from models.slowfast import SlowFast
# # model = SlowFast(config['model'])
# # model.to(DEVICE)



# # # checkpoint = torch.load("/home/talha/Data/mme/chkpts/mme2.pth")#, map_location='cpu')
# # # model.load_state_dict(checkpoint['model_state_dict'])
# # data = GEN_DATA_LISTS(config)

# # train_data, test_data = data.get_folds(config['num_fold'])

# # data_sample, sliding_windows = get_infer_sample(test_data,
# #                                                 0,
# #                                                 0)

# # infer = Inference(test_data, model)

# # preds, lbls, nfile = infer.get_preds(data_sample, sliding_windows, 
# #                                       batch_size=1, return_feats=False)
# # x = run_inference(test_data, model, overlap_sec=0, return_feats=False)



