from configs.config import config
import cv2, os, random
from tqdm import tqdm
import numpy as np
from termcolor import cprint
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from models.utils.loss import get_loss
from data.utils import video_transform#, preprocess
from torchmetrics.classification import BinaryAccuracy
from torchmetrics import Accuracy


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def dprint(input, debug=config['DEBUG']):
    if debug:
        print(input)

def update_ema(model, ema_model, alpha=0.9, epoch=None, ema_warmup_epochs=3):
     # Optionally delay EMA updates until after a specified number of epochs
    if epoch is not None and (epoch+1) < ema_warmup_epochs:
        ema_model.load_state_dict(model.state_dict())
        print(f'EMA not updated, epoch: {epoch+1}')
        return  # Skip EMA update during warm-up period
    #  exponential moving average of model parameters
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        if not param.data.shape: # scalar tensor
            ema_param.data = alpha * ema_param.data + (1 - alpha) * param.data
        else:
            ema_param.data[:] = alpha * ema_param[:].data[:] + (1 - alpha) * param[:].data[:]
    print('EMA updated...')
    return None
# array([0.66850875, 1.98360248]) class weights
class Trainer(object):
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.66850875,
                                                                  1.98360248]).to(DEVICE),
                                             label_smoothing=0.1) # CrossEntropyLoss  BCEWithLogitsLoss
        # self.criterion = nn.CrossEntropyLoss(weight=torch.tensor([ 0.27572671, 39.45401896,
        #                                                            9.68617219,  1.2504065 ,
        #                                                             2.24772325]).to(DEVICE),
                                            #  label_smoothing=0.1)
        self.accuracy = Accuracy(task="multiclass", num_classes=2)
    
    def training_step(self, batched_data):
        # shape from BTHWC -> BCTHW
        # flow = video_transform(batched_data['flow']).to(DEVICE)
        vid = video_transform(batched_data['vid']).to(DEVICE)
        
        lbl_batch = batched_data['lbl'].type(torch.long).to(DEVICE) # Bx1
        
        self.model.zero_grad()
        self.optimizer.zero_grad()

        preds = self.model.forward(vid)

        loss = self.criterion(preds, lbl_batch)
        acc = self.accuracy(preds.softmax(dim=-1).cpu().detach(),
                            lbl_batch.cpu().detach())#.argmax(1)
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item(), acc.item()
    
class Evaluator(object):
    def __init__(self, model, return_features=False):
        self.model = model
        self.accuracy = Accuracy(task="multiclass", num_classes=2)
        self.return_features = return_features

    def eval_step(self, batched_data):
        # shape from BTHWC -> BCTHW
        # flow = video_transform(batched_data['frames']).to(DEVICE)
        vid = video_transform(batched_data['vid']).to(DEVICE)
       
        lbl_batch = batched_data['lbl'].type(torch.long).to(DEVICE) # Bx1
        self.model.eval()
        with torch.no_grad():
            preds = self.model.forward(vid) 

        preds = preds.softmax(dim=-1).cpu().detach()
        # lbls = lbl_batch.argmax(1).cpu().detach()
        lbls = lbl_batch.cpu().detach()
        acc = self.accuracy(preds, lbls)

        if self.return_features:
            return acc, preds, lbls, self.model.out_feats.cpu().numpy()
        else:
            return acc, preds, lbls