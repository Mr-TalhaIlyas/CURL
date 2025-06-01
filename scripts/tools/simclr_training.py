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
from tools.nt_xnet import NT_Xent


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def dprint(input, debug=config['DEBUG']):
    if debug:
        print(input)

class SimCLR_Trainer(object):
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.criterion = NT_Xent(batch_size=config['batch_size'], temperature=0.5, world_size=1)
        
    
    def training_step(self, batched_data):
        # shape from BTHWC -> BCTHW
        # flow = video_transform(batched_data['flow']).to(DEVICE)
        vid_i = video_transform(batched_data['vid_i']).to(DEVICE)
        vid_j = video_transform(batched_data['vid_j']).to(DEVICE)
        
        self.model.zero_grad()
        self.optimizer.zero_grad()

        vid_i = vid_i.cuda(non_blocking=True)
        vid_j = vid_j.cuda(non_blocking=True)

        h_i, h_j, z_i, z_j = self.model.forward(vid_i, vid_j)

        loss = self.criterion(z_i, z_j)
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item()


class SimCLR_Evaluator(object):
    def __init__(self, model, return_features=False):
        self.model = model
        self.return_features = return_features

    def eval_step(self, batched_data):
        # shape from BTHWC -> BCTHW
        # flow = video_transform(batched_data['frames']).to(DEVICE)
        vid = video_transform(batched_data['vid']).to(DEVICE)
       
        lbl_batch = batched_data['lbl']#.type(torch.long).to(DEVICE) # Bx1
        self.model.eval()
        with torch.no_grad():
            h_i, _, _, _ = self.model.forward(vid, vid) 
            
        lbls = lbl_batch#.cpu().detach()
        
        return h_i.cpu().detach().numpy(), lbls