#%%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 11:01:32 2024

@author: talha
"""
from configs.config import config
import os
from data.utils import get_of_indices, generate_soft_labels
from configs.config import config
import decord as de
from pathlib import Path
import numpy as np
from data.dataloader import GEN_DATA_LISTS
from tqdm import tqdm



def get_infer_sample(dataset_dict, sample_idx=0, overlap_sec=0):

    index = sample_idx
    
    lbl_paths = dataset_dict['lbls']
        # loade video
    vid_paths = dataset_dict['vid_paths']
    flo_paths = dataset_dict['flow_paths']

    data_sample = {}
    # LODAING INPUT DATA
    filename = Path(vid_paths[index]).stem
    # load video in memory
    vr = de.VideoReader(vid_paths[index],
                        width=config['video_width'],
                        height=config['video_height'],
                        # ctx=gpu(1),
                        # num_threads=4
                        )
    fr = de.VideoReader(flo_paths[index],
                        width=config['video_width'],
                        height=config['video_height'],
                        # ctx=gpu(1),
                        # num_threads=4
                        )
    
    # LODAING LABELS
    full_lbls = np.load(lbl_paths[index])
    full_lbls = full_lbls[1, :].astype(int) # second row is the label
    full_lbls = [0 if x==0 else 1 if x==1 else 2 for x in full_lbls]
    # Count the number of 1s(non-movement)  before the first occurrence of 0(noise)
    try:
        to_drop = full_lbls[:full_lbls.index(0)].count(1)
    except ValueError:
        to_drop = 0
    # to_drop = 0
    full_lbls = np.asarray(full_lbls)
    full_lbls = full_lbls[to_drop:]

    if config['remove_noise']:
        # WON'T WORK HERE BECAUE HERE THE 0 APPERAS IN THE MIDDLE OF THE RECORDING
        # filtered_indices = np.where(lbls != 0)[0]
        # lbls = lbls[filtered_indices]
        # lbls = lbls - 1 # to make it 0 based index
        raise NotImplementedError('This is not implemented yet')
    
    # put a check on per frame labels and vidoe frame counts
    # put a check on per frame labels and vidoe frame counts
    # if not len(full_lbls) < vr._num_frame:
    #     # print(f"Labels >> video frames for {filename}")
    #     full_lbls = full_lbls[:vr._num_frame]
    assert len(full_lbls) <= vr._num_frame == fr._num_frame, f"Labels >> video frames for {filename}"
    # check max number of data points available in both video and ecg streams
    # after dividing them by their frequency/fps the data points should be same
    # and after shifting to max we still need to have same number of data points
    # for extraction so subtract the sample duration from the total length.
    vid_max_sift_in_seconds = len(full_lbls) // config['video_fps'] - config['sample_duration']
    
    # this is the max shift in seconds we can do0
    max_sift_in_seconds = vid_max_sift_in_seconds
    
    # here instead of getting shift seconds we will window size and instead of adding a 
    # random shift to the indices (i.e., get_of_indices) to get one window we will start
    # from 0 and keep shiftign towards the end of recording.
    # Here we'll move +window+ seconds till we reach +max_sift_in_seconds+
    window = config['sample_duration'] - overlap_sec
    
    sliding_windows = []
    for step in range(0, max_sift_in_seconds+config['sample_duration'], window):
        shift_seconds = step
                
        if (max_sift_in_seconds+config['sample_duration'])-shift_seconds < config['sample_duration']:
            # shift_seconds = shift_seconds - (config['sample_duration'] - ((max_sift_in_seconds+config['sample_duration'])-shift_seconds))
            shift_seconds = max_sift_in_seconds # simplified version of above line
        if shift_seconds == max_sift_in_seconds+config['sample_duration']:
            break
        
        sliding_windows.append(shift_seconds)
        
    sliding_windows = np.unique(np.asarray(sliding_windows))
    # print(sliding_windows)
    all_vids, all_flow, all_lbls = [],[],[]
    
    # for shift_seconds in tqdm(sliding_windows, total=len(sliding_windows), desc=f'Loading {filename}'):
    for shift_seconds in sliding_windows:
        try: # try will get index error for videos who have seizur till end of recording
            # this shift will move the window randomly on the entire recording to get the required sample
            shift = shift_seconds * config['video_fps']
            
            vid_indices = get_of_indices(config['sample_duration']) + shift + to_drop
            # print(vid_indices, vid_indices.dtype, vid_indices.shape)
            # get input sample 
            vframes = vr.get_batch(vid_indices).asnumpy()
            fframes = 0 #fr.get_batch(vid_indices).asnumpy()
            # get labels
            # print(full_lbls.shape, full_lbls.dtype)
            lbls = full_lbls[vid_indices-to_drop]
            
            temp_lbls = generate_soft_labels(lbls, config['num_classes']+1) # as we will remove noise form 0th index
            if temp_lbls[0] != 0:
                    raise IndexError(f"Error: {temp_lbls[0]} @  {filename}")
            # print('frames', fframes.min(), fframes.max())
            lbls = temp_lbls[1:]
            all_vids.append(vframes)
            all_flow.append(fframes)
            all_lbls.append(lbls)

        except IndexError as e:
            # print(f'Error: {e} @  {filename}')
            pass
    
    data_sample['vid'] = np.asarray(all_vids) 
    data_sample['flow'] = np.asarray(all_flow)
    data_sample['lbl'] = np.asarray(all_lbls)
    data_sample['filename'] = filename
    
    return data_sample, sliding_windows
#%%
# Example

# data = GEN_DATA_LISTS(config)

# train_data, test_data = data.get_folds(config['num_fold'])

# dataset_dict = test_data

# data_sample, sliding_windows = get_infer_sample(dataset_dict, sample_idx=0, overlap_sec=0)

# lbls = data_sample['lbl']
# vid = data_sample['vid']
#%%

















