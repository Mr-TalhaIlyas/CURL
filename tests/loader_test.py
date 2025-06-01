#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 21:00:00 2024

@author: user01
"""

import os
# os.chdir('/home/user01/Data/fetal/new_scripts/')
from configs.config import config
import torch
import torch.utils.data as data
from fmutils import fmutils as fmu
from data.utils import convert_labels
import decord as de
from decord import VideoReader, cpu
# from decord import cpu, gpu
import numpy as np
import cv2, os
import decord
from pathlib import Path
import pywt
from data.augmentors import video_augment, SimCLR_augment
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import imgviz
from data.augmentors import video_augment
from tqdm import trange
#%%
dataset_dict = train_data
flow_paths = dataset_dict['flow_paths']
video_paths = dataset_dict['vid_paths']
label_paths = dataset_dict['lbls']


y_seconds = config['y_seconds']
segment_length = config['sample_duration']
noise_threshold = config['noise_threshold']

chk_segments = config['chk_segments']
downsmaple_clip = config['downsmaple_clip']
# augment = augment

crop_frames = config['crop_minutes'] * 60 * config['video_fps']  # Convert crop minutes to frames


def prepare_samples():
    samples = []
    for vid_path, flo_path, lbl_path in zip(video_paths, flow_paths, label_paths):
        
        filename = Path(vid_path).stem

        labels = np.load(lbl_path)[1, :]  # Load the second row of labels
        labels = labels.astype(int)
        labels = [0 if x == 0 else 1 if x == 1 else 2 for x in labels]
        # 5 grouped classes
        # labels = convert_labels(labels, config['grouped_labels'])

        timestamps = np.load(lbl_path)[2, :]  # Load the 3rd row of timestamps
        # Crop 2 minutes from the start and end
        if len(labels) > 2 * crop_frames:
            labels = labels[crop_frames:-crop_frames]
            timestamps = timestamps[crop_frames:-crop_frames]
        else:
            continue  # Skip files that are too short

        segments = get_segments(labels)
        
            
        for segment in segments:
            clips = get_clips(segment)
            for clip in clips:
                # subtract 1 to make it 0-based
                # segment[2] is the label
                samples.append((vid_path, flo_path, clip, segment[2]-1, filename))  
    print(f"Total samples: {len(samples)}")
    return samples, timestamps

def get_segments(labels):
    segments = []
    start_idx = None
    current_label = None
    noise_count = 0
    fps = config['video_fps']
    noise_threshold_frames = noise_threshold * fps

    for idx, label in enumerate(labels):
        if label != 0:
            if current_label is None:
                current_label = label
                start_idx = idx
            elif label != current_label:
                segments.append((start_idx, idx - 1, current_label))
                current_label = label
                start_idx = idx
                noise_count = 0
        else:
            if current_label is not None:
                noise_count += 1
                if noise_count > noise_threshold_frames:
                    segments.append((start_idx, idx - noise_count, current_label))
                    current_label = None
                    start_idx = None
                    noise_count = 0
    
    if current_label is not None:
        segments.append((start_idx, len(labels) - 1, current_label))

    return segments

def get_clips(segment):
    start_idx, end_idx, label = segment
    fps = config['video_fps']
    y_frames = y_seconds * fps
    clip_frames = segment_length * fps

    # Apply clipping to avoid transition noise
    if end_idx - start_idx + 1 <= 2 * y_frames + clip_frames:
        return []

    start_idx += y_frames
    end_idx -= y_frames

    clips = []
    for idx in range(start_idx, end_idx - clip_frames + 1, clip_frames):
        clips.append(list(range(idx, idx + clip_frames)))
    
    if downsmaple_clip:
        for i in range(len(clips)):
            tmp = np.random.choice(clips[i], config['downsampled_frame_rate'],
                                   replace=False)
            tmp.sort() # place holder
            clips[i] = tmp
        
    return clips

def _is_mostly_black(frames, threshold=0.6): # more than 60% pixels are black
    total_pixels = frames.size
    black_pixels = np.sum(frames == 0)
    black_ratio = black_pixels / total_pixels
    return black_ratio > threshold

def _is_solid_color(frames):
    # return frames.max() == frames.min()
    return np.all(frames == frames[0, 0, 0, 0])
#%%
samples, timestamps = prepare_samples()
#%%

for idx in trange(len(samples)):
    # idx = 0
    data_sample = {}
    
    vid_path, flo_path, clip, label, filename = samples[idx]
    vr = decord.VideoReader(vid_path,
                            width=config['video_width'],
                            height=config['video_height'],
                            ctx=cpu(0))
    
    # adjust the clip indices by adding crop_frames to account for the cropped video
    adjusted_clip = [frame + crop_frames for frame in clip]
    
    vframes = vr.get_batch(adjusted_clip).asnumpy()

    # vframes = video_augment(vframes)

    # as augments might return float values
    if vframes.max() <= 1:
        vframes = (vframes * 255).astype(np.uint8)
        # x = list(vframes)
        # tiles = imgviz.tile(x, shape=(4, 4), border=(255,0,0))
        # plt.figure()
        # plt.imshow(tiles)
        # plt.axis('off')
        print(f'{idx}: Multi by *255')
    if vframes.max() <= 1:
        print(f'AGAIN &&& {idx}: Multi by *255')
    if _is_mostly_black(vframes) or _is_solid_color(vframes):
        print(f"{idx}:Black or solid color frame: {vid_path}")
        continue
    if vframes.max() <= 1 or vframes.max() == vframes.min():
        break










#%%
x = list(vframes)
tiles = imgviz.tile(x, shape=(4, 4), border=(255,0,0))
plt.imshow(tiles)
plt.axis('off')















