#%%
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
# decord.bridge.set_bridge('torch')
np.random.seed(0)

#%%
class GEN_DATA_LISTS():

    def __init__(self, config):
        self.folds = config['folds']
        self.vid_dir = config['vid_dir']
        self.flow_dir = config['flow_dir']
        self.lbl_dir = config['lbl_dir']

    def get_folds(self, num_fold=1):
        def read_samples(file_type):
            path = f"{self.folds}/{file_type}_fold_{num_fold}.txt"
            with open(path, 'r') as file:
                samples = [line.strip() for line in file.readlines()]
            return samples

        def generate_paths(samples):
            data = {
                'flow_paths': [f'{self.flow_dir}{sample}.mp4' for sample in samples],
                'vid_paths': [f'{self.vid_dir}{sample}.mp4' for sample in samples],
                'lbls': [f'{self.lbl_dir}{sample}.npy' for sample in samples]
            }
            return data

        train_samples = read_samples('train')
        test_samples = read_samples('test')

        train_data = generate_paths(train_samples)
        test_data = generate_paths(test_samples)

        return train_data, test_data
    
    def chk_paths(self, data):
        error_flag = False
        for key, paths in data.items():
            for path in paths:
                if not os.path.exists(path):
                    print(f"The path {path} does not exist.")
                    error_flag = True
        if not error_flag:
            print("All paths exist.")

#%%


class FetalMovementDataset(data.Dataset):
    def __init__(self, dataset_dict, config, augment=False):
        
        self.flow_paths = dataset_dict['flow_paths']
        self.video_paths = dataset_dict['vid_paths']
        self.label_paths = dataset_dict['lbls']

        self.config = config
        self.y_seconds = self.config['y_seconds']
        self.segment_length = self.config['sample_duration']
        self.noise_threshold = self.config['noise_threshold']

        self.chk_segments = self.config['chk_segments']
        self.downsmaple_clip = self.config['downsmaple_clip']
        self.augment = augment

        self.crop_frames = self.config['crop_minutes'] * 60 * self.config['video_fps']  # Convert crop minutes to frames
        self.group_labels = self.config['group_labels']

        self.samples = self.prepare_samples()
        

    def prepare_samples(self):
        samples = []
        self.test = []
        for vid_path, flo_path, lbl_path in zip(self.video_paths, self.flow_paths, self.label_paths):
            
            filename = Path(vid_path).stem

            labels = np.load(lbl_path)[1, :]  # Load the second row of labels
            labels = labels.astype(int)
            if self.group_labels:
                # 5 grouped classes
                # labels = convert_labels(labels, config['grouped_labels'])
                labels = labels
            else:
                labels = [0 if x == 0 else 1 if x == 1 else 2 for x in labels]
            self.test.append(labels)

            self.timestamps = np.load(lbl_path)[2, :]  # Load the 3rd row of timestamps
            # Crop 2 minutes from the start and end
            if len(labels) > 2 * self.crop_frames:
                labels = labels[self.crop_frames:-self.crop_frames]
                self.timestamps = self.timestamps[self.crop_frames:-self.crop_frames]
            else:
                continue  # Skip files that are too short

            segments = self.get_segments(labels)
            
            if self.chk_segments:
                self._chk_segments(segments, labels, lbl_path)
                
            for segment in segments:
                clips = self.get_clips(segment)
                for clip in clips:
                    # subtract 1 to make it 0-based, segment[2] is the label
                    samples.append((vid_path, flo_path, clip, segment[2] - 1, filename))  

        print(f"Total samples: {len(samples)}")
        return samples

    def get_segments(self, labels):
        segments = []
        start_idx = None
        current_label = None
        noise_count = 0
        fps = self.config['video_fps']
        noise_threshold_frames = self.noise_threshold * fps

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

    def get_clips(self, segment):
        start_idx, end_idx, label = segment
        fps = self.config['video_fps']
        y_frames = self.y_seconds * fps
        clip_frames = self.segment_length * fps

        # Apply clipping to avoid transition noise
        if end_idx - start_idx + 1 <= 2 * y_frames + clip_frames:
            return []

        start_idx += y_frames
        end_idx -= y_frames

        clips = []
        for idx in range(start_idx, end_idx - clip_frames + 1, clip_frames):
            clips.append(list(range(idx, idx + clip_frames)))
        
        if self.downsmaple_clip:
            for i in range(len(clips)):
                tmp = np.random.choice(clips[i], self.config['downsampled_frame_rate'],
                                       replace=False)
                tmp.sort() # place holder
                clips[i] = tmp
            
        return clips
    
    def get_labels(self):
        return [sample[3] for sample in self.samples]  # label is at index 3 && subtract 1 to make it 0-based
    
    def _chk_segments(self, seg, lbls, file_path):
        try:
            lbl_err_flag = 0
            for i in range(len(seg)):
                start, end, lbl = seg[i]
                if start != end:
                    orig = np.unique(lbls[start:end])[0]
                    if lbl == orig:
                        pass
                    else:
                        lbl_err_flag += 1
                else:
                    pass
            print(f'Errors segments in {Path(file_path).stem}: {lbl_err_flag}')
        except IndexError:
            print(f'Error in {Path(file_path).stem}: {np.unique(lbls[start:end])}')
            # print(seg)
        return None

    def _is_mostly_black(self, frames, threshold=0.6): # more than 50% pixels are black
        total_pixels = frames.size
        black_pixels = np.sum(frames == 0)
        black_ratio = black_pixels / total_pixels
        return black_ratio > threshold
    
    def _is_solid_color(self, frames):
        # return frames.max() == frames.min()
        return np.all(frames == frames[0, 0, 0, 0])
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data_sample = {}

        vid_path, flo_path, clip, label, filename = self.samples[idx]
        vr = decord.VideoReader(vid_path,
                                width=self.config['video_width'],
                                height=self.config['video_height'],
                                ctx=cpu(0))
        # fr = decord.VideoReader(flo_path, width=self.config['video_width'], height=self.config['video_height'])

        # adjust the clip indices by adding crop_frames to account for the cropped video
        adjusted_clip = [frame + self.crop_frames for frame in clip]
        vframes = vr.get_batch(adjusted_clip).asnumpy()
        # fframes = fr.get_batch(adjusted_clip).asnumpy()
        # print(vframes.shape)

        if self.augment:
            # TO apply same augmentation to both video and flow frames
            # stack video and flow frames
            # vf_frames = np.vstack((vframes, fframes))
            # vf_frames = video_augment(vf_frames)
            # # # split back 
            # vframes = vf_frames[:len(vframes)]
            # fframes = vf_frames[len(vframes):]
            vframes = video_augment(vframes)
            # fframes = video_augment(fframes)

            # as augments might return float values
            if vframes.max() <= 1:
                vframes = (vframes * 255).astype(np.uint8)
        # print(vframes.dtype, vframes.shape)
        
        if self._is_mostly_black(vframes) or self._is_solid_color(vframes):
            next_idx = (idx + 1) % len(self.samples)
            return self.__getitem__(next_idx)

        # take first channel of vframes and last two cahnnels of fframes and stack them
        # a = vframes[:, :, :, 0:1]
        # b = fframes[:, :, :, 1:]
        # stacked = np.concatenate((a, b), axis=-1)

        data_sample['vid'] = vframes # stacked  vframes
        data_sample['flow'] = 0#fframes
        
        data_sample['lbl'] = label 
        data_sample['filename'] = filename

        return data_sample
