
import numpy as np
import torch
# import fast
from tabulate import tabulate

def convert_labels(labels, grouped_labels):
    return [grouped_labels[label] for label in labels]

def normalize(clip, mean, std, inplace=False):
    """
    Args:
        clip (torch.tensor): Video clip to be normalized. Size is (C, T, H, W)
        mean (tuple): pixel RGB mean. Size is (3)
        std (tuple): pixel standard deviation. Size is (3)
    Returns:
        normalized clip (torch.tensor): Size is (C, T, H, W)
    """
    if not inplace:
        clip = clip.clone()
    mean = torch.as_tensor(mean, dtype=clip.dtype, device=clip.device)
    std = torch.as_tensor(std, dtype=clip.dtype, device=clip.device)
    clip.sub_(mean[:, None, None, None]).div_(std[:, None, None, None])
    return clip

class NormalizeVideo(object):
    """
    Normalize the video clip by mean subtraction and division by standard deviation
    Args:
        mean (3-tuple): pixel RGB mean
        std (3-tuple): pixel RGB standard deviation
        inplace (boolean): whether do in-place normalization
    """

    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor): video clip to be normalized. Size is (C, T, H, W)
        """
        return normalize(clip, self.mean, self.std, self.inplace)

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1}, inplace={2})'.format(
            self.mean, self.std, self.inplace)

apply_norm = NormalizeVideo(mean=(128,128,128), std=(128,128,128))

def to_norm_tensor(clip):
    """
    Convert tensor data type from uint8 to float, divide value by 255.0 and
    permute the dimenions of clip tensor
    Args:
        clip (torch.tensor, dtype=torch.uint8): Size is (T, H, W, C)
    Return:
        clip (torch.tensor, dtype=torch.float): Size is (C, T, H, W)
    """
    # return clip.float().permute(3, 0, 1, 2) / 255.0 # range [0, 1]
    return apply_norm(clip.float().permute(3, 0, 1, 2)) # range [-1, 1]

def video_transform(videos):
    '''
    videos: list of PIL videos
    Args:
        clip (torch.tensor, dtype=torch.uint8): Size is (B, T, H, W, C)
    Return:
        clip (torch.tensor, dtype=torch.float): Size is (B, C, T, H, W)
    '''
    inputs = []
    for vid in videos:
        inputs.append(to_norm_tensor(vid))
    inputs = torch.stack(inputs, dim=0).float().to('cuda' if torch.cuda.is_available() else 'cpu')
    return inputs

def collate(batch):
    '''
    custom Collat funciton for collating individual fetched data samples into batches.
    '''
    frames = [b['frames'] for b in batch]
    
    body = [b['body'] for b in batch]
    face = [b['face'] for b in batch]
    rh = [b['rh'] for b in batch]
    lh = [b['lh'] for b in batch]
    
    ecg = [b['ecg'] for b in batch]

    sub_lbls = [b['sub_lbls'] for b in batch]
    super_lbls = [b['super_lbls'] for b in batch]    

    filename = [b['filename'] for b in batch]
    return {'frames': frames, 'body': body, 'face': face,
            'rh': rh, 'lh': lh, 'ecg': ecg, 'sub_lbls': sub_lbls,
            'super_lbls': super_lbls, 'filename': filename}

def generate_soft_labels(sample, num_classes):
    label = np.zeros(num_classes)
    for element in sample:
        label[element] += 1
    label /= len(sample)  # Normalize to get proportions
    return label

def get_of_indices(seconds_to_sample):
    
    max_samples_per_interval = 50
    frames_per_second = 23#30
    
    frames_per_second_to_sample = int(np.ceil(max_samples_per_interval/seconds_to_sample))
    
    # Generate random indices for frame sampling
    sampled_indices = []
    for second in range(1, seconds_to_sample + 1):
        start_frame = int((second - 1) * frames_per_second)
        end_frame = int(second * frames_per_second)
        sampled_indices.extend(np.random.choice(range(start_frame, end_frame),
                                                frames_per_second_to_sample,
                                                replace=False))
    # this 48 is because slowfast network's input
    x = np.random.choice(sampled_indices, 48, replace=False) 
    x.sort()
    return x


def values_fromreport(report):
    p = report['weighted avg']['precision']
    r = report['weighted avg']['recall']
    f1 = report['weighted avg']['f1-score']
    return p,r, f1

def print_formatted_table(data):
    """
    Prints a formatted table of model performance metrics.

    Parameters:
    - data (dict): A dictionary containing model names as keys and lists of metrics as values.
                   Each list should contain floating-point numbers.
    - metrics (list): A list of strings representing the names of the metrics.
    """
    metrics = ['None_Mov', 'Movement', 'Avg', 'Precision', 'Recall', 'F1']
    rows = []

    # Prepare data for tabulation
    for model, values in data.items():
        row = [model] + [f"{value:.4f}" for value in values]
        rows.append(row)

    # Define the table headers
    headers = ['Modality'] + metrics

    # Print the table
    print(tabulate(rows, headers=headers, tablefmt="grid"))
    return None

def custom_collate_fn(batch):
    '''
    Custom collate function for collating individual fetched data samples into batches.
    '''
    vid_i = [b['vid_i'] for b in batch]
    vid_j = [b['vid_j'] for b in batch]
    lbl = [b['lbl'] for b in batch]
    filename = [b['filename'] for b in batch]

    return {
        'vid_i': torch.tensor(np.array(vid_i)),
        'vid_j': torch.tensor(np.array(vid_j)),
        'lbl': lbl,
        'filename': filename
    }

def infer_collate_fn(batch):
    '''
    Custom collate function for collating individual fetched data samples into batches.
    '''
    vid = [b['vid'] for b in batch]
    # flo = [b['flo'] for b in batch]
    lbl = [b['lbl'] for b in batch]
    start_times = [b['timestamp'][0] for b in batch]
    end_times = [b['timestamp'][1] for b in batch]
    filename = [b['filename'] for b in batch]

    return {
        'vid': torch.tensor(np.array(vid)),
        # 'flo': torch.tensor(np.array(flo)),
        'lbl': lbl,
        'timestamp': (start_times, end_times),
        'filename': filename
    }

# Function to denoise a frame using FAST
# def denoise_frame(frame):
#     # Create an image object from the frame array
#     image = fast.Image.createFromArray(frame)
    
#     # Create the Non-Local Means filter
#     nlm = fast.NonLocalMeans.create(
#         filterSize=3,
#         searchSize=11,
#         smoothingAmount=0.2,
#         inputMultiplicationWeight=0.5
#     )
#     nlm.setInputData(image)
#     nlm.update()
    
#     # Retrieve the denoised image
#     denoised_image = nlm.getOutputData()
#     return denoised_image

# def preprocess(frames):
#     # receives batched frames of shape
#     # (B, T, H, W, C)
#     B, T, H, W, C = frames.shape
#     frames = frames.reshape(-1,H,W,C).numpy() # (B*T, H, W, C)
#     # Process each frame
#     denoised_frames = []
#     for frame in frames:
#         denoised_image = denoise_frame(frame)
#         denoised_array = np.asarray(denoised_image)
#         denoised_frames.append(denoised_array)

#     denoise_frames = np.stack(denoised_frames)
#     # reshape back to original shape
#     denoise_frames = denoise_frames.reshape(B, T, H, W, C)

#     return torch.from_numpy(denoise_frames)