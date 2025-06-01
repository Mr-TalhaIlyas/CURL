import numbers
import random
import numpy as np
import PIL
import skimage.transform
import torchvision
import math
import torch
import cv2

# convert THWC video/clip to a PIL image list
def to_pil_image(clip):
    """
    Args:
    clip (np.array): shape of clip should be T x H x W x C
    Returns:
    list : list of PIL.Image
    """
    return [PIL.Image.fromarray(img.astype(np.uint8)) for img in clip]

class ColorJitter(object):
    """Randomly change the brightness, contrast and saturation and hue of the clip

    Args:
    brightness (float): How much to jitter brightness. brightness_factor
    is chosen uniformly from [max(0, 1 - brightness), 1 + brightness].
    contrast (float): How much to jitter contrast. contrast_factor
    is chosen uniformly from [max(0, 1 - contrast), 1 + contrast].
    saturation (float): How much to jitter saturation. saturation_factor
    is chosen uniformly from [max(0, 1 - saturation), 1 + saturation].
    hue(float): How much to jitter hue. hue_factor is chosen uniformly from
    [-hue, hue]. Should be >=0 and <= 0.5.
    """

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def get_params(self, brightness, contrast, saturation, hue):
        if brightness > 0:
            brightness_factor = random.uniform(
                max(0, 1 - brightness), 1 + brightness)
        else:
            brightness_factor = None

        if contrast > 0:
            contrast_factor = random.uniform(
                max(0, 1 - contrast), 1 + contrast)
        else:
            contrast_factor = None

        if saturation > 0:
            saturation_factor = random.uniform(
                max(0, 1 - saturation), 1 + saturation)
        else:
            saturation_factor = None

        if hue > 0:
            hue_factor = random.uniform(-hue, hue)
        else:
            hue_factor = None
        return brightness_factor, contrast_factor, saturation_factor, hue_factor

    def __call__(self, clip):
        """
        Args:
        clip (list): list of PIL.Image

        Returns:
        list PIL.Image : list of transformed PIL.Image
        """
        if isinstance(clip[0], np.ndarray):
            # raise TypeError(
            #     'Color jitter not yet implemented for numpy arrays')
            clip = to_pil_image(clip)
            brightness, contrast, saturation, hue = self.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        elif isinstance(clip[0], PIL.Image.Image):
            brightness, contrast, saturation, hue = self.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image' +
                            'but got list of {0}'.format(type(clip[0])))

        # Create img transform function sequence
        img_transforms = []
        if brightness is not None:
            img_transforms.append(lambda img: torchvision.transforms.functional.adjust_brightness(img, brightness))
        if saturation is not None:
            img_transforms.append(lambda img: torchvision.transforms.functional.adjust_saturation(img, saturation))
        if hue is not None:
            img_transforms.append(lambda img: torchvision.transforms.functional.adjust_hue(img, hue))
        if contrast is not None:
            img_transforms.append(lambda img: torchvision.transforms.functional.adjust_contrast(img, contrast))
        random.shuffle(img_transforms)

        # Apply to all images
        jittered_clip = []
        for img in clip:
            for func in img_transforms:
                jittered_img = func(img)
            jittered_clip.append(jittered_img)

        
        return jittered_clip

class CenterCropResize(object):
    """
    Extract a specific crop of the video and resize back to the original input size.

    Args:
        x_range (tuple): (start, end) pixel range for cropping on the x-axis.
        y_range (tuple): (start, end) pixel range for cropping on the y-axis.
    """
    def __init__(self, x_range, y_range):
        if not (isinstance(x_range, tuple) and len(x_range) == 2):
            raise ValueError('x_range must be a tuple of length 2.')
        if not (isinstance(y_range, tuple) and len(y_range) == 2):
            raise ValueError('y_range must be a tuple of length 2.')

        self.x_range = x_range
        self.y_range = y_range

    def __call__(self, clip):
        crop_x_start, crop_x_end = self.x_range
        crop_y_start, crop_y_end = self.y_range

        cropped_clip = []
        for frame in clip:
            if isinstance(frame, PIL.Image.Image):
                frame = np.asarray(frame)
            # Crop the specific region
            cropped_frame = frame[crop_y_start:crop_y_end, crop_x_start:crop_x_end, :]

            # Resize back to the original size
            resized_frame = cv2.resize(cropped_frame, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
            cropped_clip.append(resized_frame)

        return np.stack(cropped_clip, axis=0)