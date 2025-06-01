import numpy as np
from vidaug import augmentors as va
import random
from data.vid_transforms import ColorJitter, CenterCropResize

sometimes = lambda aug: va.Sometimes(0.70, aug) # Used to apply augmentor with 50% probability
oneof = lambda aug: va.OneOf(aug)
someof = lambda aug: va.SomeOf(aug, 2, random_order=True)

seq = va.Sequential([
    someof([
    sometimes(va.HorizontalFlip()),
    # sometimes(va.VerticalFlip()),
    sometimes(va.RandomTranslate(x=50, y=50)),
    # New
    sometimes(va.RandomRotate(degrees=30)),
    sometimes(CenterCropResize((50,175),(50,175))),
    sometimes(ColorJitter(0.8,0.8,0.8,0.2)),

    sometimes(va.Salt(ratio=40)),
    sometimes(va.Pepper(ratio=40)),
    
    # sometimes(va.GaussianBlur(sigma=1)),
    # sometimes(va.RandomShear(x=0.2, y=0.2)), # less than 1
    ])
    ],
    random_order = True
)


def video_augment(video):
    video_aug = seq(video)
    return np.stack(video_aug, axis=0)


seq1 = va.Sequential([
    someof([
    sometimes(CenterCropResize((50,175),(50,175))),
    sometimes(va.HorizontalFlip()),
    sometimes(ColorJitter(0.8,0.8,0.8,0.2)),
    sometimes(va.RandomTranslate(x=50, y=50)),
    sometimes(va.RandomRotate(degrees=30)),
    ])
    ],
    random_order = True
)

seq2 = va.Sequential([
    someof([
    sometimes(CenterCropResize((50,175),(50,175))),
    sometimes(va.HorizontalFlip()),
    sometimes(ColorJitter(0.8,0.8,0.8,0.2)),
    sometimes(va.RandomTranslate(x=50, y=50)),
    sometimes(va.RandomRotate(degrees=30)),
    ])
    ],
    random_order = True
)

def SimCLR_augment(video):
    video_aug1 = seq1(video)
    video_aug2 = seq2(video)
    return np.stack(video_aug1, axis=0), np.stack(video_aug2, axis=0)

#%%

# import numpy as np
# from vidaug import augmentors as va
# import random

# sometimes = lambda aug: va.Sometimes(0.99, aug) # Used to apply augmentor with 50% probability
# oneof = lambda aug: va.OneOf(aug)
# someof = lambda aug: va.SomeOf(aug, 3, random_order=True)

# seq = va.Sequential([
#     someof([
#     sometimes(va.HorizontalFlip()),
#     # sometimes(va.VerticalFlip()),
#     sometimes(va.RandomTranslate(x=50, y=50)),
#     # New
#     sometimes(va.RandomRotate(degrees=30)),
#     sometimes(va.Salt(ratio=60)),
#     sometimes(va.Pepper(ratio=60)),
#     sometimes(va.GaussianBlur(sigma=1)),
#     sometimes(va.RandomShear(x=0.2, y=0.2)), # blacks out all pixels needs fixing
#     sometimes(va.Add(value=np.random.randint(30,150))),
#     sometimes(va.Multiply(value=random.uniform(0.2,0.8))),
#     sometimes(ColorJitter(0.8,0.8,0.8,0.2)),
#     sometimes(CenterCropResize((50,175),(50,175))),
    
    
#     ])
#     ],
#     random_order = True
# )


# def video_augment(video):
#     video_aug = seq(video)
#     return np.stack(video_aug, axis=0)

# import moviepy.editor as mp
# clip = mp.VideoFileClip("/home/user01/Data/fetal/new_scripts/data/MNIST/raw/original.gif")
# x=[]
# t = [x.append(frame) for frame in clip.iter_frames()]
# x = np.stack(x, axis=0)

# y = video_augment(x)
# HTML(display_video(y).to_html5_video())

#%%

# seq1 = va.Sequential([
#     someof([
#     sometimes(va.HorizontalFlip()),
#     # sometimes(va.VerticalFlip()),
#     sometimes(va.RandomTranslate(x=50, y=50)),
#     sometimes(va.RandomRotate(degrees=30)),
#     sometimes(va.Salt(ratio=60)),
#     sometimes(va.Pepper(ratio=60)),
#     sometimes(va.GaussianBlur(sigma=1)),
#     sometimes(va.RandomShear(x=0.2, y=0.2)), 
#     sometimes(va.Add(value=np.random.randint(30,150))),
#     sometimes(va.Multiply(value=random.uniform(0.2,0.8))),
#     sometimes(ColorJitter(0.8,0.8,0.8,0.2)),
#     sometimes(CenterCropResize((50,175),(50,175))),
#     ])
#     ],
#     random_order = True
# )

# seq2 = va.Sequential([
#     someof([
#     sometimes(va.HorizontalFlip()),
#     # sometimes(va.VerticalFlip()),
#     sometimes(va.RandomTranslate(x=50, y=50)),
#     sometimes(va.RandomRotate(degrees=30)),
#     sometimes(va.Salt(ratio=60)),
#     sometimes(va.Pepper(ratio=60)),
#     sometimes(va.GaussianBlur(sigma=1)),
#     sometimes(va.RandomShear(x=0.2, y=0.2)),
#     sometimes(va.Add(value=np.random.randint(30,150))),
#     sometimes(va.Multiply(value=random.uniform(0.2,0.8))),
#     sometimes(ColorJitter(0.8,0.8,0.8,0.2)),
#     sometimes(CenterCropResize((50,175),(50,175))),
#     ])
#     ],
#     random_order = True
# )