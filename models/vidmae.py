#%%
# import os, psutil
# os.chdir('/home/user01/Data/fetal/new_scripts/')
from mmaction.models.backbones import VisionTransformer
import torch
import torch.nn as nn
from models.utils.tools import load_pretrained_chkpt

# pretrained_path ='C:/Users/talha/Desktop/MICCAI/models/pretrained/slowfast_r50_4x16x1_kinetics400-rgb.pth'


class VidMAE(nn.Module):
    def __init__(self, config):
        
        super(VidMAE, self).__init__()
        self.dropout_ratio = config['dropout_ratio']
        self.num_classes = config['num_classes']
        self.pretrained_path = config['vidmae_pretrained_chkpts']

        self.backbone = VisionTransformer(embed_dims=384, depth=12, num_heads=6,
                                          num_frames= 24)

        in_channels = 384

        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        
        self.fc = nn.Linear(in_channels, self.num_classes)# softmax

        if self.pretrained_path is not None:
            self.init_weights(self.pretrained_path)

    def forward(self, x):
        # [B, C, T, H, W]
        x = self.backbone(x)
        # (B, embed_dim)

        if self.dropout is not None:
            x = self.dropout(x)
        # [N, feats]
        self.out_feats = x
        # [N x num_classes]
        cls_conf = self.fc(x)

        return cls_conf

    def init_weights(self, pretrained_path=None):
        print('Loading SlowFast pretrained chkpts...')
        if pretrained_path is not None:
            chkpt = torch.load(pretrained_path)

            pretrained_dict = chkpt#['state_dict']
            # load model state dict
            state = self.backbone.state_dict()
            # loop over both dicts and make a new dict where name and the shape of new state match
            # with the pretrained state dict.
            matched, unmatched = [], []
            new_dict = {}
            for i, j in zip(pretrained_dict.items(), state.items()):
                pk, pv = i # pretrained state dictionary
                nk, nv = j # new state dictionary
                # if name and weight shape are same
                if pk.strip('backbone.') == nk:# and pv.shape == nv.shape: #.strip('backbone.')
                    new_dict[nk] = pv
                    matched.append(pk)
                elif pv.shape == nv.shape:
                    new_dict[nk] = pv
                    matched.append(pk)
                else:
                    unmatched.append(pk)

            state.update(new_dict)
            self.backbone.load_state_dict(state, strict=False)
            print('Pre-trained SlowFast state loaded successfully...')
            print(f'Mathed kyes: {len(matched)}, Unmatched Keys: {len(unmatched)}')
            print(40*'=')
            print(unmatched)
        # END


#%%
# from configs.config import config
# model = SlowFast(config['model'],
# pretrained_path ="/home/talha/Data/mme/scripts/models/pretrained/slowfast_r50_4x16x1_kinetics400-rgb.pth",
# )

# model.backbone.load_state_dict(pretrained_path['state_dict'])
# x = torch.randn([2,3,32,224,224])

# y = model(x)

# from torchinfo import summary
# summary(model, (8,3,48,224,224))


# %%
