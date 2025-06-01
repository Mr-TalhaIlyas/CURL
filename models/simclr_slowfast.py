#%%
# import os, psutil
# os.chdir('/home/user01/Data/fetal/new_scripts/')
from mmaction.models.backbones import ResNet3dSlowFast
import torch
import torch.nn as nn
from models.utils.tools import load_pretrained_chkpt

# pretrained_path ='C:/Users/talha/Desktop/MICCAI/models/pretrained/slowfast_r50_4x16x1_kinetics400-rgb.pth'

class SlowFast_Encoder(nn.Module):
    def __init__(self, config):
        
        super(SlowFast_Encoder, self).__init__()
        self.dropout_ratio = config['dropout_ratio']
        self.num_classes = config['num_classes']
        self.pretrained_path = config['slowfast_pretrained_chkpts']

        self.backbone = ResNet3dSlowFast()

        in_channels = 2048+256

        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

        # self.reducer = nn.Sequential(
        #     nn.Linear(in_channels, 512, bias=False),
        #     nn.ReLU(),
        #     nn.Linear(512, 512, bias=False),
        # )

        if self.pretrained_path is not None:
            self.init_weights(self.pretrained_path)

    def forward(self, x):
        # [B, C, T, H, W]
        x = self.backbone(x)
        # ([N, channel_slow, T1, H, W], [(N, channel_fast, T2, H, W)])
        x_slow, x_fast = x
        # ([N, channel_slow, 1, 1, 1], [N, channel_fast, 1, 1, 1])
        x_slow = self.avg_pool(x_slow)
        x_fast = self.avg_pool(x_fast)
        # [N, channel_fast + channel_slow, 1, 1, 1]
        x = torch.cat((x_fast, x_slow), dim=1)
        # [N, feats]
        x = x.view(x.size(0), -1)

        # x = self.reducer(x)
        
        return x

    def init_weights(self, pretrained_path=None):
        print('Loading SlowFast pretrained chkpts...')
        if pretrained_path is not None:
            chkpt = torch.load(pretrained_path)

            pretrained_dict = chkpt['state_dict']
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

class SimCLR(nn.Module):
    """
    We opt for simplicity and adopt the commonly used ResNet (He et al., 2016) to obtain hi = f(x ̃i) = ResNet(x ̃i) where hi ∈ Rd is the output after the average pooling layer.
    """

    def __init__(self, encoder, projection_dim=256, n_features=2048+256): #2048+256 # 512
        super(SimCLR, self).__init__()

        self.encoder = encoder
        self.n_features = n_features

        # We use a MLP with one hidden layer to obtain z_i = g(h_i) = W(2)σ(W(1)h_i) where σ is a ReLU non-linearity.
        self.projector = nn.Sequential(
            nn.Linear(self.n_features, self.n_features, bias=False),
            nn.ReLU(),
            nn.Linear(self.n_features, projection_dim, bias=False),
        )

    def forward(self, x_i, x_j):
        h_i = self.encoder(x_i)
        h_j = self.encoder(x_j)

        z_i = self.projector(h_i)
        z_j = self.projector(h_j)
        return h_i, h_j, z_i, z_j
