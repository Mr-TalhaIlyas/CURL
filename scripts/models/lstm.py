#%%
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np

# Feature extractor using a frozen VGG16 network
class VGG16FeatureExtractor(nn.Module):
    def __init__(self):
        super(VGG16FeatureExtractor, self).__init__()
        vgg16 = models.vgg16(pretrained=True)
        self.features = nn.Sequential(*list(vgg16.features.children()))
        self.avgpool = vgg16.avgpool
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(vgg16.classifier[0].in_features, 512)
        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

# Attention layer
class Attention(nn.Module):
    def __init__(self, input_dim):
        super(Attention, self).__init__()
        self.attention_weights = nn.Parameter(torch.Tensor(input_dim, 1))
        nn.init.xavier_uniform_(self.attention_weights)
        self.attention_bias = nn.Parameter(torch.Tensor(1, 1))
        nn.init.zeros_(self.attention_bias)

    def forward(self, x):
        e = torch.tanh(torch.matmul(x, self.attention_weights) + self.attention_bias)
        alpha = F.softmax(e, dim=1)
        context = x * alpha
        context = torch.sum(context, dim=1)
        return context

# BiLSTM with attention
class BiLSTMWithAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(BiLSTMWithAttention, self).__init__()
        self.bilstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True,
                              bidirectional=True, dropout=0.5)
        self.attention = Attention(hidden_dim * 2)

    def forward(self, x):
        x, _ = self.bilstm(x)
        x = self.attention(x)
        return x

# Complete model integrating all components
class CNN_LSTM(nn.Module):
    def __init__(self, num_classes=10, dropout_ratio=0.3):
        super(CNN_LSTM, self).__init__()
        self.num_classes = num_classes
        self.dropout_ratio = dropout_ratio
        self.feature_extractor = VGG16FeatureExtractor()
        self.bilstm_attention = BiLSTMWithAttention(input_dim=512, hidden_dim=256, num_layers=2)
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None

        self.fc = nn.Linear(256*2, self.num_classes)# softmax

    def forward(self, x):
        # permute x from BCTHW to BTCHW
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        batch_size, seq_len, channels, height, width = x.shape
        x = x.view(batch_size * seq_len, channels, height, width)
        features = self.feature_extractor(x)
        features = features.view(batch_size, seq_len, -1)
        # print(features.shape)
        features = self.bilstm_attention(features)
        # print(features.shape)
        if self.dropout is not None:
            x = self.dropout(x)
        output = self.fc(features)
        return output
#%%
# Example usage
# num_classes = 10  # Replace with the actual number of classes
# model = CNN_LSTM(num_classes)
# x = torch.randn([2,3,32,224,224])
# y = model(x)
# print(y.shape)
#%%
