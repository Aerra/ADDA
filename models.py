from torch import nn
import torch.nn.functional as F

class EncoderNet(nn.Module):
    def __init__(self, h, w, kernel_size_1, kernel_size_2, out_ch_1, out_ch_2):
        super().__init__()
        h_out = (((h - kernel_size_1 + 1)/2) - kernel_size_2 + 1)/2
        w_out = (((w - kernel_size_1 + 1)/2) - kernel_size_2 + 1)/2
        in_features = int(h_out * w_out * out_ch_2)

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, out_ch_1, kernel_size=kernel_size_1),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(out_ch_1, out_ch_2, kernel_size=kernel_size_2),
            nn.Dropout2d(),
            nn.MaxPool2d(2),
            nn.ReLU(),
        )
        self.fc1 = nn.Linear(in_features, 500)

    def forward(self, x):
        features = self.feature_extractor(x)
        features = features.view(x.shape[0], -1)
        out = self.fc1(features)
        return out

class ClassifierNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc2 = nn.Linear(500, 10)
    def forward(self, x):
        out = F.dropout(F.relu(x), training = self.training)
        out = self.fc2(out)
        return out

class AddaDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 2),
            nn.LogSoftmax()
        )

    def forward(self, x):
        overlap = self.discriminator(x)
        return overlap