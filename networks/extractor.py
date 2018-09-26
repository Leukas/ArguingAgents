import numpy as np

import torch
import torch.nn as nn

class FeatureExtractorFMRI(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv3d(1, 16, 3, padding=1),
            nn.Conv3d(16, 16, 3, padding=1, stride=(4,4,3)),
            nn.ReLU(),
            nn.Conv3d(16, 32, 3, padding=1),
            nn.Conv3d(32, 32, 3, padding=1, stride=(2,2,3)),
            nn.ReLU(),
            nn.Conv3d(32, 64, 3, padding=1),
            nn.Conv3d(64, 64, 3, padding=1, stride=(2,2,2)),
            nn.ReLU(),

        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(64, 64, 3, padding=1),
            nn.ConvTranspose3d(64, 32, 3, padding=1, stride=(2,2,2), output_padding=(1,1,1)),
            nn.ReLU(),
            nn.ConvTranspose3d(32, 32, 3, padding=1),
            nn.ConvTranspose3d(32, 16, 3, padding=1, stride=(2,2,3), output_padding=(1,1,2)),
            nn.ReLU(),
            nn.ConvTranspose3d(16, 16, 3, padding=1),
            nn.ConvTranspose3d(16, 1, 3, padding=1, stride=(4,4,3), output_padding=(3,3,2)),
            nn.Tanh(),

        )
        self.fc = nn.Linear(64*16, 40)

    def encode(self, x):
        return self.encoder(x)

    def forward(self, x):
        x = self.encoder(x)
        # print(x.size())
        x2 = x.view(-1, 64*4*4*1)
        x2 = self.fc(x2)
        # x = x.view(-1, 64,4,4,1)
        x = self.decoder(x)

        return x, x2