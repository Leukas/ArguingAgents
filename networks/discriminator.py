import numpy as np

import torch
import torch.nn as nn
from torchvision.utils import save_image
from . import device
from .noise import add_black_box

class ConditionalDiscriminator(nn.Module):
    def __init__(self, num_channels, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.label_embedding = nn.Embedding(num_classes, num_classes)

        self.conv = nn.Sequential(
            nn.Conv2d(num_channels, 32, 3, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.Dropout(0.25),
            # nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(128, 128, 3, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.25),
        )
        self.fc = nn.Linear(128, 1)

    def forward(self, img, labels=None):
        # Concatenate label embedding and image to produce input
        batch_size = img.size(0)
        x = self.conv(img)
        x = x.view(batch_size, -1)
        # x = torch.cat((x, self.label_embedding(labels)), -1)
        x = self.fc(x)
        # d_in = torch.cat((img.view(img.size(0), -1), self.label_embedding(labels)), -1)
        # validity = self.model(d_in)
        return x


    def visualize(self, img):
        boxed_imgs, dims = add_black_box(img, (10, 10), stride=1)
        boxed_imgs = boxed_imgs.unsqueeze(1)
        # boxed_imgs = boxed_imgs.view(boxed_imgs.size(0), -1)
        # labels = labels.expand(int(dims[0]*dims[1]), -1).t().flatten()

        # d_in = torch.cat((boxed_imgs, self.label_embedding(labels)), -1)
        validity = self(boxed_imgs)
        validity = validity.view(-1, 1, dims[0], dims[1])
        save_image(validity, './images/vis/d_sample.png')
        save_image(img, './images/vis/d_imgs.png')
        return validity
