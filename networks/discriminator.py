import numpy as np

import torch
import torch.nn as nn
from torchvision.utils import save_image
from . import device
from .noise import add_black_box


class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(input_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity



class ConditionalDiscriminator(nn.Module):
    def __init__(self, input_shape, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.label_embedding = nn.Embedding(num_classes, num_classes)

        self.model = nn.Sequential(
            nn.Linear(num_classes + int(np.prod(input_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1)
        )

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, 3, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.Dropout(0.25),
            # nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(128, 128, 3, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.25),
        )
        self.fc = nn.Linear(128+num_classes, 1)

    def forward(self, img, labels):
        # Concatenate label embedding and image to produce input
        batch_size = img.size(0)
        x = self.conv(img)
        x = x.view(batch_size, -1)
        x = torch.cat((x, self.label_embedding(labels)), -1)
        x = self.fc(x)
        # d_in = torch.cat((img.view(img.size(0), -1), self.label_embedding(labels)), -1)
        # validity = self.model(d_in)
        return x


    def visualize(self, img, labels):
        boxed_imgs, dims = add_black_box(img, (2,2), stride=1)
        boxed_imgs = boxed_imgs.unsqueeze(1)
        # boxed_imgs = boxed_imgs.view(boxed_imgs.size(0), -1)
        labels = labels.expand(int(dims[0]*dims[1]), -1).t().flatten()

        # d_in = torch.cat((boxed_imgs, self.label_embedding(labels)), -1)
        validity = self(boxed_imgs, labels)
        validity = validity.view(-1, 1, dims[0], dims[1])
        save_image(validity, './images/vis/sample.png')
        save_image(img, './images/vis/imgs.png')
        return validity
