import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
from . import device
from .noise import black_box_module, add_black_boxes

class ConditionalDiscriminator(nn.Module):
    def __init__(self, num_channels, num_classes):
        super().__init__()
        # self.num_classes = num_classes
        # self.label_embedding = nn.Embedding(num_classes, num_classes)

        self.conv = nn.Sequential(
            nn.Conv2d(num_channels, 8, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 8, 3, stride=2, padding=1),
            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, stride=2, padding=1),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.Dropout(0.25),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
        )

        self.fc = nn.Linear(64, 1)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, img, labels=None):
        # Concatenate label embedding and image to produce input
        batch_size = img.size(0)
        x = self.conv(img)
        x = x.view(batch_size, -1)
        x = self.fc(x)
        x = torch.sigmoid(x)
        return x

    def classify(self, img, labels=None):
        # Concatenate label embedding and image to produce input
        batch_size = img.size(0)
        x = self.conv(img)
        x = x.view(batch_size, -1)
        x = self.fc2(x)
        return x

    def visualize(self, img):
        # boxed_imgs, dims = add_black_boxes(img, (10, 10), stride=1)
        # boxed_imgs = boxed_imgs.unsqueeze(1)
        boxed_imgs, dims = black_box_module(img, (10, 10), stride=1)
        # boxed_imgs = boxed_imgs.view(boxed_imgs.size(0), -1)
        # labels = labels.expand(int(dims[0]*dims[1]), -1).t().flatten()

        # d_in = torch.cat((boxed_imgs, self.label_embedding(labels)), -1)
        validity = self(boxed_imgs)
        validity = validity.view(-1, 1, dims[0], dims[1])
        save_image(validity, './images/vis/d_sample.png')
        save_image(img, './images/vis/d_imgs.png')
        return validity

    def visualize_class(self, img, labels):
        # boxed_imgs, dims = add_black_boxes(img, (10, 10), stride=1)
        # boxed_imgs = boxed_imgs.unsqueeze(1)
        boxed_imgs, dims = black_box_module(img, (10, 10), stride=1)
        # boxed_imgs = boxed_imgs.view(boxed_imgs.size(0), -1)
        # labels = labels.expand(int(dims[0]*dims[1]), -1).t().flatten()
        classes = self.classify(boxed_imgs)
        classes = F.softmax(classes, dim=1)

        # print(classes.size(), labels.size())
        labels = labels.unsqueeze(1).expand(labels.size(0),int(dims[0]*dims[1])).contiguous().view(-1)
        classes = torch.gather(classes, 1, labels.unsqueeze(1)) # class prob for label
        classes = classes.view(-1, 1, dims[0], dims[1])
        # classes = torch.max(classes, dim=1)[1]
        save_image(classes, './images/vis/dc_sample.png')
        save_image(img, './images/vis/dc_imgs.png')
        return classes