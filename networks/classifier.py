import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
from . import device
from .noise import add_black_box


class Classifier(nn.Module):
    def __init__(self, num_channels, num_classes):
        super().__init__()

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

        self.fc = nn.Linear(128, num_classes)

    def forward(self, img):
        batch_size = img.size(0)
        x = self.conv(img)
        x = x.view(batch_size, -1)
        x = self.fc(x)

        return x

    def vis_layer(self, img, layer=5, feature_num=0):

        for feature_num in range(self.conv[layer].weight.size(0)):
            x = self.conv[0:layer+1](img)
            # print(x.size())
            x[:,:feature_num] = 0
            x[:,feature_num+1:] = 0
            # stride = self.conv[layer].stride
            # weight = torch.zeros(self.conv[layer].weight.size()).to(device)
            # weight[feature_num] = self.conv[layer].weight[feature_num]
            # weight = self.conv[layer].weight
            # x = F.conv_transpose2d(x, weight, stride=stride, padding=1)

            # print(x.size())
            irange = range(layer+1)[::-1]
            for i in irange:
                print(x.size())
                if i in [7, 14] :
                    continue 
                elif i in [1, 4, 8, 11, 13] :
                    x = F.leaky_relu(x, 0.1, inplace=True)
                else:
                    stride = self.conv[i].stride 
                    x = F.conv_transpose2d(x, self.conv[i].weight, stride=stride, padding=1)
            # print(weight.size())

            print(x.size())


            save_image(x, './images/vis/x_sample_%d.png' % feature_num, nrow=10)

    def visualize(self, img, labels):
        boxed_imgs, dims = add_black_box(img, (16,16), stride=1)
        boxed_imgs = boxed_imgs.unsqueeze(1)
        # boxed_imgs = boxed_imgs.view(boxed_imgs.size(0), -1)
        # labels = labels.expand(int(dims[0]*dims[1]), -1).t().flatten()

        # d_in = torch.cat((boxed_imgs, self.label_embedding(labels)), -1)
        classes = self(boxed_imgs)
        classes = F.softmax(classes, dim=1)

        print(classes.size(), labels.size())
        labels = labels.unsqueeze(1).expand(labels.size(0),int(dims[0]*dims[1])).contiguous().view(-1)
        print(classes.size(), labels.size())
        classes = torch.gather(classes, 1, labels.unsqueeze(1)) # class prob for label
        classes = classes.view(-1, 1, dims[0], dims[1])
        # classes = torch.max(classes, dim=1)[1]
        save_image(classes, './images/vis/c_sample.png')
        save_image(img, './images/vis/c_imgs.png')
        return classes
