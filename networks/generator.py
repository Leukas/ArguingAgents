import numpy as np
import torch
import torch.nn as nn


class ConditionalGenerator(nn.Module):
    def __init__(self, num_channels, num_classes, latent_dim=100):
        super().__init__()
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.label_emb = nn.Embedding(num_classes, num_classes)

        def block(in_feat, out_feat, normalize=True):
            layers = [  nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.init_channels = 128
        self.fc = nn.Linear(latent_dim + num_classes, self.init_channels * 8 * 8)


        self.conv = nn.Sequential(
            nn.BatchNorm2d(self.init_channels),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(self.init_channels, self.init_channels, 3, stride=1, padding=1),
            nn.Conv2d(self.init_channels, self.init_channels, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.init_channels, self.init_channels, 3, stride=1, padding=1),
            nn.BatchNorm2d(self.init_channels, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(self.init_channels, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.Conv2d(32, num_channels, 3, stride=1, padding=1),
            # nn.Conv2d(num, num_channels, 3, stride=1, padding=1),
            nn.Tanh()

        )

    def forward(self, z, labels):
        # Concatenate label embedding and image to produce input
        gen_input = torch.cat((self.label_emb(labels), z), -1)
        # print(gen_input.size())
        gen_input = self.fc(gen_input)
        gen_input = gen_input.view(-1, self.init_channels, 8, 8)
        # gen_input = gen_input.unsqueeze(1)
        # gen_input = gen_input.unsqueeze(2)
        img = self.conv(gen_input)
# 
        # print(img.size())
        # img = self.model(gen_input)
        # img = img.view(img.size(0), *self.input_shape)
        return img


    def visualize(self, z, labels):
        # Concatenate label embedding and image to produce input
        gen_input = torch.cat((self.label_emb(labels), z), -1)
        # print(gen_input.size())
        gen_input = self.fc(gen_input)
        gen_input = gen_input.view(-1, self.init_channels, 8, 8)
        # gen_input = gen_input.unsqueeze(1)
        # gen_input = gen_input.unsqueeze(2)
        img = self.conv(gen_input)
# 
        # print(img.size())
        # img = self.model(gen_input)
        # img = img.view(img.size(0), *self.input_shape)
        return img