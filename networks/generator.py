import numpy as np
import torch
import torch.nn as nn
<<<<<<< HEAD
from . import device
=======
>>>>>>> master

class Generator(nn.Module):
    def __init__(self, input_shape, latent_dim=100):
        super().__init__()
        self.input_shape = input_shape

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(input_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *self.input_shape)
        return img

<<<<<<< HEAD
=======
class ConditionalGenerator(nn.Module):
    def __init__(self, input_shape, num_classes, latent_dim=100):
        super().__init__()
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.label_emb = nn.Embedding(num_classes, num_classes)

        def block(in_feat, out_feat, normalize=True):
            layers = [  nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim+num_classes, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(input_shape))),
            nn.Tanh()
        )

        self.fc = nn.Linear(latent_dim + num_classes, 128 * 8 * 8)


        self.conv = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 1, 3, stride=1, padding=1),
            nn.Tanh()

            # nn.Conv2d()
            # nn.ConvTranspose2d(latent_dim + num_classes, 64, 3, stride=2, padding=1, output_padding=1),
            # nn.BatchNorm2d(128),

            # nn.LeakyReLU(0.2, inplace=True),
            # nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            # nn.BatchNorm2d(32),
            # nn.LeakyReLU(0.2, inplace=True),
            # nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1),
            # nn.BatchNorm2d(16),
            # nn.LeakyReLU(0.2, inplace=True),
            # nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1),
            # nn.BatchNorm2d(8),
            # nn.LeakyReLU(0.2, inplace=True),
            # nn.ConvTranspose2d(8, 1, 3, stride=2, padding=1, output_padding=1),
            # nn.LeakyReLU(0.2, inplace=True),
            # nn.Tanh(),
            # nn.ConvTranspose2d(latent_dim + num_classes, 64, 3),
        )

    def forward(self, z, labels):
        # Concatenate label embedding and image to produce input
        gen_input = torch.cat((self.label_emb(labels), z), -1)
        # print(gen_input.size())
        gen_input = self.fc(gen_input)
        gen_input = gen_input.view(-1, 128, 8, 8)
        # gen_input = gen_input.unsqueeze(1)
        # gen_input = gen_input.unsqueeze(2)
        img = self.conv(gen_input)
# 
        # print(img.size())
        # img = self.model(gen_input)
        # img = img.view(img.size(0), *self.input_shape)
        return img
>>>>>>> master
