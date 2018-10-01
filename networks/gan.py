import torch
import torch.nn as nn

class GAN(nn.Module):
    def __init__(self, generator, discriminator):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.g = self.generator
        self.d = self.discriminator

