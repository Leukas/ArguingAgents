import torch
import torch.nn as nn

class GAN(nn.Module):
    def __init__(self, generator, discriminator, classifier=None):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.classifier = classifier
        self.g = self.generator
        self.d = self.discriminator
        self.c = self.classifier
