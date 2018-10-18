# experiment.py
import os
import torch

from training import *
from data import *

from networks import device
from networks.generator import ConditionalGenerator
from networks.discriminator import ConditionalDiscriminator
from networks.classifier import Classifier
from networks.gan import GAN


# EXPERIMENT 1: Visualizing weights in a GAN w/ classifier on MNIST
def exp1():
    dataloaders = get_MNIST_dataloaders(32)
    generator = ConditionalGenerator(1, 10, 300).to(device)
    discriminator = ConditionalDiscriminator(1, 10).to(device)
    classifier = Classifier(1, 10).to(device)
    gan = GAN(generator, discriminator, classifier).to(device)

    model_path = './models/mnist/exp1.pt'

    if os.path.exists(model_path): # assume training is done
        gan.load_state_dict(torch.load(model_path))
    else:
        train_cgan(
            gan, 
            dataloaders['train'], 
            epochs=5, 
            sample_interval=2000, 
            latent_dim=300, 
            save_path=model_path)


# EXPERIMENT 2: Visualizing weights when C and D share weights on MNIST
def exp2():
    dataloaders = get_MNIST_dataloaders(32)
    generator = ConditionalGenerator(1, 10, 300).to(device)
    discriminator = ConditionalDiscriminator(1, 10).to(device)
    classifier = Classifier(1, 10).to(device)
    gan = GAN(generator, discriminator, classifier).to(device)

    model_path = './models/mnist/exp1.pt'

    if os.path.exists(model_path): # assume training is done
        gan.load_state_dict(torch.load(model_path))
    else:
        train_cgan_shared_weights(
            gan, 
            dataloaders['train'], 
            epochs=5, 
            sample_interval=2000, 
            latent_dim=300, 
            save_path=model_path)

# EXPERIMENT 3: Train model with information 
def exp3():
    pass


