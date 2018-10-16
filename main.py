import argparse
import os
import torch

from training import train_cgan, visualize_gan
from data import *

from networks import device
from networks.generator import ConditionalGenerator
from networks.discriminator import ConditionalDiscriminator
from networks.classifier import Classifier
from networks.gan import GAN

os.makedirs('images', exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=10, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=16, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
parser.add_argument('--img_size', type=int, default=32, help='size of each image dimension')
parser.add_argument('--latent_dim', type=int, default=300, help='size of latent dimension')
parser.add_argument('--cifar10', type=bool, default=False, help='To use CIFAR10, vs MNIST')
parser.add_argument('--sample_interval', type=int, default=1000, help='interval betwen image samples')
args = parser.parse_args()
print(args)

channels = 3 if args.cifar10 else 1
# img_shape = (args.channels, args.img_size, args.img_size)
# img_shape = (1, 430, 128)
if args.cifar10:
    dataloaders = get_CIFAR10_dataloaders(args.batch_size)
else:
    dataloaders = get_MNIST_dataloaders(args.batch_size)
# dataloaders = get_EEG_dataloaders(args.)


# Initialize generator and discriminator
generator = ConditionalGenerator(channels, 10, args.latent_dim).to(device)
discriminator = ConditionalDiscriminator(channels, 10).to(device)
classifier = Classifier(channels, 10).to(device)
gan = GAN(generator, discriminator, classifier).to(device)

# Load previous save

# model_path = .
# model_path = './models/cifar10/test300.pt'
# model_path = './models/mnist/testin9.pt'
# model_path = './models/mnist/test_video2.pt'
model_path = './models/mnist/testing2.pt'


if os.path.exists(model_path):
    gan.load_state_dict(torch.load(model_path))

# print(torch.cuda.is_available())
# train_cgan(gan, dataloaders['train'], epochs=args.n_epochs, sample_interval=2000, latent_dim=args.latent_dim, save_path=model_path)
visualize_gan(gan.eval(), dataloaders['test'], visualize_fake=False)
