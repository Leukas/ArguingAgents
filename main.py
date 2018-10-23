import argparse
import os
import torch
import copy

from training import *
from data import *

from networks import device
from networks.generator import ConditionalGenerator
from networks.discriminator import ConditionalDiscriminator
from networks.classifier import Classifier
from networks.gan import GAN

os.makedirs('images', exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=5, help='Number of epochs of training')
parser.add_argument('--batch_size', type=int, default=16, help='Size of the batches')
parser.add_argument('--lr', type=float, default=0.0002, help='Adam: learning rate')
parser.add_argument('--img_size', type=int, default=32, help='Size of each image dimension')
parser.add_argument('--latent_dim', type=int, default=300, help='Size of latent dimension')
parser.add_argument('--cifar10', type=bool, default=False, help='To use CIFAR10, vs MNIST')
parser.add_argument('--sample_interval', type=int, default=1000, help='Interval betwen image samples')
parser.add_argument('--train_disc', type=int, default=1, help='Train the discriminator. 0=False/1=True, default 1.')
parser.add_argument('--train_gen', type=int, default=1, help='Train the generator. 0=False/1=True, default 1.')
parser.add_argument('--shared_weights', type=int, default=0, help='Train GAN with shared weights between discriminator and classifier. Default 0. Requires train_disc and train_gen to be 1.')
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
# print(dataloaders[])


num_classes = 10

# dataloaders = get_Feret_dataloaders(args.batch_size, img_size=(64,64))

# Initialize generator and discriminator
generator = ConditionalGenerator(channels, num_classes, args.latent_dim).to(device)
discriminator = ConditionalDiscriminator(channels, num_classes).to(device)
classifier = Classifier(channels, num_classes).to(device)
gan = GAN(generator, discriminator, classifier).to(device)

# Load previous save
# model_path = './models/mnist/testing_shared4.pt'
# model_path = './models/mnist/testing_seperate.1.pt'
model_path = './models/sample_disc.pt'


if os.path.exists(model_path):
    gan.load_state_dict(torch.load(model_path))

# sample = iter(dataloaders['test'])
# sample, label = next(sample)
# sample = sample.to(device)
# label = label.to(device)

# print(sample.size())
# gan.vis_layer(img=sample, layer=0, classifier=False)
# print(torch.cuda.is_available())
if args.train_disc and args.train_gen:
    if args.shared_weights:
        train_cgan_shared_weights(gan, dataloaders['train'], epochs=args.n_epochs, sample_interval=args.sample_interval, latent_dim=args.latent_dim, save_path=model_path)
    else:
        train_cgan(gan, dataloaders['train'], epochs=args.n_epochs, sample_interval=args.sample_interval, latent_dim=args.latent_dim, save_path=model_path)
elif args.train_disc:
    gan = train_disc(gan, dataloaders['train'], latent_dim=args.latent_dim)
elif args.train_gen:
    gan = train_gen(gan, dataloaders['train'], latent_dim=args.latent_dim)

# visualize_gan(gan.eval(), dataloaders['test'], visualize_fake=True)


# visualize_gan(gan.eval(), dataloaders['test'], layer=0, visualize_fake=True)
# visualize_gan(gan.eval(), dataloaders['test'], layer=0, visualize_fake=False)
