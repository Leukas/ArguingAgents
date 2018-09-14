import argparse
import os

from training import train_gan
from data import get_MNIST_dataloaders, get_EEG_dataloaders

from networks import device
from networks.generator import Generator
from networks.discriminator import Discriminator

os.makedirs('images', exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=10, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=32, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
parser.add_argument('--img_size', type=int, default=28, help='size of each image dimension')
parser.add_argument('--channels', type=int, default=1, help='number of image channels')
parser.add_argument('--sample_interval', type=int, default=400, help='interval betwen image samples')
args = parser.parse_args()
print(args)

img_shape = (args.channels, args.img_size, args.img_size)
img_shape = (1, 430, 128)
# dataloader = get_MNIST_dataloaders(args.batch_size)
dataloaders = get_EEG_dataloaders(args.batch_size)
print(len(dataloaders['train']))
for data, labels in dataloaders['train']:
	print(data.size(), labels.size())
	break

# Initialize generator and discriminator
generator = Generator(img_shape).to(device)
discriminator = Discriminator(img_shape).to(device)
gan = {}
gan['g'] = generator
gan['d'] = discriminator


train_gan(gan, dataloaders['train'], epochs=args.n_epochs)
    