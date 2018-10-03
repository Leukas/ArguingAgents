import argparse
import os
import torch

from training import train_gan, train_cgan, visualize_gan
from data import get_MNIST_dataloaders, get_EEG_dataloaders

from networks import device
from networks.generator import ConditionalGenerator
from networks.discriminator import ConditionalDiscriminator
from networks.gan import GAN

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
# img_shape = (1, 430, 128)
dataloaders = get_MNIST_dataloaders(args.batch_size)
# dataloaders = get_EEG_dataloaders(args.batch_size)


# Initialize generator and discriminator
generator = ConditionalGenerator(img_shape, 10).to(device)
discriminator = ConditionalDiscriminator(img_shape, 10).to(device)
gan = GAN(generator, discriminator)

# Load previous save
model_path = './models/mnist/mnist.pt'
if os.path.exists(model_path):
	gan.load_state_dict(torch.load(model_path))

train_cgan(gan, dataloaders['train'], epochs=args.n_epochs, sample_interval=2000, save_path=model_path)
# visualize_gan(gan, dataloaders['test'], visualize_fake=False)