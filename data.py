# data.py
import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets

# Configure data loader
def MNIST_dataset(batch_size):
	#os.makedirs('../../data/mnist', exist_ok=True)
	if not os.path.exists(os.path.join('data/mnist')):
		os.makedirs(os.path.join('data/mnist'))
	dataloader = torch.utils.data.DataLoader(
	    datasets.MNIST('data/mnist',
	        train=True, 
	        download=True,
	        transform=transforms.Compose([
	            transforms.ToTensor(),
	            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
	        ])),
	    batch_size=batch_size, shuffle=True)
	return dataloader
