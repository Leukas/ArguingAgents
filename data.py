# data.py
import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets

# Configure data loader
def get_MNIST_dataloaders(batch_size):
	os.makedirs('../../data/mnist', exist_ok=True)
	dataloaders = {}
	dataloaders['train'] = torch.utils.data.DataLoader(
	    datasets.MNIST('../../data/mnist', 
	        train=True, 
	        download=True,
	        transform=transforms.Compose([
	            transforms.ToTensor(),
	            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
	        ])),
	    batch_size=batch_size, shuffle=True)

	dataloaders['test'] = torch.utils.data.DataLoader(
	    datasets.MNIST('../../data/mnist', 
	        train=False, 
	        download=True,
	        transform=transforms.Compose([
	            transforms.ToTensor(),
	            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
	        ])),
	    batch_size=batch_size)

	return dataloaders


# Dataset class
class EEGDataset:
    # Constructor
    def __init__(self, eeg_signals_path="./data/eeg_signals_128_sequential_band_all_with_mean_std.pth"):
        # Load EEG signals
        loaded = torch.load(eeg_signals_path)
        self.data = loaded["dataset"]
        self.labels = loaded["labels"]
        print(self.labels)
        self.images = loaded["images"]
        self.means = loaded["means"]
        self.stddevs = loaded["stddevs"]
        # Compute size
        self.size = len(self.data)

    # Get size
    def __len__(self):
        return self.size

    # Get item
    def __getitem__(self, i):
        # Process EEG
        eeg = ((self.data[i]["eeg"].float() - self.means)/self.stddevs).t()
        eeg = eeg[20:450,:].unsqueeze(0)
        # Get label
        label = self.data[i]["label"]
        # Return
        return eeg, label

# Splitter class
class Splitter:
    def __init__(self, dataset, split_path="./data/splits_by_image.pth", split_num=0, split_name="train"):
        # Set EEG dataset
        self.dataset = dataset
        # Load split
        loaded = torch.load(split_path)
        self.split_idx = loaded["splits"][split_num][split_name]
        # Filter data
        self.split_idx = [i for i in self.split_idx if 450 <= self.dataset.data[i]["eeg"].size(1) <= 600]
        # Compute size
        self.size = len(self.split_idx)

    # Get size
    def __len__(self):
        return self.size

    # Get item
    def __getitem__(self, i):
        # Get sample from dataset
        eeg, label = self.dataset[self.split_idx[i]]
        # Return
        return eeg, label

def get_EEG_dataloaders(batch_size=32, dataset_path="./data/eeg_signals_128_sequential_band_all_with_mean_std.pth",
						splits_path="./data/splits_by_image.pth"):
	# Load dataset	
	dataset = EEGDataset()
	# Create loaders
	loaders = {split: DataLoader(
		Splitter(dataset, split_name=split), 
		batch_size = batch_size, shuffle=True) 
		for split in ["train", "val", "test"]}

	return loaders