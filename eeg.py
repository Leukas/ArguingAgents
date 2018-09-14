# EEG code from the EEG to Pizza paper. No need to run, just here for reference




# Define options
import argparse
parser = argparse.ArgumentParser(description="Template")
# Dataset options
# parser.add_argument('-ed', '--eeg-dataset', default="./data/eeg_signals_band_all.pth", help="EEG dataset path")
parser.add_argument('-ed', '--eeg-dataset', default="./data/eeg_signals_128_sequential_band_all_with_mean_std.pth", help="EEG dataset path")
parser.add_argument('-sp', '--splits-path', default="./data/splits_by_image.pth", help="splits path")
parser.add_argument('-sn', '--split-num', default=0, type=int, help="split number")
# Model options
parser.add_argument('-ll', '--lstm-layers', default=1, type=int, help="LSTM layers")
parser.add_argument('-ls', '--lstm-size', default=128, type=int, help="LSTM hidden size")
parser.add_argument('-os', '--output-size', default=128, type=int, help="output layer size")
# Training options
parser.add_argument("-b", "--batch_size", default=16, type=int, help="batch size")
parser.add_argument('-o', '--optim', default="Adam", help="optimizer")
parser.add_argument('-lr', '--learning-rate', default=0.001, type=float, help="learning rate")
parser.add_argument('-lrdb', '--learning-rate-decay-by', default=0.5, type=float, help="learning rate decay factor")
parser.add_argument('-lrde', '--learning-rate-decay-every', default=10, type=int, help="learning rate decay period")
parser.add_argument('-dw', '--data-workers', default=4, type=int, help="data loading workers")
parser.add_argument('-e', '--epochs', default=100, type=int, help="training epochs")
# Backend options
parser.add_argument('--no-cuda', default=False, help="disable CUDA", action="store_true")

# Parse arguments
opt = parser.parse_args()

# Imports
import sys
import os
import random
import math
import time
import torch; torch.utils.backcompat.broadcast_warning.enabled = True
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torch.backends.cudnn as cudnn; cudnn.benchmark = True




# Load dataset
dataset = EEGDataset(opt.eeg_dataset)
# Create loaders
loaders = {split: DataLoader(Splitter(dataset, split_path = opt.splits_path, split_num = opt.split_num, split_name = split), batch_size = opt.batch_size, drop_last = True, shuffle = True) for split in ["train", "val", "test"]}

# Define model
class Model(nn.Module):

    def __init__(self, input_size, lstm_size, lstm_layers, output_size):
        # Call parent
        super().__init__()
        # Define parameters
        self.input_size = input_size
        self.lstm_size = lstm_size
        self.lstm_layers = lstm_layers
        self.output_size = output_size
        # Define internal modules
        self.lstm = nn.LSTM(input_size, lstm_size, num_layers=lstm_layers, batch_first=True)
        self.output = nn.Linear(lstm_size, output_size)

    def forward(self, x):
        # Prepare LSTM initiale state
        batch_size = x.size(0)
        lstm_init = (torch.zeros(self.lstm_layers, batch_size, self.lstm_size), torch.zeros(self.lstm_layers, batch_size, self.lstm_size))
        if x.is_cuda: lstm_init = (lstm_init[0].cuda(), lstm_init[0].cuda())
        lstm_init = (Variable(lstm_init[0], volatile=x.volatile), Variable(lstm_init[1], volatile=x.volatile))
        # Forward LSTM and get final state
        x = self.lstm(x, lstm_init)[0][:,-1,:]
        # Forward output
        x = self.output(x)
        return x

model = Model(128, opt.lstm_size, opt.lstm_layers, opt.output_size)
optimizer = getattr(torch.optim, opt.optim)(model.parameters(), lr = opt.learning_rate)
    
# Setup CUDA
if not opt.no_cuda:
    model.cuda()
    print("Copied to CUDA")

# Start training
for epoch in range(1, opt.epochs+1):
    # Initialize loss/accuracy variables
    losses = {"train": 0, "val": 0, "test": 0}
    accuracies = {"train": 0, "val": 0, "test": 0}
    counts = {"train": 0, "val": 0, "test": 0}
    # Adjust learning rate for SGD
    if opt.optim == "SGD":
        lr = opt.learning_rate * (opt.learning_rate_decay_by ** (epoch // opt.learning_rate_decay_every))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    # Process each split
    for split in ("train", "val", "test"):
        # Set network mode
        if split == "train":
            model.train()
        else:
            model.eval()
        # Process all split batches
        for i, (input, target) in enumerate(loaders[split]):
            # Check CUDA
            if not opt.no_cuda:
                input = input.cuda(async = True)
                target = target.cuda(async = True)
            # Wrap for autograd
            input = Variable(input, volatile = (split != "train"))
            target = Variable(target, volatile = (split != "train"))
            # Forward
            output = model(input)
            loss = F.cross_entropy(output, target)
            losses[split] += loss.data[0]
            # Compute accuracy
            _,pred = output.data.max(1)
            correct = pred.eq(target.data).sum()
            accuracy = correct/input.data.size(0)
            accuracies[split] += accuracy
            counts[split] += 1
            # Backward and optimize
            if split == "train":
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    # Print info at the end of the epoch
    print("Epoch {0}: TrL={1:.4f}, TrA={2:.4f}, VL={3:.4f}, VA={4:.4f}, TeL={5:.4f}, TeA={6:.4f}".format(epoch,
                                                                                                         losses["train"]/counts["train"],
                                                                                                         accuracies["train"]/counts["train"],
                                                                                                         losses["val"]/counts["val"],
                                                                                                         accuracies["val"]/counts["val"],
                                                                                                         losses["test"]/counts["test"],
                                                                                                         accuracies["test"]/counts["test"]))

