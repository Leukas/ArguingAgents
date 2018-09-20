# import scipy.io
# import numpy as np
# from PIL import Image

# data_path = './data/vim-1/EstimatedResponses2.mat'
# labels_path = './data/vim-1/Stimuli2.mat'

# data = scipy.io.loadmat(data_path)
# labels = scipy.io.loadmat(labels_path)

# data_keys = ['dataTrnS1', 'dataTrnS2', 'dataValS1', 'dataValS2', 'roiS1', 'roiS2', 'voxIdxS1', 'voxIdxS2']
# labels_keys = ['seqTrn', 'seqVal', 'stimTrn', 'stimVal']

# sam = labels[labels_keys[2]][0]
# sam = 255*(sam-sam.min())/sam.max()

# im = Image.fromarray(sam)
# im.show()

# print(data[data_keys[0]].shape)
# print(labels.keys())
# f = h5py.File(labels_path,'r') 
# print(list(f.keys()))
# f.keys

# import tables
# file = tables.open_file(labels_path)
# print(a) 
# import scipy.io
# import numpy as np

# vim_data = scipy.io.loadmat('./data/vim-1/EstimatedResponses2.mat')
# vim_labels = scipy.io.loadmat('./data/vim-1/Stimuli2.mat')

# print(vim_labels.keys())
# data_keys = data.keys()
# 'dataTrnS1', 'dataTrnS2', 'dataValS1', 'dataValS2', 'roiS1', 'roiS2', 'voxIdxS1', 'voxIdxS2'
# labels_keys = labels.keys()
# 'seqTrn', 'seqVal', 'stimTrn', 'stimVal'

# dats = vim_data['dataTrnS1']
# idxs = vim_data['voxIdxS1']
# rois = vim_data['roiS1']
# print(vim_labels['stimTrn'].shape)
# print(np.stack([idxs, vim_data['voxIdxS2']], axis=0).shape)
# print(np.bincount(rois.squeeze()))
# print(idxs.min(), idxs.max())
# a = np.reshape(dats[idxs],(64,64,18))

# print(data_keys)
# print(labels_keys)
import torch
import torch.optim as optim
import torch.nn as nn

import copy

from data import get_VIM_dataloaders
from networks import device
from networks.extractor import FeatureExtractorFMRI
from torchvision.utils import save_image

dls = get_VIM_dataloaders()

net = FeatureExtractorFMRI().to(device)


optimizer = optim.Adam(net.parameters(), lr=0.0001)
criterion = nn.MSELoss()
class_loss = nn.CrossEntropyLoss()

running_loss = 0.0
for epoch in range(10):
    for i, (data, labels) in enumerate(dls['train']):
        data = data.unsqueeze(1).to(device)
        labels = labels.to(device)
        # print(torch.isnan(data).any())
        label = copy.deepcopy(data)

        data.requires_grad = True
        net.zero_grad()
        out, out_c = net(data)
        # print(out)
        # break

        loss = criterion(out, label)
        # loss = class_loss(out_c, labels)
        # print(loss)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()

        if i % 10 == 9:
            print(running_loss)
            running_loss = 0.0

            # save_image(out.data, 'images/e%d_i%d.png' % (epoch, i), nrow=5, normalize=True)
