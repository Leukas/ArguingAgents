import scipy.io
import numpy as np
from PIL import Image

data_path = './data/vim-1/EstimatedResponses2.mat'
labels_path = './data/vim-1/Stimuli2.mat'

data = scipy.io.loadmat(data_path)
labels = scipy.io.loadmat(labels_path)

data_keys = ['dataTrnS1', 'dataTrnS2', 'dataValS1', 'dataValS2', 'roiS1', 'roiS2', 'voxIdxS1', 'voxIdxS2']
labels_keys = ['seqTrn', 'seqVal', 'stimTrn', 'stimVal']

sam = labels[labels_keys[2]][0]
sam = 255*(sam-sam.min())/sam.max()

im = Image.fromarray(sam)
im.show()

# print(data[data_keys[0]].shape)
# print(labels.keys())
# f = h5py.File(labels_path,'r') 
# print(list(f.keys()))
# f.keys

# import tables
# file = tables.open_file(labels_path)
# print(a) 