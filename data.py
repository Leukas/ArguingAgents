# data.py
import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from scipy.misc import imread
import numpy as np
import scipy.io
from PIL import Image

# Configure data loader
def get_MNIST_dataloaders(batch_size):
    os.makedirs('../../data/mnist', exist_ok=True)
    dataloaders = {}
    dataloaders['train'] = torch.utils.data.DataLoader(
        datasets.MNIST('../../data/mnist', 
            train=True, 
            download=True,
            transform=transforms.Compose([
                transforms.Resize(32),
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

def get_CIFAR10_dataloaders(batch_size):
    os.makedirs('../../data/cifar10', exist_ok=True)
    dataloaders = {}
    dataloaders['train'] = torch.utils.data.DataLoader(
        datasets.CIFAR10('../../data/cifar10', 
            train=True, 
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])),
        batch_size=batch_size, shuffle=True)

    dataloaders['test'] = torch.utils.data.DataLoader(
        datasets.CIFAR10('../../data/cifar10', 
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
    def __init__(self, download_imagenet=True, eeg_signals_path="./data/eeg_signals_128_sequential_band_all_with_mean_std.pth"):
        # Load EEG signals
        loaded = torch.load(eeg_signals_path)
        self.data = loaded["dataset"]
        self.labels = loaded["labels"]
        # print(self.labels)
        self.images = loaded["images"]
        # print(self.images)
        # return
        self.means = loaded["means"]
        self.stddevs = loaded["stddevs"]
        # Compute size
        self.size = len(self.data)
        if download_imagenet:
            self.download_imagenet()

    # Get size
    def __len__(self):
        return self.size

    # Get item
    def __getitem__(self, i):
        # Process EEG
        eeg = ((self.data[i]["eeg"].float() - self.means)/self.stddevs).t()
        eeg = eeg[20:450,:].unsqueeze(0)
        # print(self.data[i])
        # print(self.images[self.data[i]["image"]])
        # print(self.labels[self.data[i]["label"]])
        # print(self.images[i])
        # Get label
        label = self.data[i]["label"]
        img_id = self.images[self.data[i]["image"]]
        # Return
        return eeg, label, img_id

    def download_imagenet(self):
        import urllib
        url_dict = get_imagenet_url_dict()
        if not os.path.isdir('./data/imagenet/'):
            os.mkdir('./data/imagenet/')

        for key in self.images:
            url = url_dict[key]
            url_ext = url.split('.')[-1]
            x = urllib.urlretrieve(url, key + '.' + url_ext)
            print(x)
            break
        # pass

def get_imagenet_url_dict(path='./data/fall11_urls.txt'):
    url_dict = {}
    with open(path, 'rb') as f:
        for line in f:
            split = line.split()
            key, val = split[0], split[1]
            url_dict[key] = val

    return url_dict

# if __name__ == '__main__':
#   get_imagenet_url_dict()


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


def min_max_normalize(tensor):
    # print(tensor.size())
    # print(tensor.min(1)[0].size())
    # return 2*(tensor - tensor.min(1)[0].unsqueeze(1))/tensor.max(1)[0].unsqueeze(1) - 1
    # print(tensor-tensor.min())
    return 2*(tensor - tensor.min())/tensor.max() - 1

def get_VIM_dataloaders(batch_size=32):
    class VIM_Dataset(Dataset):
        def __init__(self, train=True, subjects=[0]):
            """
            
            """
            import scipy.io
            import numpy as np 
            self.train = train
            self.subjects = subjects
            vim_data = scipy.io.loadmat('./data/vim-1/EstimatedResponses2.mat')
            vim_labels = scipy.io.loadmat('./data/vim-1/Stimuli2.mat')
            if self.train:
                s1data = torch.Tensor(vim_data['dataTrnS1'].T)
                s1data[torch.isnan(s1data)] = 0
                s1data = min_max_normalize(s1data)
                s2data = torch.Tensor(vim_data['dataTrnS2'].T)
                s2data[torch.isnan(s2data)] = 0
                s2data = min_max_normalize(s2data)
                self.labels = torch.Tensor(vim_labels['stimTrn'])
            else:
                s1data = torch.Tensor(vim_data['dataValS1'].T)
                s2data = torch.Tensor(vim_data['dataValS2'].T)
                self.labels = torch.Tensor(vim_labels['stimVal'])
            self.data = [s1data, s2data]

            s1idx = torch.Tensor(vim_data['voxIdxS1']).squeeze().long()
            s2idx = torch.Tensor(vim_data['voxIdxS2']).squeeze().long()
            self.idxs = [s1idx, s2idx]
            # self.idxs = torch.Tensor(np.vstack([vim_data['voxIdxS1'],vim_data['voxIdxS2']]))

        def __len__(self):
            return self.data[0].size(0) + self.data[1].size(0)

        def __getitem__(self, i):
            idx = i % self.data[0].size(0)
            subj = i >= self.data[0].size(0)
            if len(self.subjects) == 1:
                subj = self.subjects[0]
            # subj = 0
            out = torch.zeros(64*64*18)
            out.index_copy_(0, self.idxs[subj], self.data[subj][idx])
            out = out.view(64,64,18)
            return out, self.labels[idx]

    dataloaders = {
        "train": DataLoader(VIM_Dataset(), batch_size, shuffle=True),
        "test": DataLoader(VIM_Dataset(train=False), batch_size)
    }
    return dataloaders


def get_Feret_dataloaders(batch_size=32, img_size=(64,64)):

    class Feret_Dataset(Dataset):
        def __init__(self, path_base="./data/colorferet/", img_size=(64,64)):

            subjs=[]
            # with open(path_base+"dvd2/doc/subject_counts.out", 'r') as f:
            #     for line in f:
            #         subjs.append(line.split()[0])

            self.subjects = subjs

            feret_data = scipy.io.loadmat(path_base+'imnum.mat')
            feret_data = feret_data['imnum'].squeeze()
            print(feret_data[0].shape)
            feret_label_date = scipy.io.loadmat(path_base+'label_date.mat')
            feret_label_gender = scipy.io.loadmat(path_base+'label_gender.mat')
            feret_label_pose = scipy.io.loadmat(path_base+'label_pose.mat')
            feret_label_pose = feret_label_pose['label_pose']
            feret_label_race = scipy.io.loadmat(path_base+'label_race.mat')
            feret_label_subj = scipy.io.loadmat(path_base+'label_subj.mat')
            feret_label_year = scipy.io.loadmat(path_base+'label_year.mat')
            feret_label_year = feret_label_year['label_year'].astype(np.int32)
            # print(feret_label_year['label_year'].astype(np.int32))
            # print(len(feret_label_year))
            # face_mask = ((feret_label_pose == 'fa') and (feret_label_pose == 'fb'))
            # feret_data_fa = feret_data[]
            feret_data_fa = feret_data[feret_label_pose == 'fa']
            feret_data_fb = feret_data[feret_label_pose == 'fb']
            feret_label_year_fa = feret_label_year[feret_label_pose == 'fa']
            feret_label_year_fb = feret_label_year[feret_label_pose == 'fb']
            # print(feret_data_fa.shape)
            # print(feret_data_fb.shape)
            feret_data_face = np.hstack((feret_data_fa, feret_data_fb))
            feret_label_year_face = np.hstack((feret_label_year_fa, feret_label_year_fb))
            # print(feret_data_face.shape)
            # print(feret_label_year_face.shape)
            # feret_data_face = n
            # print(feret_label_pose[feret_label_pose == 'fa'])
            # print(feret_data.shape)
            # print(feret_label_year.shape)
            self.data = torch.zeros((len(feret_data_face), img_size[0], img_size[1])).float()
            self.labels = torch.zeros(len(feret_label_year_face)).long()
            years = np.unique(feret_label_year_face)

            for i, datum in enumerate(feret_data_face):
                img = Image.fromarray(datum)
                img = img.resize((img_size[0],img_size[1]), resample=Image.LANCZOS)
                img = torch.Tensor(np.array(img))
                img = 2*img/255-1
                self.data[i] = img
                self.labels[i] = torch.LongTensor(np.where(years==feret_label_year_face[i])[0])



            self.data = self.data.unsqueeze(1)

            # print(self.data.size())
            # print(self.labels.size())
            # self.data = feret_data_face
            # bc
            # bc = np.bincount(feret_label_year_face)
            # bc = feret_label_year_face[bc!=0]
            # print(bc)
            # print(len(bc[bc!=0]))
            # self.labels = torch.LongTensor(feret_label_year_face
            # data = torch.Tensor(feret_data_face)
            # data = torch.Tensor(feret_data_face)

        def __len__(self):
            return len(self.data)

        def __getitem__(self, i):  
            return self.data[i], self.labels[i]
    
    ds = DataLoader(Feret_Dataset(img_size=img_size), batch_size,  shuffle=True)
    dataloaders = {
        "train": ds,
        "test": ds
    }

    return dataloaders
                    
