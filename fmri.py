# fmri.py
# Extracts fMRI features

import torch
import torch.optim as optim
import torch.nn as nn

import copy

from data import get_VIM_dataloaders
from networks import device
from networks.extractor import FeatureExtractorFMRI
from torchvision.utils import save_image

# def save_model(model, epoch):
#         {
#             'epoch': epoch + 1,
#             'state_dict': model.state_dict(),
#             'optimizer' : optimizer.state_dict(),
#         }


dls = get_VIM_dataloaders()

net = FeatureExtractorFMRI().to(device)


optimizer = optim.Adam(net.parameters(), lr=0.0001)
criterion = nn.MSELoss()
class_loss = nn.CrossEntropyLoss()

running_loss = 0.0
for epoch in range(10):
    for i, (data, labels) in enumerate(dls['train']):
        data = data.unsqueeze(1).to(device)

        label = data#copy.deepcopy(data)

        # labels = labels.to(device)

        net.zero_grad()
        out, out_c = net(data)
        # for param in net.parameters():
        #     param.requires_grad = False
        # print('rq', out.requires_grad)
        # break

        loss = criterion(out, label)
        # loss = class_loss(out_c, labels)
        # print(loss)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()


        if i % 100 == 99:
            print(running_loss)
            running_loss = 0.0
            torch.save(net.state_dict(),'./models/fmri/vim_e%d_i%d.pt' % (epoch, i))
            # save_image(out.data, 'images/e%d_i%d.png' % (epoch, i), nrow=5, normalize=True)
