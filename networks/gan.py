import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image

class GAN(nn.Module):
    def __init__(self, generator, discriminator, classifier=None):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.classifier = classifier
        self.g = self.generator
        self.d = self.discriminator
        self.c = self.classifier


    def vis_layer(self, img, layer=3, feature_num=0, classifier=True):
        if classifier:
            module = self.classifier
        else:
            module = self.discriminator


        for feature_num in range(module.conv[layer].weight.size(0)):
            x = module.conv[0:layer+1](img)
            # print(x.size())
            # print(x[:, feature_num].size())
            x[:, feature_num] = x[:, feature_num]-x[:, feature_num].min() 
            x[:, feature_num] = x[:, feature_num]/x[:, feature_num].max() 
            x[:, feature_num] = 2*x[:, feature_num]-1
            print(x[:, feature_num].min())
            print(x[:, feature_num].max())
            # x[:, feature_num] = (x[:, feature_num])/(x[:feature_num].max() - x[:, feature_num].min())
            # x[:,feature_num] *= 10
            x[:,:feature_num] = 0
            x[:,feature_num+1:] = 0
            # stride = self.conv[layer].stride
            # weight = torch.zeros(self.conv[layer].weight.size()).to(device)
            # weight[feature_num] = self.conv[layer].weight[feature_num]
            # weight = self.conv[layer].weight
            # x = F.conv_transpose2d(x, weight, stride=stride, padding=1)

            # print(x.size())
            irange = range(layer+1)[::-1]
            for i in irange:
                # print(x.size())
                if i in [7, 14] :
                    continue 
                elif i in [1, 4, 8, 11, 13] :
                    x = F.leaky_relu(x, 0.1, inplace=True)
                else:
                    stride = module.conv[i].stride 
                    x = F.conv_transpose2d(x, module.conv[i].weight, stride=stride, padding=1)
            # print(weight.size())

            # print(x.size())

            module_str = "classifier" if classifier else "discriminator"
            save_image(x, './images/vis/%s_sample_%04d.png' % (module_str, feature_num), nrow=10)