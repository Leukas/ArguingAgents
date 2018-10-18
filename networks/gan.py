import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import device
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


    def vis_layer(self, img, layer=6, classifier=True, filename_suffix=""):
        if classifier:
            module = self.classifier
        else:
            module = self.discriminator

        conv_layer_nums = [2, 5, 9, 12]
        layer = conv_layer_nums[layer]

        for feature_num in range(module.conv[layer].weight.size(0)):
            x = module.conv[0:layer+1](img)
            # print(x.size())
            # print(x[:, feature_num].size())
            # x[:, feature_num] = x[:, feature_num]-x[:, feature_num].min() 
            # x[:, feature_num] = x[:, feature_num]/x[:, feature_num].max() 
            # x[:, feature_num] = 2*x[:, feature_num]-1
            # print(x[:, feature_num].min())
            # print(x[:, feature_num].max())
            # x[:, feature_num] = (x[:, feature_num])/(x[:feature_num].max() - x[:, feature_num].min())
            # x[:,feature_num] *= 10
            x[:, :feature_num] = 0
            x[:, feature_num+1:] = 0
            # stride = self.conv[layer].stride
            # weight = torch.zeros(self.conv[layer].weight.size()).to(device)
            # weight[feature_num] = self.conv[layer].weight[feature_num]
            # weight = self.conv[layer].weight
            # x = F.conv_transpose2d(x, weight, stride=stride, padding=1)

            # print(x.size())
            irange = range(layer+1)[::-1]
            for i in irange:
                # print(x.size())
                if i in [7, 15] :
                    continue 
                elif i in [1, 4, 8, 11, 14] :
                    # pass
                    x = F.relu(x, inplace=True)
                else:
                    stride = module.conv[i].stride 
                    x = F.conv_transpose2d(x, module.conv[i].weight, stride=stride, padding=1, output_padding=stride[0]-1)
            # print(weight.size())

            # print(x.size())

            module_str = "classifier" if classifier else "discriminator"
            save_image(x, './images/vis/%s_'+filename_suffix+'_sample_%04d.png' % (module_str, feature_num), nrow=10)

    def vis_layer_gen(self, layer=6, filename_suffix=""):
        conv_layer_nums = [2, 6, 9, 10]
        layer = conv_layer_nums[layer]

        module = self.generator
        z = torch.FloatTensor(np.random.normal(0, 1, (module.num_classes*2, module.latent_dim))).to(device)
        gen_labels = torch.LongTensor(np.tile(np.arange(module.num_classes),2)).to(device)

        gen_input = torch.cat((module.label_emb(gen_labels), z), -1)
        gen_input = module.fc(gen_input)
        gen_input = gen_input.view(-1, 128, 8, 8)

        for feature_num in range(module.conv[layer].weight.size(0)):
            # print(feature_num)
            x = module.conv[0:layer+1](gen_input)
            # x[:, :feature_num] = 0
            # x[:, feature_num] = 0
            # x[:, feature_num+1:] = 0
            # x = module.conv[layer+1:](x)

            # print(x.min(), x.max())
            module_str = "generator"
            # x = x.clamp(0,1)
            x = x[:, feature_num].unsqueeze(1)
            # print(type(x))
            # print(x.dtype, x.size())
            save_image(x, './images/vis/%s_'+filename_suffix+'_sample_%04d.png' % (module_str, feature_num), nrow=10)


