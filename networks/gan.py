# gan.py
# Lukas Edman

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import device
from torchvision.utils import save_image
import torchvision.transforms as transforms
import torch.optim as optim

class GAN(nn.Module):
    def __init__(self, generator, discriminator, classifier=None):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.classifier = classifier
        self.g = self.generator
        self.d = self.discriminator
        self.c = self.classifier

        self.norm = {'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5)} 


    def vis_layer(self, img, layer=0, classifier=True, filename_suffix=""):
        module = self.c if classifier else self.d

        conv_layer_nums = [3, 6, 10, 13]
        layer = conv_layer_nums[layer]

        for feature_num in range(module.conv[layer].weight.size(0)):
            x = module.conv[0:layer+1](img)

            x[:, :feature_num] = 0
            x[:, feature_num+1:] = 0

            irange = range(layer+1)[::-1]
            for i in irange:
                if i in [7, 15] :
                    continue 
                elif i in [1, 4, 8, 11, 14] :
                    x = F.relu(x, inplace=True)
                else:
                    stride = module.conv[i].stride 
                    x = F.conv_transpose2d(x, module.conv[i].weight, stride=stride, padding=1, output_padding=stride[0]-1)

            module_str = "classifier" if classifier else "discriminator"
            save_image(x, './images/vis/disc_class/%s_%s_sample_%04d.png' % (module_str, filename_suffix, feature_num), nrow=10)

    def vis_layer_gen(self, layer=0, filename_suffix=""):
        conv_layer_nums = [2, 6, 9, 10]
        layer = conv_layer_nums[layer]

        module = self.generator
        z = torch.FloatTensor(np.random.normal(0, 1, (module.num_classes*2, module.latent_dim))).to(device)
        gen_labels = torch.LongTensor(np.tile(np.arange(module.num_classes),2)).to(device)

        gen_input = torch.cat((module.label_emb(gen_labels), z), -1)
        gen_input = module.fc(gen_input)
        gen_input = gen_input.view(-1, module.init_channels, 8, 8)

        for feature_num in range(module.conv[layer].weight.size(1)):
            x = module.conv[0:layer+1](gen_input)

            x = x[:, feature_num].unsqueeze(1)
            module_str = "generator"
            save_image(x, './images/vis/g/%s_%s_sample_%04d.png' % (module_str, filename_suffix, feature_num), nrow=8)

    def vis_layer_2(self, img=None, layer=0, classifier=True):
        """ Visualize layer textures """ 
        module = self.c if classifier else self.d
        orig_layer = layer

        num_iters = 200
        conv_layer_nums = [2, 6, 9, 10]
        layer = conv_layer_nums[layer]
        for ft in range(module.conv[layer].weight.size(0)):
            if img is None:
                self.sample_img = (torch.rand((1, self.g.num_channels, 32, 32)).to(device)+0.6)/10
                for ch in range(self.g.num_channels):
                    self.sample_img[ch] -= self.norm['mean'][ch]
                    self.sample_img[ch] /= self.norm['std'][ch] 
            else: 
                self.sample_img = img[0].unsqueeze(0)

            self.sample_img.requires_grad = True
            optimizer = optim.Adam([self.sample_img], lr=0.1, weight_decay=1e-6)

            for i in range(num_iters):
                optimizer.zero_grad()
                x = self.sample_img
                x = module.conv[0:layer+1](x)
                output = x[0, ft]
                loss = -torch.mean(output)
                loss.backward()
                optimizer.step()

            img_to_save = self.reconstruct_img(self.sample_img)
            save_image(img_to_save, './images/testing/L%d_F%d.png' % (orig_layer, ft))

    def reconstruct_img(self, img):
        orig_img = copy.deepcopy(img)
        for ch in range(self.g.num_channels):
            orig_img[ch] *= self.norm['std'][ch]
            orig_img[ch] += self.norm['mean'][ch]
        
        orig_img = torch.clamp(orig_img, -1, 1)

        return orig_img
