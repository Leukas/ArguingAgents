# training.py
import numpy as np 
import torch
import torch.nn
from networks import device
from torchvision.utils import save_image

def train_gan(model, dataloader, epochs=1, lr=0.0002, optimizers=None, criterion=None, 
    latent_dim=100, sample_interval=400, img_shape=None):
    
    if img_shape is None:
        img_shape = (1,28,28)

    if criterion is None:
        criterion = torch.nn.BCELoss().to(device)

    if optimizers is None:
        optimizers = {}
        optimizers['g'] = torch.optim.Adam(model['g'].parameters(), lr=lr, betas=(0.5, 0.999))
        optimizers['d'] = torch.optim.Adam(model['d'].parameters(), lr=lr, betas=(0.5, 0.999))


    for epoch in range(epochs):
        for i, (imgs, _) in enumerate(dataloader):

            # Adversarial ground truths
            valid = torch.FloatTensor(imgs.size(0), 1).fill_(1.0).to(device)
            fake = torch.FloatTensor(imgs.size(0), 1).fill_(0.0).to(device)

            # Configure input
            real_imgs = imgs.float().to(device)
            real_imgs.requires_grad = True

            # -----------------
            #  Train Generator
            # -----------------

            optimizers['g'].zero_grad()

            # Sample noise as generator input
            z = torch.FloatTensor(np.random.normal(0, 1, (imgs.shape[0], latent_dim))).to(device)
            z.requires_grad = True

            # Generate a batch of images
            gen_imgs = model['g'](z)

            # Loss measures generator's ability to fool the discriminator
            g_loss = criterion(model['d'](gen_imgs), valid)

            g_loss.backward()
            optimizers['g'].step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizers['d'].zero_grad()

            # Measure discriminator's ability to classify real from generated samples
            real_loss = criterion(model['d'](real_imgs), valid)
            fake_loss = criterion(model['d'](gen_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            optimizers['d'].step()

            if i % 100 == 0:
                print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" % (epoch, epochs, i, len(dataloader),
                                                                d_loss.item(), g_loss.item()))

            batches_done = epoch * len(dataloader) + i
            if batches_done % sample_interval == 0:
                save_image(gen_imgs.data[:25], 'images/%d.png' % batches_done, nrow=5, normalize=True)
                save_image(gen_imgs.data[:25], 'images/%d.png' % batches_done, nrow=5, normalize=True)