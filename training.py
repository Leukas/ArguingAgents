# training.py
import numpy as np 
import torch
import torch.nn as nn
from networks import device
from torchvision.utils import save_image

def train_gan(model, dataloader, epochs=1, lr=0.0002, optimizers=None, criterion=None, 
    latent_dim=100, sample_interval=400, img_shape=None):
    
    if img_shape is None:
        img_shape = (1,28,28)

    if criterion is None:
        disc_loss = nn.BCELoss().to(device)
        class_coss = nn.CrossEntropyLoss().to(device)

    if optimizers is None:
        optimizers = {}
        optimizers['g'] = torch.optim.Adam(model.generator.parameters(), lr=lr, betas=(0.5, 0.999))
        optimizers['d'] = torch.optim.Adam(model.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
        optimizers['c'] = torch.optim.Adam(model.classifier.parameters(), lr=lr, betas=(0.5, 0.999))


    for epoch in range(epochs):
        for i, (imgs, _) in enumerate(dataloader):

            # Adversarial ground truths
            valid = torch.FloatTensor(imgs.size(0), 1).fill_(1.0).to(device)
            fake = torch.FloatTensor(imgs.size(0), 1).fill_(0.0).to(device)

            # Configure input
            real_imgs = imgs.float().to(device)

            # -----------------
            #  Train Generator
            # -----------------

            optimizers['g'].zero_grad()

            # Sample noise as generator input
            z = torch.FloatTensor(np.random.normal(0, 1, (imgs.shape[0], latent_dim))).to(device)

            # Generate a batch of images
            gen_imgs = model.generator(z)

            # Loss measures generator's ability to fool the discriminator
            g_loss = disc_loss(model.discriminator(gen_imgs), valid)

            g_loss.backward()
            optimizers['g'].step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizers['d'].zero_grad()

            # Measure discriminator's ability to classify real from generated samples
            real_loss = disc_loss(model.discriminator(real_imgs), valid)
            fake_loss = disc_loss(model.discriminator(gen_imgs.detach()), fake)
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

def train_cgan(model, dataloader, epochs=1, lr=0.0002, optimizers=None, criterion=None, 
    latent_dim=100, sample_interval=400, img_shape=None, save_path=None):
    
    if img_shape is None:
        img_shape = (1,28,28)

    if criterion is None:
        disc_loss = nn.MSELoss().to(device)
        class_loss = nn.CrossEntropyLoss().to(device)

    if optimizers is None:
        optimizers = {}
        optimizers['g'] = torch.optim.Adam(model.generator.parameters(), lr=lr, betas=(0.5, 0.999))
        optimizers['d'] = torch.optim.Adam(model.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
        optimizers['c'] = torch.optim.Adam(model.classifier.parameters(), lr=lr, betas=(0.5, 0.999))


    for epoch in range(epochs):
        for i, (imgs, labels) in enumerate(dataloader):
            batch_size = imgs.size(0)
            # Adversarial ground truths
            valid = torch.FloatTensor(imgs.size(0), 1).fill_(1.0).to(device)
            fake = torch.FloatTensor(imgs.size(0), 1).fill_(0.0).to(device)

            # Configure input
            real_imgs = imgs.float().to(device)
            labels = labels.long().to(device)
            # -----------------
            #  Train Generator
            # -----------------

            optimizers['g'].zero_grad()

            # Sample noise as generator input
            z = torch.FloatTensor(np.random.normal(0, 1, (imgs.shape[0], latent_dim))).to(device)
            gen_labels = torch.LongTensor(np.random.randint(0, model.generator.num_classes, batch_size)).to(device)

            # Generate a batch of images
            gen_imgs = model.generator(z, gen_labels)

            # Loss measures generator's ability to fool the discriminator
            validity = model.discriminator(gen_imgs, gen_labels)
            g_loss = disc_loss(validity, valid)

            classibility = model.classifier(gen_imgs)
            gc_loss = class_loss(classibility, gen_labels)

            g_loss += gc_loss

            g_loss.backward()
            optimizers['g'].step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizers['d'].zero_grad()

            # Measure discriminator's ability to classify real from generated samples
            real_loss = disc_loss(model.discriminator(real_imgs, labels), valid)
            fake_loss = disc_loss(model.discriminator(gen_imgs.detach(), gen_labels), fake)
            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            optimizers['d'].step()


            # ---------------------
            #  Train Classifier
            # ---------------------
            optimizers['c'].zero_grad()

            # Measure discriminator's ability to classify real from generated samples
            real_loss = class_loss(model.classifier(real_imgs), labels)
            fake_loss = class_loss(model.classifier(gen_imgs.detach()), gen_labels)
            c_loss = (real_loss + fake_loss) / 2

            c_loss.backward()
            optimizers['c'].step()




            if i % 100 == 0:
                print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [C loss: %f]" % (epoch, epochs, i, len(dataloader),
                                                                d_loss.item(), g_loss.item(), c_loss.item()))

            batches_done = epoch * len(dataloader) + i
            if batches_done % sample_interval == 0:
                if save_path is not None:
                    torch.save(model.state_dict(), save_path)#+'%d.pt' % batches_done)

                save_image(gen_imgs.data[:25], 'images/%d.png' % batches_done, nrow=5, normalize=True)
                save_image(gen_imgs.data[:25], 'images/%d.png' % batches_done, nrow=5, normalize=True)


def visualize_gan(model, dataloader, visualize_fake=False):    
    sample = iter(dataloader)
    sample, label = next(sample)
    label = label.to(device)
    batch_size = sample.size(0)
    if visualize_fake:
        z = torch.FloatTensor(np.random.normal(0, 1, (sample.shape[0], model.g.latent_dim))).to(device)
        gen_labels = torch.LongTensor(np.random.randint(0, model.generator.num_classes, batch_size)).to(device)

        # Generate a batch of images
        gen_imgs = model.generator(z, gen_labels)
        model.c.visualize(gen_imgs, gen_labels)
    else:

        model.c.visualize(sample, label)

    # print(sample.size(), label.size())
    # print(next(sample))