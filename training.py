# training.py
import numpy as np
import torch
import torch.nn as nn
from networks import device
from networks.noise import add_black_box_random, black_box_module
from torchvision.utils import save_image

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

            # if i % 5 == 0:
                # -----------------
                #  Train Generator
                # -----------------

            optimizers['g'].zero_grad()

            # Sample noise as generator input
            z = torch.FloatTensor(np.random.normal(0, 1, (batch_size, latent_dim))).to(device)
            gen_labels = torch.LongTensor(np.random.randint(0, model.generator.num_classes, batch_size)).to(device)

            # Generate a batch of images
            gen_imgs = model.generator(z, gen_labels)
            box_size = (10, 10)
            boxed_gen_imgs = add_black_box_random(gen_imgs, box_size)
            # gen_imgs = add_black_box_random(gen_imgs, box_size)

            # Measures generator's ability to fool the discriminator
            validity = model.discriminator(boxed_gen_imgs, gen_labels)
            g_loss = disc_loss(validity, valid)

            # Measure black-box sliding performance of discriminator
            # box_slid_imgs, ns = black_box_module(gen_imgs, box_size, stride=4)
            # valid_map = model.discriminator(box_slid_imgs)
            # # print(valid_map.size())
            # vmap_loss = disc_loss(valid_map, torch.ones((1,1)).expand_as(valid_map).type_as(valid_map))
            # vmap_loss /= torch.prod(ns).to(device)
            # vmap_loss = (valid_map**2).mean(sum()
            # g_loss += vmap_loss

            

            # Measures generator's ability to create classifiable images
            classibility = model.classifier(gen_imgs)
            gc_loss = class_loss(classibility, gen_labels)

            g_loss += gc_loss
            # g_loss = gc_loss

            g_loss.backward()
            optimizers['g'].step()

            # ---------------------
            #  Train Discriminator
            # ---------------------
            # if epoch < 2:

            optimizers['d'].zero_grad()
            # Measure discriminator's ability to classify real from generated samples
            # real_loss = disc_loss(model.discriminator(real_imgs, labels), valid)
            real_loss = disc_loss(model.discriminator(add_black_box_random(real_imgs, box_size), labels), valid)
            fake_loss = disc_loss(model.discriminator(boxed_gen_imgs.detach(), gen_labels), fake)
            d_loss = (real_loss + fake_loss) / 2



            d_loss.backward()
            optimizers['d'].step()

            # for p in model.discriminator.parameters():
            #     p.data.clamp_(-0.01, 0.01)  


            # ---------------------
            #  Train Classifier
            # ---------------------
            optimizers['c'].zero_grad()
            # Measure classifier's ability to classify both real and fake images
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

                sample_gan(model, latent_dim, batches_done)
                # save_image(gen_imgs.data[:25], 'images/%d.png' % batches_done, nrow=5, normalize=True)

def sample_gan(model, latent_dim, batches_done):
    z = torch.FloatTensor(np.random.normal(0, 1, (model.g.num_classes*4, latent_dim))).to(device)
    gen_labels = torch.LongTensor(np.tile(np.arange(model.g.num_classes),4)).to(device)
    gen_imgs = model.generator(z, gen_labels)

    save_image(gen_imgs.data[:40], 'images/u%d.png' % batches_done, nrow=10, normalize=True)


def visualize_gan(model, dataloader, visualize_fake=False):    
    sample = iter(dataloader)
    sample, label = next(sample)
    sample = sample.to(device)
    label = label.to(device)
    batch_size = sample.size(0)
    if visualize_fake:
        # z = torch.FloatTensor(np.random.normal(0, 1, (sample.shape[0], model.g.latent_dim))).to(device)
        z = torch.FloatTensor(np.random.normal(0, 1, (model.g.num_classes*2, model.g.latent_dim))).to(device)
        gen_labels = torch.LongTensor(np.tile(np.arange(model.g.num_classes),2)).to(device)
    
        # gen_labels = torch.LongTensor(np.random.randint(0, model.generator.num_classes, batch_size)).to(device)

        # Generate a batch of images
        gen_imgs = model.generator(z, gen_labels)
        model.c.visualize(gen_imgs, gen_labels)
        model.d.visualize(gen_imgs)
        model.vis_layer(gen_imgs, classifier=True)
        model.vis_layer(gen_imgs, classifier=False)
    else:
        model.c.visualize(sample, label)
        model.d.visualize(sample)
        model.vis_layer(sample, classifier=True)
        model.vis_layer(sample, classifier=False)

    # print(sample.size(), label.size())
    # print(next(sample))