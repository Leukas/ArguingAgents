# training.py
# Lukas Edman

import numpy as np
import torch
import torch.nn as nn
from networks import device
from networks.noise import add_black_box_random, black_box_module
from torchvision.utils import save_image

def train_cgan(model, dataloader, epochs=1, lr=0.0002, optimizers=None, criterion=None, 
    latent_dim=100, sample_interval=400, img_shape=None, save_path=None):
    """ Train entire model """

    if img_shape is None:
        img_shape = (1,28,28)

    if criterion is None:
        disc_loss = nn.BCELoss().to(device)
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

            optimizers['d'].zero_grad()
            # Measure discriminator's ability to classify real from generated samples
            # real_loss = disc_loss(model.discriminator(real_imgs, labels), valid)
            real_loss = disc_loss(model.discriminator(add_black_box_random(real_imgs, box_size), labels), valid)
            fake_loss = disc_loss(model.discriminator(boxed_gen_imgs.detach(), gen_labels), fake)
            d_loss = (real_loss + fake_loss) / 2



            d_loss.backward()
            optimizers['d'].step()

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

def train_cgan_shared_weights(model, dataloader, epochs=1, lr=0.0002, optimizers=None, criterion=None, 
    latent_dim=100, sample_interval=400, img_shape=None, save_path=None, vis=False):
    
    if img_shape is None:
        img_shape = (1,28,28)

    if criterion is None:
        disc_loss = nn.BCELoss().to(device)
        class_loss = nn.CrossEntropyLoss().to(device)

    if optimizers is None:
        optimizers = {}
        optimizers['g'] = torch.optim.Adam(model.generator.parameters(), lr=lr, betas=(0.5, 0.999))
        optimizers['d'] = torch.optim.Adam(model.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
        optimizers['c'] = torch.optim.Adam(model.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))


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

            # Measures generator's ability to create classifiable images
            classibility = model.discriminator.classify(gen_imgs)
            gc_loss = class_loss(classibility, gen_labels)

            g_loss += gc_loss
            # g_loss = gc_loss

            g_loss.backward()
            optimizers['g'].step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizers['d'].zero_grad()
            # Measure discriminator's ability to classify real from generated samples
            # real_loss = disc_loss(model.discriminator(real_imgs, labels), valid)
            real_loss = disc_loss(model.discriminator(add_black_box_random(real_imgs, box_size), labels), valid)
            fake_loss = disc_loss(model.discriminator(boxed_gen_imgs.detach(), gen_labels), fake)
            d_loss = (real_loss + fake_loss) / 2



            d_loss.backward()
            optimizers['d'].step()


            # ---------------------
            #  Train Classifier
            # ---------------------
            optimizers['c'].zero_grad()
            # Measure classifier's ability to classify both real and fake images
            real_loss = class_loss(model.discriminator.classify(real_imgs), labels)
            fake_loss = class_loss(model.discriminator.classify(gen_imgs.detach()), gen_labels)
            c_loss = (real_loss + fake_loss) / 2

            c_loss.backward()
            optimizers['c'].step()


            if i % 100 == 0:
                print(d_loss.item())
                print(c_loss.item())
                print(g_loss.item())
                print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [C loss: %f]" % (epoch, epochs, i, len(dataloader),
                                                                d_loss.item(), g_loss.item(), c_loss.item()))

            batches_done = epoch * len(dataloader) + i
            if batches_done % sample_interval == 0:
                if save_path is not None:
                    torch.save(model.state_dict(), save_path)#+'%d.pt' % batches_done)
                if vis:
                    sample_gan(model, latent_dim, batches_done)

                # save_image(gen_imgs.data[:25], 'im


def train_disc(model, dataloader, lr=0.0002, optimizers=None, criterion=None, 
    latent_dim=100, img_shape=None):
    """ Train only the discriminator, record outputs and layer visualization """
    sample = iter(dataloader)
    sample, label = next(sample)
    sample = sample.to(device)
    label = label.to(device)

    sample_z = torch.FloatTensor(np.random.normal(0, 1, (sample.shape[0], model.g.latent_dim))).to(device)


    
    if img_shape is None:
        img_shape = (1,28,28)

    if criterion is None:
        disc_loss = nn.BCELoss().to(device)
        class_loss = nn.CrossEntropyLoss().to(device)

    if optimizers is None:
        optimizers = {}
        optimizers['g'] = torch.optim.Adam(model.generator.parameters(), lr=lr, betas=(0.5, 0.999))
        optimizers['d'] = torch.optim.Adam(model.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
        optimizers['c'] = torch.optim.Adam(model.classifier.parameters(), lr=lr, betas=(0.5, 0.999))


    for i, (imgs, labels) in enumerate(dataloader):
        if i == 10000:
            break
        batch_size = imgs.size(0)
        # Adversarial ground truths
        valid = torch.FloatTensor(imgs.size(0), 1).fill_(1.0).to(device)
        fake = torch.FloatTensor(imgs.size(0), 1).fill_(0.0).to(device)

        # Configure input
        real_imgs = imgs.float().to(device)
        labels = labels.long().to(device)

        # Sample noise as generator input
        z = torch.FloatTensor(np.random.normal(0, 1, (batch_size, latent_dim))).to(device)
        gen_labels = torch.LongTensor(np.random.randint(0, model.generator.num_classes, batch_size)).to(device)

        # Generate a batch of images
        gen_imgs = model.generator(z, gen_labels)
        box_size = (10, 10)
        boxed_gen_imgs = add_black_box_random(gen_imgs, box_size)

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

        if i % 200 == 0:
            for layer in range(4):
                visualize_gan(model, dataloader, layer=layer, sample=sample, label=label, z=sample_z, visualize_fake=False, filename_suffix='_layer%d_iter%d' % (layer, i))
                visualize_gan(model, dataloader, layer=layer, sample=sample, label=label, z=sample_z, visualize_fake=True, filename_suffix='_layer%d_iter%d' % (layer, i))

    torch.save(model.state_dict(), './models/sample_disc_shared.pt')#+'%d.pt' % batches_done)
    return model
    # visualize_gan(model,)
                # sample_gan(model, latent_dim, batches_done)
                # save_image(gen_imgs.data[:25], 'images/%d.png' % batches_done, nrow=5, normalize=True)

def train_gen(model, dataloader, lr=0.0002, optimizers=None, criterion=None, 
    latent_dim=100, img_shape=None):
    """ Train only the generator, record outputs """


    sample = iter(dataloader)
    sample, label = next(sample)
    sample = sample.to(device)
    label = label.to(device)

    sample_z = torch.FloatTensor(np.random.normal(0, 1, (sample.shape[0], model.g.latent_dim))).to(device)


    
    if img_shape is None:
        img_shape = (1,28,28)

    if criterion is None:
        disc_loss = nn.BCELoss().to(device)
        class_loss = nn.CrossEntropyLoss().to(device)

    if optimizers is None:
        optimizers = {}
        optimizers['g'] = torch.optim.Adam(model.generator.parameters(), lr=lr, betas=(0.5, 0.999))
        optimizers['d'] = torch.optim.Adam(model.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
        optimizers['c'] = torch.optim.Adam(model.classifier.parameters(), lr=lr, betas=(0.5, 0.999))

    for epoch in range(5):
        for i, (imgs, labels) in enumerate(dataloader):
            if i == 10000:
                break
            batch_size = imgs.size(0)
            # Adversarial ground truths
            valid = torch.FloatTensor(imgs.size(0), 1).fill_(1.0).to(device)
            fake = torch.FloatTensor(imgs.size(0), 1).fill_(0.0).to(device)

            # Configure input
            real_imgs = imgs.float().to(device)
            labels = labels.long().to(device)

            # Sample noise as generator input
            z = torch.FloatTensor(np.random.normal(0, 1, (batch_size, latent_dim))).to(device)
            gen_labels = torch.LongTensor(np.random.randint(0, model.generator.num_classes, batch_size)).to(device)

            # Generate a batch of images
            gen_imgs = model.generator(z, gen_labels)
            box_size = (10, 10)
            boxed_gen_imgs = add_black_box_random(gen_imgs, box_size)

            # ---------------------
            #  Train Generator
            # ---------------------

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

            # Measures generator's ability to create classifiable images
            # classibility = model.discriminator.classify(gen_imgs)
            # gc_loss = class_loss(classibility, gen_labels)

            # g_loss += gc_loss
            # g_loss = gc_loss

            g_loss.backward()
            optimizers['g'].step()

            batches_done = epoch * len(dataloader) + i
            if batches_done % 1000 == 0:
                sample_gan(model, latent_dim, batches_done)

            for layer in range(4):
                visualize_gan_gen(model, dataloader, layer=layer, sample=sample, label=label, z=sample_z, visualize_fake=True, filename_suffix='_layer%d_iter%d' % (layer, i))

    torch.save(model.state_dict(), './models/sample_disc_gen.pt')#+'%d.pt' % batches_done)
    return model



def sample_gan(model, latent_dim, batches_done):
    z = torch.FloatTensor(np.random.normal(0, 1, (model.g.num_classes*2, latent_dim))).to(device)
    gen_labels = torch.LongTensor(np.tile(np.arange(model.g.num_classes),2)).to(device)
    gen_imgs = model.generator(z, gen_labels)

    save_image(gen_imgs.data[:2*model.g.num_classes], 'images/vis/g/%06d.png' % batches_done, nrow=model.g.num_classes, normalize=True)


def visualize_gan(model, dataloader, layer, sample=None, label=None, z=None, visualize_fake=False, filename_suffix=""):    
    if sample is None:
        sample = iter(dataloader)
        sample, label = next(sample)
        sample = sample.to(device)
        label = label.to(device)
    if visualize_fake:
        if z is None:
            z = torch.FloatTensor(np.random.normal(0, 1, (sample.shape[0], model.g.latent_dim))).to(device)
        gen_labels = label
        # z = torch.FloatTensor(np.random.normal(0, 1, (model.g.num_classes*2, model.g.latent_dim))).to(device)
        # gen_labels = torch.LongTensor(np.tile(np.arange(model.g.num_classes),2)).to(device)
        # gen_labels = torch.LongTensor(np.random.randint(0, model.generator.num_classes, batch_size)).to(device)

        # Generate a batch of images
        gen_imgs = model.generator(z, gen_labels)
        # model.c.visualize(gen_imgs, gen_labels, filename_suffix=filename_suffix+'_fake_')
        model.d.visualize(gen_imgs, filename_suffix=filename_suffix+'_fake_')
        # model.d.visualize_class(gen_imgs, gen_labels)
        # model.vis_layer(gen_imgs, layer=layer, classifier=True, filename_suffix=filename_suffix+'_fake_')
        model.vis_layer(gen_imgs, layer=layer, classifier=False, filename_suffix=filename_suffix+'_fake_')
        # model.vis_layer_gen(layer=layer, filename_suffix=filename_suffix+'_fake_')
    else:
        # model.c.visualize(sample, label, filename_suffix=filename_suffix+'_real_')
        model.d.visualize(sample, filename_suffix=filename_suffix+'_real_')
        # model.d.visualize_class(sample, label)
        # model.vis_layer(sample, layer=layer, classifier=True, filename_suffix=filename_suffix+'_real_')
        model.vis_layer(sample, layer=layer, classifier=False, filename_suffix=filename_suffix+'_real_')
        # model.vis_layer_gen(layer=layer, filename_suffix=filename_suffix+'_real_')


def visualize_gan_gen(model, dataloader, layer, sample=None, label=None, z=None, visualize_fake=False, filename_suffix=""):    
    if sample is None:
        sample = iter(dataloader)
        sample, label = next(sample)
        sample = sample.to(device)
        label = label.to(device)
    if visualize_fake:
        if z is None:
            z = torch.FloatTensor(np.random.normal(0, 1, (sample.shape[0], model.g.latent_dim))).to(device)
        gen_labels = label
        # z = torch.FloatTensor(np.random.normal(0, 1, (model.g.num_classes*2, model.g.latent_dim))).to(device)
        # gen_labels = torch.LongTensor(np.tile(np.arange(model.g.num_classes),2)).to(device)
        # gen_labels = torch.LongTensor(np.random.randint(0, model.generator.num_classes, batch_size)).to(device)

        # Generate a batch of images
        gen_imgs = model.generator(z, gen_labels)
        # model.c.visualize(gen_imgs, gen_labels, filename_suffix=filename_suffix+'_fake_')
        # model.d.visualize(gen_imgs, filename_suffix=filename_suffix+'_fake_')
        # model.d.visualize_class(gen_imgs, gen_labels)
        # model.vis_layer(gen_imgs, layer=layer, classifier=True, filename_suffix=filename_suffix+'_fake_')
        # model.vis_layer(gen_imgs, layer=layer, classifier=False, filename_suffix=filename_suffix+'_fake_')
        model.vis_layer_gen(layer=layer, filename_suffix=filename_suffix+'_fake_')
    else:
        # model.c.visualize(sample, label, filename_suffix=filename_suffix+'_real_')
        # model.d.visualize(sample, filename_suffix=filename_suffix+'_real_')
        # model.d.visualize_class(sample, label)
        # model.vis_layer(sample, layer=layer, classifier=True, filename_suffix=filename_suffix+'_real_')
        # model.vis_layer(sample, layer=layer, classifier=False, filename_suffix=filename_suffix+'_real_')
        model.vis_layer_gen(layer=layer, filename_suffix=filename_suffix+'_real_')

