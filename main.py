import os
import argparse
import yaml
import collections

import numpy as np
import torch
import torch.nn as nn
from torch.nn import MSELoss, L1Loss
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms
from skimage.io import imsave

from src import models, utils
from src.utils import write_flush


def main(args, config):

    writer = SummaryWriter(os.path.join(config.tensorboard_dir, './pathgan/%s' % args.job_number))

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    write_flush(str(device))

    if torch.cuda.is_available():
        write_flush('Available devices: %d' % torch.cuda.device_count())

    if 'CAMELYON' in config.data_dir:

        x_train, y_train, x_valid, y_valid, x_test, y_test = utils.load_data(config.data_dir)
        write_flush(str(x_train.shape))

        train_loader = utils.data_generator(x_train, y_train, config.nb_batch)
        ''' 
        validation set comprises separate patient batch with differing tissue types,
        using train data for monitoring outputs give slightly nicer visualisations
        '''
        #val_loader = utils.data_generator(x_valid, y_valid, 32)
        val_loader = utils.data_generator(x_train, y_train, 32)

    else:  # CRC

        T = transforms.Compose([transforms.Resize(224),
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomVerticalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5),
                                                     (0.5, 0.5, 0.5))])

        train_ds = utils.CRCDataset(os.path.join(config.data_dir, 'NCT-CRC-HE-100K'), T)
        val_ds = utils.CRCDataset(os.path.join(config.data_dir, 'CRC-VAL-HE-7K'), T)
        
        write_flush(len(train_ds))

        train_loader = DataLoader(train_ds, batch_size=16, drop_last=True,
                                  shuffle=True, num_workers=16)
        
        val_loader = DataLoader(val_ds, batch_size=32, drop_last=True,
                                shuffle=True, num_workers=16)

        train_loader = utils.infinite_data_loader(train_loader)
        val_loader = utils.infinite_data_loader(val_loader)

    # -------------------
    #  Initialise models
    # -------------------
    backbone = True
    generator = models.Generator(64, config.x_dim, config.z_dim, backbone=backbone).to(device)

    discriminator = models.Discriminator(64).to(device)
    discriminator_gauss = models.Discriminator_Gauss(config.z_dim).to(device)

    write_flush('#params gen. %d' % utils.count_params(generator))
    write_flush('#params disc. %d' %  utils.count_params(discriminator))
    write_flush('#params disc gauss. %d' % utils.count_params(discriminator_gauss))

    optimiser_G = Adam(generator.parameters(), lr=1e-3, betas=(0, 0.999))
    optimiser_D = Adam(discriminator.parameters(), lr=1e-3, betas=(0, 0.999))
    optimiser_D_gauss = Adam(discriminator_gauss.parameters(), lr=1e-3, betas=(0, 0.999))

    init_epoch = 1
    total_iterations = config.total_iterations  # iteration over which to progress
    iteration = 0
    step = 1  # lowest resolution
    max_steps = 5
    iterations_per_step = total_iterations // max_steps
    alpha = 0
    init_size = 14

    nb_epochs = 1250
    steps_per_epoch = total_iterations // nb_epochs

    z_sample = torch.randn((32, config.z_dim)).to(device)

    if config.ckpt:
        checkpoint = torch.load(config.ckpt)
        generator = checkpoint['generator']
        discriminator = checkpoint['discriminator']
        discriminator_gauss = checkpoint['discriminator_gauss']
        optimiser_G = checkpoint['optimiser_G']
        optimiser_D = checkpoint['optimiser_D']
        optimiser_D_gauss = checkpoint['optimiser_D_gauss']
        init_epoch = checkpoint['epoch'] + 1  # start of next epoch
        iteration = checkpoint['iteration']
        step = checkpoint['step']
        z_sample = checkpoint['z_sample']

    if backbone and init_epoch == 1: 
        generator.encoder.eval()
        write_flush('Encoder frozen for initial epoch')

    for epoch in range(init_epoch, nb_epochs + 1):

        for batch_i in range(steps_per_epoch):

            if iteration == iterations_per_step and not step == max_steps:
                iteration = 0  # reset counter
                alpha = 0
                step += 1
            else:
                alpha = min(1, 2 / iterations_per_step * iteration)

            iteration += 1

            data = utils.scale_generator(*next(train_loader), init_size * 2 ** (step - 1), alpha, config.x_dim)
            fake_inputs, fake_targets = data[0].to(device), data[1].to(device)

            data = utils.scale_generator(*next(train_loader), init_size * 2 ** (step - 1), alpha, config.x_dim)
            _, real_targets = data[0].to(device), data[1].to(device)

            disc_patch = (1, 4, 4)
            fake = torch.zeros((config.nb_batch,) + disc_patch).to(device)
            valid = torch.ones((config.nb_batch,) + disc_patch).to(device)

            real_noise = torch.randn(config.nb_batch, config.z_dim).to(device)
            fake_gauss = torch.zeros((config.nb_batch)).to(device)
            real_gauss = torch.ones((config.nb_batch)).to(device)

            # -----------------
            #  Train Generator
            # -----------------

            optimiser_G.zero_grad()

            _, gen_imgs = generator(fake_inputs, step, alpha)

            if config.masked:
                top, left = 2 * [gen_imgs.shape[2] // 4]
                bottom, right = 2 * [gen_imgs.shape[2] - (gen_imgs.shape[2] // 2 - gen_imgs.shape[2] // 4)]

                reg_loss = L1Loss()(gen_imgs[:, :, top:bottom, left:right], fake_targets[:, :, top:bottom, left:right])

            else:
                reg_loss = L1Loss()(gen_imgs, fake_targets)

            d_loss = nn.MSELoss()(discriminator(gen_imgs, step, alpha), valid)

            t = (epoch - 1) * steps_per_epoch + batch_i
            reg_weight = config.lambda_l1 * max(0.5, (1 - (0.5 * t / total_iterations)))

            g_loss = d_loss + reg_weight * reg_loss

            g_loss.backward()
            optimiser_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimiser_D.zero_grad()

            real_loss = nn.MSELoss()(discriminator(real_targets, step, alpha), valid)
            fake_loss = nn.MSELoss()(discriminator(gen_imgs.detach(), step, alpha), fake)

            d_loss = 0.5 * (real_loss + fake_loss)

            d_loss.backward()
            optimiser_D.step()

            # ----------------------------------
            #  Train Generator - Regularisation
            # ----------------------------------

            optimiser_G.zero_grad()

            gen_noise = generator.encode(fake_inputs)

            d_gauss_loss = nn.MSELoss()(discriminator_gauss(gen_noise), real_gauss)

            d_gauss_loss.backward()
            optimiser_G.step()

            # --------------------------------------
            #  Train Discriminator - Regularisation
            # --------------------------------------

            optimiser_D_gauss.zero_grad()

            real_loss = nn.MSELoss()(discriminator_gauss(real_noise), real_gauss)
            fake_loss = nn.MSELoss()(discriminator_gauss(gen_noise.detach()), fake_gauss)
            d_gauss_loss = 0.5 * (real_loss + fake_loss)

            d_gauss_loss.backward()
            optimiser_D_gauss.step()

        if epoch % 5 == 1:

            generator.eval()
            x_batch, y_batch = next(val_loader)

            data = utils.scale_generator(x_batch, y_batch, init_size * 2 ** (step - 1), alpha, config.x_dim)
            _, gen_imgs = generator(data[0].to(device), step, alpha)

            mosaique = utils.create_mosaique(gen_imgs.detach().cpu(), nrows=4, ncols=8)
            imsave('%s/generated_%04d.png' % (config.outputs_dir, epoch), mosaique)
            writer.add_image('generated', np.moveaxis(mosaique, 2, 0), epoch)

            mosaique = utils.create_mosaique(data[0], nrows=4, ncols=8)
            writer.add_image('inputs', np.moveaxis(mosaique, 2, 0), epoch)

            mosaique = utils.create_mosaique(data[1], nrows=4, ncols=8)
            imsave('%s/original_%04d.png' % (config.outputs_dir, epoch), mosaique)
            writer.add_image('targets', np.moveaxis(mosaique, 2, 0), epoch)

            samples = generator.decode(z_sample, step, alpha)
            mosaique = utils.create_mosaique(samples.detach().cpu(), nrows=4, ncols=8)
            imsave('%s/samples_%04d.png' % (config.outputs_dir, epoch), mosaique)
            writer.add_image('samples', np.moveaxis(mosaique, 2, 0), epoch)

            generator.train()

        if epoch % 100 == 0:

            checkpoint = {'generator' : generator,
                          'discriminator' : discriminator,
                          'discriminator_gauss' : discriminator_gauss,
                          'optimiser_G' : optimiser_G,
                          'optimiser_D' : optimiser_D,
                          'optimiser_D_gauss' : optimiser_D_gauss,
                          'epoch' : epoch,
                          'iteration' : iteration,
                          'step' : step,
                          'z_sample' : z_sample}

            torch.save(checkpoint, '%s/%04d.ckpt' % (config.outputs_dir, epoch))

        write_flush('[Epoch %d/%d] [Batch %d/%d] [DP loss: %.02f] [DG loss %.02f] [G loss: %.02f] [Res: %d] [A: %.02f] [Reg: %.02f]'
            % (epoch, nb_epochs, batch_i, steps_per_epoch, d_loss.item(), d_gauss_loss.item(), g_loss.item(), init_size * 2 ** (step - 1), alpha, reg_weight))

        writer.add_scalar('G loss', g_loss.item(), epoch)
        writer.add_scalar('DP loss', d_loss.item(), epoch)
        writer.add_scalar('DG loss', d_gauss_loss.item(), epoch)
        writer.add_scalar('Alpha', alpha, epoch)
        writer.add_scalar('Reg weight', reg_weight, epoch)

    writer.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Progressively-grown adversarial autoencoder for visual field expansion')
    parser.add_argument('job_number', type=int)
    parser.add_argument('config', type=str)
    args = parser.parse_args()

    write_flush(str(args))
    
    with open(args.config, 'r') as fp:
        cfg = yaml.safe_load(fp)

    config = collections.namedtuple('Config', cfg.keys())(*cfg.values())

    main(args, config)
