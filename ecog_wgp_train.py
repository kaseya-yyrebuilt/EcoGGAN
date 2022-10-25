import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from models.ecog_wgp import Generator, Discriminator, weights_init
from get_config import get_config
from get_dataloader import get_dataloader
import torch.autograd as autograd


def train():
    # load training data
    config = get_config()
    trainloader = get_dataloader(config)
    device = torch.device("cuda:0" if config.use_cuda else "cpu")
    # init netD and netG
    netD = Discriminator(config).to(device)
    netD.apply(weights_init)

    netG = Generator(config).to(device)
    netG.apply(weights_init)

    # used for visualizing training process
    fixed_noise = torch.randn(config.ng, config.nz, 1, device=device)

    # optimizers
    optimizerD = optim.Adam(netD.parameters(), lr=config.lr, betas=(config.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=config.lr, betas=(config.beta1, 0.999))

    for epoch in range(config.epoch_num):
        for step, (data, _) in enumerate(trainloader):
            # training netD
            real_cpu = data.to(device)
            b_size = real_cpu.size(0)
            netD.zero_grad()

            noise = torch.randn(b_size, config.nz, 1, device=device)
            fake = netG(noise)

            # gradient penalty
            eps = torch.rand(b_size, 1, 1).to(device)
            data_penalty = eps * data + (1 - eps) * fake

            p_output = netD(data_penalty)

            label = torch.full((b_size,), 1.,
                                dtype=torch.float, device=device)
            
            data_grad = autograd.grad(outputs=p_output.view(-1), inputs=data_penalty, grad_outputs=label,
                                      create_graph=True, retain_graph=True, only_inputs=True)
            
            grad_penalty = config.p_coeff * \
                torch.mean(torch.pow(torch.norm(data_grad[0], 2, 1) - 1, 2))

            loss_D = -torch.mean(netD(real_cpu)) + \
                torch.mean(netD(fake)) + grad_penalty
                
            Dx = netD(real_cpu).view(-1).mean().item()
            
            loss_D.backward()
            optimizerD.step()

            if step % config.n_critic == 0:
                # training netG
                noise = torch.randn(b_size, config.nz, 1, device=device)

                netG.zero_grad()
                fake = netG(noise)
                loss_G = -torch.mean(netD(fake))

                netD.zero_grad()
                netG.zero_grad()
                loss_G.backward()
                optimizerG.step()

            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tDx: %.4f'
                  % (epoch, config.epoch_num, step, len(trainloader), loss_D.item(), loss_G.item(), Dx))
        
        fake = netG(fixed_noise).detach().cpu()
        np.save(f'./logs/epoch_{epoch}.npy', np.array(fake))
    # save model
    torch.save(netG, './nets/wgan_gp_netG.pkl')
    torch.save(netD, './nets/wgan_gp_netD.pkl')


if __name__ == '__main__':
    train()