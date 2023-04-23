import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import Subset
import matplotlib.pyplot as plt
import numpy as np
import torchvision.utils as utils
import time
from torchvision import models
import torch.nn.functional as F
from scipy import linalg
from torchvision.utils import make_grid
from frechet_distance_cal import calculate_fretchet
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Trainer():
    def __init__(self,
                 dataloader,
                 CRITIC_ITERATIONS,
                 LAMBDA_GP,
                 outpath,
                 gen,
                 critic,
                 n_epochs
                 ):
        
        self.dataloader = dataloader
        self.CRITIC_ITERATIONS = CRITIC_ITERATIONS
        self.LAMBDA_GP = LAMBDA_GP
        self.out_path = outpath
        self.gen = gen
        self.critic = critic
        self.n_epochs = n_epochs


        self.gen_error = []
        self.dis_error = []

        self.fretchet_distances = []
    
    def run_trainer(self):
        for epoch in range(self.n_epochs):
            print('Epoch - {}'.format(epoch))

            critic_loss, gen_loss = self._run()
            self.gen_error.append(gen_loss)
            self.dis_error.append(critic_loss)

            #save images
            noise = torch.randn(10, Z_DIM, 1, 1,device = device) 
            gen_images = self.gen(noise).detach()

            images = make_grid(gen_images)

            images = images.cpu().numpy()
            images = images/2 + 0.5
            plt.plot([1,2,3])
            plt.imshow(np.transpose(images,axes = (1,2,0)))
            plt.axis('off')
            plt.savefig(os.path.join(self.out_path,'wgan_fake.png'))

        return self.gen_error, self.dis_error, self.fretchet_distances

    
    def _run(self):
        gen_loss = []
        critic_loss = []
        for batch_idx, (real, _) in enumerate(self.dataloader):
            real = real.to(device)
            cur_batch_size = real.shape[0]

            # Train Critic: max E[critic(real)] - E[critic(fake)]
            # equivalent to minimizing the negative of that
            for _ in range(self.CRITIC_ITERATIONS):
                noise = torch.randn(cur_batch_size, Z_DIM, 1, 1).to(device)
                fake = self.gen(noise)
                critic_real = self.critic(real).reshape(-1)
                critic_fake = self.critic(fake).reshape(-1)
                gp = gradient_penalty(critic, real, fake, device=device)
                loss_critic = (
                    -(torch.mean(critic_real) - torch.mean(critic_fake)) + self.LAMBDA_GP * gp
                )
                self.critic.zero_grad()
                loss_critic.backward(retain_graph=True)
                opt_critic.step()

            # Train Generator: max E[critic(gen_fake)] <-> min -E[critic(gen_fake)]
            gen_fake = self.critic(fake).reshape(-1)
            loss_gen = -torch.mean(gen_fake)
            gen.zero_grad()
            loss_gen.backward()
            opt_gen.step()

            critic_loss.append(loss_critic.detach().cpu())
            gen_loss.append(loss_gen.detach().cpu())

            fretchet_dist = calculate_fretchet(real, fake)
            self.fretchet_distances.append(fretchet_dist)

        return np.mean(critic_loss), np.mean(gen_loss)

class Discriminator(nn.Module):
    def __init__(self, channels_img, features_d):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            # input: N x channels_img x 64 x 64
            nn.Conv2d(channels_img, features_d, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            # _block(in_channels, out_channels, kernel_size, stride, padding)
            self._block(features_d, features_d * 2, 4, 2, 1),
            self._block(features_d * 2, features_d * 4, 4, 2, 1),
            self._block(features_d * 4, features_d * 8, 4, 2, 1),
            # After all _block img output is 4x4 (Conv2d below makes into 1x1)
            nn.Conv2d(features_d * 8, 1, kernel_size=4, stride=2, padding=0),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False,
            ),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.disc(x)
class Generator(nn.Module):
    def __init__(self, channels_noise, channels_img, features_g):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            # Input: N x channels_noise x 1 x 1
            self._block(channels_noise, features_g * 16, 4, 1, 0),  # img: 4x4
            self._block(features_g * 16, features_g * 8, 4, 2, 1),  # img: 8x8
            self._block(features_g * 8, features_g * 4, 4, 2, 1),  # img: 16x16
            self._block(features_g * 4, features_g * 2, 4, 2, 1),  # img: 32x32
            nn.ConvTranspose2d(
                features_g * 2, channels_img, kernel_size=4, stride=2, padding=1
            ),
            # Output: N x channels_img x 64 x 64
            nn.Tanh(),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)
    
def initialize_weights(model):
    # Initializes weights according to the DCGAN paper
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

def gradient_penalty(critic, real, fake, device="cpu"):
    BATCH_SIZE, C, H, W = real.shape
    alpha = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * alpha + fake * (1 - alpha)

    # Calculate critic scores
    mixed_scores = critic(interpolated_images)

    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty

def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

outpath = 'wgan_out'
make_dir(outpath)
Z_DIM = 100
CHANNELS_IMG = 3
FEATURES_GEN = 16
FEATURES_CRITIC = 16
LEARNING_RATE = 1e-5
CRITIC_ITERATIONS = 5
LAMBDA_GP = 10
n_epochs = 10

gen = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN).to(device)
critic = Discriminator(CHANNELS_IMG, FEATURES_CRITIC).to(device)
initialize_weights(gen)
initialize_weights(critic)

opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))
opt_critic = optim.Adam(critic.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))

tf = transforms.Compose([transforms.Resize(64),
                         transforms.ToTensor(),
                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                        ])

trainset = torchvision.datasets.CIFAR10(root = './data', train = True, download = True,
                                     transform = tf)


testset = torchvision.datasets.CIFAR10(root = './data', train = False, download = True,
                                    transform = tf)

dataset = torch.utils.data.ConcatDataset([trainset, testset])


dataloader = torch.utils.data.DataLoader(dataset, batch_size = 256, 
                                         num_workers = 2, shuffle = True)

trainer = Trainer(dataloader, CRITIC_ITERATIONS, LAMBDA_GP, outpath, gen, critic, n_epochs)

trainer.run_trainer()