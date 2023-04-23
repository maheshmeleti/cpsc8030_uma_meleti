import os
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torchvision.utils import make_grid
import torch.optim as optim
import numpy as np
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
from scipy import linalg
from torchvision import models
from frechet_distance_cal import calculate_fretchet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Generator(nn.Module):
    
    def __init__(self):
        super(Generator,self).__init__()
        
        self.conv_trans1 = nn.Sequential(nn.ConvTranspose2d(100,512,4,1,0,bias = False),
                                   nn.ReLU(True))
        self.conv_trans2 = nn.Sequential(nn.ConvTranspose2d(512,256,4,2,1,bias = False),
                                   nn.BatchNorm2d(256),
                                   nn.ReLU(True))
        self.conv_trans3 = nn.Sequential(nn.ConvTranspose2d(256,128,4,2,1,bias = False),
                                   nn.BatchNorm2d(128),
                                   nn.ReLU(True))
        self.conv_trans4 = nn.Sequential(nn.ConvTranspose2d(128,64,4,2,1,bias = False),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU(True))
        self.conv_trans5 = nn.Sequential(nn.ConvTranspose2d(64,3,4,2,1,bias = False),
                                   nn.Tanh())
        
        
    def forward(self,x):
        
        x = x.view(-1,100,1,1)
        x = self.conv_trans1(x)
        x = self.conv_trans2(x)
        x = self.conv_trans3(x)
        x = self.conv_trans4(x)
        x = self.conv_trans5(x)
        return x
    
    
class Discriminator(nn.Module):
    
    def __init__(self):
        super(Discriminator,self).__init__()        
        
        self.conv1 = nn.Sequential(nn.Conv2d(3,64,4,2,1,bias = False),
                                    nn.BatchNorm2d(64),
                                   nn.LeakyReLU(0.2,True),
                                   nn.Dropout2d(0.5))
        self.conv2 = nn.Sequential(nn.Conv2d(64,128,4,2,1,bias = False),
                                    nn.BatchNorm2d(128),
                                   nn.LeakyReLU(0.2,True),
                                   nn.Dropout2d(0.5))
        self.conv3 = nn.Sequential(nn.Conv2d(128,256,4,2,1,bias = False),
                                    nn.BatchNorm2d(256),
                                   nn.LeakyReLU(0.2,True),
                                   nn.Dropout2d(0.5))
        self.conv4 = nn.Sequential(nn.Conv2d(256,512,4,2,1,bias = False),
                                    nn.BatchNorm2d(512),
                                   nn.LeakyReLU(0.2,True))
        self.validity_conv = nn.Sequential(nn.Conv2d(512,1,4,1,0,bias = False),
                                   nn.Sigmoid())
        
        
    def forward(self,x):
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        validity = self.validity_conv(x)
        
        validity = validity.view(-1)
        
        return validity
    

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def plot_and_save_graph(gen_loss, dis_loss, out_path):
        print(gen_loss, dis_loss)
        import matplotlib.pyplot as plt
        # fig, ax = plt.subplots(figsize=(10,5))
        plt.title("Generator and Discriminator Loss During Training")
        plt.subplot(211)
        plt.plot(gen_loss, label="Generator")
        plt.xlabel("epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.subplot(212)
        plt.plot(dis_loss, label="Discriminator")
        plt.xlabel("epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(out_path, "acgan_loss.png"))



class Trainer():
    def __init__(self,
                 generator,
                 discriminator,
                 optimG,
                 optimD,
                 validity_loss,
                 n_epochs,
                 trainloader,
                 outpath
                 ):
        
        self.gen = generator
        self.disc = discriminator
        self.optimG = optimG
        self.optimD = optimD
        self.validity_loss = validity_loss
        self.n_epochs = n_epochs
        self.trainloader = trainloader
        self.real_label = 1
        self.fake_label = 0
        self.out_path = outpath
        self.gen_error = []
        self.dis_error = []
        self.fretchet_distances = []

    def run_trainer(self):
        for epoch in range(self.n_epochs):
            print('Epoch - {}'.format(epoch))
            errorD, errorG = self._run()

            self.gen_error.append(errorG)
            self.dis_error.append(errorD)

            noise = torch.randn(10,100,device = device)  
            labels = torch.arange(0,10,dtype = torch.long,device = device)
            gen_images = self.gen(noise).detach()

            images = make_grid(gen_images)

            images = images.cpu().numpy()
            images = images/2 + 0.5
            fig1, ax1 = plt.subplots()
            ax1.plot([1,2,3])
            ax1.imshow(np.transpose(images,axes = (1,2,0)))
            ax1.axis('off')
            plt.savefig(os.path.join(self.out_path,'acgan_fake.png'))

            #plot_and_save_graph(self.gen_error, self.dis_error, self.out_path)
        return self.gen_error, self.dis_error, self.fretchet_distances
    
    def _run(self):
        batch_error_dis = []
        batch_error_gen = []
        for idx, (images,_) in enumerate(self.trainloader,0):
        
            batch_size = images.size(0)
            images = images.to(device)
            
            self.optimD.zero_grad()       
            
            # real
            validity_label = torch.full((batch_size,),self.real_label , device = device).type(torch.float32)
            pvalidity = self.disc(images).type(torch.float32)       
            errD_real = validity_loss(pvalidity, validity_label)            
            errD_real.backward()
            D_x = pvalidity.mean().item()        
            
            #fake 
            noise = torch.randn(batch_size,100,device = device) 
            fakes = self.gen(noise)
            validity_label.fill_(self.fake_label)
            pvalidity= self.disc(fakes.detach())       
            errD_fake = validity_loss(pvalidity, validity_label)
        
            errD_fake.backward()
            
            D_G_z1 = pvalidity.mean().item()
            
            #finally update the params!
            errD = errD_real + errD_fake
            
            self.optimD.step()
        
            
            # ------------------------
            #      gen
            # ------------------------
            
            
            self.optimG.zero_grad()
            validity_label.fill_(self.real_label)
            fakes = self.gen(noise)
            pvalidity = self.disc(fakes)
            errG_val = validity_loss(pvalidity, validity_label)        
            
            errG_val.backward()
            
            D_G_z2 = pvalidity.mean().item()
            
            self.optimG.step()
            batch_error_dis.append(errD.detach().cpu())
            batch_error_gen.append(errG_val.detach().cpu())

        fretchet_dist = calculate_fretchet(images, fakes)
        # print('Fretchet Distance: ', fretchet_dist)
        self.fretchet_distances.append(fretchet_dist)

        return np.mean(batch_error_dis), np.mean(batch_error_gen)


generator = Generator().to(device)
generator.apply(weights_init)

discriminator = Discriminator().to(device)
discriminator.apply(weights_init)

optimG = optim.Adam(generator.parameters(), 0.0002, betas = (0.5,0.999))
optimD = optim.Adam(discriminator.parameters(), 0.0002, betas = (0.5,0.999))


validity_loss = nn.BCELoss()

def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

epochs = 300
batch_size = 1024
out_path = 'DCGaN_Out_300'
make_dir(out_path)

tf = transforms.Compose([transforms.Resize(64),
                         transforms.ToTensor(),
                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                        ])

trainset = torchvision.datasets.CIFAR10(root = './data', train = True, download = True,
                                     transform = tf)


testset = torchvision.datasets.CIFAR10(root = './data', train = False, download = True,
                                    transform = tf)

dataset = torch.utils.data.ConcatDataset([trainset, testset])


trainloader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, 
                                         num_workers = 2, shuffle = True)

trainer = Trainer(generator,
                 discriminator,
                 optimG,
                 optimD,
                 validity_loss,
                 epochs,
                 trainloader,
                 out_path)

gen_error, dis_error, fretchet_distances = trainer.run_trainer()

with open(os.path.join(out_path, 'gen_error.txt'), 'w') as f:
    for line in gen_error:
        f.write(f"{line}\n")

with open(os.path.join(out_path, 'dis_error.txt'), 'w') as f:
    for line in dis_error:
        f.write(f"{line}\n")

with open(os.path.join(out_path, 'fretchet_distances.txt'), 'w') as f:
    for line in fretchet_distances:
        f.write(f"{line}\n")

# plot_and_save_graph(gen_error, dis_error, out_path)