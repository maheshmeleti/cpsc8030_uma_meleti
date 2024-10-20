from pdb import run
import torch
import torch.nn as nn
from torchvision.utils import make_grid
import torch.optim as optim
import numpy as np
import torchvision
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import os
from torchvision import models
from scipy import linalg
from frechet_distance_cal import calculate_fretchet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Generator(nn.Module):
    
    def __init__(self):
        super(Generator,self).__init__()
        
        #input 100*1*1
        self.layer1 = nn.Sequential(nn.ConvTranspose2d(100,512,4,1,0,bias = False),
                                   nn.ReLU(True))

        #input 512*4*4
        self.layer2 = nn.Sequential(nn.ConvTranspose2d(512,256,4,2,1,bias = False),
                                   nn.BatchNorm2d(256),
                                   nn.ReLU(True))
        #input 256*8*8
        self.layer3 = nn.Sequential(nn.ConvTranspose2d(256,128,4,2,1,bias = False),
                                   nn.BatchNorm2d(128),
                                   nn.ReLU(True))
        #input 128*16*16
        self.layer4 = nn.Sequential(nn.ConvTranspose2d(128,64,4,2,1,bias = False),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU(True))
        #input 64*32*32
        self.layer5 = nn.Sequential(nn.ConvTranspose2d(64,3,4,2,1,bias = False),
                                   nn.Tanh())
        #output 3*64*64
      
        self.embedding = nn.Embedding(10,100)
        
        
    def forward(self,noise,label):
        
        label_embedding = self.embedding(label)
        x = torch.mul(noise,label_embedding)
        x = x.view(-1,100,1,1)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return x
        

class Discriminator(nn.Module):
    
    def __init__(self):
        super(Discriminator,self).__init__()        
        
        #input 3*64*64
        self.layer1 = nn.Sequential(nn.Conv2d(3,64,4,2,1,bias = False),
                                    nn.BatchNorm2d(64),
                                   nn.LeakyReLU(0.2,True),
                                   nn.Dropout2d(0.5))
        
        #input 64*32*32
        self.layer2 = nn.Sequential(nn.Conv2d(64,128,4,2,1,bias = False),
                                    nn.BatchNorm2d(128),
                                   nn.LeakyReLU(0.2,True),
                                   nn.Dropout2d(0.5))
        #input 128*16*16
        self.layer3 = nn.Sequential(nn.Conv2d(128,256,4,2,1,bias = False),
                                    nn.BatchNorm2d(256),
                                   nn.LeakyReLU(0.2,True),
                                   nn.Dropout2d(0.5))
        #input 256*8*8
        self.layer4 = nn.Sequential(nn.Conv2d(256,512,4,2,1,bias = False),
                                    nn.BatchNorm2d(512),
                                   nn.LeakyReLU(0.2,True))
        #input 512*4*4
        self.validity_layer = nn.Sequential(nn.Conv2d(512,1,4,1,0,bias = False),
                                   nn.Sigmoid())
        
        self.label_layer = nn.Sequential(nn.Conv2d(512,11,4,1,0,bias = False),
                                   nn.LogSoftmax(dim = 1))
        
    def forward(self,x):
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        validity = self.validity_layer(x)
        plabel = self.label_layer(x)
        
        validity = validity.view(-1)
        plabel = plabel.view(-1,11)
        
        return validity,plabel


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class Trainer():
    def __init__(self,
                 generator,
                 discriminator,
                 optimG,
                 optimD,
                 n_epochs,
                 trainloader,
                 outpath,
                 validity_loss
                 ):
        
        
        self.n_epochs = n_epochs
        self.trainloader = trainloader
        self.out_path = outpath
        self.real_labels = 0.7 + 0.5 * torch.rand(10, device = device)
        self.fake_labels = 0.3 * torch.rand(10, device = device)
        self.optimG = optimG
        self.optimD = optimG
        self.validity_loss = validity_loss
        self.gen_error = []
        self.dis_error = []
        self.fretchet_distances = []

    
    def run_trainer(self):
        for epoch in range(self.n_epochs):
            print('Epoch - {}'.format(epoch))
            G_losses, D_losses = self.run_()
            self.gen_error.append(G_losses)
            self.dis_error.append(D_losses)

            noise = torch.randn(10,100,device = device)  
            labels = torch.arange(0,10,dtype = torch.long,device = device)
            gen_images = gen(noise,labels).detach()

            images = make_grid(gen_images)

            images = images.cpu().numpy()
            images = images/2 + 0.5
            plt.plot([1,2,3])
            plt.imshow(np.transpose(images,axes = (1,2,0)))
            plt.axis('off')
            plt.savefig(os.path.join(self.out_path,'acgan_fake.png'))

        return self.gen_error, self.dis_error, self.fretchet_distances

    def run_(self):
        G_losses = []
        D_losses = []
        
        for idx, (images,labels) in enumerate(self.trainloader,0):
            
            batch_size = images.size(0)
            labels= labels.to(device)
            images = images.to(device)
            
            real_label = self.real_labels[idx % 10]
            fake_label = self.fake_labels[idx % 10]
            
            fake_class_labels = 10*torch.ones((batch_size,),dtype = torch.long,device = device)
            
            if idx % 25 == 0:
                real_label, fake_label = fake_label, real_label
            
            # ---------------------
            #         disc
            # ---------------------
            
            self.optimD.zero_grad()       
            
            # real
            validity_label = torch.full((batch_size,),real_label , device = device)
    
            pvalidity, plabels = disc(images)       
            
            errD_real_val = self.validity_loss(pvalidity, validity_label)            
            errD_real_label = F.nll_loss(plabels,labels)
            
            errD_real = errD_real_val + errD_real_label
            errD_real.backward()
            
            D_x = pvalidity.mean().item()        
            
            #fake 
            noise = torch.randn(batch_size,100,device = device)  
            sample_labels = torch.randint(0,10,(batch_size,),device = device, dtype = torch.long)
            
            fakes = gen(noise,sample_labels)
            
            validity_label.fill_(fake_label)
            
            pvalidity, plabels = disc(fakes.detach())       
            
            errD_fake_val = self.validity_loss(pvalidity, validity_label)
            errD_fake_label = F.nll_loss(plabels, fake_class_labels)
            
            errD_fake = errD_fake_val + errD_fake_label
            errD_fake.backward()
            
            D_G_z1 = pvalidity.mean().item()
            
            #finally update the params!
            errD = errD_real + errD_fake
            
            self.optimD.step()
        
            
            # ------------------------
            #      gen
            # ------------------------
            
            
            self.optimG.zero_grad()
            
            noise = torch.randn(batch_size,100,device = device)  
            sample_labels = torch.randint(0,10,(batch_size,),device = device, dtype = torch.long)
            
            validity_label.fill_(1)
            
            fakes = gen(noise,sample_labels)
            pvalidity,plabels = disc(fakes)
            
            errG_val = self.validity_loss(pvalidity, validity_label)        
            errG_label = F.nll_loss(plabels, sample_labels)
            
            errG = errG_val + errG_label
            errG.backward()
            
            D_G_z2 = pvalidity.mean().item()
            
            self.optimG.step()
            
            G_losses.append(errG.item())
            D_losses.append(errD.item())

        fretchet_dist = calculate_fretchet(images, fakes)
        self.fretchet_distances.append(fretchet_dist)

        return np.mean(G_losses), np.mean(D_losses)

gen = Generator().to(device)
gen.apply(weights_init)

disc = Discriminator().to(device)
disc.apply(weights_init)

optimG = optim.Adam(gen.parameters(), 0.0002, betas = (0.5,0.999))
optimD = optim.Adam(disc.parameters(), 0.0002, betas = (0.5,0.999))

validity_loss = nn.BCELoss()




tf = transforms.Compose([transforms.Resize(64),
                         transforms.ToTensor(),
                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                        ])

trainset = torchvision.datasets.CIFAR10(root = './data', train = True, download = True,
                                     transform = tf)


testset = torchvision.datasets.CIFAR10(root = './data', train = False, download = True,
                                    transform = tf)

dataset = torch.utils.data.ConcatDataset([trainset, testset])


trainloader = torch.utils.data.DataLoader(dataset, batch_size = 100, 
                                         num_workers = 2, shuffle = True)



def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

out_path = 'ACGan_out2_300_trail'
make_dir(out_path)

n_epochs = 300
outpath = out_path

trainer = Trainer(gen,
                 disc,
                 optimG,
                 optimD,
                 n_epochs,
                 trainloader,
                 outpath,
                 validity_loss)

gen_error, dis_error, fretchet_distances = trainer.run_trainer()
