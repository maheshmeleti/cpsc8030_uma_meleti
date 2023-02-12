import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
import seaborn
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision import transforms

from torch.utils.data import TensorDataset
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from autograd_lib import autograd_lib
from collections import defaultdict
import random
#
# def seed_all(n=1998):
#     torch.manual_seed(n)
#     np.random.seed(n)
#     random.seed(n)
# seed_all(195)


class Trainer:

    def __init__(self,
                 model: torch.nn.Module,
                 device: torch.device,
                 criterion: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 training_DataLoader: torch.utils.data.Dataset,
                 validation_DataLoader: None,
                 # lr_scheduler: torch.optim.lr_scheduler = None,
                 epochs: int = 100,
                 epoch: int = 0,
                 notebook: bool = False,
                 path2write: str = None,
                 save_best=False,
                 save_final=True,
                 save_interval=10,
                 checkpoint_start_epoch=50,
                 gradient_norm=False,
                 min_ratio = True
                 ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        # self.lr_scheduler = lr_scheduler
        self.training_DataLoader = training_DataLoader
        self.validation_DataLoader = validation_DataLoader
        self.device = device
        self.epochs = epochs
        self.epoch = epoch
        self.notebook = notebook
        self.path2write = path2write
        LOG_DIR = os.path.join(path2write, 'Log')  # path2write + 'Log/'
        self.writer_train = SummaryWriter(os.path.join(LOG_DIR, "train"))
        self.writer_val = SummaryWriter(os.path.join(LOG_DIR, "val"))
        self.check_point_path = os.path.join(path2write, 'check_points')
        if not os.path.exists(self.check_point_path):
            os.makedirs(self.check_point_path)
        self.save_best = save_best
        self.save_final = save_final
        self.save_interval = save_interval
        self.checkpoint_start_epoch = checkpoint_start_epoch
        self.training_loss = []
        self.validation_loss = []
        self.learning_rate = []
        self.gradient_norm = gradient_norm
        self.grad_list = []
        self.min_ratio_list = []
        self.min_ratio = min_ratio
        self.activations = defaultdict(int)
        self.hess = defaultdict(float)

    def run_trainer(self):
        self.model.to(self.device)
        #         print(next(self.model.parameters()).device)
        if self.notebook:
            print('Notebook')
            from tqdm.notebook import tqdm, trange
        else:
            from tqdm import tqdm, trange
        #         print(self.epochs)
        progressbar = trange(self.epochs, desc='Progress', disable=True)  # don't show progressbar
        loss_max = None
        min_ratio = None
        for epoch in progressbar:
            # print(f'Epoch - {epoch}')

            # Training Block
            train_loss = self._train()
            # self.min_ratio_list.append(ratio_mean)
            self.writer_train.add_scalar("Loss", train_loss, epoch)

            # Val Block
            val_loss = self._validate()
            self.writer_val.add_scalar("Loss", val_loss, epoch)

            # lr
            self.writer_train.add_scalar("Learning Rate", self.optimizer.param_groups[0]['lr'], epoch)

            if self.save_final:
                if epoch == self.epochs - 1:
                    model_name = 'epoch-{}-loss{:.6f}'.format(epoch, val_loss)
                    torch.save(self.model.state_dict(), os.path.join(self.check_point_path, model_name))

            loss_max = val_loss

            # if self.gradient_norm:
            grad_all = 0.0
            for p in self.model.parameters():
                grad = 0.0
                if p.grad is not None:
                    grad = (p.grad.cpu().data.numpy() ** 2).sum()
                    grad_all += grad
            grad_norm = grad_all ** 0.5
            # self.grad_list.append()

            # print('Grad Norm {:.6f}'.format(grad_norm))
            if grad_norm < 0.025:
                min_ratio = self.compute_minimal_ratio(self.training_DataLoader.dataset.tensors[0], self.training_DataLoader.dataset.tensors[1])
                self.min_ratio_list.append(min_ratio)
                print('Epoch - {} Train Loss - {:.6f} Val Loss - {:.6f} Min Ratio - {:.6f}'.format(epoch, train_loss,
                                                                                                   val_loss,
                                                                                                   min_ratio))
                break


        return train_loss, min_ratio
        # return self.training_loss, self.validation_loss, self.model

    def _train(self):

        self.model.train()
        train_losses = []
        batch_iter = tqdm(enumerate(self.training_DataLoader), 'Training', total=len(self.training_DataLoader),
                          disable=True)

        for i, (x, y) in batch_iter: # x- batch X dims
            input, target = x.type(torch.float32).to(self.device), y.type(torch.float32).to(self.device)
            self.optimizer.zero_grad()
            output = self.model(input)
            loss = self.criterion(output, target)
            train_losses.append(loss.item())
            loss.backward()
            self.optimizer.step()

        self.training_loss.append(np.mean(train_losses))  # Mean batch loss
        self.learning_rate.append(self.optimizer.param_groups[0]['lr'])


        batch_iter.close()  # clean up the bar
        return np.mean(train_losses)

    def _validate(self):

        self.model.eval()
        valid_losses = []
        batch_iter = tqdm(enumerate(self.validation_DataLoader), 'validation', total=len(self.validation_DataLoader),
                          disable=True)
        for i, (x, y) in batch_iter:
            input, target = x.type(torch.float32).to(self.device), y.type(torch.float32).to(self.device)
            with torch.no_grad():
                out = self.model(input)
                loss = self.criterion(target, out)
                valid_losses.append(loss.item())
        self.validation_loss.append(np.mean(valid_losses))
        batch_iter.close()
        return np.mean(valid_losses)

    def save_activations(self, layer, A, _):
        self.activations[layer] = A

    def compute_hess(self, layer, _, B):
        A = self.activations[layer]
        BA = torch.einsum('nl,ni->nli', B, A)  # do batch-wise outer product

        # full Hessian
        self.hess[layer] += torch.einsum('nli,nkj->likj', BA, BA)

    def compute_minimal_ratio(self, train, target):
        # model.to(device)
        train = train.to(self.device)
        target = target.to(self.device)
        self.model.zero_grad()

        # compute Hessian matrix
        # save the gradient of each layer
        with autograd_lib.module_hook(self.save_activations):
            output = model(train)
            loss = self.criterion(output, target)

        # compute Hessian according to the gradient value stored in the previous step
        with autograd_lib.module_hook(self.compute_hess):
            autograd_lib.backward_hessian(output, loss='LeastSquares')

        layer_hess = list(self.hess.values())
        minimum_ratio = []

        # compute eigenvalues of the Hessian matrix
        for h in layer_hess:
            size = h.shape[0] * h.shape[1]
            h = h.reshape(size, size)
            h_eig = torch.symeig(
                h).eigenvalues  # torch.symeig() returns eigenvalues and eigenvectors of a real symmetric matrix
            num_greater = torch.sum(h_eig > 0).item()
            minimum_ratio.append((num_greater/len(h_eig)))

        ratio_mean = np.mean(minimum_ratio)
        return ratio_mean

class model1(nn.Module):
    def __init__(self, input_size=1, output_size=1):
        super().__init__()
        self.dense1 = nn.Linear(input_size, 10)
        self.dense2 = nn.Linear(10, 18)
        self.dense3 = nn.Linear(18, 15)
        self.dense4 = nn.Linear(15, 4)
        self.dense5 = nn.Linear(4, output_size)

    def forward(self, input_data):
        x1 = F.relu(self.dense1(input_data))
        x2 = F.relu(self.dense2(x1))
        x3 = F.relu(self.dense3(x2))
        x4 = F.relu(self.dense4(x3))
        x5 = F.relu(self.dense5(x4))
        return x5

class SineApproximator(nn.Module):
    def __init__(self):
        super(SineApproximator, self).__init__()
        self.regressor = nn.Sequential(nn.Linear(1, 256),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(256, 1))
    def forward(self, x):
        output = self.regressor(x)
        return output
class model2(nn.Module):
    def __init__(self, input_size=1, output_size=1):
        super(model2, self).__init__()
        self.dense1 = nn.Linear(input_size, 128)
        self.dense2 = nn.Linear(128, 64)
        self.dense3 = nn.Linear(64, 1)
    def forward(self, x):
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        out = self.dense3(x)

        return out

def prep_data(func, data_length=2500, train_ratio=0.7, batch_size=8, shuffle=True):
    X = np.linspace(1e-4, 1, data_length)
    # np.random.shuffle(X)
    y = np.array(list(map(func, X)))
    X = X.reshape(X.shape[0], 1)
    y = y.reshape(y.shape[0], 1)
    X = torch.from_numpy(X).float()
    y = torch.from_numpy(y).float()
    X_train, X_val = X[0:int(data_length * train_ratio), ], X[int(data_length * train_ratio):, ]
    y_train, y_val = y[0:int(data_length * train_ratio), ], y[int(data_length * train_ratio):, ]
    assert X_train.shape[0] == y_train.shape[0]
    assert X_val.shape[0] == y_val.shape[0]
    TrainDataLoader = DataLoader(TensorDataset(X_train, y_train), batch_size, shuffle)
    ValDataLoader = DataLoader(TensorDataset(X_val, y_val), batch_size, shuffle)

    return TrainDataLoader, ValDataLoader

gpu_id = 0
loss_fn = nn.MSELoss()
lr = 1e-4
func1 = lambda x: (np.sin(5 * (np.pi) * x)) / (5 * np.pi * x)
func2 = lambda x: np.sign(np.sin(5*np.pi*x))
training_DataLoader,  validation_DataLoader = prep_data(func=func1,batch_size=4096)
epochs =  2000
notebook = True
checkpoint_start_epoch = 5 #Not using
path2write = r"C:\Users\UMA\Desktop\grad\Deep_Learning\code\HW1\min_ratio"


# model = model1()
# model = SineApproximator()
train_losses = []
min_ratios = []
for i in tqdm(range(100)):
    # model = model2()
    model = SineApproximator()
    autograd_lib.register(model)
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    trainer = Trainer(model=model,
                          device=gpu_id,
                          criterion=loss_fn,
                          optimizer=optimizer,
                          training_DataLoader=training_DataLoader,
                          validation_DataLoader=validation_DataLoader,
                          # lr_scheduler=lr_scheduler,
                          epochs=epochs,
                          epoch=0,
                          notebook=True,
                          path2write= path2write,
                          checkpoint_start_epoch=checkpoint_start_epoch,
                          gradient_norm = True)
    train_loss, min_ratio = trainer.run_trainer()
    train_losses.append(train_loss)
    min_ratios.append(min_ratio)

fig = plt.figure(figsize=(15, 5))

# ax1 = fig.add_subplot(1,2,1)
# ax1.plot(train_losses, 'r-', label='Training Loss')
# ax1.legend()
# ax1.set_title('Training Loss')
# ax1.set_xlabel('Epoch')
# ax1.set_ylabel('Loss')
# ax1.savefig(os.path.join(path2write, 'Training Loss.png'))

ax2 = fig.add_subplot(1, 1, 1)
ax2.scatter(min_ratios, train_losses, alpha=0.5, label='Minimum Ratio')
ax2.legend()
ax2.set_title('Minimal Ratio')
ax2.set_xlabel('Minimal Ratio')
ax2.set_ylabel('Loss')

# extent = ax2.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
# fig.savefig('ax2_figure.png', bbox_inches=extent)

fig.savefig('minimal ratio')
# plt.show()




















