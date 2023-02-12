
# from trainer import Trainer
# from model import model0, model1, model2
# from dataloader import prep_data

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

from torch.utils.data import TensorDataset
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import random

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
                 checkpoint_start_epoch=50
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
        #             if self.notebook:
        #                 print('Notebook')
        #                 from tqdm.notebook import tqdm, trange
        #             else:
        #                 from tqdm import tqdm, trange
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
        for epoch in progressbar:
            # print(f'Epoch - {epoch}')

            # Training Block
            train_loss = self._train()
            self.writer_train.add_scalar("Loss", train_loss, epoch)

            # Val Block
            val_loss = self._validate()
            self.writer_val.add_scalar("Loss", val_loss, epoch)

            # lr
            self.writer_train.add_scalar("Learning Rate", self.optimizer.param_groups[0]['lr'], epoch)

            print('Epoch - {} Train Loss - {} Val Loss - {}'.format(epoch, train_loss, val_loss))
            if self.save_final:
                if epoch == self.epochs-1:
                    model_name = 'epoch-{}-loss{:.6f}'.format(epoch, val_loss)
                    torch.save(self.model.state_dict(), os.path.join(self.check_point_path, model_name))
            # if epoch == 0:
            #     loss_max = val_loss
            #     print(loss_max)
            #     pass
            # if self.save_best and epoch >= self.checkpoint_start_epoch:
            #     if val_loss < loss_max:  # loss decreased
            #         print('Saving Checkpoint at val loss dropped from {:.6f} -> {:.6f}'.format(loss_max, val_loss))
            #         model_name = 'epoch-{}-loss{:.6f}'.format(epoch, val_loss)
            #         # torch.save(self.model.state_dict(), os.path.join(self.check_point_path, model_name))
            # elif epoch % self.save_interval == 0:
            #     print('Saving Checkpoint Val loss - {:.6f}'.format(val_loss))
            #     model_name = 'epoch-{}-loss{:.6f}'.format(epoch, val_loss)
            #     torch.save(self.model.state_dict(), os.path.join(self.check_point_path, model_name))
            loss_max = val_loss

        return self.training_loss, self.validation_loss, self.model

    def _train(self):

        self.model.train()
        train_losses = []
        batch_iter = tqdm(enumerate(self.training_DataLoader), 'Training', total=len(self.training_DataLoader),
                          disable=False)
        for i, (x, y) in batch_iter:
            input, target = x.type(torch.float32).to(self.device), y.type(torch.float32).to(self.device)
            self.optimizer.zero_grad()
            output = self.model(input)
            # target = target.unsqueeze(-1)
            #             print('Target Shape - ', target.shape)
            #             print('Output Shape - ', output.shape)
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
                          disable=False)
        for i, (x, y) in batch_iter:
            input, target = x.type(torch.float32).to(self.device), y.type(torch.float32).to(self.device)
            with torch.no_grad():
                out = self.model(input)
                loss = self.criterion(target, out)
                valid_losses.append(loss.item())
        self.validation_loss.append(np.mean(valid_losses))
        batch_iter.close()
        return np.mean(valid_losses)

#example models
class model0(nn.Module):
    def __init__(self, input_size=1, output_size=1):
        super().__init__()
        self.dense_1 = nn.Linear(input_size, 5)
        self.dense_2 = nn.Linear(5, 10)
        self.dense_3 = nn.Linear(10, 10)
        self.dense_4 = nn.Linear(10, 10)
        self.dense_5 = nn.Linear(10, 10)
        self.dense_6 = nn.Linear(10, 10)
        self.dense_7 = nn.Linear(10, 5)
        self.dense_8 = nn.Linear(5, output_size)

    def forward(self, input_data):
        x1 = F.relu(self.dense_1(input_data))
        x2 = F.relu(self.dense_2(x1))
        x3 = F.relu(self.dense_3(x2))
        x4 = F.relu(self.dense_4(x3))
        x5 = F.relu(self.dense_5(x4))
        x6 = F.relu(self.dense_6(x5))
        x7 = F.relu(self.dense_7(x6))
        x8 = self.dense_8(x7)
        return x8

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
        x5 = self.dense5(x4)
        return x5

class model2(nn.Module):
    def __init__(self, input_size=1, output_size=1):
        super(model2, self).__init__()
        self.dense1 = nn.Linear(input_size, 190)
        self.dense2 = nn.Linear(190, output_size)
    def forward(self, input_data):
        x1 = F.relu(self.dense1(input_data))
        x2 = self.dense2(x1)
        return x2

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

def evaluate_model(model, eval_input_, device=0):
  output = []
  eval_input = eval_input_.reshape(eval_input_.shape[0], 1)
  eval_input = torch.from_numpy(eval_input).float()
  EvalDataLoader = DataLoader(TensorDataset(eval_input), batch_size = 1024, shuffle=False)
  for x in EvalDataLoader:
    model.eval()
    x = x[0].type(torch.float32).to(device)
    out = model(x)
    output_ = list(out.flatten().detach().cpu().numpy())
    output += output_
  return output

# model_0 = model0()
# model_1 = model1()
# model_2 = model2()

gpu_id = 0
loss_fn = nn.MSELoss()
lr = 1e-4
func1 = lambda x: (np.sin(5 * (np.pi) * x)) / (5 * np.pi * x)
func2 = lambda x: np.sign(np.sin(5*np.pi*x))
training_DataLoader,  validation_DataLoader = prep_data(func=func1,batch_size=2048*2)
epochs =  20000
notebook = True
checkpoint_start_epoch = 5 #Not using
path2write = r"C:\Users\UMA\Desktop\grad\Deep_Learning\code\HW1\sim_func"

model0 = model0()
optimizer = torch.optim.Adam(model0.parameters(), lr = lr)
trainer = Trainer(model=model0,
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
                      checkpoint_start_epoch=checkpoint_start_epoch )
training_losses_m0, validation_losses_m0, model0 = trainer.run_trainer()

model1 = model1()
optimizer = torch.optim.Adam(model1.parameters(), lr = lr)
trainer = Trainer(model=model1,
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
                      checkpoint_start_epoch=checkpoint_start_epoch )
training_losses_m1, validation_losses_m1, model1 = trainer.run_trainer()

model2 = model2()
trainer = Trainer(model=model2,
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
                      checkpoint_start_epoch=checkpoint_start_epoch )
training_losses_m2, validation_losses_m2, model2 = trainer.run_trainer()


fig = plt.figure(figsize=(15, 15))
plt.plot(np.array(training_losses_m0), label='Model0')
plt.plot(np.array(training_losses_m1), label='Model1')
plt.plot(np.array(training_losses_m2), label='Model2')
plt.title('sin(5*pi*x)/5*pi*x - Training Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE loss')
plt.legend()
plt.savefig(os.path.join(path2write, 'sin_func_loss.png'))

points = 10000
eval_input_ = np.linspace(1e-4, 1, points)
model_1_op = evaluate_model(model0, eval_input_)
model_2_op = evaluate_model(model1, eval_input_)
model_3_op = evaluate_model(model2, eval_input_)

fig = plt.figure(figsize=(15, 15))
exp_output = list(map(func1, eval_input_))
plt.plot(eval_input_, exp_output, label='Expected Output')
plt.plot(eval_input_, np.array(model_1_op), label='Model0 output')
plt.plot(eval_input_, np.array(model_2_op), label='Model1 output')
plt.plot(eval_input_, np.array(model_3_op), label='Model2_output')
plt.title('sin(5*pi*x)/5*pi*x - output')
plt.xlabel('Input')
plt.ylabel('Output')
plt.legend()
plt.xticks(np.arange(0, 11, 10)/10)
plt.savefig(os.path.join(path2write, 'sin_output.png'))
# plt.xlim(1e-4,1)
plt.show()



