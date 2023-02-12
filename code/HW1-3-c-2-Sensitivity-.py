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
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score


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
        self.training_accuracy = []
        self.validation_accuracy = []

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
            print(f'Epoch - {epoch}')

            # Training Block
            train_loss, train_accuracy = self._train()
            self.writer_train.add_scalar("Train Loss", train_loss, epoch)
            self.writer_train.add_scalar("Train Accuracy", train_accuracy, epoch)

            # Val Block
            val_loss, val_accuracy = self._validate()
            self.writer_val.add_scalar("Val Loss", val_loss, epoch)
            self.writer_val.add_scalar("Val Accuracy", val_accuracy, epoch)

            # lr
            self.writer_train.add_scalar("Learning Rate", self.optimizer.param_groups[0]['lr'], epoch)

            print(
                'Epoch - {} Train Loss - {:.6f} Val Loss - {:.6f} Train Accuracy - {:.6f} Val Accuracy - {:.6f}'.format(
                    epoch, train_loss, val_loss, train_accuracy, val_accuracy))
            if self.save_final:
                if epoch == self.epochs - 1:
                    model_name = 'epoch-{}-loss{:.6f}'.format(epoch, val_loss)
                    torch.save(self.model.state_dict(), os.path.join(self.check_point_path, model_name))
            loss_max = val_loss
        sensitivity = self.sensitivity()
        return train_loss, train_accuracy, val_loss, val_accuracy, sensitivity
        # return self.training_loss, self.validation_loss, self.model, self.training_accuracy, self.validation_accuracy

    def _train(self):

        self.model.train()
        train_losses = []
        batch_iter = tqdm(enumerate(self.training_DataLoader), 'Training', total=len(self.training_DataLoader),
                          disable=True)
        batch_acc = 0
        for i, (x, y) in batch_iter:
            input, target = x.type(torch.float32).to(self.device), y.type(torch.float32).to(self.device)
            self.optimizer.zero_grad()
            target = target.type(torch.LongTensor).to(self.device)
            output = self.model(input)
            loss = self.criterion(output, target)
            train_losses.append(loss.item())
            loss.backward()
            self.optimizer.step()
            pred = output.argmax(dim=1, keepdim=True)  # max of prob
            pred = pred.flatten()
            batch_acc += torch.mean(pred.eq(target.view_as(pred)).type(torch.FloatTensor))

        accuracy = batch_acc / len(self.training_DataLoader)
        self.training_loss.append(np.mean(train_losses))  # Mean batch loss
        self.learning_rate.append(self.optimizer.param_groups[0]['lr'])
        self.training_accuracy.append(accuracy)

        batch_iter.close()  # clean up the bar
        return np.mean(train_losses), accuracy

    def _validate(self):

        self.model.eval()
        valid_losses = []
        batch_iter = tqdm(enumerate(self.validation_DataLoader), 'validation', total=len(self.validation_DataLoader),
                          disable=True)
        batch_acc = 0
        for i, (x, y) in batch_iter:
            input, target = x.type(torch.float32).to(self.device), y.to(self.device)
            with torch.no_grad():
                output = self.model(input)
                target = target.type(torch.LongTensor).to(self.device)
                loss = self.criterion(output, target)
                valid_losses.append(loss.item())
                pred = output.argmax(dim=1, keepdim=True)
                batch_acc += torch.mean(pred.eq(target.view_as(pred)).type(torch.FloatTensor)).item()

        accuracy = batch_acc / len(self.validation_DataLoader)
        self.validation_loss.append(np.mean(valid_losses))
        self.validation_accuracy.append(accuracy)
        batch_iter.close()
        return np.mean(valid_losses), accuracy

    def sensitivity(self):
        num = 0
        FNorm = 0
        for p in self.model.parameters():
            grad = 0.0
            if p.grad is not None:
                num += 1
                grad = p.grad
                FNorm += torch.linalg.norm(grad).cpu().numpy()
        return FNorm / num

gpu_id = 0
loss_fn = nn.CrossEntropyLoss()
lr = 1e-4
epochs =  15
notebook = True
checkpoint_start_epoch = 5 #Not using
path2write = r"C:\Users\UMA\Desktop\grad\Deep_Learning\code\HW1\report\sensitivity"

class CNN2(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
    self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

    self.dense1 = nn.Linear(32*14*14, 128)
    self.dense2 = nn.Linear(128, 10)

  def forward(self, x):
    x = self.conv1(x)
    x = F.relu(x)
    x = self.pool1(x)
    x = x.view(x.shape[0], -1)
    x = self.dense1(x)
    x = self.dense2(x)
    out = F.log_softmax(x)
    return out


batch_ = [8, 32, 128, 512, 2048, 4016]
train_loss_ = []
val_loss_ = []
train_acc_ = []
val_acc_ = []
sensitivity_ = []
for batch_size in tqdm(batch_):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    dataset1 = datasets.MNIST('../data', train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST('../data', train=False, transform=transform)

    training_DataLoader = DataLoader(dataset1, batch_size=batch_size, shuffle=True)
    validation_DataLoader = DataLoader(dataset2, batch_size=batch_size, shuffle=True)

    model = CNN2()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
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
                      path2write=path2write,
                      checkpoint_start_epoch=checkpoint_start_epoch)
    training_loss, training_accuracy, validation_loss, validation_accuracy, sensitivity = trainer.run_trainer()
    train_loss_.append(training_loss)
    val_loss_.append(validation_loss)
    train_acc_.append(training_accuracy)
    val_acc_.append(validation_accuracy)
    sensitivity_.append(sensitivity)

fig, ax1 = plt.subplots(figsize=(15, 5))

ax1 = fig.add_subplot(1, 1, 1)
ax1.plot(batch_, train_loss_, 'b-', label='Train')
ax1.plot(batch_, val_loss_, 'b--', label='Validation')
ax1.legend()
ax1.set_xlabel('Batch Size')
ax1.set_ylabel('Loss', color='b')
ax1.set_title('Batch Size Vs loss Vs Sensitivity')

ax2 = ax1.twinx()
ax2.plot(batch_, sensitivity_, 'r-')
ax2.legend()
ax2.set_ylabel('Sensitivity', color='r')
fig.tight_layout()
fig.savefig(os.path.join(path2write, 'sensitivity_loss.png'))

fig, ax1 = plt.subplots(figsize=(15, 5))

ax1 = fig.add_subplot(1, 1, 1)
ax1.plot(batch_, train_acc_, 'b-', label='Train')
ax1.plot(batch_, val_acc_, 'b--', label='Validation')
ax1.legend()
ax1.set_xlabel('Batch Size')
ax1.set_ylabel('Accuracy', color='b')
ax1.set_title('Batch Size Vs Accuracy Vs Sensitivity')

ax2 = ax1.twinx()
ax2.plot(batch_, sensitivity_, 'r-')
ax2.legend()
ax2.set_ylabel('Sensitivity', color='r')
fig.tight_layout()
fig.savefig(os.path.join(path2write, 'sensitivity_accuracy.png'))
