# Define your model here, configure everything about the model
# including wheather to put on cuda device
# or if you want to do some computation on cpu and some on gpus
# If you want to split model's parameters on multiple gpus,
# define it here as well

# If you want to do data parallelism, define it here as well


import os
import random
import math
import numpy as np
from scipy.io import loadmat

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from load_data import PressToAmpFFT
import torch.nn.parallel.DistributedDataParallel as DDP


seed = 999
np.random.seed(seed)




class MLP(nn.Module):
    def __init__(self, layer_size):
        '''
        Args:
            hidden_size: a list of size of the hidden layers,
            excluding the output layer
        '''
        super(MLP, self).__init__()
        self.layer_size = layer_size
        _layers = []
        for i in range(len(layer_size)-2):
            module = nn.Sequential(
                    nn.Linear(layer_size[i], layer_size[i+1]),
                    nn.BatchNorm1d(layer_size[i+1]),
                    nn.ReLU())
            _layers.append(module)

        # output layer
        _layers.append(
                nn.Linear(layer_size[-2], layer_size[-1])
                )
        
        self.layers = nn.Sequential(*_layers)
    
    def forward(self, x):
        x = self.layers(x)
        return x

class LayerNormMLP(nn.Module):
    '''Multiple layers perceptron with layer normalization'''
    def __init__(self, layer_size):
        super(LayerNormMLP, self).__init__()
        self.layer_size = layer_size
        _layers = []
        for i in range(len(layer_size)-2):
            module = nn.Sequential(
                    nn.Linear(layer_size[i], layer_size[i+1]),
                    nn.LayerNorm(layer_size[i+1]),
                    nn.Sigmoid())
            _layers.append(module)

        # output layer
        _layers.append(
                nn.Linear(layer_size[-2], layer_size[-1])
                )
        
        self.layers = nn.Sequential(*_layers)

    def forward(self, x):
        return self.layers(x)


class Trainer(object):
    def __init__(self, dataset_train, dataset_val,
        dataset_test, model, batch_size, epochs):

        print("Size of training set is:{}".format(
            dataset_train.__len__()))

        print("Size of validation set is:{}".format(
            dataset_val.__len__()))

        print("Size of testing set is:{}".format(
            dataset_test.__len__()))

        self.train_loader = DataLoader(dataset_train,
            batch_size=batch_size, shuffle=True, num_workers=2)

        self.val_loader = DataLoader(dataset_val,
            batch_size=batch_size*10, shuffle=False, num_workers=2)

        self.test_loader = DataLoader(dataset_test,
            batch_size=batch_size*10, shuffle=False, num_workers=2)

        # load model
        self.model = model
        self.model_name = ""
        for h in self.model.layer_size:
            self.model_name+=str(h)+"_"
        
        self.epochs = epochs
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            print("cuda device is available")
        else:
            self.device = torch.device("cpu")
            print("cuda device is NOT available")


        self.model = self.model.to(self.device)

        self.optimizer = optim.Adam(self.model.parameters())

    def train(self):
        for epoch in range(1, self.epochs+1):
            self.train_one_epoch(epoch)
            self.validate(epoch)
    
    def train_one_epoch(self, epoch):
        self.model.train()
        for step, (features, target) in enumerate(self.train_loader):
            features = features.to(self.device)
            target = target.to(self.device).float()
            target = target.view(-1, 1)
            output = self.model(features)
            
            loss = self.compute_loss(output, target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if step % 50 == 0:
                # compute average error (relative to target)
                error = torch.abs(output - target) / target 
                error = torch.sum(error) / len(error)
                error = error.detach().cpu().item()
                loss = loss.detach().cpu().item()
                
                message = "Epoch: {}, Step: {}, Loss: {:0.3f}, Error: {:0.2f}%".format(
                    epoch, step, loss, error*100)
                print(message)

    def validate(self, epoch):
        self.model.eval()
        val_error = []
        for features, target in self.val_loader:
            features = features.to(self.device)
            target = target.to(self.device).float()
            target = target.view(-1, 1)

            output = self.model(features)
            error = torch.abs(output - target) / target 
            error = torch.sum(error) / len(error)
            error = error*100
            error = error.detach().cpu().item()
            val_error.append(error)
        val_error = sum(val_error) / len(val_error)
        message ="Epoch: {}, Validation Error: {:0.2f}%".format(
            epoch, val_error)
        print(message)

    def compute_loss(self, output, target):
        loss = F.mse_loss(output, target)
        return loss

    def save_model(self, ckp_dir, model_name):
        return

if __name__=="__main__":
    model = MLP([8400, 4200, 2100, 1050, 500, 100, 10 , 1])
    data_dir = "/scr1/li108/data/fft"
    file_names = ["HL_CL1_noair_dec1_datafft.mat"]
    dataset_train = PressToAmpFFT(data_dir, file_names, 'train')
    dataset_val = PressToAmpFFT(data_dir, file_names, 'validate')
    dataset_test = PressToAmpFFT(data_dir, file_names, 'test')

    trainer = Trainer(dataset_train,dataset_val,dataset_test,
         model, 16, 10)
    trainer.train()

        
