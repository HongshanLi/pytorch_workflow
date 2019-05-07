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

class PressToAmp(Dataset):
    def __init__(self, data_dir, file_names):
        self.data = []
        for file_name in file_names:
            data_path = os.path.join(data_dir, file_name)
            data = loadmat(data_path)
            data = data["data"]
            self.data.append(data)
        
        # concatenate the data
        self.data = np.concatenate(self.data, 0)

        self.data = self.data.astype('float32')

        # shuffle the data
        np.random.shuffle(self.data)

        # normalize the rpm
        rpm = self.data[:, 0] / 10000.0
        rpm = rpm.reshape(-1, 1)
        pressure = self.data[:, 35:]
        print(rpm.shape, pressure.shape)
        self.features = np.concatenate([rpm, pressure], axis=1)
    
        # targets is the average pressure on 33 blades
        self.targets = self.data[:,1:34] 
        self.targets = np.sum(self.targets, axis=1, keepdims=True)/33.0
        
        # keep 10% for validation and 10% for test
        self.purpose = None
        l = self.__len__()
        self.train_l = math.floor(l*0.8)
        self.val_l = math.floor(l*0.1)
        self.test_l = math.floor(l*0.1)

    def __len__(self):
        if self.purpose:
            return len(self.targets_sub)
        else:
            return len(self.data)

    def __getitem__(self, idx):
        return self.features_sub[idx], self.targets_sub[idx]

    def train(self):
        self.purpose = "train"
        tr = self.train_l
        self.features_sub = self.features[:tr]
        self.targets_sub = self.targets[:tr]
        return self

    def validate(self):
        self.purpose = "validate"
        tr=self.train_l
        v = self.val_l
        self.features_sub = self.features[tr:tr+v]
        self.targets_sub = self.targets[tr:tr+v]
        return self

    def test(self):
        self.purpose = "test"
        tr=self.train_l
        v=self.val_l
        t=self.test_l
        self.features_sub = self.features[tr+v:tr+v+t]
        self.targets_sub = self.targets[tr+v:tr+v+t]
        return self

class Model(nn.Module):
    def __init__(self, hidden_size):
        '''
        Args:
            hidden_size: a list of size of the hidden layers,
            excluding the output layer
        '''
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.layers = []
        self.layers.append(
            nn.Sequential(nn.Linear(10501, hidden_size[0]),
            nn.BatchNorm1d(hidden_size[0]),
            nn.ReLU()))
        for i in range(len(hidden_size) - 1):
            self.layers.append(
                nn.Sequential(nn.Linear(hidden_size[i], hidden_size[i+1]),
                nn.BatchNorm1d(hidden_size[i+1]),
                nn.ReLU()))

        self.layers.append(
            nn.Linear(hidden_size[-1], 1)
        )
        self.layers = nn.Sequential(*self.layers)
        
    
    def forward(self, x):
        x = self.layers(x)
        return x

class Trainer(object):
    def __init__(self, dataset, model, batch_size, epochs):
        self.dataset = dataset
        self.batch_size = batch_size

        # load model
        self.model = model
        self.model_name = ""
        for h in self.model.hidden_size:
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
        data_train = self.dataset.train()
        self.train_loader = DataLoader(
            data_train, batch_size=self.batch_size, num_workers=2)

        for epoch in range(1, self.epochs+1):
            self.train_one_epoch(epoch)
            #self.validate(epoch)
    
    def train_one_epoch(self, epoch):
        for step, (features, target) in enumerate(self.train_loader):
            features = features.to(self.device)
            target = target.to(self.device)
            output = self.model(features)
            
            loss = self.compute_loss(output, target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if step % 10 == 0:
                # compute average error (relative to target)
                error = torch.abs(output - target) / target 
                error = torch.sum(error) / len(error)
                error = error * 100
                error = error.detach().cpu().item()
                
                message = "Epoch: {}, Step: {}, Error: {:0.2f}%".format(
                    epoch, step, error)
                print(message)

    def compute_loss(self, output, target):
        loss = F.mse_loss(output, target)
        return loss

model = Model([5000, 2500, 1200, 600, 300, 100, 50])

data_dir = "/scr1/li108/data/press_to_amp/"
file_names = ["HL_CL1_withair_dec1_data.mat"]
dataset = PressToAmp(data_dir, file_names)
trainer = Trainer(dataset, model, 20, 20)
trainer.train()
