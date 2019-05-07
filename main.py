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

seed = 999
np.random.seed(seed)

class PressToAmp(Dataset):
    def __init__(self, data_dir, file_names, purpose='train'):
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
        self.features = np.concatenate([rpm, pressure], axis=1)
    
        # targets is the average pressure on 33 blades
        self.targets = self.data[:,1:34] 
        self.targets = np.sum(self.targets, axis=1, keepdims=True)/33.0
        
        # keep 10% for validation and 10% for test
        self.purpose = purpose
        l = self.__len__()
        tr= math.floor(l*0.8)
        v= math.floor(l*0.1)
        te = math.floor(l*0.1)

        if self.purpose == "train":
            self.features = self.features[:tr]
            self.targets = self.targets[:tr]
        elif self.purpose == "validate":
            self.features = self.features[tr:tr+v]
            self.targets = self.targets[tr:tr+v]
        elif self.purpose == "test":
            self.features = self.features[tr+v:tr+v+te]
            self.targets = self.targets[tr+v:tr+v+te]
        else:
            raise ValueError("purpose must be train, validate, or test")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


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
            nn.Sigmoid()))
        for i in range(len(hidden_size) - 1):
            self.layers.append(
                nn.Sequential(nn.Linear(hidden_size[i], hidden_size[i+1]),
                nn.BatchNorm1d(hidden_size[i+1]),
                nn.Sigmoid()))

        self.layers.append(
            nn.Linear(hidden_size[-1], 1)
        )
        self.layers = nn.Sequential(*self.layers)
        
    
    def forward(self, x):
        x = self.layers(x)
        return x

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
            batch_size=batch_size*10, shuffle=True, num_workers=2)

        self.test_loader = DataLoader(dataset_test,
            batch_size=batch_size*10, shuffle=True, num_workers=2)

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
        for epoch in range(1, self.epochs+1):
            self.train_one_epoch(epoch)
            self.validate(epoch)
    
    def train_one_epoch(self, epoch):
        self.model.train()
        for step, (features, target) in enumerate(self.train_loader):
            features = features.to(self.device)
            target = target.to(self.device)
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
            target = target.to(self.device)

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

if __name__=="__main__":
    model = Model([1000, 100, 10])
    data_dir = "/content/drive/My Drive/data/press_to_amp"
    file_names = ["HL_CL1_withair_dec1_data.mat"]
    dataset_train = PressToAmp(data_dir, file_names, 'train')
    dataset_val = PressToAmp(data_dir, file_names, 'validate')
    dataset_test = PressToAmp(data_dir, file_names, 'test')

    trainer = Trainer(dataset_train,dataset_val,dataset_test,
         model, 32, 50)
    trainer.train()
