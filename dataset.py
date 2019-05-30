# Define dataset 
# output of the dataset should be readily used for neural network
# That means all preprocess and transforms should happen here

# MyDataset must take purpose argument to specify if it is intended
# for training, validating or testing




import os
import random
import math
import numpy as np
from scipy.io import loadmat
import types

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

seed = 999
np.random.seed(seed)

class PressToAmpFFT(Dataset):
    '''Dataset for PTA in frequency domain'''
    def __init__(self, data_dir, file_names, purpose='train'):
        if not isinstance(file_names, list):
            raise TypeError("file_names needs to be a list")
        self.data = []
        for f in file_names:
            dpath=os.path.join(data_dir, f)
            data=loadmat(dpath)
            data=data['data_fft']
            self.data.append(data)
        
        # concatenate data from different files into one array
        self.data = np.concatenate(self.data, 0)
        #self.data = self.data.astype(np.float32)

        np.random.shuffle(self.data)
        
        self.rpm = self.data[:,0].real
        self.amp = self.data[:,1:34].real

        # take average of amp
        self.amp = np.mean(self.amp, axis=1)

        # pressure is a complex number
        self.pressure = self.data[:,35:]

        # split the pressure into real and complex
        _real = self.pressure.real.reshape(len(self.pressure), -1, 1)
        _complex = self.pressure.imag.reshape(len(self.pressure), -1, 1)

        # join real and complex
        self.pressure = np.concatenate([_real,_complex],axis=2).astype(
                np.float32).reshape(len(self.pressure), -1)

        # keep 10% for validation and 10% for test
        self.purpose = purpose
        l = self.__len__()
        tr= math.floor(l*0.8)
        v= math.floor(l*0.1)
        te = math.floor(l*0.1)

        if self.purpose == "train":
            self.rpm = self.rpm[:tr]
            self.pressure = self.pressure[:tr]
            self.amp = self.amp[:tr]
        elif self.purpose == "validate":
            self.rpm = self.rpm[tr:tr+v]
            self.pressure = self.pressure[tr:tr+v]
            self.amp = self.amp[tr:tr+v]
        elif self.purpose == "test":
            self.rpm = self.rpm[tr+v:tr+v+te]
            self.pressure = self.pressure[tr+v:tr+v+te]
            self.amp = self.amp[tr+v:tr+v+te]
        else:
            raise ValueError("purpose must be train, validate, or test")

    def __len__(self):
            return len(self.amp)

    def __getitem__(self, idx):
        return self.pressure[idx], self.amp[idx]





class PressToAmpV2(Dataset):
    '''
    prepare the data in the way more suitable for lstm.
    output of the data should be of the shape
    (seq_len, batch, input_size)
    '''
    
    def __init__(self, data_dir, file_names, purpose='train'):
        self.data = []
        if not isinstance(file_names, list):
            raise TypeError("file_name needs to be a list")

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
        #rpm = self.data[:, 0] / 10000.0
        #rpm = rpm.reshape(-1, 1)
     
        pressure = self.data[:, 35:]
        mmt = self.data[:, 34].astype('int32')
        print(mmt)
        # define features according to mmt
        self.features = []
        for i in range(len(mmt)):
            obs = pressure[i]
            m = mmt[i].item()
            tm = m*6
            obs = obs[35:35+tm]
            #obs = obs.reshape(6, m)
            #obs = np.transpose(obs)
            self.features.append(obs)
        
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


if __name__=="__main__":
    data_dir = "/scr1/li108/data/press_to_amp/"
    file_name = ["HL_CL1_withair_dec1_data.mat"]
    d = PressToAmpV2(data_dir, file_name)
    f, l = d[10]
    print(f.shape, f)
