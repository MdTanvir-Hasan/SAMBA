from __future__ import annotations
import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from einops import rearrange, repeat, einsum

import csv

import torch
import random
import numpy as np
import torch.utils.data
import os
import logging
from datetime import datetime
import torch.nn.functional as F
import torch.nn as nn
import math
import os
import time
import copy

import sys

import h5py
import argparse
import configparser
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

X = pd.read_csv('combined_dataframe_IXIC.csv', index_col="Date", parse_dates=True)
    # basic preprocessing: get the name, the classification
    # Save the target variable as a column in dataframe for easier dropna()
name = X["Name"][0]
del X["Name"]
cols = X.columns
X["Target"] = (X["Price"].pct_change().shift(-1) > 0).astype(int)
X.dropna(inplace=True)
    # Fit the standard scaler using the training dataset


class MinMaxNorm01(object):
    """scale data to range [0, 1]"""
    def __init__(self):
        pass

    def fit(self, x):
        self.min = x.min()
        self.max = x.max()
        #print('Min:{}, Max:{}'.format(self.min, self.max))

    def transform(self, x):
        x = 1.0 * (x - self.min) / (self.max - self.min)
        return x

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)

    def inverse_transform(self, x):
        x = x * (self.max - self.min) + self.min
        return x

a=X.to_numpy()

#data1=train_data.to_numpy()

mmn = MinMaxNorm01()

data=a


dataset = mmn.fit_transform(data)

window=5
predict=1

ran=data.shape[0]
i=0
X=[]
Y=[]
while i+window<ran:

    X.append(torch.Tensor(dataset[i:i+window,1:]))
    Y.append(torch.Tensor(dataset[i+window:i+window+predict,0]))
    i+=1

XX=torch.stack(X,dim=0)
YY=torch.stack(Y,dim=0)
YY=YY[:,:,None]




test_len = int(0.15*XX.shape[0])
val_len = int(0.05*XX.shape[0])
train_len =  XX.shape[0]-test_len-val_len



X_test=torch.Tensor.float(XX[:test_len,:,:]).cuda()


Y_test=torch.Tensor.float(YY[:test_len,:,:]).cuda()

X_train=torch.Tensor.float(XX[test_len:test_len+train_len,:,:]).cuda()
Y_train=torch.Tensor.float(YY[test_len:test_len+train_len,:,:]).cuda()

X_val=torch.Tensor.float(XX[-val_len:,:,:]).cuda()
Y_val=torch.Tensor.float(YY[-val_len:,:,:]).cuda()



def data_loader(X, Y, batch_size, shuffle=True, drop_last=True):
    cuda = True if torch.cuda.is_available() else False
    TensorFloat = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    X, Y = TensorFloat(X), TensorFloat(Y)
    data = torch.utils.data.TensorDataset(X, Y)
    dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size,
                                             shuffle=shuffle, drop_last=drop_last)
    return dataloader

train_loader = data_loader(X_train, Y_train, 64, shuffle=False, drop_last=False)
val_loader = data_loader(X_val, Y_val, 64, shuffle=False, drop_last=False)
test_loader = data_loader(X_test, Y_test, 64, shuffle=False, drop_last=False)

def masked_mae_loss(scaler, mask_value):
    def loss(preds, labels):
        if scaler:
            preds = scaler.inverse_transform(preds)
            labels = scaler.inverse_transform(labels)
        mae = MAE_torch(pred=preds, true=labels, mask_value=mask_value)
        return mae
    return loss
