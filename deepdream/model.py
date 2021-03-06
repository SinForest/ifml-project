#!/bin/env python3
import torch
import torch.nn as nn
from torch import Tensor as Ten
from torch.autograd import Variable as Var

import numpy as np
from functools import reduce
import operator

class AbstractNetwork(nn.Module):
    def __init__(self):
        super().__init__()

    def _calc_fc_size(self):
        print("Initializing model with", self.inp_size) #DEBUG!
        tmp = Var(Ten(1, 3, *self.inp_size))
        return reduce(operator.mul, self.conv(tmp).size())
    
    def num_params(self):
        return sum([reduce(operator.mul, p.size()) for p in self.parameters()])
    
    def forward(self, X):
        X = self.conv(X)
        X = self.f_con(X.view(X.size(0), -1))
        return self.sigma(X)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                m.running_mean.zero_()
                m.running_var.fill_(1)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
            elif isinstance(m, nn.PReLU):
                m.weight.data.fill_(0.1)


class SmallerNetwork(AbstractNetwork):

    def __init__(self, inp_size, n_classes):
        super().__init__() # inp-size: 268x182
        self.inp_size = inp_size # tuple!
        self.conv1 = nn.Sequential(nn.BatchNorm2d(3),
                                   nn.Conv2d(3, 16, kernel_size=7, stride=1, padding=1),
                                   nn.ELU(),
                                   nn.BatchNorm2d(16),
                                   nn.MaxPool2d((3,3)),
                                   nn.Dropout2d(p=0.25))
        self.conv2 = nn.Sequential(nn.Conv2d(16, 16, kernel_size=5, stride=1, padding=1),
                                   nn.ELU(),
                                   nn.BatchNorm2d(16),
                                   nn.MaxPool2d((2,2)),
                                   nn.Dropout2d(p=0.25))
        self.conv3 = nn.Sequential(nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
                                   nn.ELU(),
                                   nn.BatchNorm2d(32),
                                   nn.MaxPool2d((2,2)),
                                   nn.Dropout2d(p=0.25))
        self.conv4 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                                   nn.ELU(),
                                   nn.BatchNorm2d(64),
                                   nn.MaxPool2d((2,2)))
        self.conv  = nn.Sequential(self.conv1, self.conv2, self.conv3, self.conv4)
        self.f_con = nn.Sequential(nn.Dropout(0.5),
                                   nn.Linear(self._calc_fc_size(), 1024),
                                   nn.ELU(),
                                   nn.Dropout(0.5),
                                   nn.Linear(1024, 1024),
                                   nn.ELU(),
                                   nn.Linear(1024, n_classes))
        self.sigma = nn.Sigmoid()
        self.init_weights()

class SmallNetwork(AbstractNetwork):

    def __init__(self, inp_size, n_classes):
        super(SmallNetwork, self).__init__() # inp-size: 268x182
        self.inp_size = inp_size # tuple!
        self.conv1 = nn.Sequential(nn.BatchNorm2d(3),
                                   nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
                                   nn.ELU(),
                                   nn.BatchNorm2d(16),
                                   nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
                                   nn.ELU(),
                                   nn.BatchNorm2d(16),
                                   nn.MaxPool2d((2,2)),
                                   nn.Dropout2d(p=0.25))
        self.conv2 = nn.Sequential(nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
                                   nn.ELU(),
                                   nn.BatchNorm2d(16),
                                   nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
                                   nn.ELU(),
                                   nn.BatchNorm2d(16),
                                   nn.MaxPool2d((2,2)),
                                   nn.Dropout2d(p=0.25))
        self.conv3 = nn.Sequential(nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
                                   nn.ELU(),
                                   nn.BatchNorm2d(32),
                                   nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
                                   nn.ELU(),
                                   nn.BatchNorm2d(32),
                                   nn.MaxPool2d((2,2)),
                                   nn.Dropout2d(p=0.25))
        self.conv4 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                                   nn.ELU(),
                                   nn.BatchNorm2d(64),
                                   nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                                   nn.ELU(),
                                   nn.BatchNorm2d(64),
                                   nn.MaxPool2d((2,2)))
        self.conv  = nn.Sequential(self.conv1, self.conv2, self.conv3, self.conv4)
        self.f_con = nn.Sequential(nn.Dropout(0.5),
                                   nn.Linear(self._calc_fc_size(), 1024),
                                   nn.ELU(),
                                   nn.Dropout(0.5),
                                   nn.Linear(1024, 1024),
                                   nn.ELU(),
                                   nn.Linear(1024, n_classes))
        self.sigma = nn.Sigmoid()
        self.init_weights()

class MidrangeNetwork(AbstractNetwork):

    def __init__(self, inp_size, n_classes):
        super().__init__() # inp-size: 268x182
        self.inp_size = inp_size # tuple!
        self.conv1 = nn.Sequential(nn.BatchNorm2d(3),
                                   nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
                                   nn.ELU(),
                                   nn.BatchNorm2d(32),
                                   nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
                                   nn.ELU(),
                                   nn.BatchNorm2d(32),
                                   nn.MaxPool2d((2,2)),
                                   nn.Dropout2d(p=0.25))
        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                                   nn.ELU(),
                                   nn.BatchNorm2d(64),
                                   nn.Conv2d(64,64, kernel_size=3, stride=1, padding=1),
                                   nn.ELU(),
                                   nn.BatchNorm2d(64),
                                   nn.MaxPool2d((2,2)),
                                   nn.Dropout2d(p=0.25))
        self.conv3 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                                   nn.ELU(),
                                   nn.BatchNorm2d(128),
                                   nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                                   nn.ELU(),
                                   nn.BatchNorm2d(128),
                                   nn.MaxPool2d((2,2)),
                                   nn.Dropout2d(p=0.25))
        self.conv4 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                                   nn.ELU(),
                                   nn.BatchNorm2d(256),
                                   nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
                                   nn.ELU(),
                                   nn.BatchNorm2d(128),
                                   nn.MaxPool2d((2,2)))
        self.conv  = nn.Sequential(self.conv1, self.conv2, self.conv3, self.conv4)
        self.f_con = nn.Sequential(nn.Dropout(0.5),
                                   nn.Linear(self._calc_fc_size(), 2048),
                                   nn.ELU(),
                                   nn.Dropout(0.5),
                                   nn.Linear(2048, 1024),
                                   nn.ELU(),
                                   nn.Linear(1024, n_classes))
        self.sigma = nn.Sigmoid()
        self.init_weights()

class DebugNetwork(AbstractNetwork):

    def __init__(self, inp_size, n_classes):
        super(DebugNetwork, self).__init__() # inp-size: 182x268
        self.inp_size = inp_size # tuple!
        self.conv = nn.Sequential(nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
                                   nn.PReLU(),
                                   nn.BatchNorm2d(16),
                                   nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
                                   nn.PReLU(),
                                   nn.BatchNorm2d(16),
                                   nn.MaxPool2d((32,32)))
        
        self.f_con = nn.Sequential(nn.Dropout(0.5),
                                   nn.Linear(self._calc_fc_size(), n_classes))
        self.sigma = nn.Sigmoid()
        self.init_weights()

if __name__ == "__main__":
    net = SmallNetwork((182,268), 23)
    print(net)
    print("Parameters: {:,}".format(net.num_params()))
