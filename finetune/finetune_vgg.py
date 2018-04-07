#!/usr/bin/env python3
import numpy as np
import torch
import pickle
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable as Var
from torch import Tensor as Ten
from core import PosterSet
from functools import reduce
from reshapeLayer import ReshapeLayer
import operator
import sys, os

data_path = "../sets/set_splits.p"
poster_path = "../posters/"
dict_path = "../sets/gen_d.p"
sets_path = "../sets/"
if(len(sys.argv) > 2):
    load_model_name = sys.argv[1]
else: 
    load_model_name = "vgg16_010.nn"

num_epochs = 75
batch_size = 128
learning_rate = 0.001
momentum = 0.9
log_percent = 10
s_factor = 0.5
input_size = (268, 182) #posters are all 182 width, 268 heigth
CUDA_ON = True

p = pickle.load(open(data_path, 'rb'))
gen_d = pickle.load(open(dict_path, 'rb'))

train_set = PosterSet(poster_path, p, 'train', gen_d=gen_d, tv_norm=True, augment=True, resize=input_size)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
log_interval = (len(train_loader.dataset)/log_percent)//batch_size

val_set = PosterSet(poster_path, p, 'val', gen_d=gen_d, tv_norm=True, augment=True, resize=input_size)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4)

num_classes = (len(gen_d)//2)

#model = make_model("resnet50", num_classes =(len(gen_d)//2), pretrained=True, input_size=(182, 268), classifier_factory=make_classifier)
#model = make_model("vgg16", num_classes=(len(gen_d)//2), pretrained=True, input_size=(182, 268), classifier_factory=make_classifier)

model = torchvision.models.vgg16(pretrained=True)
modules = list(model.children())[:-1] #delete classification layer
model = nn.Sequential(*modules)

#disable grad requirement to keep pretrained weights
for param in model.parameters():
    param.requires_grad = False

calc_fc_tensor = Var(Ten(1, 3, *input_size)) #tensor for calculation of number of connections to the first fc layer
fc_size = reduce(operator.mul, model(calc_fc_tensor).size()) #single forward pass through the network

#custom classifier
classifier = nn.Sequential(
                ReshapeLayer(),
                nn.Linear(fc_size, 2048),
                nn.RReLU(lower=0, upper=0.7, inplace=True),
                #nn.ReLU(inplace=True),
                nn.Linear(2048, 2048),
                nn.ReLU(inplace=True),
                nn.Linear(2048, num_classes),
)

model = nn.Sequential(*modules, classifier)

optimizer = torch.optim.Adam(classifier.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=s_factor, patience=5, verbose=True)
#optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
criterion = nn.BCELoss(size_average=False)

epoch = 1
try:
    train_state = torch.load(load_model_name)
    model.load_state_dict(train_state['state_dict'])
    optimizer.load_state_dict(train_state['optim'])
    epoch = train_state['epoch'] + 1
except Exception as e:  
    print(e)

if CUDA_ON:
    model.cuda()

def train(epoch):
    total_loss = 0
    total_size = 0
    model.train()
    for batch_id, (data, target) in enumerate(train_loader):

        if CUDA_ON:
            data, target = data.cuda(), target.cuda()
        
        data, target = Var(data), Var(target)
        optimizer.zero_grad()

        output = nn.functional.sigmoid(model(data))

        loss = criterion(output, target)
        
        total_loss += loss.data[0]
        total_size += data.size(0)

        loss.backward()
        optimizer.step()
        
        if batch_id % log_interval == 0:
            print('Train Epoch: {} [{:>5d}/{:> 5d} ({:>2.0f}%)]\tCurrent loss: {:.6f}'.format(
            epoch, total_size, len(train_loader.dataset), 100. * batch_id / len(train_loader), loss.data[0]/data.size(0)))

    print('Train Epoch: {} Average loss: {:.6f}'.format(
            epoch, total_loss / total_size))
    
    return (total_loss/total_size)


def validate():
    model.eval()
    val_loss = 0
    for data, target in val_loader:
        
        if CUDA_ON:
            data, target = data.cuda(), target.cuda()
        
        data, target = Var(data, volatile = True), Var(target) 
        
        output = nn.functional.sigmoid(model(data))
        
        val_loss += criterion(output, target).data[0]
    
    val_loss /= len(val_loader.dataset)
    
    print('\nTest set: Average loss: {:.4f}\n'.format(val_loss))
    
    return val_loss

loss_list = []
val_list = []
for epoch in range(epoch, num_epochs):
    
    loss_list.append(train(epoch))
    val_loss = validate()
    scheduler.step(val_loss)
    val_list.append(val_loss)
    
    state = {'state_dict':model.state_dict(), 'optim':optimizer.state_dict(), 'epoch':epoch, 'train_loss':loss_list, 'val_loss': val_list}
    filename = "vgg16_{:03d}.nn".format(epoch)
    torch.save(state, filename)

