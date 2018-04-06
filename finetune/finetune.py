#!/usr/bin/env python3
import numpy as np
import torch
import pickle
import torchvision
import torch.nn as nn
import torch.optim as optim
from cnn_finetune import make_model
from torch.autograd import Variable
from core import PosterSet
import h5py
import sys, os

#posters are all 182 width, 268 heigth

data_path = "../sets/set_splits.p"
poster_path = "../posters/"
dict_path = "../sets/gen_d.p"
sets_path = "../sets/"
load_model_name = "model_0008.nn"

num_epochs = 200
batch_size = 32
learning_rate = 0.5
momentum = 0.9
log_interval = 100
CUDA_ON = True

p = pickle.load(open(data_path, 'rb'))
gen_d = pickle.load(open(dict_path, 'rb'))

train_set = PosterSet(poster_path, p, 'train', gen_d = gen_d, tv_norm = True, augment = True, resize = (182, 268))
train_loader = torch.utils.data.DataLoader(train_set, batch_size = batch_size, shuffle = True, num_workers = 4)

val_set = PosterSet(poster_path, p, 'val', gen_d = gen_d, tv_norm = True, augment = True, resize = (182, 268))
val_loader = torch.utils.data.DataLoader(val_set, batch_size = batch_size, shuffle = False, num_workers = 4)



def make_classifier(in_features, num_classes):
    return nn.Sequential(
        nn.Linear(in_features, 2048),
        nn.RReLU(lower = 0, upper = 0.15, inplace = True),
        nn.Linear(2048, 512),
        nn.ReLU(inplace = True),
        nn.Linear(512, num_classes),
    )

model = make_model("vgg16", num_classes = (len(gen_d)//2), pretrained = True, input_size = (182, 268), classifier_factory = make_classifier)
optimizer = torch.optim.Adam(model.parameters())
#optimizer = optim.SGD(model.parameters(), lr = learning_rate, momentum = momentum)
criterion = nn.BCELoss(size_average = False)


epoch = 1
try:
    train_state = torch.load(load_model_name)
    model.load_state_dict(train_state['state_dict'])
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
        #print(data)
        #exit()
        if CUDA_ON:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = nn.functional.sigmoid(model(data))
        #print(output)
        #exit()
        loss = criterion(output, target)
        
        total_loss += loss.data[0]
        total_size += data.size(0)
        loss.backward()
        optimizer.step()
        

        if batch_id % log_interval == 0:
            print('Train Epoch: {} [{:> 5d}/{:> 5d} ({:> 2.0f}%)]\tCurrent loss: {:.6f}'.format(
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
        data, target = Variable(data, volatile = True), Variable(target) 
        output = nn.functional.sigmoid(model(data))
        val_loss += criterion(output, target).data[0]
    
    val_loss /= len(val_loader.dataset)
    print('\nTest set: Average loss: {:.4f}\n'.format(val_loss))
    return val_loss

loss_list = []
val_list = []
for epoch in range(epoch, num_epochs):
    loss_list.append(train(epoch))
    val_list.append(validate())
    state = {'state_dict':model.state_dict(), 'optim':optimizer.state_dict(), 'epoch':epoch, 'train_loss':loss_list, 'val_loss': val_list}
    filename = "model_{:04d}.nn".format(epoch)
    torch.save(state, filename)

