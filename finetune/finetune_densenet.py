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
from core import accuracy
from functools import reduce
from reshapeLayer import ReshapeLayer
import operator
import sys, os

DATA_PATH   = "../sets/set_splits.p"
POSTER_PATH = "../posters/"
DICT_PATH   = "../sets/gen_d.p"
SETS_PATH   = "../sets/"
MODEL_PATH  = "./densenet/densenet169_100.nn"
CUDA_ON     = True
DEBUG_MODE  = False

num_epochs  = 101
batch_s     = 128
log_percent = 0.25
s_factor    = 0.5
learn_r     = 0.0001
input_size  = (268, 182) #posters are all 182 width, 268 heigth


p = pickle.load(open(DATA_PATH, 'rb'))
gen_d = pickle.load(open(DICT_PATH, 'rb'))

train_set = PosterSet(POSTER_PATH, p, 'train', gen_d=gen_d, tv_norm=True, augment=True, resize=input_size, debug=DEBUG_MODE)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_s, shuffle=True, num_workers=4)
log_interval = np.ceil((len(train_loader.dataset) * log_percent) / batch_s)

val_set = PosterSet(POSTER_PATH, p, 'val', gen_d=gen_d, tv_norm=True, augment=True, resize=input_size, debug=DEBUG_MODE)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_s, shuffle=False, num_workers=4)

test_set = PosterSet(POSTER_PATH, p, 'test', gen_d=gen_d, tv_norm=True, augment=False, resize=input_size, debug=DEBUG_MODE)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_s, shuffle=False, num_workers=4)

num_classes = (len(gen_d)//2)

model = torchvision.models.densenet169(pretrained=True)
modules = list(model.children())[:-1] #delete classification layer
model = nn.Sequential(*modules)

#disable grad requirement to keep pretrained weights
for param in model.parameters():
    param.requires_grad = False

calc_fc_tensor = Var(Ten(1, 3, *input_size)) #tensor for calculation of number of connections to the first fc layer
fc_size = reduce(operator.mul, model(calc_fc_tensor).size()) #single forward pass through the network
print(fc_size)
#custom classifier
classifier = nn.Sequential(
                ReshapeLayer(),
                nn.Linear(fc_size, 2048),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                #nn.ReLU(inplace=True),
                nn.Linear(2048, num_classes),
                #nn.ReLU(inplace=True),
                #nn.Linear(4096, num_classes),
)

model = nn.Sequential(*modules, classifier)

optimizer = torch.optim.Adam(classifier.parameters(), lr=learn_r)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=s_factor, patience=5, verbose=True)
#optimizer = optim.SGD(model.parameters(), lr=learn_r, momentum=momentum)
criterion = nn.BCELoss(size_average=False)

epoch = 1
try:
    train_state = torch.load(MODEL_PATH)
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

    print('Train Epoch: {} DenseNet average loss: {:.6f}'.format(
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
    
    print('\nTest set: DenseNet average loss: {:.4f}\n'.format(val_loss))
    
    return val_loss

def test():
    model.eval()
    test_loss = 0
    total_length = 0

    for data, target in test_loader:

        if CUDA_ON:
            data, target = data.cuda(), target.cuda()
        
        data, target = Var(data, volatile = True), Var(target)

        output = nn.functional.sigmoid(model(data))

        for i in range(output.size(0)):
            try:
                if target.data[i].sum() == 0: continue
                if target.data[i].sum() >= 12: continue    
                test_loss += accuracy(output.data[i], target.data[i])
                total_length += 1
            except Exception as e:
                print(e)
                print(target.data[i])
                sys.exit()
    return test_loss / total_length

loss_list = []
val_list = []
for epoch in range(epoch, num_epochs + 1):
    print(test())
    '''
    loss_list.append(train(epoch))
    val_loss = validate()
    scheduler.step(val_loss)
    val_list.append(val_loss)
    
    state = {'state_dict':model.state_dict(), 'optim':optimizer.state_dict(), 'epoch':epoch, 'train_loss':loss_list, 'val_loss': val_list}
    filename = "./densenet/densenet169_{:03d}.nn".format(epoch)
    torch.save(state, filename)'''