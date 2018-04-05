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

num_epochs = 200
batch_size = 32
learning_rate = 0.05
momentum = 0.9
log_interval = 10
CUDA_ON = True

p = pickle.load(open(data_path, 'rb'))
gen_d = pickle.load(open(dict_path, 'rb'))

train_set = PosterSet(poster_path, p, 'train', gen_d = gen_d, normalize = True, debug = True)
train_loader = torch.utils.data.DataLoader(train_set, batch_size = batch_size, shuffle = False, num_workers = 6)

val_set = PosterSet(poster_path, p, 'val', gen_d = gen_d, normalize = True, debug = True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size = batch_size, shuffle = False, num_workers = 6)



cuda_device = int(sys.argv[1]) if len(sys.argv) > 1 else 0
if cuda_device == -1:
    cuda_device = 0
    CUDA_ON = False

model = make_model("vgg16", num_classes = (len(gen_d)//2), pretrained = True, input_size = (182, 268))

if CUDA_ON:
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr = learning_rate, momentum = momentum)
criterion = nn.BCEWithLogitsLoss(size_average = False)

def train(epoch):
    total_loss = 0
    total_size = 0
    model.train()
    for batch_id, (data, target) in enumerate(train_loader):
        print(data)
        exit()
        if CUDA_ON:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        total_loss += loss.data[0]
        total_size += data.size(0)
        loss.backward()
        optimizer.step()

        if batch_id % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tAverage loss: {:.6f}'.format(
                    epoch, batch_id * len(data), len(train_loader.dataset), 100. * batch_id / len(train_loader), total_loss / total_size))


def validate():
    model.eval()
    val_loss = 0
    for data, target in val_loader:
        if CUDA_ON:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile = True), Variable(target) 
        output = model(data)
        val_loss += criterion(output, target).data[0]
    
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))

for epoch in range(1, num_epochs):
    train(epoch)
    validate()

