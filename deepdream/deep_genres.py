#!/usr/bin/env python3
import numpy as np
import torchvision
import torch.nn as nn
import torch
import torch.optim as optim
from PIL import Image
from torch.autograd import Variable as Var
from torch import Tensor as Ten
from functools import reduce
import model
import operator
import os, sys

#wrongly implemented, sadly it didnt work for us in time

MODEL_PATH  = "model_0260_67929.nn"
IMAGE_PATH  = "bladerunner2.jpeg"
CUDA_ON     = True

input_size  = (160, 160)
num_classes = 23
to_ten      = torchvision.transforms.ToTensor()
to_pil      = torchvision.transforms.ToPILImage()
scale       = torchvision.transforms.Resize(input_size) 
tar_class   = 15
learn_r     = 200

img = Image.open(IMAGE_PATH).convert("RGB")
img = scale(img)
img_ten = to_ten(img).unsqueeze(0)
var = Var(img_ten, requires_grad=True)

tar_ten = torch.FloatTensor(1, num_classes).zero_()
tar_ten[0][(tar_class-1)] = 1
tar = Var(tar_ten)

model = model.SmallNetwork(input_size, num_classes)
model.eval()

#optimizer = optim.SGD([var], lr=learn_r)
optimizer = torch.optim.Adam([var], lr=learn_r)
criterion = nn.BCELoss(size_average=False)

try:
    model_state = torch.load(MODEL_PATH)
    #for i in range(len(model_state['state_dict'])):
    #    print(list(model_state['state_dict'])[i])
    model.load_state_dict(model_state['state_dict'])

except Exception as e:  
    print("EXCEPTION TRIGGERED!")
    print(e)
    sys.exit()

if CUDA_ON:
    model.cuda()
    var, tar = var.cuda(), tar.cuda()

for i in range(1, 251):
    optimizer.zero_grad()
    
    
    output = model(var)
    loss = criterion(output, tar)


    loss.backward()
    optimizer.step()
    
    new_img = to_pil(var.data.cpu().squeeze(0))
    
    if i % 20 == 0:
        new_img.save("pics/bladedream_{:03d}.jpg".format(i))

    
