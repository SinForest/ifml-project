#!/usr/bin/env python3
import numpy as np
import torchvision
import torch.nn as nn
import torch
from PIL import Image
from torch.autograd import Variable as Var
from torch import Tensor as Ten
from functools import reduce
from core import PosterSet
from reshapeLayer import ReshapeLayer
import pickle
import operator
import os, sys


MODEL_PATH  = "../finetune/squeezenet/squeezenet_200.nn"
POSTER_PATH = "../posters/"
DATA_PATH   = "../sets/set_splits.p"
DICT_PATH   = "../sets/gen_d.p"
SAVE_PATH   = "saliencymaps/squeeze/squeeze"
CUDA_ON     = True
DEBUG_MODE  = True
SETS = [
    "train",
    "train",
    "val",
    "val",
    "test",
    "test",
]
ID_LIST = [
    "tt1213218",
    "tt0036332",
    "tt1258911",
    "tt0052520",
    "tt0098967",
    "tt0037075",
]

input_size  = (268, 182)
num_classes = 23
to_ten      = torchvision.transforms.ToTensor()
to_pil      = torchvision.transforms.ToPILImage()
scale       = torchvision.transforms.Resize(input_size) 
#tar_class   = 1
learn_r     = 0.001


p = pickle.load(open(DATA_PATH, 'rb'))
gen_d = pickle.load(open(DICT_PATH, 'rb'))

gradients = None

def hook_function(module, grad_in, grad_out):
    global gradients
    gradients = grad_in[0].cpu()

def relu_hook_function(module, grad_in, grad_out):
    if isinstance(module, nn.ReLU):
        return (torch.clamp(grad_in[0], min=0.0),)

model = torchvision.models.squeezenet1_1(pretrained=True)
model.num_classes = num_classes

for pos, module in model.features._modules.items():
    if isinstance(module, nn.ReLU):
        module.register_backward_hook(relu_hook_function)

first_layer = list(model.features._modules.items())[0][1]
first_layer.register_backward_hook(hook_function)

#replace classifier with one fitting number of classes
final_conv = nn.Conv2d(512, num_classes, kernel_size = 1)
classifier = nn.Sequential(
    nn.Dropout(p=0.5),
    final_conv,
    nn.ReLU(inplace=True),
    nn.AdaptiveAvgPool2d(1), #makes input size arbitrary
)
model.classifier = classifier


try:
    model_state = torch.load(MODEL_PATH)
    model.load_state_dict(model_state['state_dict'])

except Exception as e:  
    print("EXCEPTION TRIGGERED!")
    print(e)
    sys.exit()

model.cuda()
model.eval()

for set_name, id_name in zip(SETS, ID_LIST):
    try:
        img = Image.open(POSTER_PATH + id_name + ".jpg").convert("RGB")
        img.save(SAVE_PATH + id_name, "JPEG")
    except Exception as e:
        print(e)
        sys.exit()
    img_ten = to_ten(img).unsqueeze(0)
    var = Var(img_ten, requires_grad=True)
    var = var.cuda()

    labels = p[set_name]['labels'][p[set_name]['ids'].index(id_name)]
    for i in range(len(labels)):
        tar_class = gen_d[labels[i]]
        tar_ten = torch.FloatTensor(1, num_classes).zero_()
        tar_ten[0][tar_class] = 1
        tar = Var(tar_ten)
        tar = tar.cuda()
        output = model(var)
        model.zero_grad()
        output.backward(gradient=tar)
        pil_img = to_pil(gradients.data.cpu().squeeze(0))
        pil_img.save(SAVE_PATH + "_{}_{}_{}".format(set_name, id_name, labels[i]), "JPEG")




    
