import torch
import torch.nn as nn
from torch import Tensor as Ten
from torch.autograd import Variable as Var
from torch.utils.data import DataLoader

import os, sys
import random
import numpy as np
from hashlib import md5
from tqdm import tqdm, trange
import pickle
from multiprocessing import Process
import itertools

from model import SmallNetwork, DebugNetwork, SmallerNetwork, MidrangeNetwork
from core import PosterSet, plot_losses

DATASET_PATH    = "../sets/set_splits.p"
POSTER_PATH     = "../posters/"
GENRE_DICT_PATH = "../sets/gen_d.p"

CUDA_ON = True
CROP_SIZE = (160, 160)

# dict for terminal colors
TERM = {'y'  : "\33[33m",
        'g'  : "\33[32m",
        'c'  : "\33[36m",
        'm'  : "\33[35m",
        'clr': "\33[m"}

class ReduceLROnPlateauWithLog(torch.optim.lr_scheduler.ReduceLROnPlateau):
    def __init__(self, *args, **kwargs):
        super(ReduceLROnPlateauWithLog, self).__init__(*args, **kwargs)
        self.logger = []

    def _reduce_lr(self, epoch):
        self.logger.append(epoch)
        return super(ReduceLROnPlateauWithLog, self)._reduce_lr(epoch)

def hook(m, i, o):
    print(TERM['c'])
    print("####  " + str(m) + "  ####")
    print(TERM['m'])
    print("output:", o)
    print(TERM['g'])
    print("input: ", i)
    print(TERM['y'])
    print("mean: ", m.running_mean)
    print("var: ", m.running_var)
    print("bias:", m.bias.data)
    print("weight:", m.weight.data)
    print(TERM['clr'])

# prepare data
gen_d = pickle.load(open(GENRE_DICT_PATH, 'rb'))
split = pickle.load(open(DATASET_PATH, 'rb'))

tr_set = PosterSet(POSTER_PATH, split, 'train', gen_d=gen_d, augment=True, resize=None, rnd_crop=CROP_SIZE)#, debug=True)
tr_load = DataLoader(tr_set, batch_size=128, shuffle=True, num_workers=3, drop_last=True)

va_set = PosterSet(POSTER_PATH, split, 'val',  gen_d=gen_d, augment=False, resize=None, ten_crop=CROP_SIZE)#, debug=True)
va_load = DataLoader(va_set, batch_size=64, shuffle=False, num_workers=3, drop_last=True)

# prepare model and training utillity
net  = MidrangeNetwork(tr_set[0][0].size()[1:], len(gen_d) // 2)
l_fn = torch.nn.BCELoss(size_average=False)
opti = torch.optim.Adam(net.parameters())
sdlr = ReduceLROnPlateauWithLog(opti, 'min', factor=0.8, patience=12, cooldown=8)
if CUDA_ON:
    net.cuda()

if len(sys.argv) > 1:
    load_state = torch.load(sys.argv[1])
    net.load_state_dict(load_state['state_dict'])
    opti.load_state_dict(load_state['opti'])
    start_epoch = load_state['epoch']
    losses = load_state['losses']
    print("loaded mode from {}".format(sys.argv[1]))
else:
    load_state = None
    start_epoch = 1
    losses = {"train":{}, "val":{}}

"""
#DEBUG!!:
print("##################### IM HERE! #####################")
for m in net.modules():
    if isinstance(m, nn.BatchNorm2d):
        m.register_forward_hook(hook)
"""
va_delay  = 1
bg_proc = None

# generate random new folder to store all snapshots
rnd_token = None
while not rnd_token or os.path.isdir(rnd_token):
    rnd_token = "_" + md5(str(random.random()).encode()).hexdigest()[:10]
os.mkdir(rnd_token)

print("=== Starting Training with Token [{}] ===".format(rnd_token))

for epoch in tqdm(itertools.count(start_epoch), desc="epochs:"):
    tr_err = 0
    net.train() # puts model into 'train' mode (some layers behave differently)
    for X, y in tqdm(tr_load, desc="train:"): # iterate over training set (in mini-batches)
        X, y = (Var(X).cuda(), Var(y).cuda()) if CUDA_ON else (Var(X), Var(y))
        # wrap data into Variables (for back-prop) and maybe move to GPU
        
        #TRAINING-STEP:
        opti.zero_grad()         # "reset" the optimizers gradients
        pred = net(X)            # forward pass
        loss = l_fn(pred, y)     # calculate loss
        loss.backward()          # backward pass
        opti.step()              # update weights
        
        tr_err += loss.data[0]   # log training loss

    #[end] X,y in dataloader
    
    losses["train"][epoch] = tr_err / len(tr_set) # average loss over epoch and save
    tqdm.write("Epoch {} - loss: {:.5f}".format(epoch, losses["train"][epoch]))

    if epoch % va_delay == 0:  # validate and save every <va_delay>th epoch
        net.eval() # puts model into 'eval' mode (some layers behave differently)
        va_sum = []
        for X ,y in tqdm(va_load, desc="valid:"):
            bs, ncrops, c, h, w = X.size()
            if CUDA_ON:
                X, y = Var(X, volatile=True).cuda(), Var(y).cuda()
            else:
                X, y = Var(X, volatile=True), Var(y)
            
            result = net(X.view(-1, c, h, w))      # put crops into model as single instances
            result = result.view(bs, ncrops, -1).mean(1) # calc average score over all crops of same image
            va_sum.append(l_fn(result, y).data[0])
        
        losses["val"][epoch] = sum(va_sum) / len(va_set)
        tqdm.write("  -->{}Validating - loss: {:.5f}{}".format(TERM['g'], losses["val"][epoch], TERM['clr']))
        sdlr.step(losses["val"][epoch])
        losses["lr"] = sdlr.logger
        # plot loss in BG:
        if bg_proc:
            bg_proc.join(2)
            bg_proc.terminate()
        bg_proc = Process(target=plot_losses, args=(losses,rnd_token))
        bg_proc.start()

        # save model do disk
        state = {'state_dict':net.state_dict(), 'opti':opti.state_dict(), 'epoch':epoch, 'losses':losses}
        f_name = rnd_token + "/model_{:04d}_{:.0f}.nn".format(epoch, losses["val"][epoch]*10000)
        torch.save(state, f_name)
        tqdm.write("  -->{}saved model to {}{}".format(TERM['y'], f_name, TERM['clr']))
        
    #[end] epoch % va_delay == 0
#[end] epoch in range(1,101)
