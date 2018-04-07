import torch
import torch.nn as nn
from torch import Tensor as Ten
from torch.autograd import Variable as Var
from torch.utils.data import DataLoader

import os
import random
import numpy as np
from hashlib import md5
from tqdm import tqdm, trange
import pickle
from multiprocessing import Process
import itertools

from model import SmallNetwork, DebugNetwork
from core import PosterSet, plot_losses

DATASET_PATH    = "../sets/set_splits.p"
POSTER_PATH     = "../posters/"
GENRE_DICT_PATH = "../sets/gen_d.p"

CUDA_ON = True

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

tr_set = PosterSet(POSTER_PATH, split, 'train', gen_d=gen_d, augment=True, resize=None)# debug=True)
tr_load = DataLoader(tr_set, batch_size=128, shuffle=True, num_workers=3, drop_last=True)

va_set = PosterSet(POSTER_PATH, split, 'val', gen_d=gen_d, augment=False, resize=None)# debug=True)
va_load = DataLoader(va_set, batch_size=128, shuffle=False, num_workers=3, drop_last=True)

# prepare model and training utillity
net  = SmallNetwork(tr_set[0][0].size()[1:], len(gen_d) // 2)
l_fn = torch.nn.BCELoss(size_average=False)
opti = torch.optim.Adam(net.parameters())
sdlr = ReduceLROnPlateauWithLog(opti, 'min', factor=0.5)
if CUDA_ON:
    net.cuda()

"""
#DEBUG!!:
print("##################### IM HERE! #####################")
for m in net.modules():
    if isinstance(m, nn.BatchNorm2d):
        m.register_forward_hook(hook)
"""
losses = {"train":{}, "val":{}}
va_delay  = 1
bg_proc = None

# generate random new folder to store all snapshots
rnd_token = None
while not rnd_token or os.path.isdir(rnd_token):
    rnd_token = "_" + md5(str(random.random()).encode()).hexdigest()[:10]
os.mkdir(rnd_token)

print("=== Starting Training with Token [{}] ===".format(rnd_token))

for epoch in tqdm(itertools.count(1), desc="epochs:"):
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
        if CUDA_ON:
            va_sum = [l_fn(net(Var(X, volatile=True).cuda()), Var(y).cuda()).data[0] for X,y in tqdm(va_load, desc="valid:")] # val forward passes (GPU)
            # print([net(Var(X, volatile=True).cuda()).data for X,y in va_load])
        else:
            va_sum = [l_fn(net(Var(X, volatile=True)), Var(y)).data[0] for X,y in tqdm(va_load, desc="valid:")] # val forward passes (CPU)
            # print([net(Var(X, volatile=True)).data for X,y in va_load])
        
        #print([(X, y) for X,y in va_load][:2])
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
