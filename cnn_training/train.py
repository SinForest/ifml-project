import torch
import torch.nn as nn
from torch import Tensor as Ten
from torch.autograd import Variable as Var
from torch.utils.data import DataLoader

import os
import random
from hashlib import md5
from tqdm import tqdm, trange
import pickle
import matplotlib.pyplot as plt
from shutil import copyfile
from multiprocessing import Process

from model import SmallNetwork, DebugNetwork
from core import PosterSet

DATASET_PATH    = "../sets/set_splits.p"
POSTER_PATH     = "../posters/"
GENRE_DICT_PATH = "../sets/gen_d.p"

# dict for terminal colors
TERM = {'y'  : "\33[33m",
        'g'  : "\33[32m",
        'c'  : "\33[36m",
        'clr': "\33[m"}

def plot_losses(d, rnd):
    x, y = zip(*d["train"].items())
    X, Y = zip(*d["val"].items())
    plt.clf()
    plt.plot(x, y, "g-", label='training')
    plt.plot(X, Y, "b-", label='validation')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("losses during training")
    plt.savefig(rnd + "/plot_current.png")
    copyfile(rnd + "/plot_current.png", rnd + "/plot_after_{}.png".format(max(x)))
#[end] plot_losses(d)

# prepare data
gen_d = pickle.load(open(GENRE_DICT_PATH, 'rb'))
split = pickle.load(open(DATASET_PATH, 'rb'))

tr_set = PosterSet(POSTER_PATH, split, 'train', gen_d=gen_d, augment=True, debug=True)
tr_load = DataLoader(tr_set, batch_size=32, shuffle=True, num_workers=6)

va_set = PosterSet(POSTER_PATH, split, 'val', gen_d=gen_d, augment=False, debug=True)
va_load = DataLoader(va_set, batch_size=32, shuffle=False, num_workers=6)

# prepare model and training utillity
net  = DebugNetwork(tr_set[0][0].size()[1:], len(gen_d) // 2)
l_fn = torch.nn.BCEWithLogitsLoss(size_average=False)
opti = torch.optim.Adam(net.parameters()) #TODO: better optim

losses = {"train":{}, "val":{}}
va_delay  = 1 #DEBUG!
bg_proc = None

# generate random new folder to store all snapshots
rnd_token = None
while not rnd_token or os.path.isdir(rnd_token):
    rnd_token = "_" + md5(str(random.random()).encode()).hexdigest()[:10]
os.mkdir(rnd_token)

print("=== Starting Training with Token [{}] ===".format(rnd_token))

for epoch in trange(1,101):
    tr_err = 0
    net.train() # puts model into 'train' mode (some layers behave differently)
    for X, y in tr_load:         # iterate over training set (in mini-batches)
        X, y = Var(X), Var(y)    # wrap data into Variables (for back-prop)
        
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
        va_sum = [l_fn(net(Var(X, volatile=True)), Var(y)).data[0] for X,y in va_load] # val forward passes
        print([net(Var(X, volatile=True)).data for X,y in va_load])
        print([(X, y) for X,y in va_load][:2])
        losses["val"][epoch] = sum(va_sum) / len(va_set)
        tqdm.write("  -->{}Validating - loss: {:.5f}{}".format(TERM['g'], losses["val"][epoch], TERM['clr']))

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