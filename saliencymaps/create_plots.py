#!/usr/bin/env python3
import numpy as np
import core
import torch
import sys
import matplotlib.pyplot as plt
import pickle

PATH  = [   
    "../finetune/squeezenet/",
    "../finetune/densenet/",
    "../finetune/vgg/",
    "../finetune/inception/",
    "../finetune/resnet/",
]
NET_NAME = [
    "squeezenet",
    "densenet169",
    "vgg16",
    "inception_v3",
    "resnet50",
]
FOLDER      = "plots"

for name_id, path in enumerate(PATH):
    train_loss = []
    val_loss = []
    print("Loading states of model {}".format(NET_NAME[name_id]))
    for i in range(1, 101):
        STATE_PATH = path + NET_NAME[name_id] + "_{:03d}.nn".format(i)
        try: 
            state = torch.load(STATE_PATH)
        except Exception as e:  
            print("EXCEPTION TRIGGERED!")
            print(e)
            sys.exit()
        print("Read {:02d}% of data".format(i), end="\r")
        train_loss.append(state["train_loss"][-1])
        val_loss.append(state["val_loss"][-1])
    print("Generating plot for model {}".format(NET_NAME[name_id]), end="\n")
    epochs = len(train_loss)

    x = np.array(range(1,101)) - 0.5
    y = np.asarray(train_loss)
    Y = np.asarray(val_loss)
    X = np.array(range(1,101))

    plt.clf()
    plt.plot(x, y, "g-", label='training')
    plt.plot(X, Y, "b-", label='validation')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("losses during training")
    plt.savefig(FOLDER + "/plot_{}.png".format(NET_NAME[name_id]))
    loss_dict = {"train":train_loss, "val":val_loss}
    with open(FOLDER + "/loss_{}.pkl".format(NET_NAME[name_id]),"wb") as f:
        pickle.dump(loss_dict, f)

