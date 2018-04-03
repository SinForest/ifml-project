import pickle
import torchvision
import torch.nn as nn
import torch
import numpy as np
from PIL import Image
from scipy.misc import imresize
from torch.autograd import Variable
from tqdm import tqdm
import h5py
import sys, os

from extractors import *
from core import PosterSet

DATASET_PATH    = "../sets/set_splits.p"
POSTER_PATH     = "../posters/"
GENRE_DICT_PATH = "../sets/gen_d.p"
SETS_PATH = "../sets/"
CUDA_ON = True

p       = pickle.load(open(DATASET_PATH, 'rb'))
gen_d   = pickle.load(open(GENRE_DICT_PATH, 'rb'))
h5      = h5py.File("../sets/features_all.h5", 'a')
dataset = PosterSet(POSTER_PATH, p, 'all', gen_d=gen_d, normalize=True)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, num_workers=1)

h5.create_dataset("labels", data=np.stack([y for __, y in tqdm(dataset)]))
h5.create_dataset("ids", data=np.array([s.encode('utf8') for s in dataset.ids]))

cuda_device = int(sys.argv[1]) if len(sys.argv) > 1 else 0
if cuda_device == -1:
    cuda_device = 0
    CUDA_ON = False

with torch.cuda.device(cuda_device):

    for extr_name in ["alex_fc6", "alex_fc7", "vgg19bn_fc6", "vgg19bn_fc7", "res50_avg", "dense161_last"]:
        extr = eval("{}({})".format(extr_name, CUDA_ON))

        if CUDA_ON:
            feat = np.concatenate([extr(Variable(X).cuda()).cpu().data.numpy() for X, __ in tqdm(dataloader, desc=extr_name)])
        else:
            feat = np.concatenate([extr(Variable(X)).data.numpy() for X, __ in tqdm(dataloader, desc=extr_name)])
        
        h5.create_dataset(extr_name, data=feat)

d = dict(zip(h5["ids"], range(len(h5["ids"]))))
idx_all = []
for s in ["train", "val", "test"]:
    f = open(SETS_PATH + s + ".csv", 'r')
    idx = np.array([d[line.split(",")[0].encode('utf-8')] for line in tqdm(f)])
    idx_all.extend(idx)
    h5.create_dataset(s + "_idx", data=idx)

assert(list(sorted(idx_all)) == list(range(len(idx_all))))