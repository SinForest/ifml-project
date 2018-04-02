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

from extractors import *

DATASET_PATH    = "../sets/set_splits.p"
POSTER_PATH     = "../posters/"
GENRE_DICT_PATH = "../sets/gen_d.p"
CUDA_ON = True

class PosterSet(torch.utils.data.Dataset):
    def load1(self, fname):
        im = Image.open(self.path + fname + ".jpg").convert("RGB")
        return im

    def __init__(self, path, data, setname, debug=False):
        self.path = path
        if setname == 'all':
            data['all'] = {}
            data['all']['ids'] = data['train']['ids'] + data['val']['ids'] + data['test']['ids']
            data['all']['labels'] = data['train']['labels'] + data['val']['labels'] + data['test']['labels']
        if debug:
            data[setname]['ids'] = data[setname]['ids'][:65]
        self.X = [self.load1(iname) for iname in tqdm(data[setname]['ids'], desc='dataset')]
        self.y = data[setname]['labels']
        self.ids = data[setname]['ids']
        norm = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])
        scale = torchvision.transforms.Resize((224, 224))
        self.preproc = torchvision.transforms.Compose([scale, torchvision.transforms.ToTensor(), norm])

    def __getitem__(self, index):
        return self.preproc(self.X[index]), self.y[index]

    def __len__(self):
        return len(self.X)

def frac_hot(y, d, twoway=True):
    num = int(len(d) / 2) if twoway else len(d)
    a = np.zeros(num)
    [d[x] for x in y]
    a[y] = 1.0 / len(y)
    return a

p       = pickle.load(open(DATASET_PATH, 'rb'))
gen_d   = pickle.load(open(GENRE_DICT_PATH, 'rb'))
h5      = h5py.File("../sets/features_all.h5", 'a')
dataset = PosterSet(POSTER_PATH, p, 'all')
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, num_workers=1)
h5.create_dataset("labels", data=np.stack([frac_hot(y, gen_d) for __, y in tqdm(dataloader, desc="labels")]))
h5.create_dataset("ids", data=np.array([s.encode('utf8') for s in dataset.ids]))


for extr_name in ["alex_fc6", "alex_fc7", "vgg19bn_fc6", "vgg19bn_fc7", "res50_avg", "dense161_last"]:
    extr = eval("{}({})".format(extr_name, CUDA_ON))

    if CUDA_ON:
        feat = np.concatenate([extr(Variable(X).cuda()).cpu().data.numpy() for X, __ in tqdm(dataloader, desc=extr_name)])
    else:
        feat = np.concatenate([extr(Variable(X)).data.numpy() for X, __ in tqdm(dataloader, desc=extr_name)])
    
    h5.create_dataset(extr_name, data=feat)