import torch
import sys
import pickle
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.autograd import Variable as Var

from model import SmallNetwork, SmallerNetwork, MidrangeNetwork
from core import PosterSet, accuracy

INP_SIZE = (268, 182)
CROP_SIZE = (160, 160)
SNAP_PATH = "../"
DATASET_PATH    = "../sets/set_splits.p"
POSTER_PATH     = "../posters/"
GENRE_DICT_PATH = "../sets/gen_d.p"
CUDA_ON = True

gen_d = pickle.load(open(GENRE_DICT_PATH, 'rb'))
split = pickle.load(open(DATASET_PATH, 'rb'))

def test_rnd_network():
    te_set = PosterSet(POSTER_PATH, split, 'test',  gen_d=gen_d, augment=False, resize=None, ten_crop=None, debug=True)
    te_load = DataLoader(te_set, batch_size=64, shuffle=False, num_workers=3, drop_last=True)
    model = SmallNetwork(INP_SIZE, 23)
    if CUDA_ON: model.cuda()
    model.eval()
    loss = 0
    skipped = 0

    for X, y in tqdm(te_load, desc='rnd'):
        X, y = Var(X, volatile=True), Var(y)
        if CUDA_ON:
            X, y = X.cuda(), y.cuda()
    
        out = model(X)

        for i in range(out.size(0)):
            try:
                loss += accuracy(out.data[i], y.data[i])
            except:
                skipped += 1
    
    return loss / (len(te_set) - skipped)



def test_2nd_snapshot():
    te_set = PosterSet(POSTER_PATH, split, 'test',  gen_d=gen_d, augment=False, resize=None, ten_crop=None, debug=True)
    te_load = DataLoader(te_set, batch_size=64, shuffle=False, num_workers=3, drop_last=True)
    model = SmallNetwork(INP_SIZE, 23)
    state = torch.load(SNAP_PATH + "snap2nd.nn")
    model.load_state_dict(state['state_dict'])
    if CUDA_ON: model.cuda()
    model.eval()
    loss = 0
    skipped = 0

    for X, y in tqdm(te_load, desc='2nd'):
        X, y = Var(X, volatile=True), Var(y)
        if CUDA_ON:
            X, y = X.cuda(), y.cuda()
    
        out = model(X)

        for i in range(out.size(0)):
            try:
                loss += accuracy(out.data[i], y.data[i])
            except:
                skipped += 1
    
    return loss / (len(te_set) - skipped)

def test_3rd_snapshot():
    te_set = PosterSet(POSTER_PATH, split, 'test',  gen_d=gen_d, augment=False, resize=None, ten_crop=None, debug=True)
    te_load = DataLoader(te_set, batch_size=64, shuffle=False, num_workers=3, drop_last=True)
    model = SmallerNetwork(INP_SIZE, 23)
    state = torch.load(SNAP_PATH + "snap3rd.nn")
    model.load_state_dict(state['state_dict'])
    if CUDA_ON: model.cuda()
    model.eval()
    loss = 0
    skipped = 0

    for X, y in tqdm(te_load, desc='3rd'):
        X, y = Var(X, volatile=True), Var(y)
        if CUDA_ON:
            X, y = X.cuda(), y.cuda()
    
        out = model(X)

        for i in range(out.size(0)):
            try:
                loss += accuracy(out.data[i], y.data[i])
            except:
                skipped += 1
    
    return loss / (len(te_set) - skipped)

def test_4th_snapshot():
    te_set = PosterSet(POSTER_PATH, split, 'test',  gen_d=gen_d, augment=False, resize=None, ten_crop=CROP_SIZE, debug=True)
    te_load = DataLoader(te_set, batch_size=64, shuffle=False, num_workers=3, drop_last=True)
    model = SmallNetwork(CROP_SIZE, 23)
    state = torch.load(SNAP_PATH + "snap4th.nn")
    model.load_state_dict(state['state_dict'])
    if CUDA_ON: model.cuda()
    model.eval()
    loss = 0
    skipped = 0

    for X, y in tqdm(te_load, desc='4th'):
        X, y = Var(X, volatile=True), Var(y)
        
        bs, ncrops, c, h, w = X.size()

        if CUDA_ON:
            X, y = X.cuda(), y.cuda()
    
        out = model(X.view(-1, c, h, w))
        out = out.view(bs, ncrops, -1).mean(1)

        for i in range(out.size(0)):
            try:
                loss += accuracy(out.data[i], y.data[i])
            except:
                skipped += 1
    
    return loss / (len(te_set) - skipped)

def test_5th_snapshot():
    te_set = PosterSet(POSTER_PATH, split, 'test',  gen_d=gen_d, augment=False, resize=None, ten_crop=CROP_SIZE, debug=True)
    te_load = DataLoader(te_set, batch_size=64, shuffle=False, num_workers=3, drop_last=True)
    model = MidrangeNetwork(CROP_SIZE, 23)
    state = torch.load(SNAP_PATH + "snap5th.nn")
    model.load_state_dict(state['state_dict'])
    if CUDA_ON: model.cuda()
    model.eval()
    loss = 0
    skipped = 0

    for X, y in tqdm(te_load, desc='5th'):
        X, y = Var(X, volatile=True), Var(y)
        
        bs, ncrops, c, h, w = X.size()

        if CUDA_ON:
            X, y = X.cuda(), y.cuda()
    
        out = model(X.view(-1, c, h, w))
        out = out.view(bs, ncrops, -1).mean(1)

        for i in range(out.size(0)):
            try:
                loss += accuracy(out.data[i], y.data[i])
            except:
                skipped += 1
    
    return loss / (len(te_set) - skipped)

if __name__ == '__main__':
    print(test_2nd_snapshot())
    print(test_3rd_snapshot())
    print(test_4th_snapshot())
    print(test_5th_snapshot())
    print(test_rnd_network())