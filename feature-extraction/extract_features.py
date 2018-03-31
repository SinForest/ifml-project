import pickle
import torchvision.models as tvm
import torch.nn as nn
import torch
import numpy as np
from imageio import imread
from scipy.misc import imresize
from torch.autograd import Variable
from tqdm import tqdm
import h5py

DATASET_PATH = "../dataset.p"
POSTER_PATH  = "../posters/"
CUDA_ON = True

class PosterSet(torch.utils.data.Dataset):
    def load_one_sample(self, fname):
        try:
            im = imread(self.path + fname + ".jpg")
            if len(im.shape) == 2:
                im = np.stack([im]*3, axis=2)
            im = imresize(im, (227, 227, 3)).astype(np.float32)
            im = np.transpose(im, (2,0,1))
            im[0, :, :] -= 123.68
            im[1, :, :] -= 116.779
            im[2, :, :] -= 103.939
            return im
        except Exception as e:
            print("Error on: " + fname)
            print("Shape: " + str(im.shape))
            raise e


    def __init__(self, path, data):
        self.path = path
        self.X = [self.load_one_sample(iname) for iname in tqdm(data['ids'], desc='dataset')]
        self.y = data['labels']

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return len(self.X)

model = tvm.alexnet(pretrained=True)
nc = nn.Sequential(*list(model.classifier.children())[:3])
model.classifier = nc
model.eval()
if CUDA_ON: model.cuda()

h5 = h5py.File("./sets/features.h5", 'a')
p = pickle.load(open(DATASET_PATH, 'rb'))

for s in ["train", "val", "test"]:
    dataset = PosterSet(POSTER_PATH, p[s])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, num_workers=1)
    feat = np.concatenate([model(Variable(X).cuda()).cpu().data.numpy() for X, __ in tqdm(dataloader, desc=s)])
    h5.create_dataset(s, data=feat)