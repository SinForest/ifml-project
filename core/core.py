import numpy as np
from PIL import Image
from tqdm import tqdm
import torchvision
import torch


class PosterSet(torch.utils.data.Dataset):
    def load1(self, fname):
        im = Image.open(self.path + fname + ".jpg").convert("RGB")
        return im

    def __init__(self, path, data, setname, gen_d=None, normalize=False ,debug=False):
        """
        @args:
        path:      str  - path of the dataset images
        data:      dict - data split (set_splits.p)
        setname:   str  - "train", "val" or "test"
        gen_d:     dict - label transformation dict (gen_d.p)
                        > if not given, generated automatically
        normalize: bool - normalize images for torchvision.models

        """
        self.path = path
        if setname == 'all':
            data['all'] = {}
            data['all']['ids'] = data['train']['ids'] + data['val']['ids'] + data['test']['ids']
            data['all']['labels'] = data['train']['labels'] + data['val']['labels'] + data['test']['labels']
        if debug:
            data[setname]['ids'] = data[setname]['ids'][:65]
        if gen_d is None: # not tested yet!!
            genres     = {item.strip() for sublist in data[setname]['labels'] for item in sublist}
            self.gen_d = dict(zip(genres, range(len(genres))))
            gen_d2     = {val: key for key, val in self.gen_d.items()}
            self.gen_d.update(gen_d2)
            del gen_d2
        else:
            self.gen_d = gen_d

        self.X = [self.load1(iname) for iname in tqdm(data[setname]['ids'], desc='dataset')]
        self.y = data[setname]['labels']
        self.ids = data[setname]['ids']
        norm = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])
        scale = torchvision.transforms.Resize((224, 224))
        if normalize:
            self.preproc = torchvision.transforms.Compose([scale, torchvision.transforms.ToTensor(), norm])
        else:
            self.preproc = torchvision.transforms.Compose([scale, torchvision.transforms.ToTensor()])

    def __getitem__(self, index):
        return self.preproc(self.X[index]), self.frac_hot(self.y[index])

    def __len__(self):
        return len(self.X)

    def frac_hot(self, y):
        num = int(len(gen_d) / 2)
        a = np.zeros(num)
        y = [self.gen_d[x] for x in y]
        a[y] = 1
        return a