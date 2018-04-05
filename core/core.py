import numpy as np
from PIL import Image
from tqdm import tqdm
import torchvision
import torchvision.transforms as trans
import torch


class PosterSet(torch.utils.data.Dataset):
    def load1(self, fname):
        im = Image.open(self.path + fname + ".jpg").convert("RGB")
        return im

    def __init__(self, path, data, setname, gen_d=None, tv_norm=False, resize=False, augment=False, debug=False):
        """
        @args:
        path:      str   - path of the dataset images
        data:      dict  - data split (set_splits.p)
        setname:   str   - "train", "val" or "test"
        gen_d:     dict  - label transformation dict (gen_d.p)
                         > if not given, generated automatically
        tv_norm:   bool  - normalize images for torchvision.models
                         > if false, calculates own mean/std
        resize:    bool  - True resizes image to (224, 224) for torchvision
                   tuple - resizes images to (x, y)
        augment:   bool  - add minor random transformations to images
        debug:     bool  - use really small subset, if true

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

        self.X = [self.load1(iname) for iname in tqdm(data[setname]['ids'], desc='gen. ' + setname)]
        self.y = data[setname]['labels']
        self.ids = data[setname]['ids']

        to_ten = trans.ToTensor()
        self.mean = [0.485, 0.456, 0.406] if tv_norm else (np.stack(self.X)/255).mean(axis=(0,1,2))
        self.std  = [0.229, 0.224, 0.225] if tv_norm else (np.stack(self.X)/255).std(axis=(0,1,2))
        norm = trans.Normalize(mean=self.mean, std=self.std)

        if resize == True:
            scale = [trans.Resize((224, 224))]
        elif resize == False:
            scale = []
        elif type(resize) == tuple and len(resize) == 2:
            scale = [trans.Resize(resize)]
        else:
            raise RuntimeError("resize needs to be bool or 2-tuple")
        
        if augment:
            augmentation = [trans.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.03),
                            trans.RandomHorizontalFlip(),
                            trans.RandomRotation(5, resample=Image.BILINEAR)]
        else:
            augmentation = [] #TODO!
        
        self.preproc = torchvision.transforms.Compose(scale + augmentation + [to_ten, norm])

    def __getitem__(self, index):
        return self.preproc(self.X[index]), self.frac_hot(self.y[index])

    def __len__(self):
        return len(self.X)

    def frac_hot(self, y):
        num = int(len(self.gen_d) / 2)
        a = np.zeros(num)
        y = [self.gen_d[x] for x in y]
        a[y] = 1
        return torch.Tensor(a)