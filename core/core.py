import numpy as np
from PIL import Image
from tqdm import tqdm
import torchvision
import torchvision.transforms as trans
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from shutil import copyfile

class PosterSet(torch.utils.data.Dataset):
    def load1(self, fname):
        im = Image.open(self.path + fname + ".jpg").convert("RGB")
        return im

    def __init__(self, path, data, setname, gen_d=None, tv_norm=False, resize=False,
                 augment=False, rnd_crop=None, ten_crop=None, debug=False):
        """
        @args:
        path:      str   - path of the dataset images
        data:      dict  - data split (set_splits.p)
        setname:   str   - "train", "val" or "test"
        gen_d:     dict  - label transformation dict (gen_d.p)
                         > if not given, generated automatically
        tv_norm:   bool  - normalize images for torchvision.models
                         > if false, does not normalize
        resize:    bool  - True resizes image to (224, 224) for torchvision
                   tuple - resizes images to (h, w)
                   None  - resizes images to (268, 182)
        augment:   bool  - add minor random transformations to images
        debug:     bool  - use really small subset, if true
        rnd_crop:  tuple - crop randomly with size (h, w) ->use for train
        ten_crop:  tuple - deterministic crop with size (h, w) ->use for val/test
        """
        self.path = path
        if setname == 'all':
            data['all'] = {}
            data['all']['ids']    = data['train']['ids']    + data['val']['ids']    + data['test']['ids']
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

        self.X = data[setname]['ids']
        self.y = data[setname]['labels']
        self.ids = data[setname]['ids']

        if resize == True:
            scale = trans.Resize((224, 224))
        elif resize == False:
            scale = torchvision.transforms.Compose([])
        elif type(resize) == tuple and len(resize) == 2:
            scale = trans.Resize(resize)
        elif resize is None:
            scale = trans.Resize((268, 182))
        else:
            raise RuntimeError("resize needs to be bool or 2-tuple")

        
        self.mean = [0.485, 0.456, 0.406] if tv_norm else [0., 0., 0.]
        self.std  = [0.229, 0.224, 0.225] if tv_norm else [1., 1., 1.]
        norm = trans.Normalize(mean=self.mean, std=self.std)

        to_ten = trans.ToTensor()

        if augment:
            augmentation = torchvision.transforms.Compose([trans.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.03),
                                                           trans.RandomHorizontalFlip(),trans.RandomRotation(5, resample=Image.BILINEAR)])
        else:
            augmentation = torchvision.transforms.Compose([]) #TODO!
        

        if rnd_crop and ten_crop:
            raise RuntimeError("Can only use one crop type.")
        elif type(rnd_crop) == tuple and len(rnd_crop) == 2:
            rc = torchvision.transforms.RandomCrop(rnd_crop)
            self.preproc = torchvision.transforms.Compose([scale, augmentation, rc, to_ten, norm])
            print("using rnd_crop") #DEBUG!
        elif type(ten_crop) == tuple and len(ten_crop) == 2:
            tc = torchvision.transforms.TenCrop(ten_crop)
            self.preproc = lambda img: torch.stack([norm(to_ten(x)) for x in tc(augmentation(scale(img)))], 0)
            print("using ten_crop") #DEBUG!
        elif rnd_crop is None and ten_crop is None:
            self.preproc = torchvision.transforms.Compose([scale, augmentation, to_ten, norm])
        else:
            RuntimeError("Wrong parameters for ten_crop or rnd_crop.")


    def __getitem__(self, index):
        return self.preproc(self.load1(self.X[index])), self.frac_hot(self.y[index])

    def __len__(self):
        return len(self.X)

    def frac_hot(self, y):
        num = int(len(self.gen_d) / 2)
        a = np.zeros(num)
        y = [self.gen_d[x] for x in y]
        a[y] = 1
        return torch.Tensor(a)

def plot_losses(d, folder, shift=True):
    """
    @args:
    d:       dict - ["train"]: dict - {epoch: loss, ...}
                    ["val"]:   dict - {epoch: loss, ...}
                    ["lr"]:    list - [<epoch, when lr reduced>, ...]
    folder:  str  - folder to store plots in
    shift:   bool - shifts train and val by 0.5, since validation happens after training
    """
    x, y    = zip(*d["train"].items())
    X, Y    = zip(*d["val"].items())
    x       = np.array(x) - 0.5 if shift else 0
    y, X, Y = np.array(y), np.array(X), np.array(Y)
    plt.clf()
    plt.plot(x, y, "g-", label='training')
    plt.plot(X, Y, "b-", label='validation')
    plt.scatter(x[d["lr"]], y[d["lr"]], c='g', marker='v') #train
    plt.scatter(X[d["lr"]], Y[d["lr"]], c='b', marker='v') #val
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("losses during training")
    plt.savefig(folder + "/plot_current.png")
    copyfile(folder + "/plot_current.png", folder + "/plot_after_{}.png".format(max(x)))
#[end] plot_losses(d)

def accuracy(pred, label): #single instance use only
    assert label.size(0) == pred.size(0)
    N = pred.size(0)
    n_gen = label.sum()

    sor, ind = pred.sort()
    true_ind = label.nonzero()
    res = []
    for cl in true_ind:
        pos = (ind == cl).nonzero()[0,0]
        a = (pos - n_gen + 1)
        b = (N - 2*n_gen + 1)
        score = a/b
        res.append(min(1, max(0, score)))
    return sum(res) / len(res)
