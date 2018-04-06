from PIL import Image
from random import choice
from torchvision.transforms import ColorJitter, RandomHorizontalFlip, RandomRotation, Compose
from scipy.misc import imshow

SETS_PATH = "../sets/"
POSTER_PATH = "../posters/"

images = [x.split(',')[0].strip() for x in open(SETS_PATH + "train.csv")]
trans = Compose([ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.03),
                 RandomHorizontalFlip(),
                 RandomRotation(5, resample=Image.BILINEAR)])

while(True):
    im = Image.open(POSTER_PATH + choice(images) + ".jpg")
    while(True):
        try:
            tmp = trans(im)
            imshow(tmp)
        except KeyboardInterrupt:
            break
    if input() == "x":
        exit()

