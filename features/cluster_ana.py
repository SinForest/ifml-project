import h5py
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans

FEATURE_PATH = "../sets/features_all.h5"
SETS_PATH = "../sets/"

h5     = h5py.File(FEATURE_PATH, 'a')

#TDO: load train only:
feat   = h5["vgg19bn_fc6"][:100] #DEBUG!
labels = h5["labels"][:100] #DEBUG!

n_clusters = 4

kmeans = KMeans(n_clusters).fit(feat)

for i in range(n_clusters):
    curr = kmeans.labels_ == i
    cl_labels  = labels[curr]
    cl_labels /= cl_labels.sum(axis=1)[:, None]
    cl_labels  = cl_labels.sum(axis=0)
    print(i, cl_labels)
