import h5py
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

FEATURE_PATH = "../sets/features_all.h5"
SETS_PATH = "../sets/"

h5     = h5py.File(FEATURE_PATH, 'a')

idx    = np.array(h5["train_idx"][:5000])
feat   = np.array(h5["vgg19bn_fc6"])[idx] #DEBUG!
labels = np.array(h5["labels"])[idx] #DEBUG!

n_clusters = 26

kmeans = KMeans(n_clusters).fit(feat)

res = np.empty((n_clusters, labels.shape[1])).astype(float)

for i in range(n_clusters):
    curr = kmeans.labels_ == i
    cl_labels  = labels[curr]
    normalize(cl_labels, norm='l1', copy=False)# => normalize
    cl_labels  = cl_labels.sum(axis=0)
    res[i] = cl_labels

np.set_printoptions(threshold=np.inf, linewidth=np.inf, precision=2)
print(res.T)
