import h5py
import numpy as np
import pickle
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import scale as standardize
from sklearn.preprocessing import normalize

import matplotlib as mpl
import matplotlib.pyplot as plt

FEATURE_PATH    = "../sets/features_all.h5"
GENRE_DICT_PATH = "../sets/gen_d.p"

print("### feature visualization ###")
h5 = h5py.File(FEATURE_PATH, 'r')
gen_d = pickle.load(open(GENRE_DICT_PATH, 'rb'))

idx = np.array(h5["train_idx"][:5000])

labels = np.array(h5["labels"])[idx]
mask = (labels.sum(axis=1) == 1)
labels = labels[mask].astype(int)
n_lab = labels.shape[1]

labels = (labels * np.arange(n_lab)[None,:]).sum(axis=1)
l_set = np.unique(labels)

cmap = plt.cm.gist_ncar
bounds = np.linspace(0,len(l_set),len(l_set)+1)
ticks = [gen_d[i] for i in range(n_lab)]
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

# FOR LOOP HERE

print(" -> loading data")
feat = np.array(h5["res50_avg"])[idx][mask]
print(" -> standardizing data")
normalize(feat, copy=False)
feat = standardize(feat)
print(" -> PCA-ing data")
feat = PCA(50).fit_transform(feat)
print(" -> TSNE-ing")
feat = TSNE(2).fit_transform(feat)

scat = plt.scatter(feat[:,0], feat[:,1], c=labels, cmap=cmap, norm=norm)
cb = plt.colorbar(scat, spacing='proportional',ticks=bounds)
cb.ax.set_yticklabels(ticks) #weird, labens between colors...
plt.show()



