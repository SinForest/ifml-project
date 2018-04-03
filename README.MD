# Movie-Poster-Classification for IFML

## Usage

+ use `./postercollector.py byGenre <N>` to collect the `./movielist`
  + *`<N>`*: number of top-pages per genre to crawl
+ use `./postercollector.py download` to download all posters to `./posters/`
+ use `./datasetgenerator.py` to split the dataset into train/val/test
+ use `./feature-extraction/extract_features.py <dev>` to generate `./sets/features_all.h5`
  + *`<dev>`*: CUDA device number, `-1` is CPU

## Downloading the sets

The crawled, downloaded and split data can be found [here](http://datasets.thesinforest.eu).
+ `posterset_*k.zip` contains data and labels
  + after `datasetgenerator.py`
+ `posterset_*k_full.zip` additionally contains the features
  + after `extract_features.py`

The zip can be extracted in the root-dir of the repo, all files will be at their desired position.
All files are **encryped with a password**, because we live in deep fear of the copyright law.

## Using the sets

The directory "sets" contains information about the split (train/val/test) of the dataset.  
It also contains extracted features, as well as  information about the existing labels.  

### ./sets/gen_d.p

Contains a dict, that translates labels to numbers. Works in both directions.  
Labels with too less representation have been erased from the dataset.  

e.g.:
+ `gen_d["Action"] == 0`
+ `gen_d["Some Genre"] == 99`
+ `den_d[0] == "Action"`

`len(gen_d) / 2 == {number of classes}`

### ./sets/**.csv

Conatain train/val/test-sets. Each row is a sample, first entry is the imdb-id (the poster name).  
Following entrys are the labels of the sample.  

e.g.:
+ `tt12346, Action, Romance, Drama`
+ `tt57890, Adventure`

### ./sets/set_splits.p

Contains a dict with an entry for train/val/test and dropped (not used) movies.  
Each entry is a dict itself, with `["ids"]` containing a list of imdb-ids in this set,  
and `["labels"]` being a list of the corresponding genres as strings.  
(**WARNING:** could contain leading spaces).  

The same information is in the *.csv-files, but `extract_feature.py` needs this one.

### ./sets/features_all.h5

Contains features from different neural networks.  
All datasets are in the same order, so index [0] from all sets belongs to the same sample.  

Has the following datasets:
+ `["lables"]`: multi-hot-vectors, 1 for each label the sample has, 0 if not. For indices, see *gen_d*.
+ `["ids"]`: the imdb-id (=poster-id) of each sample
+ `["alex\_fc6"]`, `["alex\_fc7"]`: features from AlexNet (pytorch)
+ `["vgg19bn\_fc6"]`, `["vgg19bn\_fc7"]`: features from  VGG19 with batchnorm (pytorch)
+ `["res50\_avg"]`: features from ResNet50s layer (pytorch)
+ `["dense161\_last"]`: features from DenseNet (pytorch)

### ./crawling/movielist

Contains a list of dicts. Each dict represents a movie has the following entries:

+ `["imdb-id"]`: the imdb-id (=poster-id) as *string*.
+ `["title"]`: the movie title as *string*.
+ `["genres"]`: all genres of the movie as *list of strings*. **WARNING:** all genres have a leading blank!
+ `["poster"]`: the URL of the movie-poster as *string*.


Duplications are possible. **This file is not needed anymore**, after `datasetgenerator.py` has been successfully executed.

## Coding

When adding a new feature **in a new branch**, you can use `ln -s ../core/core.py core.py` (maybe change `..` to the projects root directory) to access features from the core utillity, like:
```python
class PosterSet(torch.utils.data.Dataset) #wrapper class for dataset, preprocessing images
```
