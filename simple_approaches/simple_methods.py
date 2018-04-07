#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 11:09:03 2018

@author: twuensche
"""
from sklearn.metrics import accuracy_score
from sklearn.metrics import hamming_loss
import pickle
import os
import numpy as np


import h5py

filename = "../sets/features_all.h5"
n_neighbors = 5
feature_sets = ['alex_fc6', 'alex_fc7', 'vgg19bn_fc6', 'vgg19bn_fc6', \
                'vgg19bn_fc7', 'res50_avg', 'dense161_last']




def load_movie_data(feature_set):
    
    
    h5 = h5py.File(filename,'r')
    
    train_idx = h5['train_idx']
    test_idx = h5['test_idx']
    features = h5[feature_set]
    labels = h5['labels']
    
    
    X_train = features[train_idx]
    X_test = features[test_idx]
    Y_train = labels[train_idx]
    Y_test = labels[test_idx]
    

    return X_train, X_test, Y_train, Y_test


    
def write_model(trained_classifier, name):
    with open('models/' + name, 'wb') as fp:
        pickle.dump(trained_classifier, fp)

def read_model(name):
    if os.path.exists(name):
        with open (name, 'rb') as fp:
            return pickle.load(fp)
    else:
        return []

def output_to_file(data_set, filename, expected, prediction):
    text_file = open('results/' + filename, "w")
    
    mean_genres_per_movie = np.mean(np.sum(prediction, axis = 1))
    accuracy = accuracy_score(expected, prediction)
    hamming = hamming_loss(expected, prediction)
    text_file.write(data_set)
    text_file.write('\n')
    text_file.write('subset accuracy: ' + str(accuracy))
    text_file.write('\n')
    text_file.write('hamming loss: ' + str(hamming))
    text_file.write('\n')
    text_file.write('mean genres per movie: ' + str(mean_genres_per_movie))

    
    text_file.close()
    