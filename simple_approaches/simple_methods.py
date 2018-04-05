#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 11:09:03 2018

@author: twuensche
"""
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import pickle
import os


import h5py

filename = "../sets/features_all.h5"
n_neighbors = 5
feature_sets = ['alex_fc6', 'alex_fc7', 'vgg19bn_fc6', 'vgg19bn_fc6', \
                'vgg19bn_fc7', 'res50_avg', 'dense161_last']




def load_movie_data(feature_set):
    
    print('load data..')
    
    h5 = h5py.File(filename,'r')
    
    train_idx = h5['train_idx']
    test_idx = h5['test_idx']
    features = h5[feature_set]
    labels = h5['labels']
    
    
    X_train = features[train_idx]
    X_test = features[test_idx]
    Y_train = labels[train_idx]
    Y_test = labels[test_idx]
    

    print('data loaded')
    return X_train, X_test, Y_train, Y_test

def help():
    print('How to use the kneighbours script:')
    print('TODO')
    
def write_model(trained_classifier, name):
    with open(name, 'wb') as fp:
        pickle.dump(trained_classifier, fp)
        print('model written to file')

def read_model(name):
    if os.path.exists(name):
        with open (name, 'rb') as fp:
            print('trained_classifier loaded')
            return pickle.load(fp)
    else:
        return []

def output_to_file(data_set, filename, expected, prediction):
    text_file = open(filename, "w")
    
    accuracy = accuracy_score(expected, prediction)
    report = classification_report(expected, prediction)
    text_file.write(data_set)
    text_file.write(accuracy)
    text_file.write(report)
    
    text_file.close()