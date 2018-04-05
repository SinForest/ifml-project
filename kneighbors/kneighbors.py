#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 11:09:03 2018

@author: twuensche
"""
import sys
from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

import h5py

filename = "../sets/features_all.h5"
n_neighbors = 5
feature_sets = ['alex_fc6', 'alex_fc7', 'vgg19bn_fc6', 'vgg19bn_fc6', \
                'vgg19bn_fc7', 'res50_avg', 'dense161_last']

def main():
    mode, X_train, X_test, Y_train, Y_test = handle_user_input()
    if mode == 'test':
        model = KNeighborsClassifier(n_neighbors=n_neighbors)
        print('begin training..')
        model.fit(X_train, Y_train)
        print('begin prediction..')
        prediction = model.predict(X_test)
        print('write report..')
        report = classification_report(Y_test, prediction)
        print(report)
        accuracy = accuracy_score(Y_test, prediction)
        print(accuracy)
        
    if mode == 'kneighbors':
        model = KNeighborsClassifier(n_neighbors=n_neighbors)
        print('begin training..')
        model.fit(X_train, Y_train)
        print('begin prediction..')
        prediction = model.predict(X_test)
        print('write report..')
        report = classification_report(Y_test, prediction)
        print(report)
        accuracy = accuracy_score(Y_test, prediction)
        print(accuracy)
    if mode == 'rforest':
        model = RandomForestClassifier()
        print('begin training..')
        model.fit(X_train, Y_train)
        print('begin prediction..')
        prediction = model.predict(X_test)
        print('write report..')
        report = classification_report(Y_test, prediction)
        print(report)
        accuracy = accuracy_score(Y_test, prediction)
        print(accuracy)

def handle_user_input():
    inputs = sys.argv[1:]
    mode = ''
    if len(inputs) > 0:
        if inputs[0] == 'test':
            X_train, X_test, Y_train, Y_test = make_test_data()
            mode = 'test'
        elif inputs[0] == 'neighbors':
            X_train, X_test, Y_train, Y_test = load_movie_data('alex_fc6')
            mode = 'kneighbors'
        elif inputs[0] == 'forest':
            X_train, X_test, Y_train, Y_test = load_movie_data('alex_fc6')
            mode = 'rforest'
         
    else:
        help()
        sys.exit()
    return mode, X_train, X_test, Y_train, Y_test

def make_test_data():
    X, Y = make_multilabel_classification(n_classes=2, n_labels=1,
                                      allow_unlabeled=False,
                                      random_state=1)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, 
                                                    test_size=0.25)
    return X_train, X_test, Y_train, Y_test

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
    
    #X_train, X_test, Y_train, Y_test = make_test_data()

    print('data loaded')
    return X_train, X_test, Y_train, Y_test

def help():
    print('How to use the kneighbours script:')
    print('TODO')
    



if  __name__ =='__main__':main()