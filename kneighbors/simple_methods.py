#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 11:09:03 2018

@author: twuensche
"""
import sys
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier
import pickle
import os


import h5py

filename = "../sets/features_all.h5"
n_neighbors = 5
feature_sets = ['alex_fc6', 'alex_fc7', 'vgg19bn_fc6', 'vgg19bn_fc6', \
                'vgg19bn_fc7', 'res50_avg', 'dense161_last']

def main():
    mode = handle_user_input()
    if mode == 'kneighbors':
        X_train, X_test, Y_train, Y_test = load_movie_data('alex_fc6')
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
        X_train, X_test, Y_train, Y_test = load_movie_data('alex_fc6')
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
    if mode == 'onevsrest':
        X_train, X_test, Y_train, Y_test = load_movie_data('alex_fc6')
        model = OneVsRestClassifier(SVC(kernel='linear'), n_jobs = -1)
        print('begin training..')
        model.fit(X_train[:100], Y_train[:100])
        print('begin prediction..')
        prediction = model.predict(X_test[:100])
        accuracy = accuracy_score(Y_test, prediction)
        print(accuracy)
        write_model(model)
    if mode == 'all':
        for set in feature_sets:
            X_train, X_test, Y_train, Y_test = load_movie_data(set)
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
        if inputs[0] == 'neighbors':
            mode = 'kneighbors'
        elif inputs[0] == 'forest':
            mode = 'rforest'
        elif inputs[0] == 'onevsrest':
            mode = 'onevsrest'
        elif inputs[0] == 'all':
            mode = 'all'
    else:
        help()
        sys.exit()
    return mode


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
    
def write_model(trained_classifier):
    with open('trained_classifier', 'wb') as fp:
        pickle.dump(trained_classifier, fp)
        print('model written to file')

def read_model():
    if os.path.exists('trained_classifier'):
        with open ('trained_classifier', 'rb') as fp:
            print('trained_classifier loaded')
            return pickle.load(fp)
    else:
        return []



if  __name__ =='__main__':main()