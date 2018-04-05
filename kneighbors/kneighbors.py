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
import numpy as np
from sklearn.metrics import classification_report

n_neighbors = 5

def main():
    mode, X_train, X_test, Y_train, Y_test = handle_user_input()
    if mode == 'test':
        model = KNeighborsClassifier(n_neighbors=n_neighbors)
        model.fit(X_train, Y_train)
        prediction = model.predict(X_test)
        report = classification_report(Y_test, prediction)
        print(report)
        

def handle_user_input():
    inputs = sys.argv[1:]
    mode = ''
    if len(inputs) > 0:
        if inputs[0] == 'test':
            X_train, X_test, Y_train, Y_test = make_test_data()
            mode = 'test'
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

def help():
    print('How to use the kneighbours script:')
    print('TODO')
    



if  __name__ =='__main__':main()