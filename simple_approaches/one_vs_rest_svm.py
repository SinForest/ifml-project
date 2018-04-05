#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 16:27:54 2018

@author: twuensche
"""

import simple_methods.py as util
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

def main():
    data_set = 'alex_fc6'
    X_train, X_test, Y_train, Y_test = util.load_movie_data(data_set)
    model = OneVsRestClassifier(SVC(kernel='linear'), n_jobs = -1)
    print('begin training..')
    model.fit(X_train[:100], Y_train[:100])
    print('begin prediction..')
    prediction = model.predict(X_test[:100])
    
    filename = data_set + '_output_one_vs_rest_svm.txt'
    util.output_to_file(data_set, filename, Y_test, prediction)
    
    util.write_model(model, data_set + '_one_vs_rest_svm')
    


if  __name__ =='__main__':main()