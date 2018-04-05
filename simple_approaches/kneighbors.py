#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 16:27:54 2018

@author: twuensche
"""

import simple_methods.py as util
from sklearn.neighbors import KNeighborsClassifier

def main():
    data_set = 'alex_fc6'
    X_train, X_test, Y_train, Y_test = util.load_movie_data(data_set)
    model = KNeighborsClassifier()
    print('begin training..')
    model.fit(X_train, Y_train)
    print('begin prediction..')
    prediction = model.predict(X_test)
    
    filename = data_set + '_output_kneighbors.txt'
    util.output_to_file(data_set, filename, Y_test, prediction)
    
    util.write_model(model, data_set + '_kneighbors')
    


if  __name__ =='__main__':main()