#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 16:25:13 2018

@author: twuensche
"""

import simple_methods as util
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier

from tqdm import tqdm

def main():
   for data_set in tqdm(util.feature_sets):    
        X_train, X_test, Y_train, Y_test = util.load_movie_data(data_set)
        model = OneVsRestClassifier(KNeighborsClassifier(), n_jobs = -1)
        model.fit(X_train, Y_train)
        prediction = model.predict(X_test)
        filename = data_set + '_output_one_vs_rest_kneighbors.txt'
        util.output_to_file(data_set, filename, Y_test, prediction)
        
        util.write_model(model, data_set + '_one_vs_rest_kneighbors')
    


if  __name__ =='__main__':main()