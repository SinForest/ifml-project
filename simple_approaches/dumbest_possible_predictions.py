#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 16:42:13 2018

@author: twuensche
"""

import simple_methods as util

def main():
    data_set = 'alex_fc6'
    X_train, X_test, Y_train, Y_test = util.load_movie_data(data_set)
    prediction = Y_test*0
    
    filename = data_set + '_output_dumb.txt'
    util.output_to_file(data_set, filename, Y_test, prediction)
    
    


if  __name__ =='__main__':main()