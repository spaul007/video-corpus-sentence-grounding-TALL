#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 12:54:01 2020

@author: spaul007
"""


import os


train_file = open('../Dataset/Charades/charades_sta_train.txt','r')
train_info = train_file.readlines()

test_file = open('../Dataset/Charades/charades_sta_test.txt','r')
test_info = test_file.readlines()


for item in train_info:
    if item.split(' ')[0] == '004QE':
        print(item)
        
for item in test_info:
    if item.split(' ')[0] == '004QE':
        print(item)
        
        
movie_length_file= open("../Dataset/Charades/ref_info/charades_movie_length_info.txt", 'r')
movie_length_info = movie_length_file.readlines()
for item in movie_length_info:
    if item.split(' ')[0]=='004QE':
        print(item)
        