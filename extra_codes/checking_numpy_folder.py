#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 13:12:28 2020

@author: spaul007
"""


import os

path = '/storage/home/spaul007/work/retrieval_from_corpus/Temporally-language-grounding/Dataset/Charades/all_fc6_unit16_overlap0.5'
if os.path.isdir(path):
    print('true')
files = os.listdir(path)
print(len(files))