#!/usr/bin/env python
# encoding: utf-8

"""
__author__ = 'FireJohnny'
@license: Apache Licence 
@file: data_process_for_english.py
@time: 2018/3/1 15:38
"""


import numpy as np
import re
import itertools
from collections import Counter
import codecs
import random
import csv

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[\n\t\r]","",string)
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_and_labels(positive_data_file, negative_data_file=None):
    # Load data from files
    x_text = []
    labels = []
    with codecs.open(positive_data_file,encoding="utf-8") as f:
        for l in f.readlines():
            _label = [1,0]

            x_text.append(clean_str(l))
            labels.append(_label)

    with codecs.open(negative_data_file,encoding="utf-8") as f:
        for l in f.readlines():
            _label = [0,1]

            x_text.append(clean_str(l))
            labels.append(_label)

    return x_text,np.array(labels)


def load_unlabel_data(unlabel_file,_label = None):
    x_text = []
    label = []
    with codecs.open(unlabel_file,encoding="utf-8") as f:
        for l in f.readlines():
            label.append(_label)
            x_text.append(clean_str(l))
    return x_text#,np.array(label)
def load_pre_data(pre_data):
    p_data = list(codecs.open(pre_data, "r",encoding="utf-8").readlines())
    data = [s.strip(" ") for s in p_data]

    return data

def batch_iter(s_data,t_data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """

    data_size = len(s_data)
    num_batches_per_epoch = int((len(s_data)-1)/batch_size)+1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            random.shuffle(s_data)
            random.shuffle(t_data)
            shuffled_s_data = s_data
            shuffled_t_data = t_data
        else:
            shuffled_s_data = s_data
            shuffled_t_data  = t_data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield (shuffled_s_data[start_index:end_index],shuffled_t_data[start_index:end_index])

