#!/usr/bin/env python
# encoding: utf-8

"""
__author__ = 'FireJohnny'
@license: Apache Licence 
@file: train_english.py
@time: 2018/4/5 19:47
"""
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import sys
sys.path.append("..")
import tensorflow as tf
from cbow_nn import NN
from tensorflow.contrib import learn
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer,TfidfVectorizer
import nltk
import io

from data_process_for_english import *

def _parse(source,target):
    FLAGS = tf.flags.FLAGS
    tf.flags.DEFINE_string("dvd_pos", "./data/dvd/dvdpositive", "dvd Data for the positive data.")
    tf.flags.DEFINE_string("dvd_neg", "./data/dvd/dvdnegative", "dvd Data for the negative data.")
    tf.flags.DEFINE_string("dvd_unl","./data/English_data/Data/DVDUnlabel.txt","unlabel dvd data")

    tf.flags.DEFINE_string("elec_pos", "./data/electronics/elecpositive","electronics Data for the positive data.")
    tf.flags.DEFINE_string("elec_neg", "./data/electronics/elecnegative","electronics Data for the negative data.")
    tf.flags.DEFINE_string("elec_unl","./data/English_data/Data/ElectronicsUnlabel.txt","unlabel elec data")

    tf.flags.DEFINE_string("book_pos", "./data/books/bookpositive","books Data for the positive data.")
    tf.flags.DEFINE_string("book_neg", "./data/books/booknegative","books Data for the negative data.")
    tf.flags.DEFINE_string("book_unl","./data/English_data/Data/BookUnlabel.txt","unlabel book data")

    tf.flags.DEFINE_string("kitchen_pos", "./data/kitchen/kitchenpositive","kitchen Data for the positive data.")
    tf.flags.DEFINE_string("kitchen_neg", "./data/kitchen/kitchennegative","kitchen Data for the negative data.")
    tf.flags.DEFINE_string("kitchen_unl","./data/English_data/Data/KitchenUnlabel.txt","unlabel kitchen data")



    if source == "dvd":
        s_train, s_labels = load_data_and_labels(FLAGS.dvd_pos,FLAGS.dvd_neg)
        un_s = load_unlabel_data(FLAGS.dvd_unl)
    elif source == "kitchen":
        s_train, s_labels = load_data_and_labels(FLAGS.kitchen_pos,FLAGS.kitchen_neg)
        un_s = load_unlabel_data(FLAGS.kitchen_unl)
    elif source == "elec":
        s_train, s_labels = load_data_and_labels(FLAGS.elec_pos,FLAGS.elec_neg)
        un_s = load_unlabel_data(FLAGS.elec_unl)
    elif source == "book":
        s_train, s_labels = load_data_and_labels(FLAGS.book_pos,FLAGS.book_neg)
        un_s = load_unlabel_data(FLAGS.book_unl)
    else:
        print("源领域输入有错 请重新输入！")
        exit(1)
    if target == "dvd":
        t_train, t_labels = load_data_and_labels(FLAGS.dvd_pos,FLAGS.dvd_neg)
        un_t = load_unlabel_data(FLAGS.dvd_unl)
    elif target == "kitchen":
        t_train, t_labels = load_data_and_labels(FLAGS.kitchen_pos,FLAGS.kitchen_neg)
        un_t = load_unlabel_data(FLAGS.kitchen_unl)
    elif target == "elec":
        t_train, t_labels = load_data_and_labels(FLAGS.elec_pos,FLAGS.elec_neg)
        un_t = load_unlabel_data(FLAGS.elec_unl)
    elif target == "book":
        t_train, t_labels = load_data_and_labels(FLAGS.book_pos,FLAGS.book_neg)
        un_t = load_unlabel_data(FLAGS.book_unl)
    else:
        print("目标领域领域输入有错 请重新输入！")
        exit(1)

    FLAGS._parse_flags()
    print("\nParameters:")
    for attr, value in sorted(FLAGS.__flags.items()):
        print("{}={}".format(attr.upper(), value))
    print("")


    with codecs.open("./stopwords-master/百度停用词表.txt",encoding="utf-8") as f:
        stop_words = set([line.strip() for line in f if line != "\n"])


    all_string = un_s+ un_t

    #CBOW
    n_gram = (1, 2)
    length =5000

    vectorizer = CountVectorizer(input="content",stop_words = stop_words,min_df=1,max_features=length)
    tfidf = TfidfVectorizer(input="content",stop_words= stop_words,min_df= 1 ,max_features=length,ngram_range=n_gram)

    x_count = vectorizer.fit(all_string)
    x_tfidf = tfidf.fit(all_string)
    s_co_train = x_count.transform(s_train).toarray()
    s_tf_train = x_tfidf.transform(s_train).toarray()

    t_co_train = x_count.transform(t_train).toarray()
    t_tf_train = x_tfidf.transform(t_train).toarray()
    #data split for train and test
    dev_sample_index = -1 * int(0.2 * float(len(t_co_train)))
    s_co_train,s_co_test = s_co_train[:dev_sample_index],s_co_train[dev_sample_index:]
    t_co_train,t_co_test = t_co_train[:dev_sample_index],t_co_train[dev_sample_index:]
    s_tf_train,s_tf_test = s_tf_train[:dev_sample_index],s_tf_train[dev_sample_index:]
    t_tf_train,t_tf_test = t_tf_train[:dev_sample_index],t_tf_train[dev_sample_index:]
    s_train_labels,s_test_labels = s_labels[:dev_sample_index],s_labels[dev_sample_index:]
    t_train_labels,t_test_labels = t_labels[:dev_sample_index],t_labels[dev_sample_index:]

    s_train = combine(s_co_train,s_tf_train)
    s_test = combine(s_co_test,s_tf_test)
    t_train = combine(t_co_train,t_tf_train)
    t_test = combine(t_co_test,t_tf_test)
    #
    # s_train = s_co_train
    # s_test = s_co_test
    # t_train = t_co_train
    # t_test = t_co_test


    return {"len":length,
            "s_train":s_tf_train,
            "s_labels":s_labels,
            "t_train":t_tf_train,
            "t_labels":t_labels,
            "s_test":s_tf_test,
            "s_test_labels":s_test_labels,
            "t_test":t_tf_test,
            "t_test_labels":t_test_labels
    }
    pass

def combine(x,y):
    return np.concatenate([x,y],axis= 1)



def train(argv):

    model = NN(argv=argv)
    # global_step = tf.Variable(initial_value=0,trainable=False,name = "global_step")
    model.build()
    model.train()



if __name__ == "__main__":
    if len(sys.argv) !=5:
        print("请输入两个不同的领域：kitchen,book,dvd,elec")
        exit(1)

    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[4]
    argv = _parse(sys.argv[1],sys.argv[2])

    argv["embed_size"] = 200
    argv["class"] = 2

    argv["num_filters"] = 128
    argv["filter_size"] = [2,3,4]
    argv["batch_size"] = 64
    argv["epoch"] =200
    argv["nb_itr"]  = 5000
    argv["dnn_method"] = sys.argv[3]#method = ["clip_value", "penalty","withoutdiffer","withdiffer","penalty_with_clip"]


    train(argv=argv)
    print("source:{}, target : {}".format(sys.argv[1],sys.argv[2])  )
    pass

pass


