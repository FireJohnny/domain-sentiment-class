#!/usr/bin/env python
# encoding: utf-8

"""
__author__ = 'FireJohnny'
@license: Apache Licence 
@file: t_sne.py
@time: 2018/5/24 16:40
"""
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import seaborn as sns
import tensorflow as tf
from cbow_nn import NN
from tensorflow.contrib import learn
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer,TfidfVectorizer
from data_process_for_english import *
import numpy as np

def data_load(source,target):
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
        un_s,label_s = load_unlabel_data(FLAGS.dvd_unl,1)
    elif source == "kitchen":
        s_train, s_labels = load_data_and_labels(FLAGS.kitchen_pos,FLAGS.kitchen_neg)
        un_s,label_s = load_unlabel_data(FLAGS.kitchen_unl,1)
    elif source == "elec":
        s_train, s_labels = load_data_and_labels(FLAGS.elec_pos,FLAGS.elec_neg)
        un_s,label_s = load_unlabel_data(FLAGS.elec_unl,1)
    elif source == "book":
        s_train, s_labels = load_data_and_labels(FLAGS.book_pos,FLAGS.book_neg)
        un_s,label_s = load_unlabel_data(FLAGS.book_unl,1)
    else:
        print("源领域输入有错 请重新输入！")
        exit(1)
    if target == "dvd":
        t_train, t_labels = load_data_and_labels(FLAGS.dvd_pos,FLAGS.dvd_neg)
        un_t,label_t = load_unlabel_data(FLAGS.dvd_unl,0)
    elif target == "kitchen":
        t_train, t_labels = load_data_and_labels(FLAGS.kitchen_pos,FLAGS.kitchen_neg)
        un_t,label_t = load_unlabel_data(FLAGS.kitchen_unl,0)
    elif target == "elec":
        t_train, t_labels = load_data_and_labels(FLAGS.elec_pos,FLAGS.elec_neg)
        un_t,label_t = load_unlabel_data(FLAGS.elec_unl,0)
    elif target == "book":
        t_train, t_labels = load_data_and_labels(FLAGS.book_pos,FLAGS.book_neg)
        un_t,label_t = load_unlabel_data(FLAGS.book_unl,0)
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


    all_string = un_s +un_t #tongjiyuanlingyu de ti-idf
    lens = len(all_string)
    #CBOW
    n_gram = (1, 2)
    length =5000

    vectorizer = CountVectorizer(input="content",stop_words = stop_words,min_df=1,)
    tfidf = TfidfVectorizer(input="content",stop_words= stop_words,min_df= 1 ,ngram_range=n_gram,max_features=5000)

    x_count = vectorizer.fit(all_string)
    x_tfidf = tfidf.fit(all_string)
    s_tf_train = x_tfidf.transform(s_train).toarray()
    t_tf_train = x_tfidf.transform(t_train).toarray()
    dev_sample_index = -1 * int(0.2 * float(len(t_tf_train)))
    s_tf_train,s_tf_test = s_tf_train[:dev_sample_index],s_tf_train[dev_sample_index:]
    t_tf_train,t_tf_test = t_tf_train[:dev_sample_index],t_tf_train[dev_sample_index:]
    s_train_labels,s_test_labels = label_s[:dev_sample_index],label_s[dev_sample_index:]
    t_train_labels,t_test_labels = label_t[:dev_sample_index],label_t[dev_sample_index:]

    return s_tf_test,s_test_labels,t_tf_test,t_test_labels

s_text,s_label,t_text,t_label=data_load("book","dvd")
label = np.concatenate((s_label,t_label))
# print (label)
data = np.vstack((s_text,t_text))
x = load_iris()
color_mapping = {0: sns.xkcd_rgb['bright purple'], 1: sns.xkcd_rgb['green']}
marker = {0:".",1:"o"}
colors = list(map(lambda x: color_mapping[x], label))
# markers = list(map(lambda x:marker[x],label))
# print (colors)
# model = TSNE(learning_rate=100, n_components=2, random_state=0, perplexity=5)
# tsne5 = model.fit_transform(data)
#
# model = TSNE(learning_rate=100, n_components=2, random_state=0, perplexity=30)
# tsne30 = model.fit_transform(data)

model = TSNE(learning_rate=100, n_components=2, random_state=0, perplexity=50)
tsne50 = model.fit_transform(data)


plt.figure(1)
plt.subplot(111)
plt.scatter(tsne50[:, 0][:400], tsne50[:, 1][:400], c=sns.xkcd_rgb['bright purple'],s=10,marker=".")
plt.scatter(tsne50[:, 0][400:], tsne50[:, 1][400:], c=sns.xkcd_rgb['green'],s=10,marker="^")


plt.show()



