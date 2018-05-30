#!/usr/bin/env python
# encoding: utf-8

"""
__author__ = 'FireJohnny'
@license: Apache Licence 
@file: cbow_nn.py
@time: 2018/3/16 20:36
"""
import tensorflow as tf
from tensorflow.contrib import slim
import numpy as np
from flip_gradient import flip_gradient
from utils import *
from termcolor import cprint
import os
from functools import partial

def leaky_relu(x, alpha=0.2):
    return tf.maximum(tf.minimum(0.0, alpha * x), x)

def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

class NN():
    def __init__(self,argv):
        self.len = argv["len"]
        self.num_class = argv["class"]

        self.dnn_method = argv["dnn_method"]


        self.batch_size = argv["batch_size"]

        self.num_epoch = argv["epoch"]
        self.nb_critic_itr = argv["nb_itr"]

        self.s_train = argv["s_train"]
        self.s_labels = argv["s_labels"]
        self.t_train = argv["t_train"]
        self.t_labels = argv["t_labels"]
        self.s_test = argv["s_test"]
        self.s_test_labels = argv["s_test_labels"]
        self.t_test = argv["t_test"]
        self.t_test_labels = argv["t_test_labels"]


        self.sess = tf.Session()
        self.RMSoptimizer = tf.train.RMSPropOptimizer(learning_rate=0.0001)
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        # group_init_ops = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        pass
    #creat placeholder
    def _placeholder(self):

        self.s_text = tf.placeholder(dtype=tf.float32, shape = [None, self.len])
        self.s_label = tf.placeholder(dtype=tf.int32, shape=[None, self.num_class])

        self.t_text = tf.placeholder(dtype=tf.float32, shape=[None, self.len])
        self.t_label = tf.placeholder(dtype=tf.int32, shape=[None, self.num_class])

        # self.domain_text = tf.placeholder(dtype=tf.int32 ,shape = [None , self.len])

        self.domain_labels = tf.placeholder(dtype=tf.int32, shape=[None, self.num_class])

        self.dw = tf.placeholder(dtype=tf.float32)

    def _feeddict(self, shuffle=True):
        data_size1 = self.s_train.shape[0]
        data_size2 = self.t_train.shape[0]
        if data_size1 > data_size2:
            temp_size = data_size2
        else :
            temp_size = data_size1

        num_batches_per_epoch = int((temp_size-1)/self.batch_size)+1

        for epoch in range(self.num_epoch):
        # Shuffle the data at each epoch
            if shuffle:
                shuffle_indices1 = np.random.permutation(np.arange(data_size1))
                shuffle_indices2 = np.random.permutation(np.arange(data_size2))
                shuffled_s_data = self.s_train[shuffle_indices1]
                shuffled_s_labels = self.s_labels[shuffle_indices1]
                shuffled_t_data = self.t_train[shuffle_indices2]
                shuffled_t_labels = self.t_labels[shuffle_indices2]
            else:
                shuffled_s_data = self.s_train
                shuffled_s_labels = self.s_labels
                shuffled_t_data  = self.t_train
                shuffled_t_labels = self.t_labels
            # print("epoch {:g}".format(epoch))
            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * self.batch_size
                end_index = min((batch_num + 1) * self.batch_size, data_size1)
                # x,y = shuffled_s_data[start_index:end_index],shuffled_t_data[start_index:end_index]
                # yield (shuffled_s_data[start_index:end_index],shuffled_t_data[start_index:end_index])

                source_l = [[0,1] for _ in range(end_index -start_index)]
                target_l = [[1,0] for _ in range(end_index -start_index)]
                domain_l = np.array(source_l+target_l)
                yield {self.s_text : shuffled_s_data[start_index:end_index],
                 self.s_label: shuffled_s_labels[start_index:end_index],
                 self.t_text: shuffled_t_data[start_index:end_index],
                 self.t_label:shuffled_t_labels[start_index:end_index],
                 self.domain_labels:domain_l,
                 self.dw:0.9
        }

    def neural_net(self, x, name = "domain_net", reuse = False):
        # x = tf.expand_dims(x,-1)
        with tf.variable_scope(name) as scope:
            if reuse:
                scope.reuse_variables()
            net = slim.fully_connected(x,1000,scope="fc1")
            net = slim.fully_connected(net,500,scope="fc2")
            net = slim.fully_connected(net,200,scope="fc3")
        return net

    def class_pred_net(self, feat, name='class_pred', reuse=False):
        with tf.variable_scope(name) as scope:
            if reuse:
                scope.reuse_variables()
            net = slim.fully_connected(feat, 200, scope='fc1')
            net = slim.fully_connected(net, 100, scope='fc2')
            net = slim.fully_connected(net, 2, activation_fn = None, scope='out')
        return net

    # DOMAIN PREDICTION
    def domain_pred_net(self,feat, name='domain_pred_1', reuse=False):
        with tf.variable_scope(name) as scope:
            if reuse:
                scope.reuse_variables()
            feat = flip_gradient(feat, self.dw) # GRADIENT REVERSAL
            net = slim.fully_connected(feat, 100, scope='fc1')
            if name == "domain_pred_2":
                net = slim.fully_connected(net, 2, activation_fn = None, scope='out')
            else :
                net = slim.fully_connected(net, 1, activation_fn = None, scope='out')
        return net

    def optimizer(self):

        self.class_loss = self.compute_class_loss(method="class_loss")
        self.t_class_loss = self.compute_class_loss(method="t_class")
        domain_class_loss = self.compute_class_loss(method="domain_class_loss")
        self.da_loss = self.compute_class_loss(method = "domain_loss_1")

        s_diff_loss = self.diff_loss(self.s_feat, self.s_feat)
        t_diff_loss = self.diff_loss(self.t_feat, self.t_feat)

        #gradient penalty


        if self.dnn_method == "penalty" or self.dnn_method == "penalty_with_clip":

            self.total_loss =  s_diff_loss + t_diff_loss + self.da_loss + self.gp
        elif self.dnn_method == "withoutdiffer":
            self.total_loss = self.da_loss
        # elif self.dnn_method == "clip_value":
        #     self.total_loss = s_diff_loss + t_diff_loss + domain_adversarial_loss #+ domain_class_loss
        else:

            self.total_loss = s_diff_loss + t_diff_loss + self.da_loss
        train_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        self.dann_variables = [v for v in train_variables if v.name.startswith("domain_pred_1")  ]#or v.name.startswith("domain_pred_1")
        self.senti = [v for v in train_variables if v.name.startswith("class_pred")]

        self.optimize_dann = self.RMSoptimizer.minimize(self.total_loss,global_step=self.global_step)
        self.optimize_class  = self.RMSoptimizer.minimize(self.class_loss,)
        self.mmd_value = tf.abs(mmd_loss(self.s_feat,self.t_feat,weight=1.0,value="value"))
        self._clip_op = clip_op(self.dann_variables,)


        # slim.convolution



    def build(self):
        self._placeholder()
        #shared feature extract
        self.s_feat = self.neural_net(self.s_text, name = "domain_net")
        self.t_feat = self.neural_net(self.t_text, reuse=True,name = "domain_net")

        # domain_feat = self.neural_net(self.domain_text,name = "domain_net")
        #private feature extract
        self.s_private = self.neural_net(self.s_text, name = "s_private")
        self.t_private = self.neural_net(self.t_text, name = "t_private")

        #text sentiment class training
        self.s_class_pred = self.class_pred_net(self.s_feat)
        self.t_class_pred = self.class_pred_net(self.t_feat,reuse=True)

        #domain class training
        self.s_c_pred = self.class_pred_net(self.s_feat,name = "domain_class")
        self.t_c_pred = self.class_pred_net(self.t_feat,name = "domain_class",reuse=True)

        #text Adversarial training
        self.s_domain = self.domain_pred_net(self.s_feat,name = "domain_pred_2")
        self.t_domain = self.domain_pred_net(self.t_feat,name = "domain_pred_2",reuse=True)

        #gradient penalty
        self.gp =self.gradient_penalty(self.s_text,self.t_text,self.neural_net)


        self.optimizer()
        self.acc_and_pred()

        self.JS_value = JSdistance(self.s_feat,self.t_feat)

    def gradient_penalty(self,s_feat,t_feat,net):
        alpha = tf.random_uniform((tf.shape(s_feat)[0], 1,),minval = 0., maxval = 1,)
        differ = s_feat - t_feat
        interp = s_feat + alpha*differ
        grads = tf.gradients(net(interp,reuse=True,name = "domain_net"),[interp])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(grads),))
        grad_penalty = tf.reduce_mean((slopes - 1.)**2)
        return grad_penalty
    def compute_class_loss(self,method = "class_loss"):
        #sentiment loss
        if method == "class_loss":
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = self.s_class_pred,labels = self.s_label))
            # loss = tf.reduce_mean(tf.losses.hinge_loss(logits = self.s_class_pred,labels = self.s_label))
            # loss = tf.reduce_mean(tf.losses.mean_squared_error(labels=self.s_label,predictions=self.s_class_pred))

        elif method == "t_class":
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = self.t_class_pred,labels = self.t_label))
        #domain class loss
        elif method == "domain_class_loss":
            #domain class loss
            domain_feat = tf.concat([self.s_c_pred,self.t_c_pred], axis = 0)
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=domain_feat,labels=self.domain_labels))

        elif method == "domain_loss_1":
        #Wgan
            loss = -tf.reduce_mean(self.s_domain) + tf.reduce_mean(self.t_domain)

        elif method == "domain_loss_2":

        #normal gan
            domain_feat = tf.concat([self.s_domain,self.t_domain], axis = 0)
            # loss = tf.reduce_mean(tf.losses.hinge_loss(labels=self.domain_labels,logits=domain_feat))
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=domain_feat,labels=self.domain_labels))
            # loss = tf.reduce_mean(tf.losses.mean_squared_error(labels=self.domain_labels,predictions=domain_feat))
        elif method == "Extract_loss_1":
            loss = -tf.reduce_mean(self.t_domain)
        elif method == "Extract_loss_2":
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.t_domain,labels=self.domain_labels))

        else:
            print("input method is not in the list!")

        return loss

    def summary(self):
        trainable_variable = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        for var in trainable_variable:
            tf.summary.histogram(var.op.name, var)

        tf.summary.scalar("dann_loss", self.total_loss)
        tf.summary.scalar("class_loss",self.class_loss)


        self.merged_summary_op = tf.summary.merge_all()
        pass

    def _restore(self,save_name = "model/"):
        out_dir = os.path.abspath(os.path.curdir)
        _dir = os.path.join(out_dir,save_name,)
        if not os.path.exists(_dir):
            os.makedirs(_dir)
        saver = tf.train.Saver(max_to_keep=5)
        # Try to restore an old model
        model_dir = os.path.abspath(os.path.join(_dir,"model"))
        log_dir = os.path.abspath(os.path.join(_dir,"logs"))
        last_saved_model = tf.train.latest_checkpoint(model_dir)

        group_init_ops = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess.run(group_init_ops)
        summary_writer = tf.summary.FileWriter(log_dir,
                                               graph=self.sess.graph,
                                               flush_secs=20)
        if last_saved_model is not None:
            saver.restore(self.sess, last_saved_model)
            cprint("[*] Restoring model  {}".format(last_saved_model), color="green")
        else:
            tf.train.global_step(self.sess, self.global_step)
            cprint("[*] New model created", color="green")
        return saver, summary_writer ,model_dir

    def summary_merge(self):
        tf.summary.scalar(name = "dnn_loss",tensor=self.da_loss)
        tf.summary.scalar(name = "class_loss", tensor=self.class_loss)
        tf.summary.scalar(name = "target_acc",tensor= self.target_acc)
        tf.summary.scalar(name = "source_acc",tensor=self.source_scc)
        tf.summary.scalar(name = "JS_estimate", tensor=self.JS_value)
        tf.summary.scalar(name = "MMD_distance",tensor=self.mmd_value)
        summary = tf.summary.merge_all()
        return summary

    def _save(self,  summary_writer, is_iter=True, extras=None):
        current_iter = self.sess.run(self.global_step)
        if not is_iter:
            # Save graph
            summary_writer.add_summary(extras, current_iter)
        # Iter saving (write variable + loss)
        else:
            # summary_dann_loss = tf.Summary(value=[
            #     tf.Summary.Value(tag="dann_loss", simple_value=extras[0]),
            # ])
            # summary_class_loss = tf.Summary(value=[
            #     tf.Summary.Value(tag="class_loss", simple_value=extras[1]),
            # ])
            # summary_t_class_loss = tf.Summary(value = [tf.Summary.Value(tag = "t_loss",simple_value = extras[4])])
            # summary_source_acc = tf.Summary(value =[tf.Summary.Value(tag="source_acc",simple_value =extras[2])])
            # summary_target_acc = tf.Summary(value =[tf.Summary.Value(tag="target_acc",simple_value =extras[3])])
            # summary_js_value = tf.Summary(value =[tf.Summary.Value(tag="JS_estimate",simple_value =extras[5])])

            summary_writer.add_summary(extras,global_step = current_iter)
            # summary_writer.add_summary(summary_dann_loss , global_step=current_iter)
            # summary_writer.add_summary(summary_class_loss , global_step=current_iter)
            # summary_writer.add_summary(summary_source_acc,global_step = current_iter)
            # summary_writer.add_summary(summary_target_acc,global_step = current_iter)
            # summary_writer.add_summary(summary_t_class_loss,global_step = current_iter)
            # summary_writer.add_summary(summary_js_value,global_step=current_iter)




    def diff_loss(self,private_samples, shared_samples, weight=1.0, name=''):
        # def difference_loss(private_samples, shared_samples, weight=1.0, name=''):
        private_samples -= tf.reduce_mean(private_samples, 0)
        shared_samples -= tf.reduce_mean(shared_samples, 0)
        private_samples = tf.nn.l2_normalize(private_samples, 1)
        shared_samples = tf.nn.l2_normalize(shared_samples, 1)
        correlation_matrix = tf.matmul( private_samples, shared_samples, transpose_a=True)
        cost = tf.reduce_mean(tf.square(correlation_matrix)) * weight
        cost = tf.where(cost > 0, cost, 0, name='value')
        #tf.summary.scalar('losses/Difference Loss {}'.format(name),cost)
        assert_op = tf.Assert(tf.is_finite(cost), [cost])
        with tf.control_dependencies([assert_op]):
            tf.losses.add_loss(cost)
        return cost

        pass

    def acc_and_pred(self,):
        #sentiment predict
        self.source_scc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.s_class_pred,1),tf.argmax(self.s_label,1)),tf.float32))
        self.target_acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.t_class_pred,1),tf.argmax(self.t_label,1)),tf.float32))
        #domain label predict
        domain_= tf.concat([self.s_domain,self.t_domain],axis=0)
        self.domain_acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(domain_,1),tf.argmax(self.domain_labels,1)),tf.float32))
        # self.t_domain_acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.t_domain,1),tf.argmax(self.t_label,1)),tf.float32))


    def train(self,):
        self.sess.run(tf.global_variables_initializer())
        import time


        t = str(int(time.time()))
        saver, summary_writer,model_dir = self._restore(save_name=t)
        merge = self.summary_merge()
        start_itr = self.sess.run(self.global_step)
        top = 0
        for itr in range(start_itr, self.nb_critic_itr):
            # if itr <25 or itr % 50 == 0:
            #     nb_critic_iteration = 20
            # else:
            nb_critic_iteration = 100
            #
            # for _ in range(nb_critic_iteration):
            feed_dict = next(self._feeddict())

            # self.sess.run(self.global_step)
            # feed_dict =self._feed_dict

            # for _ in range(nb_critic_iteration):
            if self.dnn_method == "clip_value" or self.dnn_method =="penalty_with_clip":

                self.sess.run([self.optimize_dann,self.optimize_class],feed_dict=feed_dict)
                _, = self.sess.run([self._clip_op],feed_dict=feed_dict) #是否进行梯度裁剪

            else:
                self.sess.run([self.optimize_dann,self.optimize_class],feed_dict=feed_dict)
                # self.sess.run()

            # self.sess.run(self.global_step)


            source_l = [[0,1] for _ in range(len(self.s_test))]
            target_l = [[1,0] for _ in range(len(self.t_test))]
            all_domain = np.array(source_l+target_l)
            test_feeddict = {  self.s_text :self.s_test ,
                                       self.s_label: self.s_test_labels,
                                       self.t_text: self.t_test,
                                       self.t_label:self.t_test_labels,
                                       self.domain_labels:all_domain,
                                       self.dw:0.05
                                }
            d_loss, c_loss,source_acc,target_acc ,MMD_value,_merge = self.sess.run([self.total_loss,
                                                                                        self.class_loss,
                                                                                        self.source_scc,
                                                                                        self.target_acc,
                                                                                        self.mmd_value,
                                                                                        merge],
                                                       feed_dict=test_feeddict)

            self._save( summary_writer, is_iter=True, extras=_merge)

            if target_acc>top:
                iter_num = itr
                top = target_acc
            d_loss_val = d_loss
            c_loss_val = c_loss
            print(
                        "Iter {}: \n\tdiscriminator loss: {}, \n\tclass loss: {}, \n\tsource_acc: {}, \n\ttarget acc: {} ".format(
                                                                                    itr, d_loss_val,c_loss_val,source_acc,target_acc))
            itr += 1
            if itr %100 == 0:
                if not os.path.exists(os.path.join(model_dir,self.dnn_method)):
                    os.makedirs(os.path.join(model_dir,self.dnn_method))
                saver.save(self.sess,save_path =os.path.join(model_dir ,"model"),global_step =self.global_step)
        print("dnn_method: {} ,{}\n".format(self.dnn_method,t))
        print ("iter {}, top acc {}".format(iter_num,top))
if __name__ == '__main__':
    pass


