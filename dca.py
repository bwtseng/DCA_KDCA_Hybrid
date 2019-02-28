### Import basic package
import numpy as np 
import pandas as pd 
import tensorflow as tf
import random 
import time 
import os 
from numpy.linalg import inv, norm, eigh
from sklearn.model_selection import train_test_split

### Import DL and evaluation pachage. 
import tensorflow.contrib.layers as ly 
import matplotlib.pyplot as plt 
from scipy.misc import imrotate ,imread ,imsave,imresize
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics.pairwise import polynomial_kernel
from sklearn.metrics.pairwise import laplacian_kernel
from sklearn.svm import SVC

random.seed(9)
np.random.seed(9)
class dca: 

    def __init__(self, arg): 


        self.arg = arg 
        self.ori_dim = self.arg.ori_dim
        self.com_dim = self.arg.com_dim
        self.ridge = self.arg.ridge
        self.kernel_index = self.arg.kernel
        self.epo = self.arg.epoch
        self.gamma = self.arg.gamma
        self.kernel = self.arg.kernel
        self.batch_size = self.arg.batch_size

        self.t_data, self.t_label_u, self.t_label_p, self.v_data, self.v_label_u, self.v_label_p, self.te_data, self.te_label_u, self.te_label_p =  self.sample(self.load_data)
        self.u_label_index = self.search_index(self.t_label_u, utility=True)
        self.p_label_index = self.search_index(self.t_label_p, utility=False)


        ### Simple DCA. 

        self.w_matrix = self.within_matrix(self.t_data, self.u_label_index)
        self.b_matrix = self.between_matrix(self.t_data, self.u_label_index)
        self.s_matrix = self.scatter_matrix(self.w_matrix, self.b_matrix)

        ### KERNEL DCA (from hybrid paper, there is different difinition in the Thee paper.)
        self.k_matrix = self.kernel_matrix(self.t_data, self.t_data, self.kernel, self.gamma, train=True)
        print("********** ALL MATRIX FINISH *************")
        self.k_b_matrix = self.kernel_between_matrix(self.k_matrix, self.u_label_index)
        self.k_w_matrix = self.kernel_within_matrix(self.k_matrix, self.u_label_index) 
        self.k_s_matrx = self.kernel_scatter_matrix(self.k_w_matrix, self.k_b_matrix)
        print("********** ALL KERNEL MATRIX FINISH *************")


        N = self.k_matrix.shape[0]
        one_vec = np.array([1 for i in range(N)]).reshape(-1, 1)
        C_n = (np.identity(N) - (1/N)*np.dot(one_vec, one_vec.T))

        self.eigen_val, self.eigen_vec = self.dca_formula(self.s_matrix, self.b_matrix, self.ridge)
        self.k_eigen_val, self.k_eigen_vec = self.kernel_dca_formula(self.k_matrix, self.k_s_matrx, self.k_b_matrix, self.ridge)

        self.dca_data_train = self.projection(self.eigen_vec, self.t_data)

        gg = np.dot(C_n.T, self.k_matrix)
        self.k_dca_data_train = self.projection(self.k_eigen_vec, gg)

        self.dca_data_val = self.projection(self.eigen_vec, self.v_data)


        gg = np.dot(C_n.T, self.kernel_matrix(self.t_data, self.v_data, self.kernel, self.gamma, train=False))
        #self.k_dca_data_val = self.projection(self.k_eigen_vec, self.kernel_matrix(self.t_data, self.v_data, self.kernel, self.gamma, train=False))
        self.k_dca_data_val = self.projection(self.k_eigen_vec, gg, kernel=True)

        self.dca_data = self.projection(self.eigen_vec, self.te_data)

        gg = np.dot(C_n.T, self.kernel_matrix(self.t_data, self.te_data, self.kernel, self.gamma, train=False))
        #self.k_dca_data = self.projection(self.k_eigen_vec, self.kernel_matrix(self.t_data, self.te_data, self.kernel, self.gamma, train=False))
        self.k_dca_data = self.projection(self.k_eigen_vec, gg, kernel=True)


        self.build_NN()

    def load_data(self): 

        path = "MHEALTHDATASET"
        # subject 0, 10, 2~9
        file = sorted(os.listdir(path))

        all_data = []
        privte_label = []
        utility_label = []
        
        id = 0
        for i in file: 
            if i.split('.')[-1] == 'log':
                a = open(os.path.join(path, i), 'r')
                temp_3 = [[] for i in range(12)]
                temp_1 = [[] for i in range(12)]
                temp_2 = [[] for i in range(12)]

                for j in a :    
                    temp = j[:-1].split('\t')   
                    if int(temp[-1]) == 0 : 
                        pass 
                    else :
                        temp_3[int(temp[-1])-1].append(temp[0:23])
                        temp_1[int(temp[-1])-1].append(int(temp[-1])-1)
                        temp_2[int(temp[-1])-1].append(id)


                utility_label.append(temp_1)
                privte_label.append(temp_2)
                all_data.append(temp_3)
                id += 1 

            else : 
                pass

        ### Sample data, each individual sample 1800 pieces, and each activity sample 1500 pieces.
        return all_data, utility_label, privte_label

    
    def sample(self, load_data_func):


        all_data, utility_label, privte_label = load_data_func()

        t_data = []
        t_label_u = []
        t_label_p = []

        v_data = []
        v_label_u = []
        v_label_p = []

        te_data = []
        te_label_u = []
        te_label_p = []

        ### Index for the label without enough amount !!! 
        '''
        index = []
        for k in range(10):
            index.append[[i for i in range(12)]]
        '''

        ### Sample !   
        for i in range(10):
            for lab in range(12):

                u_lab = np.array(utility_label[i][lab]).astype('int32')
                p_lab = np.array(privte_label[i][lab]).astype('int32')
                data = np.array(all_data[i][lab]).astype('float32')
                lab_num = len(utility_label[i][lab]) 

                indice = random.sample([i for i in range(lab_num)], 150)
                train_indice, temp = train_test_split(indice, test_size=1/3)
                val_indice, test_indice = train_test_split(temp, test_size=1/2)
                #index[i][lab].append(index)

                t_data.append(data[np.array(train_indice), :])
                #print(t_data[0].shape)
                t_label_u.append(u_lab[np.array(train_indice)])
                #print(t_label_u[0].shape)
                t_label_p.append(p_lab[np.array(train_indice)])
                #print(t_label_p[0].shape)

                v_data.append(data[np.array(val_indice), :])
                v_label_u.append(u_lab[np.array(val_indice)])
                v_label_p.append(p_lab[np.array(val_indice)])

                te_data.append(data[np.array(test_indice), :])
                te_label_u.append(u_lab[np.array(test_indice)])
                te_label_p.append(p_lab[np.array(test_indice)])

        t_data = np.concatenate(t_data, axis=0)
        v_data = np.concatenate(v_data, axis=0)
        te_data = np.concatenate(te_data, axis=0)

        t_label_u = np.concatenate(t_label_u, axis=0)
        v_label_u = np.concatenate(v_label_u, axis=0)
        te_label_u = np.concatenate(te_label_u, axis=0)

        t_label_p = np.concatenate(t_label_p, axis=0)
        v_label_p = np.concatenate(v_label_p, axis=0)
        te_label_p = np.concatenate(te_label_p, axis=0)

        t_data, t_label_u, t_label_p = self.shuffle(t_data, t_label_u, t_label_p)
        #print(t_data.shape)
        #print(t_label_p.shape)
        #print(t_label_u.shape)
        v_data, v_label_u, v_label_p = self.shuffle(v_data, v_label_u, v_label_p)
        te_data, te_label_u, te_label_p = self.shuffle(te_data, te_label_u, te_label_p)

        return t_data, t_label_u, t_label_p, v_data, v_label_u, v_label_p, te_data, te_label_u, te_label_p

    def shuffle(self, input, label_u, label_p):
        a = input.shape[0]
        a = [i for i in range(a)]
        random.shuffle(a)
        a = np.array(a)
        return input[[a]], label_u[[a]], label_p[[a]]


    def search_index(self, label, utility=True):

        if utility: 

            temp = [[] for i in range(12)]
            count = 0
            for i in label:
                temp[i].append(count)
                count += 1 

            return temp 

        else: 

            temp = [[] for i in range(10)]
            count = 0
            for i in label:
                temp[i].append(count)
                count += 1 

            return temp 

    def kernel_matrix(self, data_matrix, y, kernel_index, gamma, train=True):



        kernel_list = ['rbf', 'polynomial', 'laplacian', 'linear']

        if train : 

            if kernel_index =='rbf':
                K = rbf_kernel(data_matrix, gamma = gamma)          
            if kernel_index =='polynimail':
                K = polynomial_kernel(data_matrix, gamma = gamma)
            if kernel_index =='laplacian':
                K = laplacian_kernel(data_matrix, gamma = gamma)
            if kernel_index =='linear':
                K = pairwise_kernels(data_matrix, 'linear')
            return K 

        else :

            if kernel_index =='rbf':
                K = rbf_kernel(data_matrix, y, gamma = gamma)
            if kernel_index =='polynimail':
                K = polynomial_kernel(data_matrix, y, gamma = gamma)
            if kernel_index =='laplacian':
                K = laplacian_kernel(data_matrix, y, gamma = gamma)
            if kernel_index =='linear':
                K = pairwise_kernels(data_matrix, y, 'linear')
            return K        



    def within_matrix(self, train_matrix, label_index):

        # For classes, each entry of the list is a matrix: 
        N = train_matrix.shape[0]
        all_matrix = []

        for i in label_index:
            index = np.array(i)
            temp = train_matrix[index, :]
            mean_cls = np.mean(temp, axis=0).reshape(-1, 1)
            matrix = np.zeros((self.ori_dim, self.ori_dim))

            '''
            for j in temp: 
                vec = j.reshape(-1, 1) - mean_cls
                matrix += np.dot(vec, vec.T)
            '''

            matrix = np.cov(temp.T, ddof=0)
            all_matrix.append(matrix)
        matrix = np.zeros((self.ori_dim, self.ori_dim))
        for i in all_matrix:
            matrix += i 

        return matrix

    def between_matrix(self, train_matrix, label_index): 

        # nEED MODIFY
        N = train_matrix.shape[0]
        matrix = np.zeros((self.ori_dim, self.ori_dim))
        mu = np.mean(train_matrix, axis=0).reshape(-1, 1)
        for i in label_index:
            index = np.array(i)
            temp = train_matrix[index, :]
            cls_mu = np.mean(temp, axis=0).reshape(-1, 1)
            vec = cls_mu - mu
            matrix += len(i) * np.dot(vec, vec.T)

        return matrix

    def scatter_matrix(self, between_matrix, within_matrix): 

        return between_matrix + within_matrix 

    def dca_formula(self, scatter_matrix, between_matrix, ridge):

        outside = inv(scatter_matrix + ridge * np.identity(scatter_matrix.shape[0]))
        tar_matrix = np.dot(outside, between_matrix)
        w, v = eigh(tar_matrix)

        return w, v

    def kernel_scatter_matrix(self, kernel_between_matrix, kernel_within_matrix):
        ### Note that this is the sqaure version
        return kernel_between_matrix + kernel_within_matrix

    def kernel_within_matrix(self, kernel_matrix, label_index):

        N = kernel_matrix.shape[0]
        #N = train_matrix.shape[0]
        all_matrix = []


        #kernel_matrix =  self.kernel_matrix(train_matrix, train_matrix, self.kernel, self.gamma, train=True)

        count = 0
        for i in label_index:
            index = np.array(i)
            #temp = train_matrix[index, :]
            temp = kernel_matrix[index, :]
            mean_cls = np.mean(temp, axis=0).reshape(1, -1)
            #kernel_mean_cls = self.kernel_matrix(train_matrix, mean_cls, self.kernel, self.gamma, train=False).reshape(-1, 1)
            matrix = np.zeros((N, N))

            ## this inner loop can be excluded
            '''
            for j in temp: 
                vec = j.reshape(-1, 1) - mean_cls
                #vec =  self.kernel_matrix(train_matrix, j.reshape(1, -1), self.kernel, self.gamma, train=False) - kernel_mean_cls

                matrix += np.dot(vec, vec.T)

                print(count)
                count+=1 
            '''
            matrix = np.cov(temp.T, ddof=0)
            all_matrix.append(matrix)

        matrix = np.zeros((N, N))

        for i in all_matrix:
            matrix += i 
        one_vec = np.array([1 for i in range(N)]).reshape(-1, 1)
        C_n = (np.identity(N) - (1/N)*np.dot(one_vec, one_vec.T))

        output = np.dot(C_n, matrix)
        output = np.dot(output, C_n)
        return output

    def kernel_between_matrix(self, kernel_matrix, label_index):

        N = kernel_matrix.shape[0]

        #kernel_matrix =  self.kernel_matrix(train_matrix, train_matrix, self.kernel, self.gamma, train=True)
        kernel_mu = np.mean(kernel_matrix, axis=0).reshape(-1, 1)
        #mu = np.mean(train_matrix, axis=0).reshape(1, -1)
        #kernel_mu = self.kernel_matrix(train_matrix, mu, self.kernel, self.gamma, train=False).reshape(-1, 1)
        matrix = np.zeros((N, N))

        for i in label_index:
            index = np.array(i)
            #temp = train_matrix[index, :]      
            temp = kernel_matrix[index, :]  
            cls_mu = np.mean(temp, axis=0).reshape(1, -1)
            #kernel_mean_cls = self.kernel_matrix(train_matrix, cls_mu, self.kernel, self.gamma, train=False).reshape(-1, 1)
            vec = cls_mu - kernel_mu
            matrix += (len(i) * np.dot(vec, vec.T))

        one_vec = np.array([1 for i in range(N)]).reshape(-1, 1)
        C_n = (np.identity(N) - (1/N)*np.dot(one_vec, one_vec.T))

        output = np.dot(C_n, matrix)
        output = np.dot(output, C_n)

        return output                     

    def kernel_dca_formula(self, kernel_matrix, k_scatter_matrix, k_between_matrix, ridge):

        N = kernel_matrix.shape[0]
        one_vec = np.array([1 for i in range(N)]).reshape(-1, 1)
        C_n = (np.identity(N) - (1/N)*np.dot(one_vec, one_vec.T))

        k_bar = np.dot(C_n, kernel_matrix)
        k_bar = np.dot(k_bar, C_n)

        k_bar_square = k_scatter_matrix + k_between_matrix

        outside = inv(k_bar_square + ridge * k_bar)
        tar_matrix = np.dot(outside, k_between_matrix)
        w, v = eigh(tar_matrix)

        return w, v 

    def projection(self, eigen_vector, data_matrix, kernel=False):

        projection_matrix = eigen_vector[:self.com_dim]
        #print(projection_matrix.shape)
        if kernel: 
            output =  np.dot(projection_matrix, data_matrix)
        else :
            output =  np.dot(projection_matrix, data_matrix.T)
        return output.T

    def fs_layer(self, input, output_dim, activation= tf.nn.relu, bias = False):

        if bias :  
            return ly.fully_connected(input, output_dim, activation_fn = activation, weights_initializer=tf.contrib.layers.xavier_initializer())#, biases_initializer = None)
        else : 
            return ly.fully_connected(input, output_dim, activation_fn = activation, weights_initializer=tf.contrib.layers.xavier_initializer(), biases_initializer = None)


    def build_NN(self):

        ## Remember to sample class number.
        self.data_u = tf.placeholder(tf.float32, shape=[None, self.com_dim])
        self.data_p = tf.placeholder(tf.float32, shape=[None, self.com_dim])
        self.label_u = tf.placeholder(tf.int64, shape=[None])
        self.label_p = tf.placeholder(tf.int64, shape=[None])

        self.one_hot_u = tf.one_hot(self.label_u, 12)
        self.one_hot_p = tf.one_hot(self.label_p, 10)

        self.dropout = tf.placeholder(tf.float32)
        self.train_phase = tf.placeholder(tf.bool)

        with tf.variable_scope('utility'):

            out_u = self.fs_layer(self.data_u, 512, bias=True)
            #out_u = tf.nn.dropout(out_u, keep_prob = self.dropout)
            out_u = self.fs_layer(out_u, 512, bias=True)
            #out_u = tf.nn.dropout(out_u, keep_prob = self.dropout)
            #out_u = self.fs_layer(out_u, 512, bias=True)
            #out_u = tf.nn.dropout(out_u, keep_prob = self.dropout)
            self.out_u = self.fs_layer(out_u, 128, bias=True)
            out_u = tf.nn.dropout(out_u, keep_prob = self.dropout)
            self.pre_logit_u = self.fs_layer(self.out_u, 12, activation=None, bias = True)
            self.prob_u = tf.nn.softmax(self.pre_logit_u)


        with tf.variable_scope('privacy'):

            out_p = self.fs_layer(self.data_p, 512, bias=True)
            #out_p = tf.nn.dropout(out_p, keep_prob = self.dropout)
            out_p = self.fs_layer(out_p, 512, bias=True)
            #out_p = tf.nn.dropout(out_p, keep_prob = self.dropout)
            #out_p = self.fs_layer(out_p, 512, bias=True)
            #out_p = tf.nn.dropout(out_p, keep_prob = self.dropout)
            self.out_p = self.fs_layer(out_p, 128, bias=True)
            self.pre_logit_p = self.fs_layer(self.out_p, 10, activation=None, bias = True)
            self.prob_p = tf.nn.softmax(self.pre_logit_p)

        #self.accuracy = 
        self.loss_u = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels= self.one_hot_u, logits=self.pre_logit_u))
        self.loss_p = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels= self.one_hot_p, logits=self.pre_logit_p))


        self.theta_u = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='utility')
        self.theta_p = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='privacy')

        uti_update_u = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='utility')
        with tf.control_dependencies(uti_update_u):
            op_u = tf.train.AdamOptimizer(0.0001)
            self.opt_u = op_u.minimize(self.loss_u, var_list = self.theta_u)

        uti_update_p = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='privacy')
        with tf.control_dependencies(uti_update_p):
            op_p = tf.train.AdamOptimizer(0.0001)
            self.opt_p = op_p.minimize(self.loss_p, var_list = self.theta_p)


        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

    def next_batch(self, t_data, t_label_u, t_label_p, batch_size, shuffle=False):

        le = len(t_data)
        epo = le // batch_size
        leftover = le - epo * batch_size
        sup = batch_size - leftover

        if shuffle : 
            c = list(zip(t_data, t_label_U, t_label_p))
            random.shuffle(c)
            t_data , t_label_u, t_label_p = zip(*c)

        for i in range(0, le, batch_size):

            if i ==  (epo *batch_size) : 
                yield np.array(t_data[i:]) , np.array(t_label_u[i:]), np.array(t_label_p[i:])
            else : 
                yield np.array(t_data[i: i+self.batch_size]), np.array(t_label_u[i: i+self.batch_size]), np.array(t_label_p[i: i+self.batch_size])



    def train(self):

        for _ in range(self.epo):
            for i, j, k in self.next_batch(self.k_dca_data_train, self.t_label_u, self.t_label_p, self.batch_size):
                feed_dict = {}
                feed_dict[self.data_p] = i 
                feed_dict[self.data_u] = i
                feed_dict[self.label_u] = j
                feed_dict[self.label_p] = k 
                feed_dict[self.dropout] = 0.5 
                loss, _, _ = self.sess.run([self.loss_u, self.opt_u, self.opt_p], feed_dict=feed_dict)
                #print(loss)



            print("********Training set Evaluate*************")
            self.evaluate(self.k_dca_data_train, self.t_label_u, self.t_label_p)
            
            print("********Validation set Evaluate*************")
            self.evaluate(self.k_dca_data_val, self.v_label_u, self.v_label_p)
            #self.evaluate(self.dca_data_val, self.v_label_u, self.v_label_p)

            print("********Testing set Evaluate*************")
            self.evaluate(self.k_dca_data, self.te_label_u, self.te_label_p)
            #self.evaluate(self.dca_data, self.te_label_u, self.te_label_p)
            
    def svm_pred(self):

        #gamma = [1, 0.1, 0.01, 0.001, 0.0001]
        gamma = [i/10 for i in range(10)]
        C = [i for i in range(50)]

        while True: 
            g = random.sample(gamma, 1)[0]
            c = random.sample(C, 1)[0]
            clf_u = SVC(C=c, gamma=g)
            clf_p = SVC(C=c, gamma=g)
            clf_u.fit(self.k_dca_data_train, self.t_label_u) 
            clf_p.fit(self.k_dca_data_train, self.t_label_p) 

            print("gamma Value: {}.".format(g))
            print("C value: {}.".format(c))
            print("********Validation set Evaluate*************")
            predict_u = clf_u.predict(self.k_dca_data_val)
            predict_p = clf_p.predict(self.k_dca_data_val)
            print("Utility accuracy: {}".format(accuracy_score(predict_u, np.array(self.v_label_u).reshape(-1))))
            print("Privacy accuracy: {}".format(accuracy_score(predict_p, np.array(self.v_label_p).reshape(-1))))

            print("********Testing set Evaluate*************")

            predict_u = clf_u.predict(self.k_dca_data)
            predict_p = clf_p.predict(self.k_dca_data)
            print("Utility accuracy: {}".format(accuracy_score(predict_u, np.array(self.te_label_u).reshape(-1))))
            print("Privacy accuracy: {}".format(accuracy_score(predict_p, np.array(self.te_label_p).reshape(-1))))


    def evaluate(self, dca_data, te_label_u, te_label_p): 

        temp_u = []
        temp_p = []

        for i, j, k in self.next_batch(dca_data, te_label_u, te_label_p, self.batch_size):
            
            feed_dict = {}
            feed_dict[self.data_p] = i 
            feed_dict[self.data_u] = i
            feed_dict[self.label_u] = j
            feed_dict[self.label_p] = k     
            feed_dict[self.dropout] = 1
            uu, pp = self.sess.run([self.prob_u, self.prob_p], feed_dict=feed_dict)

            temp_u.append(uu)
            temp_p.append(pp)


        prob_u = np.concatenate(temp_u, axis=0)
        prob_p = np.concatenate(temp_p, axis=0)
        predict_u = np.argmax(prob_u, axis=1)
        predict_p = np.argmax(prob_p, axis=1)

        print("Utility accuracy: {}".format(accuracy_score(predict_u, np.array(te_label_u).reshape(-1))))
        print("Privacy accuracy: {}".format(accuracy_score(predict_p, np.array(te_label_p).reshape(-1))))



