
# coding: utf-8

# In[1]:

import tensorflow as tf
lib_path = './als_tf.so'
als_module = tf.load_op_library(lib_path)


# In[2]:

import numpy as np
#input data
csrVal = np.fromfile('../data/netflix/R_train_csr.data.bin',dtype=np.float32)
csrCol = np.fromfile('../data/netflix/R_train_csr.indices.bin',dtype=np.int32)
csrRow = np.fromfile('../data/netflix/R_train_csr.indptr.bin',dtype=np.int32)

cscVal = np.fromfile('../data/netflix/R_train_csc.data.bin',dtype=np.float32)
cscRow = np.fromfile('../data/netflix/R_train_csc.indices.bin',dtype=np.int32)
cscCol = np.fromfile('../data/netflix/R_train_csc.indptr.bin',dtype=np.int32)

cooRow = np.fromfile('../data/netflix/R_train_coo.row.bin',dtype=np.int32)

cooRowTest = np.fromfile('../data/netflix/R_test_coo.row.bin',dtype=np.int32)
cooColTest = np.fromfile('../data/netflix/R_test_coo.col.bin',dtype=np.int32)
cooValTest = np.fromfile('../data/netflix/R_test_coo.data.bin',dtype=np.float32)


# In[3]:

m = 17770
n = 480189
f = 100
nnz = 99072112
nnz_test = 1408395
llambda = 0.048
iters =10
xbatch = 1
thetabatch = 3


# In[4]:

with tf.device('/cpu:0'):
    hello = tf.constant('Hello, TensorFlow!')
    [thetaT,XT, rmse] = als_module.do_als(csrRow, csrCol, csrVal, cscRow, cscCol, cscVal, cooRow, 
                                          cooRowTest, cooColTest, cooValTest, m, n, f, nnz, nnz_test, 
                                          llambda, iters,xbatch, thetabatch, 0)
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
with sess:
    print sess.run(hello)    
    print sess.run(rmse)


# In[ ]:



