
# coding: utf-8

# In[1]:

#prepare netflix data as an input to to cuMF
#data should be in ./data/netflix/
#assume input is given in text format
#each line is like 
#"user_id item_id rating"
import os
import pandas as pd
from six.moves import urllib
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import csc_matrix
from scipy.sparse import coo_matrix
from scipy import sparse
from sklearn.cross_validation import train_test_split


# In[2]:

# Step 1: Download the data.
url = 'http://files.grouplens.org/datasets/movielens/'

def maybe_download(filename, expected_bytes):
  """Download a file if not present, and make sure it's the right size."""
  if not os.path.exists(filename):
    filename, _ = urllib.request.urlretrieve(url + filename, filename)
  statinfo = os.stat(filename)
  if statinfo.st_size == expected_bytes:
    print('Found and verified', filename)
  else:
    print(statinfo.st_size)
    raise Exception(
        'Failed to verify ' + filename + '. Can you get to it with a browser?')
  return filename

data_file = maybe_download('ml-10m.zip', 65566137)


# In[3]:

get_ipython().system(u'unzip -o ml-10m.zip')
#!cd ./ml-10M100K && ./split_ratings.sh


# In[4]:

#file look like
'''
1::122::5::838985046
1::185::5::838983525
1::231::5::838983392
1::292::5::838983421
1::316::5::838983392
1::329::5::838983392
1::355::5::838984474
1::356::5::838983653
1::362::5::838984885
1::364::5::838983707
'''
m = 71567
n = 65133


# In[5]:

user,item,rating, ts = np.loadtxt('ml-10M100K/ratings.dat', delimiter='::', dtype=np.int32,unpack=True)
print user
print item
print rating
print np.max(user)
print np.max(item)
print np.max(rating)
print user.size


# In[6]:

user_item = np.vstack((user, item))


# In[7]:

user_item_train, user_item_test, rating_train, rating_test = train_test_split(user_item.T, rating, test_size=1000006, random_state=42)
nnz_train = 9000048
nnz_test = 1000006


# In[8]:

#for test data, we need COO format to calculate test RMSE
#1-based to 0-based
R_test_coo = coo_matrix((rating_test,(user_item_test[:,0] - 1,user_item_test[:,1] - 1)))
assert R_test_coo.nnz == nnz_test
R_test_coo.data.astype(np.float32).tofile('R_test_coo.data.bin')
R_test_coo.row.tofile('R_test_coo.row.bin')
R_test_coo.col.tofile('R_test_coo.col.bin')


# In[9]:

print np.max(R_test_coo.data)
print np.max(R_test_coo.row)
print np.max(R_test_coo.col)
print R_test_coo.data
print R_test_coo.row
print R_test_coo.col


# In[10]:

test_data = np.fromfile('R_test_coo.data.bin',dtype=np.float32)
test_row = np.fromfile('R_test_coo.row.bin', dtype=np.int32)
test_col = np.fromfile('R_test_coo.col.bin',dtype=np.int32)
print test_data[0:10]
print test_row[0:10]
print test_col[0:10]


# In[11]:

#1-based to 0-based
R_train_coo = coo_matrix((rating_train,(user_item_train[:,0] - 1,user_item_train[:,1] - 1)))


# In[12]:

print R_train_coo.data
print R_train_coo.row
print R_train_coo.col
print np.max(R_train_coo.data)
print np.max(R_train_coo.row)
print np.max(R_train_coo.col)


# In[13]:

print np.unique(user).size
print np.unique(R_train_coo.row + 1).size
print np.unique(item).size
print np.unique(R_train_coo.col + 1).size

print np.unique(R_test_coo.row + 1).size
print np.unique(R_test_coo.col + 1).size


# In[14]:

np.min(R_test_coo.col)


# In[15]:

#for training data, we need COO format to calculate training RMSE
#we need CSR format R when calculate X from \Theta
#we need CSC format of R when calculating \Theta from X
assert R_train_coo.nnz == nnz_train
R_train_coo.row.tofile('R_train_coo.row.bin')


# In[16]:

R_train_csr = R_train_coo.tocsr()
R_train_csc = R_train_coo.tocsc()
R_train_csr.data.astype(np.float32).tofile('R_train_csr.data.bin')
R_train_csr.indices.tofile('R_train_csr.indices.bin')
R_train_csr.indptr.tofile('R_train_csr.indptr.bin')
R_train_csc.data.astype(np.float32).tofile('R_train_csc.data.bin')
R_train_csc.indices.tofile('R_train_csc.indices.bin')
R_train_csc.indptr.tofile('R_train_csc.indptr.bin')


# In[17]:

print R_train_csr.data
print R_train_csr.indptr
print R_train_csr.indices


# In[ ]:



