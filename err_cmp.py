from __future__ import print_function

import numpy as np
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors
import scipy.io
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
import pickle

filename = 'mnistdnn_100_sgd_org.pckl'
weights_log = pickle.load(open(filename, 'rb'))

weights_log_t=weights_log.transpose()
tempstd=np.std(weights_log_t,1)
idx=np.where(tempstd>0.005)

weights_log_del=weights_log_t[idx].transpose()



for x in range(95):
    print(x)
    temp_m = weights_log_del[x:x + 5]
    if x == 0:
        sliced_weights_log_dnn = temp_m.transpose()
    else:
        sliced_weights_log_dnn = np.append(sliced_weights_log_dnn, temp_m.transpose(), axis=0)

print("aaa")

"""
def weight_preprocess(l):
    temp_sub_weight_pool = np.array([])
    for j in range(6):a
        temp_sub_weight_pool = np.append(temp_sub_weight_pool, l[j].flatten())
    return  temp_sub_weight_pool

from multiprocessing.dummy import Pool as ThreadPool
pool = ThreadPool(6)


"""