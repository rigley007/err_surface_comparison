'''This example demonstrates the use of Convolution1D for text classification.

Gets to 0.89 test accuracy after 2 epochs.
90s/epoch on Intel i5 2.4Ghz CPU.
10s/epoch on Tesla K40 GPU.
'''
from __future__ import print_function

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.datasets import imdb
import numpy as np
from keras.callbacks import LambdaCallback
from keras import backend as K
from sklearn.utils import resample
import matplotlib.pyplot as plt


# set parameters:
max_features = 5000
maxlen = 400
batch_size = 128
embedding_dims = 50
filters = 250
kernel_size = 3
hidden_dims = 250
epochs = 26


import pickle
#filename = 'pred15_finalized_reg_model.sav'
#prediction_model = pickle.load(open(filename, 'rb'))


num_callback = 6
l = np.array([])
m_grad = np.array([])
temp_a = np.array([])

weights_log = np.array([])

weight_hist_length = 25




def weight_prediciton():
    global l
    global m_grad
    temp_w = model.get_weights()
    '''
    grads = model.optimizer.get_gradients(model.total_loss, model.trainable_weights)
    symb_inputs = (model._feed_inputs + model._feed_targets + model._feed_sample_weights)
    f = K.function(symb_inputs, grads)
    x, y, sample_weight = model._standardize_user_data(x_train_re, y_train_re)
    temp_gradients = f(x + y + sample_weight)
    '''
    if l.shape[0] == 0:
        l = np.append(l, temp_w)
        m_grad = np.append(m_grad, temp_gradients[1:7])
    else:
        l = np.vstack([l, temp_w])
        m_grad = np.vstack([m_grad, temp_gradients[1:7]])

print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
x_train_re,y_train_re=resample(x_train,y_train, n_samples=2500, random_state=1)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

print('Build model...')
model = Sequential()

# we start off with an efficient embedding layer which maps
# our vocab indices into embedding_dims dimensions
model.add(Embedding(max_features,
                    embedding_dims,
                    input_length=maxlen))
model.add(Dropout(0.2))

# we add a Convolution1D, which will learn filters
# word group filters of size filter_length:
model.add(Conv1D(filters,
                 kernel_size,
                 padding='valid',
                 activation='relu',
                 strides=1))
# we use max pooling:
model.add(GlobalMaxPooling1D())

# We add a vanilla hidden layer:
model.add(Dense(hidden_dims))
model.add(Dropout(0.2))
model.add(Activation('relu'))

# We project onto a single unit output layer, and squash it with a sigmoid:
model.add(Dense(1))
model.add(Activation('sigmoid'))

update_weights = LambdaCallback(on_epoch_end=lambda batch, logs: weight_prediciton())

model.compile(loss='mean_squared_error',
              optimizer='sgd',
              metrics=['accuracy'])
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test),
          callbacks = [update_weights])


temp_weight_pool = np.array([])
temp_grad_pool = np.array([])
for i in range(epochs):
    temp_sub_weight_pool = np.array([])
    temp_sub_grad_pool = np.array([])
    for j in range(l.shape[1]):
        temp_sub_weight_pool = np.append(temp_sub_weight_pool, l[i][j].flatten())
    if i == 0:
        temp_weight_pool = np.append(temp_weight_pool, temp_sub_weight_pool)
    else:
        temp_weight_pool = np.vstack([temp_weight_pool, temp_sub_weight_pool])

    for j in range(m_grad.shape[1]):
        temp_sub_grad_pool = np.append(temp_sub_grad_pool, m_grad[i][j].flatten())

    if i == 0:
        temp_grad_pool = np.append(temp_grad_pool, temp_sub_grad_pool)
    else:
        temp_grad_pool = np.vstack([temp_grad_pool, temp_sub_grad_pool])


s_x = np.arange(-1.0, 1.0, 0.05)


e_begin = 20


scores = np.array([])
for x in range(e_begin, e_begin + 5):
    latest_weight = temp_weight_pool[x]
    temp_scores = np.array([])

    for temp_x in s_x:
        latest_weight[287721] = temp_x
        temp_n = 0
        for j in range(l.shape[1]):
            tmp_shape = l[0][j].shape
            tmp_length = l[0][j].flatten().shape[0]
            l[0][j] = np.reshape(latest_weight[temp_n:temp_n + tmp_length], tmp_shape)
            temp_n = temp_n + tmp_length

        model.set_weights(l[0])
        scores_temp = model.evaluate(x_test, y_test, verbose=1)
        temp_scores = np.append(temp_scores, scores_temp[0])

    if x == e_begin:
        scores = np.append(scores, temp_scores)
    else:
        scores = np.vstack([scores, temp_scores])



plt.show()
print("aaa")