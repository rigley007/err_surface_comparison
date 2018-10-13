'''Trains a simple deep NN on the MNIST dataset.

Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''

from __future__ import print_function

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from keras.callbacks import LambdaCallback
import numpy as np
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from sklearn.preprocessing import normalize
import os

batch_size = 128
num_classes = 10
epochs = 100
num_callback = 3

temp_a = np.array([])
weights_log = np.array([])
l = np.array([])
# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_mnist100_trained_model.h5'


def weight_prediciton():
    global l
    if l.shape[0] == 0:
        l = np.append(l, model.get_weights())
    else:
        l = np.vstack([l, model.get_weights()])


x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(784,)))
#print_weights_0 = LambdaCallback(on_epoch_end=lambda batch, logs: l.append(model.layers[0].get_weights()))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
#print_weights_1 = LambdaCallback(on_epoch_end=lambda batch, logs: l.append(model.layers[2].get_weights()))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))
print_weights_2 = LambdaCallback(on_epoch_end=lambda batch, logs: weight_prediciton())
model.summary()

model.compile(loss='mean_squared_error',
              optimizer='sgd',
              metrics=['accuracy'])

#RMSprop()

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test),
                    callbacks = [print_weights_2])
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
l = np.array(l)
'''
for x in range(epochs):
    t_base=x*num_callback
    temp_a = np.append(temp_a, l[t_base + 1][0].flatten())
    temp_a = np.append(temp_a, l[t_base + 2][0].flatten())
    temp_a = np.append(temp_a, l[t_base + 3][0].flatten())
    if x == 0:
        weights_log = [temp_a]
    else:
        weights_log=np.append(weights_log,[temp_a],axis=0)
    temp_a = np.array([])
'''
#tsne = TSNE(n_components=2, random_state=0)
#weights_log_t=weights_log.transpose()
#np.random.shuffle(weights_log_t)
#weights_log_t_del = np.delete(weights_log_t, slice(0, 660000), 0)
#X_2d = tsne.fit_transform(weights_log_t_del)
#import scipy.io
#scipy.io.savemat('mnistdnn_100_sgd_org.mat', {'weights_log':weights_log})

import pickle

# save:
f = open('mnistdnn_100_sgd_org.pckl', 'wb')
pickle.dump(weights_log, f)
f.close()


"""
plt.show()
plt.figure(figsize=(6, 5))
for x in range(10):
    plt.plot(X_2d[x * 200 + 50][0], X_2d[x * 200 + 50][1], "bo")
for x in range(10):
    plt.plot(X_2d[x * 200 + 51][0], X_2d[x * 200 + 51][1], "ro")
for x in range(10):
    plt.plot(X_2d[x * 200 + 52][0], X_2d[x * 200 + 52][1], "go")
plt.show()


for x in range(10):
    temp_m = normed_matrix[5 * x:5 * x + 5]
    if x == 0:
        weights_log = temp_m.transpose()
    else:
        weights_log = np.append(weights_log, temp_m.transpose(), axis=0)


normed_matrix = normalize(weights_log_del, norm='l2')
"""

# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)