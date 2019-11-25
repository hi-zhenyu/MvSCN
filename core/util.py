'''
utility functions used in the experiments.
'''


import gzip
import random

import numpy as np
from sklearn.neighbors import NearestNeighbors

import tensorflow as tf
from keras import backend as K
from keras.callbacks import Callback
from keras.utils.data_utils import get_file


def random_index(n_all, n_train, seed):
    '''
    generated random index
    '''
    random.seed(seed)
    random_index = random.sample(range(n_all), n_all)
    train_index = random_index[0:n_train]
    test_index = random_index[n_train:n_all]

    train_index = np.array(train_index)
    test_index = np.array(test_index)
    return train_index, test_index

def load_data(data_file, url):
    '''
    loads the data from the gzip pickled files, and converts to numpy arrays
    '''
    print('loading data ...')
    path = get_file(data_file, origin=url)
    f = gzip.open(path, 'rb')
    train_set, valid_set, test_set = load_pickle(f)
    f.close()

    train_set_x, train_set_y = make_numpy_array(train_set)
    valid_set_x, valid_set_y = make_numpy_array(valid_set)
    test_set_x, test_set_y = make_numpy_array(test_set)

    return [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)]

def load_pickle(f):
    '''
    loads and returns the content of a pickled file
    it handles the inconsistencies between the pickle packages available in Python 2 and 3
    '''
    try:
        import cPickle as thepickle
    except ImportError:
        import _pickle as thepickle

    try:
        ret = thepickle.load(f, encoding='latin1')
    except TypeError:
        ret = thepickle.load(f)

    return ret

def make_numpy_array(data_xy):
    '''
    converts the input to numpy arrays
    '''
    data_x, data_y = data_xy
    data_x = np.asarray(data_x, dtype='float64')
    data_y = np.asarray(data_y, dtype='int32')
    return data_x, data_y



def make_batches(size, batch_size):
    '''
    generates a list of (start_idx, end_idx) tuples for batching data
    of the given size and batch_size
    '''
    num_batches = (size + batch_size - 1) // batch_size  # round up
    return [(i * batch_size, min(size, (i + 1) * batch_size))
            for i in range(num_batches)]

def train_gen(pairs_train, dist_train, batch_size):
    '''
    Generator used for training the SiameseNet
    '''
    batches = make_batches(len(pairs_train), batch_size)
    while 1:
        random_idx = np.random.permutation(len(pairs_train))
        for batch_start, batch_end in batches:
            p_ = random_idx[batch_start:batch_end]
            x1, x2 = pairs_train[p_, 0], pairs_train[p_, 1]
            y = dist_train[p_]
            yield([x1, x2], y)

def make_layer_list(arch, network_type=None, reg=None, dropout=0):
    '''
    Generates the list of layers specified by arch, to be stacked
    by stack_layers
    '''
    layers = []
    for i, a in enumerate(arch):
        layer = {'l2_reg': reg}
        layer.update(a)
        if network_type:
            layer['name'] = '{}_{}'.format(network_type, i)
        layers.append(layer)
        if a['type'] != 'Flatten' and dropout != 0:
            dropout_layer = {
                'type': 'Dropout',
                'rate': dropout,
                }
            if network_type:
                dropout_layer['name'] = '{}_dropout_{}'.format(network_type, i)
            layers.append(dropout_layer)
    return layers

class LearningHandler(Callback):
    '''
    Class for managing the learning rate scheduling and early stopping criteria

    Learning rate scheduling is implemented by multiplying the learning rate
    by 'drop' everytime the validation loss does not see any improvement
    for 'patience' training steps
    '''
    def __init__(self, lr, drop, lr_tensor, patience):
        '''
        lr:         initial learning rate
        drop:       factor by which learning rate is reduced by the
                    learning rate scheduler
        lr_tensor:  tensorflow (or keras) tensor for the learning rate
        patience:   patience of the learning rate scheduler
        '''
        super(LearningHandler, self).__init__()
        self.lr = lr
        self.drop = drop
        self.lr_tensor = lr_tensor
        self.patience = patience

    def on_train_begin(self, logs=None):
        '''
        Initialize the parameters at the start of training (this is so that
        the class may be reused for multiple training runs)
        '''
        self.assign_op = tf.no_op()
        self.scheduler_stage = 0
        self.best_loss = np.inf
        self.wait = 0

    def on_epoch_end(self, epoch, logs=None):
        '''
        Per epoch logic for managing learning rate and early stopping
        '''
        stop_training = False
        # check if we need to stop or increase scheduler stage
        if isinstance(logs, dict):
            loss = logs['loss']
        else:
            loss = logs
        if loss <= self.best_loss:
            self.best_loss = loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait > self.patience:
                self.scheduler_stage += 1
                self.wait = 0

        # calculate and set learning rate
        lr = self.lr * np.power(self.drop, self.scheduler_stage)
        K.set_value(self.lr_tensor, lr)

        # built in stopping if lr is way too small
        if lr <= 1e-8:
            stop_training = True

        # for keras
        if hasattr(self, 'model') and self.model is not None:
            self.model.stop_training = stop_training

        return stop_training

def get_scale(x, batch_size, n_nbrs):
    '''
    Calculates the scale* based on the median distance of the kth
    neighbors of each point of x*, a m-sized sample of x, where
    k = n_nbrs and m = batch_size
    '''
    n = len(x)

    # sample a random batch of size batch_size
    sample = x[np.random.randint(n, size=batch_size), :]
    # flatten it
    sample = sample.reshape((batch_size, np.prod(sample.shape[1:])))

    # compute distances of the nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=n_nbrs).fit(sample)
    distances, _ = nbrs.kneighbors(sample)

    # return the median distance
    return np.median(distances[:, n_nbrs - 1])

def normalize(x):
    x = (x-np.min(x)) / (np.max(x)-np.min(x))
    return x
