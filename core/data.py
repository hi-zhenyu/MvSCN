'''
load and preprocess data
'''

import numpy as np
import scipy.io as sio


from keras import backend as K
from keras.models import model_from_json

from core import util
from core import pairs

def get_data(params):
    '''
    load data
    data_list contains the multi-view data for SiameseNet and MvSCN
    '''
    # data list (views)
    data_list = []

    if params['views'] is None:
        params['views']  = range(1, params['view_size']+1)

    for i in params['views']:
        view_name = 'view'+str(i)
        print('********', 'load', params['dset'], view_name, '********')
        ret = {}
        x_train, y_train, x_test, y_test = load_data(params, i)
        print('data size (training, testing)', x_train.shape, x_test.shape)

        # Using the low-dimension data via AE
        if params['use_code_space']:
            all_data = [x_train, x_test]
            for j, d in enumerate(all_data):
                all_data[j] = embed_data(d, dset=params['dset'] , view_name=view_name)
            x_train, x_test = all_data

        # data for MvSCN
        ret['spectral'] = (x_train, y_train, x_test, y_test)
       
        # prepare the training pairs for SiameseNet
        ret['siamese'] = {}
        pairs_train, dist_train = pairs.create_pairs_from_unlabeled_data(
            x1=x_train,
            k=params['siam_k'],
            tot_pairs=params['siamese_tot_pairs'],
        )
        # data for SiameseNet
        ret['siamese'] = (pairs_train, dist_train)
    
        data_list.append(ret)

    return data_list

def load_data(params, view):
    '''
    load data for training and testing
    '''
    #  multi-view
    if params['dset'] == 'noisymnist':
        data = util.load_data('noisymnist_view'+str(view)+'.gz', 'https://www2.cs.uic.edu/~vnoroozi/noisy-mnist/noisymnist_view'+str(view)+'.gz')
        train_set_x, train_set_y = data[0]
        valid_set_x, valid_set_y = data[1]
        test_set_x, test_set_y = data[2]
        train_set_x, train_set_y = np.concatenate((train_set_x, valid_set_x), axis=0), np.concatenate((train_set_y, valid_set_y), axis=0)
    elif params['dset'] == 'Caltech101-20':
        mat = sio.loadmat('./data/'+params['dset']+'.mat')
        X = mat['X'][0]
        x = X[view-1]
        x = util.normalize(x)
        y = np.squeeze(mat['Y'])

        # split it into two partitions
        data_size = x.shape[0]
        train_index, test_index = util.random_index(data_size, int(data_size*0.5), 1)
        test_set_x = x[test_index]
        test_set_y = y[test_index]
        train_set_x = x[train_index]
        train_set_y = y[train_index]
    else:
        raise ValueError('Dataset provided ({}) is invalid!'.format(params['dset']))

    return train_set_x, train_set_y, test_set_x, test_set_y

def embed_data(x, dset, view_name):
    '''
    obtain the AE features
    '''
    if x is None:
        return None
    if not x.shape[0]:
        return np.zeros(shape=(0, 10))


    # use pretrained autoencoder from DEC
    json_path = './pretrain/ae/'+dset+'/ae_{}.json'.format(dset+'_'+view_name)
    weights_path = './pretrain/ae/'+dset+'/ae_{}_weights.h5'.format(dset+'_'+view_name)

    with open(json_path) as f:
        pt_ae = model_from_json(f.read())
    pt_ae.load_weights(weights_path)
    x = x.reshape(-1, np.prod(x.shape[1:]))

    get_embeddings = K.function([pt_ae.input],
                                  [pt_ae.layers[4].output])
    x_embedded = predict_with_K_fn(get_embeddings, x)[0]

    del pt_ae

    return x_embedded

def predict_with_K_fn(K_fn, x, bs=1000):
    '''
    Convenience function: evaluates x by K_fn(x), where K_fn is
    a Keras function, by batches of size 1000.
    '''
    if not isinstance(x, list):
        x = [x]
    num_outs = len(K_fn.outputs)
    y = [np.empty((x[0].shape[0], output_.get_shape()[1])) for output_ in K_fn.outputs]
    recon_means = []
    for i in range(int(x[0].shape[0]/bs + 1)):
        x_batch = []
        for x_ in x:
            x_batch.append(x_[i*bs:(i+1)*bs])
        temp = K_fn(x_batch)
        for j in range(num_outs):
            y[j][i*bs:(i+1)*bs] = temp[j]

    return y

