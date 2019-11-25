'''
network definitions
'''

import itertools

import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Lambda

from . import train
from . import costs
from .layer import stack_layers
from .util import LearningHandler, make_layer_list, train_gen, get_scale


class SiameseNet:
    def __init__(self, name, inputs, arch, siam_reg):
        self.orig_inputs = inputs
        self.name = name
        # set up input shapes
        self.inputs = {
                'A': inputs['Embedding'],
                'B': Input(shape=inputs['Embedding'].get_shape().as_list()[1:]),
                }
        
        # generate layers
        self.layers = []
        self.layers += make_layer_list(arch, 'siamese', siam_reg)
        self.outputs = stack_layers(self.inputs, self.layers)

        # add the distance layer
        self.distance = Lambda(costs.euclidean_distance, output_shape=costs.eucl_dist_output_shape)([self.outputs['A'], self.outputs['B']])

        #create the distance model for training
        self.net = Model([self.inputs['A'], self.inputs['B']], self.distance)
        
        # compile the siamese network
        self.net.compile(loss=costs.get_contrastive_loss(m_neg=1, m_pos=0.05), optimizer='rmsprop')

    def train(self, pairs_train, dist_train,
            lr, drop, patience, num_epochs, batch_size, pre_train=False):
        # create handler for early stopping and learning rate scheduling
        self.lh = LearningHandler(
                lr=lr,
                drop=drop,
                lr_tensor=self.net.optimizer.lr,
                patience=patience)

        # initialize the training generator
        train_gen_ = train_gen(pairs_train, dist_train, batch_size)

        validation_data = None

        # compute the steps per epoch
        steps_per_epoch = int(len(pairs_train) / batch_size)

        if pre_train:
            print('Use pretrained weights')
            self.net.load_weights('./pretrain/siamese/'+self.name+'_siamese_weight'+'.h5')
            return 0
        else:
            # train the network
            hist = self.net.fit_generator(train_gen_, epochs=num_epochs, validation_data=validation_data, steps_per_epoch=steps_per_epoch, callbacks=[self.lh])
            self.net.save_weights('./pretrain/siamese/'+self.name+'_siamese_weight'+'.h5')
            return hist


    def predict(self, x, batch_sizes):
        # compute the siamese embeddings of the input data
        return train.predict_siamese(self.outputs['A'], x=x, inputs=self.orig_inputs, batch_sizes=batch_sizes)



class MvSCN:
    def __init__(self, input_list, arch, spec_reg,
            n_clusters, scale_nbr, n_nbrs, batch_sizes, view_size,
            siamese_list, x_train_siam, lamb):
        self.n_clusters = n_clusters
        self.input_list = input_list
        self.batch_sizes = batch_sizes
        self.view_size = view_size

        # Embedding inputs shape 
        self.input_shape = {'Embedding': [], 'Orthogonal': []}
        for i in range(self.view_size):
            self.input_shape['Embedding'].append(self.input_list[i]['Embedding'])
            self.input_shape['Orthogonal'].append(self.input_list[i]['Orthogonal'])

        self.output_shape = []
        for i in range(self.view_size):
            input_item = self.input_list[i]
            self.layers = make_layer_list(arch[:-1], 'Embedding_view'+str(i), spec_reg)
            self.layers += [
                    {'type': 'tanh',
                    'size': n_clusters,
                    'l2_reg': spec_reg,
                    'name': 'Embedding_'+'view'+str(i)+'_{}'.format(len(arch)-1)},
                    {'type': 'Orthogonal', 'name':'Orthogonal'+'_view'+str(i)}
                ]

            output = stack_layers(input_item, self.layers)
            self.output_shape.append(output['Embedding'])


        self.net = Model(inputs=self.input_shape['Embedding'], outputs=self.output_shape)

        # LOSS

        loss_1 = 0

        # W
        for (x_train, siamese_net, input_shape, output) in zip(x_train_siam, siamese_list, self.input_list, self.output_shape):
            # generate affinity matrix W according to params
            input_affinity = siamese_net.outputs['A']
            x_affinity = siamese_net.predict(x_train, batch_sizes)

            # calculate scale for affinity matrix
            scale = get_scale(x_affinity, self.batch_sizes['Embedding'], scale_nbr)

            # create affinity matrix
            W = costs.knn_affinity(input_affinity, n_nbrs, scale=scale, scale_nbr=scale_nbr)
            Dy = costs.squared_distance(output) 

            loss_1 = loss_1 + K.sum(W * Dy) / (batch_sizes['Embedding'] ** 2)
        
        loss_1 = loss_1 / self.view_size

        # LOSS 2
        loss_2 = 0

        # generate permutation of different views
        ls = itertools.combinations(self.output_shape, 2)
        

        for (view_i, view_j) in ls:
            loss_2 += K.sum(costs.pairwise_distance(view_i, view_j)) / (batch_sizes['Embedding'])
        
        self.loss = (1-lamb) * loss_1 + lamb * loss_2 / (view_size**2)

        # create the train step update
        self.learning_rate = tf.Variable(0., name='spectral_net_learning_rate')
        self.train_step = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate).minimize(self.loss, var_list=self.net.trainable_weights)

        # initialize spectralnet variables
        K.get_session().run(tf.variables_initializer(self.net.trainable_weights))
        
    def train(self, x_train, lr, drop, patience, num_epochs, x_test=None, y_test=None):
        # create handler for early stopping and learning rate scheduling
        self.lh = LearningHandler(
                lr=lr,
                drop=drop,
                lr_tensor=self.learning_rate,
                patience=patience)

        losses = np.empty((num_epochs,))
        val_losses = np.empty((num_epochs,))

        # begin spectralnet training loop
        self.lh.on_train_begin()
        for i in range(num_epochs):
            # train spectralnet
            losses[i] = train.train_step(
                    return_var=[self.loss],
                    updates=self.net.updates + [self.train_step],
                    x_unlabeled=x_train,
                    inputs=self.input_shape,
                    batch_sizes=self.batch_sizes,
                    batches_per_epoch=100)[0]

            # do early stopping if necessary
            if self.lh.on_epoch_end(i, losses[i]):
                print('STOPPING EARLY')
                return losses[:i]
            # print training status
            print("Epoch: {}, loss={:2f}".format(i, losses[i]))

        return losses[:i]

    def predict(self, x):
        return train.predict(
                    self.output_shape,
                    x_unlabeled=x,
                    inputs=self.input_shape,
                    batch_sizes=self.batch_sizes,
                    view_size=self.view_size)
