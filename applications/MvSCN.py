import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from keras.layers import Input

from . import Clustering
from core import networks


def run_net(data_list, config, save=True):

    # network input shapes stored as view-list
    MvSCN_input_shape = []
    SiameseNet_input_shape = []

    # training and testing data stored as view-list
    x_train_list = []
    x_test_list = []

    # get the labels
    _, y_train, _, y_test = data_list[0]['spectral']

    batch_sizes = {
        'Embedding': config['batch_size'],
        'Orthogonal': config['batch_size_orthogonal'],
    }

    for i in range(config['view_size']):
        data_i = data_list[i]
        x_train_i, y_train_i, x_test_i, y_test_i = data_i['spectral']
        x = x_train_i

        x_test_list.append(x_test_i)
        x_train_list.append(x_train_i)

        # SiameseNet training pairs
        pairs_train, dist_train = data_i['siamese']

        # input shape
        input_shape = x.shape[1:]
        inputs = {
                'Embedding': Input(shape=input_shape,name='EmbeddingInput'+'view'+str(i)),
                'Orthogonal': Input(shape=input_shape,name='OrthogonalInput'+'view'+str(i)),
                }
        MvSCN_input_shape.append(inputs)

        print('******** SiameseNet ' + 'view'+str(i+1)+' ********')
        siamese_net = networks.SiameseNet(name=config['dset']+'_'+'view'+str(i+1),
                inputs=inputs,
                arch=config['arch'], 
                siam_reg=config['siam_reg'])
        history = siamese_net.train(pairs_train=pairs_train,
                dist_train=dist_train,
                lr=config['siam_lr'], 
                drop=config['siam_drop'], 
                patience=config['siam_patience'],
                num_epochs=config['siam_epoch'], 
                batch_size=config['siam_batch_size'], 
                pre_train=config['siam_pre_train'])

        SiameseNet_input_shape.append(siamese_net)

    # MvSCN
    mvscn = networks.MvSCN(input_list=MvSCN_input_shape,
                           arch=config['arch'],
                           spec_reg=config['spectral_reg'],
                           n_clusters=config['n_clusters'],
                           scale_nbr=config['scale_nbr'],
                           n_nbrs=config['n_nbrs'],
                           batch_sizes=batch_sizes,
                           view_size=config['view_size'],
                           siamese_list=SiameseNet_input_shape,
                           x_train_siam=x_train_list,
                           lamb=config['lamb'])

    # training
    print('********', 'Training', '********')
    mvscn.train(x_train=x_train_list,
        lr=config['spectral_lr'], 
        drop=config['spectral_drop'], 
        patience=config['spectral_patience'],
        num_epochs=config['spectral_epoch'])

    print("Training finished ")

    print('********', 'Prediction', '********')
    print('Testing data')
    # get final testing embeddings
    x_test_final_list = mvscn.predict(x_test_list)
    print('Learned representations for testing (view size, nsamples, dimension)', x_test_final_list.shape)
    
    # clustering
    y_preds, scores= Clustering.Clustering(x_test_final_list, y_test)

    return x_test_final_list, scores


