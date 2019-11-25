'''
Experiments on Caltech101-20 and NoisyMNIST.
'''

import os
import sys
# add directories in src/ to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))
# set cuda
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from applications.MvSCN import run_net
from core.Config import load_config
from core.data import get_data


# load config for NoisyMNIST 
config = load_config('./config/noisymnist.yaml')

# load config for Caltech101-20
# config = load_config('./config/Caltech101-20.yaml')

# use pretrained SiameseNet. 
config['siam_pre_train'] = True

# LOAD DATA
data_list = get_data(config)

# RUN EXPERIMENT
x_final_list, scores = run_net(data_list, config)
