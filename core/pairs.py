'''
generated data pairs from unlabeled data to training SiameseNet
'''

from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import random
from random import randint
from collections import defaultdict

from sklearn.neighbors import NearestNeighbors
from sklearn import metrics


def get_choices(arr, num_choices, valid_range=[-1, np.inf], not_arr=None, replace=False):
    '''
    Select n=num_choices choices from arr, with the following constraints for
    each choice:
        choice > valid_range[0],
        choice < valid_range[1],
        choice not in not_arr
    if replace == True, draw choices with replacement
    if arr is an integer, the pool of choices is interpreted as [0, arr]
    (inclusive)
        * in the implementation, we use an identity function to create the
        identity map arr[i] = i
    '''
    if not_arr is None:
        not_arr = []
    if isinstance(valid_range, int):
        valid_range = [0, valid_range]
    # make sure we have enough valid points in arr
    if isinstance(arr, tuple):
        if min(arr[1], valid_range[1]) - max(arr[0], valid_range[0]) < num_choices:
            raise ValueError("Not enough elements in arr are outside of valid_range!")
        n_arr = arr[1]
        arr0 = arr[0]
        arr = defaultdict(lambda: -1)
        get_arr = lambda x: x
        replace = True
    else:
        greater_than = np.array(arr) > valid_range[0]
        less_than = np.array(arr) < valid_range[1]
        if np.sum(np.logical_and(greater_than, less_than)) < num_choices:
            raise ValueError("Not enough elements in arr are outside of valid_range!")
        # make a copy of arr, since we'll be editing the array
        n_arr = len(arr)
        arr0 = 0
        arr = np.array(arr, copy=True)
        get_arr = lambda x: arr[x]
    not_arr_set = set(not_arr)
    def get_choice():
        arr_idx = randint(arr0, n_arr-1)
        while get_arr(arr_idx) in not_arr_set:
            arr_idx = randint(arr0, n_arr-1)
        return arr_idx
    if isinstance(not_arr, int):
        not_arr = list(not_arr)
    choices = []
    for _ in range(num_choices):
        arr_idx = get_choice()
        while get_arr(arr_idx) <= valid_range[0] or get_arr(arr_idx) >= valid_range[1]:
            arr_idx = get_choice()
        choices.append(int(get_arr(arr_idx)))
        if not replace:
            arr[arr_idx], arr[n_arr-1] = arr[n_arr-1], arr[arr_idx]
            n_arr -= 1
    return choices

def create_pairs_from_unlabeled_data(x1, k=5, tot_pairs=None, verbose=True):
    '''
    Generates positive and negative pairs for the SiameseNet networt from
    unlabeled data.
    '''

    n = len(x1)

    pairs_per_pt = max(1, min(k, int(tot_pairs/(n*2)))) if tot_pairs is not None else max(1, k)

    pairs = []
    pairs2 = []
    labels = []
    true = []
    
    if verbose:
        print('computing k={} nearest neighbors...'.format(k))

    if len(x1.shape)>2:
        x1_flat = x1.reshape(x1.shape[0], np.prod(x1.shape[1:]))[:n]
    else:
        x1_flat = x1[:n]


    nbrs = NearestNeighbors(n_neighbors=k+1).fit(x1_flat)
    _, Idx = nbrs.kneighbors(x1_flat)

    # for each row, remove the element itself from its list of neighbors
    # (we don't care that each point is its own closest neighbor)
    new_Idx = np.empty((Idx.shape[0], Idx.shape[1] - 1))
    assert (Idx >= 0).all()
    for i in range(Idx.shape[0]):
        try:
            new_Idx[i] = Idx[i, Idx[i] != i][:Idx.shape[1] - 1]
        except Exception as e:
            print(Idx[i, ...], new_Idx.shape, Idx.shape)
            raise e
    Idx = new_Idx.astype(np.int)
    k_max = min(Idx.shape[1], k+1)

    if verbose:
        print('creating pairs...')

    # pair generation loop (alternates between true and false pairs)
    consecutive_fails = 0
    for i in range(n):
        # get_choices sometimes fails with precomputed results. if this happens
        # too often, we relax the constraint on k
        if consecutive_fails > 5:
            k_max = min(Idx.shape[1], int(k_max*2))
            consecutive_fails = 0
        # if verbose and i % 10000 == 0:
        #     print("Iter: {}/{}".format(i,n))
        # pick points from neighbors of i for positive pairs
        try:
            choices = get_choices(Idx[i,:k_max], pairs_per_pt, replace=False)
            consecutive_fails = 0
        except ValueError:
            consecutive_fails += 1
            continue
        assert i not in choices
        # form the pairs
        new_pos = [[x1[i], x1[c]] for c in choices]
        # pick points *not* in neighbors of i for negative pairs
        try:
            choices = get_choices((0, n), pairs_per_pt, not_arr=Idx[i,:k_max], replace=False)
            consecutive_fails = 0
        except ValueError:
            consecutive_fails += 1
            continue
        # form negative pairs
        new_neg = [[x1[i], x1[c]] for c in choices]

        # add pairs to our list
        labels += [1]*len(new_pos) + [0]*len(new_neg)
        pairs += new_pos + new_neg

    # package return parameters for output
    ret = [np.array(pairs).reshape((len(pairs), 2) + x1.shape[1:])]
    ret.append(np.array(labels))

    return ret

