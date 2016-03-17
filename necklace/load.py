import cPickle as pickle
import os

import numpy as np
# File to keep where the each file should be stored
from path_settings import DATA_PATH, DICTIONARY_INIT_PATH
from parameters import N_TRAIN, N_VALID, N_TEST
from sklearn import preprocessing


def one_hot(x, n):
    if type(x) == list:
        x = np.array(x)
    x = x.flatten()
    o_h = np.zeros((len(x), n))
    o_h[np.arange(len(x)), x] = 1
    return o_h


def mnist(ntrain=N_TRAIN, nvalid=N_VALID, ntest=N_TEST, onehot=True, normalize_axes=0, ndim=2):
    f = open(os.path.join(DATA_PATH, 'mnist.pkl'))
    loaded_objs = pickle.load(f)
    f.close()
    f = open(os.path.join(DATA_PATH, 'valid_index.pkl'))
    valid_mask = pickle.load(f)
    f.close()
    trX = np.asarray(loaded_objs[0][0])
    trY = np.asarray(loaded_objs[0][1])
    vlX = np.asarray(loaded_objs[1][0])
    vlY = np.asarray(loaded_objs[1][1])
    teX = np.asarray(loaded_objs[2][0])
    teY = np.asarray(loaded_objs[2][1])

    # NEW IMPLEMENTATION
    vlX = np.concatenate((trX, vlX), axis=0)
    vlY = np.concatenate((trY, vlY), axis=0)
    if normalize_axes is not None:
        vlX = preprocessing.scale(vlX, axis=normalize_axes)
        teX = preprocessing.scale(teX, axis=normalize_axes)
    trX = np.delete(vlX, valid_mask, axis=0)
    trY = np.delete(vlY, valid_mask, axis=0)
    vlX = vlX[valid_mask]
    vlY = vlY[valid_mask]
    # END

    if onehot:
        trY = one_hot(trY, 10)
        vlY = one_hot(vlY, 10)
        teY = one_hot(teY, 10)

    if ndim == 3:
        trX = np.reshape(trX, (trX.shape[0], 1, trX.shape[1]))
        vlX = np.reshape(vlX, (vlX.shape[0], 1, vlX.shape[1]))
        teX = np.reshape(teX, (teX.shape[0], 1, teX.shape[1]))
    # if onehot:
    # 	trY = np.reshape(trY, (trY.shape[0], 1, trY.shape[1]))
    # 	vlY = np.reshape(vlY, (vlY.shape[0], 1, vlY.shape[1]))
    # 	teY = np.reshape(teY, (teY.shape[0], 1, teY.shape[1]))

    return trX[0:ntrain], vlX[0:nvalid], teX[0:ntest], trY[0:ntrain], vlY[0:nvalid], teY[0:ntest]


def load_dictionary_init(n_each=250, normalize_axes=0):
    with open(DICTIONARY_INIT_PATH, 'r') as f:
        D = pickle.load(f)
    load_index = []
    for _ in range(10):
        _load_index = range(250 * _, 250 * _ + n_each);
        load_index = load_index + _load_index;
    D = D[load_index]
    if normalize_axes is not None:
        D = preprocessing.scale(D, axis=normalize_axes)
    return D
