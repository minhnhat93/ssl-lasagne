import Image
import cPickle as pickle
import os

import numpy as np

from load import mnist


def export_wrong_samples(input_file, output_folder):
    trX, vlX, teX, trY, vlY, teY = mnist(onehot=False, ndim=2)
    with open(input_file, 'r') as f:
        train_ws, train_c, valid_ws, valid_c, test_ws, test_c = pickle.load(f)
    trX = np.asarray(trX[train_ws] * 255.0, dtype=np.uint8).reshape((-1, 28, 28))
    trY = trY[train_ws]
    vlX = np.asarray(vlX[valid_ws] * 255.0, dtype=np.uint8).reshape((-1, 28, 28))
    vlY = vlY[valid_ws]
    teX = np.asarray(teX[test_ws] * 255.0, dtype=np.uint8).reshape((-1, 28, 28))
    teY = teY[test_ws]
    path = os.path.join(output_folder, 'train')
    if not os.path.exists(path):
        os.mkdir(path)
    for _ in range(len(train_ws)):
        Image.fromarray(trX[_], 'L').save(os.path.join(path, '{}.png'.format(_)))
        with open(os.path.join(path, '{}.txt'.format(_)), 'w') as f:
            f.write("{} {}".format(trY[_], train_c[_]))
    path = os.path.join(output_folder, 'valid')
    if not os.path.exists(path):
        os.mkdir(path)
    for _ in range(len(valid_ws)):
        Image.fromarray(vlX[_], 'L').save(os.path.join(path, '{}.png'.format(_)))
        with open(os.path.join(path, '{}.txt'.format(_)), 'w') as f:
            f.write("{} {}".format(vlY[_], valid_c[_]))
    path = os.path.join(output_folder, 'test')
    if not os.path.exists(path):
        os.mkdir(path)
    for _ in range(len(test_ws)):
        Image.fromarray(teX[_], 'L').save(os.path.join(path, '{}.png'.format(_)))
        with open(os.path.join(path, '{}.txt'.format(_)), 'w') as f:
            f.write("{} {}".format(teY[_], test_c[_]))


export_wrong_samples('../models/wrong_0', '/home/nhat/wrong_samples')
