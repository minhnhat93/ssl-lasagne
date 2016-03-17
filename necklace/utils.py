import cPickle as pickle
import os

import numpy as np
from skimage.io import imsave

from load import mnist


def export_image_array(image_array, output_folder, prefix):
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    for _ in range(image_array.shape[0]):
        imsave(os.path.join(output_folder, '{}{}.png'.format(prefix, _)), image_array[_] / np.max(abs(image_array[_])))


def export_classification_info(prediction, target, output_folder, prefix):
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    for _ in range(len(prediction)):
        with open(os.path.join(output_folder, '{}{}.txt'.format(prefix, _)), 'w') as f:
            f.write("{} {}".format(target[_], prediction[_]))

def export_wrong_samples(input_file, output_folder):
    trX, vlX, teX, trY, vlY, teY = mnist(onehot=False, ndim=2)
    with open(input_file, 'r') as f:
        train_ws, train_c, valid_ws, valid_c, test_ws, test_c = pickle.load(f)
    trX = np.asarray(trX[train_ws], dtype=np.uint8).reshape((-1, 28, 28))
    trY = trY[train_ws]
    vlX = np.asarray(vlX[valid_ws], dtype=np.uint8).reshape((-1, 28, 28))
    vlY = vlY[valid_ws]
    teX = np.asarray(teX[test_ws], dtype=np.uint8).reshape((-1, 28, 28))
    teY = teY[test_ws]
    export_image_array(trX, os.path.join(output_folder, 'train'), "")
    export_classification_info(trY, train_c, os.path.join(output_folder, 'train'), "")
    export_image_array(vlX, os.path.join(output_folder, 'valid'), "")
    export_classification_info(vlY, valid_c, os.path.join(output_folder, 'valid'), "")
    export_image_array(teX, os.path.join(output_folder, 'test'), "")
    export_classification_info(teY, test_c, os.path.join(output_folder, 'test'), "")
