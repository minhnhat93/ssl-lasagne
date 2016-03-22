import cPickle as pickle
import os

import numpy as np
import theano

# File to keep where the each file should be stored
from path_settings import DATA_PATH
from parameters import N_TRAIN, N_LABELED

permute = np.random.permutation(N_LABELED)
labeled = np.zeros((N_TRAIN, 1), dtype=theano.config.floatX)
labeled[permute] = 1

with open(os.path.join(DATA_PATH, 'labeled_index.pkl'), 'w') as f:
    pickle.dump(labeled, f)
