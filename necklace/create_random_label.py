import cPickle as pickle
import os

import numpy as np
import theano
# File to keep where the each file should be stored
from path_settings import DATA_PATH

N_TRAIN = 50000
N_VALID = 10000
N_LABELED = 15000
permute = np.random.permutation(N_TRAIN)
labeled = np.zeros((N_TRAIN, 1), dtype=theano.config.floatX)
labeled[permute] = 1

with open(os.path.join(DATA_PATH,'labeled_index.pkl'), 'w') as f:
    pickle.dump([N_VALID, labeled], f)
