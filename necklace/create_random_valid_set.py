import cPickle as pickle
import os

import numpy as np
# File to keep where the each file should be stored
from path_settings import DATA_PATH
from parameters import N_TRAIN, N_VALID

MODE = 'RANDOM'  # RANDOM or LAST
if MODE == 'RANDOM':
    valid_idx = np.random.permutation(N_VALID)
else:
    valid_idx = range(N_TRAIN, N_TRAIN + N_VALID)
with open(os.path.join(DATA_PATH, 'valid_index.pkl'), 'w') as f:
    pickle.dump(valid_idx, f)
