import cPickle as pickle
import os

import numpy as np
from sklearn.decomposition import MiniBatchDictionaryLearning

from load import mnist
from path_settings import DATA_PATH

trX, vlX, _, _, _, _ = mnist()
trX = np.concatenate((trX, vlX))
dico = MiniBatchDictionaryLearning(n_components=1500, alpha=1, batch_size=1000, n_iter=10)
learned_dictionary = dico.fit(trX)
with open(os.path.join(DATA_PATH, 'dictionary_init.pkl'), 'w') as f:
    pickle.dump(learned_dictionary, f, pickle.HIGHEST_PROTOCOL)
