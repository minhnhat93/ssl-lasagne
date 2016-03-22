import os

from parameters import run_parameters

currentPath = os.path.dirname(os.path.realpath(__file__))
# Setup does not use this reference, adjust in setup.py if changing.
DATA_PATH = os.path.join(currentPath, '../data')
DICTIONARY_INIT_PATH = os.path.join(currentPath, '../data/dictionary.pkl')

BEST_MODEL_PATH = os.path.join(currentPath, '../models/best_' + str(run_parameters.run_index) + '.params')
LAST_MODEL_PATH = os.path.join(currentPath, '../models/last_' + str(run_parameters.run_index) + '.params')
WRONG_SAMPLES_PATH = os.path.join(currentPath, '../models/wrong_' + str(run_parameters.run_index))
UNSUPERVISED_PRETRAIN_PATH = os.path.join(currentPath, '../data/unsupervised_pretrain.pkl')
SUPERVISED_PRETRAIN_PATH = os.path.join(currentPath, '../data/supervised_pretrain.pkl')
