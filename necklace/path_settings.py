import os

from parameters import run_parameters

currentPath = os.path.dirname(os.path.realpath(__file__))
# Setup does not use this reference, adjust in setup.py if changing.
DATA_PATH = os.path.join(currentPath, '../data')

BEST_MODEL_PATH = os.path.join(currentPath, '../models/best' + str(run_parameters.run_index))
LAST_MODEL_PATH = os.path.join(currentPath, '../models/last' + str(run_parameters.run_index))
