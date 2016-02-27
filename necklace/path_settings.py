from parameters import run_parameters

# Setup does not use this reference, adjust in setup.py if changing.
DATA_PATH = '../data'

BEST_MODEL_PATH = '../models/best' + str(run_parameters.run_index)
LAST_MODEL_PATH = '../models/last' + str(run_parameters.run_index)
