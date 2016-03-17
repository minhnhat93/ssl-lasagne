import numpy as np

from load import mnist

trX, vlX, _, _, _, _ = mnist()
trX = np.concatenate((trX, vlX))
