from collections import namedtuple

from sparse import LISTA

RunParameters = namedtuple('RunParameters', 'sparse_algorithm'
                                            ' input_shape'
                                            ' dimension'
                                            ' tied_weight'
                                            ' necklace_link'
                                            ' residual_link'
                                            ' batch_normalization'
                                            ' p_input p_weight norm_axes'
                                            ' additional_sparse_params'
                                            ' update_lr'
                                            ' batch_size'
                                            ' losses_ratio'
                                            ' supervised_cost_fun'
                                            ' sparse_regularize_factor'
                                            ' num_epochs'
                                            ' run_index')
batch_size = 1000
run_parameters = RunParameters(
    LISTA,
    (batch_size, 784),
    # Set the dimension here, 1 list = 1 stack, 2 list = 2 stacks, etc...
    # dimensions = [[1500, 3, 200]]  # example of 1 stack
    [[1000, 2, 300]],
    True,
    True,
    False,
    True,
    0.2, 0.5, 0,
    [None, None, False],
    0.0001,  # 0.0001 seems to be a really nice learning rate
    batch_size,
    # Set learning ratio for unsupervised, supervised and weights regularization
    [1.0, 0, 0.000],
    # 'categorical_crossentropy' or 'squared_error'
    'categorical_crossentropy',
    0.01,
    5000,
    1
)

N_TRAIN = 50000
N_VALID = 10000
N_LABELED = 10000
N_TEST = 10000
