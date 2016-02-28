from collections import namedtuple

RunParameters = namedtuple('RunParameters', 'input_shape'
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
                                            ' run_index')
batch_size = 500
run_parameters = RunParameters(
    (batch_size, 784),
    # Set the dimension here, 1 list = 1 stack, 2 list = 2 stacks, etc...
    # dimensions = [[1500, 3, 200]]  # example of 1 stack
    [[1500, 3, 500]],
    True,
    True,
    True,
    True,
    0.2, 0.5, 0,
    [None, None, False],
    0.00001,
    batch_size,
    # Set learning ratio for unsupervised, supervised and weights regularization
    [1.0, 1.0, 1e-3],
    # 'categorical_crossentropy' or 'squared_error'
    'categorical_crossentropy',
    0
)

N_TRAIN = 50000
N_VALID = 10000
N_LABELED = 10000
N_TEST = 10000
