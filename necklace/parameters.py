from collections import namedtuple

from sparse import LISTA

RunParameters = namedtuple('RunParameters', ' sparse_algorithm'
                                            ' input_shape'
                                            ' dimension'
                                            ' tied_weight'
                                            ' necklace_link'
                                            ' residual_link'
                                            ' batch_normalization'
                                            ' clip_unsupervised_output'
                                            ' clip_supervised_output'
                                            ' dictionary_init'
                                            ' normalize_dictionary_after_epoch'
                                            ' clip_gradient'
                                            ' load_pretrain_unsupervised'
                                            ' load_pretrain_supervised'
                                            ' test_model'
                                            ' p_input p_weight batch_norm_axes'
                                            ' additional_sparse_params'
                                            ' update_lr'
                                            ' batch_size'
                                            ' losses_ratio'
                                            ' unsupervised_cost_fun'
                                            ' supervised_cost_fun'
                                            ' sparse_regularizer_type'
                                            ' sparse_regularize_factor'
                                            ' num_epochs'
                                            ' run_index ')
batch_size = 1000
run_parameters = RunParameters(
    LISTA,
    (batch_size, 784),
    # Set the dimension here, 1 list = 1 stack, 2 list = 2 stacks, etc...
    # dimensions = [[1500, 3, 200]]  # example of 1 stack
    [[200, 3, 50]],
    True,
    False,  # learning easier but makes A1 becomes 0
    False,  # learning easier but makes A1 becomes 0
    True,  # batch norm hoc dictionary khong duoc
    None,
    None,
    False,
    None,
    [0, 0.1],
    False,
    False,
    False,
    0.0, 0.8, 0,
    [None, None, False],
    0.0001,  # 0.0001 seems to be a really nice learning rate
    batch_size,
    # Set learning ratio for unsupervised, supervised and weights regularization
    # [1, 0, 0.001, 0.0001],
    [1, 1.0, 0.001, 0.0005],
    # 'categorical_crossentropy' or 'squared_error'
    'squared_error',
    'categorical_crossentropy',
    1,
    0.01,  # 0.05 = perfect location
    10000,
    4
)

N_TRAIN = 50000
N_VALID = 10000
N_LABELED = 50000
N_TEST = 10000
