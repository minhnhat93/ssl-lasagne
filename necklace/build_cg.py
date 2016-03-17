import theano
from lasagne.init import GlorotUniform, Uniform, Constant
from lasagne.layers import InputLayer, DropoutLayer, DenseLayer, BatchNormLayer, batch_norm
from lasagne.nonlinearities import identity, softmax
from lasagne.utils import floatX

from load import load_dictionary_init
from otherlayers import LinearCombinationLayer, TransposedDenseLayer


def NecklaceNetwork(incoming, dimensions, SparseClass, additional_sparse_params, tied_weight=False, necklace_link=False,
                    residual_link=False, batch_normalization=False, norm_axes=0, p_weight=0.5, D_init=None):
    '''
    Implementation of a necklace network. See: https://drive.google.com/file/d/0B8EOfHp2L5mNbkY1UkhlWmF2YWc/view?ts=56b3a4fc
    :param dimensions: contain information of the dimension
    :param SparseClass: Sparse Code Algorithm class
    :param p_weight: dropout factor for weight matrices
    :param linear_combination_ratio: alpha factor in x' = alpha*a + (1-alpha)*a'
    :return:
    '''
    network = incoming
    num_input = incoming.output_shape[1]
    num_of_stacks = len(dimensions)
    D_list = []
    G_list = []
    sparse_layers = []
    feature_layers = []
    last_residual_layer = network
    stack_idx = 0
    for _ in range(num_of_stacks):
        stack_str = str(stack_idx)
        stack_idx += 1
        sparse_dimensions = dimensions[_][0:2]
        output_size = dimensions[_][2]
        params_init = [GlorotUniform() if D_init is None else D_init[_],
                       GlorotUniform(0.01),
                       Uniform([0, 0.5])]
        network = SparseClass(network, sparse_dimensions, params_init, [False] + additional_sparse_params,
                              name='SPARSE_' + stack_str)
        D_list.append(network.get_dictionary_param())
        network = DropoutLayer(network, p=p_weight, name='SPARSE_DROP_' + stack_str)
        sparse_layers.append(network)
        network = DenseLayer(network, num_units=output_size, b=None, nonlinearity=identity, name='PROJ_' + stack_str)
        G_list.append(network.W)
        network = DropoutLayer(network, p=p_weight, name='PROJ_DROP_' + stack_str)
        feature_layers.append(network)
        if residual_link:
            network = LinearCombinationLayer([DropoutLayer(
                DenseLayer(last_residual_layer, num_units=network.output_shape[-1], W=Constant(0.0), b=None,
                           nonlinearity=identity, name='PROJ_FEA_' + stack_str), p=p_weight), network], [1, 1])
        if batch_normalization:
            network = BatchNormLayer(network, axes=norm_axes)
        last_residual_layer = network
    feature = network
    classification_branch = DenseLayer(feature, 10, nonlinearity=softmax)
    classification_branch = batch_norm(classification_branch, axes=norm_axes)
    classification_branch = DropoutLayer(classification_branch, p=p_weight)
    for _ in reversed(range(num_of_stacks)):
        stack_str = str(stack_idx)
        stack_idx += 1
        sparse_dimensions = [dimensions[_][0], dimensions[_][1]]
        output_size = num_input if _ == 0 else dimensions[_ - 1][2]
        params_init = [G_list[_] if tied_weight else GlorotUniform(),
                       GlorotUniform(0.01),
                       Uniform([0, 0.5])]
        # make link from encoding feature layer to decoding sparse layer. in case of the inner most feature layer,
        # feature_layer[_] and network are both the inner most layer
        if necklace_link:
            network = LinearCombinationLayer([feature_layers[_], network], [0.5, 0.5])
        network = SparseClass(network, sparse_dimensions, params_init, [tied_weight] + additional_sparse_params,
                              name='SPARSE_' + stack_str)
        network = DropoutLayer(network, p=p_weight, name='SPARSE__DROP_' + stack_str)
        if necklace_link:
            network = LinearCombinationLayer([sparse_layers[_], network], [0.5, 0.5])
        if tied_weight:
            network = TransposedDenseLayer(network, num_units=output_size, W=D_list[_], b=None, nonlinearity=identity,
                                           name='PROJ_' + stack_str)
        else:
            network = DenseLayer(network, num_units=output_size, W=GlorotUniform(0.01), b=None, nonlinearity=identity,
                                 name='PROJ_' + stack_str)
        network = DropoutLayer(network, p=p_weight, name='PROJ_DROP_' + stack_str)
        if residual_link:
            network = LinearCombinationLayer(
                [DropoutLayer(DenseLayer(last_residual_layer, num_units=network.output_shape[-1], W=Constant(0.0),
                                         b=None, nonlinearity=identity, name='PROJ_FEA_' + stack_str), p=p_weight),
                 network], [1, 1])
        if batch_normalization and _ is not 0:
            network = BatchNormLayer(network, axes=norm_axes)
        last_residual_layer = network
    return network, classification_branch, feature


def build_computation_graph(input_var, parameters):
    #dimension[-1][-1] is the last output size of last stacked layer a.k.a size of the image vector
    sparse_algorithm = parameters.sparse_algorithm
    input_shape = parameters.input_shape
    dimensions = parameters.dimension
    tied_weight = parameters.tied_weight
    necklace_link = parameters.necklace_link
    residual_link = parameters.residual_link
    batch_normalization = parameters.batch_normalization
    p_input = parameters.p_input
    p_weight = parameters.p_weight
    norm_axes = parameters.norm_axes
    additional_sparse_params = parameters.additional_sparse_params
    network = InputLayer(shape=input_shape, input_var=input_var, name='input')
    network = DropoutLayer(network, p=p_input, name='input_drop')
    D = load_dictionary_init(100, normalize_axes=None).transpose()
    D = (D - D.min(axis=0)) / (D.max(axis=0) - D.min(axis=0))
    D_init = [theano.shared(floatX(D))]
    network, classification_branch, features = NecklaceNetwork(
        network, dimensions, sparse_algorithm, additional_sparse_params, tied_weight, necklace_link, residual_link,
        batch_normalization, norm_axes, p_weight, D_init)
    return network, classification_branch, features