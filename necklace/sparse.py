import theano
import theano.tensor as T
from lasagne.init import GlorotUniform, Normal, Uniform
from lasagne.layers import Layer, DropoutLayer


def shrinkage(x, theta):
    return T.switch(T.lt(x, -theta), x + theta, T.switch(T.le(x, theta), 0, x - theta))


class SparseAlgorithm:
    def __init__(self):
        pass

    def get_dictionary_param(self):
        raise NotImplementedError

    def get_dictionary(self):
        raise NotImplementedError


class ShrinkageLayer(Layer):
    def __init__(self, incoming, dict_size, params_init=(GlorotUniform(0.01),
                                                         Normal(0.0005, mean=0.001),
                                                         None),
                 **kwargs):
        super(ShrinkageLayer, self).__init__(incoming, **kwargs)
        self.dict_size = dict_size
        self.S = self.add_param(params_init[0], [self.dict_size, self.dict_size], name='S',
                                lista=True, lista_weight_W=True, regularizable=True)
        self.theta = self.add_param(params_init[1], [self.dict_size, ], name='theta',
                                    lista=True, lista_fun_param=True, regularizable=False)
        self.B = params_init[2]

    def get_output_for(self, input, **kwargs):
        eps = 1e-6
        output = shrinkage(T.dot(input, self.S) + self.B.get_output_for(input, **kwargs), self.theta + eps)
        return output

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.dict_size)


class LISTAWithDropout(DropoutLayer, SparseAlgorithm):
    def __init__(self, incoming, dimension, params_init=(GlorotUniform(),
                                                         GlorotUniform(),
                                                         Uniform([0, 0.1])),
                 additional_parameters=[False, None, None, False], **kwargs):
        '''
        init parameters
        :param incoming: input to the LISTA layer
        :param dimension: 2 numbers list.
         dimension[0] is dict_size, length of dictionary vector in LISTA. dimension[1] is T a.k.a depth
        :param params_init: init value or init method for LISTA
        :transposed: = True if the input dictionary D is the transpose matrix of a theano.compile.SharedVariable V.
         In that case self.W = D^T = V^T^T = V
        :param kwargs: parameters of super class
        :return:
        '''
        super(LISTAWithDropout, self).__init__(incoming, **kwargs)
        num_inputs = incoming.output_shape[-1]
        self.dict_size = dimension[0]
        self.T = dimension[1]
        self.W = self.add_param(params_init[0], [num_inputs, self.dict_size], name='W',
                                lista=True, lista_weight_S=True, sparse_dictionary=True, regularizable=True)
        self.S = self.add_param(params_init[1], [self.dict_size, self.dict_size], name='S',
                                lista=True, lista_weight_W=True, regularizable=True)
        self.theta = self.add_param(params_init[2], [self.dict_size, ], name='theta',
                                    lista=True, lista_fun_param=True, regularizable=False)
        self.transposed = additional_parameters[0]
        self.p_drop_input_to_shrinkage = additional_parameters[1]
        self.p_drop_shrinkage = additional_parameters[2]
        self.rescale = additional_parameters[3]

    def dropout(self, input, p_drop, deterministic=False, **kwargs):
        """
        Parameters
        ----------
        input : tensor
            output from the previous layer
        deterministic : bool
            If true dropout and scaling is disabled, see notes
        """
        if deterministic or p_drop == 0:
            return input
        else:
            retain_prob = 1 - p_drop
            if self.rescale:
                input /= retain_prob

            # use nonsymbolic shape for dropout mask if possible
            input_shape = self.input_shape
            if any(s is None for s in input_shape):
                input_shape = input.shape

            return input * self._srng.binomial(input_shape, p=retain_prob,
                                               dtype=theano.config.floatX)

    def get_dictionary_param(self):
        return self.W

    def get_dictionary(self):
        return T.transpose(self.W) if not self.transposed else self.W

    def get_output_for(self, input, deterministic=False, **kwargs):
        eps = 1e-6
        input_to_shrinkage = T.dot(input, self.W if not self.transposed else self.W.transpose())
        if self.p_drop_input_to_shrinkage is not None and self.T is not 0:
            input_to_shrinkage = self.dropout(input_to_shrinkage, self.p_drop_input_to_shrinkage, deterministic,
                                              **kwargs)
        output = shrinkage(input_to_shrinkage, self.theta)
        for _ in range(self.T):
            output = shrinkage(T.dot(output, self.S) + input_to_shrinkage, self.theta + eps)
            if self.p_drop_shrinkage is not None and _ is not self.T - 1:
                output = self.dropout(output, self.p_drop_shrinkage, deterministic, **kwargs)
        return output

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.dict_size)


class LISTA(Layer, SparseAlgorithm):
    '''
    Class implementation for LISTA transformation
    '''

    def __init__(self, incoming, dimension, params_init=(GlorotUniform(),
                                                         GlorotUniform(),
                                                         Uniform([0, 0.1])),
                 addition_parameters=[False], **kwargs):
        '''
        init parameters
        :param incoming: input to the LISTA layer
        :param dimension: 2 numbers list.
         dimension[0] is dict_size, length of dictionary vector in LISTA. dimension[1] is T a.k.a depth
        :param params_init: init value or init method for LISTA
        :transposed: = True if the input dictionary D is the transpose matrix of a theano.compile.SharedVariable V.
         In that case self.W = D^T = V^T^T = V
        :param kwargs: parameters of super class
        :return:
        '''
        super(LISTA, self).__init__(incoming, **kwargs)
        num_inputs = incoming.output_shape[-1]
        self.dict_size = dimension[0]
        self.T = dimension[1]
        self.W = self.add_param(params_init[0], [num_inputs, self.dict_size], name='W',
                                lista=True, lista_weight_S=True, sparse_dictionary=True, regularizable=True)
        self.S = self.add_param(params_init[1], [self.dict_size, self.dict_size], name='S',
                                lista=True, lista_weight_W=True, regularizable=True)
        self.theta = self.add_param(params_init[2], [self.dict_size, ], name='theta',
                                    lista=True, lista_fun_param=True, regularizable=False)
        self.transposed = addition_parameters[0]

    def get_dictionary_param(self):
        return self.W

    def get_dictionary(self):
        return T.transpose(self.W) if not self.transposed else self.W

    def get_output_for(self, input, **kwargs):
        eps = 1e-6
        B = T.dot(input, self.W if not self.transposed else self.W.transpose())
        output = shrinkage(B, self.theta + eps)
        for _ in range(self.T):
            output = shrinkage(T.dot(output, self.S) + B, self.theta + eps)
        return output

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.dict_size)
