from collections import deque

import numpy as np
import theano
import theano.tensor as T
from lasagne.init import GlorotUniform, Normal, Uniform
from lasagne.layers import Layer, DropoutLayer
from lasagne.utils import floatX


def shrinkage(x):
    shrink_value = 1
    # return T.switch(T.lt(x, -shrink_value), x + shrink_value, T.switch(T.le(x, shrink_value), 0, x - shrink_value))
    return T.switch(T.lt(x, shrink_value), 0, x - shrink_value)

class SparseAlgorithm:
    def __init__(self):
        pass

    def get_dictionary_param(self):
        raise NotImplementedError

    def get_dictionary(self):
        raise NotImplementedError


def get_all_sparse_layers(layer, treat_as_input=None):
    """
    This function gathers all layers below one or more given :class:`Layer`
    instances, including the given layer(s). Its main use is to collect all
    layers of a network just given the output layer(s). The layers are
    guaranteed to be returned in a topological order: a layer in the result
    list is always preceded by all layers its input depends on.

    Parameters
    ----------
    layer : Layer or list
        the :class:`Layer` instance for which to gather all layers feeding
        into it, or a list of :class:`Layer` instances.

    treat_as_input : None or iterable
        an iterable of :class:`Layer` instances to treat as input layers
        with no layers feeding into them. They will show up in the result
        list, but their incoming layers will not be collected (unless they
        are required for other layers as well).

    Returns
    -------
    list
        a list of :class:`Layer` instances feeding into the given
        instance(s) either directly or indirectly, and the given
        instance(s) themselves, in topological order.

    Examples
    --------
    >>> from lasagne.layers import InputLayer, DenseLayer
    >>> l_in = InputLayer((100, 20))
    >>> l1 = DenseLayer(l_in, num_units=50)
    >>> get_all_layers(l1) == [l_in, l1]
    True
    >>> l2 = DenseLayer(l_in, num_units=10)
    >>> get_all_layers([l2, l1]) == [l_in, l2, l1]
    True
    >>> get_all_layers([l1, l2]) == [l_in, l1, l2]
    True
    >>> l3 = DenseLayer(l2, num_units=20)
    >>> get_all_layers(l3) == [l_in, l2, l3]
    True
    >>> get_all_layers(l3, treat_as_input=[l2]) == [l2, l3]
    True
    """
    # We perform a depth-first search. We add a layer to the result list only
    # after adding all its incoming layers (if any) or when detecting a cycle.
    # We use a LIFO stack to avoid ever running into recursion depth limits.
    try:
        queue = deque(layer)
    except TypeError:
        queue = deque([layer])
    seen = set()
    done = set()
    result = []

    # If treat_as_input is given, we pretend we've already collected all their
    # incoming layers.
    if treat_as_input is not None:
        seen.update(treat_as_input)

    while queue:
        # Peek at the leftmost node in the queue.
        layer = queue[0]
        if layer is None:
            # Some node had an input_layer set to `None`. Just ignore it.
            queue.popleft()
        elif layer not in seen:
            # We haven't seen this node yet: Mark it and queue all incomings
            # to be processed first. If there are no incomings, the node will
            # be appended to the result list in the next iteration.
            seen.add(layer)
            if hasattr(layer, 'input_layers'):
                queue.extendleft(reversed(layer.input_layers))
            elif hasattr(layer, 'input_layer'):
                queue.appendleft(layer.input_layer)
        else:
            # We've been here before: Either we've finished all its incomings,
            # or we've detected a cycle. In both cases, we remove the layer
            # from the queue and append it to the result list.
            queue.popleft()
            if layer not in done:
                if issubclass(layer.__class__, SparseAlgorithm):
                    result.append(layer)
                done.add(layer)

    return result



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
        self.transposed = additional_parameters[0]
        self.p_drop_input_to_shrinkage = additional_parameters[1]
        self.p_drop_shrinkage = additional_parameters[2]
        self.rescale = additional_parameters[3]
        num_inputs = incoming.output_shape[-1]
        self.dict_size = dimension[0]
        self.T = dimension[1]
        self.W = self.add_param(params_init[0], [num_inputs, self.dict_size], name='W',
                                lista=True, lista_weight_S=True, sparse_dictionary=True, regularizable=False)
        # self.S = self.add_param(params_init[1], [self.dict_size, self.dict_size], name='S',
        #                         lista=True, lista_weight_W=True, regularizable=True)
        if T > 0:
            self.S = T.eye(self.dict_size) - T.dot(self.get_dictionary(), self.get_dictionary().T)
            self.S = self.add_param(theano.shared(floatX(self.S.eval())), [self.dict_size, self.dict_size], name='S',
                                    lista=True, lista_weight_W=True, regularizable=False)
        self.theta = self.add_param(theano.shared(floatX(np.ones([self.dict_size, ]))), [self.dict_size, ], name='theta',
                                    lista=True, lista_fun_param=True, regularizable=False)
        self.theta = self.add_param(theano.shared(floatX(np.ones([self.dict_size, ]))), [self.dict_size, ],
                                    name='theta',
                                    lista=True, lista_fun_param=True, regularizable=False)
        self.eps = 1e-6
        self.clipped_theta = T.clip(self.theta, self.eps, 10)

    def get_dictionary_param(self):
        return self.W

    def get_dictionary(self):
        return self.W.T if not self.transposed else self.W

    def get_dictionary_transpose(self):
        return self.W if not self.transposed else self.W.T

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

    def shrinkage_theta(self, input):
        return self.clipped_theta * shrinkage(input / self.clipped_theta)

    def get_output_for(self, input, deterministic=False, **kwargs):
        input_to_shrinkage = T.dot(input, self.get_dictionary_transpose())
        if self.p_drop_input_to_shrinkage is not None and self.T is not 0:
            input_to_shrinkage = self.dropout(input_to_shrinkage, self.p_drop_input_to_shrinkage, deterministic,
                                              **kwargs)
        output = self.shrinkage_theta(input_to_shrinkage)
        for _ in range(self.T):
            output = self.shrinkage_theta(T.dot(output, self.S) + input_to_shrinkage)
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
        self.transposed = addition_parameters[0]
        num_inputs = incoming.output_shape[-1]
        self.dict_size = dimension[0]
        self.T = dimension[1]
        self.W = self.add_param(params_init[0], [num_inputs, self.dict_size], name='W',
                                lista=True, lista_weight_S=True, sparse_dictionary=True, regularizable=True)
        # self.S = self.add_param(params_init[1], [self.dict_size, self.dict_size], name='S',
        #                         lista=True, lista_weight_W=True, regularizable=True)
        if T > 0:
            self.S = T.eye(self.dict_size) - T.dot(self.get_dictionary(), self.get_dictionary().T)
            self.S = self.add_param(theano.shared(floatX(self.S.eval())), [self.dict_size, self.dict_size], name='S',
                                    lista=True, lista_weight_W=True, regularizable=True)
        self.theta = self.add_param(theano.shared(floatX(np.ones([self.dict_size, ]))), [self.dict_size, ],
                                    name='theta',
                                    lista=True, lista_fun_param=True, regularizable=False)
        self.eps = 1e-6
        self.clipped_theta = T.clip(self.theta, self.eps, 10)

    def get_dictionary_param(self):
        return self.W

    def get_dictionary(self):
        return self.W.T if not self.transposed else self.W

    def get_dictionary_transpose(self):
        return self.W if not self.transposed else self.W.T

    def shrinkage_theta(self, input):
        return self.clipped_theta * shrinkage(input / self.clipped_theta)

    def get_output_for(self, input, **kwargs):
        B = T.dot(input, self.get_dictionary_transpose())
        output = self.shrinkage_theta(B)
        for _ in range(self.T):
            output = self.shrinkage_theta(T.dot(output, self.S) + B)
        # output = T.clip(output, 0, float('inf'))
        return output
        # self.B = T.dot(input, self.get_dictionary_transpose())
        # shrinkage_fn = lambda x, theta: theta * shrinkage(x / theta)
        # output, _ = theano.scan(fn=lambda previous_output, S, theta, B: shrinkage_fn(T.dot(previous_output, S) + B, theta),
        #                         outputs_info=shrinkage_fn(self.B, self.theta),
        #                         non_sequences=[self.S, self.theta, self.B],
        #                         n_steps=self.T)
        # output = output[-1]
        # output = T.clip(output, 0, float('inf'))
        # return output


    def get_output_shape_for(self, input_shape):
        return (self.input_shape[0], self.dict_size)
