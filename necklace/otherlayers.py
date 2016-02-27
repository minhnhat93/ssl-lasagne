import numpy as np
import theano.tensor as T
from lasagne.init import GlorotUniform, Constant
from lasagne.layers import Layer, MergeLayer
from lasagne.nonlinearities import identity, rectify


class TransposedDenseLayer(Layer):
    def __init__(self, incoming, num_units, W=GlorotUniform(0.01),
                 b=Constant(0.), nonlinearity=rectify,
                 **kwargs):
        super(TransposedDenseLayer, self).__init__(incoming, **kwargs)
        self.nonlinearity = (identity if nonlinearity is None
                             else nonlinearity)

        self.num_units = num_units

        num_inputs = int(np.prod(self.input_shape[1:]))

        self.W = self.add_param(W, (num_inputs, num_units), name="W")
        if b is None:
            self.b = None
        else:
            self.b = self.add_param(b, (num_units,), name="b",
                                    regularizable=False)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.num_units)

    def get_output_for(self, input, **kwargs):
        if input.ndim > 2:
            # if the input has more than two dimensions, flatten it into a
            # batch of feature vectors.
            input = input.flatten(2)

        activation = T.dot(input, self.W.transpose())
        if self.b is not None:
            activation = activation + self.b.dimshuffle('x', 0)
        return self.nonlinearity(activation)


class LinearCombinationLayer(MergeLayer):
    def __init__(self, incomings, alpha, **kwargs):
        assert incomings[0].output_shape == incomings[1].output_shape
        super(LinearCombinationLayer, self).__init__(incomings, **kwargs)
        self.alpha = alpha

    def get_output_shape_for(self, input_shapes):
        return self.input_layers[0].get_output_shape_for(input_shapes[0])

    def get_output_for(self, inputs, **kwargs):
        output = (self.alpha[0] * self.input_layers[0].get_output_for(inputs[0], **kwargs) +
                  self.alpha[1] * self.input_layers[1].get_output_for(inputs[1], **kwargs))
        return output

