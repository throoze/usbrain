#!/usr/bin/env python
# ------------------------------------------------------------
# usbrain
#
# Python multilayer backpropagation neural network
# implementation
#
# Authors:
# Victor De Ponte, 05-38087, <rdbvictor19@gmail.com>
# Jorge , , <@gmail.com>
#
# Usage:
#
# ------------------------------------------------------------
import numpy as np

import theano
from theano import tensor as T
from theano import function
from theano.tensor.nnet import sigmoid

class MalformedNetworkException(Exception):
    pass

class ActivationFunctionError(Exception):
    pass

class UnknownDistributionException(Exception):
    pass

class UnknownOptimizationMethodException(Exception):
    pass

class NeuralNetwork(object):
    """
    docstring for NeuralNetwork
    """

    random_samplers_map = {
        "uniform" : np.random.random_sample,
        "normal"  : np.random.standard_normal
    }
    known_distributions = random_samplers_map.keys()

    def __init__(self,
                sizes=None,
                activation=sigmoid,
                distribution="uniform",
                optimization_method="gradient_descent",
                debug=False
                            ):
        if sizes is not None:
            if len(sizes) < 2:
                message = "Not enough layers. Must be two or more."
                raise MalformedNetworkException(message)
            for i, layer in enumerate(sizes):
                if layer < 1:
                    message = "Not enough activation units in layer %d" % (i+1)
                    raise MalformedNetworkException(message)
        else :
            raise MalformedNetworkException("Layers not provided")
        if distribution in self.known_distributions:
            self._random_generator = self.random_samplers_map[distribution]
        else:
            message = "Please choose one of the following distributions: "
            message += self.known_distributions[0]
            for dist in self.known_distributions[1:]:
                message += ", %s" % dist
            message += "."
            raise UnknownDistributionException(message)
        if optimization_method in self.known_optimizations:
            self._optimization_method = self.optimization_methods_map[optimization_method]
        else:
            message = "Please choose one of the following optimization methods: "
            message += self.known_optimizations[0]
            for opt in self.known_optimizations[1:]:
                message += ", %s" % opt
            message += "."
            raise UnknownOptimizationMethodException(message)
        self._sizes = sizes
        self._activation = activation
        self._debug = debug
        self._build()

    def _build(self):
        if self._debug:
            theano.config.compute_test_value = 'warn'
        X,W = T.matrices('X','W')
        if self._debug:
            X.tag.test_value = np.random.rand(3,1)
            W.tag.test_value = np.random.rand(5,3)
        Z = T.dot(W,X)
        A = self._activation(Z)
        self._fpropagate = function([X, W],A)
        self._layers = []
        self._generate_initial_weights()

    def _generate_initial_weights(self):
        self._weights = []
        for layer in range(len(self._sizes)-1):
            i, j = self._sizes[layer+1], self._sizes[layer]+1
            self._weights.append(self._random_generator((i,j)))

    def _forward_propagation(self, x):
        self._reset_layers()
        input_layer = np.insert(np.array(x),0,1,axis=0)
        self._layers.append(input_layer)
        for layer,weight in enumerate(self._weights):
            propagated_layer = self._fpropagate(self._layers[layer], weight)
            if layer == len(self._weights)-1:
                new_layer = propagated_layer
            else:
                new_layer = np.insert(propagated_layer, 0, 1, axis=0)
            self._layers.append(new_layer)
                

    def _reset_layers(self):
        del self._layers
        self._layers = []

    def _backward_propagation(self, y):
        pass

    def _gradient_descent(self):
        pass

    optimization_methods_map = {
        "gradient_descent" : _gradient_descent
    }
    known_optimizations = optimization_methods_map.keys()