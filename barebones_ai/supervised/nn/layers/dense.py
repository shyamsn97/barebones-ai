import numpy as np

from barebones_ai import activations


class Dense:
    """
    Dense Layer used for the DNN object
    Parameters:
        weights: numpy array() weight matrix
        output: output of the layer
        deriv: derivative of activation
        activation_name: name of activation, used to print
        activation: activation functions
        shape: output shape
        prevlayer: previous layer
        next: next layer
        type: type of layer
    """

    def __init__(self, outputdims, activation="sigmoid"):

        self.outputdims = outputdims
        self.weights = None
        self.output = 0
        self.deriv = 1
        self.activation_name = activation
        self.activation = None
        self.print_shape = (None, outputdims)
        self.shape = None
        self.prevlayer = None
        self.next = None
        self.type = "Dense"

        if activation == "sigmoid":
            self.activation = activations.sigmoid

        elif activation == "softmax":
            self.activation = activations.softmax

        elif activation == "linear":
            self.activation = self.linear

        elif activation == "relu":
            self.activation = activations.relu

    def initialize(self, prevlayer):

        self.weights = [
            np.random.uniform(-1, 1, size=(prevlayer.shape[1], self.outputdims)),
            np.random.uniform(-1, 1, size=(1, self.outputdims)),
        ]
        self.shape = (prevlayer.shape[1], self.outputdims)
        self.prevlayer = prevlayer

    def getWeights(self):

        return self.weights

    def setNext(self, layer):

        self.next = layer

    def getNext(self):

        return self.next

    def getPrev(self):

        return self.prevlayer

    def dot(self, X, t=False):

        if t:
            return self.output.T.dot(X)
        return self.output.dot(X)

    def forward(self):

        layer_mul = self.prevlayer.dot(self.weights[0]) + self.weights[1]
        self.output = self.activation(layer_mul)
        self.deriv = self.activation(layer_mul, True)

    def predict(self, X):

        return self.activation(X.dot(self.weights[0]) + self.weights[1])

    def update(self, X):

        self.weights = X

    def linear(self, X, deriv=False):

        if deriv:
            return 1
        else:
            return X

    def __repr__(self):

        string = (
            "Dense: "
            + "activation: "
            + self.activation_name
            + ", weight shape: "
            + str(self.weights[0].shape)
            + ", output shape: "
            + str(self.print_shape)
        )
        string = (
            string
            + ", parameters: "
            + str(self.weights[0].shape[0] * self.weights[0].shape[1])
        )
        string = (
            string
            + "\n"
            + "-----------------------------------------------------------------"
            + "\n"
        )
        return string

    def __str__(self):

        string = (
            "Dense: "
            + "activation: "
            + self.activation_name
            + ", weight shape: "
            + str(self.weights[0].shape)
            + ", output shape: "
            + str(self.print_shape)
        )
        string = (
            string
            + ", parameters: "
            + str(self.weights[0].shape[0] * self.weights[0].shape[1])
        )
        string = (
            string
            + "\n"
            + "-----------------------------------------------------------------"
            + "\n"
        )
        return string
