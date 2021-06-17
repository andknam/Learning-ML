import numpy as np

class Softmax:
    # A standard fully-connected layer with softmax activation

    def __init__(self, input_length, nodes):
        # divide by input_length to reduce variance of initial values
        self.weights = np.random.randn(input_length, nodes) / input_length
        self.biases = np.zeros(nodes)

    def forward(self, input):
        '''
        Performs forward pass of the softmax layer using the given input.
        Returns 1d numpy array containing the respective prob values
        - input can be any array with any dimensions
        '''
        # makes input easier to work with (no longer need shape)
        input = input.flatten()

        input_length, nodes = self.weights.shape

        # multiply input and self.weights element-wise and sum results
        totals = np.dot(input, self.weights) + self.biases

        # calc exponentials needed for Softmax
        exp = np.exp(totals)
        return exp / np.sum(exp, axis=0)