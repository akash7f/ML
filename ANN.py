import numpy as np
from ActivationFunctions import sigmoid
"""
Q = > 1 dimensional vector
q = 1 dimensional vector
q_i = value
"""
# layer
class Dense:
    def __init__(self, units, activation):
        self.units = units
        self.W = np.array(units, dtype=object) 
        self.b = np.array(units, dtype=object)
        self.activation = activation

    def out(self, A_in):
        # a_out = np.zeros(self.units)
        # for i in range(self.units):
        #     w,b_i = self.W[:, i], self.b[i]
        #     z = np.dot(A_in * w) + b_i
        #     A_out[i] = g(z)
        """Vectorized"""
        Z = np.matmul(A_in, self.W) + self.b
        A_out = self.activation(Z)
        return A_out

# artificial neural network
class Sequential:
    def __init__(self, layers):
        self.no_of_layers = len(layers)
        self.layers = layers
    
    def out(self, A_in):
        t  = A_in
        for i in range(self.no_of_layers):
            t = self.layers[i].out(t)
        return t

model = Sequential([
    Dense(units=25, activation='sigmoid'),
    Dense(units=15, activation='sigmoid'),
    Dense(units=1, activation='sigmoid')
])