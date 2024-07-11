import numpy as np
from ActivationFunctions import sigmoid, sigmoid_derivative

class Dense:
    def __init__(self, units, activation, derivative):
        self.units = units
        self.W = None
        self.b = None
        self.activation = activation
        self.derivative = derivative

    def out(self, A_in):
        Z = np.matmul(A_in, self.W) + self.b
        A_out = self.activation(Z)
        return A_out, Z

    def initialize(self, dimensions):
        self.dimensions = dimensions
        self.W = np.random.rand(self.dimensions, self.units)
        self.b = np.random.rand(self.units)
        return self.units

class Sequential:
    def __init__(self, layers):
        self.layers = layers

    def predict(self, A_in):
        a, z = self.evaluate(A_in)
        yh = a[-1].reshape(1, -1)[0]
        yh = np.where(yh < 0.5, 0, 1)
        return yh

    def evaluate(self, A_in):
        a_i = A_in
        a = []
        z = []
        
        for layer in self.layers:
            a_i, z_i = layer.out(a_i)
            a.append(a_i)
            z.append(z_i)
        return a, z
            
    def fit(self, X, y, learning_rate=0.01, epochs=10000):
        dimensions = X.shape[1]

        for layer in self.layers:
            dimensions = layer.initialize(dimensions)

        for epoch in range(epochs):
            
            a, z = self.evaluate(X)

            # Backward pass
            y = y.reshape(-1, 1)
            error = a[-1] - y
            deltas = [error * self.layers[-1].derivative(z[-1])]

            for i in range(len(self.layers) - 2, -1, -1):
                delta = np.dot(deltas[-1], self.layers[i + 1].W.T) * self.layers[i].derivative(z[i])
                deltas.append(delta)

            deltas.reverse()

            # Update weights and biases
            A_in = X
            for i, layer in enumerate(self.layers):
                layer.W -= learning_rate * np.dot(A_in.T, deltas[i]) / len(X)
                layer.b -= learning_rate * np.mean(deltas[i], axis=0)
                A_in = a[i]

if __name__ == "__main__":

    X = np.array([
    [50, 0],  # Not spam (low word count, no uppercase)
    [200, 1],  # Spam (medium word count, uppercase)
    [2500, 0], # Not spam (high word count, no uppercase)
    [100, 1],  # Spam (low word count, uppercase)
    [78, 0],   # Not spam (low word count, no uppercase)
    ])

    y = np.array([0, 1, 0, 1, 0])

    model = Sequential([
        Dense(units=3, activation=sigmoid, derivative=sigmoid_derivative),
        Dense(units=1, activation=sigmoid, derivative=sigmoid_derivative)
    ])

    model.fit(X, y)
    print(model.predict(X))
