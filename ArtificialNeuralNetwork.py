import numpy as np
import Activations

class Layer:
    def __init__(self, neurons, activation):
        self.n = neurons #number of neurons
        self.x = None    #inputs
        self.y = None    #outputs
        self.w = None    #weights
        self.b = None    #bias
        self.f = getattr(Activations, activation)   #activation function
        self.fd = getattr(Activations, activation+"_derivative")  #function derivative

    def initialize(self, features):
        self.w = np.random.random((features, self.n))
        self.b = np.random.random((1, self.n))
    
    def forward(self, inputs):
        self.x = inputs
        if self.w is None:
            self.initialize(len(inputs))
        summation = self.w.T @ self.x + self.b.T
        self.y = self.f(summation)
        return self.y
            
    def backward(self, error, lr):
        delta = self.fd(self.y) * error
        weighted_delta = self.w @ delta
        self.w -= lr * (self.x @ delta.T)
        self.b -= lr * np.sum(delta, axis = 1, keepdims=True).T
        return weighted_delta

class Model:
    def __init__(self, *layers):
        self.layers = layers
        self.n = len(layers)

    def print(self, error):
        print(np.mean(error**2))
                
    def forward(self, X):
        a = X.T
        for i in range(self.n):
            a = self.layers[i].forward(a)
        return a

    def backward(self, error, lr):
        for i in range(self.n - 1,-1,-1):
            error = self.layers[i].backward(error, lr)

    def train(self, X, y, learning_rate, epochs):
        """
        m samples of n dimensional data
        x = [[x11, x12...x1n],
             [x21, x22...[2n],
             ...
             [xm1, xm2...[mn]]
    
        m outputs for m samples of data
        y = [[y1], [y2], [y3]...[ym]]

        """
        for epoch in range(epochs):
            error = self.forward(X) - y.T
            self.print(error)
            self.backward(error, learning_rate)
    
    def predict(self, X):
        return self.forward(X)


if __name__ == "__main__":
    X = np.array([[0.6,0.4,0.3], [0.4,0.5,0.6], [0.4,0.5,0.3]])
    y = np.array([[0.1,0.8], [0.7, 0.3], [0.5, 0.5]])
    model = Model(
        Layer(4, activation='relu'),     # input layer
        Layer(3, activation='relu'),     # hidden layer
        Layer(2, activation='sigmoid'),  # output layer for 3 classes
    )
    model.train(X, y, learning_rate=0.01, epochs=5000)
    print(model.predict(X).T)