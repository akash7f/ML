import numpy as np
class SoftmaxRegression:
    def fit(self, x, y, learning_rate = 0.01, no_of_iterations = 5000):
        dimensions = len(x[0])

        a = learning_rate
        n = no_of_iterations
        self.outputs = np.unique(y)
        no_of_outputs = len(self.outputs)
        self.W = np.zeros(shape=(no_of_outputs, dimensions))
        self.b = np.zeros(no_of_outputs)

        for iteration in range(n):
            yh = self.evaluate(x)

            # for i in range(no_of_outputs):

            #     indicator = np.where(y == self.outputs[i], 1, 0)
            #     error = yh[:, i] - indicator

            #     for j in range(dimensions):
            #         self.W[i, j] -= a*(error.T * x[:, j]).mean()
            #     self.b[i] -= error.mean() 

            """ Vectorized for efficiency"""
            error = yh - np.eye(no_of_outputs)[y]
            self.W -= a * np.dot(error.T, x) / x.shape[0]
            self.b -= a * error.mean(axis=0)

                
    def evaluate(self, x):
        p = np.dot(x, self.W.T) + self.b
        # subtracting np.max(...) for numerical stability
        e = np.exp(p - np.max(p, axis=1, keepdims=True))    
        e_sum = np.sum(e, axis=1, keepdims=True)
        a = e / e_sum
        return a
    
    def predict(self, x):
        a = self.evaluate(x)
        yh = np.argmax(a, axis=1)
        return yh

if __name__ == "__main__":
    x = np.array([
        [2.0, 3.0],
        [1.0, 5.0],
        [2.0, 2.0],
        [3.0, 6.0],
        [5.0, 8.0],
        [8.0, 8.0],
        [7.0, 6.0],
        [6.0, 5.0],
        [6.0, 3.0],
        [5.0, 2.0]
    ])

    y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2])

    model = SoftmaxRegression()
    model.fit(x, y)
    print(model.predict(x))