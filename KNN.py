import numpy as np

class PolynomialRegression:
    
    def fit(self, x, y):
        """
        m samples of n dimensional data
        x = [[x11, x12...x1n],
             [x21, x22...[2n],
             ...
             [xm1, xm2...[mn]]
    
        m outputs for m samples of data
        y = [y1, y2, y3...ym]

        """
        # no of attributes in data and no of samples of data
        attributes = len(x[0])
        samples = len(x)

        test_size = samples//5
        train_size = samples - test_size

        x_train, y_train = x[ : train_size], y[ : train_size]
        x_test, y_test = x[train_size : ], y[train_size : ]

        max_k = 100
        for K in range(1, max_k, 2):
            k_error = np.ndarray(shape=(2, max_k//2))

            test_result = np.zeros(test_size)
            i = 0
            for test in x_test:
                distances = np.zeros(train_size)
                j = 0
                for train, value in x_train, y_train:
                    distances[j] = np.sqrt(np.sum((train - test)**2))
                    j+=1
                j = np.argsort(distances)
                j = j[:K]
                k_outputs = y_train[:, j]
                unique, frequencies = np.unique(k_outputs, return_counts=True)
                j = np.argsort(frequencies)
                unique = unique[:, j]
                test_result[i] = unique[0]

            error = ((y_test - test_result)**2).mean()
            k_error[0][max_k//2] = K
            k_error[1][max_k//2] = error
        
        args = np.argsort(k_error[1])
        k_error = k_error[:, args]
        K = k_error[0][0]

    def predict(self, x):
        return np.sum(self.w* np.power(x, self.p)) + self.b

    def print(self):
        print()
        print("Power : ", self.p)
        print("Slope : ", self.w)
        print("Constant : ", self.b)
        print("MSE :", self.MSE)
        print()

if __name__ == "__main__":

    x = np.array([[0.1, 0.2], [0.3, 0.3], [0.5, 0.4], [0.6, 0.7], [1, 0.9]])
    y = np.array([0.001, 0.027, 0.125, 0.343, 1])
    model = PolynomialRegression()
    model.fit(x, y)
    model.print()

    data = [8, 5]
    print(f"Predicted value for {data} : {model.predict(data)}")