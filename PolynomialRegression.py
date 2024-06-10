import numpy as np

class PolynomialRegression:
    
    def fit(self, x, y, learning_rate = 1, no_of_iterations = 10000):
        """
        Use standardized lying between 0 and 1
        m samples of one dimensional data
        x = [x1, x2, x3...xm]

        m samples of n dimensional data
        x = [[x11, x12...x1n],
             [x21, x22...[2n],
             ...
             [xm1, xm2...[mn]]
    
        m outputs for m samples of data
        y = [y1, y2, y3...ym]

        """
        if not isinstance(type(x[0]), np.ndarray):
            x = x.reshape(-1, 1)

        # no of attributes in data and no of samples of data
        attributes = len(x[0])
        samples = len(x)
        
        a = learning_rate                               #learning rate
        n = no_of_iterations                            #no of iterations
        self.p = np.ones(attributes)                    #power
        self.w = np.zeros(attributes)                   #slope
        self.b = 0                                      #constant

        for i in range(0, n):

            # predicting values using row approach
            # yh = np.zeros(samples)
            # for i in range(samples):
            #     yh[i] = self.predict(x[i])

            # predicting values column approach
            # column is preferred becuase of parallel multiplication of vector
            yh = np.zeros(samples)
            for i in range(attributes):
                attribute = x[:, i]
                yh += self.w[i] * np.power(attribute, self.p[i])
            yh += self.b
            
            # error between predicted value and true value
            error = yh - y
            
            # reducing of error using gradient descent
            # gradient descent with respect to power of each attribute
            for i in range(attributes):
                attribute = x[:, i]
                log_attribute = np.log1p(attribute)
                self.p[i] -= a * (error * self.w[i] * log_attribute * attribute ** self.p[i]).mean()

            # gradient descent with respect to slope of each attributee
            # because of slow learning rate using of changed power will not effect much
            for i in range(attributes):
                attribute = x[:, i]
                self.w[i] -= a * (error * attribute ** self.p[i]).mean()

            # gradient descent with respect to constant
            self.b -= a * error.mean()

        # mean square error on training data
        self.MSE = (error**2).mean()

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

    x = np.array([0.1, 0.3, 0.5, 0.7, 1])
    y = np.array([0.001, 0.027, 0.125, 0.343, 1])
    model = PolynomialRegression()
    model.fit(x, y)
    model.print()

    data = [8]
    print(f"Predicted value for {data} : {model.predict(data)}")