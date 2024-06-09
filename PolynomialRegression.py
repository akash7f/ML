import numpy as np

class Regression:
    
    def fit(self, x, y, learning_rate = 0.001, no_of_iterations = 100000):
        
        attributes = len(x[0])
        samples = len(y)
        
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
            yh = np.zeros(samples)
            for i in range(attributes):
                attribute = x[:, i]
                yh += self.w[i] * np.power(attribute, self.p[i])
            yh += self.b
            
            error = yh - y
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
    import time

    x = np.array([[1], [2], [3], [4], [5]])     #input values
    y = np.array([1, 4, 9, 16, 25])                                   #output values
    
    model = Regression()
    model.fit(x, y)
    model.print()

    data = [8]
    print(f"Predicted value for {data} : {model.predict(data)}")