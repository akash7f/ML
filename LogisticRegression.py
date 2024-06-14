import numpy as np

class LogisticRegression:
    
    def fit(self, x, y, learning_rate = 0.01, no_of_iterations = 80):
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
        
        a = learning_rate                               #learning rate
        n = no_of_iterations                            #no of iterations
        self.p = np.ones(attributes)                    #power
        self.w = np.zeros(attributes)                   #slope
        self.b = 0                                      #constant

        for iteraion in range(n):

            # predicting values column approach
            # column is preferred becuase of parallel multiplication of vector
            ep = np.zeros(samples)
            for i in range(attributes):
                attribute = x[:, i]
                ep += self.w[i] * np.power(attribute, self.p[i])
            ep += self.b
            e = np.exp(ep)
            yh = e/(1 + e)

            # error between predicted value and true value
            iyh = 1/yh 
            error = (1 - y * iyh) * (iyh - 1) * (iyh) 
            
            # # reducing of error using gradient descent
            # # gradient descent with respect to power of each attribute
            # for i in range(attributes):
            #     attribute = x[:, i]
            #     log_attribute = np.log1p(attribute)
            #     self.p[i] -= a * (error * self.w[i] * log_attribute * attribute ** self.p[i]).mean()

            # # gradient descent with respect to slope of each attributee
            # # because of slow learning rate using of changed power will not effect much
            # for i in range(attributes):
            #     attribute = x[:, i]
            #     self.w[i] -= a * (error * attribute ** self.p[i]).mean()

            # # gradient descent with respect to constant
            # self.b -= a * error.mean()

            # reducing of error using gradient descent merged
            for i in range(attributes):
                attribute = x[:, i]
                log_attribute = np.log1p(attribute)

                # gradient descent with respect to power of each attribute
                self.p[i] -= a * (error * self.w[i] * log_attribute * attribute ** self.p[i]).mean()
                
                # gradient descent with respect to slope of each attributee
                # because of slow learning rate using of changed power will not effect much
                self.w[i] -= a * (error * attribute ** self.p[i]).mean()
            
            # gradient descent with respect to constant
            self.b -= a * error.mean()


        # mean square error on training data
        self.MSE = (error**2).mean()

    def predict(self, x):
        ep = np.sum(self.w* np.power(x, self.p)) + self.b
        e = np.exp(ep)
        yh = e/(1+e)
        if yh < 0.5:
            return 0
        return 1

    def print(self):
        print()
        print("Power : ", self.p)
        print("Slope : ", self.w)
        print("Constant : ", self.b)
        print("MSE :", self.MSE)
        print()

if __name__ == "__main__":

    x = np.array([[0.1, 0.2], [0.3, 0.3], [0.5, 0.4], [0.6, 0.7], [1, 0.9]])
    y = np.array([0, 0, 1, 1, 1])
    model = LogisticRegression()
    model.fit(x, y)
    model.print()

    data = [0.3, 0.9]
    print(f"Predicted value for {data} : {model.predict(data)}")