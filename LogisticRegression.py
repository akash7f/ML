import numpy as np

class LogisticRegression:
    
    def fit(self, x, y, learning_rate = 0.01, no_of_iterations = 5000):
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


            yh = self.predict(x)
            
            """ 
            Function (yh, y): Either -log(yh)  if y = 1
                              or -log( 1-yh )  if y = 0 
            Function (yh, y): (1 - y)( -log( 1-yh ) ) + (y) (-log(yh))
            Cost Function = (1 / m) Summation of(i = 1 to m)  Function (yh(i), y(i))
            """
            # error between predicted value and true value
            error = yh - y 
            
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

            """ Vectorization of reducing of cost"""
            
            log_attributes = np.log(x)                                          # log of attributes
            attribute_powers = x ** self.p                                      # attribute values with their powers
            error_term = error[:, np.newaxis] * attribute_powers                # Error term is error * attribute_powers
            self.p -= a * (error_term * self.w * log_attributes).mean(axis=0)   # Gradient descent with respect to power of each attribute
            self.w -= a * error_term.mean(axis=0)                               # Gradient descent with respect to slope of each attribute
            self.b -= a * error.mean()                                          # Gradient descent with respect to constant

    def predict(self, x):
        # Read Polynomial Regression Prediction for more
        """Vectorization of prediction"""
        attribute_powers = x ** self.p                       # Attribute values with their powers
        p = np.sum(self.w * attribute_powers, axis=1)        # Compute the weighted sum of the attribute powers
        p += self.b                                          # Add the constant term
        yh = 1/(1 + np.exp(-p))                              # Sigmoid function
        return yh

    def print(self):
        print()
        print("Power : ", self.p)
        print("Slope : ", self.w)
        print("Constant : ", self.b)
        print()

if __name__ == "__main__":

    x = np.array([[1, 2], [3, 3], [5, 4], [6, 7], [1, 9]])
    y = np.array([0, 0, 1, 1, 1])
    model = LogisticRegression()
    model.fit(x, y)
    model.print()

    data = np.array([[0, 0]])
    print(f"Predicted value for {data} : {model.predict(data)}")