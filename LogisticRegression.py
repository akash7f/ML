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
        # no of dimensions in data and no of samples of data
        dimensions = len(x[0])
        
        a = learning_rate                               #learning rate
        n = no_of_iterations                            #no of iterations
        self.w = np.zeros(dimensions)                   #slope
        self.b = 0                                      #constant

        for iteraion in range(n):


            yh = self.evaluate(x)
            
            """ 
            Function (yh, y): Either -log(yh)  if y = 1
                              or -log( 1-yh )  if y = 0 
            Function (yh, y): (1 - y)( -log( 1-yh ) ) + (y) (-log(yh))
            Cost Function = (1 / m) Summation of(i = 1 to m)  Function (yh(i), y(i))

            derivation of the cost function of logistic regression is same as
            derivation of the cost function of linear regression 
            Its's better to understand linear regression first

            we can skip power too if we want as we have seen the sigmoi function
            there will be very little effect 
            """
            error = yh - y                                                      # error between evaluateed value and true value
            error_term = error[:, np.newaxis] * x                               # Error term is error * dimension_powers
            self.w -= a * error_term.mean(axis=0)                               # Gradient descent with respect to slope of each dimension
            self.b -= a * error.mean()                                          # Gradient descent with respect to constant

    def evaluate(self, x):
        p = np.sum(self.w * x, axis=1)                                          # Compute the weighted sum of the dimension powers
        p += self.b                                                             # Add the constant term
        e = 1/(1 + np.exp(-p))                                                  # Sigmoid function
        return e

    def predict(self, x):
        yh = self.evaluate(x)
        yh = np.where(yh < 0.5, 0, 1)
        return yh
        
    def print(self):
        print()
        print("Slope : ", self.w)
        print("Constant : ", self.b)
        print()

if __name__ == "__main__":

    x = np.array([
        [1, 2],
        [3, 3],
        [5, 4],
        [6, 7],
        [1, 9]
    ])
    y = np.array([0, 0, 1, 1, 1])
    model = LogisticRegression()
    model.fit(x, y)
    model.print()

    data = np.array([[0, 0]])
    print(f"Predicted value for {data} : {model.predict(data)}")