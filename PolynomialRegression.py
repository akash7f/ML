import numpy as np

class Regression:
    
    def __init__(self, x, y, learning_rate = 0.01, no_of_iterations = 100,
        power = None, slope = None, constant = 0):
        
        #if power is not given for any attribute or if power is given only for some input attributes
        if power == None:
            power = np.ones(len(x[0]))
        elif len(power) != len(x[0]):
            power = np.pad(power, (0, len(x[0]) - len(power)), mode='constant', constant_values=1)

        #if slope is not given for any attribute or if slope is given only for some input attributes
        if slope == None:
            slope = np.zeros(len(x[0]))
        elif len(slope) != len(x[0]):
            slope = np.pad(power, (0, len(x[0]) - len(slope)), mode='constant', constant_values=0)
        
        a = learning_rate                               #learning rate
        n = no_of_iterations                            #no of iterations
        self.p = power                                  #power
        self.w = slope                                  #slope
        self.b = constant                               #constant

        for i in range(0, n):

            yh = np.zeros(len(x))
            for i in range(len(x[0])):
                yh += self.w[i] * np.power(x[:, i], self.p[i])
            yh += self.b

            # MSE = ((yh - y)**2).mean()
            # print(MSE)

            for i in range(len(x[0])): 
                self.w[i] -= 2*a*(((yh - y)*x[:, i]).mean())      #gradient descent with respect to w
            self.b -= 2*a*((yh - y).mean())                       #gradient descent with respect to b

    def predict(self, x):
        return np.sum(self.w*np.power(x, self.p)) + self.b
    
    def print(self):
        print("power : ", self.p)
        print("slope : ", self.w)
        print("constant : ", self.b)

if __name__ == "__main__":
    x = np.array([[1,2,3], [2,4,5], [3,4,6], [4,5,7], [5,6,8]])     #input values
    y = np.array([4, 5, 7, 6, 7])                                   #output values
    model = Regression(x, y, power = [2,0.5])
    model.print()
    print("predicted value for [6,4,2] : ",model.predict([6,4,2]))
