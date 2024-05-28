import numpy as np

x = np.array([[1,2], [2,4], [3,4], [4,5], [5,6]])  #input values
y = np.array([4, 5, 7, 6, 7])            #output values

class Regression:
    
    def __init__(self, x, y, learning_rate = 0.01, no_of_iterations = 100,
        power = [1 for i in range(len(x[0]))], slope = np.zeros(len(x[0])), constant = 0):
                
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

            MSE = ((yh - y)**2).mean()
            print(MSE)

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
    model = Regression(x, y, power = [2,1])
    model.print()
    print(model.predict([6,4]))
