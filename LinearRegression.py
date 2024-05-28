import numpy as np
import matplotlib.pyplot as plt

x = np.array([3, 5, 7, 9.2, 8])
y = np.array([4.2, 5, 7.1, 6.5, 7.2])
plt.scatter(x, y)
w = 0
b = 0


a = 0.01                                      #learning rate

for i in range(0, 1000):
    yh = w*x + b                                  #linear function wrt w and b

    # MSE = ((yh - y)**2).mean()                #mean square error
    # print(MSE)

    w -= 2 * a * ((yh - y)*x).mean()     #gradient descent of MSE wrt to w
    b -=2 * a * (yh - y).mean()         #gradient descent of MSE wrt to b
        
plt.plot(x, yh, c='r')
plt.show()