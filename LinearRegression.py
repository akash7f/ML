import numpy as np
import matplotlib.pyplot as plt

x = np.array([3, 5, 7, 9.2, 8])
y = np.array([4.2, 5, 7.1, 6.5, 7.2])
plt.scatter(x, y)
w1 = 0
w0 = 0

yh = w1*x + w0                                  #linear function wrt w1 and w0

MSE = ((yh - y)**2).mean()                #mean square error
print(MSE)

a = 0.01                                      #learning rate

for i in range(0, 1000):
    w1 -= 2 * a * ((yh - y)*x).mean()     #gradient descent of MSE wrt to w1
    w0 -=2 * a * (yh - y).mean()         #gradient descent of MSE wrt to w0
    
    yh = w1*x + w0
    
    MSE = ((yh - y)**2).mean()
    print(MSE)
    
plt.plot(x, yh, c='r')
plt.show()