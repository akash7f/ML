import numpy as np
import matplotlib.pyplot as plt

x = np.array([2, 3, 4, 5, 6, 7, 8])
y = np.array([0, 0, 0, 1, 1, 1, 1])
plt.scatter(x, y)
w = 0
b = 0


a = 0.01                                      #learning rate

for i in range(0, 10000):
    e = np.exp(w*x + b)
    yh = e/(1 + e)                              #linear function wrt w and b
    error = yh - y

    w -= 2*a* ( error * x).mean()
    b -= 2*a* (error).mean()

plt.plot(x, yh, c='r')
plt.show()