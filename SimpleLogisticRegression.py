import numpy as np
import matplotlib.pyplot as plt

x = np.array([2, 3, 4, 5, 6, 7, 8])
y = np.array([0, 0, 0, 1, 1, 1, 1])
plt.scatter(x, y)
w = 0
b = 0


a = 0.01                                      #learning rate

for i in range(0, 90):
    e = np.exp(w*x + b)
    yh = e/(1 + e)                              #linear function wrt w and b

    # MSE = ((yh - y)**2).mean()                #mean square error
    # print(MSE)
    iyh = 1/yh 

    w -= 2*a* ((1 - y * iyh) * (iyh - 1) * (iyh) * x).mean()
    b -= 2*a* ((1 - y * iyh) * (iyh - 1) * (iyh)).mean()

plt.plot(x, yh, c='r')
plt.show()