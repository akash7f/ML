import numpy

class ABC:
    def __init__(self, val):
        self.k = val

def linear(x):
    """
    Linear Activation Function
    It is used in regression and it is also known as no function 
    """
    return x.copy()

def sigmoid(x):
    """
    Sigmoid Activation Function
    It is used in Binary Classification
    """
    # 0 < y < 1 for -infinity < x < infinity
    return 1/(1 + numpy.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


def relu(x):
    """
    Rectified Linear Unit Function
    It is used in regression where the prediction can't be negative
    """
    # if x[i] < 0 then y[i] = 0 else y[i] = x[i]
    return numpy.where(x < 0, 0, x)

if __name__  == "__main__":
    import matplotlib.pyplot as plt

    x = numpy.linspace(-5, 5, 100)
    l = linear(x)
    s = sigmoid(x)
    r = relu(x)
    plt.subplot(1, 3, 1)
    plt.plot(x, l, color='r')
    plt.title('Linear')
    plt.subplot(1, 3, 2)
    plt.plot(x, s, color='g')
    plt.title('Sigmoid')
    plt.subplot(1, 3, 3)
    plt.plot(x, r, color='b')
    plt.title('ReLu')
    plt.show()