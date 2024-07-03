import numpy

def Linear(x):
    """
    Linear Activation Function
    It is used in regression and it is also known as no function 
    """
    return x.copy()

def Sigmoid(x):
    """
    Sigmoid Activation Function
    It is used in Binary Classification
    """
    # 0 < y < 1 for -infinity < x < infinity
    return 1/(1 + numpy.exp(-x))

def ReLu(x):
    """
    Rectified Linear Unit Function
    It is used in regression where the prediction can't be negative
    """
    # if x[i] < 0 then y[i] = 0 else y[i] = x[i]
    return numpy.where(x < 0, 0, x)

if __name__  == "__main__":
    a = numpy.random.randint(-100, 100, 5)
    print(a)
    print(Sigmoid(a))
    print(ReLu(a))