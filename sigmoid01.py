def sigmoid01(x):
    """
    if x >= 0.5 then x = 1
       x <  0.5 then x = 0 
    """
    n = len(x)
    for i in range(n):
        if x[i] >= 0.5:
            x[i] = 1
        else:
            x[i] = 0
