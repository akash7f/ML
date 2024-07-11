import numpy as np

def KMeans(X, k, initial_centers = None):
    samples = len(X)

    if k < 1 or samples < k:
        return
    elif k == 1:
        return np.zeros(samples, dtype=int)
    elif samples == k:
        np.arange(samples)

    y = np.zeros(samples, dtype=int)
    centers = X[:k] if initial_centers is None else initial_centers

    while True:      

        # new_y = np.zeros(samples, dtype=int)
        # for i in range(samples) :
        #     datapoint = X[i]
        #     distances = np.sum((centers - datapoint)**2, axis=1)
        #     new_y[i] = np.argmin(distances)

        # vectorized of above
        distances = np.linalg.norm(X[:, np.newaxis] - centers, axis=2)
        new_y = np.argmin(distances, axis=1)

        if np.array_equal(y, new_y):
            return y

        y = new_y
        for i in range(k):            
            points = X[y == i]
            if len(points) > 0:
                centers[i] = np.mean(points, axis=0)
    
# def get_centre(datapoint, centers):
#     distances = np.sum((centers - datapoint)**2, axis=1)
#     return np.argmin(distances)

def KMeansPlus(X, k):
    centers = np.zeros(shape=(k, len(X[0])))
    
    centers[0] = X[0]
    for i in range(1, k):
        alpha = np.sum(centers, axis=0)/i
        distances = np.sum((X - alpha)**2, axis=1)
        index = np.argmax(distances)
        centers[i] = X[index]
    
    return KMeans(X, k, centers)


if __name__ == "__main__":

    from sklearn.datasets import load_iris
    iris = load_iris()
    X = iris.data

    print(KMeans(X, 3))
    print(KMeansPlus(X, 3))
