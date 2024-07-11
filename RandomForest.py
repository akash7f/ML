from DecisionTree import DecisionTree as DTC
import numpy as np

class RandomForest:
    def fit(self, X, y, n_trees=99):

        self.bootstrap_samples = np.empty(n_trees, dtype=object)
        self.random_forest = np.empty(n_trees, dtype=DTC)
        
        for _ in range(n_trees):

            X_bootstrap, y_bootstrap = self.BootStrapData(X, y)
            self.random_forest[_] = DTC()
            self.random_forest[_].fit(X_bootstrap, y_bootstrap)
        self.n = n_trees
 
    def BootStrapData(self, X, y):
        samples = len(X)
        indices = np.random.choice(range(samples), size=samples, replace=True)
        X_bootstrap = X[indices]
        y_bootstrap = y[indices]
        return X_bootstrap, y_bootstrap

    def predict(self, X):

        predictions = np.array([tree.predict(X) for tree in self.random_forest])
        predictions = predictions.T
        predictions = np.where(predictions == None, 0, predictions)

        yh = np.empty(len(X), dtype=object)
        for i in range(len(X)):
            unique, frequencies = np.unique(predictions[i], return_counts=True)
            t = np.argmax(frequencies)
            yh[i] = unique[t]
        return yh
    
if __name__ == "__main__":

    from sklearn.datasets import load_iris
    iris = load_iris()
    X = iris.data
    y = iris.target
    model = RandomForest()
    model.fit(X, y)
    yh = model.predict(X)
    print(yh)