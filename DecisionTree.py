import numpy as np

class DecisionTree:
    def fit(self, X, y):

        self.is_leaf = False
        if len(np.unique(y)) == 1:
            self.is_leaf = True
            self.label = y[0]
            return

        # Entropy of each attribute of the data set
        X_entropy = np.apply_along_axis(self.Entropy, 0, X, y)
        
        self.root_node = np.argmin(X_entropy)
        attribute = X[:, self.root_node]
        rule = np.unique(attribute)
        self.sub_rules = {}
                
        for value in rule:
            # Indices of a certain value of the attribute
            indices = attribute == value

            # Y values of those indices
            new_y = y[indices]
            if new_y.size == 0:
                continue
            
            # X values of those indices after removing the attribute column
            reduced_X = X[indices]
            new_X = np.delete(reduced_X, self.root_node, axis=1)
            
            sub_tree = DecisionTree()
            sub_tree.fit(new_X, new_y)
            self.sub_rules[value] = sub_tree

    def Entropy(self, c, y):
        n = len(c)
        unique, frequency = np.unique(c, return_counts=True)
        infogain = np.zeros_like(unique, dtype=float)
        
        for i, val in enumerate(unique):
            reduced_y = y[c == val]
            infogain[i] = self.Infogain(reduced_y)
        
        return np.sum(infogain * frequency) / n

    def Infogain(self, z):
        n = len(z)
        unique, frequency = np.unique(z, return_counts=True)
        frequency_n = frequency / n
        log_frequency_n = np.log(np.where(frequency_n == 0, 1, frequency_n))
        z_infogain = np.sum(frequency_n * log_frequency_n)
        return -z_infogain

    def predict_single(self, x):
        if self.is_leaf:
            return self.label
        
        subtree = self.sub_rules.get(x[self.root_node], None)
        if subtree is None:
            return -1  # Handle case where value not seen during training
        
        x_reduced = np.delete(x, self.root_node)
        return subtree.predict_single(x_reduced)
    
    def predict(self, X):
        return np.apply_along_axis(self.predict_single, 1, X)

if __name__ == "__main__":

    from sklearn.datasets import load_iris
    from sklearn.tree import DecisionTreeClassifier
    from timeit import default_timer as timer
    iris = load_iris()
    X = iris.data
    y = iris.target

    model = DecisionTree()
    start = timer()
    model.fit(X, y)
    end = timer() - start
    yh = model.predict(X)
    print(end)
    print(yh)

    model = DecisionTreeClassifier()
    start = timer()
    model.fit(X, y)
    end = timer() - start
    yh = model.predict(X)
    print(end)
    print(yh)
