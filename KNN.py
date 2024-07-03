import numpy as np

def sbr(A, i):
    sorted_indices = np.argsort(A[i])
    sorted_A = A[:, sorted_indices]
    return sorted_A

class KNN:
    
    def fit(self, x, y):
        """
        m samples of n dimensional data
        x = [[x11, x12...x1n],
             [x21, x22...[2n],
             ...
             [xm1, xm2...[mn]]
    
        m outputs for m samples of data
        y = [y1, y2, y3...ym]

        """
        self.inputs, self.outputs, self.samples = x, y, len(x)

        max_k = self.samples
        k_error = np.zeros(shape=(2, max_k//2 - 1), dtype=int)
        for self.K in range(3, max_k, 2):

            yh = self.predict(x)

            error = 0
            for i in range(0, self.samples):
                if yh[i] != self.outputs[i]:
                    error += 1

            Knot = self.K//2 - 1
            k_error[0][Knot] = self.K
            k_error[1][Knot] = error
        
        sorted_k_error = sbr(k_error, 1)
        self.K = sorted_k_error[0][0]

    def predict(self, x):
        
        yh = np.zeros(len(x))
        i = 0
        for test in x:
            distances = np.zeros(self.samples)
            j = 0
            for train in self.inputs:
                distances[j] = np.sqrt(np.sum((train - test)**2))
                j+=1

            dis_out = np.vstack([distances, self.outputs.copy()])
            sorted_dis_out = sbr(dis_out, 0)
            k_outputs = sorted_dis_out[1][:self.K]

            unique, frequencies = np.unique(k_outputs, return_counts=True)
            uni_fre = np.vstack([unique, frequencies])
            sorted_uni_fre = sbr(uni_fre, 1)

            yh[i] = sorted_uni_fre[0][-1]
            i+=1

        return yh
    
    def print(self):
        print()
        print("K : ", self.K)
        print()


if __name__ == "__main__":

    x = np.array([[0.1, 0.2], [0.3, 0.3] ,[0.6, 0.7], [1, 0.9],[1, 1], [1.5, 2], [0.9, 2], [1.1, 1], [1.2, 0.4], [0.5, 0.4]])
    y = np.array([0, 0, 0, 1, 1, 1, 1, 1, 0, 0])
    model = KNN()
    model.fit(x, y)
    model.print()
    print(model.predict([[1.1, 0.9]]))