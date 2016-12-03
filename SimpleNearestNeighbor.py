import numpy as np


class NearestNeighbor(object):
    def __init__(self, distance_type=2, euristic=False):
        self.distance_type = distance_type
        self.euristic = euristic
        pass

    def fit(self, X, y):
        """ X is N x D where each row is an example. Y is 1-dimension of size N """
        # the nearest neighbor classifier simply remembers all the training data
        self.Xtr = X
        self.ytr = y

    def predict(self, X):
        """ X is N x D where each row is an example we wish to predict label for """
        num_test = X.shape[0]
        # lets make sure that the output type matches the input type
        Ypred = np.zeros(num_test, dtype=self.ytr.dtype)

        if self.distance_type == 1:
            fun = lambda x, y, i : np.sum(np.abs(x - y[i, :]), axis=1)
        elif self.euristic:
            fun = lambda x, y, i:  np.sum(np.square(x - y[i, :]), axis=1)
        else:
            fun = lambda x, y, i: np.sqrt(np.sum(np.square(x - y[i, :]), axis=1))

        # loop over all test rows
        for i in xrange(num_test):
            # find the nearest training image to the i'th test image
            # using the L1 distance (sum of absolute value differences)
            distances = fun(self.Xtr, X, i)
            min_index = np.argmin(distances)  # get the index with smallest distance
            Ypred[i] = self.ytr[min_index]  # predict the label of the nearest example
        return Ypred
