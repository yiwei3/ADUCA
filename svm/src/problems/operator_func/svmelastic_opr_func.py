import numpy as np
from scipy.sparse import csr_matrix

"""
A's shape: (n, d), where n is the number of samples, d is the number of features.
b's shape: (n, ), labels of samples.
"""
class SVMElasticOprFunc:
    def __init__(self, data):
        self.d = data.features.shape[1]
        self.n = len(data.values)
        self.A = data.features
        self.A_T = self.A.T
        self.b = data.values
        self.A_sparse = csr_matrix(self.A)
        self.A_sparse_T = self.A_sparse.T

    def func_value(self, x):
        assert len(x) == self.d + self.n
        _x = x[:self.d]
        res = 1 - (self.b * (self.A @ _x))  # Element-wise multiplication
        return np.sum(np.maximum(res, 0)) / self.n

    def func_map(self, x):
        ret = np.zeros(self.d + self.n)
        ret[:self.d] = self.A_T @ (x[self.d:] * self.b)
        ret[self.d:] = 1 - self.b * (self.A @ x[:self.d])
        return ret / self.n

    def func_map_block(self, j, x):
        assert len(x) == self.d + self.n
        assert 1 <= j <= self.d + self.n

        if j <= self.d:
            ret = 0.0
            for t in range(self.n):
                ret += x[self.d + t] * self.b[t] * self.A[t, j - 1]
            return ret / self.n
        else:
            t = j - self.d - 1
            return -(self.b[t] * (self.A[t, :] @ x[:self.d]) - 1) / self.n
        
    def func_map_coordinate_update(self, F, uslice, uslice_, j):
        if j >= self.d:
            F[:self.d] += (uslice - uslice_) * self.b[j-self.d] * self.A[j-self.d, :] / self.n
        else:
            F[self.d:] += -self.b * (self.A[:, j] * (uslice - uslice_)) / self.n
        return F
    
    def func_map_block_update(self, F, uslice, uslice_, block: slice):
        # only update alpha
        if block.start >= self.d:
            F[:self.d] += (self.A_sparse[block.start-self.d:block.stop-self.d].T.dot((self.b[block.start-self.d:block.stop-self.d]*(uslice - uslice_)))) / self.n
        # only update x
        elif block.stop <= self.d:
            F[self.d:] += -self.b * (self.A_sparse[:, block].dot((uslice - uslice_))) / self.n
        # update x (uslice[: self.d-block.start]), and update alpha (uslice[self.d-block.start:])
        else:
            F[:self.d] += ((self.b[:block.stop-self.d] *(uslice[(self.d - block.start):] - uslice_[(self.d - block.start):])) @ self.A_sparse[:(block.stop - self.d)]) / self.n
            F[self.d:] += -self.b * (self.A_sparse[:, block.start:] @ (uslice[:(self.d - block.start)] - uslice_[:(self.d - block.start)])) / self.n
        return F

    def func_map_block_sample(self, j, t, x):
        assert len(x) == self.d + self.n
        assert 1 <= j <= self.d + self.n
        assert 1 <= t <= self.n

        if j <= self.d:
            return x[self.d + t - 1] * self.b[t - 1] * self.A[t - 1, j - 1]
        elif j - self.d == t:
            return - (self.b[t - 1] * (self.A[t - 1, :] @ x[:self.d]) - 1)
        else:
            return 0.0
