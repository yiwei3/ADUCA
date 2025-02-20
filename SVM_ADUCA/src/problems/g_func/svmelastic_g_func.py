import numpy as np

class SVMElasticGFunc:
    def __init__(self, d, n, lambda1, lambda2):
        self.lambda1 = lambda1  # Lasso regularization parameter
        self.lambda2 = lambda2  # Ridge regularization parameter
        self.d = d
        self.n = n

    def func_value(self, x):
        ret_1 = np.sum(np.abs(x[:self.d]))  # L1 regularization part
        # print(f"!!! np.sum(x[:self.d]): {np.sum(x[:self.d])} ")
        ret_2 = np.sum(x[:self.d] ** 2)     # L2 regularization part
        ret = self.lambda1 * ret_1 + (self.lambda2 / 2) * ret_2

        # Constraint check for elements of x[d+1:] to be in [-1, 0]
        # if not np.all((-1.0 <= x[self.d:]) & (x[self.d:] <= 0.0)):
        #     return -np.inf
        return ret

    def prox_opr_coordinate(self, j, u, tau):
        if j <= self.d:
            p1 = tau * self.lambda1
            p2 = 1.0 / (1.0 + tau * self.lambda2)
            return self._prox_func(u, p1, p2)
        else:
            return min(0.0, max(-1.0, u))
        
    def prox_opr_block(self, block:range, u_block, tau: np.array):
        if block.stop <= self.d:
            p1 = tau * self.lambda1
            p2 = 1.0 / (1.0 + tau * self.lambda2)
            prox = p2 * np.sign(u_block) * np.maximum(0, np.abs(u_block) - p1)
        elif block.start >= self.d:
            prox = np.minimum(0, np.maximum(-1, u_block))
        else:
            p1 = tau * self.lambda1
            p2 = 1.0 / (1.0 + tau * self.lambda2)
            prox_1 = p2 * np.sign(u_block[:self.d-block.start]) * np.maximum(0, np.abs(u_block[:self.d-block.start]) - p1)
            prox_2 = np.minimum(0, np.maximum(-1, u_block[self.d-block.start:]))
            prox = np.concatenate((prox_1, prox_2))
        return prox

    @staticmethod
    def _prox_func(_x0, p1, p2):
        if _x0 > p1:
            return p2 * (_x0 - p1)
        elif _x0 < -p1:
            return p2 * (_x0 + p1)
        else:
            return 0.0
        
    def prox_opr(self, u, τ, d):
        p1 = τ * self.lambda1
        p2 = 1.0 / (1.0 + τ * self.lambda2)
        prox = p2 * np.sign(u[:self.d]) * np.maximum(0, np.abs(u[:self.d]) - p1)
        
        p = np.minimum(0, np.maximum(-1, u[self.d:]))
        new_u = np.concatenate((prox,p))
        # print(f"!!! np.sum(p): {np.sum(p)} ")
        return new_u