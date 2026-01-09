import numpy as np
import warnings
# from scipy.linalg.blas import dgemm
from scipy.sparse import csr_matrix

class NASHOprFunc:
    def __init__(self, n, gamma, beta, c , L):
        self.n = n
        self.gamma = gamma
        self.beta = beta
        self.c = c
        self.L = L
        # Prevent zero/negative totals from hitting invalid power operations
        self._min_total_quantity = 1e-12
        self._inv_beta = 1.0 / self.beta
        self._inv_beta_plus_one = 1.0 + self._inv_beta
        self._coef_f = 1.0 / self._inv_beta_plus_one
        self._L_pow = np.power(self.L, self._inv_beta)
        self._p_const = 5000 ** (1.0 / self.gamma)
        self._dp_const = -(1.0 / self.gamma) * self._p_const
        self._p_power = -1.0 / self.gamma
        self._dp_power = -1.0 / self.gamma - 1.0

    def _clip_total_quantity(self, Q):
        return max(Q, self._min_total_quantity)

    def f(self, q):
        q = np.maximum(q, 0)
        res = self.c * q + self._coef_f * (self._L_pow * np.power(q, self._inv_beta_plus_one))
        return res
    
    def f_block(self, q, block:range):
        q_block = np.maximum(q[block], 0)
        t = self._inv_beta[block]
        res = self.c[block] * q_block + (1.0 / (1.0 + t)) * (self._L_pow[block] * np.power(q_block, 1.0 + t))
        return res

    def df(self, q):
        q = np.maximum(q, 0)
        res = self.c + self._L_pow * np.power(q, self._inv_beta)
        return res
    
    def df_block(self, q_block, block:range):
        q_block = np.maximum(q_block, 0)
        t = self._inv_beta[block]
        res = self.c[block] + self._L_pow[block] * np.power(q_block, t)
        return res
    
    def p(self, Q):
        Q = self._clip_total_quantity(Q)
        return self._p_const * (Q ** self._p_power)
        
    
    def dp(self, Q):
        Q = self._clip_total_quantity(Q)
        return self._dp_const * (Q ** self._dp_power)
    
    def func_map(self, q):
        Q = np.sum(q)
        res = self.df(q) - self.p(Q) - q*self.dp(Q)
        return res

    def func_map_block(self, q_block, Q, block):
        res = self.df_block(q_block, block) - self.p(Q) - q_block*self.dp(Q)
        return res
    
    def func_map_block_update(self, F, q, p, p_, dp, dp_, block:range):
        q_block = q[block]
        F[block] = self.df_block(q_block, block) - p - q_block * dp
        delta_p = p_ - p
        delta_dp = dp_ - dp
        if delta_p != 0.0 or delta_dp != 0.0:
            if block.start > 0:
                if delta_p != 0.0:
                    F[:block.start] += delta_p
                if delta_dp != 0.0:
                    F[:block.start] += q[:block.start] * delta_dp
            if block.stop < q.size:
                if delta_p != 0.0:
                    F[block.stop:] += delta_p
                if delta_dp != 0.0:
                    F[block.stop:] += q[block.stop:] * delta_dp
        return F

    # def func_map_block_sample(self, j, t, x):
    #     assert len(x) == self.d + self.n
    #     assert 1 <= j <= self.d + self.n
    #     assert 1 <= t <= self.n

    #     if j <= self.d:
    #         return x[self.d + t - 1] * self.b[t - 1] * self.A[t - 1, j - 1]
    #     elif j - self.d == t:
    #         return - (self.b[t - 1] * (self.A[t - 1, :] @ x[:self.d]) - 1)
    #     else:
    #         return 0.0
