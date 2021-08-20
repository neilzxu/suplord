import numpy as np

from alg.alg_dispatch import add_alg
from alg.gamma_scheduler import make_raw_gamma_coefs


@add_alg('Bonferroni')
class Bonferroni:
    def __init__(self, alpha, gamma_series, gamma_exponent=None):
        self.alpha = alpha
        gamma_vec = make_raw_gamma_coefs(range(1, 200000), gamma_series,
                                         gamma_exponent)
        self.gamma_vec = gamma_vec / np.sum(gamma_vec)
        self.alpha = self.gamma_vec * self.alpha
        self.rejset = None
        self.wealth_vec = alpha - self.alpha

    def run_fdr(self, p_values):
        self.alpha = self.alpha[:len(p_values)]
        self.wealth_vec = self.wealth_vec[:len(p_values)]
        self.rejset = p_values < self.alpha
        return self.rejset.astype(int)
