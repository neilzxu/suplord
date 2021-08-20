from typing import List, Optional
from functools import lru_cache

import numpy as np

from alg.alg_dispatch import add_alg
from alg.utils import FDP_coef, find_default_a
from alg.gamma_scheduler import GammaScheduler


@add_alg('SupLORD')
class SupLORD:
    """Online multiple testing method that controls FDP with high
    probability."""
    def __init__(self,
                 delta: float,
                 bound: float,
                 a: float,
                 threshold: int,
                 init_param: float,
                 gamma_mode: str,
                 gamma_series: str,
                 a_mode: str = 'manual',
                 alpha_strategy: int = 2,
                 gamma_decay_coef: Optional[float] = None,
                 gamma_decay_len: Optional[int] = None,
                 gamma_exponent: Optional[float] = None,
                 init_mode: str = 'flat'):
        """
        Parameters
        ----------
        delta, bound, threshold:
            The FDP is less than bound with 1 - delta probability after threshold number of rejections are made.
        a:
            hyperparameter for controlling how much wealth is gained on each rejection vs at the beginning of the algorithm.
        init_param:
            parameter related to how wealth is allocated before algorithm makes threshold rejections.
        gamma_*:
            arguments to how gamma series is produced - see documentation of GammaScheduler.
        a_mode:
            whether a is chosen using the a argument, or automatically (i.e. the canonical a).
        alpha_strategy:
            spending strategy - spends based on distance from each prior rejection (if 2) or distance from last rejection (if 3).
        init_mode:
            how the wealth is allocated before threshold rejections have been made.
            - flat: allocate all wealth on the first and last rejection in this period.
            - decay: allocate a decaying amount over all threshold rejections.
            - uniform: uniformly split wealth over all threshold rejections.
        """
        delta,

        self.delta = delta
        self.bound = bound
        self.a = a

        self.threshold = threshold
        self.gamma_scheduler = GammaScheduler(mode=gamma_mode,
                                              series=gamma_series,
                                              decay_len=gamma_decay_len,
                                              decay_coef=gamma_decay_coef,
                                              exponent=gamma_exponent)

        assert a_mode in ['manual', 'auto']
        if a_mode == 'auto':
            self.a = find_default_a(bound, delta, threshold, 0.0001)

        assert init_mode in ['flat', 'decay', 'uniform']
        self.init_mode = init_mode
        self.init_param = init_param

        self.reject_earning = bound / self._FDP_coef()
        self.init_wealth = self.reject_earning * (self.threshold) - self.a
        assert self.init_wealth > 0, f'Wealth: {self.init_wealth} too small. a: {self.a}, delta: {delta}, bound: {bound}, threshold: {threshold}'

        if init_mode == 'flat':
            assert self.init_param > 0 and self.init_param <= 1
            self.init_gain_vec = np.zeros(self.threshold)
            self.init_gain_vec[0] = self.init_param * self.init_wealth
            self.init_gain_vec[-1] += (1 - self.init_param) * self.init_wealth
        elif init_mode == 'decay':
            init_idxs = np.arange(1, self.threshold + 1)
            init_vec = make_vec_vals(init_idxs, init_param)
            self.init_gain_vec = init_vec * self.init_wealth / np.sum(init_vec)
        else:  # init_mode == 'uniform'
            self.init_gain_vec = np.array(
                [0.1 * self.reject_earning, 0.9 * self.reject_earning] +
                [self.reject_earning for _ in range(self.threshold - 2)])

        assert alpha_strategy in [2, 3]
        self.alpha_strategy = alpha_strategy

        self.w0 = self.init_gain_vec[0]
        self.alpha: List[float] = []
        self.wealth_vec = [self.w0]
        self.rej_indices: List[int] = []

    def _FDP_coef(self) -> float:
        """Returns FDP coefficient with current delta and a."""
        return FDP_coef(self.delta, self.a)

    def _FDP_estimate(self) -> float:
        """High probability estimate of FDP based on wealth spent."""
        return self._FDP_coef() * (self.a +
                                   np.sum(self.alpha)) / self.rejset.sum()

    def compute_FDP_set(self) -> np.ndarray:
        """Compute array of all FDP estimates for each step up to the present
        organized from earliest to latest."""
        return self._FDP_coef() * np.cumsum(np.array(self.alpha)) / np.maximum(
            np.cumsum(self.rejset), 1)

    def _pre_threshold_alpha(self, idx: int) -> float:
        """Calculates next alpha value when fewer than threshold rejections
        have been made.

        Parameters
        ----------
        idx:
            index of hypothesis this alpha_value is to be produced for.
        """

        all_rej_indices = np.array([-1] + self.rej_indices)[:self.threshold]
        all_pre_rej_gains = self.init_gain_vec[:len(all_rej_indices)]
        gamma_coefs = self.gamma_scheduler.make_gammas(
            dists=idx - all_rej_indices,
            wealth_vec=np.array(self.wealth_vec)[all_rej_indices + 1],
            w0=self.w0)
        assert len(gamma_coefs) == len(
            all_pre_rej_gains
        ), f'{gamma_coefs.shape}, {all_pre_rej_gains.shape}, {all_rej_indices.shape}'
        return np.sum(gamma_coefs * all_pre_rej_gains)

    def _post_threshold_alpha(self, idx: int) -> float:
        """Calculates next alpha value when more than or equal to threshold
        rejections have been made.

        Parameters
        ----------
        idx:
            index of hypothesis this alpha_value is to be produced for.
        """
        post_idxs = np.array(self.rej_indices[(self.threshold - 1):])
        post_dists = idx - post_idxs
        gamma_coefs = self.gamma_scheduler.make_gammas(
            post_dists,
            np.array(self.wealth_vec)[post_idxs + 1], self.w0)

        return np.sum(gamma_coefs * self.reject_earning)

    def _calc_exp_alpha(self) -> float:
        """Calculates alpha if alpha_strategy = 3 (based on current
        wealth and distance from last rejection)."""
        if self.rej_indices:
            last_rej_idx = self.rej_indices[-1]
        else:
            last_rej_idx = -1
        dist_to_last_rej = len(self.alpha) - last_rej_idx
        gamma_coef = self.gamma_scheduler.make_gammas(
            np.array([dist_to_last_rej]),
            np.array([self.wealth_vec[last_rej_idx + 1]]), self.w0)[0]
        return gamma_coef * self.wealth_vec[last_rej_idx + 1]

    def _calc_wealth_gain(self, idx) -> float:
        """Calculates wealth earned by algorithm if a rejection is made at
        idx."""
        if len(self.rej_indices) + 1 >= self.threshold:
            return self.reject_earning
        else:
            return self.init_gain_vec[len(self.rej_indices) + 1]

    def next_alpha(self) -> float:
        """Alpha value for the next hypothesis."""
        if self.alpha_strategy == 2:
            i = len(self.alpha)
            cur_alpha = self._pre_threshold_alpha(i)
            if len(self.rej_indices) >= self.threshold:
                cur_alpha += self._post_threshold_alpha(i)
            return cur_alpha
        else:  # alpha_strategy == 3
            return self._calc_exp_alpha()

    def process_next(self, p) -> bool:
        """Incorporates the next p-value into the algorithm.

        Returns bool of whether the p-value was rejected.
        """
        cur_alpha = self.next_alpha()
        if p < cur_alpha:
            i = len(self.alpha)
            self.rej_indices.append(i)
            self.wealth_vec.append(self.wealth_vec[-1] - cur_alpha +
                                   self._calc_wealth_gain(i))
        else:
            self.wealth_vec.append(self.wealth_vec[-1] - cur_alpha)
        self.alpha.append(cur_alpha)
        return p < cur_alpha

    def run_fdr(self, pvec):
        """Runs the algorithm on a vector of p-values."""
        assert isinstance(pvec, np.ndarray)
        for _, p in enumerate(list(pvec.tolist())):
            self.process_next(p)
        self.rejset = np.zeros(len(pvec))
        self.rejset[self.rej_indices] = 1
        return self.rejset
