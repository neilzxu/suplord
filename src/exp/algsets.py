from typing import Any, Dict, List, Tuple

from itertools import product
import os

import numpy as np

from alg import FDP_coef

AlgSpec = Tuple[str, Dict[str, Any]]


def baseline_lord(alpha: float, is_pp: bool, alpha_strategy: int) -> AlgSpec:
    return ('LORD', {
        'delta': alpha,
        'bound': None,
        'startfac': 0.1,
        'is_pp': is_pp,
        'alpha_strategy': alpha_strategy,
        'gamma_series': 'gaussian',
        'gamma_exponent': None
    })


def baseline_lordfdx(delta: float, bound: float, is_pp: bool,
                     alpha_strategy: int) -> AlgSpec:
    return ('LORD', {
        'delta': delta,
        'bound': bound,
        'startfac': 0.1,
        'is_pp': is_pp,
        'alpha_strategy': alpha_strategy,
        'gamma_series': 'gaussian',
        'gamma_exponent': None
    })


def baseline_bonferroni(alpha: float) -> AlgSpec:
    return ('Bonferroni', {
        'alpha': alpha,
        'gamma_series': 'gaussian',
        'gamma_exponent': None
    })


def suplord_v_lordfdx(delta: float,
                      bound: float,
                      r_list: List[int],
                      auto: bool = False) -> Tuple[List[str], List[AlgSpec]]:
    """names and specs for set of algorithm specs to compare SupLORD vs
    baseline online multiple testing methods."""
    suplords = [('SupLORD', {
        'delta': delta,
        'bound': bound,
        'a_mode': 'manual' if not auto else 'auto',
        'a': 1,
        'alpha_strategy': 2,
        'threshold': r,
        'init_mode': 'uniform',
        'init_param': None,
        'gamma_mode': 'static',
        'gamma_series': 'gaussian',
    }) for r in r_list]
    names = [f'SupLORD $r^*={r}$'
             for r in r_list] + ['LORDFDX', 'LORD', 'Bonferroni']
    return names, suplords + [
        baseline_lordfdx(delta, bound, is_pp=True, alpha_strategy=2),
        baseline_lord(delta, is_pp=True, alpha_strategy=2),
        baseline_bonferroni(delta)
    ]


def dynamic_comparison(
        delta: float, bound: float, threshold: int, decay_lens: List[int],
        decay_coefs: List[float]) -> Tuple[List[str], List[AlgSpec]]:
    """names and specs for set of algorithm specs to compare dynamic vs static
    schedules."""
    base_params = {
        'delta': delta,
        'bound': bound,
        'a': 1,
        'threshold': threshold,
        'init_param': None,
        'gamma_series': 'gaussian',
        'a_mode': 'manual',
        'gamma_exponent': None,
        'init_mode': 'uniform'
    }

    len_coef_list = list(product(decay_lens, decay_coefs))

    dynamics = [('SupLORD', {
        'gamma_mode': 'dynamic',
        'alpha_strategy': 2,
        'gamma_decay_coef': decay_coef,
        'gamma_decay_len': decay_len,
        **base_params
    }) for decay_len, decay_coef in len_coef_list]

    lord2 = ('SupLORD', {
        'gamma_mode': 'static',
        'alpha_strategy': 2,
        'gamma_decay_coef': None,
        'gamma_decay_len': None,
        **base_params
    })

    lord3 = ('SupLORD', {
        'gamma_mode': 'static',
        'alpha_strategy': 3,
        'gamma_decay_coef': None,
        'gamma_decay_len': None,
        **base_params
    })
    names = ['Static SupLORD1'] + [
        f'Dynamic $\ell={decay_len},\eta={decay_coef}$'
        for decay_len, decay_coef in len_coef_list
    ] + ['Static SupLORD2']
    return names, [lord2] + dynamics + [lord3]
