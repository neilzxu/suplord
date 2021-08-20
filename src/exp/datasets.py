from typing import Any, Dict, List, Tuple
from itertools import product

import numpy as np

DataSpec = Tuple[str, Dict[str, Any]]


def hmm_dataset(non_null_mean_min: int,
                non_null_mean_max: int,
                trans_min: float = 0.1,
                trans_max: float = 1,
                trans_step: float = 0.1,
                init_p: float = 0.5) -> List[DataSpec]:
    """Produces spec associated with HMM datasets.

    Range of signals in datasets span integer values [non_null_mean_min,
    non_null_mean_max] and transition probabilities at trans_step
    intervals from trans_min to trans_max (exclusive). init_p specifies
    the parameter for the Bernoulli RV determining whether the initial
    hypothesis is non-null.
    """
    param_list = list(
        product(np.arange(trans_min, trans_max, trans_step),
                range(non_null_mean_min, non_null_mean_max + 1)))
    seeds = range(322, 322 + len(param_list))
    return [('hmm', {
        'trials': 200,
        'hypotheses': 1000,
        'transition_prob': transition_prob,
        'non_null_mean': non_null_mean,
        'init_p': init_p,
        'seed': seed
    }) for seed, (transition_prob, non_null_mean) in zip(seeds, param_list)]


def signal_comp_dataset(signal_min: int,
                        signal_max: int,
                        prob_min: float = 0.1,
                        prob_max: float = 1,
                        prob_step: float = 0.1) -> List[DataSpec]:
    """Produces spec associated with constant datasets.

    Range of signals in datasets span integer values in [signal_min,
    signal_max] with probabilities of non-null ranging from prob_min to
    prob_max (exclusive) in steps of prob_step.
    """
    gaussian_params = list(
        product(np.arange(prob_min, prob_max, prob_step),
                range(signal_min, signal_max + 1)))
    gaussian_seeds = range(322, 322 + len(gaussian_params))
    return list(
        product(['gaussian'], [{
            'trials': 200,
            'hypotheses': 1000,
            'non_null_mean': mean,
            'non_null_sd': 1,
            'null_mean': 0,
            'null_sd': 1,
            'non_null_p': p,
            'seed': seed
        } for seed, (p, mean) in zip(gaussian_seeds, gaussian_params)]))
