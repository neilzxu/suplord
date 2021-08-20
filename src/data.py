from toolz.itertoolz import interleave

import numpy as np
from scipy.stats import norm, bernoulli

from alg import get_alg

_DATA_METHOD_DISPATCH = {}


def add_data_method(name):
    def add_to_dispatch(fn):
        _DATA_METHOD_DISPATCH[name] = fn
        return fn

    return add_to_dispatch


def list_data_methods():
    return _DATA_METHOD_DISPATCH.keys()


def get_data_method(name):
    return _DATA_METHOD_DISPATCH[name]


def generate_gaussian_p_values(size, non_null_p, non_null_mean, non_null_sd,
                               null_mean, null_sd):
    alternates = bernoulli.rvs(non_null_p, size=size).astype(bool)
    noise = np.random.normal(size=size)

    noise[alternates] *= non_null_sd
    noise[alternates] += non_null_mean

    noise[~alternates] *= null_sd
    noise[~alternates] += null_mean

    p_values = 1. - norm.cdf(noise, loc=null_mean, scale=null_sd)
    return p_values, alternates


@add_data_method('gaussian')
def generate_gaussian_trials(trials,
                             hypotheses,
                             non_null_p,
                             non_null_mean,
                             non_null_sd,
                             null_mean=0,
                             null_sd=1,
                             seed=None):
    """Generate trials with random p-values based on Gaussians for null and
    non-nulls.

    P-values are for a one-tailed test on whether the alternate mean is
    larger than the null mean.
    """

    if seed is not None:
        np.random.seed(seed)
    return generate_gaussian_p_values((trials, hypotheses), non_null_p,
                                      non_null_mean, non_null_sd, null_mean,
                                      null_sd)


def generate_bimodal_gaussian_trial(hypotheses, non_null_p_1, non_null_mean_1,
                                    non_null_sd_1, non_null_p_2,
                                    non_null_mean_2, non_null_sd_2, chunks,
                                    null_mean, null_sd):
    chunk_len = hypotheses // chunks
    chunk_leftover = hypotheses % chunks
    chunk_len_list = [
        chunk_len + 1 if i < chunk_leftover else chunk_len
        for i in range(chunks)
    ]

    assert sum(chunk_len_list) == hypotheses

    def param_iter():
        while True:
            yield (non_null_p_1, non_null_mean_1, non_null_sd_1)
            yield (non_null_p_2, non_null_mean_2, non_null_sd_2)

    p_value_list = []
    alternate_list = []
    for chunk_len, (non_null_p, non_null_mean,
                    non_null_sd) in zip(chunk_len_list, param_iter()):
        p_values, alternates = generate_gaussian_p_values(
            chunk_len, non_null_p, non_null_mean, non_null_sd, null_mean,
            null_sd)
        p_value_list.append(p_values)
        alternate_list.append(alternates)
    p_values, alternates = np.concatenate(p_value_list), np.concatenate(
        alternate_list)

    return p_values, alternates


@add_data_method('bimodal_gaussian')
def generate_bimodal_gaussian_trials(trials,
                                     hypotheses,
                                     non_null_p_1,
                                     non_null_mean_1,
                                     non_null_sd_1,
                                     non_null_p_2,
                                     non_null_mean_2,
                                     non_null_sd_2,
                                     chunks,
                                     null_mean=0,
                                     null_sd=1,
                                     seed=None):
    """Generate trials with random p-values based on Gaussians for null and
    non-nulls.

    P-values are for a one-tailed test on whether the alternate mean is
    larger than the null mean.
    """

    if seed is not None:
        np.random.seed(seed)

    trial_data = [
        generate_bimodal_gaussian_trial(hypotheses, non_null_p_1,
                                        non_null_mean_1, non_null_sd_1,
                                        non_null_p_2, non_null_mean_2,
                                        non_null_sd_2, chunks, null_mean,
                                        null_sd) for _ in range(trials)
    ]

    p_values, alternates = zip(*trial_data)
    p_values, alternates = np.stack(p_values), np.stack(alternates)
    assert p_values.shape == alternates.shape
    assert p_values.shape == (trials, hypotheses), p_values.shape
    return p_values, alternates


@add_data_method('hmm')
def generate_bistate_hmm_trials(trials,
                                hypotheses,
                                transition_prob,
                                non_null_mean,
                                init_p,
                                seed=None):
    """Generate trials with Gaussians but, in a HMM model.

    p-values are still one tailed test for whether alternate mean is
    larger than null mean, but probability of the ith hypothesis being
    non-null or null is now being modeled by a 2-state HMM model.
    """
    if seed is not None:
        np.random.seed(seed)
    init_states = bernoulli.rvs(init_p, size=trials).astype(bool)
    state_list = [init_states]
    for i in range(hypotheses - 1):
        cur_states = state_list[-1]
        transition = bernoulli.rvs(transition_prob, size=trials).astype(bool)
        state_list.append(cur_states ^ transition)

    alternates = np.stack(state_list, axis=1)

    noise = np.random.normal(
        size=alternates.shape) + alternates.astype(int) * non_null_mean
    p_values = 1. - norm.cdf(noise)
    return p_values, alternates


@add_data_method('adversarial')
def generate_adversarial_trials(trials,
                                hypotheses,
                                method_name,
                                alg_kwargs,
                                non_null_p,
                                non_null_mean,
                                adversarial_p,
                                epsilon,
                                seed=None):
    basic_p_values, alternates = generate_gaussian_p_values(
        (trials, hypotheses), non_null_p, non_null_mean, 1, 0, 1)
    adversarials = bernoulli.rvs(adversarial_p, size=(trials, hypotheses))
    adv_alternates = adversarials & alternates

    new_p_value_list = []
    for trial in range(trials):
        p_values = basic_p_values[trial]
        adv_alternate_mask = adv_alternates[trial]
        method = get_alg(method_name)(**alg_kwargs)
        new_p_values = []
        for p_value, is_adv_alt in zip(p_values, adv_alternate_mask):
            if is_adv_alt:
                adv_p_value = method.next_alpha() + epsilon
                method.process_next(adv_p_value)
                new_p_values.append(adv_p_value)
            else:
                method.process_next(p_value)
                new_p_values.append(p_value)
        new_p_value_list.append(np.array(new_p_values))
    new_p_values_arr = np.stack(new_p_value_list)
    return new_p_values_arr, alternates
