import numpy as np


def fdr(alternates, rejsets):
    rejected_nulls = ~alternates & rejsets
    total_rej = np.sum(rejsets)
    if total_rej == 0:
        return 0
    else:
        return np.sum(rejected_nulls) / total_rej


def power(alternates, rejsets):
    rejected_alts = alternates & rejsets
    total_alts = np.sum(alternates)
    if total_alts == 0:
        return 1
    else:
        return np.sum(rejected_alts) / total_alts


def fdx(alternates, rejsets, gamma, threshold):
    trials, hypotheses = alternates.shape

    rej_null_sum = np.cumsum((rejsets & ~alternates).astype(int), axis=1)
    rej_sum = np.cumsum(np.copy(rejsets).astype(int), axis=1)

    rej_sum = np.maximum(rej_sum, 1)

    idxs = rej_sum > threshold
    fdps = (rej_null_sum / rej_sum) * idxs.astype(int)
    worst_fdp = np.max(fdps, axis=1)

    return np.sum(worst_fdp > gamma) / trials


def rejsize(rejsets):
    return np.cumsum(np.copy(rejsets).astype(int), axis=1)


def fdp(alternates, rejsets):
    rej_null_sum = np.cumsum((rejsets & ~alternates).astype(int), axis=1)
    rej_sum = np.cumsum(np.copy(rejsets).astype(int), axis=1)
    rej_sum = np.maximum(rej_sum, 1)
    return rej_null_sum / rej_sum


def expected_sup_fdp(alternates, rejsets, threshold):
    fdps = fdp(alternates, rejsets)
    rejsizes = rejsize(rejsets)
    return np.mean(np.max(fdps * (rejsizes >= threshold).astype(int), axis=1))
