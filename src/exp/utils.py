import hashlib
import json
import multiprocessing as mp
import pickle

import numpy as np

from alg import get_alg
from data import get_data_method
from result import Result
SALT = 222
HASH_SIZE = 32


def generate_path(data, data_kwargs, alg, alg_kwargs):
    def make_name(key, value):
        if isinstance(value, float):
            return f'{key}={value:.3f}'
        else:
            return f'{key}={value}'

    msg = json.dumps([data, data_kwargs, alg, alg_kwargs])
    # msg = f'{data},' + ','.join([
    #     make_name(key, data_kwargs[key]) for key in sorted(data_kwargs.keys())
    # ]) + f',{alg},' + ','.join(
    #     [make_name(key, alg_kwargs[key]) for key in sorted(alg_kwargs.keys())])

    # bytes is called on an iterable
    hasher = hashlib.blake2b(salt=bytes([SALT]), digest_size=HASH_SIZE)
    hasher.update(msg.encode('utf-8'))
    return str(hasher.hexdigest())


def exec_exp(data,
             data_kwargs,
             alg,
             alg_kwargs,
             name=None,
             out_dir='comp_exp'):
    path = generate_path(data, data_kwargs, alg, alg_kwargs)

    final_path = f'{out_dir}/{path}.pkl'
    p_values, alternates = get_data_method(data)(**data_kwargs)
    instances = []
    rejsets = []

    trials, _ = p_values.shape
    alg_fn = get_alg(alg)
    for i in range(trials):
        state = alg_fn(**alg_kwargs)
        rejset = state.run_fdr(p_values[i]).astype(bool)
        rejsets.append(rejset)
        instances.append(state)

    result = Result(name if name is not None else alg, data, alg,
                    alg_kwargs, data_kwargs, p_values, alternates,
                    np.stack(rejsets), instances)
    with open(final_path, 'wb') as out_f:
        pickle.dump(result, out_f)
        print(f"Saved results to: {final_path}")
