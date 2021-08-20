from typing import Any, Dict, List, Union, Tuple

import argparse
from itertools import chain, product, repeat
import multiprocessing as mp
import os
import pickle
from pprint import pprint
import re
import time

import numpy as np
from tqdm import tqdm

from alg import get_alg, FDP_coef
from analysis import *
from exp.algsets import *
from exp.datasets import *
from exp import styles
from exp.utils import generate_path, exec_exp


def run_exp(out_dir: str,
            alg_kwargs_list: List[AlgSpec],
            data_kwargs_list: List[DataSpec],
            names: Union[List[str], None] = None,
            processes: int = 20) -> None:
    """Run experiments corresponding to product of configs for algs and data.

    Concurrent processing with processes sized thread pool and saves
    results of each trial in hash pathname inside out_dir.
    """

    OUT_DIR = out_dir
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    flat_args = [
        (data, data_kwargs, alg, alg_kwargs)
        for (data, data_kwargs), (
            alg, alg_kwargs) in product(data_kwargs_list, alg_kwargs_list)
    ]

    # Hash all args and check if hashing failed to get paths for saving the results of each trial.
    filenames = [generate_path(*args) for args in flat_args]
    # Check if hashing failed
    assert len(set(filenames)) == len(filenames)

    # Create final set of arguments
    if names is None:
        flat_args = [
            (*args, None, OUT_DIR) for name, args in zip(filenames, flat_args)
            if not os.path.exists(os.path.join(OUT_DIR, f'{name}.pkl'))
        ]
    else:
        flat_names = list(chain(*(repeat(names, len(data_kwargs_list)))))
        flat_args = [
            (*args, method_name, OUT_DIR) for name, args, method_name in zip(
                filenames, flat_args, flat_names)
            if not os.path.exists(os.path.join(OUT_DIR, f'{name}.pkl'))
        ]

    with mp.Pool(max(processes, len(flat_args))) as p:
        p.starmap(exec_exp, flat_args)


# yapf: disable
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--processes',
                        type=int,
                        default=20,
                        help="size of process pool used to run experiments.")
    args = parser.parse_args()

    """(Experiment) SupLORD vs LORDFDX comparison."""
    """(Data) Constant data model"""
    sup_v_fdx_names, sup_v_fdx_algs = suplord_v_lordfdx(delta=0.05,
                                                        bound=0.15,
                                                        r_list=[30])
    run_exp('results/suplord_v_lordfdx_constant', sup_v_fdx_algs,
            signal_comp_dataset(2, 3), sup_v_fdx_names, args.processes)

    def_cmap = plt.get_cmap('Set1')

    fdx_style = styles.color_palette_dicts(def_cmap, sup_v_fdx_names)

    def display_name_fn(name):
        if re.fullmatch(r'SupLORD.*', name):
            return 'SupLORD'
        else:
            return name

    gaussian_analysis('suplord_v_lordfdx_constant',
                      fdx_style,
                      lambda result: result.method_name,
                      sub_path="all",
                      display_name_fn=display_name_fn)

    """(Data) HMM data model"""
    run_exp('results/suplord_v_lordfdx_hmm', sup_v_fdx_algs, hmm_dataset(2, 3),
            sup_v_fdx_names, args.processes)
    hmm_analysis('suplord_v_lordfdx_hmm',
                 fdx_style,
                 lambda result: result.method_name,
                 display_name_fn=display_name_fn)

    """(Experiment) Comparison between different r^* values for SupLORD vs. LORDFDX"""
    """(Data) Constant data model"""

    def display_name_fn(name):
        res = re.fullmatch(r'SupLORD (.*)', name)
        if res is not None:
            return res.group(1)
        else:
            return name

    sup_v_fdx_names, sup_v_fdx_algs = suplord_v_lordfdx(delta=0.05,
                                                        bound=0.15,
                                                        auto=True,
                                                        r_list=[10, 20, 30])
    run_exp('results/suplord_v_lordfdx_intro', sup_v_fdx_algs,
            signal_comp_dataset(3, 3), sup_v_fdx_names, args.processes)

    gaussian_analysis(
        'suplord_v_lordfdx_intro',
        styles.suplord_final_style,
        lambda result: result.method_name,
        filter_fn=lambda result: re.fullmatch(r'(SupLORD.*)|(LORDFDX)', result.
                                              method_name) is not None,
        display_name_fn=display_name_fn,
        sub_path="fdx_only")

    """(Experiment) Dynamic vs static scheduling"""
    """(Data) Constant data model"""
    dyn_v_static_names, dyn_v_static_algs = dynamic_comparison(
        delta=0.05,
        bound=0.15,
        threshold=30,
        decay_lens=[200],
        decay_coefs=[0.01])

    def alt_cmap(i):
        if i == 0:
            return def_cmap(i)
        elif i == 1:
            return 'tab:purple'
        else:
            return 'tab:cyan'

    dyn_v_static_style = styles.color_palette_dicts(alt_cmap,
                                                    dyn_v_static_names)

    def display_name_fn(name):
        if re.fullmatch(r'Dynamic.*', name):
            return 'Dynamic'
        elif name[-1] == "1":
            return 'STEADY'
        else:
            return 'AGGRESSIVE'

    def filter_fn(result):
        res = result.alg_kwargs['gamma_mode'] != 'dynamic_uni'
        return res

    run_exp('results/dynamic_v_static_constant', dyn_v_static_algs,
            signal_comp_dataset(2, 3), dyn_v_static_names)
    gaussian_analysis('dynamic_v_static_constant',
                      dyn_v_static_style,
                      lambda result: result.method_name,
                      filter_fn=filter_fn,
                      display_name_fn=display_name_fn)

    """(Data) HMM data model"""
    run_exp('results/dynamic_v_static_hmm', dyn_v_static_algs,
            hmm_dataset(2, 3), dyn_v_static_names, args.processes)
    hmm_analysis('dynamic_v_static_hmm',
                 dyn_v_static_style,
                 lambda result: result.method_name,
                 filter_fn=filter_fn,
                 display_name_fn=display_name_fn)
