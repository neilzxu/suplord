from itertools import product
import os
import pickle

import matplotlib.pyplot as plt
import multiprocess as mp
import numpy as np
import pandas as pd

from alg import SupLORD, LORDFDX, Bonferroni, get_alg
from exp.algsets import suplord_v_lordfdx
from exp import styles
from exp.utils import exec_exp


def load_IMPC_data(path):
    df = pd.read_csv(
        path,
        usecols=['Latest.Mutant.Assay.Date', 'Genotype.Contribution'],
        parse_dates=['Latest.Mutant.Assay.Date'])
    return df


if __name__ == '__main__':
    df = load_IMPC_data('real_data/IMPC_ProcessedData_Continuous.csv')
    sorted_df = df.sort_values(['Latest.Mutant.Assay.Date'])
    save_dir = 'results/real_data'

    alpha = 0.05
    delta = 0.05
    gamma = 0.15

    thresholds = [100]
    gamma_modes = ['static', 'dynamic']

    results = []
    pvec = sorted_df.loc[:, 'Genotype.Contribution'].to_numpy()

    def run_exp(alg_name, alg_kwargs):
        algo = get_alg(alg_name)(**alg_kwargs)
        rejset = algo.run_fdr(pvec)
        return algo, rejset

    suplord_combos = list(product(gamma_modes, thresholds))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(f'{save_dir}/record.pkl'):
        suplord_algs = [('SupLORD', {
            'delta': delta,
            'bound': gamma,
            'a': 1,
            'threshold': threshold,
            'init_param': 'uniform',
            'gamma_mode': gamma_mode,
            'gamma_series': 'gaussian',
            'a_mode': 'auto',
            'gamma_decay_coef': 0.001,
            'gamma_decay_len': 10000,
            'init_mode': 'uniform'
        }) for gamma_mode, threshold in suplord_combos]
        suplord_3_algs = [('SupLORD', {
            'delta': delta,
            'bound': gamma,
            'a': 1,
            'threshold': threshold,
            'init_param': 'uniform',
            'gamma_mode': 'static',
            'gamma_series': 'gaussian',
            'a_mode': 'auto',
            'gamma_decay_coef': 0.001,
            'gamma_decay_len': 10000,
            'init_mode': 'uniform',
            'alpha_strategy': 3
        }) for threshold in [100]]
        lord_alg = ('LORD', {
            'delta': alpha,
            'bound': None,
            'startfac': 0.1,
            'gamma_series': 'gaussian',
            'is_pp': True,
            'alpha_strategy': 2
        })
        lordfdx_alg = ('LORD', {
            'delta': delta,
            'bound': gamma,
            'startfac': 0.1,
            'gamma_series': 'gaussian',
            'is_pp': True,
            'alpha_strategy': 2
        })

        bonf_alg = ('Bonferroni', {'alpha': alpha, 'gamma_series': 'gaussian'})
        all_algs = suplord_algs + suplord_3_algs + [
            lord_alg, lordfdx_alg, bonf_alg
        ]
        with mp.Pool(10) as p:
            results = p.starmap(run_exp, all_algs)

        with open(f'{save_dir}/record.pkl', 'wb') as in_f:
            pickle.dump(results, in_f)
    else:
        with open(f'{save_dir}/record.pkl', 'rb') as out_f:
            results = pickle.load(out_f)

    # Generating consistent style
    sup_v_fdx_names, sup_v_fdx_algs = suplord_v_lordfdx(delta=0.05,
                                                        bound=0.15,
                                                        r_list=[30])
    def_cmap = plt.get_cmap('Set1')
    colors, markers, linestyles = styles.color_palette_dicts(
        def_cmap, sup_v_fdx_names + [
            f'SupLORD 2 {gamma_mode} $r^*={thresh}$'
            for gamma_mode, thresh in suplord_combos
        ] + [f'SupLORD 3 static $r^*={thresh}$' for thresh in thresholds])

    fig = plt.figure(figsize=(8, 6))
    ax = plt.gca()
    print(colors.keys())
    print(pvec.shape)
    for alg, result in results:
        if isinstance(alg, SupLORD):
            if alg.alpha_strategy == 2 and alg.gamma_scheduler.mode == 'static':
                method_name = 'SupLORD STEADY'
            elif alg.alpha_strategy == 2 and alg.gamma_scheduler.mode == 'dynamic':
                method_name = 'SupLORD DYNAMIC'
            else:
                method_name = 'SupLORD AGGRESSIVE'
            name = f'{method_name} $r^*={alg.threshold}$'
        elif isinstance(alg, LORDFDX) and alg.bound is None:
            name = 'LORD'
        elif isinstance(alg, LORDFDX):
            name = 'LORDFDX'
        else:
            name = 'Bonferroni'
        print(name, np.sum(result))
        ax.plot(np.arange(len(result)),
                np.cumsum(result),
                label=name,
                color=colors[name])
    fig.legend(bbox_to_anchor=(0, 0.8, 1, 0.2), ncol=3)
    fig.tight_layout(rect=(0, 0, 1, 0.8))
    fig.savefig('results/real_data/method_comp.png')
