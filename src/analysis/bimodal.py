from itertools import product
import os
import pickle
import re

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

import metrics
import plot


def make_datapoints(results):
    datapoints = [{
        'alg':
        result.name,
        'pi':
        max(result.data_kwargs['non_null_p_1'],
            result.data_kwargs['non_null_p_2']),
        'FDR':
        metrics.fdr(result.alternates, result.rejsets),
        'Power':
        metrics.power(result.alternates, result.rejsets),
        'FDX':
        metrics.fdx(result.alternates, result.rejsets, gamma=0.15,
                    threshold=0),
        '$\mathbb{E}[\sup~\mathrm{FDP}]$':
        metrics.expected_sup_fdp(result.alternates,
                                 result.rejsets,
                                 threshold=10)
    } for result in results]
    return datapoints


def bimodal_analysis(exp_name, style):
    color_map, marker_map, dash_map = style
    exp_path = os.path.join('results', exp_name)
    exp_figure_path = os.path.join('figures', exp_name)
    if not os.path.exists(exp_figure_path):
        os.makedirs(exp_figure_path)

    results = [
        pickle.load(open(os.path.join(exp_path, filename), 'rb'))
        for filename in os.listdir(exp_path)
        if re.fullmatch(r'.*\.pkl', filename)
    ]
    datapoints = make_datapoints(results)

    df = pd.DataFrame.from_records(datapoints).sort_values(['alg', 'pi'])
    fig = plot.plot_many_over_pi(
        df,
        10,
        1,
        'alg', ['FDR', '$\mathbb{E}[\sup~\mathrm{FDP}]$', 'Power'],
        marker_map=marker_map,
        dash_map=dash_map,
        color_map=color_map)
    fig.savefig(f'{exp_figure_path}/perf_comp.png')
    plt.close(fig)

    def get_pi(result):
        return max(result.data_kwargs['non_null_p_1'],
                   result.data_kwargs['non_null_p_2'])

    def trial_wise_mean(x):
        return np.mean(x, axis=0)

    def result_name(result):
        return f'{result.name}'

    pi_range = [0.2]
    disp_results = [result for result in results if get_pi(result) in pi_range]

    plots = {
        'Alpha': (trial_wise_mean, [
            (result_name(result),
             np.stack([instance.alpha for instance in result.instances]))
            for result in disp_results
        ]),
        'Rejections': (trial_wise_mean, [(result_name(result),
                                          np.cumsum(result.rejsets, axis=1))
                                         for result in disp_results]),
        'Wealth': (trial_wise_mean, [
            (result_name(result),
             np.stack([instance.wealth_vec for instance in result.instances]))
            for result in disp_results
        ]),
        'FDP_Estimate':
        (trial_wise_mean, [(result_name(result),
                            np.stack([
                                instance.compute_FDP_set()
                                for instance in result.instances
                            ])) for result in disp_results
                           if result.alg == 'SupLORD']),
        'FDP': (trial_wise_mean, [(result_name(result),
                                   metrics.fdp(result.alternates,
                                               result.rejsets))
                                  for result in disp_results])
    }

    for (name,
         (agg_fn,
          groups)), conf_pct in tqdm(list(product(plots.items(),
                                                  [None, 0.95])),
                                     desc='Plotting hypotheses plots'):
        fig, ax = plot.plot_paths(list(sorted(groups, key=lambda x: x[0])),
                                  agg_fn,
                                  height=10,
                                  color_map=color_map,
                                  marker_map=marker_map,
                                  dash_map=dash_map,
                                  conf_pct=conf_pct)

        y_min, y_max = ax.get_ylim()
        y_min, y_max, y_step = plot.make_aesthetic_interval(y_min, y_max, 10)
        ax.set_yticks(np.arange(y_min, y_max + y_step, y_step))

        ax.set_xlabel('Hypotheses')
        ax.set_ylabel(name)
        fig.legend()
        fig.tight_layout()
        fig.savefig(f'{exp_figure_path}/{name}_hypotheses_{str(conf_pct)}.png')
        plt.close(fig)

    for pi in tqdm(pi_range, desc='Plotting worst dists for pis'):
        fig = plt.figure(figsize=(10, 10))
        ax = plt.gca()

        fdp_points = [
            np.max(instance.compute_FDP_set()
                   [instance.rej_indices[instance.threshold - 1]:])
            for result in results for instance in result.instances
            if get_pi(result) == pi and result.alg == 'SupLORD'
            and len(instance.rej_indices) >= instance.threshold
        ]
        if fdp_points:
            fdp_ponts = np.stack(fdp_points)
            ax.hist(fdp_points)
            fig.suptitle(
                f"Distribution of worst FDP for $\pi$: {pi}. (Count: {len(fdp_points)})"
            )
            fig.savefig(f'{exp_figure_path}/worst_fdp_dist_pi={pi}.png')
        plt.close(fig)
