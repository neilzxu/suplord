from collections import defaultdict
from itertools import product
import os
import pickle
from pprint import pprint
import re

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from analysis.utils import *
import metrics
import plot

font = {'size': 40}
matplotlib.rc('font', **font)
label = {'labelsize': 40}
matplotlib.rc('xtick', **label)
matplotlib.rc('ytick', **label)
matplotlib.rc('legend', fontsize=40)
matplotlib.rc('lines', linewidth=15)


def make_datapoints(results, result_name):

    datapoints = [{
        'alg':
        result_name(result),
        'pi':
        result.data_kwargs['non_null_p'],
        'signal':
        result.data_kwargs['non_null_mean'],
        'FDR':
        metrics.fdr(result.alternates, result.rejsets),
        'Power':
        metrics.power(result.alternates, result.rejsets),
        'FDX':
        metrics.fdx(result.alternates, result.rejsets, gamma=0.15,
                    threshold=0),
        'SupFDP':
        metrics.expected_sup_fdp(result.alternates,
                                 result.rejsets,
                                 threshold=20)
    } for result in results]
    return datapoints


def gaussian_analysis(exp_name,
                      style,
                      result_name=lambda result: result.alg,
                      ncols=2,
                      sub_path=None,
                      filter_fn=None,
                      display_name_fn=lambda x: x):
    color_map, marker_map, dash_map = style
    exp_path = os.path.join('results', exp_name)
    exp_figure_path = os.path.join('figures', exp_name)

    # Make directory (or subdirectories) if necessary
    if sub_path is not None:
        exp_figure_path = os.path.join(exp_figure_path, sub_path)
    if not os.path.exists(exp_figure_path):
        os.makedirs(exp_figure_path)

    results = [
        pickle.load(open(os.path.join(exp_path, filename), 'rb'))
        for filename in os.listdir(exp_path)
        if re.fullmatch(r'.*\.pkl', filename)
    ]
    if filter_fn is not None:
        results = [result for result in results if filter_fn(result)]

    datapoints = make_datapoints(results, result_name)
    df = pd.DataFrame.from_records(datapoints).sort_values(['alg', 'pi'])

    signals = sorted(list({point['signal'] for point in datapoints}))
    nrows = int(np.ceil(len(signals) / ncols))

    # Plot a metric for each method and data setup
    stat_cols = ['FDR', 'SupFDP', 'Power']
    for metric in stat_cols:
        fig, axes = plot.plot_subplots(df,
                                       x='pi',
                                       y=metric,
                                       series='alg',
                                       grid='signal',
                                       ncols=ncols,
                                       length=10,
                                       sharex=True,
                                       sharey=True,
                                       squeeze=False,
                                       color_map=color_map,
                                       dash_map=dash_map,
                                       marker_map=marker_map,
                                       x_name='$\pi$',
                                       x_len_const=4,
                                       grid_name=lambda x, y: f'$\mu={y}$')
        ax = axes[0, 0]
        handles, labels = axes[0, 0].get_legend_handles_labels()
        new_labels = [display_name_fn(label) for label in labels]
        leg = fig.legend(handles=handles,
                         labels=new_labels,
                         bbox_to_anchor=(0, 0.9, 1, 0.1),
                         loc='upper center',
                         ncol=4)
        if metric == 'Power':
            ax.set_ylim(-0.1, 1)
            ax.set_yticks(np.arange(0, 1.1, 0.2))
        else:
            ax.set_ylim(-0.01, 0.1)
            ax.set_yticks(np.arange(0, 0.11, 0.02))
        ax.set_xticks(np.arange(0.1, 1, 0.2))
        if 'Steady' in new_labels:
            o_property = leg.get_texts()[0]._fontproperties
            for label, t in zip(new_labels, leg.get_texts()):
                if label in ['Steady', 'Aggressive']:
                    t._fontproperties = o_property.copy()
                    t.set_variant('small-caps')

        fig.tight_layout(rect=(0, 0, 1, 0.9), pad=1, w_pad=0, h_pad=0)
        fig.savefig(f'{exp_figure_path}/{metric}.png')
        plt.close(fig)

    def get_pi(result):
        return result.data_kwargs['non_null_p']

    def hyp_filter(result):
        return result.data_kwargs['non_null_mean'] in [
            3
        ] and result.data_kwargs['non_null_p'] == 0.2 or result.data_kwargs[
            'non_null_p'] == 0.05

    hyp_results = [result for result in results if hyp_filter(result)]

    # Draw hypothesis wise plots for different metrics
    for name, data_fn in tqdm(plot_settings, desc='hypotheses-wise plot'):
        if name in ['Alpha', 'Wealth']:
            fig, axes = plt.subplots(figsize=(16, 10),
                                     nrows=1,
                                     ncols=1,
                                     sharex=True,
                                     sharey=True,
                                     squeeze=False)
            draw_ax_hypotheses_plot(
                axes.flat,
                hyp_results,
                name,
                name_fn=result_name,
                agg_fn=lambda x: np.mean(x, axis=0),
                data_fn=data_fn,
                filter_fn=lambda result: (result.data_kwargs['non_null_mean'],
                                          result.data_kwargs['non_null_p']),
                filter_name_fn=lambda x: "$\mu$=" + str(x[0]),
                conf_pct=0.0,
                color_map=color_map,
                dash_map={key: (None, None)
                          for key in dash_map},
                marker_map={key: " "
                            for key in marker_map})

            ax = axes[0, 0]
            ax.set_title("")
            if name == 'Alpha':
                ax.set_ylabel('$\\alpha$')
            handles, labels = axes[0, 0].get_legend_handles_labels()
            succinct = {
                label: handle
                for label, handle in zip(labels, handles)
            }
            labels, handles = zip(*list(succinct.items()))
            new_labels = [display_name_fn(label) for label in labels]
            leg = fig.legend(handles,
                             new_labels,
                             loc='upper center',
                             bbox_to_anchor=(0, 0.85, 1.0, 0.15),
                             ncol=3)

            if 'Steady' in new_labels:
                o_property = leg.get_texts()[0]._fontproperties
                for label, t in zip(new_labels, leg.get_texts()):
                    if label in ['Steady', 'Aggressive']:
                        t._fontproperties = o_property.copy()
                        t.set_variant('small-caps')
            fig.tight_layout(rect=(0, 0, 1, 0.85),
                             pad=0.2,
                             w_pad=0.5,
                             h_pad=0.5)

            fig.savefig(f'{exp_figure_path}/hypotheses_{name}.png')
            plt.close(fig)

    # Draw plot in paper introduction
    col_width, row_height = (10, 5)
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 8), squeeze=False)
    fdp_ax = axes[0, 0]
    power_ax = axes[0, 1]

    signal_df = df[df['signal'] == 3]
    alg_keys = plot.plot_on_ax(df=signal_df,
                               x_col='pi',
                               alg_col='alg',
                               stat_cols=['Power'],
                               ax_list=[power_ax],
                               marker_map=marker_map,
                               dash_map=dash_map,
                               color_map=color_map)
    handles, labels = power_ax.get_legend_handles_labels()
    new_labels = [display_name_fn(label) for label in labels]
    leg = fig.legend(handles=handles,
                     labels=new_labels,
                     bbox_to_anchor=(0, 0.9, 1, 0.1),
                     loc='upper center',
                     ncol=4)

    def new_filter(result):
        return result.data_kwargs['non_null_mean'] == 3 and int(
            result.data_kwargs['non_null_p'] * 10
        ) in [
            4
        ] and result.alg == 'SupLORD' and result.alg_kwargs['threshold'] == 30

    new_results = [result for result in results if new_filter(result)]
    draw_ax_hypotheses_plot([fdp_ax],
                            new_results,
                            'FDP',
                            name_fn=result_name,
                            agg_fn=lambda x: np.mean(x, axis=0),
                            data_fn=fdp_fn,
                            filter_fn=lambda result:
                            (result.data_kwargs['non_null_mean'], result.
                             data_kwargs['non_null_p']),
                            filter_name_fn=lambda x: "$\mu$=" + str(x[0]),
                            conf_pct=0.95,
                            color_map=color_map,
                            dash_map=dash_map,
                            marker_map=marker_map)

    ymax = fdp_ax.get_ylim()[1]
    new_ymax = (ymax // 0.15 + 1) * 0.15
    fdp_ax.set_yticks(np.arange(0.00, new_ymax, 0.15))
    fdp_ax.set_ylim(-0.01, new_ymax)
    fdp_ax.set_title('$\pi=0.3$')

    power_ax.set_xlabel('$\pi$')
    power_ax.set_xticks(np.arange(0.1, 1, 0.2))
    power_ax.set_ylim(-0.1, 1)
    power_ax.set_yticks(np.arange(0, 1.1, 0.2))
    fig.tight_layout(rect=(0, 0, 1, 0.9), pad=0.5, w_pad=0.5, h_pad=0.5)
    fig.savefig(f'{exp_figure_path}/intro_fig.png')

    all_disp_names = {
        display_name_fn(result_name(result)): result
        for result in hyp_results
    }
    if 'Dynamic' in all_disp_names:
        # Plot difference between alphas of dynamic vs baseline schedules for a single run
        dynamic_result = all_disp_names['Dynamic']
        for name, result in [
                item for item in all_disp_names.items() if item[0] != 'Dynamic'
        ]:
            fig = plt.figure(figsize=(20, 12))
            ax = plt.gca()
            ys = np.array(dynamic_result.instances[0].alpha) - np.array(
                result.instances[0].alpha)
            xs = np.arange(ys.shape[0])
            higher_alpha_mask = ys > 0
            ax.bar(xs[higher_alpha_mask],
                   ys[higher_alpha_mask],
                   color='green',
                   edgecolor='none')
            ax.bar(xs[~higher_alpha_mask],
                   ys[~higher_alpha_mask],
                   color='red',
                   edgecolor='none')
            ax.set_ylim(-0.03, 0.03)
            ax.set_xlabel('Hypotheses')
            ax.set_ylabel('Difference in $\\alpha$')
            fig.savefig(f'{exp_figure_path}/alpha_comp_{name}.png')
            plt.close(fig)
        # Plot alphas of a single run of each schedule
        fig = plt.figure(figsize=(20, 12))
        ax = plt.gca()
        for name, result in all_disp_names.items():
            ax.plot(np.arange(len(result.instances[0].alpha)),
                    result.instances[0].alpha,
                    color=color_map[result_name(result)],
                    s=2,
                    label=name)
        ax.set_xlabel('Hypotheses')
        ax.set_ylabel('$\\alpha$')
        fig.legend(bbox_to_anchor=(0, 0.9, 1, 0.1), loc='upper center', ncol=3)
        fig.tight_layout(rect=(0, 0, 1, 0.9), pad=0.5, w_pad=0.5, h_pad=0.5)
        fig.savefig(f'{exp_figure_path}/alpha_run.png')
        plt.close(fig)
