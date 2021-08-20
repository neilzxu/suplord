from itertools import product

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from analysis.utils import *
import plot

font = {'size': 40}
matplotlib.rc('font', **font)
matplotlib.rc('lines', linewidth=10)


def make_datapoints(results, result_name):

    datapoints = [{
        'alg':
        result_name(result),
        'pi':
        result.data_kwargs['transition_prob'],
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
                                 threshold=10)
    } for result in results]
    return datapoints


def hmm_analysis(exp_name,
                 style,
                 result_name=lambda result: result.alg,
                 ncols=2,
                 filter_fn=lambda result: True,
                 display_name_fn=lambda x: x,
                 sub_path=None):
    results, _, exp_figure_path = load_results(exp_name)
    results = [result for result in results if filter_fn(result)]
    if sub_path is not None:
        exp_figure_path = os.path.join(exp_figure_path, sub_path)
    if not os.path.exists(exp_figure_path):
        os.makedirs(exp_figure_path)

    color_map, marker_map, dash_map = style

    datapoints = make_datapoints(results, result_name)
    signals = sorted(list({point['signal'] for point in datapoints}))
    df = pd.DataFrame.from_records(datapoints).sort_values(['alg', 'pi'])

    # Plot metric for each method and data setup
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
                                       x_name='$\\xi$',
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
