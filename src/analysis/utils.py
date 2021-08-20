import os
import pickle
import re

import matplotlib.pyplot as plt
import numpy as np

import metrics
import plot


def load_results(exp_name):
    exp_path = os.path.join('results', exp_name)
    exp_figure_path = os.path.join('figures', exp_name)
    if not os.path.exists(exp_figure_path):
        os.makedirs(exp_figure_path)

    results = [
        pickle.load(open(os.path.join(exp_path, filename), 'rb'))
        for filename in os.listdir(exp_path)
        if re.fullmatch(r'.*\.pkl', filename)
    ]
    return results, exp_path, exp_figure_path


def alpha_fn(result):
    return np.stack([instance.alpha for instance in result.instances])


def rej_fn(result):
    return np.cumsum(result.rejsets, axis=1)


def wealth_fn(result):
    return np.stack([instance.wealth_vec for instance in result.instances])


def fdp_est_fn(result):
    return np.stack(
        [instance.compute_FDP_set() for instance in result.instances])


def fdp_fn(result):
    return metrics.fdp(result.alternates, result.rejsets)


def fdx_fn_maker(beta):
    def fdx_fn(result):
        fdps = metrics.fdp(result.alternates, result.rejsets)
        trials, hypotheses = fdp.shape
        df = pd.DataFrame(data=fdps[:, ::-1].T,
                          index=np.arange(0, hypotheses),
                          columns=np.arange(0, trials))
        max_fdp = df.cummax().to_numpy().T[:, ::-1]
        return something


plot_settings = [('Alpha', alpha_fn), ('Rejections', rej_fn),
                 ('Wealth', wealth_fn), ('FDP', fdp_fn)]


def draw_ax_hypotheses_plot(axes_list, results, name, name_fn, agg_fn, data_fn,
                            filter_fn, filter_name_fn, conf_pct, color_map,
                            marker_map, dash_map):

    filter_keys = list(sorted({filter_fn(result) for result in results}))
    for i, (filter_key, ax) in enumerate(zip(filter_keys, axes_list)):
        groups = [(name_fn(result), data_fn(result)) for result in results
                  if filter_fn(result) == filter_key]
        plot.make_paths(ax,
                        list(sorted(groups, key=lambda x: x[0])),
                        agg_fn,
                        color_map=color_map,
                        marker_map=marker_map,
                        dash_map=dash_map,
                        conf_pct=conf_pct)
        ax.set_xlabel('Hypotheses')
        ax.set_ylabel(name)
        ax.set_title(filter_name_fn(filter_key))
        ax.grid()


def draw_hypotheses_plot(results, name, name_fn, agg_fn, data_fn, filter_fn,
                         filter_name_fn, ncols, conf_pct, color_map,
                         marker_map, dash_map):
    filter_keys = list(sorted({filter_fn(result) for result in results}))
    nrows = int(np.ceil(len(filter_keys) / ncols))
    fig, axes = plt.subplots(figsize=(ncols * 10, nrows * 10),
                             nrows=nrows,
                             ncols=ncols,
                             sharex=True,
                             sharey=True,
                             squeeze=False)
    draw_ax_hypotheses_plot(axes.flat, results, name, name_fn, agg_fn, data_fn,
                            filter_fn, filter_name_fn, conf_pct, color_map,
                            marker_map, dash_map)
    return fig, axes
