import itertools

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams.update({'font.size': 22})


def construct_stack_grid(row_height, col_width, **kwargs):
    fig = plt.figure(figsize=(2 * col_width, 2 * row_height), **kwargs)
    gs = fig.add_gridspec(2, 2)
    fdr_plot = fig.add_subplot(gs[0, 0])
    fdp_plot = fig.add_subplot(gs[1, 0])
    power_plot = fig.add_subplot(gs[:, 1])
    return fig, [fdr_plot, fdp_plot, power_plot]


def make_xy_plot(ax, xs, ys, labels, marker_map, dash_map, color_map):
    for x, y, label in zip(xs, ys, labels):
        ax.plot(x,
                y,
                marker=marker_map[label],
                dashes=dash_map[label],
                color=color_map[label],
                label=label)


def make_aesthetic_interval(val_min, val_max, val_ct):
    interval = (val_max - val_min) / val_ct

    ten_power = np.floor(np.log10(interval))
    scalar = max(np.floor(interval / np.power(10, ten_power)), 1 / 10)
    neat_interval = scalar * np.power(10, ten_power)
    neat_min, neat_max = np.floor(
        val_min / neat_interval) * neat_interval, np.ceil(
            val_max / neat_interval) * neat_interval

    return neat_min, neat_max, neat_interval


def plot_on_ax(df, alg_col, x_col, stat_cols, ax_list, marker_map, dash_map,
               color_map):
    alg_keys = df[alg_col].unique().tolist()

    def get_data(alg, name):
        alg_df = df[(df[alg_col] == alg)]
        pis = alg_df[x_col].values
        stats = alg_df[name].values
        return alg, pis, stats

    for stat_col, ax in zip(stat_cols, ax_list):
        algs, pis_list, stats_list = zip(
            *[get_data(alg, stat_col) for alg in alg_keys])

        make_xy_plot(ax, pis_list, stats_list, algs, marker_map, dash_map,
                     color_map)
        ax.set_ylabel(stat_col)
        ax.set_xlabel(x_col)

        # get y axis range so we can evenly distributed ticks
        min_y, max_y = ax.get_ylim()
        neat_min, neat_max, neat_interval = make_aesthetic_interval(
            min_y, max_y, 10)
        ax.set_yticks(
            np.arange(neat_min, neat_max + neat_interval, neat_interval))
        ax.set_xticks(np.arange(0, 1.1, 0.1))
        ax.grid()
    return alg_keys


def plot_many_over_x(df, x_col, alg_col, stat_cols, height, ncols, marker_map,
                     dash_map, color_map):

    alg_keys = df[alg_col].unique().tolist()

    def get_data(alg, name):
        alg_df = df[(df[alg_col] == alg)]
        pis = alg_df[x_col].values
        stats = alg_df[name].values
        return alg, pis, stats

    nrows = int(np.ceil(len(stat_cols) / ncols))
    fig, axes = plt.subplots(figsize=(ncols * height, nrows * height),
                             nrows=nrows,
                             ncols=ncols,
                             sharex=True,
                             sharey=False,
                             squeeze=False)

    for i, stat_name in enumerate(stat_cols):
        algs, pis_list, stats_list = zip(
            *[get_data(alg, stat_name) for alg in alg_keys])

        grid_x, grid_y = i // ncols, i % ncols
        ax = axes[grid_x, grid_y]
        make_xy_plot(ax, pis_list, stats_list, algs, marker_map, dash_map,
                     color_map)
        ax.set_ylabel(stat_name)
        ax.set_xlabel(x_col)

        # get y axis range so we can evenly distributed ticks
        min_y, max_y = ax.get_ylim()
        neat_min, neat_max, neat_interval = make_aesthetic_interval(
            min_y, max_y, 10)
        ax.set_yticks(
            np.arange(neat_min, neat_max + neat_interval, neat_interval))
        ax.set_xticks(np.arange(0, 1.1, 0.1))
        ax.grid()
        if i == 0:
            fig.legend(loc='upper center',
                       bbox_to_anchor=(0, 0.8, 1, 0.2),
                       ncol=2)
    fig.tight_layout()
    return fig, axes


def plot_alphas(vals, ylabel, xs=None, xlabel=None):
    fig = plt.figure(figsize=(10, 10))
    ax = plt.gca()
    if xs is None:
        for name, values in vals:
            ax.plot(np.arange(1, len(values) + 1), values, label=name)
    else:
        for (name, values), xs in zip(vals, xs):
            ax.plot(xs, values, label=name)

    ax.legend()
    if xlabel is None:
        ax.set_xlabel("Hypotheses")
    else:
        ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    neat_min, neat_max, neat_interval = make_aesthetic_interval(
        *(ax.get_ylim()), 10)
    ax.set_yticks(np.arange(neat_min, neat_max + neat_interval, neat_interval))
    ax.grid()
    fig.tight_layout()
    return fig


def make_paths(ax, groups, aggregate_fn, marker_map, dash_map, color_map,
               conf_pct):
    for label, data_arr in groups:
        data_arr = np.squeeze(data_arr)
        trials, hypotheses = data_arr.shape
        color = color_map[label]
        if conf_pct is None:
            pale_color = tuple(val + 0.5 * (1 - val) for val in color)
            for i in range(trials):
                ax.plot(np.arange(0, hypotheses),
                        data_arr[i, :],
                        alpha=0.1,
                        marker=" ",
                        linewidth=0.5,
                        dashes=(None, None),
                        color=pale_color)
        elif conf_pct > 0.0:
            # Sort trials by metric of last hypothesis
            last_hypotheses = np.squeeze(data_arr[:, -1])
            trial_order = np.argsort(last_hypotheses)
            sorted_arr = data_arr[trial_order]
            trials_in_band = int(np.ceil(conf_pct * trials))
            pct_arr = sorted_arr[:trials_in_band, :]

            ax.fill_between(np.arange(0, hypotheses),
                            np.min(pct_arr, axis=0),
                            np.max(pct_arr, axis=0),
                            color=color,
                            alpha=0.15)

        agg_vals = aggregate_fn(data_arr)
        ax.plot(np.arange(0, hypotheses),
                agg_vals,
                alpha=1.0,
                dashes=dash_map[label],
                marker=" ",
                color=color,
                label=label,
                linewidth=4)


def plot_paths(groups,
               aggregate_fn,
               height,
               marker_map,
               dash_map,
               color_map,
               conf_pct=None):
    fig = plt.figure(figsize=(height, height))
    ax = plt.gca()
    make_paths(ax, groups, aggregate_fn, marker_map, dash_map, color_map,
               conf_pct)
    return fig, ax


def plot_subplots(df,
                  x,
                  y,
                  series,
                  grid,
                  ncols,
                  length,
                  color_map=None,
                  marker_map=None,
                  dash_map=None,
                  x_name=None,
                  y_name=None,
                  x_len_const=0,
                  y_len_const=2,
                  grid_name=lambda name, val: f'{name}={val}',
                  **kwargs):

    if x_name is None:
        x_name = x
    if y_name is None:
        y_name = y

    unique_grids = np.unique(df[grid].values)
    nrows = int(np.ceil(len(unique_grids) / ncols))

    fig, axes = plt.subplots(figsize=(ncols * length + x_len_const,
                                      nrows * length + y_len_const),
                             nrows=nrows,
                             ncols=ncols,
                             **kwargs)
    for i, grid_val in enumerate(unique_grids):

        row_idx, col_idx = (i // ncols, i % ncols)
        ax = axes[row_idx, col_idx]

        data = [
            (series_val, infos[x].values, infos[y].values)
            for series_val, infos in df[(df[grid] == grid_val)].groupby(series)
        ]
        for series_val, xs, ys in data:
            ax.plot(xs,
                    ys,
                    label=f'{series_val}',
                    color=color_map[series_val],
                    dashes=dash_map[series_val],
                    marker=marker_map[series_val])
        ax.grid()
        ax.set_title(grid_name(grid, grid_val))
        ax.set_xlabel(x_name)
        ax.set_ylabel(y_name)
    return fig, axes
