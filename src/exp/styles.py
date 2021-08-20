import re

import numpy as np
import matplotlib.pyplot as plt
from utils import rotate_list


def unzip_dict(tuple_dict):
    return [{key: value[i]
             for key, value in tuple_dict.items()} for i in range(3)]


def color_palette_style(cmap, length):
    colors = [cmap(i) for i in range(length)]
    all_markers = list(".,ov^<>12348spP*hH+xXDd|_") + list(range(12))
    markers = all_markers[:length]
    all_linestyles = [
        (1, 1), (2, 2), (4, 4)
    ] + [tuple([1, 2] * (i + 1) + [1, 4]) for i in range(max(0, length - 3))]
    linestyles = all_linestyles[:length]
    return colors, markers, linestyles


def color_palette_dicts(cmap, names):
    return tuple([{name: val
                   for name, val in zip(names, values)}
                  for values in color_palette_style(cmap, len(names))])


def_cmap = plt.get_cmap('Set1')
suplord_col = (0.8941176470588236, 0.10196078431372549, 0.10980392156862745,
               1.0)
lordfdx_col = (0.21568627450980393, 0.49411764705882355, 0.7215686274509804,
               1.0)
col10, col20 = np.linspace((0.15, 0, 0.25, 1), suplord_col, 3,
                           endpoint=False)[1:]
suplord_final_style = unzip_dict({
    'SupLORD $r^*=10$': (col10, 'v', (4, 4)),
    'SupLORD $r^*=20$': (col20, 'o', (1, 2, 1, 4)),
    'SupLORD $r^*=30$': (suplord_col, '.', (1, 1)),
    'LORDFDX': (lordfdx_col, ',', (2, 2)),
})

comp_algs_style = unzip_dict({
    'LORD': ((102 / 255, 0, 1), 'o', (1, 1)),
    'LORDFDX': ((102 / 255, 0.5, 1), '.', (4, 4)),
    'LORD++': ((204 / 255, 0, 1), 'o', (1, 1)),
    'LORD++FDX': ((204 / 255, 0.5, 1), 'o', (1, 1)),
    'Bonferroni': ((0, 0, 0), 'x', (2, 4)),
    'SupLORD': ((0, 1, 0), '*', (6, 4))
})
suplord_inits_style = unzip_dict({
    'flat': ((0, 1, 0), 'o', (1, 1)),
    'decay': ((0, 0.9, 0.1), 'x', (2, 4)),
    'uniform': ((0, 0.8, 0.2), '.', (4, 4)),
})
suplord_aggressive_style = unzip_dict({
    'LORD': ((0, 1, 0), 'o', (1, 1)),
    'SupLORD standard': ((0, 0.9, 0.1), 'x', (2, 4)),
    'SupLORD accelerated': ((0, 0.8, 0.2), '.', (4, 4)),
})

suplord_agg_comp_style = unzip_dict({
    'SupLORD param=1,len=50': ((0, 1, 0), 'x', (2, 4)),
    'SupLORD param=2,len=50': ((0, 0.9, 0.1), 'x', (2, 4)),
    'SupLORD param=4,len=50': ((0, 0.8, 0.2), 'x', (2, 4)),
    'SupLORD param=1,len=100': ((0, 0.7, 0.3), 'x', (2, 4)),
    'SupLORD param=2,len=100': ((0, 0.6, 0.4), 'x', (2, 4)),
    'SupLORD param=4,len=100': ((0, 0.5, 0.5), 'x', (2, 4)),
})

suplord_agg_comp_style_2 = unzip_dict({
    'SupLORD param=4,len=200': ((0, 1, 0), 'x', (2, 4)),
    'SupLORD param=16,len=200': ((0, 0.9, 0.1), 'x', (2, 4)),
    'SupLORD param=64,len=200': ((0, 0.8, 0.2), 'x', (2, 4)),
    'SupLORD param=4,len=100': ((0, 0.7, 0.3), 'x', (2, 4)),
    'SupLORD param=16,len=100': ((0, 0.6, 0.4), 'x', (2, 4)),
    'SupLORD param=64,len=100': ((0, 0.5, 0.5), 'x', (2, 4)),
})


def gen_styles(start_color, end_color, key_list):
    colors = np.linspace(start_color, end_color, len(key_list))
    return unzip_dict(
        {key: (color, '.', (2, 4))
         for color, key in zip(colors, key_list)})


suplord_agg_comp_style_3 = unzip_dict({
    'SupLORD param=0.001,len=200': ((0, 1, 0), 'x', (2, 4)),
    'SupLORD param=0.01,len=200': ((0, 0.9, 0.1), 'x', (2, 4)),
    'SupLORD param=0.1,len=200': ((0, 0.8, 0.2), 'x', (2, 4)),
    'SupLORD param=0.001,len=100': ((0, 0.7, 0.3), 'x', (2, 4)),
    'SupLORD param=0.01,len=100': ((0, 0.6, 0.4), 'x', (2, 4)),
    'SupLORD param=0.1,len=100': ((0, 0.5, 0.5), 'x', (2, 4)),
})
