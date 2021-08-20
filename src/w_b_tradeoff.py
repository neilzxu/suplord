import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.lines as mlines

font = {'family': 'normal', 'size': 35}
matplotlib.rc('font', **font)


def coef_fn(a, delta, gamma):
    return a / np.log(1 / delta) * np.log(1 + np.log(1 / delta) / a) * gamma


delta = 0.05
gamma = 0.15
threshold = 20

fig = plt.figure(figsize=(22, 10))
ax = plt.gca()
xs = np.arange(0.01, 3, 0.01)
ax.plot(xs, coef_fn(xs, delta, gamma), color='red', label='Upper bound on $b$')

ax.set_xlabel('$a$')
ax.set_ylabel('$b$')
ax.yaxis.label.set_color('red')
ax.tick_params(axis='y', colors='red')

ax2 = ax.twinx()

ys = coef_fn(xs, delta, gamma) * threshold - xs
ax2.plot(xs, ys, label=f'Upper bound on $w_0$', color='blue', linestyle='--')
ax2.set_ylabel('$w_0$')

optimal_a = 0.565794
optimal_b = coef_fn(optimal_a, delta, gamma)
optimal_w = coef_fn(optimal_a, delta, gamma) * threshold - optimal_a


def newline(p1, p2, ax):
    xmin, xmax = ax.get_xbound()

    if (p2[0] == p1[0]):
        xmin = xmax = p1[0]
        ymin, ymax = ax.get_ybound()
    else:
        ymax = p1[1] + (p2[1] - p1[1]) / (p2[0] - p1[0]) * (xmax - p1[0])
        ymin = p1[1] + (p2[1] - p1[1]) / (p2[0] - p1[0]) * (xmin - p1[0])

    l = mlines.Line2D([xmin, xmax], [ymin, ymax],
                      linestyle='dotted',
                      color='green')
    ax.add_line(l)
    return l


ymin, ymax = ax.get_ybound()
vline = newline((optimal_a, ymin), (optimal_a, ymax), ax)
ax.plot([optimal_a], [optimal_b], marker='o')
ax.annotate('Canonical $b$ = {:.3f}'.format(optimal_b),
            xy=(optimal_a, optimal_b),
            xytext=(optimal_a + 0.25, optimal_b),
            arrowprops=dict(facecolor='black', shrink=0.05),
            verticalalignment='center')
ax.annotate('Canonical $a$ = {:.3f}'.format(optimal_a),
            xy=(optimal_a, 0.01),
            xytext=(optimal_a + 0.25, 0.01),
            arrowprops=dict(facecolor='black', shrink=0.05),
            verticalalignment='center')
ax2.plot([optimal_a], [optimal_w], marker='o')
ax2.annotate('Canonical $w_0$ = {:.3f}'.format(optimal_w),
             xy=(optimal_a, optimal_w),
             xytext=(optimal_a + 0.25, optimal_w),
             arrowprops=dict(facecolor='black', shrink=0.05),
             verticalalignment='center')

fig.legend(loc='upper center', bbox_to_anchor=(0, 0.8, 1, 0.2), ncol=2)
if not os.path.exists('final_figures'):
    os.makedirs('final_figures')
fig.savefig('final_figures/w0_b0.png')
