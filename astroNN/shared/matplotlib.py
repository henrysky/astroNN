import pylab as plt

def pylab_style():
    params = {
        'boxplot.boxprops.linewidth': 10.0,
        "figure.figsize": [8, 5],
        "axes.labelsize": 15,
        "axes.labelweight": "medium",
        "axes.titleweight": "medium",
        'legend.fontsize': 15,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "font.weight": "medium",
        "lines.linewidth": 1,
        "axes.titlesize": 15,
        "ytick.minor.visible": True,
        "xtick.minor.visible": True,
        "ytick.right": True,
        "xtick.top": True,
        "ytick.direction": "in",
        "xtick.direction": "in",
        "ytick.major.size": 5,
        "ytick.major.width": 1,
        "ytick.minor.size": 3,
        "ytick.minor.width": 0.6,
        "xtick.major.size": 5,
        "xtick.major.width": 1,
        "xtick.minor.size": 3,
        "xtick.minor.width": 0.6,
    }

    plt.rcParams.update(params)
