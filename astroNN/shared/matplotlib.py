import matplotlib.pyplot as plt


def pylab_style(paper=False):
    # list of avaliable parameters
    # https://matplotlib.org/stable/tutorials/introductory/customizing.html#the-default-matplotlibrc-file
    params = {
        "boxplot.boxprops.linewidth": 10.0,
        "figure.figsize": [8, 5],
        "figure.titlesize": 20,
        "axes.labelsize": 15,
        "axes.labelweight": "medium",
        "axes.titleweight": "medium",
        "legend.fontsize": 15,
        "legend.markerscale": 3,
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
        # set back the original dpi, see https://github.com/ipython/matplotlib-inline/pull/14
        "figure.dpi": 72,
    }
    if paper:  # additional parameters for paper style
        params["text.usetex"] = True
        params["font.family"] = "serif"
        # params["font.serif"] = "Computer Modern Roman"
        params["axes.labelsize"] = 20
        params["xtick.labelsize"] = 20
        params["ytick.labelsize"] = 20
        params["axes.titlesize"] = 20
        params["savefig.dpi"] = 300  # printing quality

    plt.rcParams.update(params)
