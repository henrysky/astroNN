import matplotlib.pyplot as plt


def pylab_style(paper=True):
    """
    Set the style of matplotlib to astroNN style

    Parameters
    ----------
    paper: bool
        if True, set the style to paper style using Latex
    """
    plt.style.use("astroNN.mplstyle")

    if not paper:
        params = {
            "text.usetex": False,
            "font.family": "sans-serif",
        }
        plt.rcParams.update(params)
