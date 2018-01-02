![AstroNN Logo](astroNN_icon_withname.png)

## Getting Started and Prerequisites

astroNN is a python package to do various kind of neural networks for astronomers. It provides bayesian neural network implementation to do neural network with incomplete labels
and uncertainty analysis. Incomplete data means you have some target labels, but you only has a subset of them for some data. astroNN will look for -9999. in training data
and not backpropagate those particular labels for those particular datas. For uncertainty analysis, please see demostration section.

As of now, this is a python package developing for an undergraduate research project on deep learning application in 
stellar and galactic astronomy using SDSS APOFEE DR14 and Gaia DR1.

#### [New Draft of astroNN Documentation](https://henrysky.github.io/astroNN/)

#### [Quick Start guide](https://henrysky.github.io/astroNN/quick_start.html)

#### [v0.99 Alpha Release](https://github.com/henrysky/astroNN/releases/tag/v0.99)

## Authors

* **Henry W.H. Leung** - *Initial work and developer* - [henrysky](https://github.com/henrysky)\
Contact Henry: [henrysky.leung@mail.utoronto.ca](mailto:henrysky.leung@mail.utoronto.ca)

* **Jo Bovy** - [jobovy](https://github.com/jobovy)\
*Supervisor of **Henry W.H. Leung** on this undergraduate project*\
*Original developer of `xmatch()` of `astroNN.datasets.xmatch.xmatch()`* from his [gaia_tools](https://github.com/jobovy/gaia_tools)

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
