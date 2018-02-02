![AstroNN Logo](astroNN_icon_withname.png)

[![Documentation Status](https://readthedocs.org/projects/astronn/badge/?version=latest)](http://astronn.readthedocs.io/en/latest/?badge=latest)
[![GitHub license](https://img.shields.io/github/license/henrysky/astroNN.svg)](https://github.com/henrysky/astroNN/blob/master/LICENSE)

## Getting Started

astroNN is a python package to do various kind of neural networks for astronomers. 

Besides conventional neural network like convolutional neural net, astroNN provides bayesian neural network 
implementation to do neural network with incomplete labeled data and uncertainty analysis. 
Incomplete labeled data means you have some target labels, but you only has a subset of them for some data. astroNN 
will look for MAGIC_NUMBER (Default is -9999.) in training data and wont backpropagate those particular labels for 
those particular data. For uncertainty analysis, please see the demonstration section.

Furthermore, astroNN also included a deep learning toy dataset for astronomer - Galaxy10.

As of now, this is a python package developing for an undergraduate research project on deep learning application in 
stellar and galactic astronomy using SDSS APOGEE DR14 and Gaia DR1.

#### [astroNN Documentation](http://astronn.readthedocs.io/)

#### [Quick Start guide](http://astronn.readthedocs.io/en/latest/quick_start.html)

#### [Galaxy10 dataset](http://astronn.readthedocs.io/en/latest/galaxy10.html)

#### [Uncertainty analysis in neural net demo](https://github.com/henrysky/astroNN/tree/master/demo_tutorial/NN_uncertainty_analysis)

## Authors

* **Henry Leung** - *Initial work and developer* - [henrysky](https://github.com/henrysky)\
*Astronomy Undergrad, University of Toronto*\
*Contact Henry: henrysky.leung [at] mail.utoronto.ca*


* **Jo Bovy** - [jobovy](https://github.com/jobovy)\
*Astronomy Professor, University of Toronto*\
*Supervisor of **Henry Leung** on this undergraduate project*

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
