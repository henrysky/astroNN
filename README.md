![AstroNN Logo](astroNN_icon_withname.png)

## Getting Started

A python package to do neural network with APOGEE stellar spectra DR13/DR14 and Gaia DR1 with Tensorflow
#### !!!Still in Development!!!!

### Prerequisites

This package must be using with Tensorflow 1.4.0 (As of 18 Oct 2017, Tensorflow 1.4.0 still in beta)

Please go to one of the following link to download a wheel locally and install it\
[Tensorflow 1.4.0](https://pypi.python.org/pypi/tensorflow/1.4.0rc0)\
[Tensorflow-gpu 1.4.0](https://pypi.python.org/pypi/tensorflow-gpu/1.4.0rc0)

Only Keras with Tensorflow backend is supported

Multi-gpu training is not supported, however you can run multiple models separately on multi-gpu system.

~~This package has no Keras dependency, it have been migrated to Tensorflow. You dont need to install Keras anymore~~

```
Python 3.6 or above (Anaconda 5.0.0 64bit is tested by author)
Tensorflow 1.4.0 or above (***There is no GPU version of tensorflow for MacOS user)
Tensorflow-gpu 1.4.0 or above is recommended
Keras 2.0.8 or above
CUDA 8.0 and CuDNN 6.1 for Tensorflow 1.3.0/1.4.0
graphviz and pydot_ng are required to plot the model architecture
```

For instruction on how to install Tensorflow, please refer to their official website
[->Installing TensorFlow](https://www.tensorflow.org/install/)

### Installing

Just run the following commmand to install after you open the command line windows in the project folder

```
python setup.py install
```

Or recommanded method of installation:
```
python setup.py develop
```

## Tutorial

Please refer to tutorial section [Tutorial](tutorial)

###Folder Structure
You should create a project folder, then create a folde named `apogee_dr14` and put `allStar-l31c.2.fits` and every aspcap 
 fits under it. Always run your command line or python under the project folder.

## Authors

* **Henry Leung** - *Initial work and developer* - [henrysky](https://github.com/henrysky)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## Acknowledgments

* **Jo Bovy** - [jobovy](https://github.com/jobovy)\
*Supervisor of **Henry Leung** on this undergraduate project*\
*Original developer of `xmatch()` of `astroNN.datasets.xmatch.xmatch()`*

* **S. Fabbro et al. (2017)** - [arXiv:1709.09182 ](https://arxiv.org/abs/1709.09182)\
*This project is inspired by [StarNet](https://github.com/astroai/starnet)*
