![AstroNN Logo](astroNN_icon_withname.png)

## Getting Started

A python package to do neural network with APOGEE stellar spectra DR13/DR14 and Gaia DR1 with Tensorflow
#### !!!Still in Development!!!!

### Prerequisites


***This python package must be using with Tensorflow 1.4.0 (As of 18 Oct 2017, Tensorflow 1.4.0 still in beta)

Please go to one of the following link to download a wheel locally and install it\
[Tensorflow 1.4.0](https://pypi.python.org/pypi/tensorflow/1.4.0rc0)\
[Tensorflow-gpu 1.4.0](https://pypi.python.org/pypi/tensorflow-gpu/1.4.0rc0)

***This package has no Keras dependency, it have been migrated to Tensorflow. You dont need to install Keras anymore

```
Python 3.6 or above (Anaconda 5.0.0 64bit is recommended)
Tensorflow 1.4.0 or above (***There is no GPU version of tensorflow for MacOS user)
Tensorflow-gpu 1.4.0 or above is recommended
graphviz and pydot_ng are required to plot the model architecture
```

For instruction on how to install Tensorflow, please refer to their official website
[->Installing TensorFlow](https://www.tensorflow.org/install/)

### Installing

Just run the following commmand to install after you open the command line windows in the project folder

```
python setup.py install
```

## Tutorial

Please refer to tutorial section [Tutorial](tutorial)

## Authors

* **Henry Leung** - *Initial work and developer* - [henrysky](https://github.com/henrysky)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## Acknowledgments

* **Jo Bovy** - *Original developer of `xmatch` of `astroNN.datasets.xmatch.xmatch()`* - [jobovy](https://github.com/jobovy)