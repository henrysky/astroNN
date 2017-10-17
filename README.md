![AstroNN Logo](astroNN_icon_withname.png)

## Getting Started

A python package to do neural network with APOGEE stellar spectra DR13/DR14 and Gaia DR1 with Tensorflow
#### !!!Still in Development!!!!

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites


***This python package must be using with Tensorflow 1.4.0 (As of 17 Oct 2017, Tensorflow 1.4.0 still in beta and you have to compile it from source)

***This package has no Keras dependency, this package have been migrated to Tensorflow. You dont need to install Keras anymore

```
Python 3.6 or above (Anaconda 5.0.0 64bit is recommended)
Tensorflow 1.4.0 or above (***There is no GPU version of tensorflow for MacOS user)
Tensorflow-gpu 1.4.0 or above is recommended
```

### Installing

Just run the following commmand to install after you open the command line windows in the project folder

```
python setup.py install
```


## Authors

* **Henry Leung** - *Initial work and developer* - [henrysky](https://github.com/henrysky)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## Acknowledgments

* **Jo Bovy** - *Original developer of `xmatch` of `astroNN.datasets.xmatch.xmatch()`* - [jobovy](https://github.com/jobovy)