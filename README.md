![AstroNN Logo](astroNN_icon_withname.png)

## Getting Started

A python package to do neural network with APOGEE stellar spectra DR13/DR14 and Gaia DR1 with Tensorflow

This is a python package developing for an undergraduate research project by `Henry W.H. Leung (Toronto)` under the supervision of 
Professor `Jo Bovy`, Unviersity of Toronto Department of Astronomy and Astrophysics.
#### !!!Still in Active Development!!!!

## Version History
`13 Oct 2017` - `astroNN was created`\
`19 Oct 2017` - `astroNN 0.1 - includes basic function of downloading and compiling data, training and 
testing neural network`\
`19 Oct 2017` - `astroNN_tutorial was creacted to include tutorial jupyter notebook`



## Prerequisites

This package must be using with Tensorflow 1.4.0 or above

Only Keras with Tensorflow backend is supported

Multi-gpu training is not supported, however you can run multiple models separately on your multi-gpu system.

~~This package has no Keras dependency, it have been migrated to Tensorflow. You dont need to install Keras anymore~~

```
Python 3.6 or above (Anaconda 5.0.0 64bit is tested by author)
Tensorflow OR Tensorflow-gpu (1.4.0 or above)
Keras 2.0.8 or above
CUDA 8.0 and CuDNN 6.1 (only neccessary for Tensorflow-gpu 1.4.0)
CUDA 9.0 and CuDNN 7.0 (only neccessary for Tensorflow-gpu 1.5.0 beta, you should only use 1.5.0 beta if and only if you are using Nvidia Volta)
graphviz and pydot_ng are required to plot the model architecture
```

Please go to one of the following link to download a wheel locally and install it\
[Tensorflow](https://pypi.python.org/pypi/tensorflow/)\
[Tensorflow-gpu](https://pypi.python.org/pypi/tensorflow-gpu/)

For instruction on how to install Tensorflow, please refer to their official website
[->Installing TensorFlow](https://www.tensorflow.org/install/)

## Installing

Recommended method of installation as this python package is still in active development and will update daily:
```
python setup.py develop
```

Or just run the following command to install after you open the command line windows in the project folder:
```
python setup.py install
```

## Tutorial

Please refer to tutorial section [Tutorial](https://github.com/henrysky/astroNN_tutorial)

### Folder Structure
You should create a project folder, then create a folder named `apogee_dr14` and put `allStar-l31c.2.fits` and every aspcap 
 fits under it. Always run your command line or python under the project folder.
 
##### This folder structure guideline is temporary only, I am working on a more sensible folder structure.

## Authors

* **Henry W.H. Leung** - *Initial work and developer* - [henrysky](https://github.com/henrysky)\
Contact Henry: [henrysky.leung@mail.utoronto.ca](mailto:henrysky.leung@mail.utoronto.ca)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## Acknowledgments

* **Jo Bovy** - [jobovy](https://github.com/jobovy)\
*Supervisor of **Henry W.H. Leung** on this undergraduate project*\
*Original developer of `xmatch()` of `astroNN.datasets.xmatch.xmatch()`* from his [gaia_tools](https://github.com/jobovy/gaia_tools)

* **S. Fabbro et al. (2017)** - [arXiv:1709.09182](https://arxiv.org/abs/1709.09182)\
*This project is inspired by [StarNet](https://github.com/astroai/starnet)*
