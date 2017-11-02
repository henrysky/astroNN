![AstroNN Logo](astroNN_icon_withname.png)

## Getting Started

A python package to do neural network with APOGEE stellar spectra DR14 and Gaia DR1 with Tensorflow.
The idea is feeding spectra into neural network and train it with ASPCAP stellar parameter and eventually you can try
training with GAIA distance.

This is a python package developing for an undergraduate research project by `Henry W.H. Leung (Toronto)` under the supervision of 
Professor `Jo Bovy`, Unviersity of Toronto Department of Astronomy and Astrophysics.
#### !!!Still in Active Development!!!!

## Version History
`13 Oct 2017` - `astroNN was created`\
`19 Oct 2017` - `astroNN 0.1 - includes basic function of downloading and compiling data, training and 
testing neural network`\
`21 Oct 2017` - `astroNN_tutorial was creacted to include tutorial jupyter notebook`\
`24 Oct 2017` - `Updated the folder structure, update the code to remove unnessary I/O access, update plot`\
`26 Oct 2017` - `Performance and high RAM usage issue fix`\
`27 Oct 2017` - `The Cannon 2 comparison module and critical bug fix!!`\
`28 Oct 2017` - `Initial generative neural network for spectra noise reduction`\
`31 Oct 2017` - `Bug fix, more diagnostic info, performance fix`\
`1 Nov 2017` - `Platform independent path fix`


## Prerequisites

This package must be using with Tensorflow 1.4.0 or above

Only Keras with Tensorflow backend is supported

Multi-gpu training is not supported, however you can run multiple models separately on your multi-gpu system.

~~This package has no Keras dependency, it have been migrated to Tensorflow. You dont need to install Keras anymore~~

```
Python 3.6 or above
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

Or just run the following command to install after you open the command line windows in the package folder:
```
python setup.py install
```

## Tutorial

Please refer to tutorial section [Tutorial](https://github.com/henrysky/astroNN_tutorial)

### Folder Structure
This code depends on an environment variables and folder. The environment variables is 
* `SDSS_LOCAL_SAS_MIRROR`: top-level directory that will be used to (selectively) mirror the SDSS SAS
* A dedicated project folder is recommended to run astroNN, always run python under the root of project folder

How to set environment variable on different operating system: [Guide here](https://www.schrodinger.com/kb/1842)
 
##### The folder structure should be consistent with [APOGEE](https://github.com/jobovy/apogee/) python package by Jo Bovy, tools for dealing with APOGEE data

    $SDSS_LOCAL_SAS_MIRROR/
	dr14/
		apogee/spectro/redux/r8/stars/
					apo25m/
						4102/
							apStar-r8-2M21353892+4229507.fits
							apStar-r8-**********+*******.fits
						****/
					apo1m/
						hip/
							apStar-r8-2M00003088+5933348.fits
							apStar-r8-**********+*******.fits
						***/
					l31c/l31c.2/
						allStar-l30e.2.fits
						allVisit-l30e.2.fits
						4102/
							aspcapStar-r8-l30e.2-2M21353892+4229507.fits
							aspcapStar-r8-l30e.2-**********+*******.fits
						****/
						Cannon/
						    allStarCannon-l31c.2.fits
	dr13/
	   *similar to dr13/*
	   
### Example Plots
#### astroNN apogee_cnn_model_2 (blue) vs Cannon 2 (red)
![](https://image.ibb.co/bDsAcb/markdown1.png)

#### astroNN mimic bestfit
![](https://image.ibb.co/ikr8Vw/markdown2.png)

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
