![AstroNN Logo](astroNN_icon_withname.png)

## Getting Started

astroNN is a  python package to do neural network with APOGEE stellar spectra DR14 and Gaia DR1 with Tensorflow and Keras.
The idea is feeding spectra into neural network and train it with ASPCAP stellar parameter or feeding spectra into neural
network and train it with Gaia DR1 parallax.

This is a python package developing for an undergraduate research project by `Henry W.H. Leung (Toronto)` under the 
supervision of Professor `Jo Bovy, Unviersity of Toronto Department of Astronomy and Astrophysics.`
#### !!!Still in Active Development!!!!

### !!!Merry xmas and happy new year!!!
### !!!No update should be expected until 4th Jan 2018!!!
### !!!Currently not working, please download astroNN committed at least before 7 Dec!!!

### New Year Plan
* Reorganization of the codes and modules (a week?)
* Update Documentation (a day?)
* Possibility of ditching Keras and using Tensorflow directly (2 weeks but probably not gonna do it``)
* Using individual visit (a day)
* Using better Gaia parallax [Lauren Anderson et al. 2017](https://arxiv.org/pdf/1706.05055.pdf) (a day)
* Integrate uncertainty analysis (a day)
* Robust downloading with checksum integrity checking

### New Draft of astroNN Documentation
[https://henrysky.github.io/astroNN/](https://henrysky.github.io/astroNN/)

## Updates History
`13 Oct 2017` - `astroNN was created`\
`19 Oct 2017` - `astroNN 0.1 - includes basic function of downloading and compiling data, training and 
testing neural network`\
`21 Oct 2017` - `astroNN_tutorial was creacted to include tutorial jupyter notebook`\
`24 Oct 2017` - `Updated the folder structure, update the code to remove unnessary I/O access, update plot`\
`26 Oct 2017` - `Performance and high RAM usage issue fix`\
`27 Oct 2017` - `The Cannon 2 comparison module and critical bug fix!!`\
`28 Oct 2017` - `Initial generative neural network for spectra noise reduction`\
`31 Oct 2017` - `Bug fix, more diagnostic info, performance fix`\
`1 Nov 2017` - `Platform independent path fix`\
`2 Nov 2017` - `More disgnostic info and save hyperparameter`\
`2 Nov 2017` - `New visualizing module`\
`3 Nov 2017` - `Visualizing module fix and including in apogee_train`\
`7 Nov 2017` - `Update apogee_cnn_model_1`\
`8 Nov 2017` - `Gaia DR1 downloading function and gaia h5 compiler` \
`9 Nov 2017` - `Gaia DR1 models`\
`10 Nov 2017` - `Refractored some the code`\
`13 Nov 2017` - `Add APOKASC surface gravity checking (golden)`\
`15 Nov 2017` - `Add APOKASC surface gravity checking (all catalog)`\
`16 Nov 2017` - `Add Red Clumps Catalog checking`\
`17 Nov 2017` - `Add Experimental channel`\
`18 Nov 2017` - `Add blackbox module for NN attention checking`\
`19 Nov 2017` - `Add NNgalaxy module for plotting milkyway via NN`\
`20 Nov 2017` - `Add Jacobia of NN`\
`23 Nov 2017` - `Add model uncertainty calculation`\
`24 Nov 2017` - `Add uncertainty propagation`\
`25 Nov 2017` - `Better CPU/GPU management`\
`26 Nov 2017` - `Circular import loop fix and code refractoring`\
`27 Nov 2017` - `CPU fallback option and GPU memory management`\
`28 Nov 2017` - `Add uncertainty analysis to APOKASC and Red Clumps Catalog checking`\
`29 Nov 2017` - `Add MC dropout option`\
`1 Dec 2017` - `Reduce RAM footprint`\
`2 Dec 2017` - `Take adavantage of multi-threading data generation`\
`4 Dec 2017` - `Continuum normalization with Chebyshev polynomials`\
`5 Dec 2017` - `Integrate continuum normalization in h5 compiler`\
`7 Dec 2017` - `Uncertainty depends on inpput data (23 Nov only calculated model uncertainty)`\
`8 Dec 2017` - `Aleatoric Uncertainty Analysisa added on 7 Dec is currently not integrated`\
`10 Dec 2017` - `Add uncertainty demo`\
`11 Dec 2017` - `Fix DR13 wrong pixel info`\
`12 Dec 2017` - `Final commit before xmas and new year holiday`\
`16 Dec 2017` - `Code refactorings`\
`17 Dec 2017` - `Preparing to transist to a less braindead API`\
`21 Dec 2017` - `Object-orientation models and training API`\
`23 Dec 2017` - `Add StarNet archetiture (arXiv:1709.09182)`


## Prerequisites

This package must be using with Tensorflow 1.4.0 or above

Only Keras with Tensorflow backend is supported!!

Multi-gpu training is not supported.

```
Python 3.6 or above
Tensorflow OR Tensorflow-gpu (1.4.0 or above)
Keras 2.1.0 or above
CUDA 8.0 and CuDNN 6.1 (only neccessary for Tensorflow-gpu 1.4.0)
CUDA 9.0 and CuDNN 7.0 (only neccessary for Tensorflow-gpu 1.5.0 beta, you should only use 1.5.0 beta if and only if you are using Nvidia Volta)
graphviz and pydot_ng are required to plot the model architecture
tqdm and astroquery also required
```

Please go to one of the following link to download a wheel locally and install it\
[Tensorflow](https://pypi.python.org/pypi/tensorflow/)\
[Tensorflow-gpu](https://pypi.python.org/pypi/tensorflow-gpu/)

For instruction on how to install Tensorflow, please refer to their official website
[->Installing TensorFlow](https://www.tensorflow.org/install/)

Recommended system requirement:
```
CPU which supports AVX2 (2014 or later Intel CPU/2015 or later AMD CPU)
8GB RAM or above
Nvidia Graphics card (GTX700 series or above)
(If using GPU): At least 2GB VRAM on GPU
Only Windows and Linux is officially supported by Tensorflow-GPU
```

## Installing

Recommended method of installation as this python package is still in active development and will update daily:
```
python setup.py develop
```

Or run the following command to install after you open a command line window in the package folder:
```
python setup.py install
```

## Tutorial

Please refer to tutorial section [Tutorial](https://github.com/henrysky/astroNN_tutorial)

### Folder Structure
This code depends on an environment variables and folder. The environment variables is 
* `SDSS_LOCAL_SAS_MIRROR`: top-level directory that will be used to (selectively) mirror the SDSS SAS
* `GAIA_TOOLS_DATA`: top-level directory under which the data will be stored
* A dedicated project folder is recommended to run astroNN, always run astroNN under the root of project folder

How to set environment variable on different operating system: [Guide here](https://www.schrodinger.com/kb/1842)
 
##### The APOGEE folder structure should be consistent with [APOGEE](https://github.com/jobovy/apogee/) python package by Jo Bovy, tools for dealing with APOGEE data

##### The GAIA folder structure should be consistent with [gaia_tools](https://github.com/jobovy/gaia_tools/) python package by Jo Bovy, tools for dealing with GAIA data

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
 

    $GAIA_TOOLS_DATA/
	    gaia/tgas_source/fits/
			TgasSource_000-000-000.fits
			TgasSource_000-000-0**.fits
			
## Early result
astroNN apogee_cnn_1 model vs the Cannon 2
![](https://image.ibb.co/fDY5JG/table1.png)

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
