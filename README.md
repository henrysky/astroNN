![AstroNN Logo](astroNN_icon_withname.png)

## Getting Started and Prerequisites

astroNN is a  python package to do neural network with APOGEE stellar spectra DR14 and Gaia DR1 with Tensorflow and Keras.
The idea is feeding spectra into neural network and train it with ASPCAP stellar parameter or feeding spectra into neural
network and train it with Gaia DR1 parallax.

This is a python package developing for an undergraduate research project by `Henry W.H. Leung (Toronto)` under the 
supervision of Professor `Jo Bovy, Unviersity of Toronto Department of Astronomy and Astrophysics.`
#### Still in Active Development!!!!

#### Currently not working, please download astroNN committed at least before 7 Dec!!!

### [New Draft of astroNN Documentation](https://henrysky.github.io/astroNN/)

#### [Quick Start guide](https://henrysky.github.io/astroNN/quick-start.html)

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

* **Jo Bovy** - [jobovy](https://github.com/jobovy)\
*Supervisor of **Henry W.H. Leung** on this undergraduate project*\
*Original developer of `xmatch()` of `astroNN.datasets.xmatch.xmatch()`* from his [gaia_tools](https://github.com/jobovy/gaia_tools)

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
