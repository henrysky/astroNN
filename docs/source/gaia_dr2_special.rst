
GAIA DR2 Preparation and Possible Science
=============================================

GAIA DR2 will be released on 25 April 2018 with data collected from 25 July 2014 to 23 May 2016 with 1.5 billion sources.

astroNN will be used to train neural network to predict intrinsic brightness of stars from APOGEE spectra trained with
GAIA DR1 parallax. This page will act as a notebook for the author (Henry) and share his latest update on GAIA DR2.


Plans
-------

#. If neural network turns out very accurate when DR2 comes out, how did neural network predict those distance
#. If neural network turns out very accurate when DR2 comes out, then we can get distance for many APOGEE spectra
#. If neural network failed, is predicting intrinsic brightness from APOGEE spectra impossible, or just because the training set is too small in DR1 led to failure


2M16363993+3654060 distance disagreement between astroNN and GAIA/Anderson2017
---------------------------------------------------------------------------------

.. image:: gaia_dr2/fakemag.png

Neural Network trained on Anderson2017 parallax constantly predicted an almost constant offset with very small uncertainty
to the ground truth (Anderson2017), the star was found to be 2M16363993+3654060. astroNN agredd pretty well with APOGEE_distances BPG_dist50.
Seems like GAIA/Anderson2017 is the on which is far off.

The result:

#. astroNN Bayesian Neural Network (Trained on ASPCAP parameters and Anderson2017 parallax): :math:`2188.34 \text{ parsec} \pm 395.16 \text{ parsec}`
#. APOGEE_distances BPG_dist50: :math:`2266.15 \text{ parsec} \pm 266.1705 \text{ parsec}`
#. astroNN Bayesian Neural Network: :math:`568.08 \text{ parsec} \pm 403.86 \text{ parsec}`
#. astroNN Bayesian Neural Network: :math:`318.05 \text{ parsec} \pm 1021.73 \text{ parsec}`

Distance Prediction with APOGEE Spectra
----------------------------------------------------

The model will be uploaded here later