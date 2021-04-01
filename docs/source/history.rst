
History
=========

v1.1 series
--------------

.. topic:: v1.1.0 (xx xxx 20xx)


    | **New features:**

    * Added models : ``ApogeeKplerEchelle``
    * Input data can now be a dict, such as ``nn.train({'input': input_data, 'input': aux_input_data}, {'output': labels, 'output_aux': aux_labels})``
    * Added numerical integrator for NeuralODE
    * tqdm progress bar for model prediction
    * Added a new improved version ``Galaxy10``

    | **Improvement:**

    * Fully compatible with Tensorflow 2
    * Model training/inference should be much faster by using Tensorflow v2 eager execution (see: https://github.com/tensorflow/tensorflow/issues/33024#issuecomment-551184305)
    * Improved continuous integration testing with Github Actions, now actually test model learn properly with real world data instead of checking no syntax error with random data
    * Support `sample_weight` in all losss functions and training
    * Improved catalog coordinates matching
    * New documentation webpages

    | **Breaking Changes:**

    * Deprecated support for all Tensorflow 1.x
    * Dropped optional Keras support, now depends on Tensorflow only
    * Tested with Tensorflow 2.4.x
    * Incompatible to Tensorflow 1.x and <=2.2 due to necessary changes for Tensorflow eager execution API
    * Change neural network models ``train()``, ``test()``, ``train_on_batch()`` method to ``fit()``, ``predict()``, ``fit_on_batch()``
    * Old ``Galaxy10`` has been renamed to ``Galaxy10 SDSS`` and the new version will replace and call ``Galaxy10``

v1.0 series
--------------

.. topic:: v1.0.1 (5 March 2019)

    This release mainly targeted to the paper ``Simultaneous calibration of spectro-photometric distances and the Gaia DR2 parallax zero-point offset with deep learning``
    available at
    [`arXiv:1902.08634 <https://arxiv.org/abs/1902.08634>`_]
    [`ADS <https://ui.adsabs.harvard.edu/abs/2019MNRAS.489.2079L/abstract>`_]

    Documentation for this version is available at
    https://astronn.readthedocs.io/en/v1.0.1/

    | **New features:**

    * Better and faster with IPython tab auto-completion
    * Added models : ``ApogeeDR14GaiaDR2BCNN``

    | **Improvement:**

    * Improved data pipeline to generate data for NNs

    | **Breaking Changes:**

    * Tested with Tensorflow 1.11.0/1.12.0/1.13.1 and Keras 2.2.0/2.2.4

.. topic:: v1.0.0 (16 August 2018)

    This is the first release of astroNN. This release mainly targeted to the paper ``Deep learning of multi-element abundances from high-resolution spectroscopic data`` available at
    [`arXiv:1804.08622 <https://arxiv.org/abs/1808.04428>`_]
    [`ADS <https://ui.adsabs.harvard.edu/abs/2019MNRAS.483.3255L/abstract>`_]

    Documentation for this version is available at
    https://astronn.readthedocs.io/en/v1.0.0/

    | **New features:**

    * Initial Release!!

    | **Breaking Changes:**

    * Tested with Tensorflow 1.8.0/1.9.0 and Keras 2.2.0/2.2.2
    * Python 3.6 or above only

v0.0 series
--------------

.. topic:: v0.0.0  (13 October 2017)

    First commit of astroNN on Github!!!
