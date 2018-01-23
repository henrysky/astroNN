
Galaxy10 Dataset
===================

Introduction
---------------

Galaxy10 is a dataset contains 25753 69x69 pixels colored galaxy images (r,g and i band) separated in 10 classes.
Galaxy10 images come from Sloan Digital Sky Survey and labels come from `Galaxy Zoo`_.

::

    Galaxy10 dataset (25753 images)
    ├── Class 0 (3461 images): Disk, Face-on, No Spiral
    ├── Class 1 (6997 images): Smooth, Completely round
    ├── Class 2 (6292 images): Smooth, in-between round
    ├── Class 3 (394 images): Smooth, Cigar shaped
    ├── Class 4 (3060 images): Disk, Edge-on, Rounded Bulge
    ├── Class 5 (17 images): Disk, Edge-on, Boxy Bulge
    ├── Class 6 (1089 images): Disk, Edge-on, No Bulge
    ├── Class 7 (1932 images): Disk, Face-on, Tight Spiral
    ├── Class 8 (1466 images): Disk, Face-on, Medium Spiral
    └── Class 9 (1045 images): Disk, Face-on, Loose Spiral

Since the classes are mutually exclusive but due to the fact that Galaxy Zoo relies on human volunteers to
classify (vote) galaxy images, Galaxy10 only contains images with more than 55% of the votes, i.e. more than 55%
votes among 10 classes for that particular image, if none of the classes get more than 55%, it will not be included in
Galaxy10 as no agreement reached.

The original images are 424x424, but were cropped to 207x207 centered at the images
and then downscaled 3 times to 69x69 in order to make them manageable on most computer and graphics card memory.

There is no guarantee on the accuracy of the labels. Moreover, Galaxy10 is not a balanced dataset and it should only
be used for educational or experimental purpose. If you use Galaxy10 for research purpose, please cite Galaxy Zoo and
Sloan Digital Sky Survey.

For more information on the original classification tree: `Galaxy Zoo Decision Tree`_

.. _Galaxy Zoo Decision Tree: https://data.galaxyzoo.org/gz_trees/gz_trees.html


.. image:: galaxy10_example.png

Download Galaxy10
-------------------

Galaxy10.h5: http://astro.utoronto.ca/somewhere

SHA256: 969A6B1CEFCC36E09FFFA86FEBD2F699A4AA19B837BA0427F01B0BC6DED458AF


Load with astroNN
-------------------

.. code:: python

    from astroNN.datasets import galaxy10
    from keras.utils import np_utils

    # To load images and labels (will download automatically at the first time)
    images, labels = galaxy10.load_data()

    # To convert the labels to categorical 10 classes
    labels = np_utils.to_categorical(labels, 10)

Load with Python & h5py
----------------------------

.. code:: python

    import h5py
    import numpy as np
    from keras.utils import np_utils

    # To get the images and labels from file
    with h5py.File('Galaxy10.h5', 'r') as F:
        images = np.array(F['images'])
        labels = np.array(F['ans'])

    # To convert the labels to categorical 10 classes
    labels = np_utils.to_categorical(labels, 10)

Galaxy10 Dataset Authors
==========================

-  | **Henry Leung** - Compile the Galaxy10 - henrysky_
   | Astronomy Undergrad, University of Toronto

-  | **Jo Bovy** - Supervisor of Henry Leung - jobovy_
   | Astronomy Professor, University of Toronto

.. _henrysky: https://github.com/henrysky
.. _jobovy: https://github.com/jobovy

Acknowledgments
==================
1. Galaxy10 dataset classification labels come from `Galaxy Zoo`_
2. Galaxy10 dataset images come from Sloan Digital Sky Survey (SDSS)

Galaxy Zoo is described in `Lintott et al. 2008, MNRAS, 389, 1179`_ and the data release is described in
`Lintott et al. 2011, 410, 166`_

Funding for the Sloan Digital Sky Survey IV has been provided by the Alfred P. Sloan Foundation, the
U.S. Department of Energy Office of Science, and the Participating Institutions. SDSS-IV acknowledges
support and resources from the Center for High-Performance Computing at
the University of Utah. The SDSS web site is www.sdss.org.

.. _Galaxy Zoo: https://www.galaxyzoo.org/
.. _Lintott et al. 2008, MNRAS, 389, 1179: http://adsabs.harvard.edu/abs/2008MNRAS.389.1179L
.. _Lintott et al. 2011, 410, 166: http://adsabs.harvard.edu/abs/2011MNRAS.410..166L
