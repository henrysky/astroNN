.. automodule:: astroNN.datasets.galaxy10

Galaxy10 Dataset
===================

Introduction
---------------

Galaxy10 is a dataset contains 25753 69x69 pixels colored galaxy images (g, r and i band) separated in 10 classes.
Galaxy10 images come from `Sloan Digital Sky Survey`_ and labels come from `Galaxy Zoo`_.

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

These classes are mutually exclusive, but Galaxy Zoo relies on human volunteers to classify galaxy images and the
volunteers do not agree on all images. For this reason, Galaxy10 only contains images for which more than 55% of the
votes agree on the class. That is, more than 55% of the votes among 10 classes are for a single class for that particular
image. If none of the classes get more than 55%, the image will not be included in Galaxy10 as no agreement was reached.
As a result, 25753 images after the cut.

The justification of 55% as the threshold is based on validation. Galaxy10 is meant to be an alternative to MNIST or
Cifar10 as a deep learning toy dataset for astronomers. Thus astroNN.models.Cifar10_CNN is used with Cifar10 as a reference.
The validation was done on the same astroNN.models.Cifar10_CNN.
50% threshold will result a poor neural network classification accuracy although around 36000 images in the dataset,
many are probably misclassified and neural network has a difficult time to learn. 60% threshold result is similar to 55%
, both classification accuracy is similar to Cifar10 dataset on the same network, but 55%
threshold will have more images be included in the dataset. Thus 55% was chosen as the threshold to cut data.

The original images are 424x424, but were cropped to 207x207 centered at the images
and then downscaled 3 times via bilinear interpolation to 69x69 in order to make them manageable on most computer and
graphics card memory.

There is no guarantee on the accuracy of the labels. Moreover, Galaxy10 is not a balanced dataset and it should only
be used for educational or experimental purpose. If you use Galaxy10 for research purpose, please cite Galaxy Zoo and
Sloan Digital Sky Survey.

For more information on the original classification tree: `Galaxy Zoo Decision Tree`_

.. _Galaxy Zoo Decision Tree: https://data.galaxyzoo.org/gz_trees/gz_trees.html
.. _Cifar10: http://www.sdss.org/

.. image:: galaxy10_example.png

Download Galaxy10
-------------------

Galaxy10.h5: http://astro.utoronto.ca/~bovy/Galaxy10/Galaxy10.h5

SHA256: ``969A6B1CEFCC36E09FFFA86FEBD2F699A4AA19B837BA0427F01B0BC6DED458AF``

Size: 200 MB (210,234,548 bytes)

Or see below to load (and download automatically) the dataset with astroNN

TL;DR for Beginners
----------------------

You can view the Jupyter notebook in here: https://github.com/henrysky/astroNN/blob/master/demo_tutorial/galaxy10/Galaxy10_Tutorial.ipynb

OR you can train with astroNN and just copy and paste the following script to get and train a simple neural network on Galaxy10

Basically first we load the Galaxy10 with astroNN and split into train and test set. astroNN will split the training
set into training data and validation data as well as normalizing them automatically.

`Glaxy10CNN` is a simple 4 layered convolutional neural network consisted of 2 convolutional layers and 2 dense layers.

.. code-block:: python

    # import everything we need first
    from keras.utils import np_utils
    import numpy as np
    from sklearn.model_selection import train_test_split
    import pylab as plt

    from astroNN.models import Galaxy10CNN
    from astroNN.datasets import galaxy10
    from astroNN.datasets.galaxy10 import galaxy10cls_lookup, galaxy10_confusion

    # To load images and labels (will download automatically at the first time)
    # First time downloading location will be ~/.astroNN/datasets/
    images, labels = galaxy10.load_data()

    # To convert the labels to categorical 10 classes
    labels = np_utils.to_categorical(labels, 10)

    # Select 10 of the images to inspect
    img = None
    plt.ion()
    print('===================Data Inspection===================')
    for counter, i in enumerate(range(np.random.randint(0, labels.shape[0], size=10).shape[0])):
        img = plt.imshow(images[i])
        plt.title('Class {}: {} \n Random Demo images {} of 10'.format(np.argmax(labels[i]), galaxy10cls_lookup(labels[i]), counter+1))
        plt.draw()
        plt.pause(2.)
    plt.close('all')
    print('===============Data Inspection Finished===============')

    # To convert to desirable type
    labels = labels.astype(np.float32)
    images = images.astype(np.float32)

    # Split the dataset into training set and testing set
    train_idx, test_idx = train_test_split(np.arange(labels.shape[0]), test_size=0.1)
    train_images, train_labels, test_images, test_labels = images[train_idx], labels[train_idx], images[test_idx], labels[test_idx]

    # To create a neural network instance
    galaxy10net = Galaxy10CNN()

    # set maximium epochs the neural network can run, set 5 to get quick result
    galaxy10net.max_epochs = 5

    # To train the nerual net
    # astroNN will normalize the data by default
    galaxy10net.train(train_images, train_labels)

    # print model summary before training
    galaxy10net.keras_model.summary()

    # After the training, you can test the neural net performance
    # Please notice predicted_labels are labels predicted from neural network. test_labels are ground truth from the dataset
    predicted_labels = galaxy10net.test(test_images)

    # Convert predicted_labels to class
    prediction_class = np.argmax(predicted_labels, axis=1)

    # Convert test_labels to class
    test_class = np.argmax(test_labels, axis=1)

    # Prepare a confusion matrix
    confusion_matrix = np.zeros((10,10))

    # create the confusion matrix
    for counter, i in enumerate(prediction_class):
        confusion_matrix[i, test_class[counter]] += 1

    # Plot the confusion matrix
    galaxy10_confusion(confusion_matrix)


Load with astroNN
-------------------

.. code-block:: python

    from astroNN.datasets import galaxy10
    from keras.utils import np_utils
    import numpy as np

    # To load images and labels (will download automatically at the first time)
    # First time downloading location will be ~/.astroNN/datasets/
    images, labels = galaxy10.load_data()

    # To convert the labels to categorical 10 classes
    labels = np_utils.to_categorical(labels, 10)

    # To convert to desirable type
    labels = labels.astype(np.float32)
    images = images.astype(np.float32)

OR Load with Python & h5py
----------------------------

You should download Galaxy10.h5 first and open python at the same location and run the following to open it:

.. code-block:: python

    import h5py
    import numpy as np
    from keras.utils import np_utils

    # To get the images and labels from file
    with h5py.File('Galaxy10.h5', 'r') as F:
        images = np.array(F['images'])
        labels = np.array(F['ans'])

    # To convert the labels to categorical 10 classes
    labels = np_utils.to_categorical(labels, 10)

    # To convert to desirable type
    labels = labels.astype(np.float32)
    images = images.astype(np.float32)

Split into train and test set
----------------------------------

.. code-block:: python

    import numpy as np
    from sklearn.model_selection import train_test_split

    train_idx, test_idx = train_test_split(np.arange(labels.shape[0]), test_size=0.1)
    train_images, train_labels, test_images, test_labels = images[train_idx], labels[train_idx], images[test_idx], labels[test_idx]

Lookup Galaxy10 Class
--------------------------

You can lookup Galaxy10 class to the corresponding name by

.. code-block:: python

    from astroNN.datasets.galaxy10 import galaxy10cls_lookup
    galaxy10cls_lookup(#class_number_here)


Galaxy10 Dataset Authors
--------------------------

-  | **Henry Leung** - Compile the Galaxy10 - henrysky_
   | Astronomy Undergrad, University of Toronto

-  | **Jo Bovy** - Supervisor of Henry Leung - jobovy_
   | Astronomy Professor, University of Toronto

.. _henrysky: https://github.com/henrysky
.. _jobovy: https://github.com/jobovy

Acknowledgments
--------------------------

1. Galaxy10 dataset classification labels come from `Galaxy Zoo`_
2. Galaxy10 dataset images come from `Sloan Digital Sky Survey`_ (SDSS)

Galaxy Zoo is described in `Lintott et al. 2008, MNRAS, 389, 1179`_ and the data release is described in
`Lintott et al. 2011, 410, 166`_

Funding for the SDSS and SDSS-II has been provided by the Alfred P. Sloan Foundation, the Participating Institutions,
the National Science Foundation, the U.S. Department of Energy, the National Aeronautics and Space Administration, the
Japanese Monbukagakusho, the Max Planck Society, and the Higher Education Funding Council for England. The SDSS Web
Site is http://www.sdss.org/.

The SDSS is managed by the Astrophysical Research Consortium for the Participating Institutions. The Participating
Institutions are the American Museum of Natural History, Astrophysical Institute Potsdam, University of Basel,
University of Cambridge, Case Western Reserve University, University of Chicago, Drexel University, Fermilab, the
Institute for Advanced Study, the Japan Participation Group, Johns Hopkins University, the Joint Institute for Nuclear
Astrophysics, the Kavli Institute for Particle Astrophysics and Cosmology, the Korean Scientist Group, the Chinese
Academy of Sciences (LAMOST), Los Alamos National Laboratory, the Max-Planck-Institute for Astronomy (MPIA), the
Max-Planck-Institute for Astrophysics (MPA), New Mexico State University, Ohio State University, University of
Pittsburgh, University of Portsmouth, Princeton University, the United States Naval Observatory, and the University of
Washington.

.. _Sloan Digital Sky Survey: http://www.sdss.org/
.. _Galaxy Zoo: https://www.galaxyzoo.org/
.. _Lintott et al. 2008, MNRAS, 389, 1179: http://adsabs.harvard.edu/abs/2008MNRAS.389.1179L
.. _Lintott et al. 2011, 410, 166: http://adsabs.harvard.edu/abs/2011MNRAS.410..166L
