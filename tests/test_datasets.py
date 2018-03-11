import unittest
import numpy.testing as npt


class DatasetTestCase(unittest.TestCase):
    def test_xmatch(self):
        from astroNN.datasets import xmatch
        import numpy as np

        # Some coordinates for cat1, J2000.
        cat1_ra = np.array([36., 68., 105., 23., 96., 96.])
        cat1_dec = np.array([72., 56., 54., 55., 88., 88.])

        # Some coordinates for cat2, J2000.
        cat2_ra = np.array([23., 56., 222., 96., 245., 68.])
        cat2_dec = np.array([36., 68., 82., 88., 26., 56.])

        # Using maxdist=2 arcsecond separation threshold, because its default, so not shown here
        # Using epoch1=2000. and epoch2=2000., because its default, so not shown here
        # because both datasets are J2000., so no need to provide pmra and pmdec which represent proper motion
        idx_1, idx_2, sep = xmatch(cat1_ra, cat2_ra, colRA1=cat1_ra, colDec1=cat1_dec, colRA2=cat2_ra, colDec2=cat2_dec,
                                   swap=False)
        self.assertEqual(len(idx_1), len(idx_2))

    def test_apokasc(self):
        from astroNN.datasets.apokasc import apokasc_load

        ra, dec, logg = apokasc_load()
        gold_ra, gold_dec, gold_logg, basic_ra, basic_dec, basic_logg = apokasc_load(combine=False)

    def test_galaxy10(self):
        # import everything we need first
        from keras.utils import np_utils
        import numpy as np
        from sklearn.model_selection import train_test_split
        import pylab as plt

        from astroNN.models import Galaxy10_CNN
        from astroNN.datasets import galaxy10
        from astroNN.datasets.galaxy10 import galaxy10cls_lookup, galaxy10_confusion

        # To load images and labels (will download automatically at the first time)
        # First time downloading location will be ~/.astroNN/datasets/
        images, labels = galaxy10.load_data()

        # To convert the labels to categorical 10 classes
        labels = np_utils.to_categorical(labels, 10)

        # To convert to desirable type
        labels = labels.astype(np.float32)
        images = images.astype(np.float32)

        # Split the dataset into training set and testing set
        train_idx, test_idx = train_test_split(np.arange(labels.shape[0]), test_size=0.1)
        train_images, train_labels, test_images, test_labels = images[train_idx], labels[train_idx], images[test_idx], \
                                                               labels[test_idx]

        # To create a neural network instance
        galaxy10net = Galaxy10_CNN()

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
        confusion_matrix = np.zeros((10, 10))

        # create the confusion matrix
        for counter, i in enumerate(prediction_class):
            confusion_matrix[i, test_class[counter]] += 1

        # Plot the confusion matrix
        galaxy10_confusion(confusion_matrix)

        for i in range(10):
            print(galaxy10cls_lookup(i))


if __name__ == '__main__':
    unittest.main()
