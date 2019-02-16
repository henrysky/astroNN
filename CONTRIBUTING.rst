Contributor and Issue Reporting guide
=====================================

When contributing to this repository, please first discuss the big changes you wish to make via opening issue,
email, or any other method with the maintainers of this repository.

Submitting bug reports and feature requests
---------------------------------------------

Bug reports and feature requests should be submitted by creating an issue on https://github.com/henrysky/astroNN

Pull Request
-------------

This is a general guideline to make pull request (PR).

#. Go to https://github.com/henrysky/astroNN, click the ``Fork`` button.
#. Download your own astroNN fork to your computer by ``$git clone https://github.com/your_username/astroNN``
#. Create a new branch with a short simple name that represents the change you want to make
#. Make commits locally in that new branch, and push to your own astroNN fork repository
#. Create a pull request by clicking the ``New pull request`` button.

New Model Proposal guide
-----------------------------
astroNN acts as a platform to share astronomy-oriented neural networks, so you are welcome to do so.

To add new models:

* Import your model in ``astroNN\models\__init__.py`` and add the model class name to ``__all__``
* Add a documentation page for the new model and add link it appropriately in ``docs\source\index.rst``
* Add the new model to the tree diagram and API under appropriate class in ``docs\souce\neuralnets\basic_usage.rst``
* Add the new model to the release history in ``docs\source\history.rst``

If your new model is proposed along with a paper, add your model to the test suite in ``tests\test_paper_models.rst``
just to make sure your model works fine against future changes in astroNN.

Possible New Features and Improvement in the future
----------------------------------------------------

.. topic:: Anticipating Tensorflow 2.0

    * Drop Keras support as Keras is deeply integrated in Tensorflow 2.0?
    * Need to keep track of deprecation warning in current Tensorflow 1.1x series to better prepare for tf2.0

.. topic:: GPU performance related issues

    * Data reduction pipeline on GPU?
    * Multiple GPU support!

.. topic:: Neural Network related issues

    * Currently the Bayesian NN models only use Dropout VI, maybe introduce more methods especially from TF-Probability
    * Have some nice VAE or GAN thing, maybe on spectroscopic data first