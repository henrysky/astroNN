## Epistemic and Aleatoric Uncertainty Analysis in Bayesian Deep Learning

#### The motivation of this tutorial is to help people to understand and implement uncertainty analysis in neural networks.
#### If you are dealing with time series data with recurrent neural net or image segmentation, please refer Papers/Materials below below.

If Github has an issue (or is too slow) loading the Jupyter Notebooks, you can go to 
https://nbviewer.jupyter.org/github/henrysky/astroNN/tree/master/demo_tutorial/NN_uncertainty_analysis/

Uncertainty with Dropout Variational Inference Demonstration for Regression and Classification
-----------------------------------------------------------------------------------------------------------

astroNN was used to do a regression task with Dropout VI in a paper 
**Deep learning of multi-element abundances from high-resolution spectroscopic data** 
which the code available at [https://github.com/henrysky/astroNN_spectra_paper_figures](https://github.com/henrysky/astroNN_spectra_paper_figures)
and the paper available at [[arxiv:1808.04428](https://arxiv.org/abs/1808.04428)][[ADS](https://ui.adsabs.harvard.edu/#abs/2018arXiv180804428L/)]. 
We demonstrated Dropout VI can report reasonable uncertainty with high prediction 
accuracy trained on incomplete stellar parameters and abundances data from from high-resolution stellar spectroscopic data.

Regression: Two Jupyter notebooks provided here to fit two different functions
* Function: y=x sin(x) : [Here](Uncertainty_Demo_x_sinx.ipynb)
* Function: y=0.1+0.3x+0.4x^2 : [Here](Uncertainty_Demo_quad.ipynb)

Classification: Two Jupyter notebooks provided here to do classification on MNIST and astroNN's Galaxy10
* MNIST classification with uncertainty: [Here](Uncertainty_Demo_MNIST.ipynb)
* Galaxy10 classification with uncertainty (Unavailable, work in progress): [Here](Uncertainty_Demo_classification.ipynb)

Uncertainty with Tensorflow Probability using Flipout/Reparameterization/Local Reparameterization
-----------------------------------------------------------------------------------------------------------
Tensorflow Probability github: https://github.com/tensorflow/probability

Regression:
* Function: y=x sin(x) : [Here](Uncertainty_Demo_x_sinx_tfp.ipynb)

<br>

Papers/Materials
-----------------
For Dropout variational inference, related material:
* Paper: [Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning](https://arxiv.org/abs/1506.02142)
* Paper: [What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?](https://arxiv.org/abs/1703.04977)
* Paper: [Bayesian Convolutional Neural Networks with Bernoulli Approximate Variational Inference](https://arxiv.org/abs/1506.02158)
* Yarin Gal's Blog: [What My Deep Model Doesn't Know...](https://mlg.eng.cam.ac.uk/yarin/blog_3d801aa532c1ce.html)
* [Demo from Yarin Gal written in javascript](https://github.com/yaringal/HeteroscedasticDropoutUncertainty)

<br>

For variational Bayesian methods used by Tensorflow Probability, related material:
* Paper for Flipout: [Flipout: Efficient Pseudo-Independent Weight Perturbations on Mini-Batches](https://arxiv.org/abs/1803.04386)
* Paper for Reparameterization: [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)
* Paper for Local Reparameterization: [Variational Dropout and the Local Reparameterization Trick](https://arxiv.org/abs/1506.02557)

<br>

Other resources:
* If you are doing classification task: [Building a Bayesian deep learning classifier](https://github.com/kyle-dorman/bayesian-neural-network-blogpost)
* If you are doing recurrent neural net: [BayesianRNN](https://github.com/yaringal/BayesianRNN)
* If you interested in industrial application of this method: [Uber](https://eng.uber.com/neural-networks-uncertainty-estimation/)

<br>

Here is [astroNN](https://github.com/henrysky/astroNN), please take a look if you are interested in astronomy or how neural network applied in astronomy
* **Henry Leung** - *Astronomy student, University of Toronto* - [henrysky](https://github.com/henrysky)
* Project supervisor: **Jo Bovy** - *Professor, Department of Astronomy and Astrophysics, University of Toronto* - [jobovy](https://github.com/jobovy)
* Contact Henry: henrysky.leung [at] utoronto.ca
