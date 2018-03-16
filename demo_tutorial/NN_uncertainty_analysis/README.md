## Epistemic and Aleatoric Uncertainty Analysis in Bayesian Deep Learning with Dropout Variational Inference Demonstration for Regression and Classification

#### The motivation of this tutorial is to help people to understand and implement uncertainty analysis in neural network with dropout variational inference for regression and classification. If you are dealing with time series data with recurrent neural net or image segmentation, please refer resources below. Please do not apply the methodology of this tutorial outside the scope of regression and classification. This tutorial is originally a technological demonstration on how astroNN gets its uncertainty

Regression: Two Jupyter notebooks provided here to fit two different functions
* Function: y=x sin(x) : [Here](Uncertainty_Demo_x_sinx.ipynb)
* Function: y=0.1+0.3x+0.4x^2 : [Here](Uncertainty_Demo_quad.ipynb)

Classification: Two Jupyter notebooks provided here to do classification on MNIST and astroNN's Galaxy10
* MNIST : [Here](Uncertainty_Demo_MNIST.ipynb)
* Galaxy10 (Unavailable, work in progress): [Here](Uncertainty_Demo_classification.ipynb)

<br>

Here is [astroNN](https://github.com/henrysky/astroNN), please take a look if you are interested in astronomy or how neural network applied in astronomy
* **Henry Leung** - *Astronomy Undergraduate, University of Toronto* - [henrysky](https://github.com/henrysky)
* Project adviser: **Jo Bovy** - *Professor, Department of Astronomy and Astrophysics, University of Toronto* - [jobovy](https://github.com/jobovy)
* Contact Henry: henrysky.leung [at] mail.utoronto.ca
* You can copy and use this tutorial freely without acknowledging me (Henry Leung), but you should acknowledge the great works and papers from **Yarin Gal (Cambridge University)**
* This tutorial is created on 09/Dec/2017 with Keras 2.1.2, Tensorflow 1.4.0, Nvidia CuDNN 6.1 for CUDA 8.0 (Optional)
* Updated on 16/Mar/2018 with Keras 2.1.5, Tensorflow 1.6.0, Nvidia CuDNN 7.0 for CUDA 9.0 (Optional)

<br>

This tutorial is based on the material, ideas and theory from:
* Paper: [Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning](https://arxiv.org/abs/1506.02142)
* Paper: [What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?](https://arxiv.org/abs/1703.04977)
* Paper: [Bayesian Convolutional Neural Networks with Bernoulli Approximate Variational Inference](https://arxiv.org/abs/1506.02158)
* Yarin Gal's Blog: [What My Deep Model Doesn't Know...](http://mlg.eng.cam.ac.uk/yarin/blog_3d801aa532c1ce.html)
* [Demo from Yarin Gal written in javascript](https://github.com/yaringal/HeteroscedasticDropoutUncertainty)
 
<br>

Other resources:
* If you are doing classification task: [Building a Bayesian deep learning classifier](https://github.com/kyle-dorman/bayesian-neural-network-blogpost)
* If you are doing recurrent neural net: [BayesianRNN](https://github.com/yaringal/BayesianRNN)
* If you interested in industrial application of this method: [Uber](https://eng.uber.com/neural-networks-uncertainty-estimation/)