from setuptools import setup, find_packages
import os, os.path

setup(
    name='astroNN',
    version='0.3',
    packages=['astroNN', 
              os.path.join('astroNN','apogeetools'),
              os.path.join('astroNN','gaiatools'),
              os.path.join('astroNN','NN'),
              os.path.join('astroNN','datasets')],
    include_package_data=True,
    url='https://github.com/henrysky/astroNN/',
    license='MIT',
    author='Henry Leung',
    author_email='henryskyleung@gmail.com',
    description='A python package to do neural network in astronomy using Keras and Tensorflow'
)
