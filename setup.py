from setuptools import setup, find_packages

setup(
    name='astroNN',
    version='0.0',
    packages=['astroNN', 'astroNN\\apogeetools','astroNN\\gaiatools','astroNN\\NN', 'astroNN\\datasets'],
    include_package_data=True,
    url='https://github.com/henrysky/astroNN/',
    license='MIT',
    author='Henry Leung',
    author_email='henryskyleung@gmail.com',
    description='A python package to do neural network in astronomy using Keras and Tensorflow'
)