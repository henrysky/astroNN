import os.path

from setuptools import setup

setup(
    name='astroNN',
    version='0.8',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Topic:: Scientific / Engineering:: Astronomy'],
    packages=['astroNN',
              os.path.join('astroNN', 'apogee'),
              os.path.join('astroNN', 'gaia'),
              os.path.join('astroNN', 'NN'),
              os.path.join('astroNN', 'datasets'),
              os.path.join('astroNN', 'shared')],
    include_package_data=True,
    package_data={
            '': ['*.npy'],},
    install_requires=[
        'keras','numpy','astropy','h5py','matplotlib', 'astroquery', 'pandas', 'seaborn'],
    extras_require={
        "tensorflow": ["tensorflow>=1.4.0"],
        "tensorflow-gpu": ["tensorflow-gpu>=1.4.0"]},
    url='https://github.com/henrysky/astroNN/',
    license='MIT',
    author='Henry Leung',
    author_email='henrysky.leung@ mail.utoronto.ca',
    description='A python package to do neural network in astronomy using Keras and Tensorflow'
)
