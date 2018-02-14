import os.path

from setuptools import setup

setup(
    name='astroNN',
    version='0.9.1',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3 :: Only',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic:: Scientific / Engineering:: Astronomy'],
    packages=['astroNN',
              os.path.join('astroNN', 'apogee'),
              os.path.join('astroNN', 'gaia'),
              os.path.join('astroNN', 'models'),
              os.path.join('astroNN', 'datasets'),
              os.path.join('astroNN', 'shared')],
    include_package_data=True,
    package_data={
        'astroNN': ['data/*.npy', 'data/*.npz']},
    python_requires='>=3.5',
    install_requires=[
        'keras', 'numpy', 'astropy', 'h5py', 'matplotlib', 'astroquery', 'pandas', 'seaborn', 'scikit-learn', 'tqdm'],
    extras_require={
        "tensorflow": ["tensorflow>=1.5.0"],
        "tensorflow-gpu": ["tensorflow-gpu>=1.5.0"]},
    url='https://henrysky.github.io/astroNN/',
    license='MIT',
    author='Henry Leung',
    author_email='henrysky.leung@mail.utoronto.ca',
    description='A python package to do neural network in astronomy using Keras and Tensorflow',
)
