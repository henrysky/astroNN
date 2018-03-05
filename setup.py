import os.path

from setuptools import setup, find_packages

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
    packages=find_packages(),
    include_package_data=True,
    package_data={
        'astroNN': ['data/*.npy', 'data/*.npz']},
    python_requires='>=3.5',
    install_requires=[
        'numpy', 'astropy', 'h5py', 'matplotlib', 'astroquery', 'pandas', 'seaborn', 'scikit-learn', 'tqdm'],
    extras_require={
        "keras": ["keras>=2.1.3"],
        "tensorflow": ["tensorflow>=1.5.0"],
        "tensorflow-gpu": ["tensorflow-gpu>=1.5.0"]},
    url='https://github.com/henrysky/astroNN',
    project_urls={
        "Bug Tracker": "https://github.com/henrysky/astroNN/issues",
        "Documentation": "http://astronn.readthedocs.io/en/latest/",
        "Source Code": "https://github.com/henrysky/astroNN",
    },
    license='MIT',
    author='Henry Leung',
    author_email='henrysky.leung@mail.utoronto.ca',
    description='A python package to do neural network in astronomy using Keras and Tensorflow',
)
