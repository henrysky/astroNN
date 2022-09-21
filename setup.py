import os
import warnings
from packaging import version
from setuptools import setup, find_packages

with open(
    os.path.join(os.path.abspath(os.path.dirname(__file__)), "README.rst"),
    encoding="utf-8",
) as f:
    long_description = f.read()

tf_min_version = "2.9.0"
tfp_min_version = "0.17.0"
python_min_version = "3.7"

setup(
    name="astroNN",
    version="1.1.dev",
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: MIT License",
        f"Programming Language :: Python :: {python_min_version}",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Astronomy",
    ],
    packages=find_packages(),
    include_package_data=True,
    python_requires=f">={python_min_version}",
    install_requires=[
        "numpy",
        "astropy",
        "h5py",
        "matplotlib",
        "astroquery",
        "pandas",
        "scikit-learn",
        "tqdm",
        "packaging",
    ],
    # extra requirement as there are tensorflow-cpu
    extras_require={
        "tensorflow": [f"tensorflow>={tf_min_version}"],
        "tensorflow-probability": [f"tensorflow-probability>={tfp_min_version}"],
    },
    url="https://github.com/henrysky/astroNN",
    project_urls={
        "Bug Tracker": "https://github.com/henrysky/astroNN/issues",
        "Documentation": "http://astronn.readthedocs.io/",
        "Source Code": "https://github.com/henrysky/astroNN",
    },
    license="MIT",
    author="Henry Leung",
    author_email="henrysky.leung@utoronto.ca",
    description="Deep Learning for Astronomers with Tensorflow",
    long_description=long_description,
)

# check if user has tf and tfp installed as they are not strict requirements
try:
    import tensorflow
    if version.parse(tensorflow.__version__) < version.parse(tf_min_version):
        warnings.warn(
            "Your Tensorflow version might be too low for astroNN to work proporly"
        )
except ImportError:
    warnings.warn(
        "Tensorflow not found, please install tensorflow or tensorflow_gpu or tensorflow_cpu manually!"
    )

try:
    import tensorflow_probability
    if version.parse(tensorflow_probability.__version__) < version.parse(tfp_min_version):
        warnings.warn(
            "Your Tensorflow_probability version might be too low for astroNN to work proporly"
        )
except ImportError:
    warnings.warn(
        "tensorflow_probability not found, please install tensorflow_probability manually!"
    )
