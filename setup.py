import os
from setuptools import setup, find_packages

with open(
    os.path.join(os.path.abspath(os.path.dirname(__file__)), "README.rst"),
    encoding="utf-8",
) as f:
    long_description = f.read()

torch_min_version = "2.1.0"
python_min_version = "3.9"

setup(
    name="astroNN",
    version="1.2.dev0",
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
        f"torch>={torch_min_version}",
    ],
    url="https://github.com/henrysky/astroNN",
    project_urls={
        "Bug Tracker": "https://github.com/henrysky/astroNN/issues",
        "Documentation": "http://astronn.readthedocs.io/",
        "Source Code": "https://github.com/henrysky/astroNN",
    },
    license="MIT",
    author="Henry Leung",
    author_email="henrysky.leung@utoronto.ca",
    description="Deep Learning for Astronomers with Keras",
    long_description=long_description,
    long_description_content_type="text/x-rst",
)
