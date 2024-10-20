# setup.py
from setuptools import setup, find_packages

setup(
    name='my_ml_package',
    version='0.1',
    description='A Python package for a Machine Learning model',
    packages=find_packages(),
    install_requires=[
        'scikit-learn',
        'numpy',
        'joblib',
    ],
)
