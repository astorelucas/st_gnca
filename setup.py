import setuptools
from setuptools import find_packages

setuptools.setup(
    name='GNCA',
    install_requires=[
          'numpy',
          'pandas',
          'torch'
          ],
    packages=find_packages(),
    version='1.0',
    description='GNCA',
    long_description='graph neural network cellular automata for time series forecasting',
    long_description_content_type="text/markdown",
    author='Lucas Astore',
    author_email='astore.lucas@gmail.com',
    url='',
    download_url='',
    keywords=['time series', 'neural cellular automata', 'forecasting'],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.10',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Topic :: Scientific/Engineering',
    
    ]
)