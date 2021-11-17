# BASS
Stand alone package for the algorithm "Behavioral Action Sequence Segmentation". 

This repository contains all the code for the implementation of the motif discovery algorithm BASS from paper ["A lexical approach for identifying behavioral action sequences"](https://www.biorxiv.org/content/10.1101/2020.08.27.270694v1.abstract). 

This repository allows you to run BASS on your own dataset. In order to reproduce the results from the paper please refer to the the [original repository](https://github.com/greddy992/BASS). For a sister repository on running BASS please check [here](https://github.com/oliviermirat/BASSlibrary)

For questions, email gautam_nallamala(AT)fas.harvard.edu and gautam.sridhar(AT)icm-institute.org

## Requirements:
Python3, 
Cython, 
NumPy,
pandas,
matplotlib
SciPy
Sklearn 
editdistance 

To install editdistance:

pip3 install editdistance

## Setup BASS:

Before running any of the code, bass.pyx has to be compiled using:

Navigate to the folder BASS/    then run

python3 setup_bass.py build_ext --inplace

bass.pyx contains the implementation of the motif discovery algorithm, the specification of the mixture model and miscellaneous functions used for the analysis in the paper.  

### Important:
For each new application, a `soft’ clustering model has to be specified. 

If instead you have the data as a sequence of cluster labels, i.e., `hard' clustered data, then convert it into a sequence of probability vectors, and define a gmm model with means as the centers of the clusters and the circular standard deviation of 1.0
