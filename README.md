# BASS
Stand alone package for the algorithm "Behavioral Action Sequence Segmentation". 

This repository contains all the code for the implementation of the motif discovery algorithm BASS from paper ["A lexical approach for identifying behavioral action sequences"](https://www.biorxiv.org/content/10.1101/2020.08.27.270694v1.abstract). 

This repository allows you to run BASS on your own dataset. In order to reproduce the results from the paper please refer to the the [original repository](https://github.com/greddy992/BASS)

For questions, email gautam_nallamala(AT)fas.harvard.edu and gautam.sridhar(AT)icm-institute.org

## Requirements:
Python3, 
Cython, 
NumPy, 
SciPy
Sklearn 
editdistance 

To install editdistance:

pip3 install editdistance

## Setup BASS:

Before running any of the code, bass.pyx has to be compiled using:

python3 setup_bass.py build_ext --inplace 

To run a test case, use:

python3 main_synthetic_data.py 7 50 5 10000 0.0 0.0 0.15 4 0.0 1 &

bass.pyx contains the implementation of the motif discovery algorithm, the specification of the mixture model and miscellaneous functions used for the analysis in the paper.  

main_synthetic_dataset.py is a sample application of the algorithm. This code generates a synthetic dictionary of motifs, a dataset from that dictionary and applies the motif discovery algorithm on the dataset. The output files contain the true dictionary and the one learned by the algorithm along with the probabilities of each motif. 

### Important:
For each new application, a `soft’ clustering model has to be specified. In the code, this is implemented in the GMM_synthetic class in bass.pyx. This class has to be appropriately modified or alternatively a new class should be defined which contains the two functions defined for this class – “_compute_likelihood” and “_generate_sample_from_state”. 

If instead you have the data as a sequence of cluster labels, i.e., `hard' clustered data, then convert it into a sequence of probability vectors, where each probability vector is a unit vector. 

