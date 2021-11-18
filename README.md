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

## Dataset organization

All datasets are present in the Data/ folder. The filenames have to .npy formats, and the names of the files have to be in the form datasetname_dataset_condition(x).npy. The x stands for a number to indicate multiple conditions of the same experimental setup that you would like to try. For example, we use Condition 0 as the control, Condition 1 as the experiment. You can in addition use many conditions. Each condition dataset should be a matrix of the form nbouts x nfeatures. In addition you will need a lengths file named as datasetname_lengths_condition(x).npy. These are used if your dataset has subsets, like multiple recordings, different trajectories etc. 

For eg. if you had a dataset for the control setting with 10 recordings, datasetname_lengths_condition0.npy will be an array with 10 elements, where each element is the number of bouts in that recording. datasetname_dataset_condition0.npy will have all the bouts in all the recordings as a matrix as mentioned above.


## Code organization

4 scripts are present in the main folder along with 2 jupyter notebooks. The ideal work flow is as follow - 

1. python learn_gmm.py -c 0 1 2 (and so on)    -     The GMM can be learnt on different conditions as the same time.

Use Analyze_GMM notebook now in order to analyze your GMM and the bout types

2. python run_bass.py -c 0                     -     BASS must be run separately on different conditions
3. python compare_datasets.py -cn 0 -ch 1      -     Find the enriched motifs in one condition over another
4. python decode.py -c 0                       -     Label the motifs in your dataset after the dictionary is found

Use Analyze_decoded in order to analyze your recordings and validating the sequences found using compare_datasets.py

For all the scripts, there are multiple options other than -c that can be added. In order to check the options, use python script_name.py --help

### Important:
For each new application, a `soft’ clustering model has to be specified using a GMM.

If instead you have the data as a sequence of cluster labels, i.e., `hard' clustered data, then convert it into a sequence of probability vectors, and define a gmm model with means as the centers of the clusters and the circular standard deviation of 1.0
