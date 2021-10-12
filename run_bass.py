#Author: Gautam Reddy Nallamala. Email: gautam_nallamala@fas.harvard.edu
#Packaged by: Gautam Sridhar. Email: gautam.sridhar@icm_institute.org

#This work is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License. 
#To view a copy of this license, visit http://creativecommons.org/licenses/by-nc-sa/4.0/ or send a letter to Creative Commons, 
#PO Box 1866, Mountain View, CA 94042, USA.

import bass as md
import comparisons as cp
import numpy as np
import pandas as pd
import os
import sys
from GMM import GMM_model
from save_load_utils import *


def main(args):

    np.random.seed(args.Seed)

    # load data

    data = args.pathData + 'data_condition{}'.format(args.Condition)
    lengths = args.pathData + 'lengths_condition{}'.format(args.Condition)

    # load GMM 
    means_ = np.load(args.PathGMM + savename + "_means.npy")
    covars_ = np.load(args.PathGMM + savename + "_covars.npy")
    weights_ = np.load(args.PathGMM + savename + "_weights.npy")

    model_fit = GMM_model(len(means_))
    model_fit._read_params(means_,covars_,weights_)

    #Add names for each bout types
    class_names = ['']
    
    lengths_flat = lengths[:]
    data_flat = data[:np.sum(lengths_flat)]

    # Format data and set calculation params
    H = -model_fit.score(data_flat,args.Condition)/len(data_flat) #entropy
    Y = np.exp(model_fit._compute_log_likelihood(data_flat))/np.exp(-H)

    w_thr = 1e-4
    eps  = 0.1
    p_d  = 0.2
    p_ins = 0.2
    mu = 1.0
    H_beta_fac = 0
    Jthr = 0.15
    Sigma = YdbsharppH.shape[1]
    std = 0.05
    params = np.array([eps,p_d,p_ins, mu, w_thr,H_beta_fac, Jthr, Sigma, std], dtype =float)

    # Solve for dictionary
    P_dict, w_dict = md.solve_dictionary(Y,lengths_flat,params,model_fit,7)
    
    P_dict_sorted = []
    w_dict_sorted = []
    num_instances = []

    idx = np.argsort(-P_dict)
    motif_lengths = [len(w) for w in w_dict]
    lmean = np.mean(motif_lengths)
    for i in idx[:]:
        P_dict_sorted.append(P_dict[i])
        num_instances.append(int(P_dict[i] * len(Y)/lmean))
        w_dict_sorted.append(w_dict[i])

    save_results_raw(args, P_dict_sorted, num_instances, w_dict_sorted)
    save_results_classnames(args, P_dict_sorted, num_instances, w_dict_sorted, class_names)

    transmat_, stationary_probs_ = md.compute_transmat(Y)
    neg_log_p, empirical_freq, expected_freq = cp.test_for_markovianity(Y,w_dict,eps,p_d,transmat_, stationary_probs_)

    save_markovianity(args, w_dict, neg_log_p, empirical_freq, expected_freq, class_names)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-seed','--Seed',help="Seed for data extraction", default=42,type=int)
    parser.add_argument('-condition','--Condition', help="Condition/experiment to run BASS on",default=0 ,type=int)
    parser.add_argument('-pathData','--PathData',help="path to data",default='/Users/gautam.sridhar/Documents/Repos/BASS/Data/',type=str)
    parser.add_argument('-pathGMM','--PathGMM', help="path to GMM", default='/Users/gautam.sridhar/Dcouments/Repos/BASS/GMM/', type=str)
    parser.add_argument('-savename','--Savename',help="name of gmm to save/load", default="acid",type=str)
    parser.add_argument('-exp','--Exp',help="name of the experiment to save as", default="pHtaxis",type=str)
    parser.add_argument('-out','--Out',help="path save",default='/Users/gautam.sridhar/Documents/Repos/BASS/Results/',type=str)
    args = parser.parse_args()
    main(args)
