#Author: Gautam Reddy Nallamala. Email: gautam_nallamala@fas.harvard.edu
#Packaged by: Gautam Sridhar. Email: gautam.sridhar@icm_institute.org

#This work is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License. 
#To view a copy of this license, visit http://creativecommons.org/licenses/by-nc-sa/4.0/ or send a letter to Creative Commons, 
#PO Box 1866, Mountain View, CA 94042, USA.

import os
import sys
sys.path.append('./BASS/')
sys.path.append('./Utils/')
import argparse
import bass as md
from GMM import GMM_model
import numpy as np
import pandas as pd
from save_load_utils import *


def main(args):

    np.random.seed(args.Seed)

    # load data

    data = np.load(args.PathData + 'data_condition{}.npy'.format(args.Condition))
    lengths = np.load(args.PathData + 'lengths_condition{}.npy'.format(args.Condition))

    lengths_flat = lengths[:]
    data_flat = data[:np.sum(lengths_flat)]

    # load GMM 
    means_ = np.load(args.PathGMM + args.Savename + "_means.npy")
    covars_ = np.load(args.PathGMM + args.Savename + "_covars.npy")
    weights_ = np.load(args.PathGMM + args.Savename + "_weights.npy")

    model_fit = GMM_model(len(means_))
    model_fit._read_params(means_,covars_,weights_)

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
    Sigma = Y.shape[1]
    std = 0.05
    params = np.array([eps,p_d,p_ins, mu, w_thr,H_beta_fac, Jthr, Sigma, std], dtype =float)

    [P, num_instances, w_dict] = load_results_raw(args, condition=args.Condition)
    w_MLs,words,ls = md.decode(Y, lengths_flat, w_dict, params)
    save_decoded(args, w_MLs, words, ls)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-seed','--Seed',help="Seed for data extraction", default=42,type=int)
    parser.add_argument('-condition','--Condition', help="type of experiment to run/analyze",default=0 ,type=int)
    parser.add_argument('-pathData','--PathData',help="path to data",default='/Users/gautam.sridhar/Documents/Repos/BASS/Data/',type=str)
    parser.add_argument('-pathGMM','--PathGMM', help="path to GMM", default='/Users/gautam.sridhar/Documents/Repos/BASS/GMM/', type=str)
    parser.add_argument('-savename','--Savename',help="name of gmm to save/load", default="acid",type=str)
    parser.add_argument('-exp','--Exp',help="name of the experiment to save as", default="pHtaxis",type=str)
    parser.add_argument('-out','--Out',help="path save",default='/Users/gautam.sridhar/Documents/Repos/BASS/Results/',type=str)
    args = parser.parse_args()
    main(args)
