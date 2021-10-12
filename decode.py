#Author: Gautam Reddy Nallamala. Email: gautam_nallamala@fas.harvard.edu
#Packaged by: Gautam Sridhar. Email: gautam.sridhar@icm_institute.org

#This work is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License. 
#To view a copy of this license, visit http://creativecommons.org/licenses/by-nc-sa/4.0/ or send a letter to Creative Commons, 
#PO Box 1866, Mountain View, CA 94042, USA.

import bass as md
import numpy as np
import pandas as pd
import os
import sys
from save_load_utils import *


def main(args):

    np.random.seed(args.Seed)

    # load data

    data = args.pathData + 'data_condition{}'.format(args.Condition)
    lengths = args.pathData + 'lengths_condition{}'.format(args.Condition)

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

    [P, num_instances, w_dict] = reload_raw_results(args, condition=args.Condition)
    `
    w_MLs,words,ls = md.decode(Y, lengths_flat, w_dict, params)
    save_decoded(args, w_MLs, words, ls)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-seed','--Seed',help="Seed for data extraction", default=42,type=int)
    parser.add_argument('-condition','--Condition', help="type of experiment to run/analyze",default=0 ,type=int)
    parser.add_argument('-pathData','--PathData',help="path to data",default='/Users/gautam.sridhar/Documents/Repos/BASS/Data/',type=str)
    parser.add_argument('-exp','--Exp',help="name of the experiment to save as", default="pHtaxis",type=str)
    parser.add_argument('-out','--Out',help="path save",default='/Users/gautam.sridhar/Documents/Repos/BASS/Results/',type=str)
    args = parser.parse_args()
    main(args)
