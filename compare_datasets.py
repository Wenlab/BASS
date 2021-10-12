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
from save_load_utils import *

def main(args):

    np.random.seed(args.Seed)

    # load data

    data_null = args.pathData + 'data_condition{}'.format(args.Condition1)
    lengths_null = args.pathData + 'lengths_condition{}'.format(args.Condition1)
    
    data_hyp = args.pathData + 'data_condition{}'.format(args.Condition2)
    lengths_hyp = args.pathData + 'lengths_condition{}'.format(args.Condition2)

    # load GMM 
    means_ = np.load(args.PathGMM + savename + "_means.npy")
    covars_ = np.load(args.PathGMM + savename + "_covars.npy")
    weights_ = np.load(args.PathGMM + savename + "_weights.npy")

    model_fit = GMM_model(len(means_))
    model_fit._read_params(means_,covars_,weights_)

    #Add names for each bout types
    class_names = ['']
    
    lengths_null = lengths_null[:]
    data_null = data_null[:np.sum(lengths_null)]

    lengths_hyp = lengths_hyp[:]
    data_hyp = data_hyp[:np.sum(lengths_hyp)]

    # Load BASS results
    [P_null, n_null, w_dict_null] = reload_raw_results(args, condition=args.Condition1)
    [P_hyp, n_hyp, w_dict_hyp] = reload_raw_results(args, condition=args.Condition2)


    # Format data and set calculation params 
    w_thr = 1e-4
    eps  = 0.1
    p_d  = 0.2
    p_ins = 0.2
    mu = 1.0
    H_beta_fac = 0
    Jthr = 0.15
    Sigma = Yhyp.shape[1]
    std = 0.05
    params = np.array([eps,p_d,p_ins, mu, w_thr,H_beta_fac, Jthr, Sigma, std], dtype =float)

    for i in range(0, len(w_dict_null)):
        w_dict_null[i] = w_dict_null[i].astype('int')

    for i in range(0, len(w_dict_hyp)):
        w_dict_hyp[i] = w_dict_hyp[i].astype('int')

    # perform comparison multiple times
    niter = 10
    w_dict_combined = md.combine_dicts(w_dict_null,w_dict_hyp,params,model_fit)
    m2lnLR_hyp = np.zeros((niter,len(w_dict_combined)))
    empirical_hyp = np.zeros((niter,len(w_dict_combined)))
    expected_hyp = np.zeros((niter,len(w_dict_combined)))

    for n_ in range(niter):
        L = 0.8*len(data_hyp)

        length_per_traj = len(data_hyp)/len(lengths_hyp)
        numtrajs = int(L/length_per_traj)

        sample_lengths = np.random.choice(len(lengths_hyp),numtrajs, replace=False)
        nonsample_lengths = np.delete(np.arange(len(lengths_hyp)),sample_lengths)

        lengths_hyp_train = lengths_hyp[sample_lengths]
        for i,l in enumerate(sample_lengths):
            first = np.sum(lengths_hyp[:l])
            last = np.sum(lengths_hyp[:l+1])
            if i==0:
                data_hyp_train = data_hyp[first:last]
            else:
                data_hyp_train = np.concatenate((data_hyp_train, data_hyp[first:last]))

        lengths_null = lengths_null[:]
        data_null = data_null[:np.sum(lengths_null)]

        Hnull = -model_fit.score(data_null,0)/len(data_null) #entropy
        Ynull = np.exp(model_fit._compute_log_likelihood(data_null) + Hnull)
        #Ynull = data_null

        Hhyp_train =  -model_fit.score(data_hyp_train,0)/len(data_hyp_train)
        Yhyp_train = np.exp(model_fit._compute_log_likelihood(data_hyp_train) + Hhyp_train)
        #Yhyp_train = data_hyp_train

        m2lnLR_hyp,emps_hyp,exps_hyp = cp.compare_datasets(Ynull, lengths_null, Yhyp_train, lengths_hyp_train, w_dict_null, w_dict_hyp, params,model_fit)

        m2lnLR_hyps[n_] = m2lnLR_hyp
        emps_hyps[n_] = emps_hyp
        exps_hyps[n_] = exps_hyp

    #Find the correct threshold to only take significant difference

    Lthr_acid = 15
    filtered_L = np.prod(m2lnLR_hyps > Lthr_acid, axis = 0)
    filtered_num = np.prod(emps_hyps > exps_hyps, axis = 0)
    filtered = filtered_L*filtered_num
    np.set_printoptions(precision = 2)
    filtered_indices = []
    filtered_motifs = []
    filtered_neg_log_P = []
    filtered_empirical_value = []
    filtered_expected_value = []
    for i in range(len(filtered)):
        if filtered[i] == 1 and len(w_dict_combined[i]) > 1:
            motif = [classNamesConvertion[a] for a in w_dict_combined[i]]
            filtered_indices += [i]
            filtered_motifs.append(motif)
            filtered_neg_log_P.append(np.mean(m2nLR_hyp,axis=0)[i])
            filtered_empirical_value.append(np.mean(emps_hyp, axis=0)[i])
            filtered_expected_value.append(np.mean(exps_hyp, axis=0)[i])


    save_path = args.Out + args.Exp
    compare_dict = pd.DataFrame({'Negative log P':filtered_neg_log_P,'Frequency in test':filtered_empirical_value,'Frequency in null':filtered_expected_value, 'Motifs':motifs})
    compare_dict.to_csv(path_or_buf = save_path + '/Comparisons_cond{}vcond{}.csv'.format(args.Condition1,args.Condition2), index = False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-seed','--Seed',help="Seed for data extraction", default=42,type=int)
    parser.add_argument('-condition1','--Condition1', help="First Condition/experiment to use as null hypothesis",default=0 ,type=int)
    parser.add_argument('-condition2','--Condition2', help="Second Condition/experiment to use as the test case", default=1,type=int)
    parser.add_argument('-pathData','--PathData',help="path to data",default='/Users/gautam.sridhar/Documents/Repos/BASS/Data/',type=str)
    parser.add_argument('-pathGMM','--PathGMM', help="path to GMM", default='/Users/gautam.sridhar/Dcouments/Repos/BASS/GMM/', type=str)
    parser.add_argument('-savename','--Savename',help="name of gmm to save/load", default="acid",type=str)
    parser.add_argument('-exp','--Exp',help="name of the experiment to save as", default="pHtaxis",type=str)
    parser.add_argument('-out','--Out',help="path save",default='/Users/gautam.sridhar/Documents/Repos/BASS/Results/',type=str)
    args = parser.parse_args()
    main(args)
