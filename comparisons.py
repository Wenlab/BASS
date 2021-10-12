#The functions below are for the various tests and comparisons done in the paper. 
#Author: Gautam Reddy Nallamala. Email: gautam_nallamala@fas.harvard.edu

#This work is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License. 
#To view a copy of this license, visit http://creativecommons.org/licenses/by-nc-sa/4.0/ or send a letter to Creative Commons, 
#PO Box 1866, Mountain View, CA 94042, USA.

import bass as md
import numpy as np
import scipy.stats as stats


#Implementation of the forward backward algorithm to compute the transition matrix for the HMM.         
def compute_transmat(Y):
    """
    Initialize transition matrices
    """
    numclasses = Y.shape[1]
    transmat_ = np.zeros((numclasses,numclasses))
    stationary_probs_ = np.random.dirichlet(5*np.ones(numclasses))
    for k in range(numclasses):
        transmat_[k] = np.random.dirichlet(5*np.ones(numclasses))
    for k in range(numclasses):
        for iter_ in range(50):
            stationary_probs_ = np.einsum('i,ij', stationary_probs_, transmat_)
            
    numiter = 100
    LLs = np.zeros(numiter)
    
    #compute alphas
    for iter_ in range(numiter):
        N = Y.shape[0]
        alphas = np.zeros((N,numclasses))
        norms = np.zeros(N)
        for i in range(N):
            if i == 0:
                alphas[i] = stationary_probs_*Y[i] + 1e-12
            else:  
                alphas[i] = np.einsum('i,ij',alphas[i-1],transmat_)*Y[i] + 1e-12
            norms[i] = np.sum(alphas[i])
            alphas[i] /= norms[i]

        
        LLs[iter_] = -np.sum(np.log(norms + 1e-20))/N
        #print(iter_,LLs[iter_])
        if iter_ > 5 and np.abs(LLs[iter_] - LLs[iter_-1]) < 1e-4:
            break
        #compute betas
        betas = np.zeros((N,numclasses))
        for i in range(N-1,-1,-1):
            if i == N-1:
                betas[i] = 1.0 + 1e-12
            else:  
                betas[i] = np.einsum('j,ij,j', betas[i+1], transmat_,Y[i+1]) + 1e-12
            betas[i] /= np.sum(betas[i])

        #compute states
        gamma_ = np.zeros((N,numclasses))
        for i in range(N):
            gamma_[i] = alphas[i]*betas[i]/np.sum(alphas[i]*betas[i])

        #compute transitions:
        xi_ = np.zeros((N,numclasses,numclasses))
        for j in range(numclasses):
            for k in range(numclasses):
                xi_[1:,j,k] = alphas[:-1,j]*transmat_[j,k]*Y[1:,k]*betas[1:,k]
        xi_ /= np.sum(xi_,axis=(1,2))[:,None,None]

        #update transmat_
        for k in range(numclasses):
            transmat_[:,k] = np.sum(xi_[1:,:,k],axis=0)/np.sum(gamma_[:-1],axis=0)
        stationary_probs_ = gamma_[0]
        for k in range(numclasses):
            transmat_[k] /= np.sum(transmat_[k])
            
        for k in range(numclasses):
            for iter_ in range(50):
                stationary_probs_ = np.einsum('i,ij', stationary_probs_, transmat_)
    
    return transmat_,stationary_probs_

def test_for_markovianity(Y,w_dict,eps,p_d,transmat_, stationary_probs_):
    """
    Main function used to test markovianity of each motif. 
    """
    lengths = [len(w) for w in w_dict]
    lmean = np.mean(lengths)
    mlnPs = np.zeros(len(w_dict))
    emps =  np.zeros(len(w_dict))
    exps =  np.zeros(len(w_dict))
    for i,w in enumerate(w_dict):
        seqs,probs = get_mutated_sequences_prob(list(w),eps,p_d)
        emp = 0
        exp = 0
        for j,seq in enumerate(seqs):
            seq_arr = np.array(seq,dtype = int)
            #print(w,seq_arr,probs[i])
            emp += md.calculate_empirical_frequency_hmm(seq_arr,Y,transmat_, stationary_probs_)*probs[j]
            exp += md.calculate_expected_frequency_hmm(seq_arr,transmat_, stationary_probs_)*probs[j]

        q1 = 1 + (1.0/exp + 1.0/(1-exp) - 1)/(6.0*len(Y)) #correction to LR test
        ll = 2*len(Y)*(emp*np.log(emp/exp) + (1-emp)*np.log((1-emp)/(1-exp)))/q1
        mlnP = -np.log10(stats.chi2.sf(ll,1))
        mlnPs[i] = mlnP
        emps[i] = emp
        exps[i] = exp
        #print("%04d %04d %2.2f"%(int(emp*len(Y)),int(exp*len(Y)),mlnP),w)
    sorted_ = np.argsort(-mlnPs)
    for w in sorted_:
        if emps[w] > exps[w] and 10**(-mlnPs[w]) < 1e-3:
            print("%04d %04d %2.2f"%(int(emps[w]*len(Y)),int(exps[w]*len(Y)),mlnPs[w]),w_dict[w])
    return mlnPs,emps,exps

def print_dict(Y,w_dict,P_w):
    """
    Print dictionary
    """
    sorted_ = np.argsort(-P_w)
    lengths = [len(w) for w in w_dict]
    lmean = np.mean(lengths)
    for i in sorted_[:]:
        print("%.4f %d"%(P_w[i],int(P_w[i]*len(Y)/lmean)),w_dict[i])

def combine_dicts(w_dict1, w_dict2, params, model):
    """
    Combine two dictionaries:
    """
    w_dict = w_dict1 + w_dict2
    eps = params[0]
    params[0] = 0
    P_w = []
    w_dict = remove_duplicates_w_dict(P_w,w_dict,params,model)
    return w_dict

def compare_datasets(Y1, lengths_Y1, Y2, lengths_Y2,  w_dict1, w_dict2, params,model):
    """
    Compare the number of occurrences of each motif in two datasets. The model of course has to be the same for both datasets.     
    """
    w_dict = combine_dicts(w_dict1,w_dict2,params,model)
    P_w1 = get_P_w(Y1,lengths_Y1,w_dict,params)
    P_w2 = get_P_w(Y2,lengths_Y2,w_dict,params)
    lengths = [len(w) for w in w_dict]
    lmean = np.sum(P_w2*lengths)
    N_av2 = len(Y2)/lmean
    scores = np.zeros(len(w_dict))
    print(len(w_dict), len(w_dict1), len(w_dict2))
    emps = np.zeros(len(w_dict))
    exps = np.zeros(len(w_dict))
    for w in range(len(w_dict)):
        f_calc = P_w2[w]
        f_exp =  P_w1[w]
        q1 = 1 + (1.0/f_exp + 1.0/(1-f_exp) - 1)/(6.0*N_av2) #correction to LR test
        m2lnLR = 2*N_av2*(f_calc*np.log(f_calc/f_exp) + (1-f_calc)*np.log((1-f_calc)/(1-f_exp)))
        scores[w] = -np.log10(stats.chi2.sf(m2lnLR,1))
        emps[w] = f_calc*N_av2
        exps[w] = f_exp*N_av2
        
    sorted_ = np.argsort(-scores)
    for w in sorted_:
        if P_w2[w] > P_w1[w] and N_av2*P_w2[w] > 10 and len(w_dict[w]) > 1 and 10**(-scores[w]) < 1e-2 and N_av2*P_w1[w] > 5:
            print( "%04d %04d %.2f" %(int(N_av2*P_w1[w]),int(N_av2*P_w2[w]), scores[w]),w_dict[w])
    return scores,emps,exps

