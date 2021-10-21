#Author: Gautam Reddy Nallamala. Email: gautam_nallamala@fas.harvard.edu

#This work is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License. 
#To view a copy of this license, visit http://creativecommons.org/licenses/by-nc-sa/4.0/ or send a letter to Creative Commons, 
#PO Box 1866, Mountain View, CA 94042, USA.

import numpy as np
import scipy.stats as stats
from scipy.stats import norm
from copy import deepcopy


class GMM_synthetic:
    """
    Definition of the model. This is important. 
    You need to define your own P_ygs and sample_y. 
    The current implementation of these two functions is for 
    the synthetic data case presented in the paper.
    """
    def __init__(self,params):
        self.Sigma = int(params[7])
        self.std = float(params[8])
    def _compute_likelihood(self,y,s):
        return P_ygs(y,s,self.Sigma,self.std)
    def _generate_sample_from_state(self,s):
        return sample_y(s,self.Sigma,self.std)


class GMM_model:
    """
    This is our implementation of a GMM used to fit multiple datasets simultaneously (see paper). 
    Not used for the synthetic dataset.
    """
    def __init__(self,numclasses):
        self.numclasses = numclasses
        
    def E_step(self,datasets):
        N = datasets.shape[1]
        numsets = datasets.shape[0]
        gamma_ = np.zeros((numsets,N,self.numclasses))
        for k in range(self.numclasses):
            gamma_[:,:,k] = 1e-20 + self.weights_[:,k][:,np.newaxis]*stats.multivariate_normal.pdf(datasets,mean = self.means_[k],cov = self.covars_[k])
        gamma_ = gamma_/np.sum(gamma_,axis=2)[:,:,np.newaxis]
        return gamma_
    
    def M_step(self,datasets,gamma_):
        for k in range(self.numclasses):
            Nk = np.sum(gamma_[:,:,k])
            self.means_[k] = np.sum(np.sum(gamma_[:,:,k][:,:,None]*datasets,axis=1),axis=0)/Nk
            outerprod = (datasets - self.means_[k])[:,:,:,None]*(datasets - self.means_[k])[:,:,None,:]
            self.covars_[k] = np.sum(np.sum(gamma_[:,:,k][:,:,None,None]*outerprod,axis=1),axis=0)/Nk
            self.weights_ = np.sum(gamma_,axis=1)/self.N
            
    def LL(self,datasets):
        N = datasets.shape[1]
        numsets = datasets.shape[0]
        temp = np.zeros((numsets,N))
        for k in range(self.numclasses):
            temp += self.weights_[:,k][:,None]*stats.multivariate_normal.pdf(datasets,mean = self.means_[k],cov = self.covars_[k])
        LL = np.mean(np.log(temp + 1e-80))
        return -LL
        
    def solve(self,datasets):
        self.numsets= len(datasets)
        self.dim = datasets.shape[2]
        self.N = datasets.shape[1]
        self.means_ = np.zeros((self.numclasses,self.dim))
        self.covars_ = np.zeros((self.numclasses, self.dim,self.dim))
        self.weights_ = np.zeros((self.numsets,self.numclasses))
        
        datasets_flat = np.reshape(datasets,(-1,datasets.shape[2]))
        covar = np.cov(datasets_flat, rowvar = False)
        mean = np.mean(datasets_flat, axis = 0)
        
        numinits = 20
        means_init = np.zeros((numinits,self.numclasses,self.dim))
        covars_init = np.zeros((numinits,self.numclasses,self.dim,self.dim))
        weights_init = np.zeros((numinits,self.numsets,self.numclasses))
        LL_init = np.zeros(numinits)
        for init_ in range(numinits):
            for i in range(self.numclasses):
                means_init[init_][i] = np.random.multivariate_normal(mean,covar)
                covars_init[init_][i] = deepcopy(covar)

            for j in range(self.numsets):
                weights_init[init_][j] = np.random.dirichlet(5*np.ones(self.numclasses))
            self.means_ = means_init[init_]
            self.covars_ = covars_init[init_]
            self.weights_ = weights_init[init_]
            LL_init[init_] = self.LL(datasets)
        best = np.argmin(LL_init)
        self.means_ = means_init[best]
        self.covars_ = covars_init[best]
        self.weights_ = weights_init[best]
            
        LL_curr = self.LL(datasets)
        LL_prev = 0
        print("Initial negative log-likelihood per sample = %.4f" %LL_curr)
        num = 0
        while np.abs(LL_curr - LL_prev) > 1e-4:
            gamma_= self.E_step(datasets)
            self.M_step(datasets,gamma_)
            LL_prev = LL_curr
            LL_curr = self.LL(datasets)
            num += 1
            #print(LL_curr)
        print("Final negative log-likelihood per sample = %.4f" %LL_curr)
        print("Number of iterations = %d" %num)
        
    def _compute_posterior(self,y,set_index):
        post = np.zeros((self.numclasses,y.shape[0]))
        for k in range(self.numclasses):
            post[k] = self.weights_[set_index][k]*self._compute_likelihood(y,k)
        return post/np.sum(post,axis=0)
        
    def _compute_likelihood(self,y,s):
        return stats.multivariate_normal.pdf(y,mean = self.means_[s],cov = self.covars_[s])
    
    def _compute_log_likelihood(self,data):
        Y = np.zeros((len(data),self.numclasses))
        for k in range(self.numclasses):
            Y[:,k] = np.log(stats.multivariate_normal.pdf(data,mean = self.means_[k],cov = self.covars_[k]) + 1e-80)
        return Y
    def score(self,dataset,set_index):
        temp = np.zeros(len(dataset))
        for k in range(self.numclasses):
            temp += self.weights_[set_index,k]*stats.multivariate_normal.pdf(dataset,mean = self.means_[k],cov = self.covars_[k])
        LL = np.sum(np.log(temp + 1e-80))
        return LL
        
    def _generate_sample_from_state(self,s):
        return np.random.multivariate_normal(self.means_[s],self.covars_[s])
    
    def _read_params(self,means_,covars_,weights_):
        self.numclasses = means_.shape[0]
        self.means_ = means_
        self.covars_ = covars_
        self.weights_ = weights_
        
    def _save_params(self,filename):
        np.save(filename + "_means",self.means_)
        np.save(filename + "_covars",self.covars_)
        np.save(filename + "_weights",self.weights_)

