#Author: Gautam Reddy Nallamala. Email: gautam_nallamala@fas.harvard.edu

#This work is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License. 
#To view a copy of this license, visit http://creativecommons.org/licenses/by-nc-sa/4.0/ or send a letter to Creative Commons, 
#PO Box 1866, Mountain View, CA 94042, USA.

import numpy as np
cimport numpy as np
from libc.math cimport log
from libc.math cimport exp
from libc.math cimport ceil
from libc.math cimport pow
from libc.math cimport sqrt
from libc.math cimport cos
from libc.math cimport sin
from scipy.optimize import minimize
import scipy.stats as stats
from scipy.special import comb
from scipy.stats import poisson
import cython
import sys
from scipy.stats import norm
import editdistance
from copy import deepcopy

# cython: profile=True

@cython.cdivision(True)
cpdef double P_ygs(double [:] y, int s, int numclasses, double std):
    """
    P_ygs is the character likelihood function q(y|.) in the paper. Outputs the probability of observing y given s. 
    This is defined for the GMM_synthetic model. 
    This function specifies the emission probabilities of the mixture model. HAS to be defined.
    """
    cdef int dim = 2
    cdef double means0 = 0
    cdef double means1 = 0
    cdef double PI = 3.1415926
    if s < numclasses - 1 and numclasses > 1:
        means0 =  cos(PI*2*s/(numclasses-1))
        means1 =  sin(PI*2*s/(numclasses-1))
    cdef double prob1 =  exp(-(y[0] - means0)*(y[0] - means0)/(2*std*std))/sqrt(2*PI*std*std)
    cdef double prob2 =  exp(-(y[1] - means1)*(y[1] - means1)/(2*std*std))/sqrt(2*PI*std*std)
    return prob1*prob2


@cython.cdivision(True)
cpdef double [:] sample_y(int s, int numclasses, double std):
    """
    Sample from the mixture model. HAS to be defined.
    """ 
    cdef int dim = 2
    cdef double means0 = 0
    cdef double means1 = 0
    cdef double PI = 3.1415926
    if s < numclasses - 1:
        means0 =  cos(PI*2*s/(numclasses-1))
        means1 =  sin(PI*2*s/(numclasses-1))
        
    cdef double [:] Y = np.zeros(dim,dtype = float)
    Y[0] = means0 + std*np.random.randn()
    Y[1] = means1 + std*np.random.randn()
    return Y


cpdef double P_YgS_func_noeps(double [:,:] Y,  np.int_t[:] S):
    """
    Q(Y|m) with no action pattern noise.
    """     
    cdef int lenY = Y.shape[0]
    cdef int lenS = S.shape[0]
    cdef int i
    cdef double prod = 1
    if lenY != lenS:
        return 0
    else:
        for i in range(lenS):
            prod = prod*Y[i,S[i]]
        return prod


@cython.cdivision(True)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cpdef double P_YgS_func(double[:,:] Y, np.int_t [:] S, double [:] params, double[:,:] P_memo):
    """
    This is the implementation of Q(Y|m) using the recursive formula from the paper.
    """
    cdef int i,j
    cdef double eps,p_d,p_ins
    eps = params[0]
    p_d = params[1]
    p_ins = params[2]
    cdef int Ly = Y.shape[0]
    cdef int Ls = S.shape[0]
    cdef double P = 0
    if Ly == 0:
        return pow(eps*p_d,Ls)
    elif Ls == 0 or Ly < 0:
        return 0.0
    elif Ly > 2*Ls:
        return 0.0       
    elif P_memo[Ly-1,Ls-1] > -1:
        return P_memo[Ly-1,Ls-1]
    else:
        for i in range(min(3,Ly+1)):
            if i == 0 and eps*p_d > 1e-3:
                P += eps*p_d*P_YgS_func(Y[:],S[:Ls-1],params,P_memo)
            elif i == 1:
                P += (1-eps)*P_YgS_func(Y[:Ly-1],S[:Ls-1],params,P_memo)*Y[Ly-1,S[Ls-1]]
            elif i == 2 and eps*(1-p_d) > 1e-3: 
                P += eps*(1-p_d)*P_YgS_func(Y[:Ly-2],S[:Ls-1],params,P_memo)*Y[Ly-1,S[Ls-1]]*Y[Ly-2,S[Ls-1]]
    #Can insert further cases here if more than insertion is to be allowed
    P_memo[Ly-1,Ls-1] = P
    return P


@cython.cdivision(True)
cpdef double P_YgS(double [:,:] Y,  np.int_t[:] S, double [:] params):
    cdef int lenY = Y.shape[0]
    cdef int lenS = S.shape[0]
    if params[0] < 1e-3 or lenS == 1:
        return P_YgS_func_noeps(Y,S)
    cdef double [:,::1] P_memo = -5*np.ones((lenY,lenS),dtype = float)
    return P_YgS_func(Y,S,params,P_memo)/(1.0 - pow(params[0]*params[1],lenS)) #divide by the possibility of S -> null.


def get_lmax(w_dict,params):
    """
    Maximum length of motif instantiations.
    """
    eps = params[0]
    p_d = params[1]
    p_ins = params[2]
    lengths = [len(w) for w in w_dict]
    return 40  #we set 20 as the length of maximal motif so 40 is the upper limit on the length of a motif instantiation with one insertion per symbol. 


def get_lmin_and_max(w,params):
    """
    This function is part of the optimization of the code. 
    Gives you the minimum and maximum lengths a motif can mutate to, based on the p_d and e_p.
    """
    l = len(w)
    eps = params[0]
    p_d = params[1]
    p_ins = params[2]
    lmin = 0
    lmax = 2*l
    for k in range(l):
        if comb(l,k)*pow(eps*p_d,k)*pow(1-eps,l-k) < 5e-3:
            lmin = l - k - 1
            break
    for k in range(l + 1):
        if comb(l,k)*pow(eps*(1-p_d),k)*pow(1-eps,l-k) < 5e-3:
            lmax = l + k
            break
    return lmin,lmax
    

def append_mseq(sym,seqs):
    """
    Append sequences.
    """
    mseqs = []
    for i in range(len(seqs)):
        mseqs += [[sym] + seqs[i]]
    return mseqs


def generate_mutated_sequences(seq,eps,p_d):
    """
    Generate mutated sequences from a motif template m, drawn from P(\tilde{m}|m). 
    """
    mseqs = []
    probs_mseqs = []
    if len(seq) == 1:
        return [[seq[0]],[],[seq[0],seq[0]]],[1-eps,eps*p_d,eps*(1-p_d)]
    
    seqs,probs = generate_mutated_sequences(seq[1:],eps,p_d)
       
    mseqs += append_mseq(seq[0],seqs)
    probs_mseqs += [p*(1-eps) for p in probs]
    
    mseqs += seqs
    probs_mseqs += [p*eps*p_d for p in probs]
    
    seqs_dup = append_mseq(seq[0],seqs)
    seqs_dup = append_mseq(seq[0],seqs_dup)
    mseqs += seqs_dup
    probs_mseqs += [p*eps*(1-p_d) for p in probs]
    
    dups = []
    for i in range(len(mseqs)):
        for j in range(i+1,len(mseqs)):
            if np.array_equal(mseqs[i],mseqs[j]):
                dups += [j]
                probs_mseqs[i] += probs_mseqs[j]
                
    dups = np.array(dups)
    dups = np.unique(dups)                
    #print(dups)
    for j in range(len(dups)):
        del mseqs[dups[j]]
        del probs_mseqs[dups[j]]
        dups -= 1
        
    seqs_out = []
    probs_out = []
    for i in range(len(mseqs)):
        if probs_mseqs[i] > 5e-4:
            seqs_out += [mseqs[i]]
            probs_out += [probs_mseqs[i]]
            
    return seqs_out,probs_out


def get_mutated_sequences_prob(seq,eps,p_d):
    """
    Compute P(\tilde{m}|m) i.e., the probability of a mutated sequence given the motif template.
    """
    if len(seq) == 1:
        return [seq],[1.0]
    else:
        seqs,probs = generate_mutated_sequences(seq,eps,p_d)
        empty_prob = 0
        for i in range(len(seqs)):
            if len(seqs[i]) == 0:
                empty_prob = probs[i]
                del seqs[i]
                del probs[i]
                break
            
        for i in range(len(seqs)):
            probs[i] /= (1-empty_prob)
        return seqs,probs


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def get_W_ils(w_dict, double[:,:] Y, np.int_t[:] lengths_Y, double [:] params):
    """
    This function computes the locations (index i) and lengths (index l) where a particular motif (index w) could possibly fit 
    based on a threshold w_thr_l on Q(Y_{i-l+1:i}|w). 
    Computing this is the rate-limiting step. 
    The MLE to get p_m is much more efficient if W_il is pre-computed and then optimization performed.
    """
    cdef int L = Y.shape[0]
    cdef int D = len(w_dict)
    cdef int K = lengths_Y.shape[0]
    cdef int lmax = get_lmax(w_dict,params)
    
    Q_gw = np.zeros((L,lmax), dtype=float) 
    cdef double[:,:] Q_gw_view = Q_gw
    
    cdef np.int_t[:] w_dict_view
    
    cdef int i,w,k,j,lmin,l,kk,mlen,wtild
    cdef double [:] Pmtilds
    cdef np.int_t [:] seq_mtild
    cdef double w_thr = params[4]
    cdef double w_thr_l = 0
    W_ils = []
    
    for w in range(D):
        W_ils += [[]]
        
        w_dict_view = w_dict[w]
        lmin,lmax = get_lmin_and_max(w_dict[w],params)
        lw = w_dict[w].shape[0]
        seqs,probs = get_mutated_sequences_prob(w_dict_view, params[0],params[1])
        Pmtilds = np.array(probs,dtype = float)
        mlen = Pmtilds.shape[0]
        
        Q_gw_view = np.zeros(Q_gw.shape, dtype=float) 
        for wtild in range(mlen):
            seq_mtild = np.array(seqs[wtild],dtype = int)
            l = seq_mtild.shape[0]-1
            kk = 0
            for k in range(K):
                L = lengths_Y[k]
                for i in range(l,L):
                    Q_gw_view[i+kk,l] += P_YgS_func_noeps(Y[i+kk-l:i+kk+1],seq_mtild)*Pmtilds[wtild]
                kk += L
        
        L = Y.shape[0]
        for i in range(L):
            for l in range(min(i+1,lmax)):
                w_thr_l = pow(w_thr,l+1)
                if Q_gw_view[i,l] > w_thr_l:
                    W_ils[w] += [[i,l,Q_gw_view[i,l]]] 
    return W_ils


def get_Q_gw_from_W_il(W_ils_w,Y,w_dict,params):
    """
    Compute the likelihood of a particular motif along the dataset.
    """
    L = Y.shape[0]
    lmax = get_lmax(w_dict,params)
    Q_gw_w = np.zeros((L,lmax))
    for W_il in W_ils_w:
        Q_gw_w[<int> W_il[0],<int> W_il[1]] = W_il[2]
    return Q_gw_w


def get_Q_gw_i(double [:,:] Yseq, w_dict, double [:] params):
    """
    Compute the likelihood of all motifs at a particular locus on the dataset.
    """
    cdef int L = Yseq.shape[0]
    cdef int lmax = get_lmax(w_dict,params)
    cdef int D = len(w_dict)
    Q_gw_i = np.zeros((lmax,D),dtype = float)
    cdef double [:,:] Q_gw_i_view = Q_gw_i
    cdef int l,w
    for l in range(min(L,lmax)):
        for w in range(D):
            Q_gw_i_view[l,w] = P_YgS(Yseq[-l-1:],w_dict[w],params)
    return Q_gw_i


def evaluate_Q(double[:] P_w, W_ils, int L, int lmax, int D):
    """
    Evaluate the marginal probability Q(Y_{i-l+1:i}) at each locus i and length l.
    """
    Q = np.zeros((L,lmax), dtype=float)
    cdef double[:,:] Q_view = Q
    cdef int i,l,w,w_il_length
    
    for w in range(D):
        for W_il in W_ils[w]:
            i = <int> W_il[0]
            l = <int> W_il[1]
            Q_view[i,l] += P_w[w]*W_il[2]
    return Q


def evaluate_R(double[:,:] Q):
    """
    Evaluate R defined in the paper.
    """
    cdef int L = Q.shape[0]
    cdef int i,l,lmax
    lmax = Q.shape[1]
    R = np.zeros(L, dtype = float)
    cdef double[:] R_view = R
    cdef double prod = 1
    for i in range(L):
        R_view[i] = Q[i,0] + 1e-10
        prod = 1
        for l in range(1,min(lmax,i+1)):
            prod *= R_view[i-l]
            R_view[i] += Q[i,l]/prod
    return R


def evaluate_R1(double[:,:] Q):
    """
    Evaluate R' defined in the paper.
    """
    cdef int L = Q.shape[0]
    cdef int i,l,lmax
    lmax = Q.shape[1]
    R1 = np.zeros(L, dtype = float)
    cdef double[:] R1_view = R1
    cdef double prod = 1
    for i in range(L-1,-1,-1):
        R1_view[i] = Q[i,0] + 1e-10
        prod = 1
        for l in range(1,min(lmax,L-i)):
            prod *= R1_view[i+l]
            R1_view[i] += Q[i+l,l]/prod
    return R1


def evaluate_G(double [:] R,double [:] R1,int lmax):
    """
    Evaluate G defined in the paper.
    """
    cdef int L = R.shape[0]
    G = np.zeros((L,lmax), dtype= float)
    cdef double[:,:] G_view = G
    cdef double prod = 1
    cdef double prod2 = 1
    cdef int i,l
    for i in range(L):
        prod *= R[i]/R1[i]
        prod2 = 1
        for l in range(min(lmax,i+1)):
            prod2 *= R[i-l]
            G_view[i,l] = prod/prod2
    return G


def evaluate_F(R):
    return -np.sum(np.log(R))


def evaluate_dF(double[:] P_w, double[:,:] G, W_ils):
    """
    Evaluate gradient of free energy.
    """
    cdef int D = P_w.shape[0]
    
    dF = np.zeros(D, dtype = float)
    
    cdef double [::1] dF_view = dF
    
    cdef int i,l,w
    
    for w in range(D):
        for W_il in W_ils[w]:
            i = <int> W_il[0]
            l = <int> W_il[1]
            dF_view[w] += -G[i,l]*W_il[2] 
    return dF


def evaluate_F_seq(Y,lengths_Y, P_w, w_dict, params):
    """
    Evaluate free energy for a particular sequence.
    """
    lmax = get_lmax(w_dict,params)
    L = Y.shape[0]
    D = len(w_dict)
    
    W_ils = get_W_ils(w_dict, Y,lengths_Y, params)
    Q = evaluate_Q(P_w,W_ils,L,lmax,D)
    R = evaluate_R(Q)
    F = evaluate_F(R)
    return F


cpdef double subroutine_Nw1w2(np.int_t[:] w1w2, double [:,:] G, np.int_t [:] nz_pos, double [:,:] Q_gw1, double [:,:] Q_gw2, int l1,int l2, double [:] params):
    """
    This is used in dictionary expansion. 
    Compute the number of times the concatenated motif w1w2 occurs in the dataset upto a pre-factor.
    """
    cdef int L = G.shape[0]
    cdef double Nw1w2 = 0
    cdef double eps = params[0]
    cdef double p_d = params[1]
    cdef double temp
    cdef int lenw1w2 = w1w2.shape[0]
    if lenw1w2 == 0:
        return 0.0
    cdef int i,l,lmin,lmax,k
    lmin,lmax = get_lmin_and_max(w1w2,params)
    for i in nz_pos:
        for l in range(lmin,min(lmax,i+1)):
            temp = 0
            temp += Q_gw2[i,l]*pow(eps*p_d,l1)
            temp += Q_gw1[i,l]*pow(eps*p_d,l2)
            for k in range(l):
                temp += Q_gw2[i,k]*Q_gw1[i-k-1,l-k-1]
            Nw1w2 += temp*G[i,l]
    return Nw1w2


def sample(S,params,model):
    """
    This is used to sample an output sequence Y given a motif template S.
    """
    eps = params[0]
    p_d = params[1]
    p_ins = params[2]
    Y = []
    while len(Y) == 0:
        if len(S) == 1:
            Y += [model._generate_sample_from_state(S[0])]
        else:
            for s in S:
                randu = np.random.uniform()
                if randu < 1-eps:
                    Y += [model._generate_sample_from_state(s)]
                elif randu < 1-eps + eps*(1-p_d):
                    numy = 2#np.random.geometric(1-p_ins)
                    for j in range(numy):
                        Y += [model._generate_sample_from_state(s)]
    return Y


def convert_Y_to_Y_Ps(Y,params,model):
    """
    Convert a data vector Y to a sequence of probability vectors q(Y_i|c_j).
    """
    Sigma = int(params[7])
    Y_Ps = np.zeros((len(Y),Sigma),dtype = float)
    for i in range(len(Y)):
        for s in range(Sigma):
            Y_Ps[i,s] = model._compute_likelihood(Y[i],s)
    return Y_Ps


@cython.cdivision(True)
def get_JS(w_dict,double [:] params, model):
    """
    Compute the Jensen-Shannon divergence matrix between all pairs of motifs in the dictionary. 
    This is where the editdistance library is used i.e.,  
    in order to restrict the computation to somewhat close motifs since most motifs are separated by the maximal distance 1.
    This takes annoyingly long to compute.
    """
    cdef int niter = 500
    cdef int D = len(w_dict)
    cdef double [:,:] dij = np.zeros((D,D),dtype = float)
    cdef double [:,:] dijT = np.zeros((D,D),dtype = float)
    cdef double [:,:] JS = np.zeros((D,D),dtype = float)
    
    eps = params[0]
    cdef np.int_t [:,:] editdist = np.zeros((D,D),dtype = int)
    cdef int di,dj,i
    cdef double dist
    for di in range(D):
        for dj in range(D):
            str1 = ''.join(str(e) for e in w_dict[di])
            str2 = ''.join(str(e) for e in w_dict[dj])
            dist = editdistance.eval(str1, str2)
            if dist > 0.0 and pow(eps,dist)*pow(1-eps,len(str1)-dist) < 1e-4:
                editdist[di][dj] = 1
                
    cdef double YiSi,YiSj
    cdef double [:,:] Ydi
    for di in range(D):
        for i in range(niter):
            sample_di = np.array(sample(w_dict[di],params,model),dtype = float)
            Ydi = convert_Y_to_Y_Ps(sample_di,params,model)
            YiSi = P_YgS(Ydi,w_dict[di],params)
            for dj in range(D):
                if w_dict[di].shape[0] == 1 or w_dict[dj].shape[0] == 1 or editdist[di][dj]:
                    dij[di,dj] = log(2)
                else:
                    YiSj = P_YgS(Ydi, w_dict[dj],params)
                    dij[di,dj] += (log(YiSi) - log(0.5*YiSi + 0.5*YiSj))/niter
    for di in range(D):
        for dj in range(D):
            JS[di][dj] = (dij[di,dj] + dij[dj,di])/(2*log(2))
    return JS


cpdef double P_YgHMM(double[:,:] Y, double[:,:] transmat_hmm, double[:] rho):
    """
    This is used for the Markovianity test. Implements the forward algorithm for HMMs.
    """
    cdef int l = Y.shape[0]
    cdef int Sigma = Y.shape[1]
    alphas = np.zeros((l,Sigma),dtype = float)
    cdef double [:,:] alphas_view = alphas
    cdef double output = 0
    cdef int i,s,sp
    for i in range(l):
        for s in range(Sigma):
            if i == 0:
                alphas_view[i,s] = Y[i,s]*rho[s]
            else:
                for sp in range(Sigma):
                    alphas_view[i,s] += alphas_view[i-1,sp]*transmat_hmm[sp,s]
                alphas_view[i,s] *= Y[i,s]
    for s in range(Sigma):
        output += alphas_view[-1,s]
    return output


def calculate_expected_frequency_hmm(seq,transmat,rho):
    """
    This is used for the Markovianity test. Compute the expected counts from a Markovian model.
    """
    l = len(seq)
    prob = 1
    for i in range(l):
        if i == 0:
            prob *= rho[seq[i]]
        else:
            prob *= transmat[seq[i-1]][seq[i]]
    return prob


@cython.cdivision(True)
@cython.boundscheck(False)
cpdef double calculate_empirical_frequency_hmm(np.int_t [:] seq, double [:,:] Y, double [:,:] transmat_, double [:] stationary_probs_):
    """
    This is used for the Markovianity test. Empirical frequency of a motif.
    """
    cdef int l = seq.shape[0]
    cdef int L = Y.shape[0]
    cdef double prob = calculate_expected_frequency_hmm(seq,transmat_, stationary_probs_)
    cdef double counts = 0
    cdef int i,j
    cdef double count,likelihood
    for i in range(L-l):
        count = 1
        for j in range(l):
            count *= Y[i+j,seq[j]]
        likelihood = P_YgHMM(Y[i:i+l],transmat_, stationary_probs_)
        count /= likelihood
        count *= prob
        counts += count
    return counts/(L-l)


def create_word_array(ws,w_dict):
    """
    Convert a list of motifs to a single array of letters.
    """
    arr = np.array([])
    for w in ws:
        arr = np.concatenate((arr,w_dict[w]))
    arr = np.array(arr,dtype = int)
    return arr


def Z_partitions(seq,P_w,w_dict):
    """
    This is zeta(m) from the paper. 
    Computes the probability of all possible partitions of a motif.
    """
    L = len(seq)
    D = len(w_dict)
    lengths = [len(w) for w in w_dict]
    lmax = np.max(lengths)
    Z_p = np.zeros(L)
    for i in range(L):
        for l in range(min(i+1,lmax)):
            for w in range(D):
                if np.array_equal(seq[i-l:i+1],w_dict[w]):
                    if i >= l+1:
                        Z_p[i] += Z_p[i-l-1]*P_w[w]
                    else:
                        Z_p[i] += P_w[w]
    return Z_p[-1] 


def Nw1w2(ws, P_w, w_dict, Y, Q, Q_gw1, Q_gw2, nz_pos, params):
    """
    Computes the number of times a concatenated motif occurs in the dataset. The subroutine_Nw1w2 is called here. 
    Directly gives the p-value for the over-representation of the concatenated motif relative to random juxtaposition.
    """
    arr = create_word_array(ws,w_dict)
    lengths = [len(w) for w in w_dict]
    l1 = len(w_dict[ws[0]])
    l2 = len(w_dict[ws[1]])
    
    lmax = get_lmax([arr],params)
    
    R = evaluate_R(Q)
    R1 = evaluate_R1(Q)
    G = evaluate_G(R,R1,lmax)
    
    Fseq = -np.log(Z_partitions(arr,P_w,w_dict))
    
    N_calc = np.exp(-Fseq)*subroutine_Nw1w2(arr,G,nz_pos,Q_gw1,Q_gw2,l1,l2,params) + 1e-10
    f_calc = N_calc/len(Y)
    if N_calc < 5.0:
        return 1.0
    
    lmean = np.sum(lengths*P_w)
    N_exp = len(Y)*np.exp(-Fseq)/lmean + 1e-10
    f_exp = N_exp/len(Y)
    
    q1 = 1 + (1.0/f_exp + 1.0/(1-f_exp) - 1)/(6.0*len(Y)) #correction to LR test
    m2lnLR = 2*len(Y)*(f_calc*np.log(f_calc/f_exp) + (1-f_calc)*np.log((1-f_calc)/(1-f_exp)))/q1
    return stats.chi2.sf(m2lnLR,1)


def F_beta(betas, W_ils, L, lmax, D, params):
    """
    Free energy as a function of betas. beta_i = log(p_{-1}/p_i).
    """
    P_w = np.zeros(len(betas)+1)
    P_w[:-1] = np.exp(-betas)
    P_w[-1] = 1.0
    P_w /= np.sum(P_w)
    for i in range(len(P_w)):
        if P_w[i] != P_w[i]:
            print("Error in F_beta", i, betas[i])
            sys.stdout.flush()
    Q = evaluate_Q(P_w, W_ils, L, lmax, D)
    R = evaluate_R(Q)
    F = evaluate_F(R)
    return F/L


def dF_beta(betas, W_ils, L, lmax, D,params):
    """
    Gradient of free energy in terms of betas.
    """
    P_w = np.zeros(D)
    P_w[:-1] = np.exp(-betas)
    P_w[-1] = 1.0
    P_w /= np.sum(P_w)
    
    Q = evaluate_Q(P_w, W_ils, L, lmax, D)
    R = evaluate_R(Q)
    R1 = evaluate_R1(Q)
    
    G = evaluate_G(R,R1,lmax)
    
    dF_Pw = evaluate_dF(P_w, G, W_ils)
    
    dF_betas = np.zeros(len(betas))
    for j in range(D-1):
        dF_betas[j] = P_w[j]*(-dF_Pw[j] + np.sum(P_w*dF_Pw))
        
    return dF_betas/L


def minimize_F(Y, W_ils, L, lmax, D, params, method):
    """
    Minimizing free energy using gradient descent.
    """
    betas0 = 0.1*np.random.randn(D-1)
    bounds = []
    for i in range(len(betas0)):
        bounds += [[-20.0,20.0]]
    res = minimize(F_beta, betas0, args = (W_ils, L, lmax, D, params),method = method, jac = dF_beta, bounds = bounds, options = {'disp':False})
    return res


def get_P_w(Y, lengths_Y, w_dict, params,method = 'L-BFGS-B'):
    """
    Optimizing for P_w. MLE.
    """
    lmax = get_lmax(w_dict, params)
    L = len(Y)
    D = len(w_dict)
    W_ils = get_W_ils(w_dict, Y, lengths_Y, params)
    res = minimize_F(Y,W_ils, L, lmax, D,params,method) 
    P_w_fit = np.zeros(D)
    P_w_fit[:-1] = np.exp(-res.x)
    P_w_fit[-1] = 1.0
    P_w_fit /= np.sum(P_w_fit)
    return P_w_fit


def decode_Y(Y,P_w,w_dict,params):
    """
    This is called in the "decode" function below.
    """
    D = len(w_dict)
    L = len(Y)
    K = np.zeros(L)
    lmax = get_lmax(w_dict,params)
    decodedws = []
    decodedls = []
    argmaxws = np.zeros(L)
    argmaxls = np.zeros(L)
    for i in range(L):
        l = min(i,lmax-1)
        Q_gw_i = get_Q_gw_i(Y[i-l:i+1],w_dict,params)
        if i == 0:
            K[i] = np.log(np.max(P_w*Q_gw_i[0]))
            argmaxws[0] = np.argmax(P_w*Q_gw_i[0])
            argmaxls[0] = 0
        else:
            argmaxw = np.argmax(P_w*Q_gw_i,axis=1)
            maxw = np.max(P_w*Q_gw_i,axis=1)
            ls = []
            for l in range(min(i+1,lmax)):
                if l == i:
                    ls += [np.log(maxw[l])]
                else:
                    ls += [np.log(maxw[l]) + K[i-l-1]]
            argmaxl = np.argmax(ls)
            K[i] = np.max(ls)
            argmaxws[i] = argmaxw[argmaxl]
            argmaxls[i] = argmaxl
    i = L-1
    while i >= 0:
        decodedws += [int(argmaxws[int(i)])]
        decodedls += [int(argmaxls[int(i)])+1]
        i -= argmaxls[int(i)]+1
    ws = np.array(decodedws[::-1])
    ls = np.array(decodedls[::-1])
    
    w_ML = []
    for w in ws:
        w_ML += list(w_dict[w])
    
    return w_ML,ws,ls


def decode(Y,lengths_Y,w_dict,params):
    """
    Decoding the most likely sequence of motifs given the dataset and the dictionary of motifs using the Viterbi-like algorithm.
    """
    L = len(Y)
    D = len(w_dict)
    lmax = get_lmax(w_dict,params)
    P_w = get_P_w(Y, lengths_Y, w_dict, params)
    
    kk = 0
    w_MLs = []
    words = []
    wordlengths = []
    for lths in lengths_Y:
        w_ML,ws,ls = decode_Y(Y[kk:kk+lths],P_w,w_dict,params)
        w_MLs += [w_ML]
        words += [ws]
        wordlengths += [ls]
        kk += lths
    return w_MLs,words,wordlengths


def generate_Y(L, P_w, w_dict, params, model):
    """
    Generating synthetic data.
    """
    l = 0
    Y = []
    ws = []
    words_true = []
    while l < L:
        w = np.random.choice(len(P_w), 1, p=P_w)[0]
        ws += list(w_dict[w])
        words_true += [w]
        S = sample(w_dict[w],params,model)
        Y = Y + S
        l += len(S)
    Ydata = np.array(Y)
    return convert_Y_to_Y_Ps(Y,params,model),words_true,Ydata


def generate_w_dict(alphfreqs, D, lmean):
    """
    Generating a synthetic dictionary.
    """
    w_dict= []
    alphabetsize = len(alphfreqs)
    flag = 0
    while len(w_dict) < D:
        numletters = poisson.rvs(lmean)#np.random.geometric(1.0/lmean)#
        while numletters < 2:
            numletters = poisson.rvs(lmean)#np.random.geometric(1.0/lmean)
        w = np.zeros(numletters)
        for j in range(numletters):
            w[j] = np.random.choice(alphabetsize, 1, p=alphfreqs)[0]
        
        for ww in w_dict:
            if len(w) == len(ww) and np.prod((w == ww)):
                flag = 1
        if flag == 1:
            flag = 0
            continue
        w_dict += [w]
    for i in range(alphabetsize):
        w_dict += [np.array([i])]
    w_dict = [np.array(w, dtype = int) for w in w_dict]
    return w_dict


def remove_duplicates_w_dict(P_w,w_dict,params,model):
    """
    Removing duplicate motifs from the dictionary. This is the function that implements dictionary truncation based on JS divergence.
    """
    eps = params[0]
    dups = []
    if eps > 1e-3:
        JS = get_JS(w_dict,params,model)
        Jthr = params[6]
        P_w_weighted = np.sum((JS < Jthr)*P_w,axis=1)
        
        for i in range(len(w_dict)):#Remove duplicates
            for j in range(i+1,len(w_dict)): 
                if JS[i][j] < Jthr and P_w_weighted[i] >= P_w_weighted[j]:
                    dups += [j]
                elif JS[i][j] < Jthr and P_w_weighted[i] < P_w_weighted[j]:
                    dups += [i]
                    break
    else:
        for i in range(len(w_dict)):#Remove duplicates
            for j in range(i+1,len(w_dict)): 
                if np.array_equal(w_dict[i],w_dict[j]):
                    dups += [j]
        
    dups = np.array(dups)
    dups = np.unique(dups)                
    print(dups)
    for j in range(len(dups)):
        del w_dict[dups[j]]
        dups -= 1
    return w_dict


def truncate_w_dict(P_w,w_dict,thr = 5e-4): 
    """
    Further truncate dictionary if the motif occurs fewer than a certain number of times. 5 copies is the currently set threshold.
    """
    i = 0
    P_w = list(P_w)
    while i < len(w_dict):#Remove low probability words
        if P_w[i] < thr and len(w_dict[i]) > 1:
            del w_dict[i]
            del P_w[i]
            i-=1
        i+=1
    return w_dict


def prune_w_dict(Y,lengths_Y, w_dict, params,model):
    """
    Keep truncating until all motifs occurs more than 5 times. In its current implementation only truncates once.
    """
    D = len(w_dict)
    Dnew = 0
    i=0
    P_w = get_P_w(Y,lengths_Y, w_dict, params)
    w_dict = remove_duplicates_w_dict(P_w,w_dict,params,model)

    while Dnew != D and i < 1:
        D = len(w_dict)
        P_w = get_P_w(Y,lengths_Y, w_dict, params)
        lengths = [len(w) for w in w_dict]
        lmean = np.sum(lengths*P_w)
        thr = 5*lmean/len(Y)  #Threshold of 5 copies is set here. 
        w_dict = truncate_w_dict(P_w,w_dict,thr)
        Dnew = len(w_dict)
        i+=1
    return w_dict


def prune_letters_w_dict(Y,P_w,w_dict): 
    """
    This is used at the very end of the algorithm. Used to remove low probability single letters from the dictionary. 
    5 copies is the threshold again.
    """
    lengths = [len(w) for w in w_dict]
    lmean = np.sum(lengths*P_w)
    thr = 5*lmean/len(Y)
    
    P_w = list(P_w)
    i = 0
    while i < len(w_dict):#Remove low probability words
        if P_w[i] < thr and len(w_dict[i]) == 1: 
            del w_dict[i]
            del P_w[i]
            i-=1
        i+=1
    return w_dict


def get_words_to_add(Y,lengths_Y,w_dict,params):
    """
    This is the main function used for dictionary expansion. This function is heavily optimized.
    """
    words_to_add = []
    P_w = get_P_w(Y,lengths_Y, w_dict, params)
    lengths = [len(w) for w in w_dict]
    lmax_w_dict = np.max(lengths)
    L = len(Y)
    lmax = get_lmax(w_dict,params)
    D = len(w_dict)
    W_ils = get_W_ils(w_dict, Y,lengths_Y, params)
    Q = evaluate_Q(P_w, W_ils, L, lmax, D)
    
    R = evaluate_R(Q)
    R1 = evaluate_R1(Q)
    G = evaluate_G(R,R1,lmax)
    w_thr = params[4]
    lmean = np.sum(P_w*lengths)
    for ibeta in range(len(w_dict)):
        lmin,lmax = get_lmin_and_max(w_dict[ibeta],params)
        Nw1w2_arr = np.zeros(len(w_dict))
        Q_gw2 = get_Q_gw_from_W_il(W_ils[ibeta],Y,w_dict,params)
        
        nz_pos = np.nonzero(np.sum(Q_gw2[:,lmin:lmax],axis=1) > w_thr)
        nz_pos = np.array(nz_pos[0],dtype = int)
        
        for ialpha in range(len(w_dict)):
            ws = [ialpha,ibeta]
            arr = create_word_array(ws,w_dict)
            Q_gw1 = get_Q_gw_from_W_il(W_ils[ialpha],Y,w_dict,params)
            if len(arr) > 25: #Not sure this is required. This is the upper limit on the length of motifs. 
                continue
            pvalue_ij = Nw1w2(ws,P_w,w_dict,Y,Q,Q_gw1,Q_gw2,nz_pos, params)
            if pvalue_ij < 1e-3:  #p_value of over-representation of concatenated motif should be less than 0.001 to accept a new motif to the dictionary. 
                words_to_add += [arr]
                
    return words_to_add


def update_w_dict(Y,lengths_Y,w_dict,params,model):
    """
    One iteration of dictionary expansion and truncation.
    """
    words_to_add = get_words_to_add(Y,lengths_Y,w_dict,params) #expand              
    w_dict = w_dict + words_to_add
    print("Dictionary length %d" %len(w_dict))
    w_dict = prune_w_dict(Y,lengths_Y,w_dict,params,model) #truncate
    print("Pruned length %d" %len(w_dict))
    sys.stdout.flush()
    return w_dict


def get_entropy(Y):
    P_S = np.zeros(Y.shape[1])
    P_Y = np.sum(Y,axis=1)
    P_Si = np.mean(Y/P_Y[:,np.newaxis],axis=0)
    P_Yi = np.sum(Y*P_Si,axis=1)
    entropy = np.mean(-P_Yi*np.log(P_Yi+1e-15))
    return entropy


def solve_dictionary(Y,lengths_Y,params,model,niter = 6):
    """
    Main function used to run the algorithm. 
    Takes in sequences of probability vectors over an alphabet of size Sigma and the lengths of each sequence (the sum of lengths_Y should equal total length of Y)
    Returns the probabilities of each motif and the dictionary itself.
    """
    if np.sum(lengths_Y) != len(Y) or np.min(lengths_Y) <= 0:
        print("Invalid lengths_Y", np.sum(lengths_Y), len(Y))
        sys.stdout.flush()
        return 
    
    w_dict = []
    Sigma = Y.shape[1]
    params[7] = Sigma

    for i in range(Sigma):
        w_dict += [np.array([i], dtype = int)]
        
    P_w = get_P_w(Y,lengths_Y,w_dict,params)
    
    Fs = np.zeros(niter+1)
    Fs[0] = evaluate_F_seq(Y,lengths_Y,P_w, w_dict, params)/len(Y)
    
    w_dict = prune_w_dict(Y,lengths_Y,w_dict,params,model)
    
    for i in range(niter):
        w_dict = update_w_dict(Y,lengths_Y,w_dict,params,model)
        P_w = get_P_w(Y,lengths_Y,w_dict,params)
        Ftrain = evaluate_F_seq(Y, lengths_Y,P_w, w_dict, params)/len(Y)
        Fs[i+1] = Ftrain 
        w_dict_list = [list(w) for w in w_dict]
        print("%d iter, w_dict length = %d, Train -logL = %.3f" %(i+1,len(w_dict),Ftrain))
        sys.stdout.flush()
        if i == niter - 1 or (i > 1 and abs(Fs[i] - Fs[i+1]) < 0.002 and abs(Fs[i-1] - Fs[i]) < 0.002):
            w_dict = prune_letters_w_dict(Y,P_w,w_dict)
            w_dict = prune_w_dict(Y,lengths_Y,w_dict,params,model)
            P_w = get_P_w(Y, lengths_Y, w_dict, params)
            break
    print("Final length of w_dict = %d" %(len(w_dict)))
    print("Done, w_dict length = %d" %(len(w_dict)))
    return P_w,w_dict

