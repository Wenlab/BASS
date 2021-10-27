#Author: Gautam Reddy Nallamala. Email: gautam_nallamala@fas.harvard.edu
#Packaged by: Gautam Sridhar. Email: gautam.sridhar@icm_institute.org

#This work is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License. 
#To view a copy of this license, visit http://creativecommons.org/licenses/by-nc-sa/4.0/ or send a letter to Creative Commons, 
#PO Box 1866, Mountain View, CA 94042, USA.

import numpy as np
import pandas as pd
import pickle
import os
import sys


def save_results_raw(args, P_dict, num_instances, w_dict):
    """
    Save results from BASS as raw output. 
    Use in order to have raw dictionary to perform further decoding
  
    Parameters:
    args: Argparse object containing general parameters:
    P_dict: Probabilities of dictionary to be saved
    num_instances: Number of instances of motif
    w_dict: the motifs output by BASS in numerical form
    """

    save_path = args.Out + args.DataName +'/'+ args.Exp + '_condition_{}'.format(args.Condition)
    if not(os.path.exists(save_path)):
        os.makedirs(save_path)
    outfile = open(save_path + '/BASSresults','wb')
    pickle.dump([P_dict, num_instances, w_dict],outfile)
    outfile.close()


def save_results_classnames(args, P_dict, num_instances, w_dict, class_names):
    """
    Save results from BASS as a csv file with the class names as motifs

    Parameters:
    args: Argparse object containing general parameters
    P_dict: Probabilities of dictionary to be saved
    num_instances: Number of instances of motif
    w_dict: the motifs output in numerical form
    class_names: the associated names of the bout types`
    """

    save_path = args.Out + args.DataName +'/' + args.Exp + '_condition_{}'.format(args.Condition)
    if not(os.path.exists(save_path)):
        os.makedirs(save_path)
    motifs = []
    for i, w in enumerate(w_dict):
        motif = [class_names[a] for a in w]
        motifs.append(str(motif))

    full_dict = pd.DataFrame({'Probability':P_dict,'Number of occurences':num_instances,'Sequences':motifs})
    full_dict.to_csv(path_or_buf = save_path + '/BASS_dictionary.csv', index = False)


def save_markovianity(args, w_dict, neg_log_p, empirical_freq, expected_freq, class_names):
    """
    Save the results of the comparison to an HMM on the bout types

    Parameters:
    args: Argparse object containing general parameters:
    w_dict: the motif output in numerical form
    neg_log_p: negative log P valaues of the statistical test
    empirical_freq: Actual frequency of motifs predicted by BASS
    expected_freq: Expected frequency of motifs predicted by an HMM
    """
    
    save_path = args.Out + args.DataName +'/' + args.Exp + '_condition_{}'.format(args.Condition)
    if not(os.path.exists(save_path)):
        os.makedirs(save_path)

    tostore_nlp = []
    tostore_emp = []
    tostore_exp = []

    motifs = []
    idx = np.argsort(-neg_log_p)
    for w in idx[:]:
        if empirical_freq[w] > expected_freq[w] and 10**(-neg_log_p[w]) < 1: #used to be 1e-3 not 1
            tostore_nlp.append(neg_log_p[w])
            tostore_emp.append(empirical_freq[w])
            tostore_exp.append(expected_freq[w])
            motifs.append(str([class_names[a] for a in w_dict[w]]))

    full_dict = pd.DataFrame({'Negative log P values':tostore_nlp,'Empirical frequency':tostore_emp,'Expected frequency':tostore_exp, 'Motif': motifs})
    full_dict.to_csv(path_or_buf = save_path + '/Comparison_to_HMM.csv', index = False)


def save_decoded(args, w_MLs, words, ls):
    """
    Save decoded output of data

    Parameters:
    args: Argparse object containing general parameters
    w_MLs: characters making up words
    words: individual words detected
    ls: word lengths
    """
    chars_out = []
    words_out = []
    lengths_out = []
    currsum = 0
    for i,w_ML in enumerate(w_MLs):
        chars_out += w_ML
        for k,l in enumerate(ls[i]):
            currsum += l
            lengths_out += [currsum]
            word = words[i][k]
            for j in range(l): 
                words_out += [word]
    chars_out = np.array(chars_out)
    words_out = np.array(words_out)
    lengths_out = np.array(lengths_out)

    save_path = args.Out +args.DataName + '/' +  args.Exp + '_condition_{}'.format(args.Condition)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    np.save(save_path + "_seg_bouttypes",chars_out)
    np.save(save_path + "_seg_words",words_out)
    np.save(save_path + "_seg_lengths",lengths_out)


def load_results_raw(args, condition):
    """
    Load raw BASS results
    """

    save_path = args.Out +args.DataName + '/' + args.Exp + '_condition_{}'.format(condition)

    infile = open(save_path + '/BASSresults','rb')
    results = pickle.load(infile)
    infile.close()

    return results
