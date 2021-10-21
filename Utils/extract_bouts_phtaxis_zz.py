#Author: Gautam Reddy Nallamala. Email: gautam_nallamala@fas.harvard.edu
#Packaged by: Gautam Sridhar, Email: gautam.sridhar@icm-institute.org

#This work is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License. 
#To view a copy of this license, visit http://creativecommons.org/licenses/by-nc-sa/4.0/ or send a letter to Creative Commons, 
#PO Box 1866, Mountain View, CA 94042, USA.

# Use as example script for extracting the data in the format required for running the GMM and running BASS


import argparse
import os
import time
#data format library
#numpy and scipy
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
#data processing functions
import data_processing_zz as dp


def main(args):

    px_to_mm = 0.071
    seed=args.Seed
    maxT = 30
    npoints = 7
    fps = 160.0

    #input the name of the folder where the data is located
    foldername = args.Path +  '/Catamaran_pH_2a/'
    filenames = []
    filenames += ['Catamaran_pH_2a_t1a']
    filenames += ['Catamaran_pH_2a_t2']
    filenames += ['Catamaran_pH_2a_t3']
    filenames += ['Catamaran_pH_2a_t4']
    filenames += ['Catamaran_pH_2a_t5']
    filenames += ['Catamaran_pH_2a_t6']
    filenames += ['Catamaran_pH_2a_t7']

    welltypes = []
    for i in range(len(filenames)):
        if i < 4:
            welltypes += [[0,0,0,0,0,0,1,1,1,1,1,1]]
        else:
            welltypes += [[0,0,0,0,0,0,2,2,2,2,2,2]]
    
    #bad data
    welltypes[0][6] = 4
    welltypes[2][9] = 4
    welltypes[3][7] = 4
        
    bouts_explo1 = dp.get_bouts_textdataset(foldername, filenames, welltypes, px_to_mm, fps, maxT, seed, 0)
    bouts_dbsharppH1 = dp.get_bouts_textdataset(foldername, filenames, welltypes, px_to_mm, fps, maxT, seed,1)

    foldername = args.Path  + '/Catamaran_pH_1a/'
    filenames = []
    filenames += ['Catamaran_pH_1a_t1a']
    filenames += ['Catamaran_pH_1a_t1b']
    filenames += ['Catamaran_pH_1a_t2a']
    filenames += ['Catamaran_pH_1a_t2b']
    filenames += ['Catamaran_pH_1a_t3a']
    filenames += ['Catamaran_pH_1a_t4a']
    filenames += ['Catamaran_pH_1a_t1c']
    filenames += ['Catamaran_pH_1a_t2c']
    filenames += ['Catamaran_pH_1a_t3c']
    filenames += ['Catamaran_pH_1a_t4c']

    welltypes = []
    for i in range(len(filenames)):
        if i < 6:
            welltypes += [[0,0,0,1,1,1,1,1,1,0,0,0]]
        else:
            welltypes += [[0,0,0,0,0,0,2,2,2,2,2,2]]

    bouts_explo2 = dp.get_bouts_textdataset(foldername, filenames, welltypes, px_to_mm, fps, maxT, seed, 0)
    bouts_dbsharppH2 = dp.get_bouts_textdataset(foldername, filenames, welltypes, px_to_mm, fps, maxT, seed, 1)

    foldername = args.Path + '/Catamaran_pH_2b/'
    filenames = []
    filenames += ['Catamaran_pH_2b_t1']
    filenames += ['Catamaran_pH_2b_t2']
    filenames += ['Catamaran_pH_2b_t3']
    filenames += ['Catamaran_pH_2b_t4']
    filenames += ['Catamaran_pH_2b_t5']
    filenames += ['191119_Catamaran_pH_2b_t6']
    filenames += ['191119_Catamaran_pH_2b_t8']

    welltypes = []
    for i in range(len(filenames)):
        welltypes += [[0,0,0,0,0,0,1,1,1,1,1,1]]
        
    welltypes[0][8] = 4
    welltypes[4][7] = 4
    welltypes[5][9] = 4
    welltypes[6][8] = 4

    bouts_explo3 = dp.get_bouts_textdataset(foldername, filenames, welltypes, px_to_mm, fps, maxT, seed, 0)
    bouts_dbsharppH3 = dp.get_bouts_textdataset(foldername, filenames, welltypes, px_to_mm, fps, maxT, seed, 1)


    foldername = args.Path + '/Catamaran_pH_2c/'
    filenames = []
    filenames += ['Catamaran_pH_2c_t1']
    filenames += ['Catamaran_pH_2c_t2']
    filenames += ['Catamaran_pH_2c_t3']
    filenames += ['Catamaran_pH_2c_t4']
    filenames += ['Catamaran_pH_2c_t5']
    filenames += ['Catamaran_pH_2c_t6']
    filenames += ['Catamaran_pH_2c_t7']
    filenames += ['Catamaran_pH_2c_t8']

    welltypes = []
    for i in range(len(filenames)):
        welltypes += [[0,0,0,0,0,0,1,1,1,1,1,1]]
        
    bouts_explo4 = dp.get_bouts_textdataset(foldername, filenames, welltypes, px_to_mm, fps, maxT, seed, 0)
    bouts_dbsharppH4 = dp.get_bouts_textdataset(foldername, filenames, welltypes, px_to_mm, fps, maxT, seed, 1)

    #pool data
    bouts_explo = bouts_explo1 + bouts_explo2 + bouts_explo3 + bouts_explo4
    bouts_dbsharppH = bouts_dbsharppH1 + bouts_dbsharppH2 + bouts_dbsharppH3 + bouts_dbsharppH4

    #gather tail angle data for PCA
    ta_explo = dp.get_tailangles(bouts_explo, maxT, npoints)
    ta_dbsharppH = dp.get_tailangles(bouts_dbsharppH, maxT, npoints)

    ta_all = np.concatenate((ta_explo,ta_dbsharppH))
    pca = PCA()
    pca.fit(ta_all)

    pcs_explo = pca.transform(ta_explo)
    pcs_dbsharppH = pca.transform(ta_dbsharppH)

    #add PC components into bout information
    bouts_explo = dp.update_tail_pcas(bouts_explo,pcs_explo)
    bouts_dbsharppH = dp.update_tail_pcas(bouts_dbsharppH,pcs_dbsharppH)

    #concatenate bouts into a list
    trajs_explo_nospacings = dp.collect_trajectories_nospacings(bouts_explo, fps, px_to_mm)
    trajs_dbsharppH_nospacings = dp.collect_trajectories_nospacings(bouts_dbsharppH, fps, px_to_mm)

    #collect data used as input to BASS
    data_explo_hmm, lengths_explo_hmm = dp.collect_data_hmm(trajs_explo_nospacings)
    data_dbsharppH_hmm, lengths_dbsharppH_hmm = dp.collect_data_hmm(trajs_dbsharppH_nospacings)

    #flattened list of trajectories
    trajs_explo_flat = []
    for t in trajs_explo_nospacings:
        trajs_explo_flat += t

    trajs_dbsharppH_flat = []
    for t in trajs_dbsharppH_nospacings:
        trajs_dbsharppH_flat += t

    #collected other variables
    data_explo_hmm_other, lengths_explo_hmm_other = dp.collect_data_hmm_other(trajs_explo_nospacings)
    data_dbsharppH_hmm_other, lengths_dbsharppH_hmm_other = dp.collect_data_hmm_other(trajs_dbsharppH_nospacings)

    tail_angles_explo,_ = dp.collect_tailangles_hmm(trajs_explo_nospacings, maxT, npoints)
    tail_angles_dbsharppH,_ = dp.collect_tailangles_hmm(trajs_dbsharppH_nospacings ,maxT, npoints)

    print(data_explo_hmm.shape)
    print(tail_angles_explo.shape)
    print(data_dbsharppH_hmm.shape)
    print(tail_angles_dbsharppH.shape)

    #These are loaded in the BASS algorithm code.
    #They contain the six parameters - speed, delta head, tail length and first three PCs
    #for every bout recorded in our experiments.

    #the files with the "other" suffix contain other auxiliary variables.

    np.save(args.Out + "data_condition0",data_explo_hmm)
    np.save(args.Out + "data_condition1",data_dbsharppH_hmm)

    np.save(args.Out + "lengths_condition0",lengths_explo_hmm)
    np.save(args.Out + "lengths_condition1",lengths_dbsharppH_hmm)

    np.save(args.Out + "tailangles_condition0",tail_angles_explo)
    np.save(args.Out + "tailangles_condition1", tail_angles_dbsharppH)

    np.save(args.Out + "data_condition0_other",data_explo_hmm_other)
    np.save(args.Out + "data_condition1_other",data_dbsharppH_hmm_other)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-seed','--Seed',help="Seed for data extraction", default=42,type=int)
    parser.add_argument('-path','--Path',help="path to data",default='/Users/gautam.sridhar/Documents/Repos/BASS/CatamaranData/',type=str)
    parser.add_argument('-out','--Out',help="path save",default='/Users/gautam.sridhar/Documents/Repos/BASS/Data/',type=str)
    args = parser.parse_args()
    main(args)
