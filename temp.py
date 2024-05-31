import numpy as np
import pickle

synth_covars = np.load(".\\GMM\\synth_covars.npy")
synth_means = np.load(".\\GMM\\synth_means.npy")
synth_weights = np.load(".\\GMM\\synth_weights.npy")
synth_dataset_condition0 = np.load(".\\Data\\synth_dataset_condition0.npy")
synth_lengths_condition0 = np.load(".\\Data\\synth_lengths_condition0.npy")
synth_condition_0_seg_bouttypes = np.load(".\\Results\\synth\\synth_condition_0_seg_bouttypes.npy")
synth_condition_0_seg_lengths = np.load(".\\Results\\synth\\synth_condition_0_seg_lengths.npy")
synth_condition_0_seg_words = np.load(".\\Results\\synth\\synth_condition_0_seg_words.npy")

toy_condition_0_seg_bouttypes = np.load(".\\Results\\toy\\toy_condition_0_seg_bouttypes.npy")
toy_condition_0_seg_lengths = np.load(".\\Results\\toy\\toy_condition_0_seg_lengths.npy")
toy_condition_0_seg_words = np.load(".\\Results\\toy\\toy_condition_0_seg_words.npy")

fw = open(".\\Results\\synth\\synth_condition_0\\BASSresults",'rb')
data = pickle.load(fw)

words_len = []
for word in data[2]:
    words_len.append(len(word))

for
temp_var4 = np.load(".\\Data\\synth_dataset_condition0.npy")
