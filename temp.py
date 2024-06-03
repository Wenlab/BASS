import numpy as np
import pickle
import matplotlib.pyplot as plt

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

synth_condition_0_seg_bouttypes_recon = []
word_start = 0
for word_start_next in synth_condition_0_seg_lengths:
    words_end = word_start_next-1
    word_ID_present = synth_condition_0_seg_words[word_start]
    for idx_char in range(word_start,word_start_next):
        if synth_condition_0_seg_words[idx_char] != word_ID_present:
            print("Inconsistent word! position:",idx_char)
            print("\n")
    word_present = data[2][word_ID_present]
    if len(word_present) != word_start_next-word_start:
        print("Inconsistent word length! position:", word_start,"length difference:",len(word_present)-(word_start_next-word_start))
        print("\n")
    synth_condition_0_seg_bouttypes_recon = np.concatenate((synth_condition_0_seg_bouttypes_recon, word_present))
    word_start = word_start_next
print("\n")
if np.array_equal(synth_condition_0_seg_bouttypes_recon, synth_condition_0_seg_bouttypes):
    print("seg_bouttypes reconstruction right!")
    print("\n")
else:
    print("seg_bouttypes reconstruction wrong!")
    print("\n")

unique_elements_in_synth_condition_0_seg_words = np.unique(synth_condition_0_seg_words)
# plt.hist(synth_condition_0_seg_words, bins=31, color='skyblue', edgecolor='black')
# plt.show()

temp_var4 = np.load(".\\Data\\synth_dataset_condition0.npy")
