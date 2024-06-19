import numpy as np
import pickle
import matplotlib.pyplot as plt
import scipy.io

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

# Prepare the data. Transform mat files to npy files.
mat = scipy.io.loadmat('D:\\Nutstore\\我的坚果云\\临时\\2023_11_28-16_56_8\\behavior\\bouts_softmax_output_withHeadingVelocity_extent4.mat')
data = np.transpose(mat['softmaxOutput'])
np.save('.\\Data\\20231128_dataset_condition0.npy', data.astype(np.float64))
np.save('.\\Data\\20231128_lengths_condition0.npy', np.array([data.shape[0]], dtype=np.int64))
ldg_covars = mat['covs']
ldg_means = mat['means']
ldg_weights = mat['weights']
np.save('.\\GMM\\20231128_covars.npy', ldg_covars.astype(np.float64))
np.save('.\\GMM\\20231128_means.npy', ldg_means.astype(np.float64))
np.save('.\\GMM\\20231128_weights.npy', ldg_weights.astype(np.float64))
ldg_dataset_condition0 = np.load(".\\Data\\20231128_dataset_condition0.npy")
ldg_lengths_condition0 = np.load(".\\Data\\20231128_lengths_condition0.npy")
ldg_covars = np.load(".\\GMM\\20231128_covars.npy")
ldg_means = np.load(".\\GMM\\20231128_means.npy")
ldg_weights = np.load(".\\GMM\\20231128_weights.npy")

temp_var4 = np.load(".\\Data\\synth_dataset_condition0.npy")



import numpy as np
from scipy import stats

def make_positive_semidefinite(cov_matrix, epsilon=1e-6):
    # Make the matrix symmetric
    cov_matrix = (cov_matrix + cov_matrix.T) / 2
    # Add epsilon to the diagonal elements
    cov_matrix += np.eye(cov_matrix.shape[0]) * epsilon
    # Ensure positive semidefiniteness by checking eigenvalues
    eigvals = np.linalg.eigvalsh(cov_matrix)
    if np.any(eigvals < 0):
        min_eigval = np.min(eigvals)
        cov_matrix += np.eye(cov_matrix.shape[0]) * (-min_eigval + epsilon)
    return cov_matrix

# Modify your scoring function
def score(self, dataset, allow_singular=False):
    temp = 0
    for k in range(self.n_components):
        cov_matrix = self.covars_[k]
        cov_matrix = make_positive_semidefinite(cov_matrix)
        temp += self.weights_[set_index, k] * stats.multivariate_normal.pdf(
            dataset, mean=self.means_[k], cov=cov_matrix, allow_singular=allow_singular
        )
    return temp

# In the main function, set allow_singular=True if needed
H = -model_fit.score(data_flat, args.Condition, allow_singular=True) / len(data_flat)  # entropy
