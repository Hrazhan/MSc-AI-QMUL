import numpy as np
import os
import matplotlib.pyplot as plt
from print_values import *
from plot_data_all_phonemes import *
from plot_data import *
import random
from sklearn.preprocessing import normalize
from get_predictions import *
from plot_gaussians import *

# File that contains the data
data_npy_file = 'data/PB_data.npy'

# Loading data from .npy file
data = np.load(data_npy_file, allow_pickle=True)
data = np.ndarray.tolist(data)

# Make a folder to save the figures
figures_folder = os.path.join(os.getcwd(), 'figures')
if not os.path.exists(figures_folder):
    os.makedirs(figures_folder, exist_ok=True)

# Array that contains the phoneme ID (1-10) of each sample
phoneme_id = data['phoneme_id']
# frequencies f1 and f2
f1 = data['f1']
f2 = data['f2']

# Initialize array containing f1 & f2, of all phonemes.
X_full = np.zeros((len(f1), 2))
#########################################
# Write your code here
# Store f1 in the first column of X_full, and f2 in the second column of X_full
X_full[:,0] = f1
X_full[:,1] = f2
########################################/
X_full = X_full.astype(np.float32)

# number of GMM components
k = 3

#########################################
# Write your code here

# Create an array named "X_phonemes_1_2", containing only samples that belong to phoneme 1 and samples that belong to phoneme 2.
# The shape of X_phonemes_1_2 will be two-dimensional. Each row will represent a sample of the dataset, and each column will represent a feature (e.g. f1 or f2)
# Fill X_phonemes_1_2 with the samples of X_full that belong to the chosen phonemes
# To fill X_phonemes_1_2, you can leverage the phoneme_id array, that contains the ID of each sample of X_full

X_phonemes_1_2 = X_full[(phoneme_id == 1) | (phoneme_id == 2)]
########################################/

# Plot array containing the chosen phonemes

# Create a figure and a subplot
fig, ax1 = plt.subplots()

title_string = 'Phoneme 1 & 2'
# plot the samples of the dataset, belonging to the chosen phoneme (f1 & f2, phoneme 1 & 2)
plot_data(X=X_phonemes_1_2, title_string=title_string, ax=ax1)
# save the plotted points of phoneme 1 as a figure
plot_filename = os.path.join(os.getcwd(), 'figures', 'dataset_phonemes_1_2.png')
plt.savefig(plot_filename)


#########################################
# Write your code here
# Get predictions on samples from both phonemes 1 and 2, from a GMM with k components, pretrained on phoneme 1
# Get predictions on samples from both phonemes 1 and 2, from a GMM with k components, pretrained on phoneme 2
# Compare these predictions for each sample of the dataset, and calculate the accuracy, and store it in a scalar variable named "accuracy"


# 3 Clusters
phoneme_1_3 = np.load('data/GMM_params_phoneme_01_k_03.npy', allow_pickle=True)
phoneme_1_3 = np.ndarray.tolist(phoneme_1_3)

phoneme_2_3 = np.load('data/GMM_params_phoneme_02_k_03.npy', allow_pickle=True)
phoneme_2_3 = np.ndarray.tolist(phoneme_2_3)


z_1_3 = get_predictions(phoneme_1_3["mu"], phoneme_1_3["s"], phoneme_1_3["p"], X_phonemes_1_2) 
z_2_3 = get_predictions(phoneme_2_3["mu"], phoneme_2_3["s"], phoneme_2_3["p"], X_phonemes_1_2)



# 6 Clusters
phoneme_1_6 = np.load('data/GMM_params_phoneme_01_k_06.npy', allow_pickle=True)
phoneme_1_6 = np.ndarray.tolist(phoneme_1_6)

phoneme_2_6 = np.load('data/GMM_params_phoneme_02_k_06.npy', allow_pickle=True)
phoneme_2_6 = np.ndarray.tolist(phoneme_2_6)

z_1_6 = get_predictions(phoneme_1_6["mu"], phoneme_1_6["s"], phoneme_1_6["p"], X_phonemes_1_2)
z_2_6 = get_predictions(phoneme_2_6["mu"], phoneme_2_6["s"], phoneme_2_6["p"], X_phonemes_1_2) 

pred_3 = []
pred_6 = []
for i in range(0, len(X_phonemes_1_2)):
    pred_1_3 = sum(z_1_3[i])
    pred_2_3 = sum(z_2_3[i])

    if pred_1_3 > pred_2_3:
        pred_3.append(1)
    else:
        pred_3.append(2)


    pred_1_6 = sum(z_1_6[i])
    pred_2_6 = sum(z_2_6[i])

    if pred_1_6 > pred_2_6:
        pred_6.append(1)
    else:
        pred_6.append(2)



# print(pred_3)
phoneme_id_1_or_2 = phoneme_id[(phoneme_id == 1) | (phoneme_id == 2)]


correct_3 = 0
correct_6 = 0

for i in range(0, len(phoneme_id_1_or_2)):
    if pred_3[i] == phoneme_id_1_or_2[i]:
        correct_3 += 1

    if pred_6[i] == phoneme_id_1_or_2[i]:
        correct_6 += 1



# accuracy = (correct_6/len(phoneme_id_1_or_2)) * 100
# accuracy6 = (correct_6/len(phoneme_id_1_or_2)) * 100
if k == 3:
    accuracy = (correct_3/len(phoneme_id_1_or_2)) * 100
    error =  (1 - (correct_3/len(phoneme_id_1_or_2)) ) * 100
else:
    accuracy = (correct_6/len(phoneme_id_1_or_2)) * 100
    error = (1 - (correct_6/len(phoneme_id_1_or_2)) ) * 100
    
########################################/

print('Accuracy using GMMs with {} components: {:.2f}%'.format(k, accuracy))
print('Miss classification error: {:.2f}%'.format(error))
################################################
# enter non-interactive mode of matplotlib, to keep figures open
plt.ioff()
plt.show()
