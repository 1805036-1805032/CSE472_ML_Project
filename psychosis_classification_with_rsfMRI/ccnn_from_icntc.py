import os
import numpy as np
import pandas as pd
import torch
import warnings
import matplotlib.pyplot as plt


np.random.seed(0)
torch.manual_seed(0)
warnings.filterwarnings("ignore")

# loading the dataset
PATH_TO_BP = "/home/mmk/4_2_resources/CSE472_ML_Project/psychosis_classification_with_rsfMRI/train/BP"
PATH_TO_SZ = "/home/mmk/4_2_resources/CSE472_ML_Project/psychosis_classification_with_rsfMRI/train/SZ"

BP_folder_names = os.listdir(PATH_TO_BP)
SZ_folder_names = os.listdir(PATH_TO_SZ)
BP_folders_paths = [os.path.join(PATH_TO_BP, x) for x in BP_folder_names]
SZ_folders_paths = [os.path.join(PATH_TO_SZ, x) for x in SZ_folder_names]

print("Training Dataset\n", "-" * 10)
print("total BP:", len(BP_folder_names))
print("total SZ:", len(SZ_folder_names))
TOTAL_ENTRIES = len(BP_folder_names) + len(SZ_folder_names)
print("Total_entries:", TOTAL_ENTRIES)


PATH_TO_TEST = "/home/mmk/4_2_resources/CSE472_ML_Project/psychosis_classification_with_rsfMRI/test"
test_folder_names = os.listdir(PATH_TO_TEST)
test_folder_paths = [os.path.join(PATH_TO_TEST, x) for x in test_folder_names]
print("Test Dataset\n", "-" * 10)
print("Total_entries in test:", len(test_folder_names))

# creating test and train dataset, icn_tc dataframe
X_train_full = []
y_train_full = []
X_test_full = []

def calculate_connectome(icn_tc_array):
    correlations = np.corrcoef(icn_tc_array.T)
    return correlations

for i in range(len(BP_folders_paths)):
    icn_tc_array = np.load(os.path.join(BP_folders_paths[i], "icn_tc.npy"))
    connectome = calculate_connectome(icn_tc_array)
    X_train_full.append(connectome)
    y_train_full.append(1)


for i in range(len(SZ_folders_paths)):
    icn_tc_array = np.load(os.path.join(SZ_folders_paths[i], "icn_tc.npy"))
    connectome = calculate_connectome(icn_tc_array)
    X_train_full.append(connectome)
    y_train_full.append(0)


for i in range(len(test_folder_paths)):
    icn_tc_array = np.load(os.path.join(test_folder_paths[i], "icn_tc.npy"))
    connectome = calculate_connectome(icn_tc_array)
    X_test_full.append(connectome)


X_train_full = np.array(X_train_full)
X_test_full = np.array(X_test_full)
y_train_full = pd.Series(y_train_full)


print("-" * 10)
print("X_train_full:", X_train_full.shape)
print("y_train_full:", y_train_full.shape)
print("X_test_full:", X_test_full.shape)