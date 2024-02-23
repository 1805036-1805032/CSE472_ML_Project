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

whole_icn_df = pd.DataFrame()


for i in range(len(BP_folders_paths)):

    icn_tc_array = np.load(os.path.join(BP_folders_paths[i], "icn_tc.npy"))
    icn_tc_df = pd.DataFrame(icn_tc_array)
    
    icn_tc_df["sub_num"] = BP_folder_names[i]
    y_train_full.append(1)
    whole_icn_df = pd.concat([whole_icn_df, icn_tc_df], axis=0)


for i in range(len(SZ_folders_paths)):

    icn_tc_array = np.load(os.path.join(SZ_folders_paths[i], "icn_tc.npy"))
    icn_tc_df = pd.DataFrame(icn_tc_array)
    icn_tc_df["sub_num"] = SZ_folder_names[i]
    
    whole_icn_df = pd.concat([whole_icn_df, icn_tc_df], axis=0)

column_names = [ f"col_{str(x)}" for x in range(105)]
column_names.append("sub_num")

whole_icn_df.columns = column_names

print("Whole icn_tc dataframe is created")

i = 0

for sub_num in whole_icn_df["sub_num"].unique():
    sub_df = whole_icn_df[whole_icn_df["sub_num"] == sub_num]
    # for each subject, building the connectome / correalaion matrix
    
    connectome = np.zeros((105, 105))

    for row in range(105):
        measures_1 = sub_df[f"col_{row}"]
        for col in range(105):
            measures_2 = sub_df[f"col_{col}"]
            connectome[row, col] = np.corrcoef(measures_1, measures_2)[0, 1]        
        
            
    # # sanity check, checking if the connectome is correct
    # print(type(connectome))
    # print(connectome.shape)

    # plt.imshow(connectome, cmap='coolwarm', interpolation='nearest')
    # plt.show()
            
    X_train_full.append(connectome)
    print(f"done with sub {sub_num}")

    i += 1
    if i == 5:
        break

X_train_full = np.array(X_train_full)
X_test_full = np.array(X_test_full)
y_train_full = pd.Series(y_train_full)


print("-" * 10)
print("X_train_full:", X_train_full.shape)
print("y_train_full:", y_train_full.shape)
print("X_test_full:", X_test_full.shape)