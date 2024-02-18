import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split

PATH_TO_BP = 'Kaggle_dataset/train/BP'
PATH_TO_SZ = 'Kaggle_dataset/train/SZ'

BP_folder_names = os.listdir(PATH_TO_BP)
SZ_folder_names = os.listdir(PATH_TO_SZ)
BP_folders_paths = [ os.path.join(PATH_TO_BP, x ) for x in BP_folder_names]
SZ_folders_paths = [ os.path.join(PATH_TO_SZ, x ) for x in SZ_folder_names]

print("total BP:", len(BP_folder_names))
print("total SZ:", len(SZ_folder_names))
TOTAL_ENTRIES = len(BP_folder_names) + len(SZ_folder_names)
print("Total_entries:", TOTAL_ENTRIES)

X_train_full = []
y_train_full = []

for path in BP_folders_paths:

    fnc_array = np.load(os.path.join(path, "fnc.npy"))
    fnc_array = fnc_array.reshape(1,5460)[0].tolist()
    X_train_full.append(fnc_array)
    y_train_full.append(0)


for path in SZ_folders_paths:
    fnc_array = np.load(os.path.join(path, "fnc.npy"))
    fnc_array = fnc_array.reshape(1,5460)[0].tolist()
    X_train_full.append(fnc_array)
    y_train_full.append(1)
    
X_train_full = pd.DataFrame(X_train_full)
y_train_full = pd.Series(y_train_full)

print(X_train_full.shape)

X_train, X_test, y_train, y_test = train_test_split(X_train_full, y_train_full, test_size=0.3, random_state=42)