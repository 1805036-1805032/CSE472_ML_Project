import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

PATH_TO_BP = "/home/mmk/4_2_resources/CSE472_ML_Project/psychosis_classification_with_rsfMRI/train/BP"
PATH_TO_SZ = "/home/mmk/4_2_resources/CSE472_ML_Project/psychosis_classification_with_rsfMRI/train/SZ"

BP_folder_names = os.listdir(PATH_TO_BP)
SZ_folder_names = os.listdir(PATH_TO_SZ)
BP_folders_paths = [os.path.join(PATH_TO_BP, x) for x in BP_folder_names]
SZ_folders_paths = [os.path.join(PATH_TO_SZ, x) for x in SZ_folder_names]

print("total BP:", len(BP_folder_names))
print("total SZ:", len(SZ_folder_names))
TOTAL_ENTRIES = len(BP_folder_names) + len(SZ_folder_names)
print("Total_entries:", TOTAL_ENTRIES)


# ### Working with ICN_TC only

whole_icn_df = pd.DataFrame()

for i in range(len(BP_folders_paths)):

    icn_tc_array = np.load(os.path.join(BP_folders_paths[i], "icn_tc.npy"))
    icn_tc_df = pd.DataFrame(icn_tc_array)
    icn_tc_df["sub_num"] = BP_folder_names[i]

    whole_icn_df = pd.concat([whole_icn_df, icn_tc_df], axis=0)


for i in range(len(SZ_folders_paths)):

    icn_tc_array = np.load(os.path.join(SZ_folders_paths[i], "icn_tc.npy"))
    icn_tc_df = pd.DataFrame(icn_tc_array)
    icn_tc_df["sub_num"] = SZ_folder_names[i]

    whole_icn_df = pd.concat([whole_icn_df, icn_tc_df], axis=0)

column_names = [f"col_{str(x)}" for x in range(105)]
column_names.append("sub_num")

whole_icn_df.columns = column_names


def DisplayTimeSeries(sub_num, column):
    fig, ax = plt.subplots(figsize=(12, 6))
    sub_df = whole_icn_df[whole_icn_df["sub_num"] == sub_num]
    sns.lineplot(data=sub_df, x=sub_df.index, y=column, ax=ax)
    ax.set_title(column)
    ax.set_ylim(-40, 40)
    plt.plot()
    plt.show()


for i in range(3):
    col_name = "col_" + str(i)
    DisplayTimeSeries("sub006", col_name)
