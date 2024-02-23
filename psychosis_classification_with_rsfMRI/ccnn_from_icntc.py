import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
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
    upper_triangular = np.triu(connectome, k=1)[np.triu_indices(105, k=1)]  
    X_train_full.append(upper_triangular)
    # X_train_full.append(connectome)
    y_train_full.append(1)


for i in range(len(SZ_folders_paths)):
    icn_tc_array = np.load(os.path.join(SZ_folders_paths[i], "icn_tc.npy"))
    connectome = calculate_connectome(icn_tc_array)
    upper_triangular = np.triu(connectome, k=1)[np.triu_indices(105, k=1)]  
    X_train_full.append(upper_triangular)
    # X_train_full.append(connectome)
    y_train_full.append(0)


for i in range(len(test_folder_paths)):
    icn_tc_array = np.load(os.path.join(test_folder_paths[i], "icn_tc.npy"))
    connectome = calculate_connectome(icn_tc_array)
    upper_triangular = np.triu(connectome, k=1)[np.triu_indices(105, k=1)] 
    X_test_full.append(upper_triangular)
    # X_train_full.append(connectome)


X_train_full = np.array(X_train_full)
X_test_full = np.array(X_test_full)
y_train_full = pd.Series(y_train_full)


print("-" * 10)
print("X_train_full:", X_train_full.shape)
print("y_train_full:", y_train_full.shape)
print("X_test_full:", X_test_full.shape)


class Model:
    def __init__(self) -> None:
        self.model = LogisticRegression(max_iter = 10000, random_state = 42)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def evaluateAUC(self, X, y):
        scores = cross_val_score(self.model, X, y, cv=10, scoring="roc_auc")
        return scores.mean(), scores.std()

    def evaluateAccuracy(self, X, y):
        scores = cross_val_score(self.model, X, y, cv=10, scoring="accuracy")
        return scores.mean(), scores.std()

    def evaluateRecall(self, X, y):
        scores = cross_val_score(self.model, X, y, cv=10, scoring="recall")
        return scores.mean(), scores.std()

    def evaluatePrecision(self, X, y):
        scores = cross_val_score(self.model, X, y, cv=10, scoring="precision")
        return scores.mean(), scores.std()

    def evaluateF1(self, X, y):
        scores = cross_val_score(self.model, X, y, cv=10, scoring="f1")
        return scores.mean(), scores.std()


model = Model()


print("-" * 10)
print("Model: ", type(model))
auc, auc_std = model.evaluateAUC(X_train_full, y_train_full)
print("AUC: %.6f (%.6f)" % (auc, auc_std))
accuracy, accuracy_std = model.evaluateAccuracy(X_train_full, y_train_full)
print("Accuracy: %.6f (%.6f)" % (accuracy, accuracy_std))
recall, recall_std = model.evaluateRecall(X_train_full, y_train_full)
print("Recall: %.6f (%.6f)" % (recall, recall_std))
precision, precision_std = model.evaluatePrecision(X_train_full, y_train_full)
print("Precision: %.6f (%.6f)" % (precision, precision_std))
f1, f1_std = model.evaluateF1(X_train_full, y_train_full)
print("F1: %.6f (%.6f)" % (f1, f1_std))


# training and submission
model.fit(X_train_full, y_train_full)
y_preds = model.predict(X_test_full)
y_preds_prob = model.predict_proba(X_test_full)


output_df = pd.DataFrame(
    {"ID": pd.Series(test_folder_names), "Predicted": pd.Series(y_preds_prob[:, 1])}
)
output_df.to_csv("submission.csv", index=False)
