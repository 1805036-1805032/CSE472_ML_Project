import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC


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


# creating test adn train dataset
X_train_full = []
y_train_full = []
X_test_full = []

for path in BP_folders_paths:
    fnc_array = np.load(os.path.join(path, "fnc.npy"))
    fnc_array = fnc_array.reshape(1, 5460)[0].tolist()
    X_train_full.append(fnc_array)
    y_train_full.append(1)


for path in SZ_folders_paths:
    fnc_array = np.load(os.path.join(path, "fnc.npy"))
    fnc_array = fnc_array.reshape(1, 5460)[0].tolist()
    X_train_full.append(fnc_array)
    y_train_full.append(0)

X_train_full = pd.DataFrame(X_train_full)
y_train_full = pd.Series(y_train_full)


for path in test_folder_paths:
    fnc_array = np.load(os.path.join(path, "fnc.npy"))
    fnc_array = fnc_array.reshape(1, 5460)[0].tolist()
    X_test_full.append(fnc_array)


print("-" * 10)
print("X_train_full:", X_train_full.shape)
print("y_train_full:", y_train_full.shape)
print("X_test_full:", len(X_test_full))


from sklearn.svm import SVC


class Model:
    def __init__(self) -> None:
        self.model = SVC(random_state=0, probability=True)

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
