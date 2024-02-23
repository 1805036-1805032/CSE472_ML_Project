import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
import torch.nn as nn
import torch.optim as optim
import warnings


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


# Define the neural network
input_size = 5460
hidden_size = 128
hidden_size1 = 256
hidden_size2 = 128
output_size = 1


class ComplexNN(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(ComplexNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size2, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        out = self.sigmoid(out)
        return out

    def fit(self, X_train, y_train, num_epochs=100, batch_size=32):
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.parameters())

        for epoch in range(num_epochs):
            for i in range(0, len(X_train), batch_size):
                inputs = torch.tensor(X_train[i : i + batch_size], dtype=torch.float32)
                targets = torch.tensor(
                    y_train[i : i + batch_size], dtype=torch.float32
                ).view(-1, 1)

                # Forward pass
                outputs = self(inputs)
                loss = criterion(outputs, targets)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    def predict_proba(self, X):
        with torch.no_grad():
            inputs = torch.tensor(X, dtype=torch.float32)
            outputs = self.forward(inputs).numpy()
            return outputs

    def predict(self, X, threshold=0.5):
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)


# Perform cross-validation
def evaluate_model(
    X,
    y,
    metrics=("roc_auc", "accuracy", "recall", "precision", "f1"),
    n_splits=10,
    random_state=0,
):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # Initialize dictionaries to store scores for each metric
    scores = {metric: [] for metric in metrics}

    # Perform cross-validation
    for fold, (train_index, test_index) in enumerate(skf.split(X, y), 1):
        print(f"Fold {fold}:")

        # Split data into training and validation sets for this fold
        X_train, X_val = X[train_index], X[test_index]
        y_train, y_val = y[train_index], y[test_index]

        # Convert data to PyTorch tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)

        # Train the model
        model = ComplexNN(input_size, hidden_size1, hidden_size2, output_size)

        model.fit(X_train_tensor, y_train_tensor)

        # Predict y_preds_prob on the validation set
        y_proba = model.predict_proba(X_val_tensor)

        # Calculate evaluation metrics
        for metric in metrics:
            if metric == "roc_auc":
                score = roc_auc_score(y_val, y_proba)
            elif metric == "accuracy":
                y_pred = model.predict(X_val)
                score = accuracy_score(y_val, y_pred)
            elif metric == "precision":
                y_pred = model.predict(X_val)
                score = precision_score(y_val, y_pred)
            elif metric == "recall":
                y_pred = model.predict(X_val)
                score = recall_score(y_val, y_pred)
            elif metric == "f1":
                y_pred = model.predict(X_val)
                score = f1_score(y_val, y_pred)
            else:
                raise ValueError(f"Unknown metric: {metric}")

            print(f"{metric.capitalize()}: {score:.4f}")
            scores[metric].append(score)

    # Calculate and print average scores across folds
    avg_scores = {metric: np.mean(score_list) for metric, score_list in scores.items()}
    print("-" * 10)
    print("Average Scores:")
    for metric, score in avg_scores.items():
        print(f"{metric.capitalize()}: {score:.6f}")

    return avg_scores


scores = evaluate_model(np.array(X_train_full), np.array(y_train_full))

# done with cross validation

# train and submit
model = ComplexNN(input_size, hidden_size1, hidden_size2, output_size)
print(model)
model.fit(np.array(X_train_full), np.array(y_train_full))
y_preds_prob = model.predict_proba(np.array(X_test_full))
print("y_preds_prob shape:", y_preds_prob.shape)
print("y_preds_prob:", y_preds_prob[:5])


output_df = pd.DataFrame(
    {"ID": pd.Series(test_folder_names), "Predicted": pd.Series(y_preds_prob[:, 0])}
)
output_df.to_csv("submission.csv", index=False)
