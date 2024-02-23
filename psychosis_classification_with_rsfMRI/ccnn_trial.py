import os
import numpy as np
import pandas as pd
import torch
import warnings
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from six.moves import cPickle as pickle
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold

np.random.seed(0)
torch.manual_seed(0)
warnings.filterwarnings("ignore")

# loading the dataset
PATH_TO_BP = "./psychosis_classification_with_rsfMRI/train/BP"
PATH_TO_SZ = "./psychosis_classification_with_rsfMRI/train/SZ"
PATH_TO_TEST = "./psychosis_classification_with_rsfMRI/test"

BP_folder_names = os.listdir(PATH_TO_BP)
SZ_folder_names = os.listdir(PATH_TO_SZ)
BP_folders_paths = [os.path.join(PATH_TO_BP, x) for x in BP_folder_names]
SZ_folders_paths = [os.path.join(PATH_TO_SZ, x) for x in SZ_folder_names]

test_folder_names = os.listdir(PATH_TO_TEST)
test_folder_paths = [os.path.join(PATH_TO_TEST, x) for x in test_folder_names]

class CCNN(nn.Module):
    def __init__(self, num_channels=1, depth=64, num_hidden=96):
        super(CCNN, self).__init__()
        self.layer1 = nn.Conv2d(num_channels, depth, kernel_size=(1, 105), padding=0)
        self.layer2 = nn.Conv2d(depth, 2*depth, kernel_size=(105, 1), padding=0)
        self.fc1_input_size = 2 * depth * 1 * 1  # Adjusted based on the output size of layer2
        self.fc1 = nn.Linear(self.fc1_input_size, num_hidden)
        self.fc2 = nn.Linear(num_hidden, 1)
        
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = x.view(-1, self.fc1_input_size)  # Flatten the tensor for the fully connected layer
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def fit(self, X_train, y_train, num_epochs=100, batch_size=32):
        # Ensure X_train and y_train are numpy arrays for consistency
        X_train = np.array(X_train, dtype=np.float32)
        y_train = np.array(y_train, dtype=np.float32)

        # Convert numpy arrays to PyTorch tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)  

        # Create a TensorDataset
        dataset = TensorDataset(X_train_tensor, y_train_tensor)

        # Create a DataLoader
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        for epoch in range(num_epochs):
            self.train()
            for data, labels in train_loader:
                self.optimizer.zero_grad()
                outputs = self(data)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
            print(f'Epoch {epoch+1}, Loss: {loss.item()}')


    def predict_proba(self, X):
        self.eval()
        with torch.no_grad():
            outputs = self(X)
            probabilities = torch.sigmoid(outputs)
            return probabilities

    def predict(self, X, threshold=0.5):
        proba = self.predict_proba(X)
        predictions = (proba >= threshold).float()
        return predictions

def evaluate_model(X, y, metrics=("roc_auc", "accuracy", "recall", "precision", "f1"), n_splits=10, random_state=0):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    # Convert X and y to numpy arrays if they are not already
    X_np = np.array(X)
    y_np = np.array(y)

    # Initialize dictionaries to store scores for each metric
    scores = {metric: [] for metric in metrics}

    # Perform cross-validation
    for fold, (train_index, test_index) in enumerate(skf.split(X_np, y_np), 1):
        print(f"Fold {fold}:")

        # Split data into training and validation sets for this fold
        X_train, X_val = X_np[train_index], X_np[test_index]
        y_train, y_val = y_np[train_index], y_np[test_index]

        # Convert data to PyTorch tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

        # Re-initialize the model for each fold to reset weights
        model = CCNN()
        model.fit(X_train_tensor, y_train_tensor, num_epochs=10)  # Adjust epochs as needed

        # Predict on the validation set
        y_proba_tensor = model.predict_proba(X_val_tensor)
        y_proba = y_proba_tensor.numpy().squeeze()  # Convert to numpy array and squeeze if necessary

        # Convert probabilities to binary predictions based on a threshold (default 0.5)
        y_pred = model.predict(X_val_tensor).numpy()

        # Calculate evaluation metrics
        for metric in metrics:
            if metric == "roc_auc":
                score = roc_auc_score(y_val, y_proba)
            elif metric == "accuracy":
                score = accuracy_score(y_val, y_pred)
            elif metric == "precision":
                score = precision_score(y_val, y_pred)
            elif metric == "recall":
                score = recall_score(y_val, y_pred)
            elif metric == "f1":
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

print("Training Dataset\n", "-" * 10)
print("total BP:", len(BP_folder_names))
print("total SZ:", len(SZ_folder_names))
TOTAL_ENTRIES = len(BP_folder_names) + len(SZ_folder_names)
print("Total_entries:", TOTAL_ENTRIES)

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


X_train_full = np.array(X_train_full).astype(np.float32)
X_train_full = np.expand_dims(X_train_full, 1)  # Add channel dimension
y_train_full = np.array(y_train_full).astype(np.float32)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train_full, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_full, dtype=torch.float32).view(-1, 1)

# Initialize and train the model
evaluate_model(X_train_full, y_train_full)


model = CCNN()
print(model)
model.fit(np.array(X_train_full), np.array(y_train_full))

X_test_tensor = torch.tensor(X_test_full, dtype=torch.float32).unsqueeze(1)

y_preds_prob = model.predict_proba(X_test_tensor)

print("y_preds_prob shape:", y_preds_prob.shape)
print("y_preds_prob:", y_preds_prob[:5])


output_df = pd.DataFrame(
    {"ID": pd.Series(test_folder_names), "Predicted": pd.Series(y_preds_prob[:, 0])}
)
output_df.to_csv("submission_ccnn.csv", index=False)