import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, precision_score, f1_score
import warnings

# Set seeds for reproducibility and ignore warnings
np.random.seed(0)
torch.manual_seed(0)
warnings.filterwarnings("ignore")

# Define paths to your dataset
PATH_TO_BP = "./psychosis_classification_with_rsfMRI/train/BP"
PATH_TO_SZ = "./psychosis_classification_with_rsfMRI/train/SZ"
PATH_TO_TEST = "./psychosis_classification_with_rsfMRI/test"

# Function to load the dataset
def load_dataset(path):
    folder_names = os.listdir(path)
    folders_paths = [os.path.join(path, x) for x in folder_names]
    data = []
    for path in folders_paths:
        fnc_array = np.load(os.path.join(path, "fnc.npy")).reshape(-1, 5460)  # Keep as 2D array for now
        data.append(fnc_array)
    return np.vstack(data), folder_names  # Stack for a unified numpy array

# Load training and test data
BP_data, _ = load_dataset(PATH_TO_BP)
SZ_data, _ = load_dataset(PATH_TO_SZ)
X_train_full = np.vstack([BP_data, SZ_data])
y_train_full = np.array([1] * len(BP_data) + [0] * len(SZ_data))  # 1 for BP, 0 for SZ

X_test_full, test_folder_names = load_dataset(PATH_TO_TEST)

# Splitting training data for validation
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=0)

# Define the CCNN model
class CCNN(nn.Module):
    def __init__(self, num_channels=1, depth=64, num_hidden=96, num_labels=1):
        super(CCNN, self).__init__()
        self.layer1 = nn.Conv2d(num_channels, depth, kernel_size=(1, 5460), stride=1, padding=0)
        self.layer2 = nn.Conv2d(depth, depth*2, kernel_size=(1, 1), stride=1, padding=0)
        self.fc1 = nn.Linear(depth*2, num_hidden)
        self.fc2 = nn.Linear(num_hidden, num_labels)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = x.view(x.size(0), -1)  # Flatten the tensor for the fully connected layer
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return self.sigmoid(x)

# Prepare data for training, validation, and testing
def prepare_data(X, y=None):
    X_reshaped = X.reshape(-1, 1, 1, 5460)  # Reshape to add channel dimension
    X_tensor = torch.tensor(X_reshaped, dtype=torch.float32)
    
    if y is not None:
        y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)
        dataset = TensorDataset(X_tensor, y_tensor)
    else:
        dataset = TensorDataset(X_tensor)
    
    return dataset

train_dataset = prepare_data(X_train, y_train)
val_dataset = prepare_data(X_val, y_val)
train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=32, shuffle=False)

test_dataset = prepare_data(X_test_full)
test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)

# Initialize model, loss function, and optimizer
model = CCNN()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters())

# Training loop
num_epochs = 100
model.train()
for epoch in range(num_epochs):
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

# Evaluation on the validation set
model.eval()
y_true = []
y_pred = []
with torch.no_grad():
    for X_batch, y_batch in val_loader:
        y_val_pred = model(X_batch)
        y_true.extend(y_batch.numpy())
        y_pred.extend(y_val_pred.numpy())

y_true = np.array(y_true).flatten()
y_pred = np.array(y_pred).flatten()
y_pred_label = (y_pred > 0.5).astype(int)

# Calculate and print evaluation metrics
auc = roc_auc_score(y_true, y_pred)
accuracy = accuracy_score(y_true, y_pred_label)
recall = recall_score(y_true, y_pred_label)
precision = precision_score(y_true, y_pred_label)
f1 = f1_score(y_true, y_pred_label)

print(f'AUC: {auc:.4f}')
print(f'Accuracy: {accuracy:.4f}')
print(f'Recall: {recall:.4f}')
print(f'Precision: {precision:.4f}')
print(f'F1 Score: {f1:.4f}')

# Prediction
predictions = []
with torch.no_grad():
    for X_batch in test_loader:
        y_test_pred = model(X_batch[0])
        predictions.extend(y_test_pred.squeeze().numpy())

# Prepare submission
submission_df = pd.DataFrame({
    "ID": test_folder_names,
    "Predicted": predictions
})
submission_df.to_csv("submission_ccnn.csv", index=False)
print("Submission file created.")
