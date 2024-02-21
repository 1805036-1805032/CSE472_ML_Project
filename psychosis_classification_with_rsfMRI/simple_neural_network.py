import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
import torch
import torch.nn as nn
import torch.optim as optim

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


input_size = 5460  # Assuming 5460 features
hidden_size = 64   # You can adjust this according to your needs
hidden_size1 = 128
hidden_size2 = 64
output_size = 1    # For binary classification

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
                inputs = torch.tensor(X_train[i:i+batch_size], dtype=torch.float32)
                targets = torch.tensor(y_train[i:i+batch_size], dtype=torch.float32).view(-1, 1)
                
                # Forward pass
                outputs = self(inputs)
                loss = criterion(outputs, targets)
                
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            if (epoch+1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
    def predict_proba(self, X):
        with torch.no_grad():
            inputs = torch.tensor(X, dtype=torch.float32)
            outputs = self.forward(inputs).numpy()
            return outputs
    
    def predict(self, X, threshold=0.5):
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)
    


# Perform cross-validation
# Combine training and test data for cross-validation
X_full = np.concatenate((X_train_full, X_test_full), axis=0)
y_full = np.concatenate((y_train_full, np.zeros(len(X_test_full))), axis=0)  # Assuming test data labels are unknown (set to 0)

# Define 10-fold cross-validation
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Initialize lists to store ROC AUC scores
roc_auc_scores = []

for fold, (train_index, test_index) in enumerate(skf.split(X_full, y_full), 1):
    print(f"Fold {fold}:")
    
    # Split data into training and validation sets for this fold
    X_train, X_val = X_full[train_index], X_full[test_index]
    y_train, y_val = y_full[train_index], y_full[test_index]
    
    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    
    # Instantiate the model
    model = ComplexNN(input_size, hidden_size1, hidden_size2, output_size)
    
    # Train the model
    model.fit(X_train_tensor, y_train_tensor)
    
    # Predict probabilities on the validation set
    y_proba = model.predict_proba(X_val_tensor)
    
    # Calculate ROC AUC score
    roc_auc = roc_auc_score(y_val, y_proba)
    print(f"ROC AUC: {roc_auc:.4f}")
    
    # Store ROC AUC score
    roc_auc_scores.append(roc_auc)

# Calculate and print average ROC AUC score across folds
print("Average ROC AUC Score:", np.mean(roc_auc_scores))

## Done with cross-validation, now train the model on the full training data and make predictions on the test data
    

# Instantiate the model
model = ComplexNN(input_size, hidden_size1, hidden_size2, output_size)

# training and submission
X_train = np.array(X_train_full)
y_train = np.array(y_train_full)
model.fit(X_train, y_train)

# Example usage of prediction
# Assuming X_test_full is your test data
X_test = np.array(X_test_full)

# Example usage of predict_proba
probabilities = model.predict_proba(X_test)
print("Probabilities:", probabilities)
print("Probabilities shape:", probabilities.shape)

output_df = pd.DataFrame(
    {"ID": pd.Series(test_folder_names), "Predicted": pd.Series(probabilities[:, 0])}
)
output_df.to_csv("submission.csv", index=False)