import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from six.moves import cPickle as pickle
from torchsummary import summary



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


class CCNN(nn.Module):
    def __init__(self, num_channels=2, depth=64, num_hidden=96, num_labels=2):
        super(CCNN, self).__init__()
        self.layer1 = nn.Conv2d(num_channels, depth, kernel_size=(1, 499), padding=0)
        self.layer2 = nn.Conv2d(depth, 2*depth, kernel_size=(499, 1), padding=0)
        self.fc1_input_size = 2 * depth
        self.fc1 = nn.Linear(self.fc1_input_size, num_hidden)
        self.fc2 = nn.Linear(num_hidden, num_labels)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = x.view(-1, self.fc1_input_size)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def load_data(pickle_file):
    with open(pickle_file, 'rb') as f:
        save = pickle.load(f)
        data_tensor1 = np.squeeze(np.array(save['data_tensor1']), axis=-1)
        data_tensor2 = np.squeeze(np.array(save['data_tensor2']), axis=-1)
        labels = np.array(save['label'])
        del save
    return data_tensor1, data_tensor2, labels

def normalize_tensor(data_tensor):
    data_tensor -= np.mean(data_tensor, axis=(1, 2), keepdims=True)
    data_tensor /= np.max(np.abs(data_tensor), axis=(1, 2), keepdims=True)
    return data_tensor

def create_datasets(data_tensor1, data_tensor2, labels, test_split=0.2):
    data_tensor1 = normalize_tensor(data_tensor1)
    data_tensor2 = normalize_tensor(data_tensor2)
    combined_tensor = np.stack([data_tensor1, data_tensor2], axis=1)
    dataset = TensorDataset(torch.tensor(combined_tensor, dtype=torch.float32), torch.tensor(labels, dtype=torch.long))
    train_size = int((1 - test_split) * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    return train_dataset, test_dataset

def train_model(model, train_loader, device, num_epochs=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(num_epochs):
        model.train()
        for data, labels in train_loader:
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')

def test_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Test Accuracy: {100 * correct / total:.2f} %')


X_train, X_test, y_train, y_test = train_test_split(X_train_full, y_train_full, test_size=0.3, random_state=42)


X_train_tensor = torch.tensor(X_train.values, dtype=torch.float).view(-1, 1, 5460, 1)  # Reshape to [N, C, H, W] format
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float).view(-1, 1, 5460, 1)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)


train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)


train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CCNN().to(device)

# Model summary with torchsummary, ensure your environment supports this or remove if causing issues
try:
    summary(model, input_size=(2, 499, 499))
except Exception as e:
    print("Error generating model summary:", e)

# Train and test the model
train_model(model, train_loader, device, num_epochs=10)
test_model(model, test_loader, device)
