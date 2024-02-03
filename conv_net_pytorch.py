import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from six.moves import cPickle as pickle
from torchsummary import summary

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

if __name__ == "__main__":
    pickle_file = 'tensors_5_noiselevel.pickle'  # Adjust path as needed
    data_tensor1, data_tensor2, labels = load_data(pickle_file)
    train_dataset, test_dataset = create_datasets(data_tensor1, data_tensor2, labels)
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
