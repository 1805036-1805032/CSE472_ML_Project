import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from six.moves import cPickle as pickle

def load_data(pickle_file):
    with open(pickle_file, 'rb') as f:
        save = pickle.load(f)
        data_tensor1 = save['data_tensor1']
        data_tensor2 = save['data_tensor2']
        labels = save['label']
        del save
    return data_tensor1, data_tensor2, labels

def normalize_tensor(data_tensor):
    data_tensor -= np.mean(data_tensor, axis=(1, 2), keepdims=True)
    data_tensor /= np.max(np.abs(data_tensor), axis=(1, 2), keepdims=True)
    return data_tensor

def prepare_data(data_tensor1, data_tensor2, labels):
    data_tensor1 = normalize_tensor(data_tensor1)
    data_tensor2 = normalize_tensor(data_tensor2)
    combined_tensor = np.stack((data_tensor1, data_tensor2), axis=3)
    return combined_tensor, labels

def create_model():
    model = models.Sequential([
        layers.Conv2D(64, (1, 499), activation='relu', input_shape=(499, 499, 2)),
        layers.Conv2D(128, (499, 1), activation='relu'),
        layers.Flatten(),
        layers.Dense(96, activation='relu'),
        layers.Dense(2, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

pickle_file = 'tensors_5_noiselevel.pickle'  # Update this path
data_tensor1, data_tensor2, labels = load_data(pickle_file)
combined_tensor, labels = prepare_data(data_tensor1, data_tensor2, labels)

X_train, X_test, y_train, y_test = train_test_split(combined_tensor, labels, test_size=0.2, random_state=42)

model = create_model()
model.summary()

model.fit(X_train, y_train, epochs=10, batch_size=4, validation_split=0.2)
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_acc}, Test loss: {test_loss}')
