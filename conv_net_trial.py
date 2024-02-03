import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle

print(tf.__version__)

# Load sample data
pickle_file = 'tensors_5_noiselevel.pickle'  # Update this path as necessary
with open(pickle_file, 'rb') as f:
    save = pickle.load(f)
    data_tensor1 = save['data_tensor1']
    data_tensor2 = save['data_tensor2']
    labels = save['label']
    del save  # Hint to help gc free up memory

subjectIDs = np.arange(150)  # Adjusted to use numpy for array operations

# Define functions for cross-validation, tensor randomization and normalization, and performance calculation

def create_train_and_test_folds(num_folds, subjects):
    n = np.ceil(len(subjects) / num_folds).astype(int)
    subjects = np.random.permutation(subjects)
    if len(subjects) != n * num_folds:
        s = np.zeros(n * num_folds, dtype=int)
        s[:len(subjects)] = subjects
        subjects = s
    IDs = subjects.reshape((n, num_folds))
    return IDs

def normalize_tensor(data_tensor):
    data_tensor -= np.mean(data_tensor)
    data_tensor /= np.max(np.abs(data_tensor))
    return data_tensor

def randomize_tensor(dataset, labels):
    permutation = np.random.permutation(labels.shape[0])
    shuffled_dataset = dataset[permutation, :, :, :]
    shuffled_labels = labels[permutation, :]
    return shuffled_dataset, shuffled_labels

def create_train_and_test_data(fold, IDs, subjectIDs, labels, data_tensor):
    num_labels = len(np.unique(labels))
    labels = (np.arange(num_labels) == labels[:, None]).astype(np.float32)
    
    testIDs = np.in1d(subjectIDs, IDs[:, fold])
    
    test_data = normalize_tensor(data_tensor[testIDs, :, :, :]).astype(np.float32)
    test_labels = labels[testIDs]
    
    train_data = normalize_tensor(data_tensor[~testIDs, :, :, :]).astype(np.float32)
    train_labels = labels[~testIDs]
    train_data, train_labels = randomize_tensor(train_data, train_labels)
    
    return train_data, train_labels, test_data, test_labels

def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])

# Initialize network parameters
numROI = 499
num_channels = 2
num_labels = 2
image_size = numROI
batch_size = 4
patch_size = image_size
depth = 64
num_hidden = 96
keep_pr = 0.6

combined_tensor = np.zeros((data_tensor1.shape[0], data_tensor1.shape[1], data_tensor1.shape[2], num_channels))

combined_tensor[:, :, :, 0] = normalize_tensor(data_tensor1[:, :, :, 0])
combined_tensor[:, :, :, 1] = normalize_tensor(data_tensor2[:, :, :, 0])

subjects = np.unique(subjectIDs)

num_folds = 10
IDs = create_train_and_test_folds(num_folds, subjects)

test_labs = []
test_preds = []

# Launch TensorFlow in each fold of cross-validation
for i in range(num_folds):
    train_data, train_labels, test_data, test_labels = create_train_and_test_data(i, IDs, subjectIDs, labels, combined_tensor)
    
    train_data = train_data[:, :image_size, :image_size, :]
    test_data = test_data[:, :image_size, :image_size, :]
    
    graph = tf.Graph()
    
    with graph.as_default():
        # Input data placeholders
        tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels))
        tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
        
        # Test data is a constant
        tf_test_dataset = tf.constant(test_data)
        
        # Network weight variables: Xavier initialization for better convergence in deep layers
        layer1_weights = tf.get_variable("layer1_weights", shape=[1, patch_size, num_channels, depth], initializer=tf.contrib.layers.xavier_initializer())
        layer1_biases = tf.Variable(tf.constant(0.001, shape=[depth]))
        layer2_weights = tf.get_variable("layer2_weights", shape=[patch_size, 1, depth, 2*depth], initializer=tf.contrib.layers.xavier_initializer())
        layer2_biases = tf.Variable(tf.constant(0.001, shape=[2*depth]))
        layer3_weights = tf.get_variable("layer3_weights", shape=[2*depth, num_hidden], initializer=tf.contrib.layers.xavier_initializer())
        layer3_biases = tf.Variable(tf.constant(0.01, shape=[num_hidden]))
        layer4_weights = tf.get_variable("layer4_weights", shape=[num_hidden, num_labels], initializer=tf.contrib.layers.xavier_initializer())
        layer4_biases = tf.Variable(tf.constant(0.01, shape=[num_labels]))
        
        # Convolutional network architecture
        def model(data, keep_pr):
            conv = tf.nn.conv2d(data, layer1_weights, [1, 1, 1, 1], padding='VALID')
            hidden = tf.nn.dropout(tf.nn.relu(conv + layer1_biases), keep_pr)
            conv = tf.nn.conv2d(hidden, layer2_weights, [1, 1, 1, 1], padding='VALID')
            hidden = tf.nn.dropout(tf.nn.relu(conv + layer2_biases), keep_pr)
            shape = hidden.get_shape().as_list()
            reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])
            hidden = tf.nn.dropout(tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases), keep_pr)
            return tf.matmul(hidden, layer4_weights) + layer4_biases
        
        # Calculate loss-function (cross-entropy) in training
        logits = model(tf_train_dataset, keep_pr)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf_train_labels))
        
        # Optimizer definition
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
        
        # Calculate predictions from training data
        train_prediction = tf.nn.softmax(logits)
        # Calculate predictions from test data (keep_pr of dropout is 1!)
        test_prediction = tf.nn.softmax(model(tf_test_dataset, 1))
        
        # Number of iterations
        num_steps = 20001
    
    # Start TensorFlow session
    with tf.Session(graph=graph) as session:
        tf.global_variables_initializer().run()
        print('Initialized')
        
        for step in range(num_steps):
            offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
            
            if offset == 0:  # If we've seen all train data at least once, re-randomize the order of instances
                train_data, train_labels = randomize_tensor(train_data, train_labels)
            
            # Create batch
            batch_data = train_data[offset:(offset + batch_size), :, :, :]
            batch_labels = train_labels[offset:(offset + batch_size), :]
            
            # Feed batch data to the placeholders
            feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}
            _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
            
            # At every 2000th step, give some feedback on the progress
            if step % 2000 == 0:
                print('Minibatch loss at step %d: %f' % (step, l))
                print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
        
        # Evaluate the trained model on the test data in the given fold
        test_pred = test_prediction.eval()
        print('Test accuracy: %.1f%%' % accuracy(test_pred, test_labels))
        
        # Save test predictions and labels of this fold to a list
        test_labs.append(test_labels)
        test_preds.append(test_pred)

# Create np.array to store all predictions and labels
l = test_labs[0]
p = test_preds[0]
# Iterate through the cross-validation folds
for i in range(1, num_folds):
    l = np.vstack((l, test_labs[i]))
    p = np.vstack((p, test_preds[i]))

# Calculate final accuracy
print('Test accuracy: %.1f%%' % accuracy(p, l))

# Save data
np.savez("predictions.npz", labels=l, predictions=p, splits=IDs)
