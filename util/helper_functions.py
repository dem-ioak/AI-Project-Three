# Helper Functions
import numpy as np
import random
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def softmax(z):
    z_new = np.exp(z[:])
    return z_new / sum(z_new)

def relu(z):
    return np.maximum(0, z)

def train_test_validation_split(X, y, train_size = .8, test_size = .1, validation_size = .1):
    if train_size + test_size + validation_size != 1:
        print("Error: train + test + validation don't add up to 1.")
        return
    
    examples, _ = X.shape
    # Randomize Indices
    index_array = np.arange(examples)
    random.shuffle(index_array)

    # Split up data
    num_training = int(examples * train_size)
    num_testing = int(examples * test_size)
    train_indices = index_array[:num_training]
    test_indices = index_array[num_training:(num_training + num_testing)]
    validation_indices = index_array[(num_training + num_testing):]
    
    X_train = X[train_indices]
    y_train = y[train_indices]
    X_test = X[test_indices]
    y_test = y[test_indices]
    X_validation = X[validation_indices]
    y_validation = y[validation_indices]

    return X_train, y_train, X_test, y_test, X_validation, y_validation
    

def one_hot_encode(image_data):
    """One-Hot encode matrix of examples"""
    if not isinstance(image_data, np.ndarray):
        image_data = np.array(image_data)

    num_colors = 4
    if image_data.ndim == 1:
        examples = image_data.shape[0]
        encoded_shape = (examples, num_colors)
        encoded_image = np.zeros(encoded_shape)
        for i in range(examples):
            for color in range(num_colors):
                if color == image_data[i]:
                    encoded_image[i, color] = 1
    if image_data.ndim == 3:
        examples, rows, cols = image_data.shape
        encoded_shape = (examples, rows, cols, num_colors)
        encoded_image = np.zeros(encoded_shape)
        for i in range(examples):
            for j in range(rows):
                for k in range(cols):
                    for l in range(num_colors):
                        if l == image_data[i, j, k]:
                            encoded_image[i, j, k, l] = 1
                    
    return encoded_image

def apply_convolution(image_data, step_size = 1):
    """Apply convolution"""
    num_images, W, _, C = image_data.shape
    
    # Kernel Initialization
    kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    K = 3

    output_size = ((W - K) // step_size) + 1
    output = np.zeros((num_images, output_size, output_size, C))

    # For every image
    for n in range(num_images):

        # For every element of the feature map
        for i in range(output_size):
            for j in range(output_size):
                region = image_data[n, i : i + K, j : j + K] # (3, 3, 4) - pixel subregion
                for c in range(C):
                    output[n, i, j, c] += np.sum(kernel * region[:, :, c])
    return output

def apply_pooling(image_data, P, mode_ = "max"):
    if mode_ not in ["max", "mean"]:
        return -1

    num_images, W, _, C = image_data.shape
    output_size = W // P
    output = np.zeros((num_images, output_size, output_size, C))

    for n in range(num_images):
        for i in range(0, W - P + 1, P):
            for j in range(0, W - P + 1, P):
                region = image_data[n, i : i + P, j : j + P, :] # (3, 3, 4) - pixel subregion
                for c in range(C):
                    func_ = np.max if mode_ == "max" else np.mean
                    output[n, i // P, j // P, c] = func_(region[:, :, c])
    return output

def row_col_features(image_data):
    col_features = np.sum(image_data, axis=1)
    row_features = np.sum(image_data, axis=2)
    return np.concatenate([col_features, row_features], axis=1)

def preprocess_data_t1(image_data, label_data, processing=None):
    """Uses neither convolution or row_col_features by default (only one_hot encoding)
    
    - Use "convolution" for convolution
    - Use "rc for row_col_features
    - Use "both" for both of those features"""
    image_data = one_hot_encode(image_data)
    label_data = np.array(label_data)
    flattened_raw = image_data.reshape(image_data.shape[0], -1)
    output = np.concatenate([flattened_raw], axis = 1)
    
    if processing in ("convolution", "both"):
        convolved = apply_convolution(image_data, 1)
        pooled = apply_pooling(convolved, 2, "mean")
        flattened_pooled = pooled.reshape(pooled.shape[0], -1)
        flattened_pooled_squared = np.square(flattened_pooled)  
        output = np.concatenate([flattened_raw, flattened_pooled, flattened_pooled_squared], axis = 1)
    if processing in ("rc", "both"):
        rc_features = row_col_features(image_data)
        flattened_rc = rc_features.reshape(rc_features.shape[0], -1)
        flattened_rc_squared = np.square(flattened_rc)
        output = np.concatenate([flattened_raw, flattened_rc, flattened_rc_squared], axis = 1)
    if processing == "both":
        output = np.concatenate([flattened_raw, flattened_pooled, flattened_pooled_squared, flattened_rc, flattened_rc_squared], axis = 1)
    if processing == "noraw":
        convolved = apply_convolution(image_data, 1)
        pooled = apply_pooling(convolved, 2, "mean")
        flattened_pooled = pooled.reshape(pooled.shape[0], -1)
        flattened_pooled_squared = np.square(flattened_pooled)  
        
        rc_features = row_col_features(image_data)
        flattened_rc = rc_features.reshape(rc_features.shape[0], -1)
        flattened_rc_squared = np.square(flattened_rc)
        
        output = np.concatenate([flattened_pooled, flattened_pooled_squared, flattened_rc, flattened_rc_squared], axis = 1)

    flattened_data = np.c_[np.ones(len(output)), output] 
    return flattened_data, label_data

def preprocess_data_t2(data, processing=None):
    """Task 2 preprocessing"""
    image_data, is_safe_data, third_wire = data.image_data, data.is_safe, data.third_wires
    dangerous_images, dangerous_third_wire = [], []
    for i in range(len(is_safe_data)):
        dangerous_images.append(image_data[i])
        dangerous_third_wire.append(third_wire[i])
        
    dangerous_images = one_hot_encode(dangerous_images)
    dangerous_third_wire = one_hot_encode(dangerous_third_wire)
    output = dangerous_images.reshape(dangerous_images.shape[0], -1)
    flattened_data = np.c_[np.ones(len(output)), output] 
    return flattened_data, dangerous_third_wire


def plot_data(title, x_label, y_label, data):
    """Data should be either a list or tuple, where the list is a list of tuples.
    
    Tuple format: Data to be plotted, name of data"""
    fig, ax = plt.subplots()
    
    if isinstance(data, list):
        for d in data:
            ax.plot(d[0], label = d[1])
    else:
        ax.plot(data[0], label = data[1])
    ax.set_title(title)
    ax.set_xlabel(x_label) 
    ax.set_ylabel(y_label)
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.show()
    