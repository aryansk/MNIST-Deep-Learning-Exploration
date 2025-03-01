# -*- coding: utf-8 -*-
"""Untitled11.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1_Ycy3brwmUVdDwiARhDvD06Ev9jzWrlM
"""

# Import necessary libraries
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tensorflow.keras.datasets import mnist  # Newly added import for Keras dataset

# Load the MNIST dataset from Keras
(X_sklearn, y_sklearn), (_, _) = mnist.load_data()

# Reshape the images into a 2D array (samples, features) before scaling
X_sklearn = X_sklearn.reshape(X_sklearn.shape[0], -1)
# X_sklearn.shape[0] keeps the number of samples
# -1 automatically calculates the number of features (28*28 = 784)


# Display 4 images from the Keras MNIST dataset
plt.figure(figsize=(10, 5))
for i in range(4):
    plt.subplot(1, 4, i+1)
    # Reshape the image back to 28x28 for display
    plt.imshow(X_sklearn[i].reshape(28, 28), cmap='gray')
    plt.title(f"Label: {y_sklearn[i]}")
    plt.axis('off')
plt.tight_layout()
plt.show()

# ... (Rest of your code) ...
# Q2: Print the shape of the input data and target data
print(f"Input data shape (sklearn): {X_sklearn.shape}")  # Expected (70000, 784)
print(f"Target data shape (sklearn): {y_sklearn.shape}")  # Expected (70000,)

# Q3: Display the top 10 images using matplotlib from Scikit-learn dataset
for i in range(10):
    image = X_sklearn[i].reshape(28, 28)
    plt.subplot(2, 5, i+1)
    plt.imshow(image, cmap='gray')
    plt.axis('off')
plt.show()

# Q4: Scale the data (standardizing features) for sklearn dataset
scaler = StandardScaler()
X_scaled_sklearn = scaler.fit_transform(X_sklearn)

# Q6: Split the dataset into 80% training and 20% testing
X_train_sklearn, X_test_sklearn, y_train_sklearn, y_test_sklearn = train_test_split(X_scaled_sklearn, y_sklearn, test_size=0.2, random_state=42)

# Q7: Train FFN with one hidden layer and 64 neurons, and set max_iter to 10
mlp = MLPClassifier(hidden_layer_sizes=(64,), max_iter=10, random_state=42)
mlp.fit(X_train_sklearn, y_train_sklearn)

# Q8: Predict on the test set and print accuracy, precision, recall, and F1 scores
y_pred_sklearn = mlp.predict(X_test_sklearn)
accuracy = accuracy_score(y_test_sklearn, y_pred_sklearn)
precision, recall, f1, _ = precision_recall_fscore_support(y_test_sklearn, y_pred_sklearn, average='macro')

print(f"Accuracy (sklearn dataset): {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

# Q9: Compare different train-test splits and analyze accuracy
splits = [(0.6, 0.4), (0.75, 0.25), (0.8, 0.2), (0.9, 0.1)]
for train_size, test_size in splits:
    X_train_sklearn, X_test_sklearn, y_train_sklearn, y_test_sklearn = train_test_split(X_scaled_sklearn, y_sklearn, train_size=train_size, test_size=test_size, random_state=42)
    mlp.fit(X_train_sklearn, y_train_sklearn)
    y_pred_sklearn = mlp.predict(X_test_sklearn)
    accuracy = accuracy_score(y_test_sklearn, y_pred_sklearn)
    print(f"Train-Test Split ({train_size*100}-{test_size*100}): Accuracy = {accuracy}")

# Q10: Playing with iterations
iterations = [20, 50, 100, 150, 200]
for iter in iterations:
    mlp = MLPClassifier(hidden_layer_sizes=(64,), max_iter=iter, random_state=42)
    mlp.fit(X_train_sklearn, y_train_sklearn)
    y_pred_sklearn = mlp.predict(X_test_sklearn)
    accuracy = accuracy_score(y_test_sklearn, y_pred_sklearn)
    print(f"Iterations {iter}: Accuracy = {accuracy}")

cimport tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
import time

# Load the MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Preprocessing: Normalize the data to [0, 1] range and reshape it
X_train = X_train.reshape((X_train.shape[0], 28 * 28)) / 255.0
X_test = X_test.reshape((X_test.shape[0], 28 * 28)) / 255.0

# Convert labels to one-hot encoding
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Function to build and compile model
def build_model(hidden_layers, nodes_per_layer, activation_function='relu'):
    model = models.Sequential()
    model.add(layers.InputLayer(input_shape=(28 * 28,)))  # Input layer

    # Add hidden layers
    for _ in range(hidden_layers):
        model.add(layers.Dense(nodes_per_layer, activation=activation_function))

    # Output layer (10 neurons for 10 digit classes)
    model.add(layers.Dense(10, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Function to train and evaluate model
def train_and_evaluate(hidden_layers, nodes_per_layer, epochs, activation_function='relu'):
    model = build_model(hidden_layers, nodes_per_layer, activation_function)

    # Measure training time
    start_time = time.time()
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_data=(X_test, y_test), verbose=0)
    training_time = time.time() - start_time

    # Evaluate model on test data
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)

    # Print results
    print(f"Hidden Layers: {hidden_layers}, Nodes per Layer: {nodes_per_layer}, Epochs: {epochs}")
    print(f"Test Accuracy: {test_accuracy:.4f}, Training Time: {training_time:.4f} seconds")
    print(f"Number of Parameters: {model.count_params()}")

    return history

# --- PART 2 TASKS ---

# 1. Number of Nodes
nodes = [4, 32, 64, 128, 512, 2056]
for n in nodes:
    train_and_evaluate(hidden_layers=1, nodes_per_layer=n, epochs=10)

# 2. Number of Layers
layers_list = [4, 5, 6, 8, 16]
for l in layers_list:
    train_and_evaluate(hidden_layers=l, nodes_per_layer=64, epochs=10)

# Run the same models for 30 epochs to observe changes
print("Running models with 30 epochs...")
for l in layers_list:
    train_and_evaluate(hidden_layers=l, nodes_per_layer=64, epochs=30)

# 3. Layer-node Combinations (explore different structures)
print("Layer-node combinations")
layer_node_combinations = [(3, 64), (4, 128), (2, 256), (5, 32)]
for (l, n) in layer_node_combinations:
    train_and_evaluate(hidden_layers=l, nodes_per_layer=n, epochs=10)

# 4. Input Size (4 hidden layers, 256 nodes each, ReLU activation)
print("Input size exploration")
train_and_evaluate(hidden_layers=4, nodes_per_layer=256, epochs=10)

# 5. Dataset Split - already done in part 1, if needed, adapt to train/evaluate with different sizes

# 6. Activation Functions (Sigmoid, Tanh, ReLU)
activation_functions = ['sigmoid', 'tanh', 'relu']
for af in activation_functions:
    print(f"Using {af} activation function")
    train_and_evaluate(hidden_layers=4, nodes_per_layer=64, epochs=10, activation_function=af)

# Run the same models for 30 epochs
print("Activation functions - 30 epochs")
for af in activation_functions:
    train_and_evaluate(hidden_layers=4, nodes_per_layer=64, epochs=30, activation_function=af)

# 7. Activation Function Combinations
print("Activation function combinations")
def build_custom_activation_model():
    model = models.Sequential()
    model.add(layers.InputLayer(input_shape=(28 * 28,)))
    model.add(layers.Dense(32, activation='sigmoid'))  # Layer 1
    model.add(layers.Dense(32, activation='relu'))     # Layer 2
    model.add(layers.Dense(32, activation='tanh'))     # Layer 3
    model.add(layers.Dense(10, activation='softmax'))  # Output layer
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Train the custom activation model
model = build_custom_activation_model()
start_time = time.time()
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test), verbose=0)
training_time = time.time() - start_time
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Custom Activation Model - Test Accuracy: {test_accuracy:.4f}, Training Time: {training_time:.4f} seconds")
print(f"Number of Parameters: {model.count_params()}")