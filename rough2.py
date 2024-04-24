import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import plotly.graph_objects as go
import time

image_directory = 'datasets/'

no_tumor_images = os.listdir(image_directory + 'no/')
yes_tumor_images = os.listdir(image_directory + 'yes/')
dataset = []
label = []

INPUT_SIZE = 64

for image_name in no_tumor_images:
    if image_name.split('.')[1] == 'jpg':
        image = cv2.imread(image_directory + 'no/' + image_name)
        image = cv2.resize(image, (INPUT_SIZE, INPUT_SIZE))
        dataset.append(image.flatten())
        label.append(0)

for image_name in yes_tumor_images:
    if image_name.split('.')[1] == 'jpg':
        image = cv2.imread(image_directory + 'yes/' + image_name)
        image = cv2.resize(image, (INPUT_SIZE, INPUT_SIZE))
        dataset.append(image.flatten())
        label.append(1)

dataset = np.array(dataset)
label = np.array(label)

accuracies = []


# Repeat for 100 iterations
for i in range(10):
    # Split the dataset into train and test sets
    x_train, x_test, y_train, y_test = train_test_split(dataset, label, test_size=0.2, random_state=i)
    
    # Flatten the image data
    x_train_flat = x_train.reshape(x_train.shape[0], -1)
    x_test_flat = x_test.reshape(x_test.shape[0], -1)
    
    # Initialize SVM model
    svm_model = SVC(kernel='linear', C=1.0)
    
    # Train the SVM model
    svm_model.fit(x_train_flat, y_train)
    
    # Predict labels for test set
    y_pred = svm_model.predict(x_test_flat)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Append accuracy to list
    accuracies.append(accuracy)

# Find the highest accuracy and its corresponding iteration
max_accuracy = max(accuracies)
max_accuracy_iteration = accuracies.index(max_accuracy) + 1  # Add 1 to get the iteration number

print("List of Accuracies:")
for i, acc in enumerate(accuracies, start=1):
    print(f"Iteration {i}: Accuracy = {acc:.4f}")

print(f"\nHighest accuracy from 100 iterations: {max_accuracy:.4f}")
print(f"Occurred at iteration: {max_accuracy_iteration}")

# Plot the accuracies using Plotly
fig = go.Figure()
fig.add_trace(go.Scatter(x=np.arange(1, 101), y=accuracies, mode='lines+markers', name='Accuracy'))
fig.add_trace(go.Scatter(x=[max_accuracy_iteration], y=[max_accuracy], mode='markers', name=f'Highest Accuracy: {max_accuracy:.4f}', marker=dict(color='red', size=10)))
fig.update_layout(title='Accuracy of SVM Model (100 iterations)',
                  xaxis_title='Iteration',
                  yaxis_title='Accuracy',
                  legend=dict(x=0.02, y=0.98))
fig.show()


