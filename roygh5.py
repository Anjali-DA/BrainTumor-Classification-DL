#new code for CNN

import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import plotly.graph_objects as go

# Constants
image_directory = 'datasets/'
INPUT_SIZE = 64
ITERATIONS = 100

# Function to load and preprocess images
def load_images(directory):
    images = []
    labels = []
    for image_name in os.listdir(directory):
        if image_name.endswith('.jpg'):
            image_path = os.path.join(directory, image_name)
            image = cv2.imread(image_path)
            image = cv2.resize(image, (INPUT_SIZE, INPUT_SIZE))
            images.append(image)
            labels.append(0 if directory.endswith('no') else 1)
    return images, labels

# Load images and labels
no_tumor_directory = os.path.join(image_directory, 'no')
yes_tumor_directory = os.path.join(image_directory, 'yes')

if not os.listdir(no_tumor_directory) or not os.listdir(yes_tumor_directory):
    raise ValueError("Both classes are required for training")

no_tumor_images, no_tumor_labels = load_images(no_tumor_directory)
yes_tumor_images, yes_tumor_labels = load_images(yes_tumor_directory)

# Combine data from both classes
dataset = np.array(no_tumor_images + yes_tumor_images)
labels = np.array(no_tumor_labels + yes_tumor_labels)

# Check if dataset contains both classes
if len(set(labels)) < 2:
    raise ValueError("Both classes are required for training")

# Initialize lists to store accuracy values from each iteration
accuracies = []

# Repeat for 100 iterations
for i in range(ITERATIONS):
    # Split dataset into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(dataset, labels, test_size=0.2, random_state=i)

    # CNN model
    cnn_model = Sequential()
    cnn_model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(INPUT_SIZE, INPUT_SIZE, 3)))
    cnn_model.add(MaxPooling2D((2, 2)))
    cnn_model.add(Conv2D(64, (3, 3), activation='relu'))
    cnn_model.add(MaxPooling2D((2, 2)))
    cnn_model.add(Conv2D(64, (3, 3), activation='relu'))
    cnn_model.add(Flatten())
    cnn_model.add(Dense(64, activation='relu'))
    cnn_model.add(Dense(1, activation='sigmoid'))

    # Compile CNN model
    cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train CNN model
    cnn_model.fit(x_train, y_train, epochs=1, batch_size=16, verbose=0)

    # Evaluate CNN model
    _, test_accuracy = cnn_model.evaluate(x_test, y_test, verbose=0)
    
    # Append accuracy to list
    accuracies.append(test_accuracy)

# Find the highest accuracy and its corresponding iteration
max_accuracy = max(accuracies)
max_accuracy_iteration = accuracies.index(max_accuracy) + 1  # Add 1 to get the iteration number

# Plotting the accuracies using Plotly
fig = go.Figure()
fig.add_trace(go.Scatter(x=list(range(1, ITERATIONS + 1)), y=accuracies, mode='lines', name='Accuracy'))
fig.add_trace(go.Scatter(x=[max_accuracy_iteration], y=[max_accuracy], mode='markers', name=f'Highest Accuracy: {max_accuracy:.4f}', marker=dict(color='red', size=10)))
fig.update_layout(title='Accuracy of CNN Model (100 iterations)',
                  xaxis_title='Iteration',
                  yaxis_title='Accuracy',
                  hovermode='closest')
fig.show()

print(f"Highest accuracy from 100 iterations: {max_accuracy:.4f}")
