import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten
import matplotlib.pyplot as plt

image_directory = 'datasets/'

no_tumor_images = os.listdir(image_directory + 'no/')
yes_tumor_images = os.listdir(image_directory + 'yes/')
dataset = []
labels = []

INPUT_SIZE = 64

# Load and preprocess images
for image_name in no_tumor_images:
    if image_name.endswith('.jpg'):
        image_path = os.path.join(image_directory, 'no', image_name)
        image = cv2.imread(image_path)
        image = cv2.resize(image, (INPUT_SIZE, INPUT_SIZE))
        dataset.append(image)
        labels.append(0)

for image_name in yes_tumor_images:
    if image_name.endswith('.jpg'):
        image_path = os.path.join(image_directory, 'yes', image_name)
        image = cv2.imread(image_path)
        image = cv2.resize(image, (INPUT_SIZE, INPUT_SIZE))
        dataset.append(image)
        labels.append(1)

dataset = np.array(dataset)
labels = np.array(labels)

# Initialize lists to store accuracy values from each iteration
accuracies = []

# Repeat for 5 iterations
for i in range(5):
    # Split dataset into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(dataset, labels, test_size=0.2, random_state=i)

    # CNN model for feature extraction
    cnn_model = Sequential()
    cnn_model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(INPUT_SIZE, INPUT_SIZE, 3)))
    cnn_model.add(MaxPooling2D((2, 2)))
    cnn_model.add(Conv2D(64, (3, 3), activation='relu'))
    cnn_model.add(MaxPooling2D((2, 2)))
    cnn_model.add(Conv2D(64, (3, 3), activation='relu'))
    cnn_model.add(Flatten())

    # Extract features using CNN
    x_train_features = cnn_model.predict(x_train)
    x_test_features = cnn_model.predict(x_test)

    # SVM classifier
    svm_classifier = SVC(kernel='linear')
    svm_classifier.fit(x_train_features, y_train)

    # Predictions
    y_pred_test = svm_classifier.predict(x_test_features)

    # Calculate accuracy
    test_accuracy = accuracy_score(y_test, y_pred_test)
    
    # Append accuracy to list
    accuracies.append(test_accuracy)

# Find the highest accuracy and its corresponding iteration
max_accuracy = max(accuracies)
max_accuracy_iteration = accuracies.index(max_accuracy) + 1  # Add 1 to get the iteration number

# Plotting the accuracies using a line plot
plt.figure(figsize=(8, 6))
plt.plot(np.arange(1, 6), accuracies, marker='o', color='b')
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.title('Accuracy of SVM Model (5 iterations)')
# Mark the highest accuracy
plt.scatter(max_accuracy_iteration, max_accuracy, color='r', label=f'Highest Accuracy: {max_accuracy:.4f}')
plt.annotate(f'Highest Accuracy: {max_accuracy:.4f}', xy=(max_accuracy_iteration, max_accuracy),
             xytext=(max_accuracy_iteration + 0.2, max_accuracy - 0.05),
             arrowprops=dict(facecolor='black', shrink=0.05))
plt.legend()
plt.grid(True)
plt.show()

print(f"Highest accuracy from 5 iterations: {max_accuracy:.4f}")
