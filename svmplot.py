import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from skimage.feature import hog

# Function to load and preprocess images
def preprocess_images(image_directory, label, dataset, labels):
    for image_name in os.listdir(image_directory):
        if image_name.endswith('.jpg'):
            image_path = os.path.join(image_directory, image_name)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (INPUT_SIZE, INPUT_SIZE))
            hog_features = hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), transform_sqrt=True)
            dataset.append(hog_features)
            labels.append(label)

# Function to train and evaluate SVM
def train_and_evaluate(dataset, labels):
    x_train, x_test, y_train, y_test = train_test_split(dataset, labels, test_size=0.2, random_state=0)
    svm_classifier = SVC(kernel='linear')
    svm_classifier.fit(x_train, y_train)
    y_pred_train = svm_classifier.predict(x_train)
    y_pred_test = svm_classifier.predict(x_test)
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    return train_accuracy, test_accuracy

image_directory = 'datasets/'
no_tumor_images = os.listdir(os.path.join(image_directory, 'no'))
yes_tumor_images = os.listdir(os.path.join(image_directory, 'yes'))

INPUT_SIZE = 64
num_iterations = 100
train_accuracies = []
test_accuracies = []

# Load and preprocess images, train and evaluate SVM iteratively
for i in range(num_iterations):
    dataset = []
    labels = []
    preprocess_images(os.path.join(image_directory, 'no'), 0, dataset, labels)
    preprocess_images(os.path.join(image_directory, 'yes'), 1, dataset, labels)
    dataset = np.array(dataset)
    labels = np.array(labels)
    train_accuracy, test_accuracy = train_and_evaluate(dataset, labels)
    train_accuracies.append(train_accuracy)
    test_accuracies.append(test_accuracy)

    print(f"Iteration {i+1}: Train Accuracy = {train_accuracy:.4f}, Test Accuracy = {test_accuracy:.4f}")

