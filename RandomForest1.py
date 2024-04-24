import cv2
import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

image_directory = 'datasets/'

no_tumor_images = os.listdir(image_directory + 'no/')
yes_tumor_images = os.listdir(image_directory + 'yes/')
dataset = []
label = []
INPUT_SIZE = 64

for image_name in no_tumor_images:
    if image_name.split('.')[1] == 'jpg':
        image = cv2.imread(image_directory + 'no/' + image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((INPUT_SIZE, INPUT_SIZE))
        dataset.append(np.array(image).flatten())
        label.append(0)

for image_name in yes_tumor_images:
    if image_name.split('.')[1] == 'jpg':
        image = cv2.imread(image_directory + 'yes/' + image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((INPUT_SIZE, INPUT_SIZE))
        dataset.append(np.array(image).flatten())
        label.append(1)

dataset = np.array(dataset)
label = np.array(label)

x_train, x_test, y_train, y_test = train_test_split(dataset, label, test_size=0.2, random_state=0)

model = RandomForestClassifier(n_estimators=100, random_state=0)
model.fit(x_train, y_train)

# Predictions
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)

# Plotting the validation accuracy
plt.plot(range(1, len(model.estimators_) + 1), [estimator.score(x_test, y_test) for estimator in model.estimators_], label='Validation Accuracy')
plt.xlabel('Number of Trees')
plt.ylabel('Accuracy')
plt.title('Accuracy of Random Forest with 100 Trees')

# Highlighting the highest accuracy point
max_val_accuracy = max([estimator.score(x_test, y_test) for estimator in model.estimators_])
max_val_accuracy_tree = [i for i, v in enumerate([estimator.score(x_test, y_test) for estimator in model.estimators_], start=1) if v == max_val_accuracy][0]
plt.scatter(max_val_accuracy_tree, max_val_accuracy, color='red', marker='o', label=f'Highest Accuracy: {max_val_accuracy:.4f}')
plt.annotate(f'Highest Accuracy: {max_val_accuracy:.4f}', (max_val_accuracy_tree, max_val_accuracy), textcoords="offset points", xytext=(0,10), ha='center')

plt.legend()
plt.show()
