import cv2
import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import plotly.graph_objs as go

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

accuracies = []

# Train SVC with different kernel types
for kernel_type in ['linear', 'poly', 'rbf', 'sigmoid']:
    model = SVC(kernel=kernel_type)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)

# Plotting the validation accuracy
fig = go.Figure()
fig.add_trace(go.Bar(x=['Linear', 'Polynomial', 'RBF', 'Sigmoid'], y=accuracies, name='Validation Accuracy'))

# Highlighting the highest accuracy point
max_val_accuracy = max(accuracies)
max_val_accuracy_kernel = ['Linear', 'Polynomial', 'RBF', 'Sigmoid'][accuracies.index(max_val_accuracy)]
fig.add_trace(go.Scatter(x=[max_val_accuracy_kernel], y=[max_val_accuracy], mode='markers', name=f'Highest Accuracy: {max_val_accuracy:.4f}', marker=dict(color='red', size=10)))

fig.update_layout(title='Validation Accuracy of SVC with Different Kernel Types',
                  xaxis_title='Kernel Type',
                  yaxis_title='Validation Accuracy')

fig.show()
