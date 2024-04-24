import cv2
import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
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

for k in range(1, 101):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)

# Plotting the validation accuracy
fig = go.Figure()
fig.add_trace(go.Scatter(x=list(range(1, 101)), y=accuracies, mode='lines', name='Validation Accuracy'))

# Highlighting the highest accuracy point
max_val_accuracy = max(accuracies)
max_val_accuracy_k = accuracies.index(max_val_accuracy) + 1
fig.add_trace(go.Scatter(x=[max_val_accuracy_k], y=[max_val_accuracy], mode='markers', name=f'Highest Accuracy: {max_val_accuracy:.4f}', marker=dict(color='red', size=10)))

fig.update_layout(title='Accuracy of KNN with 100 Iterations',
                  xaxis_title='Number of Neighbors (k)',
                  yaxis_title='Accuracy')

fig.show()
