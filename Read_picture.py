import cv2
import os
import numpy as np
import datetime
# Read images from the dataset
dataset_path = 'Chinese herbal medicine Datasets'  # Dataset path
classes = os.listdir(dataset_path)  # Get the list of classes
Read_time = datetime.datetime.now()
print("Start time for reading images:", Read_time)
x_data = []  # To store image data for the training set
y_data = []  # To store labels for the training set

for class_name in classes:
    class_path = os.path.join(dataset_path, class_name)
    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)
        img = cv2.imread(img_path)  # Read the image
        img = cv2.resize(img, (224, 224))  # Resize the image to 224x224x3
        x_data.append(img)
        y_data.append(class_name)

    print(class_name, 'successfully read')

x_data = np.array(x_data)  # Convert to a numpy array
y_data = np.array(y_data)  # Convert to a numpy array






