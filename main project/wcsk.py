import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import utilssk as ut


# carica le path dinamicamente dal xml
""" data_list_path = "path_to_data_list.xml"  # Sostituisci con il percorso reale
r_paths = ut.load_paths(data_list_path)

for path in r_paths:
    print(f"Main Path: {path}")
 """


""" train_data = ut.load_images_by_label(".\\data\\UAVid\\UAVid", "train")
test_data = ut.load_images_by_label(".\\data\\UAVid\\UAVid", "test")
 """

train_data = ut.load_images_by_label(".\\data\\ACDC\\ACDC", "train")
test_data = ut.load_images_by_label(".\\data\\ACDC\\ACDC", "test")

# ut.show_images_with_labels(loaded_data[1],loaded_data[0])


# implicitly use the labs from the train part
imgs, labs, filenames = train_data[0], train_data[1], train_data[2]
lab, counts = np.unique(labs, return_counts=True)
# print the label counts
print("Label Counts:\n")
for label, count in zip(lab, counts):
    print(f"{label}:\t{count}")
print(f"\nTotal images:\t{len(imgs)}\n")


# ut.show_images_with_labels(imgs, filenames, list([10, 15, 21, 25, 30, 35]))

print("showing some images with histograms")


# calculating histograms
X_train = []
X_test = []

for img in train_data[0]:
    X_train.append(ut.calculate_histogram(img))

for img in test_data[0]:
    X_test.append(ut.calculate_histogram(img))


# display image with histogram with associated image in the training data
# ut.display_images_and_histograms(train_data[0], train_data[2], [0, 1, 2, 3], X_train)
# ut.display_images_and_histograms(test_data[0], test_data[2], [0, 1, 2, 3], X_test)


# setting the train and test labels and data
# converting the trainand test  into array and using a better naming convention
X_train = np.array(X_train)
y_train = np.array(train_data[1])

X_test = np.array(X_test)
y_test = np.array(test_data[1])


# reshaping
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)


# Split the data into training and test sets
""" X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
) """

# Create the scikit-learn model (MLPClassifier) CNN
model = make_pipeline(MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000))

# Train the model
model.fit(X_train, y_train)
# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Test Accuracy: {:.2%}".format(accuracy))


#tester 
for i in [1,20,12]:
    ind = i
    h_test = ut.calculate_histogram(test_data[0][ind])
    h_test_r = np.array(h_test).reshape(1, -1)
    y_pred = model.predict(h_test_r)
    print(y_pred)
    ut.display_images_and_histograms([test_data[0][ind]],[test_data[2][ind]],[0] , [h_test])




