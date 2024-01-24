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
from sklearn.svm import SVC
from time import time
from sklearn.ensemble import RandomForestClassifier
import pandas as pd



# carica le path dinamicamente dal xml
""" data_list_path = "path_to_data_list.xml"  # Sostituisci con il percorso reale
r_paths = ut.load_paths(data_list_path)

for path in r_paths:
    print(f"Main Path: {path}")
 """


paths = {
    "UAVid":".\\data\\UAVid\\UAVid" ,
    "ACDC":".\\data\\ACDC\\ACDC",
    "syndrone":".\\data\\syndrone_weather\\syndrone",
    "MWD":".\\data\\MWD\\MWD",
    }


path = paths["UAVid"]

print(f"\n\nWe are analyzing {path.split("\\")[-1]}\n\n")
train_data = ut.load_images_by_label(path, "train")
test_data = ut.load_images_by_label(path, "test")

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



#tryes multiple models

common_params = {'max_iter': 1000, 'random_state': 42}
common_params_rf = {'random_state': 42}

models = [
    ("MLP", MLPClassifier(hidden_layer_sizes=(100,), **common_params)),
    ("SVM Linear", SVC(kernel='linear', C=1.0, **common_params)),
    ("SVM RBF", SVC(kernel='rbf', C=1.0, gamma='scale', **common_params)),
    ("Random Forest", RandomForestClassifier(n_estimators=100, **common_params_rf))
]



filenames = np.array(test_data[2])

bad_batches = []
bad_pred = []

# tries differet 
for m in models:
    # Train the model and tracks the time
    model_name,model = m
    start_time = time()
    model.fit(X_train, y_train)
    t = time()-start_time
    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    #which are the bad ones
    bad_batch = np.where(y_pred != y_test)[0]
    bad_batches.append(bad_batch)
    bad_pred.append([filenames[bad_batch],y_test[bad_batch],y_pred[bad_batch]])
    print (f"\nTesting:\t\t {model_name}")
    print("Train time:\t\t {:.2f} s".format(t))
    print("Test Accuracy with:\t {:.2%}".format(accuracy))
    print("\n")
    


## prints the bad ones

## TO DO THERE ARE SOME THAT DON'T ADDS UP WITH THE CLASSIFICATION 
print("\n BAD BATCH\n")
for i,j in zip(models,bad_pred):
    model_name,_ = i
    print(model_name)
    matrix = np.column_stack((bad_pred[0], bad_pred[1],bad_pred[2])).T
    df = pd.DataFrame(matrix)
    print(df)
    print("\n\n")

# analizzo delle caratteristiche delle immagini classificate male 
# cerco qualche elemento in comune tra queste immagini esempio illumiunazione etc 
# 
# ne identifico un buon modo per classificarle (potrei mandare queste in una altra rete e trainarla in modo diverso)
# o con algoritmi diversi







""" #tester 
for i in [1,50,12]:
    ind = i
    h_test = ut.calculate_histogram(test_data[0][ind])
    h_test_r = np.array(h_test).reshape(1, -1)
    y_pred = model.predict(h_test_r)
    print(y_pred)
    ut.display_images_and_histograms([test_data[0][ind]],[test_data[2][ind]],[0] , [h_test])


 """

