import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

#to load the the xml 
import xml.etree.ElementTree as ET



def loadPaths(index=0, data_list_path="data_list.xml"):
    
    # load data from the xml
    data_list = ET.parse(data_list_path)
    root = data_list.getroot()
    
    train_paths,test_paths = [],[]

    for item in root.findall('dataset'):
        path = item.find('path').text
        train_paths.append(path + "\\train")
        test_paths.append(path + "\\test")


    return train_paths,test_paths











# Funzione per calcolare l'istogramma di un'immagine
def calculate_histogram(image):
    hist = cv2.calcHist(
        [image], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256]
    )
    hist = hist.flatten() / hist.sum()  # Normalizza l'istogramma
    return hist


