import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


#to load the the xml 
import xml.etree.ElementTree as ET

def loadDataset(path,index):


    return 0


# Funzione per calcolare l'istogramma di un'immagine
def calculate_histogram(image):
    hist = cv2.calcHist(
        [image], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256]
    )
    hist = hist.flatten() / hist.sum()  # Normalizza l'istogramma
    return hist