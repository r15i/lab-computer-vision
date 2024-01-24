import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from tqdm import tqdm

DEBUG = True

# debug print
def dprint(s):
    if DEBUG:
        print(s)


def load_paths(index=0, data_list_path="data_list.xml"):
    # Load data from the xml
    data_list = ET.parse(data_list_path)
    root = data_list.getroot()

    main_paths = []

    for item in root.findall("dataset"):
        path = item.find("path").text
        main_paths.append(path)

    return main_paths


def load_images_by_label(root_directory, dataset_type):
    label_path = os.path.join(root_directory, dataset_type)

    labels = []
    images = []
    if os.path.isdir(label_path):
        # dprint(label_path)
        # aggiungere caso in cui si hanno sottocartelle
        for image_file in tqdm(os.listdir(label_path), desc=f"Loading Images for {dataset_type+" set"}"):
            # find a better way to say only photos
            if image_file == "desktop.ini":
                continue
            image_path = os.path.join(label_path, image_file)
            # dprint(image_path)
            labels.append(image_file)
            images.append(cv2.imread(image_path))

    filename = labels
    
    
    if(root_directory.split("\\")[-1]=="MWD"):
        names = ["cloudy","shine","rain","sunrise"]
        for i in range(len(labels)):
            for j in names:
                if j in labels[i]:
                    labels[i] = j

    else:
        labels = [l.split("_")[0] for l in labels]

    # dprint(np.unique(labels, return_counts=True))

    return [images, labels, filename]


def calculate_histogram(image):
    # Assuming image is a 3D numpy array (RGB image)
    hist_channels = [
        cv2.calcHist([image], [i], None, [256], [0, 256]) for i in range(3)
    ]
    return hist_channels

def display_images_and_histograms(images, filenames, indices_to_display,histograms):
    num_images = len(indices_to_display)

    # Set up subplots with 2 columns for each image and its histogram
    plt.figure(figsize=(15, 5 * num_images))

    for i, idx in enumerate(indices_to_display, start=1):
        img = images[idx]
        filename = filenames[idx]
        #hist_channels = calculate_histogram(img)
        hist_channels = histograms[idx]
        # Display Image
        plt.subplot(num_images, 2, i * 2 - 1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for Matplotlib
        plt.title(f"Image: {filename}")

        # Display Histogram with different colors for each channel
        plt.subplot(num_images, 2, i * 2)
        colors = ['blue', 'green', 'red']
        for j in range(3):
            plt.plot(hist_channels[j], color=colors[j], label=f'Channel {colors[j]}')
        plt.title(f"Histogram: {filename}")
        plt.legend()

    plt.tight_layout()
    plt.show()


def show_images_with_labels(images, labels, index):
    img = np.array(images)[index]
    lab = np.array(labels)[index]

    # Mostra le immagini
    for i, l in zip(img, lab):
        cv2.imshow(f"Label: {l}", i)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

