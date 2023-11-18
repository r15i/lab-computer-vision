import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


def calc_hists(img: np.ndarray) -> list:
    """
    Calculates the histogram of the image (channel by channel).

    Args:
        img (numpy.ndarray): image to calculate the histogram

    Returns:
        list: list of histograms
    """

    chan = []
    for i in range(3):
        hist = cv.calcHist([img], [i], None, [256], [0, 256])
        chan.append(hist)
    return chan


def show_histogram(hists: list) -> np.ndarray:
    """
    Shows the histogram of the image.

    Args:
        hists (list): list of histograms to plot

    Returns:
        img (numpy.ndarray): image containing the histograms
    """

    colors = ["Blue", "Green", "Red"]

    for i in range(2):
        plt.subplot(1, 2, i + 1)
        plt.hist(hists[i], 256, range=(0, 255), color=colors[i], edgecolor="black")
        plt.title(f"Histogram {i+1}")
        plt.xlabel("Pixel Value")
        plt.ylabel("Frequency")

    canvas = plt.get_current_fig_manager().canvas
    plt.close()

    canvas.draw()

    img = np.frombuffer(canvas.buffer_rgba(), dtype="uint8")
    img = img.reshape(canvas.get_width_height()[::-1] + (4,))

    return cv.cvtColor(img, cv.COLOR_RGB2BGR)

print("test")
img = cv.imread("./data/barbecue.jpg")
hD = calc_hists(img)
histo = show_histogram(hD)
plt.imshow(histo)
plt.axis("off")
plt.title("Image")

plt.tight_layout()
plt.show()