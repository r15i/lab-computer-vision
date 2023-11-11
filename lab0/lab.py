import cv2 as cv
import numpy as np
import time


#global variable for the calls 
iavg = [-1, -1, -1]


def getAvg(event: int, x: int, y: int, flags: int, function_params):
    global iavg
    if event == cv.EVENT_LBUTTONDOWN:
        i = function_params
        csize = 0
        cube = i[y-csize-1:y+csize+2, x-csize-1:x+csize+2]
        avg = np.average(cube, axis=(0, 1)).astype(np.uint8)
        iavg = avg
        print(f"current avg {iavg}")


def grabColor(win_name, img):
    win = cv.imshow(win_name, img)
    cv.setMouseCallback(win_name, getAvg, img)
    key = cv.waitKey(0)
    return iavg

def swapColor(old_color,new_color,img,threshold_value):
    # Split the image into R, G, B channels
    R, G, B = cv.split(img)
    # Apply static threshold
    #print(f"threshold_value {threshold_value}")
    segmented_mask = (np.abs(R-old_color[0]) < threshold_value) & (np.abs(G-old_color[1]) < threshold_value) & (np.abs(B-old_color[2]) < threshold_value)
    cv.imshow("mask",segmented_mask.astype(np.uint8) *255)
    img[segmented_mask] = new_color
    title = f'Swapped colors threshold {threshold_value}'
    cv.imshow(title, img)
    cv.waitKey(0)
    cv.destroyWindow(title)






img_path = {"f1": "images/f1.jpg", "test": "images/test.png"}

img = cv.imread(img_path["f1"])
print("IMG :", type(img))
h, w, c = img.shape
print("size : "+str(h)+" x "+str(w))
print("img chann :", str(c))
print("Data type", img.dtype)


#CHOOSE COLOR FROM THE IMAG
old_color = grabColor("f1", cv.imread(img_path["f1"]))


#COPY THE  IMAGE
img_raw = img
img = np.array(img_raw, copy=True)


#SELECT THE NEW COLOR
new_color = grabColor("test", cv.imread(img_path["test"]))

key = cv.waitKey(0)
cv.destroyWindow('test')

#ok we got the data
print(f"OLD COLOR {old_color}")
print(f"NEW COLOR {new_color}")

for i in range (10,10,10):
    swapColor(old_color,new_color,img,i)




#swap RGB TO BGR ? 
img_raw = img[:, :, ::-1]
cv.imshow("BGR", img)
cv.waitKey(0)