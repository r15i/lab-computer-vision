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

    hist_r = cv.calcHist([img], [0], None, [256], [0, 256])
    hist_g = cv.calcHist([img], [1], None, [256], [0, 256])
    hist_b = cv.calcHist([img], [2], None, [256], [0, 256])
    return hist_r, hist_g, hist_b

def show_histogram(hists: list) -> np.ndarray:
    """
    Shows the histogram of the image.
    
    Args:
        hists (list): list of histograms to plot

    Returns:
        img (numpy.ndarray): image containing the histograms
     """   
    colors = ["Blue", "Green", "Red"]

    x = len(hists)
    s = 1 #scale
    plt.figure(figsize=(30*s, 30*s))

    plt.axis('off')  
    fig, axs = plt.subplots(1, 3, figsize=(10*s,10*s ), gridspec_kw={'width_ratios': [10*s for _ in range(3)]})

    for i in range(3):
        x = np.array(range(0, 256))
        axs[i].plot(x, hD[i].flatten(), color=colors[i])
        axs[i].set_title(f"Histogram {i+1}")
        axs[i].set_xlabel("Pixel Value",fontsize=50)
        axs[i].set_ylabel("Frequency",fontsize=50)
        axs[i].tick_params(axis='both', which='major', labelsize=50)  


    plt.subplots_adjust(wspace=0.6)  
    

    canvas = plt.get_current_fig_manager().canvas
    plt.close()
    canvas.draw()    

    img = np.frombuffer(canvas.buffer_rgba(), dtype='uint8')
    img = img.reshape(canvas.get_width_height()[::-1] + (4,))

    
    return cv.cvtColor(img, cv.COLOR_RGB2BGR)


def img_hist(img: np.ndarray, hists: list[np.ndarray], scaling_factor: float = 0.3) -> np.ndarray:
    """
    Fuse the image with the histogram.
    
    Args:
        img (np.ndarray): image
        hist (list(np.ndarray)): histograms of the image
        
    Returns:
        img (numpy.ndarray): image containing the original image with the histograms on top
    """
    

    img_h = show_histogram(hists)
    width = img_h.shape[1]
    h_w = img.shape[0]/img.shape[1]
    height = int(  h_w * width)
    resized_img = cv.resize(img, (width, height),fx= scaling_factor,fy=scaling_factor)
    s_image = np.vstack(( img_h,resized_img))

    return s_image
    

print("Loading image")

img = cv.imread('./data/barbecue.jpg')
hD = calc_hists(img)
hD_img = show_histogram(hD)
#img_hi = img_hist(img,hD_img,2)
print(hD)

plt.imshow(img)
plt.axis('off')  # Turn off axis labels and ticks
plt.show()






"""


img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

img_yuv = cv.cvtColor(img, cv.COLOR_RGB2YUV)

img_yuv[:,:,0] = cv.equalizeHist(img_yuv[:,:,0])

img_eq = cv.cvtColor(img_yuv, cv.COLOR_YUV2RGB)
img_hi = show_histogram(calc_hists(img_output))

img_output = img_hist(img_output,img_hi) 








plt.axis('off')  
plt.imshow(img_output)













img = cv.imread('./data/barbecue.jpg')
img_lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)











l_channel, a_channel, b_channel = cv.split(img_lab)


l_channel_eq = cv.equalizeHist(l_channel)


image_lab_eq = cv.merge((l_channel_eq, a_channel, b_channel))


image_bgr_eq = cv.cvtColor(image_lab_eq, cv.COLOR_LAB2BGR)






hD_eq = calc_hists(image_bgr_eq)
img_hi = show_histogram(hD_eq)
img_hi = show_histogram(image_bgr_eq,hD_eq)











plt.axis('off')  
plt.imshow(img_output)

















def hist_match(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
     
    
    #Adjust the pixel values of a color image such that its histogram
    #matches that of a target one.

    #Args:
    #    source (numpy.ndarray): Image to transform; the histogram is computed over the flattened array
    #    reference (numpy.ndarray): Template image; can have different dimensions from source
    #Returns:
    #    numpy.ndarray: The transformed output image
     
    assert source.shape[2] == reference.shape[2], "Images must have the same number of channels"
    assert source.shape[:2] == reference.shape[:2], "Images must have the same dimensions"

    
    src_hists = [np.histogram(source[..., i].flatten(), 256, [0,256])[0] for i in range(source.shape[2])]
    src_cdfs = [hist.cumsum() for hist in src_hists]
    src_cdfs_normalized = [cdf / float(cdf.max()) for cdf in src_cdfs]
 
    
    ref_hists = [np.histogram(reference[..., i].flatten(), 256, [0,256])[0] for i in range(reference.shape[2])]
    ref_cdfs = [hist.cumsum() for hist in ref_hists]
    ref_cdfs_normalized = [cdf / float(cdf.max()) for cdf in ref_cdfs]

    
    lookup_tables = [np.zeros(256) for _ in range(source.shape[2])]
    lookup_values = [0] * source.shape[2]
    for index in range(len(lookup_tables)):
        for src_pixel_val in range(len(src_cdfs_normalized[index])):
            lookup_values[index]
            for ref_pixel_val in range(len(ref_cdfs_normalized[index])):
                if ref_cdfs_normalized[index][ref_pixel_val] >= src_cdfs_normalized[index][src_pixel_val]:
                    lookup_values[index] = ref_pixel_val
                    break
            lookup_tables[index][src_pixel_val] = lookup_values[index]

    
    matched = np.stack([cv.LUT(source[..., i], lookup_tables[i]).astype(np.uint8) for i in range(len(lookup_tables))], axis=-1)

    return matched








imgl = cv.imread("./data/panorama_left.jpg")
imgr = cv.imread("./data/panorama_right.jpg")















































class Filter:
    def __init__(self, size: int):
        
        if size % 2 == 0:
            size += 1
        self.set_filter_size(size)
    
    def set_filter_size(self, size: int) -> None:
        
        if size % 2 == 0:
            size += 1
        self.filter_size = size
    
    def get_filter_size(self) -> int:
        
        return self.filter_size

    def __call__(self, image: np.ndarray)) -> None:
        
        
        raise NotImplementedError






class GaussianFilter(Filter):
    def __init__(self, size: int, sigma_g: float):
        super().__init__(size)
        self.sigma = sigma_g

    def set_sigma(self, s: float) -> None:
        

        
        pass

    def get_sigma(self) -> float:
        
        
        
        pass
    
    def __call__(self, image: np.ndarray, size: int, sigma: float) -> None:
        
        
        
        
        pass



class MedianFilter(Filter):
    def __init__(self, size: int):
        super().__init__(size)

    def __call__(self, image: np.ndarray, size: int) -> None:
        
        
        
        
        pass



class BilateralFilter(Filter):
    def __init__(self, size: int, sigma_s: float, sigma_r: float):
        super().__init__(size)
        self.sigma_space = sigma_s
        self.sigma_range = sigma_r

    def set_sigma_range(self, sr: float) -> None:
        

        
        pass

    def get_sigma_range(self) -> float:
        
        
        
        pass

    def set_sigma_space(self, ss: float) -> None:
        

        
        pass

    def get_sigma_space(self) -> float:
        
        
        
        pass
    
    def __call__(self, image: np.ndarray, sigma_s: float, sigma_r: float) -> None:
        
        
        
        
        pass




















































































"""