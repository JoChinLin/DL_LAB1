import cv2
import numpy as np
import matplotlib.pyplot as plt

def gammaCorrection(img, gamma=1.0):
    # Build a lookup table mapping the pixel values [0, 255] to their adjusted gamma values
    # TODO
    invGamma = 1.0 / gamma
    table = np.array([((i/255.0) ** invGamma) * 255 for i in np.arange(0,256)]).astype("uint8")

    # Apply gamma correction using the lookup table
    # TODO
    img_g=cv2.LUT(img,table)
    return img_g

def histEq(gray):
    # Calculate histogram
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).reshape(-1)
    hist = hist / gray.size
    print(hist)

    # Convert the histogram to Cumulative Distribution Function
    # TODO
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max()/ cdf.max()

    # Build a lookup table mapping the pixel values [0, 255] to their new grayscale value
    # TODO

    cdf_m = np.ma.masked_equal(cdf,0)
    cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
    cdf = np.ma.filled(cdf_m,0).astype('uint8')

    
    # Apply histogram equalization using the lookup table
    # TODO
    img_h = cdf[img]

    return img_h


# ------------------ #
#  Gamma Correction  #
# ------------------ #
name = "../data.mp4"
cap = cv2.VideoCapture(name)
success, frame = cap.read()
if success:
    print("Success reading 1 frame from {}".format(name))
else:
    print("Faild to read 1 frame from {}".format(name))
cap.release()

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
img_g1 = gammaCorrection(gray, 0.5)
img_g2 = gammaCorrection(gray, 2)
cv2.imwrite('gray.png', gray)
cv2.imwrite('data_g0.5.png', img_g1)
cv2.imwrite('data_g2.png', img_g2)

# ------------------------ #
#  Histogram Equalization  #
# ------------------------ #
name = "../hist.png"
img = cv2.imread(name, 0)

img_h = histEq(img)
img_h_cv = cv2.equalizeHist(img)
cv2.imwrite("hist_h.png", img_h)
cv2.imwrite("hist_h_cv.png", img_h_cv)

# save histogram
plt.figure(figsize=(18, 6))
plt.subplot(1,3,1)
plt.bar(range(1,257), cv2.calcHist([img], [0], None, [256], [0, 256]).reshape(-1))
plt.subplot(1,3,2)
plt.bar(range(1,257), cv2.calcHist([img_h], [0], None, [256], [0, 256]).reshape(-1))
plt.subplot(1,3,3)
plt.bar(range(1,257), cv2.calcHist([img_h_cv], [0], None, [256], [0, 256]).reshape(-1))
plt.savefig('hist_plot.png')