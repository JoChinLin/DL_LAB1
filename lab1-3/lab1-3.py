import cv2
import numpy as np

def avgFilter(img,kernel=3):
    # TODO
    img_avg = cv2.blur(img,(kernel,kernel))
    return img_avg

def midFilter(img,kernel=3):
    # TODO
    img_mid = cv2.medianBlur(img, kernel)
    return img_mid

def edgeSharpen(img):
    # TODO

    # Define Laplacian filter
    t1=list([[0,1,0],
    		[1,-4,1],
    		[0,1,0]])

    # grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_edge = img*1

    # Add a layer of edges to the original image
    img_edge = np.pad(img,((1, 1), (1, 1)), "constant", constant_values=0)

    for i in range(1,img.shape[0]):
    	for j in range(1,img.shape[1]):
    		img_edge[i,j]=abs(np.sum(img_edge[i:i+1,j:j+1]*t1))


    img_s = img - img_edge[1:img_edge.shape[0]-1,1:img_edge.shape[1]-1]


    return img_edge, img_s

# ------------------ #
#       Denoise      #
# ------------------ #
name1 = '../noise_impulse.png'
name2 = '../noise_gauss.png'
noise_imp = cv2.imread(name1, 0)
noise_gau = cv2.imread(name2, 0)

img_imp_avg = avgFilter(noise_imp)
img_imp_mid = midFilter(noise_imp)
img_gau_avg = avgFilter(noise_gau)
img_gau_mid = midFilter(noise_gau)

cv2.imwrite('img_imp_avg.png', img_imp_avg)
cv2.imwrite('img_imp_mid.png', img_imp_mid)
cv2.imwrite('img_gau_avg.png', img_gau_avg)
cv2.imwrite('img_gau_mid.png', img_gau_mid)


# ------------------ #
#       Sharpen      #
# ------------------ #
name = '../mj.tif'
img = cv2.imread(name, 0)

img_edge, img_s = edgeSharpen(img)
cv2.imwrite('mj_edge.png', img_edge)
cv2.imwrite('mj_sharpen.png', img_s)