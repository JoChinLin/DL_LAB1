import cv2
import numpy as np

def splitRGB(img):
    # TODO

    R_map = np.zeros(img.shape)
    R_map[:,:,2] = img[:,:,2]
    G_map = np.zeros(img.shape)
    G_map[:,:,1] = img[:,:,1]
    B_map = np.zeros(img.shape)
    B_map[:,:,0] = img[:,:,0]


    return R_map, G_map, B_map

def splitHSV(img):
    # TODO
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H_map,S_map,V_map = cv2.split(hsv)
    return H_map, S_map, V_map


def GetBilinearPixel(imArr, posX, posY):
	out = []
 
	#Get integer and fractional parts of numbers
	modXi = int(posX)
	modYi = int(posY)
	modXf = posX - modXi
	modYf = posY - modYi
	modXiPlusOneLim = min(modXi+1,imArr.shape[1]-1)
	modYiPlusOneLim = min(modYi+1,imArr.shape[0]-1)
 
	#Get pixels in four corners
	for chan in range(imArr.shape[2]):
		bl = imArr[modYi, modXi, chan]
		br = imArr[modYi, modXiPlusOneLim, chan]
		tl = imArr[modYiPlusOneLim, modXi, chan]
		tr = imArr[modYiPlusOneLim, modXiPlusOneLim, chan]
 
		#Calculate interpolation
		b = modXf * br + (1. - modXf) * bl
		t = modXf * tr + (1. - modXf) * tl
		pxf = modYf * t + (1. - modYf) * b
		out.append(int(pxf+0.5))
 
	return out

def resize(img, size):
    # TODO

    shape = list(map(int, [img.shape[0]*size, img.shape[1]*size, img.shape[2]]))
    img_t = np.empty(shape, dtype=np.uint8)
    rowScale = float(img.shape[0]) / float(img_t.shape[0])
    colScale = float(img.shape[1]) / float(img_t.shape[1])
    for r in range(img_t.shape[0]):
    	for c in range(img_t.shape[1]):
    		orir = r * rowScale
    		oric = c * colScale
    		img_t[r,c]=GetBilinearPixel(img,oric,orir)

    return img_t



class MotionDetect(object):
    """docstring for MotionDetect"""
    def __init__(self, shape):
        super(MotionDetect, self).__init__()

        self.shape = shape
        self.avg_map = np.zeros((self.shape[0], self.shape[1]), dtype='float')
        self.alpha = 0.8 # you can ajust your value
        self.threshold = 40 # you can ajust your value

        print("MotionDetect init with shape {}".format(self.shape))

    def getMotion(self, img):
        assert img.shape == self.shape, "Input image shape must be {}, but get {}".format(self.shape, img.shape)


        # Extract motion part (hint: motion part mask = difference between image and avg > threshold)
        # TODO

        motion_part = img - (avg_map > threshold)

        motion_map = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        Motion = motion_map - self.avg_map
        ret, mask = cv2.threshold(Motion,self.threshold,255,cv2.THRESH_BINARY)

        # Mask out unmotion part (hint: set the unmotion part to 0 with mask)
        # TODO

        motion_map[~mask] = [0, 0, 0]

        # Update avg_map
        # TODO

        self.avg_map = self.avg_map*alpha + motion_map*(1-alpha)


        return motion_map


# ------------------ #
#     RGB & HSV      #
# ------------------ #
name = "../data.png"
img = cv2.imread(name)
if img is not None:
    print("Reading {} success. Image shape {}".format(name, img.shape))
else:
    print("Faild to read {}.".format(name))

R_map, G_map, B_map = splitRGB(img)
H_map, S_map, V_map = splitHSV(img)

cv2.imwrite('data_R.png', R_map)
cv2.imwrite('data_G.png', G_map)
cv2.imwrite('data_B.png', B_map)
cv2.imwrite('data_H.png', H_map)
cv2.imwrite('data_S.png', S_map)
cv2.imwrite('data_V.png', V_map)


# ------------------ #
#   Interpolation    #
# ------------------ #
name = "../data.png"
img = cv2.imread(name)
if img is not None:
    print("Reading {} success. Image shape {}".format(name, img.shape))
else:
    print("Faild to read {}.".format(name))

height, width, channel = img.shape
img_big = resize(img, 2)
img_small = resize(img, 0.5)
img_big_cv = cv2.resize(img, (width*2, height*2))
img_small_cv = cv2.resize(img, (width//2, height//2))

cv2.imwrite('data_2x.png', img_big)
cv2.imwrite('data_0.5x.png', img_small)
cv2.imwrite('data_2x_cv.png', img_big_cv)
cv2.imwrite('data_0.5x_cv.png', img_small_cv)

# ------------------ #
#  Video Read/Write  #
# ------------------ #
name = "../data.mp4"
# Input reader
cap = cv2.VideoCapture(name)
fps = cap.get(cv2.CAP_PROP_FPS)
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

# Output writer
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output1.avi', fourcc, fps, (w, h), True)

# Motion detector
mt = MotionDetect(shape=(h,w,3))
frames=0
# Read video frame by frame
while True:
    # Get 1 frame
    success, frame = cap.read()

    if success:
        motion_map = mt.getMotion(frame)

        # Write 1 frame to output video
        out.write(motion_map)
    else:
        break

# Release resource
cap.release()
out.release()