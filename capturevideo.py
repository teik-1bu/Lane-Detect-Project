import numpy as np
import cv2
from thresholding_main import *

cap = cv2.VideoCapture('../test.mp4')

#perspective transform on undistorted images
def perspective_transform(img):
    imshape = img.shape
    #print (imshape)
    vertices = np.array([[(0.65*imshape[1], 0.6*imshape[0]), (imshape[1],imshape[0]),(0,imshape[0]),(0.35*imshape[1], 0.6*imshape[0])]], dtype=np.float32)
    #print (vertices)
    src= np.float32(vertices)
    dst = np.float32([[0.75*img.shape[1],0],[0.75*img.shape[1],img.shape[0]],
                      [0.25*img.shape[1],img.shape[0]],[0.25*img.shape[1],0]])
    #print (dst)
    M = cv2.getPerspectiveTransform(src, dst)

    Minv = cv2.getPerspectiveTransform(dst, src)
    img_size = (imshape[1], imshape[0]) 
    perspective_img = cv2.warpPerspective(img, M, img_size, flags = cv2.INTER_LINEAR)    
    return perspective_img, Minv

#region of interest
def region_of_interest(img, vertices):
    #defining a blank mask to start with
    mask = np.zeros_like(img, dtype=np.uint8)
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, np.array([vertices], dtype=np.int32), ignore_mask_color)
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

#Color thresholding, takes saturation and value images in single channel and corresponding threshold values 
def color_thr(s_img, v_img, s_threshold = (0,255), v_threshold = (0,255)):
    s_binary = np.zeros_like(s_img).astype(np.uint8)
    s_binary[(s_img > s_threshold[0]) & (s_img <= s_threshold[1])] = 1
    v_binary = np.zeros_like(s_img).astype(np.uint8)
    v_binary[(v_img > v_threshold[0]) & (v_img <= v_threshold[1])] = 1
    col = ((s_binary == 1) | (v_binary == 1))
    return col

while(cap.isOpened()):
    ret, frame = cap.read()
    imshape = frame.shape
    # vertices = np.array([[(0.625*imshape[1], 0.55*imshape[0]), (0.875*imshape[1],0.8*imshape[0]),(0.125*imshape[1],0.8*imshape[0]),(0.375*imshape[1], 0.55*imshape[0])]])
    vertices = np.array([[(0.65*imshape[1], 0.6*imshape[0]), (imshape[1],imshape[0]),(0,imshape[0]),(0.35*imshape[1], 0.6*imshape[0])]])
    roi = region_of_interest(frame, vertices)
    perspective_img, Minv = perspective_transform(roi)
    gray = cv2.cvtColor(perspective_img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(15,15),0)
    # cv2.imshow('frame1', gray)
    cv2.imshow('frame2', blur) 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()