# -*- coding=utf8 -*-
import numpy as np
import cv2
import Image
from PIL import Image
import scipy.misc


# import matplotlib.pyplot as plt

def MatrixToImage(data):
    data = data * 255
    new_im = Image.fromarray(data.astype(np.uint8))
    return new_im


## [x,y,3]
def MatrixToImage2(data):
    rgb = scipy.misc.toimage(data)
    return rgb
#     r,g,b = cv2.split(data)
# 	img_bgr = cv2.merge([b,g,r])
# 	return img_bgr

def ImageToMatrix(im):
    width, height = im.size
    im = im.convert("L")
    data = im.getdata()
    data = np.matrix(data, dtype='float') / 255.0
    new_data = np.reshape(data, (width, height))
    return new_data
    
def ImageToMatrix2(im):
    mx = scipy.misc.fromimage(im)
    return mx 

root_path = "/Users/apple2/Tracking/PredictImages/"

cap = cv2.VideoCapture("test3.mp4")

fgbg = cv2.BackgroundSubtractorMOG(500, 10, 0.2)

fnprex = "getting_mp4_"

predict_filelist = root_path + "predict_filelist.txt"
pfl = open(predict_filelist,'w')

j = 1
while (1):
    print "begining"
    ret, frame = cap.read()
    fgmask = fgbg.apply(frame)

    fgmask2 = cv2.dilate(fgmask, None, iterations=2)
    (cnts, _) = cv2.findContours(fgmask2.copy(), cv2.RETR_EXTERNAL,
                                 cv2.CHAIN_APPROX_SIMPLE)
    i = 1
    for c in cnts:
        # if the contour is too small, ignore it
        if cv2.contourArea(c) < 100:
            continue

        # compute the bounding box for the contour, draw it on the frame,
        # and update the text
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        print frame
        print frame.shape
        
        frame2 = frame.copy()
        frame2 = MatrixToImage2(frame2)
        
        box = (x, y, x + w, y + h)
        cropImg = frame2.crop(box)
        re_size = (64, 128)
        im_resize = cropImg.resize(re_size, Image.BILINEAR)
        fp = root_path + fnprex + '%s_%s.jpg' % (j, i)
        
        pfl.write(fp + " 0\n")  ##图片列表

        print im_resize
        im_resize.save(fp)
        
        cv2.imshow('frame',ImageToMatrix2(im_resize))

        # im_resize.save('test')
#         im_resize.save(fp)
		
		
        i = i + 1

    print "frame number %s" % j
    j = j + 1
    cv2.imshow('frame',ret)
    cv2.imshow('ori', frame)
    cv2.imshow('other',fgmask2)

    k = cv2.waitKey(30) & 0xff

    if k == 27:
        break
    if j > 2200:
        break


pfl.close()

cap.release()
cv2.destroyAllWindows()
