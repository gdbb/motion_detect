# -*- coding=utf8 -*-

'''
#reference:
#    histogram similarity caculation: 
#       http://www.cnblogs.com/qq78292959/archive/2013/03/22/2976239.html
'''

import numpy as np
import cv2
import Image
from PIL import Image
import scipy.misc
import math


weight_mat = \
[[0.00208333, 0.00208333, 0.00208333, 0.00208333, 0.00208333, 0.00208333
, 0.00208333, 0.00208333, 0.00208333, 0.00208333, 0.00208333, 0.00208333
, 0.00208333, 0.00208333, 0.00208333, 0.00208333],
 [0.00208333, 0.00240385, 0.00240385, 0.00240385, 0.00240385, 0.00240385
, 0.00240385, 0.00240385, 0.00240385, 0.00240385, 0.00240385, 0.00240385
, 0.00240385, 0.00240385, 0.00240385, 0.00208333],
 [0.00208333, 0.00240385, 0.00284091, 0.00284091, 0.00284091, 0.00284091
, 0.00284091, 0.00284091, 0.00284091, 0.00284091, 0.00284091, 0.00284091
, 0.00284091, 0.00284091, 0.00240385, 0.00208333],
 [0.00208333, 0.00240385, 0.00284091, 0.00347222, 0.00347222, 0.00347222
, 0.00347222, 0.00347222, 0.00347222, 0.00347222, 0.00347222, 0.00347222
, 0.00347222, 0.00284091, 0.00240385, 0.00208333],
 [0.00208333, 0.00240385, 0.00284091, 0.00347222, 0.00446429, 0.00446429
, 0.00446429, 0.00446429, 0.00446429, 0.00446429, 0.00446429, 0.00446429
, 0.00347222, 0.00284091, 0.00240385, 0.00208333],
 [0.00208333, 0.00240385, 0.00284091, 0.00347222, 0.00446429, 0.00625
, 0.00625, 0.00625, 0.00625, 0.00625, 0.00625, 0.00446429
, 0.00347222, 0.00284091, 0.00240385, 0.00208333],
 [0.00208333, 0.00240385, 0.00284091, 0.00347222, 0.00446429, 0.00625
, 0.01041667, 0.01041667, 0.01041667, 0.01041667, 0.00625, 0.00446429
, 0.00347222, 0.00284091, 0.00240385, 0.00208333],
 [0.00208333, 0.00240385, 0.00284091, 0.00347222, 0.00446429, 0.00625
, 0.01041667, 0.03125, 0.03125, 0.01041667, 0.00625, 0.00446429
, 0.00347222, 0.00284091, 0.00240385, 0.00208333],
 [0.00208333, 0.00240385, 0.00284091, 0.00347222, 0.00446429, 0.00625
, 0.01041667, 0.03125, 0.03125, 0.01041667, 0.00625, 0.00446429
, 0.00347222, 0.00284091, 0.00240385, 0.00208333],
 [0.00208333, 0.00240385, 0.00284091, 0.00347222, 0.00446429, 0.00625
, 0.01041667, 0.01041667, 0.01041667, 0.01041667, 0.00625, 0.00446429
, 0.00347222, 0.00284091, 0.00240385, 0.00208333],
 [0.00208333, 0.00240385, 0.00284091, 0.00347222, 0.00446429, 0.00625
, 0.00625, 0.00625, 0.00625, 0.00625, 0.00625, 0.00446429
, 0.00347222, 0.00284091, 0.00240385, 0.00208333],
 [0.00208333, 0.00240385, 0.00284091, 0.00347222, 0.00446429, 0.00446429
, 0.00446429, 0.00446429, 0.00446429, 0.00446429, 0.00446429, 0.00446429
, 0.00347222, 0.00284091, 0.00240385, 0.00208333],
 [0.00208333, 0.00240385, 0.00284091, 0.00347222, 0.00347222, 0.00347222
, 0.00347222, 0.00347222, 0.00347222, 0.00347222, 0.00347222, 0.00347222
, 0.00347222, 0.00284091, 0.00240385, 0.00208333],
 [0.00208333, 0.00240385, 0.00284091, 0.00284091, 0.00284091, 0.00284091
, 0.00284091, 0.00284091, 0.00284091, 0.00284091, 0.00284091, 0.00284091
, 0.00284091, 0.00284091, 0.00240385, 0.00208333],
 [0.00208333, 0.00240385, 0.00240385, 0.00240385, 0.00240385, 0.00240385
, 0.00240385, 0.00240385, 0.00240385, 0.00240385, 0.00240385, 0.00240385
, 0.00240385, 0.00240385, 0.00240385, 0.00208333],
 [0.00208333, 0.00208333, 0.00208333, 0.00208333, 0.00208333, 0.00208333
, 0.00208333, 0.00208333, 0.00208333, 0.00208333, 0.00208333, 0.00208333
, 0.00208333, 0.00208333, 0.00208333, 0.00208333]]


# import matplotlib.pyplot as plt

def MatrixToImage(data):
    data = data * 255
    new_im = Image.fromarray(data.astype(np.uint8))
    return new_im


## [x,y,3]
def MatrixToImage2(data):
    rgb = scipy.misc.toimage(data)
    return rgb
#   r,g,b = cv2.split(data)
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

'''
Functions for similarity calculation.
Histogram.
'''

def make_regalur_image(img, size = (256, 256)):
    return img.resize(size).convert('RGB')

def split_image(img, part_size = (64, 64)):
    w, h = img.size
    pw, ph = part_size
                    
    #assert w % pw == h % ph == 0
    
    return [img.crop((i, j, i+pw, j+ph)).copy() \
            for i in xrange(0, w, pw) \
            for j in xrange(0, h, ph)]

def hist_similar(lh, rh):
    #assert len(lh) == len(rh)
    return sum(1 - (0 if l == r else float(abs(l - r))/max(l, r)) for l, r in zip(lh, rh))/len(lh)

def histogram_simi(li, ri):
    #rireturn hist_similar(li.histogram(), ri.histogram())
    return sum(hist_similar(l.histogram(), r.histogram()) for l, r in zip(split_image(li), split_image(ri))) / 16.0

def calc_similarity(pic1, pic2, size, weight_mat):
	h_simi = histogram_simi(pic1, pic2)
	p_simi = pyramid_simi(pic1, pic2, size, weight_mat)

	return p_simi * 0.9 + h_simi * 0.1

def pyramid_simi(pic1, pic2, size, weight_mat):
	length = size[0]

	pic1 = pic1.resize(size)#.convert('RGB')
	pic2 = pic2.resize(size)#.convert('RGB')

	pic1 = ImageToMatrix2(pic1)
	pic2 = ImageToMatrix2(pic2)

	#print pic1.shape
	#print pic2.shape

	simi_mat = np.zeros((length, length))

	for i in range(length):
		for j in range(length):
			for k in range(3):
				simi_mat[i][j] +=  (255 - abs(pic1[i][j][k] - pic2[i][j][k])) / 255. * 0.33

	simi_mat = simi_mat * weight_mat

	return np.sum(simi_mat)


root_path = "./"

cap = cv2.VideoCapture("data/test.3gp")

fgbg = cv2.BackgroundSubtractorMOG(500, 10, 0.2)
#fgbg = cv2.BackgroundSubtractorMOG2(500, 10)

#fgbg = cv2.createBackgroundSubtractorKNN()
#fgbg.setHistory(500)
#fgbg.setShadowThreshold(10)
#fgbg.setNMixtures(10)
#fgbg.setBackgroundRatio(0.2)

fnprex = "getting_mp4_"

#predict_filelist = root_path + "predict_filelist.txt"
#pfl = open(predict_filelist,'w')


x0 = 370
y0 = 50
w0 = 60
h0 = 150
area0 = 9000 
noMove = True


'''
x0 = 23 
y0 = 15
w0 = 50
h0 = 50
area0 = 0 
noMove = True
'''


j = 1
while (1):
    #print "begining"
    ret, frame = cap.read()
    fgmask = fgbg.apply(frame)

    fgmask2 = cv2.dilate(fgmask, None, iterations=2)
    (cnts, _) = cv2.findContours(fgmask2.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    noMove = True

    i = 1
    for c in cnts:
        # if the contour is too small, ignore it
        if cv2.contourArea(c) < 100:
            continue

        # compute the bounding box for the contour, draw it on the frame,
        # and update the text
        (x, y, w, h) = cv2.boundingRect(c)
        #cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        #cv2.rectangle(frame, (x0, y0), (x0 + w0, y0 + h0), (255, 255, 0), 2)

        #coordiante of center point of last rectangle
        cx0 = x0 + w0 / 2
        cy0 = y0 + h0 / 2
        cx = x + w / 2
        cy = y + h / 2
        area = cv2.contourArea(c)

        #print x, y
        #cv2.split(frame)[x][y]

        frame_temp = frame.copy()
        frame_temp = MatrixToImage2(frame_temp)
        
        box1 = (x0, y0, x0 + w0, y0 + h0)
        pic1 = frame_temp.crop(box1)
        box2 = (x, y, x + w, y + h)
        pic2 = frame_temp.crop(box2)

        #print calc_similarity(pic1, pic2, (16, 16), weight_mat)

        if math.sqrt((cx - cx0)*(cx - cx0) + (cy - cy0)*(cy - cy0)) <= math.sqrt((x0 - cx0)*(x0 - cx0) + (y0 - cy0)*(y0 - cy0)) \
        and area >= area0 / 2 \
        and area <= area0 * 2 \
        and (pic1, pic2, (16, 16), weight_mat) >= 0.6: 
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.rectangle(frame, (x0, y0), (x0 + w0, y0 + h0), (255, 255, 0), 2)
            x0 = x
            y0 = y
            w0 = w
            h0 = h
            area0 = area
            noMove = False

        #print frame
        #print frame.shape

        frame2 = frame.copy()
        frame2 = MatrixToImage2(frame2)
        
        box = (x, y, x + w, y + h)
        cropImg = frame2.crop(box)

        re_size = (64, 128)
        im_resize = cropImg.resize(re_size, Image.BILINEAR)
        fp = root_path + fnprex + '%s_%s.jpg' % (j, i)
        
        #pfl.write(fp + " 0\n")  ##图片列表

        #print im_resize
        #im_resize.save(fp)
        
        cv2.imshow('frame',ImageToMatrix2(im_resize))

        #im_resize.save('test')
        #im_resize.save(fp)
		
		
        i = i + 1

    if noMove == True:
        cv2.rectangle(frame, (x0, y0), (x0 + w0, y0 + h0), (255, 255, 0), 2)


    #print "frame number %s" % j
    j = j + 1
    cv2.imshow('frame',ret)
    cv2.imshow('ori', frame)
    cv2.imshow('other',fgmask2)

    k = cv2.waitKey(30) & 0xff

    if k == 27:
        break
    if j > 2200:
        break


#pfl.close()

cap.release()
cv2.destroyAllWindows()
