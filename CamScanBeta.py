# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 11:32:38 2018

@author: Akshay Narla
Background Subtraction not required as it is a still image.
Trial for a document viewer and thus vehicle plate recognition from an image.
Then advance it to a video.
"""
import numpy as np
import cv2 as cv

def rectify(h):
#list of co-ordinates of end points in an order- top-left and clockwise
    h = h.reshape((4,2))
    hnew = np.zeros((4,2),dtype = np.float32)
#ordering of co-ordinates based on sum and difference   
    add = h.sum(1)
    hnew[0] = h[np.argmin(add)]
    hnew[2] = h[np.argmax(add)]

    diff = np.diff(h,axis = 1)
    hnew[1] = h[np.argmin(diff)]
    hnew[3] = h[np.argmax(diff)]    
#ordered co-ordinates
    return hnew

def four_point_transform(image, pts):
#type of transform for getting an image out of the contours or ROI\
#from ordered co-ordinates
    rect = rectify(pts)
    (tl, tr, br, bl) = rect
#the new height and width for the image    
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
#create an array of points for the scanned image     
    dst = np.array([[0, 0],[maxWidth - 1, 0],[maxWidth - 1, maxHeight - 1],[0, maxHeight - 1]], dtype = "float32")
    M = cv.getPerspectiveTransform(rect, dst)
    warped = cv.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped
#read the image file
var=cv.imread('1.jpg')
#image processing for getting proper contours
var = cv.resize(var,( 480,480 ))
ratio = var.shape[0] / 483.0
gray= cv.cvtColor(var,cv.COLOR_BGR2GRAY, cv.CV_8UC1)
gray = cv.resize(gray,( 480,480 ))
#cv.imshow('frame3',gray)
    
resize = cv.GaussianBlur( gray,(1,1),0)
edged = cv.Canny(resize, 75, 100)
#cv.imshow('frame2',edged)
ret, th3 = cv.threshold(edged ,25,200,cv.THRESH_BINARY)

im2, contours, hierarchy = cv.findContours(th3.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
cv.drawContours(im2, contours, -1, (200,50,50), 2)
#cv.imshow('frame4',im2)
#sort the contours according to their areas
cnts = sorted(contours, key = cv.contourArea, reverse = True)[:5]
for c in cnts:
    peri = cv.arcLength(c, True)
    approx = cv.approxPolyDP(c, 0.02 * peri, True)
    if len(approx) == 4:
        target = approx
        break
    
cv.drawContours(var, [target], -1, (0, 255, 0), 2)
#scanning process for the contours
scanned = four_point_transform(var, target.reshape(4, 2) * ratio)
#the threshold values can be changed or further thresholding can be done to obtain enhanced image of the scanned contour 
ret,th1 = cv.threshold(scanned,100,255,cv.THRESH_BINARY)     
#display the images
cv.imshow('Original',var)
cv.imshow('Scanned',scanned)

cv.waitKey(0)
cv.destroyAllWindows()
