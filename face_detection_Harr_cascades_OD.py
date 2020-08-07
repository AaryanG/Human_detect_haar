# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 22:06:06 2020

@author: Aaryan gupta
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np

# =============================================================================
# img1 = cv2.imread('C:/Users/Aaryan gupta/Documents/CV tutorial files/3.1 Computer-Vision-with-Python.zip/Computer-Vision-with-Python/DATA/Nadia_Murad.jpg',0)
# img2= cv2.imread('C:/Users/Aaryan gupta/Documents/CV tutorial files/3.1 Computer-Vision-with-Python.zip/Computer-Vision-with-Python/DATA/Denis_Mukwege.jpg',0)
# img3= cv2.imread('C:/Users/Aaryan gupta/Documents/CV tutorial files/3.1 Computer-Vision-with-Python.zip/Computer-Vision-with-Python/DATA/solvay_conference.jpg',0)
# plt.imshow(img3)
# =============================================================================
facecascade = cv2.CascadeClassifier('C:/Users/Aaryan gupta/Documents/CV tutorial files/3.1 Computer-Vision-with-Python.zip/Computer-Vision-with-Python/DATA/haarcascades/haarcascade_frontalface_default.xml')
def detect_face(img):
    faceimg = img.copy()
    facerects = facecascade.detectMultiScale(faceimg)
    
    for (x,y,w,h) in facerects:
        cv2.rectangle(faceimg,(x,y),(x+w,y+h),(255,255,255),10)
        
    return faceimg

# =============================================================================
# result = detect_face(img3)
# plt.imshow(result)
# plt.show()
# =============================================================================

def adjust_detect_face(img):
    faceimg = img.copy()
    facerects = facecascade.detectMultiScale(faceimg,scaleFactor=1.2,minNeighbors=5)
    
    for (x,y,w,h) in facerects:
        cv2.rectangle(faceimg,(x,y),(x+w,y+h),(255,255,255),10)
        
    return faceimg
# =============================================================================
# result = adjust_detect_face(img3)
# plt.imshow(result)
# plt.show()
# =============================================================================

# =============================================================================
# eyecascade = cv2.CascadeClassifier('C:/Users/Aaryan gupta/Documents/CV tutorial files/3.1 Computer-Vision-with-Python.zip/Computer-Vision-with-Python/DATA/haarcascades/haarcascade_eye.xml')
# 
# def detect_eyes(img):
#     faceimg = img.copy()
#     facerects = eyecascade.detectMultiScale(faceimg)
#     
#     for (x,y,w,h) in facerects:
#         cv2.rectangle(faceimg,(x,y),(x+w,y+h),(0,0,255),10)
#         
#     return faceimg
# =============================================================================

upperbodycascade = cv2.CascadeClassifier('C:/Users/Aaryan gupta/Documents/CV tutorial files/3.1 Computer-Vision-with-Python.zip/Computer-Vision-with-Python/DATA/haarcascades/haarcascade_upperbody.xml')

def detect_upperbody(img):
    faceimg = img.copy()
    facerects = upperbodycascade.detectMultiScale(faceimg)
    
    for (x,y,w,h) in facerects:
        cv2.rectangle(faceimg,(x,y),(x+w,y+h),(0,0,255),10)
        
    return faceimg
# =============================================================================
# result = detect_eyes(img2)
# plt.imshow(result)
# plt.show()
# =============================================================================

cap = cv2.VideoCapture(0)

while True:
    ret,frame = cap.read(0)
    frame1 = adjust_detect_face(frame)
    frame2 = detect_upperbody(frame)
    blended_img=cv2.addWeighted(frame1,0.5,frame2,0.5,0)
    cv2.imshow('abc',blended_img)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()



