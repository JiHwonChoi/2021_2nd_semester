import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from numpy.lib.arraypad import pad

#shape=(높이,너비)

def pad_my1d(img: np.array, kernel : np.array ):
    
    if (len(kernel.shape)==1):
        #가로길이 = kernel.shape[0]
        plus = int(kernel.shape[0]/2)  
        right_edge=img[:,img.shape[0]-1]
        first_edge=img[:,0]
        pad_right = right_edge
        pad_left=first_edge
        for _ in range(200):
            pad_right = np.column_stack([right_edge,pad_right])
            pad_left = np.column_stack([pad_left,first_edge])
        newimg=np.column_stack([pad_left,img])
        newimg=np.column_stack([newimg,pad_right]) 
        cv2.imshow('pad_my',newimg)
        return newimg
       
    else:
        
        #세로길이 = kernel.shape[0]
        plus = int (kernel.shape[0]/2)  
        top_edge=img[0,:]
        bottom_edge=img[img.shape[0]-1,:]
        pad_top=top_edge
        pad_bottom=bottom_edge
        for _ in range(200):
            pad_top=np.vstack([pad_top,top_edge])
            pad_bottom=np.vstack([pad_bottom,bottom_edge])
        newimg=np.vstack([pad_top,img])
        newimg=np.vstack([newimg,pad_bottom])
        return newimg

def pad_my2d(img: np.array, kernel : np.array ):
   
        plus = int(kernel.shape[0]/2)
        
        right_edge=img[:,img.shape[0]-1]
        first_edge=img[:,0]
        pad_right = right_edge
        pad_left=first_edge
        for _ in range(200):
            pad_right = np.column_stack([right_edge,pad_right])
            pad_left = np.column_stack([pad_left,first_edge])
        newimg=np.column_stack([pad_left,img])
        newimg=np.column_stack([newimg,pad_right])
        
       
        
        top_edge=newimg[0,:]
        bottom_edge=newimg[newimg.shape[0]-1,:]
        pad_top=top_edge
        pad_bottom=bottom_edge
        for _ in range(200):
            pad_top=np.vstack([pad_top,top_edge])
            pad_bottom=np.vstack([pad_bottom,bottom_edge])
        newimg=np.vstack([pad_top,newimg])
        newimg=np.vstack([newimg,pad_bottom])
        
        return newimg

def cross_correlation_1d(img:np.array, kernel:np.array):
    img=pad_my1d(img,kernel)
    width=int(img.shape[1])
    height=int(img.shape[0])
    
    
    #커널의 길이 파악
    if (len(kernel.shape)==1 or kernel.shape(0)==1):
        print('가로커널')
    else:
        print('세로커널')
    
    
    

# print(os.listdir(os.getcwd()))
img=cv2.imread('Computer_Vision/Assignment1/soyoonkim.JPG',cv2.IMREAD_GRAYSCALE)

#resizing----------------
# width = int(img.shape[1] * 1.2)
# height = int(img.shape[0] * 1.2)
# dim=(width,height)
# imgBig=cv2.resize(img,dim, interpolation=cv2.INTER_NEAREST )
# cv2.imshow('image',imgBig)

#padding----------
img=cv2.imread('Computer_Vision/Assignment1/lenna.png',cv2.IMREAD_GRAYSCALE)


# newimg=cross_correlation_1d(newimg,kernel)
kernel=np.array([0,0,1])

cross_correlation_1d(img,kernel)


#cv2 의 blur 써보기----
# kernel1 = np.ones((5,5), np.float32)/25
# kernel2 = np.ones((11,11), np.float32)/121
# blur1=cv2.filter2D(img,-1,kernel1)
# blur2 = cv2.filter2D(img,-1, kernel2)
# cv2.imshow('blur',blur1)
# cv2.imshow('blur2', blur2)

#커널 가로세로 구분 하기-----
# kernel=np.array([[0,0,0]]) #shape = (1,3) []없으면 (3,)
# print(kernel.shape)
# kernel=np.array([[0],[0],[0]]) #shape = (3,1)
# print(kernel.shape)

#커널의 가로가 [] 일때랑 [[]]일때랑 인덱싱 달라질까?
kernel=np.array([[0,0,0]])
print(kernel[0][1])
kernel=np.array([0,0,0])
print(kernel[0][1])

# cv2.waitKey(0)
# cv2.destroyAllWindows()


