import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import gaussian as g
from numpy.lib.arraypad import pad
from math import exp
from math import pi
import time


#shape=(높이,너비)

def pad_my1d(img: np.array, kernel : np.array ): #가로 커널인데 왜 위아래로 늘어나냐고
    
    if (kernel.shape[0]==1):
        # print('가로커널')
        #가로길이 = kernel.shape[1]
        plus = int(kernel.shape[1]/2)  
        right_edge=img[:,img.shape[1]-1]
        first_edge=img[:,0]
        
        # print('right edge',right_edge)
        # print('first edge', first_edge)
        
        pad_right = right_edge
        pad_left=first_edge
        for _ in range(plus-1):
            pad_right = np.column_stack([right_edge,pad_right])
            pad_left = np.column_stack([pad_left,first_edge])
        newimg=np.column_stack([pad_left,img])
        newimg=np.column_stack([newimg,pad_right]) 
        # cv2.imshow('pad_my',newimg)
        # print('newimg:',newimg.shape)
        return newimg
       
    else:
        # print('세로커널 들어옴')
        #세로길이 = kernel.shape[0]
        plus = int (kernel.shape[0]/2)  
        top_edge=img[0,:]
        bottom_edge=img[img.shape[0]-1,:]
        pad_top=top_edge
        pad_bottom=bottom_edge
        for _ in range(plus-1):
            pad_top=np.vstack([pad_top,top_edge])
            pad_bottom=np.vstack([pad_bottom,bottom_edge])
        newimg=np.vstack([pad_top,img])
        newimg=np.vstack([newimg,pad_bottom])
        # print('newimg:',newimg.shape)
        return newimg

def pad_my2d(img: np.array, kernel : np.array ):

        plus = int(kernel.shape[0]/2)
        
        right_edge=img[:,img.shape[1]-1]
        first_edge=img[:,0]
        pad_right = right_edge
        pad_left=first_edge
        for _ in range(plus-1):
            pad_right = np.column_stack([right_edge,pad_right])
            pad_left = np.column_stack([pad_left,first_edge])
        newimg=np.column_stack([pad_left,img])
        newimg=np.column_stack([newimg,pad_right])
        
       
        
        top_edge=newimg[0,:]
        bottom_edge=newimg[newimg.shape[0]-1,:]
        pad_top=top_edge
        pad_bottom=bottom_edge
        for _ in range(plus-1):
            pad_top=np.vstack([pad_top,top_edge])
            pad_bottom=np.vstack([pad_bottom,bottom_edge])
        newimg=np.vstack([pad_top,newimg])
        newimg=np.vstack([newimg,pad_bottom])
        
        return newimg

def cross_correlation_1d(img:np.array, kernel:np.array):
    img=pad_my1d(img,kernel)
    newimg=img.copy()
    
    width=int(img.shape[1])
    height=int(img.shape[0])
    
    #커널의 길이 파악
    if (kernel.shape[0]==1):
        len_kernel=kernel.shape[1]
        plus = int(kernel.shape[1]/2)
        for i in range(height):
            for j in range(width-(2*plus)):
                a=0
                for k in range(len_kernel):
                    # print('img:',img[i][j+k])
                    # print('kernel:',kernel[0][k])
                    a+=img[i][j+k]*kernel[0][k]
                newimg[i][j+plus]=a
                # temp_img=img[i][j]~img[i][j+len(kernel)] #배열 슬라이싱 하는거 알아야함
                # img[i][j+plus] = temp_img*kernel
        #붙였던 것 떨궈내기 (앞으로 plus 뒤로 plus)
        newimg=newimg[:,plus-1:width-plus-1]
        # cv2.imshow('after',img)       
        # print(img.shape) 
      
    else:
        plus = int(kernel.shape[0]/2)
        len_kernel=kernel.shape[0]
        for j in range(width):
            for i in range(height-(2*plus)):
                a=0
                for k in range(len_kernel):
                    a+=img[i+k][j]*kernel[k][0]
                newimg[i+k][j]=a
        newimg=newimg[plus-1:height-plus-1,:]
    
    return newimg
        
        # print('세로커널적용후:',img.shape)
        # cv2.imshow('after',newimg)

def cross_correlation_2d(img:np.array, kernel:np.array):  
    img=pad_my2d(img,kernel)
    # print(img)
    newimg=img.copy()
    
    width=int(img.shape[1])
    height=int(img.shape[0])
    plus_w=int(kernel.shape[1]/2)
    plus_h=int(kernel.shape[0]/2)
    
    for i in range(height-(2*plus_h)):
        for j in range(width-(2*plus_w)):
            a=0
            # print(i,j)
            for h in range(kernel.shape[0]):
                for w in range(kernel.shape[1]):
                    a+=img[i+h][j+w]*kernel[h][w]
            newimg[i+plus_h][j+plus_w]=a
            # print('({0},{1})'.format(i+plus_h,j+plus_w),end=' ')
            # print(' ')
    # print(newimg)
    newimg=newimg[plus_h:height-plus_h,plus_w:width-plus_w]
    # print(newimg.shape)
    # print(newimg)
    
    # cv2.imshow('newimg',newimg)
    return newimg

def get_gaussian_filter_1d(size:int, sigma:int):
    filter=np.zeros(size)
    center=int(size/2)
    for i in range(size):
        filter[i]=exp(-((center-i)**2)/2*(sigma**2))/(2*pi*(sigma**2))
    
    
    sum=0
    for i in filter:
        sum+=i
    
    return filter/sum
    
def get_gaussian_filter_2d(size:int, sigma:int):
    filter=np.zeros((size,size))
    center=int(size/2)
    for i in range(size):
        for j in range(size):
            filter[i][j]=exp(-(((i-center)**2+(j-center)**2)/(2*(sigma**2)))/(2*pi*(sigma**2)))
    
    
    sum=0
    for i in filter:
        for j in i:
            sum+=j

    return filter/sum
    
def filter9(img: np.array):
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10,40)
    fontScale              = 1
    fontColor              = (1,1,1)
    lineType               = 2
    
    sizearr=[5,11,17]
    sigmaarr=[1,6,11]
    imgrow=img.copy()
    imgcol=img.copy()
    
    for i in sizearr:
        
        for j in sigmaarr:
            kernel=get_gaussian_filter_2d(i,j)
            img_ij=cross_correlation_2d(img,kernel)
            cv2.putText(img_ij,'{0}x{0} s={1}'.format(i,j), 
            bottomLeftCornerOfText, 
            font, 
            fontScale,
            fontColor,
            lineType)
            if(j==1):
                imgrow=img_ij.copy()
            else:
                imgrow=np.column_stack([imgrow,img_ij])
        if(i==5):
            imgcol=imgrow.copy()
        else:
            imgcol=np.row_stack([imgcol,imgrow])       
    return imgcol

def filter9_1d_vertical(img: np.array):
    
    #가로 가우시안 커널만들기
    size=5
    sigma=1
    kernel=get_gaussian_filter_1d(size,sigma) #가로 가우시안 커널
    kernel_horizontal=kernel[np.newaxis]
    start_time_1d=time.time()
    img_1d_horizontal=cross_correlation_1d(img,kernel_horizontal)
    end_time_1d=time.time()
    
    kernel_vertical=kernel[:,np.newaxis]
    img_1d_vertical=cross_correlation_1d(img,kernel_vertical)
    #2d 구하기
    kernel_2d=get_gaussian_filter_2d(size,sigma)
    start_time_2d=time.time()
    img_2d=cross_correlation_2d(img,kernel_2d)
    end_time_2d=time.time()
    
    print('1D computational time:{0}'.format(end_time_1d-start_time_1d))
    print('2D computational time:{0}'.format(end_time_2d-start_time_2d))
    cv2.imshow('difference with vertical', (img_1d_vertical-img_2d))
    cv2.imshow('difference with horizontal', (img_1d_horizontal-img_2d))
    
    


# cv2.imshow('current',img)
# print(img.shape)

# kernel=np.array([[0,0,0,0,0,0,0,0,0,0,1]])
# kernel=np.array([[1],[0],[0]])
# kernel=np.array([[0,0,0],
#                 [0,0,1],
#                 [0,0,0]])
# print('kernelshape:', kernel.shape)
# img=np.array([[10,20,30],
#               [40,50,60],
#               [70,80,90],
#               [100,110,120]])
# print(img.shape)

# size=5
# sigma=1
# # get_gaussian_filter_1d(size, sigma)
# kernel=get_gaussian_filter_2d(size, sigma)

# # cross_correlation_1d(img,kernel)
# img1=cross_correlation_2d(img,kernel)

#(d)
imgName='lenna.png'
img=cv2.imread('./{0}'.format(imgName),cv2.IMREAD_GRAYSCALE)
# cv2.imshow('test',img)
cv2.imwrite('./result/part_1_gaussian_filtered_{0}'.format(imgName),filter9(img))

imgName='shapes.png'
img=cv2.imread('Computer_Vision/Assignment1/{0}'.format(imgName),cv2.IMREAD_GRAYSCALE)
cv2.imwrite('./result/part_1_gaussian_filtered_{0}'.format(imgName),filter9(img))

#(e)
imgName='lenna.png'
img=cv2.imread('./{0}'.format(imgName),cv2.IMREAD_GRAYSCALE)
filter9_1d_vertical(img)

imgName='shapes.png'
img=cv2.imread('Computer_Vision/Assignment1/{0}'.format(imgName),cv2.IMREAD_GRAYSCALE)
filter9_1d_vertical(img)

cv2.waitKey(0)
cv2.destroyAllWindows()


