import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from numpy.lib.arraypad import pad

#shape=(높이,너비)

def pad_my1d(img: np.array, kernel : np.array ): #가로 커널인데 왜 위아래로 늘어나냐고
    
    if (kernel.shape[0]==1):
        print('가로커널')
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
        print('newimg:',newimg.shape)
        return newimg
       
    else:
        print('세로커널 들어옴')
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
        print('newimg:',newimg.shape)
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
        
        # print('가로커널')
        
        plus = int(kernel.shape[1]/2)
        # print('plus',plus)
      
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
        
        print('세로커널적용후:',img.shape)
        cv2.imshow('after',newimg)

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
    
    cv2.imshow('newimg',newimg)

def get_gaussian_filter_1d(size:int, sigma:int):
    filter=np.zeros(size)
    for i in filter:
        print(i)

    

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

# cv2.imshow('current',img)
# print(img.shape)

# kernel=np.array([[0,0,0,0,0,0,0,0,0,0,1]])
# kernel=np.array([[1],[0],[0]])
kernel=np.array([[0,0,0],
                [0,0,1],
                [0,0,0]])
# print('kernelshape:', kernel.shape)
# img=np.array([[10,20,30],
#               [40,50,60],
#               [70,80,90],
#               [100,110,120]])
# print(img.shape)

size=5
sigma=1
get_gaussian_filter_1d(size, sigma)


# cross_correlation_1d(img,kernel)
# cross_correlation_2d(img,kernel)


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
# kernel=np.array([[0,0,0]])
# print(kernel[0][1])
# kernel=np.array([0,0,0])
# print(kernel[0][1]) # index call error - 이차원 배열로 표현할 수 없음
#결론: 가로커널은 무조건 [[]]형식으로 써야한다

cv2.waitKey(0)
cv2.destroyAllWindows()


