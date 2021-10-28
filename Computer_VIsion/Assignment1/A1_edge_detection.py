import gaussian as g
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import gaussian as g
from numpy.lib.arraypad import pad
from math import exp
from math import pi


def compute_image_gradient(img:np.array):
    a=([[1],[2],[1]])
    b=([[-1,0,1]])
    a=np.array(a) 
    b=np.array(b)
    sobel_x=a.dot(b)
    print(sobel_x)
    # sum=0
    # for i in range(sobel_x.shape[0]):
    #     for j in range(sobel_x.shape[1]):
    #         sum+=sobel_x[i][j]
    # sobel_x=sobel_x/sum
    
    c=([[-1],[0],[1]])
    d=([[1,2,1]])
    c=np.array(c)
    d=np.array(d) 
    sobel_y=c.dot(d)
    print(sobel_y)
    # sum=0
    # for i in range(sobel_y.shape[0]):
    #     for j in range(sobel_y.shape[1]):
    #         sum+=sobel_y[i][j]
    # sobel_y=sobel_y/sum
    
    img_sobel_x=g.cross_correlation_2d(img,sobel_x)
    
    img_sobel_y=g.cross_correlation_2d(img,sobel_y)
    
    img_gradient=((img_sobel_x**2)+(img_sobel_y**2))**(0.5)
    cv2.imshow('res',img_gradient)
    return img_gradient
    

imgName='lenna.png'
img=cv2.imread('./{0}'.format(imgName),cv2.IMREAD_GRAYSCALE)

kernel=g.get_gaussian_filter_2d(7,1.5)
img=g.cross_correlation_2d(img,kernel)

# cv2.imshow('2-1',img)

gradient=compute_image_gradient(img)
cv2.imwrite('./result/part_2_edge_sup_{0}'.format(imgName),gradient)

imgName='shapes.png'
img=cv2.imread('./{0}'.format(imgName),cv2.IMREAD_GRAYSCALE)

kernel=g.get_gaussian_filter_2d(7,1.5)
img=g.cross_correlation_2d(img,kernel)

# cv2.imshow('2-1',img)

gradient=compute_image_gradient(img)
cv2.imwrite('./result/part_2_edge_sup_{0}'.format(imgName),gradient)
cv2.waitKey(0)
cv2.destroyAllWindows()