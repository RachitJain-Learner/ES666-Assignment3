import numpy as np
import math 

def some_func():
    print('Hey Class, don\'t worry, I\'ve got your back. Just call me anytime you need help… just not too many times, I\'m on a tight loop')
    return None

def GetPixels(img, x, y): 
    x_floor = math.floor(x)
    y_floor = math.floor(y)
    dx = x - x_floor
    dy = y - y_floor

    p1 = img[y_floor, x_floor]
    p2 = img[y_floor, math.floor(x + 1)]
    p3 = img[math.floor(y + 1), x_floor]
    p4 = img[math.floor(y + 1), math.floor(x + 1)]
    
    w1 = pow(pow(dx, 2) + pow(dy, 2), -0.5)
    w2 = pow(pow(1 - dx, 2) + pow(dy, 2), -0.5)
    w3 = pow(pow(dx, 2) + pow(1 - dy, 2), -0.5)
    w4 = pow(pow(1 - dx, 2) + pow(1 - dy, 2), -0.5)

    sum_w = w1 + w2 + w3 + w4
    return (w1 * p1 + w2 * p2 + w3 * p3 + w4 * p4) / sum_w

def GetCorners(img, H):
    img_corners = np.array([
        [0, 0, 1],
        [0, img.shape[1], 1],
        [img.shape[0], 0, 1],
        [img.shape[0], img.shape[1], 1]
    ]).T
    
    img_corners_range = np.matmul(H,img_corners)

    for i in range(img_corners_range.shape[1]):
        img_corners_range[:,i] = img_corners_range[:,i]/img_corners_range[-1,i]

    return img_corners_range[0:2,:]

def Final_Panaroma(range_img, domain_img, H, offsetXY):
    H_inv = np.linalg.inv(H)
    
    for i in range(0,range_img.shape[0]): # row
        for j in range(0,range_img.shape[1]): # col
                X_domain = np.array([j+offsetXY[0],i+offsetXY[1], 1])
                X_range = np.array(np.matmul(H_inv,X_domain))
                X_range = X_range/X_range[-1]
                if (X_range[0]>0 and X_range[1]>0 and X_range[0]<domain_img.shape[1]-1 and X_range[1]<domain_img.shape[0]-1):
                    range_img[i][j] = GetPixels(domain_img, X_range[0], X_range[1])
    return range_img

