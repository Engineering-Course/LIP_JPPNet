# This function is used to get the largest square from the cropped and segmented image. It can be further used to find patterns
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time
from collections import namedtuple
import glob

def printMaxSubSquare(M): 
    """" find the largest square """  
    R = len(M) # no. of rows in M[][] 
    C = len(M[0]) # no. of columns in M[][] 
   
    S = [[0 for k in range(C)] for l in range(R)] 
    # here we have set the first row and column of S[][] 
  
    # Construct other entries 
    for i in range(1, R): 
        for j in range(1, C): 
            if (M[i][j] == 1): 
                S[i][j] = min(S[i][j-1], S[i-1][j], 
                            S[i-1][j-1]) + 1
            else: 
                S[i][j] = 0
      
    # Find the maximum entry and 
    # indices of maximum entry in S[][] 
    max_of_s = S[0][0] 
    max_i = 0
    max_j = 0
    for i in range(R): 
        for j in range(C): 
            if (max_of_s < S[i][j]): 
                max_of_s = S[i][j] 
                max_i = i 
                max_j = j 
  
    print("Maximum size sub-matrix is: ") 
    count_i = 0
    count_j = 0
    position_matrix = []
    for i in range(max_i, max_i - max_of_s, -1): 
        for j in range(max_j, max_j - max_of_s, -1): 
            position_matrix.append((i,j)) 
        count_i+=1 
        
    print('count_i :' + str(count_i))
    print('count_j :' + str(count_j))
    return position_matrix


def crop_square_portion(image_file_name):
    """" crop and save image """    
    image_file_name_list = image_file_name.split('_')
    vis_file_name = '_'.join(image_file_name_list[:2])+'_vis.png'
    save_file_name = '_'.join(image_file_name_list[:3])+'_square.png'
    cloth_type = image_file_name_list[-2]
    list_index = cloth_type_list.index(cloth_type)
    light_shade = light_shade_list[list_index]
    dark_shade = dark_shade_list[list_index]
    print(light_shade,dark_shade)
    #read input image
    img = cv2.imread(INPUT_DIR+vis_file_name,cv2.COLOR_BGR2RGB)
    
    #detect shades from vis:
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, light_shade, dark_shade)
    
    #coverting to binary array:
    np_img = np.array(mask)
    np_img[np_img == 255] = 1
    
    #coverting to binary array:
    np_img = np.array(mask)
    np_img[np_img == 255] = 1

    #find and plot the largest square
    var = printMaxSubSquare(np_img)
    for point in var:
        a,b = point
        pt = (b,a)
        cv2.circle(np_img,pt,5,(200,0,0),2)

    ##convert mask to bunary mask
    np_img[np_img != 200] = 0
    print('final mask shape:')
    print(np_img.shape)

    ##crop and save the square image
    img = cv2.imread(INPUT_DIR+image_file_name,cv2.COLOR_BGR2RGB)
    print('input image shape:')
    print(img.shape)
    x,y,w,h = cv2.boundingRect(np_img)
    crop_img = img[y:y+h,x:x+w]
    print('cropped image shape:')
    print(crop_img.shape)
    cv2.imwrite(OUTPUT_DIR+save_file_name, crop_img)
    

if __name__ == "__main__":    
    INPUT_DIR = r' set your input folder where segmented images are there'
    OUTPUT_DIR = r' set your output images'
    cloth_type_list = ['UpperClothes','Dress','Pants','Scarf','Skirt','Coat']
    light_shade_list = [(100, 240, 255),(0,255,70),(0,255,70),(10,150,125),(50,0,70),(10,100,200)]
    dark_shade_list = [(190, 255, 255),(0,255,200),(100,255,200),(100,160,130),(60,255,200),(20,255,255)]

    #for each bgcropped file read, pass to crop_image function 
    for file in glob.glob(INPUT_DIR+'*_cropped.png'):
        print(file)
        image_file_name = file.split('\\')[-1]
        crop_square_portion(image_file_name)
    
