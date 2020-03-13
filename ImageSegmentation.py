import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

border_size = 3

def openImage(image_name):
    img = cv2.imread('test_set/raw_images_test/' + image_name)
    fix_img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    return fix_img

def processImage(image):
    raw_data_list = []
    width, height = image.shape[0], image.shape[1]
    for i in range(border_size,width-border_size):
        for j in range(border_size,height-border_size):
            pixel_region = image[i-border_size:i+border_size+1,j-border_size:j+border_size+1]
            raw_data_list.append(processPixelsWithRegions(pixel_region))
    dataFrame = pd.DataFrame(data = raw_data_list,columns = columns)
    scaler = MinMaxScaler()
    scaler.fit(dataFrame)
    classification_results = random_forest_model.predict(dataFrame)
    result_matrix = classification_results.reshape((width-border_size*2,height-border_size*2))
    for i in range(0,result_matrix.shape[0]):
        for j in range(0,result_matrix.shape[1]):
            if result_matrix[i][j] == 0:
                result_image[i+border_size][j+border_size] = 0
    

def processPixelsWithRegions(pixel_region):
    #image processing methods here...
    # means of R G B channels
    red_channel = np.reshape(pixel_region[:,:,0], -1)
    green_channel = np.reshape(pixel_region[:,:,1], -1)
    blue_channel = np.reshape(pixel_region[:,:,2], -1)
    
    center_of_pixel_region = pixel_region[border_size:border_size+1,border_size:border_size+1,:]
    center_red_pixel = center_of_pixel_region[:,:,0:1].item()
    center_green_pixel = center_of_pixel_region[:,:,1:2].item()
    center_blue_pixel = center_of_pixel_region[:,:,2:3].item()

    std_red = np.std(red_channel)
    std_green = np.std(green_channel)
    std_blue = np.std(blue_channel)

    if std_red == 0 or std_green == 0 or std_blue == 0:
        return [float(center_red_pixel),float(center_green_pixel),float(center_blue_pixel),0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]

    #correlation between Red and Green channels
    corr_rg = np.corrcoef(red_channel,green_channel)[0][1]
    
    #correlation between Red and Blue channels
    corr_rb = np.corrcoef(red_channel,blue_channel)[0][1]
    
    #correlation between Green and Blue channels
    corr_gb = np.corrcoef(green_channel,blue_channel)[0][1]
  
    #means of rgb channels + standard deviation of rgb channels
    return [float(center_red_pixel),float(center_green_pixel),float(center_blue_pixel),np.mean(red_channel),np.mean(green_channel),np.mean(blue_channel),std_red,std_green,std_blue,corr_rg,corr_rb,corr_gb]

columns = ['red_pixel','green_pixel','blue_pixel','mean_red','mean_green','mean_blue','std_red','std_green','std_blue','corr_rg','corr_rb','corr_gb']
random_forest_model = pickle.load(open('randomforest_classifier_model.pkl', 'rb'))
imageName = '0323.jpg'
#imageName = 'test.jpg'
image_array = openImage(imageName)
result_image = image_array.copy()
processImage(image_array)
cv2.imwrite('segmentation_result.jpg',cv2.cvtColor(result_image,cv2.COLOR_BGR2RGB))