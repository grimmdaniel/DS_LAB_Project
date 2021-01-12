import cv2
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import datetime
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler

scaler = pickle.load(open('random_forest_scaler.pkl', 'rb'))
classifier = pickle.load(open('random_forest_model.pkl', 'rb'))

border_size = 3
number_of_trees = 100

def openImage(image_name):
    img = cv2.imread(image_name)
    fix_img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    return fix_img

def processImage(image):
    raw_data_list = []
    width, height = image.shape[0], image.shape[1]
    for i in range(border_size,width-border_size):
        for j in range(border_size,height-border_size):
            pixel_region = image[i-border_size:i+border_size+1,j-border_size:j+border_size+1]
            result = processPixelsWithRegions(pixel_region)
            raw_data_list.append(result)
    columns = ['red_pixel','green_pixel','blue_pixel','mean_red','mean_green','mean_blue','std_red','std_green','std_blue','corr_rg','corr_rb','corr_gb']
    dataFrame = pd.DataFrame(data = raw_data_list,columns=columns)
    return (dataFrame,width,height)

    
def predictNewImage(imageName,data,width,height,mask):
    classification_results = classifier.predict(data)
    result_matrix = classification_results.reshape((width-border_size*2,height-border_size*2))
    for i in range(0,result_matrix.shape[0]):
        for j in range(0,result_matrix.shape[1]):
            if result_matrix[i][j] == 0:
                mask[i+border_size][j+border_size] = 0
            else:
                mask[i+border_size][j+border_size] = 1
    return(imageName,mask)
    


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
        return [round(float(center_red_pixel),6),round(float(center_green_pixel),6),round(float(center_blue_pixel),6),0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]

    #correlation between Red and Green channels
    corr_rg = np.corrcoef(red_channel,green_channel)[0][1]
    
    #correlation between Red and Blue channels
    corr_rb = np.corrcoef(red_channel,blue_channel)[0][1]
    
    #correlation between Green and Blue channels
    corr_gb = np.corrcoef(green_channel,blue_channel)[0][1]
  
    #means of rgb channels + standard deviation of rgb channels
    return [round(float(center_red_pixel),6),round(float(center_green_pixel),6),round(float(center_blue_pixel),6),round(np.mean(red_channel),6),round(np.mean(green_channel),6),round(np.mean(blue_channel),6),round(std_red,6),round(std_green,6),round(std_blue,6),round(corr_rg,6),round(corr_rb,6),round(corr_gb,6)]

def applyErosionOnMask(mask):
    kernel = np.array ([[0, 1, 0],
                [1, 1, 1],
                [0, 1, 0]], dtype = np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.erode(src = mask, kernel = kernel, iterations = 3)
    return mask

def applyMask(originalImage, mask):
    for i in range(0,originalImage.shape[0]):
        for j in range(0,originalImage.shape[1]):
            if mask[i][j] == 0:
                originalImage[i][j] = 0
    return originalImage
    

imageName = 'images_to_test/028.jpg'
#imageName = 'test.jpg'
image_array = openImage(imageName)
image_for_final = image_array.copy()

#resize image
ratio = 1/2
image_array = cv2.resize(image_array,(0,0),image_array,ratio,ratio)
shape = image_array.shape
empty_mask = np.zeros((shape[0],shape[1]),dtype=float)


t1=datetime.datetime.utcnow()
print('Started at: ' + str(t1))
returnedData = processImage(image_array)
createdDataFrame = returnedData[0]
newImageData = createdDataFrame.iloc[:,:].values

newImageData = scaler.transform(newImageData)
imageName, mask = predictNewImage(imageName.split('/')[1],newImageData,returnedData[1],returnedData[2],empty_mask)

#Enlarge image back to its original size
ratio = 2
mask = mask.reshape((shape[0],shape[1],1))
mask = cv2.resize(mask,(0,0),mask,ratio,ratio)
mask = applyErosionOnMask(mask)

image_for_final = applyMask(image_for_final, mask)

cv2.imwrite('results/' + imageName + '_segmentation_result_' + str(number_of_trees) + '_trees_' + str(datetime.datetime.utcnow().date()) +'.jpg',cv2.cvtColor(image_for_final,cv2.COLOR_BGR2RGB))
t2=datetime.datetime.utcnow()
print('Elapsed time: ' + str(t2-t1))