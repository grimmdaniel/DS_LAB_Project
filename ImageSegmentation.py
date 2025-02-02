import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import datetime

border_size = 3
number_of_trees = 100

data_set_url = 'raw_images_dataset/output_train_set.csv'
training_set = pd.read_csv(data_set_url, header=0, sep=',')

header = ['red_pixel','green_pixel','blue_pixel','mean_red','mean_green','mean_blue','std_red','std_green','std_blue','corr_rg','corr_rb','corr_gb','inside']

X = training_set.iloc[:,:-1].values
y = training_set.iloc[:,12].values

from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model_name = 'random_forest_scaler.pkl'
pickle.dump(scaler, open(model_name, 'wb'))

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = number_of_trees , verbose=3)
classifier.fit(X_train,y_train)

from sklearn.metrics import confusion_matrix
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm)
cm_df
cm_df.columns = ['1','0']
cm_df = cm_df.rename(index={0: '1',1:'0'})
cm_df

model_name = 'random_forest_model.pkl'
pickle.dump(classifier, open(model_name, 'wb'))
#classifier = pickle.load(open(model_name, 'rb'))

from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))

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

    
def predictNewImage(imageName,data,width,height):
    classification_results = classifier.predict(data)
    result_matrix = classification_results.reshape((width-border_size*2,height-border_size*2))
    for i in range(0,result_matrix.shape[0]):
        for j in range(0,result_matrix.shape[1]):
            if result_matrix[i][j] == 0:
                result_image[i+border_size][j+border_size] = 0
    cv2.imwrite(imageName + '_segmentation_result_' + str(number_of_trees) + '_trees_' + str(datetime.datetime.utcnow().date()) +'.jpg',cv2.cvtColor(result_image,cv2.COLOR_BGR2RGB))
    


def processPixelsWithRegions(pixel_region):
    #image processing methods here...
    # means of R G B channels

    red_channel = np.reshape(pixel_region[:,:,0], -1)
    green_channel = np.reshape(pixel_region[:,:,1], -1)
    blue_channel = np.reshape(pixel_region[:,:,2], -1)
    
    #reshaped_red = red_channel.reshape((7,7,1))
    #histr = cv2.calcHist([reshaped_red],[0],None,[256],[0,256])
    
    #std_hist_r = np.std(np.histogram(blue_channel, bins=bins)[0])
    
    
    #bins = [x for x in range(0,256)]
    
    #hp_red = np.histogram(red_channel, bins=bins)
    #std_hist_r = calculateStandardDeviationHistogram(hp_red[0],hp_red[1])
    
    #hp_green = np.histogram(green_channel, bins=bins)
    #std_hist_g = calculateStandardDeviationHistogram(hp_green[0],hp_green[1])
    
    #hs_blue = np.histogram(blue_channel, bins=bins)
    #std_hist_b = calculateStandardDeviationHistogram(hs_blue[0],hs_blue[1])
    

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

imageName = 'images_to_test/0134.jpg'
#imageName = 'test.jpg'
image_array = openImage(imageName)
result_image = image_array.copy()

t1=datetime.datetime.utcnow()
print('Started at: ' + str(t1))

returnedData = processImage(image_array)
createdDataFrame = returnedData[0]
newImageData = createdDataFrame.iloc[:,:].values

#scaler = pickle.load(open('random_forest_scaler.pkl', 'rb'))
#classifier = pickle.load(open('random_forest_model.pkl', 'rb'))

newImageData = scaler.transform(newImageData)
predictNewImage(imageName.split('/')[1],newImageData,returnedData[1],returnedData[2])
t2=datetime.datetime.utcnow()
print('Elapsed time: ' + str(t2-t1))