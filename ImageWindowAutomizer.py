from os import walk
import cv2
from sys import exit

f = []
path = '/Users/grimmdaniel/Developer/DataScience/DS_Lab_project/raw_images_dataset/raw_images_train'
for (dirpath, dirnames, filenames) in walk(path):
    for filename in filenames:
        if '.jpg' in filename or '.png' in filename or '.JPG' in filename or '.PNG' in filename:
            f.append(filename)
    break
    
    
file = open('raw_images_dataset/window_location.csv','w')

#f = f[0:5]

currentImageName = '' 
leftPressed = False
windowName = 'imageWindow'

#Left press in, right press out
def getClickLocation(event,x,y,flags,param):
    
    global leftPressed
    
    if event == cv2.EVENT_LBUTTONDOWN and not leftPressed:
        currentLine = currentImageName + ',' + str(x) + ',' + str(y) + ',1,'
        file.write(currentLine)
        print(currentLine)
        leftPressed = True
    elif event == cv2.EVENT_RBUTTONDOWN and leftPressed:
        currentLine = str(x) + ',' + str(y) + ',0\n'
        file.write(currentLine)
        print(currentLine)
        leftPressed = False
        displayImage(f)
        
def displayImage(files):
    global currentImageName
    if not files:
        file.close()
        exit()
    else:
        print(len(files))
        currentFileName = files[-1]
        currentImageName = currentFileName
        del files[-1]
        
        img = cv2.imread(path + '/' +currentFileName)
        cv2.imshow(windowName,img)


cv2.namedWindow(winname=windowName)
cv2.setMouseCallback(windowName,getClickLocation)
displayImage(f)
cv2.waitKey(0)
file.close()