from os import walk
import cv2

f = []
path = '/Users/grimmdaniel/Developer/DataScience/DS_Lab_project/raw_images/'
for (dirpath, dirnames, filenames) in walk(path):
    f.extend(filenames)
    break
    
if '.DS_Store' in f:
    f.remove('.DS_Store')    
    
file = open('window_location.csv','w')

#f = f[0:5]

currentImageName = '' 
leftPressed = False
windowName = 'imageWindow'

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
        
        img = cv2.imread('raw_images/'+currentFileName)
        cv2.imshow(windowName,img)


cv2.namedWindow(winname=windowName)
cv2.setMouseCallback(windowName,getClickLocation)
displayImage(f)
cv2.waitKey(0)
file.close()