import sys
import cv2

arguments = sys.argv
if len(arguments) > 3:
    print('OK')
    
def openImage(image_path):
    img = cv2.imread(image_path)
    fix_img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    return fix_img

def createConfidenceMatrix(result):
    matrix_string = '[' + createFormattedNumberString(result[0]) + ' ,' + createFormattedNumberString(result[1]) + ' ]\n' + '[' + createFormattedNumberString(result[2]) + ' ,' + createFormattedNumberString(result[3]) + ' ]\n' + calculateAccuracy(result) 
    return matrix_string
    
def createFormattedNumberString(number, desired_length = 8):
    number_as_string = str(number)
    return_string = ' '*(desired_length - len(number_as_string)) + number_as_string
    return return_string

def calculateAccuracy(result):
    all_observations = sum(result)
    accuracy = (result[0] + result[3]) / all_observations
    return 'Accuracy: {}%'.format(round(accuracy,4))

def writeResultsToFile(results):
    with open(y_pred_file + '.txt', 'w') as file_to_write:
        for index in range(0,len(results)):
            file_to_write.write(y_test_files[index]+'\n')
            file_to_write.write(createConfidenceMatrix(results[index]))
            file_to_write.write('\n\n')
    
y_test_files = arguments[1:-1]
y_pred_file = arguments[-1]

y_test_images = [openImage(image_path) for image_path in y_test_files]
y_pred_image = openImage(y_pred_file)

shape = y_pred_image.shape
width, height = shape[1], shape[0]

negative = [0,0,0]

results = [[0]*4 for i in range(0,len(y_test_files))]
for i in range(0,width):
    for j in range(0,height):
        current_pred_element = y_pred_image[j][i]
        for imageindex in range(0,len(y_test_images)):
            current_test_pixel = list(y_test_images[imageindex][j][i])
            current_pred_pixel = list(y_pred_image[j][i])
            if current_test_pixel == negative and current_pred_pixel == negative:
                results[imageindex][3] += 1 #black both of them
            elif current_test_pixel == negative and not current_pred_pixel == negative:
                results[imageindex][1] += 1 #pred is inside but it was outside(black)
            elif not current_test_pixel == negative and current_pred_pixel == negative:
                results[imageindex][2] += 1 #pred is outside but it was inside
            elif not current_test_pixel == negative and not current_pred_pixel == negative:
                results[imageindex][0] += 1 #leaf both of them

writeResultsToFile(results)