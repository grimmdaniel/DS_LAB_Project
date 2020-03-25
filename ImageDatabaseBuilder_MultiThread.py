import numpy as np
import cv2
import sys
import datetime
import multiprocessing as mp

window_size = 106
border_size = 3

def readImageWindowLocationsFile():
	file = open('train_set/window_location.csv','r')
	return file

def createWindowLocations(line):
	splittedLine = line.split(',')
	return(splittedLine[0],(int(splittedLine[1]),int(splittedLine[2])),(int(splittedLine[4]),int(splittedLine[5])))

def openImageWindow(image):
	image_name = image[0]
	img = cv2.imread('train_set/raw_images_train/' + image_name)
	fix_img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
	
	window_inside_x = image[1][0]
	window_inside_y = image[1][1]

	window_outside_x = image[2][0]
	window_outside_y = image[2][1]
	
	window_inside = fix_img[window_inside_y:window_inside_y+window_size,window_inside_x:window_inside_x+window_size]
	window_outside = fix_img[window_outside_y:window_outside_y+window_size,window_outside_x:window_outside_x+window_size]
	return (window_inside,window_outside)

def processImageWindows(q,window,is_inside):
	for i in range(border_size,window_size-border_size):
		for j in range(border_size,window_size-border_size):
			pixel_region = window[i-border_size:i+border_size+1,j-border_size:j+border_size+1]
			result = processPixelsWithRegions(pixel_region)
			if result != '':
				writeResultToFile(q,result,is_inside)
			
	
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
		return ''

	#correlation between Red and Green channels
	corr_rg = np.corrcoef(red_channel,green_channel)[0][1]
	
	#correlation between Red and Blue channels
	corr_rb = np.corrcoef(red_channel,blue_channel)[0][1]
	
	#correlation between Green and Blue channels
	corr_gb = np.corrcoef(green_channel,blue_channel)[0][1]
  
	#means of rgb channels + standard deviation of rgb channels
	result = str(center_red_pixel) + ',' + str(center_green_pixel) + ',' + str(center_blue_pixel) + ',' + str(np.mean(red_channel)) + ',' + str(np.mean(green_channel)) + ',' + str(np.mean(blue_channel)) + ',' + str(std_red) + ',' + str(std_green) + ',' + str(std_blue)  + ',' + str(corr_rg)  + ',' + str(corr_rb)  + ',' + str(corr_gb)
	return result
	
def createFileForOutput():
	o_file = open('train_set/output_train_set_multi.csv','w+')
	return o_file
	
def writeResultToFile(q,result,is_inside):
	if is_inside:
		q.put(result+',1\n')
	else:
		q.put(result+',0\n')

t1=datetime.datetime.utcnow()
output_file = createFileForOutput()
output_file.write('red_pixel,green_pixel,blue_pixel,mean_red,mean_green,mean_blue,std_red,std_green,std_blue,corr_rg,corr_rb,corr_gb,inside\n')
file = readImageWindowLocationsFile()

def processLine(line,q):
	currentImageWindows = createWindowLocations(line)
	window_inside,window_outside = openImageWindow(currentImageWindows)
	processImageWindows(q,window_inside,True)
	processImageWindows(q,window_outside,False)
	print('finished with line: ' + line)

def listener(q):
	global output_file

	while 1:
		m = q.get()
		if m == 'kill':
			break
		output_file.write(str(m))
		output_file.flush()
		

if __name__ == "__main__":
	
	manager = mp.Manager()   
	q = manager.Queue()
	pool = mp.Pool(mp.cpu_count() + 2)

    #put listener to work first
	watcher = pool.apply_async(listener, (q,))

	jobs = []

	for line in file:
		job = pool.apply_async(processLine, (line,q))
		jobs.append(job)

	for job in jobs:
		job.get()

	q.put('kill')
	pool.close()
	pool.join()

	file.close()
	output_file.close()
	print('Database exported successfully')
	t2=datetime.datetime.utcnow()
	print('Elapsed time: ' + str(t2-t1))


