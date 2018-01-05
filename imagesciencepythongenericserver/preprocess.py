import cv2
import dlib
from skimage import io
import numpy as np

import subprocess
import re
import os
import json


def json_preprocessor(image,image_url):
        getJSONcommand = 'curl -i "http://localhost:8000/?image=http://galleries.hustler.com/_dare/interracialCheerleaderOrgy2/hustler/3kristinaRose/images/15002.jpg"'
#        #getJSONcommand = 'curl -i "http://172.17.0.2:5102/?image=%s"' %(image_url) #port and ip will change .... need to fix

        process = subprocess.Popen([getJSONcommand], shell=True, stdout=subprocess.PIPE)
        cropped_images = []
        (response_stdout, response_stderr) = process.communicate()

        #print "json_from_porn", response_stdout

        jsonInput = []
        for line in response_stdout.splitlines():
                jsonInput.append(line)

	# TBD corner cases
        input_ = json.loads(jsonInput[-1])
        input_ = input_['Detector']['Frames'][0]['Detections']

        cropped_image_list = []
        for z in input_:
                if z['Object'] == 'face':
                        for locations in z['Locs']:
                                xmin = locations['X']
                                ymin = locations['Y']
                                xmax = xmin + locations['W']
                                ymax = ymin + locations['H']
                                #print "x: ", xmin, " y: ", ymin, " xmax: ", xmax, " ymax: ", ymax
                                cropped_image_list.append(image[ymin:ymax, xmin:xmax])

        return cropped_image_list



def skin_detection(image):
	#print "TBD make skin detection that returns a mask"
	converted_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
      	skinMask_hsv = cv2.inRange(converted_hsv, (0,48,80), (20,255,255))#lower,upper)

      	converted_ycbcr = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)
      	skinMask_ycbcr = cv2.inRange(converted_ycbcr, (0,133,77), (255,173,127))
                        
      	skinMask_hsv = cv2.inRange(skinMask_hsv, (0), (5))
      	skinMask_ycbcr = cv2.inRange(skinMask_ycbcr, (0), (5))  

      	return skinMask_ycbcr, skinMask_hsv

def selective_search(image_in):
	image = np.array(image_in)
	rect_list = []
	dlib.find_candidate_object_locations(image, rect_list, min_size = 5000, max_merging_iterations = 200) #5000, 200
	
	windows_list = []
	temp = np.empty([len(rect_list),4])

	for ii in range(len(rect_list)):
		temp[ii][0] = rect_list[ii].top()
		temp[ii][1] = rect_list[ii].left()
		temp[ii][2] = rect_list[ii].bottom()
		temp[ii][3] = rect_list[ii].right()

	windows_list.append(temp)

	image_list = []
	bbox = []

	for rect in windows_list[0]:
		tmp_img = image[int(rect[1]):int(rect[3]), int(rect[0]):int(rect[2])].copy()
		w, h, c = tmp_img.shape
		if w*h > 0:
			tmp_img = cv2.cvtColor(tmp_img, cv2.COLOR_BGR2RGB)
			image_list.append(tmp_img)

			bbox.append(rect)
			# cv2.imshow("test", tmp_img)
			# cv2.waitKey(0)

	roi = []
	roi.append(bbox)

	return image_list, roi

def mser_mask(input_image):

	w = input_image.shape[1]
	h = input_image.shape[0]
	area = w*h

	mask_regions = np.zeros((h, w), np.uint8)
	
	 # int delta = 13;
	 #int img_area = img.cols*img.rows;
	 #Ptr<MSER> cv_mser = MSER::create(delta,(int)(0.00002*img_area),(int)(0.11*img_area),55,0.);


	#mser = cv2.MSER_create()
	#delta, minArea, maxArea, maxVariation, minDiversity, maxEvolution, areaThresh, minMargin, edgeBlurSize
	#mser = cv2.MSER_create(8, 5, 14400, 0.25, 0.01, 100, 1.01, 0.03, 5) # good1
	#mser = cv2.MSER_create(13, 2, 14400, 55, 0)
	#mser = cv2.MSER_create(13, int(0.00002*area),int(0.11*area), 55, 0)
	mser = cv2.MSER_create(13, 10, 14400, 55, 0) # good2
	

	#1
	#mser = cv2.MSER_create(5,5, 14400, .25, .4, 100, 1.01, 0.003, 5)
	#mser = cv2.MSER_create(5, 5, 14400, 0.25, 0.2, 200, 1.01, 0.003, 5)

	gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)	
	regions = mser.detectRegions(gray, None)
	for p in regions:
        	for pts in p:
                	x, y = pts
                	mask_regions[y,x] = 255

	return mask_regions

def face_bbox(image_in):

	# test = io.imread('/home/joshua/Desktop/wine2.jpg')
	# print type(test)

	image = np.array(image_in)
	# print "type: ",type(image)

	img_h = image.shape[0]
	img_w = image.shape[1]
	min_size_percent = (img_h*img_w)/10

	# print "width ", img_w, " height", img_h
	rect_list = []

	face_detector = dlib.get_frontal_face_detector()
	rect_list = face_detector(image,1)
	
	# print "face list: ", rect_list
	
	windows_list = []
	temp = np.empty([len(rect_list),4])

	for ii in range(len(rect_list)):
		temp[ii][0] = rect_list[ii].top()
		temp[ii][1] = rect_list[ii].left()
		temp[ii][2] = rect_list[ii].bottom()
		temp[ii][3] = rect_list[ii].right()
	
	if rect_list[ii].top() <= 0:
		temp[ii][0] = 0
	if rect_list[ii].left() <= 0:
		temp[ii][1] = 0
	if rect_list[ii].bottom() >= img_h:
		temp[ii][2] = img_h
	if rect_list[ii].right() >= img_w:
		temp[ii][3] = img_w
		
            
	windows_list.append(temp)
	#print 'len(windows_list): ',len(windows_list)
	#print "window list", windows_list

	face_images = []

	for rect in windows_list[0]:
		#tmp_img = image[int(rect[1]):int(rect[3]), int(rect[0]):int(rect[2])].copy()
		# image = image[x:x+w, y:y+h]
		tmp_img = image[int(rect[0]):int(rect[2]), int(rect[1]):int(rect[3])].copy()
		tmp_img = cv2.cvtColor(tmp_img, cv2.COLOR_BGR2RGB) 
		#tmp_img = cv2.resize(tmp_img,(224,224))
		face_images.append(tmp_img)
		# cv2.imshow("test",tmp_img)
		# cv2.waitKey(1)
	return face_images, windows_list

