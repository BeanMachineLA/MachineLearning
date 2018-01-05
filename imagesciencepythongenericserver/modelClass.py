import _init_paths
import caffe 
import numpy as np
import sys
import operator
import pandas as pd
import warnings
warnings.simplefilter(action = "ignore", category = FutureWarning)

from PIL import Image
from io import BytesIO
import cv2

from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms


from utils.timer import Timer
import matplotlib.pyplot as plt
import scipy.io as sio
import os
import argparse
from PIL import Image as PILImage


class CaffeModel:
	def __init__(self, root, deviceID, netObject):
		self.root = root
		self.deviceID = deviceID
		

		with open(root + 'labels.txt') as f:
			labels_df = pd.DataFrame([
				{
				'synset_id': l.strip().split(' ')[0],
				'name': ' '.join(l.strip().split(' ')[1:]).split(',')[0]
				}
				for l in f.readlines()
				])
			labels_df.sort('synset_id')
		
		MODEL_FILE = root + 'deploy.prototxt'
		PRETRAINED = root + 'model.caffemodel'
		

		caffe.set_mode_gpu()
		caffe.set_device(self.deviceID)
		

		if netObject == 'CNN' or netObject == 'RCNN':
			net = caffe.Classifier(MODEL_FILE, PRETRAINED,
				mean=np.load(root + 'mean.npy').mean(1).mean(1),
				channel_swap=(2,1,0),
				raw_scale=1,
				image_dims=(224, 224))  #TBD Read from Prototxt
		
		if netObject == 'NET':
			net = caffe.Net(MODEL_FILE, PRETRAINED, caffe.TEST)
		
		if netObject == 'SEGMENTER':
			net = caffe.Segmenter(MODEL_FILE, PRETRAINED)#, True)
		
		self.net = net
		self.labels_df = labels_df


	def run_model(self, images): #, bboxes):
		caffe.set_mode_gpu()
		caffe.set_device(self.deviceID)
		# print "image list len: ",len(images)
		# cv2.imwrite("out.png", images[0])
	
		data_blob_shape = self.net.blobs['data'].data.shape
		data_blob_shape = list(data_blob_shape)
		self.net.blobs['data'].reshape(len(images), data_blob_shape[1], data_blob_shape[2], data_blob_shape[3])
		self.net.blobs['data'].data[...] = map(lambda x: self.net.transformer.preprocess('data',x), images)

		# # # test 1
		# print "size: ", images[0].shape , " type: ", type(images[0])
		# transformed_image = self.net.transformer.preprocess('data', images[0])
		# self.net.blobs['data'].data[...] = transformed_image

		# #test 2
		# print "size: ", images[0].shape , " type: ", type(images[0])
		# image = cv2.resize(images[0], (224,224))
		# print "size: ", image.shape , " type: ", type(image)
		# test_im = np.float32(np.rollaxis(image, 2)[::-1]) - self.net.transformer.mean['data']
		# #net.blobs['data'].reshape(*test_im.shape)
		# self.net.blobs['data'].data[...] = test_im 

		predictions = self.net.forward(end='prob')

		objectOutput = []
		# top_objects = [] # top n objects

		# roi = list(bboxes[0])
		# for x in range(0,len(output['prob'])):
		# 	print "prediction:: ", output['prob'][x]
		# 	buffer_list = list(reversed(np.argsort(output['prob'][x])))

		# 	number = len(output['prob'][0]) # can limit list top n output
		# 	topDictObjects = []
			
		# 	for y in range(0, number):
		# 		top_objects.append(self.labels_df.iloc[ buffer_list[y], 1 ])
		# 		topDictObjects.append({"ObjectID":self.labels_df.iloc[ buffer_list[y], 1 ], 
		# 			"Confidence": int(output['prob'][0][buffer_list[y]]*100)})


		# 	final_dict = {"TopNList": topDictObjects, "ROI": list(map(int, roi[x]))}
		# 	objectOutput.append(final_dict)
		
		return predictions, objectOutput


	def run_crf(self, images):
		caffe.set_mode_cpu()
		#caffe.set_device(self.deviceID)
		image = images[0]
		#image = 255 * caffe.io.load_image('/home/joshua/maya/python-scripts/input.jpg')
		size_in = 500
		width = image.shape[0]
		height = image.shape[1]
		maxDim = max(width,height)

		if maxDim > size_in:
			aspectRatio = float(float(width)/float(height))
			#print "max : ", maxDim, " w : ", width, " h : ", height, " aspectRatio : ", aspectRatio
			if width == maxDim:
				width  = size_in
				height = int(size_in/aspectRatio)
			else:
				height = size_in
				width  = int(size_in*aspectRatio)

		image = cv2.resize(image, (height,width))

		#image = PILImage.fromarray(np.uint8(input_image))
		#image = np.array(image)

		mean_vec = np.array([103.939, 116.779, 123.68], dtype=np.float32)
		reshaped_mean_vec = mean_vec.reshape(1, 1, 3);

		im = image[:,:,::-1]
		# Subtract mean
		im = im - reshaped_mean_vec

		# Pad as necessary
		cur_h, cur_w, cur_c = im.shape
		pad_h = size_in - cur_h
		pad_w = size_in - cur_w
		im = np.pad(im, pad_width=((0, pad_h), (0, pad_w), (0, 0)), mode = 'constant', constant_values = 0)
		# print im.shape
		# print self.net
		# Get predictions
		segmentation = self.net.predict([im])
		
		threshold = 15# wine 5, table 11, car 7, person 15
		idx = segmentation[:,:] != threshold
		segmentation[idx] = 0

		segmentation2 = segmentation[0:cur_h, 0:cur_w]
		idx = segmentation2[:,:] != 0
		segmentation2[idx] = 255

		output_im = PILImage.fromarray(segmentation2)
		output_im.save("outputMask.png")

		objectOutput = []
		# top_objects = [] # top n objects
		
		# buffer_list = list(reversed(np.argsort(output['prob'][0])))
		# number = len(output['prob'][0]) # can limit list top n output
		
		# for x in range(0, number):
		# 	top_objects.append(self.labels_df.iloc[ buffer_list[x], 1 ])
		# 	objectOutput.append({"ObjectID":self.labels_df.iloc[ buffer_list[x], 1 ], "Confidence": str(output['prob'][0][buffer_list[x]])})
	
		return objectOutput

	def run_rcnnFaster(self, images):
		caffe.set_mode_gpu()
		caffe.set_device(self.deviceID)
		
		image = images[0]
		cfg.TEST.HAS_RPN = True
		
		scores, boxes = im_detect(self.net, image)

		# THRESHOLDS
		CONF_THRESH = 0.7
		NMS_THRESH = 0.3

		classes = list(self.labels_df.iloc[ :, 1 ])
		classes.remove('background')

		objectOutput = []
		
		#image_out = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

		for cls_ind, cls in enumerate(classes):
			cls_ind += 1 # because we skipped background
			cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
			cls_scores = scores[:, cls_ind]
			dets = np.hstack((cls_boxes,cls_scores[:, np.newaxis])).astype(np.float32)
			keep = nms(dets, NMS_THRESH)
			dets = dets[keep, :]
			inds = np.where(dets[:, -1] >= CONF_THRESH)[0]

			locations = []

			for i in inds:
				bbox = dets[i, :4]
				score = dets[i, -1]

				# follow mantii convention ??
				w = abs(int(bbox[0]) - int(bbox[2]))
				h = abs(int(bbox[1]) - int(bbox[3]))

				location_dict = {'Y': int(bbox[1]), 'X': int(bbox[0]), 'W': w, \
							'H': h, 'Confidence': int(score*100)}

				locations.append(location_dict)
				
				#print "class : ", cls, " roi : ", locations, " confidence: ", score
				#print "class : ", cls, " roi : ", list(map(int, bbox)), " confidence: ", score
				#print "image size: ", image.shape 
				#print "x: ", int(bbox[0])," y: ", int(bbox[1]), " w: ", w, " h: ", h
				#print "xmin: ", int(bbox[0])," ymin: ", int(bbox[1]), " xmax: ", int(bbox[2]), " ymax: ", int(bbox[3])
				#cv2.rectangle(image_out, (int(bbox[0]), int(bbox[1])), (int(bbox[0])+w,int(bbox[1])+h), (40,cls_ind*4,cls_ind*10), 3)	
                #cv2.rectangle(image_out, (int(bbox[1]), int(bbox[0])), (int(bbox[1]),int(bbox[0])), (40,cls_ind*4,cls_ind*10), 10)
                #cv2.imwrite('output/debug.png', image_out)
			
			
			if len(locations) == 0:
				continue	
			objectOutput.append({"ObjectID":cls_ind,
				"Object":cls,
				"Locs": locations})
		return scores, objectOutput 

