import logging
import logging.config
import _init_paths

import threading
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify
app = Flask(__name__) #init flask app
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False


import StringIO
import modelClass
import genderPredictor
import json

import numpy as np
import simplejson as json
from json import dumps, load
import requests

from PIL import Image
from io import BytesIO

import caffe
import cv2
import os, sys, traceback

import argparse
import sys
import pycurl

import boto.sqs
import boto3
from boto3.session import Session
import botocore

import getSQSmessage
import ast
import urllib, cStringIO
import preprocess
import utility
import time

#python profiler
#from pycallgraph import PyCallGraph
#from pycallgraph.output import GraphvizOutput

#example
#TYPE = CNN, RCNN, SEGMENTER
#python webapp.py --port 5000 --model crf-rnn --type SEGMENTER


def process_image(images):#, bbox):
	if _init_paths.args['type'] == 'CNN':
		predictions, net_json = model.run_model(images)#, bbox)
	if _init_paths.args['type'] == 'RCNN':
		predictions, net_json = model.run_rcnnFaster(images)
	if _init_paths.args['type'] == 'SEGMENTER':
		predictions = model.run_crf(images)
	return predictions, net_json

def main_detector(image, image_url):
	#check image
	if image == None:
		img_sz = (0,0)
		object_list_output = []
		final_result, deleteQmessage = utility.create_json(object_list_output, image_url, image, 0)
		return jsonify(final_result)

	#initalize
	image_list = []
	image_list.append(image)
	bboxes = []

	#preprocess
	if _init_paths.args['preprocess'] == 'json':
		image_list = preprocess.json_preprocessor(image,image_url)
	if _init_paths.args['preprocess'] == 'face':
		image_list, bboxes = preprocess.face_bbox(image)
	# if _init_paths.args['preprocess'] == 'search':
	# 	image_list, bboxes = preprocess.selective_search(image)

	net_predictions, net_dict_format = process_image(image_list)

	object_list_output = utility.json_formater_cnn(net_predictions, net_dict_format, bboxes, labels, image_list)
	final_result, deleteQmessage = utility.create_json(object_list_output, image_url, image, len(image_list))

	return final_result, deleteQmessage

@app.route("/batch")
def batch():
	print "batch process"
	return "batch"

@app.route("/heartbeat")
def heartbeat():		
	logging.config.fileConfig('logging.conf')
	final_result,deleteQmessage  = utility.create_json([], 'heartbeat', 'heartbeat', 0)
	return jsonify(final_result)


@app.route("/name")
def name_nltk():
	# usage ::: curl -i "http://localhost:5000/name?firstname=joshua"
	profileName = request.args.get('firstname')
	name = str(profileName)
	
	gender = gp.classify(name)
	
	gender = list(gender)

	final_result,deleteQmessage  = utility.create_json(gender, name, 'name', 0)
	return jsonify(final_result)
	

@app.route("/")
def index():
	lock = threading.Lock()
	logging.config.fileConfig('logging.conf')
	logger = logging.getLogger("webapp")

	headers = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) \
	AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11'}

	object_list_output = []
	if _init_paths.args['pullrequest']:
		with app.test_request_context():
			try:
				#pull url
				image_url = 'EMPTY'
				with lock: #avoid race for message
					body, receipt_handle = getSQSmessage.getMessage()
					message = ast.literal_eval(body) #security issue but works
					image_url=unicode(message['imageUrl'])

				if image_url.startswith("s://"):
					image_url = image_url.replace("s://", "http://")
				elif image_url.startswith("www."):
					image_url = image_url.replace("www.", "http://www.")
				elif image_url.startswith("s:"):
					image_url = image_url.replace("s:", "http://")
				elif image_url.startswith("http://"):
					image_url=unicode(message['imageUrl'])
				else:
					image_url=unicode('http://' + message['imageUrl'])


				response = requests.get(image_url, headers=headers, timeout=10)#, verify=False)
				
				# get image
				image = utility.get_image(BytesIO(response.content))

				# detector 
				final_result, deleteQmessage = main_detector(image, image_url)

				#post message and delete from queue only if URL read correctly and image read properly

				if deleteQmessage == True:
					if "destinationUrl" in body:
						destinationUrl= unicode(message['destinationUrl'])
						headers_post = {'Content-type': 'application/json',}
						postMessage = requests.post(destinationUrl, data=json.dumps(final_result), headers=headers_post)
					else: 
						pass
					getSQSmessage.deleteMessage(receipt_handle)

			except KeyboardInterrupt:
				raise

			except:
				#print(" Couldn't open the url")
				#tbd json the web url request did not work
				#traceback.print_exc(file=sys.stdout)

				image = "exceptionraised"
				object_list_output=[]
				if image_url == 'EMPTY':
					pass
				else:
					final_result, deleteQmessage = utility.create_json(object_list_output, image_url, image, 0)
					if deleteQmessage == True:
						if "destinationUrl" in body: 
							destinationUrl= unicode(message['destinationUrl'])
							headers_post = {'Content-type': 'application/json',}
							postMessage = requests.post(destinationUrl, data=json.dumps(final_result), headers=headers_post)
						else:
							pass
						getSQSmessage.deleteMessage(receipt_handle)
	else:
		try:
			image = None

			#get url and image , file or url
			image_file = request.args.get('image_file')
			image_url = request.args.get('image')

			if image_url is not None:
				if image_url.startswith("s://"):
					image_url = image_url.replace("s://", "http://")
				elif image_url.startswith("www."):
					image_url = image_url.replace("www.", "http://www.")
				elif image_url.startswith("s:"):
					image_url = image_url.replace("s:", "http://")

				response = requests.get(image_url, headers=headers, timeout=10)#, verify=False)
				image = utility.get_image(BytesIO(response.content))
				
		
			if image_file is not None:
				print image_file
				image = utility.get_image(image_file)

			# detector
			final_result, deleteQmessage = main_detector(image, image_url)
			return jsonify(final_result)

		except:
			#print(" Couldn't open the url")
			#tbd json the web url request did not work
			traceback.print_exc(file=sys.stdout)

			image = "exceptionraised"
			object_list_output=[]
			final_result, deleteQmessage = utility.create_json(object_list_output, image_url, image, 0)

			return jsonify(final_result)

if __name__ == "__main__":


	gp = genderPredictor.genderPredictor()
	accuracy=gp.trainAndTest()

	#load one model to device 
	model = modelClass.CaffeModel(_init_paths.root+ _init_paths.args['model'] \
			+  '/', _init_paths.args['deviceid'], _init_paths.args['type'])
	labels = model.labels_df

	print "object list names :: ", model.labels_df

	#app.run(debug=True)
	if _init_paths.args['pullrequest']:
		try:
			while(True):
				index()
		except KeyboardInterrupt:
			print "stop"

	#app.run(host=sys.argv[1], port=int(sys.argv[2]))

	#app.run(host="10.94.111.250",port=_init_paths.args['port'])#,threaded=True)

	#docker
	app.run(host='0.0.0.0',port=_init_paths.args['port'])#,threaded=True)



#selective search
# if _init_paths.args['preprocess'] == 'search':
# 	selective_structure = []

# 	for idx, rect in enumerate(bboxes[0]):
# 		buffer_pts = [int(i) for i in list(rect)]
# 		#buffer_pts.append(int(round(100*net_predictions['prob'][idx][1]))) # to check specific class
# 		buffer_pts.append(int(round(100*max(net_predictions['prob'][idx]))))
# 		selective_structure.append(buffer_pts)

# 	box_array = np.asarray(selective_structure)
# 	nms_dets = utility.nms_detections(box_array)
# 	print "dets::", nms_dets

# 	output_box = []
# 	nms_list = []
# 	for roi in list(nms_dets):
# 		tmp_img = image[int(roi[1]):int(roi[3]), int(roi[0]):int(roi[2])].copy()
# 		nms_list.append(tmp_img)

# 	net_predictions, net_dict_format = process_image(nms_list)
