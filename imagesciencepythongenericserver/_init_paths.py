# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Set up paths for Fast R-CNN."""

import os.path as osp
import sys
import argparse

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.dirname(__file__)

# Add caffe to PYTHONPATH
#caffe_path = osp.join(this_dir, '/home/joshua/maya', 'caffe-fast-rcnn', 'python')
caffe_path = osp.join(this_dir, '/home/ubuntu/machine-learning/', 'caffe', 'python')
#caffe_path = osp.join(this_dir, '/home/talnts/Env/py-faster-rcnn', 'caffe-fast-rcnn', 'python')
add_path(caffe_path)

# selective_path = osp.join(this_dir, '/home/joshua/Repos/pythongenericserver/detector')
# print selective_path
# add_path(selective_path)

# Add lib to PYTHONPATH
#lib_path = osp.join(this_dir, '/home/joshua/maya', 'lib')
lib_path = osp.join(this_dir, '/home/ubuntu/machine-learning/py-faster-rcnn/', 'lib')
add_path(lib_path)

# where models live 
root = '/home/ubuntu/models/'

detectorversion = 1.000

deleteQmessage = False
#image_url = TBD 
#destination_url = TBD

parser = argparse.ArgumentParser(description='Run the DIGITS development server')
parser.add_argument('-t', '--type', type=str, default="CNN", help='point to model type')
parser.add_argument('-pull', '--pullrequest', help="enable pullrequest mode", action="store_true", default=False)
parser.add_argument('-p', '--port', type=int, default=5000, help='Port to run app on (default 5000)')
parser.add_argument('-m', '--model', type=str, default="liveActionModel", help='point to model')
parser.add_argument('-pre', '--preprocess', type=str, default="None", help='point to preprocessing')
parser.add_argument('-dev', '--deviceid', type=int, default=0, help='point to device, number of devices')
parser.add_argument('-top', '--topoutput', type=int, default=1, help='point to number, number of top outputs')

args = vars(parser.parse_args())
