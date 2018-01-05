#import _init_paths
import sys
sys.path.insert(0, '/home/joshua/Env/py-faster-rcnn/caffe-fast-rcnn/python')
import caffe
import numpy as np

if (len(sys.argv) != 3):
    print "Usage: python convertmean.py proto.mean out.npy"
    sys.exit()

blob = caffe.proto.caffe_pb2.BlobProto()
data = open( sys.argv[1] , 'rb').read()
blob.ParseFromString(data)
arr = np.array( caffe.io.blobproto_to_array(blob) )
out = arr[0]
np.save( sys.argv[2] , out)
