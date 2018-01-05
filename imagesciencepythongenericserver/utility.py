import logging
import _init_paths
import cv2
import os, sys, traceback
from PIL import Image
from io import BytesIO
import numpy as np
#import webapp #### bad



def nms_detections(dets, overlap=0.5):
    """
    Non-maximum suppression: Greedily select high-scoring detections and
    skip detections that are significantly covered by a previously
    selected detection.

    This version is translated from Matlab code by Tomasz Malisiewicz,
    who sped up Pedro Felzenszwalb's code.

    Parameters
    ----------
    dets: ndarray
        each row is ['xmin', 'ymin', 'xmax', 'ymax', 'score']
    overlap: float
        minimum overlap ratio (0.5 default)

    Output
    ------
    dets: ndarray
        remaining after suppression.
    """
    if np.shape(dets)[0] < 1:
        return dets

    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]

    w = x2 - x1
    h = y2 - y1
    area = w * h

    s = dets[:, 4]
    ind = np.argsort(s)

    pick = []
    counter = 0
    while len(ind) > 0:
        last = len(ind) - 1
        i = ind[last]
        pick.append(i)
        counter += 1

        xx1 = np.maximum(x1[i], x1[ind[:last]])
        yy1 = np.maximum(y1[i], y1[ind[:last]])
        xx2 = np.minimum(x2[i], x2[ind[:last]])
        yy2 = np.minimum(y2[i], y2[ind[:last]])

        w = np.maximum(0., xx2 - xx1 + 1)
        h = np.maximum(0., yy2 - yy1 + 1)

        o = w * h / area[ind[:last]]

        to_delete = np.concatenate(
            (np.nonzero(o > overlap)[0], np.array([last])))
        ind = np.delete(ind, to_delete)

    return dets[pick, :]

def json_formater_cnn(output, net_dict, bboxes, labels_df, images): 
    if not bboxes:
        for image in images:
            img_sz = images[0].shape
            full_box = np.array([[0,0,img_sz[0],img_sz[1]]])
            bboxes.append(full_box)
    if _init_paths.args['type'] == 'RCNN':
        return net_dict
    else:
        objectOutput = []
        top_objects = [] # top n objects

        roi = list(bboxes[0])
        for x in range(0, len(output['prob'])):
                
            #print "prediction:: ", output['prob'][x]
            buffer_list = list(reversed(np.argsort(output['prob'][x])))

            number = _init_paths.args['topoutput']#len(output['prob'][0]) # can limit list top n output TBD
            topDictObjects = []
                
            for y in range(0, number):
                top_objects.append(labels_df.iloc[ buffer_list[y], 1 ])
                topDictObjects.append({"ObjectID":labels_df.iloc[ buffer_list[y], 1 ], 
                    "Confidence": str(output['prob'][0][buffer_list[y]])})


            #final_dict = {"TopNList": topDictObjects, "ROI": list(map(int, roi[x]))}
            #objectOutput.append(final_dict)

	    	objectID = labels_df.iloc[ buffer_list[y],1 ].strip("\t").split()
	    	confidence = str(100*output['prob'][0][buffer_list[y]])

            Locs=[]
            insideLocs = {"X": roi[x][0], "Y": roi[x][1], "H": roi[x][2], "W": roi[x][3], "Confidence": int(float(confidence))}
            Locs.append(insideLocs)

            objectOutput.append({"Object": objectID[1], "ObjectID": int(objectID[0]),"Locs": Locs})


    return objectOutput



def create_json(net_outputs, image_directory, image, number_of_regions):
    #image_info = {}
    #results = {}
    #output = results
    json_final ={}
    if image == None:
        #image is size 0 
        _init_paths.isDelete = False
        insideDetector = {}
        insideDetector["Ver"] = _init_paths.detectorversion
        insideDetector["ImgW"] = 0
        insideDetector["ImgH"] = 0
        insideDetector["Message"] = "Image none: '" +image_directory +"' "
        insideFrames = {'FrameID': 0}
        insideDetector["Frames"] = [insideFrames]
        json_final["Detector"] = insideDetector

    elif image == 'exceptionraised':
        #couldn't open url 
        logger = logging.getLogger("webapp.utility.create_json")
        _init_paths.isDelete = False
        insideDetector = {}
        insideDetector["Ver"] = _init_paths.detectorversion
        insideDetector["ImgW"] = 0
        insideDetector["ImgH"] = 0
        insideDetector["Message"] = "Could not read image: '" +image_directory +"' "
        insideFrames = {'FrameID': 0}
        insideDetector["Frames"] = [insideFrames]
        json_final["Detector"] = insideDetector
        logger.info("%s" %(json_final))

    elif image == 'heartbeat':
        logger = logging.getLogger("heartbeat")
        _init_paths.isDelete = False
        insideDetector = {}
        insideDetector["Ver"] = _init_paths.detectorversion
        insideDetector["Message"] = "heartbeat"
        insideFrames = {'FrameID': 0}
        insideDetector["Frames"] = [insideFrames]
        json_final["Detector"] = insideDetector

        logger.info("%s" %(json_final))

    elif image == 'name':
        #image is size 0 
        _init_paths.isDelete = False
        insideDetector = {}
        insideDetector["Ver"] = _init_paths.detectorversion
        insideDetector["Name"] = image_directory
        insideDetector["Gender"] = net_outputs[0]
        insideDetector["Prob"] = net_outputs[1]
        json_final["NameClassifier"] = insideDetector

    elif image != None:
        logger = logging.getLogger("webapp.utility.create_json")
        _init_paths.isDelete = True
        image_size = image.shape
        insideDetector={}
        insideDetector["Ver"] = _init_paths.detectorversion
        insideDetector["ImgH"] = image_size[0]
        insideDetector["ImgW"] = image_size[1]
        insideFrames = {"FrameID":0, "Detections":net_outputs}
        insideDetector["Frames"] = [insideFrames]
        json_final['Detector'] = insideDetector

        logger.info("%s" %(json_final))

    elif not net_outputs:
        #if no detections are detected 
        logger = logging.getLogger("webapp.utility.create_json")
        _init_paths.isDelete = True
        image_size = image.shape
        insideDetector = {}
        insideDetector["Ver"] = _init_paths.detectorversion
        insideDetector["ImgH"] = image_size[0]
        insideDetector["ImgW"] = image_size[1]
        insideDetector["Message"] = "No detections in: '" +image_directory +"' "
        insideFrames = {'FrameID': 0}
        insideDetector["Frames"] = [insideFrames]
        json_final["Detector"] = insideDetector

        logger.info("%s" %(json_final))

        #image_size = image.shape
        #image_info["image_url"] = image_directory
        #image_info["img_width"] = image_size[0]
        #image_info["img_height"] = image_size[1]
        #image_info["frame_duration"] = int(0) # tbd add sequence frames for gif image jxr
        #image_info["frames"] = [{"detections": net_outputs, "frame": int(0)}]
        #image_info["regions_nom"] = number_of_regions
        #image_info["status"] = 200


    return json_final,_init_paths.isDelete


def get_image(input_image):
    try:
        img = Image.open(input_image)
        #img = Image.open(BytesIO(response.content))
        image = np.asarray(img)
   
        if image.ndim is 2:
            temp = np.zeros((image.shape[0],image.shape[1],3))
            temp[:,:,0] = image[:,:]
            temp[:,:,1] = image[:,:]
            temp[:,:,2] = image[:,:]
            image = temp

        if image.shape[2] is 4:
            temp = np.zeros((image.shape[0],image.shape[1],3))
            temp[:,:,0] = image[:,:,0]
            temp[:,:,1] = image[:,:,1]
            temp[:,:,2] = image[:,:,2]
            image = temp
	#print "img url", response.content
	#print "image shape", image.shape       
        return image
    except:
        #print("Could not open{}".format(url))
        #traceback.print_exc(file=sys.stdout)
        return None

