/*
  This is an example of how to define a Detector class usable by the VisionServer.
  Defined herein is a "ImageDetector" which just wraps around dlib face detection and
  returns the number of faces in a given image or sequence of images.
  The ImageDetector is then run by the multi-threaded VisionServer.

*/

//#include <dlib/dnn.h>
//#include <dlib/data_io.h>
//#include <dlib/image_processing.h>
//#include <dlib/gui_widgets.h>

#include <iostream>
#include <boost/asio.hpp>
#include <boost/thread.hpp>
#include <boost/bind.hpp>
#include <boost/lexical_cast.hpp>
#include <json/json.h>
#include "./vision_server/server.hpp"
#include "caffeclassifier.h"

// #include "segmenter.h"

/* --
    Put any data that can be shared between multiple instances of the detector
    (e.g. neural net, template sift descriptors) inside the SharedDetectorData struct.
    [If there is no data that needs to be shared between multiple detectors,
    the struct must still be defined, but leave it empty.]
    The VisionServer will copy the pointer to the first instantiated detector's 
    SharedDetectorData for all subsequent detectors.
-- */

namespace image_science{
  
struct SharedDetectorData
{
  /* --
    Whatever is put in here MUST be accessible from multiple threads simultaneously.
    If the data is mutable by whatever actions are done via Detector->initialize()
    or Detector->run(), this will cause unspecified runtime errors.
    On the other hand, if this data contains a mechanism whereby only one thread can access
    it at a time, putting it here, while safe, may not be practical, since the calls to this resource
    will be synchronous.
  -- */
  
};

}

class ImageDetector : public image_science::Detector
{
  public:
    
    ImageDetector();
    ~ImageDetector();
    
    // Functions to be called upon instantiation.
    // These must be defined.
    virtual void initialize();
    virtual void initSharedData();
    
    // Functions to be called upon individual request.
    // These must be defined.
    virtual DetectorResult run(std::vector<cv::Mat>& img);
    
    
  private:
    // Put custom internal functions and member variables here:
    Json::Value make_json(bool success);
    
    // DLIB face detector
    // (This is not thread-safe, so it cannot be in SharedDetectorData.)
    //CaffeClassifier net;
    image_science::CaffeClassifier net;
    std::vector<float> places_prob;
    // dlib::frontal_face_detector face_detector_;
    
    int num_predictions_;
    std::string object_name_;
    int object_confidence_;
    int w_;
    int h_;
    float confidence_;

    std::pair<std::string, float> netoutput_;
    std::pair<std::string, float> DetectNet(cv::Mat &original_image);
    void DetectFace(cv::Mat &original_image);


};

/// VERSION: 1.0.1
