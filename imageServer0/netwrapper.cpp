//#include <dlib/dnn.h>
//#include <opencv2/highgui/highgui.hpp>
//#include <dlib/opencv.h>
//#include <dlib/data_io.h>
//#include <dlib/image_processing.h>
//#include <dlib/gui_widgets.h>
#include "netwrapper.h"
#include <opencv2/imgproc.hpp>
#include <curl/curl.h>
#include "caffeclassifier.h"
//#include "segmenter.h"
#define DEBUG true
#define DLIB_JPEG_SUPPORT
#define DLIB_PNG_SUPPORT
//using namespace dlib;
using namespace std;

/* --
  The name of your specific detector must be provided to the Detector constructor.
  That way, it can automatically wrap your return json with the correct detector name. 
-- */
ImageDetector::ImageDetector() : image_science::Detector("Net Detector"),
    net("/home/ubuntu/models/nsfw/deploy.prototxt",
        "/home/ubuntu/models/nsfw/model.caffemodel",
        "/home/ubuntu/models/nsfw/mean.binaryproto",
        "/home/ubuntu/models/nsfw/labels.txt")
{}

ImageDetector::~ImageDetector()
{}

void ImageDetector::initSharedData()
{
    // This line must be included:
    shared_data_ = std::shared_ptr<image_science::SharedDetectorData>(new image_science::SharedDetectorData);
    
    // Initialize all members of SharedDetectorData (as defined in "sample_detector.h").
    // They can now be referred to by "shared_data_->myVariable"   

}

void ImageDetector::initialize()
{
  /* --
    Initialize all necessary variables here. This will be called only once
    by the VisionServer, upon instantiation of the detector.
  -- */   
 
}


std::pair<std::string, float> ImageDetector::DetectNet(cv::Mat &original_image)
{
  // std::cout << "image size " << original_image.size() << std::endl;
  std::string colorout = "test";
  std::vector<image_science::Prediction> classID = net.Classify(original_image, 1);
  // std::cout << classID.size() << std::endl;
  //for (int i = 0; i < classID.size(); i++)
  //{
  //  std::pair<std::string, float> output = classID[i];
  //  std::cout << "output: " << output.first << " confidence: " << int(output.second*100) << std::endl;
  //}

  std::pair<std::string, float> output = classID[0]; 
  return output;
}

//template <long num_filters, typename SUBNET> using con5d = dlib::con<num_filters,5,5,2,2,SUBNET>;
//template <long num_filters, typename SUBNET> using con5  = dlib::con<num_filters,5,5,1,1,SUBNET>;

//template <typename SUBNET> using downsampler  = relu<affine<con5d<32, relu<affine<con5d<32, relu<affine<con5d<16,SUBNET>>>>>>>>>;
//template <typename SUBNET> using rcon5  = relu<affine<con5<45,SUBNET>>>;

//using net_type = loss_mmod<con<1,9,9,1,1,rcon5<rcon5<rcon5<downsampler<input_rgb_image_pyramid<pyramid_down<6>>>>>>>>;

void ImageDetector::DetectFace(cv::Mat &original_image)
{
  
  //net_type net;
  //deserialize("/home/talnts/Downloads/mmod_human_face_detector.dat") >> net; 
  //matrix<bgr_pixel> img;
  
  //load_image(img, "/home/talnts/Desktop/nsfw_test/1.jpg");
  //matrix<bgr_pixel> img(original_image);
  
  //auto dets = net(cimg);

}




ImageDetector::DetectorResult ImageDetector::run(std::vector<cv::Mat>& imgs)
{   
    cv::Mat img = imgs.at(0);

    // Handle RGBA and grayscale
    if (img.channels() == 4)
        cv::cvtColor(img, img, CV_BGRA2BGR);
    else if (img.channels() == 1)
        cv::cvtColor(img, img, CV_GRAY2BGR);
    if (img.channels() != 3)
    {
        Json::Value json = make_json(false);
        return makeDetectorResult(json);
    }
        
        
        
    // All "options" (or "flags"), sent in the request, will be in std::string format in
    // the inherited options_ vector.
    // We can set local flags like this:
    // bool detect_facial_landmarks_ = false;
    // if (std::find(options_.begin(), options_.end(), "landmarks") != options_.end())
    //     detect_facial_landmarks_ = true;
    
    try
    {
        // places_prob = net.ForwardPass(img);
        //vector<cv::Mat> imgs;
        //imgs.push_back(img);
        //std::vector<std::vector<Prediction>> predictions(imgs.size());
        //predictions = net.Classify(imgs, 1);
        //DetectFace(img); //face first
        netoutput_ = DetectNet(img);

        //cout << "size out:: " << predictions.first() << endl;
        //cout << "name: " << predictions[0] << endl;
        // Convert image to dlib format:
        //dlib::cv_image< dlib::bgr_pixel> cv_img =  dlib::cv_image< dlib::bgr_pixel>(img);
        //dlib::array2d< dlib::bgr_pixel> dlib_img;
        //dlib::assign_image(dlib_img, cv_img);
        
        // Run face detector:
        //auto faces = face_detector_(dlib_img);
        object_name_ = netoutput_.first;
        object_confidence_ = 100;
        w_ = img.rows;
        h_ = img.cols;
        //{"Detector": {"Frames": [{"Detections": [{"BrandID": 1835, "Locs": [{"Confidence": 99, "H": 243, "W": 178, "X": 291, "Y": 42}],
        //"Object": "face", "ObjectID": 1}, {"BrandID": 1838, "Locs": [{"Confidence": 99, "H": 181, "W": 159, "X": 285, "Y": 441}], 
        //"Object": "breast", "ObjectID": 4}], "FrameID": 0}], "ImgH": 1276, "ImgW": 807, "Ver": 1.0}}
        //cout << "object: " << object_name_ << " confidence: " << object_confidence_ << endl;
    }
    catch (std::exception& e)
    {
        // How to return an error message:
        // Construct a DetectorResult with "false" as the first argument, and the error message
        // string as the second argument.
        // The string passed to the DetectorResult will be exactly what is returned to the requestor
        // or posted to the destination (in vertex mode). So make sure it is in JSON format.
        std::string error_msg = e.what();
        return makeDetectorError(error_msg);
    }
    
    // Constructing a DetectorResult with a string alone (or with "true" as the first argument)
    // will just return the string and assume the detection was done successfully.
    return makeDetectorResult( make_json(true) ); 
}

// Custom non-inherited functions:
Json::Value ImageDetector::make_json(bool success)
{
  std::string json;
  
  if (success)
    {
      // std::string delim = "-";
      // auto start = 0U;
      // auto end = netoutput_.find(delim);
      //vector<string> parsecolor_;
      //while (end != std::string::npos)
      //{
      //std::cout << netoutput_.substr(start, end - start) << std::endl;
      //    parsecolor_.push_back(netoutput_.substr(start,end-start));
      //    start = end + delim.length();
      //    end = netoutput_.find(delim, start);
      //    break;
      //}
      // std::string shade_ = netoutput_.substr(end+1,netoutput_.length());
      // std::string color_ = netoutput_.substr(0,end);

      Json::Value root(Json::objectValue);
      Json::Value jframes(Json::arrayValue);
      Json::Value one_frame(Json::objectValue);
      Json::Value net_estimation(Json::objectValue);
      Json::Value j_persons(Json::arrayValue);
    
      //Place_detect["Object"] = object_name_;
      //Place_detect["X"] = 0;
      //Place_detect["W"] = w_;
      //Place_detect["H"] = h_;
      //Place_detect["Y"] = 0;
      //Place_detect["Confidence"] = object_confidence_;
      Json::Value one_person(Json::objectValue);
      one_person["ClassID"] = netoutput_.first;
      // one_person["hairShade"] = shade_;
      one_person["Confidence"] = floor(netoutput_.second*100);
      j_persons.append(one_person);
      one_frame["FrameID"] = 0;
      one_frame["Persons"] = j_persons;
      // one_frame["PercentSkin"] = m_percent_skin;
      // one_frame["SkinDensity"] = m_skin_density;
      // one_frame["SkinAreaOfImage"] = m_skin_area_image;
      // one_frame["SkinAreaAngle"] = m_skin_area_angle;            
      // one_frame["Safety"] = m_safety; 
      // one_frame["SafetyConfidence"] = m_confidence;

      jframes.append(one_frame);
      net_estimation["Frames"] = jframes;
      // m_values["Key"] = m_key;
      return net_estimation;
    }
  else
    {
      Json::Value m_answer(Json::objectValue);
      Json::Value net_estimation(Json::objectValue);

      net_estimation["Messsage"] = "Error processing or could not open image file.";
      return net_estimation;
    }

  return json;
}
