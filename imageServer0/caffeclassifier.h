#ifndef _CAFFE_CLASSIFIER_H_
#define _CAFFE_CLASSIFIER_H_

#include <caffe/caffe.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <iostream>

namespace image_science {
/* Pair (label, confidence) representing a prediction. */
typedef std::pair<std::string, float> Prediction;

class CaffeClassifier {
 public:
 
  /// The constructor takes the paths to four files:
  CaffeClassifier(const std::string& model_file,      // caffe .prototxt file that defines model
             const std::string& trained_file,    // .caffemodel file with trained weights
             const std::string& mean_file,       // .binaryproto or standard image format (jpg, png)
             const std::string& label_file);     // txt file with list of labels (in order)
                 
  /// Takes vector of Mats and returns vector of predictions for each one
  /// For each image returns top N classifications.
  std::vector<std::vector<Prediction>> Classify(std::vector<cv::Mat>& imgs, int N = 10);
  
  /// Overloaded for single image
  std::vector<Prediction> Classify(cv::Mat& img, int N = 10);
  
  /// Takes vector of Mats and returns the raw output of the net. 
  std::vector<std::vector<float>> ForwardPass(std::vector<cv::Mat>& imgs);
  
  /// Overloaded for single image
  std::vector<float> ForwardPass(cv::Mat& imgs);
  
  /// Set preprocessor to predefined method (squash, fill, crop)
  void SetPreprocessingMethod(std::string method);
  
  /// Set preprocessor to custom provided function
  /// The function provided must be of typical opencv void(const Mat& src, Mat& det) signature. 
  void SetPreprocessingMethod(std::function<void(const cv::Mat&, cv::Mat&)>& preprocessor);
  
  /// Set mean subtraction mode (default is image subtraction)
  void SetMeanImageSubtract(){ mean_subtraction_mode_ = mean_mode::SUBTRACT_IMAGE; };
  void SetMeanPixelSubtract(){ mean_subtraction_mode_ = mean_mode::SUBTRACT_PIXEL; };
 
 
 private:
  void SetMean(const std::string& mean_file);
  void WrapInputLayer(std::vector<cv::Mat>* input_channels);
  
  void CorrectMatType(cv::Mat& img);
  void SubtractMean(cv::Mat& img);

  void PreprocessSquash(const cv::Mat& in_img, cv::Mat& out_img);
  void PreprocessFill(const cv::Mat& in_img, cv::Mat& out_img);
  void PreprocessCrop(const cv::Mat& in_img, cv::Mat& out_img);
  

  enum mean_mode { SUBTRACT_PIXEL, SUBTRACT_IMAGE };
  
  mean_mode mean_subtraction_mode_;
  
  /// Pointer to actual caffe::Net
  std::shared_ptr<caffe::Net<float> > net_;
  
  /// Preprocessor function (usually just wraps one of the above Preprocess funcs)
  std::function<void(const cv::Mat&, cv::Mat&)> preprocessor_;
  
  /// Maximum batch size (consider having a set/get for this and making it non-static)
  static const int kMaxBatchSize_ = 16;
  
  /// This stuff will be etracted from loaded caffemodel
  cv::Size input_size_;
  int num_channels_;
  
  /// loaded from binaryproto file:
  cv::Mat mean_;
  cv::Scalar mean_pixel_;
  
  /// loaded from txt file
  std::vector<std::string> labels_;

};

} // namespace image_science


#endif
