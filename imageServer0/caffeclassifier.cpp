#include "caffeclassifier.h"

using std::cout;
using std::endl;
using std::vector;
using std::string;
using cv::Size;

namespace image_science{
  
using caffe::Net;
using caffe::Blob;

CaffeClassifier::CaffeClassifier(const string& model_file,
                       const string& trained_file,
                       const string& mean_file,
                       const string& label_file) {
#ifdef USE_GPU_CAFFE
  caffe::Caffe::set_mode(caffe::Caffe::GPU)
#else
  caffe::Caffe::set_mode(caffe::Caffe::CPU);
#endif

  /* Load the network. */
  net_.reset(new Net<float>(model_file, caffe::TEST));
  net_->CopyTrainedLayersFrom(trained_file);

  CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";

  Blob<float>* input_layer = net_->input_blobs()[0];
  num_channels_ = input_layer->channels();
  CHECK(num_channels_ == 3 || num_channels_ == 1)
    << "Input layer should have 1 or 3 channels.";
  input_size_ = cv::Size(input_layer->width(), input_layer->height());

  /* Load the binaryproto mean file. */
  SetMean(mean_file);

  /* Load labels. */
  std::ifstream labels(label_file.c_str());
  CHECK(labels) << "Unable to open labels file " << label_file;
  string line;
  while (std::getline(labels, line))
  {
    int space = line.find_first_of(' ');
    if (space == string::npos)
      space = -1;
    line = line.substr(space+1);
    labels_.push_back(string(line));

  }

  int num_outputs = net_->num_outputs();
  Blob<float>* output_layer = net_->output_blobs()[num_outputs-1];
  CHECK_EQ(labels_.size(), output_layer->channels())
    << "Number of labels is different from the output layer dimension.";
    
  preprocessor_ = std::bind(&CaffeClassifier::PreprocessSquash, this, std::placeholders::_1, std::placeholders::_2);
}

static bool PairCompare(const std::pair<float, int>& lhs,
                        const std::pair<float, int>& rhs) {
  return lhs.first > rhs.first;
}

/* Return the indices of the top N values of vector v. */
static std::vector<int> Argmax(const std::vector<float>& v, int N) {
  std::vector<std::pair<float, int> > pairs;
  for (size_t i = 0; i < v.size(); ++i)
    pairs.push_back(std::make_pair(v[i], i));
  std::partial_sort(pairs.begin(), pairs.begin() + N, pairs.end(), PairCompare);

  std::vector<int> result;
  for (int i = 0; i < N; ++i)
    result.push_back(pairs[i].second);
  return result;
}


/* Return the top N predictions. */
std::vector<Prediction> CaffeClassifier::Classify(cv::Mat& img, int N) {
  
  vector<cv::Mat> img_vec = {img};
  vector<vector<Prediction>> predictions = Classify(img_vec, N);
  
  return predictions[0];
}


/* Return the top N predictions. */
std::vector<std::vector<Prediction>> CaffeClassifier::Classify(vector<cv::Mat>& imgs, int N) {
  std::vector<std::vector<float>> net_output = ForwardPass(imgs);

  N = std::min<int>(labels_.size(), N);
  std::vector<std::vector<Prediction>> predictions(imgs.size());
  
  for (int k = 0; k < imgs.size(); ++k)
  {
    std::vector<int> maxN = Argmax(net_output[k], N);
    
    for (int i = 0; i < N; ++i)
    {
      int idx = maxN[i];
      predictions[k].push_back(std::make_pair(labels_[idx], net_output[k][idx]));
    }
  }

  return predictions;
}

/* Load the mean file in binaryproto format. */
void CaffeClassifier::SetMean(const string& mean_file) {
  caffe::BlobProto blob_proto;
  caffe::ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);

  /* Convert from BlobProto to Blob<float> */
  Blob<float> mean_blob;
  mean_blob.FromProto(blob_proto);
  CHECK_EQ(mean_blob.channels(), num_channels_)
    << "Number of channels of mean file doesn't match input layer.";

  /* The format of the mean file is planar 32-bit float BGR or grayscale. */
  std::vector<cv::Mat> channels;
  float* data = mean_blob.mutable_cpu_data();
  for (int i = 0; i < num_channels_; ++i)
  {
    /* Extract an individual channel. */
    cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
    channels.push_back(channel);
    data += mean_blob.height() * mean_blob.width();
  }

  /* Merge the separate channels into a single image. */
  cv::Mat mean;
  cv::merge(channels, mean);

  cv::resize(mean, mean_, Size(input_size_.width, input_size_.height));
  
  mean_pixel_ = cv::mean(mean_);

}

std::vector<std::vector<float>> CaffeClassifier::ForwardPass(std::vector<cv::Mat>& imgs)
{
  std::vector<std::vector<float>> return_val;
  
  // Determine number of forward passes
  int num_full_passes = imgs.size() / kMaxBatchSize_;
  int residual_pass_size = imgs.size() % kMaxBatchSize_;
  
  Blob<float>* input_layer = net_->input_blobs()[0];
  
  // for (int pass = 0; pass < num_full_passes; ++pass)
  // {
  auto forward_pass = [&](vector<cv::Mat>::iterator start_batch, vector<cv::Mat>::iterator end_batch){
    std::vector<std::vector<float>> output; 
    std::vector<cv::Mat> input_channels;
    WrapInputLayer(&input_channels);
    
    int total_channel_count = 0;
    for (auto it = start_batch; it != end_batch; ++it)
    {
      cv::Mat processed;
      preprocessor_(*it, processed);
      if (it->channels() == 1)
      {
        processed.copyTo(input_channels[total_channel_count]);
        ++total_channel_count;
      }
      else
      {
        std::vector<cv::Mat> temp_channels;
        cv::split(processed, temp_channels);
        for (cv::Mat& ch : temp_channels)
        {
          ch.copyTo(input_channels[total_channel_count]);
          ++total_channel_count;
        }
      }
    }
    
    // Quick check to make sure that totally worked
    CHECK(reinterpret_cast<float*>(input_channels.at(0).data)
        == net_->input_blobs()[0]->cpu_data())
        << "Input channels are not wrapping the input layer of the network." << reinterpret_cast<float*>(input_channels.at(0).data) << " vs " << net_->input_blobs()[0]->cpu_data();

    net_->ForwardPrefilled();

    /* Copy the output layer to the return val */
    Blob<float>* output_layer = net_->output_blobs()[net_->num_outputs()-1];
    const float* begin = output_layer->cpu_data();
    const float* end = begin + output_layer->shape(1);
    for (int k = 0; k < output_layer->shape(0); ++k)
    {
      output.push_back( std::vector<float>(begin, end) );
      begin = end;
      end = begin + output_layer->shape(1);
    }
    
    return output;
  };
  
  if (num_full_passes > 0)
  {
    input_layer->Reshape(kMaxBatchSize_, num_channels_, input_size_.height, input_size_.width);
    net_->Reshape();
  }
  for (int pass = 0; pass < num_full_passes; ++pass)
  {
    vector<cv::Mat>::iterator start = imgs.begin() + (pass*kMaxBatchSize_);
    vector<cv::Mat>::iterator end = start + kMaxBatchSize_;
    auto inference_output = forward_pass(start, end);
    return_val.insert(return_val.end(), inference_output.begin(), inference_output.end());
  }
  if (residual_pass_size > 0)
  {
    input_layer->Reshape(residual_pass_size, num_channels_, input_size_.height, input_size_.width);
    net_->Reshape();
    vector<cv::Mat>::iterator start = imgs.begin() + num_full_passes*kMaxBatchSize_;
    vector<cv::Mat>::iterator end = start + residual_pass_size;
    auto inference_output = forward_pass(start, end);
    return_val.insert(return_val.end(), inference_output.begin(), inference_output.end());
  }
  
  // Why? Bekuz C++11. That's why.
  return return_val;
}


std::vector<float> CaffeClassifier::ForwardPass(cv::Mat& img) {
  Blob<float>* input_layer = net_->input_blobs()[0];
  input_layer->Reshape(1, num_channels_,
                       input_size_.height, input_size_.width);
  /* Forward dimension change to all layers. */
  net_->Reshape();

  std::vector<cv::Mat> input_channels;
  WrapInputLayer(&input_channels);

  cv::Mat processed;
  preprocessor_(img, processed);
  if (num_channels_ > 0)
    cv::split(processed, input_channels);
  else
    processed.copyTo(input_channels[0]);

  net_->ForwardPrefilled();

  /* Copy the output layer to a std::vector */
  Blob<float>* output_layer = net_->output_blobs()[net_->output_blobs().size()-1];
  const float* begin = output_layer->cpu_data();
  const float* end = begin + output_layer->channels();
  return std::vector<float>(begin, end);
}

/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
void CaffeClassifier::WrapInputLayer(std::vector<cv::Mat>* input_channels)
{
  Blob<float>* input_layer = net_->input_blobs()[0];

  int width = input_layer->shape(3);
  int height = input_layer->shape(2);
  int channels = input_layer->shape(1);
  int batch_size = input_layer->shape(0);
  
  float* input_data = input_layer->mutable_cpu_data();
  for (int k = 0; k < batch_size; ++k)
  {
    for (int c = 0; c < channels; ++c)
    {
      cv::Mat channel(height, width, CV_32FC1, input_data);
      input_channels->push_back(channel);
      input_data += width * height;
    }
  }
}

void CaffeClassifier::SetPreprocessingMethod(std::string method)
{
  std::transform(method.begin(), method.end(), method.begin(), ::tolower);
  
  if (method == "squash")
    preprocessor_ = std::bind(&CaffeClassifier::PreprocessSquash, this, std::placeholders::_1, std::placeholders::_2);
  else if (method == "fill")
    preprocessor_ = std::bind(&CaffeClassifier::PreprocessFill, this, std::placeholders::_1, std::placeholders::_2);
  else if (method == "crop")
    preprocessor_ = std::bind(&CaffeClassifier::PreprocessCrop, this, std::placeholders::_1, std::placeholders::_2);
  else
  {
    std::cerr << "Invalid Preprocessing method specified in CaffeClassifier::SetPreprocessingMethod()...\n";
    std::cerr << "Using default: \"squash\"\n";
  }
}

void CaffeClassifier::SetPreprocessingMethod(std::function<void(const cv::Mat&, cv::Mat&)>& preprocessor)
{
  // Test function to make sure it provides images of the right size for input to the net:
  cv::Mat test_mat = cv::Mat::zeros(10,10, CV_8UC1);
  cv::Mat processed;
  preprocessor(test_mat, processed);
  if (! (processed.size().width == input_size_.width && processed.size().height == input_size_.height
          && processed.channels() == num_channels_) )
  {
    std::cerr << "Preprocessing function provided in CaffeClassifier::SetPreprocessingMethod() is not valid...\n";
    std::cerr << "Using default: \"squash\"\n";
    return;
  }
  
  preprocessor_ = preprocessor;  
}



void CaffeClassifier::CorrectMatType(cv::Mat& img)
{
  if (img.channels() == 3 && num_channels_ == 1)
    cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
  else if (img.channels() == 4 && num_channels_ == 1)
    cv::cvtColor(img, img, cv::COLOR_BGRA2GRAY);
  else if (img.channels() == 4 && num_channels_ == 3)
    cv::cvtColor(img, img, cv::COLOR_BGRA2BGR);
  else if (img.channels() == 1 && num_channels_ == 3)
    cv::cvtColor(img, img, cv::COLOR_GRAY2BGR);
    
  if (num_channels_ == 3)
    img.convertTo(img, CV_32FC3, 1.0f/255.0f);
  else
    img.convertTo(img, CV_32FC1, 1.0f/255.0f);
}


void CaffeClassifier::SubtractMean(cv::Mat&img)
{
  img *= 255.0f;
  if (mean_subtraction_mode_ == mean_mode::SUBTRACT_PIXEL)
    img -= mean_pixel_;
  else if (mean_subtraction_mode_ == mean_mode::SUBTRACT_IMAGE)
    img -= mean_;
  else
    img -= mean_;  
}


void CaffeClassifier::PreprocessSquash(const cv::Mat& in_img, cv::Mat& out_img)
{
  
  cv::resize(in_img, out_img, Size(input_size_.width, input_size_.height));
  CorrectMatType(out_img);
  SubtractMean(out_img);
  
}

void CaffeClassifier::PreprocessCrop(const cv::Mat& in_img, cv::Mat& out_img)
{
  
  cv::Size target_size;
  if (in_img.size().width > in_img.size().height)
  {
    float ratio = input_size_.height / float(in_img.size().height);
    target_size = cv::Size(int(ratio*in_img.size().width), input_size_.height);
    cv::resize(in_img, out_img, target_size);
    
    int center = out_img.size().width / 2;
    cv::Rect crop(center-input_size_.width/2, 0, input_size_.width, out_img.size().height);
    out_img = out_img(crop).clone();
  }
  else
  {
    float ratio = input_size_.width / float(in_img.size().width);
    target_size = cv::Size(input_size_.width, int(ratio*in_img.size().height));
    cv::resize(in_img, out_img, target_size);
    
    int center = out_img.size().height / 2;
    cv::Rect crop(0, center-input_size_.height/2, out_img.size().width, input_size_.height);
    out_img = out_img(crop).clone();
  }
  
  CorrectMatType(out_img);
  SubtractMean(out_img);

}

void CaffeClassifier::PreprocessFill(const cv::Mat& in_img, cv::Mat& out_img)
{
  cv::Size target_size;
  
  cv::Mat output_img(input_size_, in_img.type()); 
  cv::randu(output_img, cv::Scalar::all(0), cv::Scalar::all(255));
  
  if (in_img.size().width > in_img.size().height)
  {
    float ratio = input_size_.width / float(in_img.size().width);
    target_size = cv::Size(input_size_.width ,int(ratio*in_img.size().height));
    cv::resize(in_img, out_img, target_size);
    
    int center = input_size_.height / 2;
    cv::Rect insert_box(0, center - out_img.size().height/2, input_size_.width, out_img.size().height);
    out_img.copyTo(output_img(insert_box));
  }
  else
  {
    float ratio = input_size_.height / float(in_img.size().height);
    target_size = cv::Size(int(ratio*in_img.size().width), input_size_.height);
    cv::resize(in_img, out_img, target_size);
    
    int center = input_size_.width / 2;
    cv::Rect insert_box(center - out_img.size().width/2, 0, out_img.size().width, input_size_.height);
    out_img.copyTo(output_img(insert_box));
  }
  
  out_img = output_img.clone();
  
  CorrectMatType(out_img);
  SubtractMean(out_img);
}


} // namespace image_science
