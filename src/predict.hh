#pragma once
#include <opencv2/ml.hpp>
#include <map>
#include "fileInfo.hh"

class Predict
{
public:
  Predict(std::vector<FileInfo> files);
  void predict(std::vector<FileInfo> files);
  void predictUnique(std::vector<FileInfo> files);
  void predictUnique(std::string fileName);
  std::vector<FileInfo> files;
  std::map<std::string, cv::Ptr<cv::ml::SVM>> svms;

};
