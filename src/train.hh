#pragma once
#include "fileInfo.hh"
#include <opencv2/ml.hpp>
#include <opencv2/xfeatures2d.hpp>
#include "tbb/tbb.h"
#include "csvDumper.hh"
class Train
{
public:
  Train(std::vector<FileInfo> files, int sizeBow);
  void calculateDescriptor(std::string fileToSaveTo, bool shouldDump = false);
  void clusterBow(std::string fileToSaveTo="");
  void translateToBow(std::string vocFile="", std::string featureDir="");
  void svmTraining(std::string featureDir="");
  //private:
  std::vector<FileInfo> files;
  std::string featuresDir;
  std::vector<std::vector<cv::Mat>> everyFeaturesByVideo;
  cv::Mat everyFeatures;
  cv::BOWKMeansTrainer bowk;
  cv::Mat voc;
  std::map<std::string, cv::Mat> codedMat;
  float resCols;
  float resType;
  std::vector<std::string> classesName;
};
