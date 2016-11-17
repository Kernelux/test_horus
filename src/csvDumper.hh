#pragma once
#include <fstream>
#include <opencv2/opencv.hpp>

class CsvDumper
{
public:
  CsvDumper() = default;
  void dump(const cv::Mat& m, std::string fileName);
  void dump(const std::vector<cv::Mat>& m, std::string fileName);
  void operator()(const cv::Mat& m, std::string fileName);
  void operator()(const std::vector<cv::Mat>& m, std::string fileName);
};
