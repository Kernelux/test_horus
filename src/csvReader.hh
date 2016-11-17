#pragma once
#include <fstream>
#include <opencv2/opencv.hpp>

class CsvReader
{
public:
  CsvReader() = default;
  cv::Mat read(std::string fileName);
  void read(cv::Mat res, std::string fileName);
  cv::Mat operator()(std::string fileName);
  void operator()(cv::Mat res, std::string fileName);
};
