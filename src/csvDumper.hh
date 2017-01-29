#pragma once
#include <fstream>
#include <opencv2/opencv.hpp>
#include <vector>
class CsvDumper
{
public:
  CsvDumper() = default;

  void dump(std::vector<int> vec, std::string fileName)
  {
    std::ofstream dumpCSV(fileName);
    for (decltype(vec.size()) i = 0; i < vec.size(); ++i)
      {
        dumpCSV << vec.at(i) << (i != vec.size() - 1 ? "," : "");
      }
    dumpCSV << '\n';
  }
  void dump(const cv::Mat& m, std::string fileName);
  void dump(const std::vector<cv::Mat>& m, std::string fileName);
  void operator()(const cv::Mat& m, std::string fileName);
  void operator()(const std::vector<cv::Mat>& m, std::string fileName);
  void operator()(const std::vector<int>& m, std::string fileName)
  {
    dump(m, fileName);
  }
};
