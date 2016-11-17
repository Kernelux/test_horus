#include "csvReader.hh"
#include <fstream>
#include <sstream>
#include <boost/algorithm/string.hpp>

cv::Mat CsvReader::read(std::string fileName)
{
  cv::Mat res;
  std::ifstream f(fileName);
  std::string line;
  std::vector<std::string> resVec;
  std::string coma;
  int lineNumber = 0;
  while(std::getline(f, line))
    {
      boost::split(resVec, line, boost::is_any_of(","));
      res.push_back(cv::Mat::zeros(1, resVec.size(), CV_32F));
      for (auto j = 0.0; j < resVec.size(); j++)
        {
          res.at<float>(lineNumber, j) = std::stof(resVec.at(j));
        }
      lineNumber++;
      resVec.clear();
    }
  return res;
}

cv::Mat CsvReader::operator()(std::string fileName)
{
  return read(fileName);
}
