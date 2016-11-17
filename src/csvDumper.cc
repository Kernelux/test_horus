#include "csvDumper.hh"

void CsvDumper::dump(const cv::Mat& m, std::string fileName)
{
  std::ofstream dumpCSV(fileName);
  for (auto y = 0; y < m.rows; y++)
    {
      for (auto x = 0; x < m.cols; x++)
        dumpCSV << m.at<float>(y, x)
                        << (x < m.cols - 1  ? "," : "");
      dumpCSV << "\n";
    }
}

void CsvDumper::dump(const std::vector<cv::Mat>& mat, std::string fileName)
{
  std::ofstream dumpCSV(fileName);
  for (auto& m : mat)
    for (auto y = 0; y < m.rows; y++)
      {
        for (auto x = 0; x < m.cols; x++)
          dumpCSV << m.at<float>(y, x)
                  << (x < m.cols - 1  ? "," : "");
        dumpCSV << "\n";
      }
}

void CsvDumper::operator()(const cv::Mat& m, std::string fileName)
{
  dump(m, fileName);
}

void CsvDumper::operator()(const std::vector<cv::Mat>& mat, std::string fileName)
{
  dump(mat, fileName);
}
