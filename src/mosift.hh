#pragma once
#include <map>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/video.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/xfeatures2d.hpp>

enum Position
  {
    RIGHT = 1,
    LEFT = 2,
    TOP = 4,
    BOTTOM = 8,
    RIGHT_TOP = RIGHT | TOP,
    RIGHT_BOTTOM = RIGHT | BOTTOM,
    LEFT_TOP = LEFT | TOP,
    LEFT_BOTTOM = LEFT | BOTTOM,
  };

class MoSIFT
{
public:
  MoSIFT() = default;
  int findRegion(const float& a);
  int findRegion(const float& x, const float& y);
  int findRegionWithPrecalTrigo(const float& cosF, const float& sinF);
  float angleToRad(const float& x);
  std::map<unsigned int, unsigned int> initMap();
  std::map<unsigned int, unsigned int> calcHist(const cv::Mat& flowMat,
                                                const int& size,
                                                const int& x,
                                                const int& y,
                                                const cv::Point2f& original
                                                );
  void buildGaussianPyramid( const cv::Mat& base, std::vector<cv::Mat>& pyr, int nOctaves );
  cv::Mat createInitialImage( const cv::Mat& img, cv::Mat& dst, float sigma );
  void draw(const std::map<unsigned int, unsigned int>& m, const cv::Mat& d, int x, int y);
  std::vector<float> concatenateFeatures(std::vector<float>& features, std::map<unsigned int, unsigned int> m);
private:
  typedef short sift_wt;
  const int SIFT_FIXPT_SCALE = 48;
  const float SIFT_INIT_SIGMA = 0.5f;

};
