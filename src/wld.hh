#pragma once
#include <opencv2/xfeatures2d.hpp>
#include "motion.hh"
class Wld : public cv::Feature2D
{
  public:
  Wld() = default;
  void compute(cv::InputArray image,
               std::vector<cv::KeyPoint>& keypoints,
               cv::OutputArray desc) override;
  auto calcHist(cv::Mat& memoMat,
                cv::Mat& flowMat,
                const int& x,
                const int& y);
  auto subCalcHistOrien(cv::Mat& memoMat,cv::Mat& motionMat,
                        const int& x,
                        const int& y, const int& xori, const int& yori);
private:
   // Used for the flow (kind of diff between them)
  cv::Mat previousFrame;
  cv::Mat currentFrame;
  // Used for grayscale
  cv::Mat auxCurrent;
  cv::Mat auxPrevious;
  std::vector<cv::KeyPoint> keypoints;

  cv::Mat desc = cv::Mat::zeros(previousFrame.size(), CV_32F);
  std::vector<cv::Mat> totalFeatureOfVideo;
  //Used for Motiondetection in mosift.
  // Contains pyramid of previousframe
  std::vector<cv::Mat> oldPyr;
  // Contains pyramid of currentframe
  std::vector<cv::Mat> newPyr;
  // Contains pyramid of memoization
  // Aux matrix used to store flow
  cv::Mat resFlow;
  // Init of the first frame
  int nbPyr = 3;
  float sig_diff = sqrtf(fmax(1.6 * 1.6 - 0.5f * 0.5f * 4, 0.01f));
  // flow between pyramids
  std::vector<cv::Mat> pyrOptFlow;
  std::vector<double> meanMov;
  std::vector<double> stdMov;
  std::vector<std::vector<float>> features;
  const static cv::Ptr<cv::Feature2D> sift;
  Motion motion;
  bool created = false;
};
