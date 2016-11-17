#include "mosiftExtractor.hh"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/video.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <vector>
#include "tbb/tbb.h"


const cv::Ptr<cv::Feature2D> MosiftExtractor::sift = cv::xfeatures2d::SIFT::create();

void MosiftExtractor::compute(cv::InputArray image,
                             std::vector<cv::KeyPoint>& keypoints,
                             cv::OutputArray descriptors)
{
  if (!created)
    {
      previousFrame = image.getMat();
      pyrOptFlow.resize(nbPyr + 1);
      meanMov.resize(nbPyr + 1);
      stdMov.resize(nbPyr + 1);
      cv::cvtColor(previousFrame, auxPrevious, cv::COLOR_BGR2GRAY);
      cv::buildPyramid(previousFrame, oldPyr, nbPyr);
      for (auto i = 0; i < nbPyr + 1; i++)
        pyrOptFlow.at(i) = cv::Mat();
      for (size_t i = 0; i < oldPyr.size(); i++)
        {
          //We blur and init our first pyramid
          GaussianBlur(oldPyr.at(i), oldPyr.at(i), cv::Size(), sig_diff, sig_diff);
          cv::cvtColor(oldPyr.at(i), oldPyr.at(i), cv::COLOR_BGR2GRAY);
        }
      created = !created;
      keypoints.clear();
      return;
    }
  currentFrame = image.getMat();
  sift->compute(image, keypoints, desc);
  cv::buildPyramid(currentFrame, newPyr, nbPyr);
  tbb::parallel_for(size_t(0), newPyr.size(),
                    [&](size_t i)
                    {
                      cv::cvtColor(newPyr.at(i), newPyr.at(i), cv::COLOR_BGR2GRAY);
                      GaussianBlur(newPyr.at(i), newPyr.at(i), cv::Size(), sig_diff, sig_diff);
                      cv::calcOpticalFlowFarneback(oldPyr.at(i), newPyr.at(i), pyrOptFlow.at(i), 0.5, 3, nbPyr, 15, 3, 1.6, 0);
                    });
  cv::cvtColor(currentFrame, auxCurrent, cv::COLOR_BGR2GRAY);
  //We save features of sift into a vector
  features.clear();
  for (int j = 0; j < desc.rows; ++j)
    features.push_back(desc.row(j));
  // We calculate the mean of motion in each pyramid
  for (size_t t = 0; t < pyrOptFlow.size(); t++)
    {
      double movMean = 0;
      for (auto x = 0; x < pyrOptFlow.at(t).cols; x++)
        for (auto y = 0; y < pyrOptFlow.at(t).rows; y++)
          {
            auto p = pyrOptFlow.at(t).at<cv::Point2f>(y, x);
            movMean += sqrt(p.x * p.x + p.y * p.y);
          }
      meanMov.at(t) = movMean/(pyrOptFlow.at(t).cols*pyrOptFlow.at(t).rows);
    }
    for (size_t t = 0; t < pyrOptFlow.size(); t++)
    {
      double movSTD = 0;
      auto currentMean = meanMov.at(t);
      for (auto x = 0; x < pyrOptFlow.at(t).cols; x++)
        for (auto y = 0; y < pyrOptFlow.at(t).rows; y++)
          {
            auto p = pyrOptFlow.at(t).at<cv::Point2f>(y, x);
            auto tmp = sqrt(p.x * p.x + p.y * p.y) - currentMean;
            movSTD = movSTD + tmp * tmp;
          }
      stdMov.at(t) = sqrt(movSTD / (pyrOptFlow.at(t).cols * pyrOptFlow.at(t).rows));
    }

  //We remove motion without sufficient motion
  size_t k = 0;
  while (k < keypoints.size())
    {
      float coordX = keypoints.at(k).pt.x;
      float coordY = keypoints.at(k).pt.y;
      for (size_t t = 0; t < pyrOptFlow.size(); t++)
        {
          auto p = pyrOptFlow.at(t).at<cv::Point2f>(coordY, coordX);
          // This thresh cannot be tweak a lot...
          if (sqrt(p.x * p.x + p.y * p.y) < meanMov.at(t) + stdMov.at(t))
            {
              keypoints.erase(keypoints.begin() + k);
              features.erase(features.begin() + k);
              k--;
              break;
            }
          coordX /= 2.0;
          coordY /= 2.0;
        }
      k++;
    }
  // We calculate the motion part of mosift
  // and concatenate the new feature (motion) to the previous one ("angles")
  tbb::parallel_for(size_t(0), keypoints.size(),
                    [&](size_t i)
                    {
                      //we calculate an histogram of 4 * 4 * 4 * 4
                      //4 *
                      for (auto y = keypoints.at(i).pt.y - 8; y < keypoints.at(i).pt.y + 8; y +=4)
                        {
                          // 4 *
                          for (auto x = keypoints.at(i).pt.x - 8; x < keypoints.at(i).pt.x + 8; x +=4)
                            {
                              // 4 * 4
                            }
                        };
                    });
  // imshow("move2", drawing);
  // imshow("current", currentFrame);
  cv::swap(previousFrame,currentFrame);
  cv::swap(auxPrevious, auxCurrent);
  std::swap(oldPyr, newPyr);
  cv::Mat res;
  if (features.size() > 0)
    {
      res = cv::Mat::zeros(features.size(), features.at(0).size(), CV_32F);
      for (size_t y = 0; y < features.size(); y++)
        for (size_t x = 0; x < features.at(y).size(); x++)
            res.at<float>(y, x) = features.at(y).at(x);
    }
  descriptors.assign(res);
}
