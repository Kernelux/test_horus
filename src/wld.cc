#include "wld.hh"
#include <algorithm>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/optflow.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/video.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <vector>
#include "tbb/tbb.h"
#include <string>

decltype(auto) findWeberWindowPoint(const int& xPos, const int& yPos, const int& scale, const cv::Mat& mat)
{
  std::vector<cv::Point2f> pivot = {
    cv::Point2f(xPos - scale, yPos - scale),
    cv::Point2f(xPos + scale, yPos - scale),
    cv::Point2f(xPos + scale, yPos + scale),
    cv::Point2f(xPos - scale, yPos + scale),
  };
  std::vector<cv::Point2f> points;
  cv::Point2f tmp;
  for (int i = 0; i < 2 * scale; ++i)
    {
      if (pivot.at(0).x >= 0 && pivot.at(0).x < mat.cols && pivot.at(0).y >= 0 && pivot.at(0).y < mat.rows)
        points.push_back(pivot.at(0));
      pivot.at(0).x++;
      if (pivot.at(1).x >= 0 && pivot.at(1).x < mat.cols && pivot.at(1).y >= 0 && pivot.at(1).y < mat.rows)
        points.push_back(pivot.at(1));
      pivot.at(1).y++;
      if (pivot.at(2).x >= 0 && pivot.at(2).x < mat.cols && pivot.at(2).y >= 0 && pivot.at(2).y < mat.rows)
        points.push_back(pivot.at(2));
      pivot.at(2).y--;
      if (pivot.at(3).x >= 0 && pivot.at(3).x < mat.cols && pivot.at(3).y >= 0 && pivot.at(3).y < mat.rows)
        points.push_back(pivot.at(3));
      pivot.at(3).x--;
    }
  return points;
}
decltype(auto) dotProduct(std::vector<double> A, std::vector<double> B) noexcept
{
  auto res = 0.0;
  for (auto i = 0.0; i < A.size(); ++i)
      res += A.at(i) * B.at(i);
  return res;
}

decltype(auto) calcAngle(cv::Point& xc, cv::Point& x) noexcept
{
 auto l = cv::norm(xc) * cv::norm(x);
  if (l == 0)
    return acos(0);
  auto res = xc.dot(x);
  if (res >= l)
    return 0.0;
  else if (res <= -l)
    return M_PI;
  return acos(res / l);
}
decltype(auto) calcAngle(cv::Point& xc, cv::Point&& x) noexcept
{
  auto l = cv::norm(xc) * cv::norm(x);
  if (l == 0)
    return acos(0);
  auto res = xc.dot(x);
  if (res >= l)
    return 0.0;
  else if (res <= -l)
    return M_PI;
  return acos(res / l);
}
//Ameliorable de ouf ...
decltype(auto) improvedWeber(const cv::Mat& input, const int& xPos, const int& yPos, const int& scale) noexcept
{
  int nbPoint = 8 * scale;
  std::vector<double>  X(nbPoint);
  std::vector<double> Jx(nbPoint);
  std::vector<double> Jy(nbPoint);
  cv::Point xc(xPos, yPos);
  auto&& currentPoint = input.at<uchar>(yPos, xPos) + 1.0;
  auto&& points = findWeberWindowPoint(xPos, yPos, scale, input);
  int current = 0;
  auto&& angle = 0.0;
  for (auto&& p : points)
    {
      X.at(current) = ((input.at<uchar>(p) + 1) - currentPoint);
      angle = calcAngle(xc, cv::Point(p.x - xPos, p.y - yPos));
      Jx.at(current) = angle;
      current++;
    }
  std::transform(std::begin(Jx), std::end(Jx), std::begin(Jy), [](auto&& x){return sin(x);});
  std::transform(std::begin(Jx), std::end(Jx), std::begin(Jx), [](auto&& x){return cos(x);});
  auto&& alpha = 6.0f;
  auto&& coses =   alpha * dotProduct(X, Jx);
  auto&& sinuses = alpha * dotProduct(X, Jy);
  coses    = atan(coses   / currentPoint);
  sinuses  = atan(sinuses / currentPoint);
  return std::make_pair(coses, sinuses);
}


decltype(auto) mapTheta(double v3, double v4) noexcept
{
  if (v3 == 0 && v4 == 0)
    return 0.0;
  return atan2(v3, v4) + M_PI;
}

decltype(auto) orientIndToAng(const int& ind) noexcept
{
  if (ind == 1)
    return 0.0 + 0.15 * 0.5f;
  if (ind == 2)
    return (0.15 + 0.35) * 0.5f;
  if (ind == 3)
    return (0.35 + 0.5) * 0.5f;
    if (ind == 4)
      return (0.5 + 0.65) * 0.5f;
  if (ind == 5)
    return (0.65 + 0.85) * 0.5f;
  if (ind == 6)
    return (0.85 + M_PI) * 0.5f;
  if (ind == 7)
    return (M_PI + 1.15 * M_PI) * 0.5f;
    if (ind == 8)
      return (1.15 * M_PI + 1.35 * M_PI) * 0.5f;
  if (ind == 9)
    return (1.35 * M_PI + 1.5 * M_PI) * 0.5f;
    if (ind == 10)
      return (1.5 * M_PI + 1.65 * M_PI) * 0.5f;
  if (ind == 11)
    return (1.65 * M_PI + 1.85 * M_PI) * 0.5f;
  return (1.85 * M_PI + 2.0 * M_PI) * 0.5f;
}

decltype(auto) mapOrientToInd(double thetaPrime)
{
  if (thetaPrime >= 0 && thetaPrime < 0.15)
    return 1;
  if (thetaPrime >= 0.15 && thetaPrime < 0.35)
    return 2;
  if (thetaPrime >= 0.35 && thetaPrime < 0.5)
    return 3;
  if (thetaPrime >= 0.5 && thetaPrime < 0.65)
    return 4;
  if (thetaPrime >= 0.65 && thetaPrime < 0.85)
    return 5;
  if (thetaPrime >= 0.85 && thetaPrime < M_PI)
    return 6;
  if (thetaPrime >= M_PI && thetaPrime < 1.15 * M_PI)
    return 7;
  if (thetaPrime >= 1.15 * M_PI && thetaPrime < 1.35 * M_PI)
    return 8;
  if (thetaPrime >= 1.35 * M_PI && thetaPrime < 1.5 * M_PI)
    return 9;
  if (thetaPrime >= 1.5  * M_PI && thetaPrime < 1.65 * M_PI)
    return 10;
  if (thetaPrime >= 1.65 * M_PI && thetaPrime < 1.85 * M_PI)
    return  11;
  if (thetaPrime >= 1.85 * M_PI && thetaPrime <= 2.0 * M_PI)
    return  12;
  return 1;
}
auto floatMod(double a, double b)
{
  return fmod((fmod(a, b) + b),b);
}

auto findGreatest(cv::Mat& memoMat,
                  const int& x,
                  const int& y, std::vector<int>& greatest)
{
  for (int i = y - 1; i <= y + 1; ++i)
    {
      if (i >= memoMat.rows || i < 0)
        continue;
        for (int j = x - 1; j <= x + 1; ++j)
          {
            if (j >= memoMat.cols || j < 0)
              continue;
            greatest.at(mapOrientToInd(memoMat.at<cv::Point2f>(i, j).y) - 1) +=  1;
          }
    }
}
auto Wld::subCalcHistOrien(cv::Mat& memoMat,
                           cv::Mat& motionMat,
                           const int& x,
                           const int& y, const int& xori, const int& yori)
{
  //We put at the same time flow and orientation
  std::vector<double> orien(24);
  auto dist = 0.0;
  for (int i = y - 1; i <= y + 1; ++i)
    {
      if (i >= memoMat.rows || i < 0)
        continue;
        for (int j = x - 1; j <= x + 1; ++j)
          {
            if (j >= memoMat.cols || j < 0)
              continue;
            dist =  sqrt((xori - j) * (xori - j) + (yori - i) * (yori - i));
            orien.at(mapOrientToInd(memoMat.at<cv::Point2f>(i, j).y) - 1) +=  100.0 * memoMat.at<cv::Point2f>(i, j).x / (dist + 0.1f);
            //We do an offset of 12 to be in the motion part
            auto f = motionMat.at<cv::Point2f>(i, j);
            orien.at(12 + mapOrientToInd(atan2(f.y, f.x)) - 1) +=  100.0 * cv::norm(motionMat.at<cv::Point2f>(i, j)) / (dist + 0.1f);
          }
    }
  return orien;
}

auto Wld::calcHist(cv::Mat& memoMat,
                   cv::Mat& flowMat,
                   const int& x,
                   const int& y)
{
  cv::Mat saveMat = cv::Mat(auxCurrent.rows, auxCurrent.cols, CV_32FC2, cv::Scalar(0,0));
  std::vector<int> greatest(12);
  std::vector<double> f;
  auto newX = x - 3 * 2;
  auto newY = y - 3 * 2;
  //Calc greatest orientation
  for (auto i = newX; i < x + 3 * 2; i++)
    for (auto j = newY; j < y + 3 * 2; j++)
      greatest.at(mapOrientToInd(memoMat.at<cv::Point2f>(j, i).y) - 1) +=  1;
  //find major orientation
  auto ind = std::distance(std::begin(greatest), std::max_element(std::begin(greatest),std::end(greatest))) + 1.0;
  for (auto i = newX; i < x + 3 * 2; i++)
    {
      if (i >= memoMat.cols || i < 0)
        continue;
      for (auto j = newY; j < y + 3 * 2; j++)
        {
          if (j >= memoMat.rows || j < 0)
            continue;
          saveMat.at<cv::Point2f>(j, i).y = floatMod(memoMat.at<cv::Point2f>(j, i).y - orientIndToAng(ind), M_PI * 0.5f);
          saveMat.at<cv::Point2f>(j, i).x = memoMat.at<cv::Point2f>(j, i).x;
        }
    }
  int descSize = 192;
  std::vector<double> finalRes(descSize * 2);
  int c = 0;
  for (auto i = newX; i < x + 3 * 2; i += 3)
    {
      if (i >= memoMat.cols || i < 0)
        continue;
      for (auto j = newY; j < y + 3 * 2; j += 3)
        {
          if (j >= memoMat.rows || j < 0)
            continue;
          auto&& vecSub = subCalcHistOrien(saveMat, flowMat, i + 1, j + 1, x, y);
          for(auto k = 0.0; k < vecSub.size() - 12; ++k)
            {
              finalRes.at(c + k) = vecSub.at(k);
              finalRes.at(c + k + descSize) = vecSub.at(k + 12);
            }
          c += 12;
        }
    }
  return finalRes;
}

void Wld::compute(cv::InputArray image,
                  std::vector<cv::KeyPoint>& keypoints,
                  cv::OutputArray descriptors)
{
  //std::cout << "bite" << std::endl;
  cv::Ptr<cv::DenseOpticalFlow> flow = cv::optflow::createOptFlow_Farneback();
  if (!created)
    {
      previousFrame = image.getMat();
      pyrOptFlow.resize(nbPyr + 1);
      meanMov.resize(nbPyr + 1);
      stdMov.resize(nbPyr + 1);
      cv::cvtColor(previousFrame, auxPrevious, cv::COLOR_BGR2GRAY);
      cv::buildPyramid(previousFrame, oldPyr, nbPyr);
      for (auto i = 0; i < nbPyr + 1; i++)
        pyrOptFlow.at(i) = previousFrame.clone();
      for (size_t i = 0; i < oldPyr.size(); i++)
          cv::cvtColor(oldPyr.at(i), oldPyr.at(i), cv::COLOR_BGR2GRAY);
      created = !created;
      keypoints.clear();
      return;
    }
  //To empty keypoints
  keypoints.clear();
  currentFrame = image.getMat();
  cv::cvtColor(currentFrame, auxCurrent, cv::COLOR_BGR2GRAY);
  cv::buildPyramid(auxCurrent, newPyr, nbPyr);
  tbb::parallel_for(size_t(0), newPyr.size(),
                    [&](size_t i)
                    {
                      flow->calc(oldPyr.at(i), newPyr.at(i), pyrOptFlow.at(i));
                    });
    // WLD REAL PART
    cv::Mat blured;
    cv::GaussianBlur(auxCurrent, blured, cv::Size(), 1.8);

    // cv::Mat drawingA = currentFrame.clone();
    std::vector<cv::Point2f> weberKeypoints;
    //Compute gradient
    std::vector<cv::Point2f> coordinate;
    std::vector<double> norma;
    for (double i = 0.0; i < pyrOptFlow.size(); ++i)
      {
        double maxi = 0.0;
        double movMean = 0;
        for (auto it = pyrOptFlow.at(i).begin<cv::Point2f>(); it != pyrOptFlow.at(i).end<cv::Point2f>(); it++)
          {
            auto dist = cv::norm(*it);
            maxi = fmax(dist, maxi);
            movMean += dist;
          }
        meanMov.at(i) = movMean / (pyrOptFlow.at(i).cols*pyrOptFlow.at(i).rows);
        norma.emplace_back(maxi);
      }
    for (int i = 0; i < auxCurrent.rows; i++)
        for (int j = 0; j < auxCurrent.cols; j++)
            coordinate.emplace_back(cv::Point2f(j, i));
    std::vector<cv::Point2f> extrema = coordinate;
    cv::Mat currentScale;
    cv::Mat memoization;
    std::vector<cv::Point2f> newCoord;
    std::vector<cv::Point2f> extremaAux;
    for (double scale = 1; scale <=  nbPyr + 1; ++scale)
      {
        newCoord.clear();
        extremaAux.clear();
        currentScale = cv::Mat(auxCurrent.rows, auxCurrent.cols, CV_32FC2, cv::Scalar(0,0));
        tbb::parallel_for(size_t(0), coordinate.size(), [&] (auto&& i)
          {
            auto&& p = coordinate.at(i);
            auto&& pairWeber = improvedWeber(blured, p.x, p.y, scale);
            auto&& orientation = mapTheta(pairWeber.second, pairWeber.first);
            auto&& magni = sqrt(pairWeber.first * pairWeber.first + pairWeber.second * pairWeber.second);
            currentScale.at<cv::Point2f>(p) = cv::Point2f(magni, orientation);
          });
        if (scale == 1)
          memoization = currentScale;
        // cv::Mat show = cv::Mat(auxCurrent.rows, auxCurrent.cols, CV_8UC3,     cv::Scalar(0,0,0));
        // int ratio = 1;
        // for (int i = 0; i < scale; i++)
        //   ratio *= 10;
        // tbb::parallel_for(size_t(0), coordinate.size(), [&] (auto&& i)
        //   {
        //     auto&& p = coordinate.at(i);
        //     show.at<cv::Vec3b>(p) = {0, 0, static_cast<unsigned char>(memoization.at<cv::Point2f>(p).x * 10 * ratio)};
        //   });
        // imshow(std::to_string(scale), show);
        // Find local extram
        // We process motion and WLD at the same time, in order to have better performance
        for (auto&& p : extrema)
          {
            // This thresh cannot be tweak a lot...
            bool done = true;
            double i = 1;
            auto threshInit =  0.05f;
            auto thresh = threshInit;
            (void)thresh;
            for (size_t t = 0; t < pyrOptFlow.size(); t++)
              {
                auto& pt = pyrOptFlow.at(t).at<cv::Point2f>(p.y / i + 0.5f, p.x / i + 0.5f);
                // This thresh cannot be tweak a lot...
                if (cv::norm(pt) < meanMov.at(t)    || cv::norm(pt) < thresh * norma.at(t))
                  {
                    done = false;
                    break;
                  }
                i *= 2.0;
              }
            if (!done)
              continue;
            auto&& p_w = findWeberWindowPoint(p.x, p.y, scale, currentScale);
            for (auto&& f : p_w)
              {
                double i = 1;
                for (size_t t = 0; t < pyrOptFlow.size(); t++)
                  {
                    auto& pt = pyrOptFlow.at(t).at<cv::Point2f>(f.y / i + 0.5f, f.x / i + 0.5f);
                    // This thresh cannot be tweak a lot...
                    if (cv::norm(pt) < meanMov.at(t) || cv::norm(pt) < thresh * norma.at(t))
                      //if (cv::norm(pt) - meanMov.at(t) < thresh * (norma.at(t) - meanMov.at(t)))
                      {
                        done = false;
                        break;
                      }
                    i *= 2.0;
                  }
              }
            if (!done)
              continue;
            int counter = 0;
            auto v = 0.0;
            for (auto&& pp : p_w)
              {
                if (currentScale.at<cv::Point2f>(p).x > currentScale.at<cv::Point2f>(pp).x)
                  {
                    counter++;
                    v = v + currentScale.at<cv::Point2f>(pp).x;
                  }
                  else
                    break;
              }
            if (counter == scale * 8 && v != 0)
              {
                extremaAux.push_back(p);
                newCoord.push_back(p);
                if (scale < nbPyr)
                  {
                    p_w = findWeberWindowPoint(p.x, p.y, scale + 1, currentScale);
                    newCoord.insert(std::end(newCoord), std::begin(p_w),std::end(p_w));
                  }
                else
                  {
                    weberKeypoints.emplace_back(p);
                  }
              }
          }
        std::swap(extrema, extremaAux);
        std::swap(coordinate, newCoord);
      }
    auto resWLD = std::vector<std::vector<double>>();
    for (auto&& points : weberKeypoints)
      {
        // cv::circle(drawingA, points, 3, cv::Scalar(0, 0, 255));
        resWLD.emplace_back(calcHist(memoization, pyrOptFlow.at(0), points.x, points.y));
        auto v = cv::KeyPoint();
        v.pt = points;
        keypoints.push_back(v);
      }
  // cv::waitKey(1);
  // imshow("MagniSave", drawingA);
  // imshow("test", auxCurrent);

  cv::swap(previousFrame,currentFrame);
  cv::swap(auxPrevious, auxCurrent);
  std::swap(oldPyr, newPyr);
  cv::Mat res;
  if (resWLD.size() > 0)
    {
      res = cv::Mat::zeros(resWLD.size(), resWLD.at(0).size(), CV_32F);
      for (size_t y = 0; y < resWLD.size(); y++)
        for (size_t x = 0; x < resWLD.at(y).size(); x++)
          res.at<float>(y, x) = resWLD.at(y).at(x);
    }
  descriptors.assign(res);
}
