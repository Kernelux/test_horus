#include "motion.hh"

int Motion::findRegion(const float& a)
{
  int region = LEFT;
  int it = 0;
  float angle = angleToRad(a);
  std::vector<int> pos = {RIGHT , RIGHT | TOP, TOP, LEFT | TOP, LEFT};
  for (float i = 22.5; i <= 222.5; i += 45, it++)
      if (cos(angle) >= cos(angleToRad(i)))
        {
          region = pos[it];
          break;
        }
  if (sin(angle) < 0 && region >> 2 == 1)
    region = (region & ~TOP) | BOTTOM;
  return region;
}


int Motion::findRegion(const float& x, const float& y)
{
  float hyp = sqrt(x * x + y * y);
  float coss = x/hyp;
  float sins = y/hyp;
  int region = LEFT;
  int it = 0;
  std::vector<int> pos = {RIGHT , RIGHT | TOP, TOP, LEFT | TOP, LEFT};
  for (float i = 22.5; i <= 222.5; i += 45, it++)
      if (coss >= cos(angleToRad(i)))
        {
          region = pos[it];
          break;
        }
  if (sins < 0 && region >> 2 == 1)
    region = (region & ~TOP) | BOTTOM;
  return region;
}

int Motion::findRegionWithPrecalTrigo(const float& cosF, const float& sinF)
{
  auto thetaPrime = atan2(sinF, cosF);
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

float Motion::angleToRad(const float& x)
{
  return x * M_PI / 180.0;
}


decltype(auto) Motion::initMap()
{
  return std::vector<double>();
}

decltype(auto) Motion::calcHist(const cv::Mat& flowMat,
                                const int& size,
                                const int& x,
                                const int& y,
                                const cv::Point2f& original)
{
  int botX = x - size / 2;
  int botY = y - size / 2;
  int topX = x + size / 2;
  int topY = y + size / 2;
  auto histo = initMap();
  int region;
  for (int i = botY; i < topY; i++)
    {
      if (i >= flowMat.rows || i < 0)
        continue;
      for (int j = botX; j < topX; j++)
        {
          if (j >= flowMat.cols || j < 0)
            continue;
          auto flow = flowMat.at<cv::Point2f>(i, j);
          auto hyp = cv::norm(flow);
          auto dist = cv::norm(original - cv::Point2f(j, i));
          region = findRegionWithPrecalTrigo(flow.x, flow.y);
          histo.at(region) = (histo.at(region))  + 10 * (hyp / dist);
        }
    }
  return histo;
}


void Motion::draw(const std::map<unsigned int, unsigned int>& m, const cv::Mat& d, int x, int y)
{
  auto count = 8.0;
  auto i2 = 1.0;
  for (auto& v : m)
    {
      i2 += v.second;
      count++;
    }
  i2 /= count;
  cv::Point coordinate;
  for (auto bin : m)
    {
      auto coef = (bin.second + 0.0) / i2;
      if (bin.first == RIGHT)
        coordinate = cv::Point(x + coef, y);
      else if (bin.first == LEFT)
        coordinate = cv::Point(x - coef, y );
      else if (bin.first == TOP)
        coordinate = cv::Point(x, y + coef);
      else if (bin.first == BOTTOM)
        coordinate = cv::Point(x, y - coef);
      else if (bin.first == RIGHT_BOTTOM)
        coordinate = cv::Point(x + coef, y - coef);
      else if (bin.first == RIGHT_TOP)
        coordinate = cv::Point(x + coef, y + coef);
      else if (bin.first == LEFT_BOTTOM)
        coordinate = cv::Point(x - coef, y - coef);
      else if (bin.first == LEFT_TOP)
        coordinate = cv::Point(x - coef, y + coef);
      cv::arrowedLine(d, cv::Point(x, y), coordinate, cv::Scalar(255, 0, 0));
    }
}

std::vector<float> Motion::concatenateFeatures(std::vector<float>& features, std::map<unsigned int, unsigned int> m)
{
  std::for_each(m.begin(), m.end(), [&features](auto& x){ features.push_back(x.second);});
  return features;
}
