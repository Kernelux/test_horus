#include "predict.hh"
#include "wld.hh"
#include "fileInfo.hh"
#include <opencv2/ml.hpp>
#include <opencv2/xfeatures2d.hpp>
#include "tbb/tbb.h"
#include "mosiftExtractor.hh"
#include "csvReader.hh"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/video.hpp>
Predict::Predict(std::vector<FileInfo> files)
  :files(files)
{
  for (auto& f : files)
      svms[f.originalDirName] = cv::ml::SVM::load(f.pathName);
}

void Predict::predict(std::vector<FileInfo> files)
{

    std::map<std::string, std::map<std::string, int>> confMatrix;
    int count = 0;
    //for (auto i = 0.0; i < files.size(); i++)
    tbb::mutex confMatM;
    tbb::mutex countM;
    tbb::parallel_for (size_t(0),files.size(), [&](size_t i)
      {
        cv::Ptr<cv::DescriptorMatcher> matcher(new cv::FlannBasedMatcher());
        cv::Ptr<cv::DescriptorExtractor> mos = new Wld();
        //cv::Ptr<cv::Feature2D> sift = cv::xfeatures2d::SIFT::create();
        cv::BOWImgDescriptorExtractor bowDE(mos, matcher);
        CsvReader read;
        bowDE.setVocabulary(read("vocabulary.csv"));
        std::vector<cv::KeyPoint> kp;
        cv::Mat curFrame;
        cv::Mat res;
        auto fileName = files.at(i).originalFileName;
        auto className = files.at(i).originalDirName;
        cv::VideoCapture v(files.at(i).originalPathName);
        bool stop = false;
        for (;!stop;)
          {
            v >> curFrame;
            if (curFrame.empty())
              break;

            kp.resize(1);
            bowDE.compute(curFrame, kp, res);

            if (kp.size() == 0)
              continue;
            // char key = (char)cv::waitKey(10);
            // switch (key)
            //   {
            //   case 'q':
            //   case 'Q':
            //   case 27: //escape key
            //     stop = true;
            //     break;
            //   default:
            //     break;
            //   }
            // cv::Mat out;
            // cv::drawKeypoints(curFrame, kp, out, cv::Scalar(255));
            // imshow("f", out);
            res.convertTo(res, CV_32FC1);
            float m = -FLT_MAX;
            std::string c = "_ERROR_";
            for (auto& s : svms )
              {
                double f = s.second->predict(res);
                if (f > m)
                  {
                    m = f;
                    c = s.first;
                  }
                //std::cout << s.first << ": " << f << ", ";
              }
            {
              tbb::mutex::scoped_lock lock1(confMatM);
              confMatrix[className][c]++;
            }
            // std::cout << std::endl;
            // std::cout << "Is " << className << " found ";
            // std::cout << c << " " << m << std::endl;
          }
        {
          tbb::mutex::scoped_lock lock2(countM);
          count++;
        }
        std::cout << count << "/" << files.size() << std::endl;
      });
    for (auto& m : confMatrix)
      {
        auto tot = 0.0;
        std::cout << "Results for " << m.first << ": ";
        for (auto& c : m.second)
            tot += c.second;
        for (auto& c : m.second)
            std::cout << c.first << ": " <<  c.second << "("<< (100 * c.second/tot)<<"%)"<< ";";
        std::cout << std::endl;
      }
}
