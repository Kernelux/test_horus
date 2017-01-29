#include "predict.hh"
#include <string>
#include "wld.hh"
#include "fileInfo.hh"
#include <opencv2/ml.hpp>
#include <opencv2/xfeatures2d.hpp>
#include "tbb/tbb.h"
#include "mosiftExtractor.hh"
#include "csvReader.hh"
#include "csvDumper.hh"
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
    // for (auto i = 0.0; i < files.size(); i++)
    //   {
    tbb::mutex confMatM;
    tbb::mutex countM;
    CsvReader read;
    CsvDumper dump;
    cv::Mat voc = read("vocabulary.csv");
    int ind = 1;
    tbb::parallel_for(size_t(0),files.size(), [&](size_t i)
      {
        //cv::Ptr<cv::DescriptorMatcher> matcher(new cv::FlannBasedMatcher());
        cv::Ptr<cv::DescriptorMatcher> matcher(new cv::BFMatcher(cv::NORM_L1));
        cv::Ptr<cv::DescriptorExtractor> mos = new Wld();
        cv::BOWImgDescriptorExtractor bowDE(mos, matcher);
        bowDE.setVocabulary(voc);
        std::vector<cv::KeyPoint> kp;
        cv::Mat curFrame;
        cv::Mat res;
        auto fileName = files.at(i).originalFileName;
        auto className = files.at(i).originalDirName;
        cv::VideoCapture v(files.at(i).originalPathName);
        auto countFight = 0.0;
        auto countNoFight = 0.0;
        auto framNum = 0.0;
        std::vector<int> resVec;
        std::vector<std::string> windowNoise(3);
        for (;;)
          {
            v >> curFrame;
            if (curFrame.empty())
              break;
            // imshow("t", curFrame);
            // cv::waitKey(0);
            kp.resize(1);
            bowDE.compute(curFrame, kp, res);
            if (kp.size() == 0)
              continue;
            res.convertTo(res, CV_32FC1);
            // cv::normalize(res, res);
            std::string c = "unclassable";
            for (auto& s : svms)
              {
                if (s.first != "fight")
                  continue;
                auto r = s.second->predict(res);
                if (r == 1)
                  {
                    c = "fight";
                    countFight++;
                  }
                else
                  {
                    c = "nofight";
                    countNoFight++;
                  }
                resVec.push_back(r);
              }
            if (framNum < 3)
                windowNoise.at(static_cast<int>(framNum)) = static_cast<std::string>(c);
            else
              {
                std::swap(windowNoise.at(0), windowNoise.at(1));
                std::swap(windowNoise.at(1), windowNoise.at(2));
                windowNoise.at(2) = static_cast<std::string>(c);
              }
            framNum++;
            // {
            //   tbb::mutex::scoped_lock lock1(confMatM);
            //   if (framNum < 3)
            //       confMatrix[className][c]++;
            //   else
            //     {
            //       if (windowNoise.at(1) == windowNoise.at(0) && windowNoise.at(1) == windowNoise.at(2))
            //         {
            //           confMatrix[className][c]++;
            //         }
            //       else
            //         {
            //           confMatrix[className][windowNoise.at(0)]++;
            //         }
            //     }
            // }
          }
        if (countFight > countNoFight)
          {
            confMatrix[className]["fight"]++;
          }
        else
          confMatrix[className]["nofight"]++;
        {
          tbb::mutex::scoped_lock lock2(countM);
          std::cout << "Video " << std::setw(3) << ind << ' '  << fileName  << ' ' << countFight / framNum << "(fight) "<< countNoFight / framNum << "(nofight) ==> should be have been " << className << std::endl;
          std::cout << ind / framNum << std::endl;
          dump(resVec, "/tmp/" + fileName + ".txt");
          ind++;
        }
      }
      );
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


void Predict::predictUnique(std::string fileName)
{
    CsvReader read;
    cv::Mat voc = read("vocabulary.csv");
    std::ofstream fileOutput("/tmp/random.txt");
    //cv::Ptr<cv::DescriptorMatcher> matcher(new cv::FlannBasedMatcher());
    cv::Ptr<cv::DescriptorMatcher> matcher(new cv::BFMatcher(cv::NORM_L1));
    cv::Ptr<cv::DescriptorExtractor> mos = new Wld();
    cv::BOWImgDescriptorExtractor bowDE(mos, matcher);
    bowDE.setVocabulary(voc);
    std::vector<cv::KeyPoint> kp;
    cv::Mat curFrame;
    cv::Mat res;
    auto countFight = 0.0;
    auto countNoFight = 0.0;
    auto framNum = 0.0;
    int frame = 0;
    std::vector<std::string> windowNoise(3);
    for (;;)
      {
        std::stringstream ss;
        ss << fileName << "/img_" << frame++ << ".png";
        curFrame = cv::imread(ss.str());
        if (curFrame.empty())
          {
            if (std::ifstream(fileName + "/stop").good())
              break;
            else
              {
                frame--;
                continue;
              }
          }
        kp.resize(1);
        bowDE.compute(curFrame, kp, res);
        if (kp.size() == 0)
          continue;
        res.convertTo(res, CV_32FC1);
        std::string c = "unclassalbe";
        for (auto& s : svms)
          {
            if (s.first != "fight")
              continue;
            auto r = s.second->predict(res);
            if (r == 1)
              {
                c = "fight";
                fileOutput << framNum<< std::endl;
                // std::cout << 1 << std::endl;
                countFight++;
              }
            else
              {
                c = "nofight";
                fileOutput << framNum<< std::endl;
                // std::cout << 0 << std::endl;
                countNoFight++;
              }
          }
      }
}
void Predict::predictUnique(std::vector<FileInfo> files)
{
    std::map<std::string, std::map<std::string, int>> confMatrix;
    tbb::mutex confMatM;
    tbb::mutex countM;
    CsvReader read;
    cv::Mat voc = read("vocabulary.csv");
    std::ofstream fileOutput("/tmp/progress.txt");
    for(size_t i = size_t(0); i < files.size(); ++i)
      {
        //cv::Ptr<cv::DescriptorMatcher> matcher(new cv::FlannBasedMatcher());
        cv::Ptr<cv::DescriptorMatcher> matcher(new cv::BFMatcher(cv::NORM_L1));
        cv::Ptr<cv::DescriptorExtractor> mos = new Wld();
        cv::BOWImgDescriptorExtractor bowDE(mos, matcher);
        bowDE.setVocabulary(voc);
        std::vector<cv::KeyPoint> kp;
        cv::Mat curFrame;
        cv::Mat res;
        auto fileName = files.at(i).originalFileName;
        auto className = files.at(i).originalDirName;
        cv::VideoCapture v(files.at(i).originalPathName);
        auto countFight = 0.0;
        auto countNoFight = 0.0;
        auto framNum = 0.0;
        std::vector<std::string> windowNoise(3);
        for (;;)
          {
            v >> curFrame;
            if (curFrame.empty())
              break;
            kp.resize(1);
            bowDE.compute(curFrame, kp, res);
            if (kp.size() == 0)
              continue;
            res.convertTo(res, CV_32FC1);
            // cv::normalize(res, res);
            std::string c = "unclassalbe";
            for (auto& s : svms)
              {
                if (s.first != "fight")
                  continue;
                auto r = s.second->predict(res);
                //std::cout << s.second->predict(res) << std::endl;
                if (r == 1)
                  {
                    c = "fight";
                    std::cout << 1 << std::endl;
                    countFight++;
                  }
                else
                  {
                    c = "nofight";
                    std::cout << 0 << std::endl;
                    countNoFight++;
                  }
              }
            if (framNum < 3)
              windowNoise.at(static_cast<int>(framNum)) = static_cast<std::string>(c);
            else
              {
                std::swap(windowNoise.at(0), windowNoise.at(1));
                std::swap(windowNoise.at(1), windowNoise.at(2));
                windowNoise.at(2) = static_cast<std::string>(c);
              }
            framNum++;
            if (framNum < 3)
              confMatrix[className][c]++;
            else
              {
                if (windowNoise.at(1) == windowNoise.at(0) && windowNoise.at(1) == windowNoise.at(2))
                  confMatrix[className][c]++;
                else
                  {
                    confMatrix[className][windowNoise.at(0)]++;
                  }
              }
            fileOutput << framNum / v.get(cv::CAP_PROP_FRAME_COUNT)<< std::endl;
          }
        fileOutput << 1.0 << std::endl;

      }
}
