#include "train.hh"
#include "motion.hh"
#include "mosiftExtractor.hh"
#include "wld.hh"
#include "csvReader.hh"
#include <utility>

Train::Train(std::vector<FileInfo> files, int bowk)
  :files(files),
   everyFeatures(cv::Mat(0, 256, CV_32F, 0)),
   bowk(bowk)
{}

static std::vector<cv::Mat> motionDetection(cv::VideoCapture& v, FileInfo* fi=NULL)
{
  // MosiftExtractor m;
  Wld m;
  cv::Mat currentFrame;
  cv::Mat auxCurrent;
  std::vector<cv::Mat> totalFeatureOfVideo;
  std::vector<cv::KeyPoint> keypoints;
  cv::Mat descriptors;
  CsvDumper dump;
  int i = 0;
  for (;;)
    {
      v >> currentFrame;
      if (currentFrame.empty())
        break;
      cv::cvtColor(currentFrame, auxCurrent, cv::COLOR_BGR2GRAY);
      m.compute(currentFrame, keypoints, descriptors);
      if (descriptors.empty())
          continue;
      else if (fi != NULL)
        {
          std::ostringstream dstream;
          dstream << "_frame_" << i++ << ".desc";
          dump(descriptors, fi->dirName + fi->fileName + dstream.str());
        }
      totalFeatureOfVideo.push_back(descriptors.clone());
    }
  return totalFeatureOfVideo;
}

void Train::calculateDescriptor(std::string directoryToSaveTo, bool shouldDump)
{
  featuresDir = directoryToSaveTo;
  everyFeaturesByVideo.resize(files.size());
  CsvDumper dump;
  tbb::mutex _mut;
  int fileNum = 0;
  std::map<std::string, cv::Mat> byFileDesc;
  tbb::parallel_for(size_t(0), files.size(),
                    [&](size_t i)
  // for(auto i = 0.0; i < files.size(); i++)
                    {
                      auto& fi = files.at(i);

                      cv::VideoCapture capture;
                      capture.open(fi.pathName);
                      std::vector<cv::Mat> features;
                      fi.setDirPrefix(directoryToSaveTo);
                      if(capture.isOpened())
                        {
                          std::clog << "START EXTRACT FEATURE MOSIFT " << fi.pathName << std::endl;
                          if(shouldDump)
                            features = motionDetection(capture, &fi);
                          else
                            features = motionDetection(capture);
                          capture.release();
                        }
                      else
                        std::cerr << "ERR: cannot open " << fi.fileName << std::endl;
                      if (byFileDesc.find(fi.originalDirName) == std::end(byFileDesc))
                        {
                          std::cout << fi.originalDirName << std::endl;
                          byFileDesc[fi.originalDirName] = cv::Mat(0, 256, CV_32F, 0);
                        }
                      for (const auto& f : features)
                        byFileDesc[fi.originalDirName].push_back(f);
                      everyFeaturesByVideo.at(i) = features;
                      std::cout << fi.originalDirName << std::endl;
                      if (everyFeaturesByVideo.at(i).size() != 0)
                        {
                          dump(everyFeaturesByVideo.at(i), fi.dirName + fi.fileName + ".csv");
                        }
                      else
                        {
                          std::clog << "dank " << i << std::endl;
                        }
                      {
                        tbb::mutex::scoped_lock lock1(_mut);
                        fileNum++;
                        std::clog << fileNum << '/' << files.size() << " (" <<  100.0 * fileNum / files.size()<< "%)" << std::endl;
                      }
                    });
  for (auto& v : everyFeaturesByVideo)
    {
      for (auto& f : v)
        {
          everyFeatures.push_back(f);
        }
    }

  if (shouldDump)
    {
      std::clog << "Dump total feature video" << std::endl;
      dump(everyFeatures, "totalFeatures.csv");
      for (const auto& p : byFileDesc )
        {
          dump(p.second, p.first  + "TOTDESC.csv");
        }
      std::clog << "Dump total feature video DONE" << std::endl;
    }
}

void Train::clusterBow(std::string fileName)
{
  if (fileName != "")
    everyFeatures = CsvReader()(fileName);
  std::clog << "INIT CLUSTERING" << std::endl;
  for (int i = 0; i < everyFeatures.rows; i++)
    bowk.add(everyFeatures.row(i));
  std::clog << "START CLUSTERING" << std::endl;
  voc = bowk.cluster();
  std::clog << "CLUSTERING DONE" << std::endl;
  CsvDumper()(voc, "vocabulary.csv");
}

void Train::translateToBow(std::string vocFile, std::string featureDir)
  {

    codedMat.clear();
    classesName.clear();
    resCols = 0;
    resType = 0;
    tbb::mutex mut;
    // for (auto i = 0.0; i < files.size(); i++)
    tbb::parallel_for(size_t(0), files.size(), [&](size_t i)
      {
        //Pas propre mais en attendant de trouver une jolie manière de reinit
        CsvDumper dump;
        std::vector<cv::KeyPoint> kp;
        cv::Mat res;
        cv::Ptr<cv::DescriptorMatcher> matcher(new cv::BFMatcher(cv::NORM_L1));
        cv::Ptr<cv::DescriptorExtractor> mos = new Wld();
        cv::BOWImgDescriptorExtractor bowDE(mos, matcher);
        if (vocFile != "")
          {
            CsvReader reader;
            voc = reader(vocFile);
          }
        bowDE.setVocabulary(voc);
        auto fileName = files.at(i).originalFileName;
        auto className = files.at(i).originalDirName;
        if (featureDir != "")
          {
            files.at(i).setDirPrefix(featureDir);
          }
        cv::VideoCapture v(files.at(i).originalPathName);
        std::clog << fileName << std::endl;
        cv::Mat curFrame;
        int frameNum = 0;
        int count = 0;
        for (;;)
          {
            cv::Mat res;
            v >> curFrame;
            if (curFrame.empty())
              break;
            kp.resize(1);
            bowDE.compute(curFrame, kp, res);
            if (kp.size() == 0 || kp.size() != 0)
              {
                tbb::mutex::scoped_lock locker(mut);
                if (kp.size() != 0)
                  {
                    dump(res, files.at(i).pathName + "_frame_" + std::to_string(frameNum) + ".bow");
                    if (codedMat.count(className) == 0)
                      {
                        codedMat[className].create(0, res.cols, res.type());
                        resCols = res.cols;
                        resType = res.type();
                        classesName.push_back(className);
                      }
                    count++;
                    codedMat[className].push_back(res.clone());
                  }
                frameNum++;
              }
          }
        v.release();
        //std::cout << count << " " << className << std::endl;
      } );
    std::clog << "DONE" << std::endl;
  }

void Train::svmTraining(std::string featureDir)
{
  //Spec for 2 classes
  if (featureDir != "")
    {
      cv::Mat res;
      CsvReader reader;
      FileCrawler crawler;
      files = std::vector<FileInfo>();
      for (auto f : crawler.retrieveFiles(featureDir))
        {
          if (f.fileName.substr(f.fileName.length() - 3) == "bow")
            {
              files.push_back(f);
              auto className = f.originalDirName;
              res = reader(f.pathName);
              if (codedMat.count(className) == 0)
                {
                  codedMat[className].create(0, res.cols, res.type());
                  resCols = res.cols;
                  resType = res.type();
                  classesName.push_back(className);
                }
              codedMat[className].push_back(res.clone());
            }
        }
    }
  for (auto f : codedMat)
    std::clog << f.first << std::endl;
  tbb::parallel_for(size_t(0), classesName.size(),
                    [&](size_t i)
    {
      //"pas propre ballec"
      auto& n = classesName.at(i);
      std::clog << "=> START OF " << n << std::endl;
      cv::Mat samples(0, resCols, resType);
      cv::Mat labels(0, 1, CV_32SC1);
      samples.push_back(codedMat[n]);
      cv::Mat class_label = cv::Mat::ones(codedMat[n].rows, 1, CV_32SC1);
      labels.push_back(class_label);
      for (auto itCodeData = codedMat.begin(); itCodeData != codedMat.end(); itCodeData++)
        {
          if (n != itCodeData->first)
            {
              samples.push_back(codedMat[itCodeData->first]);
              class_label = cv::Mat::ones(codedMat[itCodeData->first].rows, 1, CV_32SC1) * (-1);
              labels.push_back(class_label);
            }
        }
      cv::Mat s32;
      samples.convertTo(s32, CV_32FC1);
      // for (int i = 0; i < s32.rows; i++)
      //   {
      //     cv::normalize(s32.row(i), s32.row(i));
      //   }
      cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();
      cv::ml::ParamGrid c_grid, gamma_grid, p_grid, nu_grid, coef_grid, degree_grid;
      c_grid = cv::ml::SVM::getDefaultGrid(cv::ml::SVM::C);
      gamma_grid = cv::ml::SVM::getDefaultGrid(cv::ml::SVM::GAMMA);
      p_grid = cv::ml::SVM::getDefaultGrid(cv::ml::SVM::P);
      p_grid.logStep = 1;
      nu_grid = cv::ml::SVM::getDefaultGrid(cv::ml::SVM::NU);
      nu_grid.logStep = 1;
      coef_grid = cv::ml::SVM::getDefaultGrid(cv::ml::SVM::COEF);
      coef_grid.logStep = 1;
      degree_grid = cv::ml::SVM::getDefaultGrid(cv::ml::SVM::DEGREE);
      degree_grid.logStep = 1;
      //svm->setType(cv::ml::SVM::EPS_SVR);
      svm->setType(cv::ml::SVM::NU_SVC);
      //svm->setType(cv::ml::SVM::NU_SVC);
      svm->setKernel(cv::ml::SVM::INTER);
      //svm->setKernel(cv::ml::SVM::POLY);
      // POLY/RBF
      //svm->setGamma(0.001);
      svm->setGamma(0.00001);
      // svm->setGamma(1);
      svm->setCoef0(1e4);
      //svm->setKernel(cv::ml::SVM::POLY);
      // POLY
      svm->setDegree(8.0f);
      //EPS/SVC
      svm->setC(10000.0f);
      svm->setP(0.01f);
      //svm->setNu(0.4f);
      svm->setNu(0.25f);
      std::clog << "TRAINING SVM OF " << n << std::endl;
      // svm->trainAuto(cv::ml::TrainData::create(s32, cv::ml::ROW_SAMPLE, labels), 10,
      //                c_grid, gamma_grid, p_grid, nu_grid, coef_grid, degree_grid);
      svm->train(s32, cv::ml::ROW_SAMPLE, labels);
      std::clog << "TRAINING SVM OF DONE " << n << std::endl;
      boost::filesystem::create_directory("SVM");
      boost::filesystem::create_directory("SVM/" + n);
      svm->save("SVM/" + n + "/svm.xml");
      std::clog << "DUMP OF " << n << std::endl;
      CsvDumper dump;
      dump(s32, "SVM/" + n + "/samples");
      dump(labels, "SVM/" + n + "/labels");
      std::clog << "DONE WITH " << n << "Supp Vector size " << svm->getSupportVectors().size()<< std::endl;
      std::clog << "nb sample was " << samples.rows<<std::endl;
    }
                    );
}
