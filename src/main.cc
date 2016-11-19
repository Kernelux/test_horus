// NEED HUGE CLEAN IN INCLUDES ...
#include "csvDumper.hh"
#include "csvReader.hh"
#include "fileInfo.hh"
#include "motion.hh"
#include "mosiftExtractor.hh"
#include "opencv2/optflow.hpp"
#include "tbb/tbb.h"
#include "train.hh"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/video.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <sstream>
#include <vector>
#include "predict.hh"

int main(int ac, char* av[])
{
  if (ac == 1)
    {
      std::cerr << "Options: "<< std::endl;
      std::cerr << "      --train   directory" << std::endl;
      std::cerr << "      --predict directory" << std::endl;
      return 1;
    }
  // Parser option
  // OPTION
  // --train file
  // --predict fileofsvms filewithdata
  // Vocabulary of size
  FileCrawler crawler;
  std::vector<FileInfo> files = crawler.retrieveFiles(av[2]);
  if (strcmp(av[1],"--train-svm") == 0)
    {
      Train t(files, 600);
      t.svmTraining("features");
      return 0;
    }
  if (strcmp(av[1],"--train") == 0)
    {
      Train t(files, 600);
      t.calculateDescriptor("features", true);
      t.clusterBow();
      t.translateToBow();
      t.svmTraining();
      return 0;
    }
  //Predict
  if (strcmp(av[1],"--predict") == 0)
    {
      // Kind of remove if. Need overloading of stuff otherwise. Doing it later
      std::vector<FileInfo> svmFiles;
      for (auto& f : files)
        if (f.fileName == "svm.xml")
          svmFiles.push_back(f);
      Predict p(svmFiles);
      files = crawler.retrieveFiles(av[3]);
      p.predict(files);
    }
  return 0;
}
