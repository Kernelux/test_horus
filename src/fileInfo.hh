#pragma once
#include <boost/filesystem.hpp>
#include <vector>

struct FileInfo
{
  FileInfo(boost::filesystem::path p);
  void setDirPrefix(std::string prefix);
  void renameFile(std::string newName);
  const std::string originalPathName;
  const std::string originalDirName;
  const std::string originalFileName;
  const boost::filesystem::path originalPath;

  std::string pathName;
  std::string dirName;
  std::string fileName;

  std::string prefix = "";

};

class FileCrawler
{
public:
  FileCrawler() = default;
  std::vector<FileInfo> retrieveFiles(std::string directoryName, int percent = 100, bool shuffle=false);
};
