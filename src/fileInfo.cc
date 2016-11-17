#include "fileInfo.hh"
#include <map>
#include <random>

FileInfo::FileInfo(boost::filesystem::path p)
  :originalPathName(p.string()),
   originalDirName(p.parent_path().filename().string()),
   originalFileName(p.filename().string()),
   originalPath(p)
  {
    pathName = p.string();
    fileName = p.filename().string();
    dirName =  p.parent_path().filename().string() + "/";
  }
void FileInfo::setDirPrefix(std::string prefix)
  {
    dirName = prefix + "/" + dirName;
    if (!boost::filesystem::exists(prefix))
      boost::filesystem::create_directory(prefix);
    if (!boost::filesystem::exists(dirName))
      boost::filesystem::create_directory(dirName);
    pathName = dirName + fileName;
  }

void FileInfo::renameFile(std::string newName)
  {
    fileName = newName;
    pathName = dirName + newName;
  }


std::vector<FileInfo> FileCrawler::retrieveFiles(std::string directoryName,
                                                 int percent,
                                                 bool shuffle)
{
  //std::vector<boost::filesystem::path> fileNames;
  std::vector<FileInfo> files;
  boost::filesystem::path current_dir(directoryName);
  std::map<std::string, std::vector<FileInfo>> m;
  if (boost::filesystem::exists(current_dir))
    {
      if (!boost::filesystem::is_directory(current_dir))
        {
          files.push_back(FileInfo(current_dir));
          return files;
        }
      for (boost::filesystem::recursive_directory_iterator iter(current_dir), end;
           iter != end; ++iter)
        {
          if(!boost::filesystem::is_directory(iter->path()))
            {
              auto fi = FileInfo(iter->path());
              if (m.count(fi.dirName) == 0)
                  m.insert({fi.dirName, std::vector<FileInfo>()});
              m.at(fi.dirName).push_back(fi);
            }
        }
    }
  if (percent > 100)
    percent = 100;
  else if (percent < 0)
    percent = 0;
  float size = 0;
  for (auto& vec : m)
    {
      size = vec.second.size() * percent / 100.0;
      if (shuffle)
        {
          //shuffle vector... Need information
          // on which video as been processed
        }
      for (int i = 0; i < size; i++)
        {
          files.push_back(vec.second.at(i));
        }
    }
  return files;
}
