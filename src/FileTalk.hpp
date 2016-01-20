/** @file FileTalk.hpp */
#include <iostream>
#include <fstream>
#include <string>

#define MAX_LINESIZE 160

class FileTalk
{
  private:
    
    std::ifstream _file;
    std::string   _filename;


  public:
    
    FileTalk( std::string const filename )
    : _file(("./"+filename).c_str()),
      _filename("./"+filename) {}

    void sayLine( int lineNumber )
    {
      std::cout << _filename << "(" << lineNumber << ") : ";
      char line[MAX_LINESIZE];

      _file.seekg(0);
      for(int i=0; i<lineNumber; i++)
        _file.getline(line, MAX_LINESIZE);

      std::cout << line << std::endl;
    }
};
#define SAYLINE( i ) { FileTalk(__FILE__).sayLine(i); }
//#define SAYLINE( i ) {}
#define SAYLINES( begin, end ) { for(int i=begin;i<end+1;i++) SAYLINE(i); }
