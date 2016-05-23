/**
 * Copyright 2016 Malte Zacharias
 *
 * This file is part of raptr.
 *
 * raptr is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * raptr is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with raptr.
 * If not, see <http://www.gnu.org/licenses/>.
 */

/** @file FileTalk.hpp
 * 
 *  @brief Auxiliary macro for printing code lines to stdout.
 */
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
