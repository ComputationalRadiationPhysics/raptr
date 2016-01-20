/** @file test_H5Reader.cpp */
#include "H5Reader.hpp"
#include <iostream>
#include <cstdlib>

#define FILESIZE 5140980

int main( int ac, char ** av )
{
  if(ac!=2)
  {
    std::cerr << "Wrong number of arguments.  Exspected 1: H5 filename."
              << std::endl;
    return EXIT_FAILURE;
  }

  std::string filename(av[1]);

  H5Reader reader(filename);

  float * mem = new float[FILESIZE];

  reader.read(mem);

//  int nonzero = 0;
//  for(int i=0; i<FILESIZE; i++)
//    if(mem[i]!=0.) nonzero++;
//  std::cout << "nonzero fraction: " << (float)(nonzero)/FILESIZE << std::endl;

  for(int i=0; i<FILESIZE; i++)
  {
    std::cout << mem[i] << ", ";
    if((i+1)%13==0) std::cout << std::endl;
  }


  return EXIT_SUCCESS;
}
