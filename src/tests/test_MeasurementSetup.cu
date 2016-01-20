/** @file test_MeasurementSetup.cu */
#include "cuda.h"
#include "MeasurementSetup.hpp"
#include <iostream>
#include <string>
#include "H5Reader.hpp"

#define NA  180
#define N0Z 13
#define N0Y 13
#define N1Z 13
#define N1Y 13
#define NCHANNELS NA*N0Z*N0Y*N1Z*N1Y 

typedef float val_t;

int main(int ac, char ** av)
{
  /* Treat command line args */
  if(ac < 2)
  {
    std::cerr << "too few args" << std::endl;
    exit(EXIT_FAILURE);
  }
  std::string fn(av[1]);
  
  /* Read measurement data */
  H5Reader h5reader(fn);
  val_t * meas = new val_t[NCHANNELS];
  h5reader.read(meas);

  /* Create setup object */
  DefaultMeasurementSetup<val_t>
        setup(-1., 1.,
              NA, N0Z, N0Y, N1Z, N1Y,
              2, 0.1, 0.1, 0.1);

  /* Print data */
  for(int i=0; i<NCHANNELS; i++)
  {
    if(meas[i] != 0.)
    {
      int sepCnlId[5] = {0,0,0,0,0};
      setup.sepChannelId(sepCnlId, i);
      std::cout << "ida: "  << sepCnlId[0] << ", "
                << "id0z: " << sepCnlId[1] << ", "
                << "id0y: " << sepCnlId[2] << ", "
                << "id1z: " << sepCnlId[3] << ", "
                << "id1y: " << sepCnlId[4] << "    "
                << meas[i] << std::endl;
    }
  }
  std::cout << std::endl;

  //int const sepCnlId[] = {1, 0, 0, 0, 0};
  //int const linCnlId(setup.linChannelId(sepCnlId));
  //int sepCnlId_[5];
  //setup.sepChannelId(sepCnlId_, linCnlId);

  //std::cout << linCnlId << std::endl
  //          << sepCnlId_[0] << " "
  //          << sepCnlId_[1] << " "
  //          << sepCnlId_[2] << " "
  //          << sepCnlId_[3] << " "
  //          << sepCnlId_[4] << " "
  //          << std::endl;
  
  return 0;
}
