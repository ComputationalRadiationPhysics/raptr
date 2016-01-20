/** @file test_getSystemMatrixDeviceOnly.cu */
/* Author: malte
 *
 * Created on 16. Januar 2015, 15:20 */

#include <cstdlib>
#include <iostream>
#include <fstream>

#define NBLOCKS 32

#include "wrappers.hpp"
#include "getSystemMatrixDeviceOnly.cu"
#include "real_measurementsetup_defines.h"
#include "voxelgrid_defines.h"
#include "CUDA_HandleError.hpp"
#include "typedefs.hpp"
#include "device_constant_memory.hpp"

typedef int GridSizeType;


int main(int argc, char** argv) {
  int const nargs(3);
  if(argc!=nargs+1) {
    std::cerr << "Error: Wrong number of arguments. Exspected: "
              << nargs << ":" << std::endl
              << "  filename of measurement" << std::endl
              << "  filename of output" << std::endl
              << "  number of rays" << std::endl;
    exit(EXIT_FAILURE);
  }
  std::string const fn(argv[1]);
  std::string const on(argv[2]);
  
  
  int const nrays(atoi(argv[3]));
  HANDLE_ERROR(cudaMemcpyToSymbol(nrays_const, &nrays, sizeof(int)));

  
  MS setup = MS(POS0X, POS1X, NA, N0Z, N0Y, N1Z, N1Y, DA, SEGX, SEGY, SEGZ);
  HANDLE_ERROR(cudaMemcpyToSymbol(setup_const, &setup, sizeof(MS)));
  
  
  VG grid = VG(GRIDOX, GRIDOY, GRIDOZ, GRIDDX, GRIDDY, GRIDDZ, GRIDNX, GRIDNY, GRIDNZ);
  HANDLE_ERROR(cudaMemcpyToSymbol(grid_const, &grid, sizeof(grid)));
  
  
  ListSizeType mlSize_host[1];
  int * ml_host = NULL;
  {
    int tmp(0);
    readHDF5_MeasList<float>(ml_host, tmp, fn);
    mlSize_host[0] = ListSizeType(tmp);
  }
  
  
  ListSizeType * mlSize_devi = NULL;
  int * ml_devi = NULL;
  mallocD_MeasList(ml_devi, mlSize_host[0]);
  cpyH2DAsync_MeasList(ml_devi, ml_host, mlSize_host[0]);
  
  
  MemArrSizeType const memSize = MemArrSizeType(mlSize_host[0]) * MemArrSizeType(VGRIDSIZE);
  int * cnlId_devi = NULL;
  GridSizeType * vxlId_devi = NULL;
  val_t * sme_devi = NULL;
  MemArrSizeType truckDest_host[1] = {0};
  MemArrSizeType * truckDest_devi = NULL;
  mallocD(cnlId_devi, memSize);
  mallocD(vxlId_devi, memSize);
  mallocD(sme_devi, memSize);
  mallocD(truckDest_devi, 1);
  
  memcpyH2D(truckDest_devi, truckDest_host, 1);
  
  getSystemMatrix<
        val_t, VG, Idx, Idy, Idz, MS, Id0z, Id0y, Id1z, Id1y, Ida,
        Trafo0_inplace, Trafo1_inplace, ListSizeType, GridSizeType, MemArrSizeType>
        <<<NBLOCKS, TPB>>>
      ( sme_devi, vxlId_devi, cnlId_devi, ml_devi, mlSize_devi, truckDest_devi);
  HANDLE_ERROR(cudaDeviceSynchronize());
  
  memcpyD2H(truckDest_host, truckDest_devi, 1);
  
  std::vector<val_t> sme_host(truckDest_host[0], 0.);
  memcpyD2H(&(*sme_host.begin()), sme_devi, truckDest_host[0]);
  std::stable_sort(sme_host.begin(), sme_host.end());
  
  
  val_t sum(0);
  for(MemArrSizeType i=0; i<truckDest_host[0]; i++) { sum += sme_host[i]; }
  
  std::ofstream out(on.c_str());
  if(!out) {
    std::cerr << __FILE__ << "(" << __LINE__ << "): Error: Could not open "
              << on << " for writing." << std::endl;
    exit(EXIT_FAILURE);
  }
  out << sum;
  
  
  if(sum != 0.) exit(EXIT_SUCCESS);
  exit(EXIT_FAILURE);
}

