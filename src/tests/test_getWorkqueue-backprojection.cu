/** @file test_getWorkqueue-backprojection.cu */
/* 
 * File:   test_getWorkqueue-backprojection.cu
 * Author: malte
 *
 * Created on 20. Oktober 2014, 18:34
 */

#include "getWorkqueue.hpp"
#include "VoxelGrid.hpp"
#include "MeasurementSetup.hpp"
#include "MeasurementSetupLinIndex.hpp"
#include "MeasurementSetupTrafo2CartCoord.hpp"
#include "H5File2DefaultMeasurementList.h"
#include "H5DensityWriter.hpp"
#include "real_measurementsetup_defines.h"
#include "voxelgrid_defines.h"
#include <iostream>

#include "typedefs.hpp"

template<typename T>
class GridAdapter {
public:
  GridAdapter(VG * grid) {
    _grid = grid;
  }
  
  void getOrigin( T * const origin ) const {
    origin[0] = _grid->gridox();
    origin[1] = _grid->gridoy();
    origin[2] = _grid->gridoz();
  }
  
  void getVoxelSize( T * const voxelSize ) const {
    voxelSize[0] = _grid->griddx();
    voxelSize[1] = _grid->griddy();
    voxelSize[2] = _grid->griddz();
  }
  
  void getNumberOfVoxels( int * const number ) const {
    number[0] = _grid->gridnx();
    number[1] = _grid->gridny();
    number[2] = _grid->gridnz();
  }
  
private:
  VG * _grid;
};

void test2( std::string const fn, std::string const on ) {
  VG grid =
    VG(
      GRIDOX, GRIDOY, GRIDOZ,
      GRIDDX, GRIDDY, GRIDDZ,
      GRIDNX, GRIDNY, GRIDNZ);
    
  MS setup =
    MS(
      POS0X, POS1X,
      NA, N0Z, N0Y, N1Z, N1Y,
      DA, SEGX, SEGY, SEGZ);
  
  ML list =
    H5File2DefaultMeasurementList<val_t>(fn, NA*N0Z*N0Y*N1Z*N1Y);
  
  std::vector<int> wqCnlId;
  std::vector<int> wqVxlId;
  int listId(0); int vxlId(0);
  int nFound;
  
  nFound = getWorkqueue<
        val_t,
        ML, 
        VG, Idx, Idy, Idz,
        MS, Id0z, Id0y, Id1z, Id1y, Ida,
        Trafo0, Trafo1> (
        wqCnlId, wqVxlId, listId, vxlId, &list, &grid, &setup);
  
  // Create grid memory for backprojection
  int const gridsize(grid.gridnx()*grid.gridny()*grid.gridnz());
  val_t * mem = new val_t[gridsize];
  for(int vxlId=0; vxlId<gridsize; vxlId++) {
    mem[vxlId] = 0.;
  }
  
  // Backproject workqueue on grid
  for(int wqId=0; wqId<nFound; wqId++) {
    int vxlId = wqVxlId[wqId];
    mem[vxlId] += 1.0;
  }
  
  H5DensityWriter<GridAdapter<val_t> > writer(on);
  GridAdapter<val_t> ga(&grid);
  writer.write(mem, ga);
}

int main(int argc, char** argv) {
  int const nargs(2);
  if(argc!=nargs+1) {
    std::cerr << "Error: Wrong number of arguments. Exspected: "
              << nargs << ":" << std::endl
              << "  filename of measurement" << std::endl
              << "  filename of output" << std::endl;
    exit(EXIT_FAILURE);
  }
  std::string const fn(argv[1]);
  std::string const on(argv[2]);

  test2(fn, on);
  
  return 0;
}