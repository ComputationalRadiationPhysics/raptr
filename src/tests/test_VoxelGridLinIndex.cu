/** @file test_VoxelGridLinIndex.cu */
/* 
 * File:   test_VoxelGridLinIndex.cu
 * Author: malte
 *
 * Created on 24.10.2014, 13:11:02
 */

#include <stdlib.h>
#include <iostream>
#include <cassert>
#include "VoxelGrid.hpp"
#include "VoxelGridLinIndex.hpp"
#include "voxelgrid_defines.h"

/*
 * Simple C++ Test Suite
 */

typedef float val_t;
typedef DefaultVoxelGrid<val_t> VG;

void test1() {
  // Create voxel grid
  VG vg(
        GRIDOX, GRIDOY, GRIDOZ,
        GRIDDX, GRIDDY, GRIDDZ,
        GRIDNX, GRIDNY, GRIDNZ);
  
  // Create functors
  DefaultVoxelGridLinId<VG> f_linId = DefaultVoxelGridLinId<VG>();
  DefaultVoxelGridIdx<VG>   f_Idx;
  DefaultVoxelGridIdy<VG>   f_Idy;
  DefaultVoxelGridIdz<VG>   f_Idz;
  
  // Test all indices
  for(int i=0; i<VGRIDSIZE; i++) {
    int idx   = f_Idx(i, &vg);
    int idy   = f_Idy(i, &vg);
    int idz   = f_Idz(i, &vg);
    int linId = f_linId(idx, idy, idz, &vg);
    
    assert(linId == i);
  }
}

int main(int argc, char** argv) {
  test1();
  std::cout << "test_VoxelGridLinId succeeded!" << std::endl;
  return (EXIT_SUCCESS);
}

