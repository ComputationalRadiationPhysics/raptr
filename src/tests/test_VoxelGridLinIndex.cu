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

