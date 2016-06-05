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

/** @file test_MeasurementSetupLinIndex.cu */
/* 
 * File:   test_MeasurementSetupLinIndex.cpp
 * Author: malte
 *
 * Created on 14.10.2014, 15:58:42
 */

#include <stdlib.h>
#include <iostream>
#include <cassert>
#include "MeasurementSetupLinIndex.hpp"
#include "measurementsetup_defines.h"

/*
 * Simple C++ Test Suite
 */

typedef double val_t;
typedef DefaultMeasurementSetup<val_t> MS;

void test1() {
  // Create setup
  MS setup(POS0X, POS1X,
           NA, N0Z, N0Y, N1Z, N1Y,
           DA, SEGX, SEGY, SEGZ);
  
  // Create functors
  DefaultMeasurementSetupLinId<MS> f_linId = DefaultMeasurementSetupLinId<MS>();
  DefaultMeasurementSetupId0z<MS>  f_id0z  = DefaultMeasurementSetupId0z<MS>();
  DefaultMeasurementSetupId0y<MS>  f_id0y  = DefaultMeasurementSetupId0y<MS>();
  DefaultMeasurementSetupId1z<MS>  f_id1z  = DefaultMeasurementSetupId1z<MS>();
  DefaultMeasurementSetupId1y<MS>  f_id1y  = DefaultMeasurementSetupId1y<MS>();
  DefaultMeasurementSetupIda<MS>   f_ida   = DefaultMeasurementSetupIda<MS>();
  
  // Test all indices
  for(int i=0; i<NA*N0Z*N0Y*N1Z*N1Y; i++) {
    int id0z  = f_id0z(i, &setup);
    int id0y  = f_id0y(i, &setup);
    int id1z  = f_id1z(i, &setup);
    int id1y  = f_id1y(i, &setup);
    int ida   = f_ida(i, &setup);
    int linId = f_linId(id0z, id0y, id1z, id1y, ida, &setup);
    
    assert(linId == i);
  }
}

int main(int argc, char** argv) {
  test1();
  std::cout << "test_MeasurementSetupLinIndex succeeded!" << std::endl;
  return (EXIT_SUCCESS);
}

