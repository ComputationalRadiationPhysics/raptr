/** @file test_getWorkqueue.cu */
/* 
 * File:   test_getWorkqueue.cpp
 * Author: malte
 *
 * Created on 9. Oktober 2014, 17:18
 */

#include "getWorkqueue.hpp"
#include "VoxelGrid.hpp"
#include "MeasurementSetup.hpp"
#include "MeasurementSetupLinIndex.hpp"
#include "MeasurementSetupTrafo2CartCoord.hpp"
#include "H5File2DefaultMeasurementList.h"
#include "real_measurementsetup_defines.h"
#include "voxelgrid_defines.h"
#include <iostream>
#include <cassert>

#include "typedefs.hpp"

void test1(std::string const fn) {
  std::cout << std::endl
            << "-----Test1-----"
            << std::endl;
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
  
  int wqCnlId; int wqVxlId;
  int listId(0); int vxlId(0);
  int found;
  int nFound(0);
  
  do {
    found = getWorkqueueEntry<
      val_t,
      ML,
      VG, Idx, Idy, Idz,
      MS, Id0z, Id0y, Id1z, Id1y, Ida,
      Trafo0, Trafo1> (
        &wqCnlId, &wqVxlId, listId, vxlId, &list, &grid, &setup);
    nFound += found;
//    std::cout << "wqCnlId: " << wqCnlId
//              << ", wqVxlId: " << wqVxlId
//              << ", listId: " << listId << std::endl;
  } while(found == 1);
    std::cout << "Looking for all workqueue entries found: " << nFound
              << std::endl;
}

void test2( std::string fn, int const n ) {
  std::cout << std::endl
            << "-----Test2-----"
            << std::endl;
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
                 Trafo0, Trafo1>
               (
                 wqCnlId, wqVxlId, listId, vxlId, &list, &grid, &setup, n);
  std::cout << "Found " << nFound << " workqueue entries" << std::endl;
  std::cout << "Workqueue:" << std::endl
            << "----------" << std::endl;
  for(int i=0; i<nFound; i++) {
    std::cout << "  cnlId: " << wqCnlId[i] << ",   vxlId: " << wqVxlId[i]
              << std::endl;
  }
}

void test3( std::string const fn, int const n ) {
  std::cout << std::endl
            << "-----Test3-----"
            << std::endl;
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
  int nFound(0);
  nFound = getWorkqueue<
                 val_t,
                 ML,
                 VG, Idx, Idy, Idz,
                 MS, Id0z, Id0y, Id1z, Id1y, Ida,
                 Trafo0, Trafo1>
               (
                 wqCnlId, wqVxlId, listId, vxlId, &list, &grid, &setup);
  
  std::vector<int> wqCnlId2;
  std::vector<int> wqVxlId2;
  int listId2(0); int vxlId2(0);
  int nFound2(0);
  nFound2 = getWorkqueue<
                 val_t,
                 ML,
                 VG, Idx, Idy, Idz,
                 MS, Id0z, Id0y, Id1z, Id1y, Ida,
                 Trafo0, Trafo1>
               (
                 wqCnlId2, wqVxlId2, listId2, vxlId2, &list, &grid, &setup,
                 n);
  
  std::cout << "Full search found " << nFound << " workqueue entries,"
            << std::endl
            << "search for " << n << " workqueue entries found " << nFound2
            << std::endl;
  std::cout << "Workqueue (full | limited):" << std::endl
            << "---------------------------" << std::endl;
  std::vector<int>::iterator wqCnlIdIt  = wqCnlId.begin();
  std::vector<int>::iterator wqVxlIdIt  = wqVxlId.begin();
  std::vector<int>::iterator wqCnlId2It = wqCnlId2.begin();
  std::vector<int>::iterator wqVxlId2It = wqVxlId2.begin();
  while(true) {
    if(wqCnlIdIt != wqCnlId.end()) {
      std::cout << "  cnlId: " << *wqCnlIdIt;
      wqCnlIdIt++;
    }
    if(wqVxlIdIt != wqVxlId.end()) {
      std::cout << "    vxlId: " << *wqVxlIdIt;
      wqVxlIdIt++;
    }
    if(wqCnlId2It != wqCnlId2.end()) {
      std::cout << "  |  cnlId: " << *wqCnlId2It;
      wqCnlId2It++;
    }
    if(wqVxlId2It != wqVxlId2.end()) {
      std::cout << "    vxlId: " << *wqVxlId2It;
      wqVxlId2It++;
    }
    std::cout << std::endl;
    
    if(   (wqCnlIdIt  == wqCnlId.end())
       && (wqVxlIdIt  == wqVxlId.end())
       && (wqCnlId2It == wqCnlId2.end())
       && (wqVxlId2It == wqVxlId2.end()))
      break;
  }
  
  wqCnlIdIt  = wqCnlId.begin();
  wqVxlIdIt  = wqVxlId.begin();
  wqCnlId2It = wqCnlId2.begin();
  wqVxlId2It = wqVxlId2.begin();
  while(   (wqCnlId2It != wqCnlId2.end())
        && (wqVxlId2It != wqVxlId2.end())) {
    assert((*wqCnlIdIt) == *(wqCnlId2It));
    assert((*wqVxlIdIt) == *(wqVxlId2It));
    wqCnlIdIt++;
    wqVxlIdIt++;
    wqCnlId2It++;
    wqVxlId2It++;
  }
}

int main(int argc, char ** argv) {
  int const nargs(2);
  if(argc!=nargs+1) {
    std::cerr << "Error: Wrong number of arguments. Exspected: "
              << nargs << ":" << std::endl
              << "  filename of measurement" << std::endl
              << "  number of workqueue entries to look for" << std::endl;
    exit(EXIT_FAILURE);
  }
  std::string const fn(argv[1]);
  int const         n(atoi(argv[2]));
  
  test1(fn);
  test2(fn, n);
  test3(fn, n);
  
  return 0;
}

