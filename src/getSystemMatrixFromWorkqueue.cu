/** @file getSystemMatrixFromWorkqueue.cu */
/* 
 * Author: malte
 *
 * Created on 21. Oktober 2014, 17:53
 */

#ifndef GETSYTEMMATRIXELEMENT_CU
#define	GETSYTEMMATRIXELEMENT_CU

#define DEBUG_MACRO ((defined DEBUG || defined GETSYSTEMMATRIXELEMENT_DEBUG)\
                    && (NO_GETSYSTEMMATRIXELEMENT_DEBUG==0))
#define PRINT_KERNEL 1

#include "VoxelGrid.hpp"
#include "VoxelGridLinIndex.hpp"
#include "MeasurementSetup.hpp"
#include "ChordsCalc_lowlevel.hpp"
#include <curand_kernel.h>
#include "device_constant_memory.hpp"

#define RANDOM_SEED 1251

template<
      typename T >
__host__ __device__
T intersectionLength(
      T const * const vxlCoord,
      T const * const rayCoord ) {
 
  // Which planes are intersected?  If non at all:  Return zero
  bool sects[3];
  for(int dim=0; dim<3; dim++)
    sects[dim] = rayCoord[dim] != rayCoord[3+dim];
  if(!(sects[0]||sects[1]||sects[2]))
    return 0;
  
  // Get min, max intersection parameters for each dim
  T aDimMin[3];
  T aDimMax[3];
  T temp;
  for(int dim=0; dim<3; dim++) {
    if(sects[dim]) {
      aDimMin[dim] = (vxlCoord[  dim]-rayCoord[dim])
                    /(rayCoord[3+dim]-rayCoord[dim]);
      aDimMax[dim] = (vxlCoord[3+dim]-rayCoord[dim])
                    /(rayCoord[3+dim]-rayCoord[dim]);
      if(aDimMax[dim]<aDimMin[dim]) {
        temp         = aDimMin[dim];
        aDimMin[dim] = aDimMax[dim];
        aDimMax[dim] = temp;
      }
    }
  }
  
  // Get entry and exit points
  T aMin, aMax;
  bool aMinGood, aMaxGood;
  MaxFunctor<3>()(&aMin, &aMinGood, aDimMin, sects);
  MinFunctor<3>()(&aMax, &aMaxGood, aDimMax, sects);
  
  // Really intersects?
  if(!(aMin<aMax) || (!aMinGood) || (!aMaxGood))
    return 0;
  
  return (aMax-aMin) * sqrt(  (rayCoord[3+0]-rayCoord[0]) * (rayCoord[3+0]-rayCoord[0])
                             +(rayCoord[3+1]-rayCoord[1]) * (rayCoord[3+1]-rayCoord[1])
                             +(rayCoord[3+2]-rayCoord[2]) * (rayCoord[3+2]-rayCoord[2]) );
}

/**
 * Kernel function. Calculates system matrix.
 * @param globalCnl Array of system matrix channel ids (input). In device
 * global memory. Has wqLength elements.
 * @param globalVxl Array of system matrix voxel ids (input). In device global
 * memory. Has wqLength elements.
 * @param globalVal Array for resulting system matrix elements. In device
 * global memory.
 * @param wqLength Number of entries in the workqueue. In device global memory.
 * @param nrays Number of rays to use in one channel. In device global memory.
 */
template<
      typename T
    , typename ConcreteVG
    , typename ConcreteVGIdx
    , typename ConcreteVGIdy
    , typename ConcreteVGIdz
    , typename ConcreteMS
    , typename ConcreteMSid0z
    , typename ConcreteMSid0y
    , typename ConcreteMSid1z
    , typename ConcreteMSid1y
    , typename ConcreteMSida  
    , typename ConcreteMSTrafo2CartCoordFirstPixel
    , typename ConcreteMSTrafo2CartCoordSecndPixel >
__global__
void getSystemMatrixFromWorkqueue(
      int const * const globalCnl,
      int const * const globalVxl,
      T * const  globalVal,
      int const wqLength,
      int const nrays) {
  // Global id of thread
  int const globalId = threadIdx.x + blockDim.x * blockIdx.x;
  int const globalSize = gridDim.x * blockDim.x;
  
  int wqId = globalId;
  while(wqId<wqLength) {
    if(wqId==(wqLength-1)){
      printf(
            "Thread %i(/%i) got last workqueue entry: %i (workqueue length: %i)\n",
            globalId, globalSize, wqId, wqLength);
    }
//#if DEBUG_MACRO
#if 0
  //if((threadIdx.x==(blockDim.x-1)) && (blockIdx.x==(gridDim.x-1))) {
  if(true) {
    printf("wqId: %i, wqLength: %i\n", wqId, wqLength);
  }
#endif

#if DEBUG_MACRO
  //#if 0
    if(globalId==PRINT_KERNEL) {
      printf(
            "setup segx, segy, segz: %f, %f, %f\n",
            setup_const.segx(), setup_const.segy(), setup_const.segz());
    }
  #endif

    // Create functors
    ConcreteVGIdx  f_idx;
    ConcreteVGIdy  f_idy;
    ConcreteVGIdz  f_idz;
    ConcreteMSid0z f_id0z;
    ConcreteMSid0y f_id0y;
    ConcreteMSid1z f_id1z;
    ConcreteMSid1y f_id1y;
    ConcreteMSida  f_ida;
    ConcreteMSTrafo2CartCoordFirstPixel trafo0;
    ConcreteMSTrafo2CartCoordSecndPixel trafo1;

    // Copy from workqueue to register
    int cnl(0);
    int vxl(0);
    __syncthreads();
    if(wqId<wqLength) {
      cnl = globalCnl[wqId];
      vxl = globalVxl[wqId];
    }
//#if DEBUG_MACRO
#if 0
  if((globalId%1000)==0)
    printf(
          "Kernel %i has cnl: %i\n",
          globalId, cnl);
#endif


    // Calculate voxel coordinates
    T vxlCoord[6];
    int sepVxlId[3];
    sepVxlId[0] = f_idx(vxl, &grid_const);
    sepVxlId[1] = f_idy(vxl, &grid_const);
    sepVxlId[2] = f_idz(vxl, &grid_const);
//#if DEBUG_MACRO
#if 0
  if((globalId%1000)==0)
    printf(
          "Kernel %i has sepVxlId: %i, %i, %i\n",
          globalId, sepVxlId[0], sepVxlId[1], sepVxlId[2]);
#endif
    vxlCoord[0] = grid_const.gridox() +  sepVxlId[0]   *(grid_const.griddx());
    vxlCoord[1] = grid_const.gridoy() +  sepVxlId[1]   *(grid_const.griddy());
    vxlCoord[2] = grid_const.gridoz() +  sepVxlId[2]   *(grid_const.griddz());
    vxlCoord[3] = grid_const.gridox() + (sepVxlId[0]+1)*(grid_const.griddx());
    vxlCoord[4] = grid_const.gridoy() + (sepVxlId[1]+1)*(grid_const.griddy());
    vxlCoord[5] = grid_const.gridoz() + (sepVxlId[2]+1)*(grid_const.griddz());
//#if DEBUG_MACRO
#if 0
  if((globalId%1000)==0)
    printf(
          "Kernel %i has vxlCoord: %f, %f, %f, %f, %f, %f\n",
          globalId, vxlCoord[0], vxlCoord[1], vxlCoord[2], vxlCoord[3],
          vxlCoord[4], vxlCoord[5]);
#endif

    // Initialize random number generator
    curandState rndState;
    curand_init(RANDOM_SEED, cnl, 0, &rndState);

    // Matrix element
    T a(0);

    // For rays ...
    for(int idRay=0; idRay<nrays; idRay++) {
      // ... Get 6 randoms for ray ...
      T rnd[6];
      rnd[0] = curand_uniform(&rndState);
      rnd[1] = curand_uniform(&rndState);
      rnd[2] = curand_uniform(&rndState);
      rnd[3] = curand_uniform(&rndState);
      rnd[4] = curand_uniform(&rndState);
      rnd[5] = curand_uniform(&rndState);
//#if DEBUG_MACRO
#if 0
    if((globalId%1000)==0) {
      printf("Kernel %i has rnd: %f, %f, %f, %f, %f, %f\n",
            globalId, rnd[0], rnd[1], rnd[2], rnd[3], rnd[4], rnd[5]);
    }
#endif

      // ... Calculate ray coordinates ...
      T rayCoord[6];
      int id0z = f_id0z(cnl, &setup_const);
      int id0y = f_id0y(cnl, &setup_const);
      int id1z = f_id1z(cnl, &setup_const);
      int id1y = f_id1y(cnl, &setup_const);
      int ida  = f_ida(cnl, &setup_const);
//#if DEBUG_MACRO
#if 0
    if((globalId%1000)==0) {
      printf("Kernel %i has id0z, id0y, id1z, id1y, ida: %i, %i, %i, %i, %i\n",
            globalId, id0z, id0y, id1z, id1y, ida);
    }
#endif

      trafo0(&rayCoord[0], &rnd[0], id0z, id0y, ida, &setup_const);
      trafo1(&rayCoord[3], &rnd[3], id1z, id1y, ida, &setup_const);
//#if DEBUG_MACRO
#if 0
    if((globalId%1000)==0) {
      printf("Kernel %i has rayCoord: %f, %f, %f, %f, %f, %f\n",
            globalId, rayCoord[0], rayCoord[1], rayCoord[2], rayCoord[3],
            rayCoord[4], rayCoord[5]);
    }
#endif

      // ... Calculate & add intersection length
      a += intersectionLength(vxlCoord, rayCoord);
    }

    // Divide by number of rays
    a /= nrays;

//#if DEBUG_MACRO
#if 0
  if(a!=0) {
    printf("Kernel %i got a=%f\n", globalId, a);
  }
#endif

    // Write matrix element back
    __syncthreads();
    if(wqId<wqLength) {
      globalVal[wqId] = a;
    }

    wqId += globalSize;
  }
}

#endif	/* GETSYTEMMATRIXELEMENT_CU */

