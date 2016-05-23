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

/** @file getSystemMatrixDeviceOnly.cu
 * 
 *  @brief Header file that defines the cuda kernel function that, given a
 *  measurement list, calculates the system matrix for that measurement list.
 */
#ifndef GETSYSTEMMATRIXDEVICEONLY_CU
#define GETSYSTEMMATRIXDEVICEONLY_CU

#include "VoxelGrid.hpp"
#include "VoxelGridLinIndex.hpp"
#include "MeasurementSetup.hpp"
#include "ChordsCalc_lowlevel.hpp"
#include <curand_kernel.h>
#include "device_constant_memory.hpp"

#include <cmath>
#include <vector>
#include "VoxelGrid.hpp"
#include "MeasurementSetup.hpp"
#include "MeasurementList.hpp"
#include "distancePointLine.h"

#define RANDOM_SEED 1251
#define TPB 256

/** @fn test
 * @brief Test one voxel channel combination: Might the corresponding SM
 * element have a non-zero value.
 * 
 * Having a non-zero value is equivalent to the
 * statement that annihilations in the voxel might result in an event in the
 * channel or in different words that the channel sees the voxel. This test
 * function performs an overestimation in the sense that negativly tested SM
 * elements are definitely zero but positively tested elements might still be
 * zero. The channel volume is overestimated by an enclosing cylinder.
 * 
 * @param cnlId Linear id of the channel.
 * @param vxlId Linear id of the voxel. 
 * @result false: SM element is definitely zero. true: SM element might be
 * different from zero
 */
template<
      typename T
    , typename VG
    , typename VGidx
    , typename VGidy
    , typename VGidz
    , typename MS
    , typename MSid0z
    , typename MSid0y
    , typename MSid1z
    , typename MSid1y
    , typename MSida
    , typename MSTrafo0_inplace
    , typename MSTrafo1_inplace
    , typename GridSizeType >
__device__
bool test(
      int const & cnlId, GridSizeType const & vxlId) {
  
  /* Create multidim index functors for grid and setup */
  int const idx  = VGidx()( vxlId, &grid_const);
  int const idy  = VGidy()( vxlId, &grid_const);
  int const idz  = VGidz()( vxlId, &grid_const);
  
  int const id0z = MSid0z()(cnlId, &setup_const);
  int const id0y = MSid0y()(cnlId, &setup_const);
  int const id1z = MSid1z()(cnlId, &setup_const);
  int const id1y = MSid1y()(cnlId, &setup_const);
  int const ida  = MSida()( cnlId, &setup_const);

  /* Sum of radii of voxel and detector pixel. This sum is the minimum distance
   * the voxel center must have from the connecting line between the pixel
   * centers for the test result to be negative. */
  T vxlEdges[3] = {grid_const.griddx(), grid_const.griddy(), grid_const.griddz()};
  T pxlEdges[3] = {setup_const.segx(), setup_const.segy(), setup_const.segz()};
  T sumRadii = T(0.5)*(absolute(vxlEdges)+absolute(pxlEdges));
  
  /* Get channel pixel centers, voxel center */
  T pix0Center[3] = {T(0.5), T(0.5), T(0.5)};
  MSTrafo0_inplace()(pix0Center, id0z, id0y, ida, &setup_const);

  T pix1Center[3] = {T(0.5), T(0.5), T(0.5)};
  MSTrafo1_inplace()(pix1Center, id1z, id1y, ida, &setup_const);

  T vxlCenter[3];
  vxlCenter[0] = grid_const.gridox() + (idx+T(0.5))*grid_const.griddx();
  vxlCenter[1] = grid_const.gridoy() + (idy+T(0.5))*grid_const.griddy();
  vxlCenter[2] = grid_const.gridoz() + (idz+T(0.5))*grid_const.griddz();

  /* Compare actual distance between the voxel center and the connecting line
   * of the pixel centers with with the minimum distance for negative result. */
  if(distance(pix0Center, pix1Center, vxlCenter)<sumRadii) {
    return true;
  }
  return false;
}



/** @brief Calculate the intersection length of a line with a box.
 * 
 * @param vxlCoord Coordinates of the box (voxel): 0-2: Cartesian coordinates
 * of one corner. 3-5: Cartesian coordinates of diagonally opposing corner.
 * Grid parallel edges.
 * @param rayCoord Coordinates of the line (ray): 0-2: Cartesian coordinates
 * of start point. 3-5: Cartesian coordinates of end point.
 * @return Intersection length. */
template<
      typename T >
__host__ __device__
T intersectionLength(
      T const * const vxlCoord,
      T const * const rayCoord ) {
 
  /* Which planes are intersected?  If non at all:  Return zero */
  bool sects[3];
  for(int dim=0; dim<3; dim++)
    sects[dim] = rayCoord[dim] != rayCoord[3+dim];
  if(!(sects[0]||sects[1]||sects[2]))
    return 0;
  
  /* Get min, max intersection parameters for each dim */
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
  
  /* Get line parameters for entry into and exit from the box */
  T aMin, aMax;
  bool aMinGood, aMaxGood;
  MaxFunctor<3>()(&aMin, &aMinGood, aDimMin, sects);
  MinFunctor<3>()(&aMax, &aMaxGood, aDimMax, sects);
  
  /* Do they really intersect? */
  if(!(aMin<aMax) || (!aMinGood) || (!aMaxGood))
    return 0;
  
  return (aMax-aMin) * sqrt(  (rayCoord[3+0]-rayCoord[0]) * (rayCoord[3+0]-rayCoord[0])
                             +(rayCoord[3+1]-rayCoord[1]) * (rayCoord[3+1]-rayCoord[1])
                             +(rayCoord[3+2]-rayCoord[2]) * (rayCoord[3+2]-rayCoord[2]) );
}



/** @brief Calculate one SM element given by its corresponding channel and
 * voxel. The channel is sampled by rays and the SM element value is calculated
 * as the average intersection length between the voxel and those rays.
 * 
 * @param cnl Linear id of the channel.
 * @param vxl Linear id of the voxel.
 * @return System matrix element. */
template<
      typename T
    , typename VG
    , typename VGIdx
    , typename VGIdy
    , typename VGIdz
    , typename MS
    , typename MSId0z
    , typename MSId0y
    , typename MSId1z
    , typename MSId1y
    , typename MSIda  
    , typename MSTrafo0_inplace
    , typename MSTrafo1_inplace
    , typename RayGen
    , typename GridSizeType >
__device__
T calcSme(
      int const cnl,
      GridSizeType const vxl,
      RayGen & rayGen) {

  /* Get voxel coordinates */
  T vxlCoord[6];
  int sepVxlId[3];
  sepVxlId[0] = VGIdx()(vxl, &grid_const);
  sepVxlId[1] = VGIdy()(vxl, &grid_const);
  sepVxlId[2] = VGIdz()(vxl, &grid_const);

  vxlCoord[0] = grid_const.gridox() +  sepVxlId[0]   *(grid_const.griddx());
  vxlCoord[1] = grid_const.gridoy() +  sepVxlId[1]   *(grid_const.griddy());
  vxlCoord[2] = grid_const.gridoz() +  sepVxlId[2]   *(grid_const.griddz());
  vxlCoord[3] = grid_const.gridox() + (sepVxlId[0]+1)*(grid_const.griddx());
  vxlCoord[4] = grid_const.gridoy() + (sepVxlId[1]+1)*(grid_const.griddy());
  vxlCoord[5] = grid_const.gridoz() + (sepVxlId[2]+1)*(grid_const.griddz());
  
  /* Get channel indices */
  int const id0z = MSId0z()(cnl, &setup_const);
  int const id0y = MSId0y()(cnl, &setup_const);
  int const id1z = MSId1z()(cnl, &setup_const);
  int const id1y = MSId1y()(cnl, &setup_const);
  int const ida  = MSIda()( cnl, &setup_const);
  
  /* Get functors for geometric transformation of positions within pixels */
  MSTrafo0_inplace  trafo0;
  MSTrafo1_inplace  trafo1;
  
  /* Initialize matrix element */
  T a(0.);

  /* Add up intersection lengths over all rays */
  for(int idRay=0; idRay<nrays_const; idRay++) {
    T ray[6];
    rayGen(ray, id0z, id0y, id1z, id1y, ida, trafo0, trafo1);
    a += intersectionLength(vxlCoord, ray);
  }

  /* Divide by number of rays */
  a /= nrays_const;

  return a;
}



/** @brief Functor. Calculates the linear voxel id from the getId. */
template< typename VG >
struct GetId2vxlId {
  int const _gridDim;
  
  /** Ctor. */
  __host__ __device__
  GetId2vxlId( VG & vg )
  : _gridDim(vg.gridnx()*vg.gridny()*vg.gridnz()) {}
  
  __host__ __device__
  int operator()( int const getId ) const {
    return getId % _gridDim;
  }
};



/** @brief Functor. Calculates the list id from the getId. */
template< typename VG >
struct GetId2listId {
  int const _gridDim;
  
  /** Ctor. */
  __host__ __device__
  GetId2listId( VG & vg )
  : _gridDim(vg.gridnx()*vg.gridny()*vg.gridnz()) {}

  __host__ __device__
  int operator()( int const getId ) const {
    return getId / _gridDim;
  }  
};



/** @brief Functor. Tests if getId is in the range for another getting loop. */
template<
      typename ConcreteVG
    , typename ListSizeType=uint32_t
    , typename GridSizeType=uint32_t
    , typename MemArrSizeType=uint64_t >
struct IsInRange {
  GridSizeType const _gridDim;
  ListSizeType const _listSize;
  
  /** Ctor.
   * @param vg Voxel grid.
   * @param mlSize Measurement list size. */
  __device__
  IsInRange( VG & vg, ListSizeType const mlSize )
  : _gridDim(vg.gridnx()*vg.gridny()*vg.gridnz()),
    _listSize(mlSize) {}
  
  __device__
  bool operator()( MemArrSizeType const getId ) const {
    return getId < MemArrSizeType((_gridDim * _listSize) + blockDim.x -1);
  }
};



/** @brief Functor. Tests if (vxlId, listId) refers to a valid system matrix
 * entry. */
template<
      typename ConcreteVG
    , typename GridSizeType=uint32_t
    , typename ListSizeType=uint32_t >
struct GetIsLegal {
  GridSizeType const _gridDim;
  ListSizeType const _listSize;
  
  /** Ctor.
   * @param vg Voxel grid.
   * @param mlSize measurement list size. */
  __host__ __device__
  GetIsLegal( VG vg, ListSizeType const mlSize )
  : _gridDim(vg.gridnx()*vg.gridny()*vg.gridnz()),
    _listSize(mlSize) {}
  
  __host__ __device__
  bool operator()( GridSizeType const vxlId, ListSizeType const listId ) const {
    return (listId<_listSize) && (vxlId<_gridDim);
  }
};


/** @brief Kernel function. Calculates system matrix for a given measurement
 * list. The whole range of system matrix elements (SMEs) is equally distributed
 * among all threads with no spatial ordering. Threads loop over their subrange
 * testing one element per loop pass for possibility of a non-zero value.
 * Positively tested SMEs are accumulated in shared memory. When there are at
 * least as many SMEs in shared mem as there are threads in one block, all
 * threads calculate the actual value of one of those SMEs in shared mem and
 * writes the result to global mem. This flushing takes place within the loop
 * cycles. When the loop is finished one last flush is performed if necessary.
 * 
 * @param sme_devi Array for resulting system matrix elements. In device
 * global memory.
 * @param vxlId_devi Array for resulting system matrix voxel ids. In device
 * global memory.
 * @param cnlId_devi Array for resulting system matrix channel ids. In device
 * global memory.
 * @param ml_devi Array: Measurement list. In device global memory.
 * @param mlSize_devi Measurement list size. In device global memory.
 * @param truckDest_devi Position where to write next system matrix entries. Is
 * equal to the number of entries written so far. */
template<
      typename T
    , typename ConcreteVG
    , typename ConcreteVGidx
    , typename ConcreteVGidy
    , typename ConcreteVGidz
    , typename ConcreteMS
    , typename ConcreteMSid0z
    , typename ConcreteMSid0y
    , typename ConcreteMSid1z
    , typename ConcreteMSid1y
    , typename ConcreteMSida  
    , typename ConcreteMSTrafo0
    , typename ConcreteMSTrafo1
    , typename ConcreteRayGen
    , typename ListSizeType
    , typename GridSizeType
    , typename MemArrSizeType
>
__global__
void getSystemMatrix(
      T * const            sme_devi,   // return array for system matrix elements
      GridSizeType * const vxlId_devi, // return array for voxel ids
      int * const          cnlId_devi, // return array for channel ids
      int const * const    ml_devi,
      ListSizeType const * const mlSize_devi,
      MemArrSizeType * const truckDest_devi
     ) {
  
  /* Global id of thread and global dim of the kernel call */
  int const globalId  = threadIdx.x + blockIdx.x*blockDim.x;
  int const globalDim = blockDim.x*gridDim.x;
  
  __shared__ uint           nPassed_blck;
  __shared__ int            truckCnlId_blck[TPB];
  __shared__ GridSizeType   truckVxlId_blck[TPB];
  __shared__ MemArrSizeType truckDest_blck;
  
  /* Random init of thread's ray generator */
  ConcreteRayGen rayGen(int(RANDOM_SEED), globalId);
  
  /* Master thread in block?*/
  if(threadIdx.x == 0) {
    /* Set count of positivly tested system matrix elements (SMEs) to zero */
    nPassed_blck = 0;
  }
  __syncthreads();
  
  /* Create indices related functors */
  GetId2vxlId<ConcreteVG>   f_vxlId(grid_const);
  GetId2listId<ConcreteVG>  f_listId(grid_const);
  IsInRange<ConcreteVG, ListSizeType, GridSizeType, MemArrSizeType>
    f_isInRange(grid_const, *mlSize_devi);
  GetIsLegal<ConcreteVG, GridSizeType, ListSizeType>
    f_getIsLegal(grid_const, *mlSize_devi);
  
  /* Loop over SMEs. The linear loop index getId will be decomposed within the
   * body into two indices, i.e. the index into the grid (vxlId) and the index
   * into the measurement list (listId). Each thread's getId walks through a
   * unique subset of
   * [0, ..., size(measurement list)*size(grid) + globalDim - 1]. The
   * decomposition ensures that within the whole range of getIds over all
   * threads, (1) getIds are mapped to unique pairs (vxlId, listId) and
   * (2) all possible pairs are obtained. getIds outside the SM range spanned
   * by the grid and the measurement list will be handled properly. */
  for(MemArrSizeType getId_thrd = globalId;
          f_isInRange(getId_thrd);
          getId_thrd += globalDim) {
    /* Calculate linear indices into grid and measurement list */
    GridSizeType vxlId_thrd( f_vxlId( getId_thrd));
    ListSizeType listId_thrd(f_listId(getId_thrd));
    int cnlId_thrd = -1;
    
    int writeOffset_thrd = -1;
    
    /* Do voxel index and list index point to an SME in calculable range? */
    if(f_getIsLegal(vxlId_thrd, listId_thrd)) {

      /* Get id of channel from list id */
      cnlId_thrd = ml_devi[listId_thrd];

      /* Put current SME to the test */
      bool didPass_thrd = test<
                                T
                              , ConcreteVG
                              , ConcreteVGidx
                              , ConcreteVGidy
                              , ConcreteVGidz
                              , ConcreteMS
                              , ConcreteMSid0z
                              , ConcreteMSid0y
                              , ConcreteMSid1z
                              , ConcreteMSid1y
                              , ConcreteMSida
                              , ConcreteMSTrafo0
                              , ConcreteMSTrafo1
                              , GridSizeType>
                            (cnlId_thrd, vxlId_thrd);

      /* Did current SME pass the test? */
      if(didPass_thrd) {

        /* Increase the count of passed elements in this block and get write
         * offset into shared mem */
        writeOffset_thrd = atomicAdd(&nPassed_blck, 1);

        /* Can current SME be written to shared mem during this pass of the
         * loop? */
        if(writeOffset_thrd < TPB) {

          /* Write current SME to shared mem */
          truckCnlId_blck[writeOffset_thrd] = cnlId_thrd;
          truckVxlId_blck[writeOffset_thrd] = vxlId_thrd;
        } /* end if write current SME directly */
      } /* end if current SME did pass */
    } /* end if current SME legal */
    __syncthreads();

    /* Is it time for a flush of shared mem to global mem? */
    if(nPassed_blck >= TPB) {
      
      /* Master thread? */
      if(threadIdx.x == 0) {
        /* Get write offset into global mem and decrease count of passed
         * elements in this block by the number that will now be flushed to
         * global mem */
        truckDest_blck = atomicAdd(truckDest_devi, MemArrSizeType(TPB));
        nPassed_blck -= TPB;
      }
      __syncthreads();
      
      /* Calculate the SME scheduled for flushing and flush it to global mem */
      val_t sme_thrd = calcSme<
                              T
                            , ConcreteVG
                            , ConcreteVGidx
                            , ConcreteVGidy
                            , ConcreteVGidz
                            , ConcreteMS
                            , ConcreteMSid0z
                            , ConcreteMSid0y
                            , ConcreteMSid1z
                            , ConcreteMSid1y
                            , ConcreteMSida  
                            , ConcreteMSTrafo0
                            , ConcreteMSTrafo1
                            , ConcreteRayGen >
                          ( truckCnlId_blck[threadIdx.x],
                            truckVxlId_blck[threadIdx.x], rayGen);
      
      sme_devi[  truckDest_blck+threadIdx.x]  = sme_thrd;
      cnlId_devi[truckDest_blck+threadIdx.x]  = truckCnlId_blck[threadIdx.x];
      vxlId_devi[truckDest_blck+threadIdx.x]  = truckVxlId_blck[threadIdx.x];
      __syncthreads();
      
      /* Could current SME NOT be written to shared before? */
      if(writeOffset_thrd >= TPB) {
        
        writeOffset_thrd -= TPB;
        
        /* Write current SME to shared */
        truckCnlId_blck[writeOffset_thrd] = cnlId_thrd;
        truckVxlId_blck[writeOffset_thrd] = vxlId_thrd;
      }
    }
  } /* end loop*/
  
  /* Is a last flush to global necessary? */
  if(nPassed_blck > 0) {
    
    /* Master thread? */
    if(threadIdx.x == 0) {
      /* Get write offset into global mem */
      truckDest_blck = atomicAdd(truckDest_devi, MemArrSizeType(nPassed_blck));
    }
    __syncthreads();
    
    /* Does this thread take part in the last flush? */
    if(threadIdx.x < nPassed_blck) {
      
      /* Calculate the SME scheduled for flushing and flush it to global mem */
      val_t sme_thrd = calcSme<
                              T
                            , ConcreteVG
                            , ConcreteVGidx
                            , ConcreteVGidy
                            , ConcreteVGidz
                            , ConcreteMS
                            , ConcreteMSid0z
                            , ConcreteMSid0y
                            , ConcreteMSid1z
                            , ConcreteMSid1y
                            , ConcreteMSida  
                            , ConcreteMSTrafo0
                            , ConcreteMSTrafo1
                            , ConcreteRayGen >
                          (truckCnlId_blck[threadIdx.x],
                           truckVxlId_blck[threadIdx.x], rayGen);
      
      sme_devi[  truckDest_blck+threadIdx.x]  = sme_thrd;
      cnlId_devi[truckDest_blck+threadIdx.x]  = truckCnlId_blck[threadIdx.x];
      vxlId_devi[truckDest_blck+threadIdx.x]  = truckVxlId_blck[threadIdx.x];
      __syncthreads();
    }
  }
}

#endif /* GETSYSTEMMATRIXDEVICEONLY_CU */
