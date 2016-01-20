/** @file getSystemMatrixDeviceOnly.cu */
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

/** @brief Test if a given channel maybe intersects a given voxel.
 * @param cnlId Linear id of the channel.
 * @param vxlId Linear id of the voxel. */
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
  
  /* Create functors */
  int const idx  = VGidx()( vxlId, &grid_const);
  int const idy  = VGidy()( vxlId, &grid_const);
  int const idz  = VGidz()( vxlId, &grid_const);
  
  int const id0z = MSid0z()(cnlId, &setup_const);
  int const id0y = MSid0y()(cnlId, &setup_const);
  int const id1z = MSid1z()(cnlId, &setup_const);
  int const id1y = MSid1y()(cnlId, &setup_const);
  int const ida  = MSida()( cnlId, &setup_const);

  /* Sum of radii */
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

  
  if(distance(pix0Center, pix1Center, vxlCenter)<sumRadii) {
    return true;
  }
  return false;
}



/** @brief Calculate the intersection length of a line with a box.
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
  
  /* Get entry and exit points */
  T aMin, aMax;
  bool aMinGood, aMaxGood;
  MaxFunctor<3>()(&aMin, &aMinGood, aDimMin, sects);
  MinFunctor<3>()(&aMax, &aMaxGood, aDimMax, sects);
  
  /* Really intersects? */
  if(!(aMin<aMax) || (!aMinGood) || (!aMaxGood))
    return 0;
  
  return (aMax-aMin) * sqrt(  (rayCoord[3+0]-rayCoord[0]) * (rayCoord[3+0]-rayCoord[0])
                             +(rayCoord[3+1]-rayCoord[1]) * (rayCoord[3+1]-rayCoord[1])
                             +(rayCoord[3+2]-rayCoord[2]) * (rayCoord[3+2]-rayCoord[2]) );
}



/** @brief Calculate system matrix element.
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

  /* Voxel coordinates */
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
  
  /* Channel indices */
  int const id0z = MSId0z()(cnl, &setup_const);
  int const id0y = MSId0y()(cnl, &setup_const);
  int const id1z = MSId1z()(cnl, &setup_const);
  int const id1y = MSId1y()(cnl, &setup_const);
  int const ida  = MSIda()( cnl, &setup_const);
  
  /* Functors */
  MSTrafo0_inplace  trafo0;
  MSTrafo1_inplace  trafo1;
  
  /* Matrix element */
  T a(0.);

  /* Add up intersection lengths */
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


/** @brief Kernel function. Calculates system matrix.
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
  
  /* global id and global dim */
  int const globalId  = threadIdx.x + blockIdx.x*blockDim.x;
  int const globalDim = blockDim.x*gridDim.x;
  
  __shared__ uint           nPassed_blck;
  __shared__ int            truckCnlId_blck[TPB];
  __shared__ GridSizeType   truckVxlId_blck[TPB];
  __shared__ MemArrSizeType truckDest_blck;
  
  /* Random init */
  ConcreteRayGen rayGen(int(RANDOM_SEED), globalId);
  
  /* Master thread */
  if(threadIdx.x == 0) {
    nPassed_blck = 0;
  }
  __syncthreads();
  
  GetId2vxlId<ConcreteVG>   f_vxlId(grid_const);
  GetId2listId<ConcreteVG>  f_listId(grid_const);
  IsInRange<ConcreteVG, ListSizeType, GridSizeType, MemArrSizeType>
    f_isInRange(grid_const, *mlSize_devi);
  GetIsLegal<ConcreteVG, GridSizeType, ListSizeType>
    f_getIsLegal(grid_const, *mlSize_devi);
  for(MemArrSizeType getId_thrd = globalId;
          f_isInRange(getId_thrd);
          getId_thrd += globalDim) {
    GridSizeType vxlId_thrd( f_vxlId( getId_thrd));
    ListSizeType listId_thrd(f_listId(getId_thrd));
    int cnlId_thrd = -1;
    
    int writeOffset_thrd = -1;
    
    /* Is getting another element legal? */
    if(f_getIsLegal(vxlId_thrd, listId_thrd)) {

      /* Get cnlId */
      cnlId_thrd = ml_devi[listId_thrd];

      /* Put this element to the test */
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

      /* Did it pass the test? */
      if(didPass_thrd) {

        /* Increase the count of passed elements in this block and get write
         * offset into shared mem */
        writeOffset_thrd = atomicAdd(&nPassed_blck, 1);

        /* Can this element be written to shared directly? */
        if(writeOffset_thrd < TPB) {

          /* Write element to shared */
          truckCnlId_blck[writeOffset_thrd] = cnlId_thrd;
          truckVxlId_blck[writeOffset_thrd] = vxlId_thrd;
        }
      }
    }
    __syncthreads();

    /* Is it time for a flush? */
    if(nPassed_blck >= TPB) {
      
      /* Master thread? */
      if(threadIdx.x == 0) {
        truckDest_blck = atomicAdd(truckDest_devi, MemArrSizeType(TPB));
        nPassed_blck -= TPB;
      }
      __syncthreads();
      
      /* Calculate SM element and flush */
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
      
      /* Could this element NOT be written to shared before? */
      if(writeOffset_thrd >= TPB) {
        
        writeOffset_thrd -= TPB;
        
        /* Write element to shared */
        truckCnlId_blck[writeOffset_thrd] = cnlId_thrd;
        truckVxlId_blck[writeOffset_thrd] = vxlId_thrd;
      }
    }
  }
  
  /* Is a last flush necessary? */
  if(nPassed_blck > 0) {
    
    /* Master thread? */
    if(threadIdx.x == 0) {
      truckDest_blck = atomicAdd(truckDest_devi, MemArrSizeType(nPassed_blck));
    }
    __syncthreads();
    
    /* Does this thread take part? */
    if(threadIdx.x < nPassed_blck) {
      
      /* Calculate SM element and flush */
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
