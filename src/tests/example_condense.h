/** @file example_condense.h */
/* 
 * File:   example_condense.h
 * Author: malte
 *
 * Created on 26. November 2014, 15:48
 */

#ifndef EXAMPLE_CONDENSE_H
#define	EXAMPLE_CONDENSE_H



#include "example_condense_defines.h"
#include <cuda.h>
#include <curand_kernel.h>
//#include <stdlib.h>
//#include <stdio.h>
  
typedef float val_t;

__device__
bool test( val_t const & elem ) {
//  return (elem>THRESHOLD);
  return true;
}

template<typename State>
__device__
void init_elems( State * const, int const );

template<>
__device__
void init_elems<curandState>( curandState * const state_thrd, int const start ){
  curand_init(SEED, 0, start, state_thrd);
}

struct OwnState {
  int i;
};

template<>
__device__
void init_elems<OwnState>( OwnState * const state_thrd, int const start ){
  state_thrd->i = start;
}

template<typename State>
__device__
val_t get_elem( State * const );

template<>
__device__
val_t get_elem<curandState>( curandState * const state_thrd) {
  return curand_uniform(state_thrd);
}

template<>
__device__
val_t get_elem<OwnState>( OwnState * const state_thrd) {
  state_thrd->i += 1;
  return (state_thrd->i)-1;
}



__global__
void condense(
      val_t * const passed_devi, int * truckDest_devi ) {
  
  // Initialize
  int const         globalId = threadIdx.x + blockDim.x*blockIdx.x;
  int const         globalDim = gridDim.x*blockDim.x;
  __shared__ int    nPassed_blck;
  __shared__ val_t  truck_blck[TPB];
  __shared__ int    truckDest_blck;
  
  if(threadIdx.x == 0) {
    nPassed_blck = 0;
  }
  __syncthreads();
  
//  curandState      state_thrd;
  OwnState         state_thrd;

  for(int getId_thrd = globalId;
      getId_thrd < (SIZE + blockDim.x -1);
      getId_thrd += globalDim) {
    
    int writeOffset_thrd = -1;
    val_t boat_thrd;
    
    // Is getting another element legal?
    if(getId_thrd < SIZE) {

      // Get another element
      init_elems(&state_thrd, getId_thrd);
      boat_thrd = get_elem(&state_thrd);

      // Put this element to the test
      bool didPass_thrd = test(boat_thrd);

      // Did it pass the test?
      if(didPass_thrd) {

        // Increase the count of passed elements in this block and get write
        // offset into shared mem
        writeOffset_thrd = atomicAdd(&nPassed_blck, 1);

        // Can this element be written to shared during this loop passage?
        if(writeOffset_thrd < TPB) {

          // Write element to shared
          truck_blck[writeOffset_thrd] = boat_thrd;
        }
      }
    }
    __syncthreads();

    // Is it time for a flush?
    if(nPassed_blck >= TPB) {
      
      if(threadIdx.x == 0) {
        truckDest_blck = atomicAdd(truckDest_devi, TPB);
        nPassed_blck -= TPB;
      }
      __syncthreads();
      
      // Flush
      passed_devi[truckDest_blck+threadIdx.x] = truck_blck[threadIdx.x];
//      val_t length_thrd = calcLength(truck_blck[threadIdx.x]);
//      length_devi[] = length_thrd;
      __syncthreads();
      if(writeOffset_thrd >= TPB) {
        writeOffset_thrd -= TPB;
        truck_blck[writeOffset_thrd] = boat_thrd;
      }
    }
  }
  
  // Flush
  if(nPassed_blck > 0) {
    if(threadIdx.x == 0) {
      truckDest_blck = atomicAdd(truckDest_devi, nPassed_blck);
    }
    __syncthreads();
    
    if(threadIdx.x < nPassed_blck) {
      passed_devi[truckDest_blck+threadIdx.x] = truck_blck[threadIdx.x];
//      val_t length_thrd = calcLength(truck_blck[threadIdx.x]);
//      length_devi[] = length_thrd;
    }
  }
}
  


#endif	/* EXAMPLE_CONDENSE_H */
