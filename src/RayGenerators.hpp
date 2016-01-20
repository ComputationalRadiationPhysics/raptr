/** @file RayGenerators.hpp
 * Author: malte
 *
 * Created on 12. Juni 2015, 14:33 */

#ifndef RAYGENERATORS_HPP
#define	RAYGENERATORS_HPP

#include "device_constant_memory.hpp"

template<
      typename T
    , typename MSTrafo0_inplace
    , typename MSTrafo1_inplace >
struct RandRayGen {
  __device__
  RandRayGen( int const seed, int const setNr ) {
    curand_init(seed, setNr, 0, &state_);
  }
  
  __device__
  void operator()(T * __restrict__ const ray,
        int const & id0z, int const & id0y, int const & id1z, int const & id1y,
        int const & ida,
        MSTrafo0_inplace const & trafo0,
        MSTrafo1_inplace const & trafo1) {
    
    /* Get randoms */
    for(int i=0; i<6; i++) {
      ray[i] = curand_uniform(&state_);
    }
    
    /* Transform to cartesian coordinates */
    trafo0(&ray[0], id0z, id0y, ida, &setup_const);
    trafo1(&ray[3], id1z, id1y, ida, &setup_const);
  }

private:
  curandState state_;
};

template<
      typename T
    , typename MSTrafo0_inplace
    , typename MSTrafo1_inplace >
struct ReguRayGen {
  __device__
  ReguRayGen( int const seed, int const setNr ) {}
  
  __device__
  void operator()(T * __restrict__ const ray,
        int const & id0z, int const & id0y, int const & id1z, int const & id1y,
        int const & ida,
        MSTrafo0_inplace const & trafo0,
        MSTrafo1_inplace const & trafo1) const {
    
    /* Get uniform position */
    for(int i=0; i<6; i++) {
      ray[i] = T(0.5);
    }
    
    /*Transform to cartesian coordinates */
    trafo0(&ray[0], id0z, id0y, ida, &setup_const);
    trafo1(&ray[3], id1z, id1y, ida, &setup_const);
  }
};

#endif	/* RAYGENERATORS_HPP */

