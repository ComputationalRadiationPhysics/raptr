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

/** @file RayGenerators.hpp
 *  
 *  @brief Header file that defines ray generators. A specific ray generator
 *  implements a certain way of generating rays.
 */

#ifndef RAYGENERATORS_HPP
#define	RAYGENERATORS_HPP

#include "device_constant_memory.hpp"

/**
 * @brief Functor for generating rays with start and end points randomly chosen
 * from the detector pixel volumes.
 */
template<
      typename T
    , typename MSTrafo0_inplace
    , typename MSTrafo1_inplace >
struct RandRayGen {
  /* Ctor */
  __device__
  RandRayGen( int const seed, int const setNr ) {
    curand_init(seed, setNr, 0, &state_);
  }
  
  /* Functor operation */
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

/**
 * @brief Functor for generating rays with start and end points centered in the
 * detector pixel volumes.
 */
template<
      typename T
    , typename MSTrafo0_inplace
    , typename MSTrafo1_inplace >
struct ReguRayGen {
  /* Ctor */
  __device__
  ReguRayGen( int const seed, int const setNr ) {}
  
  /* Functor operation */
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

