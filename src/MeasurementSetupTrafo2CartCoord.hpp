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

/** @file MeasurementSetupTrafo2CartCoord.hpp
 * 
 *  @brief Header file that defines the MeasurementSetupTrafo2CartCoord...
 *  templates for pixels on the 1st detector and for pixels on the 2nd detector
 *  and specialisations to use.
 *  Specialisations are trafos that are used to get the global coordinates of
 *  a point in the volume of a given detector pixel.
 */

/*
 * TODO:
 * - put transformation matrix calculation in constructor
 * - reuse transformation matrix -> calculate it only once in construction
 * - put argument "meas" in Ctor (?), so that functors are created for one
 *   specific measurement setup only
 * - asserts: are arguments within valid range?
 */

#ifndef MEASUREMENTSETUPTRAFO2CARTCOORD_HPP
#define	MEASUREMENTSETUPTRAFO2CARTCOORD_HPP

#include "MeasurementSetup.hpp"
#include <cmath>

/**
 * @brief Functor that transforms spatial coordinates from within the unit dice
 * to spatial coordinates within a given pixel on the first detector. Out of
 * place.
 * 
 * @tparam T Coordinate type.
 * @tparam ConcreteMSTrafo2CartCoordFirstPixel CRTP derived type.
 * @tparam ConcreteMS Measurement setup type.
 */
template<
      typename T
    , typename ConcreteMSTrafo2CartCoordFirstPixel
    , typename ConcreteMS >
class MeasurementSetupTrafo2CartCoordFirstPixel {
  public:
    /**
     * brief Functor operation.
     * 
     * @param cart_coords Output coordinates.
     * @param box_coords Input coodinates (within unit dice).
     * @param id0z Pixel index in z direction.
     * @param id0y Pixel index in y direction.
     * @param ida Rotation index.
     * @param meas Measurement setup.
     */
    __host__ __device__
    void operator()(
          T * const cart_coords, T const * const box_coords,
          int const id0z, int const id0y, int const ida,
          ConcreteMS const * const meas ) {
      return static_cast<ConcreteMSTrafo2CartCoordFirstPixel*>(this)->
              operator()(cart_coords, box_coords,
                         id0z, id0y, ida,
                         meas);
    }
};

/**
 * @brief Concrete derived class.
 */
template<
      typename T
    , typename ConcreteMS >
class DefaultMeasurementSetupTrafo2CartCoordFirstPixel
: public MeasurementSetupTrafo2CartCoordFirstPixel<
             T,
             DefaultMeasurementSetupTrafo2CartCoordFirstPixel<T, ConcreteMS>,
             ConcreteMS> {
  public:
    __host__ __device__
    void operator()(
          T * const cart_coords, T const * const box_coords,
          int const id0z, int const id0y, int const ida,
          ConcreteMS const * const meas ) const {
      // get pixel edge lengths
      T edges[3];
      edges[0] = meas->segx();
      edges[1] = meas->segy();
      edges[2] = meas->segz();
      // get not-yet-rotated position of pixel
      T pos[3];
      pos[0]=meas->pos0x();
      pos[1]=(id0y-T(.5)*(meas->n0y()-1))*edges[1];
      pos[2]=(id0z-T(.5)*(meas->n0z()-1))*edges[2];
      // get angular function values of channel's rotation
      T sin_ = sinf(T(M_PI)*ida*(meas->da())/T(180.0));
      T cos_ = cosf(T(M_PI)*ida*(meas->da())/T(180.0));
        
      // create transformation matrix
      T trafo[12];

      trafo[0*4 + 0] = cos_*edges[0];
      trafo[0*4 + 1] = T(0.);
      trafo[0*4 + 2] = sin_*edges[2];
      trafo[0*4 + 3] = cos_*(pos[0]-T(.5)*edges[0])\
                      +sin_*(pos[2]-T(.5)*edges[2]);

      trafo[1*4 + 0] = T(0.);
      trafo[1*4 + 1] = edges[1];
      trafo[1*4 + 2] = T(0.);
      trafo[1*4 + 3] = pos[1]-T(.5)*edges[1];

      trafo[2*4 + 0] =-sin_*edges[0];
      trafo[2*4 + 1] = T(0.);
      trafo[2*4 + 2] = cos_*edges[2];
      trafo[2*4 + 3] =-sin_*(pos[0]-T(.5)*edges[0])\
                      +cos_*(pos[2]-T(.5)*edges[2]);

      // apply transformation matrix
      cart_coords[0] =   trafo[0*4 + 0] * box_coords[0]
                       + trafo[0*4 + 1] * box_coords[1]
                       + trafo[0*4 + 2] * box_coords[2] 
                       + trafo[0*4 + 3] * T(1.);

      cart_coords[1] =   trafo[1*4 + 0] * box_coords[0]
                       + trafo[1*4 + 1] * box_coords[1]
                       + trafo[1*4 + 2] * box_coords[2] 
                       + trafo[1*4 + 3] * T(1.);

      cart_coords[2] =   trafo[2*4 + 0] * box_coords[0]
                       + trafo[2*4 + 1] * box_coords[1]
                       + trafo[2*4 + 2] * box_coords[2] 
                       + trafo[2*4 + 3] * T(1.);
    }
};
  
/**
 * @brief Functor that transforms spatial coordinates from within the unit dice
 * to spatial coordinates within a given pixel on the first detector. Inplace.
 * 
 * @tparam T Coordinate type.
 * @tparam ConcreteMSTrafo2CartCoordFirstPixel CRTP derived type.
 * @tparam ConcreteMS Measurement setup type.
 */
template<
      typename T
    , typename C
    , typename MS >
class MeasurementSetupTrafo2CartCoordFirstPixel_inplace {
  public:
    /**
     * brief Functor operation.
     * 
     * @param coords On call: Input coordinates. After termination: Output
     * coordinates.
     * @param id0z Pixel index in z direction.
     * @param id0y Pixel index in y direction.
     * @param ida Rotation index.
     * @param meas Measurement setup.
     */
    __host__ __device__
    void operator()(
          T * const coords,
          int const id0z, int const id0y, int const ida,
          MS const * const meas ) {
      return static_cast<C*>(this)->
              operator()(coords,
                         id0z, id0y, ida,
                         meas);
    }
};

/**
 * @brief Concrete derived class.
 */
template<
      typename T
    , typename MS >
class DefaultMeasurementSetupTrafo2CartCoordFirstPixel_inplace
: public MeasurementSetupTrafo2CartCoordFirstPixel_inplace<
             T,
             DefaultMeasurementSetupTrafo2CartCoordFirstPixel_inplace<T, MS>,
             MS> {
  public:
    __host__ __device__
    void operator()(
          T * const coords,
          int const id0z, int const id0y, int const ida,
          MS const * const meas ) const {
      // get pixel edge lengths
      T edges[3];
      edges[0] = meas->segx();
      edges[1] = meas->segy();
      edges[2] = meas->segz();
      // get not-yet-rotated position of pixel
      T pos[3];
      pos[0]=meas->pos0x();
      pos[1]=(id0y-T(.5)*(meas->n0y()-1))*edges[1];
      pos[2]=(id0z-T(.5)*(meas->n0z()-1))*edges[2];
      // get angular function values of channel's rotation
      T sin_ = sinf(T(M_PI)*ida*(meas->da())/T(180.0));
      T cos_ = cosf(T(M_PI)*ida*(meas->da())/T(180.0));
        
      // create transformation matrix
      T trafo[12];

      trafo[0*4 + 0] = cos_*edges[0];
      trafo[0*4 + 1] = T(0.);
      trafo[0*4 + 2] = sin_*edges[2];
      trafo[0*4 + 3] = cos_*(pos[0]-T(.5)*edges[0])\
                      +sin_*(pos[2]-T(.5)*edges[2]);

      trafo[1*4 + 0] = T(0.);
      trafo[1*4 + 1] = edges[1];
      trafo[1*4 + 2] = T(0.);
      trafo[1*4 + 3] = pos[1]-T(.5)*edges[1];

      trafo[2*4 + 0] =-sin_*edges[0];
      trafo[2*4 + 1] = T(0.);
      trafo[2*4 + 2] = cos_*edges[2];
      trafo[2*4 + 3] =-sin_*(pos[0]-T(.5)*edges[0])\
                      +cos_*(pos[2]-T(.5)*edges[2]);

      // apply transformation matrix
      T tmp[3] = {coords[0], coords[1], coords[2]};
      coords[0] =   trafo[0*4 + 0] * tmp[0]
                       + trafo[0*4 + 1] * tmp[1]
                       + trafo[0*4 + 2] * tmp[2] 
                       + trafo[0*4 + 3] * T(1.);

      coords[1] =   trafo[1*4 + 0] * tmp[0]
                       + trafo[1*4 + 1] * tmp[1]
                       + trafo[1*4 + 2] * tmp[2] 
                       + trafo[1*4 + 3] * T(1.);

      coords[2] =   trafo[2*4 + 0] * tmp[0]
                       + trafo[2*4 + 1] * tmp[1]
                       + trafo[2*4 + 2] * tmp[2] 
                       + trafo[2*4 + 3] * T(1.);
    }
};

/**
 * @brief Functor that transforms spatial coordinates from within the unit dice
 * to spatial coordinates within a given pixel on the second detector. Out of
 * place.
 * 
 * @tparam T Coordinate type.
 * @tparam ConcreteMSTrafo2CartCoordFirstPixel CRTP derived type.
 * @tparam ConcreteMS Measurement setup type.
 */
template<
      typename T
    , typename ConcreteMSTrafo2CartCoordSecndPixel
    , typename ConcreteMS >
class MeasurementSetupTrafo2CartCoordSecndPixel {
  public:
    /**
     * brief Functor operation.
     * 
     * @param cart_coords Output coordinates.
     * @param box_coords Input coodinates (within unit dice).
     * @param id1z Pixel index in z direction.
     * @param id1y Pixel index in y direction.
     * @param ida Rotation index.
     * @param meas Measurement setup.
     */__host__ __device__
    void operator()(
          T * const cart_coords, T const * const box_coords,
          int const id1z, int const id1y, int const ida,
          ConcreteMS const * const meas ) {
      return static_cast<ConcreteMSTrafo2CartCoordSecndPixel*>(this)->
              operator()(cart_coords, box_coords,
                         id1z, id1y, ida,
                         meas);
    }
};

/**
 * @brief Concrete derived class.
 */
template<
      typename T
    , typename ConcreteMS >
class DefaultMeasurementSetupTrafo2CartCoordSecndPixel
: public MeasurementSetupTrafo2CartCoordSecndPixel<
             T,
             DefaultMeasurementSetupTrafo2CartCoordSecndPixel<T, ConcreteMS>,
             ConcreteMS> {
  public:
    __host__ __device__
    void operator()(
          T * const cart_coords, T const * const box_coords,
          int const id1z, int const id1y, int const ida,
          ConcreteMS const * const meas ) const {
      // get pixel edge lengths
      T edges[3];
      edges[0] = meas->segx();
      edges[1] = meas->segy();
      edges[2] = meas->segz();
      // get not-yet-rotated position of pixel
      T pos[3];
      pos[0]=meas->pos1x();
      pos[1]=(id1y-T(.5)*(meas->n1y()-1))*edges[1];
      pos[2]=(id1z-T(.5)*(meas->n1z()-1))*edges[2];
      // get angular function values of channel's rotation
      T sin_ = sinf(T(M_PI)*ida*(meas->da())/T(180.0));
      T cos_ = cosf(T(M_PI)*ida*(meas->da())/T(180.0));

      // create transformation matrix
      T trafo[12];

      trafo[0*4 + 0] = cos_*edges[0];
      trafo[0*4 + 1] = T(0.);
      trafo[0*4 + 2] = sin_*edges[2];
      trafo[0*4 + 3] = cos_*(pos[0]-T(.5)*edges[0])\
                      +sin_*(pos[2]-T(.5)*edges[2]);

      trafo[1*4 + 0] = T(0.);
      trafo[1*4 + 1] = edges[1];
      trafo[1*4 + 2] = T(0.);
      trafo[1*4 + 3] = pos[1]-T(.5)*edges[1];

      trafo[2*4 + 0] =-sin_*edges[0];
      trafo[2*4 + 1] = T(0.);
      trafo[2*4 + 2] = cos_*edges[2];
      trafo[2*4 + 3] =-sin_*(pos[0]-T(.5)*edges[0])\
                      +cos_*(pos[2]-T(.5)*edges[2]);

      // apply transformation matrix
      cart_coords[0] =   trafo[0*4 + 0] * box_coords[0]
                       + trafo[0*4 + 1] * box_coords[1]
                       + trafo[0*4 + 2] * box_coords[2] 
                       + trafo[0*4 + 3] * T(1.);

      cart_coords[1] =   trafo[1*4 + 0] * box_coords[0]
                       + trafo[1*4 + 1] * box_coords[1]
                       + trafo[1*4 + 2] * box_coords[2] 
                       + trafo[1*4 + 3] * T(1.);

      cart_coords[2] =   trafo[2*4 + 0] * box_coords[0]
                       + trafo[2*4 + 1] * box_coords[1]
                       + trafo[2*4 + 2] * box_coords[2] 
                       + trafo[2*4 + 3] * T(1.);
    }
};

/**
 * @brief Functor that transforms spatial coordinates from within the unit dice
 * to spatial coordinates within a given pixel on the second detector. Inplace.
 * 
 * @tparam T Coordinate type.
 * @tparam ConcreteMSTrafo2CartCoordFirstPixel CRTP derived type.
 * @tparam ConcreteMS Measurement setup type.
 */
template<
      typename T
    , typename C
    , typename MS >
class MeasurementSetupTrafo2CartCoordSecndPixel_inplace {
  public:
    /**
     * brief Functor operation.
     * 
     * @param coords On call: Input coordinates. After termination: Output
     * coordinates.
     * @param id1z Pixel index in z direction.
     * @param id1y Pixel index in y direction.
     * @param ida Rotation index.
     * @param meas Measurement setup.
     */
    __host__ __device__
    void operator()(
          T * const cart_coords,
          int const id1z, int const id1y, int const ida,
          MS const * const meas ) {
      return static_cast<C*>(this)->
              operator()(cart_coords,
                         id1z, id1y, ida,
                         meas);
    }
};

/**
 * @brief Concrete derived class.
 */
template<
      typename T
    , typename MS >
class DefaultMeasurementSetupTrafo2CartCoordSecndPixel_inplace
: public MeasurementSetupTrafo2CartCoordSecndPixel_inplace<
             T,
             DefaultMeasurementSetupTrafo2CartCoordSecndPixel_inplace<T, MS>,
             MS> {
  public:
    __host__ __device__
    void operator()(
          T * const coords,
          int const id1z, int const id1y, int const ida,
          MS const * const meas ) const {
      // get pixel edge lengths
      T edges[3];
      edges[0] = meas->segx();
      edges[1] = meas->segy();
      edges[2] = meas->segz();
      // get not-yet-rotated position of pixel
      T pos[3];
      pos[0]=meas->pos1x();
      pos[1]=(id1y-T(.5)*(meas->n1y()-1))*edges[1];
      pos[2]=(id1z-T(.5)*(meas->n1z()-1))*edges[2];
      // get angular function values of channel's rotation
      T sin_ = sinf(T(M_PI)*ida*(meas->da())/T(180.0));
      T cos_ = cosf(T(M_PI)*ida*(meas->da())/T(180.0));

      // create transformation matrix
      T trafo[12];

      trafo[0*4 + 0] = cos_*edges[0];
      trafo[0*4 + 1] = T(0.);
      trafo[0*4 + 2] = sin_*edges[2];
      trafo[0*4 + 3] = cos_*(pos[0]-T(.5)*edges[0])\
                      +sin_*(pos[2]-T(.5)*edges[2]);

      trafo[1*4 + 0] = T(0.);
      trafo[1*4 + 1] = edges[1];
      trafo[1*4 + 2] = T(0.);
      trafo[1*4 + 3] = pos[1]-T(.5)*edges[1];

      trafo[2*4 + 0] =-sin_*edges[0];
      trafo[2*4 + 1] = T(0.);
      trafo[2*4 + 2] = cos_*edges[2];
      trafo[2*4 + 3] =-sin_*(pos[0]-T(.5)*edges[0])\
                      +cos_*(pos[2]-T(.5)*edges[2]);

      // apply transformation matrix
      T tmp[3] = {coords[0], coords[1], coords[2]};
      coords[0] =   trafo[0*4 + 0] * tmp[0]
                       + trafo[0*4 + 1] * tmp[1]
                       + trafo[0*4 + 2] * tmp[2] 
                       + trafo[0*4 + 3] * T(1.);

      coords[1] =   trafo[1*4 + 0] * tmp[0]
                       + trafo[1*4 + 1] * tmp[1]
                       + trafo[1*4 + 2] * tmp[2] 
                       + trafo[1*4 + 3] * T(1.);

      coords[2] =   trafo[2*4 + 0] * tmp[0]
                       + trafo[2*4 + 1] * tmp[1]
                       + trafo[2*4 + 2] * tmp[2] 
                       + trafo[2*4 + 3] * T(1.);
    }
};

#endif	/* MEASUREMENTSETUPTRAFO2CARTCOORD_HPP */

