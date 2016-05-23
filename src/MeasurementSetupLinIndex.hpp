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

/** @file MeasurementSetupLinIndex.hpp
 * 
 *  @brief Header file that defines the measurement setup index functor
 *  templates and specializations for the ePET measurements.
 *  Specializations define the mapping from multi-dim channel indices to a
 *  1-dim linearized channel index and the corresponding inverse mapping.
 */

/*
 * TODO:
 * - put argument "meas" in Ctor (?), so that functors are created for one
 *   specific measurement setup only
 * - asserts: are arguments within valid range?
 */

#ifndef MEASUREMENTSETUPLININDEX_HPP
#define	MEASUREMENTSETUPLININDEX_HPP

#include "MeasurementSetup.hpp"

/**
 * @brief Functor template. Objects perform 'multi-dim channel index ->
 * linearized channel index'. 
 * 
 * @tparam ConcreteMSLinId Type of a specialization of this template.
 * @tparam ConcreteMeasurementSetup Type of a setup that indices refer to.
 */
template<typename ConcreteMSLinId, typename ConcreteMeasurementSetup>
struct MeasurementSetupLinId {
  /**
   * @brief Functor operation.
   * 
   * @param id0z Pixel index on 1st detector in z direction.
   * @param id0y Pixel index on 1st detector in y direction.
   * @param id1z Pixel index on 2nd detector in z direction.
   * @param id1y Pixel index on 2nd detector in y direction.
   * @param ida Index of angular step of the measurement.
   * @param meas Ptr to the stetup definition object.
   * @return Linearized channel index.
   */
  __host__ __device__
  int operator()(
        int const id0z, int const id0y, int const id1z, int const id1y,
        int const ida, ConcreteMeasurementSetup const * const meas ) {
    return static_cast<ConcreteMSLinId*>(this)->
            operator()(id0z, id0y, id1z, id1y, ida, meas);
  }
};

/**
 * @brief Partial template specialization. Specializes, how the linearized
 * channel index is calculated from the multi-dim channel index. The multi-dim
 * indices ordered from least volatile to most volatile are:
 * (ida, id0z, id0y, id1z, id1y).
 * 
 * @tparam ConcreteMeasurementSetup Type of a setup that indices refer to.
 */
template<typename ConcreteMeasurementSetup>
struct DefaultMeasurementSetupLinId
: public MeasurementSetupLinId<DefaultMeasurementSetupLinId<ConcreteMeasurementSetup>,
                                ConcreteMeasurementSetup> {
  __host__ __device__
  int operator()(
        int const id0z, int const id0y, int const id1z, int const id1y,
        int const ida, ConcreteMeasurementSetup const * const meas ) {
      return ida  * (meas->n1y()*meas->n1z()*meas->n0y()*meas->n0z())
           + id0z * (meas->n1y()*meas->n1z()*meas->n0y())
           + id0y * (meas->n1y()*meas->n1z())
           + id1z * (meas->n1y())
           + id1y;
  }
};

/**
 * @brief Functor template. Objects perform 'linearized channel index ->
 * channel's index of angular step'. 
 * 
 * @tparam ConcreteMSIda Type of a specialization of this template.
 * @tparam ConcreteMeasurementSetup Type of a setup that indices refer to.
 */
template<typename ConcreteMSIda, typename ConcreteMeasurementSetup>
struct MeasurementSetupIda {
  /**
   * @brief Functor operation.
   * 
   * @param linId Linearized channel index.
   * @param meas Ptr to the stetup definition object.
   * @return Channel's index of angular step.
   */
  __host__ __device__
  int operator()(
        int const linId, ConcreteMeasurementSetup const * const meas ) {
    return static_cast<ConcreteMSIda*>(this)->
            operator()(linId, meas);
  }
};

/**
 * @brief Partial template specialization. Specializes, how the channel's
 * angular index is calculated from the linearized channel index. Corresponds to
 * DefaultMeasurementSetupLinId<typename ConcreteMeasurementSetup>.
 * 
 * @tparam ConcreteMeasurementSetup Type of a setup that indices refer to.
 */
template<typename ConcreteMeasurementSetup>
struct DefaultMeasurementSetupIda
: public MeasurementSetupIda<DefaultMeasurementSetupIda<ConcreteMeasurementSetup>,
                              ConcreteMeasurementSetup> {
  __host__ __device__
  int operator()(
        int const linId, ConcreteMeasurementSetup const * const meas ) {
    return linId /(meas->n1y()*meas->n1z()*meas->n0y()*meas->n0z());
  }
};

/**
 * @brief Functor template. Objects perform 'linearized channel index ->
 * pixel index on 1st detector in z direction.'. 
 * 
 * @tparam ConcreteMSId0z Type of a specialization of this template.
 * @tparam ConcreteMeasurementSetup Type of a setup that indices refer to.
 */
template<typename ConcreteMSId0z, typename ConcreteMeasurementSetup>
struct MeasurementSetupId0z {
  /**
   * @brief Functor operation.
   * 
   * @param linId Linearized channel index.
   * @param meas Ptr to the stetup definition object.
   * @return Channel's pixel index on 1st detector in z direction.
   */
  __host__ __device__
  int operator()(
        int const linId, ConcreteMeasurementSetup const * const meas ) {
    return static_cast<ConcreteMSId0z*>(this)->
            operator()(linId, meas);
  }
};

/**
 * @brief Partial template specialization. Specializes, how the pixel index on
 * 1st detector in z direction is calculated from the linearized channel index.
 * Corresponds to
 * DefaultMeasurementSetupLinId<typename ConcreteMeasurementSetup>.
 * 
 * @tparam ConcreteMeasurementSetup Type of a setup that indices refer to.
 */
template<typename ConcreteMeasurementSetup>
struct DefaultMeasurementSetupId0z
: public MeasurementSetupId0z<DefaultMeasurementSetupId0z<ConcreteMeasurementSetup>,
                                ConcreteMeasurementSetup> {
  __host__ __device__
  int operator()(
        int const linId, ConcreteMeasurementSetup const * const meas ) {
    int temp = linId;
    temp %=      (meas->n1y()*meas->n1z()*meas->n0y()*meas->n0z());
    return temp /(meas->n1y()*meas->n1z()*meas->n0y());
  }
};

/**
 * @brief Functor template. Objects perform 'linearized channel index ->
 * pixel index on 1st detector in y direction.'. 
 * 
 * @tparam ConcreteMSId0y Type of a specialization of this template.
 * @tparam ConcreteMeasurementSetup Type of a setup that indices refer to.
 */
template<typename ConcreteMSId0y, typename ConcreteMeasurementSetup>
struct MeasurementSetupId0y {
  /**
   * @brief Functor operation.
   * 
   * @param linId Linearized channel index.
   * @param meas Ptr to the stetup definition object.
   * @return Channel's pixel index on 1st detector in y direction.
   */
  __host__ __device__
  int operator()(
        int const linId, ConcreteMeasurementSetup const * const meas ) {
    return static_cast<ConcreteMSId0y*>(this)->
            operator()(linId, meas);
  }
};

/**
 * @brief Partial template specialization. Specializes, how the pixel index on
 * 1st detector in y direction is calculated from the linearized channel index.
 * Corresponds to
 * DefaultMeasurementSetupLinId<typename ConcreteMeasurementSetup>.
 * 
 * @tparam ConcreteMeasurementSetup Type of a setup that indices refer to.
 */
template<typename ConcreteMeasurementSetup>
struct DefaultMeasurementSetupId0y
: public MeasurementSetupId0y<DefaultMeasurementSetupId0y<ConcreteMeasurementSetup>,
                                ConcreteMeasurementSetup> {
  __host__ __device__
  int operator()(
        int const linId, ConcreteMeasurementSetup const * const meas ) {
    int temp = linId;
    temp %=      (meas->n1y()*meas->n1z()*meas->n0y()*meas->n0z());
    temp %=      (meas->n1y()*meas->n1z()*meas->n0y());
    return temp /(meas->n1y()*meas->n1z());
  }
};

/**
 * @brief Functor template. Objects perform 'linearized channel index ->
 * pixel index on 2nd detector in z direction.'. 
 * 
 * @tparam ConcreteMSId1z Type of a specialization of this template.
 * @tparam ConcreteMeasurementSetup Type of a setup that indices refer to.
 */
template<typename ConcreteMSId1z, typename ConcreteMeasurementSetup>
struct MeasurementSetupId1z {
  /**
   * @brief Functor operation.
   * 
   * @param linId Linearized channel index.
   * @param meas Ptr to the stetup definition object.
   * @return Channel's pixel index on 2nd detector in z direction.
   */
  __host__ __device__
  int operator()(
        int const linId, ConcreteMeasurementSetup const * const meas ) {
    return static_cast<ConcreteMSId1z*>(this)->
            operator()(linId, meas);
  }
};

/**
 * @brief Partial template specialization. Specializes, how the pixel index on
 * 2nd detector in z direction is calculated from the linearized channel index.
 * Corresponds to
 * DefaultMeasurementSetupLinId<typename ConcreteMeasurementSetup>.
 * 
 * @tparam ConcreteMeasurementSetup Type of a setup that indices refer to.
 */
template<typename ConcreteMeasurementSetup>
struct DefaultMeasurementSetupId1z
: public MeasurementSetupId1z<DefaultMeasurementSetupId1z<ConcreteMeasurementSetup>,
                                ConcreteMeasurementSetup> {
  __host__ __device__
  int operator()(
        int const linId, ConcreteMeasurementSetup const * const meas ) {
    int temp = linId;
    temp %=      (meas->n1y()*meas->n1z()*meas->n0y()*meas->n0z());
    temp %=      (meas->n1y()*meas->n1z()*meas->n0y());
    temp %=      (meas->n1y()*meas->n1z());
    return temp /(meas->n1y());
  }
};

/**
 * @brief Functor template. Objects perform 'linearized channel index ->
 * pixel index on 2nd detector in y direction.'. 
 * 
 * @tparam ConcreteMSId1y Type of a specialization of this template.
 * @tparam ConcreteMeasurementSetup Type of a setup that indices refer to.
 */
template<typename ConcreteMSId1y, typename ConcreteMeasurementSetup>
struct MeasurementSetupId1y {
  /**
   * @brief Functor operation.
   * 
   * @param linId Linearized channel index.
   * @param meas Ptr to the stetup definition object.
   * @return Channel's pixel index on 2nd detector in y direction.
   */
  __host__ __device__
  int operator()(
        int const linId, ConcreteMeasurementSetup const * const meas ) {
    return static_cast<ConcreteMSId1y*>(this)->
            operator()(linId, meas);
  }
};

/**
 * @brief Partial template specialization. Specializes, how the pixel index on
 * 2nd detector in y direction is calculated from the linearized channel index.
 * Corresponds to
 * DefaultMeasurementSetupLinId<typename ConcreteMeasurementSetup>.
 * 
 * @tparam ConcreteMeasurementSetup Type of a setup that indices refer to.
 */
template<typename ConcreteMeasurementSetup>
struct DefaultMeasurementSetupId1y
: public MeasurementSetupId1y<DefaultMeasurementSetupId1y<ConcreteMeasurementSetup>,
                                ConcreteMeasurementSetup> {
  __host__ __device__
  int operator()(
        int const linId, ConcreteMeasurementSetup const * const meas ) {
    int temp = linId;
    temp %=      (meas->n1y()*meas->n1z()*meas->n0y()*meas->n0z());
    temp %=      (meas->n1y()*meas->n1z()*meas->n0y());
    temp %=      (meas->n1y()*meas->n1z());
    temp %=      (meas->n1y());
    return temp;
  }
};

#endif	/* MEASUREMENTSETUPLININDEX_HPP */

