/** @file MeasurementSetupLinIndex.hpp */
/* 
 * File:   MeasurementSetupLinIndex.hpp
 * Author: malte
 *
 * Created on 12. Oktober 2014, 16:01
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

template<typename ConcreteMSLinId, typename ConcreteMeasurementSetup>
struct MeasurementSetupLinId {
  __host__ __device__
  int operator()(
        int const id0z, int const id0y, int const id1z, int const id1y,
        int const ida, ConcreteMeasurementSetup const * const meas ) {
    return static_cast<ConcreteMSLinId*>(this)->
            operator()(id0z, id0y, id1z, id1y, ida, meas);
  }
};

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

template<typename ConcreteMSIda, typename ConcreteMeasurementSetup>
struct MeasurementSetupIda {
  __host__ __device__
  int operator()(
        int const linId, ConcreteMeasurementSetup const * const meas ) {
    return static_cast<ConcreteMSIda*>(this)->
            operator()(linId, meas);
  }
};

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

template<typename ConcreteMSId0z, typename ConcreteMeasurementSetup>
struct MeasurementSetupId0z {
  __host__ __device__
  int operator()(
        int const linId, ConcreteMeasurementSetup const * const meas ) {
    return static_cast<ConcreteMSId0z*>(this)->
            operator()(linId, meas);
  }
};

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

template<typename ConcreteMSId0y, typename ConcreteMeasurementSetup>
struct MeasurementSetupId0y {
  __host__ __device__
  int operator()(
        int const linId, ConcreteMeasurementSetup const * const meas ) {
    return static_cast<ConcreteMSId0y*>(this)->
            operator()(linId, meas);
  }
};

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

template<typename ConcreteMSId1z, typename ConcreteMeasurementSetup>
struct MeasurementSetupId1z {
  __host__ __device__
  int operator()(
        int const linId, ConcreteMeasurementSetup const * const meas ) {
    return static_cast<ConcreteMSId1z*>(this)->
            operator()(linId, meas);
  }
};

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

template<typename ConcreteMSId1y, typename ConcreteMeasurementSetup>
struct MeasurementSetupId1y {
  __host__ __device__
  int operator()(
        int const linId, ConcreteMeasurementSetup const * const meas ) {
    return static_cast<ConcreteMSId1y*>(this)->
            operator()(linId, meas);
  }
};

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

