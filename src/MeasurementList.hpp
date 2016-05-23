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

/** @file MeasurementList.hpp
 * 
 *  @brief Header file that defines the MeasurementList template as a sparse
 *  vector format for keeping measurement data and a specialization that is used
 *  in the current implementation of the ePET reconstruction algorithm.
 */
#ifndef MEASUREMENTLIST_HPP
#define	MEASUREMENTLIST_HPP

#include <cstdlib>
#include <iostream>

/**
 * @brief Class template. Interface definition for a sparse vector format.
 * 
 * @tparam T Type of stored values.
 * @tparam ConcreteMeasurementList Type of a specialization of this template.
 */
template<typename T, typename ConcreteMeasurementList>
class MeasurementList {
  public:
    /**
     * @param listId Index into storage.
     * @return Linear measurement channel index of listId'th vector element.
     */
    int cnlId( int const listId ) const {
      return static_cast<ConcreteMeasurementList*>(this)->
              cnlId(listId);
    }
    
    /**
     * @param listId Index into storage.
     * @return Stored value of listId'th vector element.
     */
    T val( int const listId ) const {
      return static_cast<ConcreteMeasurementList*>(this)->
              val(listId);
    }
};

/**
 * @brief Partial template specialization.
 * 
 * @tparam T Type of stored values.
 */
template<typename T>
class DefaultMeasurementList
: public MeasurementList<T, DefaultMeasurementList<T> > {
  public:
    DefaultMeasurementList( int const size )
    : _size(size) {
      _cnlId = (int*)malloc(size*sizeof(int));
      _val   = (T*)  malloc(size*sizeof(T));
    }
    
    DefaultMeasurementList( DefaultMeasurementList const & o )
    : _size(o._size) {
      _cnlId = (int*)malloc(o._size*sizeof(T));
      _val   = (T*)  malloc(o._size*sizeof(T));
      for(int i=0; i<o._size; i++) {
        _cnlId[i] = o.cnlId(i);
        _val[i]   = o.val(i);
      }
    }
    
    ~DefaultMeasurementList() {
      free(_cnlId);
      free(_val);
    }
    
    int cnlId( int const listId ) const {
      return _cnlId[listId];
    }
    
    T val( int const listId ) const {
      return _val[listId];
    }
    
    /**
     * @return Total number of stored vector elements.
     */
    int size() const {
      return _size;
    }
    
    /**
     * @brief Set a stored vector elements' value and channel index.
     * 
     * @param listId Index into storage.
     * @param cnlId Linearized measurement channel index to set for listId'th
     * vector element.
     * @param val Value to set for listId'th vector element.
     */
    void set( int const listId, int const cnlId, T const val ) {
      if(listId>=_size) {
        std::cerr << "DefaultMeasurementList::set(...) : error : listId out of "
                  << " bounds" << std::endl;
        exit(EXIT_FAILURE);
      }
      _cnlId[listId] = cnlId;
      _val[listId]   = val;
    }
  
  private:
    int * _cnlId;
    T *   _val;
    int   _size;
};

#endif	/* MEASUREMENTLIST_HPP */

