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

