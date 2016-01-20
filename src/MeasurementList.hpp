/** @file MeasurementList.hpp */
/* 
 * File:   MeasurementList.hpp
 * Author: malte
 *
 * Created on 18. Oktober 2014, 22:18
 */

#ifndef MEASUREMENTLIST_HPP
#define	MEASUREMENTLIST_HPP

#include <cstdlib>
#include <iostream>

template<typename T, typename ConcreteMeasurementList>
class MeasurementList {
  public:
    int cnlId( int const listId ) const {
      return static_cast<ConcreteMeasurementList*>(this)->
              cnlId(listId);
    }
    
    T val( int const listId ) const {
      return static_cast<ConcreteMeasurementList*>(this)->
              val(listId);
    }
};

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
    
    int size() const {
      return _size;
    }
    
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

