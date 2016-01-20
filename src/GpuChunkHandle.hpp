/* 
 * File:   GpuChunkHandle.hpp
 * Author: malte
 *
 * Created on 8. Oktober 2015, 16:44
 */

#ifndef GPUCHUNKHANDLE_HPP
#define	GPUCHUNKHANDLE_HPP

//#include "typedefs.hpp"
typedef int MemArrSizeType;

struct GpuChunkHandle {
  MemArrSizeType nElems;
  int nMemRows;
  val_t * val;
  int * colId;
  int * rowPtr;
  bool isReady;
};

#endif	/* GPUCHUNKHANDLE_HPP */

