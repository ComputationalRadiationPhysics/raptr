/* 
 * File:   CpuChunkHandle.hpp
 * Author: malte
 *
 * Created on 8. Oktober 2015, 16:47
 */

#ifndef CPUCHUNKHANDLE_HPP
#define	CPUCHUNKHANDLE_HPP

//#include "typedefs.hpp"
typedef int MemArrSizeType;
typedef float val_t;

struct CpuChunkHandle {
  MemArrSizeType nElems;
  int nMemRows;
  val_t * val;
  int * colId;
  int * rowPtr;
  bool isReady;
};

#endif	/* CPUCHUNKHANDLE_HPP */

