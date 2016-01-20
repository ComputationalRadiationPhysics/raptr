/* 
 * File:   CpuChunkHandle.hpp
 * Author: malte
 *
 * Created on 8. Oktober 2015, 16:47
 */

#ifndef CPUCHUNKHANDLE_HPP
#define CPUCHUNKHANDLE_HPP

#include "typedefs_val_type.hpp"
#include "typedefs_array_sizes.hpp"
#include "CpuChunkMem.hpp"

class CpuChunkHandle {
public:
  class CtrError{};
  
  /* Ctor */
  CpuChunkHandle( CpuChunkMem * ctrMem, MemArrSizeType ctrNElems, int ctrNMemRows ) {
    /* Does chunk fit into mem? */
    if(ctrNElems > ctrMem->nElemSlots) {
      throw CtrError();
    }
    if(ctrNMemRows > ctrMem->nMemRowSlots) {
      throw CtrError();
    }
    
    /* Assign to mem */
    nElems   = ctrNElems;
    nMemRows = ctrNMemRows;
    mem      = ctrMem;
    val      = ctrMem->valMem;
    colId    = ctrMem->colIdMem;
    rowPtr   = ctrMem->rowPtrMem;
  }
  
  /* Dtor - DON'T FREE MEMORY!!! */
  ~CpuChunkHandle() {
  }
  
  MemArrSizeType nElems;
  int nMemRows;
  CpuChunkMem * mem;
  val_t * val;
  int * colId;
  int * rowPtr;
};

#endif	/* CPUCHUNKHANDLE_HPP */

