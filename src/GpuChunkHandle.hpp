/** @file:   GpuChunkHandle.hpp */
/* Author: malte
 *
 * Created on 8. Oktober 2015, 16:44 */

#ifndef GPUCHUNKHANDLE_HPP
#define	GPUCHUNKHANDLE_HPP

#include "typedefs_val_type.hpp"
#include "typedefs_array_sizes.hpp"
#include "GpuChunkMem.hpp"


class GpuChunkHandle {
public:
  class CtrError {};
  
  /* Ctor */
  GpuChunkHandle( GpuChunkMem * ctrMem, MemArrSizeType ctrNElems, int ctrNMemRows ) {
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
    rowId    = ctrMem->rowIdMem;
    rowPtr   = ctrMem->rowPtrMem;
  }
  
  /* Dtor - DON'T FREE MEMORY!!! */
  ~GpuChunkHandle() {
  }
  
  MemArrSizeType nElems;
  int            nMemRows;
  GpuChunkMem *  mem;
  val_t *        val;
  int *          colId;
  int *          rowId;
  int *          rowPtr;
};

#endif	/* GPUCHUNKHANDLE_HPP */

