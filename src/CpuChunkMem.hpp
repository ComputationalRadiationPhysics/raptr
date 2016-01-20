/* 
 * File:   CpuChunkMem.hpp
 * Author: malte
 *
 * Created on 26. November 2015, 11:24
 */

#ifndef CPUCHUNKMEM_HPP
#define CPUCHUNKMEM_HPP

#include <cstdlib>
#include "typedefs_val_type.hpp"
#include "typedefs_array_sizes.hpp"

class CpuChunkMem {
public:
  class InitError {};

  /* Ctor */
  CpuChunkMem()
  : nElemSlots(0), nMemRowSlots(0), valMem(NULL), colIdMem(NULL), rowPtrMem(NULL) {}
  
  /* Ctor */
  CpuChunkMem( MemArrSizeType ctrNElemSlots, int ctrNMemRowSlots )
  : nElemSlots(ctrNElemSlots), nMemRowSlots(ctrNMemRowSlots) {
    valMem    = new val_t[ctrNElemSlots];
    colIdMem  = new int[  ctrNElemSlots];
    rowPtrMem = new int[  ctrNMemRowSlots+1];
  }
  
  /* Dtor */
  ~CpuChunkMem() {
    delete[] valMem;
    delete[] colIdMem;
    delete[] rowPtrMem;
  }
  
  /* Initialization of empty object */
  void init( MemArrSizeType initNElemSlots, int initNMemRowSlots ) {
    if(nElemSlots != 0)   throw InitError();
    if(nMemRowSlots != 0) throw InitError();
    if(valMem != NULL)    throw InitError();
    if(colIdMem != NULL)  throw InitError();
    if(rowPtrMem != NULL) throw InitError();
    
    valMem    = new val_t[initNElemSlots];
    colIdMem  = new int[  initNElemSlots];
    rowPtrMem = new int[  initNMemRowSlots+1];
    nElemSlots   = initNElemSlots;
    nMemRowSlots = initNMemRowSlots;
  }
  
  
  MemArrSizeType nElemSlots;
  int nMemRowSlots;
  val_t * valMem;
  int * colIdMem;
  int * rowPtrMem;
};

#endif	/* CPUCHUNKMEM_HPP */

