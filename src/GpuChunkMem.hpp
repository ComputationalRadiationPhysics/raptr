/** @file GpuChunkMem.hpp */
/* Author: malte
 *
 * Created on 5. November 2015, 11:47 */

#ifndef GPUCHUNKMEM_HPP
#define	GPUCHUNKMEM_HPP

#include "typedefs.hpp"
#include "cuda_wrappers.hpp"


class GpuChunkMem {
public:
  class InitError {};

  /* Ctor */
  GpuChunkMem()
  : nElemSlots(0), nMemRowSlots(0), valMem(NULL), colIdMem(NULL), rowIdMem(NULL), rowPtrMem(NULL) {}
  
  /* Ctor */
  GpuChunkMem(MemArrSizeType ctrNElemSlots, int ctrNMemRowSlots)
  : nElemSlots(ctrNElemSlots), nMemRowSlots(ctrNMemRowSlots) {
    mallocD<val_t>(valMem,    ctrNElemSlots);
    mallocD<int>(  colIdMem,  ctrNElemSlots);
    mallocD<int>(  rowIdMem,  ctrNElemSlots);
    mallocD<int>(  rowPtrMem, ctrNMemRowSlots+1);
  }
  
  /* Dtor */
  ~GpuChunkMem() {
    cudaFree(valMem);
    cudaFree(colIdMem);
    cudaFree(rowIdMem);
    cudaFree(rowPtrMem);
  }
  
  /* Initialization of empty object */
  void init( MemArrSizeType initNElemSlots, int initNMemRowSlots ) {
    if(nElemSlots != 0)   throw InitError();
    if(nMemRowSlots != 0) throw InitError();
    if(valMem != NULL)    throw InitError();
    if(colIdMem != NULL)  throw InitError();
    if(rowIdMem != NULL)  throw InitError();
    if(rowPtrMem != NULL) throw InitError();
    
    mallocD<val_t>(valMem,    nElemSlots);
    mallocD<int>(  colIdMem,  nElemSlots);
    mallocD<int>(  rowIdMem,  nElemSlots);
    mallocD<int>(  rowPtrMem, nMemRowSlots+1);
    nElemSlots   = initNElemSlots;
    nMemRowSlots = initNMemRowSlots;
  }
  
  
  MemArrSizeType nElemSlots;
  int nMemRowSlots;
  val_t * valMem;
  int * colIdMem;
  int * rowIdMem;
  int * rowPtrMem;
};

#endif	/* GPUCHUNKMEM_HPP */

