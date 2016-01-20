/** @file GpuChunkCatalogue.hpp */
/* Author: malte
 *
 * Created on 8. Oktober 2015, 16:58 */

#ifndef GPUCHUNKCATALOGUE_HPP
#define GPUCHUNKCATALOGUE_HPP

#include "typedefs_val_type.hpp"
#include "typedefs_array_sizes.hpp"
#include "cuda_wrappers.hpp"
#include "GpuChunkHandle.hpp"
#include "GpuChunkMem.hpp"
#include <vector>
#include <map>
#include <list>

class GpuChunkCatalogue {
public:
  /* Ctor */
  GpuChunkCatalogue( size_t ctrMaxNumChunks, MemArrSizeType memNElemSlots, int memNMemRowSlots )
  : maxNumChunks(ctrMaxNumChunks) {
    /* Create vector of the right number of empty GpuChunkMem objects */
    memVec = MemVector(ctrMaxNumChunks, GpuChunkMem());
    
    /* Initialize GpuChunkMem objects in vec, i.e. allocate GPU memory for them */
    for(MemVector::iterator it=memVec.begin(); it<memVec.end(); it++) {
      it->init(memNElemSlots, memNMemRowSlots);
    }
    
    /* Build up list of pointers to available GpuChunkMem objects */
    for(MemVector::iterator it=memVec.begin(); it!=memVec.end(); it++) {
      availMem.push_back(&(*it));
    }
  }
  
  /**/
  GpuChunkHandle & getEmptyChunk( size_t chunkId, MemArrSizeType nElems, int nMemRows ) {
    /* Make place if necessary */
    if(availMem.size()==0) {
      drop();
    }
    
    /* Create GpuChunkHandle object in next available GpuChunkMem */
    GpuChunkMem * memSlot = *(availMem.begin());
    GpuChunkHandle chunk(memSlot, nElems, nMemRows);
    
    /* Set used GpuChunkMem to unavailable */
    availMem.pop_front();
    
    /* Insert chunk in map */
    chunkMap.insert(ChunkMap::value_type(chunkId,chunk));
    
    /* Remember chunk as latest inserted one */
    ChunkHistoryList::value_type histEntry = chunkMap.find(chunkId);
    chunkHist.push_back(histEntry);
    
    /* Return handle to empty chunk */
    return histEntry->second;
  }
  
  /**/
  bool hasChunk( size_t chunkId ) {
    return (chunkMap.find(chunkId) != chunkMap.end());
  }
  
  /**/
  GpuChunkHandle & getChunk( size_t chunkId ) {
    return chunkMap.find(chunkId)->second;
  }
  
  /**/
  void clearChunkMem(GpuChunkMem * mem) {
    /* Prepare "clear prototypes" */
    val_t * clearValMem    = new val_t[mem->nElemSlots];
    int *   clearIdMem     = new int[  mem->nElemSlots];
    int *   clearRowPtrMem = new int[  mem->nMemRowSlots+1];
    for(MemArrSizeType i=0; i<mem->nElemSlots; i++) {
      clearValMem[i] = 0.;
      clearIdMem[i]  = 0;
    }
    for(int i=0; i<mem->nMemRowSlots+1; i++) {
      clearRowPtrMem[i] = 0;
    }
    
    /* Copy prototypes to device to clear */
    memcpyH2D<val_t>(mem->valMem, clearValMem, mem->nElemSlots);
    memcpyH2D<int>(  mem->colIdMem, clearIdMem, mem->nElemSlots);
    memcpyH2D<int>(  mem->rowIdMem, clearIdMem, mem->nElemSlots);
    memcpyH2D<int>(  mem->rowPtrMem, clearRowPtrMem, mem->nMemRowSlots+1);
    
    /* Release prototypes */
    delete[] clearValMem;
    delete[] clearIdMem;
    delete[] clearRowPtrMem;
  }
  
  
private:
  typedef std::list<std::map<size_t, GpuChunkHandle>::iterator> ChunkHistoryList;
  typedef std::list<GpuChunkMem*> AvailMemList;
  typedef std::vector<GpuChunkMem> MemVector;
  typedef std::map<size_t, GpuChunkHandle> ChunkMap;
  size_t maxNumChunks;
  MemVector memVec;
  ChunkMap chunkMap;
  ChunkHistoryList chunkHist;
  AvailMemList availMem;
  
  /**/
  void drop() {
    /* Assign oldest chunk's memory as available memory */
    ChunkHistoryList::value_type oldestChunkMapElem = *(chunkHist.begin());
    availMem.push_back(oldestChunkMapElem->second.mem);
    
    /* Erase oldest chunk from map */
    chunkMap.erase(oldestChunkMapElem);
    
    /* Erase oldest chunk from chunkHist */
    chunkHist.pop_front();
  }
};

#endif	/* GPUCHUNKCATALOGUE_HPP */

