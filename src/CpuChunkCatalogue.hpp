/* 
 * File:   CpuChunkCatalogue.hpp
 * Author: malte
 *
 * Created on 8. Oktober 2015, 17:01
 */

#ifndef CPUCHUNKCATALOGUE_HPP
#define CPUCHUNKCATALOGUE_HPP

#include "typedefs_val_type.hpp"
#include "typedefs_array_sizes.hpp"
#include "CpuChunkHandle.hpp"
#include "CpuChunkMem.hpp"
#include <vector>
#include <map>
#include <list>

class CpuChunkCatalogue {
public:
  /**/
  CpuChunkCatalogue( size_t ctrMaxNumChunks, MemArrSizeType memNElemSlots, int memNMemRowSlots )
  : maxNumChunks(ctrMaxNumChunks) {
    /* Create vector of the right number of empty CpuChunkMem objects */
    memVec = MemVector(ctrMaxNumChunks, CpuChunkMem());
    
    /* Initialize CpuChunkMem objects in vec, i.e. allocate CPU memory for them */
    for(MemVector::iterator it=memVec.begin(); it<memVec.end(); it++) {
      it->init(memNElemSlots, memNMemRowSlots);
    }
    
    /* Build up list of pointers to available CpuChunkMem objects */
    for(MemVector::iterator it=memVec.begin(); it<memVec.end(); it++) {
      availMem.push_back(&(*it));
    }
  }
  
  /**/
  CpuChunkHandle & getEmptyChunk( size_t chunkId, MemArrSizeType nElems, int nMemRows ) {
    /* Make place if necessary */
    if(availMem.size()==0) {
      drop();
    }
    
    /* Create CpuChunkHandle object in next available CpuChunkMem */
    CpuChunkMem * memSlot = *(availMem.begin());
    CpuChunkHandle chunk(memSlot, nElems, nMemRows);
    
    /* Set used CpuChunkMem to 'unavailable' */
    availMem.pop_front();
    
    /* Insert chunk in map */
    chunkMap.insert(ChunkMap::value_type(chunkId, chunk));
    
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
  CpuChunkHandle & getChunk( size_t chunkId ) {
    return chunkMap.find(chunkId)->second;
  }

private:
  typedef std::list<std::map<size_t, CpuChunkHandle>::iterator> ChunkHistoryList;
  typedef std::list<CpuChunkMem*> AvailMemList;
  typedef std::vector<CpuChunkMem> MemVector;
  typedef std::map<size_t, CpuChunkHandle> ChunkMap;
  size_t maxNumChunks;
  MemVector memVec;
  ChunkMap chunkMap;
  ChunkHistoryList chunkHist;
  AvailMemList availMem;
  
  /**/
  void drop() {
    /* Look up which memory is used by the oldest chunk */
    ChunkHistoryList::value_type oldestChunkMapElem = *(chunkHist.begin());
    CpuChunkMem * memoryBeingFreed = oldestChunkMapElem->second.mem;
    
    /* Erase oldest chunk from map */
    chunkMap.erase(oldestChunkMapElem);
    
    /* Erase oldest chunk from chunkHist */
    chunkHist.pop_front();
    
    /* Assign memory as 'available' */
    availMem.push_back(memoryBeingFreed);
  }
};

#endif	/* CPUCHUNKCATALOGUE_HPP */
