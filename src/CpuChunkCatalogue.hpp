/* 
 * File:   CpuChunkCatalogue.hpp
 * Author: malte
 *
 * Created on 8. Oktober 2015, 17:01
 */

#ifndef CPUCHUNKCATALOGUE_HPP
#define	CPUCHUNKCATALOGUE_HPP

#include "CpuChunkHandle.hpp"

class CpuChunkCatalogue {
public:
  CpuChunkCatalogue(std::string ctrName, size_t ctrMaxNumChunks)
  : name(ctrName), maxNumChunks(ctrMaxNumChunks) {}
  
  CpuChunkHandle & getEmptyChunk( size_t id, MemArrSizeType nElems, int nMemRows ) {
    if(cat.size()==maxNumChunks) {
      emptySlot = drop(dropCandidate);
    } else {
      cat.insert(std::pair<size_t, CpuChunkHandle>(id, CpuChunkHandle()));
      emptySlot = cat.find(id);
    }
    
    /* Create chunk */
    CpuChunkHandle chunk;
    chunk.nElems = nElems;
    chunk.nMemRows = nMemRows;
    chunk.val    = new val_t[nElems];
    chunk.colId  = new int[nElems];
    chunk.rowPtr = new int[nMemRows+1];
    
    /* Insert chunk */
    cat.insert(std::pair<size_t, CpuChunkHandle>(id, chunk));
    
    return getChunk(id);
  }
  
  bool hasChunk( size_t chunkId );
    return (cat.find(chunkId)!=cat.end());
  }
  
  CpuChunkHandle & getChunk( size_t chunkId ) {
    return cat.find(id)->second;
  }

private:
  std::string name;
  size_t makNumChunks;
  std::map<size_t, CpuChunkHandle> cat;
};

#endif	/* CPUCHUNKCATALOGUE_HPP */
