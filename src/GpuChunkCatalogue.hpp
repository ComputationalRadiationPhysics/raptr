/* 
 * File:   GpuChunkCatalogue.hpp
 * Author: malte
 *
 * Created on 8. Oktober 2015, 16:58
 */

#ifndef GPUCHUNKCATALOGUE_HPP
#define	GPUCHUNKCATALOGUE_HPP

#include "GpuChunkHandle.hpp"

class GpuChunkCatalogue {
  GpuChunkHandle & getEmptyChunk( MemArrSizeType nElems, int nMemRows );
  
  bool hasChunk( size_t chunkId );

  GpuChunkHandle & getChunk( size_t chunkId );
};

#endif	/* GPUCHUNKCATALOGUE_HPP */

