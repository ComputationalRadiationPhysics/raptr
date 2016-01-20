/* 
 * File:   H5ChunkCatalogue.hpp
 * Author: malte
 *
 * Created on 8. Oktober 2015, 16:54
 */

#ifndef H5CHUNKCATALOGUE_HPP
#define	H5CHUNKCATALOGUE_HPP

#include "H5ChunkHandle.hpp"
#include <map>
#include <list>
#include <sstream>

class H5ChunkCatalogue {
public:
  /* Ctor */
  H5ChunkCatalogue( std::string ctrName, size_t ctrMaxNumChunks )
  : name(ctrName), maxNumChunks(ctrMaxNumChunks) {}
  
  /**/
  H5ChunkHandle & getEmptyChunk( size_t chunkId, MemArrSizeType nElems, int nMemRows ) {
    if(chunkMap.size()==maxNumChunks) {
      drop();
    }
    
    /* Create chunk */
    H5ChunkHandle chunk;
    
    /* Create the file */
    std::stringstream fn("");
    fn << name << "_" << count;
    count++;
    chunk.file = H5::H5File(fn.str().c_str(), H5F_ACC_TRUNC);
    
    /* Create data spaces */
    hsize_t valDims[1]    = {hsize_t(nElems)};
    H5::DataSpace valDataSpace(   1, valDims);
    hsize_t rowPtrDims[1] = {hsize_t(nMemRows+1)};
    H5::DataSpace rowPtrDataSpace(1, rowPtrDims);
    
    /* Create data sets */
    chunk.file.createDataSet("val",    H5::PredType::NATIVE_FLOAT, valDataSpace);
    chunk.file.createDataSet("colId",  H5::PredType::NATIVE_INT,   valDataSpace);
    chunk.file.createDataSet("rowPtr", H5::PredType::NATIVE_INT,   rowPtrDataSpace);
    
    /* Create Attributes */
    chunk.file.openDataSet("val").createAttribute("nElems",   H5::PredType::NATIVE_INT, H5S_SCALAR);
    chunk.file.openDataSet("rowPtr").createAttribute("nMemRows", H5::PredType::NATIVE_INT, H5S_SCALAR);
    
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
  H5ChunkHandle & getChunk( size_t chunkId ) {
    return chunkMap.find(chunkId)->second;
  }
  
  
private:
  typedef std::map<size_t, H5ChunkHandle> ChunkMap;
  typedef std::list<ChunkMap::iterator> ChunkHistoryList;
  
  std::string name;
  size_t maxNumChunks;
  size_t count;
  ChunkMap chunkMap;
  ChunkHistoryList chunkHist;
  
  /**/
  void drop( void ) {
    /* Look up oldest chunk */
    ChunkHistoryList::value_type oldestChunkMapElem = *(chunkHist.begin());
    
    /* Erase oldest chunk from map */
    chunkMap.erase(oldestChunkMapElem);
    
    /* Erase oldest chunk from history */
    chunkHist.pop_front();
  }
};

#endif	/* H5CHUNKCATALOGUE_HPP */

