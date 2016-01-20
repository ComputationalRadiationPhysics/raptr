/** @file chunkSaving.hpp
 *  @author malte
 *
 *  Created on 13. Juli 2015, 11:47 */

#ifndef CHUNKSAVING_HPP
#define	CHUNKSAVING_HPP

#include "typedefs.hpp"
#include "cuda_wrappers.hpp"

/**
 * @brief Struct: One system matrix chunk in CPU memory.
 */
struct HostChunk {
  MemArrSizeType nElems;
  val_t * val;
  int * colId;
  int * rowPtr;
};

/**
 * @brief Class for saving the system matrix to and loading it from CPU memory.
 */
class ChunkStorage {
public:
  /**
   * @brief Save a system matrix chunk to CPU memory.
   * @param nElems Number of system matrix elements that will be saved.
   * @param nMemRows Number of matrix rows.
   * @param val Ptr to matrix elements values.
   * @param colId Ptr to matrix elements column indices.
   * @param rowPtr Ptr to matrix rowPtrs.
   */
  void saveChunk(MemArrSizeType nElems, int nMemRows, val_t * val, int * colId, int * rowPtr) {
    HostChunk chunk;
    chunk.nElems = nElems;
    chunk.val = new val_t[nElems];
    chunk.colId = new int[nElems];
    chunk.rowPtr = new int[nMemRows+1];
    memcpyD2H(chunk.val, val, nElems);
    memcpyD2H(chunk.colId, colId, nElems);
    memcpyD2H(chunk.rowPtr, rowPtr, nMemRows+1);
    storage.push_back(chunk);
  }
  
  /**
   * @brief Initialize the sequential loading of chunks from CPU memory.
   */
  void initLoadingChunks() {
    it = storage.begin();
  }
  
  /**
   * @brief Load the next system matrix chunk from CPU memory.
   * @param nElems Reference as return: Number of system matrix elements that were loaded.
   * @param nMemRows Number of matrix rows.
   * @param val Ptr to load matrix elements values into.
   * @param colId Ptr to load matrix elements column indices into.
   * @param rowPtr Ptr to load matrix rowPtrs into.
   */
  void loadChunk(MemArrSizeType & nElems, int nMemRows, val_t * val, int * colId, int * rowPtr) {
    nElems = it->nElems;
    memcpyH2D(val, it->val, nElems);
    memcpyH2D(colId, it->colId, nElems);
    memcpyH2D(rowPtr, it->rowPtr, nMemRows+1);
    it++;
  }
  
private:
  std::vector<HostChunk> storage;
  std::vector<HostChunk>::iterator it;
};

#endif	/* CHUNKSAVING_HPP */

