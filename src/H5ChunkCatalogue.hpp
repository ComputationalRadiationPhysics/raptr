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

//template<typename Key, typename T, typename Compare=less<Key>,
//         typename Alloc = allocator<pair<const Key,T> > >
//class DropMap : private std::map<Key, T, Compare, Alloc> {
//public:
//  /* Ctr */
//  DropMap(size_t const ctrMaxNumChunks)
//  : std::map<Key, T, Compare, Alloc>(), maxNumChunks(ctrMaxNumChunks) {}
//  
//  /* Insert a single element */
//  std::pair<typename DropMap::<Key, T, Compare, Alloc>::iterator, bool> insert( typename DropMap::<Key, T, Compare, Alloc>::value_type const & val ) {
//    if(this->size()==maxNumChunks) {
//      drop();
//    }
//    pair<typename DropMap<Key, T, Compare, Alloc>::iterator, bool> newlyInserted = static_cast<map<Key, T, Compare, Alloc>*>(this)->insert(val);
//    l.push_back(newlyInserted.first);
//    
//    return newlyInserted;
//  }
//  
//private:
//  size_t maxNumChunks;
//  std::list<typename DropMap<Key, T, Compare, Alloc>::iterator> l;
//  
//  void drop() {
//    erase(*l.begin());
//    l.erase(l.begin());
//  }
//};

class H5ChunkCatalogue {
public:
  typedef std::map<size_t, H5ChunkHandle> MapType;
  
//  H5ChunkCatalogue(std::string ctrName)
  H5ChunkCatalogue(std::string ctrName, size_t ctrMaxNumChunks)
  : name(ctrName), maxNumChunks(ctrMaxNumChunks) {}
  
  H5ChunkHandle & getEmptyChunk( size_t id, MemArrSizeType nElems, int nMemRows ) {
    if(cat.size()==maxNumChunks) {
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
    
    
    /* Insert */
    cat.insert(std::pair<size_t, H5ChunkHandle>(id, chunk));
    
    return getChunk(id);
  }
  
  bool hasChunk( size_t chunkId ) {
    return (cat.find(chunkId)!=cat.end());
  }

  H5ChunkHandle & getChunk( size_t id ) {
    return cat.find(id)->second;
  }

private:
  std::string name;
  size_t count;
  size_t maxNumChunks;
  MapType cat;
  std::list<MapType::iterator> insertionList;
  
  void drop( void ) {
    cat.erase(*insertionList.begin());
    insertionList.erase(insertionList.begin());
  }
};

#endif	/* H5CHUNKCATALOGUE_HPP */

