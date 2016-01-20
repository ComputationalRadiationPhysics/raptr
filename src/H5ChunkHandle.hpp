/* 
 * File:   H5ChunkHandle.hpp
 * Author: malte
 *
 * Created on 8. Oktober 2015, 16:48
 */

#ifndef H5CHUNKHANDLE_HPP
#define	H5CHUNKHANDLE_HPP

#include "typedefs_val_type.hpp"
#include "typedefs_array_sizes.hpp"
#include <H5Cpp.h>

struct H5ChunkHandle {
  H5::H5File file;
};

#endif	/* H5CHUNKHANDLE_HPP */

