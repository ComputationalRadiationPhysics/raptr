/** @file typedefs_array_sizes.hpp */
/* Author: malte
 *
 * Created on 26. November 2015, 11:57
 */

#ifndef TYPEDEFS_ARRAY_SIZES_HPP
#define TYPEDEFS_ARRAY_SIZES_HPP

#include "stdint.h"

/**
 * @var typedef uint32_t MemArrSizeType
 * @brief Type definition for a size type. Which is an array size type. Which
 * are arrays that make up sparse representations of matrices. Which are
 * matrices like the system matrix.
 */
typedef unsigned long long int MemArrSizeType;
//typedef uint32_t MemArrSizeType;

/**
 * @var typedef uint32_t ListSizeType
 * @brief Type definition for the size type of measurement lists and sparse
 * measurement vectors.
 */
typedef uint32_t ListSizeType;

/**
 * @var uint32_t ChunkGridSizeType
 * @brief Type definition for the type of the indices of matrix chunks.
 */
typedef uint32_t ChunkGridSizeType;

#endif	/* TYPEDEFS_ARRAY_SIZES_HPP */

