/**
 * Copyright 2016 Malte Zacharias
 *
 * This file is part of raptr.
 *
 * raptr is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * raptr is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with raptr.
 * If not, see <http://www.gnu.org/licenses/>.
 */

/** @file typedefs_array_sizes.hpp
 * 
 *  @brief Header file: Typedefs for array size types.
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

