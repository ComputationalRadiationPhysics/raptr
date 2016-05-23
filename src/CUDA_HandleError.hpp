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

/** @file CUDA_HandleError.hpp
 * 
 *  @brief Auxiliary macro for printing CUDA error messages.
 */
#ifndef CUDA_HANDLE_ERROR
#define CUDA_HANDLE_ERROR

#include <cuda_runtime.h>
#include <iostream>

class cudaException: public std::exception{};

/**
 * Wrapper for CUDA functions. On CUDA error prints error message and exits program.
 * @param err CUDA error object.
 * @param file Part of error message: Name of file in that the error occurred.
 * @param line Part of error message: Line in that the error occurred.
 */
void HandleError( cudaError_t err, const char * file, int line )
{
  if(err != cudaSuccess)
    {
      std::cerr << file << "(" << line << "): error: " << cudaGetErrorString( err ) << std::endl;
      throw cudaException();
    }
}

/**
 * Wrapper macro for HandleError(). Arguments 'file' and 'line' that accord to
 * the place where HANDLE_ERROR() is used are passed to HandleError()
 * automatically.
 */
#define HANDLE_ERROR( err ) { HandleError( err, __FILE__, __LINE__ ); }

#endif  // #ifndef CUDA_HANDLE_ERROR
