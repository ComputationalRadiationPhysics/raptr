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

/** @file convertCsr2Ecsr.hpp
 * 
 *  @brief Header file that defines function for conversion betwenn sparse
 *  matrix formats.
 */
#ifndef CONVERTCSR2ECSR_HPP
#define	CONVERTCSR2ECSR_HPP

#include <thrust/device_ptr.h>

/**
 * @brief Convert a matrix of CSR type to ECSR.
 * @param ecsrRowPtr
 * @param yRowId Array of matrix row ids that refer to rows that (possibly)
 * contain non-zero elements. This defines the effective column vector space.
 * Has length nnzr, which is the dimension of the effective column vector space.
 * @param nnzr Number of non-zero rows. In other words: Dimension of effective
 * column vector space.
 * @param csrRowPtr Array that is part of the definition of the matrix in CSR
 * format. Has length lengthCsrRowPtr which is the dimension of the full column
 * vector space + 1.
 * @param m Number of rows. In other words: Dimension of full column vector
 * space.
 */
//void convertCsr2Ecsr(
//      int * const ecsrRowPtr,
//      int const * const yRowId,
//      int const ner,
//      int const * const csrRowPtr,
//      int const m) {
void convertCsr2Ecsr(
      int * const ecsrRowPtr,
      int * const yRowId,
      int const nnzr,
      int * const csrRowPtr,
      int const m) {
  thrust::device_ptr<int> ee = thrust::device_pointer_cast<int>(ecsrRowPtr);
  thrust::device_ptr<int> cc = thrust::device_pointer_cast<int>(csrRowPtr);
  thrust::device_ptr<int> yy = thrust::device_pointer_cast<int>(yRowId);
  for(int i=0; i<nnzr; i++) {
    ee[i] = cc[yy[i]];
  }
  ee[nnzr] = cc[m];
}

#endif	/* CONVERTCSR2ECSR_HPP */

