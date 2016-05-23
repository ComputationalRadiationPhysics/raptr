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

/** @file csrmv.hpp
 * 
 *  @brief Header file that defines functor for matrix-vector multiplication
 *  with csr matrices.
 */
#ifndef CSRMV_HPP
#define	CSRMV_HPP

#include <cusparse.h>

/**
 * @brief Functor: y = a*op(A)*x + b*y
 * 
 * @tparam T Type of array elements, vector x elements, vector y elements,
 * scalar alpha and scalar beta
 */
template<typename T>
struct CSRmv {
  /**
   * @brief Functor operation
   *
   * @param handle Handle to the cuSPARSE context.
   * @param transA Operation type. Indicates, if matrix is to be used as
   * non-transposed, transposed or conjugated transposed.
   * @param m Number of rows of matrix A.
   * @param n Number of columns of matrix A.
   * @param nnz Number of non-zero elements of matrix A.
   * @param alpha Scalar factor for multiplication.
   * @param descrA Descriptor of matrix A.
   * @param csrValA Array of nnz non-zero elements of matrix A.
   * @param csrRowPtrA Integer array of m+1 elements that contains the start
   * of every row and the end of the last row plus one.
   * @param csrColIdA Integer array of nnz ( = csrRowPtrA(m) - csrRowPtrA(0) )
   * column indices of the nonzero elements of matrix A.
   * @param x Vector of n elements, if non-transposed and m elements if
   * transposed or conjugated transposed.
   * @param beta Scalar factor for multiplication.
   * @param y Vector of m elements, if non-transposed and n elements if
   * transposed or conjugated transposed.
   */
  void operator()(
        cusparseHandle_t handle, cusparseOperation_t transA,
        int m, int n, int nnz, T * const alpha,
        cusparseMatDescr_t descrA,
        T * const csrValA, int * const csrRowPtrA, int * const csrColIdA,
        T * const x, T * const beta, T * const y);
};

/**
 * @brief Template specialization.
 */
template<>
struct CSRmv<float> {
  void operator()(
        cusparseHandle_t handle, cusparseOperation_t transA,
        int m, int n, int nnz, float * const alpha,
        cusparseMatDescr_t descrA,
        float * const csrValA, int * const csrRowPtrA, int * const csrColIdA,
        float * const x, float * const beta, float * const y) {
    cusparseScsrmv(handle, transA, m, n, nnz, alpha, descrA, csrValA,
          csrRowPtrA, csrColIdA, x, beta, y);
  }
};

/**
 * @brief Template specialization.
 */
template<>
struct CSRmv<double> {
  void operator()(
        cusparseHandle_t handle, cusparseOperation_t transA,
        int m, int n, int nnz, double * const alpha,
        cusparseMatDescr_t descrA,
        double * const csrValA, int * const csrRowPtrA, int * const csrColIdA,
        double * const x, double * const beta, double * const y) {
    cusparseDcsrmv(handle, transA, m, n, nnz, alpha, descrA, csrValA,
          csrRowPtrA, csrColIdA, x, beta, y);
  }
};

/**
 * @brief Template specialization.
 */
template<>
struct CSRmv<cuComplex> {
  void operator()(
        cusparseHandle_t handle, cusparseOperation_t transA,
        int m, int n, int nnz, cuComplex * const alpha,
        cusparseMatDescr_t descrA,
        cuComplex * const csrValA, int * const csrRowPtrA,
        int * const csrColIdA, cuComplex * const x, cuComplex * const beta,
        cuComplex * const y) {
    cusparseCcsrmv(handle, transA, m, n, nnz, alpha, descrA, csrValA,
          csrRowPtrA, csrColIdA, x, beta, y);
  }
};

/**
 * @brief Template specialization.
 */
template<>
struct CSRmv<cuDoubleComplex> {
  void operator()(
        cusparseHandle_t handle, cusparseOperation_t transA,
        int m, int n, int nnz, cuDoubleComplex * const alpha,
        cusparseMatDescr_t descrA,
        cuDoubleComplex * const csrValA, int * const csrRowPtrA,
        int * const csrColIdA, cuDoubleComplex * const x,
        cuDoubleComplex * const beta, cuDoubleComplex * const y) {
    cusparseZcsrmv(handle, transA, m, n, nnz, alpha, descrA, csrValA,
          csrRowPtrA, csrColIdA, x, beta, y);
  }
};

#endif	/* CSRMV_HPP */

