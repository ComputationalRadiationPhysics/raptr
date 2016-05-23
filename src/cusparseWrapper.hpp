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

/** @file cusparseWrapper.hpp
 * 
 *  @brief Header file that defines cusparse wrapper functions.
 */
#ifndef CUSPARSEWRAPPER_HPP
#define	CUSPARSEWRAPPER_HPP

#include <cusparse.h>

/**
 * @brief Customize a cusparse matrix descriptor.
 * @param descr Descriptor to customize.
 * @param handle Handle to the cuSPARSE library context.
 * @return Return status of last cuSPARSE operation.
 */
cusparseStatus_t customizeMatDescr(
      cusparseMatDescr_t & descr,
      cusparseHandle_t const & handle) {
  cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
  return cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);
}

/**
 * @brief Convert a sparse matrix of COO type to CSR. 
 * @param csrRowPtr Integer array of m+1 elements. After function execution it
 * contains the start of every row and the end of the last row plus one.
 * @param cooRowId Integer array of nnz uncompressed row indices.
 * @param handle Handle to the cuSPARSE library context.
 * @param nnz Number of non-zero elements of the sparse matrix
 * @param m Number of rows of the sparse matrix.
 * @return Return status of last cuSPARSE operation.
 */
cusparseStatus_t convertCoo2Csr(
      int * const csrRowPtr,
      int const * const cooRowId,
      cusparseHandle_t const & handle,
      int const nnz, int const m) {
  return cusparseXcoo2csr(handle, cooRowId, nnz, m, csrRowPtr,
                          CUSPARSE_INDEX_BASE_ZERO);
}

#endif	/* CUSPARSEWRAPPER_HPP */

