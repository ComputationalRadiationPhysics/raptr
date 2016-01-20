/**
 * @file cusparseWrapper.hpp
 */
/*
 * Author: malte
 *
 * Created on 3. Februar 2015, 13:50
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

