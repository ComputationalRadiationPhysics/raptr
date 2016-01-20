/** 
 * @file csrmv.hpp
 */
/*
 * Author: malte
 *
 * Created on 30. Januar 2015, 15:34
 */

#ifndef CSRMV_HPP
#define	CSRMV_HPP

#include <cusparse.h>

template<typename T>
struct CSRmv {
  void operator()(
        cusparseHandle_t handle, cusparseOperation_t transA,
        int m, int n, int nnz, T * const alpha,
        cusparseMatDescr_t descrA,
        T * const csrValA, int * const csrRowPtrA, int * const csrColIdA,
        T * const x, T * const beta, T * const y);
};

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

