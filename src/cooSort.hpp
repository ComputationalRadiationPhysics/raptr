/** 
 * @file cooSort.hpp
 */
/* Author: malte
 *
 * Created on 30. Januar 2015, 15:48
 */

#ifndef COOSORT_HPP
#define	COOSORT_HPP

#include <thrust/sort.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <thrust/iterator/zip_iterator.h>

/**
 * @brief Type definition for a COO matrix index type.
 */
typedef thrust::tuple<int, int> cooId_t;

/**
 * @brief Functor for comparison of two cooId_t. Defines an order of cooId_t
 * objects.
 */
struct cooIdLess : public thrust::binary_function<cooId_t, cooId_t, bool> {
  __host__ __device__
  /**
   * @brief Functor operation.
   * @param lhs Left-hand-side operand.
   * @param rhs Right-hand-side operand.
   * @return Truth value of "lhs comes before rhs".
   */
  bool operator()(cooId_t const & lhs, cooId_t const rhs) const {
    // First elements are compared
    if(thrust::get<0>(lhs) < thrust::get<0>(rhs))
      return true;
    else if(thrust::get<0>(lhs) == thrust::get<0>(rhs))
      // If first elements equal, second elements are compared
      if(thrust::get<1>(lhs) < thrust::get<1>(rhs))
        return true;
    return false;
  }
};

/**
 * @brief Sort COO matrix elements in row major column minor order. In place.
 * @tparam T Type of matrix values.
 * @param cooVal Array of matrix elements.
 * @param cooRowId Array of matrix elements' row indices.
 * @param cooColId Array of matrix elements' column indices.
 */
template<typename T>
__host__
void cooSort(
      T * const   cooVal,
      int * const cooRowId,
      int * const cooColId,
      int const N) {
  // Wrap raw pointers (to make accessible by thrust algorithms)
  thrust::device_ptr<T>   val = thrust::device_pointer_cast(cooVal);
  thrust::device_ptr<int> row = thrust::device_pointer_cast(cooRowId);
  thrust::device_ptr<int> col = thrust::device_pointer_cast(cooColId);
  
  // Sort arrays by key (row, col) according to cooIdLess
  thrust::sort_by_key(
        thrust::make_zip_iterator(thrust::make_tuple(row,   col)),
        thrust::make_zip_iterator(thrust::make_tuple(row+N, col+N)),
        val, cooIdLess()
  );
}

#endif	/* COOSORT_HPP */

