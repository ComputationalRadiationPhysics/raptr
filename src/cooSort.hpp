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

/** @file cooSort.hpp
 * 
 *  @brief Header file that defines function for sorting matrices of format
 *  coordinate list to row major column minor ordering.
 */
#ifndef COOSORT_HPP
#define	COOSORT_HPP

// THRUST
#include <thrust/sort.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <thrust/iterator/zip_iterator.h>

// RAPTR
#include "Debug.hpp"

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
    /* First elements are compared */
    if(thrust::get<0>(lhs) < thrust::get<0>(rhs))
      return true;
    else if(thrust::get<0>(lhs) == thrust::get<0>(rhs))
      /* If first elements equal, second elements are compared */
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
 * @param N Number of matrix elements.
 */
template<typename T>
__host__
void cooSort(
      T * const   cooVal,
      int * const cooRowId,
      int * const cooColId,
      int const N) {
  /* Wrap raw pointers (to make accessible by thrust algorithms) */
#if RAPTR_DEBUG >= RAPTR_DEBUG_MINIMAL
  BOOST_TEST_MESSAGE( "** Wrap raw pointers ..." );
#endif
  thrust::device_ptr<T>   val = thrust::device_pointer_cast(cooVal);
  thrust::device_ptr<int> row = thrust::device_pointer_cast(cooRowId);
  thrust::device_ptr<int> col = thrust::device_pointer_cast(cooColId);
  
  /* Sort arrays by key (row, col) according to cooIdLess */
#if RAPTR_DEBUG >= RAPTR_DEBUG_MINIMAL
  BOOST_TEST_MESSAGE( "** Sort arrays ..." );
#endif
  thrust::sort_by_key(
        thrust::make_zip_iterator(thrust::make_tuple(row,   col)),
        thrust::make_zip_iterator(thrust::make_tuple(row+N, col+N)),
        val, cooIdLess()
  );
}

#endif	/* COOSORT_HPP */
