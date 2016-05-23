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

/** @file cooSortUT.cpp */

// BOOST
#include <boost/test/unit_test.hpp>

// STL
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <random>

// THRUST
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/tuple.h>

// RAPTR
#include "Debug.hpp"
#include "CUDA_HandleError.hpp"
#include "cooSort.hpp"


/*******************************************************************************
 * Test Suites
 ******************************************************************************/
BOOST_AUTO_TEST_SUITE( test_cooSort )

/** @brief Seed for shuffling */
unsigned const test_cooSort_seed = 1234;

/** @brief Number of rows in COO matrix */
int const test_cooSort_nrows = 2;

/** @brief Number of columns in COO matrix */
int const test_cooSort_ncols = 2;

/** @brief Density of COO matrix (between 0.0 and 1.0) */
float const test_cooSort_dens = 0.7;

/** @brief Test the function cooSort for the scenerio where every coordinate
 * occurs exactly once (dense matrix).
 * @param seed Random seed for shuffling the COO matrix.
 * @param nrows Number of matrix rows.
 * @param ncols Number of matrix columns.
 */
void test_cooSort_denseCase_shuffled( unsigned const seed, int const nrows, int const ncols ) {
  int const n = nrows*ncols;
#if RAPTR_DEBUG >= RAPTR_DEBUG_MINIMAL
  BOOST_TEST_MESSAGE( "Running test_cooSort_denseCase_shuffled with ..." );
  BOOST_TEST_MESSAGE( "... seed=" << seed );
  BOOST_TEST_MESSAGE( "... nrows=" << nrows );
  BOOST_TEST_MESSAGE( "... ncols=" << ncols );
  BOOST_TEST_MESSAGE( "... n=" << n );
#endif
  
  // Create host vectors
#if RAPTR_DEBUG >= RAPTR_DEBUG_MINIMAL
  BOOST_TEST_MESSAGE( "* Create host vectors" );
#endif
  thrust::host_vector<int> val_host(n, 0);
  thrust::host_vector<int> row_host(n, 0);
  thrust::host_vector<int> col_host(n, 0);
  
  std::iota(val_host.begin(), val_host.end(), 0);
  for(auto i=0; i<decltype(i)(val_host.size()); i++) {
    row_host[i] = val_host[i]/ncols;
    col_host[i] = val_host[i]%ncols;
  }
  
  thrust::host_vector<int> compare_val_host(val_host);
  thrust::host_vector<int> compare_row_host(row_host);
  thrust::host_vector<int> compare_col_host(col_host);
  
  std::shuffle(val_host.begin(), val_host.end(), std::default_random_engine(seed));
  std::shuffle(row_host.begin(), row_host.end(), std::default_random_engine(seed));
  std::shuffle(col_host.begin(), col_host.end(), std::default_random_engine(seed));
  
  // Create and copy into device vectors
#if RAPTR_DEBUG >= RAPTR_DEBUG_MINIMAL
  BOOST_TEST_MESSAGE( "* Create and copy into device vectors" );
#endif
  thrust::device_vector<int> val_devi(val_host);
  thrust::device_vector<int> row_devi(row_host);
  thrust::device_vector<int> col_devi(col_host);
#if RAPTR_DEBUG >= RAPTR_DEBUG_MINIMAL
  BOOST_TEST_MESSAGE( "* device vector sizes:" );
  BOOST_TEST_MESSAGE( "    val_devi: " << val_devi.size() );
  BOOST_TEST_MESSAGE( "    row_devi: " << row_devi.size() );
  BOOST_TEST_MESSAGE( "    col_devi: " << col_devi.size() );
#endif
  
  // Sort
#if RAPTR_DEBUG >= RAPTR_DEBUG_MINIMAL
  if(cudaGetLastError() != cudaSuccess) {
    BOOST_TEST_MESSAGE( "Cuda Error!" );
  } else {
    BOOST_TEST_MESSAGE( "No cuda Error here" );
  }
  BOOST_TEST_MESSAGE( "* Sort" );
#endif
  cooSort<int>(thrust::raw_pointer_cast(val_devi.data()),
               thrust::raw_pointer_cast(row_devi.data()),
               thrust::raw_pointer_cast(col_devi.data()),
               n);
    
  // Copy back to host
#if RAPTR_DEBUG >= RAPTR_DEBUG_MINIMAL
  BOOST_TEST_MESSAGE( "* Copy back to host" );
#endif
  val_host = val_devi;
  row_host = row_devi;
  col_host = col_devi;
  
  // Check results
#if RAPTR_DEBUG >= RAPTR_DEBUG_MINIMAL
  BOOST_TEST_MESSAGE( "* Check results" );
#endif
  for(int i=0; i<n; i++) {
    BOOST_CHECK_EQUAL(val_host[i], compare_val_host[i]);
    BOOST_CHECK_EQUAL(row_host[i], compare_row_host[i]);
    BOOST_CHECK_EQUAL(col_host[i], compare_col_host[i]);
  }
}

/*******************************************************************************
 * Case: Dense matrix, non-zero finite number of shuffled elements.
 ******************************************************************************/
BOOST_AUTO_TEST_CASE( test_cooSort_denseCase_shuffled_1 ){
  // Seed for shuffling
  unsigned const seed = test_cooSort_seed;

  // Size of COO matrix
  int const nrows = test_cooSort_nrows;
  int const ncols = test_cooSort_ncols;

  test_cooSort_denseCase_shuffled( seed, nrows, ncols );
}

/*******************************************************************************
 * Case: Empty "dense" matrix of 0 rows and 0 columns.
 ******************************************************************************/
BOOST_AUTO_TEST_CASE( test_cooSort_denseCase_shuffled_empty ){
  // Seed for shuffling
  unsigned const seed = test_cooSort_seed;

  // Size of COO matrix
  int const nrows = 0;
  int const ncols = 0;

  test_cooSort_denseCase_shuffled( seed, nrows, ncols );  
}

/*******************************************************************************
 * Case: Matrix of 0 elements, NULL pointers
 ******************************************************************************/
BOOST_AUTO_TEST_CASE( test_cooSort_denseCase_shuffled_empty_NULL ){
  int * val_devi = NULL;
  int * row_devi = NULL;
  int * col_devi = NULL;
  int const n = 0;
  cooSort<int>(val_devi, row_devi, col_devi, n);
}


/** @brief Test the function cooSort for the scenerio where every coordinate
 * occurs at max once (sparse matrix).
 * @param seed Random seed for shuffling the COO matrix.
 * @param nrows Number of matrix rows.
 * @param ncols Number of matrix columns.
 * @param dens Density of matrix: Fraction of non-zero elements.
 */
void test_cooSort_sparseCase_shuffled( unsigned const seed, int const nrows, int const ncols, float const dens ) {
  int const n = nrows*ncols;
  int const nDens = int(n*dens);
#if RAPTR_DEBUG >= RAPTR_DEBUG_MINIMAL
  BOOST_TEST_MESSAGE( "Running test_cooSort_sparseCase_shuffled with ..." );
  BOOST_TEST_MESSAGE( "... seed=" << seed );
  BOOST_TEST_MESSAGE( "... nrows=" << nrows );
  BOOST_TEST_MESSAGE( "... ncols=" << ncols );
  BOOST_TEST_MESSAGE( "... n=" << n );
  BOOST_TEST_MESSAGE( "... nDens=" << nDens );
#endif
  
  // Create host vectors
#if RAPTR_DEBUG >= RAPTR_DEBUG_MINIMAL
  BOOST_TEST_MESSAGE( "* Create host vectors" );
#endif
  thrust::host_vector<int> val_host(n, 0);
  thrust::host_vector<int> row_host(n, 0);
  thrust::host_vector<int> col_host(n, 0);
  
  std::iota(val_host.begin(), val_host.end(), 0);
  for(auto i=0; i<decltype(i)(val_host.size()); i++) {
    row_host[i] = val_host[i]/ncols;
    col_host[i] = val_host[i]%ncols;
  }
  
  std::shuffle(val_host.begin(), val_host.end(), std::default_random_engine(seed));
  std::shuffle(row_host.begin(), row_host.end(), std::default_random_engine(seed));
  std::shuffle(col_host.begin(), col_host.end(), std::default_random_engine(seed));
  
  thrust::host_vector<int> compare_val_host(val_host.begin(), val_host.begin()+nDens);
  val_host = compare_val_host;
  thrust::host_vector<int> compare_row_host(row_host.begin(), row_host.begin()+nDens);
  row_host = compare_row_host;
  thrust::host_vector<int> compare_col_host(col_host.begin(), col_host.begin()+nDens);
  col_host = compare_col_host;
  
  // Create and copy into device vectors
#if RAPTR_DEBUG >= RAPTR_DEBUG_MINIMAL
  BOOST_TEST_MESSAGE( "* Create and copy into device vectors" );
#endif
  thrust::device_vector<int> val_devi(val_host);
  thrust::device_vector<int> row_devi(row_host);
  thrust::device_vector<int> col_devi(col_host);
#if RAPTR_DEBUG >= RAPTR_DEBUG_MINIMAL
  BOOST_TEST_MESSAGE( "* device vector sizes:" );
  BOOST_TEST_MESSAGE( "    val_devi: " << val_devi.size() );
  BOOST_TEST_MESSAGE( "    row_devi: " << row_devi.size() );
  BOOST_TEST_MESSAGE( "    col_devi: " << col_devi.size() );
#endif
  
  // Sort
#if RAPTR_DEBUG >= RAPTR_DEBUG_MINIMAL
  if(cudaGetLastError() != cudaSuccess) {
    BOOST_TEST_MESSAGE( "Cuda Error!" );
  } else {
    BOOST_TEST_MESSAGE( "No cuda Error here" );
  }
  BOOST_TEST_MESSAGE( "* Sort" );
#endif
  cooSort<int>(thrust::raw_pointer_cast(val_devi.data()),
               thrust::raw_pointer_cast(row_devi.data()),
               thrust::raw_pointer_cast(col_devi.data()),
               nDens);
  
  // Copy back to host
#if RAPTR_DEBUG >= RAPTR_DEBUG_MINIMAL
  BOOST_TEST_MESSAGE( "* Copy back to host" );
#endif
  val_host = val_devi;
  row_host = row_devi;
  col_host = col_devi;
  
  // Check results
#if RAPTR_DEBUG >= RAPTR_DEBUG_MINIMAL
  BOOST_TEST_MESSAGE( "* Check results" );
#endif
  for(int i=0; i<nDens; i++) {
    bool found = false;
    for(int j=0; j<nDens; j++) {
      if(  (row_host[i] == compare_row_host[j])
         &&(col_host[i] == compare_col_host[j])
         &&(val_host[i] == compare_val_host[j])) {
        found = true;
      }
    }
    BOOST_CHECK(found == true);
    
    if(i<(nDens-1)){
      BOOST_CHECK(row_host[i] <= row_host[i+1]);
    
      if(row_host[i] == row_host[i+1]){
        BOOST_CHECK(col_host[i] < col_host[i+1]);
      }
    }
  }
}

/*******************************************************************************
 * Case: Sparse matrix, non-zero finite number of shuffled elements.
 ******************************************************************************/
BOOST_AUTO_TEST_CASE( test_cooSort_sparseCase_shuffled_1 ){
  // Seed for shuffling
  unsigned const seed = test_cooSort_seed;

  // Size of COO matrix
  int const nrows = test_cooSort_nrows;
  int const ncols = test_cooSort_ncols;

  // Density
  float const dens = test_cooSort_dens;
  
  test_cooSort_sparseCase_shuffled( seed, nrows, ncols, dens );
}

BOOST_AUTO_TEST_SUITE_END() // end BOOST_AUTO_TEST_SUITE( test_cooSort )

//EOF
