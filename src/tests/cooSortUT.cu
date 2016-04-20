/** @file cooSortUT.cpp */

// BOOST
#include <boost/test/unit_test.hpp>

// STL
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <random>

// RAPTR
#include "CUDA_HandleError.hpp"
#include "cooSort.hpp"


/*******************************************************************************
 * Test Suites
 ******************************************************************************/
BOOST_AUTO_TEST_SUITE( test_cooSort )

// Seed for shuffling
unsigned const seed = 1234;

// Size of COO matrix
int const nrows = 6;
int const ncols = 6;
int n = nrows*ncols;

/** @brief Test the function cooSort for the scenerio where every coordinate
 * occurs exactly once.
 * @param seed Random seed for shuffling the COO matrix.
 */
void test_cooSort_unique_shuffled( unsigned seed ) {
  // Create host arrays
  std::vector<int> val_host(n, 0);
  std::vector<int> row_host(n, 0);
  std::vector<int> col_host(n, 0);
  
  std::iota(val_host.begin(), val_host.end(), 0);
  for(auto i=0; i<decltype(i)(val_host.size()); i++) {
    row_host[i] = val_host[i]/ncols;
    col_host[i] = val_host[i]%ncols;
  }
  
  std::vector<int> compare_val_host(val_host);
  std::vector<int> compare_row_host(row_host);
  std::vector<int> compare_col_host(col_host);
  
  std::shuffle(val_host.begin(), val_host.end(), std::default_random_engine(seed));
  std::shuffle(row_host.begin(), row_host.end(), std::default_random_engine(seed));
  std::shuffle(col_host.begin(), col_host.end(), std::default_random_engine(seed));
  
  // Create and copy into device arrays
  int * val_devi = NULL;
  HANDLE_ERROR(cudaMalloc((void**)&val_devi, sizeof(val_devi[0]) * n));
  HANDLE_ERROR(cudaMemcpy(val_devi, &val_host[0], sizeof(val_devi[0]) * n, cudaMemcpyHostToDevice));
  int * row_devi = NULL;
  HANDLE_ERROR(cudaMalloc((void**)&row_devi, sizeof(row_devi[0]) * n));
  HANDLE_ERROR(cudaMemcpy(row_devi, &row_host[0], sizeof(row_devi[0]) * n, cudaMemcpyHostToDevice));
  int * col_devi = NULL;
  HANDLE_ERROR(cudaMalloc((void**)&col_devi, sizeof(col_devi[0]) * n));
  HANDLE_ERROR(cudaMemcpy(col_devi, &col_host[0], sizeof(col_devi[0]) * n, cudaMemcpyHostToDevice));
  
  // Sort
  cooSort<int>(val_devi, row_devi, col_devi, n);
  
  // Copy back to host
  HANDLE_ERROR(cudaMemcpy(&val_host[0], val_devi, sizeof(val_devi[0]) * n, cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(&row_host[0], row_devi, sizeof(row_devi[0]) * n, cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(&col_host[0], col_devi, sizeof(col_devi[0]) * n, cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaDeviceSynchronize());
  
  // Check results
  for(int i=0; i<n; i++) {
    BOOST_CHECK_EQUAL(val_host[i], compare_val_host[i]);
    BOOST_CHECK_EQUAL(row_host[i], compare_row_host[i]);
    BOOST_CHECK_EQUAL(col_host[i], compare_col_host[i]);
  }
  
  // Release memory
  HANDLE_ERROR(cudaFree(val_devi));
  HANDLE_ERROR(cudaFree(row_devi));
  HANDLE_ERROR(cudaFree(col_devi));
}

BOOST_AUTO_TEST_CASE( test_cooSort_unique_shuffled_1 ){
  test_cooSort_unique_shuffled( seed );
}

BOOST_AUTO_TEST_SUITE_END() // end test_cooSort

//EOF
