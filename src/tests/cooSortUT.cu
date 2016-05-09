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
  BOOST_TEST_MESSAGE( "Running test_cooSort_denseCase_shuffled with ..." );
  BOOST_TEST_MESSAGE( "... seed=" << seed );
  BOOST_TEST_MESSAGE( "... nrows=" << nrows );
  BOOST_TEST_MESSAGE( "... ncols=" << ncols );
  BOOST_TEST_MESSAGE( "... n=" << n );
  
  // Create host vectors
  BOOST_TEST_MESSAGE( "* Create host vectors" );
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
  BOOST_TEST_MESSAGE( "* Create and copy into device vectors" );
  thrust::device_vector<int> val_devi(val_host);
  thrust::device_vector<int> row_devi(row_host);
  thrust::device_vector<int> col_devi(col_host);
  BOOST_TEST_MESSAGE( "* device vector sizes:" );
  BOOST_TEST_MESSAGE( "    val_devi: " << val_devi.size() );
  BOOST_TEST_MESSAGE( "    row_devi: " << row_devi.size() );
  BOOST_TEST_MESSAGE( "    col_devi: " << col_devi.size() );
  
  thrust::zip_iterator<
    thrust::tuple<
      thrust::device_vector<int>::iterator,
      thrust::device_vector<int>::iterator
    >
  > end =
  thrust::make_zip_iterator(thrust::make_tuple(row_devi.data()+1, col_devi.data()+1));
  
  thrust::zip_iterator<
    thrust::tuple<
      thrust::device_vector<int>::iterator,
      thrust::device_vector<int>::iterator
    >
  > begin =
  thrust::make_zip_iterator(thrust::make_tuple(row_devi.data(), col_devi.data()));
  
  thrust::device_ptr<int> val_devi_ptr = val_devi.data();
  
  cooIdLess comp;
  
  BOOST_TEST_MESSAGE( "* col_devi[1]: " << col_devi[1] );
  BOOST_TEST_MESSAGE( "* accessing col_devi[1] as in sort: " << thrust::get<1>(*end) );
  BOOST_TEST_MESSAGE( "* row_devi[1]: " << row_devi[1] );
  BOOST_TEST_MESSAGE( "* accessing row_devi[1] as in sort: " << thrust::get<0>(*end) );
  BOOST_TEST_MESSAGE( "* val_devi[1]: " << val_devi[1] );
  BOOST_TEST_MESSAGE( "* accessing val_devi[1] as in sort: " << val_devi_ptr[1] );
  BOOST_TEST_MESSAGE( "* comp(*begin, *end)= " << comp(*begin, *end) );
 
  
  // Sort
  BOOST_TEST_MESSAGE( "* Sort" );
//  cooSort<int>(thrust::raw_pointer_cast(val_devi.data()),
//               thrust::raw_pointer_cast(row_devi.data()),
//               thrust::raw_pointer_cast(col_devi.data()),
//               n);
  if(cudaGetLastError() != cudaSuccess) {
    BOOST_TEST_MESSAGE( "Cuda Error!" );
  } else {
    BOOST_TEST_MESSAGE( "No cuda Error here" );
  }
    
  thrust::sort_by_key(
    begin,
    end,
    val_devi_ptr,
    cooIdLess()
  );
  
  // Copy back to host
  BOOST_TEST_MESSAGE( "* Copy back to host" );
  val_host = val_devi;
  row_host = row_devi;
  col_host = col_devi;
  
  // Check results
  BOOST_TEST_MESSAGE( "* Check results" );
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
  
  // Create host arrays
  std::vector<int> val_host(n, 0);
  std::vector<int> row_host(n, 0);
  std::vector<int> col_host(n, 0);
  
  std::iota(val_host.begin(), val_host.end(), 0);
  for(auto i=0; i<decltype(i)(val_host.size()); i++) {
    row_host[i] = val_host[i]/ncols;
    col_host[i] = val_host[i]%ncols;
  }
  
  std::shuffle(val_host.begin(), val_host.end(), std::default_random_engine(seed));
  std::shuffle(row_host.begin(), row_host.end(), std::default_random_engine(seed));
  std::shuffle(col_host.begin(), col_host.end(), std::default_random_engine(seed));
  
  std::vector<int> compare_val_host(val_host.begin(), val_host.begin()+nDens);
  val_host = compare_val_host;
  std::vector<int> compare_row_host(row_host.begin(), row_host.begin()+nDens);
  row_host = compare_row_host;
  std::vector<int> compare_col_host(col_host.begin(), col_host.begin()+nDens);
  col_host = compare_col_host;
  
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
  cooSort<int>(val_devi, row_devi, col_devi, nDens);
  
  // Copy back to host
  HANDLE_ERROR(cudaMemcpy(&val_host[0], val_devi, sizeof(val_devi[0]) * n, cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(&row_host[0], row_devi, sizeof(row_devi[0]) * n, cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(&col_host[0], col_devi, sizeof(col_devi[0]) * n, cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaDeviceSynchronize());
  
  // Check results
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
    }
    
    if(row_host[i] == row_host[i+1]){
      BOOST_CHECK(col_host[i] < col_host[i+1]);
    }
  }
  
  // Release memory
  HANDLE_ERROR(cudaFree(val_devi));
  HANDLE_ERROR(cudaFree(row_devi));
  HANDLE_ERROR(cudaFree(col_devi));
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
