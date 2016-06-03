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

/** @file forwardprojection.cu 
 *
 *  @brief Main file that reads voxel data (x) from a file, performs a
 *  forwardprojection (A*x) of that data into measurement space and writes the
 *  resulting measurement data (y = A*x) to another file.
 */

#define NBLOCKS 32

#include "cuda_wrappers.hpp"
#include "wrappers.hpp"
#include "CUDA_HandleError.hpp"
#include "CUSPARSE_HandleError.hpp"
#include "measure_time.hpp"
#include "typedefs.hpp"
#include "csrmv.hpp"
#include "mlemOperations.hpp"
#include "RayGenerators.hpp"

#include <cusparse.h>
#include <sstream>
#include <cstdlib>
#include <fstream>
#include <mpi.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <algorithm>
#include <cassert>

namespace SystemMatrixDeviceLimits {
  
/* [512 * 1024 * 1024 / 4] (512 MiB of float or int); max # of elems in COO
 * matrix arrays on GPU */
MemArrSizeType const LIMBYTES(512*1024*1024);
MemArrSizeType const LIMNNZ(LIMBYTES/MemArrSizeType(sizeof(val_t)));

/* Max # of channels in COO matrix arrays */
ListSizeType const LIMM(LIMNNZ/VGRIDSIZE);

} // SystemMatrixDeviceLimits

void parse_commandline(
        char ** argv,
        std::string & voxelDataInputFilename,
        std::string & forwardprojectionOutputFilename,
        int & nrays ) {
  voxelDataInputFilename = std::string(argv[1]);
  forwardprojectionOutputFilename = std::string(argv[2]);
  nrays = atoi(argv[3]);
}

void set_setup( DefaultMeasurementSetup<val_t> & setup ) {
  setup = MS(POS0X, POS1X, NA, N0Z, N0Y, N1Z, N1Y, DA, SEGX, SEGY, SEGZ);
}

void set_grid( DefaultVoxelGrid<val_t> & grid ) {
  grid = VG(GRIDOX, GRIDOY, GRIDOZ, GRIDDX, GRIDDY, GRIDDZ, GRIDNX, GRIDNY, GRIDNZ);
}

void copy_to_device_constant_memory(
        int const & nrays,
        DefaultMeasurementSetup<val_t> const & setup,
        DefaultVoxelGrid<val_t> const & grid ) {
  HANDLE_ERROR(cudaMemcpyToSymbol(nrays_const, &nrays, sizeof(int)));
  HANDLE_ERROR(cudaMemcpyToSymbol(setup_const, &setup, sizeof(MS)));
  HANDLE_ERROR(cudaMemcpyToSymbol(grid_const, &grid, sizeof(grid)));
}

void set_measurement_list_complete(
        thrust::host_vector<int> & measList,
        DefaultMeasurementSetup<val_t> const & setup ) {
  int size = setup.na() * setup.n0z() * setup.n0y() * setup.n1z() * setup.n1y();
  measList = std::vector<int>(size, 0);
  std::iota(measList.begin(), measList.end(), 0);
}

void assert_feasibility ( MemArrSizeType const & maxNnz ) {
  // Calculate number of SM chunks needed
  ChunkGridSizeType NChunks = nChunks<ChunkGridSizeType, MemArrSizeType>(
          maxNnz, MemArrSizeType(SystemMatrixDeviceLimits::LIMM*VGRIDSIZE));
  
  assert(NChunks <= 1);
}
  


int main(int argc, char ** argv) {
  std::string                     voxelIfn;
  std::string                     projectionOfn;
  int                             nrays;
  
  parse_commandline(argv, voxelIfn, projectionOfn, nrays);
  
  DefaultMeasurementSetup<val_t>  setup;
  DefaultVoxelGrid<val_t>         grid;
  
  set_setup(setup);
  set_grid(grid);
  copy_to_device_constant_memory(nrays, setup, grid);
  
  // Read voxel data, copy to device
  thrust::host_vector<val_t>   voxelData = readHDF5_Density<val_t>( voxelIfn );
  thrust::device_vector<val_t> voxelData_devi( voxelData );
  
  std::cout << "voxelData.size(): " << voxelData.size() << std::endl;
  
  // Set measurement list, copy to device
  thrust::host_vector<int>   measList;
  set_measurement_list_complete(measList, setup);
  thrust::device_vector<int> measList_devi( measList );
  
  std::cout << "measList: ";
  for(unsigned i=0; i<measList.size(); i++) std::cout << measList[i] << ", ";
  std::cout << std::endl;
  
  // Calculate maximum possible number of non-zeros in SM
  MemArrSizeType maxNnz = MemArrSizeType(measList.size()) * MemArrSizeType(VGRIDSIZE);
  
  std::cout << "maxNnz: " << maxNnz << std::endl;
  
  // Assert: Running this program on one GPU without looping is feasible 
  assert_feasibility(maxNnz);
  
  /* Initialize cusparse related variables - needed for SM calculation and
   * projection */
  cusparseHandle_t handle = NULL; /* handle to cusparse context */
  cusparseMatDescr_t A = NULL;    /* SM matrix descriptor */
  HANDLE_CUSPARSE_ERROR(cusparseCreate(&handle));
  HANDLE_CUSPARSE_ERROR(cusparseCreateMatDescr(&A));
  HANDLE_CUSPARSE_ERROR(customizeMatDescr(A, handle));
  val_t zero = val_t(0.);
  val_t one = val_t(1.);
  
  // Create empty system matrix
  thrust::device_vector<int>   aCnlId_devi;
  thrust::device_vector<int>   aCsrCnlPtr_devi;
  thrust::device_vector<int>   aEcsrCnlPtr_devi;
  thrust::device_vector<int>   aVxlId_devi;
  thrust::device_vector<val_t> aVal_devi;
  create_SystemMatrix<val_t> (
          aCnlId_devi,
          aCsrCnlPtr_devi,
          aEcsrCnlPtr_devi,
          aVxlId_devi,
          aVal_devi,
          NCHANNELS,
          SystemMatrixDeviceLimits::LIMM,
          VGRIDSIZE);
  
  /*****************************************************************************
   * Prepare variables for SM calculation
   ****************************************************************************/
  
  // System matrix
  int * aCnlId_dptr         = thrust::raw_pointer_cast(aCnlId_devi.data());
  int * aCsrCnlPtr_dptr     = thrust::raw_pointer_cast(aCsrCnlPtr_devi.data());
  int * aEcsrCnlPtr_dptr    = thrust::raw_pointer_cast(aEcsrCnlPtr_devi.data());
  int * aVxlId_dptr         = thrust::raw_pointer_cast(aVxlId_devi.data());
  val_t * aVal_dptr         = thrust::raw_pointer_cast(aVal_devi.data());
  
  // For return: Number of non-zero SM elements
  thrust::host_vector<MemArrSizeType>   nnz_host(1, 0);
  thrust::device_vector<MemArrSizeType> nnz_devi(nnz_host);
  MemArrSizeType * nnz_dptr = thrust::raw_pointer_cast(nnz_devi.data());
  
  // Array of cnl ids: Calculate those SM lines
  int * yRowId_dptr = thrust::raw_pointer_cast(measList_devi.data()); 
  // Number of SM lines to calculate
  ListSizeType m = measList.size();
  
  
  
  // Calculate system matrix
  systemMatrixCalculation<
          val_t,
          ListSizeType,
          int,
          MemArrSizeType,
          RandRayGen<
                  val_t,
                  Trafo0_inplace,
                  Trafo1_inplace
          >
  > (
          aEcsrCnlPtr_dptr, aVxlId_dptr, aVal_dptr,
          nnz_dptr,
          aCnlId_dptr, aCsrCnlPtr_dptr,
          yRowId_dptr, &m,
          handle);
  HANDLE_ERROR(cudaDeviceSynchronize());
  
  // Copy number of non-zero elements to host
  nnz_host = nnz_devi;
  
  std::cout << "nnz: " << nnz_host[0] << std::endl;
  for(MemArrSizeType i=0; i<nnz_host[0]; i++) std::cout << aVal_devi[i];
  std::cout << std::endl;
  
  // Forwardprojection
  thrust::device_vector<val_t> projection_devi(measList.size(), 0);
  CSRmv<
          val_t
  > () ( 
          handle,   // handle to cusparse context
          CUSPARSE_OPERATION_NON_TRANSPOSE, // operation type
          m,        // number of rows of matrix
          VGRIDSIZE,// number of columns of matrix
          nnz_host[0],// number of nnz elements
          &one,   // first mult-scalar
          A,        // matrix descriptor
          aVal_dptr,// matrix values
          aEcsrCnlPtr_dptr, // 
          aVxlId_dptr,  //
          thrust::raw_pointer_cast(voxelData_devi.data()),  //
          &zero,    // second mult-scalar
          thrust::raw_pointer_cast(projection_devi.data())    // 
  );
  HANDLE_ERROR(cudaDeviceSynchronize());
  
  
  
  // Copy forwardprojection result back to host
  thrust::host_vector<val_t> projection( projection_devi );
  
  std::vector<val_t> std_projection(projection.size(), 0);
  for(unsigned i=0; i<projection.size(); i++) std_projection[i] = projection[i];
  
  // Write result to file
  writeHDF5_MeasVct(std_projection, projectionOfn, setup);

  return 0;
}
