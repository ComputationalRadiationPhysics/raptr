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

/* [512 * 1024 * 1024 / 4] (512 MiB of float or int); max # of elems in COO
 * matrix arrays on GPU */
MemArrSizeType const LIMBYTES(512*1024*1024);
MemArrSizeType const LIMNNZ(LIMBYTES/MemArrSizeType(sizeof(val_t)));

/* Max # of channels in COO matrix arrays */
ListSizeType const LIMM(LIMNNZ/VGRIDSIZE);

int main(int argc, char ** argv) {
  /* NUMBER OF RAYS PER CHANNEL */
  int const nrays(atoi(argv[3]));
  HANDLE_ERROR(cudaMemcpyToSymbol(nrays_const, &nrays, sizeof(int)));
  
  // Measurement setup
  DefaultMeasurementSetup<val_t> setup = MS(POS0X, POS1X, NA, N0Z, N0Y, N1Z, N1Y, DA, SEGX, SEGY, SEGZ);
  HANDLE_ERROR(cudaMemcpyToSymbol(setup_const, &setup, sizeof(MS)));
  
  /* VOXEL GRID */
  VG grid = VG(GRIDOX, GRIDOY, GRIDOZ, GRIDDX, GRIDDY, GRIDDZ, GRIDNX, GRIDNY, GRIDNZ);
  HANDLE_ERROR(cudaMemcpyToSymbol(grid_const, &grid, sizeof(grid)));
  
  
  // Parse command line arguments
  std::string const voxelIfn(argv[1]);    // Voxel data file filename
  std::string const ofn(argv[2]);         // Output filename
  
  // Read voxel data
  thrust::host_vector<val_t> voxelData;
  voxelData = readHDF5_Density<val_t>( voxelIfn );
  std::cout << "voxelData.size(): " << voxelData.size() << std::endl;
  
  // Copy voxel data to device
  thrust::device_vector<val_t> voxelData_devi( voxelData );
  
  // Write channel data
  int channelDataNum = setup.na() * setup.n0z() * setup.n0y() * setup.n1z() * setup.n1y();
  std::vector<int> channelData(channelDataNum, 0);
  std::iota(channelData.begin(), channelData.end(), 0);
  std::cout << "channelData: ";
  for(unsigned i=0; i<channelData.size(); i++) std::cout << channelData[i] << ", ";
  std::cout << std::endl;
  
  // Max number of non-zeros in SM to be exspected
  MemArrSizeType maxNnz(MemArrSizeType(channelDataNum) * MemArrSizeType(VGRIDSIZE));
  std::cout << "maxNnz: " << maxNnz << std::endl;
  
  ChunkGridSizeType NChunks(nChunks<ChunkGridSizeType, MemArrSizeType>(maxNnz, MemArrSizeType(LIMM*VGRIDSIZE)));
  if(NChunks>1) {
    exit(EXIT_FAILURE);
  }
  
  // Copy channel data to device
  thrust::device_vector<int> channelData_devi( channelData );

  // Stuff for mv
  cusparseHandle_t handle = NULL; cusparseMatDescr_t A = NULL;
  HANDLE_CUSPARSE_ERROR(cusparseCreate(&handle));
  HANDLE_CUSPARSE_ERROR(cusparseCreateMatDescr(&A));
  HANDLE_CUSPARSE_ERROR(customizeMatDescr(A, handle));
  val_t zero = val_t(0.); val_t one = val_t(1.);
  
  // Create empty system matrix
  thrust::device_vector<int>   aCnlId_devi;
  thrust::device_vector<int>   aCsrCnlPtr_devi;
  thrust::device_vector<int>   aEcsrCnlPtr_devi;
  thrust::device_vector<int>   aVxlId_devi;
  thrust::device_vector<val_t> aVal_devi;
  create_SystemMatrix<
          val_t
  > (
          aCnlId_devi,
          aCsrCnlPtr_devi,
          aEcsrCnlPtr_devi,
          aVxlId_devi,
          aVal_devi,
          NCHANNELS,
          LIMM,
          VGRIDSIZE
  );
  thrust::device_vector<MemArrSizeType> nnz_devi(1, 0);
  thrust::host_vector<MemArrSizeType> nnz_host(1, 0);
  
  MemArrSizeType * nnz_dptr = thrust::raw_pointer_cast(nnz_devi.data());
  int * aCnlId_dptr         = thrust::raw_pointer_cast(aCnlId_devi.data());
  int * aCsrCnlPtr_dptr     = thrust::raw_pointer_cast(aCsrCnlPtr_devi.data());
  int * aEcsrCnlPtr_dptr    = thrust::raw_pointer_cast(aEcsrCnlPtr_devi.data());
  int * aVxlId_dptr         = thrust::raw_pointer_cast(aVxlId_devi.data());
  val_t * aVal_dptr         = thrust::raw_pointer_cast(aVal_devi.data());
  
  
  // Array of cnl ids: Calculate those SM lines
  int * yRowId_dptr = thrust::raw_pointer_cast(channelData_devi.data()); 
  // Number of SM lines to calculate
  ListSizeType m = channelDataNum;
  
  
  std::cout << "0" << std::endl;
  
  
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
  nnz_host = nnz_devi;
  std::cout << "nnz: " << nnz_host[0] << std::endl;
  
  for(int i=0; i<nnz_host[0]; i++) std::cout << aVal_devi[i];
  std::cout << std::endl;
  
  
  
  // Project
  thrust::device_vector<val_t> projection_devi(channelDataNum, 0);
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
  
  
  std::cout << "2" << std::endl;
  
  
  // Copy projection back to host
  thrust::host_vector<val_t> projection( projection_devi );
  std::vector<val_t> std_projection(projection.size(), 0);
  for(unsigned i=0; i<projection.size(); i++) std_projection[i] = projection[i];
  
  
  std::cout << "3" << std::endl;
  
  
  // Write Projection to file
  writeHDF5_MeasVct(std_projection, ofn, setup);

  return 0;
}
