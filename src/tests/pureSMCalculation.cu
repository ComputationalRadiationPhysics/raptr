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

/** @file   pureSMCalculation.cu */
/* Author: malte
 *
 * Created on 18. Februar 2015, 14:13 */

#define NBLOCKS 32

#include "cuda_wrappers.hpp"
#include "wrappers.hpp"
#include "CUDA_HandleError.hpp"
#include "CUSPARSE_HandleError.hpp"
#include "measure_time.hpp"
#include "typedefs.hpp"
#include "csrmv.hpp"
#include "RayGenerators.hpp"

#include <mpi.h>
#include <vector>


/* [512 * 1024 * 1024 / 4] (512 MiB of float or int); max # of elems in COO
 * matrix arrays on GPU */
MemArrSizeType const LIMNNZ(134217728);

/* Max # of channels in COO matrix arrays */
ListSizeType const LIMM(LIMNNZ/VGRIDSIZE);

int main(int argc, char** argv) {

#if MEASURE_TIME
  clock_t time1 = clock();
#endif
  
  int mpi_rank;
  int mpi_size;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  
  int const nargs(2);
  if(argc!=nargs+1) {
    std::cerr << "Error: Wrong number of arguments. Exspected: "
              << nargs << ":" << std::endl
              << "  filename of measurement" << std::endl
              << "  number of rays" << std::endl;
    exit(EXIT_FAILURE);
  }
  std::string const fn(argv[1]);
  
  /* NUMBER OF RAYS PER CHANNEL */
  int const nrays(atoi(argv[2]));
  HANDLE_ERROR(cudaMemcpyToSymbol(nrays_const, &nrays, sizeof(int)));
  
  /* MEASUREMENT SETUP */
  MS setup = MS(POS0X, POS1X, NA, N0Z, N0Y, N1Z, N1Y, DA, SEGX, SEGY, SEGZ);
  HANDLE_ERROR(cudaMemcpyToSymbol(setup_const, &setup, sizeof(MS)));
  
  /* VOXEL GRID */
  VG grid = VG(GRIDOX, GRIDOY, GRIDOZ, GRIDDX, GRIDDY, GRIDDZ, GRIDNX, GRIDNY, GRIDNZ);
  HANDLE_ERROR(cudaMemcpyToSymbol(grid_const, &grid, sizeof(grid)));
  
  /* MEASUREMENT LIST */
  /* Number of non-zeros, row indices */
  ListSizeType effM; std::vector<int> yRowId_host;
  
  {
    int tmp_effM(0);
    readHDF5_MeasList<val_t>(yRowId_host, tmp_effM, fn);
    effM = ListSizeType(tmp_effM);
  }
  
  int * yRowId_devi = NULL;
  mallocD_MeasList(yRowId_devi, effM);
  cpyH2DAsync_MeasList(yRowId_devi, &(yRowId_host[0]), effM);
  
  
  /* STUFF FOR MV */
  cusparseHandle_t handle = NULL; cusparseMatDescr_t A = NULL;
  HANDLE_CUSPARSE_ERROR(cusparseCreate(&handle));
  HANDLE_CUSPARSE_ERROR(cusparseCreateMatDescr(&A));
  HANDLE_CUSPARSE_ERROR(customizeMatDescr(A, handle));

  /* MAX NUMBER OF NON_ZEROS IN SYSTEM MATRIX */
  MemArrSizeType maxNnz(MemArrSizeType(effM) * MemArrSizeType(VGRIDSIZE));
  
  /* SYSTEM MATRIX */
  /* Row (channel) ids, row pointers, effective row pointers, column (voxel)
   * ids, values, number of non-zeros (host, devi) */
  int * aCnlId_devi = NULL; int * aCsrCnlPtr_devi = NULL;
  int * aEcsrCnlPtr_devi = NULL; int * aVxlId_devi = NULL;
  val_t * aVal_devi = NULL;
  mallocD_SystemMatrix<val_t>(aCnlId_devi, aCsrCnlPtr_devi,
        aEcsrCnlPtr_devi, aVxlId_devi, aVal_devi, NCHANNELS, LIMM, VGRIDSIZE);
  MemArrSizeType * nnz_devi = NULL;
  mallocD<MemArrSizeType>(nnz_devi,          1);
  
  /* SM CALCULATION */
  ChunkGridSizeType NChunks(nChunks<ChunkGridSizeType, MemArrSizeType>
        (maxNnz, MemArrSizeType(LIMM)*MemArrSizeType(VGRIDSIZE))
  );

#if MEASURE_TIME
  std::vector<clock_t> chunkTimes;
  chunkTimes.reserve(size_t(NChunks+1));
  chunkTimes.push_back(clock());
  printTimeDiff(*(chunkTimes.end()-1), time1, "Time before SM calculation: ");
#endif /* MEASURE_TIME */

#if DEBUG
  MemArrSizeType totalNnz(0);
  std::cout << "Calculate system matrix with #LOR = " << effM
            << " and #voxels = " << VGRIDSIZE
            << " . Max number of non-zero elements is consequently " << maxNnz
            << " . Partition the system matrix into chunks of maximum #LOR = " << LIMM
            << ", which means " << NChunks
            << " chunks have to be calculated." << std::endl;
#endif

  for(ChunkGridSizeType chunkId =  ChunkGridSizeType(mpi_rank);
                        chunkId <  NChunks;
                        chunkId += ChunkGridSizeType(mpi_size)) {
    ListSizeType m = nInChunk(chunkId, effM, LIMM);
    ListSizeType ptr = chunkPtr(chunkId, LIMM);
    
    MemArrSizeType nnz_host[1] = {0};
    memcpyH2D<MemArrSizeType>(nnz_devi, nnz_host, 1);

    /* Get system matrix */
    systemMatrixCalculation<val_t, ListSizeType, int, MemArrSizeType,
            RandRayGen<val_t, Trafo0_inplace, Trafo1_inplace> > (
          aEcsrCnlPtr_devi, aVxlId_devi, aVal_devi,
          nnz_devi,
          aCnlId_devi, aCsrCnlPtr_devi,
          &(yRowId_devi[ptr]), &m,
          handle);
    HANDLE_ERROR(cudaDeviceSynchronize());
    memcpyD2H<MemArrSizeType>(nnz_host, nnz_devi, 1);

#if MEASURE_TIME
    chunkTimes.push_back(clock());
    printTimeDiff(*(chunkTimes.end()-1), *(chunkTimes.end()-2), "Time for latest chunk: ");
#endif

#if DEBUG
    totalNnz += nnz_host[0];
    std::cout << "Finished chunk " << chunkId
              << " of " << NChunks
              << ", found " << nnz_host[0] << " elements." << std::endl;
#endif

  }

#if MEASURE_TIME
  clock_t time3 = clock();
  printTimeDiff(time3, *chunkTimes.begin(), "Time for SM calculation: ");
#endif /* MEASURE_TIME */

#if DEBUG
  std::cout << "Found: " << totalNnz << " elements." << std::endl;
#endif
          
  /* Cleanup */
  cudaFree(yRowId_devi);
  cudaFree(aCnlId_devi);
  cudaFree(aVxlId_devi);
  cudaFree(aVal_devi);
  cudaFree(nnz_devi);

  MPI_Finalize();

#if MEASURE_TIME
  clock_t time4 = clock();
  printTimeDiff(time4, time3, "Time after SM calculation: ");
#endif /* MEASURE_TIME */
  
  return 0;
}

