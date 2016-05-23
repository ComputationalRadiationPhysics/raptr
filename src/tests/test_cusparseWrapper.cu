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

/** @file test_cusparseWrapper.cu
 * Author: malte
 *
 * Created on 3. Februar 2015, 13:55
 */

#include <cstdlib>
#include <cusparse.h>

#include "CUDA_HandleError.hpp"
#include "CUSPARSE_HandleError.hpp"
#include "cusparseWrapper.hpp"

#define NNZ 9
#define M 4



int main(int argc, char** argv) {
  /* Create sparse COO matrix on host
   *   1      
   * 5   4   6
   *   3   2  
   * 8     7 9
   */
  float cooVal_host[NNZ] = {1, 5, 4, 6, 3, 2, 8, 7, 9};
  int cooRowId_host[NNZ] = {0, 1, 1, 1, 2, 2, 3, 3, 3};
  int cooColId_host[NNZ] = {1, 0, 2, 4, 1, 3, 0, 3, 4};
  
  /* Copy COO matrix to device */
  float * cooVal_devi = NULL;
  HANDLE_ERROR(cudaMalloc((void**)&cooVal_devi, sizeof(cooVal_devi[0]) * NNZ));
  HANDLE_ERROR(cudaMemcpy(cooVal_devi, cooVal_host, sizeof(cooVal_devi[0]) * NNZ, cudaMemcpyHostToDevice));
  int * cooRowId_devi = NULL;
  HANDLE_ERROR(cudaMalloc((void**)&cooRowId_devi, sizeof(cooRowId_devi[0]) * NNZ));
  HANDLE_ERROR(cudaMemcpy(cooRowId_devi, cooRowId_host, sizeof(cooRowId_devi[0]) * NNZ, cudaMemcpyHostToDevice));
  int * cooColId_devi = NULL;
  HANDLE_ERROR(cudaMalloc((void**)&cooColId_devi, sizeof(cooColId_devi[0]) * NNZ));
  HANDLE_ERROR(cudaMemcpy(cooColId_devi, cooColId_host, sizeof(cooColId_devi[0]) * NNZ, cudaMemcpyHostToDevice));
  
  
  cusparseHandle_t handle = NULL;
  HANDLE_CUSPARSE_ERROR(
        cusparseCreate(&handle));
  
  cusparseMatDescr_t descr = NULL;
  HANDLE_CUSPARSE_ERROR(
        cusparseCreateMatDescr(&descr));
  HANDLE_CUSPARSE_ERROR(
        customizeMatDescr(descr, handle));
  
  int * csrRowPtr_devi = NULL; 
  HANDLE_ERROR(cudaMalloc((void**)&csrRowPtr_devi, sizeof(csrRowPtr_devi[0]) * (M+1)));
  HANDLE_CUSPARSE_ERROR(
        convertCoo2Csr(csrRowPtr_devi, cooRowId_devi, handle, NNZ, M));
          
  cudaFree(cooVal_devi);
  cudaFree(cooRowId_devi);
  cudaFree(cooColId_devi);
  cusparseDestroy(handle);
  cusparseDestroyMatDescr(descr);
  cudaFree(csrRowPtr_devi);
  
  
  return 0;
}

