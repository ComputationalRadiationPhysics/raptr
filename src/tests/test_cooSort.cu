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

/**
 * @file test_cooSort.cu
 */
/* 
 * Author: malte
 *
 * Created on 3. Februar 2015, 10:08
 */

#include <iostream>

#include "CUDA_HandleError.hpp"
#include "cooSort.hpp"

int main(int argc, char** argv) {
  // Create host arrays
  int A_host[6] = {1, 3, 2, 2, 1, 3};
  int B_host[6] = {2, 2, 1, 2, 1, 1};
  int C_host[6] = {2, 6, 3, 4, 1, 5};
  
  // Create and copy into device arrays
  int * A_devi = NULL;
  HANDLE_ERROR(cudaMalloc((void**)&A_devi, sizeof(A_devi[0]) * 6));
  HANDLE_ERROR(cudaMemcpy(A_devi, A_host, sizeof(A_devi[0]) * 6, cudaMemcpyHostToDevice));
  int * B_devi = NULL;
  HANDLE_ERROR(cudaMalloc((void**)&B_devi, sizeof(B_devi[0]) * 6));
  HANDLE_ERROR(cudaMemcpy(B_devi, B_host, sizeof(B_devi[0]) * 6, cudaMemcpyHostToDevice));
  int * C_devi = NULL;
  HANDLE_ERROR(cudaMalloc((void**)&C_devi, sizeof(C_devi[0]) * 6));
  HANDLE_ERROR(cudaMemcpy(C_devi, C_host, sizeof(C_devi[0]) * 6, cudaMemcpyHostToDevice));
  
  // Sort
  cooSort(C_devi, A_devi, B_devi, 6);  
  HANDLE_ERROR(cudaDeviceSynchronize());
  
  // Copy back to host
  HANDLE_ERROR(cudaMemcpy(A_host, A_devi, sizeof(A_devi[0]) * 6, cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(B_host, B_devi, sizeof(B_devi[0]) * 6, cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(C_host, C_devi, sizeof(C_devi[0]) * 6, cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaDeviceSynchronize());
  
  // Print results
  for(int i=0; i<6; i++) {
    std::cout << "i: " << i
              << " A: " << A_host[i]
              << " B: " << B_host[i]
              << " C: " << C_host[i] << std::endl;
  }
  
  // Release memory
  HANDLE_ERROR(cudaFree(A_devi));
  HANDLE_ERROR(cudaFree(B_devi));
  HANDLE_ERROR(cudaFree(C_devi));
  
  return 0;
}

