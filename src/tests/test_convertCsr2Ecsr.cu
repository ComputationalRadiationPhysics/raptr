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

/** @file test_convertCsr2Ecsr.cu */
#include <iostream>
#include "convertCsr2Ecsr.hpp"
#include "CUDA_HandleError.hpp"

int main() {
  int yRid_host[] = {2, 4, 6};
  int aRptr_host[] = {0, 0, 0, 3, 3, 3, 3, 5};
  
  int * yRid_devi = NULL;
  HANDLE_ERROR(
        cudaMalloc((void**)&yRid_devi, sizeof(yRid_devi[0]) * 3));
  HANDLE_ERROR(
        cudaMemcpy(yRid_devi, yRid_host, sizeof(yRid_devi[0]) * 3, cudaMemcpyHostToDevice));
  int * aRptr_devi = NULL;
  HANDLE_ERROR(
        cudaMalloc((void**)&aRptr_devi, sizeof(aRptr_devi[0]) * 8));
  HANDLE_ERROR(
        cudaMemcpy(aRptr_devi, aRptr_host, sizeof(aRptr_devi[0]) * 8, cudaMemcpyHostToDevice));
   
  int aERptr_host[4];
  int * aERptr_devi = NULL;
  HANDLE_ERROR(
        cudaMalloc((void**)&aERptr_devi, sizeof(aERptr_devi[0]) * 4));
  
  convertCsr2Ecsr(aERptr_devi, yRid_devi, 3, aRptr_devi, 7);
  
  HANDLE_ERROR(
        cudaMemcpy(aERptr_host, aERptr_devi, sizeof(aRptr_devi[0]) * 4, cudaMemcpyDeviceToHost));
  
  for(int i=0; i<4; i++) {
    std::cout << aERptr_host[i] << std::endl;
  }
  
  return 0;
}
