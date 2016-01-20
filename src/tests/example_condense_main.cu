/** @file example_condense_main.cu */
/* 
 * File:   example_condense_main.cu
 * Author: malte
 *
 * Created on 26. November 2014, 16:15
 */

#include <cstdlib>
#include <vector>
#include <algorithm>
#include <string>
#include <fstream>

#include "CUDA_HandleError.hpp"
#include "example_condense.h"

using namespace std;

/*
 * 
 */
int main(int argc, char** argv) {
  int const nargs(2);
  if(argc!=nargs+1) {
    std::cerr << "Error: Wrong number of arguments. Exspected " << nargs
              << ":" << std::endl
              << "    output mode (b: binary/c: charakter)" << std::endl
              << "    output filename" << std::endl
              ;
    exit(EXIT_FAILURE);
  }
  std::string const mode(argv[1]);
  std::string const out_fn(argv[2]);
  if((mode!=string("b"))&&(mode!=string("c"))) {
    std::cerr << "Error: Invalid mode specification (b/c)" << std::endl;
    exit(EXIT_FAILURE);
  }
  
  std::vector<val_t> passed_host(SIZE, 0.);
  int   memId_host[1] = {0};
  
  val_t * passed_devi = NULL;
  int *   memId_devi  = NULL;
  HANDLE_ERROR(cudaMalloc((void**)&passed_devi, sizeof(passed_devi[0]) * SIZE));
  HANDLE_ERROR(cudaMalloc((void**)&memId_devi,  sizeof(memId_devi[0])));
  HANDLE_ERROR(cudaMemcpy(passed_devi, &passed_host[0], sizeof(passed_devi[0]) * SIZE, cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(memId_devi,  &memId_host[0],  sizeof(memId_devi[0]),         cudaMemcpyHostToDevice));
  
  condense<<<NBLOCKS, TPB>>>(passed_devi, memId_devi);
  HANDLE_ERROR(cudaDeviceSynchronize());
  
  HANDLE_ERROR(cudaMemcpy(&passed_host[0], passed_devi, sizeof(passed_host[0]) * SIZE, cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(&memId_host[0],  memId_devi,  sizeof(memId_host[0]),         cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaDeviceSynchronize());
  
//  std::cout << "Found: " << *memId_host << std::endl;
//  for(int i=0; i<*memId_host; i++) {
//    std::cout << "passed[" << i 
//              << "]: " << passed_host[i]
//              << ", stuff: " << stuff_host[i]
//              << ", block: " << block_host[i]
//              << std::endl;
//  }
  
  std::sort(passed_host.begin(), passed_host.end());
  std::ofstream out(out_fn.c_str(), std::ofstream::trunc|std::ios_base::binary);
  if(!out.is_open()) {
    std::cerr << "Error: Could not open file " << out_fn << std::endl;
    
    cudaFree(passed_devi);
    cudaFree(memId_devi);
    
    exit(EXIT_FAILURE);
  }
  
  if(mode==string("b")) {
    for(int i=0; i<passed_host.size(); i++) {
      out.write((char*)&passed_host[i], sizeof(passed_host[0]));
    }
  } else {
    for(int i=0; i<passed_host.size(); i++) {
      out << passed_host[i];
    }
  }
  
  cudaFree(passed_devi);
  cudaFree(memId_devi);
  
  return 0;
}

