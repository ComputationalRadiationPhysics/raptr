/** @file test_mlemOperations.cu */
/* Author: malte
 *
 * Created on 4. Februar 2015, 14:15 */

#include <iostream>
#include "CUDA_HandleError.hpp"
#include "mlemOperations.hpp"

#define N 10

template<typename T>
cudaError_t malloc_devi(T * & devi, int const n) {
  return cudaMalloc((void**)&devi, sizeof(devi[0]) * n);
}

template<typename T>
cudaError_t memcpy_h2d(T * const devi, T const * const host, int const n) {
  return cudaMemcpy(devi, host, sizeof(devi[0]) * n, cudaMemcpyHostToDevice);
}

template<typename T>
cudaError_t memcpy_d2h(T * const host, T const * const devi, int const n) {
  return cudaMemcpy(host, devi, sizeof(devi[0]) * n, cudaMemcpyDeviceToHost);
}

template<typename T>
cudaError_t memcpy_d2d(T * const devi0, T const * const devi1, int const n) {
  return cudaMemcpy(devi0, devi1, sizeof(devi0[0]) * n, cudaMemcpyDeviceToDevice);
}

typedef float val_t;

int main(int argc, char** argv) {
  /* Create host arrays */
  val_t A_host[N]; val_t B_host[N]; val_t C_host[N]; val_t D_host[N];
  
  /* Fill host arrays */
  for(int i=0; i<N; i++) {
    A_host[i] = (i+1)*(i+1);
    C_host[i] = i+1;
    B_host[i] = 10;
    D_host[i] = 0;
  }
  
  /* Create device arrays */
  val_t * A_devi = NULL;
  HANDLE_ERROR(malloc_devi(A_devi, N));
  val_t * B_devi = NULL;
  HANDLE_ERROR(malloc_devi(B_devi, N));
  val_t * C_devi = NULL;
  HANDLE_ERROR(malloc_devi(C_devi, N));
  val_t * D_devi = NULL;
  HANDLE_ERROR(malloc_devi(D_devi, N));
  
  /* Copy to device arrays */
  HANDLE_ERROR(memcpy_h2d(A_devi, A_host, N));
  HANDLE_ERROR(memcpy_h2d(B_devi, B_host, N));
  HANDLE_ERROR(memcpy_h2d(C_devi, C_host, N));
  HANDLE_ERROR(memcpy_h2d(D_devi, D_host, N));
  
  /* Divides */
  divides<val_t>(D_devi, A_devi, C_devi, N);
  HANDLE_ERROR(cudaDeviceSynchronize());
  
  HANDLE_ERROR(memcpy_d2h(D_host, D_devi, N));
  HANDLE_ERROR(cudaDeviceSynchronize());
  std::cout << "D = A / C = " << std::endl;
  for(int i=0; i<N; i++) { std::cout << D_host[i] << std::endl; }
  std::cout << std::endl;
  
  /* Divides multiplies */
  dividesMultiplies<val_t>(D_devi, A_devi, B_devi, C_devi, N);
  HANDLE_ERROR(cudaDeviceSynchronize());
  
  HANDLE_ERROR(memcpy_d2h(D_host, D_devi, N));
  HANDLE_ERROR(cudaDeviceSynchronize());
  std::cout << "D = A / B * C = " << std::endl;
  for(int i=0; i<N; i++) { std::cout << D_host[i] << std::endl; }
  std::cout << std::endl;

  /* Sum */
  val_t norm = sum<val_t>(D_devi, N);
  HANDLE_ERROR(cudaDeviceSynchronize());
  
  std::cout << "norm =" << std::endl << norm << std::endl << std::endl;
    
  /* Scales */
  scales<val_t>(D_devi, (1./norm), N);
  
  HANDLE_ERROR(memcpy_d2h(D_host, D_devi, N));
  HANDLE_ERROR(cudaDeviceSynchronize());
  std::cout << "D = D * " << 1./norm << " = " << std::endl;
  for(int i=0; i<N; i++) { std::cout << D_host[i] << std::endl; }
  std::cout << std::endl;
  
  cudaFree(A_devi);
  cudaFree(B_devi);
  cudaFree(C_devi);
  cudaFree(D_devi);
  
  return 0;
}

