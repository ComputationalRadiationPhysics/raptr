/** @file cuda_wrappers.hpp
 *  @author malte
 *
 * Created on 4. Mai 2015, 13:45 */

#ifndef CUDA_WRAPPERS_HPP
#define	CUDA_WRAPPERS_HPP

#include "CUDA_HandleError.hpp"

/** @brief Wrapper function for device memory allocation.
 * @tparam T Type of memory.
 * @param devi Pointer to allocate memory for.
 * @param n Number of elements to allocate memory for. */
template<typename T>
void mallocD(T * & devi, int const n) {
  HANDLE_ERROR(cudaMalloc((void**)&devi, sizeof(devi[0]) * n));
}

/** @brief Wrapper function for memcpy from host to device. Function only
 * returns after a device synchronization.
 * @tparam T Type of memory.
 * @param devi Target memory on device.
 * @param host Source memory on host.
 * @param n Number of elements of type T that are copied. */
template<typename T>
void memcpyH2D(T * const devi, T const * const host, int const n) {
  HANDLE_ERROR(cudaMemcpy(devi, host, sizeof(devi[0]) * n, cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaDeviceSynchronize());
}

/** @brief Wrapper function for memcpy from host to device. Function returns
 * without device synchronization.
 * @tparam T Type of memory.
 * @param devi Target memory on device.
 * @param host Source memory on host.
 * @param n Number of elements of type T that are copied. */
template<typename T>
void memcpyH2DAsync(T * const devi, T const * const host, int const n) {
  HANDLE_ERROR(cudaMemcpy(devi, host, sizeof(devi[0]) * n, cudaMemcpyHostToDevice));
}

/** @brief Wrapper function for memcpy from device to host. Function only
 * returns after a device synchronization.
 * @tparam T Type of memory.
 * @param host Target memory on host.
 * @param devi Source memory on device.
 * @param n Number of elements of type T that are copied. */
template<typename T>
void memcpyD2H(T * const host, T const * const devi, int const n) {
  HANDLE_ERROR(cudaMemcpy(host, devi, sizeof(host[0]) * n, cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaDeviceSynchronize());
}

/** @brief Wrapper function for memcpy from device to host. Function returns 
 * without device synchronization.
 * @tparam T Type of memory.
 * @param host Target memory on host.
 * @param devi Source memory on device.
 * @param n Number of elements of type T that are copied. */
template<typename T>
void memcpyD2HAsync(T * const host, T const * const devi, int const n) {
  HANDLE_ERROR(cudaMemcpy(host, devi, sizeof(host[0]) * n, cudaMemcpyDeviceToHost));
}

/** @brief Wrapper function for memcpy from device to device. Function only
 * returns after a device synchronization.
 * @tparam T Type of memory.
 * @param host Target memory on host.
 * @param devi Source memory on device.
 * @param n Number of elements of type T that are copied. */
template<typename T>
void memcpyD2D(T * const devi1, T const * const devi0, int const n) {
  HANDLE_ERROR(cudaMemcpy(devi1, devi0, sizeof(devi1[0]) * n, cudaMemcpyDeviceToDevice));
  HANDLE_ERROR(cudaDeviceSynchronize());
}

/** @brief Wrapper function for memcpy from device to device. Function returns
 * without device synchronization.
 * @tparam T Type of memory.
 * @param host Target memory on host.
 * @param devi Source memory on device.
 * @param n Number of elements of type T that are copied. */
template<typename T>
void memcpyD2DAsync(T * const devi1, T const * const devi0, int const n) {
  HANDLE_ERROR(cudaMemcpy(devi1, devi0, sizeof(devi1[0]) * n, cudaMemcpyDeviceToDevice));
}

#endif	/* CUDA_WRAPPERS_HPP */

