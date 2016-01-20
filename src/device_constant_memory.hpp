/**
 * @file device_constant_memory.hpp
 * @brief Header file that defines objects in device constant memory.
 */
#ifndef DEVICE_CONSTANT_MEMORY_H
#define DEVICE_CONSTANT_MEMORY_H

#include "typedefs.hpp"

/**
 * @var VG grid_const
 * @brief Voxel grid definition.
 */
__device__ __constant__ VG grid_const;

/**
 * @var MS setup_const
 * @brief Measurement setup definition.
 */
__device__ __constant__ MS setup_const;

/**
 * @var int nrays_const
 * @brief Number of rays to use in one channel in system matrix calculation.
 */
__device__ __constant__ int nrays_const;

#endif  // #ifndef DEVICE_CONSTANT_MEMORY_H
