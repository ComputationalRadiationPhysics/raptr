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

/** @file device_constant_memory.hpp
 * 
 *  @brief Header file that defines objects in device constant memory.
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
