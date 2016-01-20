/**
 * @file typedefs.hpp
 * @brief Header file that globally defines types for easy use in main.
 */
/* 
 * Author: malte
 *
 * Created on 24. Oktober 2014, 11:41
 */

#ifndef TYPEDEFS_HPP
#define	TYPEDEFS_HPP

#include "VoxelGrid.hpp"
#include "VoxelGridLinIndex.hpp"
#include "MeasurementSetup.hpp"
#include "MeasurementSetupLinIndex.hpp"
#include "MeasurementSetupTrafo2CartCoord.hpp"
#include "MeasurementList.hpp"
#include "stdint.h"

/**
 * @var typedef float val_t
 * @brief Type definition for value type.
 */
typedef float                              val_t;

/**
 * @var typedef DefaultVoxelGrid<val_t> VG
 * @brief Type definition for voxel grid type.
 */
typedef DefaultVoxelGrid<val_t>            VG;

/**
 * @var typedef DefaultVoxelGridIdx<VG> Idx
 * @brief Type definition for functor 'voxel index in x direction'.
 */
typedef DefaultVoxelGridIdx<VG>            Idx;

/**
 * @var typedef DefaultVoxelGridIdy<VG> Idy
 * @brief Type definition for functor 'voxel index in y direction'.
 */
typedef DefaultVoxelGridIdy<VG>            Idy;

/**
 * @var typedef  DefaultVoxelGridIdz<VG> Idz
 * @brief Type definition for functor 'voxel index in z direction'.
 */
typedef DefaultVoxelGridIdz<VG>            Idz;

/**
 * @var typedef DefaultMeasurementSetup<val_t> MS
 * @brief Type definition for measurement detup type.
 */
typedef DefaultMeasurementSetup<val_t>     MS;

/**
 * @var typedef DefaultMeasurementSetupId0z<MS> Id0z
 * @brief Type definition for functor 'index in z direction of pixel on first
 * detector'.
 */
typedef DefaultMeasurementSetupId0z<MS>    Id0z;

/**
 * @var typedef DefaultMeasurementSetupId0y<MS> Id0y
 * @brief Type definition for functor 'index in y direction of pixel on first
 * detector'.
 */
typedef DefaultMeasurementSetupId0y<MS>    Id0y;

/**
 * @var typedef DefaultMeasurementSetupId1z<MS> Id1z
 * @brief Type definition for functor 'index in z direction of pixel on second
 * detector'.
 */
typedef DefaultMeasurementSetupId1z<MS>    Id1z;

/**
 * @var typedef DefaultMeasurementSetupId1y<MS> Id1y
 * @brief Type definition for functor 'index in y direction of pixel on second
 * detector'.
 */
typedef DefaultMeasurementSetupId1y<MS>    Id1y;

/**
 * @var typedef DefaultMeasurementSetupIda<MS> Ida
 * @brief Type definition for functor 'angular index'.
 */
typedef DefaultMeasurementSetupIda<MS>     Ida;

/**
 * @var typedef DefaultMeasurementList<val_t> ML
 * @brief Type definition for measurement list type.
 */
typedef DefaultMeasurementList<val_t>      ML;

/**
 * @var typedef DefaultMeasurementSetupTrafo2CartCoordFirstPixel<val_t, MS> Trafo0
 * @brief Type definition for functor 'transformation of uniform coordinates
 * in a particular pixel on the first detector for a particular angular index
 * to global cartesian coordinates'.
 */
typedef DefaultMeasurementSetupTrafo2CartCoordFirstPixel<val_t, MS>  Trafo0;

/**
 * @var typedef DefaultMeasurementSetupTrafo2CartCoordSecndPixel<val_t, MS> Trafo1
 * @brief Type definition for functor 'transformation of uniform coordinates
 * in a particular pixel on the second detector for a particular angular index
 * to global cartesian coordinates'.
 */
typedef DefaultMeasurementSetupTrafo2CartCoordSecndPixel<val_t, MS>  Trafo1;

/**
 * @var typedef DefaultMeasurementSetupTrafo2CartCoordFirstPixel_inplace<val_t, MS> Trafo0_inplace
 * @brief Type definition for functor 'inplace transformation of uniform coordinates
 * in a particular pixel on the first detector for a particular angular index
 * to global cartesian coordinates'.
 */
typedef DefaultMeasurementSetupTrafo2CartCoordFirstPixel_inplace<val_t, MS>  Trafo0_inplace;

/**
 * @var typedef DefaultMeasurementSetupTrafo2CartCoordSecndPixel<val_t, MS> Trafo1
 * @brief Type definition for functor 'inplace transformation of uniform coordinates
 * in a particular pixel on the second detector for a particular angular index
 * to global cartesian coordinates'.
 */
typedef DefaultMeasurementSetupTrafo2CartCoordSecndPixel_inplace<val_t, MS>  Trafo1_inplace;

/**
 * @var typedef uint32_t MemArrSizeType
 * @brief Type definition for a size type. Which is an array size type. Which
 * are arrays that make up sparse representations of matrices. Which are
 * matrices like the system matrix.
 */
typedef unsigned long long int MemArrSizeType;
//typedef uint32_t MemArrSizeType;

/**
 * @var typedef uint32_t ListSizeType
 * @brief Type definition for the size type of measurement lists and sparse
 * measurement vectors.
 */
typedef uint32_t ListSizeType;

/**
 * @var uint32_t ChunkGridSizeType
 * @brief Type definition for the type of the indices of matrix chunks.
 */
typedef uint32_t ChunkGridSizeType;

#endif	/* TYPEDEFS_HPP */
