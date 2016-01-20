/** @file typedefs_geometry.hpp */
/* Author: malte
 *
 * Created on 26. November 2015, 11:52
 */

#ifndef TYPEDEFS_GEOMETRY_HPP
#define TYPEDEFS_GEOMETRY_HPP

#include "typedefs_val_type.hpp"

#include "VoxelGrid.hpp"
#include "VoxelGridLinIndex.hpp"
#include "MeasurementSetup.hpp"
#include "MeasurementSetupLinIndex.hpp"
#include "MeasurementSetupTrafo2CartCoord.hpp"
#include "MeasurementList.hpp"

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

#endif	/* TYPEDEFS_GEOMETRY_HPP */

