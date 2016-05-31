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

/** @file voxelgrid_defines.h
 *
 *  @brief Header file that defines the voxel grid on which to reconstruct the
 *  activity distribution.
 */

#ifndef VOXELGRID_DEFINES_H
#define	VOXELGRID_DEFINES_H

#ifdef	__cplusplus
extern "C" {
#endif

  
#ifdef GRID64

#define GRIDNX 64                       /** x dimension of voxel grid */
#define GRIDNY 64                       /** y dimension of voxel grid */
#define GRIDNZ 64                       /** z dimension od voxel grid */
#define GRIDOX -0.030                   /** x origin of voxel grid [m] */
#define GRIDOY -0.030                   /** y origin of voxel grid [m] */
#define GRIDOZ -0.030                   /** z origin of voxel grid [m] */
#define GRIDDX  0.0009375               /** x edge length of one voxel [m] */
#define GRIDDY  0.0009375               /** y edge length of one voxel [m] */
#define GRIDDZ  0.0009375               /** z edge length of one voxel [m] */
#define VGRIDSIZE (GRIDNX*GRIDNY*GRIDNZ)/** Number of voxel in the voxel grid */

#else
#ifdef GRID52
  
#define GRIDNX 52                       /** x dimension of voxel grid */
#define GRIDNY 52                       /** y dimension of voxel grid */
#define GRIDNZ 52                       /** z dimension od voxel grid */
#define GRIDOX -0.026                   /** x origin of voxel grid [m] */
#define GRIDOY -0.026                   /** y origin of voxel grid [m] */
#define GRIDOZ -0.026                   /** z origin of voxel grid [m] */
#define GRIDDX  0.001                   /** x edge length of one voxel [m] */
#define GRIDDY  0.001                   /** y edge length of one voxel [m] */
#define GRIDDZ  0.001                   /** z edge length of one voxel [m] */
#define VGRIDSIZE (GRIDNX*GRIDNY*GRIDNZ)/** Number of voxel in the voxel grid */
  
#else
#ifdef GRID32
  
#define GRIDNX 32                       /** x dimension of voxel grid */
#define GRIDNY 32                       /** y dimension of voxel grid */
#define GRIDNZ 32                       /** z dimension od voxel grid */
#define GRIDOX -0.030                   /** x origin of voxel grid [m] */
#define GRIDOY -0.030                   /** y origin of voxel grid [m] */
#define GRIDOZ -0.030                   /** z origin of voxel grid [m] */
#define GRIDDX  0.001875                /** x edge length of one voxel [m] */
#define GRIDDY  0.001875                /** y edge length of one voxel [m] */
#define GRIDDZ  0.001875                /** z edge length of one voxel [m] */
#define VGRIDSIZE (GRIDNX*GRIDNY*GRIDNZ)/** Number of voxel in the voxel grid */
  
#else
#ifdef GRID20

#define GRIDNX 20                       /** x dimension of voxel grid */
#define GRIDNY 20                       /** y dimension of voxel grid */
#define GRIDNZ 20                       /** z dimension od voxel grid */
#define GRIDOX -0.030                   /** x origin of voxel grid [m] */
#define GRIDOY -0.030                   /** y origin of voxel grid [m] */
#define GRIDOZ -0.030                   /** z origin of voxel grid [m] */
#define GRIDDX  0.003                   /** x edge length of one voxel [m] */
#define GRIDDY  0.003                   /** y edge length of one voxel [m] */
#define GRIDDZ  0.003                   /** z edge length of one voxel [m] */
#define VGRIDSIZE (GRIDNX*GRIDNY*GRIDNZ)/** Number of voxel in the voxel grid */

#else
#ifdef GRID10
  
#define GRIDNX 10                       /** x dimension of voxel grid */
#define GRIDNY 10                       /** y dimension of voxel grid */
#define GRIDNZ 10                       /** z dimension od voxel grid */
#define GRIDOX -0.350                   /** x origin of voxel grid [m] */
#define GRIDOY -0.350                   /** y origin of voxel grid [m] */
#define GRIDOZ -0.350                   /** z origin of voxel grid [m] */
#define GRIDDX  0.070                   /** x edge length of one voxel [m] */
#define GRIDDY  0.070                   /** y edge length of one voxel [m] */
#define GRIDDZ  0.070                   /** z edge length of one voxel [m] */
#define VGRIDSIZE (GRIDNX*GRIDNY*GRIDNZ)/** Number of voxel in the voxel grid */

#else
#ifdef GRID128

#define GRIDNX 128                      /** x dimension of voxel grid */
#define GRIDNY 128                      /** y dimension of voxel grid */
#define GRIDNZ 128                      /** z dimension od voxel grid */
#define GRIDOX -0.030                   /** x origin of voxel grid [m] */
#define GRIDOY -0.030                   /** y origin of voxel grid [m] */
#define GRIDOZ -0.030                   /** z origin of voxel grid [m] */
#define GRIDDX  0.00046875              /** x edge length of one voxel [m] */
#define GRIDDY  0.00046875              /** y edge length of one voxel [m] */
#define GRIDDZ  0.00046875              /** z edge length of one voxel [m] */
#define VGRIDSIZE (GRIDNX*GRIDNY*GRIDNZ)/** Number of voxel in the voxel grid */

#else
#ifdef GRID2

#define GRIDNX 2                      /** x dimension of voxel grid */
#define GRIDNY 2                      /** y dimension of voxel grid */
#define GRIDNZ 2                      /** z dimension od voxel grid */
#define GRIDOX -1.                   /** x origin of voxel grid [m] */
#define GRIDOY -1.                   /** y origin of voxel grid [m] */
#define GRIDOZ -1.                   /** z origin of voxel grid [m] */
#define GRIDDX  1.              /** x edge length of one voxel [m] */
#define GRIDDY  1.              /** y edge length of one voxel [m] */
#define GRIDDZ  1.              /** z edge length of one voxel [m] */
#define VGRIDSIZE (GRIDNX*GRIDNY*GRIDNZ)/** Number of voxel in the voxel grid */

#endif /* GRID2 */
#endif /* GRID128 */
#endif /* GRID10 */
#endif /* GRID20 */
#endif /* GRID32 */
#endif /* GRID52 */
#endif /* GRID64 */


#ifdef	__cplusplus
}
#endif

#endif	/* VOXELGRID_DEFINES_H */

