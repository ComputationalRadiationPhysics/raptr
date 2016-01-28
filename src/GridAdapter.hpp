/** @file GridAdapter.hpp
 * 
 *  @brief Header file that defines GridAdapter. GridAdapter takes a VoxelGrid
 *  and supplies the interface to make activity data on that VoxelGrid writable
 *  by H5DensityWriter.
 */

#ifndef GRIDADAPTER_HPP
#define	GRIDADAPTER_HPP

#include "H5DensityWriter.hpp"

/**
 * @brief Implements WritableGrid as defined in H5DensityWriter.hpp.
 * 
 * @tparam TVoxelGrid Has to fulfill interface VoxelGrid as defined in
 * VoxelGrid.hpp.
 */
template<typename TVoxelGrid, typename T>
class GridAdapter : public WritableGrid<GridAdapter<TVoxelGrid, T>, WritableGridTraits<T> > {
public:
  GridAdapter(TVoxelGrid const * const grid) 
  : _grid(grid) {}
  
  void getOrigin( T * const origin ) const {
    origin[0] = _grid->gridox();
    origin[1] = _grid->gridoy();
    origin[2] = _grid->gridoz();
  }
  
  void getVoxelSize( T * const voxelSize ) const {
    voxelSize[0] = _grid->griddx();
    voxelSize[1] = _grid->griddy();
    voxelSize[2] = _grid->griddz();
  }
  
  void getNumberOfVoxels( int * const number ) const {
    number[0] = _grid->gridnx();
    number[1] = _grid->gridny();
    number[2] = _grid->gridnz();
  }
  
private:
  TVoxelGrid const * const _grid;
};

#endif	/* GRIDADAPTER_HPP */

