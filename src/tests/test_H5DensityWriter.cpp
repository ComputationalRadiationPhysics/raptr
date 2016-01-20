/** @file test_H5DensityWriter.cpp */
#include "H5DensityWriter.hpp"

class MyGrid
{
  public:
    
    typedef float CoordType;
    
    MyGrid( float o0, float o1, float o2,
            float v0, float v1, float v2,
            int n0,   int n1,   int n2 )
    {
      _origin[0] = o0; _origin[1] = o1; _origin[2] = o2;
      _voxelSize[0] = v0; _voxelSize[1] = v1; _voxelSize[2] = v2;
      _numberOfVoxels[0] = n0; _numberOfVoxels[1] = n1; _numberOfVoxels[2] = n2;
    }

    void getOrigin( float * origin ) const
    {
      for(int dim=0; dim<3; dim++)
        origin[dim] = _origin[dim];
    }

    void getVoxelSize( float * voxelSize ) const
    {
      for(int dim=0; dim<3; dim++)
        voxelSize[dim] = _voxelSize[dim];
    }

    void getNumberOfVoxels( int * numberOfVoxels ) const
    {
      for(int dim=0; dim<3; dim++)
        numberOfVoxels[dim] = _numberOfVoxels[dim];
    }


  private:
    
    float _origin[3], _voxelSize[3];
    int _numberOfVoxels[3];
};



int main()
{
  /* Specify numbers of voxels in each direction */
  int const nx( 5); int const ny( 6); int const nz( 7);

  /* Specify indices of active voxel in each direction */
  int const idx(1); int const idy(2); int const idz(3);
  
  /* Create grid object */
  MyGrid grid(0.1, 0.2, 0.3,
              1.0, 2.0, 3.0,
              nx,  ny,  nz);
  
  /* Create writer object */
  H5DensityWriter<MyGrid> writer(std::string("test_H5DensityWriter_output.h5"));
  
  /* Allocate memory for voxel data */
  float * mem = new float[nx*ny*nz];
  
  /* Set activity in voxels */
  for(int linId=0; linId<nx*ny*nz; linId++)
    mem[linId] = 0.;

  mem[idx*ny*nz + idy*nz + idz] = 1.;

  /* Write voxel data */
  writer.write(mem, grid);

  /* Free memory and return */
  delete mem;
  return 0;
}
