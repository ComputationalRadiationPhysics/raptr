/** @file VoxelGrid.hpp */

#ifndef VOXELGRID_HPP
#define VOXELGRID_HPP

template<typename T, typename ConcreteVoxelGrid>
class VoxelGrid
{
  public:
    
//    VoxelGrid( T const   gridO0, T const   gridO1, T const   gridO2,
//               T const   gridD0, T const   gridD1, T const   gridD2,
//               int const gridN0, int const gridN1, int const gridN2 )
//    {}
    
    __host__ __device__
    T gridox() const
    {
        return static_cast<ConcreteVoxelGrid*>(this)->gridox();
    }

    __host__ __device__
    T gridoy() const
    {
        return static_cast<ConcreteVoxelGrid*>(this)->gridoy();
    }

    __host__ __device__
    T gridoz() const
    {
        return static_cast<ConcreteVoxelGrid*>(this)->gridoz();
    }
    
    __host__ __device__
    T griddx() const
    {
        return static_cast<ConcreteVoxelGrid*>(this)->griddx();
    }

    __host__ __device__
    T griddy() const
    {
        return static_cast<ConcreteVoxelGrid*>(this)->griddy();
    }

    __host__ __device__
    T griddz() const
    {
        return static_cast<ConcreteVoxelGrid*>(this)->griddz();
    }
    
    __host__ __device__
    int gridnx() const
    {
        return static_cast<ConcreteVoxelGrid*>(this)->gridnx();
    }

    __host__ __device__
    int gridny() const
    {
        return static_cast<ConcreteVoxelGrid*>(this)->gridny();
    }

    __host__ __device__
    int gridnz() const
    {
        return static_cast<ConcreteVoxelGrid*>(this)->gridnz();
    }

//    // Deprecated!
//    __host__ __device__
//    int linVoxelId( int const * const sepVoxelId ) const
//    {
//        return static_cast<ConcreteVoxelGrid*>(this)->linVoxelId(sepVoxelId);
//    }
//    
//    // Deprecated!
//    __host__ __device__
//    void sepVoxelId( int * const sepVoxelId, int const linVoxelId ) const
//    {
//        return static_cast<ConcreteVoxelGrid*>(this)->
//                sepVoxelId(sepVoxelId, linVoxelId);
//    }
};



template<typename T>
class DefaultVoxelGrid : public VoxelGrid<T, DefaultVoxelGrid<T> >
{
  private:
    T   gridO[3];
    T   gridD[3];
    int gridN[3];
  
  public:
    
    __host__ __device__
    DefaultVoxelGrid()
    {}
    
    __host__ __device__
    DefaultVoxelGrid( T const   gridO0, T const   gridO1, T const   gridO2,
                      T const   gridD0, T const   gridD1, T const   gridD2,
                      int const gridN0, int const gridN1, int const gridN2 )
//    : VoxelGrid<T, DefaultVoxelGrid<T> >(gridO0, gridO1, gridO2,
//                                         gridD0, gridD1, gridD2,  
//                                         gridN0, gridN1, gridN2)
    {
      gridO[0]=gridO0; gridO[1]=gridO1; gridO[2]=gridO2;
      gridD[0]=gridD0; gridD[1]=gridD1; gridD[2]=gridD2;
      gridN[0]=gridN0; gridN[1]=gridN1; gridN[2]=gridN2;
    }

    __host__ __device__
    DefaultVoxelGrid( DefaultVoxelGrid const & o )
//    : VoxelGrid<T, DefaultVoxelGrid<T> >(o.gridox(), o.gridoy(), o.gridoz(),
//                                         o.griddx(), o.griddy(), o.griddz(),  
//                                         o.gridnx(), o.gridny(), o.gridnz())
    {
      gridO[0]=o.gridox(); gridO[1]=o.gridoy(); gridO[2]=o.gridoz();
      gridD[0]=o.griddx(); gridD[1]=o.griddy(); gridD[2]=o.griddz();
      gridN[0]=o.gridnx(); gridN[1]=o.gridny(); gridN[2]=o.gridnz();
    }
    
    __host__ __device__
    T gridox() const
    {
        return gridO[0];
    }
    
    __host__ __device__
    T gridoy() const
    {
        return gridO[1];
    }
    
    __host__ __device__
    T gridoz() const
    {
        return gridO[2];
    }
    
    __host__ __device__
    T griddx() const
    {
        return gridD[0];
    }
    
    __host__ __device__
    T griddy() const
    {
        return gridD[1];
    }
    
    __host__ __device__
    T griddz() const
    {
        return gridD[2];
    }
    
    __host__ __device__
    int gridnx() const
    {
        return gridN[0];
    }
    
    __host__ __device__
    int gridny() const
    {
        return gridN[1];
    }
    
    __host__ __device__
    int gridnz() const
    {
        return gridN[2];
    }
    
    // Deprecated!
    __host__ __device__
    int linVoxelId( int const * const sepVoxelId ) const
    {
        return   sepVoxelId[0]
               + sepVoxelId[1]*gridN[0]
               + sepVoxelId[2]*gridN[0]*gridN[1];
    }
    
    // Deprecated!
    __host__ __device__
    void sepVoxelId( int * const sepVoxelId, int const linVoxelId ) const
    {
        int tmp = linVoxelId;
        sepVoxelId[2] = tmp/ (gridN[0]*gridN[1]);
        tmp          %=      (gridN[0]*gridN[1]);
        sepVoxelId[1] = tmp/ (gridN[0]);
        tmp          %=      (gridN[0]);
        sepVoxelId[0] = tmp;
    }
};

#endif  // #define VOXELGRID_HPP
