/** @file bigtest.cpp */
#include "Ply.hpp"
//#include "MeasVct.hpp"
#include "Ray.hpp"
#include "TemplateVertex.hpp"
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <iostream>
#include <iomanip>

#define PI 3.1415927

// Seed random numbers
//std::srand(std::time(NULL));


// Define coordinate and vertex types
typedef float                     CoordType;
typedef TemplateVertex<CoordType> VertexType;


// Define ray type
struct MyRayTraits
{
  typedef VertexType Vertex_t;
};

class MyRay : public Ray<MyRay, MyRayTraits>, public PlyLine<MyRayTraits::Vertex_t>
{
  public:
    
    // Parametric Constructor
    MyRay( Vertex_t const p0, Vertex_t const p1 )
    : PlyLine<Vertex_t>(std::string(""),p0,p1) {}
    
    // Default Constructor
    MyRay( void )
    : PlyLine<Vertex_t>(std::string(""),Vertex_t(0.,0.,0.),Vertex_t(0.,0.,0.)) {}

    // Copy Constructor
    MyRay( MyRay const & ori )
    : PlyLine<Vertex_t>(std::string(""),ori._p0,ori._p1) {}

    // Copy Assignment
    void operator=( MyRay const & ori )
    {
      _p0 = ori._p0;
      _p1 = ori._p1;
    }

    Vertex_t start() const  { return PlyLine<Vertex_t>::_p0; }
    
    Vertex_t end() const    { return PlyLine<Vertex_t>::_p1; }
    
    Vertex_t::Coord_t length() const
    {
      return std::sqrt( (_p1.x-_p0.x)*(_p1.x-_p0.x) +
                        (_p1.y-_p0.y)*(_p1.y-_p0.y) +
                        (_p1.z-_p0.z)*(_p1.z-_p0.z)
             );
    }
};


class PlyRepr : public CompositePlyGeometry
{
  public:
    
    // Default Constructor
    PlyRepr()
    : CompositePlyGeometry(std::string("")) {}

    // Copy Constructor
    PlyRepr( PlyRepr const & ori )
    : CompositePlyGeometry(std::string(""))
    {
      _geometryList = ori._geometryList;
    }
};


// Define channel type
class MyChannel
{
  public:
    
    friend class MySetup;

    typedef CoordType Coord_t;


  private:
    
    int _angle;
    Coord_t _pos0[3]; 
    Coord_t _pos1[3];
    int _nrays;
    MyRay * _rays;

    bool _updateRayMemSize( int nrays )
    {
#ifdef DEBUG
      std::cout << "MyChannel::_updateRayMemSize(int)" << std::endl;
#endif
      // Any changes needed?
      if(nrays==_nrays)
        return false;
      
      if(_rays)
        delete[] _rays;
      _rays = new MyRay[nrays];
      _nrays = nrays;

      return true;
    }

   /** 
    * @brief Write the transformation matrix: {random(0.,1.)**3} ->
    * {random point within detector segment} into mem_trafo.
    *
    * The transformation consists of 4 consecutive steps:
    * - translate by (-0.5,-0.5,-0.5) (accounts for segment position==center of
    *   segment box)
    * - scale by (edgelength_x,edgelength_y,edgelength_z)
    * - translate by detector segment basic position
    * - rotate by angle of channel
    * 
    * This can, in homogeneous coordinates, be perceived as:
    *   (R_phi * T_segpos * S_edgelengths * T_{-.5}) * (random_{(0..1)^3},1)
    * 
    * Which can be expressed as:
    *   B_transform                                  * (random_{(0..1)^3},1)
    * 
    * B_transform is referred to as transformation matrix.
    */
    void _trafo(
          Coord_t * const mem_trafo, Coord_t * const edges,
          Coord_t * const pos, Coord_t const sin, Coord_t const cos )
    {
#ifdef DEBUG
      std::cout << "MyChannel::_trafo(Coord_t * const, Coord_t * const, Coord_t"
                << " * const, Coord_t, Coord_t)" << std::endl;
#endif
      mem_trafo[0][0] = edges[0]*cos;
      mem_trafo[0][1] = 0.;
      mem_trafo[0][2] = edges[2]*sin;
      mem_trafo[0][3] = cos*(pos[0]-.5*edges[0])\
                       +sin*(pos[2]-.5*edges[2]);
      mem_trafo[1][0] = 0.;
      mem_trafo[1][1] = edges[1];
      mem_trafo[1][2] = 0.;
      mem_trafo[1][3] = pos[1]-.5*edges[1];
      mem_trafo[2][0] =-edges[0]*sin;
      mem_trafo[2][1] = 0.;
      mem_trafo[2][2] = edges[2]*cos;
      mem_trafo[2][3] =-sin*(pos[0]-.5*edges[0])\
                       +cos*(pos[2]-.5*edges[2]);
    }


  public:
    
    /**
     * Write central position of detector 0 segment in basic position (i.e. for
     * angle = 0 degrees) into mem_pos0
     */
    void getPos0( Coord_t * mem_pos0 )
    {
#ifdef DEBUG
      std::cout << "MyChannel::getPos0(Coord_t*)" << std::endl;
#endif
      mem_pos0[0] = _pos0[0]; // x [mm]
      mem_pos0[1] = _pos0[1]; // y [mm]
      mem_pos0[2] = _pos0[2]; // z [mm]
    }
    
    /**
     * Write central position of detector 1 segment in basic position (i.e. for
     * angle = 0 degrees) into mem_pos1
     */
    void getPos1( Coord_t * mem_pos1 )
    {
#ifdef DEBUG
      std::cout << "MyChannel::getPos1(Coord_t*)" << std::endl;
#endif
      mem_pos1[0] = _pos1[0];
      mem_pos1[1] = _pos1[1];
      mem_pos1[2] = _pos1[2];
    }

    /**
     * Return angular positon of the detector heads in degrees
     */
    int getAngle()
    {
#ifdef DEBUG
      std::cout << "MyChannel::getAngle()" << std::endl;
#endif
      return _angle;
    }
    
    /**
     * Write the detectors segments' edge lengths into mem_edges
     */
    void getEdges( Coord_t * mem_edges )
    {
#ifdef DEBUG
      std::cout << "MyChannel::getEdges(Coord_t*)" << std::endl;
#endif
      mem_edges[0] = 20.;  // x [mm]
      mem_edges[1] = 4.;   // y [mm]
      mem_edges[2] = 4.;   // z [mm]
    }
    
    /**
     * Write a number of randomly chosen rays representing the channel into
     * mem_rays.
     */
    void setRays( int nrays )
    {
#ifdef DEBUG
      std::cout << "MyChannel::setRays(int)" << std::endl;
#endif
      if(!_updateRayMemSize(nrays))
        return;

      // Get positions, edges, angle of det0, det1 segments
      Coord_t pos0[3];
      Coord_t pos1[3];
      Coord_t edges[3];
      getPos0(pos0);
      getPos1(pos1);
      getEdges(edges);
      Coord_t angle = (Coord_t)(getAngle())/180.*PI;
      Coord_t cos = std::cos(angle);
      Coord_t sin = std::sin(angle);

      // Generate transformation matrices for det0, det1
      Coord_t trafo0[3][4];
      Coord_t trafo1[3][4];
      _trafo(trafo0, edges, pos0, sin, cos);
      _trafo(trafo1, edges, pos1, sin, cos);

      // Iterate over rays
      for(int ray_id=0; ray_id<nrays; ray_id++)
      {
        // Generate cartesian randoms in homogeneous coordinates
        Coord_t s[4];
        Coord_t e[4];
        for(int i=0; i<3; i++)
        {
          s[i] = (Coord_t)(rand())/RAND_MAX;
          e[i] = (Coord_t)(rand())/RAND_MAX;
        }
        s[3] = 1.;
        e[3] = 1.;
        
        // Transform to obtain start, end
        Coord_t start[3];
        Coord_t end[3];
        for(int i=0; i<3; i++)
        {
          start[i] = 0.;
          end[i]   = 0.;

          for(int j=0; j<4; j++)
          {
            start[i] += trafo0[i][j] * s[j];
            end[i]   += trafo1[i][j] * e[j];
          }
        }

        // Write ray
        _rays[ray_id] = MyRay(VertexType(start[0],start[1],start[2]),
                              VertexType(end[0],  end[1],  end[2]));
      }
    }

    PlyRepr getPlyRepr()
    {
#ifdef DEBUG
      std::cout << "MyChannel::getPlyRepr" << std::endl;
#endif
      PlyRepr g;

// TODO: Add boxes representing the two detector segments.  Problem: There
//       is no way to rotate PlyBoxes at the moment.      
//      // Get positions, edges, angle of det0, det1 segments
//      Coord_t pos0[3];
//      Coord_t pos1[3];
//      Coord_t edges[3];
//      getPos0(pos0);
//      getPos1(pos1);
//      getEdges(edges);
//      Coord_t angle = (Coord_t)(getAngle())/180.*PI;
//      Coord_t cos = std::cos(angle);
//      Coord_t sin = std::sin(angle);
//
//      // Get transformation matrices
//      Coord_t trafo0[3][4];
//      Coord_t trafo1[3][4];
//      _trafo(trafo0, edges, pos0, sin, cos);
//      _trafo(trafo1, edges, pos1, sin, cos);

      for(int i=0; i<_nrays; i++)
      {
        g.add(&_rays[i]);
      }

      return g;
    }

    // Constructor
    MyChannel( int angle, Coord_t * pos0, Coord_t * pos1 )
    : _angle(angle), _nrays(0), _rays(0)
    {
#ifdef DEBUG
      std::cout << "MyChannel::MyChannel(int,Coord_t*,Coord_t*)" << std::endl;
#endif
      for(int i=0; i<3; i++)
      {
        _pos0[i] = pos0[i];
        _pos1[i] = pos1[i];
      }
    }

    // Default Constructor
    MyChannel( void )
    : _nrays(0), _rays(0)
    {
#ifdef DEBUG
      std::cout << "MyChannel::MyChannel()" << std::endl;
#endif
    }

    // Copy Constructor
    MyChannel( MyChannel const & ori )
    : _angle(ori._angle), _nrays(0), _rays(0)
    {
#ifdef DEBUG
      std::cout << "MyChannel::MyChannel(MyChannel const &)" << std::endl;
#endif
      for(int i=0; i<3; i++)
      {
        _pos0[0] = ori._pos0[i];
        _pos1[0] = ori._pos1[i];
      }

      _updateRayMemSize(ori._nrays);
      for(int i=0; i<_nrays; i++)
      {
        _rays[i] = ori._rays[i];
      }
    }

    // Copy Assignment
    void operator=( MyChannel const & ori )
    {
#ifdef DEBUG
      std::cout << "MyChannel::operator=(MyChannel const &)" << std::endl;
#endif
      _angle = ori._angle;
      
      for(int i=0; i<3; i++)
      {
        _pos0[0] = ori._pos0[i];
        _pos1[0] = ori._pos1[i];
      }
      
      _updateRayMemSize(ori._nrays);
      for(int i=0; i<ori._nrays; i++)
      {
        _rays[i] = ori._rays[i];
      }
    }
};
 

// Define setup type
class MySetup
{
  public:
    
    typedef MyChannel Channel_t;

    MySetup( void ) {};

    int linearChannelIndex(
          int angle, int det0segz, int det0segx, int det1segz, int det1segx )
    {
      return   angle    *13*13*13*13\
             + det0segz *13*13*13\
             + det0segx *13*13\
             + det1segz *13
             + det1segx;
    }

    int getExtentNdims()
    {
      return 5;
    }

    void dimensionalChannelIndex( int i_linear, int * d_index )
    {
      d_index[0] =  i_linear    /(13*13*13*13);
      d_index[1] = (i_linear% 180) /(13*13*13);
      d_index[2] = (i_linear%(180*13))/(13*13);
      d_index[3] = (i_linear%(180*13*13)) /13;
      d_index[4] =  i_linear%(180*13*13*13);
    }

    Channel_t getChannel( int i )
    {
      float_t pos0_[3];
      float_t pos1_[3];
      int d_index[5];
      dimensionalChannelIndex(i, d_index);
      pos0(d_index, pos0_);
      pos1(d_index, pos1_);
      
      return Channel_t(2*d_index[0], pos0_, pos1_);
    }


  private:
    
    void pos0( int * d_index, float_t * mem_pos0 )
    {
      mem_pos0[0] = -457.;
      mem_pos0[1] = (6-d_index[1])*4.;
      mem_pos0[2] = (6-d_index[2])*4.;
    }

    void pos1( int * d_index, float_t * mem_pos1 )
    {
      mem_pos1[0] = 457.;
      mem_pos1[1] = (6-d_index[3])*4.;
      mem_pos1[2] = (6-d_index[4])*4.;
    }
};


int main()
{
  std::cout << "MySetup     aSetup;" << std::endl;
  MySetup     aSetup;

  std::cout << "MyChannel   aChannel = aSetup.getChannel(0);" << std::endl;
  MyChannel   aChannel = aSetup.getChannel(0);

  std::cout << "aChannel.setRays(10);" << std::endl;
  aChannel.setRays(10);

  std::cout << "PlyRepr     aPlyRepr = aChannel.getPlyRepr();" << std::endl;
  PlyRepr     aPlyRepr = aChannel.getPlyRepr();

  std::cout << "PlyWriter   writer(\"bigtest.ply\");" << std::endl;
  PlyWriter   writer("bigtest.ply");

  std::cout << "writer.write(aPlyRepr);" << std::endl;
  writer.write(aPlyRepr);

  return 0;
}


// 
// 
// struct MyMeasVctTraits
// {
//   typedef MySetup Setup_t;
//   typedef float Intensity_t;
// };
// 
// 
// class MyMeasVct : public MeasVct<MyMeasVct, MyMeasVctTraits>
// {
//   public:
//     
//     Channel_t getChannel( int i, Setup_t setup );
//     Intensity_t getIntensity( int i, Setup_t setup );
// };
// 
// MyMeasVct::getChannel( int i, MyMeasVct::Setup_t setup )
// {
//   return setup.getChannel(i);
// }
// 
// MyMeasVct::getIntensity( int i, MyMeasVct::Setup_t setup )
// {
// }
