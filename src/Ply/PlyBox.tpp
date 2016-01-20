#include "PlyBox.hpp"

#include <sstream>

#ifdef DEBUG_PLY
#include <iostream>
#endif


template<typename Vertex>
PlyBox<Vertex>::PlyBox( std::string const name,
                Vertex const o,
                typename Vertex::Coord_t const dx,
                typename Vertex::Coord_t const dy,
                typename Vertex::Coord_t const dz )
: PlyGeometry(name),
  _p0(o),
  _p1(Vertex(o.x+dx, o.y,    o.z)),
  _p2(Vertex(o.x+dx, o.y+dy, o.z)),
  _p3(Vertex(o.x,    o.y+dy, o.z)),
  _p4(Vertex(o.x,    o.y,    o.z+dz)),
  _p5(Vertex(o.x+dx, o.y,    o.z+dz)),
  _p6(Vertex(o.x+dx, o.y+dy, o.z+dz)),
  _p7(Vertex(o.x,    o.y+dy, o.z+dz))
{
#ifdef DEBUG_PLY
  std::cout << "PlyBox<Vertex>::PlyBox(std::string const, Vertex const,"
            << "coord_type const, coord_type const, coord_type const)"
            << std::endl;
#endif
}


template<typename Vertex>
PlyBox<Vertex>::~PlyBox()
{
#ifdef DEBUG_PLY
  std::cout << "PlyBox<Vertex>::~PlyBox()" << std::endl;
#endif
}


template<typename Vertex>
int PlyBox<Vertex>::numVertices()
{
#ifdef DEBUG_PLY
  std::cout << "PlyBox<Vertex>::numVertices()" << std::endl;
#endif
  return 8;
}


template<typename Vertex>
int PlyBox<Vertex>::numFaces()
{
#ifdef DEBUG_PLY
  std::cout << "PlyBox<Vertex>::numFaces()" << std::endl;
#endif
  return 6;
}


template<typename Vertex>
std::string PlyBox<Vertex>::verticesStr()
{
#ifdef DEBUG_PLY
  std::cout << "PlyBox<Vertex>::verticesStr()" << std::endl;
#endif
  std::stringstream ss;
  ss.str() = "";
  ss\
    << _p0.x << " " << _p0.y << " " << _p0.z << " 0 0 0 " << std::endl
    << _p1.x << " " << _p1.y << " " << _p1.z << " 0 0 0 " << std::endl
    << _p2.x << " " << _p2.y << " " << _p2.z << " 0 0 0 " << std::endl
    << _p3.x << " " << _p3.y << " " << _p3.z << " 0 0 0 " << std::endl
    << _p4.x << " " << _p4.y << " " << _p4.z << " 0 0 0 " << std::endl
    << _p5.x << " " << _p5.y << " " << _p5.z << " 0 0 0 " << std::endl
    << _p6.x << " " << _p6.y << " " << _p6.z << " 0 0 0 " << std::endl
    << _p7.x << " " << _p7.y << " " << _p7.z << " 0 0 0 " << std::endl;
  return ss.str();
}


template<typename Vertex>
std::string PlyBox<Vertex>::facesStr()
{
#ifdef DEBUG_PLY
  std::cout << "PlyBox<Vertex>::facesStr()" << std::endl;
#endif
  std::stringstream ss;
  ss.str() = "";
  ss\
    << "4 0 3 7 4 " << std::endl
    << "4 1 2 6 5 " << std::endl
    << "4 0 4 5 1 " << std::endl
    << "4 3 7 6 2 " << std::endl
    << "4 0 1 2 3 " << std::endl
    << "4 4 5 6 7 " << std::endl;
  return ss.str();
}


template<typename Vertex>
std::string PlyBox<Vertex>::facesStr( int & v )
{
#ifdef DEBUG_PLY
  std::cout << "PlyBox<Vertex>::facesStr(int &)" << std::endl;
#endif
  std::stringstream ss;
  ss.str() = "";
  ss\
    << "4 " << v   << " " << v+3 << " " << v+7 << " " << v+4 << std::endl
    << "4 " << v+1 << " " << v+2 << " " << v+6 << " " << v+5 << std::endl
    << "4 " << v   << " " << v+4 << " " << v+5 << " " << v+1 << std::endl
    << "4 " << v+3 << " " << v+7 << " " << v+6 << " " << v+2 << std::endl
    << "4 " << v   << " " << v+1 << " " << v+2 << " " << v+3 << std::endl
    << "4 " << v+4 << " " << v+5 << " " << v+6 << " " << v+7 << std::endl;
  v += 8;
  return ss.str();
}


template<typename Vertex>
Vertex & PlyBox<Vertex>::p0()
{
  return _p0;
}


template<typename Vertex>
Vertex & PlyBox<Vertex>::p1()
{
  return _p1;
}


template<typename Vertex>
Vertex & PlyBox<Vertex>::p2()
{
  return _p2;
}


template<typename Vertex>
Vertex & PlyBox<Vertex>::p3()
{
  return _p3;
}


template<typename Vertex>
Vertex & PlyBox<Vertex>::p4()
{
  return _p4;
}


template<typename Vertex>
Vertex & PlyBox<Vertex>::p5()
{
  return _p5;
}


template<typename Vertex>
Vertex & PlyBox<Vertex>::p6()
{
  return _p6;
}


template<typename Vertex>
Vertex & PlyBox<Vertex>::p7()
{
  return _p7;
}

