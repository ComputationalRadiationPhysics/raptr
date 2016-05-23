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

#include "PlyLine.hpp"

#include <sstream>

#ifdef DEBUG_PLY
#include <iostream>
#endif


template<typename Vertex>
PlyLine<Vertex>::PlyLine( std::string const name, Vertex const p0, Vertex const p1 )
: PlyGeometry(name), _p0(p0), _p1(p1)
{
#ifdef DEBUG_PLY
  std::cout << "PlyLine<Vertex>::PLyLine(std::string const, Vertex const, Vertex const)"
            << std::endl;
#endif
}


template<typename Vertex>
PlyLine<Vertex>::~PlyLine()
{
#ifdef DEBUG_PLY
  std::cout << "PlyLine<Vertex>::~PLyLine()" << std::endl;
#endif
}


template<typename Vertex>
int PlyLine<Vertex>::numVertices()
{
#ifdef DEBUG_PLY
  std::cout << "PlyLine<Vertex>::numVertices()" << std::endl;
#endif
  return 3;
}


template<typename Vertex>
int PlyLine<Vertex>::numFaces()
{
#ifdef DEBUG_PLY
  std::cout << "PlyLine<Vertex>::numFaces()" << std::endl;
#endif
  return 1;
}


template<typename Vertex>
std::string PlyLine<Vertex>::verticesStr()
{
#ifdef DEBUG_PLY
  std::cout << "PlyLine<Vertex>::verticesStr()" << std::endl;
#endif
  std::stringstream ss;
  ss.str() = "";
  ss\
    << _p0.x << " " << _p0.y << " " << _p0.z << " 0 0 0 " << std::endl
    << 0.5*(_p0.x+_p1.x) << " "
    << 0.5*(_p0.y+_p1.y) << " "
    << 0.5*(_p0.z+_p1.z)                     << " 0 0 0 " << std::endl
    << _p1.x << " " << _p1.y << " " << _p1.z << " 0 0 0 " << std::endl;
  return ss.str();
}


template<typename Vertex>
std::string PlyLine<Vertex>::facesStr()
{
#ifdef DEBUG_PLY
  std::cout << "PlyLine<Vertex>::facesStr()" << std::endl;
#endif
  return std::string("3 0 1 2");
}


template<typename Vertex>
std::string PlyLine<Vertex>::facesStr( int & v )
{
#ifdef DEBUG_PLY
  std::cout << "PlyLine<Vertex>::facesStr(int &)" << std::endl;
#endif
  std::stringstream ss;
  ss.str() = "";
  ss\
    << "3 " << v << " " << v+1 << " " << v+2 << std::endl;
  v += 3;
  return ss.str();
}


template<typename Vertex>
Vertex & PlyLine<Vertex>::p0()
{
#ifdef DEBUG_PLY
  std::cout << "PlyLine<Vertex>::p0()" << std::endl;
#endif
  return _p0;
}


template<typename Vertex>
Vertex & PlyLine<Vertex>::p1()
{
#ifdef DEBUG_PLY
  std::cout << "PlyLine<Vertex>::p1()" << std::endl;
#endif
  return _p1;
}
