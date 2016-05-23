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

#include "PlyRectangle.hpp"

#include <sstream>

#ifdef DEBUG_PLY
#include <iostream>
#endif

template<typename Vertex>
PlyRectangle<Vertex>::PlyRectangle( std::string const name,
                            Vertex const p0,
                            Vertex const p1,
                            Vertex const p2,
                            Vertex const p3 )
: PlyGeometry(name),
  _p0(p0), _p1(p1), _p2(p2), _p3(p3)
{
#ifdef DEBUG_PLY
  std::cout << "PlyRectangle<Vertex>::PlyRectangle(std::string const, Vertex const,"
            << "Vertex const, Vertex const, Vertex const)"
            << std::endl;
#endif
}

template<typename Vertex>
PlyRectangle<Vertex>::~PlyRectangle()
{
#ifdef DEBUG_PLY
  std::cout << "PlyRectangle<Vertex>::~PlyRectangle()" << std::endl;
#endif
}

template<typename Vertex>
int PlyRectangle<Vertex>::numVertices()
{
#ifdef DEBUG_PLY
  std::cout << "PlyRectangle<Vertex>::numVertices()" << std::endl;
#endif
  return 4;
}

template<typename Vertex>
int PlyRectangle<Vertex>::numFaces()
{
#ifdef DEBUG_PLY
  std::cout << "PlyRectangle<Vertex>::numFaces()" << std::endl;
#endif
  return 1;
}

template<typename Vertex>
std::string PlyRectangle<Vertex>::verticesStr()
{
#ifdef DEBUG_PLY
  std::cout << "PlyRectangle<Vertex>::verticesStr()" << std::endl;
#endif
  std::stringstream ss;
  ss.str() = "";
  ss << _p0.x << " " << _p0.y << " " << _p0.z << " 0 0 0 " << std::endl
     << _p1.x << " " << _p1.y << " " << _p1.z << " 0 0 0 " << std::endl
     << _p2.x << " " << _p2.y << " " << _p2.z << " 0 0 0 " << std::endl
     << _p3.x << " " << _p3.y << " " << _p3.z << " 0 0 0 " << std::endl;
  return ss.str();
}

template<typename Vertex>
std::string PlyRectangle<Vertex>::facesStr()
{
#ifdef DEBUG_PLY
  std::cout << "PlyRectangle<Vertex>::facesStr()" << std::endl;
#endif
  std::stringstream ss;
  ss.str() = "";
  ss << "4 0 1 2 3 " << std::endl;
  return ss.str();
}

template<typename Vertex>
std::string PlyRectangle<Vertex>::facesStr( int & vertexId )
{
#ifdef DEBUG_PLY
  std::cout << "PlyRectangle<Vertex>::facesStr(int &)" << std::endl;
#endif
  std::stringstream ss;
  ss.str() = "";
  ss << "4 " << vertexId   << " " << vertexId+1 << " "
             << vertexId+2 << " " << vertexId+3 << std::endl;
  vertexId += 4;
  return ss.str();
}


template<typename Vertex>
Vertex & PlyRectangle<Vertex>::p0()
{
  return _p0;
}

template<typename Vertex>
Vertex & PlyRectangle<Vertex>::p1()
{
  return _p1;
}

template<typename Vertex>
Vertex & PlyRectangle<Vertex>::p2()
{
  return _p2;
}

template<typename Vertex>
Vertex & PlyRectangle<Vertex>::p3()
{
  return _p3;
}

