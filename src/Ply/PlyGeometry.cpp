#include "PlyGeometry.hpp"


#ifdef DEBUG_PLY
#include <iostream>
#endif

PlyGeometry::~PlyGeometry()
{
#ifdef DEBUG_PLY
  std::cout << "PlyGeometry::~PlyGeometry()" << std::endl;
#endif
}

std::string PlyGeometry::name()
{
#ifdef DEBUG_PLY
  std::cout << "PlyGeometry::name()" << std::endl;
#endif
  return _name;
}


int PlyGeometry::numVertices()
{
#ifdef DEBUG_PLY
  std::cout << "PlyGeometry::numVertices()" << std::endl;
#endif
  return 0;
}

int PlyGeometry::numFaces()
{
#ifdef DEBUG_PLY
  std::cout << "PlyGeometry::numFaces()" << std::endl;
#endif
  return 0;
}

std::string PlyGeometry::verticesStr()
{
#ifdef DEBUG_PLY
  std::cout << "PlyGeometry::verticesStr()" << std::endl;
#endif
  return std::string("");
}

std::string PlyGeometry::facesStr()
{
#ifdef DEBUG_PLY
  std::cout << "PlyGeometry::facesStr()" << std::endl;
#endif
  return std::string("");
}

std::string PlyGeometry::facesStr( int & vertexId )
{
#ifdef DEBUG_PLY
  std::cout << "PlyGeometry::facesStr(int &)" << std::endl;
#endif
  return std::string("");
}


void PlyGeometry::add( PlyGeometry * g_ptr )
{
#ifdef DEBUG_PLY
  std::cout << "PlyGeometry::add(PlyGeometry *)" << std::endl;
#endif
  throw 1;
}

void PlyGeometry::remove( PlyGeometry * g_ptr )
{
#ifdef DEBUG_PLY
  std::cout << "PlyGeometry::remove(PlyGeometry *)" << std::endl;
#endif
  throw 1;
}


PlyGeometry::Iterator_t PlyGeometry::createIterator()
{
#ifdef DEBUG_PLY
  std::cout << "PlyGeometry::createIterator()" << std::endl;
#endif
  return Iterator_t(0);
}


PlyGeometry::PlyGeometry( std::string const name )
: _name(name)
{
#ifdef DEBUG_PLY
  std::cout << "PlyGeometry::PlyGeometry(std::string const)" << std::endl;
#endif
}
