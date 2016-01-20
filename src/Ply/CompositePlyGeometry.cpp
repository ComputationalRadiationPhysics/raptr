#include "CompositePlyGeometry.hpp"

#ifdef DEBUG
#include <iostream>
#endif

CompositePlyGeometry::~CompositePlyGeometry()
{
#ifdef DEBUG
  std::cout << "CompositePlyGeometry::~CompositePlyGeometry()" << std::endl;
#endif
}

int CompositePlyGeometry::numVertices()
{
#ifdef DEBUG
  std::cout << "CompositePlyGeometry::numVertices()" << std::endl;
#endif
  int total = 0;

//  Iterator<PlyGeometry *> * it = CompositePlyGeometry::createIterator();
//  for(it->first(); !it->isDone(); it->next()) {
//    total += it->currentItem()->numVertices();
//  }
//
//  delete it;
  Iterator_t it = createIterator();
  for(it=_geometryList.begin(); it!=_geometryList.end(); it++)
    total += (*it)->numVertices();

  return total;
}

int CompositePlyGeometry::numFaces()
{
#ifdef DEBUG
  std::cout << "CompositePlyGeometry::numFaces()" << std::endl;
#endif
  int total = 0;

//  Iterator<PlyGeometry *> * it = CompositePlyGeometry::createIterator();
//  for(it->first(); !it->isDone(); it->next()) {
//    total += it->currentItem()->numFaces();
//  }
//
//  delete it;
  Iterator_t it = createIterator();
  for(it=_geometryList.begin(); it!=_geometryList.end(); it++)
    total += (*it)->numFaces();
  
  return total;
}

std::string CompositePlyGeometry::verticesStr()
{
#ifdef DEBUG
  std::cout << "CompositePlyGeometry::verticesStr()" << std::endl;
#endif
  std::string str("");

//  Iterator<PlyGeometry *> * it = CompositePlyGeometry::createIterator();
//  for(it->first(); !it->isDone(); it->next()) {
//    str += it->currentItem()->verticesStr();
//  }
//
//  delete it;
  Iterator_t it = createIterator();
  for(it=_geometryList.begin(); it!=_geometryList.end(); it++)
    str += (*it)->verticesStr();

  return str;
}

std::string CompositePlyGeometry::facesStr()
{
#ifdef DEBUG
  std::cout << "CompositePlyGeometry::facesStr()" << std::endl;
#endif
  int vertexId = 0;
  std::string str("");
  
//  Iterator<PlyGeometry *> * it = CompositePlyGeometry::createIterator();
//  for(it->first(); !it->isDone(); it->next()) {
//    str += it->currentItem()->facesStr(vertexId);
//  }
//
//  delete it;
  Iterator_t it = createIterator();
  for(it=_geometryList.begin(); it!=_geometryList.end(); it++)
    str += (*it)->facesStr(vertexId);

  return str;
}

std::string CompositePlyGeometry::facesStr( int & vertexId )
{
#ifdef DEBUG
  std::cout << "CompositePlyGeometry::facesStr(int &)" << std::endl;
#endif
  std::string str("");

//  Iterator<PlyGeometry *> * it = CompositePlyGeometry::createIterator();
//  for(it->first(); !it->isDone(); it->next()) {
//    str += it->currentItem()->facesStr(vertexId);
//  }
//
//  delete it;
  Iterator_t it = createIterator();
  for(it=_geometryList.begin(); it!=_geometryList.end(); it++)
    str += (*it)->facesStr(vertexId);

  return str;
}


void CompositePlyGeometry::add( PlyGeometry * g_ptr )
{
#ifdef DEBUG
  std::cout << "CompositePlyGeometry::add()" << std::endl;
#endif
  //_geometryList.append( g_ptr );
  _geometryList.push_back(g_ptr);
}

void CompositePlyGeometry::remove( PlyGeometry * g_ptr )
{
#ifdef DEBUG
  std::cout << "CompositePlyGeometry::remove()" << std::endl;
#endif
  //_geometryList.remove( g_ptr );
  Iterator_t it;
  for(it=_geometryList.begin(); it!=_geometryList.end(); it++)
  {
    if((*it)==g_ptr)
      _geometryList.erase(it);
  }
}

CompositePlyGeometry::Iterator_t CompositePlyGeometry::createIterator()
{
#ifdef DEBUG
  std::cout << "CompositePlyGeometry::createIterator()" << std::endl;
#endif
  //return new ListIterator<PlyGeometry *>(&_geometryList);
  std::list<PlyGeometry *>::iterator it;
  return it;
}


CompositePlyGeometry::CompositePlyGeometry( std::string const name )
: PlyGeometry(name)
{
#ifdef DEBUG
  std::cout << "CompositePlyGeometry::CompositePlyGeometry(std::string const)" << std::endl;
#endif
}
