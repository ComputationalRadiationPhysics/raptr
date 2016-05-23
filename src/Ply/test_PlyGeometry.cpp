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

#include "PlyGeometry.hpp"
#include "CompositePlyGeometry.hpp"
#include "PlyRectangle.hpp"
#include "TemplateVertex.hpp"

#include <iostream>
#include <string>

class Scene : public CompositePlyGeometry
{
  public:
    
    Scene( std::string const name )
    : CompositePlyGeometry(name) {}

    int count() {return this->_geometryList.size();}

    Iterator_t begin() {return this->_geometryList.begin();}
    Iterator_t end() {return this->_geometryList.end();}
};


typedef double                    CoordType;
typedef TemplateVertex<CoordType> VertexType;



int main( void )
{
  PlyRectangle<VertexType> rect( std::string("rect"),
                                 VertexType(0.,0.,0.),
                                 VertexType(0.,1.,0.),
                                 VertexType(1.,1.,0.),
                                 VertexType(1.,0.,0.) );
  
  Scene scene( std::string("scene") );

  scene.add(&rect);
  scene.add(&rect);
  scene.add(&rect);
  std::cout << "count: " << scene.count() << std::endl;

  PlyGeometry::Iterator_t it = scene.createIterator();
  std::cout << "starting iterations..." << std::endl;
  for(it=scene.begin(); it!=scene.end(); it++) {
    std::cout << (*it)->name() << std::endl << std::flush;
  }

  std::cout << scene.verticesStr() << scene.facesStr() << std::endl;

  return 0;
}
