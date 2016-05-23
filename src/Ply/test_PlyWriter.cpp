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
#include "CompositePlyGeometry.hpp"
#include "PlyWriter.hpp"
#include "TemplateVertex.hpp"

#include <iostream>


class Scene : public CompositePlyGeometry
{
  public:
    
    Scene( std::string const name )
    : CompositePlyGeometry(name) {}

    Iterator_t begin() {return this->_geometryList.begin();}
    Iterator_t end() {return this->_geometryList.end();}
};


typedef double                     CoordType;
typedef TemplateVertex<CoordType>  VertexType;



int main()
{
  PlyRectangle<VertexType> rect1( std::string("rect1"),
                      VertexType(0.,0.,0.),
                      VertexType(1.,0.,0.),
                      VertexType(1.,1.,0.),
                      VertexType(0.,1.,0.)
                    );
  PlyRectangle<VertexType> rect2( std::string("rect2"),
                      VertexType(0.,0.,1.),
                      VertexType(1.,0.,1.),
                      VertexType(1.,1.,1.),
                      VertexType(0.,1.,1.)
                    );
  PlyRectangle<VertexType> rect3( std::string("rect3"),
                      VertexType(0.,0.,2.),
                      VertexType(1.,0.,2.),
                      VertexType(1.,1.,2.),
                      VertexType(0.,1.,2.)
                    );
  PlyRectangle<VertexType> rect4( std::string("rect4"),
                      VertexType(-1., 0., 0.),
                      VertexType(0., -1., 0.),
                      VertexType(0., -1., 1.),
                      VertexType(-1., 0., 1.)
                    );
  
  Scene scene1("scene1");
  scene1.add(&rect1);
  scene1.add(&rect2);
  scene1.add(&rect3);

  Scene scene2("scene2");
  scene2.add(&rect4);
  
  Scene combined("combined");
  combined.add(&scene1);
  combined.add(&scene2);

  Scene::Iterator_t it = combined.createIterator();
  for(it=combined.begin(); it!=combined.end(); it++) {
    std::cout << (*it)->name() << std::endl;
  }

  PlyWriter writer("test_PlyWriter_output.ply");
  writer.write(combined);
  writer.close();

  return 0;
}
