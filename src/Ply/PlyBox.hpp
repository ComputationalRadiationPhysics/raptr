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

#ifndef PLYBOX_HPP
#define PLYBOX_HPP

#include "PlyGeometry.hpp"

/* A Leaf class */
template<typename Vertex>
class PlyBox : public PlyGeometry
{
  public:
    
    /* Constructor */
    PlyBox(
          std::string const, Vertex const,
          typename Vertex::Coord_t const,
          typename Vertex::Coord_t const,
          typename Vertex::Coord_t const );
    
    /* Destructor */
    virtual ~PlyBox();


    /* Get number of vertices */
    virtual int numVertices();
    
    /* Get number of faces */
    virtual int numFaces();
    
    /* Get vertices string */
    virtual std::string verticesStr();
    
    /* Get faces string */
    virtual std::string facesStr();


    Vertex & p0();

    Vertex & p1();

    Vertex & p2();

    Vertex & p3();

    Vertex & p4();
    
    Vertex & p5();
    
    Vertex & p6();
    
    Vertex & p7();


  protected:
    
    /* Get number of faces, intermediate call */
    virtual std::string facesStr( int & );


//  private:
    
    Vertex _p0, _p1, _p2, _p3, _p4, _p5, _p6, _p7;
};
#include "PlyBox.tpp"

#endif  // #define PLYBOX_HPP
