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

// Composite Pattern - Composite class header file

#ifndef COMPOSITEPLYGEOMETRY_HPP
#define COMPOSITEPLYGEOMETRY_HPP

#include "PlyGeometry.hpp"
#include <list>

/* The Composite class */
class CompositePlyGeometry : public PlyGeometry
{
  public:
    
    /* Destructor */
    virtual ~CompositePlyGeometry();


    virtual int numVertices();

    virtual int numFaces();

    virtual std::string verticesStr();

    virtual std::string facesStr();


    virtual void add( PlyGeometry * );

    virtual void remove( PlyGeometry * );

    //virtual Iterator<PlyGeometry *> * createIterator();
    virtual Iterator_t createIterator();
    
    
  protected:
    
    CompositePlyGeometry( std::string const );

    virtual std::string facesStr( int & );


//  private:
    
    //List<PlyGeometry *> _geometryList;
    std::list<PlyGeometry *> _geometryList;
};

#endif  // #define COMPOSITEPLYGEOMETRY_HPP
