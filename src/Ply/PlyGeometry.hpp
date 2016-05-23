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

// Composite Pattern - Component class header file

#ifndef PLYGEOMETRY_HPP
#define PLYGEOMETRY_HPP

#include <list>
#include <string>

class CompositePlyGeometry;

/* The Component class */
class PlyGeometry
{
  public:
    
    typedef std::list<PlyGeometry *>::iterator Iterator_t;
    
    friend class CompositePlyGeometry;
    

    /* Destructor */
    virtual ~PlyGeometry();
    
    /* Get name */
    std::string name();
    
    
    /* Get number of vertices */
    virtual int numVertices();
    
    /* Get number of faces */
    virtual int numFaces();
    
    /* Get vertices string */
    virtual std::string verticesStr();
    
    /* Get faces string */
    virtual std::string facesStr();
    
    
    /* Add a component */
    virtual void add( PlyGeometry * );
    
    /* Remove a component */
    virtual void remove( PlyGeometry * );
    
    /* Get an iterator that iterates over the components */
    //virtual Iterator<PlyGeometry *> * createIterator();
    virtual Iterator_t createIterator();

   
  protected:
    
    /* Constructor */
    PlyGeometry( std::string const );
    
    
    /* Get faces string, intermediate call */
    virtual std::string facesStr( int & );
    
     
  private:
    
    std::string  _name;
};

#endif  // #define PLYGEOMETRY_HPP
