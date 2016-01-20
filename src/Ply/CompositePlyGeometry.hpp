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
