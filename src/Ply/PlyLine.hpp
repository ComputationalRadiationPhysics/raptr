#ifndef PLYLINE_HPP
#define PLYLINE_HPP

#include "PlyGeometry.hpp"

/* A Leaf class */
template<typename Vertex>
class PlyLine : public PlyGeometry
{
  public:
    
    /* Constructor */
    PlyLine( std::string const, Vertex const, Vertex const );
    
    /* Destructor */
    virtual ~PlyLine();


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


  protected:
    
    /* Get number of faces, intermediate call */
    virtual std::string facesStr( int & );

    Vertex _p0, _p1;
};
#include "PlyLine.tpp"

#endif  // #define PLYLINE_HPP
