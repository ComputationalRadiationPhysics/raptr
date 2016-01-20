#ifndef PLYRECTANGLE_HPP
#define PLYRECTANGLE_HPP

#include "PlyGeometry.hpp"

/* A Leaf class */
template<typename Vertex>
class PlyRectangle : public PlyGeometry
{
  public:
    
    /* Constructor */
    PlyRectangle(
          std::string const, Vertex const, Vertex const,
          Vertex const, Vertex const );
    
    /* Destructor */
    virtual ~PlyRectangle();


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


  protected:
    
    /* Get number of faces, intermediate call */
    virtual std::string facesStr( int & );


  private:
    
    Vertex _p0, _p1, _p2, _p3;
};
#include "PlyRectangle.tpp"

#endif  // #define PLYRECTANGLE_HPP
