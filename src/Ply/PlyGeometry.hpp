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
