#ifndef PLYWRITER_HPP
#define PLYWRITER_HPP

#include "PlyGeometry.hpp"
#include "CompositePlyGeometry.hpp"
#include "PlyRectangle.hpp"
#include <string>
#include <sstream>
#include <fstream>

class PlyWriter
{
  public:
    
    /* Constructor */
    PlyWriter( std::string const );

    /* Write to file */
    void write( PlyGeometry & );

    /* Close */
    void close();


  protected:
    
    std::ofstream _file;

    std::string header( PlyGeometry & );


  private:
    
};

#endif  // #define PLYWRITER_HPP
