#include "PlyWriter.hpp"

#ifdef DEBUG_PLY
#include <iostream>
#endif

PlyWriter::PlyWriter( std::string const fn )
: _file(fn.c_str())
{
#ifdef DEBUG_PLY
  std::cout << "PlyWriter::PlyWriter(std::string const)" << std::endl;
#endif
}


void PlyWriter::write( PlyGeometry & pg )
{
#ifdef DEBUG_PLY
  std::cout << "PlyWriter::write(PlyGeometry &)" << std::endl;
#endif
  _file\
    << header(pg)
    << pg.verticesStr()
    << pg.facesStr();
}

void PlyWriter::close()
{
#ifdef DEBUG_PLY
  std::cout << "PlyWriter::close()" << std::endl;
#endif
  _file.close();
}

std::string PlyWriter::header( PlyGeometry & pg )
{
#ifdef DEBUG_PLY
  std::cout << "PlyWriter::header(PlyGeometry const&)" << std::endl;
#endif
  std::stringstream ss("");
  
  ss\
    << "ply" << std::endl
    << "format ascii 1.0" << std::endl
    << "comment Created by object PlyWriter" << std::endl
    << "element vertex "
    << pg.numVertices() << std::endl
    << "property float x" << std::endl
    << "property float y" << std::endl
    << "property float z" << std::endl
    << "property float nx" << std::endl
    << "property float ny" << std::endl
    << "property float nz" << std::endl
    << "element face "
    << pg.numFaces() << std::endl
    << "property list uchar uint vertex_indices" << std::endl
    << "end_header" << std::endl;
  
  return ss.str();
}
