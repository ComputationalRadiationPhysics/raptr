#include "PlyLine.hpp"
#include "PlyWriter.hpp"
#include "TemplateVertex.hpp"


typedef double                     CoordType;
typedef TemplateVertex<CoordType>  VertexType;



int main()
{
  PlyLine<VertexType> line( "line",
                            VertexType(0.,0.,0.),
                            VertexType(1.,1.,1.) );

  PlyWriter writer("test_PlyLine_output.ply");
  writer.write(line);
  writer.close();

  return 0;
}
