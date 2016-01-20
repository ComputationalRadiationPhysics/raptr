#include "PlyGrid.hpp"
#include "PlyWriter.hpp"
#include "TemplateVertex.hpp"


typedef double                    CoordType;
typedef TemplateVertex<CoordType> VertexType;



int main()
{
  PlyGrid<VertexType> grid( "grid",
                            VertexType(0.,0.,0.),
                            3, 4, 5,
                            1., 1., 1. );

  PlyWriter writer("test_PlyGrid_output.ply");
  writer.write(grid);
  writer.close();

  return 0;
}
