#include "CompositePlyGeometry.hpp"
#include "PlyBox.hpp"
#include "PlyWriter.hpp"
#include "TemplateVertex.hpp"

class Scene : public CompositePlyGeometry
{
  public:
    
    Scene( std::string const name )
    : CompositePlyGeometry(name) {}
};


typedef double                     CoordType;
typedef TemplateVertex<CoordType>  VertexType;



int main()
{
  Scene scene("scene");
  
  PlyBox<VertexType> box1("box1", VertexType(0.,  0.,  0.), 1., 1., 1.);
  PlyBox<VertexType> box2("box2", VertexType(1.5, 1.5, 0.), 0.5, 0.5, 0.5);
  PlyBox<VertexType> box3("box3", VertexType(0.,  2.,  0.), 1., 5., 1.);
  PlyBox<VertexType> box4("box4", VertexType(0.,  -1., 0.), 1., -5., 1.);

  scene.add(&box1); scene.add(&box2); scene.add(&box3); scene.add(&box4);

  PlyWriter writer("test_PlyBox_output.ply");
  writer.write(scene);
  writer.close();
  
  return 0;
}
