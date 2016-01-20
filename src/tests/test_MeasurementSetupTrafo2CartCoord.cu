/** @file test_MeasurementSetupTrafo2CartCoord.cu */
/* 
 * File:   test_MeasurementSetupTrafo2CartCoord.cpp
 * Author: malte
 *
 * Created on 16.10.2014, 18:10:07
 */

#ifndef MEASUREMENTSETUP_DEFINES
#define MEASUREMENTSETUP_DEFINES

#define N0Z 13      // 1st detector's number of segments in z
#define N0Y 13      // 1st detector's number of segments in y
#define N1Z 13      // 2nd detector's number of segments in z
#define N1Y 13      // 2nd detector's number of segments in y
#define NA  180     // number of angular positions
#define DA     2.     // angular step
#define POS0X -045.7  // position of 1st detector's center in x [cm]
#define POS1X  045.7  // position of 2nd detector's center in x [cm]
#define SEGX   002.0  // x edge length of one detector segment [cm]
#define SEGY   000.4  // y edge length of one detector segment [cm]
#define SEGZ   000.4  // z edge length of one detector segment [cm]

#define NCHANNELS NA*N0Z*N0Y*N1Z*N1Y

#endif  // #ifndef MEASUREMENTSETUP_DEFINES


#include <stdlib.h>
#include <iostream>
#include <iomanip>
#include "MeasurementSetup.hpp"
#include "MeasurementSetupTrafo2CartCoord.hpp"
#include "Ply.hpp"
#include "TemplateVertex.hpp"

/*
 * Simple C++ Test Suite
 */

#define IDA 13

typedef double val_t;
typedef DefaultMeasurementSetup<val_t> MS;
typedef DefaultMeasurementSetupTrafo2CartCoordFirstPixel<val_t, MS> Trafo0;
typedef DefaultMeasurementSetupTrafo2CartCoordSecndPixel<val_t, MS> Trafo1;
typedef TemplateVertex<val_t> Vertex;

template<typename VertexT>
class Pixel : public PlyBox<VertexT> {
  public:
    Pixel(val_t const * const c, val_t const dx, val_t const dy, val_t const dz,
          val_t const r)
    : PlyBox<VertexT>(std::string("pixel"),
                      VertexT(c[0]-.5*dx, c[1]-.5*dy, c[2]-.5*dz),
                      2*r, 2*r, 2*r) {}
    
    Pixel()
    : PlyBox<VertexT>(std::string("pixel"),
                      VertexT(0., 0., 0.),
                      0., 0., 0.){}
    
    Pixel(Pixel & o) {
      this->p0() = o.p0();
      this->p1() = o.p1();
      this->p2() = o.p2();
      this->p3() = o.p3();
      this->p4() = o.p4();
      this->p5() = o.p5();
      this->p6() = o.p6();
      this->p7() = o.p7();
    }
   
    void operator=(Pixel & o) {
      this->p0() = o.p0();
      this->p1() = o.p1();
      this->p2() = o.p2();
      this->p3() = o.p3();
      this->p4() = o.p4();
      this->p5() = o.p5();
      this->p6() = o.p6();
      this->p7() = o.p7();
    }
};

void test1() {
  // Create setup
  MS setup(POS0X, POS1X,
           NA, N0Z, N0Y, N1Z, N1Y,
           DA, SEGX, SEGY, SEGZ);
  
  // Create transformation functor for detectors
  Trafo0 trafo0;
  Trafo1 trafo1;
  
  // Create scene and alloc pixel memory
  PlyScene scene0("scene0");
  PlyScene scene1("scene1");
  Pixel<Vertex> plyPixels0[N0Z*N0Y];
  Pixel<Vertex> plyPixels1[N1Z*N1Y];
  
  // Box center in relative in-pixel coordinates
  val_t const boxCenter[3] = {0.5, 0.5, 0.5};    
  
  for(int id0z=0; id0z<N0Z; id0z++) {
    for(int id0y=0; id0y<N0Y; id0y++) {
      val_t cartCenter[3];
      
      // Do transformation
      trafo0(cartCenter, boxCenter, id0z, id0y, 0, &setup);
      std::cout << "x: "  << std::setw(6) << cartCenter[0]
                << " y: " << std::setw(6) << cartCenter[1]
                << " z: " << std::setw(6) << cartCenter[2] << std::endl;
      
      // Make detector pixel
      Pixel<Vertex> temp(cartCenter, SEGX, SEGY, SEGZ, 0.1*SEGY);
      plyPixels0[id0z + id0y*N0Z] = temp;
      scene0.add(&plyPixels0[id0z + id0y*N0Z]);
    }
  }
  
  for(int id1z=0; id1z<N1Z; id1z++) {
    for(int id1y=0; id1y<N1Y; id1y++) {
      val_t cartCenter[3];
      
      // Do transformation
      trafo1(cartCenter, boxCenter, id1z, id1y, 0, &setup);
      std::cout << "x: "  << std::setw(6) << cartCenter[0]
                << " y: " << std::setw(6) << cartCenter[1]
                << " z: " << std::setw(6) << cartCenter[2] << std::endl;
      
      // Make detector pixel
      Pixel<Vertex> temp(cartCenter, SEGX, SEGY, SEGZ, 0.1*SEGY);
      plyPixels1[id1z + id1y*N0Z] = temp;
      scene1.add(&plyPixels1[id1z + id1y*N1Z]);  
    }
  }
  
  PlyWriter writer0("test_MeasurementSetupTrafo2CartCoord_test1_scene0.ply");
  writer0.write(scene0);
  writer0.close();
  PlyWriter writer1("test_MeasurementSetupTrafo2CartCoord_test1_scene1.ply");
  writer1.write(scene1);
  writer1.close();
}

void test2() {
  // Create setup
  MS setup(POS0X, POS1X,
           NA, N0Z, N0Y, N1Z, N1Y,
           DA, SEGX, SEGY, SEGZ);
  
  // Create transformation functor for detectors
  Trafo0 trafo0;
  Trafo1 trafo1;
  
  // Create scene and alloc pixel memory
  PlyScene scene2("scene2");
  PlyScene scene3("scene3");
  Pixel<Vertex> plyPixels2[N0Z*N0Y];
  Pixel<Vertex> plyPixels3[N1Z*N1Y];
  
  // Box center in relative in-pixel coordinates
  val_t const boxCenter[3] = {0.5, 0.5, 0.5};    
  
  for(int id0z=0; id0z<N0Z; id0z++) {
    for(int id0y=0; id0y<N0Y; id0y++) {
      val_t cartCenter[3];
      
      // Do transformation
      trafo0(cartCenter, boxCenter, id0z, id0y, IDA, &setup);
      std::cout << "x: "  << std::setw(6) << cartCenter[0]
                << " y: " << std::setw(6) << cartCenter[1]
                << " z: " << std::setw(6) << cartCenter[2] << std::endl;
      
      // Make detector pixel
      Pixel<Vertex> temp(cartCenter, SEGX, SEGY, SEGZ, 0.1*SEGY);
      plyPixels2[id0z + id0y*N0Z] = temp;
      scene2.add(&plyPixels2[id0z + id0y*N0Z]);
    }
  }
  
  for(int id1z=0; id1z<N1Z; id1z++) {
    for(int id1y=0; id1y<N1Y; id1y++) {
      val_t cartCenter[3];
      
      // Do transformation
      trafo1(cartCenter, boxCenter, id1z, id1y, IDA, &setup);
      std::cout << "x: "  << std::setw(6) << cartCenter[0]
                << " y: " << std::setw(6) << cartCenter[1]
                << " z: " << std::setw(6) << cartCenter[2] << std::endl;
      
      // Make detector pixel
      Pixel<Vertex> temp(cartCenter, SEGX, SEGY, SEGZ, 0.1*SEGY);
      plyPixels3[id1z + id1y*N0Z] = temp;
      scene3.add(&plyPixels3[id1z + id1y*N1Z]);  
    }
  }
  
  PlyWriter writer2("test_MeasurementSetupTrafo2CartCoord_test2_scene2.ply");
  writer2.write(scene2);
  writer2.close();
  PlyWriter writer3("test_MeasurementSetupTrafo2CartCoord_test2_scene3.ply");
  writer3.write(scene3);
  writer3.close();
}

int main(int argc, char** argv) {
  test1();
  test2();
  
  return (EXIT_SUCCESS);
}

