/* 
 * File:   PlyScene.hpp
 * Author: malte
 *
 * Created on 16. Oktober 2014, 18:30
 */

#ifndef PLYSCENE_HPP
#define	PLYSCENE_HPP

#include "CompositePlyGeometry.hpp"

class PlyScene : public CompositePlyGeometry
{
  public:
    
    PlyScene( std::string const name )
    : CompositePlyGeometry(name) {}
};

#endif	/* PLYSCENE_HPP */

