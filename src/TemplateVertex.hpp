/**
 * Copyright 2016 Malte Zacharias
 *
 * This file is part of raptr.
 *
 * raptr is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * raptr is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with raptr.
 * If not, see <http://www.gnu.org/licenses/>.
 */

/** @file TemplateVertex.hpp */
#ifndef TEMPLATEVERTEX_HPP
#define TEMPLATEVERTEX_HPP

template<typename T>
struct TemplateVertex {

  public:

    typedef T Coord_t;
    
    /* Parametric Constructor */
    TemplateVertex( T const x_, T const y_, T const z_ );
    
    /* Copy Constructor */
    TemplateVertex( TemplateVertex const & v );

    /* Copy Assignment */
    void operator=( TemplateVertex const & v );
    
    T operator[]( int dim );


    T x, y, z;
};
#include "TemplateVertex.tpp"

#endif  // #define TEMPLATEVERTEX_HPP
