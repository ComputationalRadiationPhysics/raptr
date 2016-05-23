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

/** @file TemplateVertex.tpp */
#include "TemplateVertex.hpp"

template<class T>
TemplateVertex<T>::TemplateVertex( T const x_, T const y_, T const z_ )
: x(x_), y(y_), z(z_) {}

template<class T>
TemplateVertex<T>::TemplateVertex( TemplateVertex const & v )
: x(v.x), y(v.y), z(v.z) {}

template<class T>
void TemplateVertex<T>::operator=( TemplateVertex<T> const & v )
{
  x=v.x; y=v.y; z=v.z;
}

template<typename T>
T TemplateVertex<T>::operator[]( int dim )
{
  if(dim==0) return x;
  else if(dim==1) return y;
  else if(dim==2) return z;
  else throw 1;
}
