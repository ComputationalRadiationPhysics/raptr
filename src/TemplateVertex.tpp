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
