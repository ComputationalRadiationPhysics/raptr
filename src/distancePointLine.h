/** @file distancePointLine.h
 * 
 * @brief Header file that defines a function for calculation the orthogonal
 * distance of a point from a line.
 */

#ifndef DISTANCEPOINTLINE_H
#define	DISTANCEPOINTLINE_H

#include <cstdio>

/**
 * @brief Calculate absolute value of a 3d vector.
 * 
 * @param a A 3 component array (= "vector").
 * @return Absolute value (= length) of vector.
 */
template<typename T>
__host__ __device__
inline T absolute( T const * const a )
{
    return sqrtf(a[0]*a[0] + a[1]*a[1] + a[2]*a[2]);
}

/**
 * @brief Calculate the scalar product of two 3d vectors
 * 
 * @param a A 3 component array (= "vector"). One factor.
 * @param b A 3 component array (= "vector"). One factor.
 * @return Scalar product.
 */
template<typename T>
__host__ __device__
inline T scalarProduct( T const * const a, T const * const b )
{
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

/**
 * @brief Calculate the cross product of two 3d vectors.
 * 
 * @param result A 3 component array.
 * @param a A 3 component array (= "vector"). One factor.
 * @param b A 3 component array (= "vector"). One factor. */
template<typename T>
__host__ __device__
inline void crossProduct( T * const result, T const * const a, T const * const b )
{
  for(int dim=0; dim<3; dim++) {
    result[dim] = a[(dim+1)%3]*b[(dim+2)%3] - a[(dim+2)%3]*b[(dim+1)%3];
  }
}

/**
 * @brief In 3d calculate the orthogonal distance between a point and a line.
 * 
 * The line is defined by two different points on it. In other words: It is
 * defined by a line segment. Still the result is always the orthogonal distance
 * in respect to the whole line. Expressly, this can be smaller than the point's
 * distance from either of the points defining the line segment.
 * 
 * @param a A 3 component array (= "vector"). Position vector of one point one the line.
 * @param b A 3 component array (= "vector"). Position vector of another point on the line.
 * @param p A 3 component array (= "vector"). Position vector of the point.
 * @return Distance.
 */
template<typename T>
__host__ __device__
T distance( T const * const a, T const * const b,
                T const * const p )
{
  T ap[3];
  T ab[3];
  T c[3];
  for(int dim=0; dim<3; dim++) {
    ap[dim] = p[dim]-a[dim];
    ab[dim] = b[dim]-a[dim];
  }
  crossProduct(c, ab, ap);
  return absolute(c) / absolute(ab);
}

#endif	/* DISTANCEPOINTLINE_H */

