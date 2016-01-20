/** @file mlemOperations.hpp */
/* Author: malte
 *
 * Created on 6. Februar 2015, 14:12 */

#ifndef MLEMOPERATIONS_HPP
#define	MLEMOPERATIONS_HPP

#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>

/**
 * @brief Divide two arrays elementwise.
 * @tparam T Type of array elements.
 * @param result Array of resulting quotient of length n.
 * @param dividend Array of dividends of length n.
 * @param divisor Array of divisors of length n.
 * @param n Length of arrays.
 */
template<typename T>
void divides(
      T * const & result,
      T * const & dividend,
      T * const & divisor,
      int const n) {
  thrust::device_ptr<T> dvd = thrust::device_pointer_cast<T>(dividend);
  thrust::device_ptr<T> dvs = thrust::device_pointer_cast<T>(divisor);
  thrust::device_ptr<T> res = thrust::device_pointer_cast<T>(result);
  thrust::transform(dvd, dvd+n, dvs, res, thrust::divides<T>());
}

/**
 * @brief Functor: (a, b, c) -> result : result = a * b / c
 * @tparam T Type of argument tuple elements and result.
 */
template<typename T>
struct dividesMultiplies_F : public thrust::unary_function<thrust::tuple<T, T, T>, T> {
  __host__ __device__
  /**
   * @brief Functor operation.
   * @param t Argument tuple. Elements: a, b, c.
   * @return a * b / c.
   */
  T operator()(
        thrust::tuple<T, T, T> const & t) const {
    T dvs(thrust::get<2>(t));
    dvs += T(dvs==0.);
    return thrust::get<0>(t) * thrust::get<1>(t) / dvs;
  }
};

/**
 * @brief Perform elementwise: result = a * b / c.
 * @tparam T Type of array elements.
 * @param result Array of results of length n.
 * @param a Array of first factors of length n.
 * @param b Array of second factors of length n.
 * @param c Array of divisors of length n.
 * @param n Length of arrays.
 */
template<typename T>
void dividesMultiplies(
      T * const & result,
      T * const & a,
      T * const & b,
      T * const & c,
      int const n) {
  thrust::device_ptr<T> aa = thrust::device_pointer_cast<T>(a);
  thrust::device_ptr<T> bb = thrust::device_pointer_cast<T>(b);
  thrust::device_ptr<T> cc = thrust::device_pointer_cast<T>(c);
  thrust::device_ptr<T> res = thrust::device_pointer_cast<T>(result);
  thrust::transform(
        thrust::make_zip_iterator(thrust::make_tuple(aa, bb, cc)),
        thrust::make_zip_iterator(thrust::make_tuple(aa+n, bb+n, cc+n)),
        res,
        dividesMultiplies_F<T>());
}

/**
 * @brief Functor: t -> result : result = t * alpha.
 * @tparam T Type of argument and result.
 */
template<typename T>
struct scales_F : public thrust::unary_function<T, T> {
  
/**
 * @brief Ctor.
 * @param alpha_ Factor to multiply arguments with.
 */
  __host__ __device__
  scales_F<T>(T const alpha_)
  : alpha(alpha_) {}

/**
 * @brief Functor operation.
 * @param t Argument.
 * @return Result.
 */
  __host__ __device__
  T operator()(T const & t) const {
    return alpha * t;
  }
  
private:
  T alpha;
};

/**
 * @brief Perform elementwise: x = x * alpha.
 * @tparam T Type of array elements.
 * @param x Array of length n.
 * @param alpha Scaling factor.
 * @param n Length of array.
 */
template<typename T>
void scales(
      T * const & x,
      T const alpha,
      int const n) {
  thrust::device_ptr<T> xx = thrust::device_pointer_cast<T>(x);
  thrust::transform(xx, xx+n, xx, scales_F<T>(alpha));
}

/**
 * @brief Sum of array.
 * @tparam T Type of array elements.
 * @param x Array of summands of length n.
 * @param n Length of array.
 * @return Sum.
 */
template<typename T>
T sum(
      T * const & x,
      int const n) {
  thrust::device_ptr<T> xx = thrust::device_pointer_cast<T>(x);
  return thrust::reduce(xx, xx+n, T(0.), thrust::plus<T>());
}

#endif	/* MLEMOPERATIONS_HPP */

