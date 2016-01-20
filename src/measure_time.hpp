/** @file measure_time.hpp
 * Author: malte
 *
 * Created on 8. April 2015, 11:39 */

#ifndef MEASURE_TIME_HPP
#define	MEASURE_TIME_HPP

#include <ctime>
#include <iostream>
#include <string>

void printTimeDiff(clock_t const end, clock_t const beg,
      std::string const & mes) {
  std::cout << mes << (float(end-beg)/CLOCKS_PER_SEC) << " s" << std::endl;
}

void takeAndPrintTimeDiff( clock_t & tak, clock_t const beg,
        std::string const & mes ) {
  tak = clock();
  printTimeDiff(tak, beg, mes);
}

#endif	/* MEASURE_TIME_HPP */

