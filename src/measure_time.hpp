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

/** @file measure_time.hpp
 * 
 *  @brief Rudimentary time taking and printing to stdout.
 */
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

