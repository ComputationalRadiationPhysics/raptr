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

/** @file MeasurementWritingTraits.hpp
 *
 *  @brief Header file that defines traits used for writing files for class
 *  template DefaultMeasurementSetup.
 */

#pragma once

#include "MeasurementSetup.hpp"

template<>
class MeasurementWritingTraits<DefaultMeasurementSetup<val_t> > {
public:  
  MeasurementWritingTraits()
  {}
  
  unsigned dim0(DefaultMeasurementSetup<val_t> const & setup) {
    return unsigned(setup.na());
  }
  
  unsigned dim1(DefaultMeasurementSetup<val_t> const & setup) {
    return unsigned(setup.n0z());
  }
  
  unsigned dim2(DefaultMeasurementSetup<val_t> const & setup) {
    return unsigned(setup.n0y());
  }
  
  unsigned dim3(DefaultMeasurementSetup<val_t> const & setup) {
    return unsigned(setup.n1z());
  }
  
  unsigned dim4(DefaultMeasurementSetup<val_t> const & setup) {
    return unsigned(setup.n1y());
  }
  
  float dim0_step(DefaultMeasurementSetup<val_t> const & setup) {
    return float(setup.da());
  }
};
