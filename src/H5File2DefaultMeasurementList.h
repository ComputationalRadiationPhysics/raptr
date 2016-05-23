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

/** @file H5File2DefaultMeasurementList.h
 * 
 *  @brief Header file that defines function template
 *  H5File2DefaultMeasurementList which reads measurement data from file and
 *  returns it as DefaultMeasurementList object.
 */
#ifndef HDF5FILE2MEASUREMENTLIST_H
#define	HDF5FILE2MEASUREMENTLIST_H

#include "MeasurementList.hpp"
#include "H5Reader.hpp"
#include <string>
#include <iostream>
#include <cstdlib>

template<typename T>
DefaultMeasurementList<T> H5File2DefaultMeasurementList(std::string const fn,
                                                     int const rawSize) {
  H5Reader reader(fn);
  if(!reader.is_open()) {
    std::cerr << "Could not open file " << fn << std::endl;
    exit(EXIT_FAILURE);
  }
  
  // Read raw data from file
  int const rawFileSize = reader.sizeOfFile();
  if(rawFileSize!=rawSize) {
    std::cerr << "File " << fn << " does not have raw size " << rawSize
              << std::endl;
    exit(EXIT_FAILURE);
  }
  T * mem = new T[rawSize];
  reader.read(mem);
  
  // Count those channels, that have values != 0.
  int listId(0);
  for(int cnlId=0; cnlId<rawSize; cnlId++)
    if(mem[cnlId] != 0.)
      listId++;
  int const nonZeroSize(listId);
  
  // Create measurement list
  DefaultMeasurementList<T> list(nonZeroSize);
  listId=0;
  for(int cnlId=0; cnlId<rawSize; cnlId++) {
    if(mem[cnlId] != 0) {
      list.set(listId, cnlId, mem[cnlId]);
      listId++;
    }
  }
  
  return list;
}

#endif	/* HDF5FILE2MEASUREMENTLIST_H */

