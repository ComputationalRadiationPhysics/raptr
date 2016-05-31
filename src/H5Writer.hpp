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

/** @file H5Writer.hpp
 *
 *  @brief Header file that defines class H5Writer that writes measurement data
 *  to HDF5 measurement files.
 */

#pragma once

#include <string>
#include "H5Cpp.h"

template<typename TMeasurementSetup>
class MeasurementWritingTraits
{
  using MeasurementSetup = TMeasurementSetup;
};

template<typename TMeasurementSetup>
class H5Writer {
private:
  std::string _filename;
  
public:
  H5Writer( std::string const & filename )
  : _filename(filename)
  {}
  
  void write( float const * const mem, TMeasurementSetup const & setup )
  {
    MeasurementWritingTraits<TMeasurementSetup> traits;
    hsize_t dim0 = hsize_t(traits.dim0(setup));
    hsize_t dim1 = hsize_t(traits.dim1(setup));
    hsize_t dim2 = hsize_t(traits.dim2(setup));
    hsize_t dim3 = hsize_t(traits.dim3(setup));
    hsize_t dim4 = hsize_t(traits.dim4(setup));
    float dim0_step = traits.dim0_step(setup);
    
    // Create file (fail if exists)
    H5::H5File * file = new H5::H5File(_filename.c_str(), H5F_ACC_EXCL);
    
    // Create dataspace
    hsize_t measdims[5] = {dim0, dim1, dim2, dim3, dim4};
    H5::DataSpace measspace(5, measdims);
    
    // Create dataset
    H5::DataSet * measdataset;
    measdataset = new H5::DataSet(file->createDataSet(
            "messung", H5::PredType::IEEE_F32LE, measspace));
    
    // Write dataset
    measdataset->write(mem, H5::PredType::IEEE_F32LE);
    
    // Create and write attribute for dim0
    H5::Attribute dim0att;
    H5::DataSpace dim0dataspace(1, &dim0);
    dim0att = measdataset->createAttribute(
            "dim0_rot", H5::PredType::IEEE_F32LE, dim0dataspace);
    float * dim0mem = new float[dim0];
    for(hsize_t i = 0; i<dim0; i++) dim0mem[i] = float(i*dim0_step);
    dim0att.write(H5::PredType::IEEE_F32LE, dim0mem);
    delete[] dim0mem;
    
    // Create and write attribute for dim1
    H5::Attribute dim1att;
    H5::DataSpace dim1dataspace(1, &dim1);
    dim1att = measdataset->createAttribute(
            "dim1_Det0z", H5::PredType::STD_U32LE, dim1dataspace);
    unsigned * dim1mem = new unsigned[dim1];
    for(hsize_t i = 0; i<dim1; i++) dim1mem[i] = i;
    dim1att.write(H5::PredType::STD_U32LE, dim1mem);
    delete[] dim1mem;
    
    // Create and write attribute for dim2
    H5::Attribute dim2att;
    H5::DataSpace dim2dataspace(1, &dim2);
    dim2att = measdataset->createAttribute(
            "dim2_Det0y", H5::PredType::STD_U32LE, dim2dataspace);
    unsigned * dim2mem = new unsigned[dim2];
    for(hsize_t i = 0; i<dim2; i++) dim2mem[i] = i;
    dim2att.write(H5::PredType::STD_U32LE, dim2mem);
    delete[] dim2mem;
    
    // Create and write attribute for dim3
    H5::Attribute dim3att;
    H5::DataSpace dim3dataspace(1, &dim3);
    dim3att = measdataset->createAttribute(
            "dim3_Det1z", H5::PredType::STD_U32LE, dim3dataspace);
    unsigned * dim3mem = new unsigned[dim3];
    for(hsize_t i = 0; i<dim3; i++) dim3mem[i] = i;
    dim3att.write(H5::PredType::STD_U32LE, dim3mem);
    delete[] dim3mem;
    
    // Create and write attribute for dim4
    H5::Attribute dim4att;
    H5::DataSpace dim4dataspace(1, &dim4);
    dim4att = measdataset->createAttribute(
            "dim4_Det1y", H5::PredType::STD_U32LE, dim4dataspace);
    unsigned * dim4mem = new unsigned[dim4];
    for(hsize_t i = 0; i<dim4; i++) dim4mem[i] = i;
    dim4att.write(H5::PredType::STD_U32LE, dim4mem);
    delete[] dim4mem;
    
    // Release resources
    delete measdataset;
    delete file;
  }
};
