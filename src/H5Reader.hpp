/** @file H5Reader.hpp */
#ifndef H5READER_HPP
#define H5READER_HPP

#include "H5Cpp.h"
#include <string>
#include <iostream>
#include <cstdlib>

class H5Reader {
public:

  typedef float Value_t;

  H5Reader( std::string filename )
  : _filename(filename), _datasetname("messung") {
#ifdef DEBUG
    std::cout << "H5Reader::H5Reader(std::string)" << std::endl;
#endif
    /* Open file, dataset */
    _file = 0;
    _is_open = true;
    try {
      _file = new H5::H5File(_filename.c_str(), H5F_ACC_RDWR);
    }
    catch(const H5::FileIException &) {
      _is_open = false;
    }
  }

  ~H5Reader() {
    delete _file;
  }

  bool is_open() const {
    return _is_open;
  }

  void read( Value_t * mem ) {
#ifdef DEBUG
    std::cout << "H5Reader::read(Value_t*)" << std::endl;
#endif
    if(!is_open())
      throw H5::FileIException();

    H5::DataSet dataset     = _file->openDataSet(_datasetname.c_str());

    dataset.read(mem, H5::PredType::NATIVE_FLOAT);

  }

  int sizeOfFile() {
    if(!is_open())
      throw H5::FileIException();

    // Open dataset
    H5::DataSet     dataset = _file->openDataSet(_datasetname.c_str());
    // Open dataspace of dataset
    H5::DataSpace   dataspace = dataset.getSpace();
    // Get ndims of dataspace
    const int ndims = dataspace.getSimpleExtentNdims();

    // Get dims
    hsize_t dims[ndims];
    dataspace.getSimpleExtentDims(dims, NULL);
    hsize_t linDim(1);
    for(int i=0; i<ndims; i++)
      linDim *= dims[i];
    return linDim;
  }

  
protected:

  std::string _filename, _datasetname;
  H5::H5File * _file;
  bool _is_open;
};

#endif  // #ifndef H5READER_HPP
