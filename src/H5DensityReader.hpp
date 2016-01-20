/** @file H5DensityReader.hpp */
/* Author: malte
 *
 * Created on 9. April 2015, 15:14 */

#ifndef H5DENSITYREADER_HPP
#define	H5DENSITYREADER_HPP

//#include "H5Reader.hpp"
//
//class H5DensityReader : public H5Reader {
//public:
//  
//  H5DensityReader( std::string filename )
//  : H5Reader(filename) {
//#ifdef DEBUG
//  std::cout << "H5DensityReader::H5DensityReader(std::string)" << std::endl;
//#endif
//  
//    _datasetname = std::string("density");
//  
//    /* Open file, dataset */
//    _file = 0;
//    _is_open = true;
//    try {
//      _file = new H5::H5File(_filename.c_str(), H5F_ACC_RDWR);
//    }
//    catch(const H5::FileIException &) {
//      _is_open = false;
//    }
//  }
//
//  ~H5DensityReader() {
//    delete _file;
//  }
//};
  
#include "H5Cpp.h"

class H5DensityReader {
public:
  
  typedef float Value_t;
  
  H5DensityReader( std::string const filename )
  : _filename(filename), _datasetname("density") {
    _file = 0;
    _is_open = true;
    try {
      _file = new H5::H5File(_filename.c_str(), H5F_ACC_RDWR);
    }
    catch(const H5::FileIException &) {
      _is_open = false;
    }
  }
  
  ~H5DensityReader() {
    delete _file;
  }
  
  bool is_open() const {
    return _is_open;
  }
  
  void read( Value_t * const mem ) {
    if(!is_open())
      throw H5::FileIException();
    
    H5::DataSet dataset = _file->openDataSet(_datasetname.c_str());
    
    dataset.read(mem, H5::PredType::NATIVE_FLOAT);
  }
  
  int sizeOfFile() {
    if(!is_open())
      throw H5::FileIException();

    /* Open dataset */
    H5::DataSet     dataset = _file->openDataSet(_datasetname.c_str());
    /* Open dataspace of dataset */
    H5::DataSpace   dataspace = dataset.getSpace();
    /* Get ndims of dataspace */
    const int ndims = dataspace.getSimpleExtentNdims();

    /* Get dims */
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

#endif	/* H5DENSITYREADER_HPP */

