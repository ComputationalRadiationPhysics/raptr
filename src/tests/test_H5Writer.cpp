#include "H5Writer.hpp"

// Doesn't do anything really
class FakeSetup
{};

// Enhances other class with stuff necessary for writing HDF5 measurement file
template<>
class MeasurementWritingTraits<FakeSetup>
{
public:
  
  hsize_t dim0( FakeSetup const & setup ) {
    return hsize_t(5);
  }
  
  hsize_t dim1( FakeSetup const & setup ) {
    return hsize_t(3);
  }
  
  hsize_t dim2( FakeSetup const & setup ) {
    return hsize_t(4);
  }
  
  hsize_t dim3( FakeSetup const & setup ) {
    return hsize_t(3);
  }
  
  hsize_t dim4( FakeSetup const & setup ) {
    return hsize_t(4);
  }
  
  float dim0_step( FakeSetup const & setup ) {
    return float(72.);
  }
};

// Number of elements in measurement data
template<typename TSetup>
unsigned linDataSize( TSetup setup ) {
  MeasurementWritingTraits<TSetup> traits;
  return traits.dim0(setup) *
    traits.dim1(setup) *
    traits.dim2(setup) *
    traits.dim3(setup) *
    traits.dim4(setup);
}

// Set data elements to some values
template<typename TSetup>
void fakeData( float * const data, TSetup setup ) {
  unsigned n = linDataSize<TSetup>(setup);
  for(unsigned i=0; i<n; i++) data[i] = float(i);
}

int main() {
  FakeSetup setup;
  unsigned n = linDataSize<FakeSetup>(setup);
  float * data = new float[n];
  fakeData<FakeSetup>(data, setup);
  
  std::string fn("test_H5Writer_output.h5");
  
  H5Writer<FakeSetup> writer(fn);
  
  writer.write(data, setup);
  
  delete[] data;
  return 0;
}
