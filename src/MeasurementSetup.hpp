/** @file MeasurementSetup.hpp */
#ifndef MEASEREMENTSETUP_HPP
#define MEASEREMENTSETUP_HPP

template<typename T, typename ConcreteMeasurementSetup>
class MeasurementSetup
{
  public:
    
    // x posision of 1st detector
    __host__ __device__ T   pos0x() const
    {
      return static_cast<ConcreteMeasurementSetup*>(this)->pos0x();
    }

    // x posision of 2nd detector
    __host__ __device__ T   pos1x() const
    {
      return static_cast<ConcreteMeasurementSetup*>(this)->pos1x();
    }

    // number of angular steps
    __host__ __device__ int na() const
    {
      return static_cast<ConcreteMeasurementSetup*>(this)->na();
    }

    // number of detector segments 1st det, z direction
    __host__ __device__ int n0z() const
    {
      return static_cast<ConcreteMeasurementSetup*>(this)->n0z();
    }

    // number of detector segments 1st det, y direction
    __host__ __device__ int n0y() const
    {
      return static_cast<ConcreteMeasurementSetup*>(this)->n0y();
    }

    // number of detector segments 2nd det, z direction
    __host__ __device__ int n1z() const
    {
      return static_cast<ConcreteMeasurementSetup*>(this)->n1z();
    }

    // number of detector segments 2nd det, y direction
    __host__ __device__ int n1y() const
    {
      return static_cast<ConcreteMeasurementSetup*>(this)->n1y();
    }

    // angular step [°]
    __host__ __device__ T   da() const
    {
      return static_cast<ConcreteMeasurementSetup*>(this)->da();
    }

    // x edge length of one detector segment (same for both detectors)
    __host__ __device__ T   segx() const
    {
      return static_cast<ConcreteMeasurementSetup*>(this)->segx();
    }

    // y edge length of one detector segment (same for both detectors)
    __host__ __device__ T   segy() const
    {
      return static_cast<ConcreteMeasurementSetup*>(this)->segy();
    }
    
    // z edge length of one detector segment (same for both detectors)
    __host__ __device__ T   segz() const
    {
      return static_cast<ConcreteMeasurementSetup*>(this)->segz();
    }

//    /**
//     * @brief Break linear channel index up into separate indices of channel
//     *        configuration.
//     *
//     * @param sepChannelId Result memory (int[5])
//     * @param linChannelId Linear index of channel
//     */
//    __host__ __device__ void sepChannelId(
//          int * const sepChannelId_, int const linChannelId ) const
//    {
//      return static_cast<ConcreteMeasurementSetup*>(this)->
//            sepChannelId(sepChannelId_, linChannelId);
//    }

//    /**
//     * @brief Linearized index of channel.
//     *
//     * @param sepChannelId Separate indices of channel configuration
//     */
//    __host__ __device__ int linChannelId(
//          int const * const sepChannelId ) const
//    {
//      return static_cast<ConcreteMeasurementSetup*>(this)->
//            linChannelId(sepChannelId);
//    }
    
//    /**
//     * @brief Get a channel's geometrical properties.
//     *
//     * @param pos0 Result memory (val_t[3]), position of detector0 segment's center
//     * @param pos1 Result memory (val_t[3]), position of detector1 segment's center
//     * @param edges Result memory (val_t[3]), lengths of detector segments' edges
//     * @param sin_ Result memory (val_t *), sine of angle
//     * @param cos_ Result memory (val_t *), cosine of angle
//     * @param 5DChannelId Indices of channel configuration
//     * @param setup Measurement setup description
//     */
//    __host__ __device__ void getGeomProps(
//          T * const pos0,  T * const pos1,
//          T * const edges,
//          T * const sin_,  T * const cos_,
//          int const * const          sepChannelId ) const
//    {
//      return static_cast<ConcreteMeasurementSetup*>(this)->
//            getGeomProps(pos0, pos1, edges, sin_, cos_, sepChannelId);
//    }
};

template<typename T>
class DefaultMeasurementSetup : public MeasurementSetup<T, DefaultMeasurementSetup<T> >
{
  private:
    
    T   _pos0x;
    T   _pos1x;
    int _na;
    T   _da;
    int _n0z;
    int _n0y;
    int _n1z;
    int _n1y;
    T   _segx;
    T   _segy;
    T   _segz;

//    __host__ __device__
//    void getDims( int * const dims ) const
//    {
//      dims[0] = _na;
//      dims[1] = _n0z;
//      dims[2] = _n0y;
//      dims[3] = _n1z;
//      dims[4] = _n1y;
//    }


  public:
    
    DefaultMeasurementSetup()
    {}
    
    DefaultMeasurementSetup(
          T   pos0x, T   pos1x,
          int na,    int n0z,   int n0y,  int n1z, int n1y,
          T   da,    T   segx,  T   segy, T   segz )
    : _pos0x(pos0x), _pos1x(pos1x), _na(na), _n0z(n0z), _n0y(n0y), _n1z(n1z),
      _n1y(n1y), _da(da), _segx(segx), _segy(segy), _segz(segz)
    {}

    DefaultMeasurementSetup( DefaultMeasurementSetup const & o )
    : _pos0x(o._pos0x), _pos1x(o._pos1x),
      _na(o._na), _n0z(o._n0z), _n0y(o._n0y), _n1z(o._n1z), _n1y(o._n1y),
      _da(o._da), _segx(o._segx), _segy(o._segy), _segz(o._segz)
    {}
    
    ~DefaultMeasurementSetup()
    {}
    
    __host__ __device__
    T   pos0x() const
    {
      return _pos0x;
    }

    __host__ __device__
    T   pos1x() const
    {
      return _pos1x;
    }

    __host__ __device__
    int na() const
    {
      return _na;
    }

    __host__ __device__
    T   da() const
    {
      return _da;
    }

    __host__ __device__
    int n0z() const
    {
      return _n0z;
    }

    __host__ __device__
    int n0y() const
    {
      return _n0y;
    }

    __host__ __device__
    int n1z() const
    {
      return _n1z;
    }

    __host__ __device__
    int n1y() const
    {
      return _n1y;
    }

    __host__ __device__
    T   segx() const
    {
      return _segx;
    }

    __host__ __device__
    T   segy() const
    {
      return _segy;
    }

    __host__ __device__
    T   segz() const
    {
      return _segz;
    }
    
//    __host__ __device__
//    void sepChannelId(
//          int * const sepChannelId_, int const linChannelId ) const
//    {
//      int dims[5];
//      getDims(dims);
//
//      int temp( linChannelId );
//      sepChannelId_[0] = temp / (dims[4]*dims[3]*dims[2]*dims[1]); // angular id
//      temp %= (dims[4]*dims[3]*dims[2]*dims[1]);
//      sepChannelId_[1] = temp / (dims[4]*dims[3]*dims[2]);         // det0z index
//      temp %= (dims[4]*dims[3]*dims[2]);
//      sepChannelId_[2] = temp / (dims[4]*dims[3]);                 // det0y index
//      temp %= (dims[4]*dims[3]);
//      sepChannelId_[3] = temp / (dims[4]);                         // det1z index
//      temp %= (dims[4]);
//      sepChannelId_[4] = temp;                                     // det1y index
//    }

//    __host__ __device__
//    int linChannelId(
//          int const * const sepChannelId ) const
//    {
//      int dims[5];
//      getDims(dims);
//
//      return   sepChannelId[0] * dims[1]*dims[2]*dims[3]*dims[4]
//             + sepChannelId[1] *         dims[2]*dims[3]*dims[4]
//             + sepChannelId[2] *                 dims[3]*dims[4]
//             + sepChannelId[3] *                         dims[4]
//             + sepChannelId[4];
//    }
    
//    __host__ __device__
//    void getGeomProps(
//          T * const pos0, T * const pos1,
//          T * const edges,
//          T * const sin_, T * const cos_,
//          int const * const sepChannelId ) const
//    {
//      pos0[0]  = this->pos0x();
//      pos0[1]  = (sepChannelId[2]-0.5*this->n0y()+0.5)*this->segy();
//      pos0[2]  = (sepChannelId[1]-0.5*this->n0z()+0.5)*this->segz();
//      pos1[0]  = this->pos1x();
//      pos1[1]  = (sepChannelId[4]-0.5*this->n1y()+0.5)*this->segy();
//      pos1[2]  = (sepChannelId[3]-0.5*this->n1z()+0.5)*this->segz();
//      edges[0] = this->segx();
//      edges[1] = this->segy();
//      edges[2] = this->segz();
//      //sin_[0]  = sin(sepChannelId[0]*this->da()); // !!!!!
//      sin_[0]  = sin(sepChannelId[0]*this->da()/180.*PI);
//      //cos_[0]  = cos(sepChannelId[0]*this->da()); // !!!!!
//      cos_[0]  = cos(sepChannelId[0]*this->da()/180.*PI); // !!!!!
//    }
};

#endif  // #define MEASEREMENTSETUP_HPP

