/** @file VoxelGridLinIndex.hpp
 * 
 *  @brief Header file that defines the voxel grid index functor templates and
 *  specializations.
 *  Specializations define the mapping from multi-dim voxel indices to a 1-dim
 *  linearized grid index and the corresponding inverse mapping.
 */

#ifndef VOXELGRIDLININDEX_HPP
#define	VOXELGRIDLININDEX_HPP

/**
 * @brief Functor template. Objects perform 'multi-dim voxel index ->
 * linearized voxel index'.
 
 * @tparam ConcreteVGLinId Type of a specialization of this template.
 * @tparam ConcreteVG Type of a grid that indices refer to.
 */
template<
      typename ConcreteVGLinId
    , typename ConcreteVG >
struct VoxelGridLinId {
  /**
   * @brief Functor operation.
   * 
   * @param idx Voxel index in x direction.
   * @param idy Voxel index in y direction.
   * @param idz Voxel index in z direction.
   * @param vg Ptr to the grid definition object.
   * @return Linearized voxel index.
   */
  __host__ __device__
  int operator()(
        int const idx, int const idy, int const idz,
        ConcreteVG const * const vg ) {
    return static_cast<ConcreteVG*>(this)->
      operator()(idx, idy, idz, vg);
  }
};

/**
 * @brief Partial template specialization. Specializes, how the linearized
 * voxel index is calculated from the multi-dim voxel index. The multi-dim
 * indices ordered from least volatile to most volatile are:
 * (idx, idy, idz).
 * 
 * @tparam ConcreteVG Type of a grid that indices refer to.
 */
template<
      typename ConcreteVG >
struct DefaultVoxelGridLinId
: public VoxelGridLinId<DefaultVoxelGridLinId<ConcreteVG>, ConcreteVG> {
  __host__ __device__
  int operator()(
        int const idx, int const idy, int const idz,
        ConcreteVG const * const vg ) {
    return   idx *(vg->gridnz()*vg->gridny())
           + idy *(vg->gridnz())
           + idz;
  }
};

/**
 * @brief Functor template. Objects perform 'linearized voxel index ->
 * voxel index in x direction'.
 
 * @tparam ConcreteVGIdx Type of a specialization of this template.
 * @tparam ConcreteVG Type of a grid that indices refer to.
 */
template<
      typename ConcreteIdx,
      typename ConcreteVG >
struct VoxelGridIdx {
  /**
   * @brief Functor operation.
   * 
   * @param linId Linearized voxel index.
   * @param vg Ptr to the grid definition object.
   * @return Voxel index in x direction.
   */
  __host__ __device__
  int operator()(
        int const linId,
        ConcreteVG const * const vg ) {
    return static_cast<ConcreteVG*>(this)->
      operator()(linId, vg);
  }
};

/**
 * @brief Partial template specialization. Specializes, how the voxel index in
 * x direction is calculated from the linearized voxel index. Corresponds to
 * DefaultVoxelGridLinId<typename ConcreteVG>.
 * 
 * @tparam ConcreteVG Type of a grid that indices refer to.
 */
template<
      typename ConcreteVG >
struct DefaultVoxelGridIdx
: public VoxelGridIdx<DefaultVoxelGridIdx<ConcreteVG>, ConcreteVG> {
  __host__ __device__
  int operator()(
        int const linId, ConcreteVG const * const vg ) {
    return linId/(vg->gridnz()*vg->gridny());
  }
};

/**
 * @brief Functor template. Objects perform 'linearized voxel index ->
 * voxel index in y direction'.
 
 * @tparam ConcreteVGIdy Type of a specialization of this template.
 * @tparam ConcreteVG Type of a grid that indices refer to.
 */
template<
      typename ConcreteIdy,
      typename ConcreteVG >
struct VoxelGridIdy {
  /**
   * @brief Functor operation.
   * 
   * @param linId Linearized voxel index.
   * @param vg Ptr to the grid definition object.
   * @return Voxel index in y direction.
   */
  __host__ __device__
  int operator()(
        int const linId,
        ConcreteVG const * const vg ) {
    return static_cast<ConcreteVG*>(this)->
      operator()(linId, vg);
  }
};

/**
 * @brief Partial template specialization. Specializes, how the voxel index in
 * y direction is calculated from the linearized voxel index. Corresponds to
 * DefaultVoxelGridLinId<typename ConcreteVG>.
 * 
 * @tparam ConcreteVG Type of a grid that indices refer to.
 */
template<
      typename ConcreteVG >
struct DefaultVoxelGridIdy
: public VoxelGridIdy<DefaultVoxelGridIdy<ConcreteVG>, ConcreteVG> {
  __host__ __device__
  int operator()(
        int const linId, ConcreteVG const * const vg ) {
    int temp(linId);
    temp %= (vg->gridnz()*vg->gridny());
    return temp/vg->gridnz();
  }
};

/**
 * @brief Functor template. Objects perform 'linearized voxel index ->
 * voxel index in z direction'.
 
 * @tparam ConcreteVGIdz Type of a specialization of this template.
 * @tparam ConcreteVG Type of a grid that indices refer to.
 */
template<
      typename ConcreteIdz,
      typename ConcreteVG >
struct VoxelGridIdz {
  /**
   * @brief Functor operation.
   * 
   * @param linId Linearized voxel index.
   * @param vg Ptr to the grid definition object.
   * @return Voxel index in z direction.
   */
  __host__ __device__
  int operator()(
        int const linId,
        ConcreteVG const * const vg ) {
    return static_cast<ConcreteVG*>(this)->
      operator()(linId, vg);
  }
};

/**
 * @brief Partial template specialization. Specializes, how the voxel index in
 * z direction is calculated from the linearized voxel index. Corresponds to
 * DefaultVoxelGridLinId<typename ConcreteVG>.
 * 
 * @tparam ConcreteVG Type of a grid that indices refer to.
 */
template<
      typename ConcreteVG >
struct DefaultVoxelGridIdz
: public VoxelGridIdz<DefaultVoxelGridIdz<ConcreteVG>, ConcreteVG> {
  __host__ __device__
  int operator()(
        int const linId, ConcreteVG const * const vg ) {
    int temp(linId);
    temp %= (vg->gridnz()*vg->gridny());
    temp %=  vg->gridnz();
    return temp;
  }
};
#endif	/* VOXELGRIDLININDEX_HPP */

