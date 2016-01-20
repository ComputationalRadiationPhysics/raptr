#ifndef PLYGRID_HPP
#define PLYGRID_HPP

#include "CompositePlyGeometry.hpp"

template<typename Vertex>
class PlyGrid : public CompositePlyGeometry
{
  public:
    
    PlyGrid(
          std::string const,
          Vertex,
          int const &, int const &, int const &,
          typename Vertex::Coord_t const &,
          typename Vertex::Coord_t const &,
          typename Vertex::Coord_t const & );
};
#include "PlyGrid.tpp"

#endif  // #define PLYGRID_HPP
