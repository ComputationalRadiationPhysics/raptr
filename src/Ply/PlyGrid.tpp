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

#include "PlyGrid.hpp"
#include "PlyRectangle.hpp"

#ifdef DEBUG_PLY
#include <iostream>
#endif


template<typename Vertex>
PlyGrid<Vertex>::PlyGrid(
      std::string const name, Vertex o,
      int const & nx,
      int const & ny,
      int const & nz,
      typename Vertex::Coord_t const & dx,
      typename Vertex::Coord_t const & dy,
      typename Vertex::Coord_t const & dz )
: CompositePlyGeometry(name)
{
  for(int idx=0; idx<nx; idx++) {
    this->add(new PlyRectangle<Vertex>(
                        std::string(""),
                        Vertex(o.x + idx*dx,  o.y,           o.z),
                        Vertex(o.x + idx*dx,  o.y+(ny-1)*dy, o.z),
                        Vertex(o.x + idx*dx,  o.y+(ny-1)*dy, o.z+(nz-1)*dz),
                        Vertex(o.x + idx*dx,  o.y,           o.z+(nz-1)*dz)
                  )
             );
  }
  for(int idy=0; idy<ny; idy++) {
    this->add(new PlyRectangle<Vertex>(
                        std::string(""),
                        Vertex(o.x,           o.y + idy*dy,  o.z),
                        Vertex(o.x,           o.y + idy*dy,  o.z+(nz-1)*dz),
                        Vertex(o.x+(nx-1)*dx, o.y + idy*dy,  o.z+(nz-1)*dz),
                        Vertex(o.x+(nx-1)*dx, o.y + idy*dy,  o.z)
                  )
             );
  }
  for(int idz=0; idz<nz; idz++) {
    this->add(new PlyRectangle<Vertex>(
                        std::string(""),
                        Vertex(o.x,           o.y,           o.z + idz*dz),
                        Vertex(o.x+(nx-1)*dx, o.y,           o.z + idz*dz),
                        Vertex(o.x+(nx-1)*dx, o.y+(ny-1)*dy, o.z + idz*dz),
                        Vertex(o.x,           o.y+(ny-1)*dy, o.z + idz*dz)
                  )
             );
  }
}
