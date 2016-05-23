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

#include "PlyLine.hpp"
#include "PlyWriter.hpp"
#include "TemplateVertex.hpp"


typedef double                     CoordType;
typedef TemplateVertex<CoordType>  VertexType;



int main()
{
  PlyLine<VertexType> line( "line",
                            VertexType(0.,0.,0.),
                            VertexType(1.,1.,1.) );

  PlyWriter writer("test_PlyLine_output.ply");
  writer.write(line);
  writer.close();

  return 0;
}
