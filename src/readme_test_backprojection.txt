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

/** @file readme_test_backprojection.txt */
This is the readme file for the test unit test_backprojection.

To run the test execute 'run_test_backprojection.sh' from the command line.

To validate the test results, open the '*_x.h5' files and examine the contents.
The exspected contents are as follows:

epet_POS102_testA.mess.h5_x.h5 :
  One line in x direction at the middle of the y and z ranges

epet_POS102_testB.mess.h5_x.h5 :
  Four lines of increasing brightness:
  - at   0   degrees (weakest)
  - at  45   degrees
  - at  67.5 degrees
  - at 100   degrees (brightest)

epet_POS102_testC.mess.h5_x.h5 :
  Four lines of increasing brightness in x direction at the middle of z range:
  - at y == 0 (weakest)
  - at y == 2
  - at y == 4 
  - at y == 8 (brightest)
  (indices are detector segment indices)

epet_POS102_testD.mess.h5_x.h5 :
  Two crossing lines of same intensity in x-y plane at the middle of z range:
  - lower  y at x == 0, higher y at x == 12 (max)
  - higher y at x == 0, lower  y at x == 12
  (indices are detector segment indices)

epet_POS102_testE.mess.h5_x.h5 :
  Four lines of identical brightness in x direction at the middle of z range:
  - at y == 0
  - at y == 2
  - at y == 4 
  - at y == 8
  (indices are detector segment indices)
