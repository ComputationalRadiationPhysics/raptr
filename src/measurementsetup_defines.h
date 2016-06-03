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

/** @file measurementsetup_defines.h
 * 
 *  @brief Header file that defines the setup of the measurement that led to
 *  the data which is to be reconstructed.
 */
#ifndef MEASUREMENTSETUP_DEFINES
#define MEASUREMENTSETUP_DEFINES

#ifdef SETUP_REAL

#define N0Z 13      // 1st detector's number of segments in z
#define N0Y 13      // 1st detector's number of segments in y
#define N1Z 13      // 2nd detector's number of segments in z
#define N1Y 13      // 2nd detector's number of segments in y
#define NA  180     // number of angular positions
#define DA     2.     // angular step
#define POS0X -0.457  // position of 1st detector's center in x [m]
#define POS1X  0.457  // position of 2nd detector's center in x [m]
#define SEGX   0.020  // x edge length of one detector segment [m]
#define SEGY   0.004  // y edge length of one detector segment [m]
#define SEGZ   0.004  // z edge length of one detector segment [m]
#define NCHANNELS NA*N0Z*N0Y*N1Z*N1Y

#else
#ifdef SETUP_SMALL

#define N0Z 3      // 1st detector's number of segments in z
#define N0Y 3      // 1st detector's number of segments in y
#define N1Z 3      // 2nd detector's number of segments in z
#define N1Y 3      // 2nd detector's number of segments in y
#define NA  1     // number of angular positions
#define DA     90.     // angular step
#define POS0X -2.5  // position of 1st detector's center in x [m]
#define POS1X  2.5  // position of 2nd detector's center in x [m]
#define SEGX   1.  // x edge length of one detector segment [m]
#define SEGY   1.  // y edge length of one detector segment [m]
#define SEGZ   1.  // z edge length of one detector segment [m]
#define NCHANNELS NA*N0Z*N0Y*N1Z*N1Y

#endif  // SETUP_SMALL
#endif  // SETUP_REAL

#endif  // #ifndef MEASUREMENTSETUP_DEFINES

