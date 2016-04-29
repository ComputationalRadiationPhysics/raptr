/** @file MeasurementSetupTrafo2CartCoordUT.cu */

// BOOST
#include <boost/test/unit_test.hpp>

// RAPTR
#include "MeasurementSetup.hpp"
#include "MeasurementSetupTrafo2CartCoord.hpp"


/*******************************************************************************
 * Test Suites
 ******************************************************************************/
BOOST_AUTO_TEST_SUITE( test_MeasurementSetupTrafo2CartCoord )

#ifndef MEASUREMENTSETUP_DEFINES
#define MEASUREMENTSETUP_DEFINES

#define N0Z    4      // 1st detector's number of segments in z
#define N0Y    4      // 1st detector's number of segments in y
#define N1Z    4      // 2nd detector's number of segments in z
#define N1Y    4      // 2nd detector's number of segments in y
#define NA     4      // number of angular positions
#define DA     090.0  // angular step
#define POS0X -003.5  // position of 1st detector's center in x [cm]
#define POS1X  003.5  // position of 2nd detector's center in x [cm]
#define SEGX   001.0  // x edge length of one detector segment [cm]
#define SEGY   001.0  // y edge length of one detector segment [cm]
#define SEGZ   001.0  // z edge length of one detector segment [cm]

#define NCHANNELS NA*N0Z*N0Y*N1Z*N1Y

#endif  // #ifndef MEASUREMENTSETUP_DEFINES

typedef double val_t;
typedef DefaultMeasurementSetup<val_t> MS;
typedef DefaultMeasurementSetupTrafo2CartCoordFirstPixel<val_t, MS> Trafo0;
typedef DefaultMeasurementSetupTrafo2CartCoordSecndPixel<val_t, MS> Trafo1;

/*******************************************************************************
 * Case: det0, pixel 00, corner 000, no turn
 ******************************************************************************/
BOOST_AUTO_TEST_CASE( test_MeasurementSetupTrafo2CartCoord_1 ){
  MS setup(POS0X, POS1X,
           NA, N0Z, N0Y, N1Z, N1Y,
           DA, SEGX, SEGY, SEGZ);
  Trafo0 trafo0;
  
  // Set dice coordinates to corner 000
  val_t diceCoords[3] = {0., 0., 0.};
  
  // Set output coordinates to a defined value
  val_t outCoords[3] = {0., 0., 0.};
  
  // Transform for pixel position id0z=0, id0y=0, ida=0
  trafo0(outCoords, diceCoords, 0, 0, 0, &setup);
  
  /* Check: 0th output coordinate is
   * the x position of det0's center
   * minus one half length of one pixel in x direction
   */
  BOOST_CHECK_EQUAL(outCoords[0], setup.pos0x() - 0.5 *     int(1)  * setup.segx());
  
  /* Check: 1st output coordinate is
   * half the number of det0's pixels in y direction
   * times the length of one pixel in y direction
   * in negative direction
   */
  BOOST_CHECK_EQUAL(outCoords[1],     val_t(0.) - 0.5 * setup.n0y() * setup.segy());
  
  /* Check: 2nd output coordinate is
   * half the number of det0's pixels in z direction
   * times the length of one pixel in z direction
   * in negative direction
   */
  BOOST_CHECK_EQUAL(outCoords[2],     val_t(0.) - 0.5 * setup.n0z() * setup.segz());  
}

/*******************************************************************************
 * Case: det0, pixel 00, corner 100, no turn
 ******************************************************************************/
BOOST_AUTO_TEST_CASE( test_MeasurementSetupTrafo2CartCoord_2 ){
  MS setup(POS0X, POS1X,
           NA, N0Z, N0Y, N1Z, N1Y,
           DA, SEGX, SEGY, SEGZ);
  Trafo0 trafo0;
  
  // Set dice coordinates to corner 100
  val_t diceCoords[3] = {1., 0., 0.};
  
  // Set output coordinates to a defined value
  val_t outCoords[3] = {0., 0., 0.};
  
  // Transform for pixel position id0z=0, id0y=0, ida=0
  trafo0(outCoords, diceCoords, 0, 0, 0, &setup);
  
  /* Check: 0th output coordinate is
   * the x position of det0's center
   * plus one half length of one pixel in x direction
   */
  BOOST_CHECK_EQUAL(outCoords[0], setup.pos0x() + 0.5 *     int(1)  * setup.segx());
  
  /* Check: 1st output coordinate is
   * half the number of det0's pixels in y direction
   * times the length of one pixel in y direction
   * in negative direction
   */
  BOOST_CHECK_EQUAL(outCoords[1],     val_t(0.) - 0.5 * setup.n0y() * setup.segy());
  
  /* Check: 2nd output coordinate is
   * half the number of det0's pixels in z direction
   * times the length of one pixel in z direction
   * in negative direction
   */
  BOOST_CHECK_EQUAL(outCoords[2],     val_t(0.) - 0.5 * setup.n0z() * setup.segz());  
}

/*******************************************************************************
 * Case: det0, pixel 00, corner 010, no turn
 ******************************************************************************/
BOOST_AUTO_TEST_CASE( test_MeasurementSetupTrafo2CartCoord_3 ){
  MS setup(POS0X, POS1X,
           NA, N0Z, N0Y, N1Z, N1Y,
           DA, SEGX, SEGY, SEGZ);
  Trafo0 trafo0;
  
  // Set dice coordinates to corner 010
  val_t diceCoords[3] = {0., 1., 0.};
  
  // Set output coordinates to a defined value
  val_t outCoords[3] = {0., 0., 0.};
  
  // Transform for pixel position id0z=0, id0y=0, ida=0
  trafo0(outCoords, diceCoords, 0, 0, 0, &setup);
  
  /* Check: 0th output coordinate is
   * the x position of det0's center
   * minus one half length of one pixel in x direction
   */
  BOOST_CHECK_EQUAL(outCoords[0], setup.pos0x() - 0.5 *              int(1) * setup.segx());
  
  /* Check: 1st output coordinate is
   * minus one plus half the number of det0's pixels in y direction
   * times the length of one pixel in y direction
   * in negative direction
   */
  BOOST_CHECK_EQUAL(outCoords[1],     val_t(0.) - ((0.5 * setup.n0y()) - 1) * setup.segy());
  
  /* Check: 2nd output coordinate is
   * half the number of det0's pixels in z direction
   * times the length of one pixel in z direction
   * in negative direction
   */
  BOOST_CHECK_EQUAL(outCoords[2],     val_t(0.) - 0.5 * setup.n0z()         * setup.segz());  
}

/*******************************************************************************
 * Case: det0, pixel 00, corner 001, no turn
 ******************************************************************************/
BOOST_AUTO_TEST_CASE( test_MeasurementSetupTrafo2CartCoord_4 ){
  MS setup(POS0X, POS1X,
           NA, N0Z, N0Y, N1Z, N1Y,
           DA, SEGX, SEGY, SEGZ);
  Trafo0 trafo0;
  
  // Set dice coordinates to corner 001
  val_t diceCoords[3] = {0., 0., 1.};
  
  // Set output coordinates to a defined value
  val_t outCoords[3] = {0., 0., 0.};
  
  // Transform for pixel position id0z=0, id0y=0, ida=0
  trafo0(outCoords, diceCoords, 0, 0, 0, &setup);
  
  /* Check: 0th output coordinate is
   * the x position of det0's center
   * minus one half length of one pixel in x direction
   */
  BOOST_CHECK_EQUAL(outCoords[0], setup.pos0x() - 0.5 *              int(1) * setup.segx());
    
  /* Check: 1st output coordinate is
   * half the number of det0's pixels in y direction
   * times the length of one pixel in y direction
   * in negative direction
   */
  BOOST_CHECK_EQUAL(outCoords[1],     val_t(0.) - 0.5 * setup.n0y()         * setup.segy());  
  
  /* Check: 2nd output coordinate is
   * minus one plus half the number of det0's pixels in z direction
   * times the length of one pixel in z direction
   * in negative direction
   */
  BOOST_CHECK_EQUAL(outCoords[2],     val_t(0.) - ((0.5 * setup.n0z()) - 1) * setup.segz());
}

/*******************************************************************************
 * Case: det0, pixel 00, inner dice volume, no turn
 ******************************************************************************/
BOOST_AUTO_TEST_CASE( test_MeasurementSetupTrafo2CartCoord_5 ){
  MS setup(POS0X, POS1X,
           NA, N0Z, N0Y, N1Z, N1Y,
           DA, SEGX, SEGY, SEGZ);
  Trafo0 trafo0;
  
  // Set dice coordinates to an inner position (it's safe to change it)
  val_t diceCoords[3] = {0.1, 0.1, 0.1};
  
  // Set output coordinates to a defined value
  val_t outCoords[3] = {0., 0., 0.};
  
  // Transform for pixel position id0z=0, id0y=0, ida=0
  trafo0(outCoords, diceCoords, 0, 0, 0, &setup);
  
  /* Check: 0th output coordinate is
   * the x position of det0's center
   * minus one half length of one pixel in x direction
   * plus the dice coordinate in x direction times the length of one pixel in x direction
   */
  BOOST_CHECK_EQUAL(outCoords[0], setup.pos0x() - ((0.5 * int(1.)) - diceCoords[0])     * setup.segx());
    
  /* Check: 1st output coordinate is
   * minus half the number of det0's pixels in y direction
   * times the length of one pixel in y direction
   * plus the dice coordinate in y direction times the length of one pixel in y direction
   */
  BOOST_CHECK_EQUAL(outCoords[1],     val_t(0.) - ((0.5 * setup.n0y()) - diceCoords[1]) * setup.segy());  
  
  /* Check: 2nd output coordinate is
   * minus half the number of det0's pixels in z direction
   * times the length of one pixel in z direction
   * plus the dice coordinate in y direction times the length of one pixel in z direction
   */
  BOOST_CHECK_EQUAL(outCoords[2],     val_t(0.) - ((0.5 * setup.n0z()) - diceCoords[2]) * setup.segz());
}

BOOST_AUTO_TEST_SUITE_END() // end test_MeasurementSetupTrafo2CartCoord

