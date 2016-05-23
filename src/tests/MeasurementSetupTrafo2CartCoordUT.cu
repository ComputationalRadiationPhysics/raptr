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

/** @file MeasurementSetupTrafo2CartCoordUT.cu */

// STD
#include <cmath>

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
#define NA     6      // number of angular positions
#define DA     060.0  // angular step
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
typedef DefaultMeasurementSetupTrafo2CartCoordFirstPixel_inplace<val_t, MS> Trafo0i;
typedef DefaultMeasurementSetupTrafo2CartCoordSecndPixel_inplace<val_t, MS> Trafo1i;

/**
 * @brief Functor template. Calculates correct transformed coordinates for
 * cases without rotation. Used to check results of the transformation functors.
 * 
 * @tparam T_Trafo Type of transformation functor.
 * @param checkCoords Output array.
 * @param pixelIdZ Index of detector pixel in z direction.
 * @param pixelIdY Index of detector pixel in y direction.
 * @param diceCoordX Relative coordinate within pixel along direction that is
 * x axis in non-rotated position.
 * @param diceCoordY Relative coordinate within pixel along direction that is
 * y axis in non-rotated position.
 * @param diceCoordZ Relative coordinate within pixel along direction that is
 * z axis in non-rotated position.
 * @param setup Measurement setup object.
 */
template<
        typename T_Trafo
>
struct CalcCheckCoordsNoRot {
  void operator()( val_t * const checkCoords,
          int const pixelIdZ, int const pixelIdY,
          val_t const diceCoordX, val_t const diceCoordY, val_t const diceCoordZ,
          MS const & setup );
};

template<>
struct CalcCheckCoordsNoRot<Trafo0> {
  void operator()( val_t * const checkCoords,
          int const pixelIdZ, int const pixelIdY,
          val_t const diceCoordX, val_t const diceCoordY, val_t const diceCoordZ,
          MS const & setup ) {
    checkCoords[0] = setup.pos0x() + ( -0.5                           + diceCoordX) * setup.segx();
    checkCoords[1] =                 ((-0.5 * setup.n0y() + pixelIdY) + diceCoordY) * setup.segy();
    checkCoords[2] =                 ((-0.5 * setup.n0z() + pixelIdZ) + diceCoordZ) * setup.segz();
  }
};

template<>
struct CalcCheckCoordsNoRot<Trafo1> {
  void operator()( val_t * const checkCoords,
          int const pixelIdZ, int const pixelIdY,
          val_t const diceCoordX, val_t const diceCoordY, val_t const diceCoordZ,
          MS const & setup ) {
    checkCoords[0] = setup.pos1x() + ( -0.5                           + diceCoordX) * setup.segx();
    checkCoords[1] =                 ((-0.5 * setup.n1y() + pixelIdY) + diceCoordY) * setup.segy();
    checkCoords[2] =                 ((-0.5 * setup.n1z() + pixelIdZ) + diceCoordZ) * setup.segz();
  }
};

template<>
struct CalcCheckCoordsNoRot<Trafo0i> {
  void operator()( val_t * const checkCoords,
          int const pixelIdZ, int const pixelIdY,
          val_t const diceCoordX, val_t const diceCoordY, val_t const diceCoordZ,
          MS const & setup ) {
    checkCoords[0] = setup.pos0x() + ( -0.5                           + diceCoordX) * setup.segx();
    checkCoords[1] =                 ((-0.5 * setup.n0y() + pixelIdY) + diceCoordY) * setup.segy();
    checkCoords[2] =                 ((-0.5 * setup.n0z() + pixelIdZ) + diceCoordZ) * setup.segz();
  }
};

template<>
struct CalcCheckCoordsNoRot<Trafo1i> {
  void operator()( val_t * const checkCoords,
          int const pixelIdZ, int const pixelIdY,
          val_t const diceCoordX, val_t const diceCoordY, val_t const diceCoordZ,
          MS const & setup ) {
    checkCoords[0] = setup.pos1x() + ( -0.5                           + diceCoordX) * setup.segx();
    checkCoords[1] =                 ((-0.5 * setup.n1y() + pixelIdY) + diceCoordY) * setup.segy();
    checkCoords[2] =                 ((-0.5 * setup.n1z() + pixelIdZ) + diceCoordZ) * setup.segz();
  }
};

/**
 * @brief Functor template. Calculates screwed transformed coordinates. Used to
 * check if tests fail when the should.
 * 
 * @tparam T_CalcCheckFunctor Type of functor for calculating coordinates to
 * check against whose results will be screwed.
 * @param checkCoords Output array.
 * @param pixelIdZ Index of detector pixel in z direction.
 * @param pixelIdY Index of detector pixel in y direction.
 * @param diceCoordX Relative coordinate within pixel along direction that is
 * x axis in non-rotated position.
 * @param diceCoordY Relative coordinate within pixel along direction that is
 * y axis in non-rotated position.
 * @param diceCoordZ Relative coordinate within pixel along direction that is
 * z axis in non-rotated position.
 * @param setup Measurement setup object.
 */
template<
        typename T_CalcCheckFunctor
>
struct ScrewedCalcCheckCoords {
  void operator()( val_t * const checkCoords,
          int const pixelIdZ, int const pixelIdY,
          val_t const diceCoordX, val_t const diceCoordY, val_t const diceCoordZ,
          MS const & setup ) {
    T_CalcCheckFunctor()(checkCoords,
            pixelIdZ, pixelIdY, diceCoordX, diceCoordY, diceCoordZ, setup);
    checkCoords[0] += 1.;
    checkCoords[1] += 1.;
    checkCoords[2] += 1.;
  }
};

/**
 * @brief Functor that performs BOOST_CHECK_EQUAL.
 */
struct BoostEqualityChecker {
  void operator()( val_t const first, val_t const second ) {
    BOOST_CHECK_EQUAL( first, second );
  }
};

/**
 * @brief Functor that performs BOOST_CHECK with inequality condition.
 */
struct BoostInequalityChecker {
  void operator()( val_t const first, val_t const second ) {
    BOOST_CHECK( first != second );
  }
};

/**
 * @brief Template argument struct for in place operation.
 */
struct TInPlace {};

/**
 * @brief Template argument struct for out of place operation.
 */
struct TOutOfPlace {};

/**
 * @brief Functor template. Tests a transformation functor for cases without
 * rotation.
 * 
 * @tparam T_Trafo Type of transformation functor.
 * @tparam T_Place In place / out of place.
 * @tparam T_CalcCheckFunctor Type of functor for calculating coordinates to
 * check against.
 * @param pixelIdZ Index of detector pixel in z direction.
 * @param pixelIdY Index of detector pixel in y direction.
 * @param diceCoordX Relative coordinate within pixel along direction that is
 * x axis in non-rotated position.
 * @param diceCoordY Relative coordinate within pixel along direction that is
 * y axis in non-rotated position.
 * @param diceCoordZ Relative coordinate within pixel along direction that is
 * z axis in non-rotated position.
 */
template<
        typename T_Trafo,
        typename T_CalcCheckFunctor,
        typename T_Place=TOutOfPlace,
        typename T_Checker=BoostEqualityChecker
>
struct TestTrafoNoRot {};

template<
        typename T_Trafo,
        typename T_CalcCheckFunctor,
        typename T_Checker
>
struct TestTrafoNoRot<
        T_Trafo,
        T_CalcCheckFunctor,
        TOutOfPlace,
        T_Checker
> {
  void operator()( int const pixelIdZ, int const pixelIdY, 
          val_t const diceCoordX, val_t const diceCoordY, val_t const diceCoordZ ) {
    // Create measurement setup object
    MS setup(POS0X, POS1X,
             NA, N0Z, N0Y, N1Z, N1Y,
             DA, SEGX, SEGY, SEGZ);

    // Calculate correct output coordinates
    val_t checkCoords[3];
    T_CalcCheckFunctor()(checkCoords,
            pixelIdZ, pixelIdY, diceCoordX, diceCoordY, diceCoordZ, setup);
    
    // Set dice coordinates
    val_t diceCoords[3] = {diceCoordX, diceCoordY, diceCoordZ};

    // Set output coordinates to a defined value
    val_t outCoords[3] = {0., 0., 0.};

    // Create transformation functor
    T_Trafo trafo;

    // Transform for pixel (pixelIdZ, pixelIdY) in rotation position 0
    trafo(outCoords, diceCoords, pixelIdZ, pixelIdY, 0, &setup);

    // Check output coordinates
    T_Checker()(outCoords[0], checkCoords[0]);
    T_Checker()(outCoords[1], checkCoords[1]);
    T_Checker()(outCoords[2], checkCoords[2]);  
  }
};

template<
        typename T_Trafo,
        typename T_CalcCheckFunctor,
        typename T_Checker
>
struct TestTrafoNoRot<
        T_Trafo,
        T_CalcCheckFunctor,
        TInPlace,
        T_Checker
> {
  void operator()( int const pixelIdZ, int const pixelIdY, 
          val_t const diceCoordX, val_t const diceCoordY, val_t const diceCoordZ ) {
    // Create measurement setup object
    MS setup(POS0X, POS1X,
             NA, N0Z, N0Y, N1Z, N1Y,
             DA, SEGX, SEGY, SEGZ);

    // Calculate correct output coordinates
    val_t checkCoords[3];
    T_CalcCheckFunctor()(checkCoords,
            pixelIdZ, pixelIdY, diceCoordX, diceCoordY, diceCoordZ, setup);
    
    // Set dice coordinates
    val_t coords[3] = {diceCoordX, diceCoordY, diceCoordZ};

    // Create transformation functor
    T_Trafo trafo;

    // Transform for pixel (pixelIdZ, pixelIdY) in rotation position 0
    trafo(coords, pixelIdZ, pixelIdY, 0, &setup);

    // Check output coordinates
    T_Checker()(coords[0], checkCoords[0]);
    T_Checker()(coords[1], checkCoords[1]);
    T_Checker()(coords[2], checkCoords[2]);  
  }
};

/**
 * @brief Rotation functor.
 */
struct Rotate {
  void operator()( val_t * const checkCoords,
          int const rotationId,
          MS const & setup ) {
    val_t oldZ = checkCoords[2];
    val_t oldX = checkCoords[0];
    val_t sin_ = sinf( val_t(M_PI)/val_t(180.) * rotationId * setup.da());
    val_t cos_ = cosf( val_t(M_PI)/val_t(180.) * rotationId * setup.da());
    checkCoords[2] = cos_*oldZ - sin_*oldX;
    checkCoords[0] = sin_*oldZ + cos_*oldX;
  }
};

/**
 * @brief Functor template. Tests a transformation functor for cases with
 * rotation.
 * 
 * @tparam T_Trafo Type of transformation functor.
 * @tparam T_Place In place / out of place.
 * @tparam T_CalcCheckFunctor Type of functor for calculating coordinates to
 * check against.
 * @param pixelIdZ Index of detector pixel in z direction.
 * @param pixelIdY Index of detector pixel in y direction.
 * @param diceCoordX Relative coordinate within pixel along direction that is
 * x axis in non-rotated position.
 * @param diceCoordY Relative coordinate within pixel along direction that is
 * y axis in non-rotated position.
 * @param diceCoordZ Relative coordinate within pixel along direction that is
 * z axis in non-rotated position.
 */
template<
        typename T_Trafo,
        typename T_CalcCheckFunctor,
        typename T_Place=TOutOfPlace,
        typename T_Checker=BoostEqualityChecker
>
struct TestTrafoRot {};

template<
        typename T_Trafo,
        typename T_CalcCheckFunctor,
        typename T_Checker
>
struct TestTrafoRot<
        T_Trafo,
        T_CalcCheckFunctor,
        TOutOfPlace,
        T_Checker
> {
  void operator()( int const pixelIdZ, int const pixelIdY, int const rotationId,
          val_t const diceCoordX, val_t const diceCoordY, val_t const diceCoordZ ) {
    // Create measurement setup object
    MS setup(POS0X, POS1X,
             NA, N0Z, N0Y, N1Z, N1Y,
             DA, SEGX, SEGY, SEGZ);

    // Calculate correct output coordinates
    val_t checkCoords[3];
    T_CalcCheckFunctor()(checkCoords,
            pixelIdZ, pixelIdY, diceCoordX, diceCoordY, diceCoordZ, setup);
    Rotate()(checkCoords, rotationId, setup);
    
    // Set dice coordinates
    val_t diceCoords[3] = {diceCoordX, diceCoordY, diceCoordZ};

    // Set output coordinates to a defined value
    val_t outCoords[3] = {0., 0., 0.};

    // Create transformation functor
    T_Trafo trafo;

    // Transform for pixel (pixelIdZ, pixelIdY) in rotation position 0
    trafo(outCoords, diceCoords, pixelIdZ, pixelIdY, rotationId, &setup);

    // Check output coordinates
    T_Checker()(outCoords[0], checkCoords[0]);
    T_Checker()(outCoords[1], checkCoords[1]);
    T_Checker()(outCoords[2], checkCoords[2]);  
  }
};

/*******************************************************************************
 * Case: det0, pixel 00, corner 000, no rotation
 ******************************************************************************/
BOOST_AUTO_TEST_CASE( test_MeasurementSetupTrafo2CartCoord_001 ){
  TestTrafoNoRot<Trafo0, CalcCheckCoordsNoRot<Trafo0> >()(0, 0, 0., 0., 0.);
}

/*******************************************************************************
 * Case: det0, pixel 00, corner 100, no rotation
 ******************************************************************************/
BOOST_AUTO_TEST_CASE( test_MeasurementSetupTrafo2CartCoord_002 ){
  TestTrafoNoRot<Trafo0, CalcCheckCoordsNoRot<Trafo0> >()(0, 0, 1., 0., 0.);
}

/*******************************************************************************
 * Case: det0, pixel 00, corner 010, no rotation
 ******************************************************************************/
BOOST_AUTO_TEST_CASE( test_MeasurementSetupTrafo2CartCoord_003 ){
  TestTrafoNoRot<Trafo0, CalcCheckCoordsNoRot<Trafo0> >()(0, 0, 0., 1., 0.);
}

/*******************************************************************************
 * Case: det0, pixel 00, corner 001, no rotation
 ******************************************************************************/
BOOST_AUTO_TEST_CASE( test_MeasurementSetupTrafo2CartCoord_004 ){
  TestTrafoNoRot<Trafo0, CalcCheckCoordsNoRot<Trafo0> >()(0, 0, 0., 0., 1.);
}

/*******************************************************************************
 * Case: det0, pixel 00, inner dice volume, no rotation
 ******************************************************************************/
BOOST_AUTO_TEST_CASE( test_MeasurementSetupTrafo2CartCoord_005 ){
  TestTrafoNoRot<Trafo0, CalcCheckCoordsNoRot<Trafo0> >()(0, 0, 0.1, 0.1, 0.1);
}

/*******************************************************************************
 * Case: det0, pixel 00, inner dice volume, no rotation, fail case
 ******************************************************************************/
BOOST_AUTO_TEST_CASE( test_MeasurementSetupTrafo2CartCoord_006 ){
  TestTrafoNoRot<
          Trafo0,
          ScrewedCalcCheckCoords<CalcCheckCoordsNoRot<Trafo0> >,
          TOutOfPlace,
          BoostInequalityChecker
  >()(0, 0, 0.1, 0.1, 0.1);
}

/*******************************************************************************
 * Case: det1, pixel 00, corner 000, no rotation
 ******************************************************************************/
BOOST_AUTO_TEST_CASE( test_MeasurementSetupTrafo2CartCoord_007 ){
  TestTrafoNoRot<Trafo1, CalcCheckCoordsNoRot<Trafo1> >()(0, 0, 0., 0., 0.);
}

/*******************************************************************************
 * Case: det1, pixel 00, corner 100, no rotation
 ******************************************************************************/
BOOST_AUTO_TEST_CASE( test_MeasurementSetupTrafo2CartCoord_008 ){
  TestTrafoNoRot<Trafo1, CalcCheckCoordsNoRot<Trafo1> >()(0, 0, 1., 0., 0.);
}

/*******************************************************************************
 * Case: det1, pixel 00, corner 010, no rotation
 ******************************************************************************/
BOOST_AUTO_TEST_CASE( test_MeasurementSetupTrafo2CartCoord_009 ){
  TestTrafoNoRot<Trafo1, CalcCheckCoordsNoRot<Trafo1> >()(0, 0, 0., 1., 0.);
}

/*******************************************************************************
 * Case: det1, pixel 00, corner 001, no rotation
 ******************************************************************************/
BOOST_AUTO_TEST_CASE( test_MeasurementSetupTrafo2CartCoord_010 ){
  TestTrafoNoRot<Trafo1, CalcCheckCoordsNoRot<Trafo1> >()(0, 0, 0., 0., 1.);
}

/*******************************************************************************
 * Case: det1, pixel 00, inner dice volume, no rotation
 ******************************************************************************/
BOOST_AUTO_TEST_CASE( test_MeasurementSetupTrafo2CartCoord_011 ){
  TestTrafoNoRot<Trafo1, CalcCheckCoordsNoRot<Trafo1> >()(0, 0, 0.1, 0.1, 0.1);
}

/*******************************************************************************
 * Case: det1, pixel 00, inner dice volume, no rotation, fail case
 ******************************************************************************/
BOOST_AUTO_TEST_CASE( test_MeasurementSetupTrafo2CartCoord_012 ){
  TestTrafoNoRot<
          Trafo1,
          ScrewedCalcCheckCoords<CalcCheckCoordsNoRot<Trafo1> >,
          TOutOfPlace,
          BoostInequalityChecker
  >()(0, 0, 0.1, 0.1, 0.1);
}

/*******************************************************************************
 * Case: det0, pixel 00, corner 000, no rotation
 ******************************************************************************/
BOOST_AUTO_TEST_CASE( test_MeasurementSetupTrafo2CartCoord_013 ){
  TestTrafoNoRot<
          Trafo0i,
          CalcCheckCoordsNoRot<Trafo0i>,
          TInPlace
  >()(0, 0, 0., 0., 0.);
}

/*******************************************************************************
 * Case: det0, in place, pixel 00, corner 100, no rotation
 ******************************************************************************/
BOOST_AUTO_TEST_CASE( test_MeasurementSetupTrafo2CartCoord_014 ){
  TestTrafoNoRot<
          Trafo0i,
          CalcCheckCoordsNoRot<Trafo0i>,
          TInPlace
  >()(0, 0, 1., 0., 0.);
}

/*******************************************************************************
 * Case: det0, in place, pixel 00, corner 010, no rotation
 ******************************************************************************/
BOOST_AUTO_TEST_CASE( test_MeasurementSetupTrafo2CartCoord_015 ){
  TestTrafoNoRot<
          Trafo0i,
          CalcCheckCoordsNoRot<Trafo0i>,
          TInPlace
  >()(0, 0, 0., 1., 0.);
}

/*******************************************************************************
 * Case: det0, in place, pixel 00, corner 001, no rotation
 ******************************************************************************/
BOOST_AUTO_TEST_CASE( test_MeasurementSetupTrafo2CartCoord_016 ){
  TestTrafoNoRot<
          Trafo0i,
          CalcCheckCoordsNoRot<Trafo0i>,
          TInPlace
  >()(0, 0, 0., 0., 1.);
}

/*******************************************************************************
 * Case: det0, in place, pixel 00, inner dice volume, no rotation
 ******************************************************************************/
BOOST_AUTO_TEST_CASE( test_MeasurementSetupTrafo2CartCoord_017 ){
  TestTrafoNoRot<
          Trafo0i,
          CalcCheckCoordsNoRot<Trafo0i>,
          TInPlace
  >()(0, 0, 0.1, 0.1, 0.1);
}

/*******************************************************************************
 * Case: det0, in place, pixel 00, inner dice volume, no rotation, fail case
 ******************************************************************************/
BOOST_AUTO_TEST_CASE( test_MeasurementSetupTrafo2CartCoord_018 ){
  TestTrafoNoRot<
          Trafo0,
          ScrewedCalcCheckCoords<CalcCheckCoordsNoRot<Trafo0> >,
          TOutOfPlace,
          BoostInequalityChecker
  >()(0, 0, 0.1, 0.1, 0.1);
}

/*******************************************************************************
 * Case: det1, in place, pixel 00, corner 000, no rotation
 ******************************************************************************/
BOOST_AUTO_TEST_CASE( test_MeasurementSetupTrafo2CartCoord_019 ){
    TestTrafoNoRot<
          Trafo1i,
          CalcCheckCoordsNoRot<Trafo1i>,
          TInPlace
  >()(0, 0, 0., 0., 0.);
}

/*******************************************************************************
 * Case: det1, in place, pixel 00, corner 100, no rotation
 ******************************************************************************/
BOOST_AUTO_TEST_CASE( test_MeasurementSetupTrafo2CartCoord_020 ){
    TestTrafoNoRot<
          Trafo1i,
          CalcCheckCoordsNoRot<Trafo1i>,
          TInPlace
  >()(0, 0, 1., 0., 0.);
}

/*******************************************************************************
 * Case: det1, in place, pixel 00, corner 010, no rotation
 ******************************************************************************/
BOOST_AUTO_TEST_CASE( test_MeasurementSetupTrafo2CartCoord_021 ){
    TestTrafoNoRot<
          Trafo1i,
          CalcCheckCoordsNoRot<Trafo1i>,
          TInPlace
  >()(0, 0, 0., 1., 0.);
}

/*******************************************************************************
 * Case: det1, in place, pixel 00, corner 001, no rotation
 ******************************************************************************/
BOOST_AUTO_TEST_CASE( test_MeasurementSetupTrafo2CartCoord_022 ){
    TestTrafoNoRot<
          Trafo1i,
          CalcCheckCoordsNoRot<Trafo1i>,
          TInPlace
  >()(0, 0, 0., 0., 1.);
}

/*******************************************************************************
 * Case: det1, in place, pixel 00, inner dice volume, no rotation
 ******************************************************************************/
BOOST_AUTO_TEST_CASE( test_MeasurementSetupTrafo2CartCoord_023 ){
    TestTrafoNoRot<
          Trafo1i,
          CalcCheckCoordsNoRot<Trafo1i>,
          TInPlace
  >()(0, 0, 0.1, 0.1, 0.1);
}

/*******************************************************************************
 * Case: det1, in place, pixel 00, inner dice volume, no rotation, fail case
 ******************************************************************************/
BOOST_AUTO_TEST_CASE( test_MeasurementSetupTrafo2CartCoord_024 ){
  TestTrafoNoRot<
          Trafo1i,
          ScrewedCalcCheckCoords<CalcCheckCoordsNoRot<Trafo1i> >,
          TInPlace,
          BoostInequalityChecker
  >()(0, 0, 0.1, 0.1, 0.1);
}

/*******************************************************************************
 * Case: det0, pixel 00, rotation 1, corner 000
 ******************************************************************************/
BOOST_AUTO_TEST_CASE( test_MeasurementSetupTrafo2CartCoord_025 ){
  TestTrafoRot<
          Trafo0,
          CalcCheckCoordsNoRot<Trafo0>
  >()(0, 0, 1, 0., 0., 0.);
}

BOOST_AUTO_TEST_SUITE_END() // end test_MeasurementSetupTrafo2CartCoord

