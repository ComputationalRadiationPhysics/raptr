project(raptr)
cmake_minimum_required(VERSION 3.3)

###############################################################################
# Options
###############################################################################
option(MEASURE_TIME "Perform time measurements during execution." ON)
if(MEASURE_TIME)
  add_definitions("-DMEASURE_TIME")
endif(MEASURE_TIME)

if("${GRID}" STREQUAL "")
  set(GRID "GRID64")
endif()
set(_GRID ${GRID} CACHE STRING "Description.")

# Add voxel grid macro names as options here:
set_property(CACHE _GRID PROPERTY STRINGS "GRID10" "GRID20" "GRID32" "GRID52" "GRID64" "GRID128")
add_definitions("-D${_GRID}")

## Additional debug information
#set(CUDA_CXX_FLAGS "${CUDA_CXX_FLAGS} -Xptxas=-v --source-in-ptx -rdynamic -lineinfo --keep --keep-dir nvcc_tm")


###############################################################################
# Find Boost
###############################################################################
find_package(Boost 1.60.0 COMPONENTS unit_test_framework REQUIRED)
include_directories(SYSTEM ${Boost_INCLUDE_DIRS})
set(LIBS ${LIBS} ${Boost_LIBRARIES})


###############################################################################
# Find CUDA
###############################################################################
find_package(CUDA 7.5 REQUIRED)

## Warning for Thrust-related Bug
function(warn_path_cuda PATH_VARIABLE)
    string(TOLOWER "$ENV{${PATH_VARIABLE}}" CPLUS_TOLOWER)
    string(FIND "${CPLUS_TOLOWER}" "cuda" CPLUS_TOLOWER_FOUND)
    if(NOT "${CPLUS_TOLOWER_FOUND}" MATCHES "-1")
        message(WARNING 
            "Your ${PATH_VARIABLE} seems to contain CUDA includes.\n"
            "This might result in the following error:\n"
            "'error: kernel launches from templates are not allowed in system files'\n"
            "To fix the problem, remove the CUDA includes from the path."
            "(A better place for these includes might be your CPATH)\n"
            "source: https://code.google.com/p/thrust/issues/detail?id=359#c5 \n"
            "issue:  https://github.com/ComputationalRadiationPhysics/haseongpu/issues/26"
            "CPATH:  https://gcc.gnu.org/onlinedocs/cpp/Environment-Variables.html \n")
    endif(NOT "${CPLUS_TOLOWER_FOUND}" MATCHES "-1")
endfunction(warn_path_cuda)

warn_path_cuda("CPLUS_INCLUDE_PATH")
warn_path_cuda("C_INCLUDE_PATH")


################################################################################
# Find MPI
################################################################################
find_package(MPI REQUIRED)
include_directories(SYSTEM ${MPI_C_INCLUDE_PATH})
set(LIBS ${LIBS} ${MPI_C_LIBRARIES})


################################################################################
# Find HDF5
################################################################################
find_package(HDF5 REQUIRED COMPONENTS CXX C)
include_directories(SYSTEM ${HDF5_INCLUDE_DIRS})
set(LIBS ${LIBS} ${HDF5_LIBRARIES})
if (${HDF5_IS_PARALLEL})
  message(FATAL_ERROR "HDF5 must be non-parallel!")
endif()


################################################################################
# Flags
################################################################################
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wextra")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-parameter")

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -arch=sm_35)


###############################################################################
# Inludes
###############################################################################
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/Ply)
cuda_include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
cuda_include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/Ply)


###############################################################################
# Targets
###############################################################################

## Target for libepetreco.a library
add_library( epetreco
    src/Ply/PlyGeometry.cpp
    src/Ply/CompositePlyGeometry.cpp
    src/Ply/PlyWriter.cpp
    src/Ply/PlyGeometry.hpp
    src/Ply/CompositePlyGeometry.hpp
    src/Ply/PlyWriter.hpp
    src/Ply/Ply.hpp
)

## Create build target for reco.cu
cuda_add_executable( reco src/tests/reco.cu )
target_link_libraries( reco epetreco )
target_link_libraries( reco ${LIBS} )
target_link_libraries( reco ${CUDA_cusparse_LIBRARY} )
target_link_libraries( reco ${CUDA_cublas_LIBRARY} )

## Create build target for backprojection.cu
cuda_add_executable( backprojection src/tests/backprojection.cu )
target_link_libraries( backprojection epetreco )
target_link_libraries( backprojection ${LIBS} )
target_link_libraries( backprojection ${CUDA_cusparse_LIBRARY} )
target_link_libraries( backprojection ${CUDA_cublas_LIBRARY} )

## Create build target for pureSMCalculation.cu
cuda_add_executable( pureSMCalculation src/tests/pureSMCalculation.cu )
target_link_libraries( pureSMCalculation epetreco )
target_link_libraries( pureSMCalculation ${LIBS} )
target_link_libraries( pureSMCalculation ${CUDA_cusparse_LIBRARY} )
target_link_libraries( pureSMCalculation ${CUDA_cublas_LIBRARY} )

## Create build target for each test file
add_custom_target(TESTING_EXECS)

list(APPEND CU_TEST_FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/tests/test_convertCsr2Ecsr.cu")
list(APPEND CU_TEST_FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/tests/test_cooSort.cu")
list(APPEND CU_TEST_FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/tests/test_CudaSMLine.cu")
list(APPEND CU_TEST_FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/tests/test_cusparseWrapper.cu")
list(APPEND CU_TEST_FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/tests/test_getSystemMatrixFromWorkqueue.cu")
list(APPEND CU_TEST_FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/tests/test_getWorkqueue.cu")
list(APPEND CU_TEST_FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/tests/test_MeasurementSetupLinIndex.cu")
list(APPEND CU_TEST_FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/tests/test_MeasurementSetupTrafo2CartCoord.cu")
list(APPEND CU_TEST_FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/tests/test_mlemOperations.cu")
list(APPEND CU_TEST_FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/tests/test_VoxelGridLinIndex.cu")

list(APPEND CPP_TEST_FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/tests/test_FileTalk.cpp")
list(APPEND CPP_TEST_FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/tests/test_H5DensityWriter.cpp")
list(APPEND CPP_TEST_FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/tests/test_H5Reader.cpp")
list(APPEND CPP_TEST_FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/tests/test_H5Writer.cpp")
list(APPEND CPP_TEST_FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/create_forwardprojection_input.cpp")


#file(GLOB CU_TEST_FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/tests/test_*.cu")
#file(GLOB CPP_TEST_FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/tests/test_*.cpp")

foreach(CU_FILE ${CU_TEST_FILES})
    get_filename_component(CU_OUT ${CU_FILE} NAME_WE)
    cuda_add_executable( ${CU_OUT} EXCLUDE_FROM_ALL ${CU_FILE})
    target_link_libraries( ${CU_OUT} epetreco )
    target_link_libraries( ${CU_OUT} ${LIBS} )
    target_link_libraries( ${CU_OUT} ${CUDA_cusparse_LIBRARY} )
    target_link_libraries( ${CU_OUT} ${CUDA_cublas_LIBRARY} )
    add_dependencies(TESTING_EXECS ${CU_OUT})
endforeach()

foreach(CPP_FILE ${CPP_TEST_FILES})
    get_filename_component(CPP_OUT ${CPP_FILE} NAME_WE)
    add_executable( ${CPP_OUT} EXCLUDE_FROM_ALL ${CPP_FILE} )
    target_link_libraries( ${CPP_OUT} epetreco )
    target_link_libraries( ${CPP_OUT} ${LIBS} )
    add_dependencies(TESTING_EXECS ${CPP_OUT})
endforeach()


# UT test cases
list(APPEND CPP_UT_FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/tests/UT.cpp")

list(APPEND CU_UT_FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/tests/cooSortUT.cu")
list(APPEND CU_UT_FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/tests/MeasurementSetupTrafo2CartCoordUT.cu")

cuda_add_executable(check_all EXCLUDE_FROM_ALL ${CPP_UT_FILES} ${CU_UT_FILES})
target_link_libraries(check_all ${LIBS})




# for later...
#
#find_package(Doxygen)
#if(DOXYGEN_FOUND)
#  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Doxygen.conf ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)
#  add_custom_target(doc
#    ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
#    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
#    COMMENT "Generating API documentation with Doxygen" VERBATIM
#    )
#endif(DOXYGEN_FOUND)
