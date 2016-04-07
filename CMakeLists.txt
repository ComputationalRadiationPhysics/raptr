project(raptr)
cmake_minimum_required(VERSION 2.8.10.1)

###############################################################################
# Find CUDA
###############################################################################
find_package(CUDA 6.5 REQUIRED)

## Warning for Thrust-related Bug
function(warn_path_cuda PATH_VARIABLE)
    string(TOLOWER "$ENV{${PATH_VARIABLE}}" CPLUS_TOLOWER)
    string(FIND "${CPLUS_TOLOWER}" "cuda" CPLUS_TOLOWER_FOUND)
    if(NOT "${CPLUS_TOLOWER_FOUND}" MATCHES "-1")
        message(WARNING 
            "Your ${PATH_VARIABLE} seems to contain CUDA includes."
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
include_directories(${MPI_C_INCLUDE_PATH})
set(LIBS ${LIBS} ${MPI_C_LIBRARIES})


################################################################################
# Find HDF5
################################################################################
find_package(HDF5 REQUIRED COMPONENTS CXX C)
include_directories(${HDF5_INCLUDE_DIRS})
set(LIBS ${LIBS} ${HDF5_LIBRARIES})
if (${HDF5_IS_PARALLEL})
  message(FATAL_ERROR "HDF5 must be non-parallel!")
endif()


################################################################################
# Flags
################################################################################
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")

set(CUDA_CXX_FLAGS "${CUDA_CXX_FLAGS} -arch=sm_35 -Xcompiler")

## Additional debug information
#set(CUDA_CXX_FLAGS "${CUDA_CXX_FLAGS} -Xptxas=-v --source-in-ptx -rdynamic -lineinfo --keep --keep-dir nvcc_tm")

add_definitions("-DGRID64")
add_definitions("-DMEASURE_TIME")


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

## Create build target for each test file
file(GLOB CU_TEST_FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/tests/test_*.cu")
file(GLOB CPP_TEST_FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/tests/test_*.cpp")

add_custom_target(ttt )

foreach(CU_FILE ${CU_TEST_FILES})
    get_filename_component(CU_OUT ${CU_FILE} NAME_WE)
    cuda_add_executable( ${CU_OUT} EXCLUDE_FROM_ALL ${CU_FILE})
    target_link_libraries( ${CU_OUT} epetreco )
    target_link_libraries( ${CU_OUT} ${LIBS} )
    target_link_libraries( ${CU_OUT} ${CUDA_cusparse_LIBRARY} )
    target_link_libraries( ${CU_OUT} ${CUDA_cublas_LIBRARY} )
    add_dependencies(ttt ${CU_OUT})
endforeach()

foreach(CPP_FILE ${CPP_TEST_FILES})
    get_filename_component(CPP_OUT ${CPP_FILE} NAME_WE)
    add_executable( ${CPP_OUT} EXCLUDE_FROM_ALL ${CPP_FILE} )
    target_link_libraries( ${CPP_OUT} epetreco )
    target_link_libraries( ${CPP_OUT} ${LIBS} )
    add_dependencies(ttt ${CU_OUT})
endforeach()







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
