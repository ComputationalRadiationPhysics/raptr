SRC = ./src
TESTS_SRC = ./src/tests
PLY = $(SRC)/Ply

CPPC = g++
override CPPCFLAGS += -Wall
CPPO = -c
INC = -I$(PLY) -I$(SRC) -I$(HDF5_ROOT)/include -I$(MPIROOT)/include
LIB = -L$(HDF5_ROOT)/lib -L./ -lhdf5 -lhdf5_cpp -lepetreco

CUC = nvcc
override CUCFLAGS += -DMEASURE_TIME -arch=sm_35 -Xcompiler -Wall
#override CUCFLAGS += -DMEASURE_TIME -DGRID64 -arch=sm_35 -Xptxas=-v --source-in-ptx -Xcompiler -rdynamic -lineinfo --keep --keep-dir nvcc_tmp --use_fast_math
CUCINC = -I$(CUDA_ROOT)/include -I$(PLY) -I$(SRC) -I$(HDF5_ROOT)/include -I$(MPIROOT)/include
CUCLIB = -L$(CUDA_ROOT)/lib64 -L./ -lcublas -lcusparse -L$(HDF5_ROOT)/lib -lhdf5 -lhdf5_cpp -lepetreco -L$(MPIROOT)/lib -lmpi -lmpi_cxx

# ##############################################################################
# ### PUT ALL DEFAULT TARGETS HERE
# ##############################################################################
# Define simply expanded variable for all executables. Synchronizes the clean
# rule with the default rule - everything that is compiled by default will also
# be cleaned up.
EXECUTABLES := \
      test_PlyGeometry.out \
      test_PlyBox.out \
      test_PlyWriter.out \
      test_PlyGrid.out \
      test_PlyLine.out \
      test_H5Reader.out \
      test_H5DensityWriter.out \
      test_CudaSMLine.out \
      test_MeasurementSetupLinIndex.out \
      test_MeasurementSetupTrafo2CartCoord.out \
      test_getWorkqueue.out \
      test_getSystemMatrixFromWorkqueue.out \
      test_VoxelGridLinIndex.out \
      test_cooSort.out \
      test_cusparseWrapper.out \
      test_mlemOperations.out \
      test_convertCsr2Ecsr.out \
      reco.out \
      backprojection.out \
      pureSMCalculation.out \
      test_H5DensityReader.out \
      example_condense_main.out \
      test_getSystemMatrixDeviceOnly.out \
      test_getWorkqueue-backprojection.out

default : $(EXECUTABLES)

clean:
	rm \
      $(EXECUTABLES) \
      ./libepetreco.a \
      ./*.ply \
      ./*.h5 2>/dev/null; \
  cd ./src/external_library_examples/ && make clean 2>/dev/null


libepetreco.a : \
        PlyGeometry.o \
        CompositePlyGeometry.o \
        PlyWriter.o
	rm -f $@
	ar rcs $@ $^


.INTERMEDIATE : \
        PlyGeometry.o \
        CompositePlyGeometry.o \
        PlyWriter.o



%.out : \
        $(TESTS_SRC)/%.cu
	CPLUS_INCLUDE_PATH= ; \
  $(CUC) $(CUCFLAGS) $(CUCINC) $(CUCLIB) $^ -o $@

%.out : \
      $(TESTS_SRC)/%.cpp
	$(CPPC) $(CPPCFLAGS) $(INC) $(LIB) $^ -o $@

%.out : \
        $(SRC)/%.cu
	CPLUS_INCLUDE_PATH= ; \
  $(CUC) $(CUCFLAGS) $(CUCINC) $(CUCLIB) $^ -o $@

test_Ply%.out : \
      $(PLY)/test_Ply%.cpp \
      libepetreco.a
	$(CPPC) $(CPPCFLAGS) $(INC) $^ -o $@

test_%_Siddon.out : \
      $(TESTS_SRC)/test_%_Siddon.cpp \
      $(SRC)/Siddon.hpp \
      $(SRC)/Siddon_helper.hpp \
      libepetreco.a
	$(CPPC) $(CPPFLAGS) $(INC) $^ -o $@

test_%.out : \
      $(TESTS_SRC)/test_%.cpp \
      $(SRC)/%.hpp \
      libepetreco.a
	$(CPPC) $(CPPCFLAGS) $(INC) $^ $(LIB) -o $@

test_Siddon.out : \
      $(TESTS_SRC)/test_Siddon.cpp\
      $(SRC)/Siddon.hpp
	$(CPPC) $(CPPFLAGS) $(INC) $^ -o $@

bigtest.out : \
      $(TESTS_SRC)/bigtest.cpp \
      libepetreco.a
	$(CPPC) $(CPPCFLAGS) $(INC) $^ -o $@



PlyGeometry.o : \
      $(PLY)/PlyGeometry.cpp \
      $(PLY)/PlyGeometry.hpp
	$(CPPC) $(CPPCFLAGS) $(INC) -c $< -o $@

CompositePlyGeometry.o : \
      $(PLY)/CompositePlyGeometry.cpp \
      $(PLY)/CompositePlyGeometry.hpp \
      $(PLY)/PlyGeometry.hpp
	$(CPPC) $(CPPCFLAGS) $(INC) -c $< -o $@

PlyWriter.o : \
      $(PLY)/PlyWriter.cpp\
      $(PLY)/PlyWriter.hpp
	$(CPPC) $(CPPCFLAGS) $(INC) -c $< -o $@



# build tests
build-tests: .build-tests-post

.build-tests-pre:
# Add your pre 'build-tests' code here...

.build-tests-post: .build-tests-impl
# Add your post 'build-tests' code here...


# run tests
test: .test-post

.test-pre:
# Add your pre 'test' code here...

.test-post: .test-impl
# Add your post 'test' code here...
