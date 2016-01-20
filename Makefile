SRC = ./src
TESTS_SRC = $(SRC)/tests
PLY = $(SRC)/Ply

CPPC = g++
override CPPCFLAGS += -Wall
CPPO = -c
INC = -I$(PLY) -I$(SRC) -I$(HDF5_ROOT)/include -I$(MPIROOT)/include
LIB = -L$(HDF5_ROOT)/lib -lhdf5 -lhdf5_cpp

EPETLIB = -L./ -lepetreco

CUC = nvcc
override CUCFLAGS += -DMEASURE_TIME -DGRID64 -arch=sm_35 -Xcompiler -Wall
#override CUCFLAGS += -DMEASURE_TIME -DGRID64 -arch=sm_35 -Xptxas=-v --source-in-ptx -Xcompiler -rdynamic -lineinfo --keep --keep-dir nvcc_tmp --use_fast_math
CUCINC = -I$(CUDA_ROOT)/include -I$(PLY) -I$(SRC) -I$(HDF5_ROOT)/include -I$(MPIROOT)/include
CUCLIB = -L$(CUDA_ROOT)/lib64 -lcublas -lcusparse -L$(HDF5_ROOT)/lib -lhdf5 -lhdf5_cpp -L$(MPIROOT)/lib -lmpi -lmpi_cxx

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
      pureSMCalculation.out
#      test_getWorkqueue-backprojection.out \
#      example_condense_main.out \
#      test_getSystemMatrixDeviceOnly.out \

default : $(EXECUTABLES)

clean:
	rm \
        $(EXECUTABLES) \
        ./libepetreco.a

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
      $(TESTS_SRC)/%.cu \
      libepetreco.a
	CPLUS_INCLUDE_PATH= ; \
        $(CUC) $(CUCFLAGS) $(CUCINC) $< $(CUCLIB) $(EPETLIB) -o $@

%.out : \
      $(TESTS_SRC)/%.cpp \
      libepetreco.a
	$(CPPC) $(CPPCFLAGS) $(INC) $< $(LIB) $(EPETLIB) -o $@

%.out : \
      $(SRC)/%.cu \
      libepetreco.a
	CPLUS_INCLUDE_PATH= ; \
        $(CUC) $(CUCFLAGS) $(CUCINC) $< $(CUCLIB) $(EPETLIB) -o $@

%.out: \
      $(SRC)/%.cpp \
      libepetreco.a
	$(CPPC) $(CPPCFLAGS) $(INC) $< $(LIB) $(EPETLIB) -o $@

test_Ply%.out : \
      $(PLY)/test_Ply%.cpp \
      libepetreco.a
	$(CPPC) $(CPPCFLAGS) $(INC) $< $(EPETLIB) -o $@

test_%.out : \
      $(TESTS_SRC)/test_%.cpp \
      $(SRC)/%.hpp \
      libepetreco.a
	$(CPPC) $(CPPCFLAGS) $(INC) $< $(LIB) $(EPETLIB) -o $@



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
