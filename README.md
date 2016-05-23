**raptr**
=====

Intro
-----
**raptr** stands for *reconstruction of activity from PET data using rays*. It is a code developped at the HZDR for iterative algebraic image reconstruction of activity densities from PET measurements. **raptr** uses CUDA devices for acceleration and runs parallely across multiple nodes.

*******************************************************************************


General description of PET and reconstruction
-----
PET is an imaging technique based on the annihilation of electrons from the object of study with positrons. The imaging makes use of the fact that the two photons emerging from the annihilation are emitted in opposite directions at an angle of nearly 180 degrees. Detecting both photons justifies the statement that an annihilation has occured somewhere on the line that connects both detection places (line of response, LOR). This mapping of annihilation places onto lines of response can be described as a Radon transformation. The inversion of this surjective function using a large number of measured LORs is called image reconstruction.
The positrons can have various origins. For example beta decay of radioactive isotopes (radio pharmaceuticals) or irradiation induced creation (pair creation near nuclei) may be utilized.

Imaging techniques can on the abstract level be described as follows: An object of the image space is mapped to an object in measurement space by some detector system. Subsequently, this mapping is inverted to obtain an image of the object. This second step is called reconstruction. It may be seen as two distinct tasks:

1. Create a theoretical model of the detector system.

2. Use some method to produce an image from the knowledge of the detector model and measurement data.

There are many methods for image reconstructiion from PET, including:

* Simple summed backprojection -- Spatially sum up all measured LORs. Fast, easily implemented, low image quality.

* Filtered backprojection -- For perfectly line-like LORs this is the analytical inverse of the Radon transformation. Fast but real-world volume-like LORs give rise to problems.

* Algebraic reconstruction techniques (ART).

*******************************************************************************


Description of the used algorithm
-----
Here we describe the algorithm used for the reconstruction part of the imaging process. As outlined above, it consists firstly of theoretically modelling the detector and secondly the calculational inversion of the mapping introduced by the measurement.

For the second task, inverting the mapping, two methods have been implemented so far. One is the *summed backprojection*, the other is known as *maximum likelihood exspectation maximization* or *MLEM*. Both are commen recontruction techniques for PET and are thoroughly described in the literature.

For the first task, modelling the detector system, our approach is described below. The detector system model is represented by a system matrix. Algorithm:

* Read the measurement data (list mode).
* For each detector channel in the measurement:
    * Find a - preferably small - set of voxels that has as a subset all voxels that are seen by the detector channel. Currently that set is found by overestimating channels by their enclosing cylinders.
* For each detector channel in the measurement:
    * For each voxel in the set:
        * Calculate the corresponding system matrix element by sampling the channel with rays defined by randomly distributed start and end points within the detector pixel volumes. The system matrix element is calculated as the intersection length between a ray and the voxel averaged over all sample rays for that channel.

*******************************************************************************


Glossary
-----
The following terms are used in the sense as defined below.

* **Channel**:
  What is often called a *line of response* (*LOR*) in the context of PET. A channel is fully characterized by specification of two detector pixels that can record a PET event simultaneously and, for rotatable detectors, the detector rotation angle(s).

* <a id="image-space">**Image space**</a>:
  Discretized natural euklidian space. Referred to as *X*. Described as a 3D grid of *N* voxels that covers all allowed annihilation places. An object in image space is defined by a full 3D vector of *N* values. Objects in image space are called *images* or sometimes (somewhat unprecisely) *densities*. In formulas they are referred to as *x* and their serialized index which enumerates th voxels is *i*.

* **Measurement setup**:
  The geometrical configuration of the measurement and the disretization of measurement channels introduced by the used detector system.

* <a id="measurement-space">**Measurement space**</a>:
  Discretized space of allowed measurements. Referred to as *Y*. Its dimension *M* is the number of channels which is the same as the number of distinguishable measurement events. *Y* is naturally defined by the setup of the measurement. Objects in measurement space are *measurements* referred to as *y*. Their serialized index which enumerates the channels is *j*.

* **Ray**:
  Mathematical line segment defined by a start point and an end point.

* **System matrix**:
  Defines the theoretical (linear) mapping from image space to measurement space. Referred to as *A*. With an activity distribution *x* and a measurement *y*, the mapping is modelled as *y = A x*. The system matrix element *(i, j)* (indices as defined in [image space](#image-space) and [measurement space](#measurement-space)) is proportional to the probability of an annihilation within voxel *j* under the premise of its detection in channel *i*.

* **Voxel grid**:
  [See *image space*](#image-space)

*******************************************************************************


Install
-----
### Mandatory requirements
* **gcc** (getestet: 4.8.2)
* [**CUDA (including CUDA Toolkit)**](https://developer.nvidia.com/cuda-downloads) (getestet: 6.5)
* at least one **CUDA** capable **GPU**
    - Compute capability **sm_35** or higher
* **OpenMPI** (getestet: 1.8.4)
* **HDF5** (getestet: 1.8.14, non-parallel, c++)
* **git** (getestet: 1.7.9.5)

### Mandatory environment variables
* `CUDA_ROOT`: CUDA installation directory
* `MPIROOT`: MPI installation directory
* `HDF5_ROOT`: HDF5 installation directory

### <a id="installation-step-by-step">Installation step by step</a>
The following steps will install **raptr** in the standard configuration. But you can also [use a different configuration](#configuration).

1. **Get access to the source code**:
    1. [Sign up for GitHub](https://help.github.com/articles/signing-up-for-a-new-github-account/)
    2. Send an E-Mail to m.zacharias@hzdr.de **asking** for *access to raptr* and **telling** your GitHub username
    3. Wait to be added as a collaborator.
2. **Setup directories**: `mkdir ~/src`
3. **Download the source code**:
    1. `git clone https://github.com/ComputationalRadiationPhysics/raptr.git ~/src/raptr`
        - *optional*: update the source code with `cd ~/src/raptr && git pull`
4. **Compile**: `cd ~/src/raptr && make reco.out && make backprojection.out`

*******************************************************************************


Usage
-----
Usable **raptr** executables include `reco.out`, `backprojection.out` and `pureSMCalculation.out`. 

### <a id="configuration">Configuration</a>
**raptr** executables are always compiled for a *specific voxel grid* and a *specific measurement setup* and time measurement can either be on or off. The standard configuration is:

* Voxel grid with 64x64x64 voxels, for details see [voxelgrid_defines.h](src/voxelgrid_defines.h)
* Measurement setup with 180x13x13x13x13 channels, for details see [real_measurementsetup_defines.h](src/real_measurementsetup_defines.h)
* Time measurement on

For a different configuration, alterations have to be made before step *"4. Compile"* during the [installation](#installation-step-by-step).

* **Use a different voxel grid**:
   1. *Optional*: Define your voxel grid in `voxelgrid_defines.h` and make it switchable by a precompiler macro like the others
   2. Set your voxel grid's macro in the `Makefile`

* **Use a different measurement setup**: *This will only work for setups similar to ours!*
   * Define your measurement setup by altering the numerical values in `real_measurementsetup_defines.h`

* **Use/don't use time measurement during code execution**:
   * Set/unset the precompiler macro `MEASURE_TIME`

### Run
**raptr** executables have to be started with `mpiexec`. To run them:

1. Initialize the runtime environment. This could be: Start a job on your job system-managed cluster.

2. In the runtime environment run: 
   ```bash
   mpiexec --prefix $MPIHOME --npernode NPN -x LD_LIBRARY_PATH -x LIBRARY_PATH -n N EXEC <ARGS>
   ```
   
      * `NPN`:
         * Number of processes per node.
         * MUST NOT exceed the number of CUDA devices in a single node.
      
      * `N`:
         * Total number of processes.
         * MUST NOT exceed the total number CUDA devices available in the runtime environment.
      
      * `EXEC`:
         * The executable.
      
      * `<ARGS>`:
         * Runtime arguments.
         * Specific to `EXEC`.

**Important note on how executables use GPUs:** 
MPI instances of **raptr** executables will use a CUDA device that is visible to them. That may lead to conflicting CUDA device access attempts. To avoid such conflicts, make sure that no device is visible to more than one process. E.g. set the environment variable `CUDA_VISIBLE_DEVICES` accordingly.

### Programs in particular
#### reco.out
... is the executable that actually performs the reconstruction.

**reco.out** is called like this:
```bash
reco.out MEAS_FN ACTI_FN NRAYS SENS_FN NIT GUESS_FN
```
All arguments are mandatory. Their meaning is as follows:

* `MEAS_FN`:
  Filename of measurement file.

* `ACTI_FN`:
  Filename suffix for activity (output) files. The suffix will be prepanded with the current iteration of the reconstruction followed by an underscore `_`.

* `NRAYS`:
  Number of rays that are used for sampling a single detector channel.

* `SENS_FN`:
  Filename of sensitivity file. Sensitivity files are also density files.

* `NIT`:
  Number of reconstruction iterations.

* `GUESS_FN`:
  Filename of first guess file.

#### **backprojection.out**
... is the executable that performs an unfiltered backprojection of measurement data onto the voxel grid.

**backprojection.out** is called like this:
```bash
backprojection.out MEAS_FN SUMBP_FN NRAYS
```
All arguments are mandatory. Their meaning is as follows:

* `MEAS_FN`:
  Filename of measurement file.

* `SUMBP_FN`:
  Filename for backprojected density (output) file.

* `NRAYS`:
  Number of rays that are used for sampling a single detector channel.

#### **pureSMCalculation.out**
... is an executable that is mainly used for time measuring.

**pureSMCalculation.out** is called like this:
```bash
pureSMCalculation.out MEAS_FN NRAYS
```
All arguments are mandatory. Their meaning is as follows:

* `MEAS_FN`:
  Filename of measurement file.

* `NRAYS`:
  Number of rays that are used for sampling a single detector channel.

*******************************************************************************


Data Formats
-----

### Input Data
Several **raptr** executables, including the reconstruction executables, need a PET measurement file as input. These files store information about event counts in each of the measurement channels. Measurement files must be HDF5 files with a special structure that represents the spatial ordering of the detector channels:

**Structure of measurement files:**

> MUST have dataset "**messung**":
> - datatype float (H5T_IEEE_F32LE)
> - stores event count values
> - 5-dimensional simple dataspace with (**dim0**, **dim1**, **dim2**, **dim3**, **dim4**) representing:
> - **dim0**: rotation angle
> - **dim1**: z-axis on detector 0
> - **dim2**: y-axis on detector 0
> - **dim3**: z-axis on detector 1
> - **dim4**: y-axis on detector 1

In the current implementation, the input data files are assumed to have:
> (**dim0**, **dim1**, **dim2**, **dim3**, **dim4**) = (180, 13, 13, 13, 13)

Additional content in the input files is allowed but not part of the data format.



### Output Data
The executable **reco.out** writes the reconstructed activity into a density file. Activity is stored as one value per voxel of the reconstruction voxel grid. The density file is of HDF5 type and has a special structure that represents the spatial ordering of the voxels:

**Structure of density files:**

> MUST have dataset "**density**":
>
> - datatype float (H5T_IEEE_F32LE)
> - 3-dimensional simple dataspace with (**dim0**, **dim1**, **dim2**) = (**xnbin**, **ynbin**, **znbin**)
>
>   scalar attributes:
> - **xmin** (float), position of lower x voxel grid surface
> - **xmax** (float), postion of higher x voxel grid surface
> - **xnbin** (int: H5T_STD_I32LE), number of voxels in x direction
> - **ymin** (float), position of lower y voxel grid surface
> - **ymax** (float), postion of higher y voxel grid surface
> - **ynbin** (int), number of voxels in y direction
> - **zmin** (float), position of lower z voxel grid surface
> - **zmax** (float), postion of higher z voxel grid surface
> - **znbin** (int), number of voxels in z direction
