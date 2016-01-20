/** @file test_CudaSMLine.cu */
#include <cuda.h>
#include <cuda_runtime.h>
#include "CUDA_HandleError.hpp"
#include "FileTalk.hpp"
#include "ChordsCalc_lowlevel.hpp"

#define N_SEGY 2
#define N_SEGZ 2
#define N_CHANNELS ((N_SEGZ*N_SEGY)*(N_SEGZ*N_SEGY))

#define EDGE_SEGX 2.0
#define EDGE_SEGY 0.4
#define EDGE_SEGZ 0.4
#define DET0POSX -40.0
#define DET0POSY 0.0
#define DET0POSZ 0.0
#define DET1POSX 40.0
#define DET1POSY 0.0
#define DET1POSZ 0.0

#define N_VOXELS   24
#define N_RAYSPERCHANNEL 100


inline __host__ __device__ size_t id0z( size_t const channelId ) {
  return  channelId / (N_SEGY * N_SEGZ * N_SEGY); }

inline __host__ __device__ size_t id0y( size_t const channelId ) {
  return (channelId % N_SEGZ) / (N_SEGZ * N_SEGY); }

inline __host__ __device__ size_t id1z( size_t const channelId ) {
  return (channelId %(N_SEGZ*N_SEGY)) /  (N_SEGY); }

inline __host__ __device__ size_t id1y( size_t const channelId ) {
  return  channelId %(N_SEGZ*N_SEGY*N_SEGZ); }


struct Setup
{
  __host__ __device__ void getRay(float * ray, size_t channelId)
  {
    float seg0pos[3];
    float seg1pos[3];
    seg0pos[0] = DET0POSX;
    seg0pos[1] = DET0POSY + (id0y(channelId)-0.5*(N_SEGY-1))*EDGE_SEGY;
    seg0pos[2] = DET0POSZ + (id0z(channelId)-0.5*(N_SEGZ-1))*EDGE_SEGZ;
    seg1pos[0] = DET1POSX;
    seg1pos[1] = DET1POSY + (id1y(channelId)-0.5*(N_SEGY-1))*EDGE_SEGY;
    seg1pos[2] = DET1POSZ + (id1z(channelId)-0.5*(N_SEGZ-1))*EDGE_SEGZ;

    ray[0] = seg0pos[0];
    ray[1] = seg0pos[1];
    ray[2] = seg0pos[2];
    ray[3] = seg1pos[0];
    ray[4] = seg1pos[1];
    ray[5] = seg1pos[2];
  }
};


// !!!
__host__ __device__ void calcChords( float * chords, size_t * voxelIds, float * ray )
{
  for(int chordId=0; chordId<N_VOXELS; chordId++)
  {
    chords[chordId] = 1.*chordId/N_RAYSPERCHANNEL;
    voxelIds[chordId] = chordId; // !!!
  }
}

__global__ void calcSM(float * sm, float * gridO, float * gridD, int * gridN)
{
  __shared__ float  chords[N_VOXELS];
  __shared__ int    voxelIds[N_VOXELS];
  
  size_t channelId = blockIdx.x;
  size_t offset    = blockIdx.x * N_VOXELS;
  size_t rayId     = threadIdx.x;

  /* Create ray */
  float ray[6];
  Setup().getRay(ray, channelId);
 
  // THIS CAUSES 'unspecified launch failure'
  /* Calc chords */
//  calcChords(chords, voxelIds, ray);
  getChords(chords, voxelIds, N_VOXELS, ray, gridO, gridD, gridN);
//
//  /* Write to sm */
//  for(int chordId=0; chordId<N_VOXELS; chordId++)
//    atomicAdd(&sm[offset+voxelIds[chordId]], chords[chordId]);
}


int main()
{
  /* Create grid */
  float gridO_host[] = {-0.2, -0.3, -0.4};
  float * gridO_devi;
  HANDLE_ERROR( cudaMalloc((void**)&gridO_devi, 3*sizeof(float)) );
  HANDLE_ERROR( cudaMemcpy(gridO_devi, gridO_host, 3*sizeof(float), cudaMemcpyHostToDevice) );
  float * gridD_devi;
  float gridD_host[] = {0.2, 0.2, 0.2};
  HANDLE_ERROR( cudaMalloc((void**)&gridD_devi, 3*sizeof(float)) );
  HANDLE_ERROR( cudaMemcpy(gridD_devi, gridD_host, 3*sizeof(float), cudaMemcpyHostToDevice) );
  int   * gridN_devi;
  int   gridN_host[] = {2, 3, 4};
  HANDLE_ERROR( cudaMalloc((void**)&gridN_devi, 3*sizeof(int)) );
  HANDLE_ERROR( cudaMemcpy(gridN_devi, gridN_host, 3*sizeof(int), cudaMemcpyHostToDevice) );

  /* Create system matrix */
  SAYLINES(__LINE__+1, __LINE__+4);
  float * sm_host;
  float * sm_devi;
  sm_host = (float*) malloc(N_CHANNELS*N_VOXELS*sizeof(*sm_host));
  HANDLE_ERROR( cudaMalloc((void**)&sm_devi, N_CHANNELS*N_VOXELS*sizeof(*sm_devi)) );
  
  /* Initialize host sm */
  for(int i=0; i<N_CHANNELS*N_VOXELS; i++)
    sm_host[i] = 0.;
  
  /* Compute sm */
  SAYLINES(__LINE__+1, __LINE__+2);
  calcSM<<<N_CHANNELS, N_RAYSPERCHANNEL>>>(sm_devi, gridO_devi, gridD_devi, gridN_devi);
  HANDLE_ERROR( cudaDeviceSynchronize() );

  /* Copy sm to host */
  SAYLINE(__LINE__+1);
  HANDLE_ERROR( cudaMemcpy(sm_host, sm_devi, N_CHANNELS*N_VOXELS*sizeof(*sm_devi), cudaMemcpyDeviceToHost) );

  /* Check results */
  SAYLINE(__LINE__-1);
  for(int i=0; i<N_VOXELS*N_CHANNELS; i++)
  {
    std::cout << sm_host[i] << " ";
    if((i+1)%N_VOXELS==0)
      std::cout << std::endl;
  }

  SAYLINE(__LINE__+1);
  HANDLE_ERROR( cudaFree(sm_devi) );
  return 0;
}
