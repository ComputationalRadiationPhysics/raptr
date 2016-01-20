/* 
 * File:   chunkCopying.hpp
 * Author: malte
 *
 * Created on 8. Oktober 2015, 17:08
 */

#ifndef CHUNKCOPYING_HPP
#define	CHUNKCOPYING_HPP

//#include "GpuChunkHandle.hpp"
#include "CpuChunkHandle.hpp"
#include "H5ChunkHandle.hpp"
//#include "cuda_wrappers.hpp"

void cpyChunkH52Cpu(  CpuChunkHandle & target, H5ChunkHandle &  origin ) {
  /* Open DataSets, Attributes */
  H5::DataSet valDataSet     = origin.file.openDataSet("val");
  H5::DataSet colIdDataSet   = origin.file.openDataSet("colId");
  H5::DataSet rowPtrDataSet  = origin.file.openDataSet("rowPtr");
  H5::Attribute nElemsAttr   = valDataSet.openAttribute("nElems");
  H5::Attribute nMemRowsAttr = rowPtrDataSet.openAttribute("nMemRows");

  /* Copy data */
  valDataSet.read(   target.val,    H5::PredType::NATIVE_FLOAT);
  colIdDataSet.read( target.colId,  H5::PredType::NATIVE_INT);
  rowPtrDataSet.read(target.rowPtr, H5::PredType::NATIVE_INT);
  nElemsAttr.read(   H5::PredType::NATIVE_INT, &(target.nElems));
  nMemRowsAttr.read(     H5::PredType::NATIVE_INT, &(target.nMemRows));
}

void cpyChunkCpu2H5(  H5ChunkHandle &  target, CpuChunkHandle & origin) {
  /* Open DataSets, Attributes */
  H5::DataSet valDataSet     = target.file.openDataSet("val");
  H5::DataSet colIdDataSet   = target.file.openDataSet("colId");
  H5::DataSet rowPtrDataSet  = target.file.openDataSet("rowPtr");
  H5::Attribute nElemsAttr   = valDataSet.openAttribute("nElems");
  H5::Attribute nMemRowsAttr = rowPtrDataSet.openAttribute("nMemRows");

  /* Write */
  valDataSet.   write(origin.val,    H5::PredType::NATIVE_FLOAT);
  colIdDataSet. write(origin.colId,  H5::PredType::NATIVE_INT);
  rowPtrDataSet.write(origin.rowPtr, H5::PredType::NATIVE_INT);
  nElemsAttr.write(  H5::PredType::NATIVE_INT, &(origin.nElems));
  nMemRowsAttr.write(H5::PredType::NATIVE_INT, &(origin.nMemRows));
}

//void cpyChunkCpu2Gpu( GpuChunkHandle & target, CpuChunkHandle & origin) {
//  memcpyH2D<val_t>(         target.val,         origin.val,         origin.nElems);
//  memcpyH2D<int>  (         target.colId,       origin.colId,       origin.nElems);
//  memcpyH2D<int>  (         target.rowPtr,      origin.rowPtr,      (origin.nMemRows)+1);
//  memcpyH2D<MemArrSizeType>(&(target.nElems),   &(origin.nElems),   1);
//  memcpyH2D<int>           (&(target.nMemRows), &(origin.nMemRows), 1);
//}
//
//void cpyChunkGpu2Cpu( CpuChunkHandle & target, GpuChunkHandle & origin) {
//  memcpyD2H<val_t>(         target.val,         origin.val,         origin.nElems);
//  memcpyD2H<int>  (         target.colId,       origin.colId,       origin.nElems);
//  memcpyD2H<int>  (         target.rowPtr,      origin.rowPtr,      (origin.nMemRows)+1);
//  memcpyD2H<MemArrSizeType>(&(target.nElems),   &(origin.nElems),   1);
//  memcpyD2H<int>           (&(target.nMemRows), &(origin.nMemRows), 1);
//}

#endif	/* CHUNKCOPYING_HPP */

