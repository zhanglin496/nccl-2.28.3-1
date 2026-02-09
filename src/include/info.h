/*************************************************************************
 * Copyright (c) 2019-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_INFO_H_
#define NCCL_INFO_H_

#include "nccl.h"
#include "collectives.h"
#include "core.h"
#include "utils.h"

// Used to pass NCCL call information between functions
struct ncclInfo {
//集合操作类型
  ncclFunc_t coll;
  //操作名称
  const char* opName;
  
  // NCCL Coll Args
  //发送缓冲区
  const void* sendbuff;
  //接收缓冲区
  void* recvbuff;
  //buff中的元素个数
  //buff中有多少个datatype类型的数据
  size_t count;
  
  //数据类型
  ncclDataType_t datatype;
  //归约操作类型
  ncclRedOp_t op;
  int root; // peer for p2p operations
  
  ncclComm_t comm;
  cudaStream_t stream;
  
  // Algorithm details
  //块数
  int chunkSteps;
  //切片数
  int sliceSteps;
};

#endif
