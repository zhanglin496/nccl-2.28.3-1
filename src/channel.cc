/*************************************************************************
 * Copyright (c) 2015-2022, NVIDIA CORPORATION. All rights reserved.
 * 版权所有 (c) 2015-2022, NVIDIA CORPORATION. 保留所有权利。
 *
 * See LICENSE.txt for license information
 * 请参阅 LICENSE.txt 获取许可证信息
 ************************************************************************/

// 引入通道头文件，包含通道相关的结构和函数声明
#include "channel.h"
// 引入参数头文件，包含 NCCL 参数定义和获取函数
#include "param.h"
// 引入 GPU Direct RDMA 包装头文件，提供 GPU Direct 相关功能
#include "gdrwrap.h"
// 引入传输层头文件，包含各种传输方式的实现
#include "transport.h"

// 函数实现：初始化通道
// 此函数初始化指定的通信通道，分配必要的资源并建立数据结构
// 参数 comm: NCCL 通信器指针
// 参数 channelId: 要初始化的通道 ID
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
ncclResult_t initChannel(struct ncclComm* comm, int channelId) {
  // 获取指定通道的指针
  struct ncclChannel* channel = &comm->channels[channelId];
  // 检查通道是否已经初始化（id 不为 -1 表示已初始化）
  if (channel->id != -1)
    return ncclSuccess;                                // 如果已初始化，直接返回成功

  int nRanks = comm->nRanks;                            // 获取通信域中的 rank 总数
  int nvlsRanks = comm->localRanks;                     // 获取本地 rank 数量（用于 NVLS）
  // 计算对等节点总数：普通 ranks + CollNet 网络节点 + NVLS 节点
  int nPeers = nRanks + 1 /* Collnet */ + nvlsRanks /* NVLS */;
  //设置channel id
  // 设置通道 ID
  channel->id = channelId;                              // 设置通道的 ID
  channel->workFifoProduced = 0;                        // 初始化工作 FIFO 生产计数为 0

  // 获取共享资源指针（包含可以在多个通信器间共享的资源）
  struct ncclSharedResources* sharedRes = comm->sharedRes;
  cudaStream_t deviceStream;                            // 声明 CUDA 流变量
  // 获取设备流，用于 CUDA 操作
  // ncclCudaGraphNone() 表示不使用 CUDA 图
  // concurrent=false 表示不与其他操作并发执行
  NCCLCHECK(ncclStrongStreamAcquire(ncclCudaGraphNone(), &sharedRes->deviceStream, /*concurrent=*/false, &deviceStream));

  // 检查通道的对等节点指针是否已分配
  if (channel->peers == NULL) {
    // The extra on nRanks+1 is for collnet root (i.e. network)
    // nRanks+1 中的额外一个是为 CollNet 根节点（即网络节点）
    // Allocate everything related to sharedRes with ncclCalloc as this can be
    // 使用 ncclCalloc 分配与 sharedRes 相关的所有内容，因为这可以
    // shared between communicators hence should not be tied to comm.
    // 在通信器之间共享，因此不应绑定到单个通信器
    // 检查共享资源中的对等节点数组是否已分配
    if (sharedRes->peers[channelId] == NULL) {
      // 分配线程对本地 rank 数量的对等节点数组
      NCCLCHECK(ncclCalloc(sharedRes->peers + channelId, sharedRes->tpNRanks));
    }
    // 为当前通道分配对等节点数组（从永久内存栈中分配，不会自动释放）
    channel->peers = ncclMemoryStackAlloc<struct ncclChannelPeer*>(&comm->memPermanent, nPeers);
    // 初始化对等节点指针数组
    for (int r = 0; r < nRanks; r++) {
      // 获取共享资源中对等节点的指针（使用 topParentRanks 数组进行映射）
      channel->peers[r] = comm->sharedRes->peers[channelId] + comm->topParentRanks[r];
      // 增加对等节点的引用计数（表示有多少个通道在使用此对等节点）
      ncclAtomicRefCountIncrement(&channel->peers[r]->refCount);
    }
  }


  // 检查设备侧对等节点指针是否已分配
  if (channel->devPeers == NULL) {
    // 检查共享资源中的设备对等节点数组是否已分配
    if (sharedRes->devPeers[channelId] == NULL) {
        //分配GPU内存
        // 异步分配 GPU 内存（在线性内存中）
      NCCLCHECK(ncclCudaCallocAsync(sharedRes->devPeers + channelId, sharedRes->tpNRanks, deviceStream));
    }
    /* channel->devPeers is not shared, so just free it when calling commFree() */
    /* channel->devPeers 不共享，因此在调用 commFree() 时直接释放 */
    // 为当前通道分配设备对等节点数组（非共享）
    NCCLCHECK(ncclCudaCallocAsync(&channel->devPeers, nPeers, deviceStream));
    // 将此内存推送到通信器的 CUDA 释放队列，通信器销毁时自动释放
    ncclCommPushCudaFree(comm, channel->devPeers);
    // 分配主机侧的设备对等节点指针数组
    NCCLCHECK(ncclCalloc(&channel->devPeersHostPtr, nPeers));
    // 初始化设备对等节点指针数组
    for (int r = 0; r < nRanks; r++) {
      // 获取共享资源中设备对等节点的地址
      uintptr_t addr = (uintptr_t)(comm->sharedRes->devPeers[channelId] + comm->topParentRanks[r]);
      // 将设备对等节点指针复制到 GPU 内存
      NCCLCHECK(ncclCudaMemcpyAsync((uintptr_t*)(channel->devPeers + r), (uintptr_t*)&addr, 1, deviceStream));
      // 保存主机侧的设备对等节点指针
      channel->devPeersHostPtr[r] = (struct ncclDevChannelPeer*)addr;
    }
  }

  // 分配环形算法的用户 rank 数组（存储所有 rank 的编号）
  channel->ring.userRanks = ncclMemoryStackAlloc<int>(&comm->memPermanent, nRanks);
  // 在设备上分配环形用户 rank 数组
  NCCLCHECK(ncclCudaCallocAsync(&channel->devRingUserRanks, nRanks, deviceStream));
  // 将此内存推送到通信器的 CUDA 释放队列
  ncclCommPushCudaFree(comm, channel->devRingUserRanks);

  /* guarantee addr has been copied into channel->devPeers */
  /* 保证地址已经复制到 channel->devPeers 中 */
  // 释放设备流
  NCCLCHECK(ncclStrongStreamRelease(ncclCudaGraphNone(), &sharedRes->deviceStream, /*concurrent=*/false));
  // 同步设备流，确保所有异步操作完成
  NCCLCHECK(ncclStrongStreamSynchronize(&sharedRes->deviceStream));
  return ncclSuccess;                                  // 返回成功状态
}

// 函数实现：初始化 NVLS 通道
// 此函数为通道添加 NVLS（NVLink fabric）支持
// 参数 comm: NCCL 通信器指针
// 参数 channelId: 通道 ID
// 参数 parent: 父通信器指针（用于共享资源）
// 参数 share: 是否共享父通信器的资源
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
ncclResult_t initNvlsChannel(struct ncclComm* comm, int channelId, struct ncclComm* parent, bool share) {
  // 获取指定通道的指针
  struct ncclChannel* channel = &comm->channels[channelId];
  // 获取共享资源指针
  struct ncclSharedResources* sharedRes = comm->sharedRes;
  cudaStream_t deviceStream;                            // 声明 CUDA 流变量

  // 检查 NVLS 对等节点是否已初始化
  if (channel->nvlsPeers != NULL)
    return ncclSuccess;                                // 如果已初始化，直接返回成功

  // 如果通道尚未初始化，先初始化基础通道
  if (channel->id == -1)
    NCCLCHECK(initChannel(comm, channelId));

  // 获取设备流，用于 CUDA 操作
  NCCLCHECK(ncclStrongStreamAcquire(ncclCudaGraphNone(), &sharedRes->deviceStream, /*concurrent=*/false, &deviceStream));

  int nvlsRanks = comm->localRanks;                     // 获取本地 rank 数量（NVLS 节点数量）

  // 根据是否共享资源采用不同的初始化策略
  if (share) {                                          // 如果共享父通信器的资源
    // 直接使用父通信器的 NVLS 资源指针
    channel->nvlsPeers = parent->channels[channelId].nvlsPeers;
    channel->nvlsDevPeers = parent->channels[channelId].nvlsDevPeers;
    // 初始化 NVLS 对等节点
    for (int r = 0; r < nvlsRanks; ++r) {
      // 获取父通信器中的本地 rank 索引
      int tr = comm->topParentLocalRanks[r];
      // 获取父通信器中 NVLS 设备对等节点的地址
      uintptr_t addr = (uintptr_t)(parent->channels[channelId].nvlsDevPeers + tr);
      // 设置对等节点指针（指向父通信器的 NVLS 对等节点）
      channel->peers[comm->nRanks + 1 + r] = parent->channels[channelId].nvlsPeers + tr;
      // 将设备对等节点指针复制到 GPU 内存
      NCCLCHECK(ncclCudaMemcpyAsync((uintptr_t*)(channel->devPeers + comm->nRanks + 1 + r), (uintptr_t*)&addr, 1, deviceStream));
      // 保存主机侧的设备对等节点指针
      channel->devPeersHostPtr[comm->nRanks + 1 + r] = (struct ncclDevChannelPeer*)addr;
      // 增加父通信器 NVLS 对等节点的引用计数
      ncclAtomicRefCountIncrement(&parent->channels[channelId].nvlsPeers[tr].refCount);
    }
  } else {                                              // 如果不共享资源，需要分配新的
    // 分配 NVLS 对等节点数组
    NCCLCHECK(ncclCalloc(&channel->nvlsPeers, nvlsRanks));
    // 在设备上分配 NVLS 设备对等节点数组
    NCCLCHECK(ncclCudaCallocAsync(&channel->nvlsDevPeers, nvlsRanks, deviceStream));
    // 初始化 NVLS 对等节点
    for (int r = 0; r < nvlsRanks; ++r) {
      // 获取 NVLS 设备对等节点的地址
      uintptr_t addr = (uintptr_t)(channel->nvlsDevPeers + r);
      // 设置对等节点指针
      channel->peers[comm->nRanks + 1 + r] = channel->nvlsPeers + r;
      // 将设备对等节点指针复制到 GPU 内存
      NCCLCHECK(ncclCudaMemcpyAsync((uintptr_t*)(channel->devPeers + comm->nRanks + 1 + r), (uintptr_t*)&addr, 1, deviceStream));
      // 保存主机侧的设备对等节点指针
      channel->devPeersHostPtr[comm->nRanks + 1 + r] = (struct ncclDevChannelPeer*)addr;
      // 增加对等节点的引用计数
      ncclAtomicRefCountIncrement(&channel->nvlsPeers[r].refCount);
    }
  }

  // 释放设备流
  NCCLCHECK(ncclStrongStreamRelease(ncclCudaGraphNone(), &sharedRes->deviceStream, /*concurrent=*/false));
  // 同步设备流，确保所有异步操作完成
  NCCLCHECK(ncclStrongStreamSynchronize(&sharedRes->deviceStream));

  return ncclSuccess;                                  // 返回成功状态
}

// 函数实现：初始化 CollNet 通道
// 此函数为通道添加集合网络支持
// 参数 comm: NCCL 通信器指针
// 参数 channelId: 通道 ID
// 参数 parent: 父通信器指针（用于共享资源）
// 参数 share: 是否共享父通信器的资源
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
ncclResult_t initCollnetChannel(struct ncclComm* comm, int channelId, struct ncclComm* parent, bool share) {
  // 获取指定通道的指针
  struct ncclChannel* channel = &comm->channels[channelId];
  // 获取共享资源指针
  struct ncclSharedResources* sharedRes = comm->sharedRes;
  uintptr_t addr;                                      // 声明地址变量
  cudaStream_t deviceStream;                           // 声明 CUDA 流变量

  // 检查 CollNet 对等节点是否已初始化
  if (channel->collnetPeers != NULL)
    return ncclSuccess;                                // 如果已初始化，直接返回成功

  // 如果通道尚未初始化，先初始化基础通道
  if (channel->id == -1)
    NCCLCHECK(initChannel(comm, channelId));

  // 获取设备流，用于 CUDA 操作
  NCCLCHECK(ncclStrongStreamAcquire(ncclCudaGraphNone(), &sharedRes->deviceStream, /*concurrent=*/false, &deviceStream));

  // 根据是否共享资源采用不同的初始化策略
  if (share) {                                          // 如果共享父通信器的资源
    // 直接使用父通信器的 CollNet 资源指针
    channel->collnetPeers = parent->channels[channelId].collnetPeers;
    channel->collnetDevPeers = parent->channels[channelId].collnetDevPeers;
    // 获取父通信器中 CollNet 设备对等节点的地址
    addr = (uintptr_t)parent->channels[channelId].collnetDevPeers;
    // 设置对等节点指针（指向父通信器的 CollNet 对等节点）
    channel->peers[comm->nRanks] = parent->channels[channelId].collnetPeers;
    // 将设备对等节点指针复制到 GPU 内存（位置在普通 ranks 之后）
    NCCLCHECK(ncclCudaMemcpyAsync((uintptr_t*)(channel->devPeers + comm->nRanks), (uintptr_t*)&addr, 1, deviceStream));
    // 保存主机侧的设备对等节点指针
    channel->devPeersHostPtr[comm->nRanks] = (struct ncclDevChannelPeer*)addr;
    // 增加父通信器 CollNet 对等节点的引用计数
    ncclAtomicRefCountIncrement(&parent->channels[channelId].collnetPeers->refCount);
  } else {                                              // 如果不共享资源，需要分配新的
    // 分配 CollNet 对等节点（只需要一个，代表网络节点）
    NCCLCHECK(ncclCalloc(&channel->collnetPeers, 1));
    // 在设备上分配 CollNet 设备对等节点
    NCCLCHECK(ncclCudaCallocAsync(&channel->collnetDevPeers, 1, deviceStream));
    // 获取 CollNet 设备对等节点的地址
    addr = (uintptr_t)channel->collnetDevPeers;
    // 设置对等节点指针
    channel->peers[comm->nRanks] = channel->collnetPeers;
    // 将设备对等节点指针复制到 GPU 内存
    NCCLCHECK(ncclCudaMemcpyAsync((uintptr_t*)(channel->devPeers + comm->nRanks), (uintptr_t*)&addr, 1, deviceStream));
    // 保存主机侧的设备对等节点指针
    channel->devPeersHostPtr[comm->nRanks] = (struct ncclDevChannelPeer*)addr;
    // 增加对等节点的引用计数
    ncclAtomicRefCountIncrement(&channel->collnetPeers->refCount);
  }

  // 释放设备流
  NCCLCHECK(ncclStrongStreamRelease(ncclCudaGraphNone(), &sharedRes->deviceStream, /*concurrent=*/false));
  // 同步设备流，确保所有异步操作完成
  NCCLCHECK(ncclStrongStreamSynchronize(&sharedRes->deviceStream));

  return ncclSuccess;                                  // 返回成功状态
}

// 函数实现：释放通道资源
// 此函数释放通道相关的所有资源
// 参数 channel: 要释放的通道指针
// 参数 nRanks: 普通 rank 的数量
// 参数 collnetNRanks: CollNet rank 的数量
// 参数 nvlsNRanks: NVLS rank 的数量
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
ncclResult_t freeChannel(struct ncclChannel* channel, int nRanks, int collnetNRanks, int nvlsNRanks) {
  // 计算对等节点总数
  int nPeers = nRanks + collnetNRanks + nvlsNRanks;
  /* channel peers are only valid when async init thread completes commAlloc() and
   * the channel is initialized with initChannel(); if either is not done, this channel
   * should never be free. */
  /* 通道对等节点仅在异步初始化线程完成 commAlloc() 并且通道通过 initChannel() 初始化后才有效；
   * 如果这两个条件未满足，此通道永远不应该被释放。 */
  // 检查通道是否已初始化（id 为 -1 或 peers 为 NULL 表示未初始化）
  if (channel->id == -1 || channel->peers == NULL) return ncclSuccess;

  // Free transport proxy resources
  // 释放传输层代理资源
  // Note: free all send resources first due to CollNet arrangement
  // 注意：由于 CollNet 的安排，先释放所有发送资源
  for (int r = 0; r < nPeers; r++) {
    // 获取对等节点指针
    struct ncclChannelPeer* peer = channel->peers[r];
    if (peer) {                                          // 如果对等节点存在
      // 减少引用计数，如果计数降为 0，则释放资源
      if (ncclAtomicRefCountDecrement(&peer->refCount) == 0) {
        // 遍历所有连接（发送和接收）
        for (int b=0; b<NCCL_MAX_CONNS; b++) {
          // 释放发送连接的传输层资源
          if (peer->send[b].transportComm) NCCLCHECK(peer->send[b].transportComm->free(peer->send+b));
          // 释放接收连接的传输层资源
          if (peer->recv[b].transportComm) NCCLCHECK(peer->recv[b].transportComm->free(peer->recv+b));
        }
        // 根据对等节点类型释放特定资源
        if (r == nRanks) {                              // CollNet 节点（在普通 ranks 之后）
          free(channel->collnetPeers);                 // 释放 CollNet 对等节点
          ncclCudaFree(channel->collnetDevPeers);       // 释放 CollNet 设备对等节点
        } else if (r == nPeers - 1) {                   // NVLS 节点（最后一个）
          free(channel->nvlsPeers);                    // 释放 NVLS 对等节点
          ncclCudaFree(channel->nvlsDevPeers);         // 释放 NVLS 设备对等节点
        }
      }
    }
  }

  // 释放主机侧的设备对等节点指针数组
  free(channel->devPeersHostPtr);
  return ncclSuccess;                                  // 返回成功状态
}

