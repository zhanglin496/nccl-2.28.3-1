/*************************************************************************
 * Copyright (c) 2015-2019, NVIDIA CORPORATION. All rights reserved.
 * 版权所有 (c) 2015-2019, NVIDIA CORPORATION. 保留所有权利。
 *
 * See LICENSE.txt for license information
 * 请参阅 LICENSE.txt 获取许可证信息
 ************************************************************************/

// 防止头文件重复包含的保护宏开始
// 如果未定义 NCCL_CHANNEL_H_ 宏，则定义它
#ifndef NCCL_CHANNEL_H_
#define NCCL_CHANNEL_H_

// 引入通信器头文件，包含 ncclComm 结构体和相关操作的声明
#include "comm.h"
// 引入工具函数头文件，包含各种辅助工具函数和宏定义
#include "utils.h"

// 引入 C++ 标准库算法头文件
// 提供常用的算法函数，如排序、查找等
#include <algorithm>

// 函数声明：初始化通道
// 此函数初始化指定的通信通道，分配必要的资源并建立数据结构
// 参数 comm: NCCL 通信器指针
// 参数 channelid: 要初始化的通道 ID
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
ncclResult_t initChannel(struct ncclComm* comm, int channelid);

// 函数声明：初始化 NVLS 通道
// 此函数为通道添加 NVLS（NVLink fabric）支持
// 参数 comm: NCCL 通信器指针
// 参数 channelId: 通道 ID
// 参数 parent: 父通信器指针（用于共享资源）
// 参数 share: 是否共享父通信器的资源标志
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
ncclResult_t initNvlsChannel(struct ncclComm* comm, int channelId, struct ncclComm* parent, bool share);

// 函数声明：初始化 CollNet 通道
// 此函数为通道添加集合网络支持
// 参数 comm: NCCL 通信器指针
// 参数 channelId: 通道 ID
// 参数 parent: 父通信器指针（用于共享资源）
// 参数 share: 是否共享父通信器的资源标志
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
ncclResult_t initCollnetChannel(struct ncclComm* comm, int channelId, struct ncclComm* parent, bool share);

// 函数声明：释放通道资源
// 此函数释放通道相关的所有资源
// 参数 channel: 要释放的通道指针
// 参数 nRanks: 普通 rank 的数量
// 参数 collnetNRanks: CollNet rank 的数量
// 参数 nvlsNRanks: NVLS rank 的数量
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
ncclResult_t freeChannel(struct ncclChannel* channel, int nRanks, int collnetNRanks, int nvlsNRanks);

// 内联函数：获取 P2P 通道的基准 ID（用于指定轮次）
// 此函数根据 P2P 轮次计算对应的通道基准 ID，使用位反转来优化通道分布
// 参数 comm: NCCL 通信器指针
// 参数 p2pRound: P2P 通信的轮次编号
// 返回值：uint8_t 类型，返回计算得到的通道基准 ID
inline uint8_t ncclP2pChannelBaseForRound(struct ncclComm* comm, int p2pRound) {
  int base;                                            // 声明基准 ID 变量
  // 检查是否为多节点环境
  if (comm->nNodes > 1) {                             // 如果节点数大于 1（多节点环境）
    // 计算节点偏移量：当前轮次除以每个节点的最大本地 rank 数
    // 这确定当前轮次属于哪个节点的批次
    int nodeDelta = p2pRound/comm->maxLocalRanks;
    // 计算本地偏移量：当前轮次对每个节点的最大本地 rank 数取余
    // 这确定当前轮次在节点内的本地位置
    int localDelta = p2pRound%comm->maxLocalRanks;
    // 计算基准 ID：节点偏移乘以每个批次的通道数（向上取整）
    // divUp 向上取整确保分配足够的通道空间
    base = nodeDelta*divUp(comm->maxLocalRanks, NCCL_MAX_DEV_WORK_P2P_PER_BATCH);
    // 加上本地偏移对应的通道数（每 NCCL_MAX_DEV_WORK_P2P_PER_BATCH 个 P2P 工作占一个通道）
    base += localDelta/NCCL_MAX_DEV_WORK_P2P_PER_BATCH;
  } else {                                              // 单节点环境
    // 单节点情况下，基准 ID 就是轮次编号
    base = p2pRound;
  }
  // 对基准 ID 进行位反转，使用 log2 向上取整计算位数
  // 位反转有助于优化通道在 GPU 内存中的访问模式
  return reverseBits(base, log2Up(comm->p2pnChannels));
}

// 头文件保护结束宏
// 与开头的 #ifndef NCCL_CHANNEL_H_ 配对，防止头文件重复包含
#endif

