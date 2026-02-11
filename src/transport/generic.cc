/*************************************************************************
 * Copyright (c) 2015-2025, NVIDIA CORPORATION. All rights reserved.
 * 版权所有 (c) 2015-2025，NVIDIA 公司。保留所有权利。
 *
 * See LICENSE.txt for license information
 * 有关许可信息，请参见 LICENSE.txt 文件
 ************************************************************************/

// 包含通信域头文件，定义 ncclComm 结构体和通信相关接口
#include "comm.h"
// 包含传输层头文件，定义传输层的接口和数据结构
#include "transport.h"
// 包含引导启动头文件，定义初始化和连接建立的接口
#include "bootstrap.h"

// 函数功能：建立 Ring（环形）拓扑的传输连接
// Ring 拓扑中每个 rank 只与左右两个邻居通信，形成环形结构
// 参数说明：
//   - comm: 通信上下文指针，包含通信域的所有信息
// 返回值：ncclSuccess 表示成功，其他值表示失败
ncclResult_t ncclTransportRingConnect(struct ncclComm* comm) {
  // 定义内部结构体：Ring 连接信息
  struct ringConnInfo {
    // 是否使用网络代理（Proxy eXtension Network）
    bool useNetPXN;
    // 是否使用 GPU Direct RDMA (GDR)
    bool useGdr;
  };
  // 声明指针：Ring 连接信息数组，用于收集所有 rank 的配置
  struct ringConnInfo* ringInfo = NULL;
  // 初始化返回值为成功
  ncclResult_t ret = ncclSuccess;
  // 检查 comm 指针是否有效，并且 rank 数量大于 1（需要通信）
  if (comm && comm->nRanks > 1) {
    // 默认启用 GPU Direct RDMA
    // GDR 允许网卡直接访问 GPU 内存，减少 CPU 拷贝开销
    comm->useGdr = true;
    // 默认不使用网络代理扩展
    comm->useNetPXN = false;

  //遍历所有的通道，
  // 每个通道可以独立进行数据传输，提高并行度
    for (int c = 0; c < comm->nChannels; c++) {
      // 获取当前通道的指针
      // comm->channels: 通道数组首地址
      // + c: 偏移到第 c 个通道
      struct ncclChannel* channel = comm->channels + c;
      // 建立 P2P 连接：连接前一个和后一个 rank
      // 参数说明：
      //   - comm: 通信上下文
      //   - c: 通道索引
      //   - 1: 发送方连接数（连接 1 个前驱）
      //   - &channel->ring.prev: 前驱 rank 的指针（接收数据来源）
      //   - 1: 接收方连接数（连接 1 个后继）
      //   - &channel->ring.next: 后继 rank 的指针（发送数据目标）
      //   - 0: 不使用偏移量
      NCCLCHECKGOTO(ncclTransportP2pConnect(comm, c, 1, &channel->ring.prev, 1, &channel->ring.next, 0), ret, fail);
    }

    // 设置 P2P 传输层，建立传输图和资源
    // NCCL_ALGO_RING: Ring 算法标识
    // 0: 传输方向偏移（0 表示双向）
    NCCLCHECKGOTO(ncclTransportP2pSetup(comm, &comm->graphs[NCCL_ALGO_RING], 0), ret, fail);
    // 检查是否启用了内存注册功能（本地注册或图注册）
    if (ncclParamLocalRegister() || ncclParamGraphRegister()) {
      // 分配 Ring 连接信息数组内存
      // 大小为 nRanks，用于存储每个 rank 的配置信息
      NCCLCHECK(ncclCalloc(&ringInfo, comm->nRanks));
      //设置本rank的ring信息
      // 保存当前 rank 的 GDR 配置
      ringInfo[comm->rank].useGdr = comm->useGdr;
      // 保存当前 rank 的 PXN 配置
      ringInfo[comm->rank].useNetPXN = comm->useNetPXN;

      //同步通信组内的ringConnInfo信息
      // 使用 AllGather 操作收集所有 rank 的配置信息
      // bootstrap: 引导程序，用于初始化连接
      // ringInfo: 收集信息的缓冲区
      // sizeof(struct ringConnInfo): 每个元素的大小
      NCCLCHECKGOTO(bootstrapAllGather(comm->bootstrap, ringInfo, sizeof(struct ringConnInfo)), ret, fail);

      // 遍历所有 rank 的配置信息
      for (int i = 0; i < comm->nRanks; ++i) {
        //不支持gdr，设置为false
        // 如果任何一个 rank 不支持 GDR，全局禁用 GDR
        if (!ringInfo[i].useGdr)
            comm->useGdr = false;
        // 如果任何一个 rank 使用 PXN，全局启用 PXN
        if (ringInfo[i].useNetPXN)
            comm->useNetPXN = true;
        // 检查配置组合：GDR 禁用且 PXN 启用
        if (comm->useGdr == false && comm->useNetPXN == true)
          // 这种组合是有效的，可以退出循环
            break;
      }
    }
    // 输出日志：Ring 连接建立完成，显示 PXN 和 GDR 的配置状态
    INFO(NCCL_INIT, "Connected all rings, use ring PXN %d GDR %d", comm->useNetPXN, comm->useGdr);
  }
// 正常退出标签
exit:
  // 释放 Ring 连接信息数组内存
  free(ringInfo);
  // 返回结果状态码
  return ret;
// 失败标签
fail:
  // 跳转到退出标签进行清理
  goto exit;
}

// 函数功能：建立 Tree（树形）拓扑的传输连接
// Tree 拓扑中 rank 组织成树形结构，根节点汇聚或分发数据
// 参数说明：
//   - comm: 通信上下文指针
// 返回值：ncclSuccess 表示成功
ncclResult_t ncclTransportTreeConnect(struct ncclComm* comm) {
  // 初始化返回值为成功
  ncclResult_t ret = ncclSuccess;
  // 检查 comm 指针是否有效，并且 rank 数量大于 1
  if (comm && comm->nRanks > 1) {
    // Connect Trees
    // 连接树形拓扑：每个节点连接多个子节点和一个父节点
    // 遍历所有通道
    for (int c = 0; c < comm->nChannels; c++) {
      // 获取当前通道的指针
      struct ncclChannel* channel = comm->channels + c;
      // 建立到子节点的连接（下行方向）
      // 参数说明：
      //   - c: 通道索引
      //   - NCCL_MAX_TREE_ARITY: 树的最大分支数（子节点数量）
      //   - channel->tree.down: 子节点 rank 数组
      //   - 1: 上行连接数（连接 1 个父节点）
      //   - &channel->tree.up: 父节点 rank 的指针
      //   - 0: 不使用偏移量
      NCCLCHECKGOTO(ncclTransportP2pConnect(comm, c, NCCL_MAX_TREE_ARITY, channel->tree.down, 1, &channel->tree.up, 0), ret, fail);
      // 建立到父节点的连接（上行方向）
      // 参数说明：
      //   - 1: 下行连接数（连接 1 个父节点）
      //   - &channel->tree.up: 父节点 rank
      //   - NCCL_MAX_TREE_ARITY: 树的最大分支数（子节点数量）
      //   - channel->tree.down: 子节点 rank 数组
      NCCLCHECKGOTO(ncclTransportP2pConnect(comm, c, 1, &channel->tree.up, NCCL_MAX_TREE_ARITY, channel->tree.down, 0), ret, fail);
    }
    // 设置 P2P 传输层，建立 Tree 拓扑的传输图
    // NCCL_ALGO_TREE: Tree 算法标识
    NCCLCHECKGOTO(ncclTransportP2pSetup(comm, &comm->graphs[NCCL_ALGO_TREE], 0), ret, fail);
    // 输出日志：所有树形连接建立完成
    INFO(NCCL_INIT, "Connected all trees");
  }
// 正常退出标签
exit:
  // 返回结果状态码
  return ret;
// 失败标签
fail:
  // 跳转到退出标签
  goto exit;
}

// 函数功能：建立 PAT（Pairwise Allreduce Tree）二项式树拓扑的传输连接
// PAT 使用二项式树结构，适用于 AllReduce 和 AllGather 等操作
// 参数说明：
//   - comm: 通信 context 指针
// 返回值：ncclSuccess 表示成功
ncclResult_t ncclTransportPatConnect(struct ncclComm* comm) {
  // 初始化返回值为成功
  ncclResult_t ret = ncclSuccess;
  // 检查 comm 指针是否有效，并且 rank 数量大于 1
  if (comm && comm->nRanks > 1) {
    // 建立二项式树的连接
    // mask 表示步长，从 1 开始，每次左移一位（1, 2, 4, 8, ...）
    // 这样可以构建 log2(nRanks) 层的二项式树
    for (int mask=1; mask<comm->nRanks; mask<<=1) {
      // 计算前驱 peer：当前 rank + mask，对 nRanks 取模
      // 用于 ReduceScatter 阶段的数据汇聚
      int prevPeer = (comm->rank + mask) % comm->nRanks;
      // 计算后继 peer：当前 rank - mask，对 nRanks 取模
      // 用于 AllGather 阶段的数据分发
      int nextPeer = (comm->rank + comm->nRanks - mask) % comm->nRanks;
      // 遍历所有通道
      for (int c = 0; c < comm->nChannels; c++) {
        // 建立 ReduceScatter 阶段的连接
        // 从 prevPeer 接收数据，向 nextPeer 发送数据
        NCCLCHECKGOTO(ncclTransportP2pConnect(comm, c, 1, &prevPeer, 1, &nextPeer, 0), ret, fail); // ReduceScatter
      }
      // 设置传输层，建立 ReduceScatter 阶段的传输图
      NCCLCHECKGOTO(ncclTransportP2pSetup(comm, &comm->graphs[NCCL_ALGO_TREE], 0), ret, fail);
      // 遍历所有通道
      for (int c = 0; c < comm->nChannels; c++) {
        // 建立 AllGather 阶段的连接
        // 方向与 ReduceScatter 相反
        NCCLCHECKGOTO(ncclTransportP2pConnect(comm, c, 1, &nextPeer, 1, &prevPeer, 0), ret, fail); // AllGather
      }
      // 设置传输层，建立 AllGather 阶段的传输图
      NCCLCHECKGOTO(ncclTransportP2pSetup(comm, &comm->graphs[NCCL_ALGO_TREE], 0), ret, fail);
    }
    // 输出日志：二项式树连接建立完成
    INFO(NCCL_INIT, "Connected binomial trees");
  }
// 正常退出标签
exit:
  // 返回结果状态码
  return ret;
// 失败标签
fail:
  // 跳转到退出标签
  goto exit;
}
