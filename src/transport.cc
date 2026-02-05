/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

/*
 * transport.cc - NCCL 传输层管理模块
 *
 * 本文件负责 NCCL 传输层的选择、建立和管理，是连接拓扑和具体传输实现的桥梁
 *
 * 主要功能：
 * 1. 传输层选择 - 根据拓扑和 peer 信息选择最优传输方式（P2P/SHM/NET/CollNet）
 * 2. P2P 连接建立 - 建立点对点连接通道
 * 3. CollNet 连接 - 集合网络专用传输
 * 4. 连接信息交换 - 通过 Bootstrap 交换连接句柄
 * 5. 进度报告和同步 - 连接进度跟踪和全局同步
 */

#include "comm.h"       // 通信域结构定义
#include "info.h"       // 信息查询接口
#include "bootstrap.h"  // Bootstrap 网络初始化
#define ENABLE_TIMER 0  // 计时器开关
#include "timer.h"      // 性能计时
#include "transport.h"  // 传输层接口定义

// ============================================================================
// 传输层优先级表
// ============================================================================
// 按顺序优先级选择传输方式
// 数组索引越小，优先级越高
struct ncclTransport* ncclTransports[NTRANSPORTS+1] = {
  &p2pTransport,        // 1. P2P 传输 - 最优先，GPU 间直接访问（NVLink/PCIe）
  &shmTransport,        // 2. SHM 传输 - 共享内存，同节点不同进程
  &netTransport,        // 3. NET 传输 - 网络传输，跨节点（IB/RoCE/Socket）
  &collNetTransport,    // 4. CollNet 传输 - 集合网络加速器
  &profilerTransport    // 5. Profiler 传输 - 性能分析（不用于实际数据传输）
};
// 注意：NTRANSPORTS 是编译时常量，值为 4（不包括 profilerTransport）
// 数组大小为 NTRANSPORTS+1 是为了包含 profilerTransport

// ============================================================================
// selectTransport - 选择并建立传输层连接
// ============================================================================
// 模板函数：type=0 表示接收连接，type=1 表示发送连接
//
// 参数：
//   comm       - 通信域
//   graph      - 拓扑图
//   connect    - 连接信息数组（用于存储连接句柄）
//   channelId  - 通道 ID
//   peer       - 对端 rank 号
//   connIndex  - 连接索引
//   transportType - 输出参数，返回选择的传输类型
//
// 返回：ncclSuccess 成功，否则返回错误码
//
// 选择逻辑：
// 1. 遍历所有传输类型（P2P → SHM → NET → CollNet）
// 2. 对每种传输调用 canConnect() 检查是否支持
// 3. 第一个支持的传输被选中，调用其 setup() 函数
// 4. 设置 connector 的 transportComm 指针
template <int type>
static ncclResult_t selectTransport(struct ncclComm* comm, struct ncclTopoGraph* graph, struct ncclConnect* connect, int channelId, int peer, int connIndex, int* transportType) {
  // 获取本端 peer 信息
  struct ncclPeerInfo* myInfo = comm->peerInfo+comm->rank;
  // 获取对端 peer 信息
  struct ncclPeerInfo* peerInfo = comm->peerInfo+peer;

  // 根据 type 选择 connector：
  // type=1: 发送连接 (send connector)
  // type=0: 接收连接 (recv connector)
  struct ncclConnector* connector = (type == 1) ? comm->channels[channelId].peers[peer]->send + connIndex :
                                                  comm->channels[channelId].peers[peer]->recv + connIndex;

  // 遍历所有传输类型，按优先级顺序
  for (int t=0; t<NTRANSPORTS; t++) {
    struct ncclTransport *transport = ncclTransports[t];
    // 根据 type 选择发送或接收的传输通信接口
    struct ncclTransportComm* transportComm = type == 1 ? &transport->send : &transport->recv;
    int ret = 0;

    // 调用该传输的 canConnect 方法，判断是否支持本连接
    // canConnect 会检查：
    // - P2P: GPU 是否支持直接访问，是否在同一节点
    // - SHM: 是否在同节点但不同进程
    // - NET: 是否跨节点
    NCCLCHECK(transport->canConnect(&ret, comm, graph, myInfo, peerInfo));

    // 找到第一个支持的传输，选中并退出
    if (ret) {
      // 设置 connector 的传输通信接口
      connector->transportComm = transportComm;

      // 调用该传输的 setup 方法
      // setup 方法会：
      // 1. 分配传输资源（如创建 socket、分配共享内存等）
      // 2. 填充 connect 结构体（包含连接句柄）
      // 3. 这些信息后续会通过 Bootstrap 发送给对端
      NCCLCHECK(transportComm->setup(comm, graph, myInfo, peerInfo, connect, connector, channelId, connIndex));

      // 记录传输类型（用于调试）
      if (transportType)
        *transportType = t;
      return ncclSuccess;
    }
  }

  // 所有传输类型都不支持，报错
  WARN("No transport found for rank %d[%lx] -> rank %d[%lx]", myInfo->rank, myInfo->busId, peerInfo->rank, peerInfo->busId);
  return ncclSystemError;
}

// ============================================================================
// ncclTransportP2pConnect - 标记需要建立的 P2P 连接
// ============================================================================
// 此函数只是标记哪些 peer 需要建立连接，实际连接在 ncclTransportP2pSetup 中完成
//
// 参数：
//   comm       - 通信域
//   channelId  - 通道 ID
//   nrecv      - 需要接收的 peer 数量
//   peerRecv   - 接收 peer 列表
//   nsend      - 需要发送的 peer 数量
//   peerSend   - 发送 peer 列表
//   connIndex  - 连接索引
//
// 返回：ncclSuccess
//
// 工作原理：
// 使用位掩码（bitmask）标记每个 peer 在哪些通道需要连接
// connectRecv[peer] 的每一位代表一个通道是否需要接收连接
// connectSend[peer] 的每一位代表一个通道是否需要发送连接
ncclResult_t ncclTransportP2pConnect(struct ncclComm* comm, int channelId, int nrecv, int* peerRecv, int nsend, int* peerSend, int connIndex) {
  TRACE(NCCL_INIT, "nsend %d nrecv %d", nsend, nrecv);
  struct ncclChannel* channel = &comm->channels[channelId];

  // 构造通道位掩码：第 channelId 位为 1
  // 例如 channelId=3，则 mask = 0b1000 = 8
  uint64_t mask = 1UL << channel->id;

  // 遍历所有接收 peer
  for (int i=0; i<nrecv; i++) {
    int peer = peerRecv[i];
    // 跳过无效 peer：-1（无 peer）、越界、自身、已连接
    if (peer == -1 || peer >= comm->nRanks || peer == comm->rank || channel->peers[peer]->recv[connIndex].connected)
        continue;
    // 在 connectRecv[peer] 中标记此通道需要接收连接
    comm->connectRecv[peer] |= mask;
  }

  // 遍历所有发送 peer
  for (int i=0; i<nsend; i++) {
    int peer = peerSend[i];
    // 跳过无效 peer
    if (peer == -1 || peer >= comm->nRanks || peer == comm->rank || channel->peers[peer]->send[connIndex].connected)
        continue;
    // 在 connectSend[peer] 中标记此通道需要发送连接
    comm->connectSend[peer] |= mask;
  }

  return ncclSuccess;
}

// ============================================================================
// dumpData - 调试函数：打印连接信息
// ============================================================================
// 以十六进制格式转储 ncclConnect 结构体内容，用于调试
void dumpData(struct ncclConnect* data, int ndata) {
  for (int n=0; n<ndata; n++) {
    printf("[%d] ", n);
    uint8_t* d = (uint8_t*)data;
    // 打印每个字节的十六进制值
    for (int i=0; i<sizeof(struct ncclConnect); i++) printf("%02x", d[i]);
    printf("\n");
  }
}

// ============================================================================
// 环境变量参数定义
// ============================================================================
NCCL_PARAM(ConnectRoundMaxPeers, "CONNECT_ROUND_MAX_PEERS", 128);  // 每轮连接的最大 peer 数
NCCL_PARAM(ReportConnectProgress, "REPORT_CONNECT_PROGRESS", 0);   // 是否报告连接进度

#include <sys/time.h>  // 时间相关函数

// ============================================================================
// ncclTransportCheckP2pType - 检查 P2P 类型
// ============================================================================
// 检测系统中所有 GPU 之间的 P2P 连接类型
//
// 参数：
//   comm          - 通信域
//   isAllDirectP2p - 输出：是否所有 GPU 对都支持直接 P2P
//   directMode    - 输出：是否存在同进程内的直接 P2P
//
// 返回：ncclSuccess
//
// P2P 类型说明：
// 1. Direct P2P: GPU 之间可以直接访问（如 NVLink、PCIe P2P）
// 2. Intermediate P2P: 需要通过中间 GPU 转发
// 3. SHM: 同节点不同进程，使用共享内存
ncclResult_t ncclTransportCheckP2pType(struct ncclComm* comm, bool* isAllDirectP2p, bool* directMode) {
  bool supportFlag = true;  // 是否所有 GPU 对都支持直接 P2P
  bool directFlag = false;  // 是否存在同进程内的直接 P2P

  // 特殊情况：当前节点只有 1 个 GPU
  // 不需要 P2P，直接标记为不支持
  if (comm->localRanks == 1) {
    supportFlag = false;
  } else {
  // 遍历本地所有 GPU 对，检查 P2P 支持
    for (int i = 0; i < comm->localRanks; ++i) {
      for (int j = i + 1; j < comm->localRanks; ++j) {
        // 获取全局 rank 号
        int ipeer = comm->localRankToRank[i];
        int jpeer = comm->localRankToRank[j];

        struct ncclPeerInfo* ipeerInfo = &comm->peerInfo[ipeer];
        struct ncclPeerInfo* jpeerInfo = &comm->peerInfo[jpeer];

        int canConnect = 0;           // 是否能连接
        int intermediateRank = -1;    // 中间 GPU rank（如果需要）

        // 检查两个 GPU 之间的 P2P 连接能力
        // ncclTopoCheckP2p 会：
        // 1. 检查 GPU 是否在同一节点
        // 2. 检查是否支持 P2P（CUDA p2p access）
        // 3. 如果不支持直接 P2P，检查是否可以通过中间 GPU 转发
        NCCLCHECK(ncclTopoCheckP2p(comm, comm->topo, ipeerInfo->rank, jpeerInfo->rank, &canConnect, NULL, &intermediateRank));

        // 如果不能连接，或者需要经过中间 GPU，则不是完全直接 P2P
        if (!canConnect || intermediateRank != -1) {
          supportFlag = false;
        }

        // 检查是否是同进程内的 P2P（直接模式）
        // hostHash 相同 = 同一主机
        // pidHash 相同 = 同一进程
        // 这种情况下可以使用 CUDA IPC 或直接指针访问
        if (ipeerInfo->hostHash == jpeerInfo->hostHash && ipeerInfo->pidHash == jpeerInfo->pidHash)
            directFlag = true;

        // 如果已经确定为不支持直接 P2P 且存在直接模式，可以提前退出
        if (!supportFlag && directFlag)
            break;
      }
    }
  }

  *isAllDirectP2p = supportFlag;
  *directMode = directFlag;

  // 只有 rank 0 打印信息，避免重复输出
  if (comm->rank == 0)
    INFO(NCCL_INIT, "Check P2P Type isAllDirectP2p %d directMode %d", supportFlag, directFlag);

  return ncclSuccess;
}

// ============================================================================
// ncclTransportP2pSetup - 建立所有 P2P 连接
// ============================================================================
// 这是传输层连接的核心函数，负责：
// 1. 为每个通道选择合适的传输方式
// 2. 准备连接信息（句柄、地址等）
// 3. 通过 Bootstrap 交换连接信息
// 4. 完成连接建立
// 5. 同步所有 rank
//
// 参数：
//   comm       - 通信域
//   graph      - 拓扑图
//   connIndex  - 连接索引
//
// 返回：ncclSuccess 或错误码
//
// 连接建立流程：
// ┌─────────────────────────────────────────────────────────────┐
// │ 1. 选择传输层 → 2. 准备连接信息 → 3. 交换连接信息            │
// │                 ↓                   ↓                       │
// │          调用各传输的 setup()   通过 Bootstrap Send/Recv  │
// │                 ↓                   ↓                       │
// │ 4. 建立连接 → 5. 拷贝连接信息到设备 → 6. 全局同步          │
// │    调用 connect()   cudaMemcpyAsync      Bootstrap Barrier  │
// └─────────────────────────────────────────────────────────────┘
ncclResult_t ncclTransportP2pSetup(struct ncclComm* comm, struct ncclTopoGraph* graph, int connIndex) {
  // Stream used during transport setup; need for P2P pre-connect + CUDA Graph
  // 传输层设置期间使用的流；用于 P2P 预连接和 CUDA Graph 支持
  ncclResult_t ret = ncclSuccess;

  // data: 存储中间的 send/recvData 结构，用于连接信息交换
  // 这是一个二维数组：data[peer_index][channel_index]
  struct ncclConnect** data;
  struct ncclConnect** recvData = NULL;  // 指向 data 中接收连接的部分
  struct ncclConnect** sendData = NULL;  // 指向 data 中发送连接的部分
  int done = 0;  // 已处理完成的 peer 数量
  int maxPeers = ncclParamConnectRoundMaxPeers();  // 每轮处理的最大 peer 数

  struct timeval timeStart, timeLast;  // 用于计时和进度报告
  gettimeofday(&timeStart, NULL);
  timeLast = timeStart; // struct copy
  bool timeReported = false;
  cudaStream_t hostStream, deviceStream;  // 主机流和设备流

  // 分配内存
  NCCLCHECK(ncclCalloc(&data, maxPeers));
  NCCLCHECKGOTO(ncclCalloc(&recvData, maxPeers), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&sendData, maxPeers), ret, fail);

  // 获取主机流和设备流（强流，支持 CUDA Graph）
  NCCLCHECKGOTO(ncclStrongStreamAcquire(ncclCudaGraphNone(), &comm->sharedRes->hostStream, /*concurrent=*/false, &hostStream), ret, fail);
  NCCLCHECKGOTO(ncclStrongStreamAcquire(ncclCudaGraphNone(), &comm->sharedRes->deviceStream, /*concurrent=*/false, &deviceStream), ret, fail);

  // ========================================================================
  // 第一阶段：选择传输层并准备连接信息
  // ========================================================================
  // First time initialization
  // 遍历所有可能的 peer（距离当前 rank 的距离）
  for (int i=1; i<comm->nRanks; i++) {
    // 构造 Bootstrap 标签：
    // - 高 8 位：peer 距离 (i)
    // - 中 8 位：图 ID + 1
    // 这样不同的 peer 和图使用不同的标签，避免混淆
    int bootstrapTag = (i<<8) + (graph ? graph->id+1 : 0);

    // 计算 peer 号：环形拓扑
    // recvPeer: 环形前驱（接收来源）
    // sendPeer: 环形后继（发送目标）
    int recvPeer = (comm->rank - i + comm->nRanks) % comm->nRanks;
    int sendPeer = (comm->rank + i) % comm->nRanks;

    // 获取需要连接的通道位掩码
    uint64_t recvMask = comm->connectRecv[recvPeer];
    uint64_t sendMask = comm->connectSend[sendPeer];

    // Data[i] 包含与特定发送/接收 peer 的所有发送和接收连接的 ncclConnect 信息
    // 这些数据基于与这些 peer 连接的 sendChannels 和 recvChannels 数量打包在数组中
    // 前 N 个条目包含 recvData（接收连接信息）
    // 后 M 个条目包含 sendData（发送连接信息）
    // 不保证每个 data 条目有相同的总连接数或发送/接收连接数
    int p = i-(done+1);
    if (recvMask || sendMask) {
      if (data[p] == NULL)
        // 分配 2*MAXCHANNELS 个连接信息
        // 前 MAXCHANNELS 用于接收，后 MAXCHANNELS 用于发送
        NCCLCHECKGOTO(ncclCalloc(data + p, 2 * MAXCHANNELS), ret, fail);
      else
        // 清零现有数据
        memset(data[p], 0, 2 * MAXCHANNELS * sizeof(struct ncclConnect));
    }

    // recvData 指向 data 数组的前半部分（接收连接信息）
    recvData[p] = data[p];
    int sendChannels = 0, recvChannels = 0;
    int type;

    TIME_START(0);
    // ========================================================================
    // 为所有接收连接选择传输层
    // ========================================================================
    // 选择一个传输层实现
    for (int c=0; c<MAXCHANNELS; c++) {
      // 使用 long 类型的 bit 表示通道 id，所以限制了 64 个通道
      // 为每个通道选择一个传输层实现
      if (recvMask & (1UL<<c)) {
        // type = 0 表示接收
        // 调用 setup 函数
        // selectTransport 会：
        // 1. 遍历传输类型（P2P → SHM → NET）
        // 2. 调用 canConnect 检查是否支持
        // 3. 调用选中传输的 setup 方法
        // 4. setup 会填充 recvData[p][recvChannels] 结构
        NCCLCHECKGOTO(selectTransport<0>(comm, graph, recvData[p]+recvChannels++, c, recvPeer, connIndex, &type), ret, fail);
      }
    }
    TIME_STOP(0);

    TIME_START(1);
    // sendData 紧跟在 recvData 之后
    sendData[p] = recvData[p]+recvChannels;
    // ========================================================================
    // 为所有发送连接选择传输层
    // ========================================================================
    for (int c=0; c<MAXCHANNELS; c++) {
      if (sendMask & (1UL<<c)) {
        // type = 1 表示发送
        // 调用 setup 函数
        NCCLCHECKGOTO(selectTransport<1>(comm, graph, sendData[p]+sendChannels++, c, sendPeer, connIndex, &type), ret, fail);
      }
    }
    TIME_STOP(1);

    TIME_START(2);
    // ========================================================================
    // 通过 Bootstrap 交换连接信息
    // ========================================================================
    if (sendPeer == recvPeer) {
      // 发送和接收是同一个 peer 的情况
      if (recvChannels+sendChannels) {
        // 发送连接信息给 peer
        NCCLCHECKGOTO(bootstrapSend(comm->bootstrap, recvPeer, bootstrapTag, data[p], sizeof(struct ncclConnect)*(recvChannels+sendChannels)), ret, fail);
        // 从 peer 接收连接信息
        NCCLCHECKGOTO(bootstrapRecv(comm->bootstrap, recvPeer, bootstrapTag, data[p], sizeof(struct ncclConnect)*(recvChannels+sendChannels)), ret, fail);
        // 调整指针位置
        sendData[p] = data[p];
        recvData[p] = data[p]+sendChannels;
      }
    } else {
      // 发送和接收是不同 peer 的情况
      // 分别发送和接收
      if (recvChannels)
        NCCLCHECKGOTO(bootstrapSend(comm->bootstrap, recvPeer, bootstrapTag, recvData[p], sizeof(struct ncclConnect)*recvChannels), ret, fail);
      if (sendChannels)
        NCCLCHECKGOTO(bootstrapSend(comm->bootstrap, sendPeer, bootstrapTag, sendData[p], sizeof(struct ncclConnect)*sendChannels), ret, fail);
      if (sendChannels)
        NCCLCHECKGOTO(bootstrapRecv(comm->bootstrap, sendPeer, bootstrapTag, sendData[p], sizeof(struct ncclConnect)*sendChannels), ret, fail);
      if (recvChannels)
        NCCLCHECKGOTO(bootstrapRecv(comm->bootstrap, recvPeer, bootstrapTag, recvData[p], sizeof(struct ncclConnect)*recvChannels), ret, fail);
    }
    TIME_STOP(2);

    // ========================================================================
    // 批量建立连接
    // ========================================================================
    // 当积累了 maxPeers 个 peer 或处理完所有 peer 时，开始建立连接
    if (i-done == maxPeers || i == comm->nRanks-1) {
      // 循环直到所有通道与所有 rank 的连接都完成
      bool allChannelsConnected;
      allChannelsConnected = false;
      while (!allChannelsConnected) {
        allChannelsConnected = true;
        // 遍历当前批次的 peer
        for (int j=done+1; j<=i; j++) {
          int recvPeer = (comm->rank - j + comm->nRanks) % comm->nRanks;
          int sendPeer = (comm->rank + j) % comm->nRanks;
          uint64_t recvMask = comm->connectRecv[recvPeer];
          uint64_t sendMask = comm->connectSend[sendPeer];

          int p = j-(done+1);
          int sendDataOffset = 0;
          int recvDataOffset = 0;

          // 遍历所有通道，建立连接
          for (int c=0; c<MAXCHANNELS; c++) {
            // ================================================================
            // 建立发送连接
            // ================================================================
            TIME_START(3);
            if (sendMask & (1UL<<c)) {
              struct ncclConnector* conn = comm->channels[c].peers[sendPeer]->send + connIndex;
              // 这个连接器还未完成连接
              if (conn->connected == 0) {
                // 调用传输层的 connect 方法
                // connect 方法会：
                // 1. 使用对端传来的连接信息（sendData[p]）
                // 2. 建立实际连接（socket 连接、共享内存映射等）
                // 3. 返回 ncclSuccess（连接完成）或 ncclInProgress（连接中）
                NCCLCHECKGOTO(conn->transportComm->connect(comm, sendData[p] + sendDataOffset, 1, comm->rank, conn), ret, fail);
                if (ret == ncclSuccess) {
                  // 标记为已连接
                  conn->connected = 1;
                  /* comm->channels[c].devPeers[sendPeer]->send[connIndex] 是设备内存访问 */
                  // 将连接信息拷贝到设备端
                  // 设备内核需要这些信息来访问对端的资源
                  CUDACHECKGOTO(cudaMemcpyAsync(&comm->channels[c].devPeersHostPtr[sendPeer]->send[connIndex], &conn->conn, sizeof(struct ncclConnInfo), cudaMemcpyHostToDevice, hostStream), ret, fail);
                } else if (ret == ncclInProgress) {
                  // 连接还在进行中，需要继续等待
                  allChannelsConnected = false;
                }
              }
              sendDataOffset++;
            }
            TIME_STOP(3);

            // ================================================================
            // 建立接收连接
            // ================================================================
            // 从接收通道开始
            TIME_START(4);
            if (recvMask & (1UL<<c)) {
              struct ncclConnector* conn = comm->channels[c].peers[recvPeer]->recv + connIndex;
              // 这个连接器还未完成连接
              if (conn->connected == 0) {
                NCCLCHECKGOTO(conn->transportComm->connect(comm, recvData[p] + recvDataOffset, 1, comm->rank, conn), ret, fail);
                if (ret == ncclSuccess) {
                  conn->connected = 1;
                  /* comm->channels[c].devPeers[recvPeer]->recv[connIndex] 是设备内存访问 */
                  CUDACHECKGOTO(cudaMemcpyAsync(&comm->channels[c].devPeersHostPtr[recvPeer]->recv[connIndex], &conn->conn, sizeof(struct ncclConnInfo), cudaMemcpyHostToDevice, hostStream), ret, fail);
                } else if (ret == ncclInProgress) {
                  allChannelsConnected = false;
                }
              }
              recvDataOffset++;
            }
            TIME_STOP(4);
          }
        }

        // ================================================================
        // 报告连接进度
        // ================================================================
        if (ncclParamReportConnectProgress() && comm->rank == 0 && done > 0) {
          struct timeval now;
          gettimeofday(&now, NULL);
          // 每秒报告一次
          if (((now.tv_sec - timeLast.tv_sec) * 1.0 + (now.tv_usec - timeLast.tv_usec) * 1e-6) > 1) {
            float elapsed = (now.tv_sec - timeStart.tv_sec) * 1.0 + (now.tv_usec - timeStart.tv_usec) * 1e-6;
            float remaining = elapsed * (comm->nRanks - done) / done;
            printf("%sP2p connect: %g%% Elapsed %d:%02d Remaining %d:%02d                                       ",
              timeReported ? "\r" : "", done * 100.0 / comm->nRanks, ((int)elapsed) / 60, ((int)elapsed) % 60, ((int)remaining) / 60, ((int)remaining) % 60);
            fflush(stdout);
            timeReported = true;
            timeLast = now; // struct copy;
          }
        }
      }
      done = i;  // 标记这批 peer 已处理完成
    }
  }

  // ========================================================================
  // 打印最终耗时
  // ========================================================================
  {
    struct timeval now;
    gettimeofday(&now, NULL);
    float elapsed = (now.tv_sec - timeStart.tv_sec)*1.0 + (now.tv_usec-timeStart.tv_usec)*1e-6;
    if (elapsed > 1.0)
        INFO(NCCL_PROFILE, "timings: rank %d nranks %d P2p connect done in %.2f", comm->rank, comm->nRanks, elapsed);
    if (timeReported) {
      printf("\rP2p connect done in %d:%02d                                                                       \n",
             ((int)elapsed)/60, ((int)elapsed)%60);
      fflush(stdout);
    }
  }

  /* We need to sync ranks here since some ranks might run too fast after connection setup
   * and start to destroy the connection after returning from this function; however, the
   * others might still be trying to connect and import the buffer. No sync can lead to invalid
   * shmem/cuda buffer. In addition, we also clear all connect masks and free each connectInfo array */
  // ========================================================================
  // 全局同步
  // ========================================================================
  // 这里需要同步所有 rank，原因：
  // 1. 某些 rank 可能连接建立很快，完成后立即销毁资源
  // 2. 其他 rank 可能还在尝试连接和导入缓冲区
  // 3. 不同步会导致 shmem/cuda 缓冲区失效
  // 4. 同时清空所有连接掩码，释放连接信息数组
  for (int i = 1; i < comm->nRanks; i++) {
    // 构造同步标签（与连接标签不同）
    int bootstrapTag = (i << 8) + (1 << 7) + (graph ? graph->id + 1 : 0);
    int recvPeer = (comm->rank - i + comm->nRanks) % comm->nRanks;
    int sendPeer = (comm->rank + i) % comm->nRanks;

    if (recvPeer != sendPeer) {
      // 发送和接收是不同 peer
      // 发送空消息表示完成
      if (comm->connectSend[sendPeer] != 0UL)
        NCCLCHECKGOTO(bootstrapSend(comm->bootstrap, sendPeer, bootstrapTag, NULL, 0), ret, fail);
      if (comm->connectRecv[recvPeer] != 0UL)
        NCCLCHECKGOTO(bootstrapSend(comm->bootstrap, recvPeer, bootstrapTag, NULL, 0), ret, fail);
      // 接收确认消息
      if (comm->connectSend[sendPeer] != 0UL)
        NCCLCHECKGOTO(bootstrapRecv(comm->bootstrap, sendPeer, bootstrapTag, NULL, 0), ret, fail);
      if (comm->connectRecv[recvPeer] != 0UL)
        NCCLCHECKGOTO(bootstrapRecv(comm->bootstrap, recvPeer, bootstrapTag, NULL, 0), ret, fail);
    } else {
      // 发送和接收是同一个 peer
      if (comm->connectSend[sendPeer] != 0UL || comm->connectRecv[recvPeer] != 0UL) {
        NCCLCHECKGOTO(bootstrapSend(comm->bootstrap, sendPeer, bootstrapTag, NULL, 0), ret, fail);
        NCCLCHECKGOTO(bootstrapRecv(comm->bootstrap, sendPeer, bootstrapTag, NULL, 0), ret, fail);
      }
    }
    // 清空连接掩码
    comm->connectRecv[recvPeer] = comm->connectSend[sendPeer] = 0UL;
  }

  TIME_PRINT("P2P Setup/Connect");
exit:
  // 释放资源
  for(int i=0; i<maxPeers; ++i){
    if(data[i])
        free(data[i]);
  }
  free(data);
  if (sendData)
    free(sendData);
  if (recvData)
    free(recvData);

  // 等待设备流完成主机流的操作
  NCCLCHECK(ncclStreamWaitStream(deviceStream, hostStream, comm->sharedRes->scratchEvent));
  // 释放流
  NCCLCHECK(ncclStrongStreamRelease(ncclCudaGraphNone(), &comm->sharedRes->hostStream, /*concurrent=*/false));
  NCCLCHECK(ncclStrongStreamRelease(ncclCudaGraphNone(), &comm->sharedRes->deviceStream, /*concurrent=*/false));
  return ret;
fail:
  goto exit;
}

// ============================================================================
// CollNet 传输相关函数
// ============================================================================
extern struct ncclTransport collNetTransport;

// ============================================================================
// ncclTransportCollNetSetup - 设置 CollNet 连接
// ============================================================================
// CollNet（集合网络）是使用网络硬件加速器来执行集合操作的传输方式
// 所有 rank 必须参与此调用
//
// 如果 CollNet 设置失败，会回退到 P2P 网络，因此不使用 NCCLCHECK
//
// 参数：
//   comm                  - 通信域
//   collNetGraph          - CollNet 拓扑图
//   channel               - 通道
//   masterRank            - 主节点 rank
//   masterPeer            - 主节点 peer
//   collNetGraphChannelId - CollNet 图通道 ID
//   type                  - 连接类型（collNetSend 或 collNetRecv）
//   connect               - 连接信息（用于存储/传递连接句柄）
//
// 返回：true 表示设置成功，false 表示失败
bool ncclTransportCollNetSetup(struct ncclComm* comm, struct ncclTopoGraph* collNetGraph, struct ncclChannel* channel, int masterRank, int masterPeer, int collNetGraphChannelId, int type, ncclConnect* connect) {
  ncclResult_t ret = ncclSuccess;
  int rank = comm->rank;
  int nranks = comm->nRanks;
  int nMasters = comm->nNodes;  // 主节点数量（每个节点一个）
  int isMaster = (rank == masterRank) ? 1 : 0;  // 当前 rank 是否是主节点

  // 检查是否能连接到 CollNet，其 root 是第 nranks 个 rank
  struct ncclPeerInfo *myInfo = comm->peerInfo+rank, *peerInfo = comm->peerInfo+nranks;
  peerInfo->rank = nranks;

  if (isMaster && type == collNetSend) {
    TRACE(NCCL_INIT, "CollNet [send] : rank %d collNetRank %d collNetNranks %d received connect from rank %d", rank, comm->node, nMasters, masterPeer);
  }

  // ========================================================================
  // 选择 CollNet 传输
  // ========================================================================
  // select
  struct ncclChannelPeer* root = channel->peers[nranks];  // CollNet root peer
  // connector index: 0 for recv, 1 for send
  struct ncclConnector* conn = (type == collNetRecv) ? root->recv+type : root->send+type;
  struct ncclTransportComm* transportComm = (type == collNetRecv) ? &(collNetTransport.recv) : &(collNetTransport.send);
  conn->transportComm = transportComm;

  // ========================================================================
  // 准备连接信息
  // ========================================================================
  // setup
  struct ncclConnect myConnect = { 0 };
  struct {
    int isMaster;
    ncclConnect connect;
  } *allConnects = NULL;
  ncclConnect *masterConnects = NULL;

  // 只有主节点调用 setup
  if (isMaster) {
    NCCLCHECK(transportComm->setup(comm, collNetGraph, myInfo, peerInfo, &myConnect, conn, collNetGraphChannelId, type));
  }

  // 准备连接句柄
  NCCLCHECK(ncclCalloc(&masterConnects, nMasters));

  if (type == collNetRecv) {  // 接收端：使用 AllGather
    // 所有 rank 都必须参与
    NCCLCHECKGOTO(ncclCalloc(&allConnects, nranks), ret, cleanup);
    allConnects[rank].isMaster = isMaster;
    memcpy(&(allConnects[rank].connect), &myConnect, sizeof(struct ncclConnect));

    // AllGather 所有 rank 的连接信息
    NCCLCHECKGOTO(bootstrapAllGather(comm->bootstrap, allConnects, sizeof(*allConnects)), ret, cleanup);

    // 整合：只保留主节点的连接信息
    int c = 0;
    for (int r = 0; r < nranks; r++) {
      if (allConnects[r].isMaster) {
        memcpy(masterConnects+c, &(allConnects[r].connect), sizeof(struct ncclConnect));
        c++;
      }
    }
  } else { // 发送端：复制从 peer 接收主节点接收到的连接信息
    if (isMaster) memcpy(masterConnects+comm->node, connect, sizeof(struct ncclConnect));
  }

  // ========================================================================
  // 建立 CollNet 连接
  // ========================================================================
  // connect
  if (isMaster) {
    // 调用 CollNet 传输的 connect 方法
    NCCLCHECKGOTO(transportComm->connect(comm, masterConnects, nMasters, comm->node, conn), ret, cleanup);

    // 将连接信息拷贝到设备端
    struct ncclDevChannelPeer* devRoot;
    CUDACHECKGOTO(cudaMemcpy(&devRoot, channel->devPeers + nranks, sizeof(struct ncclDevChannelPeer*), cudaMemcpyDeviceToHost), ret, cleanup);
    struct ncclConnInfo* devConnInfo = (type == collNetRecv) ? devRoot->recv + type : devRoot->send + type;
    CUDACHECKGOTO(cudaMemcpy(devConnInfo, &conn->conn, sizeof(struct ncclConnInfo), cudaMemcpyHostToDevice), ret, cleanup);
  }

  // 接收端主节点将连接信息复制到输出参数（用于发送端使用）
  if (isMaster && type == collNetRecv) {
    memcpy(connect, masterConnects+comm->node, sizeof(struct ncclConnect));
    TRACE(NCCL_INIT, "CollNet [recv] : rank %d collNetRank %d collNetNranks %d sent connect to rank %d", rank, comm->node, nMasters, masterPeer);
  }

cleanup:
  if (allConnects != NULL) free(allConnects);
  if (masterConnects != NULL) free(masterConnects);
  return ret != ncclSuccess;  // 返回 true 表示失败（注意这里的逻辑！）
}

// ============================================================================
// ncclTransportCollNetCheck - 检查 CollNet 设置是否全部成功
// ============================================================================
// 所有 rank 参与 AllGather，检查是否有任何 rank 的 CollNet 设置失败
//
// 参数：
//   comm              - 通信域
//   collNetSetupFail  - 当前 rank 的 CollNet 设置失败标志
//
// 返回：ncclSuccess 表示全部成功，ncclSystemError 表示有失败
ncclResult_t ncclTransportCollNetCheck(struct ncclComm* comm, int collNetSetupFail) {
  // AllGather CollNet 设置结果
  int allGatherFailures[NCCL_MAX_LOCAL_RANKS] = {0};
  allGatherFailures[comm->localRank] = collNetSetupFail;

  // 在节点内 AllGather 失败标志
  NCCLCHECK(bootstrapIntraNodeAllGather(comm->bootstrap, comm->localRankToRank, comm->localRank, comm->localRanks, allGatherFailures, sizeof(int)));

  // 检查是否有任何失败
  for (int i=0; i<comm->localRanks; i++) {
    if (allGatherFailures[i] != 0) {
      collNetSetupFail = 1;
      break;
    }
  }

  if (collNetSetupFail) {
    if (comm->localRank == 0) WARN("Cannot initialize CollNet, using point-to-point network instead");
    return ncclSystemError;
  }
  return ncclSuccess;
}

// ============================================================================
// ncclTransportCollNetFree - 释放 CollNet 资源
// ============================================================================
// 释放 CollNet 相关的传输资源
//
// 参数：
//   comm - 通信域
//
// 返回：ncclSuccess
ncclResult_t ncclTransportCollNetFree(struct ncclComm* comm) {
  // Free collNet resources
  // 释放每个通道的 CollNet 资源
  for (int r=0; r<comm->nChannels; r++) {
    struct ncclChannel* channel = comm->channels+r;
    struct ncclChannelPeer* peer = channel->peers[comm->nRanks];  // CollNet root peer
    if (peer) {
      // 使用引用计数管理共享资源
      if (ncclAtomicRefCountDecrement(&peer->refCount) == 0) {
        // 释放所有发送连接资源
        for (int b=0; b<NCCL_MAX_CONNS; b++) {
          struct ncclConnector* send = peer->send + b;
          if (send->transportResources && send->transportComm) NCCLCHECK(send->transportComm->free(send));
          send->transportResources = NULL; // 避免重复释放
        }
        // 释放所有接收连接资源
        for (int b=0; b<NCCL_MAX_CONNS; b++) {
          struct ncclConnector* recv = peer->recv + b;
          if (recv->transportResources && recv->transportComm) NCCLCHECK(recv->transportComm->free(recv));
          recv->transportResources = NULL; // 避免重复释放
        }
      }
    }
  }
  return ncclSuccess;
}
