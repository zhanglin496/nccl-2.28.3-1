/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION. All rights reserved.
 * 版权所有 (c) 2022-2025，NVIDIA 公司。保留所有权利。
 *
 * See LICENSE.txt for license information
 * 有关许可信息，请参见 LICENSE.txt 文件
 ************************************************************************/

// 包含内存注册相关的头文件，定义了内存注册的接口和数据结构
#include "register.h"
// 包含传输层相关的头文件，定义了传输层接口和 P2P 连接功能
#include "transport.h"
// 包含入队操作相关的头文件，定义了任务入队的接口
#include "enqueue.h"
// 包含内联的内存注册函数，提供性能关键的优化实现
#include "register_inline.h"

// 静态函数：检查 P2P 连接是否需要内存注册
// 参数说明：
//   - comm: 通信上下文指针，包含通信域的所有信息
//   - conn: 连接器指针，描述与对等体的连接信息
//   - graph: 拓扑图指针，描述系统拓扑结构
//   - peer: 对等体的 rank 编号
//   - needReg: 输出参数，指示是否需要内存注册
// 返回值：ncclSuccess 表示成功
static ncclResult_t registerCheckP2PConnection(struct ncclComm* comm, struct ncclConnector* conn, struct ncclTopoGraph* graph, int peer, bool* needReg) {
  // 检查连接是否已建立
  // connected 字段表示连接是否已经初始化并建立
  if (conn->connected) {
    // 连接已建立，检查连接标志是否包含 P2P 读写标志
    // conn.flags: 连接标志位，描述连接的类型和属性
    // NCCL_P2P_READ: 支持 P2P 读取操作（可以直接读取对等 GPU 的内存）
    // NCCL_P2P_WRITE: 支持 P2P 写入操作（可以直接写入对等 GPU 的内存）
    // 使用位与运算检查是否设置了任一标志
    if (conn->conn.flags & (NCCL_P2P_READ | NCCL_P2P_WRITE)) {
      // 如果是 P2P 连接，需要注册内存以实现跨 GPU 直接访问
      // 内存注册可以确保内存物理地址固定，便于 DMA 访问
      *needReg = true;
    } else {
      // network connection
      // 网络连接，不需要 P2P 内存注册
      // 网络通信使用不同的内存传输机制，通常通过 RDMA 或网络堆栈
      *needReg = false;
    }
  } else {
    // 连接尚未建立，需要检查是否可以建立 P2P 连接
    // 获取对等体的信息（peerInfo）
    // comm->peerInfo: 数组，存储所有 rank 的对等体信息
    // peer: 对等体的 rank 编号
    struct ncclPeerInfo* peerInfo = &comm->peerInfo[peer];
    // 获取本地节点（当前 rank）的信息
    // comm->rank: 当前进程在通信域中的 rank 编号
    struct ncclPeerInfo* myInfo = &comm->peerInfo[comm->rank];
    // 声明变量用于存储是否可以建立连接的结果
    // 初始化为 0，表示默认不能连接
    int canConnect = 0;
    // 查询传输层是否可以建立 P2P 连接
    // ncclTransports[0]: 第一个传输层实例（通常是 P2P 传输层）
    // canConnect: 输出参数，返回是否可以连接
    // comm: 通信上下文
    // graph: 拓扑图，用于判断拓扑可达性
    // myInfo: 本地节点信息
    // peerInfo: 对等体信息
    NCCLCHECK(ncclTransports[0]->canConnect(&canConnect, comm, graph, myInfo, peerInfo));
    // 根据查询结果设置是否需要内存注册
    if (canConnect) {
      // 可以建立 P2P 连接，因此需要内存注册
      *needReg = true;
    } else {
      // 不能建立 P2P 连接，不需要内存注册
      *needReg = false;
    }
  }
  // 返回成功状态码
  return ncclSuccess;
}

// 函数功能：为 NVLS (NVLink Fabric) 算法注册集体通信缓冲区
// NVLS 是 NVIDIA 的 NVLink 交换机技术，提供跨节点的 GPU 间高速互连
// 参数说明：
//   - comm: 通信上下文指针
//   - info: 集体通信任务信息，包含操作类型、缓冲区指针、数据类型等
//   - outRegBufSend: 输出数组，每个本地 rank 的发送缓冲区注册信息
//   - outRegBufRecv: 输出数组，每个本地 rank 的接收缓冲区注册信息
//   - cleanupQueue: 清理队列，用于存储注册后的回调函数
//   - regNeedConnect: 输出参数，指示注册后是否仍需建立连接
// 返回值：ncclSuccess 表示成功
ncclResult_t ncclRegisterCollNvlsBuffers(
    struct ncclComm* comm, struct ncclTaskColl* info,
    void* outRegBufSend[NCCL_MAX_LOCAL_RANKS],
    void* outRegBufRecv[NCCL_MAX_LOCAL_RANKS],
    struct ncclIntruQueue<struct ncclCommCallback, &ncclCommCallback::next>* cleanupQueue,
    bool* regNeedConnect
  ) {
  // 初始化返回值为成功
  ncclResult_t result = ncclSuccess;

  // 设置缓冲区类型为常规缓冲区（未注册）
  // NCCL_REGULAR_BUFFER: 表示使用常规内存，需要额外的内存拷贝
  info->regBufType = NCCL_REGULAR_BUFFER;
  // 默认情况下，注册后仍需要建立连接
  // 如果内存注册成功，此标志会被设置为 false
  *regNeedConnect = true;
  // 检查是否启用内存注册功能
  // ncclParamLocalRegister(): 检查是否启用本地内存注册参数
  // comm->planner.persistent: 检查是否使用持久化图模式
  // ncclParamGraphRegister(): 检查是否启用图内存注册参数
  // 如果两种注册方式都未启用，直接跳转到 exit
  if (!(ncclParamLocalRegister() || (comm->planner.persistent && ncclParamGraphRegister()))) goto exit;
// CUDA 运行时版本检查：确保 CUDA 版本至少为 11.3
// NVLS 功能需要 CUDA 11.3 或更高版本的支持
#if CUDART_VERSION >= 11030
  // 检查算法是否为 NVLS 或 NVLS_TREE
  // NCCL_ALGO_NVLS: NVLS 算法，使用 NVLink 交换机
  // NCCL_ALGO_NVLS_TREE: NVLS 树算法，使用树形拓扑的 NVLS
  if (info->algorithm == NCCL_ALGO_NVLS || info->algorithm == NCCL_ALGO_NVLS_TREE) {
    // 检查 NVLS 注册支持
    // comm->nvlsRegSupport: 表示系统是否支持 NVLS 内存注册
    // info->opDev.op == ncclDevPreMulSum: 预乘求和操作不支持 NVLS 注册
    if (!comm->nvlsRegSupport || info->opDev.op == ncclDevPreMulSum) goto exit;
    // NVLS 注册标志：0 表示未注册，非 0 表示已注册
    int nvlsReged = 0;
    // CollNet（集体网络）注册标志
    int collnetReged = 0;
    // 获取发送缓冲区指针
    // info->sendbuff: 原始发送缓冲区地址
    const void *sendbuff = info->sendbuff;
    // 获取接收缓冲区指针
    // info->recvbuff: 原始接收缓冲区地址
    void *recvbuff = info->recvbuff;
    // 声明内存句柄变量，用于存储注册后的内存句柄
    // sendHandle: 发送缓冲区的内存句柄
    // recvHandle: 接收缓冲区的内存句柄
    void *recvHandle = NULL, *sendHandle = NULL;
    // AllGather 操作不需要发送缓冲区（每个 rank 只贡献自己的数据）
    // 因此将 sendbuff 设置为 NULL
    if (info->func == ncclFuncAllGather) sendbuff = NULL;
    // ReduceScatter 操作不需要接收缓冲区（结果分散在各个 rank）
    // 因此将 recvbuff 设置为 NULL
    if (info->func == ncclFuncReduceScatter) recvbuff = NULL;
    // 计算数据类型的大小（字节数）
    // ncclTypeSize: 根据数据类型（如 int、float）返回每个元素的字节数
    size_t elementSize = ncclTypeSize(info->datatype);
    // 计算发送缓冲区的总大小
    // ncclFuncSendCount: 根据操作类型和 rank 数计算发送的元素数量
    // comm->nRanks: 通信域中的总 rank 数
    // info->count: 每个元素的数量
    size_t sendbuffSize = elementSize*ncclFuncSendCount(info->func, comm->nRanks, info->count);
    // 计算接收缓冲区的总大小
    // ncclFuncRecvCount: 根据操作类型和 rank 数计算的接收元素数量
    size_t recvbuffSize = elementSize*ncclFuncRecvCount(info->func, comm->nRanks, info->count);

    /* first try graph registration. */
    // 首先尝试图注册（Graph Registration）
    // 图注册是在 CUDA Graph 捕获期间进行的内存注册
    // 优点：可以预注册所有缓冲区，避免运行时注册开销
    // 条件：持久化规划模式 + 启用图注册参数
    if (comm->planner.persistent && ncclParamGraphRegister()) {
      // 执行 NVLS 图注册
      // 参数说明：
      //   - comm: 通信上下文
      //   - sendbuff: 发送缓冲区地址
      //   - recvbuff: 接收缓冲区地址
      //   - sendbuffSize: 发送缓冲区大小
      //   - recvbuffSize: 接收缓冲区大小
      //   - &nvlsReged: 输出注册状态
      //   - outRegBufSend: 输出注册的发送缓冲区信息
      //   - outRegBufRecv: 输出注册的接收缓冲区信息
      //   - cleanupQueue: 清理队列
      //   - &info->nCleanupQueueElts: 清理队列元素数量
      ncclNvlsGraphRegisterBuffer(comm, sendbuff, recvbuff, sendbuffSize, recvbuffSize, &nvlsReged, outRegBufSend, outRegBufRecv, cleanupQueue, &info->nCleanupQueueElts);
    }

    // 如果图注册失败（nvlsReged == 0），尝试本地注册
    // 本地注册在运行时进行，不使用 CUDA Graph
    if (nvlsReged == 0 && ncclParamLocalRegister()) {
      // 执行 NVLS 本地注册
      ncclNvlsLocalRegisterBuffer(comm, sendbuff, recvbuff, sendbuffSize, recvbuffSize, &nvlsReged, outRegBufSend, outRegBufRecv);
    }

    // 如果 NVLS 注册成功，并且是多节点环境，使用 NVLS 算法
    // NVLS 在多节点环境下需要额外的 CollNet 注册
    if (nvlsReged && comm->nNodes > 1 && info->algorithm == NCCL_ALGO_NVLS) {
      // 首先尝试图注册 CollNet 缓冲区
      if (comm->planner.persistent && ncclParamGraphRegister()) {
        // 根据不同的操作类型进行不同的注册
        // AllGather: 只注册发送缓冲区（每个 rank 贡献数据）
        if (info->func == ncclFuncAllGather) {
          // 注册 AllGather 的发送缓冲区到 CollNet
          // collNetSend: 表示这是 CollNet 发送方向
          ncclCollnetGraphRegisterBuffer(comm, info->sendbuff, sendbuffSize, collNetSend, &collnetReged, &sendHandle, cleanupQueue, &info->nCleanupQueueElts);
        } else if (info->func == ncclFuncReduceScatter) {
          // ReduceScatter: 只注册接收缓冲区（结果分散在各个 rank）
          // collNetRecv: 表示这是 CollNet 接收方向
          ncclCollnetGraphRegisterBuffer(comm, info->recvbuff, recvbuffSize, collNetRecv, &collnetReged, &recvHandle, cleanupQueue, &info->nCleanupQueueElts);
        } else if (info->func == ncclFuncAllReduce) {
          // AllReduce: 需要同时注册接收缓冲区（先接收再发送）
          // 先注册接收方向
          ncclCollnetGraphRegisterBuffer(comm, info->recvbuff, recvbuffSize, collNetRecv, &collnetReged, &recvHandle, cleanupQueue, &info->nCleanupQueueElts);
          // 如果接收注册成功，再注册发送方向
          if (collnetReged) ncclCollnetGraphRegisterBuffer(comm, info->recvbuff, recvbuffSize, collNetSend, &collnetReged, &sendHandle, cleanupQueue, &info->nCleanupQueueElts);
        }
      }

      // 如果图注册失败（collnetReged == 0），尝试本地注册
      if (collnetReged == 0 && ncclParamLocalRegister()) {
        // 本地注册 CollNet 缓冲区
        if (info->func == ncclFuncAllGather) {
          ncclCollnetLocalRegisterBuffer(comm, info->sendbuff, sendbuffSize, collNetSend, &collnetReged, &sendHandle);
        } else if (info->func == ncclFuncReduceScatter) {
          ncclCollnetLocalRegisterBuffer(comm, info->recvbuff, recvbuffSize, collNetRecv, &collnetReged, &recvHandle);
        } else if (info->func == ncclFuncAllReduce) {
          ncclCollnetLocalRegisterBuffer(comm, info->recvbuff, recvbuffSize, collNetRecv, &collnetReged, &recvHandle);
          if (collnetReged) ncclCollnetLocalRegisterBuffer(comm, info->recvbuff, recvbuffSize, collNetSend, &collnetReged, &sendHandle);
        }
      }
    }

    // 处理 NVLS 注册成功的后续操作
    if (nvlsReged) {
      // NVLS 注册成功，不需要再建立连接（已通过注册建立）
      *regNeedConnect = 0;
      /* tweak NVLS channels usage; for registered NVLS buffer to saturate bandwidth. */
      // 调整 NVLS 通道使用量，使已注册的 NVLS 缓冲区能够充分利用带宽
      // recChannels: 推荐的通道数量
      int recChannels;
      // 查询 NVLS 注册资源，获取推荐的通道数量
      NCCLCHECK(ncclNvlsRegResourcesQuery(comm, info, &recChannels));
      // 设置最大通道数为推荐值
      info->nMaxChannels = recChannels;
      // 标记缓冲区类型为 NVLS 已注册缓冲区
      info->regBufType |= NCCL_NVLS_REG_BUFFER;
    }

    // 处理 CollNet 注册成功的后续操作
    if (collnetReged) {
      // 标记缓冲区类型为网络已注册缓冲区
      info->regBufType |= NCCL_NET_REG_BUFFER;
      // 保存发送缓冲区的内存句柄
      // 发送句柄用于后续的网络传输操作
      info->sendMhandle = sendHandle;
      // 保存接收缓冲区的内存句柄
      // 接收句柄用于后续的网络接收操作
      info->recvMhandle = recvHandle;
    }
  }
// 跳转标签：函数出口点
exit:
// 结束 CUDA 版本条件编译
#endif
  // 返回结果状态码
  return result;
}

// 函数功能：注册集体通信的缓冲区
// 这个函数是通用的集体通信缓冲区注册函数，支持多种算法和传输方式
// 参数说明：
//   - comm: 通信上下文指针
//   - info: 集体通信任务信息
//   - outRegBufSend: 输出数组，每个本地 rank 的发送缓冲区注册信息
//   - outRegBufRecv: 输出数组，每个本地 rank 的接收缓冲区注册信息
//   - cleanupQueue: 清理队列，用于存储注册后的回调函数
//   - regNeedConnect: 输出参数，指示注册后是否仍需建立连接
// 返回值：ncclSuccess 表示成功
ncclResult_t ncclRegisterCollBuffers(
    struct ncclComm* comm, struct ncclTaskColl* info,
    void* outRegBufSend[NCCL_MAX_LOCAL_RANKS],
    void* outRegBufRecv[NCCL_MAX_LOCAL_RANKS],
    struct ncclIntruQueue<struct ncclCommCallback, &ncclCommCallback::next>* cleanupQueue,
    bool* regNeedConnect
  ) {
  // 初始化返回值为成功
  ncclResult_t result = ncclSuccess;

  // 设置缓冲区类型为常规缓冲区（未注册）
  info->regBufType = NCCL_REGULAR_BUFFER;
  // 默认情况下，注册后仍需要建立连接
  *regNeedConnect = true;
  // 检查是否启用内存注册功能
  // 如果两种注册方式都未启用，直接跳转到 exit
  if (!(ncclParamLocalRegister() || (comm->planner.persistent && ncclParamGraphRegister()))) goto exit;
// CUDA 运行时版本检查
#if CUDART_VERSION >= 11030
  // 检查算法是否为 NVLS 或 NVLS_TREE
  if (info->algorithm == NCCL_ALGO_NVLS || info->algorithm == NCCL_ALGO_NVLS_TREE) {
    /* this part of nvls reg code is temporarily not used and obsolete. */
    // 这部分 NVLS 注册代码暂时未被使用且已过时
    // 检查 NVLS 注册支持
    if (!comm->nvlsRegSupport || info->opDev.op == ncclDevPreMulSum) goto exit;
    // NVLS 注册标志
    int nvlsReged = 0;
    // CollNet 注册标志
    int collnetReged = 0;
    // 获取发送和接收缓冲区指针
    const void *sendbuff = info->sendbuff;
    void *recvbuff = info->recvbuff;
    // 声明内存句柄变量
    void *recvHandle = NULL, *sendHandle = NULL;
    // 根据操作类型调整缓冲区指针
    if (info->func == ncclFuncAllGather) sendbuff = NULL;
    if (info->func == ncclFuncReduceScatter) recvbuff = NULL;
    // 计算数据类型大小和缓冲区总大小
    size_t elementSize = ncclTypeSize(info->datatype);
    size_t sendbuffSize = elementSize*ncclFuncSendCount(info->func, comm->nRanks, info->count);
    size_t recvbuffSize = elementSize*ncclFuncRecvCount(info->func, comm->nRanks, info->count);

    /* first try local registration. */
    // 首先尝试本地注册（与 ncclRegisterCollNvlsBuffers 不同，这里先尝试本地注册）
    if (ncclParamLocalRegister()) {
      // 执行 NVLS 本地注册
      ncclNvlsLocalRegisterBuffer(comm, sendbuff, recvbuff, sendbuffSize, recvbuffSize, &nvlsReged, outRegBufSend, outRegBufRecv);
    }

    // 如果本地注册失败，尝试图注册
    if (nvlsReged == 0 && comm->planner.persistent && ncclParamGraphRegister()) {
      // 执行 NVLS 图注册
      ncclNvlsGraphRegisterBuffer(comm, sendbuff, recvbuff, sendbuffSize, recvbuffSize, &nvlsReged, outRegBufSend, outRegBufRecv, cleanupQueue, &info->nCleanupQueueElts);
    }

    if (comm->nNodes > 1 && info->algorithm == NCCL_ALGO_NVLS) {
      if (ncclParamLocalRegister()) {
        ncclCollnetLocalRegisterBuffer(comm, info->recvbuff, recvbuffSize, collNetSend, &collnetReged, &sendHandle);
        if (collnetReged) ncclCollnetLocalRegisterBuffer(comm, info->recvbuff, recvbuffSize, collNetRecv, &collnetReged, &recvHandle);
      }

      if (collnetReged == 0 && comm->planner.persistent && ncclParamGraphRegister()) {
        ncclCollnetGraphRegisterBuffer(comm, info->recvbuff, recvbuffSize, collNetSend, &collnetReged, &sendHandle, cleanupQueue, &info->nCleanupQueueElts);
        if (collnetReged) ncclCollnetGraphRegisterBuffer(comm, info->recvbuff, recvbuffSize, collNetRecv, &collnetReged, &recvHandle, cleanupQueue, &info->nCleanupQueueElts);
      }
    }

    if (nvlsReged) {
      *regNeedConnect = 0;
      /* tweak NVLS channels usage; for registered NVLS buffer, we only need 4/5 channels to
       * saturate bandwidth. */
      if (comm->nNodes == 1) {
        if (info->func == ncclFuncReduceScatter)
          info->nMaxChannels = std::max(comm->config.minCTAs, std::min(comm->config.maxCTAs, 5));
        else
          info->nMaxChannels = std::max(comm->config.minCTAs, std::min(comm->config.maxCTAs, 4));
      } else {
        info->nMaxChannels = std::max(comm->config.minCTAs, std::min(comm->config.maxCTAs, 6));
      }
      info->regBufType |= NCCL_NVLS_REG_BUFFER;
    }

    if (collnetReged) {
      info->regBufType |= NCCL_NET_REG_BUFFER;
      info->sendMhandle = sendHandle;
      info->recvMhandle = recvHandle;
    }
  } else if (info->protocol == NCCL_PROTO_SIMPLE) {
    // IPC buffer registration
    if (info->func == ncclFuncReduceScatter && info->algorithm != NCCL_ALGO_COLLNET_DIRECT) goto exit;
    if (info->algorithm == NCCL_ALGO_RING && ((info->func == ncclFuncAllReduce && info->sendbuff == info->recvbuff) || info->func == ncclFuncReduce)) goto exit;
    if (info->algorithm == NCCL_ALGO_TREE && info->sendbuff == info->recvbuff) goto exit;
    if (info->algorithm == NCCL_ALGO_COLLNET_CHAIN && info->sendbuff == info->recvbuff && comm->maxLocalRanks > 1) goto exit;
    if (info->func == ncclFuncAllGather && info->algorithm == NCCL_ALGO_PAT) goto exit;

    int peerRanks[NCCL_MAX_LOCAL_RANKS];
    int nPeers = 0;
    size_t elementSize = ncclTypeSize(info->datatype);
    size_t sendbuffSize = elementSize*ncclFuncSendCount(info->func, comm->nRanks, info->count);
    size_t recvbuffSize = elementSize*ncclFuncRecvCount(info->func, comm->nRanks, info->count);
    int regBufFlag = 0;
    memset(peerRanks, 0xff, sizeof(int) * NCCL_MAX_LOCAL_RANKS);

    if (info->algorithm == NCCL_ALGO_COLLNET_DIRECT) {
      struct ncclChannel* channel = comm->channels;
      int ipcRegFlag = 0, netSendRegFlag = 0, netRecvRegFlag = 0;
      void *sendHandle, *recvHandle;
      if (info->func != ncclFuncReduceScatter && comm->isAllDirectP2p) {
        for (int r = 0; r < NCCL_MAX_DIRECT_ARITY; ++r) {
          for (int down = 0; down < 2; ++down) {
            int peer = down ? channel->collnetDirect.down[r] : channel->collnetDirect.up[r];
            if (peer != -1) {
              struct ncclConnector* peerConn = &channel->peers[peer]->recv[0];
              bool needReg = false;

              NCCLCHECK(registerCheckP2PConnection(comm, peerConn, &comm->graphs[NCCL_ALGO_COLLNET_DIRECT], peer, &needReg));
              if (needReg) {
                bool found = false;
                for (int p = 0; p < nPeers; ++p) {
                  if (peerRanks[p] == peer) {
                    found = true;
                    break;
                  }
                }
                if (!found) peerRanks[nPeers++] = peer;
              }
            }
          }
        }

        if (nPeers > 0) {
          if (comm->planner.persistent && ncclParamGraphRegister()) {
            ncclIpcGraphRegisterBuffer(comm, info->sendbuff, sendbuffSize, peerRanks, nPeers, NCCL_IPC_COLLECTIVE, &ipcRegFlag, &info->sendbuffOffset, &info->sendbuffRmtAddrs, cleanupQueue, &info->nCleanupQueueElts);
            if (ipcRegFlag) ncclIpcGraphRegisterBuffer(comm, info->recvbuff, recvbuffSize, peerRanks, nPeers, NCCL_IPC_COLLECTIVE, &ipcRegFlag, &info->recvbuffOffset, &info->recvbuffRmtAddrs, cleanupQueue, &info->nCleanupQueueElts);
          }
          if (!ipcRegFlag && ncclParamLocalRegister()) {
            ncclIpcLocalRegisterBuffer(comm, info->sendbuff, sendbuffSize, peerRanks, nPeers, NCCL_IPC_COLLECTIVE, &ipcRegFlag, &info->sendbuffOffset, &info->sendbuffRmtAddrs);
            if (ipcRegFlag) ncclIpcLocalRegisterBuffer(comm, info->recvbuff, recvbuffSize, peerRanks, nPeers, NCCL_IPC_COLLECTIVE, &ipcRegFlag, &info->recvbuffOffset, &info->recvbuffRmtAddrs);
          }
        }
        if (ipcRegFlag) {
          info->regBufType |= NCCL_IPC_REG_BUFFER;
        }
      }

      // register collnet buffer
      if (info->opDev.op != ncclDevPreMulSum && info->opDev.op != ncclDevSumPostDiv && !(info->func == ncclFuncAllReduce && !comm->isOneRPN)) {
        if (comm->planner.persistent && ncclParamGraphRegister()) {
          ncclCollnetGraphRegisterBuffer(comm, info->sendbuff, sendbuffSize, collNetSend, &netSendRegFlag, &sendHandle, cleanupQueue, &info->nCleanupQueueElts);
          info->sendMhandle = sendHandle;
          if (netSendRegFlag) {
            ncclCollnetGraphRegisterBuffer(comm, info->recvbuff, recvbuffSize, collNetRecv, &netRecvRegFlag, &recvHandle, cleanupQueue, &info->nCleanupQueueElts);
            info->recvMhandle = recvHandle;
          }
        }

        if ((netSendRegFlag == 0 || netRecvRegFlag == 0) && ncclParamLocalRegister()) {
          if (!netSendRegFlag) {
            ncclCollnetLocalRegisterBuffer(comm, info->sendbuff, sendbuffSize, collNetSend, &netSendRegFlag, &sendHandle);
            info->sendMhandle = sendHandle;
          }
          if (netSendRegFlag && !netRecvRegFlag) {
            ncclCollnetLocalRegisterBuffer(comm, info->recvbuff, recvbuffSize, collNetRecv, &netRecvRegFlag, &recvHandle);
            info->recvMhandle = recvHandle;
          }
        }
      }

      if (netSendRegFlag && netRecvRegFlag) {
        if (comm->isOneRPN) info->nMaxChannels = 1;
        info->regBufType |= NCCL_NET_REG_BUFFER;
      }
    } else if (info->algorithm == NCCL_ALGO_RING) {
      struct ncclReg* recvRegRecord = NULL;
      struct ncclReg* sendRegRecord = NULL;
      int sendNetPeers = comm->nChannels;
      int recvNetPeers = comm->nChannels;
      struct ncclConnector** sendNetConns = NULL;
      struct ncclConnector** recvNetConns = NULL;
      void** sendNetHandles = NULL;
      void** recvNetHandles = NULL;
      void** srecvNetHandles = NULL;
      bool hasRecvNetPeer = false;
      bool hasSendNetPeer = false;

      NCCLCHECK(ncclRegFind(comm, info->recvbuff, recvbuffSize, &recvRegRecord));
      if (recvRegRecord == NULL && !(comm->planner.persistent && ncclParamGraphRegister())) goto exit;
      NCCLCHECK(ncclRegFind(comm, info->sendbuff, sendbuffSize, &sendRegRecord));
      if (sendRegRecord == NULL && !(comm->planner.persistent && ncclParamGraphRegister())) goto exit;
      NCCLCHECK(ncclCalloc(&sendNetConns, comm->nChannels));
      NCCLCHECK(ncclCalloc(&sendNetHandles, comm->nChannels));
      NCCLCHECK(ncclCalloc(&recvNetConns, comm->nChannels));
      NCCLCHECK(ncclCalloc(&recvNetHandles, comm->nChannels));
      NCCLCHECK(ncclCalloc(&srecvNetHandles, comm->nChannels));

      for (int c = 0; c < comm->nChannels; ++c) {
        struct ncclChannel* channel = comm->channels + c;
        for (int r = 0; r < 2; ++r) {
          int peer;
          struct ncclConnector* peerConn;
          if (r == 0) {
            peer = channel->ring.prev;
            peerConn = &channel->peers[peer]->recv[0];
            if (peerConn->conn.flags & NCCL_DIRECT_NIC) {
              recvNetConns[c] = peerConn;
              hasRecvNetPeer = true;
            }
          } else {
            peer = channel->ring.next;
            peerConn = &channel->peers[peer]->send[0];
            if (peerConn->conn.flags & NCCL_DIRECT_NIC) {
              sendNetConns[c] = peerConn;
              hasSendNetPeer = true;
            }
          }
          if (peerConn->conn.flags & (NCCL_P2P_READ | NCCL_P2P_WRITE)) {
            bool found = false;
            for (int p = 0; p < nPeers; ++p) {
              if (peerRanks[p] == peer) {
                found = true;
                break;
              }
            }
            if (!found) peerRanks[nPeers++] = peer;
          }
        }
      }
      if (nPeers > 0 && comm->isAllDirectP2p) {
        if (comm->planner.persistent && ncclParamGraphRegister()) {
          ncclIpcGraphRegisterBuffer(comm, info->recvbuff, recvbuffSize, peerRanks, nPeers, NCCL_IPC_COLLECTIVE, &regBufFlag, &info->recvbuffOffset, &info->recvbuffRmtAddrs, cleanupQueue, &info->nCleanupQueueElts);
        }
        if (!regBufFlag && ncclParamLocalRegister()) {
          ncclIpcLocalRegisterBuffer(comm, info->recvbuff, recvbuffSize, peerRanks, nPeers, NCCL_IPC_COLLECTIVE, &regBufFlag, &info->recvbuffOffset, &info->recvbuffRmtAddrs);
        }
      }
      if (regBufFlag) {
        info->regBufType = NCCL_IPC_REG_BUFFER;
      }

      // start net registration
      regBufFlag = 0;
      if (!comm->useNetPXN && comm->useGdr && comm->netDeviceType != NCCL_NET_DEVICE_UNPACK) {
        if (comm->planner.persistent && ncclParamGraphRegister()) {
          if (hasSendNetPeer) {
            ncclNetGraphRegisterBuffer(comm, info->sendbuff, sendbuffSize, sendNetConns, sendNetPeers, &regBufFlag, sendNetHandles, cleanupQueue, &info->nCleanupQueueElts);
            if (regBufFlag)
              ncclNetGraphRegisterBuffer(comm, info->recvbuff, recvbuffSize, sendNetConns, sendNetPeers, &regBufFlag, srecvNetHandles, cleanupQueue, &info->nCleanupQueueElts);
          }
          if ((regBufFlag || !hasSendNetPeer) && hasRecvNetPeer)
            ncclNetGraphRegisterBuffer(comm, info->recvbuff, recvbuffSize, recvNetConns, recvNetPeers, &regBufFlag, recvNetHandles, cleanupQueue, &info->nCleanupQueueElts);
        }
        if (!regBufFlag && ncclParamLocalRegister()) {
          if (hasSendNetPeer) {
            ncclNetLocalRegisterBuffer(comm, info->sendbuff, sendbuffSize, sendNetConns, sendNetPeers, &regBufFlag, sendNetHandles);
            if (regBufFlag)
              ncclNetLocalRegisterBuffer(comm, info->recvbuff, recvbuffSize, sendNetConns, sendNetPeers, &regBufFlag, srecvNetHandles);
          }
          if ((regBufFlag || !hasSendNetPeer) && hasRecvNetPeer)
            ncclNetLocalRegisterBuffer(comm, info->recvbuff, recvbuffSize, recvNetConns, recvNetPeers, &regBufFlag, recvNetHandles);
        }
      }

      if (regBufFlag) {
        info->regBufType |= NCCL_NET_REG_BUFFER;
        info->sendNetHandles = sendNetHandles;
        info->recvNetHandles = recvNetHandles;
        info->srecvNetHandles = srecvNetHandles;
        if (comm->isOneRPN && (info->func == ncclFuncAllGather || info->func == ncclFuncBroadcast)) {
          info->nMaxChannels = 1;
        }
      } else {
        free(sendNetHandles);
        free(recvNetHandles);
        free(srecvNetHandles);
      }

      free(sendNetConns);
      free(recvNetConns);
    } else if (info->algorithm == NCCL_ALGO_TREE || info->algorithm == NCCL_ALGO_COLLNET_CHAIN) {
      struct ncclReg* recvRegRecord;
      int netSendRegFlag = 0, netRecvRegFlag = 0;
      void *sendHandle, *recvHandle;
      NCCLCHECK(ncclRegFind(comm, info->recvbuff, recvbuffSize, &recvRegRecord));
      if (recvRegRecord == NULL && !(comm->planner.persistent && ncclParamGraphRegister())) goto exit;
      if (comm->isAllDirectP2p) {
        for (int c = 0; c < comm->nChannels; ++c) {
          struct ncclChannel* channel = comm->channels + c;
          struct ncclTree* tree = NULL;
          int peers[NCCL_MAX_TREE_ARITY + 1];

          if (info->algorithm == NCCL_ALGO_TREE)
            tree = &channel->tree;
          else
            tree = &channel->collnetChain;
          for (int p = 0; p < NCCL_MAX_TREE_ARITY; ++p) peers[p] = tree->down[p];
          peers[NCCL_MAX_TREE_ARITY] = tree->up;
          for (int p = 0; p < NCCL_MAX_TREE_ARITY + 1; ++p) {
            int peer = peers[p];
            bool peerNeedReg = false;
            struct ncclConnector* recvConn = NULL;
            // P2P transport
            if (peer == -1 || peer == comm->nRanks) continue;
            recvConn = &channel->peers[peer]->recv[0];
            NCCLCHECK(registerCheckP2PConnection(comm, recvConn, &comm->graphs[info->algorithm], peer, &peerNeedReg));

            if (peerNeedReg) {
              bool found = false;
              for (int pindex = 0; pindex < nPeers; ++pindex) {
                if (peerRanks[pindex] == peer) {
                  found = true;
                  break;
                }
              }
              if (!found) peerRanks[nPeers++] = peer;
            }
          }
        }
        if (nPeers > 0) {
          if (comm->planner.persistent && ncclParamGraphRegister()) {
            ncclIpcGraphRegisterBuffer(comm, info->recvbuff, recvbuffSize, peerRanks, nPeers, NCCL_IPC_COLLECTIVE, &regBufFlag, &info->recvbuffOffset, &info->recvbuffRmtAddrs, cleanupQueue, &info->nCleanupQueueElts);
          }
          if (!regBufFlag && ncclParamLocalRegister()) {
            ncclIpcLocalRegisterBuffer(comm, info->recvbuff, recvbuffSize, peerRanks, nPeers, NCCL_IPC_COLLECTIVE, &regBufFlag, &info->recvbuffOffset, &info->recvbuffRmtAddrs);
          }
        }
        if (regBufFlag) {
          info->regBufType = NCCL_IPC_REG_BUFFER;
        }
      }

      // register collnet chain 1RPN buffer
      if (info->algorithm == NCCL_ALGO_COLLNET_CHAIN && info->opDev.op != ncclDevPreMulSum && info->opDev.op != ncclDevSumPostDiv && comm->isOneRPN) {
        if (comm->planner.persistent && ncclParamGraphRegister()) {
          ncclCollnetGraphRegisterBuffer(comm, info->sendbuff, sendbuffSize, collNetSend, &netSendRegFlag, &sendHandle, cleanupQueue, &info->nCleanupQueueElts);
          info->sendMhandle = sendHandle;
          if (netSendRegFlag) {
            ncclCollnetGraphRegisterBuffer(comm, info->recvbuff, recvbuffSize, collNetRecv, &netRecvRegFlag, &recvHandle, cleanupQueue, &info->nCleanupQueueElts);
            info->recvMhandle = recvHandle;
          }
        }

        if ((netSendRegFlag == 0 || netRecvRegFlag == 0) && ncclParamLocalRegister()) {
          if (!netSendRegFlag) {
            ncclCollnetLocalRegisterBuffer(comm, info->sendbuff, sendbuffSize, collNetSend, &netSendRegFlag, &sendHandle);
            info->sendMhandle = sendHandle;
          }
          if (netSendRegFlag && !netRecvRegFlag) {
            ncclCollnetLocalRegisterBuffer(comm, info->recvbuff, recvbuffSize, collNetRecv, &netRecvRegFlag, &recvHandle);
            info->recvMhandle = recvHandle;
          }
        }
      }

      if (netSendRegFlag && netRecvRegFlag) {
        if (comm->isOneRPN) info->nMaxChannels = 1;
        info->regBufType |= NCCL_NET_REG_BUFFER;
      }
    }

    if (info->regBufType == NCCL_IPC_REG_BUFFER && comm->nNodes == 1 && 16 < info->nMaxChannels && info->nMaxChannels <= 24) {
      info->nMaxChannels = 16;
    }
  }
exit:
#endif
  return result;
}
