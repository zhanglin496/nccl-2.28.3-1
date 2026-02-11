/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 * 版权所有 (c) 2016-2022, NVIDIA CORPORATION. 保留所有权利。
 *
 * See LICENSE.txt for license information
 * 详见 LICENSE.txt 获取许可证信息
 ************************************************************************/

// ============================================================================
// 头文件引入区域 - 引入NCCL核心组件和工具模块
// ============================================================================
#include "comm.h"         // 通信上下文相关定义
#include "coll_net.h"     // 集合网络(CollNet)接口定义
#include "graph.h"        // CUDA图支持相关定义
#include "proxy.h"        // 代理线程相关定义
#include "gdrwrap.h"      // GPUDirect RDMA封装
#include "transport.h"    // 传输层抽象接口
#include "assert.h"       // 断言宏定义
#include "bootstrap.h"    // Bootstrap初始化协议
#include "channel.h"      // 通信通道管理
#include "register_inline.h"  // 内存注册内联函数

// 前向声明 - 获取GDR Copy同步和刷新相关的环境变量参数
int64_t ncclParamGdrCopySyncEnable();   // GDR Copy同步使能参数
int64_t ncclParamGdrCopyFlushEnable();  // GDR Copy刷新使能参数

// ============================================================================
// CollNet接收端连接信息结构
// ============================================================================
struct collNetRecvConnectInfo {
  collNetHandle_t collNetHandle;  // CollNet句柄，用于标识接收端的CollNet连接
};
// 编译期断言：确保接收端连接信息大小不超过连接信息大小限制
static_assert(sizeof(collNetRecvConnectInfo) <= CONNECT_SIZE, "Collnet Recv Connect info is too large");

// ============================================================================
// CollNet发送端连接信息结构
// ============================================================================
struct collNetSendConnectInfo {
  void* mhandles[NCCL_NUM_PROTOCOLS];  // 每种协议的内存句柄数组（用于LL、LL128、SIMPLE等协议）
  void* reqFifo;                        // 请求FIFO队列指针，用于发送端和接收端同步
};
// 编译期断言：确保发送端连接信息大小不超过连接信息大小限制
static_assert(sizeof(collNetSendConnectInfo) <= CONNECT_SIZE, "Collnet Send Connect info is too large");

// ============================================================================
// CollNet分组相关宏定义
// ============================================================================
#define COLLNET_GROUP_NSUBS 8                                    // 每个组包含的子操作数量
#define COLLNET_MAX_GROUPS (NCCL_PROXY_MAX_SUBS/COLLNET_GROUP_NSUBS)  // 最大分组数

// ============================================================================
// 内存映射类型枚举 - 用于connectMap中区分不同类型的内存区域
// ============================================================================
#define NCCL_NET_MAP_HOSTMEM 0          // 主机内存（Host Memory）
#define NCCL_NET_MAP_DEVMEM 1           // 设备内存（Device Memory/GPU显存）
#define NCCL_NET_MAP_SHARED_HOSTMEM 2   // 共享主机内存（Shared Host Memory）
#define NCCL_NET_MAP_SHARED_DEVMEM 3    // 共享设备内存（Shared Device Memory）
#define NCCL_NET_MAP_GDCMEM 4           // GDR Copy内存（GPUDirect Copy Memory）
#define NCCL_NET_MAP_MEMS 5             // 内存类型总数

// ============================================================================
// 偏移量掩码定义 - 用于编码内存类型和偏移量的位操作
// ============================================================================
#define NCCL_NET_MAP_MASK_DEVMEM 0x40000000  // 设备内存掩码（位30）
#define NCCL_NET_MAP_MASK_SHARED 0x80000000  // 共享内存掩码（位31）
#define NCCL_NET_MAP_MASK_USED   0x20000000  // 已使用掩码（位29）
#define NCCL_NET_MAP_MASK_OFFSET 0x1fffffff  // 偏移量掩码（低29位，最大512MB）

// ============================================================================
// 宏函数：从偏移量中提取内存bank编号
// 参数：
//   mapStruct - connectMap结构体指针
//   offsetName - 偏移量字段名称
// 返回：内存bank索引（0-4）
// ============================================================================
#define NCCL_NET_MAP_OFFSET_BANK(mapStruct, offsetName) \
  ((mapStruct)->offsets.offsetName >> 30)  // 右移30位获取高3位作为bank索引

// ============================================================================
// 宏函数：检查偏移量是否为NULL
// 参数：
//   mapStruct - connectMap结构体指针
//   offsetName - 偏移量字段名称
// 返回：如果偏移量为NULL则返回true（高2位为0）
// ============================================================================
#define NCCL_NET_MAP_OFFSET_NULL(mapStruct, offsetName) \
  (((mapStruct)->offsets.offsetName >> 29) == 0)  // 检查高2位是否为0

// ============================================================================
// 宏函数：根据偏移量获取实际的CPU或GPU指针
// 参数：
//   mapStruct - connectMap结构体指针
//   cpuOrGpu - "cpu"或"gpu"，用于选择指针类型
//   offsetName - 偏移量字段名称
// 返回：计算后的指针，如果偏移量为NULL则返回NULL
// ============================================================================
#define NCCL_NET_MAP_GET_POINTER(mapStruct, cpuOrGpu, offsetName) \
  (NCCL_NET_MAP_OFFSET_NULL(mapStruct, offsetName) ? NULL : \   // 如果偏移量为NULL，直接返回NULL指针
   (mapStruct)->mems[NCCL_NET_MAP_OFFSET_BANK(mapStruct, offsetName)].cpuOrGpu##Ptr + ((mapStruct)->offsets.offsetName & NCCL_NET_MAP_MASK_OFFSET))  // 否则计算实际指针地址

// ============================================================================
// 宏函数：检查偏移量是否指向设备内存
// 参数：
//   mapStruct - connectMap结构体指针
//   offsetName - 偏移量字段名称
// 返回：如果指向设备内存则返回true
// ============================================================================
#define NCCL_NET_MAP_DEV_MEM(mapStruct, offsetName) \
  (((mapStruct)->offsets.offsetName & NCCL_NET_MAP_MASK_DEVMEM) != 0)  // 检查设备内存位是否置位

// ============================================================================
// 宏函数：向connectMap中添加指针映射
// 参数：
//   mapStruct - connectMap结构体指针
//   shared - 是否为共享内存（0=非共享，1=共享）
//   dev - 是否为设备内存（0=主机，1=设备）
//   memSize - 内存大小
//   offsetName - 偏移量字段名称
// 功能：根据内存类型计算bank值，并更新偏移量和内存大小
// ============================================================================
#define NCCL_NET_MAP_ADD_POINTER(mapStruct, shared, dev, memSize, offsetName) do { \
    int bank = NCCL_NET_MAP_MASK_USED + (dev)*NCCL_NET_MAP_MASK_DEVMEM + (shared)*NCCL_NET_MAP_MASK_SHARED; \  // 计算bank标识码
    if ((shared) == 0) { \  // 非共享内存路径
      if (dev) { \  // 设备内存
        (mapStruct)->offsets.offsetName = bank + (mapStruct)->mems[NCCL_NET_MAP_DEVMEM].size; \  // 设置偏移量
        (mapStruct)->mems[NCCL_NET_MAP_DEVMEM].size += memSize; \  // 增加设备内存池大小
      } else { \  // 主机内存
        (mapStruct)->offsets.offsetName = bank + (mapStruct)->mems[NCCL_NET_MAP_HOSTMEM].size; \  // 设置偏移量
        (mapStruct)->mems[NCCL_NET_MAP_HOSTMEM].size += memSize; \  // 增加主机内存池大小
      } \
    } else { \  // 共享内存路径
      (mapStruct)->offsets.offsetName = bank; \  // 共享内存偏移量仅为bank标识
    } \
} while (0);  // do-while(0)确保宏可以安全地在任何地方使用

// ============================================================================
// 连接映射内存结构 - 描述一块内存区域的CPU/GPU指针和大小
// ============================================================================
struct connectMapMem{
  char* gpuPtr;   // GPU指针（设备端地址）
  char* cpuPtr;   // CPU指针（主机端地址）
  int size;       // 内存区域大小（字节）
};

// ============================================================================
// 连接映射结构 - 管理CollNet连接中的所有内存映射
// ============================================================================
struct connectMap {
  int shared;  // 是否使用共享内存模式
  // First 3 bits of offsets determine the mem bank. 001 is host mem, 011 is dev mem, 101 is shared host mem and 111 is shared dev mem.
  // 偏移量的高3位确定内存bank：001=主机内存，011=设备内存，101=共享主机内存，111=共享设备内存
  struct connectMapMem mems[NCCL_NET_MAP_MEMS];  // 5种内存类型的数组
  // Offsets. 3 MSBs indicate mem bank, 111 indicates NULL.
  // 偏移量结构。最高3位表示内存bank，111表示NULL指针
  struct {
    uint32_t sendMem;                // 发送内存偏移量
    uint32_t recvMem;                // 接收内存偏移量
    uint32_t buffs[NCCL_NUM_PROTOCOLS];  // 每种协议的缓冲区偏移量数组
  } offsets;
};

// ============================================================================
// 请求槽结构 - 用于sendProxy和recvProxy之间的同步
// ============================================================================
struct reqSlot {
  bool turnIsSendNotRecv;  // 轮次标志：true=发送端轮次，false=接收端轮次
  int size;                // 本次操作的数据大小
};

// ============================================================================
// CollNet发送端资源结构 - 管理发送端的所有资源
// ============================================================================
struct sendResources {
  struct connectMap map;                    // 连接映射结构，管理内存布局
  void* collNetComm;                        // CollNet通信句柄
  struct ncclSendMem* sendMem;              // 发送内存结构指针
  struct ncclRecvMem* recvMem;              // 接收内存结构指针（用于从接收端读取状态）

  int rank;                                 // 当前进程在通信域中的rank
  int nranks;                               // 通信域中的总rank数
  int netDev;                               // 网络设备ID
  enum ncclTopoGdrMode useGdr;              // GPUDirect RDMA模式
  int useDmaBuf;                            // 是否使用DMA-BUF
  uint64_t* gdcSync;                        // GDR Copy同步指针（用于发送端控制）
  void* gdrDesc;                            // GDR描述符
  void* sendMhandles[NCCL_NUM_PROTOCOLS];   // 发送内存句柄数组（每种协议）
  void* recvMhandles[NCCL_NUM_PROTOCOLS];   // 接收内存句柄数组（从接收端获取）
  uint64_t step;                            // 当前步骤号
  struct reqSlot (*reqFifo)[NCCL_STEPS];    // 请求FIFO队列指针（从接收端获取）
  int collNetRank;                          // CollNet通信域中的rank
  size_t maxCollBytes;                      // 单次CollNet操作支持的最大字节数
};

// ============================================================================
// CollNet接收端资源结构 - 管理接收端的所有资源
// ============================================================================
struct recvResources {
  struct connectMap map;                    // 连接映射结构，管理内存布局
  void* collNetComm;                        // CollNet通信句柄
  struct ncclSendMem* sendMem;              // 发送内存结构指针（用于读取发送端状态）
  struct ncclRecvMem* recvMem;              // 接收内存结构指针

  int rank;                                 // 当前进程在通信域中的rank
  int nranks;                               // 通信域中的总rank数
  int netDev;                               // 网络设备ID
  enum ncclTopoGdrMode useGdr;              // GPUDirect RDMA模式
  int useDmaBuf;                            // 是否使用DMA-BUF
  int needFlush;                            // 是否需要刷新GDR缓冲区
  uint64_t* gdcSync;                        // GDR Copy同步指针
  uint64_t* gdcFlush;                       // GDR Copy刷新指针
  void* gdrDesc;                            // GDR描述符
  void* mhandles[NCCL_NUM_PROTOCOLS];       // 内存句柄数组（每种协议）
  uint64_t step;                            // 当前步骤号
  struct reqSlot reqFifo[COLLNET_MAX_GROUPS][NCCL_STEPS];  // 请求FIFO队列（二维数组：分组x步骤）
  int collNetRank;                          // CollNet通信域中的rank
  size_t maxCollBytes;                      // 单次CollNet操作支持的最大字节数
};

// ============================================================================
// 函数: canConnect - 检查两个peer是否可以使用CollNet传输层连接
// 参数:
//   ret - 输出参数，返回是否可以连接（1=可以，0=不可以）
//   comm - 通信上下文
//   graph - 拓扑图
//   info1 - 第一个peer的信息
//   info2 - 第二个peer的信息
// 返回: ncclResult_t - 操作结果状态码
// 说明: CollNet传输层专用于集合通信，不支持点对点(p2p)连接
// ============================================================================
static ncclResult_t canConnect(int* ret, struct ncclComm* comm, struct ncclTopoGraph* graph, struct ncclPeerInfo* info1, struct ncclPeerInfo* info2) {
  // This transport cannot be used for p2p
  // 此传输层不能用于点对点通信
  *ret = 0;  // 返回0表示不能连接
  return ncclSuccess;
}

// ============================================================================
// 函数: getHandleForAddressRangeFlags - 获取cuMemGetHandleForAddressRange调用的标志
// 参数:
//   useGdr - GPUDirect RDMA模式
// 返回: int - CUDA内存句柄标志
// 说明: 在CUDA 12.8+版本中，支持强制指定DMA-BUF映射类型为PCIe
// ============================================================================
static inline int getHandleForAddressRangeFlags(ncclTopoGdrMode useGdr) {
  int flags = 0;  // 初始化标志为0
#if CUDA_VERSION >= 12080  // 仅在CUDA 12.8及以上版本编译此代码
  // Force mapping on PCIe on systems with both PCI and C2C attachments.
  // 在同时具有PCIe和C2C(Cache-to-Cache)连接的系统中，强制使用PCIe映射
  if (useGdr == ncclTopoGdrModePci) flags = CU_MEM_RANGE_FLAG_DMA_BUF_MAPPING_TYPE_PCIE;  // 设置PCIe映射标志
#endif
  return flags;  // 返回计算出的标志
}

// ============================================================================
// 结构: setupReq - CollNet设置请求，用于在proxy线程中初始化资源
// ============================================================================
struct setupReq {
  int netDev;                            // 网络设备ID
  enum ncclTopoGdrMode useGdr;           // GPUDirect RDMA使用模式
  int needFlush;                         // 接收端是否需要刷新GDR缓冲区
  struct ncclCollNetSharedRes* collNet;  // CollNet共享资源指针
};


// ============================================================================
/* Setup send connector, and return connect information for others in the coll
 * communicator to connect to me */
/* 设置发送端连接器，并返回连接信息供集合通信域中的其他rank连接到此rank */
static ncclResult_t sendSetup(struct ncclComm* comm, struct ncclTopoGraph* graph, struct ncclPeerInfo* myInfo, struct ncclPeerInfo* peerInfo, struct ncclConnect* connectInfo, struct ncclConnector* send, int channelId, int connIndex) {
  struct setupReq req = { 0 };  // 初始化设置请求结构体

  int proxyRank;  // Proxy线程的rank
  int64_t netId;  // 网络ID
  // 获取网络设备信息：根据rank、通道ID等参数查询网络设备和proxy rank
  NCCLCHECK(ncclTopoGetNetDev(comm, myInfo->rank, graph, channelId, -1, &netId, &req.netDev, &proxyRank));
  // 检查是否可以使用GPUDirect RDMA（发送端参数send=1）
  NCCLCHECK(ncclTopoCheckGdr(comm->topo, myInfo->rank, netId, 1, &req.useGdr));
  // 如果使用GDR，则在连接标志中设置NCCL_DIRECT_NIC位
  send->conn.flags |= req.useGdr ? NCCL_DIRECT_NIC : 0;

  // 设置proxy连接的本地rank（用于进程间通信）
  send->proxyConn.tpLocalRank = comm->topParentLocalRanks[comm->localRank];
  // 连接到proxy线程（TRANSPORT_COLLNET类型，isSend=1表示发送端）
  NCCLCHECK(ncclProxyConnect(comm, TRANSPORT_COLLNET, 1, myInfo->rank, &send->proxyConn));
  // 增加CollNet共享资源的引用计数
  ncclAtomicRefCountIncrement(&comm->collNetSharedRes->refCount);
  req.collNet = comm->collNetSharedRes;  // 设置共享资源指针
  // 阻塞调用proxy的setup函数
  NCCLCHECK(ncclProxyCallBlocking(comm, &send->proxyConn, ncclProxyMsgSetup, &req, sizeof(req), NULL, 0));

  // 输出初始化日志信息：显示通道ID、连接索引、rank、CollNet名称、网络设备、GDR模式等
  INFO(NCCL_INIT|NCCL_NET,"CollNet %02d/%1d : %d [send] via COLLNET/%s/%d%s%s", channelId, connIndex, myInfo->rank, collNetName(comm), req.netDev,
      req.useGdr ? "/GDRDMA" : "", req.useGdr==ncclTopoGdrModePci ? "(PCI)" : "");
  return ncclSuccess;
}

// ============================================================================
// 函数: recvSetup - 设置接收端连接器
// 参数: 与sendSetup类似，但用于接收端
// 返回: ncclResult_t - 操作结果状态码
// 说明: 接收端需要额外判断是否需要刷新GDR缓冲区
// ============================================================================
static ncclResult_t recvSetup(struct ncclComm* comm, struct ncclTopoGraph* graph, struct ncclPeerInfo* myInfo, struct ncclPeerInfo* peerInfo, struct ncclConnect* connectInfo, struct ncclConnector* recv, int channelId, int connIndex) {
  struct setupReq req = { 0 };  // 初始化设置请求结构体

  int proxyRank;  // Proxy线程的rank
  int64_t netId;  // 网络ID
  // 获取网络设备信息
  NCCLCHECK(ncclTopoGetNetDev(comm, myInfo->rank, graph, channelId, -1, &netId, &req.netDev, &proxyRank));
  // 检查是否可以使用GPUDirect RDMA（recv=0表示接收端）
  NCCLCHECK(ncclTopoCheckGdr(comm->topo, myInfo->rank, netId, 0, &req.useGdr));
  recv->conn.flags |= req.useGdr ? NCCL_DIRECT_NIC : 0;  // 设置GDR标志
  // Determine whether we need to flush the GDR buffer on recv or not
  // 确定接收端是否需要刷新GDR缓冲区（某些硬件配置需要显式刷新以确保数据一致性）
  if (req.useGdr) NCCLCHECK(ncclTopoNeedFlush(comm, netId, req.netDev, myInfo->rank, &req.needFlush));

  // 设置proxy连接的本地rank
  recv->proxyConn.tpLocalRank = comm->topParentLocalRanks[comm->localRank];
  // 连接到proxy线程（TRANSPORT_COLLNET类型，isSend=0表示接收端）
  NCCLCHECK(ncclProxyConnect(comm, TRANSPORT_COLLNET, 0, myInfo->rank, &recv->proxyConn));
  // 编译期断言：确保接收端连接信息大小不超过ncclConnect结构体大小
  static_assert(sizeof(collNetRecvConnectInfo) <= sizeof(struct ncclConnect), "Collnet Recv Connect info is too big");
  struct collNetRecvConnectInfo* info = (struct collNetRecvConnectInfo*) connectInfo;  // 类型转换
  // 增加CollNet共享资源的引用计数
  ncclAtomicRefCountIncrement(&comm->collNetSharedRes->refCount);
  req.collNet = comm->collNetSharedRes;  // 设置共享资源指针
  // 阻塞调用proxy的setup函数，获取CollNet句柄
  NCCLCHECK(ncclProxyCallBlocking(comm, &recv->proxyConn, ncclProxyMsgSetup, &req, sizeof(req), &info->collNetHandle, sizeof(collNetHandle_t)));

  // 输出初始化日志信息
  INFO(NCCL_INIT|NCCL_NET,"CollNet %02d/%1d : %d [receive] via COLLNET/%s/%d%s%s", channelId, connIndex, myInfo->rank, collNetName(comm), req.netDev,
      req.useGdr ? "/GDRDMA" : "", req.useGdr==ncclTopoGdrModePci ? "(PCI)" : "");
  return ncclSuccess;
}

// ============================================================================
// 函数: collNetDumpMap - 调试函数，打印connectMap的详细信息
// 参数: map - 连接映射结构指针
// 返回: ncclResult_t - 操作结果状态码
// 说明: 主要用于调试，打印所有内存bank和偏移量信息
// ============================================================================
static ncclResult_t collNetDumpMap(struct connectMap* map) {
  printf("Dump map\n");  // 打印标题
  // 打印内存bank 0：主机内存
  struct connectMapMem *mem = map->mems+NCCL_NET_MAP_HOSTMEM;
  printf("Mem 0: Host mem (%x B) CPU %p GPU %p\n", mem->size, mem->cpuPtr, mem->gpuPtr);
  // 打印内存bank 1：设备内存
  mem = map->mems+NCCL_NET_MAP_DEVMEM;
  printf("Mem 1: Vid  mem CPU (%x B) %p GPU %p\n", mem->size, mem->cpuPtr, mem->gpuPtr);
  // 打印内存bank 2：共享主机内存
  mem = map->mems+NCCL_NET_MAP_SHARED_HOSTMEM;
  printf("Mem 2: Shared Host mem (%x B) CPU %p GPU %p\n", mem->size, mem->cpuPtr, mem->gpuPtr);
  // 打印内存bank 3：共享设备内存
  mem = map->mems+NCCL_NET_MAP_SHARED_DEVMEM;
  printf("Mem 3: Shared Vid  (%x B) mem CPU %p GPU %p\n", mem->size, mem->cpuPtr, mem->gpuPtr);
  // 打印发送内存信息：使用状态、bank编号、偏移量、CPU指针、GPU指针
  printf("SendMem -> Used %d Bank %d Offset %x, cpu %p gpu %p\n",
      map->offsets.sendMem & NCCL_NET_MAP_MASK_USED ? 1 : 0,  // 是否已使用
      NCCL_NET_MAP_OFFSET_BANK(map, sendMem),  // bank编号
      map->offsets.sendMem & NCCL_NET_MAP_MASK_OFFSET,  // 偏移量
      NCCL_NET_MAP_GET_POINTER(map, cpu, sendMem),  // CPU指针
      NCCL_NET_MAP_GET_POINTER(map, gpu, sendMem));  // GPU指针
  // 打印接收内存信息
  printf("RecvMem -> Used %d Bank %d Offset %x, cpu %p gpu %p\n",
      map->offsets.recvMem & NCCL_NET_MAP_MASK_USED ? 1 : 0,
      NCCL_NET_MAP_OFFSET_BANK(map, recvMem),
      map->offsets.recvMem & NCCL_NET_MAP_MASK_OFFSET,
      NCCL_NET_MAP_GET_POINTER(map, cpu, recvMem),
      NCCL_NET_MAP_GET_POINTER(map, gpu, recvMem));
  // 打印每种协议的缓冲区信息
  for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
    printf("Proto %d -> Used %d Bank %d Offset %x, cpu %p, gpu %p\n", p,  // 协议索引
        map->offsets.buffs[p] & NCCL_NET_MAP_MASK_USED ? 1 : 0,  // 使用状态
        NCCL_NET_MAP_OFFSET_BANK(map, buffs[p]),  // bank编号
        map->offsets.buffs[p] & NCCL_NET_MAP_MASK_OFFSET,  // 偏移量
        NCCL_NET_MAP_GET_POINTER(map, cpu, buffs[p]),  // CPU指针
        NCCL_NET_MAP_GET_POINTER(map, gpu, buffs[p]));  // GPU指针
  }
  printf("End of dump\n");  // 打印结束标记
  return ncclSuccess;
}

// ============================================================================
// 结构: collNetConnectArgs - CollNet连接参数，用于连接阶段
// ============================================================================
struct collNetConnectArgs {
  int rank;                     // 当前rank
  int nranks;                   // 总rank数
  struct ncclConnect* connectInfos;  // 所有rank的连接信息数组
};

// 前向声明：proxy进度推进函数
static ncclResult_t sendProxyProgress(struct ncclProxyState* proxyState, struct ncclProxyArgs* args);

// ============================================================================
// 函数: sendConnect - 建立发送端连接
// 参数:
//   comm - 通信上下文
//   connectInfos - 所有rank的连接信息数组
//   nranks - 总rank数
//   rank - 当前rank
//   send - 发送端连接器
// 返回: ncclResult_t - 操作结果状态码
// 说明: 调用proxy建立连接，获取内存映射并设置连接器指针
// ============================================================================
static ncclResult_t sendConnect(struct ncclComm* comm, struct ncclConnect* connectInfos, int nranks, int rank, struct ncclConnector* send) {
  // We're on the same process as the proxy. We can pass a pointer to a struct.
  // 我们与proxy在同一进程中，可以直接传递结构体指针
  struct collNetConnectArgs args = { rank, nranks, connectInfos };  // 构造连接参数
  struct connectMap* map;  // 连接映射指针
  // 阻塞调用proxy的connect函数，获取connectMap
  NCCLCHECK(ncclProxyCallBlocking(comm, &send->proxyConn, ncclProxyMsgConnect, &args, sizeof(struct collNetConnectArgs), &map, sizeof(struct connectMap*)));

  // If collnet connect failed, propagate error to fallback on regular p2p
  // 如果CollNet连接失败，传播错误以便回退到常规P2P传输
  if (map == NULL) return ncclSystemError;  // 返回系统错误，触发回退

  //NCCLCHECK(collNetDumpMap(map));  // 调试代码：打印内存映射（已注释）

  // 获取发送内存结构的GPU指针
  struct ncclSendMem *sendMem = (struct ncclSendMem*) NCCL_NET_MAP_GET_POINTER(map, gpu, sendMem);
  // 获取GDC内存（如果存在），否则使用sendMem中的head指针
  void* gdcMem = map->mems[NCCL_NET_MAP_GDCMEM].gpuPtr;
  send->conn.head = gdcMem ? (uint64_t*)gdcMem : &sendMem->head;  // 设置head指针

  // 获取接收内存结构的GPU指针
  struct ncclRecvMem *recvMem = (struct ncclRecvMem*) NCCL_NET_MAP_GET_POINTER(map, gpu, recvMem);
  send->conn.tail = &recvMem->tail;  // 设置tail指针（从接收端读取）
  send->conn.connFifo = recvMem->connFifo;  // 设置连接FIFO队列
  // 初始化所有连接FIFO槽位
  for (int i=0; i<NCCL_STEPS; i++) {
    send->conn.connFifo[i].size = -1;  // 初始大小设为-1（表示无效）
    send->conn.connFifo[i].mode = NCCL_MODE_OFFSET;  // 设置模式为偏移量模式
  }

  // 设置每种协议的缓冲区指针
  for (int p=0; p<NCCL_NUM_PROTOCOLS; p++)
    send->conn.buffs[p] = NCCL_NET_MAP_GET_POINTER(map, gpu, buffs[p]);

  // 设置proxy进度推进函数指针
  send->proxyConn.proxyProgress = sendProxyProgress;

  return ncclSuccess;
}

// 前向声明：proxy进度推进函数
static ncclResult_t recvProxyProgress(struct ncclProxyState* proxyState, struct ncclProxyArgs* args);

// ============================================================================
// 函数: recvConnect - 建立接收端连接
// 参数: 与sendConnect类似，但用于接收端
// 返回: ncclResult_t - 操作结果状态码
// ============================================================================
static ncclResult_t recvConnect(struct ncclComm* comm, struct ncclConnect* connectInfos, int nranks, int rank, struct ncclConnector* recv) {
  // We're on the same process as the proxy. We can pass a pointer to a struct.
  struct collNetConnectArgs args = { rank, nranks, connectInfos };
  struct connectMap* map;
  NCCLCHECK(ncclProxyCallBlocking(comm, &recv->proxyConn, ncclProxyMsgConnect, &args, sizeof(struct collNetConnectArgs), &map, sizeof(struct connectMap*)));

  // If collnet connect failed, propagate error to fallback on regular p2p
  if (map == NULL) return ncclSystemError;

  //NCCLCHECK(collNetDumpMap(map));  // 调试代码（已注释）

  // 获取发送内存结构的GPU指针
  struct ncclSendMem *sendMem = (struct ncclSendMem*) NCCL_NET_MAP_GET_POINTER(map, gpu, sendMem);
  recv->conn.head = &sendMem->head;  // 设置head指针（从发送端读取）

  // 获取接收内存结构的GPU指针
  struct ncclRecvMem *recvMem = (struct ncclRecvMem*) NCCL_NET_MAP_GET_POINTER(map, gpu, recvMem);
  // 获取GDC内存（如果存在），否则使用recvMem中的tail指针
  void* gdcMem = map->mems[NCCL_NET_MAP_GDCMEM].gpuPtr;
  recv->conn.tail = gdcMem ? (uint64_t*)gdcMem : &recvMem->tail;  // 设置tail指针
  recv->conn.connFifo = recvMem->connFifo;  // 设置连接FIFO队列
  // 初始化所有连接FIFO槽位
  for (int i=0; i<NCCL_STEPS; i++) {
    recv->conn.connFifo[i].mode = NCCL_MODE_OFFSET;  // 设置模式为偏移量模式
  }

  // 设置每种协议的缓冲区指针
  for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
    recv->conn.buffs[p] = NCCL_NET_MAP_GET_POINTER(map, gpu, buffs[p]);
  }

  // 设置proxy进度推进函数指针
  recv->proxyConn.proxyProgress = recvProxyProgress;

  return ncclSuccess;
}

// ============================================================================
// 函数: sendFree - 释放发送端连接器资源
// 参数: send - 发送端连接器指针
// 返回: ncclResult_t - 操作结果状态码
// 说明: CollNet的发送端资源在proxy端管理，此处无需额外操作
// ============================================================================
static ncclResult_t sendFree(struct ncclConnector* send) {
  return ncclSuccess;  // 直接返回成功（资源在proxy端管理）
}

// ============================================================================
// 函数: recvFree - 释放接收端连接器资源
// 参数: recv - 接收端连接器指针
// 返回: ncclResult_t - 操作结果状态码
// 说明: CollNet的接收端资源在proxy端管理，此处无需额外操作
// ============================================================================
static ncclResult_t recvFree(struct ncclConnector* recv) {
  return ncclSuccess;  // 直接返回成功（资源在proxy端管理）
}

// ============================================================================
// 函数: sendProxySetup - 在proxy线程中设置发送端资源
// 参数:
//   connection - proxy连接结构
//   proxyState - proxy状态
//   reqBuff - 请求缓冲区（包含setupReq）
//   reqSize - 请求大小
//   respBuff - 响应缓冲区（此函数不使用）
//   respSize - 响应大小
//   done - 完成标志（输出）
// 返回: ncclResult_t - 操作结果状态码
// ============================================================================
static ncclResult_t sendProxySetup(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState, void* reqBuff, int reqSize, void* respBuff, int respSize, int* done) {
  struct setupReq* req = (struct setupReq*)reqBuff;  // 解析请求
  if (reqSize != sizeof(struct setupReq)) return ncclInternalError;  // 验证请求大小

  struct sendResources* resources;  // 发送端资源指针
  NCCLCHECK(ncclCalloc(&resources, 1));  // 分配并初始化资源结构
  connection->transportResources = resources;  // 保存资源指针
  connection->shared = 1;  // 标记为共享连接

  // 保存网络设备和GDR配置
  resources->netDev = req->netDev;  // 网络设备ID
  resources->useGdr = req->useGdr;  // GDR模式
  ncclNetProperties_t props;  // 网络属性结构
  NCCLCHECK(proxyState->ncclCollNet->getProperties(req->netDev, &props));  // 获取网络属性
  connection->collNet = req->collNet;  // 保存CollNet共享资源指针
  /* DMA-BUF support */
  // DMA-BUF支持：检查是否同时满足GDR、DMA-BUF支持和网络设备的DMA-BUF指针支持
  resources->useDmaBuf = resources->useGdr && proxyState->dmaBufSupport && (props.ptrSupport & NCCL_PTR_DMABUF);
  /* collective size limits*/
  // 集合操作大小限制：获取单次CollNet操作支持的最大字节数
  resources->maxCollBytes = props.maxCollBytes;
  // 验证maxCollBytes的合法性
  if((resources->maxCollBytes <= 0) || (resources->maxCollBytes > NCCL_MAX_NET_SIZE_BYTES)) {
    WARN("sendProxySetup: collnet plugin returned invalid value for maxCollBytes %ld \
      [allowed range: %ld - %ld] \n", resources->maxCollBytes, 0L, NCCL_MAX_NET_SIZE_BYTES);
    return ncclInternalError;  // 返回内部错误
  }
  return ncclSuccess;
}

// ============================================================================
// 结构: sharedResources - CollNet共享资源，在所有连接之间共享
// ============================================================================
struct sharedResources {
  void* collNetListenComms[MAXCHANNELS];  // 每个通道的监听通信句柄数组
  void* collNetComms[MAXCHANNELS];        // 每个通道的CollNet通信句柄数组
  int commRefCount[NCCL_MAX_NETDEVS];     // 每个网络设备的通信引用计数
};

// ============================================================================
// 函数: sharedListen - 启动CollNet监听，准备接收连接
// 参数:
//   proxyState - proxy状态
//   netDev - 网络设备ID
//   collNet - CollNet共享资源
//   collNetHandle - 输出的CollNet句柄
// 返回: ncclResult_t - 操作结果状态码
// ============================================================================
static ncclResult_t sharedListen(struct ncclProxyState* proxyState, int netDev, struct ncclCollNetSharedRes* collNet, void* collNetHandle) {
  struct sharedResources* resources = (struct sharedResources*)collNet->resources;  // 获取共享资源
  if (resources == NULL) {  // 如果尚未初始化
    NCCLCHECK(ncclCalloc(&resources, 1));  // 分配共享资源结构
    collNet->resources = resources;  // 保存到CollNet共享资源中
  }
  // 如果该网络设备的监听通信尚未建立，则创建监听通信
  if (resources->collNetComms[netDev] == NULL)
    NCCLCHECK(proxyState->ncclCollNet->listen(proxyState->collNetContext, netDev, collNetHandle, resources->collNetListenComms + netDev));
  return ncclSuccess;
}

// ============================================================================
// 函数: sharedConnect - 建立CollNet连接
// 参数:
//   proxyState - proxy状态
//   netDev - 网络设备ID
//   connectInfos - 所有rank的连接信息
//   nranks - 总rank数
//   rank - 当前rank
//   collNet - CollNet共享资源
//   collNetComm - 输出的CollNet通信句柄
// 返回: ncclResult_t - 操作结果状态码
// ============================================================================
static ncclResult_t sharedConnect(struct ncclProxyState* proxyState, int netDev, struct ncclConnect* connectInfos, int nranks, int rank, struct ncclCollNetSharedRes* collNet, void** collNetComm) {
  struct sharedResources* resources = (struct sharedResources*)collNet->resources;  // 获取共享资源
  if (resources->collNetComms[netDev] == NULL) {  // 如果尚未建立连接
    // Connect to coll comm
    // 连接到CollNet通信域
    collNetHandle_t** handlePtrs = NULL;  // 句柄指针数组
    NCCLCHECK(ncclCalloc(&handlePtrs, nranks));  // 分配句柄指针数组
    // 提取所有rank的CollNet句柄
    for (int i = 0; i < nranks; i++) {
      struct collNetRecvConnectInfo* info = (struct collNetRecvConnectInfo*)(connectInfos+i);
      handlePtrs[i] = &(info->collNetHandle);  // 保存句柄指针
    }
    // 调用CollNet插件的connect函数建立集合通信域
    ncclResult_t ret = proxyState->ncclCollNet->connect((void**)handlePtrs, nranks, rank,
          resources->collNetListenComms[netDev],  // 监听通信句柄
          resources->collNetComms+netDev);        // 输出：通信句柄
    free(handlePtrs);  // 释放句柄指针数组
    if (ret == ncclSuccess) {  // 如果连接成功
      // Close listen comm
      // 关闭监听通信（不再需要）
      NCCLCHECK(proxyState->ncclCollNet->closeListen(resources->collNetListenComms[netDev]));
    } else {
      resources->collNetListenComms[netDev] = NULL;  // 连接失败，清空监听句柄
    }
  }
  *collNetComm = resources->collNetComms[netDev];  // 返回通信句柄
  if (*collNetComm) resources->commRefCount[netDev]++;  // 增加引用计数
  return ncclSuccess;
}

// ============================================================================
// 函数: sharedFree - 释放CollNet共享资源
// 参数:
//   proxyState - proxy状态
//   collNet - CollNet共享资源
//   netDev - 网络设备ID
// 返回: ncclResult_t - 操作结果状态码
// ============================================================================
static ncclResult_t sharedFree(struct ncclProxyState* proxyState, struct ncclCollNetSharedRes* collNet, int netDev) {
  struct sharedResources* resources = (struct sharedResources*)collNet->resources;  // 获取共享资源
  resources->commRefCount[netDev]--;  // 减少引用计数
  if (resources->commRefCount[netDev] == 0) {  // 如果引用计数归零
    NCCLCHECK(proxyState->ncclCollNet->closeColl(resources->collNetComms[netDev]));  // 关闭CollNet通信
  }
  // 检查是否所有网络设备的引用计数都归零
  for (int n=0; n<NCCL_MAX_NETDEVS; n++) if (resources->commRefCount[n]) return ncclSuccess;
  collNet->resources = NULL;  // 清空共享资源指针
  free(resources);  // 释放共享资源结构
  return ncclSuccess;
}

// ============================================================================
// 函数: sharedBuffersInit - 初始化共享缓冲区
// 参数:
//   collNet - CollNet共享资源
//   cuda - 是否为CUDA内存（1=设备内存，0=主机内存）
//   gpuPtr - 输出的GPU指针
//   cpuPtr - 输出的CPU指针
//   size - 输出的缓冲区大小
// 返回: ncclResult_t - 操作结果状态码
// 说明: 为所有通道分配共享缓冲区，用于发送和接收数据
// ============================================================================
static ncclResult_t sharedBuffersInit(struct ncclCollNetSharedRes* collNet, int cuda, char** gpuPtr, char** cpuPtr, int* size) {
  if (collNet->size == 0) {  // 如果缓冲区大小尚未计算
    // 计算总大小：通道数 * 2（发送/接收）* 缓冲区大小
    collNet->size = 2 * collNet->nChannels * collNet->buffSize;
  }

  *size = collNet->size;  // 返回缓冲区大小

  if (cuda && collNet->cudaBuff == NULL) {  // 如果需要CUDA内存且尚未分配
    NCCLCHECK(ncclCudaCalloc(&collNet->cudaBuff, *size));  // 分配CUDA设备内存
    cudaMemset(collNet->cudaBuff, 0x33, *size/2);  // 前半部分填充0x33模式（用于调试）
    cudaMemset((char*)collNet->cudaBuff + *size/2, 0x66, *size/2);  // 后半部分填充0x66模式
  }
  if (!cuda && collNet->hostBuff == NULL) {  // 如果需要主机内存且尚未分配
    NCCLCHECK(ncclCudaHostCalloc(&collNet->hostBuff, *size));  // 分配主机锁定内存
  }
  *gpuPtr = *cpuPtr = cuda ? collNet->cudaBuff : collNet->hostBuff;  // 设置指针（CUDA内存时GPU=CPU指针相同）
  return ncclSuccess;
}

// ============================================================================
// 函数: sharedBuffersGet - 获取共享缓冲区中特定槽位的偏移量
// 参数:
//   collNet - CollNet共享资源
//   type - 类型（0=发送，1=接收）
//   slot - 步骤槽位（0到NCCL_STEPS-1）
//   channel - 通道索引
//   offset - 输出的偏移量
// 返回: ncclResult_t - 操作结果状态码
// 说明: 为不同通道和发送/接收类型分配独立的缓冲区池
// ============================================================================
static ncclResult_t sharedBuffersGet(struct ncclCollNetSharedRes* collNet, int type, int slot, int channel, int* offset) {
  // Use different pools for different channels and also separate send/recv.
  // 为不同通道使用不同的池，并分离发送/接收
  int slotSize = collNet->buffSize / NCCL_STEPS;  // 计算每个槽位的大小
  // 计算全局槽位索引：(类型*步骤数+槽位)*通道数+通道
  int globalSlot = (type * NCCL_STEPS + slot) * collNet->nChannels + channel;
  *offset = slotSize * globalSlot;  // 计算字节偏移量
  return ncclSuccess;
}

// ============================================================================
// 函数: sharedBuffersDestroy - 销毁共享缓冲区
// 参数:
//   collNet - CollNet共享资源
// 返回: ncclResult_t - 操作结果状态码
// 说明: 释放设备内存和主机内存，使用size==0作为标志确保只释放一次
// ============================================================================
static ncclResult_t sharedBuffersDestroy(struct ncclCollNetSharedRes* collNet) {
  if (collNet->size == 0) return ncclSuccess;  // 如果已经释放，直接返回
  NCCLCHECK(ncclCudaFree(collNet->cudaBuff));  // 释放CUDA设备内存
  NCCLCHECK(ncclCudaHostFree(collNet->hostBuff));  // 释放主机锁定内存
  // This will be called multiple times, with multiple channels and send/recv. Make sure we only do it once.
  // 这会被多次调用（多个通道和发送/接收），确保只执行一次
  collNet->size = 0;  // 设置标志表示已释放
  return ncclSuccess;
}

// ============================================================================
// 函数: recvProxySetup - 在proxy线程中设置接收端资源
// 参数: 与sendProxySetup类似
// 返回: ncclResult_t - 操作结果状态码
// 说明: 接收端需要额外的needFlush标志，并返回CollNet句柄供发送端连接
// ============================================================================
static ncclResult_t recvProxySetup(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState, void* reqBuff, int reqSize, void* respBuff, int respSize, int* done) {
  struct setupReq* req = (struct setupReq*)reqBuff;  // 解析请求
  if (reqSize != sizeof (struct setupReq)) return ncclInternalError;  // 验证请求大小

  struct recvResources* resources;  // 接收端资源指针
  NCCLCHECK(ncclCalloc(&resources, 1));  // 分配并初始化资源结构
  connection->transportResources = resources;  // 保存资源指针
  connection->shared = 1;  // 标记为共享连接

  // 保存配置
  resources->netDev = req->netDev;  // 网络设备ID
  resources->useGdr = req->useGdr;  // GDR模式
  resources->needFlush = req->needFlush;  // 是否需要刷新
  ncclNetProperties_t props;  // 网络属性结构
  NCCLCHECK(proxyState->ncclCollNet->getProperties(req->netDev, &props));  // 获取网络属性
  connection->collNet = req->collNet;  // 保存CollNet共享资源指针
  /* DMA-BUF support */
  resources->useDmaBuf = resources->useGdr && proxyState->dmaBufSupport && (props.ptrSupport & NCCL_PTR_DMABUF);
  resources->maxCollBytes = props.maxCollBytes;  // 保存最大集合操作字节数
  // 验证maxCollBytes的合法性
  if((resources->maxCollBytes <= 0) || (resources->maxCollBytes > NCCL_MAX_NET_SIZE_BYTES)) {
    WARN("sendProxySetup: collnet plugin returned invalid value for maxCollBytes %ld \
      [allowed range: %ld - %ld] \n", resources->maxCollBytes, 0L, NCCL_MAX_NET_SIZE_BYTES);
    return ncclInternalError;
  }

  // 返回CollNet句柄给发送端
  collNetHandle_t* netHandle = (collNetHandle_t*) respBuff;  // 类型转换
  if (respSize != sizeof(collNetHandle_t)) return ncclInternalError;  // 验证响应缓冲区大小

  NCCLCHECK(sharedListen(proxyState, req->netDev, req->collNet, netHandle));  // 启动监听，生成句柄
  return ncclSuccess;
}

// ============================================================================
// 函数: sendProxyConnect - 在proxy线程中建立发送端连接
// 参数:
//   connection - proxy连接结构
//   proxyState - proxy状态
//   reqBuff - 请求缓冲区（包含collNetConnectArgs）
//   reqSize - 请求大小
//   respBuff - 响应缓冲区（返回connectMap指针）
//   respSize - 响应大小
//   done - 完成标志（输出）
// 返回: ncclResult_t - 操作结果状态码
// 说明: 分配内存、注册内存区域，并建立内存映射
// ============================================================================
static ncclResult_t sendProxyConnect(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState, void* reqBuff, int reqSize, void* respBuff, int respSize, int* done) {
  ncclResult_t ret = ncclSuccess;  // 初始化返回值
  // 验证请求大小
  if (reqSize != sizeof(struct collNetConnectArgs)) { WARN("sendProxyConnect: reqSize is %d != %ld", reqSize, sizeof(struct collNetConnectArgs)); return ncclInternalError; }
  struct collNetConnectArgs* args = (struct collNetConnectArgs*)reqBuff;  // 解析连接参数
  // 编译期断言：确保发送端连接信息大小不超过ncclConnect结构体
  static_assert(sizeof(collNetSendConnectInfo) <= sizeof(struct ncclConnect), "Collnet Send Connect info is too big");
  // 获取当前rank对应的发送端连接信息（从接收端传回）
  struct collNetSendConnectInfo* info = (struct collNetSendConnectInfo*)(args->connectInfos+args->rank);

  struct sendResources* resources = (struct sendResources*)(connection->transportResources);  // 获取发送端资源

  // Get info from recv side
  // 从接收端获取信息
  resources->collNetRank = args->rank;  // 保存CollNet rank
  resources->reqFifo = (struct reqSlot (*)[NCCL_STEPS])(info->reqFifo);  // 保存请求FIFO指针

  // 获取接收端的内存句柄（用于接收数据）
  for (int p=0; p<NCCL_NUM_PROTOCOLS; p++)
    resources->recvMhandles[p] = info->mhandles[p];

  // 建立CollNet连接
  NCCLCHECK(sharedConnect(proxyState, resources->netDev, args->connectInfos, args->nranks, args->rank, connection->collNet, &resources->collNetComm));

  // Collnet connect is allowed to fail. Gracefully handle that case by returning NULL to the caller.
  // CollNet连接允许失败。通过返回NULL给调用者来优雅处理这种情况。
  if (respSize != sizeof(struct connectMap*)) {
    WARN("sendProxyConnect: respSize is %d != %ld", respSize, sizeof(void*));
    return ncclInternalError;
  }
  if (resources->collNetComm == NULL) {  // 如果连接失败
    *((struct connectMap**)respBuff) = NULL;  // 返回NULL
    return ncclSuccess;
  }
  // 设置proxy追加指针（用于异步操作队列）
  connection->proxyAppendPtr = connection->collNet->proxyAppend + 2 * resources->netDev;

  struct connectMap* map = &resources->map;  // 获取内存映射结构

  // 添加发送内存和接收内存到映射（非共享、主机内存）
  NCCL_NET_MAP_ADD_POINTER(map, 0, 0, sizeof(struct ncclSendMem), sendMem);
  NCCL_NET_MAP_ADD_POINTER(map, 0, 0, sizeof(struct ncclRecvMem), recvMem);

  // 分配主机内存（sendMem和recvMem）
  NCCLCHECK(ncclCudaHostCalloc(&map->mems[NCCL_NET_MAP_HOSTMEM].cpuPtr, map->mems[NCCL_NET_MAP_HOSTMEM].size));
  map->mems[NCCL_NET_MAP_HOSTMEM].gpuPtr = map->mems[NCCL_NET_MAP_HOSTMEM].cpuPtr;  // GPU指针=CPU指针（主机内存）
  // 如果启用GDR Copy同步，分配GDC同步内存
  if (ncclGdrCopy && ncclParamGdrCopySyncEnable()) {
    uint64_t *cpuPtr, *gpuPtr;
    NCCLCHECK(ncclGdrCudaCalloc(&cpuPtr, &gpuPtr, 1, &resources->gdrDesc));  // 分配GDR内存

    resources->gdcSync = cpuPtr;  // 保存CPU指针
    struct connectMapMem* gdcMem = map->mems+NCCL_NET_MAP_GDCMEM;  // 获取GDC内存结构
    gdcMem->cpuPtr = (char*)cpuPtr;  // 保存CPU指针
    gdcMem->gpuPtr = (char*)gpuPtr;  // 保存GPU指针
    gdcMem->size = sizeof(uint64_t); // sendMem->head（head指针的大小）
  }

  // 获取sendMem和recvMem的CPU指针
  resources->sendMem = (struct ncclSendMem*) NCCL_NET_MAP_GET_POINTER(map, cpu, sendMem);
  resources->recvMem = (struct ncclRecvMem*) NCCL_NET_MAP_GET_POINTER(map, cpu, recvMem);
  // Don't give credits yet in shared mode.
  // 在共享模式下暂不提供credit（初始化为负值，表示无可用槽位）
  (resources->gdcSync ? *resources->gdcSync : resources->sendMem->head) = -NCCL_STEPS;

  // Allocate & Register shared buffers for the Simple protocol
  // 为Simple协议分配并注册共享缓冲区
  int bank = resources->useGdr ? NCCL_NET_MAP_SHARED_DEVMEM : NCCL_NET_MAP_SHARED_HOSTMEM;  // 选择内存bank
  struct connectMapMem* mapMem = map->mems+bank;  // 获取内存结构
  NCCLCHECK(sharedBuffersInit(connection->collNet, resources->useGdr, &mapMem->gpuPtr, &mapMem->cpuPtr, &mapMem->size));  // 初始化共享缓冲区
  NCCL_NET_MAP_ADD_POINTER(map, 1, resources->useGdr ? 1 : 0, mapMem->size, buffs[NCCL_PROTO_SIMPLE]);  // 添加到映射

  int dmabuf_fd = -1;  // DMA-BUF文件描述符
#if CUDA_VERSION >= 11070  // CUDA 11.7及以上版本支持DMA-BUF
  /* DMA-BUF support */
  if (resources->useGdr && resources->useDmaBuf) {  // 如果使用DMA-BUF
    // 获取DMA-BUF文件描述符
    CUCHECK(cuMemGetHandleForAddressRange((void *)&dmabuf_fd, (CUdeviceptr)mapMem->cpuPtr, mapMem->size, CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, getHandleForAddressRangeFlags(resources->useGdr)));
    // 使用DMA-BUF注册内存
    NCCLCHECKGOTO(proxyState->ncclCollNet->regMrDmaBuf(resources->collNetComm, mapMem->cpuPtr, mapMem->size,
                                                       NCCL_PTR_CUDA, 0ULL, dmabuf_fd,
                                                       &resources->sendMhandles[NCCL_PROTO_SIMPLE]),
                  ret, fail);
    (void)close(dmabuf_fd);  // 关闭文件描述符
  } else // FALL-THROUGH to nv_peermem GDR path
  // 否则使用常规的peer memory GDR路径
#endif
  {
    // 常规内存注册
    NCCLCHECK(proxyState->ncclCollNet->regMr(resources->collNetComm, mapMem->cpuPtr, mapMem->size,
                                            resources->useGdr ? NCCL_PTR_CUDA : NCCL_PTR_HOST,
                                            &resources->sendMhandles[NCCL_PROTO_SIMPLE]));
  }

  *((struct connectMap**)respBuff) = &resources->map;  // 返回内存映射

exit:
  return ret;  // 返回结果
fail:
  if (dmabuf_fd != -1) {  // 清理：关闭DMA-BUF文件描述符
    (void)close(dmabuf_fd);
  }
  goto exit;  // 跳转到退出
}

// ============================================================================
// 函数: recvProxyConnect - 在proxy线程中建立接收端连接
// 参数: 与sendProxyConnect类似
// 返回: ncclResult_t - 操作结果状态码
// 说明: 与sendProxyConnect类似，但接收端需要gdcFlush支持
// ============================================================================
static ncclResult_t recvProxyConnect(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState, void* reqBuff, int reqSize, void* respBuff, int respSize, int* done) {
  ncclResult_t ret = ncclSuccess;  // 初始化返回值
  // 验证请求大小
  if (reqSize != sizeof(struct collNetConnectArgs)) { WARN("recvProxyConnect: reqSize is %d != %ld", reqSize, sizeof(struct collNetConnectArgs)); return ncclInternalError; }
  struct collNetConnectArgs* args = (struct collNetConnectArgs*)reqBuff;  // 解析连接参数

  struct recvResources* resources = (struct recvResources*)(connection->transportResources);  // 获取接收端资源
  struct collNetSendConnectInfo* info = (struct collNetSendConnectInfo*)(args->connectInfos+args->rank);  // 获取发送端连接信息
  resources->collNetRank = args->rank;  // 保存CollNet rank

  // 建立CollNet连接
  NCCLCHECK(sharedConnect(proxyState, resources->netDev, args->connectInfos, args->nranks, args->rank, connection->collNet, &resources->collNetComm));

  // Collnet connect is allowed to fail. Gracefully handle that case by returning NULL to the caller.
  // CollNet连接允许失败
  if (respSize != sizeof(struct connectMap*)) { WARN("sendProxyConnect: respSize is %d != %ld", respSize, sizeof(void*)); return ncclInternalError; }
  if (resources->collNetComm == NULL) {  // 如果连接失败
    *((struct connectMap**)respBuff) = NULL;  // 返回NULL
    return ncclSuccess;
  }
  // 设置proxy追加指针（接收端为netDev*2+1）
  connection->proxyAppendPtr = connection->collNet->proxyAppend + 2 * resources->netDev + 1;

  struct connectMap* map = &resources->map;  // 获取内存映射结构

  // 添加发送内存和接收内存到映射
  NCCL_NET_MAP_ADD_POINTER(map, 0, 0, sizeof(struct ncclSendMem), sendMem);
  NCCL_NET_MAP_ADD_POINTER(map, 0, 0, sizeof(struct ncclRecvMem), recvMem);

  // 分配主机内存
  NCCLCHECK(ncclCudaHostCalloc(&map->mems[NCCL_NET_MAP_HOSTMEM].cpuPtr, map->mems[NCCL_NET_MAP_HOSTMEM].size));
  map->mems[NCCL_NET_MAP_HOSTMEM].gpuPtr = map->mems[NCCL_NET_MAP_HOSTMEM].cpuPtr;
  // 接收端需要GDR Copy支持（同步和刷新）
  if (ncclGdrCopy) {
    uint64_t *cpuPtr, *gpuPtr;
    NCCLCHECK(ncclGdrCudaCalloc(&cpuPtr, &gpuPtr, 2, &resources->gdrDesc));  // 分配2个uint64（sync和flush）

    // 如果启用GDR Copy同步，设置gdcSync
    if (ncclParamGdrCopySyncEnable()) {
      resources->gdcSync = cpuPtr;  // 第一个uint64用于同步
      struct connectMapMem* gdcMem = map->mems+NCCL_NET_MAP_GDCMEM;
      gdcMem->cpuPtr = (char*)cpuPtr;
      gdcMem->gpuPtr = (char*)gpuPtr;
      gdcMem->size = sizeof(uint64_t);
    }
    // 如果启用GDR Copy刷新，设置gdcFlush
    if (ncclParamGdrCopyFlushEnable()) resources->gdcFlush = cpuPtr + 1;  // 第二个uint64用于刷新
  }

  // 获取sendMem和recvMem的CPU指针
  resources->sendMem = (struct ncclSendMem*) NCCL_NET_MAP_GET_POINTER(map, cpu, sendMem);
  resources->recvMem = (struct ncclRecvMem*) NCCL_NET_MAP_GET_POINTER(map, cpu, recvMem);

  // Allocate & Register shared buffers for the Simple protocol
  int bank = resources->useGdr ? NCCL_NET_MAP_SHARED_DEVMEM : NCCL_NET_MAP_SHARED_HOSTMEM;
  struct connectMapMem* mapMem = map->mems+bank;
  NCCLCHECK(sharedBuffersInit(connection->collNet, resources->useGdr, &mapMem->gpuPtr, &mapMem->cpuPtr, &mapMem->size));
  NCCL_NET_MAP_ADD_POINTER(map, 1, resources->useGdr ? 1 : 0, mapMem->size, buffs[NCCL_PROTO_SIMPLE]);

  int dmabuf_fd = -1;
#if CUDA_VERSION >= 11070
  /* DMA-BUF support */
  if (resources->useGdr && resources->useDmaBuf) {
    CUCHECK(cuMemGetHandleForAddressRange((void *)&dmabuf_fd, (CUdeviceptr)mapMem->cpuPtr, mapMem->size, CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, getHandleForAddressRangeFlags(resources->useGdr)));
    NCCLCHECKGOTO(proxyState->ncclCollNet->regMrDmaBuf(resources->collNetComm, mapMem->cpuPtr, mapMem->size,
                                                       NCCL_PTR_CUDA, 0ULL, dmabuf_fd,
                                                       &resources->mhandles[NCCL_PROTO_SIMPLE]),
                  ret, fail);
    (void)close(dmabuf_fd);
  } else // FALL-THROUGH to nv_peermem GDR path
#endif
  {
    NCCLCHECK(proxyState->ncclCollNet->regMr(resources->collNetComm, mapMem->cpuPtr, mapMem->size,
                                            resources->useGdr ? NCCL_PTR_CUDA : NCCL_PTR_HOST,
                                            &resources->mhandles[NCCL_PROTO_SIMPLE]));
  }

  // Pass info to send side
  // 传递信息给发送端
  info->reqFifo = resources->reqFifo;  // 传递请求FIFO指针
  for (int p=0; p<NCCL_NUM_PROTOCOLS; p++)
    info->mhandles[p] = resources->mhandles[p];  // 传递内存句柄

  if (respSize != sizeof(struct connectMap*)) { WARN("recvProxyConnect: respSize is %d != %ld", respSize, sizeof(void*)); return ncclInternalError; }
  *((struct connectMap**)respBuff) = &resources->map;  // 返回内存映射

exit:
  return ret;
fail:
  if (dmabuf_fd != -1) {  // 清理：关闭DMA-BUF文件描述符
    (void)close(dmabuf_fd);
  }
  goto exit;
}

// ============================================================================
// 函数: sendProxyFree - 在proxy线程中释放发送端资源
// 参数:
//   connection - proxy连接结构
//   proxyState - proxy状态
// 返回: ncclResult_t - 操作结果状态码
// ============================================================================
static ncclResult_t sendProxyFree(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState) {
  struct sendResources* resources = (struct sendResources*)(connection->transportResources);  // 获取资源

  if (resources) {  // 如果资源存在
    // 注销所有发送内存句柄
    for (int p = 0; p < NCCL_NUM_PROTOCOLS; p++) {
      if (resources->sendMhandles[p]) {
        NCCLCHECK(proxyState->ncclCollNet->deregMr(resources->collNetComm, resources->sendMhandles[p]));
      }
    }
    // 释放所有内存
    struct connectMapMem* mems = resources->map.mems;
    NCCLCHECK(ncclCudaHostFree(mems[NCCL_NET_MAP_HOSTMEM].cpuPtr));  // 释放主机内存
    NCCLCHECK(ncclCudaFree(mems[NCCL_NET_MAP_DEVMEM].cpuPtr));  // 释放设备内存
    if (mems[NCCL_NET_MAP_GDCMEM].cpuPtr) NCCLCHECK(ncclGdrCudaFree(resources->gdrDesc));  // 释放GDR内存
    NCCLCHECK(sharedBuffersDestroy(connection->collNet));  // 销毁共享缓冲区
    NCCLCHECK(sharedFree(proxyState, connection->collNet, resources->netDev));  // 释放共享资源
    // 如果引用计数归零，释放CollNet共享资源
    if (ncclAtomicRefCountDecrement(&connection->collNet->refCount) == 0) free(connection->collNet);
    free(connection->transportResources);  // 释放资源结构
  }
  return ncclSuccess;
}

// ============================================================================
// 函数: recvProxyFree - 在proxy线程中释放接收端资源
// 参数: 与sendProxyFree类似
// 返回: ncclResult_t - 操作结果状态码
// ============================================================================
static ncclResult_t recvProxyFree(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState) {
  struct recvResources* resources = (struct recvResources*)(connection->transportResources);  // 获取资源

  if (resources) {  // 如果资源存在
    // 注销所有内存句柄
    for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
      if (resources->mhandles[p]) {
        NCCLCHECK(proxyState->ncclCollNet->deregMr(resources->collNetComm, resources->mhandles[p]));
      }
    }
    // 释放所有内存
    struct connectMapMem* mems = resources->map.mems;
    NCCLCHECK(ncclCudaHostFree(mems[NCCL_NET_MAP_HOSTMEM].cpuPtr));  // 释放主机内存
    NCCLCHECK(ncclCudaFree(mems[NCCL_NET_MAP_DEVMEM].cpuPtr));  // 释放设备内存
    if (mems[NCCL_NET_MAP_GDCMEM].cpuPtr) NCCLCHECK(ncclGdrCudaFree(resources->gdrDesc));  // 释放GDR内存
    NCCLCHECK(sharedBuffersDestroy(connection->collNet));  // 销毁共享缓冲区
    NCCLCHECK(sharedFree(proxyState, connection->collNet, resources->netDev));  // 释放共享资源
    // 如果引用计数归零，释放CollNet共享资源
    if (ncclAtomicRefCountDecrement(&connection->collNet->refCount) == 0) free(connection->collNet);
    free(connection->transportResources);  // 释放资源结构
  }
  return ncclSuccess;
}

// ============================================================================
// 函数: calcAlgoOffset - 计算算法偏移量（用于AllGather和ReduceScatter）
// 参数:
//   args - proxy参数
//   isAllNotOne - 是否为All（1）而非One（0）
//   sub - 子操作索引
//   step - 步骤号
// 返回: size_t - 计算出的偏移量
// ============================================================================
static size_t calcAlgoOffset(struct ncclProxyArgs* args, int isAllNotOne, int sub, uint64_t step) {
  int chunkSize = args->chunkSize;  // 块大小
  int nNodes = args->specifics.collnetDirect.nNodes;  // 节点数
  int node = args->specifics.collnetDirect.node;  // 当前节点
  size_t sizePerRank = args->specifics.collnetDirect.sizePerRank;  // 每个rank的大小
  // 基础偏移量：(步骤*子操作数+子操作)*块大小
  size_t offset = (step*(args->nsubs) + sub)*chunkSize;
  if (isAllNotOne) {  // 如果是"All"操作（接收全部数据）
    offset = std::min<size_t>(offset, nNodes*sizePerRank);  // 限制在总大小范围内
  } else {  // 如果是"One"操作（只接收自己的那部分）
    offset = std::max<size_t>(offset, (node+0)*sizePerRank);  // 下限：当前节点的起始位置
    offset = std::min<size_t>(offset, (node+1)*sizePerRank);  // 上限：当前节点的结束位置
  }
  return offset;  // 返回计算出的偏移量
}

// ============================================================================
// 函数: calcRegionOffset - 计算共享缓冲区中的区域偏移量
// 参数:
//   args - proxy参数
//   isRecvNotSend - 是否为接收而非发送（1=接收，0=发送）
//   sub - 子操作索引
//   step - 步骤号
//   side - 区域边界（0=开始，1=结束）
// 返回: ssize_t - 计算出的区域偏移量
// ============================================================================
static ssize_t calcRegionOffset(
    struct ncclProxyArgs* args, int isRecvNotSend, int sub, uint64_t step,
    int side // 0=begin, 1=end（0=开始位置，1=结束位置）
  ) {
  struct ncclCollNetSharedRes* collNet = args->subs[0].connection->collNet;  // 获取CollNet共享资源
  ssize_t slotSize = collNet->buffSize/NCCL_STEPS;  // 计算每个槽位的大小
  ssize_t chunkSize = args->chunkSize;  // 块大小
  // 计算基础偏移量：（接收标志*步骤数+步骤号）*通道数*槽位大小
  ssize_t base = isRecvNotSend*NCCL_STEPS + (step%NCCL_STEPS);
  base *= collNet->nChannels*slotSize;
  if (args->coll == ncclFuncAllReduce) {  // 如果是AllReduce操作
    return base + (sub+side)*chunkSize;  // 简单计算：基础+（子操作+边界）*块大小
  } else {  // AllGather或ReduceScatter操作
    int isAllNotOne = isRecvNotSend ^ (args->coll == ncclFuncReduceScatter);  // 计算是否为All模式
    int sub0 = sub - (sub%COLLNET_GROUP_NSUBS);  // 获取组内第一个子操作
    size_t off = sub0*slotSize;  // 组内起始偏移
    // 加上算法偏移量的差值
    off += calcAlgoOffset(args, isAllNotOne, sub+side, step)
         - calcAlgoOffset(args, isAllNotOne, sub0, step);
    return base + off;  // 返回总偏移量
  }
}

// ============================================================================
// 宏: LAST_OF_GROUP - 检查是否为组的最后一个子操作
// 参数:
//   args - proxy参数
//   s - 子操作索引
// 返回: 如果是组最后一个或全部最后一个则返回true
// ============================================================================
#define LAST_OF_GROUP(args, s) \
  ((s)%COLLNET_GROUP_NSUBS == COLLNET_GROUP_NSUBS-1 || (s) == (args)->nsubs-1)

// ============================================================================
// 函数: calcStepsPerGroup - 计算每组步骤数
// 参数: nGroups - 组数
// 返回: int - 每组步骤数
// ============================================================================
static constexpr int calcStepsPerGroup(int nGroups) {
  //return NCCL_STEPS/nGroups;  // 可选：按比例分配
  return NCCL_STEPS;  // 当前实现：每组使用全部步骤
}

// ============================================================================
// 函数: collNetRegIallreduce - 注册缓冲区的IallReduce操作
// 参数:
//   proxyState - proxy状态
//   resources - 发送端资源
//   args - proxy参数
//   sub - 子操作参数
//   groupStart - 组内起始索引
//   nBytesInOut - 输入输出字节数
//   request - 输出的请求句柄
// 返回: ncclResult_t - 操作结果状态码
// 说明: 使用用户注册的缓冲区进行IallReduce
// ============================================================================
static ncclResult_t collNetRegIallreduce(struct ncclProxyState* proxyState, struct sendResources *resources, struct ncclProxyArgs *args, struct ncclProxySubArgs *sub, int groupStart, ssize_t *nBytesInOut, void **request) {
  ssize_t loopSize, winOffset, nBytes;  // 循环大小、窗口偏移、字节数
  ssize_t eltSize = ncclTypeSize((ncclDataType_t)args->dtype);  // 元素大小
  // for UB iallreduce 1RPN case, user's send and recv buffers are both directly accessed by collnet network.
  // 对于用户缓冲区1RPN的情况，发送和接收缓冲区都被CollNet网络直接访问。
  // we can just issue maximal collnet bytes by resources->maxCollBytes for each iallreduce.
  // 我们可以每次发送maxCollBytes的最大字节数。
  // for multi-RPN case, we have to consider pipeline, so each time we only send groupSize * chunkSize (i.e., nBytesInOut)
  // 对于多RPN情况，需要考虑流水线，每次只发送groupSize*chunkSize
  // sub->loopOffset is data offset to the buffer for this head rank in each loop
  // sub->loopOffset是每个循环中此head rank的数据偏移
  // winOffset is used to find actual offset from send and recv buffer for this iallreduce
  // winOffset用于查找本次iallreduce的发送和接收缓冲区实际偏移
  // loopSize is all bytes sent by all channels and head ranks in each loop.
  // loopSize是每个循环中所有通道和head rank发送的总字节数。
  // send and recv mem handle are retrieved from sub in which user buffer mem handles are stored.
  // 发送和接收内存句柄从sub中获取，用户缓冲区内存句柄存储在其中。
  if (sub->isOneRPN) {  // 如果是单RPN
    winOffset = 0;  // 窗口偏移为0
    nBytes = std::min((size_t)sub->nbytes, resources->maxCollBytes);  // 限制在maxCollBytes范围内
    loopSize = nBytes;  // 循环大小等于字节数
  } else {  // 多RPN情况
    winOffset = sub->loopOffset + groupStart * args->chunkSize;  // 计算窗口偏移
    nBytes = std::min(sub->nbytes - winOffset, *nBytesInOut);  // 计算本次操作字节数
    loopSize = sub->loopSize;  // 使用预设的循环大小
  }

  if (nBytes > 0) {  // 如果有数据需要处理
    // 调用CollNet插件的iallreduce函数
    NCCLCHECK(proxyState->ncclCollNet->iallreduce(resources->collNetComm, sub->sendbuff + winOffset, sub->recvbuff + winOffset, nBytes / eltSize, (ncclDataType_t)args->dtype, (ncclRedOp_t)args->redOp, sub->sendMhandle, sub->recvMhandle, request));
    if (*request) {  // 如果请求成功提交
      // if issued successfully, we need to move the pointer forward and reduce the existing nbytes.
      // 如果成功提交，需要前移指针并减少剩余字节数
      sub->nbytes -= loopSize;  // 减少剩余字节数
      sub->sendbuff += loopSize;  // 前移发送缓冲区指针
      sub->recvbuff += loopSize;  // 前移接收缓冲区指针
      TRACE(NCCL_NET, "sendProxy [%ld/%d/%d] registered Iallreduce posted sendbuff %p recvbuff %p size %ld loopSize %ld winOffset %ld isOneRPN %d req %p", (long)sub->transmitted, sub->nsteps, groupStart, sub->sendbuff, sub->recvbuff, nBytes, loopSize, winOffset, sub->isOneRPN, *request);
    }
  }
  *nBytesInOut = nBytes;  // 返回实际字节数
  return ncclSuccess;
}

// ============================================================================
// 函数: collNetIallreduce - 非注册缓冲区的IallReduce操作
// 参数:
//   proxyState - proxy状态
//   resources - 发送端资源
//   args - proxy参数
//   sub - 子操作参数
//   nBytes - 操作字节数
//   sendBeg - 发送缓冲区起始偏移
//   recvBeg - 接收缓冲区起始偏移
//   request - 输出的请求句柄
// 返回: ncclResult_t - 操作结果状态码
// 说明: 使用中间共享缓冲区进行IallReduce
// ============================================================================
static ncclResult_t collNetIallreduce(struct ncclProxyState* proxyState, struct sendResources *resources, struct ncclProxyArgs *args, struct ncclProxySubArgs *sub, ssize_t nBytes, ssize_t sendBeg, ssize_t recvBeg, void **request) {
  void *sendMhandle = resources->sendMhandles[NCCL_PROTO_SIMPLE];  // 获取发送内存句柄
  void *recvMhandle = resources->recvMhandles[NCCL_PROTO_SIMPLE];  // 获取接收内存句柄
  char *region = NCCL_NET_MAP_GET_POINTER(&resources->map, gpu, buffs[NCCL_PROTO_SIMPLE]);  // 获取共享缓冲区指针
  ssize_t eltSize = ncclTypeSize((ncclDataType_t)args->dtype);  // 获取元素大小
  // non-UB iallreduce, region is intermediate buffer and sendBeg/recvBeg is the corresponding offset
  // 非用户缓冲区iallreduce，region是中间缓冲区，sendBeg/recvBeg是对应的偏移量
  // for send and recv data. The send and recv mem handle are retrieved from resources.
  // 用于发送和接收数据。发送和接收内存句柄从resources中获取。
  NCCLCHECK(proxyState->ncclCollNet->iallreduce(resources->collNetComm, region + sendBeg, region + recvBeg, nBytes / eltSize, (ncclDataType_t)args->dtype, (ncclRedOp_t)args->redOp, sendMhandle, recvMhandle, request));
  if (*request)  // 如果请求成功提交
    TRACE(NCCL_NET, "sendProxy [%ld/%d] Iallreduce posted size %ld sendBeg %ld recvBeg %ld req %p", (long)sub->transmitted, sub->nsteps, nBytes, sendBeg, recvBeg, *request);
  return ncclSuccess;
}

// ============================================================================
// 函数: collNetRegIallgather - 注册缓冲区的IallGather操作
// 参数:
//   proxyState - proxy状态
//   resources - 发送端资源
//   args - proxy参数
//   sub - 子操作参数
//   nBytesIn - 输入字节数
//   allBeg - All缓冲区起始偏移
//   recvBeg - 接收缓冲区起始偏移
//   recvMhandle - 接收内存句柄
//   request - 输出的请求句柄
// 返回: ncclResult_t - 操作结果状态码
// ============================================================================
static ncclResult_t collNetRegIallgather(struct ncclProxyState* proxyState, struct sendResources *resources, struct ncclProxyArgs *args, struct ncclProxySubArgs *sub, ssize_t nBytesIn, ssize_t allBeg, ssize_t recvBeg, void *recvMhandle, void **request) {
  ncclNetSGE_t recvParts;  // 接收部分的scatter-gather条目
  ssize_t sizePerRank = args->specifics.collnetDirect.sizePerRank;  // 每个rank的大小
  char *region = NCCL_NET_MAP_GET_POINTER(&resources->map, gpu, buffs[NCCL_PROTO_SIMPLE]);  // 获取共享缓冲区
  ssize_t nBytes;
  ssize_t winOffset;
  void *sendbuff;
  // UB iallgather 1RPN logic is the same as iallreduce.
  // 用户缓冲区iallgather的1RPN逻辑与iallreduce相同。
  // If iallgather is not 1RPN, we can let collnet network directly access sendbuff but not recvbuff;
  // 如果iallgather不是1RPN，我们可以让collnet网络直接访问sendbuff但不能访问recvbuff；
  // the main reason is non-1RPN case will cause non-contiguous recv data from network, so
  // 主要原因是非1RPN情况会导致来自网络的接收数据不连续，所以
  // we have to use intermediate buffer "region" to recv data and copy into the recvbuff.
  // 我们必须使用中间缓冲区"region"接收数据然后复制到recvbuff。
  // so allBeg and recvMhandle, which are global window offset of recv buffer and mem handle for region,
  // 所以allBeg和recvMhandle（接收缓冲区的全局窗口偏移和region的内存句柄）
  // are only used in multi-RPN case.
  // 只在多RPN情况下使用。
  if (sub->isOneRPN) {  // 如果是单RPN
    nBytes = std::min((size_t)sub->nbytes, resources->maxCollBytes);  // 限制在maxCollBytes
    winOffset = sub->offset;  // 使用子操作偏移
    recvParts.mhandle = sub->recvMhandle;  // 使用用户缓冲区的内存句柄
    recvParts.address = sub->recvbuff;  // 直接写入用户缓冲区
  } else {  // 多RPN情况
    nBytes = nBytesIn;  // 使用输入字节数
    winOffset = allBeg;  // 使用All缓冲区偏移
    recvParts.mhandle = recvMhandle;  // 使用region的内存句柄
    recvParts.address = region + recvBeg;  // 写入region
  }
  recvParts.size = nBytes;  // 设置大小
  // 计算发送缓冲区：如果窗口偏移落在当前节点范围内，需要加上偏移
  if (winOffset / sizePerRank == args->specifics.collnetDirect.node) {
    sendbuff = sub->sendbuff + winOffset % sizePerRank;  // 当前节点负责的数据
  } else {
    sendbuff = sub->sendbuff;  // 发送整个缓冲区
  }
  // 调用CollNet插件的iallgather函数
  NCCLCHECK(proxyState->ncclCollNet->iallgather(resources->collNetComm, sendbuff, 1, &recvParts, sizePerRank, winOffset, nBytes, sub->sendMhandle, request));
  if (*request) {  // 如果请求成功提交
    if (sub->isOneRPN) {  // 单RPN情况需要更新指针
      sub->recvbuff += nBytes;  // 前移接收缓冲区指针
      sub->nbytes -= nBytes;  // 减少剩余字节数
      sub->offset += nBytes;  // 前移偏移
    }
    TRACE(NCCL_NET, "sendProxy [%ld/%d] registered Iallgather posted sizePerRank %ld winOffset %ld recvSize %ld isOneRPN %d request %p", sub->transmitted, sub->nsteps, sizePerRank, winOffset, nBytes, sub->isOneRPN, *request);
  }
  return ncclSuccess;
}

// ============================================================================
// 函数: collNetIallgather - 非注册缓冲区的IallGather操作
// 参数: 与collNetRegIallgather类似，但使用中间缓冲区
// 返回: ncclResult_t - 操作结果状态码
// ============================================================================
static ncclResult_t collNetIallgather(struct ncclProxyState* proxyState, struct sendResources *resources, struct ncclProxyArgs *args, struct ncclProxySubArgs *sub, ssize_t nBytes, ssize_t allBeg, ssize_t sendBeg, ssize_t recvBeg, void *sendMhandle, void *recvMhandle, void **request) {
  ncclNetSGE_t recvParts;  // 接收部分的scatter-gather条目
  ssize_t sizePerRank = args->specifics.collnetDirect.sizePerRank;  // 每个rank的大小
  char *region = NCCL_NET_MAP_GET_POINTER(&resources->map, gpu, buffs[NCCL_PROTO_SIMPLE]);  // 获取共享缓冲区
  recvParts.mhandle = recvMhandle;  // 使用region的内存句柄
  recvParts.address = region + recvBeg;  // 写入region
  recvParts.size = nBytes;  // 设置大小
  // non-UB iallgather, we use intermidate region buffers for both send and recv data.
  // 非用户缓冲区iallgather，我们对发送和接收数据都使用中间region缓冲区。
  // sendMhandle and recvMhandle are send and recv mem handles for region, and allBeg is
  // sendMhandle和recvMhandle是region的发送和接收内存句柄，allBeg是
  // the global window offset of recv buffer. sendBeg and recvBeg are offset to the region
  // 接收缓冲区的全局窗口偏移。sendBeg和recvBeg是region的偏移量
  // for intermediate data.
  // 用于中间数据。
  NCCLCHECK(proxyState->ncclCollNet->iallgather(resources->collNetComm, region + sendBeg, 1, &recvParts, sizePerRank, allBeg, nBytes, sendMhandle, request));
  if (*request)  // 如果请求成功提交
    TRACE(NCCL_NET, "sendProxy [%ld/%d] Iallgather posted sizePerRank %ld winOffset %ld recvSize %ld request %p", sub->transmitted, sub->nsteps, sizePerRank, allBeg, nBytes, *request);
  return ncclSuccess;
}

// ============================================================================
// 函数: collNetRegIreducescatter - 注册缓冲区的IreduceScatter操作
// 参数:
//   proxyState - proxy状态
//   resources - 发送端资源
//   args - proxy参数
//   sub - 子操作参数
//   nBytesIn - 输入字节数
//   allBeg - All缓冲区起始偏移
//   sendBeg - 发送缓冲区起始偏移
//   sendMhandle - 发送内存句柄
//   request - 输出的请求句柄
// 返回: ncclResult_t - 操作结果状态码
// ============================================================================
static ncclResult_t collNetRegIreducescatter(struct ncclProxyState* proxyState, struct sendResources *resources, struct ncclProxyArgs *args, struct ncclProxySubArgs *sub, ssize_t nBytesIn, ssize_t allBeg, ssize_t sendBeg, void *sendMhandle, void **request) {
  ncclNetSGE_t sendParts;  // 发送部分的scatter-gather条目
  ssize_t sizePerRank = args->specifics.collnetDirect.sizePerRank;  // 每个rank的大小
  char *region = NCCL_NET_MAP_GET_POINTER(&resources->map, gpu, buffs[NCCL_PROTO_SIMPLE]);  // 获取共享缓冲区
  ssize_t nBytes;
  size_t winOffset;
  void *recvbuff;
  // Similar to iallgather, if ireducescatter is not 1RPN, we can let collnet network
  // 与iallgather类似，如果ireducescatter不是1RPN，我们可以让collnet网络
  // directly access recvbuff but not sendbuff. We use intermediate buffer "region" to
  // 直接访问recvbuff但不能访问sendbuff。我们使用中间缓冲区"region"来
  // send data and directly recv into the recvbuff.
  // 发送数据并直接接收到recvbuff。
  if (sub->isOneRPN) {  // 如果是单RPN
    nBytes = std::min((size_t)sub->nbytes, resources->maxCollBytes);  // 限制在maxCollBytes
    winOffset = sub->offset;  // 使用子操作偏移
    sendParts.mhandle = sub->sendMhandle;  // 使用用户缓冲区的内存句柄
    sendParts.address = sub->sendbuff;  // 直接从用户缓冲区读取
  } else {  // 多RPN情况
    nBytes = nBytesIn;  // 使用输入字节数
    winOffset = allBeg;  // 使用All缓冲区偏移
    sendParts.mhandle = sendMhandle;  // 使用region的内存句柄
    sendParts.address = region + sendBeg;  // 从region读取
  }
  sendParts.size = nBytes;  // 设置大小
  // 计算接收缓冲区：如果窗口偏移落在当前节点范围内，需要加上偏移
  if (winOffset / sizePerRank == args->specifics.collnetDirect.node) {
    recvbuff = sub->recvbuff + winOffset % sizePerRank;  // 当前节点负责的数据
  } else {
    recvbuff = sub->recvbuff;  // 接收整个缓冲区
  }
  // 调用CollNet插件的ireducescatter函数
  NCCLCHECK(proxyState->ncclCollNet->ireducescatter(resources->collNetComm, 1, &sendParts, recvbuff, sizePerRank, winOffset, nBytes, (ncclDataType_t)args->dtype, (ncclRedOp_t)args->redOp, sub->recvMhandle, request));
  if (*request) {  // 如果请求成功提交
    if (sub->isOneRPN) {  // 单RPN情况需要更新指针
      sub->sendbuff += nBytes;  // 前移发送缓冲区指针
      sub->nbytes -= nBytes;  // 减少剩余字节数
      sub->offset += nBytes;  // 前移偏移
    }
    TRACE(NCCL_NET, "sendProxy [%ld/%d] registered Ireducescatter posted sizePerRank %ld winOffset %ld sendSize %ld isOneRPN %d request %p", sub->transmitted, sub->nsteps, sizePerRank, winOffset, nBytes, sub->isOneRPN, *request);
  }
  return ncclSuccess;
}

// ============================================================================
// 函数: collNetIreducescatter - 非注册缓冲区的IreduceScatter操作
// 参数: 与collNetRegIreducescatter类似，但使用中间缓冲区
// 返回: ncclResult_t - 操作结果状态码
// ============================================================================
static ncclResult_t collNetIreducescatter(struct ncclProxyState* proxyState, struct sendResources *resources, struct ncclProxyArgs *args, struct ncclProxySubArgs *sub, ssize_t nBytes, ssize_t allBeg, ssize_t sendBeg, ssize_t recvBeg, void *sendMhandle, void *recvMhandle, void **request) {
  ncclNetSGE_t sendParts;  // 发送部分的scatter-gather条目
  ssize_t sizePerRank = args->specifics.collnetDirect.sizePerRank;  // 每个rank的大小
  char *region = NCCL_NET_MAP_GET_POINTER(&resources->map, gpu, buffs[NCCL_PROTO_SIMPLE]);  // 获取共享缓冲区
  sendParts.mhandle = sendMhandle;  // 使用region的内存句柄
  sendParts.address = region + sendBeg;  // 从region读取
  sendParts.size = nBytes;  // 设置大小
  // non-UB ireducescatter is the same as non-UB iallgather but in the reverse direction.
  // 非用户缓冲区ireducescatter与非用户缓冲区iallgather相同，但方向相反。
  NCCLCHECK(proxyState->ncclCollNet->ireducescatter(resources->collNetComm, 1, &sendParts, region + recvBeg, sizePerRank, allBeg, nBytes, (ncclDataType_t)args->dtype, (ncclRedOp_t)args->redOp, recvMhandle, request));
  if (*request)  // 如果请求成功提交
    TRACE(NCCL_NET, "sendProxy [%ld/%d] Ireducescatter posted sizePerRank %ld winOffset %ld sendSize %ld request %p", sub->transmitted, sub->nsteps, sizePerRank, allBeg, nBytes, *request);
  return ncclSuccess;
}

// ============================================================================
// 函数: sendProxyProgress - 发送端proxy进度推进函数
// 参数:
//   proxyState - proxy状态
//   args - proxy参数（包含所有子操作）
// 返回: ncclResult_t - 操作结果状态码
// 说明: 这是发送端的主要进度推进函数，处理post/receive/transmit/done四个阶段
// ============================================================================
static ncclResult_t sendProxyProgress(struct ncclProxyState* proxyState, struct ncclProxyArgs* args) {
  // ========== 初始化阶段 ==========
  if (args->state == ncclProxyOpReady) {  // 如果操作刚启动
    for (int s=0; s<args->nsubs; s++) {  // 遍历所有子操作
      struct ncclProxySubArgs* sub = args->subs+s;  // 获取子操作
      struct sendResources* resources = (struct sendResources*) (sub->connection->transportResources);  // 获取资源
      // Round to next multiple of sliceSteps
      // 向上取整到sliceSteps的倍数
      sub->base = ROUNDUP(resources->step, args->chunkSteps);  // 设置基础步骤号
      sub->posted = sub->received = sub->transmitted = sub->done = 0;  // 重置所有计数器
      resources->step = sub->base + sub->nsteps;  // 更新资源步骤号
      //adjust nsteps for registerd buffers as device signals a single step
      // 为注册缓冲区调整nsteps，因为设备只发信号一次
      if (sub->reg && sub->isOneRPN) sub->nsteps = DIVUP((size_t)sub->nbytes, resources->maxCollBytes);  // 计算需要的步骤数
    }
    args->state = ncclProxyOpProgress;  // 转换到进度状态
  }
  args->idle = 1;  // 假设空闲（如果有任何操作执行会设为0）

  // ========== 进度推进阶段 ==========
  if (args->state == ncclProxyOpProgress) {
    int p = NCCL_PROTO_SIMPLE;  // 使用Simple协议
    int nGroups = DIVUP(args->nsubs, COLLNET_GROUP_NSUBS);  // 计算组数
    for (int s=0; s<args->nsubs; s++) {  // 遍历所有子操作
      struct ncclProxySubArgs* sub = args->subs+s;  // 获取子操作
      struct sendResources* resources = (struct sendResources*) (sub->connection->transportResources);  // 获取资源
      void* sendMhandle = resources->sendMhandles[p];  // 获取发送内存句柄
      void* recvMhandle = resources->recvMhandles[p];  // 获取接收内存句柄
      auto reqFifo = resources->reqFifo;  // 获取请求FIFO
      int group = s/COLLNET_GROUP_NSUBS;  // 计算组编号
      int groupStart = s - (s%COLLNET_GROUP_NSUBS);  // 计算组内起始索引

      // ========== 阶段1: 发送credit通知接收端 ==========
      if (sub->posted < sub->nsteps && sub->posted < sub->done + NCCL_STEPS) {
        int buffSlot = (sub->base+sub->posted)%NCCL_STEPS;  // 计算缓冲区槽位
        // 如果是非注册缓冲区或者是ReduceScatter，设置连接FIFO偏移
        if (sub->reg == 0 || (!sub->isOneRPN && args->coll == ncclFuncReduceScatter)) {
          resources->recvMem->connFifo[buffSlot].offset = calcRegionOffset(args, 0, s, sub->posted, 0);  // 计算区域偏移
          __sync_synchronize();  // 同步内存
        }
        volatile uint64_t* sendHead = resources->gdcSync ? resources->gdcSync : &resources->sendMem->head;  // 获取head指针
        TRACE(NCCL_NET, "sendProxy [%ld/%d/%d/%d] posted offset %d @ %p signal %ld->%ld", long(sub->posted), group, buffSlot, sub->nsteps, resources->recvMem->connFifo[buffSlot].offset, &resources->recvMem->connFifo[buffSlot].offset, long(*sendHead), long(sub->base + sub->posted + args->sliceSteps - NCCL_STEPS));
        sub->posted += args->sliceSteps;  // 增加posted计数
        // Only post one credit for registered buffer
        // 注册缓冲区只发送一个credit
        if (sub->reg == 0 || !sub->isOneRPN || sub->posted == args->sliceSteps) *sendHead = sub->base + sub->posted - NCCL_STEPS;  // 更新head
        if (resources->gdcSync) wc_store_fence(); // Flush out WC write（刷新写合并缓冲）
      }

      // ========== 阶段2: 接收来自接收端的完成通知 ==========
      if (sub->received < sub->posted && sub->received < sub->done + calcStepsPerGroup(nGroups)) {
        int buffSlot = (sub->base+sub->received)%NCCL_STEPS;  // 计算缓冲区槽位
        volatile struct ncclConnFifo* connFifo = (volatile struct ncclConnFifo*)resources->recvMem->connFifo;  // 获取连接FIFO
        volatile uint64_t* recvTail = &resources->recvMem->tail;  // 获取tail指针
        //device progresses tail by only 1 for registered buffers
        // 注册缓冲区的设备只推进tail 1步
        uint64_t tail = sub->base + (sub->reg && sub->isOneRPN ? 0 : sub->received);  // 计算tail
        if ((connFifo[buffSlot].size != -1 || sub->reg) && (*recvTail > tail)) {  // 如果有数据到达
          if (args->coll != ncclFuncAllReduce && sub->reg == 0) {  // 如果不是AllReduce且非注册缓冲区
            int sendBeg = calcRegionOffset(args, 0, s, sub->received, 0);  // 计算发送起始偏移
            int sendEnd = calcRegionOffset(args, 0, s, sub->received, 1);  // 计算发送结束偏移
            if (sendEnd-sendBeg != connFifo[buffSlot].size) {  // 验证大小
              WARN("CollNet sizes: want=%d got=%ld", sendEnd-sendBeg, connFifo[buffSlot].size);
              return ncclInternalError;  // 返回错误
            }
          }
          connFifo[buffSlot].size = -1;  // 标记为已消费
          sub->received += args->sliceSteps;  // 增加received计数
          args->idle = 0;  // 标记为非空闲
        }
      }

      // ========== 阶段3: 发起集合操作 ==========
      // Enforce collective ordering of collnet ops.
      // 强制执行集合操作的顺序
      bool ordered = s==0 ? args->subs[args->nsubs-1].transmitted == sub->transmitted  // 检查是否有序
                          : sub->transmitted < (sub-1)->transmitted;
      if (ordered && (sub->transmitted < sub->received)) {  // 如果有序且有数据需要传输
        if (LAST_OF_GROUP(args, s)) {  // 如果是组内最后一个
          int buffSlot = (sub->base+sub->transmitted)%NCCL_STEPS;  // 计算缓冲区槽位
          if (!reqFifo[group][buffSlot].turnIsSendNotRecv) continue;  // 如果不是发送端轮次，跳过

          // 计算各种偏移量
          ssize_t allBeg = calcAlgoOffset(args, 1, groupStart, sub->transmitted);  // All起始偏移
          ssize_t allEnd = calcAlgoOffset(args, 1, s+1, sub->transmitted);  // All结束偏移
          ssize_t sendBeg = calcRegionOffset(args, 0, groupStart, sub->transmitted, 0);  // 发送起始偏移
          ssize_t sendEnd = calcRegionOffset(args, 0, s, sub->transmitted, 1);  // 发送结束偏移
          ssize_t recvBeg = calcRegionOffset(args, 1, groupStart, sub->transmitted, 0);  // 接收起始偏移
          ssize_t recvEnd = calcRegionOffset(args, 1, s, sub->transmitted, 1);  // 接收结束偏移
          reqFifo[group][buffSlot].size = recvEnd - recvBeg;  // 设置接收大小

          if (sendBeg==sendEnd && recvBeg==recvEnd) {  // 如果没有数据
            sub->requests[buffSlot] = nullptr; // trivally finished request（空请求）
          } else {  // 有数据需要处理
            ssize_t nBytes = 0;  // 字节数
            if (args->coll == ncclFuncAllReduce) {  // AllReduce操作
              nBytes = sendEnd - sendBeg;  // 计算字节数
              if (sub->reg) {  // 注册缓冲区
                NCCLCHECK(collNetRegIallreduce(proxyState, resources, args, sub, groupStart, &nBytes, &sub->requests[buffSlot]));
              } else {  // 非注册缓冲区
                NCCLCHECK(collNetIallreduce(proxyState, resources, args, sub, nBytes, sendBeg, recvBeg, &sub->requests[buffSlot]));
              }
            } else if (args->coll == ncclFuncAllGather) {  // AllGather操作
              nBytes = allEnd - allBeg;  // 计算字节数
              if (sub->reg) {  // 注册缓冲区
                NCCLCHECK(collNetRegIallgather(proxyState, resources, args, sub, nBytes, allBeg, recvBeg, recvMhandle, &sub->requests[buffSlot]));
              } else {  // 非注册缓冲区
                NCCLCHECK(collNetIallgather(proxyState, resources, args, sub, nBytes, allBeg, sendBeg, recvBeg, sendMhandle, recvMhandle, &sub->requests[buffSlot]));
              }
            } else {  // ReduceScatter操作
              // reducescatter
              nBytes = allEnd - allBeg;  // 计算字节数
              if (sub->reg) {  // 注册缓冲区
                NCCLCHECK(collNetRegIreducescatter(proxyState, resources, args, sub, nBytes, allBeg, sendBeg, sendMhandle, &sub->requests[buffSlot]));
              } else {  // 非注册缓冲区
                NCCLCHECK(collNetIreducescatter(proxyState, resources, args, sub, nBytes, allBeg, sendBeg, recvBeg, sendMhandle, recvMhandle, &sub->requests[buffSlot]));
              }
            }
            if (nBytes > 0 && sub->requests[buffSlot] == nullptr) continue;  // 如果字节数>0但无请求，跳过
          }
        }
        sub->transmitted += args->sliceSteps;  // 增加transmitted计数
        args->idle = 0;  // 标记为非空闲
        continue;  // 继续下一个子操作
      }

      // ========== 阶段4: 检查网络完成情况 ==========
      // Check whether the network has completed some send operations.
      // 检查网络是否完成了一些发送操作
      if (LAST_OF_GROUP(args, s) && sub->done < sub->transmitted) {  // 如果是组内最后一个且有未完成的请求
        int done, size;
        int buffSlot = (sub->base+sub->done)%NCCL_STEPS;  // 计算缓冲区槽位
        done = 1;  // 假设完成
        if (sub->requests[buffSlot]) NCCLCHECK(proxyState->ncclCollNet->test((void*)(sub->requests[buffSlot]), &done, &size));  // 测试请求
        if (done) {  // 如果完成
          TRACE(NCCL_NET, "sendProxy [%ld/%d/%d] request %p done, size %d", (long)sub->done, group, buffSlot, sub->requests[buffSlot], size);
          sub->requests[buffSlot] = nullptr;  // 清空请求
          reqFifo[group][buffSlot].turnIsSendNotRecv = false; // Notify recvProxy（通知接收端）
          for (int i=groupStart; i<=s; i++) args->subs[i].done += args->sliceSteps;  // 更新组内所有子操作的done计数
          args->idle = 0;  // 标记为非空闲
          int allDone = 1;  // 检查是否全部完成
          for (int i=0; i<args->nsubs; i++) {
            if (args->subs[i].done < args->subs[i].nsteps) { allDone = 0; break; }  // 如果有未完成的，设置标志
          }
          if (allDone) {  // 如果全部完成
            args->state = ncclProxyOpNone;  // 转换到完成状态
            TRACE(NCCL_NET, "sendProxy [%ld/%d] stopped", (long)sub->done, s);
          }
        }
      }
    }
  }
  return ncclSuccess;
}

// ============================================================================
// 函数: collNetRecvFlush - 接收端GDR缓冲区刷新操作
// 参数:
//   proxyState - proxy状态
//   resources - 接收端资源
//   args - proxy参数
//   sub - 子操作参数
//   groupStart - 组内起始索引
//   nBytesIn - 输入字节数
//   recvBeg - 接收缓冲区起始偏移
//   request - 输出的请求句柄
// 返回: ncclResult_t - 操作结果状态码
// 说明: 用于GDR模式下的缓冲区刷新，确保数据从NIC刷新到GPU内存
// ============================================================================
static ncclResult_t collNetRecvFlush(struct ncclProxyState* proxyState, struct recvResources *resources, struct ncclProxyArgs *args, struct ncclProxySubArgs *sub, int groupStart, ssize_t nBytesIn, ssize_t recvBeg, void **request) {
  char *region = NCCL_NET_MAP_GET_POINTER(&resources->map, gpu, buffs[NCCL_PROTO_SIMPLE]);  // 获取共享缓冲区
  if (sub->reg && (sub->isOneRPN || args->coll != ncclFuncAllGather)) {  // 如果是注册缓冲区且需要刷新
    ssize_t nBytes, loopSize;
    ssize_t offset = sub->offset + groupStart * args->chunkSize;  // 计算偏移
    if (sub->isOneRPN) {  // 单RPN情况
      nBytes = std::min((size_t)sub->nbytes, resources->maxCollBytes);  // 限制在maxCollBytes
      loopSize = nBytes;
    } else {  // 多RPN情况
      nBytes = std::min(sub->nbytes - sub->loopOffset, nBytesIn);  // 计算剩余字节数
      loopSize = sub->loopSize;
    }
    if (nBytes > 0) {  // 如果有数据需要刷新
      if (args->coll == ncclFuncReduceScatter) {  // ReduceScatter需要特殊处理
        ssize_t sizePerRank = args->specifics.collnetDirect.sizePerRank;  // 每个rank的大小
        ssize_t groupStartOffset = sub->offset + groupStart * args->chunkSize;  // 组起始偏移
        ssize_t groupEndOffset = groupStartOffset + nBytes;  // 组结束偏移
        int node = args->specifics.collnetDirect.node;  // 当前节点
        int startNode = groupStartOffset / sizePerRank;  // 起始节点
        int lastNode = groupEndOffset / sizePerRank;  // 结束节点
        // 根据当前节点位置计算刷新范围
        if (startNode == node) {  // 当前节点是起始节点
          offset = groupStartOffset % sizePerRank;  // 使用组内偏移
          nBytes = std::min(sizePerRank - offset, nBytes);  // 限制在sizePerRank范围内
        } else if (startNode < node && node < lastNode) {  // 当前节点在中间
          offset = 0;  // 偏移为0
          nBytes = sizePerRank;  // 刷新完整的rank大小
        } else if (node == lastNode) {  // 当前节点是结束节点
          offset = 0;
          nBytes = groupEndOffset % sizePerRank;  // 刷新部分大小
        } else {
          // dummy flush（虚拟刷新）
          offset = 0;
        }
      }
      // 调用CollNet插件的iflush函数刷新缓冲区
      NCCLCHECK(proxyState->ncclCollNet->iflush(resources->collNetComm, sub->recvbuff + offset + sub->loopOffset, nBytes, sub->recvMhandle, request));
      if (*request) {  // 如果请求成功提交
        sub->nbytes -= loopSize;  // 减少剩余字节数
        sub->offset += loopSize;  // 前移偏移
      }
    }
  } else {  // 非注册缓冲区或AllGather
    // 使用共享缓冲区刷新
    NCCLCHECK(proxyState->ncclCollNet->iflush(resources->collNetComm, region + recvBeg, nBytesIn, resources->mhandles[NCCL_PROTO_SIMPLE], request));
  }
  return ncclSuccess;
}

// ============================================================================
// 函数: recvProxyProgress - 接收端proxy进度推进函数
// 参数:
//   proxyState - proxy状态
//   args - proxy参数（包含所有子操作）
// 返回: ncclResult_t - 操作结果状态码
// 说明: 这是接收端的主要进度推进函数，处理post/receive/flush/transmit/done五个阶段
// ============================================================================
static ncclResult_t recvProxyProgress(struct ncclProxyState* proxyState, struct ncclProxyArgs* args) {
  // ========== 初始化阶段 ==========
  if (args->state == ncclProxyOpReady) {  // 如果操作刚启动
    for (int s=0; s<args->nsubs; s++) {  // 遍历所有子操作
      struct ncclProxySubArgs* sub = args->subs+s;  // 获取子操作
      struct recvResources* resources = (struct recvResources*) (sub->connection->transportResources);  // 获取资源
      // Round to next multiple of sliceSteps
      sub->base = ROUNDUP(resources->step, args->chunkSteps);  // 向上取整
      sub->posted = sub->received = sub->flushed = sub->transmitted = sub->done = 0;  // 重置所有计数器
      resources->step = sub->base + sub->nsteps;  // 更新资源步骤号
      //adjust nsteps for registerd buffers as device signals a single step
      if (sub->reg && sub->isOneRPN) sub->nsteps = DIVUP((size_t)sub->nbytes, resources->maxCollBytes);  // 调整步骤数
      memset(sub->requests, 0, sizeof(sub->requests));  // 清空请求数组
    }
    args->state = ncclProxyOpProgress;  // 转换到进度状态
  }
  args->idle = 1;  // 假设空闲

  // ========== 进度推进阶段 ==========
  if (args->state == ncclProxyOpProgress) {
    int nGroups = DIVUP(args->nsubs, COLLNET_GROUP_NSUBS);  // 计算组数
    for (int s=0; s<args->nsubs; s++) {  // 遍历所有子操作
      int group = s/COLLNET_GROUP_NSUBS;  // 计算组编号
      int groupStart = s - (s%COLLNET_GROUP_NSUBS);  // 计算组内起始索引
      struct ncclProxySubArgs* sub = args->subs+s;  // 获取子操作
      struct recvResources* resources = (struct recvResources*) (sub->connection->transportResources);  // 获取资源
      auto reqFifo = resources->reqFifo;  // 获取请求FIFO

      // ========== 阶段1: 发布缓冲区 ==========
      // Enforce sync between operations of the same group.
      // 强制执行同组操作之间的同步
      if (LAST_OF_GROUP(args, s) && (sub->posted < sub->done + calcStepsPerGroup(nGroups)) && (sub->posted < sub->nsteps)) {
        int buffSlot = (sub->base+sub->posted)%NCCL_STEPS;  // 计算缓冲区槽位
        reqFifo[group][buffSlot].turnIsSendNotRecv = true;  // 设置标志为发送端轮次
        TRACE(NCCL_NET, "recvProxy [%ld/%d/%d] posted buffer", (long)sub->posted, group, buffSlot);
        sub->posted += args->sliceSteps;  // 增加posted计数
        args->idle = 0;  // 标记为非空闲
        continue;  // 继续下一个子操作
      }

      // ========== 阶段2: 检测发送端完成 ==========
      if (LAST_OF_GROUP(args, s) && (sub->received < sub->posted)) {
        int buffSlot = (sub->base+sub->received)%NCCL_STEPS;  // 计算缓冲区槽位
        if (!reqFifo[group][buffSlot].turnIsSendNotRecv) { // Buffer is cleared : coll is complete（缓冲区已清空：集合操作完成）
          ssize_t recvBeg = calcRegionOffset(args, 1, groupStart, sub->received, 0);  // 计算接收起始偏移
          ssize_t recvEnd = calcRegionOffset(args, 1, s, sub->received, 1);  // 计算接收结束偏移
          ssize_t totalSize = recvEnd - recvBeg;  // 计算总大小
          TRACE(NCCL_NET, "recvProxy [%ld/%d/%d] received, size %ld chunkSize=%ld", (long)sub->received, group, buffSlot, totalSize, args->chunkSize);
          sub->received += args->sliceSteps;  // 增加received计数
          // 如果需要GDR刷新
          if ((reqFifo[group][buffSlot].size > 0 || sub->reg) && resources->useGdr && resources->needFlush) {
            // GDRCOPY support
            if (resources->gdcFlush) {  // 如果使用GDC flush
#if defined (__x86_64__)
              // Force a PCI-E read from GPU memory
              // 强制从GPU内存进行PCIe读取（刷新缓存）
              asm volatile ("mov (%0), %%eax" :: "l"(resources->gdcFlush) : "%eax");  // x86_64汇编：读取内存
#else
              WARN("NET: GDR Flush only supported on x86_64");  // 不支持非x86_64架构
              return ncclInternalError;
#endif
            } else {  // 使用iflush刷新
              NCCLCHECK(collNetRecvFlush(proxyState, resources, args, sub, groupStart, totalSize, recvBeg, &sub->requests[buffSlot]));
            }
          }
          args->idle = 0;  // 标记为非空闲
          continue;  // 继续下一个子操作
        }
      }

      // ========== 阶段3: 等待刷新完成 ==========
      if (LAST_OF_GROUP(args, s) && (sub->flushed < sub->received)) {
        // Progress flush operations
        // 推进刷新操作
        int buffSlot = (sub->base + sub->flushed)%NCCL_STEPS;  // 计算缓冲区槽位
        int done = 1;  // 假设完成
        if (sub->requests[buffSlot]) NCCLCHECK(proxyState->ncclCollNet->test(sub->requests[buffSlot], &done, NULL));  // 测试请求
        if (done) {  // 如果刷新完成
          sub->requests[buffSlot] = nullptr;  // 清空请求
          TRACE(NCCL_NET, "recvProxy [%ld/%d/%d] flushed", (long)sub->flushed, group, buffSlot);
          for (int i=group*COLLNET_GROUP_NSUBS; i<=s; i++) args->subs[i].flushed += args->sliceSteps;  // 更新组内flushed计数
          args->idle = 0;  // 标记为非空闲
          //continue;
        }
      }
      if (sub->transmitted < sub->flushed) {
        if (sub->reg == 0 || (!sub->isOneRPN && args->coll == ncclFuncAllGather)) {
          int buffSlot = (sub->base + sub->transmitted)%NCCL_STEPS;
          volatile struct ncclConnFifo* connFifo = (volatile struct ncclConnFifo*)resources->recvMem->connFifo;
          connFifo[buffSlot].offset = calcRegionOffset(args, 1, s, sub->transmitted, 0);
          __sync_synchronize();
        }
        volatile uint64_t* recvTail = resources->gdcSync ? resources->gdcSync : &resources->recvMem->tail;
        if (sub->reg && sub->isOneRPN) {
          // We may have bumped net steps, but reg operations only have a single step w.r.t. the GPU.
          if (sub->flushed == sub->nsteps) *recvTail = sub->base + args->sliceSteps;
        } else {
          *recvTail = sub->base + sub->flushed;
        }
        if (resources->gdcSync) wc_store_fence(); // Flush out WC write
        sub->transmitted += args->sliceSteps;
        args->idle = 0;
        continue;
      }
      // Enforce sync here to make sure the last sub doesn't increase "done" before all others in the group have
      // reached the same point, otherwise we would start posting buffers to the send proxy before we're done
      // processing all the shared buffer.
      bool groupSync = s==0 ? args->subs[args->nsubs-1].done == sub->done
                            : (sub-1)->done > sub->done;
      volatile uint64_t* sendHead = &resources->sendMem->head;
      int done = sub->reg && sub->isOneRPN ? 0 : sub->done;
      if (groupSync && sub->done < sub->transmitted && sub->base + done < *sendHead) {
        sub->done += args->sliceSteps;
        args->idle = 0;
        if (sub->done == sub->nsteps && s == args->nsubs-1) {
          args->state = ncclProxyOpNone;
          TRACE(NCCL_NET, "recvProxy [%ld/%d] stopped", (long)sub->done, s);
        }
      }
    }
  }
  return ncclSuccess;
}

// ============================================================================
// 结构: collnetRegInfo - CollNet缓冲区注册信息
// ============================================================================
struct collnetRegInfo {
  uintptr_t buffer;   // 缓冲区地址
  size_t size;        // 缓冲区大小
};

// ============================================================================
// 函数: collnetRegisterBuffer - 内部函数，注册CollNet缓冲区
// 参数:
//   comm - 通信上下文
//   userbuff - 用户缓冲区指针
//   buffSize - 缓冲区大小
//   type - 类型（collNetRecv或collNetSend）
//   regRecord - 注册记录
//   outRegBufFlag - 输出标志（0=失败，1=新注册，2=复用）
//   outHandle - 输出的内存句柄
// 返回: ncclResult_t - 操作结果状态码
// ============================================================================
static ncclResult_t collnetRegisterBuffer(struct ncclComm* comm, const void* userbuff, size_t buffSize, int type, struct ncclReg* regRecord, int* outRegBufFlag, void** outHandle) {
  ncclResult_t ret = ncclSuccess;
  int gdrEnable = -1;  // -1表示未知，0表示禁用，1表示启用
  if (regRecord) {  // 如果有注册记录
    if (regRecord->state & COLLNET_REG_COMPLETE) {  // 如果已经完成注册
      // reuse previous registration
      // 复用之前的注册
      *outRegBufFlag = 2;  // 设置标志为复用
      *outHandle = regRecord->collnetHandle;  // 返回已有句柄
      INFO(NCCL_REG, "rank %d - COLLNET reuse register userbuff %p (handle %p), buffSize %ld, type %s", comm->rank, userbuff, regRecord->collnetHandle, buffSize, type == collNetRecv ? "Recv" : "Send");
      goto exit;  // 直接退出
    } else {  // 需要新注册
      /* start register collnet buffer */
      // 开始注册CollNet缓冲区
      struct collnetRegInfo info = { regRecord->begAddr, regRecord->endAddr - regRecord->begAddr };  // 构造注册信息
      void* handle = NULL;  // 初始化句柄为NULL
      // 获取连接信息（根据类型选择接收或发送连接）
      struct ncclConnInfo* conn = (type == collNetRecv) ? &comm->channels[0].peers[comm->nRanks]->recv[type].conn : &comm->channels[0].peers[comm->nRanks]->send[type].conn;

      if (conn->flags & NCCL_DIRECT_NIC) {  // 如果支持直接NIC访问（GDR）
        // 获取proxy连接器
        struct ncclProxyConnector* proxyconn = (type == collNetRecv) ? &comm->channels[0].peers[comm->nRanks]->recv[type].proxyConn : &comm->channels[0].peers[comm->nRanks]->send[type].proxyConn;
        gdrEnable = 1;  // 标记GDR已启用
        // 调用proxy进行阻塞式注册
        NCCLCHECKGOTO(ncclProxyCallBlocking(comm, proxyconn, ncclProxyMsgRegister, &info, sizeof(struct collnetRegInfo), &handle, sizeof(void*)), ret, fail);
        if (handle) {  // 如果注册成功
          regRecord->state |= COLLNET_REG_COMPLETE;  // 标记注册完成
          regRecord->collnetProxyconn = proxyconn;  // 保存proxy连接器
          *outHandle = regRecord->collnetHandle = handle;  // 保存句柄
          *outRegBufFlag = 1;  // 设置标志为新注册
          INFO(NCCL_REG, "rank %d - COLLNET register userbuff %p (handle %p), buffSize %ld, type %s", comm->rank, userbuff, handle, buffSize, type == collNetRecv ? "Recv" : "Send");
        }
      } else {  // 不支持直接NIC访问
        gdrEnable = 0;
        goto fail;  // 跳转到失败处理
      }
    }
  }
exit:
  return ret;  // 返回结果
fail:
  *outRegBufFlag = 0;  // 设置标志为失败
  *outHandle = NULL;  // 设置句柄为NULL
  INFO(NCCL_REG, "rank %d - COLLNET failed to register userbuff %p, buffSize %ld, type %s, GDR %d", comm->rank, userbuff, buffSize, type == collNetRecv ? "Recv" : "Send", gdrEnable);
  goto exit;  // 跳转到退出
}

// ============================================================================
// 函数: ncclCollnetLocalRegisterBuffer - 本地注册CollNet缓冲区
// 参数:
//   comm - 通信上下文
//   userbuff - 用户缓冲区指针
//   buffSize - 缓冲区大小
//   type - 类型（collNetRecv或collNetSend）
//   outRegBufFlag - 输出标志
//   outHandle - 输出的内存句柄
// 返回: ncclResult_t - 操作结果状态码
// 说明: 运行时注册接口，用于非CUDA Graph场景
// ============================================================================
ncclResult_t ncclCollnetLocalRegisterBuffer(struct ncclComm* comm, const void* userbuff, size_t buffSize, int type, int* outRegBufFlag, void** outHandle) {
  ncclResult_t ret = ncclSuccess;
  struct ncclReg *regRecord = NULL;  // 注册记录指针
  bool isValid = false;  // 是否有效
  void *base = NULL;  // 基地址
  size_t baseSize = 0;  // 基础大小

  *outRegBufFlag = 0;  // 初始化标志为失败
  *outHandle = NULL;  // 初始化句柄为NULL
  if (comm && userbuff && buffSize > 0) {  // 验证输入参数
    NCCLCHECKGOTO(ncclRegFind(comm, userbuff, buffSize, &regRecord), ret, fail);  // 查找注册记录
    NCCLCHECKGOTO(ncclRegLocalIsValid(regRecord, &isValid), ret, fail);  // 检查记录是否有效
    if (isValid) {  // 如果有效
      CUCHECKGOTO(cuMemGetAddressRange((CUdeviceptr *)&base, &baseSize, (CUdeviceptr)userbuff), ret, fail);  // 获取CUDA内存地址范围
      if ((uint64_t)base + baseSize < (uint64_t)userbuff + buffSize) goto exit;  // 检查缓冲区是否在范围内
    }
    NCCLCHECKGOTO(collnetRegisterBuffer(comm, userbuff, buffSize, type, regRecord, outRegBufFlag, outHandle), ret, fail);  // 调用内部注册函数
  }
exit:
  return ret;
fail:
  *outRegBufFlag = 0;  // 设置标志为失败
  goto exit;
}

// ============================================================================
// 结构: ncclCollnetCleanupCallback - CollNet清理回调结构
// 说明: 用于CUDA Graph场景，在图执行后清理注册
// ============================================================================
struct ncclCollnetCleanupCallback {
  struct ncclCommCallback base;  // 基础回调结构
  struct ncclComm *comm;         // 通信上下文
  struct ncclReg *reg;           // 注册记录
};

// ============================================================================
// 函数: cleanupCollnet - 清理CollNet注册的回调函数
// 参数:
//   comm - 通信上下文
//   cb - 回调结构指针
// 返回: ncclResult_t - 操作结果状态码
// 说明: 用于CUDA Graph场景，在图执行完成后注销内存注册
// ============================================================================
static ncclResult_t cleanupCollnet(struct ncclComm* comm, struct ncclCommCallback* cb) {
  struct ncclCollnetCleanupCallback* obj = (struct ncclCollnetCleanupCallback*)cb;  // 类型转换
  NCCLCHECK(ncclCommGraphDeregister(obj->comm, obj->reg));  // 注销图形内存注册
  free(obj);  // 释放回调对象
  return ncclSuccess;
}

// ============================================================================
// 函数: ncclCollnetGraphRegisterBuffer - CUDA Graph场景的缓冲区注册
// 参数:
//   comm - 通信上下文
//   userbuff - 用户缓冲区指针
//   buffSize - 缓冲区大小
//   type - 类型（collNetRecv或collNetSend）
//   outRegBufFlag - 输出标志（0=失败，1=新注册，2=复用）
//   outHandle - 输出的内存句柄
//   cleanupQueue - 清理回调队列
//   nCleanupQueueElts - 队列元素计数
// 返回: ncclResult_t - 操作结果状态码
// 说明: 用于CUDA Graph的持久化缓冲区注册，注册后会添加清理回调
// ============================================================================
ncclResult_t ncclCollnetGraphRegisterBuffer(struct ncclComm* comm, const void* userbuff, size_t buffSize, int type, int* outRegBufFlag, void** outHandle, struct ncclIntruQueue<struct ncclCommCallback, &ncclCommCallback::next>* cleanupQueue, int* nCleanupQueueElts) {
  ncclResult_t ret = ncclSuccess;  // 初始化返回值
  struct ncclCollnetCleanupCallback* record = NULL;  // 清理回调记录
  struct ncclReg *regRecord = NULL;  // 注册记录
  void *base = NULL;  // CUDA内存基地址
  size_t baseSize = 0;  // CUDA内存基础大小

  *outRegBufFlag = 0;  // 初始化标志为失败
  if (comm && userbuff && buffSize > 0) {  // 验证输入参数
    // 获取CUDA内存地址范围
    CUCHECKGOTO(cuMemGetAddressRange((CUdeviceptr *)&base, &baseSize, (CUdeviceptr)userbuff), ret, fail);
    // 检查缓冲区是否完全包含在CUDA内存范围内
    if ((uint64_t)base + baseSize < (uint64_t)userbuff + buffSize) goto exit;  // 超出范围，直接退出
    // 执行图形内存注册（会复用已有注册）
    NCCLCHECKGOTO(ncclCommGraphRegister(comm, base, baseSize, (void**)&regRecord), ret, fail);
    // 调用内部函数完成CollNet缓冲区注册
    NCCLCHECKGOTO(collnetRegisterBuffer(comm, userbuff, buffSize, type, regRecord, outRegBufFlag, outHandle), ret, fail);

    if (*outRegBufFlag) {  // 如果注册成功
      // 创建清理回调记录
      record = (struct ncclCollnetCleanupCallback*)malloc(sizeof(struct ncclCollnetCleanupCallback));  // 分配内存
      record->base.fn = cleanupCollnet;  // 设置清理函数指针
      record->comm = comm;  // 保存通信上下文
      record->reg = regRecord;  // 保存注册记录
      ncclIntruQueueEnqueue(cleanupQueue, (struct ncclCommCallback*)record);  // 加入清理队列
      *nCleanupQueueElts += 1;  // 增加队列元素计数
    } else {  // 如果注册失败
      NCCLCHECKGOTO(ncclCommGraphDeregister(comm, regRecord), ret, fail);  // 注销图形内存注册
    }
  }

exit:
  return ret;  // 返回结果
fail:
  *outRegBufFlag = 0;  // 设置标志为失败
  *outHandle = NULL;  // 设置句柄为NULL
  goto exit;  // 跳转到退出
}

// ============================================================================
// 函数: ncclCollnetDeregBuffer - 注销CollNet缓冲区
// 参数:
//   comm - 通信上下文
//   proxyconn - proxy连接器
//   handle - 内存句柄
// 返回: ncclResult_t - 操作结果状态码
// 说明: 通过proxy线程注销已注册的内存区域
// ============================================================================
ncclResult_t ncclCollnetDeregBuffer(struct ncclComm* comm, struct ncclProxyConnector* proxyconn, void* handle) {
  // 阻塞调用proxy的注销函数
  NCCLCHECK(ncclProxyCallBlocking(comm, proxyconn, ncclProxyMsgDeregister, &handle, sizeof(void*), NULL, 0));
  INFO(NCCL_REG, "rank %d - COLLNET deregistered buffer handle %p", comm->rank, handle);  // 输出日志
  return ncclSuccess;
}

// ============================================================================
// 函数: sendProxyRegBuffer - 发送端proxy注册缓冲区
// 参数:
//   connection - proxy连接结构
//   proxyState - proxy状态
//   reqBuff - 请求缓冲区（包含collnetRegInfo）
//   reqSize - 请求大小
//   respBuff - 响应缓冲区（返回句柄）
//   respSize - 响应大小
//   done - 完成标志（输出）
// 返回: ncclResult_t - 操作结果状态码
// 说明: 在proxy线程中注册发送端用户缓冲区，支持DMA-BUF和普通GDR
// ============================================================================
static ncclResult_t sendProxyRegBuffer(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState, void* reqBuff, int reqSize, void* respBuff, int respSize, int* done) {
  void* handle;  // 内存句柄
  struct collnetRegInfo* info = (struct collnetRegInfo*)reqBuff;  // 解析注册信息
  struct sendResources* resources = (struct sendResources*)(connection->transportResources);  // 获取发送端资源
  ncclResult_t ret = ncclSuccess;  // 初始化返回值
  bool needReg = true;  // 是否需要常规注册（默认需要）

  assert(reqSize == sizeof(struct collnetRegInfo));  // 验证请求大小
  assert(respSize == sizeof(void*));  // 验证响应大小

  int dmabuf_fd = -1;  // DMA-BUF文件描述符，初始为-1
#if CUDART_VERSION >= 11070  // CUDA 11.7及以上版本支持DMA-BUF
  /* DMA-BUF support */
  if (resources->useGdr && resources->useDmaBuf) {  // 如果使用GDR和DMA-BUF
    // 获取DMA-BUF文件描述符
    CUCHECKGOTO(cuMemGetHandleForAddressRange((void *)&dmabuf_fd, (CUdeviceptr)info->buffer, info->size, CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, getHandleForAddressRangeFlags(resources->useGdr)), ret, peermem);
    // 使用DMA-BUF注册内存
    NCCLCHECKGOTO(proxyState->ncclCollNet->regMrDmaBuf(resources->collNetComm, (void*)info->buffer, info->size, NCCL_PTR_CUDA, 0ULL, dmabuf_fd, &handle), ret, peermem);
    needReg = false;  // 不需要常规注册
  }
#endif
peermem:  // peer memory GDR路径的标签
  if (dmabuf_fd != -1) {  // 如果打开了DMA-BUF文件描述符
    (void)close(dmabuf_fd);  // 关闭文件描述符
    dmabuf_fd = -1;  // 重置为-1
  }
  if (needReg) {  // 如果需要常规注册
    // 使用常规的regMr函数注册内存（使用peer memory GDR）
    NCCLCHECKGOTO(proxyState->ncclCollNet->regMr(resources->collNetComm, (void*)info->buffer, info->size, NCCL_PTR_CUDA, &handle), ret, fail);
  }

exit:
  memcpy(respBuff, (void*)&handle, sizeof(void*));  // 将句柄复制到响应缓冲区
  *done = 1;  // 设置完成标志
  return ncclSuccess;  // 返回成功
fail:
  handle = NULL;  // 设置句柄为NULL表示失败
  goto exit;  // 跳转到退出
}

// ============================================================================
// 函数: recvProxyRegBuffer - 接收端proxy注册缓冲区
// 参数: 与sendProxyRegBuffer类似
// 返回: ncclResult_t - 操作结果状态码
// 说明: 在proxy线程中注册接收端用户缓冲区，支持DMA-BUF和普通GDR
// ============================================================================
static ncclResult_t recvProxyRegBuffer(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState, void* reqBuff, int reqSize, void* respBuff, int respSize, int* done) {
  void* handle;  // 内存句柄
  struct collnetRegInfo* info = (struct collnetRegInfo*)reqBuff;  // 解析注册信息
  struct recvResources* resources = (struct recvResources*)(connection->transportResources);  // 获取接收端资源
  ncclResult_t ret = ncclSuccess;  // 初始化返回值
  bool needReg = true;  // 是否需要常规注册

  assert(reqSize == sizeof(struct collnetRegInfo));  // 验证请求大小
  assert(respSize == sizeof(void*));  // 验证响应大小
  int dmabuf_fd = -1;  // DMA-BUF文件描述符
  #if CUDART_VERSION >= 11070  // CUDA 11.7及以上版本
  /* DMA-BUF support */
  if (resources->useGdr && resources->useDmaBuf) {  // 如果使用GDR和DMA-BUF
    // 获取DMA-BUF文件描述符
    CUCHECKGOTO(cuMemGetHandleForAddressRange((void *)&dmabuf_fd, (CUdeviceptr)info->buffer, info->size, CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, getHandleForAddressRangeFlags(resources->useGdr)), ret, peermem);
    // 使用DMA-BUF注册内存
    NCCLCHECKGOTO(proxyState->ncclCollNet->regMrDmaBuf(resources->collNetComm, (void*)info->buffer, info->size, NCCL_PTR_CUDA, 0ULL, dmabuf_fd, &handle), ret, peermem);
    needReg = false;  // 不需要常规注册
  }
#endif
peermem:  // peer memory GDR路径
  if (dmabuf_fd != -1) {  // 如果打开了DMA-BUF文件描述符
    (void)close(dmabuf_fd);  // 关闭文件描述符
    dmabuf_fd = -1;  // 重置
  }
  if (needReg) {  // 如果需要常规注册
    // 使用常规的regMr函数注册内存
    NCCLCHECKGOTO(proxyState->ncclCollNet->regMr(resources->collNetComm, (void*)info->buffer, info->size, NCCL_PTR_CUDA, &handle), ret, fail);
  }

exit:
  memcpy(respBuff, (void*)&handle, sizeof(void*));  // 复制句柄到响应
  *done = 1;  // 设置完成标志
  return ncclSuccess;
fail:
  handle = NULL;  // 设置句柄为NULL
  goto exit;
}

// ============================================================================
// 函数: sendProxyDeregBuffer - 发送端proxy注销缓冲区
// 参数:
//   connection - proxy连接结构
//   proxyState - proxy状态
//   reqBuff - 请求缓冲区（包含句柄）
//   reqSize - 请求大小
//   done - 完成标志（输出）
// 返回: ncclResult_t - 操作结果状态码
// ============================================================================
static ncclResult_t sendProxyDeregBuffer(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState, void* reqBuff, int reqSize, int* done) {
  void* handle;  // 内存句柄
  struct sendResources* resources = (struct sendResources*)(connection->transportResources);  // 获取发送端资源

  assert(reqSize == sizeof(void*));  // 验证请求大小
  memcpy(&handle, reqBuff, sizeof(void*));  // 从请求中提取句柄
  NCCLCHECK(proxyState->ncclCollNet->deregMr(resources->collNetComm, handle));  // 注销内存
  *done = 1;  // 设置完成标志
  return ncclSuccess;
}

// ============================================================================
// 函数: recvProxyDeregBuffer - 接收端proxy注销缓冲区
// 参数: 与sendProxyDeregBuffer类似
// 返回: ncclResult_t - 操作结果状态码
// ============================================================================
static ncclResult_t recvProxyDeregBuffer(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState, void* reqBuff, int reqSize, int* done) {
  void* handle;  // 内存句柄
  struct recvResources* resources = (struct recvResources*)(connection->transportResources);  // 获取接收端资源

  assert(reqSize == sizeof(void*));  // 验证请求大小
  memcpy(&handle, reqBuff, sizeof(void*));  // 从请求中提取句柄
  NCCLCHECK(proxyState->ncclCollNet->deregMr(resources->collNetComm, handle));  // 注销内存
  *done = 1;  // 设置完成标志
  return ncclSuccess;
}

// ============================================================================
// 函数: ncclCollNetChainBufferSetup - 设置CollNet链式缓冲区连接
// 参数:
//   comm - 通信上下文
// 返回: ncclResult_t - 操作结果状态码
// 说明: 建立CollNet Chain算法的P2P连接，用于树形归约
// ============================================================================
ncclResult_t ncclCollNetChainBufferSetup(ncclComm_t comm) {
  ncclResult_t ret = ncclSuccess;  // 初始化返回值
  char line[1024];  // 日志缓冲区

  if (comm->config.collnetEnable == 0) goto exit;  // 如果CollNet未启用，直接退出
  // Connect Collnet + chain
  // 连接CollNet + 链式拓扑
  // 第一阶段：连接上行（up）和下行（down）
  for (int c = 0; c < comm->nChannels; c++) {  // 遍历所有通道
    struct ncclChannel* channel = comm->channels + c;  // 获取通道
    // 建立P2P连接：1个上行peer，1个下行peer数组，isSend=0表示接收端
    NCCLCHECKGOTO(ncclTransportP2pConnect(comm, c, 1, &channel->collnetChain.up, 1, channel->collnetChain.down, 0), ret, fail);
  }
  // 设置P2P连接（接收端）
  NCCLCHECKGOTO(ncclTransportP2pSetup(comm, &comm->graphs[NCCL_ALGO_COLLNET_CHAIN], 0), ret, fail);
  // 第二阶段：连接下行和上行
  for (int c = 0; c < comm->nChannels; c++) {  // 遍历所有通道
    struct ncclChannel* channel = comm->channels + c;  // 获取通道
    // 建立P2P连接：1个下行peer，1个上行peer，isSend=1表示发送端
    NCCLCHECKGOTO(ncclTransportP2pConnect(comm, c, 1, channel->collnetChain.down, 1, &channel->collnetChain.up, 1), ret, fail);
  }
  // 设置P2P连接（发送端）
  NCCLCHECKGOTO(ncclTransportP2pSetup(comm, &comm->graphs[NCCL_ALGO_COLLNET_CHAIN], 1), ret, fail);

  // 构建日志信息字符串
  line[0] = '\0';  // 清空缓冲区
  for (int c = 0; c < comm->nChannels; c++) {  // 遍历所有通道
    struct ncclTree* chain = &comm->channels[c].collnetChain;  // 获取链式树结构
    // 格式化输出：[通道号] 下行rank -> 当前rank -> 上行rank
    snprintf(line + strlen(line), 1023 - strlen(line), " [%d] %d->%d->%d",
      c, chain->down[0], comm->rank, chain->up);
  }
  line[1023] = '\0';  // 确保字符串终止

  INFO(NCCL_INIT, "Connected Collnet Chains %s", line);  // 输出连接信息

exit:
  return ret;
fail:
  goto exit;
}

// ============================================================================
// 函数: ncclCollNetDirectBufferSetup - 设置CollNet直接缓冲区连接
// 参数:
//   comm - 通信上下文
// 返回: ncclResult_t - 操作结果状态码
// 说明: 建立CollNet Direct算法的P2P连接，用于直接的节点间通信
// ============================================================================
ncclResult_t ncclCollNetDirectBufferSetup(ncclComm_t comm) {
  ncclResult_t ret = ncclSuccess;  // 初始化返回值

  if (comm->config.collnetEnable == 0) goto exit;  // 如果CollNet未启用，直接退出

  // Connect intra-node CollNet + Direct
  // 连接节点内CollNet + Direct
  // 第一阶段：连接接收端
  for (int c = 0; c < comm->nChannels; c++) {  // 遍历所有通道
    struct ncclChannel* channelRecv = comm->channels + c;  // 获取接收通道
    // 建立P2P连接：NCCL_MAX_DIRECT_ARITY个上行peer，NCCL_MAX_DIRECT_ARITY个下行peer
    NCCLCHECKGOTO(ncclTransportP2pConnect(comm, c, NCCL_MAX_DIRECT_ARITY, channelRecv->collnetDirect.up, NCCL_MAX_DIRECT_ARITY, channelRecv->collnetDirect.down, 0), ret, fail);
  }
  // 设置P2P连接（接收端，isSend=0）
  NCCLCHECKGOTO(ncclTransportP2pSetup(comm, &comm->graphs[NCCL_ALGO_COLLNET_DIRECT], 0), ret, fail);

  // 第二阶段：连接发送端
  for (int c = 0; c < comm->nChannels; c++) {  // 遍历所有通道
    struct ncclChannel* channelSend = comm->channels + c;  // 获取发送通道
    // 建立P2P连接：NCCL_MAX_DIRECT_ARITY个下行peer，NCCL_MAX_DIRECT_ARITY个上行peer
    NCCLCHECKGOTO(ncclTransportP2pConnect(comm, c, NCCL_MAX_DIRECT_ARITY, channelSend->collnetDirect.down, NCCL_MAX_DIRECT_ARITY, channelSend->collnetDirect.up, 1), ret, fail);
  }
  // 设置P2P连接（发送端，isSend=1）
  NCCLCHECKGOTO(ncclTransportP2pSetup(comm, &comm->graphs[NCCL_ALGO_COLLNET_DIRECT], 1), ret, fail);

  INFO(NCCL_INIT, "rank %d Connected CollNet", comm->rank);  // 输出连接信息

exit:
  return ret;
fail:
  goto exit;
}

// ============================================================================
// 函数: collNetInitRailRankMap - 初始化Rail rank映射
// 参数:
//   comm - 通信上下文
// 返回: ncclResult_t - 操作结果状态码
// 说明: 建立用户rank和dense rank之间的双向映射，用于CollNet多rail通信
//       Dense rank考虑了head rank和非head rank的位置
// ============================================================================
static ncclResult_t collNetInitRailRankMap(ncclComm_t comm) {
  int rank = comm->rank;  // 当前rank
  uint64_t nonHeadMask = (1ull << comm->localRanks) - 1;  // 初始化非head掩码（所有本地rank）

  // 分配dense到用户rank的映射数组
  comm->collNetDenseToUserRank = ncclMemoryStackAlloc<int>(&comm->memPermanent, comm->nRanks);
  // 分配用户到dense rank的映射数组
  comm->collNetUserToDenseRank = ncclMemoryStackAlloc<int>(&comm->memPermanent, comm->nRanks);
  // initialize collNetUserToDenseRank[rank]
  // 初始化当前rank的dense rank为-1
  comm->collNetUserToDenseRank[rank] = -1;
  // 遍历所有head rank
  for (int h = 0; h < comm->collNetHeadsNum; h++) {
    // 从nonHeadMask中清除当前head的位
    nonHeadMask ^= 1ull << comm->rankToLocalRank[comm->collNetHeads[h]];
    // 如果当前rank是head，设置dense rank为head索引
    if (comm->collNetHeads[h] == rank) { comm->collNetUserToDenseRank[rank] = h; break; }
  }
  // 如果当前rank不是head
  if (comm->collNetUserToDenseRank[rank] == -1) {
    // 计算dense rank：统计nonHeadMask中比当前localRank小的rank数量
    comm->collNetUserToDenseRank[rank] = __builtin_popcountll(nonHeadMask & ((1ull << comm->localRank) - 1));
  }
  // 加上节点偏移：节点号 * 本地rank数
  comm->collNetUserToDenseRank[rank] += comm->node * comm->localRanks;

  // 全局收集所有rank的dense rank映射
  NCCLCHECK(bootstrapAllGather(comm->bootstrap, comm->collNetUserToDenseRank, sizeof(int)));
  // 构建反向映射：dense rank -> 用户rank
  for (int r = 0; r < comm->nRanks; r++) {
    comm->collNetDenseToUserRank[comm->collNetUserToDenseRank[r]] = r;
  }
  return ncclSuccess;
}

// ============================================================================
// 函数: ncclCollNetSetup - 设置CollNet通信
// 参数:
//   comm - 通信上下文
//   parent - 父通信上下文（用于资源共享）
//   graphs - 拓扑图数组
// 返回: ncclResult_t - 操作结果状态码
// 说明: 主要的CollNet设置函数，建立head rank、共享资源检查、通道初始化等
// ============================================================================
ncclResult_t ncclCollNetSetup(ncclComm_t comm, ncclComm_t parent, struct ncclTopoGraph* graphs[]) {
  ncclResult_t ret = ncclSuccess;  // 初始化返回值
  int rank = comm->rank;  // 当前rank
  int collNetSetupFail = 0;  // 设置失败标志
  bool share;  // 是否共享资源

  // 定义共享信息结构
  struct collnetShareInfo {
    int headPosition;  // head位置索引
    int isMaster;      // 是否为master（head rank）
  };
  struct collnetShareInfo* infos = NULL;  // 共享信息数组

  struct ncclTopoGraph* collNetGraph;  // CollNet拓扑图

  // ========== 阶段1: 确定head rank列表 ==========
  if (!comm->nvlsSupport) {  // 如果不支持NVLS
    collNetGraph = graphs[NCCL_ALGO_COLLNET_DIRECT];  // 使用Direct图
    NCCLCHECKGOTO(ncclCalloc(&comm->collNetHeads, collNetGraph->nChannels), ret, fail);  // 分配head数组
    uint64_t mask = 0;  // 用于去重的掩码
    // Head GPU index is always 0
    // head GPU索引总是0（每个节点的第一个本地rank）
    for (int c = 0; c < collNetGraph->nChannels; c++) {  // 遍历所有通道
      int head = collNetGraph->intra[c * comm->localRanks + 0];  // 获取该通道的head rank
      assert(comm->rankToNode[head] == comm->node);  // 断言head在同一节点
      uint64_t mask0 = mask;  // 保存旧掩码
      mask |= 1ull<<comm->rankToLocalRank[head];  // 设置head对应的位
      if (mask != mask0) comm->collNetHeads[comm->collNetHeadsNum++] = head;  // 如果是新的head，添加到列表
    }
  } else {  // 如果支持NVLS
    // Use the NVLS graph to get the head ranks for collnet setup. comm->nvlsHeads already has unique heads.
    // 使用NVLS图获取collnet设置的head rank。comm->nvlsHeads已经有唯一的head。
    // nHeads is the same on all the channels, see connectNvls function
    // 所有通道的nHeads相同，参见connectNvls函数
    collNetGraph = graphs[NCCL_ALGO_NVLS];  // 使用NVLS图
    NCCLCHECKGOTO(ncclCalloc(&comm->collNetHeads, collNetGraph->nChannels), ret, fail);
    comm->collNetHeadsNum = comm->channels[0].nvls.nHeads;  // 从NVLS通道获取head数量
    // Copy over comm->collNetHeads from comm->nvlsHeads since they are freed in different places.
    // 从comm->nvlsHeads复制到comm->collNetHeads，因为它们在不同地方被释放
    memcpy(comm->collNetHeads, comm->nvlsHeads, comm->collNetHeadsNum * sizeof(int));
  }

  // ========== 阶段2: 检查是否可以共享父通信域资源 ==========
  if (parent && parent->config.collnetEnable && parent->nNodes == comm->nNodes) {  // 如果有父通信域且都启用了CollNet
    if (!parent->shareResources) {  // 如果父通信域不允许共享
      collNetSetupFail = 1;  // 标记失败
      goto fail;
    }
    NCCLCHECKGOTO(ncclCalloc(&infos, comm->nRanks), ret, fail);  // 分配共享信息数组
    /* check whether child can share collnet resources of parent. Since parent builds each collnet communicator
     * based on heads with the same head position in each node, as long as the collnet heads of child comm
     * can match parent's heads, we can let child communicator share parent's collnet resources. */
    // 检查子通信域是否可以共享父通信域的collnet资源。由于父通信域基于每个节点中相同head位置的head构建每个collnet通信器，
    // 只要子通信域的collnet head能够匹配父通信域的head，我们就可以让子通信域共享父通信域的collnet资源。
    for (int h = 0; h < comm->collNetHeadsNum; ++h) {  // 遍历每个head
      int prev = INT_MIN;  // 上一个head位置
      struct collnetShareInfo* myinfo;  // 当前rank的共享信息

      share = true;  // 初始化共享标志为true
      myinfo = infos + comm->rank;  // 获取当前rank的共享信息
      memset(myinfo, 0, sizeof(struct collnetShareInfo));  // 清零
      /* find the child head position in parent collnet heads. */
      // 查找子通信域head在父通信域collnet head中的位置
      if (comm->collNetHeads[h] == comm->rank) {  // 如果当前rank是head
        myinfo->headPosition = -1;  // 初始化为-1
        myinfo->isMaster = 1;  // 标记为master
        // 在父通信域的head列表中查找匹配的位置
        for (int th = 0; th < parent->collNetHeadsNum; ++th)
          if (parent->topParentRanks[parent->collNetHeads[th]] == comm->topParentRanks[comm->rank]) {
            myinfo->headPosition = th;  // 找到匹配位置
            break;
          }
      }

      // 全局收集所有rank的共享信息
      NCCLCHECKGOTO(bootstrapAllGather(comm->bootstrap, infos, sizeof(struct collnetShareInfo)), ret, fail);
      // 验证所有master rank的head位置是否一致
      for (int i = 0; i < comm->nRanks; ++i) {
        if (infos[i].isMaster) {  // 如果是master
          if (prev == INT_MIN)  // 第一个master
            prev = infos[i].headPosition;  // 记录位置

          if (infos[i].headPosition == -1 || prev != infos[i].headPosition) {  // 如果位置不匹配
            share = false;  // 不能共享
            break;
          }
        }
      }

      if (share) {  // 如果可以共享
        if (myinfo->isMaster) {  // 如果当前rank是master
          comm->collNetSharedRes = parent->collNetSharedRes;  // 共享父通信域的CollNet资源
          for (int c = 0; c < comm->nChannels; ++c)  // 初始化所有通道
            NCCLCHECKGOTO(initCollnetChannel(comm, c, parent, true), ret, fail);  // 初始化通道（共享模式）
        }

        NCCLCHECKGOTO(collNetInitRailRankMap(comm), ret, fail);  // 初始化rail rank映射
      } else {  // 如果不能共享
        collNetSetupFail = 1;  // 标记失败
        if (comm->rank == 0) {  // rank 0输出警告
          WARN("Child comms (nRanks %d) fails to share parent comms (nRanks %d) sharp resources", comm->nRanks, parent->nRanks);
        }
        goto fail;
      }
    }
    share = true;  // 标记为共享模式
  } else {  // ========== 阶段3: 不共享资源，创建新的CollNet通信器 ==========
    /* this allocated buffer will be freed on proxy side */
    // 这个分配的缓冲区将在proxy端释放
    NCCLCHECK(ncclCalloc(&comm->collNetSharedRes, 1));  // 分配共享资源结构
    comm->collNetSharedRes->nChannels = comm->nChannels;  // 设置通道数
    comm->collNetSharedRes->buffSize = comm->buffSizes[NCCL_PROTO_SIMPLE];  // 设置缓冲区大小

    NCCLCHECKGOTO(collNetInitRailRankMap(comm), ret, fail);  // 初始化rail rank映射

    for (int c = 0; c < comm->nChannels; c++) {  // 遍历所有通道
      struct ncclChannel* channel = comm->channels + c;  // 获取通道
      NCCLCHECKGOTO(initCollnetChannel(comm, c, parent, false), ret, fail);  // 初始化通道（非共享模式）
      for (int h = 0; h < comm->collNetHeadsNum; h++) {  // 遍历所有head
        const int head = comm->collNetHeads[h];  // 获取head rank
        ncclConnect connect;  // 连接信息
        // 设置接收端CollNet连接
        collNetSetupFail |= ncclTransportCollNetSetup(comm, collNetGraph, channel, head, head, h, collNetRecv, &connect);
        // 如果接收端成功，设置发送端CollNet连接
        if (!collNetSetupFail) collNetSetupFail |= ncclTransportCollNetSetup(comm, collNetGraph, channel, head, head, h, collNetSend, &connect);
      }
      // Verify CollNet setup across ranks after trying the first channel
      // 在尝试第一个通道后验证所有rank的CollNet设置
      if (c == 0) {
        NCCLCHECKGOTO(ncclTransportCollNetCheck(comm, collNetSetupFail), ret, fail);
      }
    }
    share = false;  // 标记为非共享模式
  }

  // ========== 阶段4: 设置支持矩阵（归约操作和数据类型支持） ==========
  if (share) {  // 如果共享资源
    // 从父通信域复制支持矩阵
    memcpy(comm->collNetSupportMatrix, parent->collNetSupportMatrix, sizeof(comm->collNetSupportMatrix));
  } else {  // 不共享，需要重新查询
    do {
      /* Initialize all entries in collNetSupportMatrix[redop][type]. Since some
      ranks don't connect to sharp we enable a (redop,type) if any rank claims
      support. */
      // 初始化collNetSupportMatrix[redop][type]的所有条目。由于某些rank不连接到sharp，
      // 我们启用任何rank声称支持的(redop,type)组合。
      uint8_t(*matrix)[4][ncclNumTypes];  // 支持矩阵：rank x op x type
      bool isHead = false;  // 是否为head rank
      matrix = nullptr;  // 初始化为空
      NCCLCHECKGOTO(ncclCalloc(&matrix, comm->nRanks), ret, matrix_end);  // 分配矩阵
      // 检查当前rank是否为head
      for (int h = 0; h < comm->collNetHeadsNum; h++) isHead |= (comm->collNetHeads[h] == comm->rank);
      if (isHead) {  // 如果是head rank，查询支持情况
        for (int ty=0; ty < ncclNumTypes; ty++) {  // 遍历所有数据类型
          for (int op=0; op < 4; op++) {  // 遍历所有归约操作（最多4个）
            int support = 0;  // 支持标志
            NCCLCHECKGOTO(collNetReduceSupport(comm, (ncclDataType_t)ty, (ncclRedOp_t)op, &support), ret, matrix_end);  // 查询支持
            // bit 0 = not supported, bit 1 = supported
            // 位0 = 不支持，位1 = 支持
            matrix[rank][op][ty] = 1<<(support ? 1 : 0);  // 编码支持信息
          }
        }
      }
      // 全局收集所有rank的支持矩阵
      NCCLCHECKGOTO(bootstrapAllGather(comm->bootstrap, matrix, sizeof(*matrix)), ret, matrix_end);
      // 计算最终的支持矩阵
      for (int ty=0; ty < ncclNumTypes; ty++) {  // 遍历所有数据类型
        for (int op=0; op < 4; op++) {  // 遍历所有归约操作
          uint8_t accum = 0;  // 累积位
          for (int r=0; r < comm->nRanks; r++) accum |= matrix[r][op][ty];  // OR所有rank的支持位
          // We support (redop, type) if some rank supports it and no rank doesn't support it
          // 如果某些rank支持且没有rank不支持，我们支持(redop, type)
          comm->collNetSupportMatrix[op][ty] = (accum == (1<<1));  // 只有当所有支持位都是"支持"时才为true
        }
      }
    matrix_end:  // 清理标签
      free(matrix);  // 释放矩阵
      if (ret != ncclSuccess) goto fail;  // 如果有错误，跳转到失败处理
    } while (0);  // 单次执行的do-while（用于跳转）
  }

  // Verify CollNet setup across ranks after trying all channels
  // 在尝试所有通道后验证所有rank的CollNet设置
  NCCLCHECKGOTO(ncclTransportCollNetCheck(comm, collNetSetupFail), ret, fail);
  TRACE(NCCL_INIT, "rank %d Connected inter-node CollNet", rank);  // 输出跟踪日志

exit:
  free(infos);  // 释放共享信息数组
  return ret;  // 返回结果
fail:
  ncclTransportCollNetFree(comm);  // 释放CollNet资源
  comm->config.collnetEnable = 0;  // 禁用CollNet
  goto exit;  // 跳转到退出
}

// ============================================================================
// 结构: collNetTransport - CollNet传输层接口定义
// 说明: 这是NCCL传输层抽象接口的CollNet实现，注册到NCCL框架中
// 字段说明:
//   name - 传输层名称（"COL" = CollNet）
//   canConnect - 检查是否可以连接的函数（CollNet不支持P2P，始终返回0）
//   send/recv - 发送端和接收端的函数指针数组
// ============================================================================
struct ncclTransport collNetTransport = {
  "COL",  // 传输层名称标识
  canConnect,  // 连接检查函数（CollNet专用于集合通信，不支持P2P）
  // 发送端函数指针数组：setup, connect, free, NULL, proxySetup, proxyConnect, proxyFree, proxyProgress, regBuffer, deregBuffer
  { sendSetup, sendConnect, sendFree, NULL, sendProxySetup, sendProxyConnect, sendProxyFree, sendProxyProgress, sendProxyRegBuffer, sendProxyDeregBuffer },
  // 接收端函数指针数组：setup, connect, free, NULL, proxySetup, proxyConnect, proxyFree, proxyProgress, regBuffer, deregBuffer
  { recvSetup, recvConnect, recvFree, NULL, recvProxySetup, recvProxyConnect, recvProxyFree, recvProxyProgress, recvProxyRegBuffer, recvProxyDeregBuffer }
};
// ============================================================================
// 文件结束 - CollNet传输层实现完成
// ============================================================================
