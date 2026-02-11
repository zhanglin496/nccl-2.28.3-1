/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 * 版权所有 (c) 2016-2022，NVIDIA 公司。保留所有权利。
 *
 * See LICENSE.txt for license information
 * 有关许可信息，请参见 LICENSE.txt 文件
 ************************************************************************/

// 包含通信域头文件，定义 ncclComm 结构体和通信相关接口
#include "comm.h"
// 包含网络头文件，定义网络传输层的接口和数据结构
#include "net.h"
// 包含图头文件，定义 CUDA Graph 相关接口
#include "graph.h"
// 包含代理头文件，定义代理操作的接口
#include "proxy.h"
// 包含集合通信头文件，定义集合操作的相关宏和函数
#include "collectives.h"
// 包含 GPU Direct RDMA 包装头文件，定义 GDR 相关接口
#include "gdrwrap.h"
// 包含共享内存工具头文件，定义共享内存操作接口
#include "shmutils.h"
// 包含 P2P 头文件，定义点对点通信接口
#include "p2p.h"
// 包含性能分析头文件，定义性能分析接口
#include "profiler.h"
// 包含传输层头文件，定义传输层的接口和数据结构
#include "transport.h"
// 包含共享内存头文件，定义共享内存通信接口
#include "shm.h"
// 包含断言头文件，提供断言宏
#include <assert.h>
// 包含内联注册头文件，定义内存注册的内联函数
#include "register_inline.h"

// 静态断言：检查网络句柄的大小是否超过连接信息的大小限制
// ncclNetHandle_t: 网络句柄类型，用于存储网络连接的标识信息
// CONNECT_SIZE: 连接信息的最大大小（通常为 1024 字节）
// 如果超过大小限制，编译时会报错，防止连接信息过大导致问题
static_assert(sizeof(ncclNetHandle_t) <= CONNECT_SIZE, "NET Connect info is too large");

// 定义内存映射类型的枚举值，用于标识不同的内存区域
// NCCL_NET_MAP_HOSTMEM (0): 主机内存，用于 CPU 访问
#define NCCL_NET_MAP_HOSTMEM 0
// NCCL_NET_MAP_DEVMEM (1): 设备内存（GPU 内存），用于 GPU 访问
#define NCCL_NET_MAP_DEVMEM 1
// NCCL_NET_MAP_SHARED_HOSTMEM (2): 共享主机内存，用于进程间共享的主机内存
#define NCCL_NET_MAP_SHARED_HOSTMEM 2
// NCCL_NET_MAP_SHARED_DEVMEM (3): 共享设备内存，用于进程间共享的 GPU 内存
#define NCCL_NET_MAP_SHARED_DEVMEM 3
// NCCL_NET_MAP_GDCMEM (4): GPU Direct Cuda 内存，特殊的 GPU 直接访问内存
#define NCCL_NET_MAP_GDCMEM 4
// NCCL_NET_MAP_MEMS (5): 内存类型的总数，用于数组大小定义
#define NCCL_NET_MAP_MEMS 5

// 定义偏移量掩码，用于编码内存类型信息到 32 位偏移量中
// 第 30 位（0x40000000）：标记是否为设备内存
#define NCCL_NET_MAP_MASK_DEVMEM 0x40000000
// 第 31 位（0x80000000）：标记是否为共享内存
#define NCCL_NET_MAP_MASK_SHARED 0x80000000
// 第 29 位（0x20000000）：标记该偏移量是否已被使用
#define NCCL_NET_MAP_MASK_USED   0x20000000
// 低 29 位（0x1fffffff）：存储实际的偏移量值（最大 512MB-1）
#define NCCL_NET_MAP_MASK_OFFSET 0x1fffffff

// 宏：从偏移量中提取内存库（bank）索引
// 参数说明：
//   - mapStruct: connectMap 结构体指针
//   - offsetName: 偏移量字段名称
// 实现原理：右移 30 位，提取高 2 位（第 30-31 位），得到内存库索引（0-3）
#define NCCL_NET_MAP_OFFSET_BANK(mapStruct, offsetName) \
  ((mapStruct)->offsets.offsetName >> 30)

// 宏：检查偏移量是否为 NULL
// 参数说明：
//   - mapStruct: connectMap 结构体指针
//   - offsetName: 偏移量字段名称
// 实现原理：右移 29 位后检查是否为 0，即检查第 29 位是否为 0
// 如果第 29 位（USED 标志）为 0，说明该偏移量未使用，返回 NULL
#define NCCL_NET_MAP_OFFSET_NULL(mapStruct, offsetName) \
  (((mapStruct)->offsets.offsetName >> 29) == 0)

// 宏：从映射结构中获取指针（CPU 或 GPU 指针）
// 参数说明：
//   - mapStruct: connectMap 结构体指针
//   - cpuOrGpu: 指针类型前缀（cpu 或 gpu），用于选择 cpuPtr 或 gpuPtr
//   - offsetName: 偏移量字段名称
// 实现原理：
//   1. 首先检查偏移量是否为 NULL
//   2. 如果不为 NULL，计算指针地址：
//      a. 使用 NCCL_NET_MAP_OFFSET_BANK 提取内存库索引
//      b. 从对应的内存库获取基址（cpuPtr 或 gpuPtr）
//      c. 加上偏移量的低 29 位作为实际偏移
#define NCCL_NET_MAP_GET_POINTER(mapStruct, cpuOrGpu, offsetName) \
  (NCCL_NET_MAP_OFFSET_NULL(mapStruct, offsetName) ? NULL : \
   (mapStruct)->mems[NCCL_NET_MAP_OFFSET_BANK(mapStruct, offsetName)].cpuOrGpu##Ptr + ((mapStruct)->offsets.offsetName & NCCL_NET_MAP_MASK_OFFSET))

// 宏：检查偏移量是否指向设备内存
// 参数说明：
//   - mapStruct: connectMap 结构体指针
//   - offsetName: 偏移量字段名称
// 实现原理：检查第 30 位（DEVMEM 标志）是否被设置
#define NCCL_NET_MAP_DEV_MEM(mapStruct, offsetName) \
  (((mapStruct)->offsets.offsetName & NCCL_NET_MAP_MASK_DEVMEM) != 0)

// 宏：向映射结构中添加一个指针，分配内存空间
// 参数说明：
//   - mapStruct: connectMap 结构体指针
//   - shared: 是否为共享内存（0=非共享，1=共享）
//   - dev: 是否为设备内存（0=主机内存，1=设备内存）
//   - memSize: 要分配的内存大小
//   - offsetName: 要设置的偏移量字段名称
// 实现原理：
//   1. 计算 bank 值：设置 USED 标志，根据 dev 设置 DEVMEM 标志，根据 shared 设置 SHARED 标志
//   2. 如果是非共享内存（shared == 0）：
//      a. 对于设备内存：使用当前 DEVMEM 的大小作为偏移，然后增加 DEVMEM 的大小
//      b. 对于主机内存：使用当前 HOSTMEM 的大小作为偏移，然后增加 HOSTMEM 的大小
//   3. 如果是共享内存（shared == 1）：
//      a. 直接使用 bank 值作为偏移（共享内存的偏移从 0 开始）
//   4. 使用 do-while(0) 包装，使宏可以像函数一样安全使用
#define NCCL_NET_MAP_ADD_POINTER(mapStruct, shared, dev, memSize, offsetName) do { \
    int bank = NCCL_NET_MAP_MASK_USED + (dev)*NCCL_NET_MAP_MASK_DEVMEM + (shared)*NCCL_NET_MAP_MASK_SHARED; \
    if ((shared) == 0) { \
      if (dev) { \
        (mapStruct)->offsets.offsetName = bank + (mapStruct)->mems[NCCL_NET_MAP_DEVMEM].size; \
        (mapStruct)->mems[NCCL_NET_MAP_DEVMEM].size += memSize; \
      } else { \
        (mapStruct)->offsets.offsetName = bank + (mapStruct)->mems[NCCL_NET_MAP_HOSTMEM].size; \
        (mapStruct)->mems[NCCL_NET_MAP_HOSTMEM].size += memSize; \
      } \
    } else { \
      (mapStruct)->offsets.offsetName = bank; \
    } \
} while (0);

// 结构体：连接映射的内存信息
// 用于管理一个内存区域的主机和设备指针，以及相关的 IPC 描述符
struct connectMapMem{
  // GPU 指针，指向设备内存（GPU 内存）的地址
  char* gpuPtr;
  // CPU 指针，指向主机内存（CPU 可访问）的地址
  char* cpuPtr;
  // 内存区域的大小（字节数）
  int size;
  // IPC 描述符，用于跨进程共享 GPU 内存
  ncclIpcDesc ipcDesc;
  // 附加描述符，用于附加到已存在的共享内存段
  ncclShmIpcDesc_t attachDesc;
  // 创建描述符，用于创建新的共享内存段
  ncclShmIpcDesc_t createDesc;
};

// 结构体：连接映射
// 用于管理网络连接中的所有内存区域和偏移量
struct connectMap {
  // 标志：连接的对等体是否在同一进程中
  // sameProcess=1 表示同一进程，可以使用更高效的通信方式（如共享内存）
  int sameProcess;
  // 标志：是否使用共享内存
  // shared=1 表示使用共享内存进行通信
  int shared;
  // CUDA 设备编号，指定关联的 GPU 设备
  int cudaDev;
  // 偏移量的高 3 位决定内存库：
  // 001 表示主机内存，011 表示设备内存，101 表示共享主机内存，111 表示共享设备内存
  // First 3 bits of offsets determine the mem bank. 001 is host mem, 011 is dev mem, 101 is shared host mem and 111 is shared dev mem.
  struct connectMapMem mems[NCCL_NET_MAP_MEMS];
  // 偏移量结构，高 3 位表示内存库，111 表示 NULL
  // Offsets. 3 MSBs indicate mem bank, 111 indicates NULL.
  struct {
    // 发送内存的偏移量，指向 ncclSendMem 结构体
    uint32_t sendMem;
    // 接收内存的偏移量，指向 ncclRecvMem 结构体
    uint32_t recvMem;
    // 不同协议的缓冲区偏移量数组
    // NCCL_NUM_PROTOCOLS: 协议数量（如 SIMPLE, LL, LL128）
    uint32_t buffs[NCCL_NUM_PROTOCOLS];
  } offsets;
};

// 结构体：发送端网络资源
// 包含网络发送所需的所有资源信息
struct sendNetResources {
  // 连接映射，管理所有相关的内存区域
  struct connectMap map;
  // 网络发送通信句柄，由网络插件提供，用于实际的网络发送操作
  void* netSendComm;
  // 发送内存指针，指向 ncclSendMem 结构体，用于存储发送控制信息
  struct ncclSendMem* sendMem;
  // 接收内存指针，指向 ncclRecvMem 结构体，用于存储接收控制信息
  struct ncclRecvMem* recvMem;

  // 传输平面（Transport Plane）的 rank 编号
  int tpRank;
  // 传输平面的本地 rank 编号（在同一节点内的编号）
  int tpLocalRank;
  // 传输平面的远程 rank 编号（对等体的 rank）
  int tpRemoteRank;
  // 网络设备编号，指定使用哪个网络设备（如 IB 设备）
  int netDev;
  // GPU Direct RDMA (GDR) 使用模式
  // ncclTopoGdrMode 枚举值：NCCL_TOPO_GDR_NONE=不使用，NCCL_TOPO_GDR_DRAM=通过 DRAM，NCCL_TOPO_GDR_CUDA=直接 CUDA
  enum ncclTopoGdrMode useGdr;
  // 是否使用 DMA 缓冲区
  // useDmaBuf=1 表示使用 DMA-BUF，允许网卡直接访问 GPU 内存
  int useDmaBuf;
  // 最大接收数量，控制可以挂起的接收操作数量
  int maxRecvs;
  // GDC（GPU Direct Cache）同步指针，用于同步 GPU 和网卡之间的缓存
  uint64_t* gdcSync;
  // GDR 描述符，用于 GPU Direct RDMA 操作
  void* gdrDesc;
  // 标志：是否使用共享内存
  int shared;
  // 通道 ID，标识使用哪个通信通道
  int channelId;
  // 连接索引，标识这是第几个连接
  int connIndex;
  // 不同协议的缓冲区指针数组
  // NCCL_NUM_PROTOCOLS: 协议数量
  char* buffers[NCCL_NUM_PROTOCOLS];
  // 不同协议的缓冲区大小数组
  int buffSizes[NCCL_NUM_PROTOCOLS];
  // 不同协议的内存句柄数组，用于网络设备的内存注册
  void* mhandles[NCCL_NUM_PROTOCOLS];
  // 步进计数器，用于跟踪通信进度
  uint64_t step;
  // LL（Long Jump）协议上次清理的时间戳
  uint64_t llLastCleaning;
  // 网络设备版本号
  int netDeviceVersion;
  // 网络设备类型（如 IB, RoCE, Socket 等）
  ncclNetDeviceType netDeviceType;
  // 网络设备句柄，由网络插件提供
  ncclNetDeviceHandle_t* netDeviceHandle;
  // 最大点对点传输字节数
  size_t maxP2pBytes;
};

// 结构体：接收端网络资源
// 包含网络接收所需的所有资源信息
struct recvNetResources {
  // 连接映射，管理所有相关的内存区域
  struct connectMap map;
  // 网络监听通信句柄，用于监听传入的连接请求
  void* netListenComm;
  // 网络接收通信句柄，用于实际的网络接收操作
  void* netRecvComm;
  // 发送内存指针，指向 ncclSendMem 结构体
  struct ncclSendMem* sendMem;
  // 接收内存指针，指向 ncclRecvMem 结构体
  struct ncclRecvMem* recvMem;

  // 传输平面（Transport Plane）的 rank 编号
  int tpRank;
  // 传输平面的本地 rank 编号（在同一节点内的编号）
  int tpLocalRank;
  // 传输平面的远程 rank 编号（对等体的 rank）
  int tpRemoteRank;
  // 传输平面的远程代理 rank 编号（对等体的代理 rank）
  int tpRemoteProxyRank;
  // 网络设备编号
  int netDev;
  // GPU Direct RDMA (GDR) 使用模式
  enum ncclTopoGdrMode useGdr;
  // 是否使用 DMA 缓冲区
  int useDmaBuf;
  // 是否需要刷新缓存
  // needFlush=1 表示需要手动刷新 GPU 缓存以确保数据一致性
  int needFlush;
  // 最大接收数量
  int maxRecvs;
  // GDC（GPU Direct Cache）同步指针
  uint64_t* gdcSync;
  // GDC 刷新指针，用于触发缓存刷新
  uint64_t* gdcFlush;
  // GDR 描述符
  void* gdrDesc;
  // 标志：是否使用共享内存
  int shared;
  // 通道 ID
  int channelId;
  // 连接索引
  int connIndex;
  // 不同协议的缓冲区指针数组
  char* buffers[NCCL_NUM_PROTOCOLS];
  // 不同协议的缓冲区大小数组
  int buffSizes[NCCL_NUM_PROTOCOLS];
  // 不同协议的内存句柄数组
  void* mhandles[NCCL_NUM_PROTOCOLS];
  // 步进计数器
  uint64_t step;
  // LL 协议上次清理的时间戳
  uint64_t llLastCleaning;
  // 网络设备版本号
  int netDeviceVersion;
  // 网络设备类型
  ncclNetDeviceType netDeviceType;
  // 网络设备句柄
  ncclNetDeviceHandle_t* netDeviceHandle;
  // 最大点对点传输字节数
  size_t maxP2pBytes;
};

// 结构体：网络注册信息
// 用于存储内存注册的相关信息
struct netRegInfo {
  // 缓冲区地址（转换为整数类型）
  uintptr_t buffer;
  // 缓冲区大小
  size_t size;
};

/* Determine if two peers can communicate with NET */
/* 判断两个对等体是否可以通过 NET（网络传输层）进行通信 */
// 函数功能：检查两个 rank 之间是否可以建立网络连接
// 参数说明：
//   - ret: 输出参数，返回是否可以连接（1=可以连接，0=不可以连接）
//   - comm: 通信上下文指针
//   - graph: 拓扑图指针（本函数中未使用）
//   - info1: 第一个对等体的信息
//   - info2: 第二个对等体的信息
// 返回值：ncclSuccess 表示成功
static ncclResult_t canConnect(int* ret, struct ncclComm* comm, struct ncclTopoGraph* graph, struct ncclPeerInfo* info1, struct ncclPeerInfo* info2) {
  // 默认返回可以连接
  *ret = 1;
  // 检查两个对等体是否在同一主机上
  // hostHash: 主机哈希值，用于标识主机
  if (info1->hostHash == info2->hostHash) {
    // If on the same host, check intra-node net is not disabled.
    // 如果在同一主机上，检查节点内网络是否被禁用
    // ncclTopoCheckNet: 拓扑检查函数，验证网络连接是否可用
    NCCLCHECK(ncclTopoCheckNet(comm->topo, info1->rank, info2->rank, ret));
  }
  // 返回成功状态码
  return ncclSuccess;
}

// 定义参数：是否使用共享缓冲区
// NetSharedBuffers: 参数名称
// "NET_SHARED_BUFFERS": 环境变量名称
// -2: 默认值，表示自动决定是否使用共享缓冲区
NCCL_PARAM(NetSharedBuffers, "NET_SHARED_BUFFERS", -2);
// 定义参数：是否使用共享通信
// NetSharedComms: 参数名称
// "NET_SHARED_COMMS": 环境变量名称
// 1: 默认值，表示启用共享通信
NCCL_PARAM(NetSharedComms, "NET_SHARED_COMMS", 1);

// 结构体：设置请求
// 用于在连接建立阶段交换配置信息
struct setupReq {
  // 传输平面的 rank 编号
  int tpRank;
  // 传输平面的本地 rank 编号
  int tpLocalRank;
  // 传输平面的远程 rank 编号
  int tpRemoteRank;
  // 标志：是否使用共享内存
  int shared;
  // 网络设备编号
  int netDev;
  // GPU Direct RDMA 使用模式
  enum ncclTopoGdrMode useGdr;
  // 标志：是否需要刷新缓存
  int needFlush;
  // 通道 ID
  int channelId;
  // 连接索引
  int connIndex;
};

// 定义参数：是否使用可选的接收完成
// NetOptionalRecvCompletion: 参数名称
// "NET_OPTIONAL_RECV_COMPLETION": 环境变量名称
// 1: 默认值，表示启用可选接收完成优化
NCCL_PARAM(NetOptionalRecvCompletion, "NET_OPTIONAL_RECV_COMPLETION", 1);

// 静态断言：检查连接信息的大小是否足够
// 确保 ncclConnect 结构体能够容纳 ncclNetHandle_t 和 useGdr 标志
static_assert(sizeof(ncclNetHandle_t) + sizeof(int) <= CONNECT_SIZE, "Not large enough ncclConnect to hold ncclNetHandle_t and useGdr flag");

// Common function to initialize network attributes from a ncclComm
// 通用函数：从 ncclComm 初始化网络属性
// 参数说明：
//   - comm: 通信上下文指针
//   - conn: 连接器指针
//   - netAttr: 输出参数，返回网络属性
static void populateCommNetAttrs(struct ncclComm* comm, struct ncclConnector* conn, ncclNetAttr_t* netAttr) {
  // 初始化网络属性为默认值
  *netAttr = NCCL_NET_ATTR_INIT;
  // 设置发送端最小并发对等体数量为 1
  netAttr->sendCommAttr.minConcurrentPeers = 1;
  // 设置发送端每个对等体的最小流数量为 1
  netAttr->sendCommAttr.minFlowsPerPeer = 1;

  // 设置接收端最小并发对等体数量为 1
  netAttr->recvCommAttr.minConcurrentPeers = 1;
  // 设置接收端每个对等体的最小流数量为 1
  netAttr->recvCommAttr.minFlowsPerPeer = 1;

  // 检查是否为仅 P2P 模式
  // p2pOnly=1 表示只使用点对点通信，不使用集合通信
  if (conn->p2pOnly) {
    // 计算最大并发对等体数量
    // p2pnChannels: P2P 通道数量
    // NCCL_MAX_DEV_WORK_P2P_PER_BATCH: 每批次最大 P2P 工作项数
    size_t maxConcPeers = comm->p2pnChannels * NCCL_MAX_DEV_WORK_P2P_PER_BATCH;
    // 如果 rank 总数小于计算值，使用 rank 总数
    if (comm->nRanks < maxConcPeers) maxConcPeers = comm->nRanks;

    // 设置发送端最大并发对等体数量
    netAttr->sendCommAttr.maxConcurrentPeers = maxConcPeers;
    // 设置发送端每个对等体的最大流数量
    netAttr->sendCommAttr.maxFlowsPerPeer = comm->p2pnChannelsPerPeer;
    // 设置接收端最大并发对等体数量
    netAttr->recvCommAttr.maxConcurrentPeers = maxConcPeers;
    // 设置接收端每个对等体的最大流数量
    netAttr->recvCommAttr.maxFlowsPerPeer = comm->p2pnChannelsPerPeer;
    // 设置支持的操作类型（使用位掩码）
    // BIT(ncclFuncSend): 发送操作
    // BIT(ncclFuncRecv): 接收操作
    // BIT(ncclFuncAlltoAll): 全对全操作
    // BIT(ncclFuncScatter): 散发操作
    // BIT(ncclFuncGather): 收集操作
    netAttr->op = BIT(ncclFuncSend) | BIT(ncclFuncRecv) |
                  BIT(ncclFuncAlltoAll) | BIT(ncclFuncScatter) | BIT(ncclFuncGather);
  } else {
    // 非仅 P2P 模式，计算最大并发对等体数量
    // NCCL_MAX_TREE_ARITY: 树形拓扑的最大分支数
    // (NCCL_MAX_TREE_ARITY - 1) * 2: 考虑上行和下行连接
    size_t maxConcPeers = (NCCL_MAX_TREE_ARITY - 1) * 2;
    // 如果 rank 总数小于计算值，使用 rank 总数
    if (comm->nRanks < maxConcPeers) maxConcPeers = comm->nRanks;
    // 设置发送端最大并发对等体数量
    netAttr->sendCommAttr.maxConcurrentPeers = maxConcPeers;
    // 设置发送端每个对等体的最大流数量（使用所有通道）
    netAttr->sendCommAttr.maxFlowsPerPeer = comm->nChannels;
    // 设置接收端最大并发对等体数量
    netAttr->recvCommAttr.maxConcurrentPeers = maxConcPeers;
    // 设置接收端每个对等体的最大流数量（使用所有通道）
    netAttr->recvCommAttr.maxFlowsPerPeer = comm->nChannels;
  }
}

// Apply the netAttr to the netComm
// 函数功能：将网络属性应用到网络通信句柄
// 参数说明：
//   - proxyState: 代理状态指针
//   - netAttr: 网络属性指针
void setNetAttrs(struct ncclProxyState* proxyState, ncclNetAttr_t* netAttr)
{
  // 检查网络插件是否提供了设置网络属性的函数
  if (proxyState->ncclNet->setNetAttr) {
    // 调用网络插件的 setNetAttr 函数设置网络属性
    // netContext: 网络上下文
    // netAttr: 要设置的网络属性
    proxyState->ncclNet->setNetAttr(proxyState->netContext, netAttr);
    // 保存网络属性到代理状态中
    proxyState->netAttr = *netAttr;
  }
}

// 函数功能：打印网络属性信息（用于调试）
// 参数说明：
//   - netAttr: 网络属性指针
//   - task: 任务名称，用于日志输出
void printNetAttrs(ncclNetAttr_t* netAttr, const char *task)
{
  // 计算操作名称缓冲区的长度
  // ncclNumFuncs: 操作的数量（Send, Recv, AllReduce 等）
  // 每个操作名称最多 32 字节
  const int opBufLen = ncclNumFuncs*32;
  // 操作名称缓冲区，初始化为空字符串
  char opBuf[opBufLen] = "";
  // 计算算法名称缓冲区的长度
  // NCCL_NUM_ALGORITHMS: 算法数量（Ring, Tree, CollNet 等）
  const int algoBufLen = NCCL_NUM_ALGORITHMS*32;
  // 算法名称缓冲区
  char algoBuf[algoBufLen] = "";
  // 计算协议名称缓冲区的长度
  // NCCL_NUM_PROTOCOLS: 协议数量（Simple, LL, LL128 等）
  const int protoBufLen = NCCL_NUM_PROTOCOLS*32;
  // 协议名称缓冲区
  char protoBuf[protoBufLen] = "";

  // 将操作位掩码转换为字符串
  // netAttr->op: 操作位掩码
  // MASK(ncclNumFuncs): 生成操作数量的掩码
  // ncclFuncToString: 将操作编号转换为字符串的函数
  // "*": 分隔符
  ncclBitsToString(netAttr->op, MASK(ncclNumFuncs), (const char* (*)(int))ncclFuncToString, opBuf, opBufLen, "*");
  // 将算法位掩码转换为字符串
  ncclBitsToString(netAttr->algo, MASK(NCCL_NUM_ALGORITHMS), ncclAlgoToString, algoBuf, algoBufLen, "*");
  // 将协议位掩码转换为字符串
  ncclBitsToString(netAttr->proto, MASK(NCCL_NUM_PROTOCOLS), ncclProtoToString, protoBuf, protoBufLen, "*");

  // 输出网络属性的跟踪日志
  // 显示发送端和接收端的并发对等体数量、流数量，以及支持的操作、算法、协议
  TRACE(NCCL_NET, "%s hints, send peers/flows: [%d-%d][%d-%d] recv peers/flows: [%d-%d][%d-%d] op: %s algo: %s proto: %s",
        task, netAttr->sendCommAttr.minConcurrentPeers, netAttr->sendCommAttr.maxConcurrentPeers,
        netAttr->sendCommAttr.minFlowsPerPeer, netAttr->sendCommAttr.maxFlowsPerPeer,
        netAttr->recvCommAttr.minConcurrentPeers, netAttr->recvCommAttr.maxConcurrentPeers,
        netAttr->recvCommAttr.minFlowsPerPeer, netAttr->recvCommAttr.maxFlowsPerPeer,
        opBuf, algoBuf, protoBuf);
}

// Set the netAttr for a transfer operation
// 函数功能：为传输操作设置网络属性
// 参数说明：
//   - proxyState: 代理状态指针
//   - args: 代理参数指针，包含传输操作的信息
//   - send: 是否为发送操作（1=发送，0=接收）
void setXferNetAttrs(struct ncclProxyState* proxyState, struct ncclProxyArgs* args, int send)
{
  // 声明网络属性变量
  ncclNetAttr_t netAttr;

  // 检查网络插件是否支持设置网络属性
  if (!proxyState->ncclNet->setNetAttr)
    return;

  // 复制当前的网络属性作为基础
  netAttr = proxyState->netAttr;

  // 根据是发送还是接收操作，设置相应的属性
  if (send) {
    // 设置发送端的最大和最小并发对等体数量
    netAttr.sendCommAttr.maxConcurrentPeers = args->nPeers;
    netAttr.sendCommAttr.minConcurrentPeers = args->nPeers;
    // 设置发送端每个对等体的最大和最小流数量
    netAttr.sendCommAttr.maxFlowsPerPeer = args->nChannels;
    netAttr.sendCommAttr.minFlowsPerPeer = args->nChannels;
  } else {
    // 设置接收端的最大和最小并发对等体数量
    netAttr.recvCommAttr.maxConcurrentPeers = args->nPeers;
    netAttr.recvCommAttr.minConcurrentPeers = args->nPeers;
    // 设置接收端每个对等体的最大和最小流数量
    netAttr.recvCommAttr.maxFlowsPerPeer = args->nChannels;
    netAttr.recvCommAttr.minFlowsPerPeer = args->nChannels;
  }

  // 设置操作类型（使用位掩码）
  // args->collAPI: 集合 API 类型
  netAttr.op = BIT(args->collAPI);
  // algo/proto are undefined for p2p
  // 对于 P2P 操作，算法和协议未定义
  if (args->collAPI < NCCL_NUM_FUNCTIONS) {
    // 对于集合操作，设置算法和协议
    netAttr.algo = BIT(args->algorithm);
    netAttr.proto = BIT(args->protocol);
  }

  // 检查网络属性是否发生变化
  // memcmp: 内存比较函数，比较两个网络属性结构体
  if (memcmp(&proxyState->netAttr, &netAttr, sizeof(netAttr))) {
    // 属性发生变化，更新网络属性
    setNetAttrs(proxyState, &netAttr);
    // 打印新的网络属性
    printNetAttrs(&netAttr, send ? "send" : "recv");
  }
}

// Forward declaration
// 前向声明：sendProxyProgress 函数
static ncclResult_t sendProxyProgress(struct ncclProxyState* proxyState, struct ncclProxyArgs* args);

// Returns the flags to be used by a call to cuMemGetHandleForAddressRange.
// 函数功能：返回调用 cuMemGetHandleForAddressRange 时使用的标志
// 参数说明：
//   - useGdr: GPU Direct RDMA 使用模式
// 返回值：标志位值
static inline int getHandleForAddressRangeFlags(ncclTopoGdrMode useGdr) {
  // 初始化标志为 0
  int flags = 0;
#if CUDA_VERSION >= 12080
  // Force mapping on PCIe on systems with both PCI and C2C attachments.
  // 在同时有 PCI 和 C2C（Cache-to-Cache）连接的系统中，强制使用 PCIe 映射
  // ncclTopoGdrModePci: PCIe 模式的 GDR
  // CU_MEM_RANGE_FLAG_DMA_BUF_MAPPING_TYPE_PCIE: CUDA DMA 缓冲区 PCIe 映射标志
  if (useGdr == ncclTopoGdrModePci) flags = CU_MEM_RANGE_FLAG_DMA_BUF_MAPPING_TYPE_PCIE;
#endif
  // 返回标志
  return flags;
}

/* Determine if we will use this transport for this peer and return connect
 * information for this peer */
/* 确定是否为此对等体使用此传输层，并返回此对等体的连接信息 */
// 函数功能：设置发送端连接，建立发送端到接收端的网络连接
// 参数说明：
//   - comm: 通信上下文指针
//   - graph: 拓扑图指针（可能为 NULL）
//   - myInfo: 本地对等体信息
//   - peerInfo: 远程对等体信息
//   - connectInfo: 输出参数，返回连接信息
//   - send: 发送连接器指针
//   - channelId: 通道 ID
//   - connIndex: 连接索引
// 返回值：ncclSuccess 表示成功
static ncclResult_t sendSetup(struct ncclComm* comm, struct ncclTopoGraph* graph, struct ncclPeerInfo* myInfo, struct ncclPeerInfo* peerInfo, struct ncclConnect* connectInfo, struct ncclConnector* send, int channelId, int connIndex) {
  // 声明并初始化设置请求结构体
  struct setupReq req = { 0 };

  // 设置是否使用共享缓冲区
  // graph || connIndex == 0: 如果有拓扑图或者是第一个连接，不使用共享缓冲区
  // ncclParamNetSharedBuffers() != -2: 检查用户是否明确设置了共享缓冲区参数
  // 如果用户设置了，使用用户设置的值；否则默认使用共享缓冲区（1）
  send->conn.shared = req.shared = graph || connIndex == 0 ? 0 : ncclParamNetSharedBuffers() != -2 ? ncclParamNetSharedBuffers() : 1;
  // 设置通道 ID
  req.channelId = channelId;
  // 设置连接索引
  req.connIndex = connIndex;

  // 声明代理 rank 变量，用于存储将执行网络操作的代理 rank
  int proxyRank;
  // 声明网络 ID 变量，用于存储网络设备的唯一标识符
  int64_t netId;
  // 获取网络设备信息
  // myInfo->rank: 本地 rank
  // graph: 拓扑图
  // channelId: 通道 ID
  // peerInfo->rank: 对等体 rank
  // &netId: 输出网络 ID
  // &req.netDev: 输出网络设备编号
  // &proxyRank: 输出代理 rank
  NCCLCHECK(ncclTopoGetNetDev(comm, myInfo->rank, graph, channelId, peerInfo->rank, &netId, &req.netDev, &proxyRank));
  // 检查是否可以使用 GPU Direct RDMA
  // myInfo->rank: 本地 rank
  // netId: 网络 ID
  // 1: 发送方向
  // &req.useGdr: 输出 GDR 使用模式
  NCCLCHECK(ncclTopoCheckGdr(comm->topo, myInfo->rank, netId, 1, &req.useGdr));
  // 如果可以使用 GDR，设置 NCCL_DIRECT_NIC 标志
  // NCCL_DIRECT_NIC: 表示网卡可以直接访问 GPU 内存
  send->conn.flags |= req.useGdr ? NCCL_DIRECT_NIC : 0;
  // 如果第一个连接不能使用 GDR，全局禁用 GDR
  if (!req.useGdr && connIndex == 0) comm->useGdr = 0;
  // 如果代理不是本地 rank，启用网络代理扩展（PXN）
  if (proxyRank != myInfo->rank && connIndex == 0) comm->useNetPXN = true;

  // 连接到代理
  // TRANSPORT_NET: 传输类型为网络
  // 1: 发送方向
  // proxyRank: 代理 rank
  // &send->proxyConn: 输出代理连接
  NCCLCHECK(ncclProxyConnect(comm, TRANSPORT_NET, 1, proxyRank, &send->proxyConn));
  // 设置传输平面的本地 rank（同一节点内的 rank）
  req.tpLocalRank = comm->topParentLocalRanks[comm->localRank];
  // 设置传输平面的本地 rank（全局）
  req.tpRank = comm->topParentRanks[myInfo->rank];
  // 设置传输平面的远程 rank（对等体的全局 rank）
  req.tpRemoteRank = comm->topParentRanks[peerInfo->rank];
  // 向代理发送设置请求，阻塞等待完成
  NCCLCHECK(ncclProxyCallBlocking(comm, &send->proxyConn, ncclProxyMsgSetup, &req, sizeof(req), NULL, 0));

  // 输出连接建立的日志信息
  if (proxyRank == myInfo->rank) {
    // 本地代理情况：不显示代理 rank
    INFO(NCCL_INIT|NCCL_NET,"Channel %02d/%d : %d[%d] -> %d[%d] [send] via NET/%s/%d%s%s%s", channelId, connIndex, myInfo->rank, myInfo->nvmlDev, peerInfo->rank, peerInfo->nvmlDev, comm->ncclNet->name, req.netDev,
        req.useGdr ? "/GDRDMA" : "", req.useGdr==ncclTopoGdrModePci ? "(PCI)" : "",
        req.shared ? "/Shared" : "");
  } else {
    // 远程代理情况：显示代理 rank
    INFO(NCCL_INIT|NCCL_NET,"Channel %02d/%d : %d[%d] -> %d[%d] [send] via NET/%s/%d(%d)%s%s%s", channelId, connIndex, myInfo->rank, myInfo->nvmlDev, peerInfo->rank, peerInfo->nvmlDev, comm->ncclNet->name, req.netDev,
        proxyRank,
        req.useGdr ? "/GDRDMA" : "", req.useGdr==ncclTopoGdrModePci ? "(PCI)" : "",
        req.shared ? "/Shared" : "");
  }
  // 将代理 rank 存储到连接信息中
  *((int*)connectInfo) = comm->topParentRanks[proxyRank];
  // 将 GDR 使用模式复制到连接信息的尾部
  // connectInfo 包含 ncclNetHandle_t，我们在其后追加 useGdr 标志
  memcpy((uint8_t*)connectInfo + sizeof(ncclNetHandle_t), &req.useGdr, sizeof(int));
  // 返回成功状态码
  return ncclSuccess;
}

// GDRCOPY support: TAIL_ENABLE When enabled locates the RX proxy tail in CUDA memory
// GDRCOPY 支持：TAIL_ENABLE 当启用时，将接收代理的尾部定位在 CUDA 内存中
// GDRCOPY_SYNC_ENABLE: 环境变量名称
// 1: 默认值，表示启用同步
NCCL_PARAM(GdrCopySyncEnable, "GDRCOPY_SYNC_ENABLE", 1);
// GDRCOPY support: FLUSH_ENABLE When enabled uses a PCI-E read to flush GDRDMA buffers
// GDRCOPY 支持：FLUSH_ENABLE 当启用时，使用 PCI-E 读取来刷新 GDRDMA 缓冲区
// GDRCOPY_FLUSH_ENABLE: 环境变量名称
// 0: 默认值，表示不启用刷新
NCCL_PARAM(GdrCopyFlushEnable, "GDRCOPY_FLUSH_ENABLE", 0);

/* Setup recv connector */
/* 设置接收连接器 */
// 函数功能：设置接收端连接，建立接收端到发送端的网络连接
// 参数说明：
//   - comm: 通信上下文指针
//   - graph: 拓扑图指针（可能为 NULL）
//   - myInfo: 本地对等体信息（接收方）
//   - peerInfo: 远程对等体信息（发送方）
//   - connectInfo: 输出参数，返回连接信息
//   - recv: 接收连接器指针
//   - channelId: 通道 ID
//   - connIndex: 连接索引
// 返回值：ncclSuccess 表示成功
static ncclResult_t recvSetup(struct ncclComm* comm, struct ncclTopoGraph* graph, struct ncclPeerInfo* myInfo, struct ncclPeerInfo* peerInfo, struct ncclConnect* connectInfo, struct ncclConnector* recv, int channelId, int connIndex) {
  // 声明并初始化设置请求结构体
  struct setupReq req = { 0 };

  // 设置是否使用共享缓冲区（与 sendSetup 相同的逻辑）
  recv->conn.shared = req.shared = graph || connIndex == 0 ? 0 : ncclParamNetSharedBuffers() != -2 ? ncclParamNetSharedBuffers() : 1;
  // 设置通道 ID
  req.channelId = channelId;
  // 设置连接索引
  req.connIndex = connIndex;

  // Use myInfo->rank as the receiver uses its own NIC
  // 使用 myInfo->rank，因为接收方使用自己的网卡
  int proxyRank;
  int64_t netId;
  // 获取网络设备信息（接收方使用自己的 NIC，所以第四个参数是 myInfo->rank）
  NCCLCHECK(ncclTopoGetNetDev(comm, myInfo->rank, graph, channelId, myInfo->rank, &netId, &req.netDev, &proxyRank));
  // 检查是否可以使用 GPU Direct RDMA（接收方向，第四个参数为 0）
  NCCLCHECK(ncclTopoCheckGdr(comm->topo, myInfo->rank, netId, 0, &req.useGdr));
  // 如果可以使用 GDR，设置 NCCL_DIRECT_NIC 标志
  recv->conn.flags |= req.useGdr ? NCCL_DIRECT_NIC : 0;
  // 如果第一个连接不能使用 GDR，全局禁用 GDR
  if (!req.useGdr && connIndex == 0) comm->useGdr = 0;

  // Determine whether we need to flush the GDR buffer on recv or not
  // 确定是否需要在接收时刷新 GDR 缓冲区
  if (req.useGdr) NCCLCHECK(ncclTopoNeedFlush(comm, netId, req.netDev, myInfo->rank, &req.needFlush));

  // We don't support PXN on receive yet
  // 我们在接收端尚不支持 PXN（网络代理扩展）
  // TRANSPORT_NET: 传输类型为网络
  // 0: 接收方向
  // myInfo->rank: 使用本地 rank 作为代理
  NCCLCHECK(ncclProxyConnect(comm, TRANSPORT_NET, 0, myInfo->rank, &recv->proxyConn));

  // 设置传输平面的本地 rank（同一节点内的 rank）
  req.tpLocalRank = comm->topParentLocalRanks[comm->localRank];
  // 设置传输平面的本地 rank（全局）
  req.tpRank = comm->topParentRanks[myInfo->rank];
  // 设置传输平面的远程 rank（对等体的全局 rank）
  req.tpRemoteRank = comm->topParentRanks[peerInfo->rank];
  // 向代理发送设置请求，并接收连接信息（网络句柄）
  NCCLCHECK(ncclProxyCallBlocking(comm, &recv->proxyConn, ncclProxyMsgSetup, &req, sizeof(req), connectInfo, sizeof(ncclNetHandle_t)));
  // 将 GDR 使用模式复制到连接信息的尾部
  memcpy((uint8_t*)connectInfo + sizeof(ncclNetHandle_t), &req.useGdr, sizeof(int));
  // 输出连接建立的日志信息
  INFO(NCCL_INIT|NCCL_NET,"Channel %02d/%d : %d[%d] -> %d[%d] [receive] via NET/%s/%d%s%s%s", channelId, connIndex, peerInfo->rank, peerInfo->nvmlDev, myInfo->rank, myInfo->nvmlDev, comm->ncclNet->name, req.netDev,
      req.useGdr ? "/GDRDMA" : "", req.useGdr==ncclTopoGdrModePci ? "(PCI)" : "",
      req.shared ? "/Shared" : "");
  // 返回成功状态码
  return ncclSuccess;
}

// 函数功能：映射共享内存到进程地址空间
// 参数说明：
//   - comm: 通信上下文指针
//   - proxyConn: 代理连接器指针
//   - mem: 内存映射信息结构体，包含共享内存的描述符
// 返回值：ncclSuccess 表示成功
static ncclResult_t netMapShm(struct ncclComm *comm, struct ncclProxyConnector* proxyConn, struct connectMapMem* mem) {
  // 导入可共享的缓冲区
  // proxyConn->rank: 代理的 rank
  // &mem->createDesc: 创建描述符
  // &mem->cpuPtr: 输出 CPU 指针
  // &mem->gpuPtr: 输出 GPU 指针
  // &mem->attachDesc: 输出附加描述符
  NCCLCHECK(ncclShmImportShareableBuffer(comm, proxyConn->rank, &mem->createDesc, (void**)&mem->cpuPtr, (void**)&mem->gpuPtr, &mem->attachDesc));
  // 返回成功状态码
  return ncclSuccess;
}

// 函数功能：创建共享内存区域
// 参数说明：
//   - proxyState: 代理状态指针
//   - mem: 内存映射信息结构体，包含内存大小和描述符
// 返回值：ncclSuccess 表示成功
static ncclResult_t netCreateShm(struct ncclProxyState* proxyState, struct connectMapMem* mem) {
  // 分配可共享的缓冲区
  // mem->size: 要分配的内存大小
  // false: 不使用统一内存
  // &mem->createDesc: 输出创建描述符
  // &mem->cpuPtr: 输出 CPU 指针
  // &mem->gpuPtr: 输出 GPU 指针
  NCCLCHECK(ncclShmAllocateShareableBuffer(mem->size, false, &mem->createDesc, (void**)&mem->cpuPtr, (void**)&mem->gpuPtr));
  // 返回成功状态码
  return ncclSuccess;
}

// 函数功能：打印连接映射的详细信息（用于调试）
// 参数说明：
//   - map: 连接映射指针
// 返回值：ncclSuccess 表示成功
static ncclResult_t netDumpMap(struct connectMap* map) {
  // 打印映射的基本信息
  printf("Dump map same process %d shared %d\n", map->sameProcess, map->shared);
  // 获取主机内存区域的指针
  struct connectMapMem *mem = map->mems+NCCL_NET_MAP_HOSTMEM;
  // 打印主机内存信息（Mem 0）
  printf("Mem 0: Host mem (%x B) CPU %p GPU %p\n", mem->size, mem->cpuPtr, mem->gpuPtr);
  // 获取设备内存区域的指针
  mem = map->mems+NCCL_NET_MAP_DEVMEM;
  // 打印设备内存信息（Mem 1）
  printf("Mem 1: Vid  mem (%x B) CPU %p GPU %p\n", mem->size, mem->cpuPtr, mem->gpuPtr);
  // 获取共享主机内存区域的指针
  mem = map->mems+NCCL_NET_MAP_SHARED_HOSTMEM;
  // 打印共享主机内存信息（Mem 2）
  printf("Mem 2: Shared Host mem (%x B) CPU %p GPU %p\n", mem->size, mem->cpuPtr, mem->gpuPtr);
  // 获取共享设备内存区域的指针
  mem = map->mems+NCCL_NET_MAP_SHARED_DEVMEM;
  // 打印共享设备内存信息（Mem 3）
  printf("Mem 3: Shared Vid mem (%x B) CPU %p GPU %p\n", mem->size, mem->cpuPtr, mem->gpuPtr);
  // 打印发送内存的映射信息
  // 显示是否使用、内存库索引、偏移量、CPU 指针、GPU 指针
  printf("SendMem -> Used %d Bank %d Offset %x, cpu %p gpu %p\n",
      map->offsets.sendMem & NCCL_NET_MAP_MASK_USED ? 1 : 0,
      NCCL_NET_MAP_OFFSET_BANK(map, sendMem), map->offsets.sendMem & NCCL_NET_MAP_MASK_OFFSET,
      NCCL_NET_MAP_GET_POINTER(map, cpu, sendMem), NCCL_NET_MAP_GET_POINTER(map, gpu, sendMem));
  // 打印接收内存的映射信息
  printf("RecvMem -> Used %d Bank %d Offset %x, cpu %p gpu %p\n",
      map->offsets.recvMem & NCCL_NET_MAP_MASK_USED ? 1 : 0,
      NCCL_NET_MAP_OFFSET_BANK(map, recvMem), map->offsets.recvMem & NCCL_NET_MAP_MASK_OFFSET,
      NCCL_NET_MAP_GET_POINTER(map, cpu, recvMem), NCCL_NET_MAP_GET_POINTER(map, gpu, recvMem));
  // 遍历所有协议，打印每个协议缓冲区的映射信息
  for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
    printf("Proto %d -> Used %d Bank %d Offset %x, cpu %p, gpu %p\n", p,
        map->offsets.buffs[p] & NCCL_NET_MAP_MASK_USED ? 1 : 0,
        NCCL_NET_MAP_OFFSET_BANK(map, buffs[p]), map->offsets.buffs[p] & NCCL_NET_MAP_MASK_OFFSET,
        NCCL_NET_MAP_GET_POINTER(map, cpu, buffs[p]), NCCL_NET_MAP_GET_POINTER(map, gpu, buffs[p]));
  }
  printf("End of dump\n");
  return ncclSuccess;
}

struct netSendConnectArgs {
  ncclNetHandle_t handle;
  ncclNetAttr_t netAttr;
};

struct netRecvConnectArgs {
  int proxyRank;
  ncclNetAttr_t netAttr;
};

static ncclResult_t sendConnect(struct ncclComm* comm, struct ncclConnect* connectInfo, int nranks, int rank, struct ncclConnector* send) {
  struct connectMap* map = (connectMap*) send->transportResources;
  void* opId;
  int recvUseGdr;

  memcpy(&recvUseGdr, (uint8_t*)connectInfo + sizeof(ncclNetHandle_t), sizeof(int));
  if (!recvUseGdr) send->conn.flags &= ~NCCL_DIRECT_NIC;

  // map isn't allocated thus this op hasn't been submitted yet
  if (!map) {
    // Setup device pointers
    NCCLCHECK(ncclCalloc(&map, 1));
    send->transportResources = map;
    opId = send;
    INFO(NCCL_PROXY, "sendConnect ncclProxyCallAsync opId=%p", opId);
    netSendConnectArgs args = {0};
    memcpy(&args.handle, connectInfo, sizeof(ncclNetHandle_t));

    populateCommNetAttrs(comm, send, &args.netAttr);

    NCCLCHECK(ncclProxyCallAsync(comm, &send->proxyConn, ncclProxyMsgConnect, &args, sizeof(netSendConnectArgs), sizeof(struct connectMap), opId));
  } else {
    opId =  send;
  }

  ncclResult_t ret;
  ret = ncclPollProxyResponse(comm, &send->proxyConn, map, opId);
  if (ret != ncclSuccess) {
    if (ret != ncclInProgress) {
      free(map);
      send->transportResources = NULL;
    }
    return ret;
  }
  INFO(NCCL_PROXY, "sendConnect ncclPollProxyResponse opId=%p", opId);

  // 检查是否为同一进程且未启用 cuMem
  if (map->sameProcess && !ncclCuMemEnable()) {
    // 检查是否为不同的 GPU 设备
    if (map->cudaDev != comm->cudaDev) {
      // Enable P2P access for Legacy IPC
      // 为传统 IPC 启用 P2P 访问
      cudaError_t err = cudaDeviceEnablePeerAccess(map->cudaDev, 0);
      // 检查是否已经启用了 P2P 访问（这是正常情况）
      if (err == cudaErrorPeerAccessAlreadyEnabled) {
        // 清除错误状态
        cudaGetLastError();
      } else if (err != cudaSuccess) {
        // P2P 访问启用失败，记录警告
        WARN("failed to peer with device %d: %d %s", map->cudaDev, err, cudaGetErrorString(err));
        // 返回内部错误
        return ncclInternalError;
      }
    }
  // 检查是否为不同进程或不同设备
  } else if (!(map->sameProcess && map->cudaDev == comm->cudaDev)) {
    // 如果不是同一进程，映射共享主机内存
    if (!map->sameProcess) NCCLCHECK(netMapShm(comm, &send->proxyConn, map->mems + NCCL_NET_MAP_HOSTMEM));
    // 检查是否有设备内存需要映射
    if (map->mems[NCCL_NET_MAP_DEVMEM].size) {
      // 初始化 GPU 指针为 NULL
      map->mems[NCCL_NET_MAP_DEVMEM].gpuPtr = NULL;
      // 导入可共享的设备内存缓冲区
      // send->proxyConn.rank: 代理的 rank
      // size: 设备内存大小
      // ipcDesc: IPC 描述符
      // gpuPtr: 输出 GPU 指针
      NCCLCHECK(ncclP2pImportShareableBuffer(comm, send->proxyConn.rank,
                                             map->mems[NCCL_NET_MAP_DEVMEM].size,
                                             &map->mems[NCCL_NET_MAP_DEVMEM].ipcDesc,
                                             (void**)&map->mems[NCCL_NET_MAP_DEVMEM].gpuPtr));
      // CPU 指针为 NULL（设备内存没有 CPU 可访问地址）
      map->mems[NCCL_NET_MAP_DEVMEM].cpuPtr = NULL;
    }
    // 检查是否有共享设备内存需要映射
    if (map->mems[NCCL_NET_MAP_SHARED_DEVMEM].size) {
      // 获取共享设备内存指针（按本地 rank 索引）
      void** sharedDevMemPtr = comm->proxyState->sharedDevMems + send->proxyConn.tpLocalRank;
      // 如果共享设备内存尚未分配
      if (*sharedDevMemPtr == NULL) {
        // 初始化 GPU 指针为 NULL
        map->mems[NCCL_NET_MAP_SHARED_DEVMEM].gpuPtr = NULL;
        // 导入可共享的设备内存缓冲区
        NCCLCHECK(ncclP2pImportShareableBuffer(comm, send->proxyConn.rank,
                                               map->mems[NCCL_NET_MAP_SHARED_DEVMEM].size,
                                               &map->mems[NCCL_NET_MAP_SHARED_DEVMEM].ipcDesc,
                                               sharedDevMemPtr));
      }
      // 使用共享设备内存指针
      map->mems[NCCL_NET_MAP_SHARED_DEVMEM].gpuPtr = (char*)(*sharedDevMemPtr);
      // CPU 指针为 NULL
      map->mems[NCCL_NET_MAP_SHARED_DEVMEM].cpuPtr = NULL;
    }
  }
  //NCCLCHECK(netDumpMap(map));  // 调试用：打印映射信息

  // 获取发送内存指针（GPU 指针）
  struct ncclSendMem *sendMem = (struct ncclSendMem*) NCCL_NET_MAP_GET_POINTER(map, gpu, sendMem);
  // 获取 GDC（GPU Direct Cache）内存指针
  void* gdcMem = map->mems[NCCL_NET_MAP_GDCMEM].gpuPtr;
  // 设置连接的头部指针（使用 GDC 内存或发送内存的头部）
  send->conn.head = gdcMem ? (uint64_t*)gdcMem : &sendMem->head;

  // 获取接收内存指针（GPU 指针）
  struct ncclRecvMem *recvMem = (struct ncclRecvMem*) NCCL_NET_MAP_GET_POINTER(map, gpu, recvMem);
  // 设置连接的尾部指针
  send->conn.tail = &recvMem->tail;
  // 计算步进大小（每个步进的缓冲区大小）
  // NCCL_PROTO_SIMPLE: Simple 协议
  // NCCL_STEPS: 步进数量
  send->conn.stepSize = comm->buffSizes[NCCL_PROTO_SIMPLE]/NCCL_STEPS;
  // 设置连接 FIFO
  send->conn.connFifo = recvMem->connFifo;
  // Only fuse P2P buffers, continue to allocate dedicated buffers for ring/tree
  // 仅融合 P2P 缓冲区，继续为 ring/tree 分配专用缓冲区
  // 初始化连接 FIFO
  for (int i=0; i<NCCL_STEPS; i++) {
    // 设置偏移量为 -1（表示未使用）
    send->conn.connFifo[i].offset = -1;
    // 设置模式：共享内存使用偏移模式，否则使用普通模式
    recvMem->connFifo[i].mode = map->shared ? NCCL_MODE_OFFSET : NCCL_MODE_NORMAL;
  }

  // 遍历所有协议，设置缓冲区指针
  for (int p=0; p<NCCL_NUM_PROTOCOLS; p++)
    // 获取每个协议的缓冲区指针（GPU 指针）
    send->conn.buffs[p] = NCCL_NET_MAP_GET_POINTER(map, gpu, buffs[p]);

  // 检查是否为同一进程
  if (send->proxyConn.sameProcess) {
    // 检查是否有网络设备句柄
    if (send->proxyConn.connection->netDeviceHandle) {
      // 复制网络设备句柄
      send->conn.netDeviceHandle = *send->proxyConn.connection->netDeviceHandle;

      // 遍历所有协议，复制内存句柄
      for (int p=0; p<NCCL_NUM_PROTOCOLS; p++)
        // 复制每个协议的内存句柄
        send->conn.mhandles[p] = send->proxyConn.connection->mhandles[p];
    }

    // 检查是否需要代理进度
    if (send->proxyConn.connection->needsProxyProgress) {
      // 设置代理进度函数
      send->proxyConn.proxyProgress = sendProxyProgress;
    } else {
      // 不需要代理进度
      send->proxyConn.proxyProgress = NULL;
    }
  } else {
    // 不同进程，总是需要代理进度
    send->proxyConn.proxyProgress = sendProxyProgress;
  }

  // 返回成功状态码
  return ncclSuccess;
}

// Forward declare
// 前向声明：recvProxyProgress 函数
static ncclResult_t recvProxyProgress(struct ncclProxyState* proxyState, struct ncclProxyArgs* args);

/* Connect to this peer */
/* 连接到此对等体 */
// 函数功能：建立接收端连接
// 参数说明：
//   - comm: 通信上下文指针
//   - connectInfo: 连接信息（从发送端获取）
//   - nranks: rank 总数
//   - rank: 当前 rank
//   - recv: 接收连接器指针
// 返回值：ncclSuccess 表示成功，ncclInProgress 表示连接进行中
static ncclResult_t recvConnect(struct ncclComm* comm, struct ncclConnect* connectInfo, int nranks, int rank, struct ncclConnector* recv) {
  // 获取连接映射指针
  struct connectMap* map = (connectMap*) recv->transportResources;
  // 声明操作 ID
  void* opId;
  // 声明发送端是否使用 GDR 的标志
  int sendUseGdr;

  // 从连接信息中提取发送端的 GDR 使用标志
  memcpy(&sendUseGdr, (uint8_t*)connectInfo + sizeof(ncclNetHandle_t), sizeof(int));
  // 如果发送端不使用 GDR，清除接收端的 NCCL_DIRECT_NIC 标志
  if (!sendUseGdr) recv->conn.flags &= ~NCCL_DIRECT_NIC;

  // 检查 map 是否已分配
  if (!map) {
    // 分配并初始化连接映射结构体
    NCCLCHECK(ncclCalloc(&map, 1));
    // 保存映射到接收连接器
    recv->transportResources = map;
    // Use recv connector as unique identifier
    // 使用接收连接器作为唯一标识符
    opId = recv;
    // 输出日志：异步调用代理连接
    INFO(NCCL_PROXY, "recvConnect ncclProxyCallAsync opId=%p &recv->proxyConn=%p connectInfo=%p",
       opId, &recv->proxyConn, connectInfo);
    // 声明并初始化接收连接参数
    netRecvConnectArgs args = {0};
    // 设置代理 rank（从连接信息中提取）
    args.proxyRank = *((int*)connectInfo);

    // 填充网络属性
    populateCommNetAttrs(comm, recv, &args.netAttr);

    // 异步调用代理进行连接
    NCCLCHECK(ncclProxyCallAsync(comm, &recv->proxyConn, ncclProxyMsgConnect, &args, sizeof(netRecvConnectArgs), sizeof(struct connectMap), opId));
  } else {
    // 如果 map 已分配，复用操作 ID
    opId = recv;
  }

  // 声明结果变量
  ncclResult_t ret;
  // 轮询代理响应
  NCCLCHECK(ret = ncclPollProxyResponse(comm, &recv->proxyConn, map, opId));
  // 检查结果
  if (ret != ncclSuccess) {
    // 如果连接失败且不是进行中状态
    if (ret != ncclInProgress) {
      // 释放映射内存
      free(map);
      // 清空传输资源指针
      recv->transportResources = NULL;
    }
    // 返回结果
    return ret;
  }
  // 输出日志：代理响应完成
  INFO(NCCL_PROXY, "recvConnect ncclPollProxyResponse opId=%p", opId);
  //NCCLCHECK(netDumpMap(map));  // 调试用：打印映射信息

  // 获取发送内存指针（GPU 指针）
  struct ncclSendMem *sendMem = (struct ncclSendMem*) NCCL_NET_MAP_GET_POINTER(map, gpu, sendMem);
  // 设置连接的头部指针
  recv->conn.head = &sendMem->head;

  // 获取接收内存指针（GPU 指针）
  struct ncclRecvMem *recvMem = (struct ncclRecvMem*) NCCL_NET_MAP_GET_POINTER(map, gpu, recvMem);
  // 获取 GDC（GPU Direct Cache）内存指针
  void* gdcMem = map->mems[NCCL_NET_MAP_GDCMEM].gpuPtr;
  // 设置连接的尾部指针（使用 GDC 内存或接收内存的尾部）
  recv->conn.tail = gdcMem ? (uint64_t*)gdcMem : &recvMem->tail;
  // 计算步进大小
  recv->conn.stepSize = comm->buffSizes[NCCL_PROTO_SIMPLE]/NCCL_STEPS;
  // 设置连接 FIFO
  recv->conn.connFifo = recvMem->connFifo;
  // Only fuse P2P buffers, continue to allocate dedicated buffers for ring/tree
  // 初始化连接 FIFO
  for (int i=0; i<NCCL_STEPS; i++) {
    // 设置模式
    recvMem->connFifo[i].mode = map->shared ? NCCL_MODE_OFFSET : NCCL_MODE_NORMAL;
  }

  // 遍历所有协议，设置缓冲区指针
  for (int p=0; p<NCCL_NUM_PROTOCOLS; p++)
    // 获取每个协议的缓冲区指针（GPU 指针）
    recv->conn.buffs[p] = NCCL_NET_MAP_GET_POINTER(map, gpu, buffs[p]);

  // 检查是否为同一进程
  if (recv->proxyConn.sameProcess) {
    // 检查是否有网络设备句柄
    if (recv->proxyConn.connection->netDeviceHandle) {
      // 复制网络设备句柄
      recv->conn.netDeviceHandle = *recv->proxyConn.connection->netDeviceHandle;

      // 遍历所有协议，复制内存句柄
      for (int p=0; p<NCCL_NUM_PROTOCOLS; p++)
        // 复制每个协议的内存句柄
        recv->conn.mhandles[p] = recv->proxyConn.connection->mhandles[p];
    }

    // 检查是否需要代理进度
    if (recv->proxyConn.connection->needsProxyProgress) {
      // 设置代理进度函数
      recv->proxyConn.proxyProgress = recvProxyProgress;
    } else {
      // 不需要代理进度
      recv->proxyConn.proxyProgress = NULL;
    }
  } else {
    // 不同进程，总是需要代理进度
    recv->proxyConn.proxyProgress = recvProxyProgress;
  }

  // 返回成功状态码
  return ncclSuccess;
}

static ncclResult_t sendFree(struct ncclConnector* send) {
  struct connectMap* map = (struct connectMap*)(send->transportResources);
  if (map) {
    int cudaDev;
    CUDACHECK(cudaGetDevice(&cudaDev));
    if (map->cudaDev != cudaDev && map->mems[NCCL_NET_MAP_DEVMEM].size) {
      if (ncclCuMemEnable()) {
        // cuMem API support
        NCCLCHECK(ncclP2pFreeShareableBuffer(&map->mems[NCCL_NET_MAP_DEVMEM].ipcDesc));
        NCCLCHECK(ncclCuMemFree(map->mems[NCCL_NET_MAP_DEVMEM].gpuPtr));
      } else {
        // Legacy CUDA IPC support
        CUDACHECK(cudaIpcCloseMemHandle(map->mems[NCCL_NET_MAP_DEVMEM].gpuPtr));
      }
    }
    if (!map->sameProcess) {
      NCCLCHECK(ncclShmIpcClose(&map->mems[NCCL_NET_MAP_HOSTMEM].attachDesc));
    }
    free(map);
  }

  return ncclSuccess;
}

static ncclResult_t recvFree(struct ncclConnector* recv) {
  if (recv->transportResources) free(recv->transportResources);
  return ncclSuccess;
}

#define NCCL_SHARED_STEPS 16
static ncclResult_t sharedNetBuffersInit(struct ncclProxyState* proxyState, int cuda, int tpLocalRank, int type, int sameProcess,
    int nChannels, char** gpuPtr, char** cpuPtr, int* size, ncclIpcDesc *ipcDesc) {
  if (cuda == 0 && sameProcess == 0) {
      WARN("PXN should not use host buffers for data");
      return ncclInternalError;
  }
  struct ncclProxyProgressState* progressState = &proxyState->progressState;
  if (progressState->localPeers == NULL) {
    NCCLCHECK(ncclCalloc(&progressState->localPeers, proxyState->tpLocalnRanks));
  }
  struct ncclProxyPeer** localPeers = progressState->localPeers;
  if (localPeers[tpLocalRank] == NULL) {
    NCCLCHECK(ncclCalloc(localPeers + tpLocalRank, 1));
  }
  struct ncclProxyPeer* peer = localPeers[tpLocalRank];
  struct ncclProxySharedP2p *state2 = (type == 0)? &peer->send : &peer->recv;
  
  state2->refcount++;
  if (state2->size == 0) {
    state2->size = nChannels * NCCL_SHARED_STEPS * proxyState->p2pChunkSize;
  }

  if (size) 
    *size = state2->size;

//cuda为1，分配cuda内存
  if (cuda && state2->cudaBuff == NULL) {
    //不在同一个进程内或者启用了CUDA内存管理
    if (sameProcess == 0 || ncclCuMemEnable()) {
        //并返回一个IPC描述符（state2->ipcDesc），其他进程可以通过这个ipc描述符来映射这块内存。
      NCCLCHECK(ncclP2pAllocateShareableBuffer(state2->size, 0, &state2->ipcDesc, (void**)&state2->cudaBuff));
    } else {
    //在同一个进程内，并且没有启用CUDA内存管理，那么直接使用普通的CUDA内存分配即可，
    //因为进程内的线程可以共享同一个地址空间，无需IPC
      NCCLCHECK(ncclCudaCalloc(&state2->cudaBuff, state2->size));
    }
    }
  }

  //cuda为0，分配主机内存
  // 如果需要主机内存且主机缓冲区未分配
  if (!cuda && state2->hostBuff == NULL) {
    // 分配锁定页面的主机内存（CUDA pinned memory）
    NCCLCHECK(ncclCudaHostCalloc(&state2->hostBuff, state2->size));
  }

  // 如果 cpuPtr 指针有效，返回 CPU 指针
  if (cpuPtr)
    // 根据 cuda 参数选择返回 CUDA 缓冲区或主机缓冲区
    *cpuPtr = cuda ? state2->cudaBuff : state2->hostBuff;

  // 如果 gpuPtr 指针有效，返回 GPU 指针
  if (gpuPtr) *
    // 如果在同一进程内且有 CPU 指针，GPU 指针与 CPU 指针相同；否则为 NULL
    gpuPtr = (cpuPtr && sameProcess) ? *cpuPtr : NULL;

  // 如果 ipcDesc 指针有效，复制 IPC 描述符
  if (ipcDesc)
    memcpy(ipcDesc, &state2->ipcDesc, sizeof(state2->ipcDesc));

  // 返回成功状态码
  return ncclSuccess;
}

// 函数功能：获取共享缓冲区的偏移量和大小
// 参数说明：
//   - proxyState: 代理状态指针
//   - channel: 通道 ID
//   - slot: 槽位索引
//   - offset: 输出参数，返回缓冲区偏移量
//   - size: 输出参数，返回缓冲区大小
// 返回值：ncclSuccess 表示成功
static ncclResult_t sharedBuffersGet(struct ncclProxyState* proxyState, int channel, int slot, int* offset, size_t* size) {
  // Use different pools for different channels and also separate send/recv.
  // 为不同通道使用不同的内存池，并且分离发送/接收
  // 计算全局槽位索引
  int globalSlot = (channel*NCCL_SHARED_STEPS)+slot;
  // 计算偏移量：全局槽位索引 * P2P 块大小
  *offset = proxyState->p2pChunkSize * globalSlot;
  // 如果 size 指针有效，返回 P2P 块大小
  if (size) *size = proxyState->p2pChunkSize;
  // 返回成功状态码
  return ncclSuccess;
}

// 函数功能：销毁共享网络缓冲区
// 参数说明：
//   - proxyState: 代理状态指针
//   - tpLocalRank: 传输平面的本地 rank
//   - type: 类型（0=发送，1=接收）
//   - connection: 代理连接指针
// 返回值：ncclSuccess 表示成功
static ncclResult_t sharedNetBuffersDestroy(struct ncclProxyState* proxyState, int tpLocalRank, int type, struct ncclProxyConnection* connection) {
  // 检查本地对等体数组是否存在
  if (proxyState->progressState.localPeers == NULL) NCCLCHECK(ncclInternalError);
  // 获取对等体指针
  struct ncclProxyPeer* peer = proxyState->progressState.localPeers[tpLocalRank];
  // 检查对等体是否存在
  if (peer == NULL) NCCLCHECK(ncclInternalError);
  // 根据类型获取发送或接收状态
  struct ncclProxySharedP2p* state = type == 0 ? &peer->send : &peer->recv;
  // 检查状态大小是否有效
  if (state->size == 0) NCCLCHECK(ncclInternalError);
  // 原子递减引用计数，检查是否为最后一个引用
  if (ncclAtomicRefCountDecrement(&state->refcount) == 0) {
    // 如果是最后一个引用，释放资源
    // 检查是否有 CUDA 缓冲区
    if (state->cudaBuff) {
      // 如果不在同一进程或启用了 cuMem，释放可共享缓冲区
      if (!connection->sameProcess || ncclCuMemEnable()) {
        NCCLCHECK(ncclP2pFreeShareableBuffer(&state->ipcDesc));
      }
      // 释放 CUDA 内存
      NCCLCHECK(ncclCudaFree(state->cudaBuff));
    }
    // 检查是否有主机缓冲区，释放之
    if (state->hostBuff) NCCLCHECK(ncclCudaHostFree(state->hostBuff));
  }

  // 如果对等体的发送或接收仍有引用，不释放对等体结构
  if (peer->send.refcount || peer->recv.refcount) return ncclSuccess;

  // 释放对等体结构
  free(peer);
  // 清空本地对等体数组中的指针
  proxyState->progressState.localPeers[tpLocalRank] = NULL;
  // 检查是否所有对等体都已释放
  for (int r = 0; r < proxyState->tpLocalnRanks; r++) {
    if (proxyState->progressState.localPeers[r]) return ncclSuccess;
  }
  // All peers are freed, free array
  // 所有对等体都已释放，释放数组
  free(proxyState->progressState.localPeers);
  proxyState->progressState.localPeers = NULL;
  // 返回成功状态码
  return ncclSuccess;
}

// 函数功能：初始化代理共享缓冲区
// 参数说明：
//   - connection: 代理连接指针
//   - proxyState: 代理状态指针
//   - nChannels: 通道数量
// 返回值：ncclSuccess 表示成功
static ncclResult_t proxySharedInit(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState, int nChannels) {
  // 初始化发送端共享网络缓冲区
  // 参数：proxyState, cuda=1, 本地 rank, type=0(发送), 同进程标志, 通道数量
  NCCLCHECK(sharedNetBuffersInit(proxyState, 1, connection->tpLocalRank, 0, connection->sameProcess, nChannels, NULL, NULL, NULL, NULL));
  // 返回成功状态码
  return ncclSuccess;
}

// 函数功能：发送端代理设置
// 参数说明：
//   - connection: 代理连接指针
//   - proxyState: 代理状态指针
//   - reqBuff: 请求缓冲区（包含 setupReq）
//   - reqSize: 请求大小
//   - respBuff: 响应缓冲区
//   - respSize: 响应大小
//   - done: 输出参数，指示操作是否完成
// 返回值：ncclSuccess 表示成功
static ncclResult_t sendProxySetup(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState, void* reqBuff, int reqSize, void* respBuff, int respSize, int* done) {
  // 获取设置请求指针
  struct setupReq* req = (struct setupReq*) reqBuff;
  // 验证请求大小
  if (reqSize != sizeof(struct setupReq)) return ncclInternalError;

  // 声明发送端网络资源指针
  struct sendNetResources* resources;
  // 分配并初始化资源结构体
  NCCLCHECK(ncclCalloc(&resources, 1));
  // 保存资源到代理连接
  connection->transportResources = resources;

  // 从请求中复制传输平面 rank 信息
  resources->tpRank = req->tpRank;
  resources->tpLocalRank = req->tpLocalRank;
  resources->tpRemoteRank = req->tpRemoteRank;
  // 设置网络设备编号
  resources->netDev = req->netDev;
  // 设置共享标志
  resources->shared = connection->shared = req->shared;
  // 设置 GDR 使用模式
  resources->useGdr = req->useGdr;
  // 设置通道 ID 和连接索引
  resources->channelId = req->channelId;
  resources->connIndex = req->connIndex;
  // 声明网络属性结构体
  ncclNetProperties_t props;
  // 获取网络设备属性
  NCCLCHECK(proxyState->ncclNet->getProperties(req->netDev, &props));
  /* DMA-BUF support */
  // 检查是否支持 DMA-BUF
  // 条件：使用 GDR && 代理支持 DMA-BUF && 网络设备支持 DMA-BUF 指针
  resources->useDmaBuf = resources->useGdr && proxyState->dmaBufSupport && (props.ptrSupport & NCCL_PTR_DMABUF);
  // 设置最大接收数量
  resources->maxRecvs = props.maxRecvs;
  // 设置网络设备版本和类型
  resources->netDeviceVersion = props.netDeviceVersion;
  resources->netDeviceType = props.netDeviceType;

  /* point-to-point size limits*/
  // 设置点对点传输大小限制
  resources->maxP2pBytes = props.maxP2pBytes;
  // 验证 maxP2pBytes 是否在有效范围内
  if((resources->maxP2pBytes <= 0) || (resources->maxP2pBytes > NCCL_MAX_NET_SIZE_BYTES)) {
    // 值无效，记录警告并返回错误
    WARN("sendProxySetup: net plugin returned invalid value for maxP2pBytes %ld \
      [allowed range: %ld - %ld] \n", resources->maxP2pBytes, 0L, NCCL_MAX_NET_SIZE_BYTES);
    return ncclInternalError;
  }

  // We don't return any data
  // 我们不返回任何数据
  if (respSize != 0) return ncclInternalError;
  // 标记操作完成
  *done = 1;
  // 返回成功状态码
  return ncclSuccess;
}

// 函数功能：接收端代理设置
// 参数说明：
//   - connection: 代理连接指针
//   - proxyState: 代理状态指针
//   - reqBuff: 请求缓冲区（包含 setupReq）
//   - reqSize: 请求大小
//   - respBuff: 响应缓冲区（返回网络句柄）
//   - respSize: 响应大小
//   - done: 输出参数，指示操作是否完成
// 返回值：ncclSuccess 表示成功
static ncclResult_t recvProxySetup(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState, void* reqBuff, int reqSize, void* respBuff, int respSize, int* done) {
  // 获取设置请求指针
  struct setupReq* req = (struct setupReq*) reqBuff;
  // 验证请求大小
  if (reqSize != sizeof(struct setupReq)) return ncclInternalError;

  // 声明接收端网络资源指针
  struct recvNetResources* resources;
  // 分配并初始化资源结构体
  NCCLCHECK(ncclCalloc(&resources, 1));
  // 保存资源到代理连接
  connection->transportResources = resources;

  // 从请求中复制传输平面 rank 信息
  resources->tpRank = req->tpRank;
  resources->tpLocalRank = req->tpLocalRank;
  resources->tpRemoteRank = req->tpRemoteRank;
  // 设置网络设备编号
  resources->netDev = req->netDev;
  // 设置共享标志
  resources->shared = connection->shared = req->shared;
  // 设置 GDR 使用模式
  resources->useGdr = req->useGdr;
  // 设置是否需要刷新缓存
  resources->needFlush = req->needFlush;
  // 设置通道 ID 和连接索引
  resources->channelId = req->channelId;
  resources->connIndex = req->connIndex;
  // 声明网络属性结构体
  ncclNetProperties_t props;
  // 获取网络设备属性
  NCCLCHECK(proxyState->ncclNet->getProperties(req->netDev, &props));
  /* DMA-BUF support */
  // 检查是否支持 DMA-BUF
  resources->useDmaBuf = resources->useGdr && proxyState->dmaBufSupport && (props.ptrSupport & NCCL_PTR_DMABUF);
  // 设置最大接收数量
  resources->maxRecvs = props.maxRecvs;
  // 设置网络设备版本和类型
  resources->netDeviceVersion = props.netDeviceVersion;
  resources->netDeviceType = props.netDeviceType;
  /* point-to-point size limits*/
  // 设置点对点传输大小限制
  resources->maxP2pBytes = props.maxP2pBytes;
  // 验证 maxP2pBytes 是否在有效范围内
  if((resources->maxP2pBytes <= 0) || (resources->maxP2pBytes > NCCL_MAX_NET_SIZE_BYTES)) {
    // 值无效，记录警告并返回错误
    WARN("recvProxySetup: net plugin returned invalid value for maxP2pBytes %ld \
      [allowed range: %ld - %ld] \n", resources->maxP2pBytes, 0L, NCCL_MAX_NET_SIZE_BYTES);
    return ncclInternalError;
  }

  // 验证响应缓冲区大小（应该能容纳网络句柄）
  if (respSize != sizeof(ncclNetHandle_t)) return ncclInternalError;
  // 开始监听网络连接
  // respBuff: 返回网络句柄
  // netListenComm: 输出监听通信句柄
  NCCLCHECK(proxyState->ncclNet->listen(proxyState->netContext, req->netDev, respBuff, &resources->netListenComm));
  // 标记操作完成
  *done = 1;

  // 返回成功状态码
  return ncclSuccess;
}

// This function embeds plugin-specific rules given the current versions
// 此函数根据当前版本包含特定于插件的规则
// 函数功能：根据网络设备类型和版本确定是否需要设备句柄
// 参数说明：
//   - type: 网络设备类型
//   - version: 设备版本
//   - isRecv: 是否为接收端
//   - handle: 输出参数，返回设备句柄指针
// 返回值：ncclSuccess 表示成功
static ncclResult_t ncclNetGetDeviceHandle(ncclNetDeviceType type, int version, bool isRecv, ncclNetDeviceHandle_t** handle) {
  // 标志：是否需要设备句柄
  bool needsDeviceHandle  = false;

  // 检查设备类型是否为 UNPACK（解包设备）
  if (type == NCCL_NET_DEVICE_UNPACK) {
    // 检查版本是否为 UNPACK 版本且为接收端
    if (version == NCCL_NET_DEVICE_UNPACK_VERSION && isRecv) {
      // 需要设备句柄
      needsDeviceHandle  = true;
    }
  }

  // Don't re-alloc netDeviceHandles
  // 不要重新分配网络设备句柄
  // 如果需要设备句柄且句柄尚未分配
  if (needsDeviceHandle && (*handle == NULL)) {
    // 分配设备句柄结构体
    NCCLCHECK(ncclCalloc(handle, 1));
    // 设置设备类型
    (*handle)->netDeviceType = type;
    // 设置设备版本
    (*handle)->netDeviceVersion = version;
  // 如果不需要设备句柄，设置为 NULL
  } else if (!needsDeviceHandle) {
    *handle = NULL;
  }

  // 返回成功状态码
  return ncclSuccess;
}

// 函数功能：发送端代理连接
// 参数说明：
//   - connection: 代理连接指针
//   - proxyState: 代理状态指针
//   - reqBuff: 请求缓冲区（包含 netSendConnectArgs）
//   - reqSize: 请求大小
//   - respBuff: 响应缓冲区（包含 connectMap）
//   - respSize: 响应大小
//   - done: 输出参数，指示操作是否完成
// 返回值：ncclSuccess 表示成功，ncclInProgress 表示连接进行中
static ncclResult_t sendProxyConnect(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState, void* reqBuff, int reqSize, void* respBuff, int respSize, int* done) {
  // 获取发送端网络资源指针
  struct sendNetResources* resources = (struct sendNetResources*)(connection->transportResources);
  // 验证请求大小
  if (reqSize != sizeof(netSendConnectArgs)) return ncclInternalError;
  // 声明结果变量
  ncclResult_t ret = ncclSuccess;
  // 获取发送连接参数指针
  netSendConnectArgs* req = (netSendConnectArgs*) reqBuff;

  // 设置网络属性
  setNetAttrs(proxyState, &req->netAttr);

  // 获取网络设备句柄（false 表示发送端）
  NCCLCHECK(ncclNetGetDeviceHandle(resources->netDeviceType, resources->netDeviceVersion, false /*isRecv*/, &resources->netDeviceHandle));
  // 检查是否使用共享缓冲区
  if (resources->shared) {
    // Shared buffers
    // 共享缓冲区模式
    // 获取代理进度状态
    struct ncclProxyProgressState* progressState = &proxyState->progressState;
    // 检查本地对等体数组是否已分配
    if (progressState->localPeers == NULL) {
      // 分配本地对等体数组
      NCCLCHECK(ncclCalloc(&progressState->localPeers, proxyState->tpLocalnRanks));
    }
    // 获取本地对等体数组指针
    struct ncclProxyPeer** localPeers = progressState->localPeers;
    // 检查指定 rank 的对等体是否已分配
    if (localPeers[resources->tpLocalRank] == NULL) {
      // 分配对等体结构体
      NCCLCHECK(ncclCalloc(localPeers + resources->tpLocalRank, 1));
    }
    // 设置代理追加指针（指向共享的追加位置）
    connection->proxyAppendPtr = localPeers[resources->tpLocalRank]->send.proxyAppend + resources->channelId;

    // 检查是否支持多接收且启用了共享通信
    if (resources->maxRecvs > 1 && ncclParamNetSharedComms()) {
      // Connect or reuse connection for a netdev/remote rank.
      // 为网络设备/远程 rank 连接或复用连接
      // 检查网络设备的通信数组是否已分配
      if (progressState->netComms[resources->netDev] == NULL) {
        // 分配通信数组（为每个远程 rank 分配一个结构）
        NCCLCHECK(ncclCalloc(progressState->netComms + resources->netDev, proxyState->tpnRanks));
      }
      // 获取共享网络通信结构
      struct ncclSharedNetComms* comms = progressState->netComms[resources->netDev] + resources->tpRemoteRank;
      // let only one localrank connect to a tpRemoteRank to avoid duplicate connections
      // 只允许一个本地 rank 连接到远程 rank，以避免重复连接
      if (comms->activeConnect[resources->channelId] == 0)
        // 标记哪个本地 rank 正在进行连接（+1 是因为 0 表示未连接）
        comms->activeConnect[resources->channelId] = (resources->tpLocalRank + 1);
      // 如果发送通信尚未连接且当前 rank 是激活的连接者
      if (comms->sendComm[resources->channelId] == NULL
          && comms->activeConnect[resources->channelId] == (resources->tpLocalRank + 1)) {
        // 建立网络连接
        ret = proxyState->ncclNet->connect(proxyState->netContext, resources->netDev, req->handle,
            comms->sendComm + resources->channelId, &resources->netDeviceHandle);
      }
      // 使用共享的发送通信
      resources->netSendComm = comms->sendComm[resources->channelId];
      // 增加引用计数
      if (comms->sendComm[resources->channelId]) comms->sendRefCount[resources->channelId]++;
    } else {
      // 不使用共享通信，直接建立连接
      ret = proxyState->ncclNet->connect(proxyState->netContext, resources->netDev, req->handle, &resources->netSendComm, &resources->netDeviceHandle);
    }
  } else {
    // Connect to remote peer
    // 连接到远程对等体
    ret = proxyState->ncclNet->connect(proxyState->netContext, resources->netDev, req->handle, &resources->netSendComm, &resources->netDeviceHandle);
    // 设置代理追加指针为本地追加位置
    connection->proxyAppendPtr = &connection->proxyAppend;
  }

  // 检查连接结果
  if (ret != ncclSuccess) {
    // 连接失败，关闭发送通信
    if (resources->netSendComm) proxyState->ncclNet->closeSend(resources->netSendComm);
    // 返回错误
    NCCLCHECK(ret);
  }
  // 如果发送通信尚未建立，返回进行中状态
  if (resources->netSendComm == NULL) {
    *done = 0;
    return ncclInProgress;
  }
  // 打印网络属性（用于调试）
  printNetAttrs(&req->netAttr, "send connect");
  // 标记操作完成
  *done = 1;

  // 检查是否有设备句柄
  if (resources->netDeviceHandle) {
    // 复制设备句柄到连接
    connection->netDeviceHandle = resources->netDeviceHandle;
    // 设置是否需要代理进度
    connection->needsProxyProgress = connection->netDeviceHandle->needsProxyProgress;
  } else {
    // 没有设备句柄，默认需要代理进度
    connection->needsProxyProgress = 1;
  }

  // Create structures
  // 创建连接映射结构
  struct connectMap* map = &resources->map;
  // 设置标志
  map->sameProcess = connection->sameProcess;
  map->shared = resources->shared;
  // 获取当前 CUDA 设备
  CUDACHECK(cudaGetDevice(&map->cudaDev));

  // 检查是否为非共享模式
  if (resources->shared == 0) { // Only allocate dedicated buffers for ring/tree, not for p2p
    // 遍历所有协议，分配专用缓冲区
    for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
      // 添加缓冲区指针到映射
      // p != NCCL_PROTO_LL: 非 LL 协议
      // resources->useGdr: 使用 GDR
      // proxyState->buffSizes[p]: 缓冲区大小
      NCCL_NET_MAP_ADD_POINTER(map, 0, p!= NCCL_PROTO_LL && resources->useGdr ? 1 : 0, proxyState->buffSizes[p], buffs[p]);
      // 保存缓冲区大小
      resources->buffSizes[p] = proxyState->buffSizes[p];
    }
  } else {
    // Get shared buffers
    // 获取共享缓冲区
    // 根据是否使用 GDR 选择内存库类型
    int bank = resources->useGdr ? NCCL_NET_MAP_SHARED_DEVMEM : NCCL_NET_MAP_SHARED_HOSTMEM;
    // 获取内存映射结构
    struct connectMapMem* mapMem = map->mems+bank;
    // 初始化共享网络缓冲区
    // 参数：代理状态, useGdr, 本地 rank, type=0(发送), 同进程标志, P2P 通道数量
    NCCLCHECK(sharedNetBuffersInit(
          proxyState, resources->useGdr, resources->tpLocalRank, 0, map->sameProcess, proxyState->p2pnChannels,
          &mapMem->gpuPtr, &mapMem->cpuPtr, &mapMem->size, &mapMem->ipcDesc));
    // 设置 Simple 协议的缓冲区大小
    resources->buffSizes[NCCL_PROTO_SIMPLE] = mapMem->size;

    // 检查是否需要分配 P2P 网络 LL 缓冲区
    if (proxyState->allocP2pNetLLBuffers) {
      // 添加 LL 协议缓冲区（主机内存）
      NCCL_NET_MAP_ADD_POINTER(map, 0, 0 /*p == NCCL_PROTO_LL*/, proxyState->buffSizes[NCCL_PROTO_LL], buffs[NCCL_PROTO_LL]);
      // 保存 LL 协议缓冲区大小
      resources->buffSizes[NCCL_PROTO_LL] = proxyState->buffSizes[NCCL_PROTO_LL];
    }

    // 添加 Simple 协议缓冲区指针到映射
    // shared=1, useGdr 决定是否为设备内存
    NCCL_NET_MAP_ADD_POINTER(map, 1, resources->useGdr ? 1 : 0, mapMem->size, buffs[NCCL_PROTO_SIMPLE]);
  }

  // 添加发送内存结构体指针（主机内存）
  NCCL_NET_MAP_ADD_POINTER(map, 0, 0, sizeof(struct ncclSendMem), sendMem);
  // 添加接收内存结构体指针（主机内存）
  NCCL_NET_MAP_ADD_POINTER(map, 0, 0, sizeof(struct ncclRecvMem), recvMem);

  // 检查是否有设备内存需要分配
  if (map->mems[NCCL_NET_MAP_DEVMEM].size) {
    // 非共享模式，需要分配设备内存
    if (resources->shared == 0) {
      // 检查是否需要 IPC（不同进程或启用 cuMem）
      if (!map->sameProcess || ncclCuMemEnable()) {
        // 对齐大小到 CUDA IPC 最小值
        ALIGN_SIZE(map->mems[NCCL_NET_MAP_DEVMEM].size, CUDA_IPC_MIN);
        // 分配可共享的设备内存
        NCCLCHECK(ncclP2pAllocateShareableBuffer(map->mems[NCCL_NET_MAP_DEVMEM].size, 0, &map->mems[NCCL_NET_MAP_DEVMEM].ipcDesc,
                                                 (void**)&map->mems[NCCL_NET_MAP_DEVMEM].gpuPtr));
      } else {
        // 同一进程且未启用 cuMem，直接分配 CUDA 内存
        NCCLCHECK(ncclCudaCalloc(&map->mems[NCCL_NET_MAP_DEVMEM].gpuPtr, map->mems[NCCL_NET_MAP_DEVMEM].size));
      }
      // CPU 指针与 GPU 指针相同（可访问的统一内存）
      map->mems[NCCL_NET_MAP_DEVMEM].cpuPtr = map->mems[NCCL_NET_MAP_DEVMEM].gpuPtr;
    }
  }
  // 检查是否为同一进程
  if (map->sameProcess) {
    // 分配锁定页面的主机内存
    NCCLCHECK(ncclCudaHostCalloc(&map->mems[NCCL_NET_MAP_HOSTMEM].cpuPtr, map->mems[NCCL_NET_MAP_HOSTMEM].size));
    // GPU 指针与 CPU 指针相同
    map->mems[NCCL_NET_MAP_HOSTMEM].gpuPtr = map->mems[NCCL_NET_MAP_HOSTMEM].cpuPtr;
  } else {
    // 不同进程，创建共享内存
    NCCLCHECK(netCreateShm(proxyState, map->mems+NCCL_NET_MAP_HOSTMEM));
    // 获取发送和接收内存的 CPU 指针
    void* sendMem = (void*)NCCL_NET_MAP_GET_POINTER(map, cpu, sendMem);
    void* recvMem = (void*)NCCL_NET_MAP_GET_POINTER(map, cpu, recvMem);
    // 清零内存
    memset(sendMem, 0, sizeof(struct ncclSendMem));
    memset(recvMem, 0, sizeof(struct ncclRecvMem));
  }
  // 检查是否使用 GDR Copy 且需要同步
  if (ncclGdrCopy && map->sameProcess && ncclParamGdrCopySyncEnable()) {
    // 声明 CPU 和 GPU 指针
    uint64_t *cpuPtr, *gpuPtr;
    // 分配 GDR Copy 同步内存
    NCCLCHECK(ncclGdrCudaCalloc(&cpuPtr, &gpuPtr, 1, &resources->gdrDesc));

    // 保存 CPU 同步指针
    resources->gdcSync = cpuPtr;
    // 设置 GDC 内存映射
    struct connectMapMem* gdcMem = map->mems+NCCL_NET_MAP_GDCMEM;
    gdcMem->cpuPtr = (char*)cpuPtr;
    gdcMem->gpuPtr = (char*)gpuPtr;
    gdcMem->size = sizeof(uint64_t); // sendMem->head
  }

  // 获取发送和接收内存的 CPU 指针
  resources->sendMem = (struct ncclSendMem*) NCCL_NET_MAP_GET_POINTER(map, cpu, sendMem);
  resources->recvMem = (struct ncclRecvMem*) NCCL_NET_MAP_GET_POINTER(map, cpu, recvMem);

  // Don't give credits yet in shared mode.
  // 共享模式下暂不给予信用（credits）
  (resources->gdcSync ? *resources->gdcSync : resources->sendMem->head) =
    (map->shared ? -NCCL_STEPS : 0);
  // 初始化连接 FIFO 的大小为 -1（未使用）
  for (int i=0; i<NCCL_STEPS; i++) resources->recvMem->connFifo[i].size = -1;

  // 遍历所有协议，设置缓冲区指针
  for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
    // 获取协议缓冲区的 CPU 指针
    resources->buffers[p] = NCCL_NET_MAP_GET_POINTER(map, cpu, buffs[p]);
    // 检查缓冲区是否存在
    if (resources->buffers[p]) {
#if CUDA_VERSION >= 11070
      /* DMA-BUF support */
      // DMA-BUF 支持
      // 确定内存类型：CUDA 内存或主机内存
      int type = NCCL_NET_MAP_DEV_MEM(map, buffs[p]) ? NCCL_PTR_CUDA : NCCL_PTR_HOST;
      // 如果是 CUDA 内存且支持 DMA-BUF
      if (type == NCCL_PTR_CUDA && resources->useDmaBuf) {
        // 声明 DMA-BUF 文件描述符
        int dmabuf_fd;
        // 获取地址范围的 DMA-BUF 句柄
        // 参数：dmabuf_fd 输出, 缓冲区地址, 大小, DMA-BUF 类型, 标志
        CUCHECK(cuMemGetHandleForAddressRange((void *)&dmabuf_fd, (CUdeviceptr)resources->buffers[p], resources->buffSizes[p], CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, getHandleForAddressRangeFlags(resources->useGdr)));
        // 使用 DMA-BUF 注册内存区域
        NCCLCHECK(proxyState->ncclNet->regMrDmaBuf(resources->netSendComm, resources->buffers[p], resources->buffSizes[p], type, 0ULL, dmabuf_fd, &resources->mhandles[p]));
        // 关闭 DMA-BUF 文件描述符
        (void)close(dmabuf_fd);
      } else // FALL-THROUGH to nv_peermem GDR path
      // 否则，使用标准的 nv_peermem GDR 路径
#endif
      {
        // 标准内存注册
        NCCLCHECK(proxyState->ncclNet->regMr(resources->netSendComm, resources->buffers[p], resources->buffSizes[p], NCCL_NET_MAP_DEV_MEM(map, buffs[p]) ? NCCL_PTR_CUDA : NCCL_PTR_HOST, &resources->mhandles[p]));
      }

      // Copy the mhandle dptr, if implemented
      // 如果实现了，复制内存句柄的设备指针
      if (resources->netDeviceHandle && proxyState->ncclNet->getDeviceMr)
        NCCLCHECK(proxyState->ncclNet->getDeviceMr(resources->netSendComm, resources->mhandles[p], &connection->mhandles[p]));
    }
  }

  //NCCLCHECK(netDumpMap(map));  // 调试用：打印映射信息
  // 验证响应缓冲区大小
  if (respSize != sizeof(struct connectMap)) return ncclInternalError;
  // 复制连接映射到响应缓冲区
  memcpy(respBuff, map, sizeof(struct connectMap));
  // 返回成功状态码
  return ncclSuccess;
}

// 函数功能：接收端代理连接
// 参数说明：
//   - connection: 代理连接指针
//   - proxyState: 代理状态指针
//   - reqBuff: 请求缓冲区（包含 netRecvConnectArgs）
//   - reqSize: 请求大小
//   - respBuff: 响应缓冲区（包含 connectMap）
//   - respSize: 响应大小
//   - done: 输出参数，指示操作是否完成
// 返回值：ncclSuccess 表示成功，ncclInProgress 表示连接进行中
static ncclResult_t recvProxyConnect(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState, void* reqBuff, int reqSize, void* respBuff, int respSize, int* done) {
  // 验证请求大小
  if (reqSize != sizeof(netRecvConnectArgs)) return ncclInternalError;
  // 获取接收端网络资源指针
  struct recvNetResources* resources = (struct recvNetResources*)(connection->transportResources);
  // 获取接收连接参数指针
  netRecvConnectArgs* req = (netRecvConnectArgs*) reqBuff;
  // 设置远程代理 rank
  resources->tpRemoteProxyRank = req->proxyRank;
  // 声明结果变量
  ncclResult_t ret = ncclSuccess;

  // 设置网络属性
  setNetAttrs(proxyState, &req->netAttr);

  // 获取网络设备句柄（true 表示接收端）
  NCCLCHECK(ncclNetGetDeviceHandle(resources->netDeviceType, resources->netDeviceVersion, true /*isRecv*/, &resources->netDeviceHandle));
  // Finish connection establishment from remote peer
  // 完成来自远程对等体的连接建立
  // 检查是否使用共享缓冲区
  if (resources->shared) {
    // Shared buffers
    // 共享缓冲区模式
    // 获取代理进度状态
    struct ncclProxyProgressState* progressState = &proxyState->progressState;
    // 检查本地对等体数组是否已分配
    if (progressState->localPeers == NULL) {
      // 分配本地对等体数组
      NCCLCHECK(ncclCalloc(&progressState->localPeers, proxyState->tpLocalnRanks));
    }
    // 获取本地对等体数组指针
    struct ncclProxyPeer** localPeers = progressState->localPeers;
    // 检查指定 rank 的对等体是否已分配
    if (localPeers[resources->tpLocalRank] == NULL) {
      // 分配对等体结构体
      NCCLCHECK(ncclCalloc(localPeers + resources->tpLocalRank, 1));
    }
    // 设置代理追加指针（指向共享的追加位置）
    connection->proxyAppendPtr = localPeers[resources->tpLocalRank]->recv.proxyAppend + resources->channelId;

    // 检查是否支持多接收且启用了共享通信
    if (resources->maxRecvs > 1 && ncclParamNetSharedComms()) {
      // Connect or reuse connection for a netdev/remote rank.
      // 为网络设备/远程 rank 连接或复用连接
      // 检查网络设备的通信数组是否已分配
      if (progressState->netComms[resources->netDev] == NULL) {
        // 分配通信数组
        NCCLCHECK(ncclCalloc(progressState->netComms + resources->netDev, proxyState->tpnRanks));
      }
      // 获取共享网络通信结构（使用远程代理 rank）
      struct ncclSharedNetComms* comms = progressState->netComms[resources->netDev] + resources->tpRemoteProxyRank;
      // reuse handle to for netdev/remote rank to avoid duplicate connections
      // 复用句柄以避免重复连接
      if (comms->activeAccept[resources->channelId] == 0)
        // 标记哪个本地 rank 正在接受连接
        comms->activeAccept[resources->channelId] = (resources->tpLocalRank + 1);
      //try connecting while comm is null
      // 尝试连接（当通信为空时）
      if (comms->recvComm[resources->channelId] == NULL
         && comms->activeAccept[resources->channelId] == (resources->tpLocalRank + 1)) {
        // 接受网络连接
        ret = proxyState->ncclNet->accept(resources->netListenComm,
            comms->recvComm+resources->channelId, &resources->netDeviceHandle);
      }
      // 使用共享的接收通信
      resources->netRecvComm = comms->recvComm[resources->channelId];
      // 增加引用计数
      if (comms->recvComm[resources->channelId]) comms->recvRefCount[resources->channelId]++;
    } else {
      // 不使用共享通信，直接接受连接
      ret = proxyState->ncclNet->accept(resources->netListenComm, &resources->netRecvComm, &resources->netDeviceHandle);
    }
  } else {
    // Connect to remote peer
    // 连接到远程对等体
    ret = proxyState->ncclNet->accept(resources->netListenComm, &resources->netRecvComm, &resources->netDeviceHandle);
    // 设置代理追加指针为本地追加位置
    connection->proxyAppendPtr = &connection->proxyAppend;
  }

  // 检查连接结果
  NCCLCHECK(ret);
  // 如果接收通信尚未建立，返回进行中状态
  if (resources->netRecvComm == NULL) {
    *done = 0;
    return ncclInProgress;
  }
  // 打印网络属性（用于调试）
  printNetAttrs(&req->netAttr, "recv connect");
  // 标记操作完成
  *done = 1;

  // 检查是否有设备句柄
  if (resources->netDeviceHandle) {
    // 复制设备句柄到连接
    connection->netDeviceHandle = resources->netDeviceHandle;
    // 设置是否需要代理进度
    connection->needsProxyProgress = connection->netDeviceHandle->needsProxyProgress;
  } else {
    // 没有设备句柄，默认需要代理进度
    connection->needsProxyProgress = 1;
  }

  // 关闭监听通信
  NCCLCHECK(proxyState->ncclNet->closeListen(resources->netListenComm));

  // Create structures
  // 创建连接映射结构
  struct connectMap* map = &resources->map;
  // 设置标志
  map->sameProcess = connection->sameProcess;
  // 检查是否为同一进程（接收端不支持远程代理）
  if (map->sameProcess == 0) return ncclInternalError; // We don't support remote proxy for recv
  // 设置共享标志
  map->shared = resources->shared;

  // 检查是否为非共享模式
  if (resources->shared == 0) { // Only allocate dedicated buffers for ring/tree, not for p2p
    // 遍历所有协议，分配专用缓冲区
    for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
      // 添加缓冲区指针到映射（根据 useGdr 决定是否为设备内存）
      NCCL_NET_MAP_ADD_POINTER(map, 0, resources->useGdr ? 1 : 0, proxyState->buffSizes[p], buffs[p]);
      // 保存缓冲区大小
      resources->buffSizes[p] = proxyState->buffSizes[p];
    }
  } else {
    // Get shared buffers
    // 获取共享缓冲区
    // 根据是否使用 GDR 选择内存库类型
    int bank = resources->useGdr ? NCCL_NET_MAP_SHARED_DEVMEM : NCCL_NET_MAP_SHARED_HOSTMEM;
    // 获取内存映射结构
    struct connectMapMem* mapMem = map->mems+bank;
    // 初始化共享网络缓冲区
    // 参数：代理状态, useGdr, 本地 rank, type=1(接收), sameProcess=1, P2P 通道数量
    NCCLCHECK(sharedNetBuffersInit(
          proxyState, resources->useGdr, resources->tpLocalRank, 1, 1, proxyState->p2pnChannels,
          &mapMem->gpuPtr, &mapMem->cpuPtr, &mapMem->size, NULL));
    // 设置 Simple 协议的缓冲区大小
    resources->buffSizes[NCCL_PROTO_SIMPLE] = mapMem->size;
    // 添加 Simple 协议缓冲区指针到映射
    NCCL_NET_MAP_ADD_POINTER(map, 1, resources->useGdr ? 1 : 0, mapMem->size, buffs[NCCL_PROTO_SIMPLE]);
  }

  // 添加发送内存结构体指针（主机内存）
  NCCL_NET_MAP_ADD_POINTER(map, 0, 0, sizeof(struct ncclSendMem), sendMem);
  // 添加接收内存结构体指针（主机内存）
  NCCL_NET_MAP_ADD_POINTER(map, 0, 0, sizeof(struct ncclRecvMem), recvMem);

  // 检查是否需要分配 P2P 网络 LL 缓冲区
  if (proxyState->allocP2pNetLLBuffers) {
    // 添加 LL 协议缓冲区（主机内存）
    NCCL_NET_MAP_ADD_POINTER(map, 0, 0 /*devMem*/, proxyState->buffSizes[NCCL_PROTO_LL], buffs[NCCL_PROTO_LL]);
    // 保存 LL 协议缓冲区大小
    resources->buffSizes[NCCL_PROTO_LL] = proxyState->buffSizes[NCCL_PROTO_LL];
  }

  // 检查是否有设备内存需要分配
  if (map->mems[NCCL_NET_MAP_DEVMEM].size) {
    // 非共享模式，需要分配设备内存
    if (resources->shared == 0) {
      // 检查是否启用了 cuMem
      if (ncclCuMemEnable()) {
        // 分配可共享的设备内存
        NCCLCHECK(ncclP2pAllocateShareableBuffer(map->mems[NCCL_NET_MAP_DEVMEM].size, 0, &map->mems[NCCL_NET_MAP_DEVMEM].ipcDesc,
                                                 (void**)&map->mems[NCCL_NET_MAP_DEVMEM].gpuPtr));
      } else {
        // 直接分配 CUDA 内存
        NCCLCHECK(ncclCudaCalloc(&map->mems[NCCL_NET_MAP_DEVMEM].gpuPtr, map->mems[NCCL_NET_MAP_DEVMEM].size));
      }
      // CPU 指针与 GPU 指针相同
      map->mems[NCCL_NET_MAP_DEVMEM].cpuPtr = map->mems[NCCL_NET_MAP_DEVMEM].gpuPtr;
    }
  }
  // 分配锁定页面的主机内存
  NCCLCHECK(ncclCudaHostCalloc(&map->mems[NCCL_NET_MAP_HOSTMEM].cpuPtr, map->mems[NCCL_NET_MAP_HOSTMEM].size));
  // GPU 指针与 CPU 指针相同
  map->mems[NCCL_NET_MAP_HOSTMEM].gpuPtr = map->mems[NCCL_NET_MAP_HOSTMEM].cpuPtr;
  // 检查是否使用 GDR Copy 且为同一进程
  if (ncclGdrCopy && map->sameProcess) {
    // 声明 CPU 和 GPU 指针
    uint64_t *cpuPtr, *gpuPtr;
    // 分配 GDR Copy 内存（2 个 uint64_t，用于同步和刷新）
    NCCLCHECK(ncclGdrCudaCalloc(&cpuPtr, &gpuPtr, 2, &resources->gdrDesc));

    // 检查是否启用同步
    if (ncclParamGdrCopySyncEnable()) {
      // 保存同步指针
      resources->gdcSync = cpuPtr;
      // 设置 GDC 内存映射
      struct connectMapMem* gdcMem = map->mems+NCCL_NET_MAP_GDCMEM;
      gdcMem->cpuPtr = (char*)cpuPtr;
      gdcMem->gpuPtr = (char*)gpuPtr;
      gdcMem->size = sizeof(uint64_t);
    }
    // 检查是否启用刷新
    if (ncclParamGdrCopyFlushEnable()) resources->gdcFlush = cpuPtr + 1;
  }

  // 获取发送和接收内存的 CPU 指针
  resources->sendMem = (struct ncclSendMem*) NCCL_NET_MAP_GET_POINTER(map, cpu, sendMem);
  resources->recvMem = (struct ncclRecvMem*) NCCL_NET_MAP_GET_POINTER(map, cpu, recvMem);
  // 初始化连接 FIFO 的大小为 -1（未使用）
  for (int i = 0; i < NCCL_STEPS; i++) resources->recvMem->connFifo[i].size = -1;
  // 遍历所有协议，设置缓冲区指针并注册内存
  for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
    // 获取协议缓冲区的 CPU 指针
    resources->buffers[p] = NCCL_NET_MAP_GET_POINTER(map, cpu, buffs[p]);
    // 检查缓冲区是否存在
    if (resources->buffers[p]) {
#if CUDA_VERSION >= 11070
      /* DMA-BUF support */
      // DMA-BUF 支持
      // 确定内存类型
      int type = NCCL_NET_MAP_DEV_MEM(map, buffs[p]) ? NCCL_PTR_CUDA : NCCL_PTR_HOST;
      // 如果是 CUDA 内存且支持 DMA-BUF
      if (type == NCCL_PTR_CUDA && resources->useDmaBuf) {
        // 声明 DMA-BUF 文件描述符
        int dmabuf_fd;
        // 获取地址范围的 DMA-BUF 句柄
        CUCHECK(cuMemGetHandleForAddressRange((void *)&dmabuf_fd, (CUdeviceptr)resources->buffers[p], resources->buffSizes[p], CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, getHandleForAddressRangeFlags(resources->useGdr)));
        // 使用 DMA-BUF 注册内存区域
        NCCLCHECK(proxyState->ncclNet->regMrDmaBuf(resources->netRecvComm, resources->buffers[p], resources->buffSizes[p], type, 0ULL, dmabuf_fd, &resources->mhandles[p]));
        // 关闭 DMA-BUF 文件描述符
        (void)close(dmabuf_fd);
      } else // FALL-THROUGH to nv_peermem GDR path
      // 否则，使用标准的 GDR 路径
#endif
      {
        // 标准内存注册
        NCCLCHECK(proxyState->ncclNet->regMr(resources->netRecvComm, resources->buffers[p], resources->buffSizes[p], NCCL_NET_MAP_DEV_MEM(map, buffs[p]) ? NCCL_PTR_CUDA : NCCL_PTR_HOST, &resources->mhandles[p]));
      }

      // Copy the mhandle dptr
      if (resources->netDeviceType != NCCL_NET_DEVICE_HOST && proxyState->ncclNet->getDeviceMr)
        NCCLCHECK(proxyState->ncclNet->getDeviceMr(resources->netRecvComm, resources->mhandles[p], &connection->mhandles[p]));
    }
  }

  //NCCLCHECK(netDumpMap(map));
  if (respSize != sizeof(struct connectMap)) 
    return ncclInternalError;
  
  memcpy(respBuff, map, sizeof(struct connectMap));
  return ncclSuccess;
}

static ncclResult_t sendProxyFree(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState) {
  struct sendNetResources* resources = (struct sendNetResources*)(connection->transportResources);
  if (connection->state == connSharedInitialized) { // NVB Preconnect
    NCCLCHECK(sharedNetBuffersDestroy(proxyState, connection->tpLocalRank, 0, connection));
    return ncclSuccess;
  }

  if (connection->state == connConnected) {
    for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
      if (resources->buffers[p]) {
        NCCLCHECK(proxyState->ncclNet->deregMr(resources->netSendComm, resources->mhandles[p]));
      }
    }
    struct connectMapMem* mems = resources->map.mems;
    if (resources->map.sameProcess) {
      NCCLCHECK(ncclCudaHostFree(mems[NCCL_NET_MAP_HOSTMEM].cpuPtr));
    } else {
      NCCLCHECK(ncclShmIpcClose(&mems[NCCL_NET_MAP_HOSTMEM].createDesc));
    }
    NCCLCHECK(ncclCudaFree(mems[NCCL_NET_MAP_DEVMEM].cpuPtr));
    if (!resources->map.sameProcess || ncclCuMemEnable()) {
      // cuMem API support
      if (mems[NCCL_NET_MAP_DEVMEM].size) {
        NCCLCHECK(ncclP2pFreeShareableBuffer(&mems[NCCL_NET_MAP_DEVMEM].ipcDesc));
      }
    }
    if (mems[NCCL_NET_MAP_GDCMEM].cpuPtr) NCCLCHECK(ncclGdrCudaFree(resources->gdrDesc));
    if (resources->shared) {
      NCCLCHECK(sharedNetBuffersDestroy(proxyState, resources->tpLocalRank, 0, connection));
      if (resources->maxRecvs > 1 && ncclParamNetSharedComms()) {
        struct ncclSharedNetComms* comms = proxyState->progressState.netComms[resources->netDev]+resources->tpRemoteRank;
        comms->sendRefCount[resources->channelId]--;
        if (comms->sendRefCount[resources->channelId] == 0) NCCLCHECK(proxyState->ncclNet->closeSend(comms->sendComm[resources->channelId]));
      } else {
        NCCLCHECK(proxyState->ncclNet->closeSend(resources->netSendComm));
      }
    } else {
      NCCLCHECK(proxyState->ncclNet->closeSend(resources->netSendComm));
    }
  }

  if (resources) free(resources);
  return ncclSuccess;
}

static ncclResult_t recvProxyFree(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState) {
  struct recvNetResources* resources = (struct recvNetResources*)(connection->transportResources);
  if (connection->state == connSharedInitialized) { // NVB Preconnect
    NCCLCHECK(sharedNetBuffersDestroy(proxyState, connection->tpLocalRank, 1, connection));
    return ncclSuccess;
  }

  if (connection->state == connConnected) {
    for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
      if (resources->buffers[p]) {
        NCCLCHECK(proxyState->ncclNet->deregMr(resources->netRecvComm, resources->mhandles[p]));
      }
    }
    struct connectMapMem* mems = resources->map.mems;
    NCCLCHECK(ncclCudaHostFree(mems[NCCL_NET_MAP_HOSTMEM].cpuPtr));
    NCCLCHECK(ncclCudaFree(mems[NCCL_NET_MAP_DEVMEM].cpuPtr));
    if (!resources->map.sameProcess || ncclCuMemEnable()) {
      // cuMem API support
      if (mems[NCCL_NET_MAP_DEVMEM].size) {
        NCCLCHECK(ncclP2pFreeShareableBuffer(&mems[NCCL_NET_MAP_DEVMEM].ipcDesc));
      }
    }
    if (mems[NCCL_NET_MAP_GDCMEM].cpuPtr) NCCLCHECK(ncclGdrCudaFree(resources->gdrDesc));
    if (resources->shared) {
      NCCLCHECK(sharedNetBuffersDestroy(proxyState, resources->tpLocalRank, 1, connection));
      if (resources->maxRecvs > 1 && ncclParamNetSharedComms()) {
        struct ncclSharedNetComms* comms = proxyState->progressState.netComms[resources->netDev] + resources->tpRemoteProxyRank;
        comms->recvRefCount[resources->channelId]--;
        if (comms->recvRefCount[resources->channelId] == 0) NCCLCHECK(proxyState->ncclNet->closeRecv(comms->recvComm[resources->channelId]));
      } else {
        NCCLCHECK(proxyState->ncclNet->closeRecv(resources->netRecvComm));
      }
    } else {
      NCCLCHECK(proxyState->ncclNet->closeRecv(resources->netRecvComm));
    }
  }

  if (resources) free(resources);
  return ncclSuccess;
}

// 静态断言：检查步进数量是否超过网络最大请求数
// NCCL_STEPS: 步进数量
// NCCL_NET_MAX_REQUESTS: 网络最大请求数
static_assert(NCCL_STEPS <= NCCL_NET_MAX_REQUESTS, "Not enough net requests to cover for steps");

// 函数功能：发送端代理进度跟踪
// 这是一个非常复杂的函数，负责处理发送操作的异步进度
// 参数说明：
//   - proxyState: 代理状态指针
//   - args: 代理参数指针，包含操作的所有信息
// 返回值：ncclSuccess 表示成功，ncclInProgress 表示操作进行中
static ncclResult_t sendProxyProgress(struct ncclProxyState* proxyState, struct ncclProxyArgs* args) {
  // 声明是否已检查网络属性的标志
  int checkedNetAttr = 0;
  // 检查操作状态是否为准备就绪
  if (args->state == ncclProxyOpReady) {
    // 遍历所有子操作
    for (int s=0; s<args->nsubs; s++) {
      // 获取子操作指针
      struct ncclProxySubArgs* sub = args->subs+s;
      // 获取发送端网络资源指针
      struct sendNetResources* resources = (struct sendNetResources*) (sub->connection->transportResources);
      // Round to next multiple of sliceSteps
      // 向上取整到 sliceSteps 的下一个倍数
      // chunkSteps: 每个块的步进数
      sub->base = ROUNDUP(resources->step, args->chunkSteps);
      // Set step base for next op
      // 为下一个操作设置步进基准
      resources->step = sub->base + sub->nsteps;
      // 初始化计数器
      sub->posted = sub->transmitted = sub->done = 0;
      // 记录性能分析事件：操作进入进行中状态
      ncclProfilerRecordProxyOpEventState(s, args, ncclProfilerProxyOpInProgress_v4);
      // 检查是否需要注册内存
      if (!sub->reg)
        // 使用预先注册的内存句柄
        sub->sendMhandle = resources->mhandles[args->protocol];
    }
    // 将操作状态设置为进行中
    args->state = ncclProxyOpProgress;
  }
  // 设置空闲标志（初始假设为空闲）
  args->idle = 1;
  // 检查操作状态是否为进行中
  if (args->state == ncclProxyOpProgress) {
    // 获取协议类型
    int p = args->protocol;
    // 计算最大深度（不能超过 NCCL_STEPS 或共享步进数/子操作数）
    int maxDepth = std::min(NCCL_STEPS, NCCL_SHARED_STEPS/args->nsubs);
    // 遍历所有子操作
    for (int s=0; s<args->nsubs; s++) {
      // 获取子操作指针
      struct ncclProxySubArgs* sub = args->subs+s;
      // 获取当前的步进 ID
      int postedStepId = sub->posted;
      int transmittedStepId = sub->transmitted;
      int doneStepId = sub->done;
      // 检查是否已完成所有步进
      if (sub->done == sub->nsteps) continue;
      // 获取发送端网络资源指针
      struct sendNetResources* resources = (struct sendNetResources*) (sub->connection->transportResources);
      // 获取连接 FIFO 指针（volatile 用于多线程访问）
      volatile struct ncclConnFifo* connFifo = (volatile struct ncclConnFifo*)resources->recvMem->connFifo;
      int stepSize = resources->buffSizes[p] / NCCL_STEPS;
      char* localBuff = NCCL_NET_MAP_GET_POINTER(&resources->map, cpu, buffs[p]);
      // Post buffers to the GPU
      if (sub->posted < sub->nsteps && sub->posted < sub->done + maxDepth) {
        ncclProfilerStartSendProxyStepEvent(s, args, postedStepId);
        int buffSlot = (sub->base+sub->posted)%NCCL_STEPS;
        if (resources->shared) {
          if (!sub->reg) {
            int sharedBuffSlot = sub->posted%maxDepth;
            int offset;
            NCCLCHECK(sharedBuffersGet(proxyState, sub->channelId, sharedBuffSlot*args->nsubs+s, &offset, NULL));
            resources->recvMem->connFifo[buffSlot].offset = offset;
            __sync_synchronize();
          }
          volatile uint64_t* sendHead = resources->gdcSync ? resources->gdcSync : &resources->sendMem->head;
          sub->posted += args->sliceSteps;
          *sendHead = sub->base + sub->posted - NCCL_STEPS;
          if (resources->gdcSync) wc_store_fence(); // Flush out WC write
        } else {
          sub->posted += args->sliceSteps;
        }
        ncclProfilerRecordProxyStepEventState(s, args, postedStepId, ncclProfilerProxyStepSendGPUWait);
        args->idle = 0;
        continue;
      }
      // Check whether we received data from the GPU and send it to the network
      if (sub->transmitted < sub->posted && sub->transmitted < sub->done + NCCL_STEPS) {
        int buffSlot = (sub->base+sub->transmitted)%NCCL_STEPS;
        volatile uint64_t* recvTail = &resources->recvMem->tail;
        uint64_t tail = sub->base + sub->transmitted;
        if (connFifo[buffSlot].size != -1 && (*recvTail > tail || p == NCCL_PROTO_LL)) {
          // We have something to receive, let's check if it's completely ready.
          int size = connFifo[buffSlot].size;
          bool shared = (p == NCCL_PROTO_SIMPLE) && resources->shared;
          char* buff = shared ? localBuff+connFifo[buffSlot].offset : localBuff+buffSlot*stepSize;
          int ready = 1;
          if (p == NCCL_PROTO_LL128) {
            ready = resources->useGdr;
            if (!ready) {
              // When data is in sysmem, we need to wait until all flags are correct since the GPU only
              // called threadfence()
              uint64_t flag = sub->base+sub->transmitted+1;
              int nFifoLines = DIVUP(connFifo[buffSlot].size, sizeof(uint64_t)*NCCL_LL128_LINEELEMS);
              volatile uint64_t* lines = (volatile uint64_t*)buff;
              ready = 1;
              for (int i=0; i<nFifoLines; i++) {
                if (lines[i*NCCL_LL128_LINEELEMS+NCCL_LL128_DATAELEMS] != flag) { ready = 0; break; }
              }
            }
          } else if (p == NCCL_PROTO_LL) {
            uint32_t flag = NCCL_LL_FLAG(sub->base+sub->transmitted+1);
            int nFifoLines = DIVUP(size, sizeof(union ncclLLFifoLine));
            union ncclLLFifoLine* lines = (union ncclLLFifoLine*)buff;
            for (int i=0; i<nFifoLines; i++) {
              volatile uint32_t *f1 = &lines[i].flag1;
              volatile uint32_t *f2 = &lines[i].flag2;
              if (f1[0] != flag || f2[0] != flag) { ready = 0; break; }
            }
          } else if (p == NCCL_PROTO_SIMPLE) {
            if (resources->shared) {
              buff = sub->reg ? (char*)sub->sendbuff + sub->transmitted * NCCL_MAX_NET_SIZE : localBuff + resources->recvMem->connFifo[buffSlot].offset;
            } else if (sub->reg) {
              size_t sendSize;
              sub->ringAlgo->getNextSendAddr(sub->transmitted, (uint8_t**)&buff, &sendSize, &sub->sendMhandle);
              assert(sendSize == size);
            }
          }
          if (ready) {
            ncclProfilerRecordProxyStepEventState(s, args, transmittedStepId, ncclProfilerProxyStepSendPeerWait_v4);
            // Data is ready, try to send.
            // Coverity complains about the size here as pointing to an out-of-scope temporary.  Which is nonsense,
            // since size is a plain integer.
            // coverity[use_invalid:FALSE]
            void* phandle = &sub->pHandles[DIVUP(transmittedStepId, args->sliceSteps)%NCCL_STEPS];
            if (!checkedNetAttr++)
              setXferNetAttrs(proxyState, args, 1);
            NCCLCHECK(proxyState->ncclNet->isend(resources->netSendComm, buff, size, resources->tpRank, sub->sendMhandle, phandle, sub->requests+buffSlot));
            if (sub->requests[buffSlot] != NULL) {
              TRACE(NCCL_NET, "sendProxy [%ld/%d/%d] Isend posted, req %p, buff %p, size %d, proto %d, myRank %d, channelId %d, mhandle %p", sub->transmitted, buffSlot, sub->nsteps, sub->requests[buffSlot], buff, size, p, proxyState->tpRank, sub->channelId, sub->sendMhandle);
              sub->transSize = size;
              sub->transmitted += args->sliceSteps;
              ncclProfilerRecordProxyStepEventState(s, args, transmittedStepId, ncclProfilerProxyStepSendWait);
              args->idle = 0;
              continue;
            }
          }
        }
      }
      // Check whether the network has completed some send operations.
      if (sub->done < sub->transmitted) {
        int done;
        int size;
        int buffSlot = (sub->base+sub->done)%NCCL_STEPS;
        NCCLCHECK(proxyState->ncclNet->test(sub->requests[buffSlot], &done, &size));
        if (done) {
          // Make sure size is reset to -1 before we update the head.
          connFifo[buffSlot].size = -1;
          __sync_synchronize();
          TRACE(NCCL_NET, "sendProxy [%ld/%d/%d] request %p done", sub->done, buffSlot, sub->nsteps, sub->requests[buffSlot]);
          sub->done += args->sliceSteps;
          ncclProfilerStopProxyStepEvent(s, args, doneStepId);

          if (resources->shared == 0) {
            volatile uint64_t* sendHead = resources->gdcSync ? resources->gdcSync : &resources->sendMem->head;
            *sendHead = sub->base + sub->done;
            if (resources->gdcSync) wc_store_fence(); // Flush out WC write
          }
          args->idle = 0;
          if (sub->done == sub->nsteps) {
            args->done++;
            if (sub->ringAlgo && sub->ringAlgo->decRefCount() == 0) delete sub->ringAlgo;
            sub->ringAlgo = NULL;
          }
        }
      }
    }
    if (args->done == args->nsubs) {
      for (int s=0; s<args->nsubs; s++) {
        ncclProfilerStopProxyOpEvent(s, args);
      }
      args->state = ncclProxyOpNone;
    }
  }
  return ncclSuccess;
}

static ncclResult_t recvProxyProgress(struct ncclProxyState* proxyState, struct ncclProxyArgs* args) {
  int checkedNetAttr = 0;
  if (args->state == ncclProxyOpReady) {
    // Initialize subs and group them by same recvComm.
    void* recvComm;
    int groupSize = 0;
    int maxRecvs = 1;
    for (int s=0; s<args->nsubs; s++) {
      struct ncclProxySubArgs* sub = args->subs+s;
      if (groupSize == maxRecvs) {
        groupSize = 0;
      } else if (s>0) { // Find next sub with the same recvComm
        int next;
        for (next=s; next<args->nsubs; next++) {
          struct recvNetResources* nextRes = (struct recvNetResources*) (args->subs[next].connection->transportResources);
          if (nextRes->netRecvComm == recvComm)
            break;
        }
        if (next == args->nsubs) { // Not found
          groupSize = 0;
        } else if (s != next) { // We found a sub later with the same recvComm ; swap subs
          struct ncclProxySubArgs temp;
          memcpy(&temp, sub, sizeof(struct ncclProxySubArgs));
          memcpy(sub, args->subs+next, sizeof(struct ncclProxySubArgs));
          memcpy(args->subs+next, &temp, sizeof(struct ncclProxySubArgs));
        }
      }
      groupSize++;
      struct recvNetResources* resources = (struct recvNetResources*) (sub->connection->transportResources);
      maxRecvs = resources->maxRecvs;
      recvComm = resources->netRecvComm;
      // Round to next multiple of sliceSteps
      sub->base = ROUNDUP(resources->step, args->chunkSteps);
      // Set step base for next op
      resources->step = sub->base + sub->nsteps;
      sub->posted = sub->received = sub->transmitted = sub->done = 0;
      sub->regBufferReady = 0;
      for (int i=0; i<groupSize; i++) sub[-i].groupSize = groupSize;
      ncclProfilerRecordProxyOpEventState(s, args, ncclProfilerProxyOpInProgress_v4);
      if (!sub->reg)
        sub->recvMhandle = resources->mhandles[args->protocol];
    }
    args->state = ncclProxyOpProgress;
  }
  args->idle = 1;
  if (args->state == ncclProxyOpProgress) {
    int p = args->protocol;
    int maxDepth = std::min(NCCL_STEPS, NCCL_SHARED_STEPS/args->nsubs);
    for (int s=0; s<args->nsubs; s+=args->subs[s].groupSize) {
      struct ncclProxySubArgs* subGroup = args->subs+s;
      int subCount = 0;
      void* ptrs[NCCL_PROXY_MAX_SUBS];
      size_t sizes[NCCL_PROXY_MAX_SUBS];
      int tags[NCCL_PROXY_MAX_SUBS];
      void* mhandles[NCCL_PROXY_MAX_SUBS];
      void* phandles[NCCL_PROXY_MAX_SUBS];
      for (int i=0; i<subGroup->groupSize; i++) {
        struct ncclProxySubArgs* sub = subGroup + i;
        int postedStepId = sub->posted;
        if (sub->posted < sub->nsteps) {
          if (sub->posted >= sub->done + maxDepth) { subCount = 0; break; }
          ncclProfilerStartRecvProxyStepEvent(s+i, args, postedStepId);
          struct recvNetResources* resources = (struct recvNetResources*) (sub->connection->transportResources);
          int stepSize = resources->buffSizes[p] / NCCL_STEPS;
          char* localBuff = NCCL_NET_MAP_GET_POINTER(&resources->map, cpu, buffs[p]);
          int buffSlot = (sub->base+sub->posted)%NCCL_STEPS;
          volatile struct ncclConnFifo* connFifo = (volatile struct ncclConnFifo*)resources->recvMem->connFifo;
          if (p == NCCL_PROTO_SIMPLE) {
            if (resources->shared) {
              if (sub->reg) {
                // Wait until CUDA kernel has started before we access the user buffer directly.
                if (!sub->regBufferReady && connFifo[sub->base % NCCL_STEPS].size == -1) continue;
                sub->regBufferReady = 1;
                ptrs[subCount] = sub->recvbuff + sub->posted * NCCL_MAX_NET_SIZE;
                sizes[subCount] = std::min(NCCL_MAX_NET_SIZE, (ssize_t)(sub->nbytes - sub->posted * NCCL_MAX_NET_SIZE));
              } else {
                int sharedBuffSlot = sub->posted % maxDepth;
                int offset;
                NCCLCHECK(sharedBuffersGet(proxyState, sub->channelId, sharedBuffSlot * args->nsubs + s + i, &offset, sizes + subCount));
                connFifo[buffSlot].offset = offset;
                ptrs[subCount] = localBuff + offset;
              }
            } else {
              if (sub->reg) {
                if (!sub->regBufferReady && connFifo[sub->base % NCCL_STEPS].size == -1) continue;
                sub->regBufferReady = 1;
                sub->ringAlgo->getNextRecvAddr(sub->posted, (uint8_t**)&ptrs[subCount], &sizes[subCount], &sub->recvMhandle);
              } else {
                ptrs[subCount] = localBuff + buffSlot * stepSize;
                sizes[subCount] = stepSize * args->sliceSteps;
              }
            }
          } else {
            ptrs[subCount] = localBuff+buffSlot*stepSize;
            sizes[subCount] = stepSize*args->sliceSteps;
          }
          if (sub->nbytes < sizes[subCount]) 
            sizes[subCount] = sub->nbytes;
          tags[subCount] = resources->tpRemoteRank;
          mhandles[subCount] = sub->recvMhandle;
          phandles[subCount] = &sub->pHandles[DIVUP(postedStepId, args->sliceSteps)%NCCL_STEPS];
          subCount++;
        }
      }
      if (subCount) {
        uint64_t step = subGroup->posted;
        struct recvNetResources* resources = (struct recvNetResources*) (subGroup->connection->transportResources);
        void** requestPtr = subGroup->requests+(step%NCCL_STEPS);
        bool ignoreCompletion = ncclParamNetOptionalRecvCompletion() && ((args->protocol == NCCL_PROTO_LL128) || (args->protocol == NCCL_PROTO_LL)) && (subCount == 1);
        if (!checkedNetAttr++)
          setXferNetAttrs(proxyState, args, 0);
        if (ignoreCompletion) 
            *requestPtr = (void *)NCCL_NET_OPTIONAL_RECV_COMPLETION;
        NCCLCHECK(proxyState->ncclNet->irecv(resources->netRecvComm, subCount, ptrs, sizes, tags, mhandles, phandles, requestPtr));
        if (*requestPtr) {
          subGroup->recvRequestsCache[step%NCCL_STEPS] = *requestPtr;
          subGroup->recvRequestsSubCount = subCount;
          for (int i=0; i<subGroup->groupSize; i++) {
            struct ncclProxySubArgs* sub = subGroup+i;
            int postedStepId = sub->posted;
            TRACE(NCCL_NET, "recvProxy [%ld/%ld/%d] Irecv posted, buff %p, size %ld, myRank %d, channelId %d, mhandle %p", sub->posted, (sub->base + sub->posted) % NCCL_STEPS, sub->nsteps, ptrs[i], sizes[i], proxyState->tpRank, sub->channelId, mhandles[i]);
            sub->posted += args->sliceSteps;
            ncclProfilerRecordProxyStepEventState(s+i, args, postedStepId, ncclProfilerProxyStepRecvWait);
          }
          args->idle = 0;
        }
      }
    }
    if (args->idle == 0) return ncclSuccess;

    for (int s=0; s<args->nsubs; s+=args->subs[s].groupSize) {
      struct ncclProxySubArgs* subGroup = args->subs+s;
      if (subGroup->posted > subGroup->received) {
        uint64_t step = subGroup->received;
        int done;
        void* ptrs[NCCL_PROXY_MAX_SUBS];
        int sizes[NCCL_PROXY_MAX_SUBS];
        void* mhandles[NCCL_PROXY_MAX_SUBS];
        for (int i=0; i<NCCL_PROXY_MAX_SUBS; i++) sizes[i] = 0;
        NCCLCHECK(proxyState->ncclNet->test(subGroup->requests[step%NCCL_STEPS], &done, sizes));
        if (done) {
          int needFlush = 0;
          int totalSize = 0;
          for (int i=0; i<NCCL_PROXY_MAX_SUBS; i++) totalSize += sizes[i];
          for (int i=0; i<subGroup->groupSize; i++) {
            struct ncclProxySubArgs* sub = subGroup + i;
            int receivedStepId = sub->received;
            int buffSlot = (sub->base + sub->received) % NCCL_STEPS;
            struct recvNetResources* resources = (struct recvNetResources*)(sub->connection->transportResources);
            volatile struct ncclConnFifo* connFifo = (volatile struct ncclConnFifo*)resources->recvMem->connFifo;
            connFifo[buffSlot].size = -1;
            sub->transSize = sizes[i];
            sub->received += args->sliceSteps;
            ncclProfilerRecordProxyStepEventState(s+i, args, receivedStepId, ncclProfilerProxyStepRecvFlushWait);
            if (step < sub->nsteps) {
              struct recvNetResources* resources = (struct recvNetResources*) (sub->connection->transportResources);
              if (resources->useGdr) needFlush |= resources->needFlush;
            }
          }
          subGroup->requests[step%NCCL_STEPS] = NULL;
          if (totalSize > 0 && p == NCCL_PROTO_SIMPLE && needFlush) {
            // GDRCOPY support
            struct recvNetResources* resources = (struct recvNetResources*) (subGroup->connection->transportResources);
            if (resources->gdcFlush) {
#if defined (__x86_64__)
              // Force a PCI-E read from GPU memory
              asm volatile ("mov (%0), %%eax" :: "l"(resources->gdcFlush) : "%eax");
#else
              WARN("NET: GDR Flush only supported on x86_64");
              return ncclInternalError;
#endif
            } else {
              int subCount = 0;
              for (int i=0; i<subGroup->groupSize; i++) {
                struct ncclProxySubArgs* sub = subGroup + i;
                if (step < sub->nsteps) {
                  struct recvNetResources* resources = (struct recvNetResources*) (sub->connection->transportResources);
                  int stepSize = resources->buffSizes[p] / NCCL_STEPS;
                  char* localBuff = NCCL_NET_MAP_GET_POINTER(&resources->map, cpu, buffs[p]);
                  int buffSlot = (sub->base+sub->received-args->sliceSteps)%NCCL_STEPS;
                  if (resources->shared) {
                    ptrs[subCount] = sub->reg ? (char*)sub->recvbuff + step * NCCL_MAX_NET_SIZE : localBuff + resources->recvMem->connFifo[buffSlot].offset;
                  } else {
                    if (sub->reg) {
                      sub->ringAlgo->getNextRecvAddr(step, (uint8_t**)&ptrs[subCount], NULL, &sub->recvMhandle);
                    } else {
                      ptrs[subCount] = localBuff + buffSlot * stepSize;
                    }
                  }
                  mhandles[subCount] = sub->recvMhandle;
                  subCount++;
                }
              }
              struct recvNetResources* resources = (struct recvNetResources*) (subGroup->connection->transportResources);
              NCCLCHECK(proxyState->ncclNet->iflush(resources->netRecvComm, subCount, ptrs, sizes, mhandles, subGroup->requests+(step%NCCL_STEPS)));
            }
          }
          args->idle = 0;
        }
      }
    }
    if (args->idle == 0) return ncclSuccess;

    for (int s=0; s<args->nsubs; s+=args->subs[s].groupSize) {
      struct ncclProxySubArgs* subGroup = args->subs+s;
      if (subGroup->received > subGroup->transmitted) {
        uint64_t step = subGroup->transmitted;
        int done = 1;
        void* request = subGroup->requests[step%NCCL_STEPS];
        if (request) NCCLCHECK(proxyState->ncclNet->test(request, &done, NULL));
        if (done) {
          for (int i=0; i<subGroup->groupSize; i++) {
            struct ncclProxySubArgs* sub = subGroup + i;
            int transmittedStepId = sub->transmitted;

            sub->transmitted += args->sliceSteps;
            ncclProfilerRecordProxyStepEventState(s+i, args, transmittedStepId, ncclProfilerProxyStepRecvGPUWait);
            if (step < sub->nsteps) {
              __sync_synchronize();
              struct recvNetResources* resources = (struct recvNetResources*) (sub->connection->transportResources);
              volatile uint64_t* recvTail = resources->gdcSync ? resources->gdcSync : &resources->recvMem->tail;
              *recvTail = sub->base + sub->transmitted;
              if (resources->gdcSync) wc_store_fence(); // Flush out WC write
            }
          }
          args->idle = 0;
        }
      }
    }
    if (args->idle == 0) return ncclSuccess;

    for (int s=0; s<args->nsubs; s+=args->subs[s].groupSize) {
      struct ncclProxySubArgs* subGroup = args->subs+s;
      for (int i=0; i<subGroup->groupSize; i++) {
        struct ncclProxySubArgs* sub = subGroup + i;
        if (sub->done == sub->nsteps) continue;
        if (sub->transmitted > sub->done) {
          struct recvNetResources* resources = (struct recvNetResources*) (sub->connection->transportResources);
          volatile uint64_t* sendHead = &resources->sendMem->head;
          uint64_t done = *sendHead;
          while (done > sub->base + sub->done &&
              // LL and LL128 can acknowledge 0-bytes send before they even happen. Don't go past what we transmitted.
              sub->transmitted > sub->done) {
            if (subGroup->recvRequestsCache[sub->done%NCCL_STEPS]) {
              // the multirecv requests are only cached in the first sub.
              if (proxyState->ncclNet->irecvConsumed)
                NCCLCHECK(proxyState->ncclNet->irecvConsumed(resources->netRecvComm, subGroup->recvRequestsSubCount, subGroup->recvRequestsCache[sub->done%NCCL_STEPS]));
              subGroup->recvRequestsCache[sub->done%NCCL_STEPS] = NULL;
            }
            int doneStepId = sub->done;
            sub->done += args->sliceSteps;
            ncclProfilerStopProxyStepEvent(s+i, args, doneStepId);
            args->idle = 0;
            if (sub->done == sub->nsteps) {
              args->done++;
              if (sub->ringAlgo && sub->ringAlgo->decRefCount() == 0) delete sub->ringAlgo;
              sub->ringAlgo = NULL;
              break;
            }
          }
        }
      }
    }
    if (args->done == args->nsubs) {
      args->state = ncclProxyOpNone;
      for (int s=0; s<args->nsubs; s++) {
        ncclProfilerStopProxyOpEvent(s, args);
      }
    }
  }
  return ncclSuccess;
}

// 函数功能：注销网络缓冲区
// 参数说明：
//   - comm: 通信上下文指针
//   - proxyConn: 代理连接器指针
//   - handle: 要注销的内存句柄
// 返回值：ncclSuccess 表示成功
ncclResult_t ncclNetDeregBuffer(struct ncclComm* comm, struct ncclProxyConnector* proxyConn, void* handle) {
  // 向代理发送阻塞式注销请求
  NCCLCHECK(ncclProxyCallBlocking(comm, proxyConn, ncclProxyMsgDeregister, &handle, sizeof(void*), NULL, 0));
  // 记录日志
  INFO(NCCL_REG, "rank %d - deregistered net buffer handle %p", comm->rank, handle);
  // 返回成功状态码
  return ncclSuccess;
}

// 函数功能：注册网络缓冲区（内部函数）
// 参数说明：
//   - comm: 通信上下文指针
//   - userbuff: 用户缓冲区地址
//   - buffSize: 缓冲区大小
//   - peerConns: 对等体连接器数组
//   - nPeers: 对等体数量
//   - regRecord: 注册记录指针
//   - outRegBufFlag: 输出参数，返回是否已注册
//   - outHandle: 输出参数，返回内存句柄
// 返回值：ncclSuccess 表示成功
static ncclResult_t netRegisterBuffer(ncclComm* comm, const void* userbuff, size_t buffSize, struct ncclConnector** peerConns, int nPeers, struct ncclReg* regRecord, int* outRegBufFlag, void** outHandle) {
  // 初始化返回值为成功
  ncclResult_t ret = ncclSuccess;
  // GDR 标志，默认支持 GDR
  int gdrFlag = 1;

  // 检查是否有注册记录
  if (regRecord) {
    // 遍历所有对等体
    for (int p = 0; p < nPeers; ++p) {
      // 获取对等体连接器
      struct ncclConnector* peerConn = peerConns[p];
      // 声明代理连接器指针
      struct ncclProxyConnector* peerProxyConn = NULL;
      // 声明网络句柄指针
      struct ncclRegNetHandles* netHandle = NULL;
      // 是否找到匹配的句柄
      bool found = false;
      // 检查连接器是否有效
      if (peerConn == NULL) continue;
      // 获取代理连接器
      peerProxyConn = &peerConn->proxyConn;
      // 获取网络句柄链表头
      netHandle = regRecord->netHandleHead;
      // 遍历网络句柄链表，查找匹配的代理连接
      while (netHandle) {
        // 检查是否匹配
        if (netHandle->proxyConn == peerProxyConn) {
          found = true;
          break;
        }
        // 移动到下一个节点
        netHandle = netHandle->next;
      }
      // 如果找到了匹配的句柄，复用它
      if (found) {
        // 设置注册标志
        *outRegBufFlag = 1;
        // 返回已存在的句柄
        outHandle[p] = netHandle->handle;
        // 记录日志：复用缓冲区
        INFO(NCCL_REG, "rank %d - NET reuse buffer %p size %ld (baseAddr %p size %ld) handle %p", comm->rank, userbuff, buffSize, (void*)regRecord->begAddr, regRecord->endAddr - regRecord->begAddr, netHandle->handle);
      } else {
        // 需要注册新的缓冲区
        // 创建注册信息结构
        struct netRegInfo info = { regRecord->begAddr, regRecord->endAddr - regRecord->begAddr };
        // 声明句柄变量
        void* handle = NULL;

        // 检查是否支持直接网卡访问（GDR）
        if (peerConn->conn.flags & NCCL_DIRECT_NIC) {
          // 向代理发送阻塞式注册请求
          NCCLCHECKGOTO(ncclProxyCallBlocking(comm, peerProxyConn, ncclProxyMsgRegister, &info, sizeof(struct netRegInfo), &handle, sizeof(void*)), ret, fail);
          // 检查是否成功获取句柄
          if (handle) {
            // 分配网络句柄节点
            struct ncclRegNetHandles* netHandle;
            // 设置注册完成标志
            regRecord->state |= NET_REG_COMPLETE;
            // 分配网络句柄节点内存
            NCCLCHECK(ncclCalloc(&netHandle, 1));
            // 保存内存句柄
            netHandle->handle = handle;
            // 保存代理连接器指针
            netHandle->proxyConn = peerProxyConn;
            // 插入到链表头部
            netHandle->next = regRecord->netHandleHead;
            regRecord->netHandleHead = netHandle;
            // 返回句柄
            outHandle[p] = handle;
            // 设置注册成功标志
            *outRegBufFlag = 1;
            // 记录日志：注册成功
            INFO(NCCL_REG, "rank %d - NET register userbuff %p (handle %p), buffSize %ld", comm->rank, userbuff, handle, buffSize);
          } else {
            // 注册失败，跳转到失败标签
            goto fail;
          }
        } else {
          // 不支持 GDR，设置标志并失败
          gdrFlag = 0;
          goto fail;
        }
      }
    }
  }

// 正常退出标签
exit:
  // 返回结果状态码
  return ret;
// 失败标签
fail:
  // 清除注册标志
  *outRegBufFlag = 0;
  // 记录失败日志
  INFO(NCCL_REG, "rank %d failed to NET register userbuff %p buffSize %ld GDR flag %d", comm->rank, userbuff, buffSize, gdrFlag);
  // 跳转到退出标签
  goto exit;
}

// 函数功能：本地注册网络缓冲区
// 参数说明：
//   - comm: 通信上下文指针
//   - userbuff: 用户缓冲区地址
//   - buffSize: 缓冲区大小
//   - peerConns: 对等体连接器数组
//   - nPeers: 对等体数量
//   - outRegBufFlag: 输出参数，返回是否已注册
//   - outHandle: 输出参数，返回内存句柄
// 返回值：ncclSuccess 表示成功
ncclResult_t ncclNetLocalRegisterBuffer(ncclComm* comm, const void* userbuff, size_t buffSize, struct ncclConnector** peerConns, int nPeers, int* outRegBufFlag, void** outHandle) {
  // 初始化返回值为成功
  ncclResult_t ret = ncclSuccess;
  // 声明注册记录指针
  struct ncclReg *regRecord = NULL;
  // 有效性标志
  bool isValid = false;
  // 基地址和大小
  void *base = NULL;
  size_t baseSize = 0;

  // 初始化注册标志为 0
  *outRegBufFlag = 0;
  // 检查参数有效性
  if (comm && userbuff && buffSize > 0 && nPeers > 0) {
    // 查找注册记录
    NCCLCHECKGOTO(ncclRegFind(comm, userbuff, buffSize, &regRecord), ret, fail);
    // 检查注册记录是否有效
    NCCLCHECKGOTO(ncclRegLocalIsValid(regRecord, &isValid), ret, fail);
    // 如果注册记录有效
    if (isValid) {
      // 获取地址范围（cuMem API）
      CUCHECKGOTO(cuMemGetAddressRange((CUdeviceptr *)&base, &baseSize, (CUdeviceptr)userbuff), ret, fail);
      // 检查地址范围是否覆盖整个缓冲区
      if ((uint64_t)base + baseSize < (uint64_t)userbuff + buffSize) goto exit;
      // 注册网络缓冲区
      NCCLCHECKGOTO(netRegisterBuffer(comm, userbuff, buffSize, peerConns, nPeers, regRecord, outRegBufFlag, outHandle), ret, fail);
    }
  }

// 正常退出标签
exit:
  // 返回结果状态码
  return ret;
// 失败标签
fail:
  // 清除注册标志
  *outRegBufFlag = 0;
  // 跳转到退出标签
  goto exit;
}

// 结构体：网络清理回调
// 用于在 CUDA Graph 销毁时清理网络注册
struct ncclNetCleanupCallback {
  // 基础回调结构
  struct ncclCommCallback base;
  // 通信上下文指针
  struct ncclComm *comm;
  // 注册记录指针
  struct ncclReg *reg;
};

// 函数功能：清理网络注册（回调函数）
// 参数说明：
//   - comm: 通信上下文指针
//   - cb: 回调对象指针
// 返回值：ncclSuccess 表示成功
static ncclResult_t cleanupNet(struct ncclComm* comm, struct ncclCommCallback* cb) {
  // 转换为网络清理回调对象
  struct ncclNetCleanupCallback* obj = (struct ncclNetCleanupCallback*)cb;
  // 注销图注册
  NCCLCHECK(ncclCommGraphDeregister(obj->comm, obj->reg));
  // 释放回调对象
  free(obj);
  // 返回成功状态码
  return ncclSuccess;
}

ncclResult_t ncclNetGraphRegisterBuffer(ncclComm* comm, const void* userbuff, size_t buffSize, struct ncclConnector** peerConns, int nPeers, int* outRegBufFlag, void** outHandle, struct ncclIntruQueue<struct ncclCommCallback, &ncclCommCallback::next>* cleanupQueue, int* nCleanupQueueElts) {
  ncclResult_t ret = ncclSuccess;
  struct ncclNetCleanupCallback *record = NULL;
  struct ncclReg *regRecord = NULL;
  void *base = NULL;
  size_t baseSize = 0;

  *outRegBufFlag = 0;
  if (comm && userbuff && buffSize > 0 && nPeers > 0) {
    CUCHECKGOTO(cuMemGetAddressRange((CUdeviceptr *)&base, &baseSize, (CUdeviceptr)userbuff), ret, fail);
    if ((uint64_t)base + baseSize < (uint64_t)userbuff + buffSize) goto exit;
    NCCLCHECKGOTO(ncclCommGraphRegister(comm, base, baseSize, (void**)&regRecord), ret, fail);
    NCCLCHECKGOTO(netRegisterBuffer(comm, userbuff, buffSize, peerConns, nPeers, regRecord, outRegBufFlag, outHandle), ret, fail);
    if (*outRegBufFlag) {
      NCCLCHECKGOTO(ncclCalloc(&record, 1), ret, fail);
      record->base.fn = cleanupNet;
      record->comm = comm;
      record->reg = regRecord;
      ncclIntruQueueEnqueue(cleanupQueue, (struct ncclCommCallback*)record);
      if (nCleanupQueueElts) *nCleanupQueueElts += 1;
    } else {
      NCCLCHECKGOTO(ncclCommGraphDeregister(comm, regRecord), ret, fail);
    }
  }
exit:
  return ret;
fail:
  *outRegBufFlag = 0;
  goto exit;
}

static ncclResult_t sendProxyRegBuffer(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState, void* reqBuff, int reqSize, void* respBuff, int respSize, int* done) {
  void* handle;
  struct netRegInfo* info = (struct netRegInfo*)reqBuff;
  struct sendNetResources* resources = (struct sendNetResources*)(connection->transportResources);
  ncclResult_t ret = ncclSuccess;
  bool needReg = true;

  assert(reqSize == sizeof(struct netRegInfo));
  assert(respSize == sizeof(void*));

#if CUDART_VERSION >= 11070
  /* DMA-BUF support */
  if (resources->useDmaBuf) {
    int dmabuf_fd;
    CUCHECKGOTO(cuMemGetHandleForAddressRange((void*)&dmabuf_fd, (CUdeviceptr)info->buffer, info->size, CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, getHandleForAddressRangeFlags(resources->useGdr)), ret, peermem);
    NCCLCHECKGOTO(proxyState->ncclNet->regMrDmaBuf(resources->netSendComm, (void*)info->buffer, info->size, NCCL_PTR_CUDA, 0ULL, dmabuf_fd, &handle), ret, peermem);
    (void)close(dmabuf_fd);
    needReg = false;
  }
peermem:
#endif
  if (needReg) {
    NCCLCHECKGOTO(proxyState->ncclNet->regMr(resources->netSendComm, (void*)info->buffer, info->size, NCCL_PTR_CUDA, &handle), ret, fail);
  }

exit:
  memcpy(respBuff, (void*)&handle, sizeof(void*));
  *done = 1;
  return ncclSuccess;
fail:
  handle = NULL;
  goto exit;
}

// 函数功能：接收端代理注册缓冲区
// 参数说明：
//   - connection: 代理连接指针
//   - proxyState: 代理状态指针
//   - reqBuff: 请求缓冲区（包含 netRegInfo）
//   - reqSize: 请求大小
//   - respBuff: 响应缓冲区（返回句柄）
//   - respSize: 响应大小
//   - done: 输出参数，指示操作是否完成
// 返回值：ncclSuccess 表示成功
static ncclResult_t recvProxyRegBuffer(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState, void* reqBuff, int reqSize, void* respBuff, int respSize, int* done) {
  // 声明句柄变量
  void* handle;
  // 获取注册信息指针
  struct netRegInfo* info = (struct netRegInfo*)reqBuff;
  // 获取接收端网络资源指针
  struct recvNetResources* resources = (struct recvNetResources*)(connection->transportResources);
  // 初始化返回值
  ncclResult_t ret = ncclSuccess;
  // 是否需要注册的标志
  bool needReg = true;

  // 验证请求和响应大小
  assert(reqSize == sizeof(struct netRegInfo));
  assert(respSize == sizeof(void*));

#if CUDART_VERSION >= 11070
  /* DMA-BUF support */
  // DMA-BUF 支持
  // 检查是否支持 DMA-BUF
  if (resources->useDmaBuf) {
    // 声明 DMA-BUF 文件描述符
    int dmabuf_fd;
    // 获取地址范围的 DMA-BUF 句柄
    CUCHECKGOTO(cuMemGetHandleForAddressRange((void*)&dmabuf_fd, (CUdeviceptr)info->buffer, info->size, CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, getHandleForAddressRangeFlags(resources->useGdr)), ret, peermem);
    // 使用 DMA-BUF 注册内存
    NCCLCHECKGOTO(proxyState->ncclNet->regMrDmaBuf(resources->netRecvComm, (void*)info->buffer, info->size, NCCL_PTR_CUDA, 0ULL, dmabuf_fd, &handle), ret, peermem);
    // 关闭 DMA-BUF 文件描述符
    (void)close(dmabuf_fd);
    // 不需要标准注册
    needReg = false;
  }
// peermem 标签：DMA-BUF 失败后的跳转点
peermem:
#endif
  // 如果需要标准注册
  if (needReg) {
    // 标准内存注册（peer-memory）
    NCCLCHECKGOTO(proxyState->ncclNet->regMr(resources->netRecvComm, (void*)info->buffer, info->size, NCCL_PTR_CUDA, &handle), ret, fail);
  }

// 正常退出标签
exit:
  // 复制句柄到响应缓冲区
  memcpy(respBuff, (void*)&handle, sizeof(void*));
  // 标记操作完成
  *done = 1;
  // 返回成功状态码
  return ncclSuccess;
// 失败标签
fail:
  // 设置句柄为 NULL
  handle = NULL;
  // 跳转到退出标签
  goto exit;
}

// 函数功能：发送端代理注销缓冲区
// 参数说明：
//   - connection: 代理连接指针
//   - proxyState: 代理状态指针
//   - reqBuff: 请求缓冲区（包含句柄）
//   - reqSize: 请求大小
//   - done: 输出参数，指示操作是否完成
// 返回值：ncclSuccess 表示成功
static ncclResult_t sendProxyDeregBuffer(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState, void* reqBuff, int reqSize, int* done) {
  // 声明句柄变量
  void* handle;
  // 获取发送端网络资源指针
  struct sendNetResources* resources = (struct sendNetResources*)(connection->transportResources);

  // 验证请求大小
  assert(reqSize == sizeof(void*));
  // 从请求缓冲区复制句柄
  memcpy(&handle, reqBuff, sizeof(void*));
  // 注销内存注册
  NCCLCHECK(proxyState->ncclNet->deregMr(resources->netSendComm, handle));
  // 标记操作完成
  *done = 1;
  // 返回成功状态码
  return ncclSuccess;
}

// 函数功能：接收端代理注销缓冲区
// 参数说明：
//   - connection: 代理连接指针
//   - proxyState: 代理状态指针
//   - reqBuff: 请求缓冲区（包含句柄）
//   - reqSize: 请求大小
//   - done: 输出参数，指示操作是否完成
// 返回值：ncclSuccess 表示成功
static ncclResult_t recvProxyDeregBuffer(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState, void* reqBuff, int reqSize, int* done) {
  // 声明句柄变量
  void* handle;
  // 获取接收端网络资源指针
  struct recvNetResources* resources = (struct recvNetResources*)(connection->transportResources);

  // 验证请求大小
  assert(reqSize == sizeof(void*));
  // 从请求缓冲区复制句柄
  memcpy(&handle, reqBuff, sizeof(void*));
  // 注销内存注册
  NCCLCHECK(proxyState->ncclNet->deregMr(resources->netRecvComm, handle));
  // 标记操作完成
  *done = 1;
  // 返回成功状态码
  return ncclSuccess;
}

// 网络传输层结构体定义
// 包含所有网络传输相关的函数指针
struct ncclTransport netTransport = {
  // 传输名称："NET"
  "NET",
  // 检查连接能力的函数
  canConnect,
  // 发送端操作函数集合
  // - sendSetup: 发送端设置
  // - sendConnect: 发送端连接
  // - sendFree: 释放发送端资源
  // - proxySharedInit: 初始化共享代理
  // - sendProxySetup: 发送端代理设置
  // - sendProxyConnect: 发送端代理连接
  // - sendProxyFree: 释放发送端代理资源
  // - sendProxyProgress: 发送端代理进度跟踪
  // - sendProxyRegBuffer: 发送端代理注册缓冲区
  // - sendProxyDeregBuffer: 发送端代理注销缓冲区
  { sendSetup, sendConnect, sendFree, proxySharedInit, sendProxySetup, sendProxyConnect, sendProxyFree, sendProxyProgress, sendProxyRegBuffer, sendProxyDeregBuffer },
  // 接收端操作函数集合
  // - recvSetup: 接收端设置
  // - recvConnect: 接收端连接
  // - recvFree: 释放接收端资源
  // - proxySharedInit: 初始化共享代理
  // - recvProxySetup: 接收端代理设置
  // - recvProxyConnect: 接收端代理连接
  // - recvProxyFree: 释放接收端代理资源
  // - recvProxyProgress: 接收端代理进度跟踪
  // - recvProxyRegBuffer: 接收端代理注册缓冲区
  // - recvProxyDeregBuffer: 接收端代理注销缓冲区
  { recvSetup, recvConnect, recvFree, proxySharedInit, recvProxySetup, recvProxyConnect, recvProxyFree, recvProxyProgress, recvProxyRegBuffer, recvProxyDeregBuffer }
};
