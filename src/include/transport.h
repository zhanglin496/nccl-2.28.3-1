/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// 头文件保护宏，防止重复包含
#ifndef NCCL_TRANSPORT_H_
#define NCCL_TRANSPORT_H_

// 包含设备相关头文件，定义GPU设备和操作
#include "device.h"
// 包含图相关头文件，定义拓扑图结构
#include "graph.h"
// 包含NVML包装层头文件，用于GPU管理
#include "nvmlwrap.h"
// 包含核心定义头文件
#include "core.h"

// 定义传输层类型数量（不包括profiler）
#define NTRANSPORTS 4
// 未定义的传输类型
#define TRANSPORT_UNDEFINED -1
// P2P传输类型（GPU直接GPU通信）
#define TRANSPORT_P2P 0
// 共享内存传输类型（同一节点内GPU通信）
#define TRANSPORT_SHM 1
// 网络传输类型（跨节点通信）
#define TRANSPORT_NET 2
// 集合网络传输类型（专用集合网络硬件）
#define TRANSPORT_COLLNET 3
// 性能分析器传输类型
#define TRANSPORT_PROFILER 4

// 包含代理相关头文件，用于代理模式
#include "proxy.h"
// 包含通信域相关头文件
#include "comm.h"
// 包含引导启动相关头文件
#include "bootstrap.h"

// 声明各种传输层的实例（在对应的.c文件中定义）
extern struct ncclTransport p2pTransport;
extern struct ncclTransport shmTransport;
extern struct ncclTransport netTransport;
extern struct ncclTransport collNetTransport;
extern struct ncclTransport profilerTransport;

// 传输层指针数组，按传输类型索引
extern struct ncclTransport* ncclTransports[];
// 前向声明，避免循环依赖
struct ncclRing;
struct ncclConnector;
struct ncclComm;

// 对等节点信息结构，用于描述远程GPU/进程的属性
struct ncclPeerInfo {
  //全局rank号
  int rank;
  //绑定的dev号
  int cudaDev;

  // NVML设备索引
  int nvmlDev;
  //是否支持gdr，GPUDirect RDMA
  int gdrSupport;

  //主机hash值，检查是否在同一个主机上
  uint64_t hostHash;
  //检查是否是同一个进程
  uint64_t pidHash;

//共享内存设备号
  dev_t shmDev;
  //gpu的busid号
  int64_t busId;
  //属于哪个通信器
  struct ncclComm* comm;
  //cuda计算能力
  int cudaCompCap;
  //全局显存
  size_t totalGlobalMem;
  //mulit-node nvlink
  // MNNVL support，多节点nvlink
  nvmlGpuFabricInfoV_t fabricInfo;
  //是否启用了cumem，这里只表示尝试启用，不代表硬件一定支持cumem
  int cuMemSupport;
  //nccl的版本号
  int version;
};

// 定义连接信息的大小（字节）
#define CONNECT_SIZE 256
// 最大页大小（512MB）
#define NCCL_MAX_PAGE_SIZE (512L * 1024L * 1024L)
// 推荐页大小（2MB）
#define NCCL_REC_PAGE_SIZE (2L * 1024L * 1024L)
// 连接信息结构体，用于存储建立连接所需的数据
struct ncclConnect {
  // 连接数据缓冲区
  char data[CONNECT_SIZE];
};

// 条件编译：仅在CUDA运行时版本>=12.1时编译NVLS相关代码
#if CUDART_VERSION >= 12010

// NVLS（NVLink Fabric）句柄大小
#define NVLS_HANDLE_SIZE 64
// NVLS共享资源结构体
struct ncclNvlsSharedRes {
  // 引用计数，跟踪有多少个用户在使用此资源
  int refCount;
  // 初始化标志
  bool inited;
  // 缓冲区的多播对象属性
  CUmulticastObjectProp bufProp;
  // 信号的多播对象属性
  CUmulticastObjectProp signalProp;
  // 内存访问描述符
  CUmemAccessDesc accessDesc;
  // GPU设备号
  int dev;
  // 单播credit缓冲区大小
  size_t creditUCSize;
  // 多播credit缓冲区大小
  size_t creditMCSize;
  // 单播数据缓冲区大小
  size_t buffUCSize;
  // 多播数据缓冲区大小
  size_t buffMCSize;
  // 多播句柄：用于NVLS数据缓冲区
  CUmemGenericAllocationHandle mcBuffHandle; // Multicast handle for NVLS buffer
  // 多播句柄：用于NVLS credit缓冲区
  CUmemGenericAllocationHandle mcCreditHandle; // Multicast handle for NVLS credit buffer
  // 多播NVLS缓冲区地址
  char* mcBuff; // Multicast NVLS buffer address
  // 多播NVLS credit地址
  char* mcCredit; // Multicast NVLS credit address
  // 单播句柄：用于NVLS数据缓冲区
  CUmemGenericAllocationHandle ucBuffHandle; // Unicast Handle for NVLS buffer
  // 单播句柄：用于NVLS credit缓冲区
  CUmemGenericAllocationHandle ucCreditHandle; // Unicast Handle for NVLS credit buffer
  // 单播NVLS缓冲区地址
  char* ucBuff; // Unicast NVLS buffer address
  // 单播NVLS credit地址
  char* ucCredit; // Unicast NVLS credit address
  // 通道数量
  int nChannels;
  // 头数量（每个通道的并发连接数）
  int nHeads;
  // NVLS共享内存集合缓冲区
  struct ncclShmemCollBuff nvlsShmem;
  // NVLS共享内存句柄
  void *nvlsShmemHandle;
};

#endif /* CUDART_VERSION >= 12010 */

// 集合网络共享资源结构体
struct ncclCollNetSharedRes {
  // 引用计数
  int refCount;
  // 资源大小
  int size;
  // CUDA缓冲区指针（设备内存）
  char* cudaBuff;
  // 主机缓冲区指针（系统内存）
  char* hostBuff;
  // 代理附加参数数组（每个网络设备2个）
  struct ncclProxyArgs* proxyAppend[2*NCCL_MAX_NETDEVS];
  // 资源指针（指向底层集合网络资源）
  void* resources;
  // 通道数量
  int nChannels;
  // 缓冲区大小
  size_t buffSize;
};

// 传输层通信操作函数表结构
// 定义了传输层的各种操作函数指针
struct ncclTransportComm {
  // setup: 设置传输层连接
  // 参数：comm-通信域, graph-拓扑图, 本地对等节点信息, 远程对等节点信息,
  //       连接信息数组, 连接器数组, 通道ID, 连接索引
  ncclResult_t (*setup)(struct ncclComm* comm, struct ncclTopoGraph* graph, struct ncclPeerInfo*, struct ncclPeerInfo*, struct ncclConnect*, struct ncclConnector*, int channelId, int connIndex);
  // connect: 建立到对等节点的连接
  // 参数：comm-通信域, 连接信息数组, rank数量, 本地rank, 连接器数组
  ncclResult_t (*connect)(struct ncclComm* comm, struct ncclConnect*, int nranks, int rank, struct ncclConnector*);
  // free: 释放连接器资源
  ncclResult_t (*free)(struct ncclConnector*);
  // proxySharedInit: 初始化代理共享资源
  ncclResult_t (*proxySharedInit)(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState, int nChannels);
  // proxySetup: 设置代理连接
  ncclResult_t (*proxySetup)(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState, void* reqBuff, int reqSize, void* respBuff, int respSize, int* done);
  // proxyConnect: 通过代理建立连接
  ncclResult_t (*proxyConnect)(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState, void* reqBuff, int reqSize, void* respBuff, int respSize, int* done);
  // proxyFree: 释放代理连接资源
  ncclResult_t (*proxyFree)(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState);
  // proxyProgress: 推进代理操作进度
  ncclResult_t (*proxyProgress)(struct ncclProxyState* proxyState, struct ncclProxyArgs*);
  // proxyRegister: 通过代理注册内存
  ncclResult_t (*proxyRegister)(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState, void* reqBuff, int reqSize, void* respBuff, int respSize, int* done);
  // proxyDeregister: 通过代理注销内存
  ncclResult_t (*proxyDeregister)(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState, void* reqBuff, int reqSize, int* done);
};

// 传输层结构体
// 定义了每种传输类型的基本信息和操作接口
struct ncclTransport {
  // 传输层名称（最多8个字符）
  const char name[8];
  // canConnect: 检查是否可以建立连接
  // 参数：connections-输出连接数组, comm-通信域, graph-拓扑图, 本地对等节点信息, 远程对等节点信息
  ncclResult_t (*canConnect)(int*, struct ncclComm* comm, struct ncclTopoGraph* graph, struct ncclPeerInfo*, struct ncclPeerInfo*);
  // 发送方向的通信操作
  struct ncclTransportComm send;
  // 接收方向的通信操作
  struct ncclTransportComm recv;
};

// P2P传输层：建立P2P连接
// 参数：comm-通信域, channelId-通道ID, nrecv-接收对等节点数量, peerRecv-接收对等节点数组,
//       nsend-发送对等节点数量, peerSend-发送对等节点数组, connIndex-连接索引
ncclResult_t ncclTransportP2pConnect(struct ncclComm* comm, int channelId, int nrecv, int* peerRecv, int nsend, int* peerSend, int connIndex);
// P2P传输层：设置P2P拓扑图
// 参数：comm-通信域, graph-拓扑图, connIndex-连接索引
ncclResult_t ncclTransportP2pSetup(struct ncclComm* comm, struct ncclTopoGraph* graph, int connIndex);
// P2P传输层：检查P2P类型（是否为直接P2P）
// 参数：comm-通信域, isAllDirectP2p-输出是否全部为直接P2P, directMode-输出直接模式标志
ncclResult_t ncclTransportCheckP2pType(struct ncclComm* comm, bool* isAllDirectP2p, bool* directMode);
// P2P传输层：检查是否全部为直接P2P
// 参数：comm-通信域, isAllDirectP2p-输出是否全部为直接P2P
ncclResult_t ncclTransportIsAllDirectP2p(struct ncclComm* comm, int* isAllDirectP2p);

// NVLS传输层：初始化NVLS
// 参数：comm-通信域
ncclResult_t ncclNvlsInit(struct ncclComm* comm);
// NVLS传输层：设置NVLS（父子通信域）
// 参数：comm-通信域, parent-父通信域
ncclResult_t ncclNvlsSetup(struct ncclComm* comm, struct ncclComm* parent);
// NVLS传输层：设置NVLS缓冲区
// 参数：comm-通信域
ncclResult_t ncclNvlsBufferSetup(struct ncclComm* comm);
// NVLS传输层：建立NVLS树连接
// 参数：comm-通信域
ncclResult_t ncclNvlsTreeConnect(struct ncclComm* comm);
// NVLS传输层：在图模式下注册缓冲区
// 参数：comm-通信域, sendbuff-发送缓冲区, recvbuff-接收缓冲区, sendbuffSize-发送缓冲区大小,
//       recvbuffSize-接收缓冲区大小, outRegBufUsed-输出是否使用了注册缓冲区,
//       outRegBufSend-输出注册的发送缓冲区, outRegBufRecv-输出注册的接收缓冲区,
//       cleanupQueue-清理队列, nCleanupQueueElts-清理队列元素数量
ncclResult_t ncclNvlsGraphRegisterBuffer(struct ncclComm *comm, const void *sendbuff, void *recvbuff, size_t sendbuffSize, size_t recvbuffSize, int *outRegBufUsed, void **outRegBufSend, void **outRegBufRecv, struct ncclIntruQueue<struct ncclCommCallback, &ncclCommCallback::next>* cleanupQueue, int* nCleanupQueueElts);
// NVLS传输层：本地注册缓冲区
// 参数：comm-通信域, sendbuff-发送缓冲区, recvbuff-接收缓冲区, sendbuffSize-发送缓冲区大小,
//       recvbuffSize-接收缓冲区大小, outRegBufUsed-输出是否使用了注册缓冲区,
//       outRegBufSend-输出注册的发送缓冲区, outRegBufRecv-输出注册的接收缓冲区
ncclResult_t ncclNvlsLocalRegisterBuffer(struct ncclComm *comm, const void *sendbuff, void *recvbuff, size_t sendbuffSize, size_t recvbuffSize, int *outRegBufUsed, void **outRegBufSend, void **outRegBufRecv);
// NVLS传输层：注销缓冲区
// 参数：comm-通信域, mcHandler-多播句柄, ptr-设备指针, dev-设备号, ucsize-单播大小, mcsize-多播大小
ncclResult_t ncclNvlsDeregBuffer(struct ncclComm* comm, CUmemGenericAllocationHandle *mcHandler, CUdeviceptr ptr, int dev, size_t ucsize, size_t mcsize);
// NVLS传输层：释放NVLS资源
// 参数：comm-通信域
ncclResult_t ncclNvlsFree(struct ncclComm* comm);

// 集合网络类型枚举
enum { collNetRecv=0, collNetSend=1 };
// 设置集合网络传输层：初始化集合网络通信的传输层
// 参数：comm-通信域，collNetGraph-集合网络拓扑图，channel-通信通道，masterRank-主节点秩，masterPeer-主节点对等节点，collNetGraphChannelId-集合网络拓扑图通道ID，type-网络类型（接收/发送），connect-连接信息
bool ncclTransportCollNetSetup(struct ncclComm* comm, struct ncclTopoGraph* collNetGraph, struct ncclChannel* channel, int masterRank, int masterPeer, int collNetGraphChannelId, int type, ncclConnect* connect);
// 检查集合网络传输层设置状态：验证集合网络传输层是否成功初始化
// 参数：comm-通信域，collNetSetupFail-集合网络设置失败标志
ncclResult_t ncclTransportCollNetCheck(struct ncclComm* comm, int collNetSetupFail);
// 释放集合网络传输层资源：清理集合网络传输层的资源
// 参数：comm-通信域
ncclResult_t ncclTransportCollNetFree(struct ncclComm* comm);
// 注册本地集合网络缓冲区：在本地内存中注册用于集合网络通信的缓冲区
// 参数：comm-通信域，userbuff-用户缓冲区指针，buffSize-缓冲区大小，type-缓冲区类型，outRegBufUsed-输出缓冲区使用标志，outHandle-输出句柄
ncclResult_t ncclCollnetLocalRegisterBuffer(struct ncclComm* comm, const void* userbuff, size_t buffSize, int type, int* outRegBufUsed, void** outHandle);
// 注册图集合网络缓冲区：在图拓扑中注册用于集合网络通信的缓冲区
// 参数：comm-通信域，userbuff-用户缓冲区指针，buffSize-缓冲区大小，type-缓冲区类型，outRegBufFlag-输出缓冲区标志，outHandle-输出句柄，cleanupQueue-清理队列，nCleanupQueueElts-清理队列元素数量
ncclResult_t ncclCollnetGraphRegisterBuffer(struct ncclComm* comm, const void* userbuff, size_t buffSize, int type, int* outRegBufFlag, void** outHandle, struct ncclIntruQueue<struct ncclCommCallback, &ncclCommCallback::next>* cleanupQueue, int* nCleanupQueueElts);
// 注销集合网络缓冲区：释放已注册的集合网络缓冲区资源
// 参数：comm-通信域，proxyconn-代理连接器，handle-缓冲区句柄
ncclResult_t ncclCollnetDeregBuffer(struct ncclComm* comm, struct ncclProxyConnector* proxyconn, void* handle);

// 设置环形传输层连接：建立环形拓扑的传输层连接
// 参数：comm-通信域
ncclResult_t ncclTransportRingConnect(struct ncclComm* comm);
// 设置树形传输层连接：建立树形拓扑的传输层连接
// 参数：comm-通信域
ncclResult_t ncclTransportTreeConnect(struct ncclComm* comm);
// 设置模式传输层连接：建立特定模式的传输层连接
// 参数：comm-通信域
ncclResult_t ncclTransportPatConnect(struct ncclComm* comm);

// 初始化集合网络：设置集合网络通信的基础结构
// 参数：comm-通信句柄，parent-父通信句柄，graphs-拓扑图数组
ncclResult_t ncclCollNetSetup(ncclComm_t comm, ncclComm_t parent, struct ncclTopoGraph* graphs[]);
// 设置链式集合网络缓冲区：初始化用于链式通信模式的集合网络缓冲区
// 参数：comm-通信句柄
ncclResult_t ncclCollNetChainBufferSetup(ncclComm_t comm);
// 设置直接集合网络缓冲区：初始化用于直接通信模式的集合网络缓冲区
// 参数：comm-通信句柄
ncclResult_t ncclCollNetDirectBufferSetup(ncclComm_t comm);

// 注销网络缓冲区：释放已注册的网络传输缓冲区资源
// 参数：comm-通信域，proxyConn-代理连接器，handle-缓冲区句柄
ncclResult_t ncclNetDeregBuffer(struct ncclComm* comm, struct ncclProxyConnector* proxyConn, void* handle);
// 注册本地网络缓冲区：在本地内存中注册用于网络通信的缓冲区
// 参数：comm-通信域，userbuff-用户缓冲区指针，buffSize-缓冲区大小，peerConns-对等连接器数组，nPeers-对等节点数量，outRegBufFlag-输出缓冲区标志，outHandle-输出句柄
ncclResult_t ncclNetLocalRegisterBuffer(ncclComm* comm, const void* userbuff, size_t buffSize, struct ncclConnector** peerConns, int nPeers, int* outRegBufFlag, void** outHandle);
// 注册图网络缓冲区：在图拓扑中注册用于网络通信的缓冲区
// 参数：comm-通信域，userbuff-用户缓冲区指针，buffSize-缓冲区大小，peerConns-对等连接器数组，nPeers-对等节点数量，outRegBufFlag-输出缓冲区标志，outHandle-输出句柄，cleanupQueue-清理队列，nCleanupQueueElts-清理队列元素数量
ncclResult_t ncclNetGraphRegisterBuffer(ncclComm* comm, const void* userbuff, size_t buffSize, struct ncclConnector** peerConns, int nPeers, int* outRegBufFlag, void** outHandle, struct ncclIntruQueue<struct ncclCommCallback, &ncclCommCallback::next>* cleanupQueue, int* nCleanupQueueElts);

// 注册P2P IPC缓冲区：注册用于进程间通信的对等P2P缓冲区
// 参数：comm-通信域，userbuff-用户缓冲区指针，size-缓冲区大小，peerRank-对等节点秩，regFlag-注册标志，regAddr-注册地址，cleanupQueue-清理队列
ncclResult_t ncclRegisterP2pIpcBuffer(struct ncclComm* comm, void* userbuff, size_t size, int peerRank, int* regFlag, void** regAddr, struct ncclIntruQueue<struct ncclCommCallback, &ncclCommCallback::next>* cleanupQueue);
// 注册P2P网络缓冲区：注册用于网络传输的对等P2P缓冲区
// 参数：comm-通信域，userbuff-用户缓冲区指针，size-缓冲区大小，conn-连接器，regFlag-注册标志，handle-句柄，cleanupQueue-清理队列
ncclResult_t ncclRegisterP2pNetBuffer(struct ncclComm* comm, void* userbuff, size_t size, struct ncclConnector* conn, int* regFlag, void** handle, struct ncclIntruQueue<struct ncclCommCallback, &ncclCommCallback::next>* cleanupQueue);
// 注册集合缓冲区：为集合操作注册发送和接收缓冲区
// 参数：comm-通信域，info-集合任务信息，outRegBufSend-输出发送缓冲区数组，outRegBufRecv-输出接收缓冲区数组，cleanupQueue-清理队列，regNeedConnect-注册需要连接标志
ncclResult_t ncclRegisterCollBuffers(struct ncclComm* comm, struct ncclTaskColl* info, void* outRegBufSend[NCCL_MAX_LOCAL_RANKS], void* outRegBufRecv[NCCL_MAX_LOCAL_RANKS], struct ncclIntruQueue<struct ncclCommCallback, &ncclCommCallback::next>* cleanupQueue, bool* regNeedConnect);
// 注册NVLS集合缓冲区：为NVLS集合操作注册发送和接收缓冲区
// 参数：comm-通信域，info-集合任务信息，outRegBufSend-输出发送缓冲区数组，outRegBufRecv-输出接收缓冲区数组，cleanupQueue-清理队列，regNeedConnect-注册需要连接标志
ncclResult_t ncclRegisterCollNvlsBuffers(struct ncclComm* comm, struct ncclTaskColl* info, void* outRegBufSend[NCCL_MAX_LOCAL_RANKS], void* outRegBufRecv[NCCL_MAX_LOCAL_RANKS], struct ncclIntruQueue<struct ncclCommCallback, &ncclCommCallback::next>* cleanupQueue, bool* regNeedConnect);
// 查询NVLS注册资源：查询NVLS注册所需的资源信息
// 参数：comm-通信域，info-集合任务信息，recChannels-接收通道数量输出
ncclResult_t ncclNvlsRegResourcesQuery(struct ncclComm* comm, struct ncclTaskColl* info, int* recChannels);

#if CUDART_VERSION >= 12010
// 创建NVLS组：创建多播对象组用于NVLS通信
// 参数：comm-通信域，prop-多播对象属性，rank-节点秩，nranks-节点数量，mcHandle-内存分配句柄，shareableHandle-可共享句柄
ncclResult_t ncclNvlsGroupCreate(struct ncclComm *comm, CUmulticastObjectProp *prop, int rank, unsigned int nranks, CUmemGenericAllocationHandle *mcHandle, char *shareableHandle);
// 连接NVLS组：连接到已有的多播对象组
// 参数：comm-通信域，shareableHandle-可共享句柄，rank-节点秩，mcHandle-内存分配句柄
ncclResult_t ncclNvlsGroupConnect(struct ncclComm *comm, char *shareableHandle, int rank, CUmemGenericAllocationHandle *mcHandle);
#endif

#endif
