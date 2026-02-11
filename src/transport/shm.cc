/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 * 版权所有 (c) 2016-2022，NVIDIA 公司。保留所有权利。
 *
 * See LICENSE.txt for license information
 * 有关许可信息，请参见 LICENSE.txt 文件
 ************************************************************************/

// 包含通信域头文件，定义 ncclComm 结构体和通信相关接口
#include "comm.h"
// 包含共享内存工具头文件，提供共享内存操作的工具函数
#include "shmutils.h"
// 包含共享内存头文件，定义共享内存传输层的接口和数据结构
#include "shm.h"
// 包含传输层头文件，定义传输层的接口
#include "transport.h"

// 定义宏：共享内存路径最大长度
// 用于限制共享内存路径字符串的长度
#define SHM_PATH_MAX 128
// 定义宏：共享内存句柄类型
// 使用 CUDA cuMem 的句柄类型
#define SHM_HANDLE_TYPE ncclCuMemHandleType

// 定义结构体：共享内存缓冲区信息
// 用于存储共享内存缓冲区的指针信息
struct shmBuffInfo {
  // hptr: 主机端指针（Host Pointer）
  // 指向主机可访问的共享内存地址
  void *hptr;
  // dptr: 设备端指针（Device Pointer）
  // 指向设备可访问的共享内存地址（可能与 hptr 相同）
  void *dptr;
};

// 定义结构体：共享内存连接信息
// 用于在对等节点之间交换连接所需的共享内存信息
struct shmConnectInfo {
  // rank: 本节点的 rank 编号
  // 用于标识连接发起方的身份
  int rank;
  // desc: 共享内存 IPC 描述符
  // 用于导入/导出共享内存到其他进程
  ncclShmIpcDesc_t desc;
  // buf: 缓冲区信息
  // 包含主机端和设备端的指针
  struct shmBuffInfo buf;
};

// 定义结构体：共享内存发送端资源
// 用于存储发送端在共享内存传输中所需的资源
struct shmSendResources {
  // remHostMem: 远端主机内存指针
  // 指向接收端在主机端的内存区域
  struct ncclRecvMem* remHostMem;
  // devRemHostMem: 远端主机内存的设备映射指针
  // 设备端可以访问的远端主机内存地址
  struct ncclRecvMem* devRemHostMem;
  // remDesc: 远端共享内存 IPC 描述符
  // 用于导入远端的共享内存
  ncclShmIpcDesc_t remDesc;
  // hostMem: 本地主机内存指针
  // 指向发送端在主机端的内存区域
  struct ncclSendMem* hostMem;
  // devHostMem: 本地主机内存的设备映射指针
  // 设备端可以访问的本地主机内存地址
  struct ncclSendMem* devHostMem;
};

// 定义结构体：共享内存接收端资源
// 用于存储接收端在共享内存传输中所需的资源
struct shmRecvResources {
  // remHostMem: 远端主机内存指针
  // 指向发送端在主机端的内存区域
  struct ncclSendMem* remHostMem;
  // devRemHostMem: 远端主机内存的设备映射指针
  // 设备端可以访问的远端主机内存地址
  struct ncclSendMem* devRemHostMem;
  // remDesc: 远端共享内存 IPC 描述符
  // 用于导入远端的共享内存
  ncclShmIpcDesc_t remDesc;
  // hostMem: 本地主机内存指针
  // 指向接收端在主机端的内存区域
  struct ncclRecvMem* hostMem;
  // devHostMem: 本地主机内存的设备映射指针
  // 设备端可以访问的本地主机内存地址
  struct ncclRecvMem* devHostMem;
};

// 定义结构体：共享内存代理信息
// 用于代理线程（Proxy Thread）处理共享内存传输
struct shmProxyInfo {
  // ceRecvMem: Copy Engine 接收内存指针
  // 用于存储接收端的内存区域信息
  struct ncclRecvMem* ceRecvMem;
  // devFifo: 设备端 FIFO 缓冲区指针
  // 指向设备端的 FIFO 队列，用于数据传输
  char* devFifo;
  // shmFifo: 共享内存 FIFO 缓冲区指针
  // 指向共享内存中的 FIFO 队列
  char* shmFifo;
  // sendMem: 发送内存指针
  // 指向发送端的内存区域
  struct ncclSendMem* sendMem;
  // recvMem: 接收内存指针
  // 指向接收端的内存区域
  struct ncclRecvMem* recvMem;

  // used by progress only
  // 以下字段仅由进度函数使用
  // step: 当前传输的步骤号
  // 用于跟踪传输进度
  uint64_t step;
  // stream: CUDA 流
  // 用于异步执行 CUDA 操作
  cudaStream_t stream;
  // events: CUDA 事件数组
  // 用于同步 CUDA 操作，每个步骤对应一个事件
  cudaEvent_t events[NCCL_STEPS];

  // ipc desc
  // IPC 描述符
  // desc: 共享内存 IPC 描述符
  ncclShmIpcDesc_t desc;
};

// 定义结构体：共享内存请求
// 用于请求分配共享内存缓冲区
struct shmRequest {
  // size: 请求的缓冲区大小
  size_t size;
  // legacy: 是否使用传统 IPC 模式
  // true 表示使用传统 CUDA IPC，false 表示使用 cuMem API
  bool legacy;
};

// 定义宏：发送端（Sender Side）
// 值为 1，表示缓冲区分配在发送端
#define SHM_SEND_SIDE 1
// 定义宏：接收端（Receiver Side）
// 值为 2，表示缓冲区分配在接收端
#define SHM_RECV_SIDE 2
// 定义参数：禁用共享内存传输
// 环境变量 NCCL_SHM_DISABLE，默认值为 0（不禁用）
NCCL_PARAM(ShmDisable, "SHM_DISABLE", 0);
// 定义参数：使用 CUDAMemcpy 进行共享内存传输
// 环境变量 NCCL_SHM_USE_CUDA_MEMCPY，默认值为 0（不使用）
NCCL_PARAM(ShmUseCudaMemcpy, "SHM_USE_CUDA_MEMCPY", 0);
// 定义参数：CUDAMemcpy 模式
// 环境变量 NCCL_SHM_MEMCPY_MODE，默认为发送端
// 1 is sender-side, 2 is receiver-side, 3 is both
// 1 表示发送端，2 表示接收端，3 表示两端都使用
NCCL_PARAM(ShmMemcpyMode, "SHM_MEMCPY_MODE", SHM_SEND_SIDE);
// 静态变量：发送端是否使用 cudaMemcpy
// 初始化为 0，由 initCeOperation 函数设置
static int useMemcpySend = 0;
// 静态变量：接收端是否使用 cudaMemcpy
// 初始化为 0，由 initCeOperation 函数设置
static int useMemcpyRecv = 0;
// 定义参数：共享内存局部性
// 环境变量 NCCL_SHM_LOCALITY，默认为接收端
// 1 is sender-size, 2 is receiver-size
// 1 表示发送端分配，2 表示接收端分配
NCCL_PARAM(ShmLocality, "SHM_LOCALITY", SHM_RECV_SIDE);
// 静态变量：共享内存局部性配置
// 初始化为 0，由 initCeOperation 函数设置
static int shmLocality = 0;
// 前向声明：初始化 Copy Engine 操作
// 这个函数会根据环境变量配置 SHM 传输的行为
static void initCeOperation();

/* Determine two peers can communicate with SHM */
/* 确定两个对等节点是否可以通过共享内存通信 */
// 函数功能：检查两个对等节点是否可以使用共享内存进行通信
// 参数说明：
//   - ret: 输出参数，返回是否可以连接（0=不可连接，1=可连接）
//   - comm: 通信上下文指针
//   - graph: 拓扑图指针
//   - info1: 对等节点 1 的信息
//   - info2: 对等节点 2 的信息
// 返回值：ncclSuccess 表示成功
static ncclResult_t shmCanConnect(int* ret, struct ncclComm* comm, struct ncclTopoGraph* graph, struct ncclPeerInfo* info1, struct ncclPeerInfo* info2) {
  // 初始化返回值为不可连接
  *ret = 0;
  // 初始化 Copy Engine 操作（配置 SHM 传输参数）
  initCeOperation();

  // 检查是否禁用了共享内存传输
  // ncclParamShmDisable: 读取环境变量 NCCL_SHM_DISABLE
  // 如果值为 1，表示禁用共享内存
  if (ncclParamShmDisable() == 1)
    // 返回成功（但 ret=0，表示不可连接）
    return ncclSuccess;

  // 声明变量：是否使用网络传输
  int useNet = 0;
  //检查NET是否比shm更优
  // 调用拓扑检查函数，判断网络传输是否比共享内存更优
  // 某些情况下（如跨节点），网络传输可能更高效
  NCCLCHECK(ncclTopoCheckNet(comm->topo, info1->rank, info2->rank, &useNet));
  // 如果网络传输更优
  if (useNet)
    // 返回成功（但 ret=0，表示不使用共享内存）
    return ncclSuccess;

  // Same host?
  // 是否在同一主机上？
  // 输出跟踪日志，显示两个对等节点的主机哈希值
  // hostHash: 主机哈希值，用于标识是否在同一台物理机器上
  TRACE(NCCL_INIT|NCCL_SHM, "peer1 hostHash %lx peer2 hostHash %lx", info1->hostHash, info2->hostHash);
    //不是同主机，shm不可用
  // 检查两个对等节点的主机哈希值是否相同
  // 如果不同，表示不在同一台物理机器上，共享内存不可用
  if (info1->hostHash != info2->hostHash)
    // 返回成功（但 ret=0，表示不可连接）
    return ncclSuccess;


  // Common /dev/shm (between containers) ?
  // 是否有共同的 /dev/shm（容器之间）？
  //没有共享/dev/shm ，则shm不可用
  // 输出跟踪日志，显示两个对等节点的共享内存设备哈希值
  // shmDev: 共享内存设备的哈希值，用于标识是否可以访问同一共享内存区域
  TRACE(NCCL_INIT|NCCL_SHM, "peer1 shmDev %lx peer2 shmDev %lx", info1->shmDev, info2->shmDev);
  // 检查两个对等节点的共享内存设备哈希值是否相同
  // 如果不同，表示不能访问同一共享内存区域（例如在不同容器中）
  if (info1->shmDev != info2->shmDev)
    // 返回成功（但 ret=0，表示不可连接）
    return ncclSuccess;

  // 所有条件都满足，可以共享内存通信
  // 设置返回值为 1（可以连接）
  *ret = 1;

  // 返回成功状态码
  return ncclSuccess;
}

// 定义宏：共享内存名称最大长度
// 用于限制共享内存对象名称的长度
#define MAX_SHM_NAME_LEN 1024

/* Create and return connect structures for this peer to connect to me */
/* 创建并返回连接结构，供对等节点连接到本节点 */
// 函数功能：设置发送端的共享内存连接
// 参数说明：
//   - comm: 通信上下文指针
//   - graph: 拓扑图指针
//   - myInfo: 本节点的信息
//   - peerInfo: 对等节点的信息
//   - connectInfo: 输出参数，返回连接信息
//   - send: 发送连接器指针
//   - channelId: 通道 ID
//   - connIndex: 连接索引
// 返回值：ncclSuccess 表示成功
static ncclResult_t shmSendSetup(struct ncclComm* comm, struct ncclTopoGraph* graph, struct ncclPeerInfo* myInfo, struct ncclPeerInfo* peerInfo, struct ncclConnect* connectInfo, struct ncclConnector* send, int channelId, int connIndex) {
  // 声明指针：发送端资源结构体
  struct shmSendResources* resources;
  // 将连接信息转换为共享内存连接信息类型
  struct shmConnectInfo* info = (struct shmConnectInfo*)connectInfo;
  // 声明变量：共享内存大小
  // 初始化为发送端内存结构体的大小
  size_t shmSize = sizeof(struct ncclSendMem);
  // 声明变量：共享内存请求
  struct shmRequest req;

  // 编译时断言：确保连接信息结构体不超过 ncclConnect 的大小
  // ncclConnect 是通用的连接信息容器，各种传输层都需要放入其中
  static_assert(sizeof(struct shmConnectInfo) <= sizeof(struct ncclConnect), "shm Connect Info is too big");

  // 分配并清零发送端资源结构体内存
  NCCLCHECK(ncclCalloc(&resources, 1));
  // 将资源结构体保存到发送连接器中
  send->transportResources = resources;

  // 检查共享内存局部性是否为发送端
  // 如果是，需要在发送端分配数据缓冲区
  if (shmLocality == SHM_SEND_SIDE) {
    // 遍历所有协议（LL, LL128, SIMPLE）
    // 将每个协议的缓冲区大小加到总大小中
    for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) shmSize += comm->buffSizes[p];
  }
  // 设置请求的共享内存大小
  req.size = shmSize;
  // 判断是否使用传统 IPC 模式
  // 条件：主机哈希值相同 && 进程哈希值相同
  // 相同主机和进程意味着可以直接使用传统 CUDA IPC
  if (myInfo->hostHash == peerInfo->hostHash && myInfo->pidHash == peerInfo->pidHash)
    // 使用传统 IPC
    req.legacy = true;
  else
    // 使用 cuMem API（跨进程场景）
    req.legacy = false;

  // 连接到代理线程
  // TRANSPORT_SHM: 传输层类型为共享内存
  // 1: 表示这是发送端
  // myInfo->rank: 本节点的 rank
  // &send->proxyConn: 输出代理连接
  NCCLCHECK(ncclProxyConnect(comm, TRANSPORT_SHM, 1, myInfo->rank, &send->proxyConn));
  // 调用代理的 Setup 消息处理函数（阻塞调用）
  // 向代理线程发送共享内存分配请求
  // ncclProxyMsgSetup: 消息类型为 Setup
  // &req: 请求数据（包含大小和模式）
  // sizeof(struct shmRequest): 请求大小
  // info: 响应数据（包含连接信息和缓冲区指针）
  // sizeof(struct shmConnectInfo): 响应大小
  NCCLCHECK(ncclProxyCallBlocking(comm, &send->proxyConn, ncclProxyMsgSetup, (void*)&req, sizeof(struct shmRequest), (void*)info, sizeof(struct shmConnectInfo)));

  // 设置连接信息中的 rank
  info->rank = comm->rank;
  // 保存主机端内存指针到资源结构体
  resources->hostMem = (struct ncclSendMem*)info->buf.hptr;
  // 保存设备端内存指针到资源结构体
  resources->devHostMem = (struct ncclSendMem*)info->buf.dptr;

  // 输出信息日志，显示连接详情
  // 格式：通道 ID : 本节点[设备] -> 对等节点[设备] via SHM/发送端模式/接收端模式
  // CE 表示使用 Copy Engine（cudaMemcpy），direct 表示直接访问
  INFO(NCCL_INIT|NCCL_SHM,"Channel %02d : %d[%d] -> %d[%d] via SHM/%s/%s", channelId, myInfo->rank, myInfo->nvmlDev, peerInfo->rank, peerInfo->nvmlDev, useMemcpySend?"CE":"direct", useMemcpyRecv?"CE":"direct");
  // 返回成功状态码
  return ncclSuccess;
}

// 函数功能：设置接收端的共享内存连接
// 参数说明：
//   - comm: 通信上下文指针
//   - graph: 拓扑图指针
//   - myInfo: 本节点的信息
//   - peerInfo: 对等节点的信息
//   - connectInfo: 输出参数，返回连接信息
//   - recv: 接收连接器指针
//   - channelId: 通道 ID
//   - connIndex: 连接索引
// 返回值：ncclSuccess 表示成功
static ncclResult_t shmRecvSetup(struct ncclComm* comm, struct ncclTopoGraph* graph, struct ncclPeerInfo* myInfo, struct ncclPeerInfo* peerInfo, struct ncclConnect* connectInfo, struct ncclConnector* recv, int channelId, int connIndex) {
  // 声明指针：接收端资源结构体
  struct shmRecvResources* resources;
  // 将连接信息转换为共享内存连接信息类型
  struct shmConnectInfo* info = (struct shmConnectInfo*)connectInfo;
  // 声明变量：共享内存大小
  // 初始化为接收端内存结构体的大小
  size_t shmSize = sizeof(struct ncclRecvMem);
  // 声明变量：共享内存请求
  struct shmRequest req;

  // 分配并清零接收端资源结构体内存
  NCCLCHECK(ncclCalloc(&resources, 1));
  // 将资源结构体保存到接收连接器中
  recv->transportResources = resources;

  // 编译时断言：确保连接信息结构体不超过 ncclConnect 的大小
  static_assert(sizeof(struct shmConnectInfo) <= sizeof(struct ncclConnect), "shm Connect Info is too big");

  // 检查共享内存局部性是否为接收端
  // 如果是，需要在接收端分配数据缓冲区
  if (shmLocality == SHM_RECV_SIDE) {
    // 遍历所有协议，将每个协议的缓冲区大小加到总大小中
    for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) shmSize += comm->buffSizes[p];
  }
  // 设置请求的共享内存大小
  req.size = shmSize;
  // 判断是否使用传统 IPC 模式
  if (myInfo->hostHash == peerInfo->hostHash && myInfo->pidHash == peerInfo->pidHash)
    // 使用传统 IPC
    req.legacy = true;
  else
    // 使用 cuMem API
    req.legacy = false;

  // 连接到代理线程
  // TRANSPORT_SHM: 传输层类型为共享内存
  // 0: 表示这是接收端
  // myInfo->rank: 本节点的 rank
  // &recv->proxyConn: 输出代理连接
  NCCLCHECK(ncclProxyConnect(comm, TRANSPORT_SHM, 0, myInfo->rank, &recv->proxyConn));
  // 调用代理的 Setup 消息处理函数（阻塞调用）
  NCCLCHECK(ncclProxyCallBlocking(comm, &recv->proxyConn, ncclProxyMsgSetup, (void*)&req, sizeof(struct shmRequest), (void*)info, sizeof(struct shmConnectInfo)));

  // 设置连接信息中的 rank
  info->rank = comm->rank;
  // 保存主机端内存指针到资源结构体
  resources->hostMem = (struct ncclRecvMem*)info->buf.hptr;
  // 保存设备端内存指针到资源结构体
  resources->devHostMem = (struct ncclRecvMem*)info->buf.dptr;

  // 返回成功状态码
  return ncclSuccess;
}

/* Connect to this peer */
/* 连接到此对等节点 */
// 函数功能：发送端连接到对等节点
// 参数说明：
//   - comm: 通信上下文指针
//   - connectInfo: 连接信息（包含对等节点的共享内存描述符）
//   - nranks: 总 rank 数量
//   - rank: 要连接的对等节点 rank
//   - send: 发送连接器指针
// 返回值：ncclSuccess 表示成功
static ncclResult_t shmSendConnect(struct ncclComm* comm, struct ncclConnect* connectInfo, int nranks, int rank, struct ncclConnector* send) {
  // Setup device pointers
  // 设置设备指针
  // 将连接信息转换为共享内存连接信息类型
  struct shmConnectInfo* info = (struct shmConnectInfo*)connectInfo;
  // 获取发送端资源结构体指针
  struct shmSendResources* resources = (struct shmSendResources*)send->transportResources;
  // 声明指针：数据缓冲区指针
  char* buff;

  // 导入对等节点的共享内存缓冲区
  // info->rank: 对等节点的 rank（代理线程的 rank）
  // &info->desc: 共享内存 IPC 描述符
  // &resources->remHostMem: 输出远端主机内存指针
  // &resources->devRemHostMem: 输出远端设备内存指针
  // &resources->remDesc: 输出远端共享内存描述符
  NCCLCHECK(ncclShmImportShareableBuffer(comm, info->rank, &info->desc, (void**)&resources->remHostMem, (void**)&resources->devRemHostMem, &resources->remDesc));

  // 确定数据缓冲区的起始位置
  // 根据共享内存局部性决定使用本地还是远端的缓冲区
  // +1: 跳过 ncclSendMem/ncclRecvMem 结构体头部，从数据区开始
  buff = shmLocality == SHM_SEND_SIDE ? (char*)(resources->devHostMem + 1) : (char*)(resources->devRemHostMem + 1);
  // 遍历所有协议，为每个协议设置缓冲区指针
  for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
    // 设置当前协议的缓冲区指针
    send->conn.buffs[p] = buff;
    // 移动缓冲区指针到下一个协议的位置
    buff += comm->buffSizes[p];
  }
  // 设置远端尾指针（接收端的写位置）
  // 发送端需要读取这个指针来知道接收端已经消费了多少数据
  send->conn.tail = &resources->devRemHostMem->tail;
  // 设置本地头指针（发送端的写位置）
  // 发送端通过这个指针通知接收端有多少新数据
  send->conn.head = &resources->devHostMem->head;
  // 设置每个步骤的大小
  // SIMPLE 协议的缓冲区被分成 NCCL_STEPS 个步骤
  send->conn.stepSize = comm->buffSizes[NCCL_PROTO_SIMPLE]/NCCL_STEPS;

  // 检查是否在接收端使用 cudaMemcpy
  if (useMemcpyRecv) {
    // 设置连接 FIFO 指针
    // 用于在接收端使用 Copy Engine 进行数据传输
    send->conn.connFifo = resources->devRemHostMem->connFifo;
  }
  // 检查是否在发送端使用 cudaMemcpy
  if (useMemcpySend) {
    // 创建代理信息结构体
    // ceRecvMem: NULL（由代理线程分配）
    // devFifo: NULL（由代理线程分配）
    // shmFifo: 共享内存 FIFO 指针
    // sendMem: 发送端内存指针
    // recvMem: 接收端内存指针（远端）
    struct shmProxyInfo proxyInfo = { NULL, NULL, send->conn.buffs[NCCL_PROTO_SIMPLE], resources->hostMem, resources->remHostMem };
    // 调用代理的 Connect 消息处理函数
    // 代理线程将创建设备 FIFO 和必要的 CUDA 资源
    NCCLCHECK(ncclProxyCallBlocking(comm, &send->proxyConn, ncclProxyMsgConnect, &proxyInfo, sizeof(struct shmProxyInfo), &proxyInfo, sizeof(struct shmProxyInfo)));
    // 使用代理分配的设备 FIFO 替换原来的共享内存 FIFO
    send->conn.buffs[NCCL_PROTO_SIMPLE] = proxyInfo.devFifo;
    // 使用代理的接收端内存的尾指针
    send->conn.tail = &proxyInfo.ceRecvMem->tail;
    // 使用代理的连接 FIFO
    send->conn.connFifo = proxyInfo.ceRecvMem->connFifo;
  }

  // We must assign the proxyConn's proxyProgress property for proper checking at enqueue-time
  // 我们必须设置 proxyConn 的 proxyProgress 属性，以便在入队时进行正确检查
  send->proxyConn.proxyProgress = shmTransport.send.proxyProgress;

  // 返回成功状态码
  return ncclSuccess;
}

// 函数功能：接收端连接到对等节点
// 参数说明：
//   - comm: 通信上下文指针
//   - connectInfo: 连接信息（包含对等节点的共享内存描述符）
//   - nranks: 总 rank 数量
//   - rank: 要连接的对等节点 rank
//   - recv: 接收连接器指针
// 返回值：ncclSuccess 表示成功
static ncclResult_t shmRecvConnect(struct ncclComm* comm, struct ncclConnect* connectInfo, int nranks, int rank, struct ncclConnector* recv) {
  // Setup device pointers
  // 设置设备指针
  // 获取接收端资源结构体指针
  struct shmRecvResources* resources = (struct shmRecvResources*)recv->transportResources;
  // 将连接信息转换为共享内存连接信息类型
  struct shmConnectInfo* info = (struct shmConnectInfo*)connectInfo;
  // 声明指针：数据缓冲区指针
  char* buff;

  // 导入对等节点的共享内存缓冲区
  NCCLCHECK(ncclShmImportShareableBuffer(comm, info->rank, &info->desc, (void**)&resources->remHostMem, (void**)&resources->devRemHostMem, &resources->remDesc));

  // 确定数据缓冲区的起始位置
  buff = shmLocality == SHM_RECV_SIDE ? (char*)(resources->devHostMem + 1) : (char*)(resources->devRemHostMem + 1);
  // 遍历所有协议，为每个协议设置缓冲区指针
  for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
    // 设置当前协议的缓冲区指针
    recv->conn.buffs[p] = buff;
    // 移动缓冲区指针到下一个协议的位置
    buff += comm->buffSizes[p];
  }
  // 设置远端头指针（发送端的写位置）
  // 接收端需要读取这个指针来知道发送端有多少新数据
  recv->conn.head = &resources->devRemHostMem->head;
  // 设置本地尾指针（接收端的读位置）
  // 接收端通过这个指针通知发送端已经消费了多少数据
  recv->conn.tail = &resources->devHostMem->tail;
  // 设置每个步骤的大小
  recv->conn.stepSize = comm->buffSizes[NCCL_PROTO_SIMPLE]/NCCL_STEPS;

  // 检查是否在接收端使用 cudaMemcpy
  if (useMemcpyRecv) {
    // 创建代理信息结构体
    struct shmProxyInfo proxyInfo = { NULL, NULL, recv->conn.buffs[NCCL_PROTO_SIMPLE], resources->remHostMem, resources->hostMem };
    // 调用代理的 Connect 消息处理函数
    NCCLCHECK(ncclProxyCallBlocking(comm, &recv->proxyConn, ncclProxyMsgConnect, &proxyInfo, sizeof(struct shmProxyInfo), &proxyInfo, sizeof(struct shmProxyInfo)));
    // 使用代理分配的设备 FIFO 替换原来的共享内存 FIFO
    recv->conn.buffs[NCCL_PROTO_SIMPLE] = proxyInfo.devFifo;
    // 使用代理的接收端内存的尾指针
    recv->conn.tail = &proxyInfo.ceRecvMem->tail;
  }

  // We must assign the proxyConn's proxyProgress property for proper checking at enqueue-time
  // 我们必须设置 proxyConn 的 proxyProgress 属性，以便在入队时进行正确检查
  recv->proxyConn.proxyProgress = shmTransport.recv.proxyProgress;

  // 返回成功状态码
  return ncclSuccess;
}

// 函数功能：释放发送端的共享内存连接资源
// 参数说明：
//   - send: 发送连接器指针
// 返回值：ncclSuccess 表示成功
static ncclResult_t shmSendFree(struct ncclConnector* send) {
  // 获取发送端资源结构体指针
  // 注意：这里实际使用的是 shmRecvResources 类型，因为发送端和接收端资源结构体布局相同
  struct shmRecvResources* resources = (struct shmRecvResources*)send->transportResources;
  // 检查资源指针是否有效
  if (resources) {
    // 关闭远端的共享内存 IPC 描述符
    // 释放导入的对等节点的共享内存
    NCCLCHECK(ncclShmIpcClose(&resources->remDesc));
    // 释放资源结构体内存
    free(resources);
    // 清空连接器的资源指针
    send->transportResources = NULL;
  }
  // 返回成功状态码
  return ncclSuccess;
}

// 函数功能：释放接收端的共享内存连接资源
// 参数说明：
//   - recv: 接收连接器指针
// 返回值：ncclSuccess 表示成功
static ncclResult_t shmRecvFree(struct ncclConnector* recv) {
  // 获取接收端资源结构体指针
  struct shmRecvResources* resources = (struct shmRecvResources*)recv->transportResources;
  // 检查资源指针是否有效
  if (resources) {
    // 关闭远端的共享内存 IPC 描述符
    // 释放导入的对等节点的共享内存
    NCCLCHECK(ncclShmIpcClose(&resources->remDesc));
    // 释放资源结构体内存
    free(resources);
    // 清空连接器的资源指针
    recv->transportResources = NULL;
  }
  // 返回成功状态码
  return ncclSuccess;
}

// 函数功能：发送端代理连接处理
// 这是代理线程中处理的函数，用于建立发送端的 Copy Engine 连接
// 参数说明：
//   - connection: 代理连接指针
//   - proxyState: 代理状态指针
//   - reqBuff: 请求缓冲区（包含 shmProxyInfo）
//   - reqSize: 请求大小
//   - respBuff: 响应缓冲区
//   - respSize: 响应大小
//   - done: 输出参数，指示是否完成
// 返回值：ncclSuccess 表示成功
static ncclResult_t shmSendProxyConnect(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState, void* reqBuff, int reqSize, void* respBuff, int respSize, int* done) {
  // 初始化返回值为成功
  ncclResult_t ret = ncclSuccess;
  // 检查请求和响应的大小是否正确
  if (reqSize != sizeof(struct shmProxyInfo) || respSize != sizeof(struct shmProxyInfo)) return ncclInternalError;
  // 声明指针：代理信息结构体
  struct shmProxyInfo* proxyInfo;
  // 将请求缓冲区转换为代理信息类型
  struct shmProxyInfo* reqInfo = (struct shmProxyInfo*)reqBuff;

  // 获取连接的代理资源指针
  proxyInfo = (struct shmProxyInfo*)connection->transportResources;
  // 保存共享内存 FIFO 指针
  proxyInfo->shmFifo = reqInfo->shmFifo;
  // 保存发送端内存指针
  proxyInfo->sendMem = reqInfo->sendMem;
  // 保存接收端内存指针
  proxyInfo->recvMem = reqInfo->recvMem;
  // 在设备上分配 FIFO 缓冲区
  // 大小为 SIMPLE 协议的缓冲区大小
  NCCLCHECKGOTO(ncclCudaCalloc(&proxyInfo->devFifo, proxyState->buffSizes[NCCL_PROTO_SIMPLE]), ret, fail);
  // 在主机上分配 Copy Engine 接收内存（可被 CUDA 访问的主机内存）
  NCCLCHECKGOTO(ncclCudaHostCalloc(&proxyInfo->ceRecvMem, 1), ret, fail);
  // 创建 CUDA 流（非阻塞模式）
  CUDACHECKGOTO(cudaStreamCreateWithFlags(&proxyInfo->stream, cudaStreamNonBlocking), ret, fail);
  // 为每个步骤创建 CUDA 事件
  // NCCL_STEPS 是总步骤数，每个步骤需要一个事件用于同步
  for (int i=0; i<NCCL_STEPS; i++) {
    // 创建 CUDA 事件
    CUDACHECKGOTO(cudaEventCreate(proxyInfo->events+i), ret, fail);
  }
  // 设置代理的追加指针
  connection->proxyAppendPtr = &connection->proxyAppend;
  // 保存代理资源到连接中
  connection->transportResources = proxyInfo;
  // 再次检查响应大小
  if (respSize != sizeof(struct shmProxyInfo)) return ncclInternalError;
  // 将代理信息复制到响应缓冲区
  memcpy(respBuff, proxyInfo, respSize);
  // 设置完成标志
  *done = 1;
// 正常退出标签
exit:
  // 返回结果状态码
  return ret;
// 失败处理标签
fail:
  // 清理已分配的资源
  if (proxyInfo->ceRecvMem) ncclCudaHostFree(proxyInfo->ceRecvMem);
  if (proxyInfo->devFifo) (void)ncclCudaFree(proxyInfo->devFifo);
  // 释放代理信息结构体
  free(proxyInfo);
  // 跳转到正常退出
  goto exit;
}

// 函数功能：接收端代理连接处理
// 这是代理线程中处理的函数，用于建立接收端的 Copy Engine 连接
// 参数说明：
//   - connection: 代理连接指针
//   - proxyState: 代理状态指针
//   - reqBuff: 请求缓冲区（包含 shmProxyInfo）
//   - reqSize: 请求大小
//   - respBuff: 响应缓冲区
//   - respSize: 响应大小
//   - done: 输出参数，指示是否完成
// 返回值：ncclSuccess 表示成功
static ncclResult_t shmRecvProxyConnect(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState, void* reqBuff, int reqSize, void* respBuff, int respSize, int* done) {
  // 初始化返回值为成功
  ncclResult_t ret = ncclSuccess;
  // 检查请求和响应的大小是否正确
  if (reqSize != sizeof(struct shmProxyInfo) || respSize != sizeof(struct shmProxyInfo)) return ncclInternalError;
  // 声明指针：代理信息结构体
  struct shmProxyInfo* proxyInfo;
  // 将请求缓冲区转换为代理信息类型
  struct shmProxyInfo* reqInfo = (struct shmProxyInfo*)reqBuff;

  // 获取连接的代理资源指针
  proxyInfo = (struct shmProxyInfo*)connection->transportResources;
  // 保存共享内存 FIFO 指针
  proxyInfo->shmFifo = reqInfo->shmFifo;
  // 保存发送端内存指针
  proxyInfo->sendMem = reqInfo->sendMem;
  // 保存接收端内存指针
  proxyInfo->recvMem = reqInfo->recvMem;
  // 在设备上分配 FIFO 缓冲区
  NCCLCHECKGOTO(ncclCudaCalloc(&proxyInfo->devFifo, proxyState->buffSizes[NCCL_PROTO_SIMPLE]), ret, fail);
  // 在主机上分配 Copy Engine 接收内存
  NCCLCHECKGOTO(ncclCudaHostCalloc(&proxyInfo->ceRecvMem, 1), ret, fail);
  // 创建 CUDA 流（非阻塞模式）
  CUDACHECKGOTO(cudaStreamCreateWithFlags(&proxyInfo->stream, cudaStreamNonBlocking), ret, fail);
  // 为每个步骤创建 CUDA 事件
  for (int i=0; i<NCCL_STEPS; i++) {
    CUDACHECKGOTO(cudaEventCreate(proxyInfo->events+i), ret, fail);
  }
  // 设置代理的追加指针
  connection->proxyAppendPtr = &connection->proxyAppend;
  // 将代理信息复制到响应缓冲区
  memcpy(respBuff, proxyInfo, respSize);
  // 设置完成标志
  *done = 1;
// 正常退出标签
exit:
  // 返回结果状态码
  return ret;
// 失败处理标签
fail:
  // 清理已分配的资源
  if (proxyInfo->ceRecvMem) ncclCudaHostFree(proxyInfo->ceRecvMem);
  if (proxyInfo->devFifo) (void)ncclCudaFree(proxyInfo->devFifo);
  // 释放代理信息结构体
  free(proxyInfo);
  // 跳转到正常退出
  goto exit;
}

// 函数功能：释放发送端代理的共享内存资源
// 参数说明：
//   - connection: 代理连接指针
//   - proxyState: 代理状态指针
// 返回值：ncclSuccess 表示成功
static ncclResult_t shmSendProxyFree(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState) {
  // 获取代理资源结构体指针
  struct shmProxyInfo* resources = (struct shmProxyInfo*)connection->transportResources;

  // 检查资源指针是否有效
  if (resources) {
    // 检查是否在发送端使用了 cudaMemcpy
    if (useMemcpySend) {
      // 销毁 CUDA 流
      CUDACHECK(cudaStreamDestroy(resources->stream));
      // 释放设备端 FIFO 缓冲区
      NCCLCHECK(ncclCudaFree(resources->devFifo));
      // 释放主机端的 Copy Engine 接收内存
      NCCLCHECK(ncclCudaHostFree(resources->ceRecvMem));
      // 遍历并销毁所有 CUDA 事件
      for (int i=0; i<NCCL_STEPS; i++) {
        CUDACHECK(cudaEventDestroy(resources->events[i]));
      }
    }
    // 关闭共享内存 IPC 描述符
    NCCLCHECK(ncclShmIpcClose(&resources->desc));
    // 释放代理资源结构体内存
    free(connection->transportResources);
    // 清空连接的资源指针
    connection->transportResources = NULL;
  }
  // 返回成功状态码
  return ncclSuccess;
}

// 函数功能：释放接收端代理的共享内存资源
// 参数说明：
//   - connection: 代理连接指针
//   - proxyState: 代理状态指针
// 返回值：ncclSuccess 表示成功
static ncclResult_t shmRecvProxyFree(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState) {
  // 获取代理资源结构体指针
  struct shmProxyInfo* resources = (struct shmProxyInfo*)connection->transportResources;

  // 检查资源指针是否有效
  if (resources) {
    // 检查是否在接收端使用了 cudaMemcpy
    if (useMemcpyRecv) {
      // 销毁 CUDA 流
      CUDACHECK(cudaStreamDestroy(resources->stream));
      // 释放设备端 FIFO 缓冲区
      NCCLCHECK(ncclCudaFree(resources->devFifo));
      // 释放主机端的 Copy Engine 接收内存
      NCCLCHECK(ncclCudaHostFree(resources->ceRecvMem));
      // 遍历并销毁所有 CUDA 事件
      for (int i=0; i<NCCL_STEPS; i++) {
        CUDACHECK(cudaEventDestroy(resources->events[i]));
      }
    }
    // 关闭共享内存 IPC 描述符
    NCCLCHECK(ncclShmIpcClose(&resources->desc));
    // 释放代理资源结构体内存
    free(connection->transportResources);
    // 清空连接的资源指针
    connection->transportResources = NULL;
  }
  // 返回成功状态码
  return ncclSuccess;
}

// 函数功能：发送端代理进度处理函数
// 这个函数在代理线程中被调用，处理使用 Copy Engine（cudaMemcpy）的发送端数据传输
// 参数说明：
//   - proxyState: 代理状态指针
//   - args: 代理参数指针，包含操作的状态和子任务信息
// 返回值：ncclSuccess 表示成功
static ncclResult_t shmSendProxyProgress(struct ncclProxyState* proxyState, struct ncclProxyArgs* args) {
  // 检查操作状态是否为 Ready（准备阶段）
  if (args->state == ncclProxyOpReady) {
    // 遍历所有子任务
    for (int s=0; s<args->nsubs; s++) {
      // 获取子任务指针
      struct ncclProxySubArgs* sub = args->subs+s;
      // 获取子任务的共享内存代理资源
      struct shmProxyInfo* resources = (struct shmProxyInfo*) (sub->connection->transportResources);
      // Round to next multiple of sliceSteps
      // 向上取整到 sliceSteps 的倍数
      // 这样可以确保传输对齐到步骤边界
      sub->base = ROUNDUP(resources->step, args->chunkSteps);
      // 初始化子任务的计数器
      // posted: 已发布的步骤数
      // transmitted: 已传输的步骤数
      // done: 已完成的步骤数
      sub->posted = sub->transmitted = sub->done = 0;
    }
    // 将操作状态设置为 Progress（进行中）
    args->state = ncclProxyOpProgress;
  }
  // 设置空闲标志为 1（假设没有工作要做）
  args->idle = 1;
  // 检查操作状态是否为 Progress
  if (args->state == ncclProxyOpProgress) {
    // 获取协议类型
    int p = args->protocol;
    // 计算每个步骤的大小
    // 总缓冲区大小除以步骤数
    int stepSize = proxyState->buffSizes[p] / NCCL_STEPS;
    // 遍历所有子任务
    for (int s=0; s<args->nsubs; s++) {
      // 获取子任务指针
      struct ncclProxySubArgs* sub = args->subs+s;
      // 获取子任务的共享内存代理资源
      struct shmProxyInfo* resources = (struct shmProxyInfo*) (sub->connection->transportResources);
      // 检查协议类型
      // 只有 SIMPLE 协议使用 cudaMemcpy，其他协议直接访问共享内存
      if (p != NCCL_PROTO_SIMPLE) { // Only Simple uses cudaMemcpy
          // 更新资源步骤位置为 base + nsteps
          resources->step = sub->base + sub->nsteps;
          // 增加完成计数
          args->done++;
          // 跳过此子任务的处理
          continue;
      }
      // 检查是否可以传输更多数据
      // 条件：已传输 < 已完成 + NCCL_STEPS && 已传输 < 总步骤数
      // 这确保传输管道中有足够的缓冲空间
      if (sub->transmitted < sub->done + NCCL_STEPS && sub->transmitted < sub->nsteps) {
        // 计算当前使用的缓冲区槽位
        // 使用模运算实现循环缓冲区
        int buffSlot = (sub->base+sub->transmitted)%NCCL_STEPS;
        // 获取连接 FIFO 指针（用于获取数据大小）
        volatile struct ncclConnFifo* connFifo = resources->ceRecvMem->connFifo;
        // 获取接收尾指针（GPU 已经发送的位置）
        volatile uint64_t* recvTail = &resources->ceRecvMem->tail;
        // Check GPU has sent everything
        // 检查 GPU 是否已经发送了数据
        // 条件：接收尾 > 当前传输位置
        if ((*recvTail > sub->base+sub->transmitted)) {
          // 从 FIFO 中获取数据大小
          int size = connFifo[buffSlot].size;
          // 异步复制数据：从设备 FIFO 到共享内存 FIFO
          // 这是 Copy Engine 操作，将 GPU 数据复制到主机共享内存
          CUDACHECK(cudaMemcpyAsync(resources->shmFifo+buffSlot*stepSize, resources->devFifo+buffSlot*stepSize, size, cudaMemcpyDeviceToHost, resources->stream));
          // 在流中记录事件，用于同步
          CUDACHECK(cudaEventRecord(resources->events[buffSlot], resources->stream));
          // 将数据大小写入接收端的 FIFO
          resources->recvMem->connFifo[buffSlot].size = size;
          // 内存同步屏障
          // 确保 connFifo[].size 的写入对其他线程可见
          __sync_synchronize(); // make sure connFifo[].size is visible
          // 增加已传输的步骤数
          sub->transmitted += args->sliceSteps;
        }
      }
      // 检查是否有已传输但未完成的步骤
      if (sub->done < sub->transmitted) {
        // 计算当前检查的缓冲区槽位
        int buffSlot = (sub->base+sub->done)%NCCL_STEPS;
        // 查询 CUDA 事件状态
        cudaError_t res = cudaEventQuery(resources->events[buffSlot]);
        // 如果事件不是 NotReady 状态（即已完成或出错）
        if (res != cudaErrorNotReady) CUDACHECK(res);
        // 如果复制已完成
        if (res == cudaSuccess) {
          // 增加完成的步骤数
          sub->done += args->sliceSteps;
          // Notify SHM
          // 通知共享内存：更新接收端的尾指针
          // 这告诉接收端有新数据可用
          resources->recvMem->tail = sub->base + sub->done;
        }
        // 检查子任务是否全部完成
        if (sub->done == sub->nsteps) {
          // 更新资源步骤位置
          resources->step = sub->base + sub->nsteps;
          // 增加总完成计数
          args->done++;
        }
      }
    }
    // 检查所有子任务是否都完成
    if (args->done == args->nsubs) {
      // 将操作状态设置为 None（操作完成）
      args->state = ncclProxyOpNone;
    }
  }
  // 返回成功状态码
  return ncclSuccess;
}

// 函数功能：接收端代理进度处理函数
// 这个函数在代理线程中被调用，处理使用 Copy Engine（cudaMemcpy）的接收端数据传输
// 参数说明：
//   - proxyState: 代理状态指针
//   - args: 代理参数指针，包含操作的状态和子任务信息
// 返回值：ncclSuccess 表示成功
static ncclResult_t shmRecvProxyProgress(struct ncclProxyState* proxyState, struct ncclProxyArgs* args) {
  // 检查操作状态是否为 Ready（准备阶段）
  if (args->state == ncclProxyOpReady) {
    // 遍历所有子任务
    for (int s=0; s<args->nsubs; s++) {
      // 获取子任务指针
      struct ncclProxySubArgs* sub = args->subs+s;
      // 获取子任务的共享内存代理资源
      struct shmProxyInfo* resources = (struct shmProxyInfo*) (sub->connection->transportResources);
      // Round to next multiple of sliceSteps
      // 向上取整到 sliceSteps 的倍数
      sub->base = ROUNDUP(resources->step, args->chunkSteps);
      // 初始化子任务的计数器
      sub->posted = sub->transmitted = sub->done = 0;
    }
    // 将操作状态设置为 Progress（进行中）
    args->state = ncclProxyOpProgress;
  }
  // 设置空闲标志为 1（假设没有工作要做）
  args->idle = 1;
  // 检查操作状态是否为 Progress
  if (args->state == ncclProxyOpProgress) {
    // 获取协议类型
    int p = args->protocol;
    // 计算每个步骤的大小
    int stepSize = proxyState->buffSizes[p] / NCCL_STEPS;
    // 遍历所有子任务
    for (int s=0; s<args->nsubs; s++) {
      // 获取子任务指针
      struct ncclProxySubArgs* sub = args->subs+s;
      // 获取子任务的共享内存代理资源
      struct shmProxyInfo* resources = (struct shmProxyInfo*) (sub->connection->transportResources);
      // 检查协议类型
      // 只有 SIMPLE 协议使用 cudaMemcpy，其他协议直接访问共享内存
      if (p != NCCL_PROTO_SIMPLE) { // Only Simple uses cudaMemcpy
          // 更新资源步骤位置为 base + nsteps
          resources->step = sub->base + sub->nsteps;
          // 增加完成计数
          args->done++;
          // 跳过此子任务的处理
          continue;
      }
      // 检查是否可以传输更多数据
      // 条件：已传输 < 已完成 + NCCL_STEPS && 已传输 < 总步骤数
      if (sub->transmitted < sub->done + NCCL_STEPS && sub->transmitted < sub->nsteps) {
        // 计算当前使用的缓冲区槽位
        int buffSlot = (sub->base+sub->transmitted)%NCCL_STEPS;
        // 获取连接 FIFO 指针（用于获取数据大小）
        volatile struct ncclConnFifo* connFifo = resources->recvMem->connFifo;
        // 获取接收尾指针（发送端已经写入共享内存的位置）
        volatile uint64_t* recvTail = &resources->recvMem->tail;
        // Check data is ready in SHM
        // 检查共享内存中是否有新数据
        // 条件：接收尾 > 当前传输位置
        if ((*recvTail > sub->base+sub->transmitted)) {
          // 从 FIFO 中获取数据大小
          int size = connFifo[buffSlot].size;
          // 异步复制数据：从共享内存 FIFO 到设备 FIFO
          // 这是 Copy Engine 操作，将主机共享内存数据复制到 GPU
          CUDACHECK(cudaMemcpyAsync(resources->devFifo+buffSlot*stepSize, resources->shmFifo+buffSlot*stepSize, size, cudaMemcpyHostToDevice, resources->stream));
          // 在流中记录事件，用于同步
          CUDACHECK(cudaEventRecord(resources->events[buffSlot], resources->stream));
          // 增加已传输的步骤数
          sub->transmitted += args->sliceSteps;
        }
      }
      // 检查是否有已传输但未完成的步骤
      if (sub->done < sub->transmitted) {
        // 计算当前检查的缓冲区槽位
        int buffSlot = (sub->base+sub->done)%NCCL_STEPS;
        // 查询 CUDA 事件状态
        cudaError_t res = cudaEventQuery(resources->events[buffSlot]);
        // 如果事件不是 NotReady 状态（即已完成或出错）
        if (res != cudaErrorNotReady) CUDACHECK(res);
        // 如果复制已完成
        if (res == cudaSuccess) {
          // 增加完成的步骤数
          sub->done += args->sliceSteps;
          // Notify GPU
          // 通知 GPU：更新 Copy Engine 接收端的尾指针
          // 这告诉 GPU 有新数据可用
          resources->ceRecvMem->tail = sub->base + sub->done;
        }
        // 检查子任务是否全部完成
        if (sub->done == sub->nsteps) {
          // 更新资源步骤位置
          resources->step = sub->base + sub->nsteps;
          // 增加总完成计数
          args->done++;
        }
      }
    }
    // 检查所有子任务是否都完成
    if (args->done == args->nsubs) {
      // 将操作状态设置为 None（操作完成）
      args->state = ncclProxyOpNone;
    }
  }
  // 返回成功状态码
  return ncclSuccess;
}

// 函数功能：发送端代理设置处理函数
// 这是代理线程中处理的函数，用于为发送端分配共享内存
// 参数说明：
//   - connection: 代理连接指针
//   - proxyState: 代理状态指针
//   - reqBuff: 请求缓冲区（包含 shmRequest）
//   - reqSize: 请求大小
//   - respBuff: 响应缓冲区（返回 shmConnectInfo）
//   - respSize: 响应大小
//   - done: 输出参数，指示是否完成
// 返回值：ncclSuccess 表示成功
static ncclResult_t shmSendProxySetup(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState, void* reqBuff, int reqSize, void* respBuff, int respSize, int* done) {
  // 初始化返回值为成功
  ncclResult_t result = ncclSuccess;
  // 将请求缓冲区转换为共享内存请求类型
  struct shmRequest* req = (struct shmRequest*)reqBuff;
  /* check message size */
  /* 检查消息大小是否正确 */
  if (reqSize != sizeof(struct shmRequest)) return ncclInternalError;
  // 检查响应大小是否正确
  if (respSize != sizeof(struct shmConnectInfo)) return ncclInternalError;

  // 将响应缓冲区转换为共享内存连接信息类型
  struct shmConnectInfo* info = (struct shmConnectInfo*)respBuff;
  // 声明指针：代理信息结构体
  struct shmProxyInfo* proxyInfo;

  // 分配并清零代理信息结构体内存
  NCCLCHECK(ncclCalloc(&proxyInfo, 1));
  // 分配可共享的缓冲区
  // req->size: 请求的缓冲区大小
  // req->legacy: 是否使用传统 IPC 模式
  // &proxyInfo->desc: 输出 IPC 描述符
  // &info->buf.hptr: 输出主机端指针
  // &info->buf.dptr: 输出设备端指针
  NCCLCHECKGOTO(ncclShmAllocateShareableBuffer(req->size, req->legacy, &proxyInfo->desc, &info->buf.hptr, &info->buf.dptr), result, fail);
  // 将 IPC 描述符复制到连接信息中
  memcpy(&info->desc, &proxyInfo->desc, sizeof(ncclShmIpcDesc_t));
  // 保存代理资源到连接中
  connection->transportResources = proxyInfo;
// 正常退出标签
exit:
  // 返回结果状态码
  return result;
// 失败处理标签
fail:
  // 释放代理信息结构体
  free(proxyInfo);
  // 跳转到正常退出
  goto exit;
}

// 函数功能：接收端代理设置处理函数
// 这是代理线程中处理的函数，用于为接收端分配共享内存
// 参数说明：
//   - connection: 代理连接指针
//   - proxyState: 代理状态指针
//   - reqBuff: 请求缓冲区（包含 shmRequest）
//   - reqSize: 请求大小
//   - respBuff: 响应缓冲区（返回 shmConnectInfo）
//   - respSize: 响应大小
//   - done: 输出参数，指示是否完成
// 返回值：ncclSuccess 表示成功
static ncclResult_t shmRecvProxySetup(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState, void* reqBuff, int reqSize, void* respBuff, int respSize, int* done) {
  // 初始化返回值为成功
  ncclResult_t result = ncclSuccess;
  // 将请求缓冲区转换为共享内存请求类型
  struct shmRequest* req = (struct shmRequest*)reqBuff;
  /* check message size */
  /* 检查消息大小是否正确 */
  if (reqSize != sizeof(struct shmRequest)) return ncclInternalError;
  // 检查响应大小是否正确
  if (respSize != sizeof(struct shmConnectInfo)) return ncclInternalError;

  // 将响应缓冲区转换为共享内存连接信息类型
  struct shmConnectInfo* info = (struct shmConnectInfo*)respBuff;
  // 声明指针：代理信息结构体
  struct shmProxyInfo* proxyInfo;

  // 分配并清零代理信息结构体内存
  NCCLCHECK(ncclCalloc(&proxyInfo, 1));
  // 分配可共享的缓冲区
  // req->size: 请求的缓冲区大小
  // req->legacy: 是否使用传统 IPC 模式
  // &proxyInfo->desc: 输出 IPC 描述符
  // &info->buf.hptr: 输出主机端指针
  // &info->buf.dptr: 输出设备端指针
  NCCLCHECKGOTO(ncclShmAllocateShareableBuffer(req->size, req->legacy, &proxyInfo->desc, &info->buf.hptr, &info->buf.dptr), result, fail);
  // 将 IPC 描述符复制到连接信息中
  memcpy(&info->desc, &proxyInfo->desc, sizeof(ncclShmIpcDesc_t));
  // 保存代理资源到连接中
  connection->transportResources = proxyInfo;
// 正常退出标签
exit:
  // 返回结果状态码
  return result;
// 失败处理标签
fail:
  // 释放代理信息结构体
  free(proxyInfo);
  // 跳转到正常退出
  goto exit;
}

// 函数功能：初始化 Copy Engine 操作
// 这个函数根据环境变量配置 SHM 传输的行为
// 使用静态变量确保只初始化一次
static void initCeOperation() {
  // 静态变量：初始化标志
  // 0 表示未初始化，1 表示已初始化
  static int init = 0;
  // 检查是否已初始化
  if (!init) {
    // 计算发送端是否使用 cudaMemcpy
    // 条件：启用了 SHM_USE_CUDA_MEMCPY && SHM_MEMCPY_MODE 的 bit 0 被设置
    // bit 0 (值 1) 表示发送端使用
    useMemcpySend = ncclParamShmUseCudaMemcpy() && (ncclParamShmMemcpyMode() & 1);
    // 计算接收端是否使用 cudaMemcpy
    // 条件：启用了 SHM_USE_CUDA_MEMCPY && SHM_MEMCPY_MODE 的 bit 1 被设置
    // bit 1 (值 2) 表示接收端使用
    useMemcpyRecv = ncclParamShmUseCudaMemcpy() && (ncclParamShmMemcpyMode() & 2);
    // 如果发送端使用 cudaMemcpy，设置相应的代理函数
    if (useMemcpySend) {
      // 设置代理连接函数
      shmTransport.send.proxyConnect = shmSendProxyConnect;
      // 设置代理进度函数
      shmTransport.send.proxyProgress = shmSendProxyProgress;
    }
    // 如果接收端使用 cudaMemcpy，设置相应的代理函数
    if (useMemcpyRecv) {
      // 设置代理连接函数
      shmTransport.recv.proxyConnect = shmRecvProxyConnect;
      // 设置代理进度函数
      shmTransport.recv.proxyProgress = shmRecvProxyProgress;
    }
    // 获取共享内存局部性配置
    // 决定缓冲区分配在发送端还是接收端
    shmLocality = ncclParamShmLocality();
    // 验证局部性配置的有效性
    if (shmLocality != SHM_SEND_SIDE && shmLocality != SHM_RECV_SIDE) {
      // 输出警告日志，配置无效
      WARN("Ignoring SHM locality, must be 1 (sender side) or 2 (receiver side, default)");
      // 使用默认值（接收端）
      shmLocality = SHM_RECV_SIDE;
    }
    // 标记为已初始化
    init = 1;
  }
}

// 函数功能：分配可共享的缓冲区
// 这个函数分配可以在进程间共享的内存缓冲区
// 参数说明：
//   - size: 要分配的缓冲区大小
//   - legacy: 是否使用传统 IPC 模式（true=传统 CUDA IPC，false=cuMem API）
//   - desc: 输出参数，返回 IPC 描述符（用于导入/导出）
//   - hptr: 输出参数，返回主机端指针
//   - dptr: 输出参数，返回设备端指针（可选，可以为 NULL）
// 返回值：ncclSuccess 表示成功
ncclResult_t ncclShmAllocateShareableBuffer(size_t size, bool legacy, ncclShmIpcDesc_t *desc, void **hptr, void **dptr) {
  // 检查参数有效性
  if (desc == NULL || hptr == NULL) {
    // 输出警告日志
    WARN("Invalid argument desc %p, hptr %p", desc, hptr);
    // 返回无效参数错误
    return ncclInvalidArgument;
  }
// 条件编译：CUDA 运行时版本 >= 12.2
// 12.2 版本引入了 cuMem API，支持更高效的跨进程内存共享
#if CUDART_VERSION >= 12020
  // 检查是否启用 cuMem API 且不使用传统模式
  if (ncclCuMemEnable() && ncclCuMemHostEnable() && !legacy) {
    // cuMem API support
    // cuMem API 支持
    // 获取句柄类型
    CUmemAllocationHandleType type = SHM_HANDLE_TYPE;
    // 声明变量：cuMem 分配句柄
    CUmemGenericAllocationHandle handle;

    // 分配可导出的主机内存
    // 这块内存可以被导出并在其他进程中导入
    NCCLCHECK(ncclCuMemHostAlloc(hptr, &handle, size));
    // 检查句柄类型是否为 POSIX 文件描述符
    if (type == CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) {
      // Return the native cuMem handle for later Export/Import via UDS
      // 返回原生 cuMem 句柄，用于后续通过 Unix Domain Socket 导出/导入
      memcpy(&desc->shmci.data, &handle, sizeof(handle));
    } else {
      // 其他类型（如 fabric handle），需要导出为可共享句柄
      CUCHECK(cuMemExportToShareableHandle(&desc->shmci.handle, handle, type, 0));
    }
    // 保存缓冲区大小到描述符
    desc->shmci.size = size;
    // 保存缓冲区指针到描述符
    desc->shmci.ptr = *hptr;
    // 如果提供了设备端指针，将其设置为主机端指针（对于 cuMem，两者相同）
    if (dptr) *dptr = *hptr;
    // 标记为非传统模式
    desc->legacy = false;
    // 输出信息日志
    INFO(NCCL_SHM, "CUMEM allocated shareable buffer %p size %zi", desc->shmci.ptr, desc->shmci.size);
  } else {
    // 使用传统 mmap 模式（通过 /dev/shm）
    // 声明变量：共享内存路径
    char shmPath[SHM_PATH_MAX] = { '\0' };
    // 保存缓冲区大小到描述符
    desc->shmli.shmSize = size;
    // 打开或创建共享内存对象
    // shmPath: 输出参数，返回共享内存路径
    // size: 缓冲区大小
    // hptr: 输出主机端指针
    // dptr: 输出设备端指针
    // 1: 创建标志（1=创建，-1=打开）
    // &desc->shmli.handle: 输出共享内存文件描述符
    NCCLCHECK(ncclShmOpen(shmPath, sizeof(shmPath), size, hptr, dptr, 1, &desc->shmli.handle));
    // 保存共享内存路径后缀（去掉 "/dev/shm/nccl-" 前缀）
    memcpy(desc->shmli.shmSuffix, shmPath + sizeof("/dev/shm/nccl-") - 1, sizeof(desc->shmli.shmSuffix));
    // 标记为传统模式
    desc->legacy = true;
    // 输出信息日志
    INFO(NCCL_SHM, "MMAP allocated shareable host buffer %s size %zi ptr %p", shmPath, desc->shmli.shmSize, *hptr);
  }
// CUDA 运行时版本 < 12.2，只支持传统 mmap 模式
#else /* CUDART_VERSION >= 12020 */
  // 声明变量：共享内存路径
  char shmPath[SHM_PATH_MAX] = { '\0' };
  // 保存缓冲区大小到描述符
  desc->shmli.shmSize = size;
  // 打开或创建共享内存对象
  NCCLCHECK(ncclShmOpen(shmPath, sizeof(shmPath), size, hptr, dptr, 1, &desc->shmli.handle));
  // 保存共享内存路径后缀
  memcpy(desc->shmli.shmSuffix, shmPath + sizeof("/dev/shm/nccl-") - 1, sizeof(desc->shmli.shmSuffix));
  // 标记为传统模式
  desc->legacy = true;
  // 输出信息日志
  INFO(NCCL_SHM, "MMAP allocated shareable host buffer %s size %zi ptr %p", shmPath, size, *hptr);
#endif /* CUDART_VERSION >= 12020 */
  // 返回成功状态码
  return ncclSuccess;
}

// 函数功能：导入可共享的缓冲区
// 这个函数导入由其他进程分配的可共享内存缓冲区
// 参数说明：
//   - comm: 通信上下文指针
//   - proxyRank: 代理 rank（分配缓冲区的进程的 rank）
//   - desc: IPC 描述符（包含如何导入共享内存的信息）
//   - hptr: 输出参数，返回主机端指针
//   - dptr: 输出参数，返回设备端指针（可选，可以为 NULL）
//   - descOut: 输出参数，返回导入后的 IPC 描述符
// 返回值：ncclSuccess 表示成功
ncclResult_t ncclShmImportShareableBuffer(struct ncclComm *comm, int proxyRank, ncclShmIpcDesc_t *desc, void **hptr, void **dptr, ncclShmIpcDesc_t *descOut) {
  // 检查参数有效性
  if (comm == NULL || desc == NULL || hptr == NULL || descOut == NULL) {
    // 输出警告日志
    WARN("Invalid argument comm %p, desc %p, hptr %p, descOut %p", comm, desc, hptr, descOut);
    // 返回无效参数错误
    return ncclInvalidArgument;
  }
// 条件编译：CUDA 运行时版本 >= 12.2
#if CUDART_VERSION >= 12020
  // 检查是否启用 cuMem API 且不使用传统模式
  if (ncclCuMemEnable() && ncclCuMemHostEnable() && !desc->legacy) {
    // cuMem API support
    // cuMem API 支持
    // 声明变量：主机端指针（CUdeviceptr 类型）
    CUdeviceptr hostptr = 0;
    // 获取句柄类型
    CUmemAllocationHandleType type = SHM_HANDLE_TYPE;
    // 声明变量：cuMem 分配句柄
    CUmemGenericAllocationHandle handle;
    // 声明变量：CUDA 设备编号
    int cudaDev;
    // 声明变量：当前 CUDA 设备
    CUdevice currentDev;
    // 声明变量：内存访问描述符
    CUmemAccessDesc accessDesc = {};
    // 声明变量：CPU NUMA 节点 ID
    int cpuNumaNodeId;
    // 声明变量：分配粒度
    size_t granularity;
    // 声明变量：缓冲区大小
    size_t size = desc->shmci.size;
    // 声明变量：内存分配属性
    CUmemAllocationProp prop = {};

    // Import and map the remote memory descriptor to the local GPU
    // 导入并映射远端内存描述符到本地 GPU
    // 检查句柄类型是否为 POSIX 文件描述符
    if (type == CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) {
      // UDS fd support
      // Unix Domain Socket 文件描述符支持
      // 声明变量：文件描述符
      int fd = -1;
      // Send cuMem handle to remote for conversion to an fd
      // 发送 cuMem 句柄到远端，转换为文件描述符
      // 通过代理客户端获取文件描述符
      NCCLCHECK(ncclProxyClientGetFdBlocking(comm, proxyRank, &desc->shmci.data, &fd));
      // 从文件描述符导入句柄
      CUCHECK(cuMemImportFromShareableHandle(&handle, (void *)(uintptr_t)fd, type));
      // 关闭文件描述符（句柄已导入）
      (void) close(fd);
    } else {
      // 从共享句柄直接导入（fabric handle）
      CUCHECK(cuMemImportFromShareableHandle(&handle, &desc->shmci.handle, type));
    }

    // Get cpu numa id
    // 获取 CPU NUMA 节点 ID
    // 获取当前 CUDA 设备
    CUDACHECK(cudaGetDevice(&cudaDev));
    // 获取 CU 设备
    CUCHECK(cuDeviceGet(&currentDev, cudaDev));
    // 获取设备的 NUMA 节点属性
    CUCHECK(cuDeviceGetAttribute(&cpuNumaNodeId, CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID, currentDev));
    // 如果 NUMA 节点 ID 无效，使用默认值 0
    if (cpuNumaNodeId < 0) cpuNumaNodeId = 0;

    // Get granularity
    // 获取分配粒度
    // 设置位置类型为 NUMA 节点
    prop.location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;
    // 设置分配类型为固定内存（pinned memory）
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    // 设置请求的句柄类型
    prop.requestedHandleTypes = type;
    // 设置 NUMA 节点 ID
    prop.location.id = cpuNumaNodeId;
    // 获取最小分配粒度
    CUCHECK(cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));

    // 对齐大小到粒度的倍数
    ALIGN_SIZE(size, granularity);

    // Reserve and map address
    // 保留并映射地址空间
    // 保留地址空间
    CUCHECK(cuMemAddressReserve(&hostptr, size, /* alignment */ 0, /* addr */ 0, /* flags */ 0));
    // 映射内存到地址空间
    CUCHECK(cuMemMap(hostptr, size, /* offset */ 0, handle, /* flags */ 0));

    // Allow access by the local GPU
    // 允许本地 GPU 访问
    // 设置访问位置为设备
    accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    // 设置设备 ID
    accessDesc.location.id = cudaDev;
    // 设置访问权限为读写
    accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    // 设置访问权限
    CUCHECK(cuMemSetAccess(hostptr, size, &accessDesc, 1));

    // Allow access by the local numa
    // 允许本地 NUMA 节点访问
    // 设置访问位置为 NUMA 节点
    accessDesc.location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;
    // 设置 NUMA 节点 ID
    accessDesc.location.id = cpuNumaNodeId;
    // 设置访问权限为读写
    accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    // 设置访问权限
    CUCHECK(cuMemSetAccess(hostptr, size, &accessDesc, 1));

    // 保存指针到输出描述符
    descOut->shmci.ptr = *hptr = (void *)hostptr;
    // 标记为非传统模式
    descOut->legacy = false;
    // 如果提供了设备端指针，将其设置为主机端指针
    if (dptr) *dptr = (void *)hostptr;
    // 输出信息日志
    INFO(NCCL_SHM, "CUMEM imported shareable host buffer from proxyRank %d size %zi ptr %p, granularity %ld", proxyRank, desc->shmci.size, descOut->shmci.ptr, granularity);
  } else {
    // 使用传统 mmap 模式
    // 声明变量：共享内存路径
    char shmPath[SHM_PATH_MAX];
    // 构造共享内存路径
    snprintf(shmPath, sizeof(shmPath), "/dev/shm/nccl-%s", desc->shmli.shmSuffix);
    // 打开共享内存对象
    // -1 表示打开（不是创建）
    NCCLCHECK(ncclShmOpen(shmPath, sizeof(shmPath), desc->shmli.shmSize, hptr, dptr, -1, &descOut->shmli.handle));
    // 标记为传统模式
    descOut->legacy = true;
    // 输出信息日志
    INFO(NCCL_SHM, "MMAP imported shareable host buffer %s size %zi ptr %p", shmPath, desc->shmli.shmSize, *hptr);
  }
// CUDA 运行时版本 < 12.2，只支持传统 mmap 模式
#else /* CUDART_VERSION >= 12020 */
  // 声明变量：共享内存路径
  char shmPath[SHM_PATH_MAX];
  // 构造共享内存路径
  snprintf(shmPath, sizeof(shmPath), "/dev/shm/nccl-%s", desc->shmli.shmSuffix);
  // 打开共享内存对象
  NCCLCHECK(ncclShmOpen(shmPath, sizeof(shmPath), desc->shmli.shmSize, hptr, dptr, -1, &descOut->shmli.handle));
  // 标记为传统模式
  descOut->legacy = true;
  // 输出信息日志
  INFO(NCCL_SHM, "MMAP imported shareable host buffer %s size %zi ptr %p", shmPath, desc->shmli.shmSize, *hptr);
#endif
  // 返回成功状态码
  return ncclSuccess;
}

// 函数功能：关闭共享内存 IPC 描述符
// 释放之前导入或分配的共享内存资源
// 参数说明：
//   - desc: 共享内存 IPC 描述符指针
// 返回值：ncclSuccess 表示成功
ncclResult_t ncclShmIpcClose(ncclShmIpcDesc_t *desc) {
  // 检查描述符指针是否有效
  if (desc) {
// 条件编译：CUDA 运行时版本 >= 12.2
#if CUDART_VERSION >= 12020
    // 检查是否使用 cuMem API 且非传统模式
    if (ncclCuMemEnable() && ncclCuMemHostEnable() && !desc->legacy) {
      // 使用 cuMem API 释放内存
      NCCLCHECK(ncclCuMemHostFree(desc->shmci.ptr));
    } else {
      // 使用传统方式关闭共享内存
      NCCLCHECK(ncclShmClose(desc->shmli.handle));
    }
// CUDA 运行时版本 < 12.2，只支持传统方式
#else
    // 使用传统方式关闭共享内存
    NCCLCHECK(ncclShmClose(desc->shmli.handle));
#endif
  }

  // 返回成功状态码
  return ncclSuccess;
}

// 定义共享内存传输层结构体
// 这个结构体定义了 SHM 传输层的所有函数指针和配置
struct ncclTransport shmTransport = {
  // 传输层名称："SHM"
  "SHM",
  // 检查是否可以连接的函数指针
  shmCanConnect,
  // 发送端操作函数集合
  { // 发送端设置函数：分配发送端资源
    shmSendSetup,
    // 发送端连接函数：连接到对等节点
    shmSendConnect,
    // 发送端释放函数：释放发送端资源
    shmSendFree,
    // 发送端数据接收函数：未使用（NULL）
    NULL,
    // 发送端代理设置函数：代理线程中分配共享内存
    shmSendProxySetup,
    // 发送端代理连接函数：代理线程中建立连接
    NULL,
    // 发送端代理释放函数：代理线程中释放资源
    shmSendProxyFree,
    // 发送端代理进度函数：代理线程中处理传输进度
    NULL },
  // 接收端操作函数集合
  { // 接收端设置函数：分配接收端资源
    shmRecvSetup,
    // 接收端连接函数：连接到对等节点
    shmRecvConnect,
    // 接收端释放函数：释放接收端资源
    shmRecvFree,
    // 接收端数据发送函数：未使用（NULL）
    NULL,
    // 接收端代理设置函数：代理线程中分配共享内存
    shmRecvProxySetup,
    // 接收端代理连接函数：代理线程中建立连接
    NULL,
    // 接收端代理释放函数：代理线程中释放资源
    shmRecvProxyFree,
    // 接收端代理进度函数：代理线程中处理传输进度
    NULL }
};
