/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 * 版权所有 (c) 2016-2022, NVIDIA CORPORATION. 保留所有权利。
 *
 * See LICENSE.txt for license information
 * 请参阅 LICENSE.txt 获取许可证信息
 ************************************************************************/

// 引入通信器头文件，包含 ncclComm 结构体和相关操作的声明
#include "comm.h"
// 引入图拓扑头文件，包含通信拓扑相关结构
#include "graph.h"
// 引入工具函数头文件，包含各种辅助工具函数和宏定义
#include "utils.h"
// 引入共享内存工具头文件，包含共享内存操作相关函数
#include "shmutils.h"
// 引入 P2P 头文件，包含点对点通信相关定义
#include "p2p.h"
// 引入传输层头文件，包含传输方式的接口定义
#include "transport.h"
// 引入 C 标准断言头文件，提供断言宏
#include <assert.h>
// 引入共享内存头文件，包含共享内存操作函数
#include "shm.h"
// 引入内联注册头文件，包含内存注册相关内联函数
#include "register_inline.h"

// P2P 传输类型枚举
// 定义四种 P2P 传输模式，根据 GPU 位置和进程关系选择最优方式
enum p2pType {
    P2P_DIRECT, // 同节点，同进程/多线程GPU，直接指针访问。最快方式，无需 IPC
    P2P_INTERMEDIATE,// 间接传输（通过中间GPU）用于无直接 P2P 路径的情况
    P2P_IPC, // 同节点，同进程/不同进程，Legacy CUDA IPC 传统 IPC 方式
    P2P_CUMEM    // 同节点，同进程/不同进程，cuMem API (CUDA 11.3+) 新版 IPC
};

// P2P 缓冲区结构体
// 用于存储 P2P 传输所需的缓冲区信息和 IPC 描述符
struct ncclP2pBuff {
  void* directPtr;                                       // 直接指针（用于 P2P_DIRECT 模式）
  size_t size;                                           // 缓冲区大小
  ncclIpcDesc ipcDesc;                                   // IPC 描述符（用于跨进程共享）
};

// P2P 请求结构体
// 用于请求分配 P2P 缓冲区时的参数
struct ncclP2pRequest {
  size_t size;                                           // 请求的缓冲区大小
  int refcount;                                          // 引用计数（用于多 GPU 共享缓冲区）
};

// P2P 连接信息结构体
// 用于在连接建立阶段交换连接参数
struct p2pConnectInfo {
  int rank;                                              // 目标 peer 的 rank 编号
  int read;                                              // 是否使用 P2P Read 模式（否则为 Write 模式）
  struct ncclP2pBuff p2pBuff;                           // P2P 缓冲区信息
  // Used by CE memcpy
  // 用于 CUDA Copy Engine（CE）memcpy 模式
  ncclShmIpcDesc_t desc;                                // 共享内存 IPC 描述符
};
// 编译时断言：确保 p2pConnectInfo 结构体大小不超过连接信息缓冲区大小
static_assert(sizeof(struct p2pConnectInfo) <= CONNECT_SIZE, "p2pConnectInfo is too large");

// P2P IPC 导出信息结构体
// 用于跨进程导出 IPC 内存描述符
struct p2pIpcExpInfo {
  ncclIpcDesc ipcDesc;                                   // IPC 描述符
  bool legacyIpcCap;                                     // 是否支持传统 IPC
  int impFd;                                             // 导入文件描述符（用于 POSIX fd 类型）
  size_t size;                                           // 内存区域大小
  uintptr_t offset;                                      // 偏移量（相对于基地址）
};

// P2P 共享内存结构体
// 定义了发送和接收内存的结构
struct p2pShm {
  struct ncclSendMem sendMem;                            // 发送内存结构
  struct ncclRecvMem recvMem;                            // 接收内存结构
};

// P2P 共享内存代理信息结构体
// 用于 CE memcpy 模式下的代理进程信息
struct p2pShmProxyInfo {
  // Shared memory between proxy and receiving GPU
  // 代理进程和接收 GPU 之间的共享内存
  struct p2pShm* shm;                                    // 主机侧共享内存指针
  struct p2pShm* devShm;                                 // 设备侧共享内存指针
  ncclShmIpcDesc_t desc;                                // 共享内存 IPC 描述符

  // Intermediate step for sender
  // 发送方的中间缓冲步骤
  struct ncclRecvMem* ceRecvMem;                         // CE 接收内存指针
  char* ceDevBuff;                                       // CE 设备缓冲区指针

  // Receiver buffer
  // 接收方缓冲区
  char* recvFifo;                                        // 接收 FIFO 队列

  // Used by CE memcpy progress only
  // 仅用于 CE memcpy 进度跟踪
  uint64_t step;                                         // 当前步骤编号
  cudaStream_t stream;                                   // CUDA 流（用于异步操作）
  cudaEvent_t events[NCCL_STEPS];                        // CUDA 事件数组（每个步骤一个事件）
};
// 编译时断言：确保 P2P 连接信息大小不超过限制
static_assert(sizeof(p2pConnectInfo) <= CONNECT_SIZE, "P2P Connect info is too large");

// P2P 资源结构体
// 存储 P2P 连接的所有资源和状态信息
struct p2pResources {
  enum p2pType type;                                     // P2P 传输类型
  union {                                                // 联合体：发送或接收内存的二选一
    struct ncclSendMem* sendDevMem;                      // 发送设备内存指针
    struct ncclRecvMem* recvDevMem;                      // 接收设备内存指针
  };
  void* sendMemIpc;                                      // 发送内存 IPC 指针
  int sendMemSameProc;                                   // 发送内存是否在同一进程
  void* recvMemIpc;                                      // 接收内存 IPC 指针
  int recvMemSameProc;                                   // 接收内存是否在同一进程
  // CE memcpy support
  // CE memcpy 支持
  struct p2pShmProxyInfo proxyInfo;                     // 代理信息（CE 模式使用）
  struct p2pShm* shm;                                    // 共享内存指针
  struct p2pShm* devShm;                                 // 设备共享内存指针
  ncclShmIpcDesc_t desc;                                // 共享内存 IPC 描述符
};

// cuMem API support
// cuMem API 支持
struct p2pCuMemProxyInfo {
  struct ncclP2pBuff p2pBuff;                           // P2P 缓冲区信息
};

// 引入系统类型头文件，提供系统级数据类型定义
#include <sys/types.h>

// 定义 NCCL 环境变量参数：是否使用传统 CUDA 注册方式
// LEGACY_CUDA_REGISTER=1 时使用传统注册方式
NCCL_PARAM(LegacyCudaRegister, "LEGACY_CUDA_REGISTER", 0);

/* Convert a PCI busId string into a local cudaDev device index (cf. CUDA_VISIBLE_DEVICES) */
/* 将 PCI 总线 ID 字符串转换为本地 CUDA 设备索引（考虑 CUDA_VISIBLE_DEVICES 环境变量） */
static int busIdToCudaDev(int64_t busId) {
  int ndev;                                              // 声明设备数量变量
  // 获取可见 CUDA 设备数量
  if (cudaGetDeviceCount(&ndev) != cudaSuccess)
    return -1;                                           // 获取失败，返回 -1
  // 遍历所有可见 CUDA 设备
  for (int i = 0; i < ndev; i++) {
    char devBusIdStr[NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE]; // 声明 PCI 总线 ID 字符串缓冲区
    // 获取设备的 PCI 总线 ID 字符串
    if (cudaDeviceGetPCIBusId(devBusIdStr, NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE, i) != cudaSuccess)
      return -1;                                         // 获取失败，返回 -1
    int64_t devBusId;                                    // 声明设备总线 ID 变量
    // 将 PCI 总线 ID 字符串转换为整数
    NCCLCHECK(busIdToInt64(devBusIdStr, &devBusId));
    // 比较总线 ID，如果匹配则返回设备索引
    if (busId == devBusId) return i;
  }
  // BusId was not found in our locally visible CUDA devices
  // 在本地可见的 CUDA 设备中未找到该总线 ID
  return -1;                                             // 未找到匹配的设备，返回 -1
}

// CE memcpy support
// 用于强制使用 CUDA Copy Engine 的 cudaMemcpy 操作替代默认的 GPU Kernel 直接内存访问
// 为了兼容性考虑，某些 GPU 或驱动版本可能不支持直接的 P2P Kernel 访问，此时可回退到更稳定的 cudaMemcpy 方式
// 环境变量：NCCL_P2P_USE_CUDA_MEMCPY=1 启用此模式
NCCL_PARAM(P2pUseCudaMemcpy, "P2P_USE_CUDA_MEMCPY", 0);
static int useMemcpy = 0;                                   // 全局变量：是否使用 CE memcpy 模式
static void initCeOperation();                              // CE 操作初始化函数声明


// 声明外部函数：获取 MNNVL（多节点 NVLink）启用状态
extern int64_t ncclParamMNNVLEnable();

/* Determine if two peers can communicate through p2p */
/* 确定两个 peer 是否可以通过 P2P 进行通信 */
/* 参数 ret: 输出参数，1 表示可以使用 P2P，0 表示不能 */
/* 参数 comm: NCCL 通信器指针 */
/* 参数 graph: 拓扑图指针 */
/* 参数 info1: 第一个 peer 的信息 */
/* 参数 info2: 第二个 peer 的信息 */
ncclResult_t p2pCanConnect(int* ret, struct ncclComm* comm, struct ncclTopoGraph* graph, struct ncclPeerInfo* info1, struct ncclPeerInfo* info2) {
  initCeOperation();                                        // 初始化 CE 操作模式

  // Check topology / p2p level.
  // 检查拓扑 / P2P 级别
  int intermediateRank;                                    // 声明中间 peer rank 变量（用于间接 P2P）
  // 检查两个 peer 之间是否可以建立 P2P 连接
  // 返回值 ret: 1 表示可以 P2P，0 表示不能
  // 返回值 intermediateRank: 如果需要中间 peer，返回其 rank，否则为 -1
  NCCLCHECK(ncclTopoCheckP2p(comm, comm->topo, info1->rank, info2->rank, ret, NULL, &intermediateRank));
  if (*ret == 0)                                            // 如果返回值为 0，表示不能 P2P
    return ncclSuccess;                                    // 直接返回成功

   //ret=1，表示支持p2p
  // 检查是否需要通过中间 peer 进行间接 P2P
  if (intermediateRank != -1) {                            // 如果存在中间 peer
    //启用了cudaMemcpy，直接返回0
    if (useMemcpy)                                         // 如果启用了 CE memcpy 模式
        *ret = 0;                                          // 不支持间接 P2P，返回 0
    return ncclSuccess;                                    // 返回成功
  }

  // Check if NET would work better
  // 检查使用网络传输是否会更好
  int useNet = 0;                                          // 声明网络传输标志
  // 检查是否应该使用网络传输而不是 P2P
  NCCLCHECK(ncclTopoCheckNet(comm->topo, info1->rank, info2->rank, &useNet));
  if (useNet) {                                            // 如果应该使用网络
    //使用net，返回0
    *ret = 0;                                              // 设置返回值为 0，表示不使用 P2P
    return ncclSuccess;                                    // 返回成功
  }

  //不同的节点，不支持p2p
  // 检查 peer 是否在同一节点上
  // 比较 hostHash：如果不相同，表示在不同节点
  if (info1->hostHash != comm->peerInfo[comm->rank].hostHash ||
      info1->hostHash != info2->hostHash) {
    // If either peer is non-local then we are done.
    // 如果任一 peer 不是本地的，则无法使用 P2P
    return ncclSuccess;                                    // 返回成功（ret 保持为 0）
  }

  // Convert the peer's busId into a local cudaDev index (cf. CUDA_VISIBLE_DEVICES)
  // 将 peer 的总线 ID 转换为本地 CUDA 设备索引（考虑 CUDA_VISIBLE_DEVICES）
  int cudaDev1 = busIdToCudaDev(info1->busId);            // 获取第一个 peer 的 CUDA 设备索引
  int cudaDev2 = busIdToCudaDev(info2->busId);            // 获取第二个 peer 的 CUDA 设备索引
  if (cudaDev1 == -1 || cudaDev2 == -1) {                 // 如果任一设备未找到
#if CUDART_VERSION >= 10010                                // 如果 CUDA 版本 >= 10.1
    // CUDA 10.1 and later can use P2P with invisible devices.
    // CUDA 10.1 及更高版本可以对不可见设备使用 P2P
    return ncclSuccess;                                    // 返回成功（允许继续尝试）
#else
    // Peer's CUDA device is not visible in this process : we can't communicate with it.
    // Peer 的 CUDA 设备在此进程中不可见：我们无法与其通信
    *ret = 0;                                              // 设置返回值为 0
    return ncclSuccess;                                    // 返回成功
#endif
  }

  // Check that CUDA can do P2P
  // 检查 CUDA 是否支持 P2P
  int p2p;                                                 // 声明 P2P 能力变量
  // 查询两个设备是否可以互相访问（P2P）
  if (cudaDeviceCanAccessPeer(&p2p, cudaDev1, cudaDev2) != cudaSuccess) {
    // 查询失败，记录日志
    INFO(NCCL_INIT|NCCL_P2P,"peer query failed between dev %d(=%lx) and dev %d(=%lx)",
         cudaDev1, info1->busId, cudaDev2, info2->busId);
    *ret = 0;                                              // 设置返回值为 0
    return ncclSuccess;                                    // 返回成功
  }

  // This will always fail when using NCCL_CUMEM_ENABLE=1
  // 当使用 NCCL_CUMEM_ENABLE=1 时，下面的检查总是会失败
  if (p2p != 0 && !ncclCuMemEnable()) {                    // 如果支持 P2P 且未启用 cuMem 模式
    // Cached result of the legacyIPC detection
    // 缓存的 legacy IPC 检测结果（避免重复检测）
    static int legacyIPC = -1;                             // 静态变量：缓存 legacy IPC 支持状态
    if (legacyIPC >= 0) {                                  // 如果已经检测过
      *ret = legacyIPC;                                    // 使用缓存的结果
      return ncclSuccess;                                  // 返回成功
    }

    // Check that legacy IPC support is available (WSL WAR)
    //检查是否支持IPC
    // 检查是否支持传统 IPC（WSL 环境的变通方案）
    char *dummy;                                           // 声明虚拟指针
    cudaIpcMemHandle_t ipc;                                // 声明 IPC 内存句柄
    NCCLCHECK(ncclCudaMalloc(&dummy, CUDA_IPC_MIN));       // 分配最小 IPC 内存用于测试
    if (cudaIpcGetMemHandle(&ipc, dummy) != cudaSuccess) { // 如果获取失败
      INFO(NCCL_INIT|NCCL_P2P,"Legacy IPC not supported"); // 记录日志：不支持传统 IPC
      *ret = 0;                                            // 设置返回值为 0
    }
    
    NCCLCHECK(ncclCudaFree(dummy));                        // 释放测试内存
    legacyIPC = *ret;                                      // 缓存检测结果
    return ncclSuccess;                                    // 返回成功
  }

  // 检查 P2P 是否为 0（不支持 P2P）
  if (p2p == 0) {                                          // 如果不支持 P2P
    INFO(NCCL_INIT|NCCL_P2P,"Could not enable P2P between dev %d(=%lx) and dev %d(=%lx)",
         cudaDev1, info1->busId, cudaDev2, info2->busId);   // 记录日志：无法启用 P2P
    *ret = 0;                                              // 设置返回值为 0
    return ncclSuccess;                                    // 返回成功
  }
  
  return ncclSuccess;                                      // 返回成功（ret 保持为 1）
}

// 定义跟踪转储 IPC 句柄的宏
// 用于调试时打印 IPC 句柄的内容（8 个 unsigned long 值）
#define TRACE_DUMP_IPC(DEVIPC)                                                             \
  do {                                                                                     \
    unsigned long *devIpc = (unsigned long *) (DEVIPC);                                    \
    TRACE(P2P,"IPC: %016lx %016lx %016lx %016lx", devIpc[0], devIpc[1], devIpc[2], devIpc[3]); \
    TRACE(P2P,"IPC: %016lx %016lx %016lx %016lx", devIpc[4], devIpc[5], devIpc[6], devIpc[7]); \
  } while (0)

// cuMem API support
// 函数实现：分配可共享的 P2P 缓冲区
// 此函数分配一个可以在进程间共享的 GPU 内存缓冲区
// 参数 size: 缓冲区大小
// 参数 refcount: 引用计数（用于多 GPU 共享）
// 参数 ipcDesc: 输出参数，接收 IPC 描述符
// 参数 ptr: 输出参数，接收分配的内存指针
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
ncclResult_t ncclP2pAllocateShareableBuffer(size_t size, int refcount, ncclIpcDesc *ipcDesc, void **ptr) {
  if (ncclCuMemEnable()) {                                  // 如果启用了 cuMem API
#if CUDART_VERSION >= 11030                                // 检查 CUDA 版本是否 >= 11.3
    CUmemAllocationHandleType type = ncclCuMemHandleType;  // 获取 cuMem 句柄类型

    // cuMem API support
    // cuMem API 支持
    CUmemGenericAllocationHandle handle;                   // 声明 cuMem 分配句柄
    NCCLCHECK(ncclCuMemAlloc(ptr, &handle, type, size));   // 分配 cuMem 内存
    if (type == CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) { // 如果是 POSIX 文件描述符类型
      // Return the native cuMem handle for later Export/Import via UDS
      // 返回原生 cuMem 句柄，用于后续通过 UDS 导出/导入
      memcpy(&ipcDesc->cuDesc.data, &handle, sizeof(handle)); // 复制句柄到数据字段
    } else {                                                // 其他类型（如 fabric handle）
      CUCHECK(cuMemExportToShareableHandle(&ipcDesc->cuDesc, handle, type, 0)); // 导出为可共享句柄
    }
    if (refcount) {                                        // 如果需要引用计数（多 GPU 共享）
      memcpy(&ipcDesc->memHandle, &handle, sizeof(handle)); // 复制句柄到 memHandle 字段
      for (int r = 0; r < refcount; ++r) {                 // 对每个引用计数
        CUCHECK(cuMemRetainAllocationHandle(&handle, *ptr)); // 保留分配句柄以增加引用计数
      }
    }
#else
    return ncclInternalError;                              // CUDA 版本不支持，返回内部错误
#endif
  } else {                                                  // 未启用 cuMem API，使用传统 CUDA IPC
    // Allocate a CUDA buffer and generate an IPC handle for it
    // 分配 CUDA 缓冲区并为其生成 IPC 句柄
    NCCLCHECK(ncclCudaCalloc((char **)ptr, size));         // 在 GPU 上分配并清零内存
    cudaError_t res = cudaIpcGetMemHandle(&ipcDesc->devIpc, *ptr); // 获取内存的 IPC 句柄
    if (res != cudaSuccess) {                              // 如果获取失败
      WARN("cudaIpcGetMemHandle failed : %s", cudaGetErrorString(res)); // 输出警告信息
      ncclCudaFree(*ptr);                                  // 释放已分配的内存
      CUDACHECK(res);                                      // 检查并抛出错误
    }
  }
  INFO(NCCL_P2P|NCCL_ALLOC, "Allocated shareable buffer %p size %zu ipcDesc %p", *ptr, size, ipcDesc); // 记录分配信息

  return ncclSuccess;                                       // 返回成功
}

// 函数实现：释放可共享的 P2P 缓冲区
// 此函数释放之前分配的可共享缓冲区
// 参数 ipcDesc: IPC 描述符
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
ncclResult_t ncclP2pFreeShareableBuffer(ncclIpcDesc *ipcDesc) {
  return ncclSuccess;                                       // 当前实现为空，直接返回成功
}

// 函数实现：导入远端的可共享缓冲区
// 此函数从远端进程导入一个可共享的 GPU 内存缓冲区
// 参数 comm: NCCL 通信上下文
// 参数 peer: 远端对等进程的 rank
// 参数 size: 缓冲区大小
// 参数 ipcDesc: IPC 描述符
// 参数 devMemPtr: 输出参数，接收导入的设备内存指针
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
ncclResult_t ncclP2pImportShareableBuffer(struct ncclComm *comm, int peer, size_t size, ncclIpcDesc *ipcDesc, void **devMemPtr) {

  if (ncclCuMemEnable()) {                                  // 如果启用了 cuMem API
#if CUDART_VERSION >= 11030                                // 检查 CUDA 版本是否 >= 11.3
    // cuMem API support
    // cuMem API 支持
    CUdeviceptr dptr = 0;                                   // 设备指针初始化为 0
    CUmemAllocationHandleType type = ncclCuMemHandleType;  // 获取 cuMem 句柄类型
    CUmemGenericAllocationHandle handle;                   // 声明分配句柄
    ncclCuDesc *cuDesc = &ipcDesc->cuDesc;                 // 获取 cuMem 描述符
    CUmemAllocationProp prop = {};                         // 声明分配属性并清零
    size_t granularity = 0;                                 // 声明内存粒度

    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;             // 设置为固定内存类型
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;       // 位置类型为设备
    prop.requestedHandleTypes = type;                      // 请求的句柄类型
    prop.location.id = comm->cudaDev;                      // 设置目标 CUDA 设备
    CUCHECK(cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM)); // 获取分配粒度
    ALIGN_SIZE(size, granularity);                         // 将大小对齐到粒度边界

    // Import and map the remote memory descriptor to the local GPU
    // 导入远端内存描述符并映射到本地 GPU
    if (type == CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) { // 如果是 POSIX 文件描述符类型
      // UDS fd support
      // UDS（Unix Domain Socket）文件描述符支持
      int fd = -1;                                         // 文件描述符初始化为 -1
      // Send cuMem handle to remote for conversion to an fd
      // 向远端发送 cuMem 句柄以转换为文件描述符
      //获取fd
      NCCLCHECK(ncclProxyClientGetFdBlocking(comm, peer, &cuDesc->data, &fd)); // 通过代理获取文件描述符
      INFO(NCCL_P2P, "UDS converted handle 0x%lx to fd %d on remote peer %d", *(uint64_t*)&cuDesc->data, fd, peer); // 记录转换信息
      //通过fd转化为远端的handle
      CUCHECK(cuMemImportFromShareableHandle(&handle, (void *)(uintptr_t)fd, type)); // 从文件描述符导入句柄
      SYSCHECK(close(fd), "close");                        // 关闭文件描述符
    } else {                                                // 其他类型（如 fabric handle）
      CUCHECK(cuMemImportFromShareableHandle(&handle, cuDesc, type)); // 从共享句柄导入
    }
    CUCHECK(cuMemAddressReserve(&dptr, size, /* alignment */ 0, /* addr */ 0, /* flags */ 0)); // 保留地址空间
    //dptr通过handle映射到远端的物理内存
    CUCHECK(cuMemMap(dptr, size, /* offset */ 0, handle, /* flags */ 0)); // 映射内存到地址空间

    TRACE(NCCL_P2P, "Imported shareable buffer size %zu handle 0x%llx dptr %p", size, handle, (void*)dptr); // 跟踪导入信息

    // Allow access by the local GPU
    // 允许本地 GPU 访问
    CUmemAccessDesc accessDesc = {};                       // 声明访问描述符并清零
    accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE; // 位置类型为设备
    accessDesc.location.id = comm->cudaDev;                // 设置目标 CUDA 设备
    accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE; // 设置读写权限
    CUCHECK(cuMemSetAccess(dptr, size, &accessDesc, 1));   // 设置内存访问权限
    TRACE(NCCL_P2P, "Set Access for %p size %zu on dev %d", (void*)dptr, size, accessDesc.location.id); // 跟踪访问权限设置

    *devMemPtr = (void *)dptr;                             // 返回设备内存指针
#else
    return ncclInternalError;                              // CUDA 版本不支持，返回内部错误
#endif
  } else {                                                  // 未启用 cuMem API，使用传统 CUDA IPC
    // Legacy CUDA IPC
    // 传统 CUDA IPC 方式
    CUDACHECK(cudaIpcOpenMemHandle(devMemPtr, ipcDesc->devIpc, cudaIpcMemLazyEnablePeerAccess)); // 打开 IPC 内存句柄
  }

  INFO(NCCL_P2P, "Imported shareable buffer device %d size %zu ptr %p", comm->cudaDev, size, *devMemPtr); // 记录导入信息

  return ncclSuccess;                                       // 返回成功
}

// Setting this to non zero causes P2P to use Reads rather than Writes
// 将此值设置为非零会导致 P2P 使用读操作而不是写操作
// 定义环境变量 P2P_READ_ENABLE，默认值为 -2（自动选择）
NCCL_PARAM(P2pReadEnable, "P2P_READ_ENABLE", -2);
// 定义环境变量 P2P_DIRECT_DISABLE，默认值为 0（不禁用直接 P2P）
NCCL_PARAM(P2pDirectDisable, "P2P_DIRECT_DISABLE", 0);

// 宏定义：判断两个对等节点是否在同一进程中
// 通过比较 hostHash 和 pidHash 来判断
#define P2P_SAME_PID(MYINFO, PEERINFO) ((MYINFO->hostHash == PEERINFO->hostHash) && (MYINFO->pidHash == PEERINFO->pidHash))

// 函数实现：获取 P2P 连接信息
// 此函数查询拓扑信息以确定两个 GPU 之间的 P2P 连接类型
// 参数 comm: NCCL 通信上下文
// 参数 info1: 第一个对等节点的信息
// 参数 info2: 第二个对等节点的信息
// 参数 read: 输出参数，指示是否使用 P2P 读操作
// 参数 intermediateRank: 输出参数，中间节点的 rank（如果有）
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
static ncclResult_t p2pGetInfo(struct ncclComm* comm, struct ncclPeerInfo* info1, struct ncclPeerInfo* info2, int* read, int* intermediateRank) {
  int p2p;                                                  // P2P 连接状态
  // Queries the topology to see if the GPUs are Ampere and
  // connected via NVLink, if so we enable P2P Read by default
  // 查询拓扑以检查 GPU 是否为 Ampere 架构且通过 NVLink 连接
  // 如果是，则默认启用 P2P 读操作
  NCCLCHECK(ncclTopoCheckP2p(comm, comm->topo, info1->rank, info2->rank, &p2p, read, intermediateRank)); // 检查 P2P 连接

  int readEnable = ncclParamP2pReadEnable();               // 获取环境变量 P2P_READ_ENABLE 的值
  if (readEnable != -2) 
    *read = readEnable;                // 如果用户指定了值，则使用用户指定的值
  return ncclSuccess;                                       // 返回成功
}

// 函数实现：映射 P2P 缓冲区
// 此函数将远端的 P2P 缓冲区映射到本地地址空间
// 参数 comm: NCCL 通信上下文
// 参数 proxyConn: 代理连接器
// 参数 myInfo: 本节点的信息
// 参数 peerInfo: 对等节点的信息
// 参数 p2pBuff: P2P 缓冲区描述符
// 参数 devMem: 输出参数，接收映射的设备内存指针
// 参数 ipcPtr: 输出参数，接收 IPC 指针
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
static ncclResult_t p2pMap(struct ncclComm *comm, struct ncclProxyConnector* proxyConn, struct ncclPeerInfo* myInfo, struct ncclPeerInfo* peerInfo, struct ncclP2pBuff* p2pBuff, void** devMem, void** ipcPtr) {
  if (P2P_SAME_PID(myInfo, peerInfo)) {                    // 如果在同一进程中
    if (peerInfo->cudaDev != myInfo->cudaDev) {             // 如果是不同的 GPU
      // Same PID different GPUs, enable P2P access
      // 同一进程的不同 GPU，启用 P2P 访问
      // Legacy CUDA IPC
      // 传统 CUDA IPC 方式
      cudaError_t err = cudaDeviceEnablePeerAccess(peerInfo->cudaDev, 0); // 启用对等访问
      if (err == cudaErrorPeerAccessAlreadyEnabled) {       // 如果已经启用
        cudaGetLastError();                                 // 清除错误状态
      } else if (err != cudaSuccess) {                      // 如果启用失败
        WARN("failed to peer with device %d(=%lx): %d %s", // 输出警告信息
            peerInfo->cudaDev, peerInfo->busId, err, cudaGetErrorString(err));
        return ncclInternalError;                          // 返回内部错误
      }

      //优先使用cumem
      if (ncclCuMemEnable()) {                              // 如果启用了 cuMem API
        // for intra-process ranks, we should map memHandle of the peers to increase refcount.
        // Otherwise, if peers abort and free the buffer, the rank can suffer invalid access.
        // 对于进程内的 rank，我们应该映射对等节点的 memHandle 以增加引用计数
        // 否则，如果对等节点中止并释放缓冲区，该 rank 可能会遇到无效访问
        NCCLCHECK(ncclCuMemAllocAddr(devMem, &p2pBuff->ipcDesc.memHandle, p2pBuff->size)); // 分配地址并映射句柄
        CUCHECK(cuMemRelease(p2pBuff->ipcDesc.memHandle));  // 释放句柄（引用计数已增加）
        *ipcPtr = *devMem;                                  // 设置 IPC 指针
      } else {                                              // 未启用 cuMem API
        *devMem = p2pBuff->directPtr;                       // 直接使用指针
        *ipcPtr = NULL;                                     // IPC 指针为空
      }
      
    } else {                                                // 同一 GPU
      *devMem = p2pBuff->directPtr;                         // 直接使用指针
      *ipcPtr = NULL;                                       // IPC 指针为空
    }
  } else {                                                  // 不同进程
    // Different PID
    // 不同进程，需要导入可共享缓冲区
    NCCLCHECK(ncclP2pImportShareableBuffer(comm, peerInfo->rank, p2pBuff->size, &p2pBuff->ipcDesc, devMem)); // 导入可共享缓冲区
    *ipcPtr = *devMem;                                      // 设置 IPC 指针
  }
  
  return ncclSuccess;                                       // 返回成功
}

/* Send: Create and return connect structures for this peer to connect to me */
/* 发送端设置：创建并返回连接结构，供对等节点连接到本节点 */
ncclResult_t p2pSendSetup(struct ncclComm* comm, struct ncclTopoGraph* graph, struct ncclPeerInfo* myInfo, struct ncclPeerInfo* peerInfo,
    struct ncclConnect* connectInfo, struct ncclConnector* send, int channelId, int connIndex) {
  struct p2pResources* resources;                          // P2P 资源结构
  struct ncclP2pRequest req;                               // P2P 请求结构
  
  NCCLCHECK(ncclCalloc(&resources, 1));                    // 分配并清零资源结构
  send->transportResources = resources;                    // 保存资源到连接器
  
  int useRead, intermediateRank;                           // 是否使用读操作，中间节点 rank
  NCCLCHECK(p2pGetInfo(comm, myInfo, peerInfo, &useRead, &intermediateRank)); // 获取 P2P 信息
  if (useMemcpy)                                           // 如果使用 cudaMemcpy
    useRead = 0;                                           // 不使用读操作

  static_assert(sizeof(struct p2pConnectInfo) <= sizeof(struct ncclConnect), "p2p Connect Info is too big"); // 编译时检查大小

  struct p2pConnectInfo* info = (struct p2pConnectInfo*)connectInfo; // 类型转换
  info->read = useRead;                                    // 设置是否使用读操作

  // For CollNet, use write for scatter-reduce (conn 1), read for broadcast-gather (conn 0)
  // 对于 CollNet，scatter-reduce 使用写操作（连接 1），broadcast-gather 使用读操作（连接 0）
  if (graph && connIndex == 1)                             // 如果是 CollNet 且是连接 1
    info->read = 0;                                        // 使用写操作
  const char* useReadStr = info->read ? "/read" : "";      // 读取模式字符串

  int sendSize = sizeof(struct ncclSendMem);              // 发送内存大小
  // For P2P Read the SIMPLE buffer is tagged on the end of the ncclSendMem structure
  // 对于 P2P 读，SIMPLE 缓冲区附加在 ncclSendMem 结构的末尾
  if (info->read)                                          // 如果使用读操作
    sendSize += comm->buffSizes[NCCL_PROTO_SIMPLE];       // 增加 SIMPLE 协议缓冲区大小

  ALIGN_SIZE(sendSize, CUDA_IPC_MIN);                     // 对齐到 CUDA IPC 最小大小

  if (intermediateRank == -1) {                            // 如果没有中间节点（直接连接）
    //设置P2P类型
    info->rank = myInfo->rank;                             // 设置本节点的 rank
    //同进程内/多线程的不同 GPU，才能使用P2P_DIRECT模式
    if (P2P_SAME_PID(myInfo, peerInfo) && ncclParamP2pDirectDisable() == 0 && useMemcpy == 0) { // 同进程且不禁用直接 P2P 且不使用 memcpy
      resources->type = P2P_DIRECT;                        // 设置为直接 P2P 类型
      INFO(NCCL_INIT|NCCL_P2P, "Channel %02d/%01d : %d[%d] -> %d[%d] via P2P/direct pointer%s",
          channelId, connIndex, myInfo->rank, myInfo->nvmlDev, peerInfo->rank, peerInfo->nvmlDev, useReadStr); // 记录日志
    } else {                                                // 同进程和不同进程都可以使用cumem或者ipc类型
      // cuMem API support
      // cuMem API 支持
      if (ncclCuMemEnable()) {                              // 如果启用了 cuMem API
        resources->type = P2P_CUMEM;                       // 设置为 CUMEM 类型
        const char *MNNVL = comm->MNNVL ? "MNNVL" : "CUMEM"; // 确定使用的类型名称
        INFO(NCCL_INIT|NCCL_P2P,"Channel %02d/%01d : %d[%d] -> %d[%d] via P2P/%s%s%s",
             channelId, connIndex, myInfo->rank, myInfo->nvmlDev, peerInfo->rank, peerInfo->nvmlDev, MNNVL, useReadStr, useMemcpy ? "/CE" : "");; // 记录日志
      } else {                                              // 未启用 cuMem API
        // Legacy CUDA IPC
        // 传统 CUDA IPC 方式
        resources->type = P2P_IPC;                         // 设置为 IPC 类型
        INFO(NCCL_INIT|NCCL_P2P,"Channel %02d/%01d : %d[%d] -> %d[%d] via P2P/IPC%s%s",
             channelId, connIndex, myInfo->rank, myInfo->nvmlDev, peerInfo->rank, peerInfo->nvmlDev, useReadStr, useMemcpy ? "/CE" : ""); // 记录日志
      }
    }
    send->conn.flags |= info->read ? NCCL_P2P_READ : NCCL_P2P_WRITE; // 设置连接标志
  } else {                                                  // 有中间节点（间接连接）
    resources->type = P2P_INTERMEDIATE;                    // 设置为中间节点类型
    info->rank = intermediateRank;                         // 设置中间节点的 rank
    INFO(NCCL_INIT|NCCL_P2P, "Channel %02d/%01d : %d[%d] -> %d[%d] via P2P/indirect/%d[%d]%s",
        channelId, connIndex, myInfo->rank, myInfo->nvmlDev, peerInfo->rank, peerInfo->nvmlDev, intermediateRank,
	  comm->peerInfo[intermediateRank].nvmlDev, useReadStr); // 记录日志
  }

  memset(&req, '\0', sizeof(req));                         // 清零请求结构
  req.size = sendSize;                                     // 设置请求大小
  req.refcount = 0;                                        // 初始化引用计数
  if (P2P_SAME_PID((comm->peerInfo + info->rank), peerInfo) && (comm->peerInfo[info->rank].cudaDev != peerInfo->cudaDev)) // 如果目标节点和远端对等节点在同一进程且不同 GPU
    req.refcount++;                                        // 增加引用计数
  if (P2P_SAME_PID((comm->peerInfo + info->rank), myInfo) && (comm->peerInfo[info->rank].cudaDev != myInfo->cudaDev)) // 如果目标节点和本节点在同一进程且不同 GPU
    req.refcount++;                                        // 增加引用计数

   //和proxy建立tcp连接
  NCCLCHECK(ncclProxyConnect(comm, TRANSPORT_P2P, 1, info->rank, &send->proxyConn)); // 连接到代理

  if (useMemcpy) {                                         // 如果使用 cudaMemcpy
    NCCLCHECK(ncclProxyCallBlocking(comm, &send->proxyConn, ncclProxyMsgSetup, NULL, 0, &resources->proxyInfo, sizeof(struct p2pShmProxyInfo))); // 调用代理设置
    memcpy(&info->desc, &resources->proxyInfo.desc, sizeof(ncclShmIpcDesc_t)); // 复制描述符
  } else {                                                  // 不使用 cudaMemcpy
    NCCLCHECK(ncclProxyCallBlocking(comm, &send->proxyConn, ncclProxyMsgSetup, &req, sizeof(struct ncclP2pRequest), &info->p2pBuff, sizeof(struct ncclP2pBuff))); // 调用代理设置
    NCCLCHECK(p2pMap(comm, &send->proxyConn, myInfo, comm->peerInfo+info->rank, &info->p2pBuff, (void**)&resources->sendDevMem, &resources->sendMemIpc)); // 映射 P2P 缓冲区
    resources->sendMemSameProc = P2P_SAME_PID(myInfo, (comm->peerInfo + info->rank)); // 记录是否同一进程
  }

  return ncclSuccess;                                       // 返回成功
}

/* Create and return connect structures for this peer to connect to me */
/* 接收端设置：创建并返回连接结构，供对等节点连接到本节点 */
ncclResult_t p2pRecvSetup(struct ncclComm* comm, struct ncclTopoGraph* graph, struct ncclPeerInfo* myInfo, struct ncclPeerInfo* peerInfo,
    struct ncclConnect* connectInfo, struct ncclConnector * recv, int channelId, int connIndex) {
  struct p2pResources* resources;                          // P2P 资源结构
  struct ncclP2pRequest req;                               // P2P 请求结构

  NCCLCHECK(ncclCalloc(&resources, 1));                    // 分配并清零资源结构
  recv->transportResources = resources;                    // 保存资源到连接器

  int useRead, intermediateRank;                           // 是否使用读操作，中间节点 rank
  NCCLCHECK(p2pGetInfo(comm, myInfo, peerInfo, &useRead, &intermediateRank)); // 获取 P2P 信息

  static_assert(sizeof(struct p2pConnectInfo) <= sizeof(struct ncclConnect), "p2p Connect Info is too big"); // 编译时检查大小

  struct p2pConnectInfo* info = (struct p2pConnectInfo*)connectInfo; // 类型转换
  info->read = useRead;                                    // 设置是否使用读操作
  // For CollNet, use write for scatter-reduce (conn 1), read for broadcast-gather (conn 0)
  // 对于 CollNet，scatter-reduce 使用写操作（连接 1），broadcast-gather 使用读操作（连接 0）
  if (graph && connIndex == 1)                             // 如果是 CollNet 且是连接 1
    info->read = 0;                                        // 使用写操作

  int recvSize = sizeof(struct ncclRecvMem);              // 接收内存大小
  // For P2P Read the SIMPLE buffer is tagged on the end of the ncclSendMem structure
  // 对于 P2P 读，SIMPLE 缓冲区附加在 ncclSendMem 结构的末尾
  // 遍历所有协议，如果不是读操作或者不是 SIMPLE 协议，则添加缓冲区大小
  for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) if (!(info->read && p == NCCL_PROTO_SIMPLE)) recvSize += comm->buffSizes[p];
  ALIGN_SIZE(recvSize, CUDA_IPC_MIN);                     // 对齐到 CUDA IPC 最小大小

  if (intermediateRank == -1) {                            // 如果没有中间节点（直接连接）
    info->rank = myInfo->rank;                             // 设置本节点的 rank
    if (P2P_SAME_PID(myInfo, peerInfo) && ncclParamP2pDirectDisable() == 0 && useMemcpy == 0) { // 同进程且不禁用直接 P2P 且不使用 memcpy
      resources->type = P2P_DIRECT;                        // 设置为直接 P2P 类型
    } else {                                                // 同进程和不同进程都可以使用cumem或者ipc类型
      if (ncclCuMemEnable()) {                              // 如果启用了 cuMem API
        // cuMem API support
        // cuMem API 支持
        resources->type = P2P_CUMEM;                       // 设置为 CUMEM 类型
        TRACE(NCCL_INIT|NCCL_P2P,"Ring %02d : %d[%d] <- %d[%d] via P2P/CUMEM",
              channelId, myInfo->rank, myInfo->nvmlDev, peerInfo->rank, peerInfo->nvmlDev); // 记录跟踪日志
      } else {                                              // 未启用 cuMem API
        // Legacy CUDA IPC
        // 传统 CUDA IPC 方式
        resources->type = P2P_IPC;                         // 设置为 IPC 类型
      }
    }
    recv->conn.flags |= info->read ? NCCL_P2P_READ : NCCL_P2P_WRITE; // 设置连接标志
  } else {                                                  // 有中间节点（间接连接）
    resources->type = P2P_INTERMEDIATE;                    // 设置为中间节点类型
    info->rank = intermediateRank;                         // 设置中间节点的 rank
  }

  memset(&req, '\0', sizeof(req));                         // 清零请求结构
  req.size = recvSize;                                     // 设置请求大小
  req.refcount = 0;                                        // 初始化引用计数
  if (P2P_SAME_PID((comm->peerInfo + info->rank), peerInfo) && (comm->peerInfo[info->rank].cudaDev != peerInfo->cudaDev)) 
    req.refcount++; // 如果目标节点和远端对等节点在同一进程且不同 GPU，增加引用计数
  if (P2P_SAME_PID((comm->peerInfo + info->rank), myInfo) && (comm->peerInfo[info->rank].cudaDev != myInfo->cudaDev)) 
    req.refcount++; // 如果目标节点和本节点在同一进程且不同 GPU，增加引用计数

  NCCLCHECK(ncclProxyConnect(comm, TRANSPORT_P2P, 0, info->rank, &recv->proxyConn)); // 连接到代理
  NCCLCHECK(ncclProxyCallBlocking(comm, &recv->proxyConn, ncclProxyMsgSetup, &req, sizeof(struct ncclP2pRequest), &info->p2pBuff, sizeof(struct ncclP2pBuff))); // 调用代理设置

  NCCLCHECK(p2pMap(comm, &recv->proxyConn, myInfo, comm->peerInfo+info->rank, &info->p2pBuff, (void**)&resources->recvDevMem, &resources->recvMemIpc)); // 映射 P2P 缓冲区
  resources->recvMemSameProc = P2P_SAME_PID(myInfo, (comm->peerInfo + info->rank)); // 记录是否同一进程
  return ncclSuccess;                                       // 返回成功
}

/* Connect/Send to this peer */
/* 连接到对等节点并设置发送端 */
static ncclResult_t p2pSendConnect(struct ncclComm* comm, struct ncclConnect* connectInfo, int nranks, int rank, struct ncclConnector* send) {
  struct p2pResources* resources = (struct p2pResources*)send->transportResources; // 获取 P2P 资源
  struct ncclRecvMem* remDevMem = NULL;                    // 远端设备内存指针
  struct p2pConnectInfo* info = (struct p2pConnectInfo*)connectInfo; // 连接信息

  NCCLCHECK(p2pMap(comm, &send->proxyConn, comm->peerInfo+rank, comm->peerInfo+info->rank, &info->p2pBuff, (void**)&remDevMem, &resources->recvMemIpc)); // 映射 P2P 缓冲区
  resources->recvMemSameProc = P2P_SAME_PID((comm->peerInfo + rank), (comm->peerInfo + info->rank)); // 记录是否同一进程

  char* buff = (char*)(remDevMem+1);                       // 缓冲区指针（跳过 ncclRecvMem 头部）
  for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {               // 遍历所有协议
    if (info->read && p == NCCL_PROTO_SIMPLE) {            // 如果使用读操作且是 SIMPLE 协议
      /* For P2P Read the SIMPLE buffer is local (ncclSendMem) */
      /* 对于 P2P 读，SIMPLE 缓冲区是本地的（ncclSendMem）*/
      if (resources->sendDevMem == NULL)                   // 如果发送设备内存为空
        return ncclInternalError;                          // 返回内部错误：不应该使用读 + memcpy
      send->conn.buffs[p] = (char*)(resources->sendDevMem+1); // 设置本地缓冲区
    } else {                                                // 其他协议
      send->conn.buffs[p] = buff;                          // 设置远端缓冲区
      buff += comm->buffSizes[p];                          // 移动到下一个缓冲区
    }
  }
  send->conn.stepSize = comm->buffSizes[NCCL_PROTO_SIMPLE]/NCCL_STEPS; // 计算步长大小

  if (useMemcpy) {                                         // 如果使用 cudaMemcpy
    send->conn.tail = &resources->proxyInfo.ceRecvMem->tail; // 设置尾指针
    send->conn.connFifo = resources->proxyInfo.ceRecvMem->connFifo; // 设置连接 FIFO
    send->conn.head = &resources->proxyInfo.devShm->sendMem.head; // 设置头指针
    // Send SIMPLE buff to proxy, and replace it by local buffer
    // 发送 SIMPLE 缓冲区到代理，并用本地缓冲区替换它
    NCCLCHECK(ncclProxyCallBlocking(comm, &send->proxyConn, ncclProxyMsgConnect, &send->conn.buffs[NCCL_PROTO_SIMPLE], sizeof(void*), NULL, 0)); // 调用代理连接
    send->conn.buffs[NCCL_PROTO_SIMPLE] = resources->proxyInfo.ceDevBuff; // 使用代理的设备缓冲区
  } else {                                                  // 不使用 cudaMemcpy
    send->conn.tail = &remDevMem->tail;                    // 设置尾指针（远端接收内存的尾）
    send->conn.head = &resources->sendDevMem->head;        // 设置头指针（本地发送内存的头）
    send->conn.ptrExchange = &resources->sendDevMem->ptrExchange; // 设置指针交换位置
    send->conn.redOpArgExchange = resources->sendDevMem->redOpArgExchange; // 设置归约操作参数交换位置
  }
  // We must assign the proxyConn's proxyProgress property for proper checking at enqueue-time
  // 我们必须为 proxyConn 分配 proxyProgress 属性，以便在入队时进行正确检查
  send->proxyConn.proxyProgress = p2pTransport.send.proxyProgress; // 设置代理进度函数
  return ncclSuccess;                                       // 返回成功
}

/* Connect/Recv from this peer */
/* 连接到对等节点并设置接收端 */
ncclResult_t p2pRecvConnect(struct ncclComm* comm, struct ncclConnect* connectInfo, int nranks, int rank, struct ncclConnector* recv) {
  struct p2pResources* resources = (struct p2pResources*)recv->transportResources; // 获取 P2P 资源
  struct p2pConnectInfo* info = (struct p2pConnectInfo*)connectInfo; // 连接信息

  struct ncclSendMem* remDevMem = NULL;                    // 远端设备内存指针

  if (useMemcpy) {                                         // 如果使用 cudaMemcpy
    // Attach to peer's SHM segment
    // 附加到对等节点的共享内存段
    NCCLCHECK(ncclShmImportShareableBuffer(comm, info->rank, &info->desc, (void**)&resources->shm, (void**)&resources->devShm, &resources->desc)); // 导入共享缓冲区

    recv->conn.tail = &resources->devShm->recvMem.tail;    // 设置尾指针
    recv->conn.head = &resources->devShm->sendMem.head;    // 设置头指针
  } else {                                                  // 不使用 cudaMemcpy
    NCCLCHECK(p2pMap(comm, &recv->proxyConn, comm->peerInfo+rank, comm->peerInfo+info->rank, &info->p2pBuff, (void**)&remDevMem, &resources->sendMemIpc)); // 映射 P2P 缓冲区
    resources->sendMemSameProc = P2P_SAME_PID((comm->peerInfo + rank), (comm->peerInfo + info->rank)); // 记录是否同一进程

    struct ncclRecvMem* devMem = resources->recvDevMem;    // 获取接收设备内存
    recv->conn.tail = &devMem->tail;                       // 设置尾指针（本地接收内存的尾）
    recv->conn.head = &remDevMem->head;                    // 设置头指针（远端发送内存的头）
    recv->conn.ptrExchange = &remDevMem->ptrExchange;      // 设置指针交换位置
    recv->conn.redOpArgExchange = remDevMem->redOpArgExchange; // 设置归约操作参数交换位置
  }
  recv->conn.stepSize = comm->buffSizes[NCCL_PROTO_SIMPLE]/NCCL_STEPS; // 计算步长大小

  char* buff = (char*)(resources->recvDevMem+1);           // 缓冲区指针（跳过 ncclRecvMem 头部）
  for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {               // 遍历所有协议
    if (info->read && p == NCCL_PROTO_SIMPLE) {            // 如果使用读操作且是 SIMPLE 协议
      if (remDevMem == NULL) return ncclInternalError;     // 如果远端内存为空，返回错误：不应该使用读 + memcpy
      /* For P2P Read the SIMPLE buffer is remote (ncclSendMem) */
      /* 对于 P2P 读，SIMPLE 缓冲区是远端的（ncclSendMem）*/
      recv->conn.buffs[p] = (char*)(remDevMem+1);          // 设置远端缓冲区
    } else {                                                // 其他协议
      recv->conn.buffs[p] = buff;                          // 设置本地缓冲区
      buff += comm->buffSizes[p];                          // 移动到下一个缓冲区
    }
  }
  return ncclSuccess;                                       // 返回成功
}

// 函数实现：释放发送端连接资源
// 此函数释放发送端连接使用的 P2P 资源
// 参数 send: 发送端连接器
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
ncclResult_t p2pSendFree(struct ncclConnector* send) {
  struct p2pResources* resources = (struct p2pResources*)send->transportResources; // 获取 P2P 资源
  if (resources) {                                          // 如果资源存在
    if (ncclCuMemEnable()) {                                // 如果启用了 cuMem API
      // cuMem API support
      // cuMem API 支持
      if (resources->sendMemIpc) {                          // 如果发送内存 IPC 指针存在
        if (resources->sendMemSameProc) {                   // 如果同一进程
          NCCLCHECK(ncclCuMemFreeAddr(resources->sendMemIpc)); // 释放 cuMem 地址
        } else {                                            // 不同进程
          NCCLCHECK(ncclCudaFree(resources->sendMemIpc));   // 释放 CUDA 内存
        }
      }

      if (resources->recvMemIpc) {                          // 如果接收内存 IPC 指针存在
        if (resources->recvMemSameProc) {                   // 如果同一进程
          NCCLCHECK(ncclCuMemFreeAddr(resources->recvMemIpc)); // 释放 cuMem 地址
        } else {                                            // 不同进程
          NCCLCHECK(ncclCudaFree(resources->recvMemIpc));   // 释放 CUDA 内存
        }
      }
    }
    else {                                                  // 未启用 cuMem API
      if (resources->sendMemIpc)
        CUDACHECK(cudaIpcCloseMemHandle(resources->sendMemIpc)); // 关闭发送内存 IPC 句柄
      if (resources->recvMemIpc)
        CUDACHECK(cudaIpcCloseMemHandle(resources->recvMemIpc)); // 关闭接收内存 IPC 句柄
    }
    free(resources);                                        // 释放资源结构
  }
  return ncclSuccess;                                       // 返回成功
}

// 函数实现：释放接收端连接资源
// 此函数释放接收端连接使用的 P2P 资源
// 参数 recv: 接收端连接器
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
ncclResult_t p2pRecvFree(struct ncclConnector* recv) {
  struct p2pResources* resources = (struct p2pResources*)recv->transportResources; // 获取 P2P 资源
  if (resources) {                                          // 如果资源存在
    if (ncclCuMemEnable()) {                                // 如果启用了 cuMem API
      // cuMem API support
      // cuMem API 支持
      if (resources->sendMemIpc) {                          // 如果发送内存 IPC 指针存在
        if (resources->sendMemSameProc) {                   // 如果同一进程
          NCCLCHECK(ncclCuMemFreeAddr(resources->sendMemIpc)); // 释放 cuMem 地址
        } else {                                            // 不同进程
          NCCLCHECK(ncclCudaFree(resources->sendMemIpc));   // 释放 CUDA 内存
        }
      }

      if (resources->recvMemIpc) {                          // 如果接收内存 IPC 指针存在
        if (resources->recvMemSameProc) {                   // 如果同一进程
          NCCLCHECK(ncclCuMemFreeAddr(resources->recvMemIpc)); // 释放 cuMem 地址
        } else {                                            // 不同进程
          NCCLCHECK(ncclCudaFree(resources->recvMemIpc));   // 释放 CUDA 内存
        }
      }
    }
    else {                                                  // 未启用 cuMem API
      if (resources->sendMemIpc) CUDACHECK(cudaIpcCloseMemHandle(resources->sendMemIpc)); // 关闭发送内存 IPC 句柄
      if (resources->recvMemIpc) CUDACHECK(cudaIpcCloseMemHandle(resources->recvMemIpc)); // 关闭接收内存 IPC 句柄
      if (useMemcpy) {                                      // 如果使用 cudaMemcpy
        NCCLCHECK(ncclShmIpcClose(&resources->desc));       // 关闭共享内存 IPC
      }
    }
    free(resources);                                        // 释放资源结构
  }
  return ncclSuccess;                                       // 返回成功
}

// 函数实现：发送端代理设置
// 此函数在代理端设置发送端的 P2P 资源
// 参数 connection: 代理连接
// 参数 proxyState: 代理状态
// 参数 reqBuff: 请求缓冲区
// 参数 reqSize: 请求大小
// 参数 respBuff: 响应缓冲区
// 参数 respSize: 响应大小
// 参数 done: 输出参数，指示是否完成
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
static ncclResult_t p2pSendProxySetup(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState, void* reqBuff, int reqSize, void* respBuff, int respSize, int* done) {
  if (useMemcpy) {                                         // 如果使用 cudaMemcpy
    // CE memcpy support
    // CE（Copy Engine）memcpy 支持
    struct p2pShmProxyInfo* proxyInfo;                     // 共享内存代理信息
    size_t shmSize;                                        // 共享内存大小

    if (respSize != sizeof(struct p2pShmProxyInfo)) 
        return ncclInternalError; // 检查响应大小
        
    NCCLCHECK(ncclCalloc(&proxyInfo, 1));                  // 分配并清零代理信息
    connection->transportResources = proxyInfo;            // 保存到连接

    NCCLCHECK(ncclCudaCalloc(&proxyInfo->ceDevBuff, proxyState->buffSizes[NCCL_PROTO_SIMPLE])); // 分配 CE 设备缓冲区

    // Create a SHM segment for the peer to attach to
    // 创建共享内存段供对等节点附加
    shmSize = sizeof(struct ncclSendMem) + sizeof(struct ncclRecvMem); // 计算共享内存大小
    NCCLCHECK(ncclShmAllocateShareableBuffer(shmSize, false, &proxyInfo->desc, (void**)&proxyInfo->shm, (void**)&proxyInfo->devShm)); // 分配共享缓冲区

    NCCLCHECK(ncclCudaHostCalloc(&proxyInfo->ceRecvMem, 1)); // 分配 CE 接收内存（主机内存）
    memcpy(respBuff, proxyInfo, sizeof(struct p2pShmProxyInfo)); // 复制代理信息到响应缓冲区
  } else {                                                  // 不使用 cudaMemcpy
    struct ncclP2pRequest* req = (struct ncclP2pRequest*)reqBuff; // 获取请求
    if (reqSize != sizeof(struct ncclP2pRequest)) 
        return ncclInternalError; // 检查请求大小
    int size = req->size;                                  // 获取请求大小
    if (respSize != sizeof(struct ncclP2pBuff)) 
        return ncclInternalError; // 检查响应大小
        
    struct ncclP2pBuff* p2pBuff = (struct ncclP2pBuff*)respBuff; // P2P 缓冲区
    NCCLCHECK(ncclP2pAllocateShareableBuffer(size, req->refcount, &p2pBuff->ipcDesc, &p2pBuff->directPtr)); // 分配可共享缓冲区
    p2pBuff->size = size;                                  // 设置缓冲区大小
    
    if (ncclCuMemEnable()) {                                // 如果启用了 cuMem API
      // cuMem API support
      // cuMem API 支持
      struct p2pCuMemProxyInfo* proxyInfo;                 // cuMem 代理信息
      NCCLCHECK(ncclCalloc(&proxyInfo, 1));                // 分配并清零代理信息
      memcpy(&proxyInfo->p2pBuff, p2pBuff, sizeof(*p2pBuff)); // 复制 P2P 缓冲区信息
      connection->transportResources = proxyInfo;          // 保存到连接
    } else {                                                // 未启用 cuMem API
      connection->transportResources = p2pBuff->directPtr; // 直接保存指针
    }
  }
  *done = 1;                                               // 标记为完成
  return ncclSuccess;                                       // 返回成功
}

// 函数实现：接收端代理设置
// 此函数在代理端设置接收端的 P2P 资源
// 参数 connection: 代理连接
// 参数 proxyState: 代理状态
// 参数 reqBuff: 请求缓冲区
// 参数 reqSize: 请求大小
// 参数 respBuff: 响应缓冲区
// 参数 respSize: 响应大小
// 参数 done: 输出参数，指示是否完成
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
static ncclResult_t p2pRecvProxySetup(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState, void* reqBuff, int reqSize, void* respBuff, int respSize, int* done) {
  struct ncclP2pRequest* req = (struct ncclP2pRequest*)reqBuff; // 获取请求
  if (reqSize != sizeof(struct ncclP2pRequest)) return ncclInternalError; // 检查请求大小
  int size = req->size;                                    // 获取请求大小
  if (respSize != sizeof(struct ncclP2pBuff)) return ncclInternalError; // 检查响应大小
  struct ncclP2pBuff* p2pBuff = (struct ncclP2pBuff*)respBuff; // P2P 缓冲区
  NCCLCHECK(ncclP2pAllocateShareableBuffer(size, req->refcount, &p2pBuff->ipcDesc, &p2pBuff->directPtr)); // 分配可共享缓冲区
  p2pBuff->size = size;                                    // 设置缓冲区大小
  if (ncclCuMemEnable()) {                                  // 如果启用了 cuMem API
    // cuMem API support
    // cuMem API 支持
    struct p2pCuMemProxyInfo* proxyInfo;                   // cuMem 代理信息
    NCCLCHECK(ncclCalloc(&proxyInfo, 1));                  // 分配并清零代理信息
    memcpy(&proxyInfo->p2pBuff, p2pBuff, sizeof(*p2pBuff)); // 复制 P2P 缓冲区信息
    connection->transportResources = proxyInfo;            // 保存到连接
  } else {                                                  // 未启用 cuMem API
    connection->transportResources = p2pBuff->directPtr;   // 直接保存指针
  }
  *done = 1;                                               // 标记为完成
  return ncclSuccess;                                       // 返回成功
}

// 函数实现：发送端代理连接
// 此函数在代理端建立发送端连接
// 参数 connection: 代理连接
// 参数 proxyState: 代理状态
// 参数 reqBuff: 请求缓冲区
// 参数 reqSize: 请求大小
// 参数 respBuff: 响应缓冲区
// 参数 respSize: 响应大小
// 参数 done: 输出参数，指示是否完成
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
static ncclResult_t p2pSendProxyConnect(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState, void* reqBuff, int reqSize, void* respBuff, int respSize, int* done) {
  struct p2pShmProxyInfo* proxyInfo = (struct p2pShmProxyInfo*)connection->transportResources; // 获取代理信息

  if (reqSize != sizeof(void*)) return ncclInternalError; // 检查请求大小
  proxyInfo->recvFifo = *((char**)reqBuff);                // 获取接收 FIFO 指针

  CUDACHECK(cudaStreamCreateWithFlags(&proxyInfo->stream, cudaStreamNonBlocking)); // 创建非阻塞 CUDA 流
  for (int i=0; i<NCCL_STEPS; i++) {                       // 为每个步骤创建 CUDA 事件
    CUDACHECK(cudaEventCreate(proxyInfo->events+i));       // 创建事件
  }
  connection->proxyAppendPtr = &connection->proxyAppend;   // 设置代理追加指针
  return ncclSuccess;                                       // 返回成功
}

// 函数实现：释放发送端代理资源
// 此函数释放发送端代理使用的资源
// 参数 connection: 代理连接
// 参数 proxyState: 代理状态
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
static ncclResult_t p2pSendProxyFree(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState) {
  // CE memcpy support
  // CE（Copy Engine）memcpy 支持
  if (useMemcpy) {                                         // 如果使用 cudaMemcpy
    struct p2pShmProxyInfo* proxyInfo = (struct p2pShmProxyInfo*)connection->transportResources; // 获取代理信息
    if (proxyInfo) {                                        // 如果代理信息存在
      NCCLCHECK(ncclShmIpcClose(&proxyInfo->desc));        // 关闭共享内存 IPC
      NCCLCHECK(ncclCudaHostFree(proxyInfo->ceRecvMem));   // 释放 CE 接收内存（主机内存）
      NCCLCHECK(ncclCudaFree(proxyInfo->ceDevBuff));       // 释放 CE 设备缓冲区
      CUDACHECK(cudaStreamDestroy(proxyInfo->stream));     // 销毁 CUDA 流
      for (int i=0; i<NCCL_STEPS; i++) {                   // 销毁所有 CUDA 事件
        CUDACHECK(cudaEventDestroy(proxyInfo->events[i])); // 销毁事件
      }
      free(proxyInfo);                                      // 释放代理信息结构
    }
  } else {                                                  // 不使用 cudaMemcpy
    if (ncclCuMemEnable()) {                                // 如果启用了 cuMem API
      // cuMem API support
      // cuMem API 支持
      struct p2pCuMemProxyInfo *proxyInfo = (struct p2pCuMemProxyInfo *) connection->transportResources; // 获取代理信息
      if (proxyInfo) {                                      // 如果代理信息存在
        struct ncclP2pBuff *p2pBuff = &proxyInfo->p2pBuff; // 获取 P2P 缓冲区
        ncclP2pFreeShareableBuffer(&p2pBuff->ipcDesc);     // 释放可共享缓冲区
        ncclCudaFree(p2pBuff->directPtr);                  // 释放直接指针
        free(proxyInfo);                                    // 释放代理信息结构
      }
    } else {                                                // 未启用 cuMem API
      // Do not check return code as CUDA may have already shut down
      // 不检查返回码，因为 CUDA 可能已经关闭
      ncclCudaFree(connection->transportResources);        // 释放传输资源
    }
  }
  return ncclSuccess;                                       // 返回成功
}

// 函数实现：释放接收端代理资源
// 此函数释放接收端代理使用的资源
// 参数 connection: 代理连接
// 参数 proxyState: 代理状态
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
static ncclResult_t p2pRecvProxyFree(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState) {
  if (ncclCuMemEnable()) {                                  // 如果启用了 cuMem API
    struct p2pCuMemProxyInfo *proxyInfo = (struct p2pCuMemProxyInfo *) connection->transportResources; // 获取代理信息
    if (proxyInfo) {                                        // 如果代理信息存在
      struct ncclP2pBuff *p2pBuff = &proxyInfo->p2pBuff;   // 获取 P2P 缓冲区
      ncclP2pFreeShareableBuffer(&p2pBuff->ipcDesc);       // 释放可共享缓冲区
      ncclCudaFree(p2pBuff->directPtr);                    // 释放直接指针
      free(proxyInfo);                                      // 释放代理信息结构
    }
  } else {                                                  // 未启用 cuMem API
    // Do not check return code as CUDA may have already shut down
    // 不检查返回码，因为 CUDA 可能已经关闭
    ncclCudaFree(connection->transportResources);          // 释放传输资源
  }
  return ncclSuccess;                                       // 返回成功
}

// CE memcpy support
// CE（Copy Engine）memcpy 支持
// 函数实现：发送端代理进度
// 此函数在代理端处理发送进度，使用 cudaMemcpy 进行数据传输
// 参数 proxyState: 代理状态
// 参数 args: 代理参数
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
static ncclResult_t p2pSendProxyProgress(struct ncclProxyState* proxyState, struct ncclProxyArgs* args) {
  if (args->state == ncclProxyOpReady) {                   // 如果操作状态为就绪
    for (int s=0; s<args->nsubs; s++) {                     // 遍历所有子操作
      struct ncclProxySubArgs* sub = args->subs+s;         // 获取子操作参数
      struct p2pShmProxyInfo* resources = (struct p2pShmProxyInfo*) (sub->connection->transportResources); // 获取资源
      // Round to next multiple of sliceSteps
      // 向上舍入到 sliceSteps 的下一个倍数
      sub->base = ROUNDUP(resources->step, args->chunkSteps); // 计算基础步骤
      sub->posted = sub->transmitted = sub->done = 0;       // 初始化计数器
    }
    args->state = ncclProxyOpProgress;                     // 更新状态为进行中
  }
  args->idle = 1;                                          // 设置空闲标志
  if (args->state == ncclProxyOpProgress) {                // 如果操作状态为进行中
    int p = args->protocol;                                // 获取协议类型
    int stepSize = proxyState->buffSizes[p] / NCCL_STEPS;  // 计算步长大小
    for (int s=0; s<args->nsubs; s++) {                     // 遍历所有子操作
      struct ncclProxySubArgs* sub = args->subs+s;         // 获取子操作参数
      struct p2pShmProxyInfo* resources = (struct p2pShmProxyInfo*) (sub->connection->transportResources); // 获取资源
      if (p != NCCL_PROTO_SIMPLE) { // Only Simple uses cudaMemcpy
          // 只有 SIMPLE 协议使用 cudaMemcpy
          resources->step = sub->base + sub->nsteps;       // 更新步骤
          args->done++;                                     // 增加完成计数
          continue;                                        // 继续下一个子操作
      }
      // 处理传输逻辑
      if (sub->transmitted < sub->done + NCCL_STEPS && sub->transmitted < sub->nsteps) { // 如果可以传输更多数据
        int buffSlot = (sub->base+sub->transmitted)%NCCL_STEPS; // 计算缓冲区槽位
        volatile struct ncclConnFifo* connFifo = resources->ceRecvMem->connFifo; // 获取连接 FIFO
        volatile uint64_t* recvTail = &resources->ceRecvMem->tail; // 获取接收尾指针
        // Check GPU has sent everything
        // 检查 GPU 是否已发送所有数据
        if ((*recvTail > sub->base+sub->transmitted)) {    // 如果接收尾指针大于当前传输位置
          int size = connFifo[buffSlot].size;              // 获取数据大小
          CUDACHECK(cudaMemcpyAsync(resources->recvFifo+buffSlot*stepSize, resources->ceDevBuff+buffSlot*stepSize, size, cudaMemcpyDeviceToDevice, resources->stream)); // 异步复制数据
          CUDACHECK(cudaEventRecord(resources->events[buffSlot], resources->stream)); // 记录事件
          sub->transmitted += args->sliceSteps;            // 增加传输计数
        }
      }
      // 检查完成状态
      if (sub->done < sub->transmitted) {                   // 如果完成的步骤少于传输的步骤
        int buffSlot = (sub->base+sub->done)%NCCL_STEPS;   // 计算缓冲区槽位
        cudaError_t res = cudaEventQuery(resources->events[buffSlot]); // 查询事件状态
        if (res != cudaErrorNotReady) CUDACHECK(res);      // 如果不是未就绪状态，检查错误
        if (res == cudaSuccess) {                           // 如果事件完成
          sub->done += args->sliceSteps;                   // 增加完成计数
          // Notify SHM
          // 通知共享内存
          resources->shm->recvMem.tail = sub->base + sub->done; // 更新共享内存尾指针
        }
        if (sub->done == sub->nsteps) {                     // 如果所有步骤都完成
          resources->step = sub->base + sub->nsteps;       // 更新资源步骤
          args->done++;                                     // 增加完成计数
        }
      }
    }
    if (args->done == args->nsubs) {                        // 如果所有子操作都完成
      args->state = ncclProxyOpNone;                       // 更新状态为无操作
    }
  }
  return ncclSuccess;                                       // 返回成功
}

// 函数实现：注册 IPC 缓冲区
// 此函数将用户缓冲区注册为可在进程间共享的 IPC 缓冲区
// 参数 comm: NCCL 通信上下文
// 参数 userbuff: 用户缓冲区指针
// 参数 buffSize: 缓冲区大小
// 参数 peerRanks: 对等节点的 rank 数组
// 参数 nPeers: 对等节点数量
// 参数 type: IPC 注册类型（集体操作或点对点）
// 参数 regRecord: 注册记录
// 参数 regBufFlag: 输出参数，指示是否已注册
// 参数 offsetOut: 输出参数，偏移量
// 参数 peerRmtAddrsOut: 输出参数，对等节点的远端地址数组
// 参数 isLegacyIpc: 输出参数，指示是否使用传统 IPC
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
static ncclResult_t ipcRegisterBuffer(ncclComm* comm, const void* userbuff, size_t buffSize, int* peerRanks, int nPeers, ncclIpcRegType type, struct ncclReg* regRecord, int* regBufFlag, uintptr_t* offsetOut, uintptr_t** peerRmtAddrsOut, bool* isLegacyIpc) {
ncclResult_t ret = ncclSuccess;                           // 返回值初始化
  struct ncclIpcRegInfo* newInfo = NULL;                  // 新的 IPC 注册信息
  uintptr_t* peerRmtAddrs = NULL;                         // 对等节点远端地址数组
  int legacyIpcCap = 0;                                   // 传统 IPC 能力标志
  size_t baseSize = 0;                                    // 基地址大小
  void* baseAddr = NULL;                                  // 基地址
  bool needUpdate = false;                                // 是否需要更新标志

  *regBufFlag = 0;                                        // 初始化注册标志为 0
  *offsetOut = 0;                                         // 初始化偏移量为 0
  *peerRmtAddrsOut = NULL;                                // 初始化远端地址数组为 NULL
  if (isLegacyIpc) *isLegacyIpc = false;                  // 如果提供了指针，初始化为 false
  if (regRecord) {                                        // 如果存在注册记录
    // buffer was registered by by users, we need to start to register or reuse it
    // 缓冲区已由用户注册，我们需要开始注册或重用它
    int peerLocalRank = -1;                               // 对等节点的本地 rank
    for (int p = 0; p < nPeers; p++) {                     // 遍历所有对等节点
      int peerRank = peerRanks[p];                        // 获取对等节点的 rank
      peerLocalRank = comm->rankToLocalRank[peerRank];    // 转换为本地 rank
      if (regRecord->ipcInfos[peerLocalRank]) {            // 如果该对等节点的 IPC 信息已存在
        // We already have IPC info for peerLocalRank, no need to register it, we can reuse it
        // 我们已经有了该对等节点的 IPC 信息，无需注册，可以重用
        *regBufFlag = 1;                                  // 设置注册标志为 1（已注册）
        if (isLegacyIpc) *isLegacyIpc = regRecord->ipcInfos[peerLocalRank]->impInfo.legacyIpcCap; // 设置传统 IPC 标志
        INFO(NCCL_REG, "rank %d - IPC reuse buffer %p size %ld (baseAddr %p size %ld) to peer %d regAddr %p", comm->rank, userbuff, buffSize, (void*)regRecord->begAddr, regRecord->endAddr - regRecord->begAddr, peerRank, regRecord->ipcInfos[peerLocalRank]->impInfo.rmtRegAddr); // 记录重用信息
      } else {                                             // 该对等节点的 IPC 信息不存在
        // Register buffer with peerLocalRank
        // 向该对等节点注册缓冲区
        struct ncclProxyConnector* proxyConn = NULL;      // 代理连接器
        struct p2pIpcExpInfo ipcInfo;                     // IPC 导出信息

        if (baseAddr == NULL) {                           // 如果基地址为空（第一次）
          CUCHECKGOTO(cuMemGetAddressRange((CUdeviceptr*)&baseAddr, &baseSize, (CUdeviceptr)userbuff), ret, fail); // 获取地址范围
          CUCHECKGOTO(cuPointerGetAttribute((void*)&legacyIpcCap, CU_POINTER_ATTRIBUTE_IS_LEGACY_CUDA_IPC_CAPABLE, (CUdeviceptr)baseAddr), ret, fail); // 获取传统 IPC 能力
        }
        if (comm->gproxyConn[peerRank].initialized == false) // 如果代理连接未初始化
          NCCLCHECKGOTO(ncclProxyConnect(comm, TRANSPORT_P2P, 1, peerRank, &comm->gproxyConn[peerRank]), ret, fail); // 连接到代理
        proxyConn = &comm->gproxyConn[peerRank];          // 获取代理连接

        // Get the mem handle for that buffer. It may have been allocated through cudaMalloc in which case we'll
        // get the CUDA legacy mem handle, or through cuMem*.
        // 获取该缓冲区的内存句柄。它可能是通过 cudaMalloc 分配的，这种情况下我们将获得 CUDA 传统内存句柄，或通过 cuMem* 分配。
        if (ncclCuMemEnable()) {                           // 如果启用了 cuMem API
          CUmemGenericAllocationHandle handle;            // cuMem 分配句柄
          if (CUPFN(cuMemRetainAllocationHandle(&handle, baseAddr)) != CUDA_SUCCESS) { // 尝试保留分配句柄
            // if cuMem* export fails, retry legacy export
            // 如果 cuMem* 导出失败，重试传统导出
            if (comm->directMode || !ncclParamLegacyCudaRegister()) // 如果是直接模式或禁用传统 CUDA 注册
                goto fail;                                // 跳转到失败处理
            CUDACHECKGOTO(cudaIpcGetMemHandle(&ipcInfo.ipcDesc.devIpc, baseAddr), ret, fail); // 获取传统 CUDA IPC 句柄
            ipcInfo.legacyIpcCap = true;                  // 设置传统 IPC 标志
            if (isLegacyIpc) *isLegacyIpc = true;         // 输出传统 IPC 标志
          } else {                                         // cuMem 导出成功
            ipcInfo.legacyIpcCap = false;                 // 设置非传统 IPC
            if (isLegacyIpc) *isLegacyIpc = false;        // 输出非传统 IPC 标志
            // cuMem* export to file descriptor or fabric handle
            // cuMem* 导出为文件描述符或 fabric 句柄
            if (proxyConn->sameProcess) {                  // 如果是同一进程
              memcpy(&ipcInfo.ipcDesc.memHandle, &handle, sizeof(CUmemGenericAllocationHandle)); // 直接复制句柄
            } else {                                        // 不同进程
              if (ncclCuMemHandleType == CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) { // 如果是 POSIX 文件描述符类型
                int expFd = -1;                           // 导出的文件描述符
                CUCHECKGOTO(cuMemExportToShareableHandle(&expFd, handle, ncclCuMemHandleType, 0), ret, fail); // 导出为文件描述符
                NCCLCHECKGOTO(ncclProxyClientQueryFdBlocking(comm, proxyConn, expFd, &ipcInfo.impFd), ret, fail); // 通过代理查询文件描述符
                SYSCHECKGOTO(close(expFd), "close", ret, fail); // 关闭文件描述符
              } else {                                      // 其他类型（如 fabric handle）
                // Allow this to silently fail for cases where the user buff cannot be registered
                // 允许静默失败，用于用户缓冲区无法注册的情况
                if (CUPFN(cuMemExportToShareableHandle(&ipcInfo.ipcDesc.cuDesc.handle, handle, ncclCuMemHandleType, 0)) != CUDA_SUCCESS) { // 尝试导出
                  CUCHECKGOTO(cuMemRelease(handle), ret, fail); // 释放句柄
                  goto fail;                              // 跳转到失败处理
                }
              }
            }
            CUCHECKGOTO(cuMemRelease(handle), ret, fail);  // 释放句柄
          }
        } else if (legacyIpcCap) {                          // 未启用 cuMem API 但支持传统 IPC
          // legacy export
          // 传统导出
          if (comm->directMode || !ncclParamLegacyCudaRegister()) // 如果是直接模式或禁用传统 CUDA 注册
            goto fail;                                    // 跳转到失败处理
          CUDACHECKGOTO(cudaIpcGetMemHandle(&ipcInfo.ipcDesc.devIpc, baseAddr), ret, fail); // 获取传统 CUDA IPC 句柄
          ipcInfo.legacyIpcCap = true;                    // 设置传统 IPC 标志
          if (isLegacyIpc)                                // 如果提供了指针
            *isLegacyIpc = true;                          // 输出传统 IPC 标志
        } else {                                            // 既不支持 cuMem 也不支持传统 IPC
          // nothing works, just return
          // 没有任何方法可行，直接返回
          goto fail;                                      // 跳转到失败处理
        }

        void* rmtRegAddr = NULL;                             // 远端注册地址
        ipcInfo.size = baseSize;                            // 设置缓冲区大小
        ipcInfo.offset = regRecord->begAddr - (uintptr_t)baseAddr; // 计算偏移量
        // Now ipcInfo contains all necessary registration info. Start to register buffer on proxy side
        // and get the remote register address back.
        // 现在 ipcInfo 包含所有必要的注册信息。开始在代理端注册缓冲区并获取远端注册地址。
        if (proxyConn) {                                    // 如果代理连接存在
          INFO(NCCL_REG, "rank %d - IPC registering buffer %p size %ld (baseAddr %p size %ld) to peer %d", comm->rank, userbuff, buffSize, (void*)regRecord->begAddr, ipcInfo.size, peerRank); // 记录注册信息
          NCCLCHECKGOTO(ncclProxyCallBlocking(comm, proxyConn, ncclProxyMsgRegister, &ipcInfo, sizeof(p2pIpcExpInfo), &rmtRegAddr, sizeof(void*)), ret, fail); // 调用代理注册
        }
        if (rmtRegAddr) {                                   // 如果远端注册地址有效
          NCCLCHECKGOTO(ncclCalloc(&newInfo, 1), ret, fail); // 分配新的 IPC 信息
          assert(regRecord->ipcInfos[peerLocalRank] == NULL); // 断言该位置为空
          regRecord->state |= IPC_REG_COMPLETE;            // 设置注册完成标志
          newInfo->peerRank = peerRank;                    // 设置对等节点 rank
          newInfo->baseAddr = baseAddr;                    // 设置基地址
          newInfo->impInfo.rmtRegAddr = rmtRegAddr;        // 设置远端注册地址
          newInfo->impInfo.offset = ipcInfo.offset;        // 设置偏移量
          newInfo->impInfo.legacyIpcCap = ipcInfo.legacyIpcCap; // 设置传统 IPC 标志
          newInfo->ipcProxyconn = proxyConn;               // 设置代理连接
          regRecord->ipcInfos[peerLocalRank] = newInfo;    // 保存 IPC 信息
          if (regRecord->regIpcAddrs.hostPeerRmtAddrs == NULL) { // 如果主机端远端地址数组不存在
            NCCLCHECKGOTO(ncclCalloc(&regRecord->regIpcAddrs.hostPeerRmtAddrs, comm->localRanks), ret, fail); // 分配地址数组
          }
          regRecord->regIpcAddrs.hostPeerRmtAddrs[peerLocalRank] = (uintptr_t)rmtRegAddr; // 保存远端地址
          needUpdate = true;                               // 设置需要更新标志
          *regBufFlag = 1;                                 // 设置注册完成标志
          INFO(NCCL_REG, "rank %d - IPC register buffer %p size %ld (baseAddr %p size %ld) to peer %d regAddr %p offsetOut %ld", comm->rank, userbuff, buffSize, (void*)regRecord->begAddr, ipcInfo.size, peerRank, rmtRegAddr, (uintptr_t)userbuff - regRecord->begAddr); // 记录成功信息
        }
      }
    }

    if (*regBufFlag) {                                     // 如果已注册
      if (type == NCCL_IPC_COLLECTIVE) {                   // 如果是集体操作类型
        // for collective, store registered remote buffers into dev memory for future reference
        // 对于集体操作，将已注册的远端缓冲区存储到设备内存中以供将来参考
        if (regRecord->regIpcAddrs.devPeerRmtAddrs == NULL || needUpdate) { // 如果设备端地址不存在或需要更新
          cudaStream_t hostStream, deviceStream;           // 主机流和设备流
          NCCLCHECKGOTO(ncclStrongStreamAcquire(ncclCudaGraphNone(), &comm->sharedRes->hostStream, /*concurrent=*/false, &hostStream), ret, fail); // 获取主机流
          NCCLCHECKGOTO(ncclStrongStreamAcquire(ncclCudaGraphNone(), &comm->sharedRes->deviceStream, /*concurrent=*/false, &deviceStream), ret, fail); // 获取设备流
          if (regRecord->regIpcAddrs.devPeerRmtAddrs == NULL) // 如果设备端地址不存在
            NCCLCHECKGOTO(ncclCudaCallocAsync(&regRecord->regIpcAddrs.devPeerRmtAddrs, comm->localRanks, hostStream), ret, fail); // 在设备上分配内存
          if (needUpdate)                                  // 如果需要更新
            NCCLCHECKGOTO(ncclCudaMemcpyAsync(regRecord->regIpcAddrs.devPeerRmtAddrs, regRecord->regIpcAddrs.hostPeerRmtAddrs, comm->localRanks, hostStream), ret, fail); // 异步复制到设备
          NCCLCHECKGOTO(ncclStreamWaitStream(deviceStream, hostStream, comm->sharedRes->scratchEvent), ret, fail); // 设备流等待主机流
          NCCLCHECKGOTO(ncclStrongStreamRelease(ncclCudaGraphNone(), &comm->sharedRes->hostStream, /*concurrent=*/false), ret, fail); // 释放主机流
          NCCLCHECKGOTO(ncclStrongStreamRelease(ncclCudaGraphNone(), &comm->sharedRes->deviceStream, /*concurrent=*/false), ret, fail); // 释放设备流
        }
        peerRmtAddrs = regRecord->regIpcAddrs.devPeerRmtAddrs; // 使用设备端地址
      } else {                                              // 点对点类型
        assert(nPeers == 1);                               // 断言只有一个对等节点
        // p2p always returns remote addr here since remote buffer addr is passed in ncclDevWorkP2p struct
        // p2p 总是返回远端地址，因为远端缓冲区地址在 ncclDevWorkP2p 结构中传递
        peerRmtAddrs = (uintptr_t*)regRecord->regIpcAddrs.hostPeerRmtAddrs[peerLocalRank]; // 使用主机端地址
      }
      *offsetOut = (uintptr_t)userbuff - regRecord->begAddr; // 计算偏移量
      *peerRmtAddrsOut = peerRmtAddrs;                     // 输出远端地址数组
    }
  }
exit:
  return ret;
fail:
  *regBufFlag = 0;
  *offsetOut = 0;
  *peerRmtAddrsOut = NULL;
  if (newInfo) 
    free(newInfo);
  INFO(NCCL_REG, "rank %d failed to IPC register userbuff %p buffSize %ld nPeers %d isLegacyIpc %d type %s", comm->rank, userbuff, buffSize, nPeers, isLegacyIpc ? *isLegacyIpc : -1, ncclCuMemHandleType == CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR ? "POSIX_FD" : "FABRIC");
  goto exit;
}

// 函数功能：本地 IPC 缓冲区注册
// 这是一个运行时（非 CUDA Graph）模式下使用的 IPC 内存注册函数
// 用于将用户缓冲区注册为 IPC 可共享的内存，使同一节点内的其他 GPU 可以直接访问
// 参数说明：
//   - comm: NCCL 通信上下文指针，包含通信域的所有信息
//   - userbuff: 用户提供的缓冲区指针，需要注册为 IPC 可共享内存
//   - buffSize: 缓冲区大小（字节数）
//   - peerRanks: 对等节点的 rank 数组，指明要与哪些对等节点共享此缓冲区
//   - nPeers: 对等节点的数量
//   - type: IPC 注册类型（NCCL_IPC_P2P 点对点类型 或 NCCL_IPC_COLLECTIVE 集合操作类型）
//   - regBufFlag: 输出参数，指示缓冲区是否已成功注册（0=未注册，1=已注册）
//   - offsetOut: 输出参数，返回 userbuff 相对于注册记录起始地址的偏移量
//   - peerRmtAddrsOut: 输出参数，返回对等节点可以使用的远端地址数组指针
// 返回值：ncclSuccess 表示成功，其他值表示失败
ncclResult_t ncclIpcLocalRegisterBuffer(ncclComm* comm, const void* userbuff, size_t buffSize, int* peerRanks, int nPeers, ncclIpcRegType type, int* regBufFlag, uintptr_t* offsetOut, uintptr_t** peerRmtAddrsOut) {
  // 初始化返回值为成功
  ncclResult_t ret = ncclSuccess;
  // 声明指针：内存注册记录
  // 用于查找或创建与 userbuff 对应的内存注册记录
  struct ncclReg *regRecord = NULL;
  // 声明变量：注册记录是否有效的标志
  // 用于检查本地引用计数是否大于 0（即内存是否已被本地注册）
  bool isValid = false;
  // 声明指针：CUDA 内存分配的基地址
  // cuMemGetAddressRange 会返回包含 userbuff 的完整 CUDA 内存分配块的起始地址
  void *baseAddr = NULL;
  // 声明变量：CUDA 内存分配块的大小
  // cuMemGetAddressRange 会返回完整内存分配块的大小
  size_t baseSize = 0;

  // 初始化输出参数：注册标志设置为 0（未注册）
  *regBufFlag = 0;
  // 初始化输出参数：偏移量设置为 0
  *offsetOut = 0;
  // 初始化输出参数：远端地址数组指针设置为 NULL
  *peerRmtAddrsOut = NULL;
  // 检查所有输入参数的有效性
  // 条件：comm 不为空 && userbuff 不为空 && buffSize 大于 0 && nPeers 大于 0
  if (comm && userbuff && buffSize > 0 && nPeers > 0) {
    // 查找或创建与 userbuff 对应的内存注册记录
    // ncclRegFind: 在 comm->regCache 中查找是否已有包含此缓冲区的注册记录
    // 如果找到，返回对应的 ncclReg 指针；如果未找到，返回 NULL
    // NCCLCHECKGOTO: 如果失败，跳转到 fail 标签进行错误处理
    NCCLCHECKGOTO(ncclRegFind(comm, userbuff, buffSize, &regRecord), ret, fail);
    // 检查注册记录的本地引用计数是否有效
    // ncclRegLocalIsValid: 检查 regRecord->localRefs 是否大于 0
    // 大于 0 表示内存已被本地注册（通过 ncclCommRegister 或 ncclCommGraphRegister）
    // isValid: 输出参数，true 表示有效，false 表示无效
    NCCLCHECKGOTO(ncclRegLocalIsValid(regRecord, &isValid), ret, fail);
    // 只有当注册记录有效时，才进行 IPC 注册
    // 条件：本地引用计数大于 0（内存已被本地注册）
    if (isValid) {
      // 获取包含 userbuff 的 CUDA 内存分配块的地址范围
      // cuMemGetAddressRange: CUDA 驱动 API，获取包含指定地址的完整内存分配块
      // (CUdeviceptr *)&baseAddr: 输出参数，返回内存块的起始地址
      // &baseSize: 输出参数，返回内存块的大小
      // (CUdeviceptr)userbuff: 输入参数，用户缓冲区的地址
      // CUCHECKGOTO: 如果 CUDA API 调用失败，跳转到 fail 标签
      CUCHECKGOTO(cuMemGetAddressRange((CUdeviceptr *)&baseAddr, &baseSize, (CUdeviceptr)userbuff), ret, fail);
      // 检查用户缓冲区是否完全在内存分配块的范围内
      // 条件：基地址 + 块大小 < 用户缓冲区地址 + 缓冲区大小
      // 使用 uint64_t 进行类型转换，避免指针运算时的溢出问题
      // 如果此条件为真，表示用户缓冲区超出了内存分配块的范围，这是一种异常情况
      if ((uint64_t)baseAddr + baseSize < (uint64_t)userbuff + buffSize)
        // 跳转到 exit 标签，正常退出（不进行注册）
        // 这种情况下不进行 IPC 注册，但也不返回错误
        goto exit;
      // 调用内部 IPC 注册函数
      // ipcRegisterBuffer: 执行实际的 IPC 注册操作
      // comm: NCCL 通信上下文
      // userbuff: 用户缓冲区地址
      // buffSize: 缓冲区大小
      // peerRanks: 对等节点 rank 数组
      // nPeers: 对等节点数量
      // type: IPC 注册类型（P2P 或 COLLECTIVE）
      // regRecord: 内存注册记录
      // regBufFlag: 输出注册标志
      // offsetOut: 输出偏移量
      // peerRmtAddrsOut: 输出远端地址数组
      // NULL: 不需要输出 isLegacyIpc 标志（本地注册不需要此信息）
      NCCLCHECKGOTO(ipcRegisterBuffer(comm, userbuff, buffSize, peerRanks, nPeers, type, regRecord, regBufFlag, offsetOut, peerRmtAddrsOut, NULL), ret, fail);
    }
  }

// 正常退出标签
// 函数执行成功或参数无效时的退出点
exit:
  // 返回结果状态码
  return ret;
// 失败处理标签
// 发生错误时的退出点，进行清理工作
fail:
  // 清除注册标志（设置为未注册状态）
  *regBufFlag = 0;
  // 清除偏移量（设置为 0）
  *offsetOut = 0;
  // 清除远端地址数组指针（设置为 NULL）
  *peerRmtAddrsOut = NULL;
  // 跳转到正常退出标签，返回错误状态码
  goto exit;
}

// 结构体：IPC 清理回调
// 用于在 CUDA Graph 中清理 IPC 注册的缓冲区
struct ncclIpcCleanupCallback {
  struct ncclCommCallback base;                            // 基础回调结构
  struct ncclComm *comm;                                   // NCCL 通信上下文
  struct ncclReg *reg;                                     // 注册记录
};

// 函数实现：清理 IPC 注册
// 此函数清理 IPC 注册的缓冲区
// 参数 comm: NCCL 通信上下文
// 参数 cb: 回调结构
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
static ncclResult_t cleanupIpc(struct ncclComm* comm, struct ncclCommCallback* cb) {
  struct ncclIpcCleanupCallback* obj = (struct ncclIpcCleanupCallback*)cb; // 获取回调对象
  NCCLCHECK(ncclCommGraphDeregister(obj->comm, obj->reg)); // 注销图形注册
  free(obj);                                                // 释放对象
  return ncclSuccess;                                       // 返回成功
}

// 函数实现：在 CUDA Graph 中注册 IPC 缓冲区
// 此函数在 CUDA Graph 环境中注册用户缓冲区为 IPC 缓冲区
// 参数 comm: NCCL 通信上下文
// 参数 userbuff: 用户缓冲区指针
// 参数 buffSize: 缓冲区大小
// 参数 peerRanks: 对等节点的 rank 数组
// 参数 nPeers: 对等节点数量
// 参数 type: IPC 注册类型
// 参数 regBufFlag: 输出参数，指示是否已注册
// 参数 offsetOut: 输出参数，偏移量
// 参数 peerRmtAddrsOut: 输出参数，对等节点的远端地址数组
// 参数 cleanupQueuePtr: 清理队列指针
// 参数 nCleanupQueueElts: 输出参数，清理队列元素数量
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
ncclResult_t ncclIpcGraphRegisterBuffer(ncclComm* comm, const void* userbuff, size_t buffSize, int* peerRanks, int nPeers, ncclIpcRegType type, int* regBufFlag, uintptr_t* offsetOut, uintptr_t** peerRmtAddrsOut, void* cleanupQueuePtr, int* nCleanupQueueElts) {
  ncclResult_t ret = ncclSuccess;                           // 返回值初始化
  void* baseAddr;                                           // 基地址
  size_t baseSize;                                          // 基地址大小
  struct ncclIntruQueue<struct ncclCommCallback, &ncclCommCallback::next>* cleanupQueue = reinterpret_cast<struct ncclIntruQueue<struct ncclCommCallback, &ncclCommCallback::next>*>(cleanupQueuePtr); // 清理队列
  bool isLegacyIpc = false;                                 // 传统 IPC 标志
  struct ncclReg *regRecord = NULL;                         // 注册记录

  *regBufFlag = 0;                                          // 初始化注册标志为 0
  *offsetOut = 0;                                           // 初始化偏移量为 0
  *peerRmtAddrsOut = NULL;                                  // 初始化远端地址数组为 NULL
  if (comm && userbuff && buffSize > 0 && nPeers > 0) {     // 检查参数有效性
    CUCHECKGOTO(cuMemGetAddressRange((CUdeviceptr*)&baseAddr, &baseSize, (CUdeviceptr)userbuff), ret, fail); // 获取地址范围
    if ((uint64_t)baseAddr + baseSize < (uint64_t)userbuff + buffSize) 
        goto exit; // 检查缓冲区是否在范围内
        
    NCCLCHECKGOTO(ncclCommGraphRegister(comm, baseAddr, baseSize, (void**)&regRecord), ret, fail); // 在图形中注册
    NCCLCHECKGOTO(ipcRegisterBuffer(comm, userbuff, buffSize, peerRanks, nPeers, type, regRecord, regBufFlag, offsetOut, peerRmtAddrsOut, &isLegacyIpc), ret, fail); // 调用 IPC 注册函数

    if (*regBufFlag) {                                      // 如果注册成功
      struct ncclIpcCleanupCallback* record;               // 清理回调记录
      NCCLCHECKGOTO(ncclCalloc(&record, 1), ret, fail);    // 分配记录
      record->base.fn = cleanupIpc;                        // 设置清理函数
      record->comm = comm;                                  // 保存通信上下文
      record->reg = regRecord;                              // 保存注册记录
      if (isLegacyIpc) {                                    // 如果是传统 IPC
        ncclIntruQueueEnqueue(&comm->legacyRegCleanupQueue, (struct ncclCommCallback*)record); // 加入传统清理队列
      } else {                                              // 非 traditional IPC
        ncclIntruQueueEnqueue(cleanupQueue, (struct ncclCommCallback*)record); // 加入清理队列
        if (nCleanupQueueElts)
            *nCleanupQueueElts += 1;     // 增加清理队列元素计数
      }
    } else {                                                // 注册失败
      NCCLCHECKGOTO(ncclCommGraphDeregister(comm, regRecord), ret, fail); // 注销图形注册
    }
  }

exit:                                                      // 正常退出标签
  // coverity[leaked_storage:FALSE] => normally, addrsRecord is added to the cleanupQueue
  // coverity 注释：通常情况下，addrsRecord 被添加到 cleanupQueue 中，不存在内存泄漏
  return ret;                                              // 返回结果
fail:                                                      // 失败处理标签
  *regBufFlag = 0;                                         // 清除注册标志
  *offsetOut = 0;                                          // 清除偏移量
  *peerRmtAddrsOut = NULL;                                 // 清除远端地址数组
  goto exit;                                               // 跳转到退出
}

// 函数实现：注销 IPC 缓冲区
// 此函数注销之前注册的 IPC 缓冲区
// 参数 comm: NCCL 通信上下文
// 参数 regInfo: IPC 注册信息
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
ncclResult_t ncclIpcDeregBuffer(struct ncclComm* comm, struct ncclIpcRegInfo* regInfo) {
  NCCLCHECK(ncclProxyCallBlocking(comm, regInfo->ipcProxyconn, ncclProxyMsgDeregister, &regInfo->impInfo, sizeof(struct ncclIpcImpInfo), NULL, 0)); // 调用代理注销
  INFO(NCCL_REG, "rank %d - IPC deregistered buffer %p peer %d ipc remote buffer %p", comm->rank, regInfo->baseAddr, regInfo->peerRank, regInfo->impInfo.rmtRegAddr); // 记录注销信息
  return ncclSuccess;                                       // 返回成功
}

// 函数实现：代理端注册 IPC 缓冲区
// 此函数在代理端注册远端的 IPC 缓冲区
// 参数 connection: 代理连接
// 参数 proxyState: 代理状态
// 参数 reqBuff: 请求缓冲区（包含 IPC 导出信息）
// 参数 reqSize: 请求大小
// 参数 respBuff: 响应缓冲区（返回注册地址）
// 参数 respSize: 响应大小
// 参数 done: 输出参数，指示是否完成
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
static ncclResult_t p2pProxyRegister(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState, void* reqBuff, int reqSize, void* respBuff, int respSize, int* done) {
  struct p2pIpcExpInfo* ipcExpInfo = (struct p2pIpcExpInfo*)reqBuff; // IPC 导出信息
  void* regAddr = NULL;                                     // 注册地址
  ncclResult_t ret = ncclSuccess;                           // 返回值初始化
  bool mapped = false;                                      // 映射标志
  bool imported = false;                                    // 导入标志
  CUmemGenericAllocationHandle handle;                      // cuMem 分配句柄

  assert(sizeof(struct p2pIpcExpInfo) == reqSize);          // 断言请求大小正确
  assert(sizeof(void*) == respSize);                        // 断言响应大小正确

  INFO(NCCL_REG, "Proxy rank %d register reqBuff %p size %ld offset %ld legacyIpcCap %d sameProcess %d", proxyState->tpRank, reqBuff, ipcExpInfo->size, ipcExpInfo->offset, ipcExpInfo->legacyIpcCap, connection->sameProcess); // 记录注册信息

  // request peer passes all necessary buffer info to import. The proxy thread would register
  // the buffer locally and return register addr back
  // 请求对等节点传递所有必要的缓冲区信息以导入。代理线程将在本地注册缓冲区并返回注册地址
  if (ipcExpInfo->legacyIpcCap) {                           // 如果是传统 IPC
    // legacy import
    // 传统导入方式
    CUDACHECKGOTO(cudaIpcOpenMemHandle(&regAddr, ipcExpInfo->ipcDesc.devIpc, cudaIpcMemLazyEnablePeerAccess), ret, fail); // 打开 IPC 内存句柄
    regAddr = (void*)((uintptr_t)regAddr + ipcExpInfo->offset); // 加上偏移量
  } else {                                                   // cuMem 导入
    // cuMem import
    // cuMem 导入方式
    if (connection->sameProcess) {                          // 如果代理和请求对等节点在同一进程
      // if proxy is same process as request peer, we just need to map the handle.
      // 如果代理和请求对等节点在同一进程，我们只需要映射句柄
      memcpy(&handle, &ipcExpInfo->ipcDesc.memHandle, sizeof(CUmemGenericAllocationHandle)); // 复制句柄
    } else {                                                 // 不同进程
      if (ncclCuMemHandleType == CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) { // 如果是 POSIX 文件描述符类型
        CUCHECKGOTO(cuMemImportFromShareableHandle(&handle, (void*)(uintptr_t)ipcExpInfo->impFd, ncclCuMemHandleType), ret, fail); // 从文件描述符导入
        SYSCHECKGOTO(close(ipcExpInfo->impFd), "close", ret, fail); // 关闭文件描述符
      } else {                                               // 其他类型（如 fabric handle）
        CUCHECKGOTO(cuMemImportFromShareableHandle(&handle, (void*)&ipcExpInfo->ipcDesc.cuDesc, ncclCuMemHandleType), ret, fail); // 从共享句柄导入
      }
    }
    imported = true;                                        // 设置导入标志
    CUCHECKGOTO(cuMemAddressReserve((CUdeviceptr*)&regAddr, ipcExpInfo->size, /* alignment */ 0, /* addr */ 0, /* flags */ 0), ret, fail); // 保留地址空间
    CUCHECKGOTO(cuMemMap((CUdeviceptr)regAddr, ipcExpInfo->size, /* offset */ 0, handle, /* flags */ 0), ret, fail); // 映射内存
    mapped = true;                                          // 设置映射标志
    // Allow access by the local GPU
    // 允许本地 GPU 访问
    CUmemAccessDesc accessDesc = {};                        // 访问描述符
    accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE; // 位置类型为设备
    accessDesc.location.id = proxyState->cudaDev;          // 设置目标 CUDA 设备
    accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE; // 设置读写权限
    CUCHECKGOTO(cuMemSetAccess((CUdeviceptr)regAddr, ipcExpInfo->size, &accessDesc, 1), ret, fail); // 设置访问权限
    regAddr = (void*)((uintptr_t)regAddr + ipcExpInfo->offset); // 加上偏移量
  }
  INFO(NCCL_REG, "Proxy rank %d register success regAddr %p size %ld offset %ld legacyIpcCap %d sameProcess %d", proxyState->tpRank, regAddr, ipcExpInfo->size, ipcExpInfo->offset, ipcExpInfo->legacyIpcCap, connection->sameProcess); // 记录成功信息

exit:                                                      // 正常退出标签
  memcpy(respBuff, (void*)&regAddr, sizeof(void*));        // 复制注册地址到响应缓冲区
  *done = 1;                                               // 标记为完成
  return ret;                                              // 返回结果
fail:                                                      // 失败处理标签
  if (!ipcExpInfo->legacyIpcCap) {                         // 如果不是传统 IPC
    if (mapped) 
        CUCHECK(cuMemUnmap((CUdeviceptr)regAddr, ipcExpInfo->size)); // 取消映射
    if (regAddr) 
        CUCHECK(cuMemAddressFree((CUdeviceptr)regAddr, ipcExpInfo->size)); // 释放地址空间
    if (imported) 
        CUCHECK(cuMemRelease(handle));           // 释放句柄
  }
  regAddr = NULL;                                          // 清空地址
  goto exit;                                               // 跳转到退出
}

// 函数实现：代理端注销 IPC 缓冲区
// 此函数在代理端注销之前注册的 IPC 缓冲区
// 参数 connection: 代理连接
// 参数 proxyState: 代理状态
// 参数 reqBuff: 请求缓冲区（包含 IPC 导入信息）
// 参数 reqSize: 请求大小
// 参数 done: 输出参数，指示是否完成
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
static ncclResult_t p2pProxyDeregister(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState, void* reqBuff, int reqSize, int* done) {
  ncclResult_t ret = ncclSuccess;                           // 返回值初始化
  struct ncclIpcImpInfo* ipcInfo = (struct ncclIpcImpInfo*)reqBuff; // IPC 导入信息
  assert(sizeof(struct ncclIpcImpInfo) == reqSize);         // 断言请求大小正确

  if (ipcInfo->legacyIpcCap) {                             // 如果是传统 IPC
    CUDACHECKGOTO(cudaIpcCloseMemHandle((void*)((uintptr_t)ipcInfo->rmtRegAddr - ipcInfo->offset)), ret, fail); // 关闭 IPC 内存句柄（减去偏移量）
  } else {                                                   // cuMem
    if (connection->sameProcess) {                          // 如果同一进程
      NCCLCHECKGOTO(ncclCuMemFreeAddr((void*)((uintptr_t)ipcInfo->rmtRegAddr - ipcInfo->offset)), ret, fail); // 释放 cuMem 地址（减去偏移量）
    } else {                                                 // 不同进程
      NCCLCHECKGOTO(ncclCudaFree((void*)((uintptr_t)ipcInfo->rmtRegAddr - ipcInfo->offset)), ret, fail); // 释放 CUDA 内存（减去偏移量）
    }
  }

exit:                                                      // 正常退出标签
  *done = 1;                                               // 标记为完成
  return ret;                                              // 返回结果
fail:                                                      // 失败处理标签
  goto exit;                                               // 跳转到退出
}

// P2P 传输层结构体
// 定义了 P2P 传输层的所有函数指针和配置
struct ncclTransport p2pTransport = {
  "P2P",                                                   // 传输层名称
  p2pCanConnect,                                           // 检查是否可以建立 P2P 连接
  { p2pSendSetup, p2pSendConnect, p2pSendFree, NULL, p2pSendProxySetup, NULL, p2pSendProxyFree, NULL, p2pProxyRegister, p2pProxyDeregister }, // 发送端操作函数集合
  { p2pRecvSetup, p2pRecvConnect, p2pRecvFree, NULL, p2pRecvProxySetup, NULL, p2pRecvProxyFree, NULL, p2pProxyRegister, p2pProxyDeregister }  // 接收端操作函数集合
};

// 函数实现：初始化 CE（Copy Engine）操作
// 此函数初始化使用 cudaMemcpy 的 P2P 操作
static void initCeOperation() {
  static int init = 0;                                      // 初始化标志
  if (!init) {                                              // 如果未初始化
    useMemcpy = ncclParamP2pUseCudaMemcpy();              // 获取环境变量，决定是否使用 cudaMemcpy
    if (useMemcpy) {                                        // 如果使用 cudaMemcpy
      p2pTransport.send.proxyConnect = p2pSendProxyConnect; // 设置代理连接函数
      p2pTransport.send.proxyProgress = p2pSendProxyProgress; // 设置代理进度函数
    }
    init = 1;                                               // 标记为已初始化
  }
}
