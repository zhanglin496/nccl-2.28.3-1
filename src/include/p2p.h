/*************************************************************************
 * Copyright (c) 2015-2020, NVIDIA CORPORATION. All rights reserved.
 * 版权所有 (c) 2015-2020, NVIDIA CORPORATION. 保留所有权利。
 *
 * See LICENSE.txt for license information
 * 请参阅 LICENSE.txt 获取许可证信息
 ************************************************************************/

// 引入 C 标准库头文件，提供内存分配、进程控制等通用功能
#include <stdlib.h>

// 防止头文件重复包含的保护宏开始
// 如果未定义 NCCL_P2P_H_ 宏，则定义它
#ifndef NCCL_P2P_H_
#define NCCL_P2P_H_

// 引入 CUDA 驱动 API 头文件，提供底层 CUDA 驱动接口
#include <cuda.h>
// 引入 CUDA 运行时 API 头文件，提供高层 CUDA 运行时接口
#include <cuda_runtime.h>

// 引入 NCCL 核心头文件，包含基础类型定义和常量
#include "core.h"

// 检查 CUDA 运行时版本是否低于 12.3
#if CUDART_VERSION < 12030
// MNNVL: FABRIC handle support lifted from CUDA 12.3
// MNNVL: 从 CUDA 12.3 移植的 FABRIC 句柄支持
// MNNVL (Multi-Node NVLink) 是 NVIDIA 的多节点 NVLink 互连技术
// FABRIC 句柄用于在不同节点间共享 GPU 内存
// 定义设备属性：是否支持 FABRIC 句柄类型
#define CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED ((CUdevice_attribute)128)
// 定义内存句柄类型：FABRIC 类型（值为 0x8）
#define CU_MEM_HANDLE_TYPE_FABRIC ((CUmemAllocationHandleType)0x8ULL)
// 定义 IPC 句柄大小为 64 字节
#define CU_IPC_HANDLE_SIZE 64
// 定义 FABRIC 句柄结构体（v1 版本）
// 用于存储 FABRIC 类型的内存句柄数据
typedef struct CUmemFabricHandle_st {
    unsigned char data[CU_IPC_HANDLE_SIZE];                // 64 字节的句柄数据
} CUmemFabricHandle_v1;
// 将 v1 版本的 FABRIC 句柄类型别名为标准类型名
typedef CUmemFabricHandle_v1 CUmemFabricHandle;
// 结束 CUDA 版本条件编译
#endif

// 联合体：cuMem 描述符
// 用于存储不同类型的 cuMem 句柄
typedef union {
  uint64_t data;                                           // 数据字段（64 位整数）
  // Needs to hold a CUmemGenericAllocationHandle for UDS fd support
  // 需要能够存储 CUmemGenericAllocationHandle 以支持 UDS 文件描述符
  // UDS (Unix Domain Socket) 用于进程间传递文件描述符
  CUmemFabricHandle handle;                               // FABRIC 句柄（用于跨节点内存共享）
} ncclCuDesc;

// 联合体：IPC 描述符
// 用于存储跨进程内存共享所需的描述信息
typedef union {
  // Legacy CUDA IPC
  // 传统 CUDA IPC（进程间通信）方式
  cudaIpcMemHandle_t devIpc;                              // CUDA IPC 内存句柄
  
  // cuMem API support
  // cuMem API 支持（CUDA 11.3+ 的新版内存管理 API）
  struct {
    ncclCuDesc cuDesc;                                    // cuMem 描述符
    CUmemGenericAllocationHandle memHandle;               // cuMem 通用分配句柄
  };
} ncclIpcDesc;

// 枚举：IPC 注册类型
// 定义不同的 IPC 注册使用场景
enum ncclIpcRegType {
  NCCL_IPC_SENDRECV = 0,                                  // 点对点发送/接收模式
  NCCL_IPC_COLLECTIVE = 1                                 // 集合通信模式（如 AllReduce）
};

// 结构体：IPC 导入信息
// 存储从远端导入的 IPC 内存相关信息
struct ncclIpcImpInfo {
  void* rmtRegAddr;                                       // 远端注册地址（导入后的本地映射地址）
  bool legacyIpcCap;                                      // 是否支持传统 IPC 能力
  uintptr_t offset;                                       // 偏移量（相对于内存基地址）
};

// 结构体：IPC 注册信息
// 存储与远端 peer 的 IPC 注册相关信息
struct ncclIpcRegInfo {
  int peerRank;                                           // 远端 peer 的 rank 编号
  void* baseAddr;                                         // 内存基地址
  struct ncclProxyConnector* ipcProxyconn;               // IPC 代理连接器（用于代理模式）
  struct ncclIpcImpInfo impInfo;                          // IPC 导入信息
};

// 函数声明：分配可共享的 P2P 缓冲区
// 此函数分配一个可以在进程间共享的 GPU 内存缓冲区，并生成 IPC 描述符
// 参数 size: 缓冲区大小（字节数）
// 参数 directMap: 直接映射标志（控制内存分配方式）
// 参数 ipcDesc: 输出参数，接收生成的 IPC 描述符
// 参数 ptr: 输出参数，接收分配的内存指针
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
ncclResult_t ncclP2pAllocateShareableBuffer(size_t size, int directMap, ncclIpcDesc *ipcDesc, void **ptr);

// 函数声明：释放可共享的 P2P 缓冲区
// 此函数释放之前分配的可共享缓冲区
// 参数 ipcDesc: IPC 描述符指针（标识要释放的缓冲区）
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
ncclResult_t ncclP2pFreeShareableBuffer(ncclIpcDesc *ipcDesc);

// 函数声明：导入远端的可共享缓冲区
// 此函数将远端进程的共享缓冲区导入到本地进程的地址空间
// 参数 comm: NCCL 通信器指针
// 参数 peer: 远端 peer 的 rank 编号
// 参数 size: 缓冲区大小（字节数）
// 参数 ipcDesc: IPC 描述符指针（包含远端缓冲区的描述信息）
// 参数 devMemPtr: 输出参数，接收导入后的设备内存指针
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
ncclResult_t ncclP2pImportShareableBuffer(struct ncclComm *comm, int peer, size_t size, ncclIpcDesc *ipcDesc, void **devMemPtr);

// 函数声明：本地注册 IPC 缓冲区
// 此函数将用户提供的缓冲区注册为可跨进程共享的 IPC 内存
// 参数 comm: NCCL 通信器指针
// 参数 userbuff: 用户缓冲区指针（要注册的内存地址）
// 参数 buffSize: 缓冲区大小（字节数）
// 参数 peerRanks: peer rank 数组（要与之共享的 peer 列表）
// 参数 nPeers: peer 数量
// 参数 type: IPC 注册类型（SENDRECV 或 COLLECTIVE）
// 参数 regBufFlag: 输出参数，缓冲区是否已注册的标志
// 参数 offsetOut: 输出参数，相对于基地址的偏移量
// 参数 peerRmtAddrsOut: 输出参数，远端地址数组指针
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
ncclResult_t ncclIpcLocalRegisterBuffer(ncclComm* comm, const void* userbuff, size_t buffSize, int* peerRanks, int nPeers, ncclIpcRegType type, int* regBufFlag, uintptr_t* offsetOut, uintptr_t** peerRmtAddrsOut);

// 函数声明：在 CUDA Graph 中注册 IPC 缓冲区
// 此函数用于 CUDA Graph 环境，注册 IPC 缓冲区并添加清理回调
// 参数 comm: NCCL 通信器指针
// 参数 userbuff: 用户缓冲区指针（要注册的内存地址）
// 参数 buffSize: 缓冲区大小（字节数）
// 参数 peerRanks: peer rank 数组（要与之共享的 peer 列表）
// 参数 nPeers: peer 数量
// 参数 type: IPC 注册类型（SENDRECV 或 COLLECTIVE）
// 参数 regBufFlag: 输出参数，缓冲区是否已注册的标志
// 参数 offsetOut: 输出参数，相对于基地址的偏移量
// 参数 peerRmtAddrsOut: 输出参数，远端地址数组指针
// 参数 cleanupQueuePtr: 清理队列指针（用于 CUDA Graph 清理）
// 参数 nCleanupQueueElts: 输出参数，清理队列元素数量
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
ncclResult_t ncclIpcGraphRegisterBuffer(ncclComm* comm, const void* userbuff, size_t buffSize, int* peerRanks, int nPeers, ncclIpcRegType type, int* regBufFlag, uintptr_t* offsetOut, uintptr_t** peerRmtAddrsOut, void* cleanupQueuePtr, int* nCleanupQueueElts);

// 函数声明：注销 IPC 缓冲区
// 此函数注销之前注册的 IPC 缓冲区，释放相关资源
// 参数 comm: NCCL 通信器指针
// 参数 regInfo: IPC 注册信息指针（包含要注销的缓冲区信息）
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
ncclResult_t ncclIpcDeregBuffer(struct ncclComm* comm, struct ncclIpcRegInfo* regInfo);

// 头文件保护结束宏
// 与开头的 #ifndef NCCL_P2P_H_ 配对，防止头文件重复包含
#endif
