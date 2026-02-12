/*************************************************************************
 * Copyright (c) 2016-2024, NVIDIA CORPORATION. All rights reserved.
 * 版权所有 (c) 2016-2024，NVIDIA 公司。保留所有权利。
 *
 * See LICENSE.txt for license information
 * 详见 LICENSE.txt 获取许可证信息
 ************************************************************************/
// 头文件保护宏开始：防止头文件被重复包含
#ifndef NCCL_SHM_H_
// 定义头文件保护宏
#define NCCL_SHM_H_

// 包含 NCCL 通信器头文件，定义通信相关的数据结构和常量
#include "comm.h"

// 传统 IPC 共享内存结构体（不使用 CUDA IPC）
// 功能：封装传统 IPC（进程间通信）共享内存的标识符
// 用于在不支持 CUDA IPC 的平台或需要使用传统 IPC 机制
struct shmLegacyIpc {
    // 共享内存后缀字符串（用于构建共享内存名称）
    char shmSuffix[7];
    // 共享内存句柄（用于标识共享内存区域）
    ncclShmHandle_t handle;
    // 共享内存大小（字节数）
    size_t shmSize;
};

// CUDA IPC 共享内存结构体（使用 CUDA IPC）
// 功能：封装 CUDA IPC（进程间通信）共享内存的标识符
// 优势：相比传统 IPC，提供更好的性能和更简单的接口
struct shmCuIpc {
    // 联合体：可以存储不同类型的共享内存句柄
    union {
        // Fabric 句柄类型（用于 NVIDIA 专有互联技术）
        CUmemFabricHandle handle;
        // 数据句柄类型（包含多个分配属性）
        CUmemGenericAllocationHandle data;
    };
    // 映射后的设备端指针（指向 GPU 可访问的内存地址）
    void *ptr;
    // 共享内存大小（字节数）
    size_t size;
};

// 共享内存 IPC 描述符（统一的共享内存描述格式）
// 功能：支持传统和 CUDA IPC 两种模式
// 使用联合体在柄类型上实现类型切换
struct shmIpcDesc {
    // 联合体：根据 legacy 标志选择不同的 IPC 类型
    union {
        // 传统 IPC 结构（不使用 CUDA IPC）
        struct shmLegacyIpc shmli;
        // CUDA IPC 结构（使用 CUDA IPC）
        struct shmCuIpc shmci;
    } u;
    // 传统标志：true 表示使用传统 IPC，false 表示使用 CUDA IPC
    bool legacy;
};

// 类型定义：共享内存 IPC 描述符类型
// 使用 typedef 简化类型名称，提高代码可读性
typedef struct shmIpcDesc ncclShmIpcDesc_t;

// 外部函数声明：分配可共享的缓冲区
// 功能：分配一个可以被其他进程共享的缓冲区
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
// 参数说明：
//   size: 要分配的缓冲区大小
//   legacy: 是否使用传统 IPC（true）或 CUDA IPC（false）
//   descOut: 输出参数，返回共享内存描述符
//   hptr: 输出参数，返回主机端指针
//   dptr: 输出参数，返回设备端指针
ncclResult_t ncclShmAllocateShareableBuffer(size_t size, bool legacy, ncclShmIpcDesc_t *descOut, void **hptr, void **dptr);

// 外部函数声明：导入可共享的缓冲区
// 功能：导入其他进程分配的可共享缓冲区
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
// 参数说明：
//   comm: NCCL 通信器指针
//   proxyRank: 代理进程的排名（用于标识要导入哪个进程的缓冲区）
//   desc: 输入参数，包含共享内存描述符
//   hptr: 输出参数，返回主机端指针
//   dptr: 输出参数，返回设备端指针
//   descOut: 输出参数，返回导出后的共享内存描述符
ncclResult_t ncclShmImportShareableBuffer(struct ncclComm *comm, int proxyRank, ncclShmIpcDesc_t *desc, void **hptr, void **dptr, ncclShmIpcDesc_t *descOut);

// 外部函数声明：关闭共享内存 IPC 描述符
// 功能：释放共享内存资源，关闭 IPC 句柄
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
// 参数说明：
//   desc: 共享内存 IPC 描述符指针
ncclResult_t ncclShmIpcClose(ncclShmIpcDesc_t *desc);

// 条件编译结束：结束头文件保护宏的条件编译块
#endif
