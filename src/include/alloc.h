/*************************************************************************
 * Copyright (c) 2019-2022, NVIDIA CORPORATION. All rights reserved.
 * 版权所有 (c) 2019-2022，NVIDIA 公司。保留所有权利。
 *
 * See LICENSE.txt for license information
 * 详见 LICENSE.txt 获取许可证信息
 ************************************************************************/
// 头文件保护宏开始：防止头文件被重复包含
#ifndef NCCL_ALLOC_H_
// 定义头文件保护宏
#define NCCL_ALLOC_H_
// 包含 NCCL 公共头文件，定义了 NCCL 的基本数据类型和枚举
#include "nccl.h"
// 包含检查相关头文件，提供各种断言和错误检查宏
#include "checks.h"
// 包含位操作头文件，提供各种位操作辅助函数
#include "bitops.h"
// 包含工具函数头文件，提供各种通用工具函数
#include "utils.h"
// 包含 P2P 头文件，定义点对点通信相关结构和函数
#include "p2p.h"
// 包含系统内存管理头文件，提供内存分配相关接口
#include <sys/mman.h>
// 包含 C 标准库的 unistd 头文件，提供 POSIX 操作系统 API
#include <unistd.h>
// 包含 C 标准库的 stdlib 头文件，提供内存分配、进程控制等函数
#include <stdlib.h>
// 包含 C 标准库的 string 头文件，提供字符串操作函数
#include <string.h>

// 声明 clockNano 函数：返回纳秒级的时间戳
// from utils.h with which we have a circular dependency
// 从 utils.h 引入，存在循环依赖关系
uint64_t clockNano();

// 编译时类型大小计算模板：返回给定类型的大小
// 用于模板元编程中获取类型的字节大小
template<typename T>
constexpr size_t ncclSizeOfT() { return sizeof(T); }

// 编译时类型大小计算模板特化：void 类型特化版本
// 用于处理 void 类型的大小计算，返回 1 字节
template<>
constexpr size_t ncclSizeOfT<void>() { return 1; }

// 条件编译：如果 CUDA 运行时版本 >= 12020（CUDA 12.0 及以上）
// 检查是否支持 CUDA 12.0 及以上版本的 CU Memory API
#if CUDART_VERSION >= 12020
  // 包含 CUDA 驱动 API 头文件，提供 CUDA Driver API 的接口
  #include <cuda.h>
  // 包含 CUDA DA wrapper 头文件，提供 CUDA Driver API 的 C++ 封装
  #include "cudawrap.h"
  // CU Memory Host 分配函数（使用 CU Memory API）
  // 功能：在主机端分配通过 CU Memory API 管理的设备内存，GPU也可以访问，
  // 参数说明：
  //   ptr: 二级指针，用于返回分配的设备内存指针
  //   handlep: 指针，用于返回 CU 内存句柄
  //   size: 要分配的内存大小（字节数）
  // 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
  static inline ncclResult_t ncclCuMemHostAlloc(void** ptr, CUmemGenericAllocationHandle *handlep, size_t size) {
    // 初始化返回值为成功
    ncclResult_t result = ncclSuccess;
    // 声明并初始化内存分配粒度为 0
    // 粒度：内存分配的最小单位，用于对齐以提高性能
    size_t granularity = 0;
    // 声明当前 CUDA 设备变量
    CUdevice currentDev;
    // 声明 CU 内存分配属性结构体并清零初始化
    CUmemAllocationProp prop = {};
    // 声明 CU 内存通用分配句柄
    CUmemGenericAllocationHandle handle;
    // 声明 CUDA 设备变量并初始化为 -1
    int cudaDev;
    // 声明 CPU NUMA 节点 ID 变量并初始化为 -1
    // NUMA（Non-Uniform Memory Access）：非统一内存访问架构
    int cpuNumaNodeId = -1;
    // 声明 CU 内存句柄类型变量
    CUmemAllocationHandleType type = ncclCuMemHandleType;

    // 获取当前 CUDA 设备
    CUDACHECK(cudaGetDevice(&cudaDev));
    // 获取 CUDA 设备指针（通过设备索引获取）
    CUCHECK(cuDeviceGet(&currentDev, cudaDev));
    // 获取当前设备在系统中的 NUMA 节点 ID 属性
    // CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID：获取主机 NUMA 节点 ID
    CUCHECK(cuDeviceGetAttribute(&cpuNumaNodeId, CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID, currentDev));
    // 如果获取 NUMA 节点 ID 失败（返回值为 -1），则设置为 0
    if (cpuNumaNodeId < 0) 
        cpuNumaNodeId = 0;

    // 设置内存分配位置类型为 HOST NUMA（非统一内存访问）
    prop.location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;
    // 设置内存分配类型为固定内存（页锁定，防止被交换到磁盘）
    // 固定内存对于 DMA 操作很重要，可以确保内存驻留在物理内存中
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    // 设置请求的句柄类型，使得内存可以被导出（用于跨进程共享）
    // So it can be exported
    prop.requestedHandleTypes = type;

    // 设置分配的 NUMA 节点 ID，以便在正确的 NUMA 节点上分配内存
    prop.location.id = cpuNumaNodeId;

    // 查询内存分配的最小粒度
    // CU_MEM_ALLOC_GRANULARITY_MINIMUM：获取最小的分配粒度，用于内存对齐
    CUCHECK(cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
    // 向上对齐大小到页边界（granularity 通常是页大小）
    // ALIGN_SIZE 宏：将 size 向上对齐到 granularity 的倍数
    ALIGN_SIZE(size, granularity);

    /* 在设备上分配物理内存 */
    /* Allocate physical memory on device */
    // 调用 CU Memory API 创建内存分配
    // 参数说明：
    //   &handle: 输出参数，返回内存句柄
    //   size: 要分配的内存大小（已对齐）
    //   &prop: 内存分配属性（位置、类型等）
    //   0: 标志位，通常为 0
    CUCHECK(cuMemCreate(&handle, size, &prop, 0));

    /* 保留虚拟地址范围 */
    /* Reserve a virtual address range */
    // 在虚拟地址空间中预留一段地址范围
    // 参数说明：
    //   (CUdeviceptr*)ptr: 输出参数，转换后的设备指针
    //   size: 预留的地址范围大小（已对齐）
    //   granularity: 对齐粒度
    //   0: 标志位，通常为 0
    //   0: 标志位，通常为 0
    CUCHECK(cuMemAddressReserve((CUdeviceptr*)ptr, size, granularity, 0, 0));

    /* 将虚拟地址范围映射到物理分配 */
    /* Map virtual address range to physical allocation */
    // 将预留的虚拟地址范围映射到实际的物理内存分配
    // 参数说明：
    //   (CUdeviceptr)*ptr: 虚拟地址（强制转换为 CUdeviceptr*）
    //   size: 映射的内存大小
    //   0: 标志位，通常为 0
    //   handle: 内存句柄
    CUCHECK(cuMemMap((CUdeviceptr)*ptr, size, 0, handle, 0));

    /* 允许本地 GPU 对新映射的内存进行读写访问 */
    /* Now allow RW access to newly mapped memory for local GPU */
    // 设置内存访问描述符属性
    CUmemAccessDesc accessDesc = {};
    // 设置访问位置类型为设备
    accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    // 设置访问的设备 ID（指定哪个 GPU 可以访问）
    accessDesc.location.id = cudaDev;
    // 设置访问权限标志为读写模式
    accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    // 调用 CU Memory API 设置内存访问权限
    // 参数说明：
    //   (CUdeviceptr)*ptr: 设备指针
    //   size: 内存大小
    //   &accessDesc: 访问描述符
    //   1: 仅有一个设备（single device flag）
    CUCHECK(cuMemSetAccess((CUdeviceptr)*ptr, size, &accessDesc, 1));

    /* 允许本地 CPU 对新映射的内存进行读写访问 */
    /* Now allow RW access to newly mapped memory from CPU */
    // 修改访问描述符以允许 CPU 访问
    accessDesc.location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;
    // 设置访问的 NUMA 节点 ID
    accessDesc.location.id = cpuNumaNodeId;
    // 保持读写访问权限
    accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    // 调用 CU Memory API 设置 CPU 访问权限
    CUCHECK(cuMemSetAccess((CUdeviceptr)*ptr, size, &accessDesc, 1));

    /* 如果提供了句柄指针，则保存句柄 */
    /* if handle pointer provided, save handle */
    // 检查句柄指针参数是否非空
    if (handlep)
      // 保存返回的内存句柄到指针指向的位置
      *handlep = handle;

    // 输出调试信息：打印分配详情
    // 参数说明：
    //   size: 分配的内存大小
    //   *ptr: 返回的设备内存指针
    //   handle: 内存句柄（十六进制格式）
    //   cpuNumaNodeId: NUMA 节点 ID（十进制）
    //   cudaDev: CUDA 设备 ID
    //   granularity: 分配粒度
    INFO(NCCL_ALLOC, "CUMEM Host Alloc Size %zi pointer %p handle %llx numa %d dev %d granularity %ld", size, *ptr, handle, cpuNumaNodeId, cudaDev, granularity);
    // 返回操作结果
    return result;
  }

  // CU Memory Host 释放函数（使用 CU Memory API）
  // 功能：释放通过 ncclCuMemHostAlloc 分配的内存
  // 参数说明：
  //   ptr: 指向要释放的设备内存指针
  // 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
  static inline ncclResult_t ncclCuMemHostFree(void* ptr) {
    // 如果指针为空，直接返回成功
    if (ptr == NULL) return ncclSuccess;
    // 初始化返回值为成功
    ncclResult_t result = ncclSuccess;
    // 声明 CU 内存通用分配句柄
    CUmemGenericAllocationHandle handle;
    // 声明并初始化内存大小为 0
    size_t size = 0;
    // 从指针获取内存句柄（需要先调用 cuMemGetAllocationHandle）
    CUCHECK(cuMemRetainAllocationHandle(&handle, ptr));
    // 输出调试跟踪信息
    TRACE(NCCL_ALLOC, "CUMEM Host Free Size %zi pointer %p handle 0x%llx", size, ptr, handle);
    // 取消内存映射（解除虚拟地址到物理内存的映射）
    CUCHECK(cuMemUnmap((CUdeviceptr)ptr, size));
    // 释放内存地址预留（释放虚拟地址空间）
    CUCHECK(cuMemAddressFree((CUdeviceptr)ptr, size));
    // 释放 CU 内存句柄
    CUCHECK(cuMemRelease(handle));
    // 返回操作结果
    return result;
  }

#else /* CUDART_VERSION >= 12020 */
  // 如果 CUDA 运行时版本 < 12020（不支持 CU Memory API）
  // 输出警告信息：CUMEM Host 在 CUDA 12.0 之前不支持
  static inline ncclResult_t ncclCuMemHostAlloc(void** ptr, CUmemGenericAllocationHandle *handlep, size_t size) {
    WARN("CUMEM Host is not supported prior to CUDA 12.2");
    // 返回内部错误（表示不支持）
    return ncclInternalError;
  }

  // 如果 CUDA 运时版本 < 12020（不支持 CU Memory API）
  // 输出警告信息：CUMEM Host 在 CUDA 12.0 之前不支持
  static inline ncclResult_t ncclCuMemHostFree(void* ptr) {
    WARN("CUMEM Host is not supported prior to CUDA 12.2");
    // 返回内部错误（表示不支持）
    return ncclInternalError;
  }

#endif /* CUDART_VERSION >= 12020 */

// 调试模式下 CUDA Host 分配函数（带文件名和行号跟踪）
// 功能：在主机端分配内存，并记录分配位置的文件名和行号
// 参数说明：
//   ptr: 二级指针，用于返回分配的内存地址
//   nelem: 要分配的元素数量（不是字节数）
//   filefunc: 源文件名字符串，用于跟踪分配位置
//   line: 源代码行号，用于跟踪分配位置
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
template <typename T>
ncclResult_t ncclCudaHostCallocDebug(T** ptr, size_t nelem, const char *filefunc, int line) {
  // 初始化返回值为成功
  ncclResult_t result = ncclSuccess;
  // 声明 CUDA 流捕获模式变量（用于 CUDA Graph 捕获）
  cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed;
  // 声明指针并初始化为空
  *ptr = nullptr;
  // 交换当前线程的流捕获模式（为后续操作准备流环境）
  CUDACHECK(cudaThreadExchangeStreamCaptureMode(&mode));

  // 如果要分配的元素数量大于 0
  if (nelem > 0) {
    // 调用 CUDA 主机端分配函数
    // 参数说明：
    //   ptr: 输出参数，返回分配的内存指针
    //   nelem * sizeof(T): 计算总字节数
    //   cudaHostAllocMapped: 分配可被映射的固定内存
    //   result: 用于返回操作状态
    CUDACHECKGOTO(cudaHostAlloc(ptr, nelem*ncclSizeOfT<T>(), cudaHostAllocMapped), result, finish);
    // 清零分配的内存（初始化为全 0）
    memset(*ptr, 0, nelem*ncclSizeOfT<T>());
  }

finish:
  // 恢复线程的流捕获模式（交换回原始模式）
  CUDACHECK(cudaThreadExchangeStreamCaptureMode(&mode));
  // 如果分配失败（指针为空且元素数量大于 0）
  if (*ptr == nullptr && nelem > 0) 
    WARN("Failed to CUDA host alloc %ld bytes", nelem*ncclSizeOfT<T>());
  // 输出调试信息：记录分配详情
  INFO(NCCL_ALLOC, "%s:%d Cuda Host Alloc Size %ld pointer %p", filefunc, line, nelem*ncclSizeOfT<T>(), *ptr);
  // 返回操作结果
  return result;
}

// CUDA Host 释放函数（释放通过 ncclCudaHostCallocDebug 分配的内存）
// 功能：释放通过 CUDA API 分配的主机端内存
// 参数说明：
//   ptr: 指向要释放的内存指针
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
static inline ncclResult_t ncclCudaHostFree(void* ptr) {
  // 调用 CUDA 主机端释放函数
  CUDACHECK(cudaFreeHost(ptr));
  // 返回成功状态
  return ncclSuccess;
}

// 宏定义：调用调试版本的 CUDA Host 分配函数
// 参数：可变参数，传递类型、指针、元素数量、文件名和行号
#define ncclCudaHostCalloc(...) ncclCudaHostCallocDebug(__VA_ARGS__, __FILE__, __LINE__)

// 通用内存重新分配函数模板（调整已分配内存块的大小）
// 功能：重新分配内存，保留原数据，返回新的更大/更小的内存块
// 参数说明：
//   ptr: 二级指针，指向要调整的内存块指针
//   oldNelem: 原始元素数量
//   nelem: 新的元素数量
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
template <typename T>
ncclResult_t ncclRealloc(T** ptr, size_t oldNelem, size_t nelem) {
  // 获取旧指针内容（指向原始内存块）
  T* oldp = *ptr;

  // 参数验证：如果新数量小于旧数量，或旧指针为空但旧数量大于 0，则返回错误
  // 这防止无效的重新分配操作
  if (nelem < oldNelem || (oldp == NULL && oldNelem > 0)) return ncclInternalError;
  // 如果新旧数量相同，直接返回成功（无需操作）
  if (nelem == oldNelem) return ncclSuccess;

  // 分配新的内存块
  T* p = (T*)malloc(nelem*ncclSizeOfT<T>());
  // 如果分配失败，返回系统错误
  if (p == NULL) {
    WARN("Failed to malloc %ld bytes", nelem*ncclSizeOfT<T>());
    return ncclSystemError;
  }
  // 如果旧指针非空，需要复制旧数据到新内存
  if (oldp && oldNelem) {
    // 计算复制的数据量（取新旧数量的较小值）
    size_t copySize = oldNelem < nelem ? oldNelem : nelem;
    // 调用 memcpy 复制数据
    memcpy(p, oldp, copySize*ncclSizeOfT<T>());
    // 释放旧内存
    free(oldp);
  }
  // 清零新增部分的内存（从旧数量到新数量之间的部分）
  memset(p+oldNelem, 0, (nelem-oldNelem)*ncclSizeOfT<T>());
  // 更新指针指向新内存
  *ptr = (T*)p;
  // 输出调试信息：记录重新分配详情
  INFO(NCCL_ALLOC, "Mem Realloc old size %ld, new size %ld pointer %p", oldNelem*ncclSizeOfT<T>(), nelem*ncclSizeOfT<T>(), *ptr);
  // 返回成功状态
  return ncclSuccess;
}

// 条件编译：如果 CUDA 运行时版本 >= 11030（CUDA 11.0 及以上）
#if CUDART_VERSION >= 11030
  // 包含 CUDA Runtime API 头文件
  #include <cuda.h>
  // 包含 CUDA DA wrapper 头文件，提供 CUDA Driver API 的 C++ 封装
  #include "cudawrap.h"

  // 注释：ncclCuMemAllocAddr 接受内存句柄和大小，返回映射的地址指针
  // Comments: ncclCuMemAllocAddr takes memory handle and size and returns mapped address pointer
  // 功能：分配并映射 CU Memory 内存，返回可直接访问的设备指针
  // 参数说明：
  //   ptr: 二级指针，用于返回映射后的设备内存地址
  //   handleIn: 内存句柄指针，指向 CU 内存句柄
  //   size: 要映射的内存大小（字节数）
  // 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
  static inline ncclResult_t ncclCuMemAllocAddr(void **ptr, CUmemGenericAllocationHandle *handleIn, size_t size) {
    // 初始化返回值为成功
    ncclResult_t result = ncclSuccess;
    // 声明并初始化内存分配粒度为 0
    // 粒度：内存分配的最小单位，用于对齐以提高性能
    size_t granularity = 0;
    // 声明 CU 内存分配属性结构体并清零初始化
    CUmemAllocationProp prop = {};
    // 声明 CU 内存访问描述符结构体并清零初始化
    CUmemAccessDesc accessDesc = {};
    // 声明 CU 内存通用分配句柄
    CUmemGenericAllocationHandle handle;
    // 声明 CUDA 设备变量并初始化为 -1
    int cudaDev;

    // 获取当前 CUDA 设备
    CUDACHECK(cudaGetDevice(&cudaDev));
    // 获取 CUDA 设备指针（通过设备索引获取）
    CUCHECK(cuDeviceGet(&currentDev, cudaDev));
    // 设置内存分配类型为固定内存（页锁定，防止被交换到磁盘）
    // 固定内存对于 DMA 操作很重要，可以确保内存驻留在物理内存中
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    // 设置请求的句柄类型，使得内存可以被导出（用于跨进程共享）
    prop.requestedHandleTypes = ncclCuMemHandleType;
    // 设置分配的设备 ID（指定在哪个 GPU 上分配）
    prop.location.id = currentDev;

    /* 查询设备以查看是否支持 RDMA（GPUDirect RDMA）*/
    /* Query device to see if RDMA (GPUDirect RDMA) is supported */
    // 声明标志变量并初始化为 0
    int flag = 0;
    // 查询设备属性：GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED
    // 如果支持 GPU Direct RDMA with CUDA VMM，则设置标志为 1
    CUCHECK(cuDeviceGetAttribute(&flag, CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED, currentDev));
    // 如果支持 RDMA，则设置分配标志为 GPU Direct RDMA 可用
    if (flag) 
        prop.allocFlags.gpuDirectRDMACapable = 1;

    /* 获取内存分配粒度 */
    /* Get memory allocation granularity */
    // 查询内存分配的最小粒度
    // CU_MEM_ALLOC_GRANULARITY_MINIMUM：获取最小的分配粒度，用于内存对齐
    CUCHECK(cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
    // 向上对齐大小到页边界（granularity 通常是页大小）
    // ALIGN_SIZE 宏：将 size 向上对齐到 granularity 的倍数
    ALIGN_SIZE(size, granularity);

    /* 在设备上分配物理内存 */
    /* Allocate physical memory on device */
    // 调用 CU Memory API 创建内存分配
    // 参数说明：
    //   &handle: 输出参数，返回内存句柄
    //   size: 要分配的内存大小（已对齐）
    //   &prop: 内存分配属性（位置、类型、RDMA 支持等）
    //   0: 标志位，通常为 0
    CUCHECK(cuMemCreate(&handle, size, &prop, 0));

    /* 保留虚拟地址范围 */
    /* Reserve a virtual address range */
    // 在虚拟地址空间中预留一段地址范围
    // 参数说明：
    //   (CUdeviceptr*)ptr: 输出参数，转换后的设备指针
    //   size: 预留的地址范围大小（已对齐）
    //   granularity: 对齐粒度
    //   0: 标志位，通常为 0
    //   0: 标志位，通常为 0
    CUCHECK(cuMemAddressReserve((CUdeviceptr*)ptr, size, granularity, 0, 0));

    /* 将虚拟地址范围映射到物理分配 */
    /* Map virtual address range to physical allocation */
    // 将预留的虚拟地址范围映射到实际的物理内存分配
    // 参数说明：
    //   (CUdeviceptr)*ptr: 虚拟地址（强制转换为 CUdeviceptr*）
    //   size: 映射的内存大小
    //   0: 标志位，通常为 0
    //   handle: 内存句柄
    CUCHECK(cuMemMap((CUdeviceptr)*ptr, size, 0, handle, 0));

    /* 允许本地 GPU 对新映射的内存进行读写访问 */
    /* Now allow RW access to newly mapped memory for local GPU */
    // 设置内存访问描述符属性
    CUmemAccessDesc accessDesc = {};
    // 设置访问位置类型为设备
    accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    // 设置访问的设备 ID（指定哪个 GPU 可以访问）
    accessDesc.location.id = currentDev;
    // 设置访问权限标志为读写模式
    accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    // 调用 CU Memory API 设置内存访问权限
    // 参数说明：
    //   (CUdeviceptr)*ptr: 设备指针
    //   size: 内存大小
    //   &accessDesc: 访问描述符
    //   1: 仅有一个设备（single device flag）
    CUCHECK(cuMemSetAccess((CUdeviceptr)*ptr, size, &accessDesc, 1));

    /* 如果提供了句柄指针，则保存句柄 */
    /* if handle pointer provided, save handle */
    // 检查句柄指针参数是否非空
    if (handleIn)
      // 保存返回的内存句柄到指针指向的位置
      *handleIn = handle;

    // 输出调试信息：打印分配详情
    // 参数说明：
    //   size: 分配的内存大小
    //   *ptr: 返回的设备内存指针
    //   handle: 内存句柄（十六进制格式）
    //   currentDev: 当前 CUDA 设备 ID
    //   granularity: 分配粒度
    TRACE(NCCL_ALLOC, "CuMem Map Size %zu pointer %p handle %llx", size, *ptr, handle);
    // 返回操作结果
    return result;
  }

  // CU Memory 释放函数（释放通过 ncclCuMemAllocAddr 分配的内存）
  // 功能：释放通过 CU Memory API 分配并映射的设备内存
  // 参数说明：
  //   ptr: 指向要释放的设备内存指针
  // 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
  static inline ncclResult_t ncclCuMemFreeAddr(void *ptr) {
    // 如果指针为空，直接返回成功
    if (ptr == NULL) return ncclSuccess;
    // 初始化返回值为成功
    ncclResult_t result = ncclSuccess;
    // 声明 CU 内存通用分配句柄
    CUmemGenericAllocationHandle handle;
    // 声明并初始化内存大小为 0
    size_t size = 0;
    // 从指针获取内存句柄（需要先调用 cuMemGetAllocationHandle）
    CUCHECK(cuMemRetainAllocationHandle(&handle, ptr));
    // 输出调试跟踪信息
    TRACE(NCCL_ALLOC, "CuMem Unmap Size %zu pointer %p handle 0x%llx", size, ptr, handle);
    // 取消内存映射（解除虚拟地址到物理内存的映射）
    CUCHECK(cuMemUnmap((CUdeviceptr)ptr, size));
    // 释放内存地址预留（释放虚拟地址空间）
    CUCHECK(cuMemAddressFree((CUdeviceptr)ptr, size));
    // 释放 CU 内存句柄
    CUCHECK(cuMemRelease(handle));
    // 返回操作结果
    return result;
  }

  // CU Memory 分配函数（使用 CU Memory API，不返回映射地址）
  // 功能：分配 CU Memory 管理的设备内存，但不返回映射地址
  // 参数说明：
  //   ptr: 二级指针，用于返回分配的内存指针
  //   handlep: 内存句柄指针，用于返回 CU 内存句柄
  //   size: 要分配的内存大小（字节数）
  // 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
  static inline ncclResult_t ncclCuMemAlloc(void **ptr, CUmemGenericAllocationHandle *handlep, CUmemAllocationHandleType type, size_t size) {
    // 初始化返回值为成功
    ncclResult_t result = ncclSuccess;
    // 声明并初始化内存分配粒度为 0
    size_t granularity = 0;
    // 声明 CU 内存分配属性结构体并清零初始化
    CUmemAllocationProp prop = {};
    // 声明 CU 内存访问描述符结构体并清零初始化
    CUmemAccessDesc accessDesc = {};
    // 声明 CU 内存通用分配句柄
    CUmemGenericAllocationHandle handle;
    // 声明 CUDA 设备变量并初始化为 -1
    int cudaDev;
    // 声明标志变量并初始化为 0
    int flag = 0;

    // 获取当前 CUDA 设备
    CUDACHECK(cudaGetDevice(&cudaDev));
    // 获取 CUDA 设备指针（通过设备索引获取）
    CUCHECK(cuDeviceGet(&currentDev, cudaDev));
    // 设置内存分配类型为固定内存（页锁定，防止被交换到磁盘）
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    // 设置请求的句柄类型，可以是文件描述符或 fabric handle
    prop.requestedHandleTypes = type;
    // 设置分配的设备 ID（指定在哪个 GPU 上分配）
    prop.location.id = currentDev;

    /* 查询设备以查看是否支持 RDMA（GPUDirect RDMA）*/
    /* Query device to see if RDMA (GPUDirect RDMA) is supported */
    // 查询设备属性：GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED
    // 如果支持 GPU Direct RDMA with CUDA VMM，则设置标志为 1
    CUCHECK(cuDeviceGetAttribute(&flag, CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED, currentDev));
    // 如果支持 RDMA，则设置分配标志为 GPU Direct RDMA 可用
    if (flag)
        prop.allocFlags.gpuDirectRDMACapable = 1;

    /* 获取内存分配粒度 */
    /* Get memory allocation granularity */
    // 查询内存分配的最小粒度
    // CU_MEM_ALLOC_GRANULARITY_MINIMUM：获取最小的分配粒度，用于内存对齐
    CUCHECK(cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
    // 向上对齐大小到页边界（granularity 通常是页大小）
    // ALIGN_SIZE 宏：将 size 向上对齐到 granularity 的倍数
    ALIGN_SIZE(size, granularity);

    /* 在设备上分配物理内存 */
    /* Allocate physical memory on device */
    // 调用 CU Memory API 创建内存分配
    // 参数说明：
    //   &handle: 输出参数，返回内存句柄
    //   size: 要分配的内存大小（已对齐）
    //   &prop: 内存分配属性（位置、类型、RDMA 支持等）
    //   0: 标志位，通常为 0
    CUCHECK(cuMemCreate(&handle, size, &prop, 0));

    /* 如果提供了句柄指针，则保存句柄 */
    /* if handle pointer provided, save handle */
    // 检查句柄指针参数是否非空
    if (handlep) 
        *handlep = handle;
    // 输出调试信息：打印分配详情
    TRACE(NCCL_ALLOC, "CuMem Alloc Size %zu pointer %p handle %llx", size, *ptr, handle);
    // 返回操作结果
    return result;
  }

  // CU Memory 释放函数（释放通过 ncclCuMemAlloc 分配的内存）
  // 功能：释放通过 CU Memory API 分配的设备内存
  // 参数说明：
  //   ptr: 指向要释放的设备内存指针
  // 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
  static inline ncclResult_t ncclCuMemFree(void *ptr) {
    // 如果指针为空，直接返回成功
    if (ptr == NULL) 
        return ncclSuccess;
    // 初始化返回值为成功
    ncclResult_t result = ncclSuccess;
    // 声明 CU 内存通用分配句柄
    CUmemGenericAllocationHandle handle;
    // 声明并初始化内存大小为 0
    size_t size = 0;
    // 从指针获取内存句柄（需要先调用 cuMemGetAllocationHandle）
    CUCHECK(cuMemRetainAllocationHandle(&handle, ptr));
    // 输出调试跟踪信息
    TRACE(NCCL_ALLOC, "CuMem Free Size %zu pointer %p handle 0x%llx", size, ptr, handle);
    // 取消内存映射（如果该内存被映射过）
    CUCHECK(cuMemGetAddressRange(NULL, &size, (CUdeviceptr)ptr));
    // 解除映射（通过 cuMemUnmap 释放虚拟地址到物理内存的映射）
    CUCHECK(cuMemUnmap((CUdeviceptr)ptr, size));
    // 释放内存地址预留（释放虚拟地址空间）
    CUCHECK(cuMemAddressFree((CUdeviceptr)ptr, size));
    // 释放 CU 内存句柄
    CUCHECK(cuMemRelease(handle));
    // 返回操作结果
    return result;
  }

#else /* CUDART_VERSION >= 11030 */
  // 如果 CUDA 运行时版本 < 11030（不支持 CU Memory API）
  // 输出警告信息：CUMEM 在 CUDA 11.3 之前不支持
  // 声明 CU Memory 不支持的函数（CUDA 11.3 之前版本）
  static inline ncclResult_t ncclCuMemAllocAddr(void **ptr, CUmemGenericAllocationHandle *handleIn, size_t size) {
    WARN("CUMEM not supported prior to CUDA 11.3");
    // 返回内部错误（表示不支持）
    return ncclInternalError;
  }

  // 如果 CUDA 运行时版本 < 11030（不支持 CU Memory API）
  // 输出警告信息：CUMEM 在 CUDA 11.3 之前不支持
  static inline ncclResult_t ncclCuMemFreeAddr(void *ptr) {
    WARN("CUMEM not supported prior to CUDA 11.3");
    // 返回内部错误（表示不支持）
    return ncclInternalError;
  }

#endif /* CUDART_VERSION >= 11030 */

// CU Memory 地址分配函数（CUDA 11.3+，带地址）
// 功能：分配 CU Memory 管理的设备内存并返回映射地址
// 参数说明：
//   ptr: 二级指针，用于返回分配的设备内存地址
//   handleIn: 内存句柄指针，指向 CU 内存句柄
//   size: 要分配的内存大小（字节数）
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
static inline ncclResult_t ncclCuMemAllocAddr(void **ptr, CUmemGenericAllocationHandle *handleIn, size_t size) {
  // 输出警告信息：CUMEM 在 CUDA 11.3 之前不支持
  WARN("CUMEM not supported prior to CUDA 11.3");
  // 返回内部错误（表示不支持）
  return ncclInternalError;
}

// CU Memory 地址释放函数（CUDA 11.3+）
// 功能：释放通过 CU Memory API 分配的设备内存
// 参数说明：
//   ptr: 指向要释放的设备内存地址
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
static inline ncclResult_t ncclCuMemFreeAddr(void *ptr) {
  // 输出警告信息：CUMEM 在 CUDA 11.3 之前不支持
  WARN("CUMEM not supported prior to CUDA 11.3");
  // 返回内部错误（表示不支持）
  return ncclInternalError;
}

#endif /* CUDART_VERSION >= 11030 */

// 外部函数声明：查询 CU Memory 是否启用
// 功能：外部函数，用于判断 CU Memory 功能是否可用
// 返回值：int 类型，非 0 表示启用，0 表示未启用
extern int ncclCuMemEnable();

// 调试模式下的 CUDA 设备内存分配函数（带文件名和行号跟踪）
// 功能：在设备端分配内存，使用传统 CUDA API
// 参数说明：
//   ptr: 二级指针，用于返回分配的设备内存地址
//   nelem: 要分配的元素数量（不是字节数）
//   filefunc: 源文件名字符串，用于跟踪分配位置
//   line: 源代码行号，用于跟踪分配位置
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
template <typename T>
ncclResult_t ncclCudaMallocDebug(T** ptr, size_t nelem, const char *filefunc, int line) {
  // 初始化返回值为成功
  ncclResult_t result = ncclSuccess;
  // 声明 CUDA 流捕获模式变量（用于 CUDA Graph 捕获）
  cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed;
  // 声明指针并初始化为空
  *ptr = nullptr;
  // 交换当前线程的流捕获模式（为后续操作准备流环境）
  CUDACHECK(cudaThreadExchangeStreamCaptureMode(&mode));

  // 如果 CU Memory API 启用，执行特殊分配逻辑
  if (ncclCuMemEnable()) {
    // Need a side stream so as not to interfere with graph capture.
    // 需要一个侧流，以免干扰 CUDA Graph 捕获
    cudaStream_t stream;
    // 创建非阻塞 CUDA 流（不与默认流同步）
    CUDACHECKGOTO(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), result, finish);
    // 如果 CU Memory 启用，使用 cuMem API 分配
    //   NULL: 第二个参数为 NULL，表示分配新内存
    //   ncclCuMemHandleType: CU Memory 句柄类型
    //   nelem * sizeof(T): 计算总字节数
    NCCLCHECKGOTO(ncclCuMemAlloc((void **)ptr, NULL, ncclCuMemHandleType, nelem*ncclSizeOfT<T>()), result, finish);
  } else {
    // 否则使用传统的 CUDA malloc API 分配内存
    // Need a side stream so as not to interfere with graph capture.
    // 需要一个侧流，以免干扰 CUDA Graph 捕获
    cudaStream_t stream;
    // 创建非阻塞 CUDA 流（不与默认流同步）
    CUDACHECKGOTO(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), result, finish);
    // 使用传统的 cudaMalloc 分配设备内存
    CUDACHECKGOTO(cudaMalloc(ptr, nelem*ncclSizeOfT<T>()), result, finish);
  }

finish:
  // 同步流以确保所有操作完成
  CUDACHECKGOTO(cudaStreamSynchronize(stream), result, finish);
  // 销毁 CUDA 流
  CUDACHECKGOTO(cudaStreamDestroy(stream), result, finish);

  // 如果分配失败（指针为空且元素数量大于 0）
  if (*ptr == nullptr && nelem > 0) WARN("Failed to CUDA malloc %ld bytes", nelem*ncclSizeOfT<T>());
  // 输出调试信息：记录分配详情
  INFO(NCCL_ALLOC, "%s:%d Cuda Alloc Size %ld pointer %p", filefunc, line, nelem*ncclSizeOfT<T>(), *ptr);
  // 返回操作结果
  return result;
}

// 宏定义：调用调试版本的 CUDA 设备内存分配函数
// 参数：可变参数，传递类型、指针、元素数量、文件名和行号
#define ncclCudaMalloc(...) ncclCudaMallocDebug(__VA_ARGS__, __FILE__, __LINE__)

// 调试模式下的 CUDA 设备内存分配并异步复制函数（带文件名和行号跟踪）
// 功能：在设备端分配内存并异步复制数据（使用 CUDA Stream）
// 参数说明：
//   dst: 目标缓冲区指针（设备内存）
//   src: 源数据指针（主机内存）
//   nelem: 要复制的元素数量
//   返回值：ncclResult_t 类型，表示操作成功或失败的状态码
template <typename T>
ncclResult_t ncclCudaMemcpy(T* dst, T* src, size_t nelem) {
  // 初始化返回值为成功
  ncclResult_t result = ncclSuccess;
  // 声明 CUDA 流捕获模式变量（用于 CUDA Graph 捕获）
  cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed;
  // 声明指针并初始化为空
  *ptr = nullptr;
  // 交换当前线程的流捕获模式（为后续操作准备流环境）
  CUDACHECK(cudaThreadExchangeStreamCaptureMode(&mode));

  // Need a side stream so as not to interfere with graph capture.
  // 需要一个侧流，以免干扰 CUDA Graph 捕获
  cudaStream_t stream;
  // 创建非阻塞 CUDA 流（不与默认流同步）
  CUDACHECKGOTO(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), result, finish);

  // 如果 CU Memory API 启用，使用特殊流和内存异步复制
  if (ncclCuMemEnable()) {
    // Need a side stream so as not to interfere with graph capture.
    // 需要一个侧流，以免干扰 CUDA Graph 捕获
    cudaStream_t stream2;
    // 创建非阻塞 CUDA 流（不与默认流同步）
    CUDACHECKGOTO(cudaStreamCreateWithFlags(&stream2, cudaStreamNonBlocking), result, finish2);
    // 调用 CUDA 内存异步设置和复制函数（使用 CU Memory）
    // 参数说明：
    //   *dst: 目标缓冲区指针
    //   0: 清零标志
    //   nelem * sizeof(T): 要复制的总字节数
    //   cudaMemcpyDeviceToDevice: 从主机到设备的复制方向
    //   stream2: 使用侧流进行异步操作
    CUDACHECKGOTO(cudaMemsetAsync(*dst, 0, nelem*ncclSizeOfT<T>(), cudaMemcpyDeviceToDevice, stream2), result, finish2);
  } else {
    // 否则使用标准 CUDA Runtime API 异步复制
    // 调用 CUDA 异步内存复制函数
    // 参数说明：
    //   dst: 目标缓冲区指针
    //   0: 清零标志
    //   nelem * sizeof(T): 要复制的总字节数
    //   cudaMemcpyDeviceToDevice: 从主机到设备的复制方向
    //   stream: 使用主 CUDA 流进行异步操作
    CUDACHECKGOTO(cudaMemcpyAsync(dst, 0, nelem*ncclSizeOfT<T>(), cudaMemcpyDeviceToDevice, stream), result, finish);
  }

finish:
  // 同步所有流以确保操作完成
  if (ncclCuMemEnable()) {
    // 如果使用了 CU Memory API，需要同步两个流
    CUDACHECKGOTO(cudaStreamSynchronize(stream), result, finish);
    CUDACHECKGOTO(cudaStreamSynchronize(stream2), result, finish);
    CUDACHECKGOTO(cudaStreamDestroy(stream), result, finish);
    CUDACHECKGOTO(cudaStreamDestroy(stream2), result, finish);
  } else {
    // 如果使用标准 CUDA API，只需要同步一个流
    CUDACHECKGOTO(cudaStreamSynchronize(stream), result, finish);
    CUDACHECKGOTO(cudaStreamDestroy(stream), result, finish);
  }

  // 返回操作结果
  return result;
}

// CUDA 设备内存释放函数
// 功能：释放通过 CUDA API 分配的设备内存
// 参数说明：
//   ptr: 指向要释放的设备内存指针
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
template <typename T>
ncclResult_t ncclCudaFree(T* ptr) {
  // 初始化返回值为成功
  ncclResult_t result = ncclSuccess;
  // 记录跟踪信息：打印释放的指针地址
  TRACE(NCCL_ALLOC, "Cuda Free pointer %p", ptr);
  // 检查 CU Memory API 是否启用
  if (ncclCuMemEnable()) {
    // 如果启用 CU Memory，使用 cuMemFree 释放内存
    NCCLCHECKGOTO(ncclCuMemFree((void *)ptr), result, finish);
  } else {
    // 否则使用标准 CUDA Runtime API 释放内存
    CUDACHECKGOTO(cudaFree(ptr), result, finish);
  }

finish:
  // 返回操作结果
  return result;
}

// 宏定义：调用 CUDA 设备内存分配函数
// 参数：可变参数，传递类型、指针、元素数量
#define ncclCudaMalloc(...) ncclCudaMallocDebug(__VA_ARGS__, __FILE__, __LINE__)

// 分配可能被 ibv_reg_mr'd 的内存
// 功能：分配内存时需要考虑 IB verbs 注册的内存页特性
// 说明：这些内存页将被标记为 MADVISE（避免被 fork）
//       并且如果它们是共享的，可能在子进程中导致崩溃
// Allocate memory to be potentially ibv_reg_mr'd. This needs to be
// allocated on separate pages as those pages will be marked MADVISE
// and if they are shared that could cause a crash in a child process.
inline ncclResult_t ncclIbMallocDebug(void** ptr, size_t size, const char *filefunc, int line) {
  // 如果要分配的大小大于 0
  if (size > 0) {
    // 获取系统页面大小（用于内存对齐）
    long page_size = sysconf(_SC_PAGESIZE);
    // 如果获取页面大小失败，返回系统错误
    if (page_size < 0) return ncclSystemError;
    // 声明指针变量并初始化为空
    void* p;
    // 计算对齐后的大小（向上对齐到页边界）
    int size_aligned = ROUNDUP(size, page_size);
    // 调用 posix_memalign 分配对齐的内存
    // 参数说明：
    //   &p: 输出参数，返回分配的内存指针
    //   page_size: 对齐边界（页大小）
    //   size_aligned: 对齐后的分配大小
    // 返回值：0 表示成功，非 0 表示失败
    int ret = posix_memalign(&p, page_size, size_aligned);
    // 如果分配失败，返回系统错误
    if (ret != 0) return ncclSystemError;
    // 清零分配的内存
    memset(p, 0, size);
    // 保存分配的指针到输出参数
    *ptr = p;
  } else {
    // 如果大小为 0，则将指针设置为空
    *ptr = NULL;
  }
  // 输出调试信息：记录分配详情
  INFO(NCCL_ALLOC, "%s:%d Ib Alloc Size %ld pointer %p", filefunc, line, size, *ptr);
  // 返回成功状态
  return ncclSuccess;
}

// 宏定义：调用调试版本的 IB 内存分配函数
// 参数：可变参数，传递类型、指针、大小、文件名和行号
#define ncclIbMalloc(...) ncclIbMallocDebug(__VA_ARGS__, __FILE__, __LINE__)

#endif /* CUDART_VERSION >= 11030 */

// 包含 CUDA Runtime API 头文件（用于传统 CUDA 内存管理）
// 条件编译：如果 CUDA 运行时版本 >= 11030
// 检查是否支持 CUDA 12.0 及以上版本的 CU Memory API
#if CUDART_VERSION >= 11030
  #include <cuda.h>
  #include "cudawrap.h"

// 注释：ncclCuMemAllocAddr 接受内存句柄和大小，返回映射的地址指针
// Comments: ncclCuMemAllocAddr takes memory handle and size and returns mapped address pointer
  // 功能：分配并映射 CU Memory 内存，返回可直接访问的设备指针
  // 参数说明：
  //   ptr: 二级指针，用于返回映射后的设备内存地址
  //   handleIn: 内存句柄指针，指向 CU 内存句柄
  //   size: 要映射的内存大小（字节数）
  // 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
  static inline ncclResult_t ncclCuMemAllocAddr(void **ptr, CUmemGenericAllocationHandle *handleIn, size_t size) {
    // 初始化返回值为成功
    ncclResult_t result = ncclSuccess;
    // 声明并初始化内存分配粒度为 0
    size_t granularity = 0;
    // 声明 CU 内存分配属性结构体并清零初始化
    CUmemAllocationProp prop = {};
    // 声明 CU 内存访问描述符结构体并清零初始化
    CUmemAccessDesc accessDesc = {};
    // 声明 CU 内存通用分配句柄
    CUmemGenericAllocationHandle handle;
    // 声明 CUDA 设备变量并初始化为 -1
    int cudaDev;

    // 获取当前 CUDA 设备
    CUDACHECK(cudaGetDevice(&cudaDev));
    // 获取 CUDA 设备指针（通过设备索引获取）
    CUCHECK(cuDeviceGet(&currentDev, cudaDev));
    // 设置内存分配类型为固定内存（页锁定，防止被交换到磁盘）
    // 固定内存对于 DMA 操作很重要，可以确保内存驻留在物理内存中
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    // 设置请求的句柄类型，使得内存可以被导出（用于跨进程共享）
    // So it can be exported
    prop.requestedHandleTypes = type;
    // 设置分配的设备 ID（指定在哪个 GPU 上分配）
    prop.location.id = currentDev;

    /* 查询设备以查看是否支持 RDMA（GPUDirect RDMA）*/
    /* Query device to see if RDMA (GPUDirect RDMA) is supported */
    // 声明标志变量并初始化为 0
    int flag = 0;
    // 查询设备属性：GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED
    // 如果支持 GPU Direct RDMA with CUDA VMM，则设置标志为 1
    CUCHECK(cuDeviceGetAttribute(&flag, CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED, currentDev));
    // 如果支持 RDMA，则设置分配标志为 GPU Direct RDMA 可用
    if (flag) 
        prop.allocFlags.gpuDirectRDMACapable = 1;

    /* 获取内存分配粒度 */
    /* Get memory allocation granularity */
    // 查询内存分配的最小粒度
    // CU_MEM_ALLOC_GRANULARITY_MINIMUM：获取最小的分配粒度，用于内存对齐
    CUCHECK(cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
    // 向上对齐大小到页边界（granularity 通常是页大小）
    // ALIGN_SIZE 宏：将 size 向上对齐到 granularity 的倍数
    ALIGN_SIZE(size, granularity);

    /* 在设备上分配物理内存 */
    /* Allocate physical memory on device */
    // 调用 CU Memory API 创建内存分配
    // 参数说明：
    //   &handle: 输出参数，返回内存句柄
    //   size: 要分配的内存大小（已对齐）
    //   &prop: 内存分配属性（位置、类型、RDMA 支持等）
    //   0: 标志位，通常为 0
    CUCHECK(cuMemCreate(&handle, size, &prop, 0));

    /* 保留虚拟地址范围 */
    /* Reserve a virtual address range */
    // 在虚拟地址空间中预留一段地址范围
    // 参数说明：
    //   (CUdeviceptr*)ptr: 输出参数，转换后的设备指针
    //   size: 预留的地址范围大小（已对齐）
    //   granularity: 对齐粒度
    //   0: 标志位，通常为 0
    //   0: 标志位，通常为 0
    CUCHECK(cuMemAddressReserve((CUdeviceptr*)ptr, size, granularity, 0, 0));

    /* 将虚拟地址范围映射到物理分配 */
    /* Map virtual address range to physical allocation */
    // 将预留的虚拟地址范围映射到实际的物理内存分配
    // 参数说明：
    //   (CUdeviceptr)*ptr: 虚拟地址（强制转换为 CUdeviceptr*）
    //   size: 映射的内存大小
    //   0: 标志位，通常为 0
    //   handle: 内存句柄
    CUCHECK(cuMemMap((CUdeviceptr)*ptr, size, 0, handle, 0));

    /* 允许本地 GPU 对新映射的内存进行读写访问 */
    /* Now allow RW access to newly mapped memory for local GPU */
    // 设置内存访问描述符属性
    CUmemAccessDesc accessDesc = {};
    // 设置访问位置类型为设备
    accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    // 设置访问的设备 ID（指定哪个 GPU 可以访问）
    accessDesc.location.id = currentDev;
    // 设置访问权限标志为读写模式
    accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    // 调用 CU Memory API 设置内存访问权限
    // 参数说明：
    //   (CUdeviceptr)*ptr: 设备指针
    //   size: 内存大小
    //   &accessDesc: 访问描述符
    //   1: 仅有一个设备（single device flag）
    CUCHECK(cuMemSetAccess((CUdeviceptr)*ptr, size, &accessDesc, 1));

    /* 如果提供了句柄指针，则保存句柄 */
    /* if handle pointer provided, save handle */
    // 检查句柄指针参数是否非空
    if (handleIn)
      // 保存返回的内存句柄到指针指向的位置
      *handleIn = handle;

    // 输出调试信息：打印分配详情
    // 参数说明：
    //   size: 分配的内存大小
    //   *ptr: 返回的设备内存指针
    //   handle: 内存句柄（十六进制格式）
    //   currentDev: 当前 CUDA 设备 ID
    //   granularity: 分配粒度
    TRACE(NCCL_ALLOC, "CuMem Map Size %zu pointer %p handle %llx", size, *ptr, handle);
    // 返回操作结果
    return result;
  }

  // CU Memory 释放函数（释放通过 ncclCuMemAllocAddr 分配的内存）
  // 功能：释放通过 CU Memory API 分配并映射的设备内存
  // 参数说明：
  //   ptr: 指向要释放的设备内存指针
  // 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
  static inline ncclResult_t ncclCuMemFreeAddr(void *ptr) {
    // 如果指针为空，直接返回成功
    if (ptr == NULL) return ncclSuccess;
    // 初始化返回值为成功
    ncclResult_t result = ncclSuccess;
    // 声明 CU 内存通用分配句柄
    CUmemGenericAllocationHandle handle;
    // 声明并初始化内存大小为 0
    size_t size = 0;
    // 从指针获取内存句柄（需要先调用 cuMemGetAllocationHandle）
    CUCHECK(cuMemRetainAllocationHandle(&handle, ptr));
    // 输出调试跟踪信息
    TRACE(NCCL_ALLOC, "CuMem Unmap Size %zu pointer %p handle 0x%llx", size, ptr, handle);
    // 取消内存映射（解除虚拟地址到物理内存的映射）
    CUCHECK(cuMemUnmap((CUdeviceptr)ptr, size));
    // 释放内存地址预留（释放虚拟地址空间）
    CUCHECK(cuMemAddressFree((CUdeviceptr)ptr, size));
    // 释放 CU 内存句柄
    CUCHECK(cuMemRelease(handle));
    // 返回操作结果
    return result;
  }

  // CU Memory 分配函数（使用 CU Memory API，不返回映射地址）
  // 功能：分配 CU Memory 管理的设备内存，但不返回映射地址
  // 参数说明：
  //   ptr: 二级指针，用于返回分配的内存指针
  //   handlep: 内存句柄指针，用于返回 CU 内存句柄
  //   size: 要分配的内存大小（字节数）
  // 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
  static inline ncclResult_t ncclCuMemAlloc(void **ptr, CUmemGenericAllocationHandle *handlep, CUmemAllocationHandleType type, size_t size) {
    // 初始化返回值为成功
    ncclResult_t result = ncclSuccess;
    // 声明并初始化内存分配粒度为 0
    size_t granularity = 0;
    // 声明 CU 内存分配属性结构体并清零初始化
    CUmemAllocationProp prop = {};
    // 声明 CU 内存访问描述符结构体并清零初始化
    CUmemAccessDesc accessDesc = {};
    // 声明 CU 内存通用分配句柄
    CUmemGenericAllocationHandle handle;
    // 声明 CUDA 设备变量并初始化为 -1
    int cudaDev;
    // 声明标志变量并初始化为 0
    int flag = 0;

    // 获取当前 CUDA 设备
    CUDACHECK(cudaGetDevice(&cudaDev));
    // 获取 CUDA 设备指针（通过设备索引获取）
    CUCHECK(cuDeviceGet(&currentDev, cudaDev));
    // 设置内存分配类型为固定内存（页锁定，防止被交换到磁盘）
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    // 设置请求的句柄类型，可以是文件描述符或 fabric handle
    prop.requestedHandleTypes = type;
    // 设置分配的设备 ID（指定在哪个 GPU 上分配）
    prop.location.id = currentDev;

    /* 查询设备以查看是否支持 RDMA（GPUDirect RDMA）*/
    /* Query device to see if RDMA (GPUDirect RDMA) is supported */
    // 查询设备属性：GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED
    // 如果支持 GPU Direct RDMA with CUDA VMM，则设置标志为 1
    CUCHECK(cuDeviceGetAttribute(&flag, CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED, currentDev));
    // 如果支持 RDMA，则设置分配标志为 GPU Direct RDMA 可用
    if (flag) prop.allocFlags.gpuDirectRDMACapable = 1;

    /* 获取内存分配粒度 */
    /* Get memory allocation granularity */
    // 查询内存分配的最小粒度
    // CU_MEM_ALLOC_GRANULARITY_MINIMUM：获取最小的分配粒度，用于内存对齐
    CUCHECK(cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
    // 向上对齐大小到页边界（granularity 通常是页大小）
    // ALIGN_SIZE 宏：将 size 向上对齐到 granularity 的倍数
    ALIGN_SIZE(size, granularity);

    /* 在设备上分配物理内存 */
    /* Allocate physical memory on device */
    // 调用 CU Memory API 创建内存分配
    // 参数说明：
    //   &handle: 输出参数，返回内存句柄
    //   size: 要分配的内存大小（已对齐）
    //   &prop: 内存分配属性（位置、类型、RDMA 支持等）
    //   0: 标志位，通常为 0
    CUCHECK(cuMemCreate(&handle, size, &prop, 0));

    /* 如果提供了句柄指针，则保存句柄 */
    /* if handle pointer provided, save handle */
    // 检查句柄指针参数是否非空
    if (handlep) *handlep = handle;
    // 输出调试信息：打印分配详情
    TRACE(NCCL_ALLOC, "CuMem Alloc Size %zu pointer %p handle %llx", size, *ptr, handle);
    // 返回操作结果
    return result;
  }

  // CU Memory 释放函数（释放通过 ncclCuMemAlloc 分配的内存）
  // 功能：释放通过 CU Memory API 分配的设备内存
  // 参数说明：
  //   ptr: 指向要释放的设备内存指针
  // 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
  static inline ncclResult_t ncclCuMemFree(void *ptr) {
    // 如果指针为空，直接返回成功
    if (ptr == NULL) return ncclSuccess;
    // 初始化返回值为成功
    ncclResult_t result = ncclSuccess;
    // 声明 CU 内存通用分配句柄
    CUmemGenericAllocationHandle handle;
    // 声明并初始化内存大小为 0
    size_t size = 0;
    // 从指针获取内存句柄（需要先调用 cuMemGetAllocationHandle）
    CUCHECK(cuMemRetainAllocationHandle(&handle, ptr));
    // 输出调试跟踪信息
    TRACE(NCCL_ALLOC, "CuMem Free Size %zu pointer %p handle 0x%llx", size, ptr, handle);
    // 取消内存映射（如果该内存被映射过）
    CUCHECK(cuMemGetAddressRange(NULL, &size, (CUdeviceptr)ptr));
    // 解除映射（通过 cuMemUnmap 释放虚拟地址到物理内存的映射）
    CUCHECK(cuMemUnmap((CUdeviceptr)ptr, size));
    // 释放内存地址预留（释放虚拟地址空间）
    CUCHECK(cuMemAddressFree((CUdeviceptr)ptr, size));
    // 释放 CU 内存句柄
    CUCHECK(cuMemRelease(handle));
    // 返回操作结果
    return result;
  }

#else /* CUDART_VERSION >= 11030 */
  // 如果 CUDA 运行时版本 < 11030（不支持 CU Memory API）
  // 输出警告信息：CUMEM 在 CUDA 11.3 之前不支持
  // 声明 CU Memory 不支持的函数（CUDA 11.3 之前版本）
  static inline ncclResult_t ncclCuMemAllocAddr(void **ptr, CUmemGenericAllocationHandle *handleIn, size_t size) {
    WARN("CUMEM not supported prior to CUDA 11.3");
    // 返回内部错误（表示不支持）
    return ncclInternalError;
  }

  // 如果 CUDA 运行时版本 < 11030（不支持 CU Memory API）
  // 输出警告信息：CUMEM 在 CUDA 11.3 之前不支持
  static inline ncclResult_t ncclCuMemAllocAddr(void **ptr, CUmemGenericAllocationHandle *handleIn, size_t size) {
    WARN("CUMEM not supported prior to CUDA 11.3");
    // 返回内部错误（表示不支持）
    return ncclInternalError;
  }

#endif /* CUDART_VERSION >= 11030 */

// 外部函数声明：查询 CU Memory 是否启用
// 功能：外部函数，用于判断 CU Memory 功能是否可用
// 返回值：int 类型，非 0 表示启用，0 表示未启用
extern int ncclCuMemEnable();

// 调试模式下的 CUDA 设备内存分配函数（带文件名和行号跟踪）
// 功能：在设备端分配内存，使用传统 CUDA API
// 参数说明：
//   ptr: 二级指针，用于返回分配的设备内存地址
//   nelem: 要分配的元素数量（不是字节数）
//   filefunc: 源文件名字符串，用于跟踪分配位置
//   line: 源代码行号，用于跟踪分配位置
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
template <typename T>
ncclResult_t ncclCudaMallocDebug(T** ptr, size_t nelem, const char *filefunc, int line) {
  // 初始化返回值为成功
  ncclResult_t result = ncclSuccess;
  // 声明 CUDA 流捕获模式变量（用于 CUDA Graph 捕获）
  cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed;
  // 声明指针并初始化为空
  *ptr = nullptr;
  // 交换当前线程的流捕获模式（为后续操作准备流环境）
  CUDACHECK(cudaThreadExchangeStreamCaptureMode(&mode));

  // 如果 CU Memory API 启用，执行特殊分配逻辑
  if (ncclCuMemEnable()) {
    // Need a side stream so as not to interfere with graph capture.
    // 需要一个侧流，以免干扰 CUDA Graph 捕获
    cudaStream_t stream;
    // 创建非阻塞 CUDA 流（不与默认流同步）
    CUDACHECKGOTO(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), result, finish);
    // 如果 CU Memory 启用，使用 cuMem API 分配
    //   NULL: 第二个参数为 NULL，表示分配新内存
    //   ncclCuMemHandleType: CU Memory 句柄类型
    //   nelem * sizeof(T): 计算总字节数
    NCCLCHECKGOTO(ncclCuMemAlloc((void **)ptr, NULL, ncclCuMemHandleType, nelem*ncclSizeOfT<T>()), result, finish);
  } else {
    // 否则使用传统的 CUDA malloc API 分配内存
    // Need a side stream so as not to interfere with graph capture.
    // 需要一个侧流，以免干扰 CUDA Graph 捕获
    cudaStream_t stream;
    // 创建非阻塞 CUDA 流（不与默认流同步）
    CUDACHECKGOTO(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), result, finish);
    // 使用传统的 cudaMalloc 分配设备内存
    CUDACHECKGOTO(cudaMalloc(ptr, nelem*ncclSizeOfT<T>()), result, finish);
  }

finish:
  // 同步流以确保所有操作完成
  CUDACHECKGOTO(cudaStreamSynchronize(stream), result, finish);
  // 销毁 CUDA 流
  CUDACHECKGOTO(cudaStreamDestroy(stream), result, finish);

  // 如果分配失败（指针为空且元素数量大于 0）
  if (*ptr == nullptr && nelem > 0) WARN("Failed to CUDA malloc %ld bytes", nelem*ncclSizeOfT<T>());
  // 输出调试信息：记录分配详情
  INFO(NCCL_ALLOC, "%s:%d Cuda Alloc Size %ld pointer %p", filefunc, line, nelem*ncclSizeOfT<T>(), *ptr);
  // 返回操作结果
  return result;
}

// 宏定义：调用调试版本的 CUDA 设备内存分配函数
// 参数：可变参数，传递类型、指针、元素数量、文件名和行号
#define ncclCudaMalloc(...) ncclCudaMallocDebug(__VA_ARGS__, __FILE__, __LINE__)

// 调试模式下的 CUDA 设备内存分配并异步复制函数（带文件名和行号跟踪）
// 功能：在设备端分配内存并异步复制数据（使用 CUDA Stream）
// 参数说明：
//   dst: 目标缓冲区指针（设备内存）
//   src: 源数据指针（主机内存）
//   nelem: 要复制的元素数量
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
template <typename T>
ncclResult_t ncclCudaMemcpy(T* dst, T* src, size_t nelem) {
  // 初始化返回值为成功
  ncclResult_t result = ncclSuccess;
  // 声明 CUDA 流捕获模式变量（用于 CUDA Graph 捕获）
  cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed;
  // 声明指针并初始化为空
  *ptr = nullptr;
  // 交换当前线程的流捕获模式（为后续操作准备流环境）
  CUDACHECK(cudaThreadExchangeStreamCaptureMode(&mode));

  // Need a side stream so as not to interfere with graph capture.
  // 需要一个侧流，以免干扰 CUDA Graph 捕获
  cudaStream_t stream;
  // 创建非阻塞 CUDA 流（不与默认流同步）
  CUDACHECKGOTO(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), result, finish);

  // 如果 CU Memory API 启用，使用特殊流和内存异步复制
  if (ncclCuMemEnable()) {
    // Need a side stream so as not to interfere with graph capture.
    // 需要一个侧流，以免干扰 CUDA Graph 捕获
    cudaStream_t stream2;
    // 创建非阻塞 CUDA 流（不与默认流同步）
    CUDACHECKGOTO(cudaStreamCreateWithFlags(&stream2, cudaStreamNonBlocking), result, finish2);
    // 调用 CUDA 内存异步设置和复制函数（使用 CU Memory）
    // 参数说明：
    //   *dst: 目标缓冲区指针
    //   0: 清零标志
    //   nelem * sizeof(T): 要复制的总字节数
    //   cudaMemcpyDeviceToDevice: 从主机到设备的复制方向
    //   stream2: 使用侧流进行异步操作
    CUDACHECKGOTO(cudaMemsetAsync(*dst, 0, nelem*ncclSizeOfT<T>(), cudaMemcpyDeviceToDevice, stream2), result, finish2);
  } else {
    // 否则使用标准 CUDA Runtime API 异步复制
    // 调用 CUDA 异步内存复制函数
    // 参数说明：
    //   dst: 目标缓冲区指针
    //   0: 清零标志
    //   nelem * sizeof(T): 要复制的总字节数
    //   cudaMemcpyDeviceToDevice: 从主机到设备的复制方向
    //   stream: 使用主 CUDA 流进行异步操作
    CUDACHECKGOTO(cudaMemcpyAsync(dst, 0, nelem*ncclSizeOfT<T>(), cudaMemcpyDeviceToDevice, stream), result, finish);
  }

finish:
  // 同步所有流以确保操作完成
  if (ncclCuMemEnable()) {
    // 如果使用了 CU Memory API，需要同步两个流
    CUDACHECKGOTO(cudaStreamSynchronize(stream), result, finish);
    CUDACHECKGOTO(cudaStreamSynchronize(stream2), result, finish2);
    CUDACHECKGOTO(cudaStreamDestroy(stream), result, finish);
    CUDACHECKGOTO(cudaStreamDestroy(stream2), result, finish);
  } else {
    // 如果使用标准 CUDA API，只需要同步一个流
    CUDACHECKGOTO(cudaStreamSynchronize(stream), result, finish);
    CUDACHECKGOTO(cudaStreamDestroy(stream), result, finish);
  }

  // 返回操作结果
  return result;
}

// CUDA 设备内存释放函数
// 功能：释放通过 CUDA API 分配的设备内存
// 参数说明：
//   ptr: 指向要释放的设备内存指针
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
template <typename T>
ncclResult_t ncclCudaFree(T* ptr) {
  // 初始化返回值为成功
  ncclResult_t result = ncclSuccess;
  // 记录跟踪信息：打印释放的指针地址
  TRACE(NCCL_ALLOC, "Cuda Free pointer %p", ptr);
  // 检查 CU Memory API 是否启用
  if (ncclCuMemEnable()) {
    // 如果启用 CU Memory，使用 cuMemFree 释放内存
    NCCLCHECKGOTO(ncclCuMemFree((void *)ptr), result, finish);
  } else {
    // 否则使用标准 CUDA Runtime API 释放内存
    CUDACHECKGOTO(cudaFree(ptr), result, finish);
  }

finish:
  // 返回操作结果
  return result;
}

// 宏定义：调用 CUDA 设备内存分配函数
// 参数：可变参数，传递类型、指针、元素数量
#define ncclCudaMalloc(...) ncclCudaMallocDebug(__VA_ARGS__, __FILE__, __LINE__)

// 分配可能被 ibv_reg_mr'd 的内存
// 功能：分配内存时需要考虑 IB verbs 注册的内存页特性
// 说明：这些内存页将被标记为 MADVISE（避免被 fork）
//       并且如果它们是共享的，可能在子进程中导致崩溃
// Allocate memory to be potentially ibv_reg_mr'd. This needs to be
// allocated on separate pages as those pages will be marked MADVISE
// and if they are shared that could cause a crash in a child process.
inline ncclResult_t ncclIbMallocDebug(void** ptr, size_t size, const char *filefunc, int line) {
  // 如果要分配的大小大于 0
  if (size > 0) {
    // 获取系统页面大小（用于内存对齐）
    long page_size = sysconf(_SC_PAGESIZE);
    // 如果获取页面大小失败，返回系统错误
    if (page_size < 0) return ncclSystemError;
    // 声明指针变量并初始化为空
    void* p;
    // 计算对齐后的大小（向上对齐到页边界）
    int size_aligned = ROUNDUP(size, page_size);
    // 调用 posix_memalign 分配对齐的内存
    // 参数说明：
    //   &p: 输出参数，返回分配的内存指针
    //   page_size: 对齐边界（页大小）
    //   size_aligned: 对齐后的分配大小
    // 返回值：0 表示成功，非 0 表示失败
    int ret = posix_memalign(&p, page_size, size_aligned);
    // 如果分配失败，返回系统错误
    if (ret != 0) return ncclSystemError;
    // 清零分配的内存
    memset(p, 0, size);
    // 保存分配的指针到输出参数
    *ptr = p;
  } else {
    // 如果大小为 0，则将指针设置为空
    *ptr = NULL;
  }
  // 输出调试信息：记录分配详情
  INFO(NCCL_ALLOC, "%s:%d Ib Alloc Size %ld pointer %p", filefunc, line, size, *ptr);
  // 返回成功状态
  return ncclSuccess;
}

// 宏定义：调用调试版本的 IB 内存分配函数
// 参数：可变参数，传递类型、指针、大小、文件名和行号
#define ncclIbMalloc(...) ncclIbMallocDebug(__VA_ARGS__, __FILE__, __LINE__)

#endif /* CUDART_VERSION >= 11030 */
