/*************************************************************************
 * Copyright (c) 2019-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// 头文件保护宏，防止该头文件被多次包含
#ifndef NCCL_ALLOC_H_
#define NCCL_ALLOC_H_

// 引入NCCL核心头文件，包含基本类型定义和函数声明
#include "nccl.h"
// 引入错误检查宏，如CUDACHECK、CUCHECK、NCCLCHECK等
#include "checks.h"
// 引入位操作工具函数，如对齐、位操作宏等
#include "bitops.h"
// 引入工具函数，但声明clockNano避免循环依赖
#include "utils.h"
// 引入P2P（点对点）通信相关定义和函数
#include "p2p.h"
// 引入系统内存管理相关头文件，用于mmap等操作
#include <sys/mman.h>
// 引入Unix标准头文件，用于sysconf等系统调用
#include <unistd.h>
// 引入C标准库内存管理函数，如malloc、free
#include <stdlib.h>
// 引入C标准库字符串操作函数，如memset、memcpy
#include <string.h>

// 如果CUDA运行时版本 >= 11.3，引入CUDA驱动API和封装函数
// CUDA 11.3引入了CUDA Unified Memory (CUMEM) API，支持更灵活的内存管理
#if CUDART_VERSION >= 11030
// 引入CUDA驱动API头文件，提供cuMem*系列函数
#include <cuda.h>
// 引入CUDA驱动API的错误检查封装
#include "cudawrap.h"
#endif

// 前向声明clockNano函数，来自utils.h
// 由于存在循环依赖问题（utils.h可能包含alloc.h），这里只做声明
uint64_t clockNano(); // from utils.h with which we have a circular dependency

/**
 * ncclSizeOfT - 获取类型的字节大小
 *
 * 这是一个模板函数，用于在编译时获取类型T的大小
 * constexpr关键字确保在编译期计算，零运行时开销
 *
 * @tparam T: 要获取大小的类型
 * @return: 类型T的字节大小
 */
template<typename T>
constexpr size_t ncclSizeOfT() { return sizeof(T); }

/**
 * ncclSizeOfT<void> - void类型的特化版本
 *
 * void类型没有大小（sizeof(void)在C++中是非法的），
 * 但在内存分配时我们需要一个单位大小，因此特化为1字节
 * 这允许模板代码在处理void类型时也能正常工作
 *
 * @return: 固定返回1（表示1字节）
 */
template<>
constexpr size_t ncclSizeOfT<void>() { return 1; }

#if CUDART_VERSION >= 12020

/**
 * ncclCuMemHostAlloc - 使用CUDA Unified Memory API分配主机内存
 *
 * 该函数使用CUDA 12.2+引入的CUMEM API分配可被GPU和CPU同时访问的主机内存
 * 相比传统的cudaHostAlloc，CUMEM提供了更细粒度的控制和更好的性能
 *
 * @param ptr: 输出参数，分配的内存指针
 * @param handlep: 输出参数，内存句柄指针，可用于导出和在其他进程中导入
 * @param size: 要分配的内存大小（字节）
 * @return ncclResult_t: 返回操作状态码，成功返回ncclSuccess
 */
static inline ncclResult_t ncclCuMemHostAlloc(void** ptr, CUmemGenericAllocationHandle *handlep, size_t size) {
  // 初始化返回值为成功
  ncclResult_t result = ncclSuccess;
  // 内存分配粒度，CUDA要求内存按特定粒度对齐
  size_t granularity = 0;
  // 当前CUDA设备
  CUdevice currentDev;
  // 内存分配属性结构体，用于描述分配的内存类型和位置
  CUmemAllocationProp prop = {};
  // 内存访问描述符，用于设置内存的访问权限
  CUmemAccessDesc accessDesc = {};
  // 通用内存分配句柄，用于标识这块内存
  CUmemGenericAllocationHandle handle;
  // CUDA设备序号（0, 1, 2...）
  int cudaDev;
  // CPU NUMA节点ID，用于指定内存在哪个NUMA节点上分配
  int cpuNumaNodeId = -1;
  // 内存句柄类型，决定了句柄如何被导出（如IPC、文件描述符等）
  CUmemAllocationHandleType type = ncclCuMemHandleType;

  // 获取当前CUDA设备序号
  CUDACHECK(cudaGetDevice(&cudaDev));
  // 根据设备序号获取CUdevice结构
  CUCHECK(cuDeviceGet(&currentDev, cudaDev));
  // 查询设备关联的CPU NUMA节点ID
  // 这有助于在离GPU最近的CPU内存上分配，提高访问性能
  CUCHECK(cuDeviceGetAttribute(&cpuNumaNodeId, CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID, currentDev));
  // 如果获取NUMA节点ID失败（返回-1），默认使用节点0
  if (cpuNumaNodeId < 0) cpuNumaNodeId = 0;

  // 设置内存分配属性
  // 指定内存位置类型为HOST_NUMA（主机NUMA节点）
  prop.location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;
  // 设置内存类型为PINNED（页锁定内存，不会被swap出去）
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  // 设置句柄类型，使内存可以被导出到其他进程或GPU
  prop.requestedHandleTypes = type; // So it can be exported
  // 指定具体的NUMA节点ID
  prop.location.id = cpuNumaNodeId;

  // 获取内存分配的最小粒度
  // CUDA要求内存分配必须按此粒度对齐
  CUCHECK(cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
  // 将size向上对齐到granularity的倍数
  ALIGN_SIZE(size, granularity);

  /* Allocate the physical memory on the device */
  // 创建物理内存分配，返回句柄
  CUCHECK(cuMemCreate(&handle, size, &prop, 0));

  /* Reserve a virtual address range */
  // 在进程的虚拟地址空间中预留一段地址范围
  // 这段地址范围稍后会被映射到物理内存
  CUCHECK(cuMemAddressReserve((CUdeviceptr*)ptr, size, granularity, 0, 0));

  /* Map the virtual address range to the physical allocation */
  // 将虚拟地址范围映射到之前创建的物理内存
  // 这样通过ptr就可以访问实际的物理内存
  CUCHECK(cuMemMap((CUdeviceptr)*ptr, size, 0, handle, 0));

  /* Now allow RW access to the newly mapped memory for local GPU */
  // 设置GPU对这块内存的访问权限
  // 指定位置类型为DEVICE（GPU设备）
  accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  // 指定具体的GPU设备ID
  accessDesc.location.id = cudaDev;
  // 设置访问权限为读写（PROT_READWRITE）
  accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  // 应用访问权限设置
  CUCHECK(cuMemSetAccess((CUdeviceptr)*ptr, size, &accessDesc, 1));

  /* Now allow RW access to the newly mapped memory from the CPU */
  // 设置CPU对这块内存的访问权限
  // 指定位置类型为HOST_NUMA（CPU NUMA节点）
  accessDesc.location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;
  // 指定具体的NUMA节点ID
  accessDesc.location.id = cpuNumaNodeId;
  // 设置访问权限为读写
  accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  // 应用访问权限设置
  CUCHECK(cuMemSetAccess((CUdeviceptr)*ptr, size, &accessDesc, 1));

  // 如果调用者提供了handlep指针，将句柄输出
  if (handlep) *handlep = handle;

  // 记录分配信息到日志
  INFO(NCCL_ALLOC, "CUMEM Host Alloc Size %zi pointer %p handle %llx numa %d dev %d granularity %ld", size, *ptr, handle, cpuNumaNodeId, cudaDev, granularity);
  return result;
}

/**
 * ncclCuMemHostFree - 释放通过ncclCuMemHostAlloc分配的主机内存
 *
 * 该函数释放CUMEM分配的主机内存，包括解除映射、释放物理内存和虚拟地址
 *
 * @param ptr: 要释放的内存指针
 * @return ncclResult_t: 返回操作状态码，成功返回ncclSuccess
 */
static inline ncclResult_t ncclCuMemHostFree(void* ptr) {
  // 如果指针为NULL，直接返回成功
  if (ptr == NULL) return ncclSuccess;
  // 初始化返回值为成功
  ncclResult_t result = ncclSuccess;
  // 内存分配句柄
  CUmemGenericAllocationHandle handle;
  // 内存大小
  size_t size = 0;

  // 通过指针获取内存句柄
  // cuMemRetainAllocationHandle会增加句柄的引用计数
  CUCHECK(cuMemRetainAllocationHandle(&handle, ptr));
  // 释放句柄（减少引用计数）
  CUCHECK(cuMemRelease(handle));
  // 获取内存地址范围的信息（主要是大小）
  CUCHECK(cuMemGetAddressRange(NULL, &size, (CUdeviceptr)ptr));

  // 记录释放信息到跟踪日志
  TRACE(NCCL_ALLOC, "CUMEM Host Free Size %zi pointer %p handle 0x%llx", size, ptr, handle);

  // 解除虚拟地址到物理内存的映射
  CUCHECK(cuMemUnmap((CUdeviceptr)ptr, size));
  // 释放物理内存（这会真正释放内存资源）
  CUCHECK(cuMemRelease(handle));
  // 释放虚拟地址范围
  CUCHECK(cuMemAddressFree((CUdeviceptr)ptr, size));
  return result;
}

#else /* CUDART_VERSION >= 12020 */

/**
 * ncclCuMemHostAlloc - CUMEM Host内存分配的存根函数（CUDA < 12.2版本）
 *
 * 当CUDA版本低于12.2时，CUMEM Host API不可用
 * 这个存根函数用于提供统一的接口，但会返回错误
 *
 * @param ptr: 输出参数，内存指针（不会被设置）
 * @param handlep: 输出参数，句柄指针（不会被设置）
 * @param size: 要分配的大小（会被忽略）
 * @return ncclResult_t: 返回ncclInternalError错误码
 */
static inline ncclResult_t ncclCuMemHostAlloc(void** ptr, void* handlep, size_t size) {
  // 输出警告信息，告知用户CUMEM Host需要CUDA 12.2或更高版本
  WARN("CUMEM Host is not supported prior to CUDA 12.2");
  return ncclInternalError;
}

/**
 * ncclCuMemHostFree - CUMEM Host内存释放的存根函数（CUDA < 12.2版本）
 *
 * 当CUDA版本低于12.2时，此函数不可用
 *
 * @param ptr: 要释放的内存指针（会被忽略）
 * @return ncclResult_t: 返回ncclInternalError错误码
 */
static inline ncclResult_t ncclCuMemHostFree(void* ptr) {
  // 输出警告信息，告知用户CUMEM Host需要CUDA 12.2或更高版本
  WARN("CUMEM Host is not supported prior to CUDA 12.2");
  return ncclInternalError;
}

#endif  /* CUDART_VERSION >= 12020 */

/**
 * ncclCudaHostCallocDebug - 分配并清零CUDA主机（可锁页）内存（调试版本）
 *
 * 该函数使用cudaHostAlloc分配可被GPU直接访问的主机内存
 * 这种内存在物理内存中被"锁定"，不会被操作系统swap到磁盘
 * cudaHostAllocMapped标志创建一个映射，使得GPU和CPU都能访问同一块物理内存
 *
 * @tparam T: 要分配的数据类型
 * @param ptr: 输出参数，分配的内存指针
 * @param nelem: 要分配的元素个数（总大小 = nelem * sizeof(T)）
 * @param filefunc: 调用此函数的源文件和函数名（用于调试）
 * @param line: 调用此函数的行号（用于调试）
 * @return ncclResult_t: 返回操作状态码，成功返回ncclSuccess
 */
template <typename T>
ncclResult_t ncclCudaHostCallocDebug(T** ptr, size_t nelem, const char *filefunc, int line) {
  // 初始化返回值为成功
  ncclResult_t result = ncclSuccess;
  // CUDA流捕获模式，用于与CUDA图捕获兼容
  cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed;
  // 初始化输出指针为空指针
  *ptr = nullptr;

  // 交换当前的流捕获模式为Relaxed模式
  // 这样在分配内存时不会干扰CUDA图的捕获过程
  CUDACHECK(cudaThreadExchangeStreamCaptureMode(&mode));

  // 只有在元素数量大于0时才进行内存分配
  if (nelem > 0) {
    // 使用cudaHostAlloc分配锁页主机内存
    // cudaHostAllocMapped标志创建GPU可直接访问的映射
    CUDACHECKGOTO(cudaHostAlloc(ptr, nelem*ncclSizeOfT<T>(), cudaHostAllocMapped), result, finish);
    // 将分配的内存清零
    memset(*ptr, 0, nelem*ncclSizeOfT<T>());
  }

finish:
  // 恢复之前的流捕获模式
  CUDACHECK(cudaThreadExchangeStreamCaptureMode(&mode));

  // 如果分配失败且请求的大小大于0，输出警告
  if (*ptr == nullptr && nelem > 0) WARN("Failed to CUDA host alloc %ld bytes", nelem*ncclSizeOfT<T>());

  // 记录分配信息到日志，包含文件名、行号、大小和指针
  INFO(NCCL_ALLOC, "%s:%d Cuda Host Alloc Size %ld pointer %p", filefunc, line, nelem*ncclSizeOfT<T>(), *ptr);
  return result;
}

/**
 * ncclCudaHostFree - 释放通过ncclCudaHostCalloc分配的主机内存
 *
 * 该函数释放由cudaHostAlloc分配的主机内存
 *
 * @param ptr: 要释放的内存指针
 * @return ncclResult_t: 返回操作状态码，成功返回ncclSuccess
 */
static inline ncclResult_t ncclCudaHostFree(void* ptr) {
  // 调用CUDA运行时API释放主机内存
  CUDACHECK(cudaFreeHost(ptr));
  return ncclSuccess;
}

// 宏定义：自动添加文件名和行号参数的包装器
// 使用__VA_ARGS__接收可变参数，并附加__FILE__和__LINE__
#define ncclCudaHostCalloc(...) ncclCudaHostCallocDebug(__VA_ARGS__, __FILE__, __LINE__)

/**
 * ncclCallocDebug - 分配并清零普通主机内存（调试版本）
 *
 * 该函数使用标准的malloc分配主机内存，并将内存清零
 * 与ncclCudaHostCalloc不同，这个函数分配的是普通主机内存，GPU无法直接访问
 *
 * @tparam T: 要分配的数据类型
 * @param ptr: 输出参数，分配的内存指针
 * @param nelem: 要分配的元素个数（总大小 = nelem * sizeof(T)）
 * @param filefunc: 调用此函数的源文件和函数名（用于调试）
 * @param line: 调用此函数的行号（用于调试）
 * @return ncclResult_t: 返回操作状态码，成功返回ncclSuccess
 */
template <typename T>
ncclResult_t ncclCallocDebug(T** ptr, size_t nelem, const char *filefunc, int line) {
  // 只有在元素数量大于0时才进行内存分配
  if (nelem > 0) {
    // 使用malloc分配内存
    T* p = (T*)malloc(nelem*ncclSizeOfT<T>());
    // 检查内存分配是否成功
    if (p == NULL) {
      // 分配失败，输出警告并返回系统错误
      WARN("Failed to malloc %ld bytes", nelem*ncclSizeOfT<T>());
      return ncclSystemError;
    }
    // 注释掉的日志：通常不需要记录每次malloc（注释保留以备调试需要）
    //INFO(NCCL_ALLOC, "%s:%d malloc Size %ld pointer %p", filefunc, line, nelem*ncclSizeOfT<T>(), p);
    // 将分配的内存清零
    memset(p, 0, nelem*ncclSizeOfT<T>());
    // 设置输出指针
    *ptr = p;
  } else {
    // 元素数量为0，设置输出指针为NULL
    *ptr = NULL;
  }
  return ncclSuccess;
}
// 宏定义：自动添加文件名和行号参数的包装器
#define ncclCalloc(...) ncclCallocDebug(__VA_ARGS__, __FILE__, __LINE__)

/**
 * ncclRealloc - 重新分配内存（仅支持扩展）
 *
 * 该函数重新分配更大的内存块，并保留原有数据
 * 注意：NCCL的realloc只支持扩展内存，不支持缩小
 * 这是为了简化内存管理逻辑，避免数据丢失的风险
 *
 * @tparam T: 数据类型
 * @param ptr: 指向原内存指针的指针（会被更新）
 * @param oldNelem: 原内存的元素个数
 * @param nelem: 新的元素个数（必须 >= oldNelem）
 * @return ncclResult_t: 返回操作状态码，成功返回ncclSuccess
 */
template <typename T>
ncclResult_t ncclRealloc(T** ptr, size_t oldNelem, size_t nelem) {
  // 保存原指针
  T* oldp = *ptr;

  // 参数有效性检查
  // 新元素个数不能小于旧元素个数（不支持缩小）
  // 如果原指针为NULL但旧元素个数大于0，是非法状态
  if (nelem < oldNelem || (oldp == NULL && oldNelem > 0)) return ncclInternalError;

  // 如果新旧元素个数相同，无需操作
  if (nelem == oldNelem) return ncclSuccess;

  // 分配新的内存块
  T* p = (T*)malloc(nelem*ncclSizeOfT<T>());
  if (p == NULL) {
    // 分配失败，输出警告并返回系统错误
    WARN("Failed to malloc %ld bytes", nelem*ncclSizeOfT<T>());
    return ncclSystemError;
  }

  // 如果原内存有数据，复制到新内存
  if (oldp && oldNelem) memcpy(p, oldp, oldNelem * ncclSizeOfT<T>());

  // 释放原内存
  if (oldp) free(oldp);

  // 将新增的内存部分清零（从oldNelem到nelem）
  memset(p+oldNelem, 0, (nelem-oldNelem)*ncclSizeOfT<T>());

  // 更新指针
  *ptr = (T*)p;

  // 记录重新分配的信息到日志
  INFO(NCCL_ALLOC, "Mem Realloc old size %ld, new size %ld pointer %p", oldNelem*ncclSizeOfT<T>(), nelem*ncclSizeOfT<T>(), *ptr);
  return ncclSuccess;
}

#if CUDART_VERSION >= 11030

// 引入CUDA驱动API头文件，提供cuMem*系列函数
#include <cuda.h>
// 引入CUDA驱动API的错误检查封装
#include "cudawrap.h"

/**
 * ncclCuMemAllocAddr - 将已有的CUMEM句柄映射到当前进程的地址空间
 *
 * 该函数用于导入由其他进程或GPU分配的内存
 * 通过内存句柄可以在多个进程间共享同一块物理内存
 *
 * @param ptr: 输出参数，映射后的内存指针
 * @param handleIn: 输入的内存句柄（由其他进程或GPU分配时产生）
 * @param size: 要映射的内存大小
 * @return ncclResult_t: 返回操作状态码，成功返回ncclSuccess
 */
// ncclCuMemAllocAddr takes memory handle and size and returns the mapped address pointer
static inline ncclResult_t ncclCuMemAllocAddr(void **ptr, CUmemGenericAllocationHandle *handleIn, size_t size) {
  // 初始化返回值为成功
  ncclResult_t result = ncclSuccess;
  // 内存分配粒度，CUDA要求内存按特定粒度对齐
  size_t granularity = 0;
  // 内存分配属性结构体
  CUmemAllocationProp prop = {};
  // 内存访问描述符
  CUmemAccessDesc accessDesc = {};
  // CUDA设备序号
  int cudaDev;

  // 获取当前CUDA设备序号
  CUDACHECK(cudaGetDevice(&cudaDev));
  // 从内存句柄中获取内存分配属性
  // 这可以让我们了解原始内存是如何分配的
  CUCHECK(cuMemGetAllocationPropertiesFromHandle(&prop, *handleIn));
  // 根据内存属性获取分配粒度
  CUCHECK(cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
  // 将size向上对齐到granularity的倍数
  ALIGN_SIZE(size, granularity);

  /* Reserve a virtual address range */
  // 在当前进程的虚拟地址空间中预留一段地址范围
  CUCHECK(cuMemAddressReserve((CUdeviceptr *)ptr, size, granularity, 0, 0));

  /* Map the virtual address range to the physical allocation */
  // 将虚拟地址范围映射到由handleIn指定的物理内存
  CUCHECK(cuMemMap((CUdeviceptr)*ptr, size, 0, *handleIn, 0));

  /* Now allow RW access to the newly mapped memory */
  // 设置当前GPU对这块内存的访问权限
  // 指定位置类型为DEVICE（GPU设备）
  accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  // 指定具体的GPU设备ID
  accessDesc.location.id = cudaDev;
  // 设置访问权限为读写
  accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  // 应用访问权限设置
  CUCHECK(cuMemSetAccess((CUdeviceptr)*ptr, size, &accessDesc, 1));

  // 记录映射信息到跟踪日志
  TRACE(NCCL_ALLOC, "CuMem Map Size %zu pointer %p handle %llx", size, *ptr, *handleIn);
  return result;
}

/**
 * ncclCuMemFreeAddr - 解除并释放通过ncclCuMemAllocAddr映射的内存地址
 *
 * 该函数解除内存映射并释放虚拟地址空间
 * 注意：这只释放当前进程的地址映射，不会影响物理内存本身
 *
 * @param ptr: 要解除映射的内存指针
 * @return ncclResult_t: 返回操作状态码，成功返回ncclSuccess
 */
static inline ncclResult_t ncclCuMemFreeAddr(void *ptr) {
  // 如果指针为NULL，直接返回成功
  if (ptr == NULL) return ncclSuccess;
  // 初始化返回值为成功
  ncclResult_t result = ncclSuccess;
  // 内存大小
  size_t size = 0;
  // 获取内存地址范围的大小
  CUCHECK(cuMemGetAddressRange(NULL, &size, (CUdeviceptr)ptr));
  // 解除内存映射
  CUCHECK(cuMemUnmap((CUdeviceptr)ptr, size));
  // 释放虚拟地址范围
  CUCHECK(cuMemAddressFree((CUdeviceptr)ptr, size));
  return result;
}

/**
 * ncclCuMemAlloc - 分配CUDA设备内存（使用CUMEM API）
 *
 * 该函数使用CUDA 11.3+引入的CUMEM API分配GPU设备内存
 * 支持GPU Direct RDMA，允许网络设备直接访问GPU内存
 *
 * @param ptr: 输出参数，分配的内存指针
 * @param handlep: 输出参数，内存句柄指针（可选，用于导出内存）
 * @param type: 内存句柄类型，决定句柄如何被导出
 * @param size: 要分配的内存大小（字节）
 * @return ncclResult_t: 返回操作状态码，成功返回ncclSuccess
 */
static inline ncclResult_t ncclCuMemAlloc(void **ptr, CUmemGenericAllocationHandle *handlep, CUmemAllocationHandleType type, size_t size) {
  // 初始化返回值为成功
  ncclResult_t result = ncclSuccess;
  // 内存分配粒度
  size_t granularity = 0;
  // 当前CUDA设备
  CUdevice currentDev;
  // 内存分配属性结构体
  CUmemAllocationProp prop = {};
  // 内存访问描述符
  CUmemAccessDesc accessDesc = {};
  // 通用内存分配句柄
  CUmemGenericAllocationHandle handle;
  // CUDA设备序号
  int cudaDev;
  // 用于查询设备属性的标志位
  int flag = 0;

  // 获取当前CUDA设备序号
  CUDACHECK(cudaGetDevice(&cudaDev));
  // 根据设备序号获取CUdevice结构
  CUCHECK(cuDeviceGet(&currentDev, cudaDev));

  // 设置内存分配属性
  // 设置内存类型为PINNED（固定在GPU上，不会被换页）
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  // 指定内存位置为DEVICE（GPU设备）
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  // 设置句柄类型，使内存可以被导出
  prop.requestedHandleTypes = type;
  // 指定具体的GPU设备ID
  prop.location.id = currentDev;

  // Query device to see if RDMA support is available
  // 查询设备是否支持GPU Direct RDMA
  // GPU Direct RDMA允许网络设备直接访问GPU内存，无需CPU介入
  CUCHECK(cuDeviceGetAttribute(&flag, CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED, currentDev));
  // 如果支持RDMA，设置相应标志
  if (flag) prop.allocFlags.gpuDirectRDMACapable = 1;

  // 获取内存分配的最小粒度
  CUCHECK(cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
  // 将size向上对齐到granularity的倍数
  ALIGN_SIZE(size, granularity);

  /* Allocate the physical memory on the device */
  // 创建物理内存分配，返回句柄
  CUCHECK(cuMemCreate(&handle, size, &prop, 0));

  /* Reserve a virtual address range */
  // 在GPU的虚拟地址空间中预留一段地址范围
  CUCHECK(cuMemAddressReserve((CUdeviceptr *)ptr, size, granularity, 0, 0));

  /* Map the virtual address range to the physical allocation */
  // 将虚拟地址范围映射到物理内存
  CUCHECK(cuMemMap((CUdeviceptr)*ptr, size, 0, handle, 0));

  /* Now allow RW access to the newly mapped memory */
  // 设置GPU对这块内存的访问权限
  // 指定位置类型为DEVICE（GPU设备）
  accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  // 指定具体的GPU设备ID
  accessDesc.location.id = currentDev;
  // 设置访问权限为读写
  accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  // 应用访问权限设置
  CUCHECK(cuMemSetAccess((CUdeviceptr)*ptr, size, &accessDesc, 1));

  // 如果调用者提供了handlep指针，将句柄输出
  if (handlep) *handlep = handle;

  // 记录分配信息到跟踪日志
  TRACE(NCCL_ALLOC, "CuMem Alloc Size %zu pointer %p handle %llx", size, *ptr, handle);
  return result;
}

/**
 * ncclCuMemFree - 释放通过ncclCuMemAlloc分配的设备内存
 *
 * 该函数释放CUMEM分配的设备内存，包括解除映射、释放物理内存和虚拟地址
 *
 * @param ptr: 要释放的内存指针
 * @return ncclResult_t: 返回操作状态码，成功返回ncclSuccess
 */
static inline ncclResult_t ncclCuMemFree(void *ptr) {
  // 如果指针为NULL，直接返回成功
  if (ptr == NULL) return ncclSuccess;
  // 初始化返回值为成功
  ncclResult_t result = ncclSuccess;
  // 内存分配句柄
  CUmemGenericAllocationHandle handle;
  // 内存大小
  size_t size = 0;

  // 通过指针获取内存句柄（增加引用计数）
  CUCHECK(cuMemRetainAllocationHandle(&handle, ptr));
  // 立即释放句柄（减少引用计数）
  // 这里看起来多余，但实际上是为了在获取size前确保handle有效
  CUCHECK(cuMemRelease(handle));
  // 获取内存地址范围的大小
  CUCHECK(cuMemGetAddressRange(NULL, &size, (CUdeviceptr)ptr));

  // 记录释放信息到跟踪日志
  TRACE(NCCL_ALLOC, "CuMem Free Size %zu pointer %p handle 0x%llx", size, ptr, handle);

  // 解除虚拟地址到物理内存的映射
  CUCHECK(cuMemUnmap((CUdeviceptr)ptr, size));
  // 释放物理内存
  CUCHECK(cuMemRelease(handle));
  // 释放虚拟地址范围
  CUCHECK(cuMemAddressFree((CUdeviceptr)ptr, size));
  return result;
}

#else

// 外部声明：检查是否启用了CUMEM支持（通过环境变量或配置）
// 在某些情况下，即使CUDA版本支持，用户也可能选择不使用CUMEM
extern int ncclCuMemEnable();

/**
 * ncclCuMemAlloc - CUMEM设备内存分配的存根函数（CUDA < 11.3版本）
 *
 * 当CUDA版本低于11.3时，CUMEM API不可用
 * 这个存根函数用于提供统一的接口，但会返回错误
 *
 * @param ptr: 输出参数，内存指针（不会被设置）
 * @param handlep: 输出参数，句柄指针（不会被设置）
 * @param type: 句柄类型（会被忽略）
 * @param size: 要分配的大小（会被忽略）
 * @return ncclResult_t: 返回ncclInternalError错误码
 */
static inline ncclResult_t ncclCuMemAlloc(void **ptr, void *handlep, int type, size_t size) {
  // 输出警告信息，告知用户CUMEM需要CUDA 11.3或更高版本
  WARN("CUMEM not supported prior to CUDA 11.3");
  return ncclInternalError;
}

/**
 * ncclCuMemFree - CUMEM设备内存释放的存根函数（CUDA < 11.3版本）
 *
 * 当CUDA版本低于11.3时，此函数不可用
 *
 * @param ptr: 要释放的内存指针（会被忽略）
 * @return ncclResult_t: 返回ncclInternalError错误码
 */
static inline ncclResult_t ncclCuMemFree(void *ptr) {
  // 输出警告信息，告知用户CUMEM需要CUDA 11.3或更高版本
  WARN("CUMEM not supported prior to CUDA 11.3");
  return ncclInternalError;
}

/**
 * ncclCuMemAllocAddr - CUMEM地址映射的存根函数（CUDA < 11.3版本）
 *
 * 当CUDA版本低于11.3时，此函数不可用
 *
 * @param ptr: 输出参数，内存指针（不会被设置）
 * @param handleIn: 输入的内存句柄（会被忽略）
 * @param size: 要映射的大小（会被忽略）
 * @return ncclResult_t: 返回ncclInternalError错误码
 */
static inline ncclResult_t ncclCuMemAllocAddr(void **ptr, CUmemGenericAllocationHandle *handleIn, size_t size) {
  // 输出警告信息，告知用户CUMEM需要CUDA 11.3或更高版本
  WARN("CUMEM not supported prior to CUDA 11.3");
  return ncclInternalError;
}

/**
 * ncclCuMemFreeAddr - CUMEM地址释放的存根函数（CUDA < 11.3版本）
 *
 * 当CUDA版本低于11.3时，此函数不可用
 *
 * @param ptr: 要释放的内存指针（会被忽略）
 * @return ncclResult_t: 返回ncclInternalError错误码
 */
static inline ncclResult_t ncclCuMemFreeAddr(void *ptr) {
  // 输出警告信息，告知用户CUMEM需要CUDA 11.3或更高版本
  WARN("CUMEM not supported prior to CUDA 11.3");
  return ncclInternalError;
}
#endif

/**
 * ncclCudaMallocDebug - 分配CUDA设备内存（调试版本）
 *
 * 该函数分配GPU设备内存，支持使用CUMEM API（如果启用）
 * 设备内存只能被GPU访问，CPU无法直接访问
 *
 * @tparam T: 要分配的数据类型
 * @param ptr: 输出参数，分配的内存指针
 * @param nelem: 要分配的元素个数（总大小 = nelem * sizeof(T)）
 * @param filefunc: 调用此函数的源文件和函数名（用于调试）
 * @param line: 调用此函数的行号（用于调试）
 * @return ncclResult_t: 返回操作状态码，成功返回ncclSuccess
 */
template <typename T>
ncclResult_t ncclCudaMallocDebug(T** ptr, size_t nelem, const char *filefunc, int line) {
  // 初始化返回值为成功
  ncclResult_t result = ncclSuccess;
  // CUDA流捕获模式，用于与CUDA图捕获兼容
  cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed;
  // 初始化输出指针为空指针
  *ptr = nullptr;
  // 交换当前的流捕获模式为Relaxed模式
  // 这样在分配内存时不会干扰CUDA图的捕获过程
  CUDACHECK(cudaThreadExchangeStreamCaptureMode(&mode));
  // 只有在元素数量大于0时才进行内存分配
  if (nelem > 0) {
    // 检查是否启用了CUMEM支持
    if (ncclCuMemEnable()) {
      // 使用CUMEM API分配内存，支持GPU Direct RDMA
      // NULL表示不需要导出句柄
      NCCLCHECKGOTO(ncclCuMemAlloc((void **)ptr, NULL, ncclCuMemHandleType, nelem*ncclSizeOfT<T>()), result, finish);
    } else {
      // 使用传统的CUDA运行时API分配内存
      CUDACHECKGOTO(cudaMalloc(ptr, nelem*ncclSizeOfT<T>()), result, finish);
    }
  }
finish:
  // 恢复之前的流捕获模式
  CUDACHECK(cudaThreadExchangeStreamCaptureMode(&mode));
  // 如果分配失败且请求的大小大于0，输出警告
  if (*ptr == nullptr && nelem > 0) WARN("Failed to CUDA malloc %ld bytes", nelem*ncclSizeOfT<T>());
  // 记录分配信息到日志
  INFO(NCCL_ALLOC, "%s:%d Cuda Alloc Size %ld pointer %p", filefunc, line, nelem*ncclSizeOfT<T>(), *ptr);
  return result;
}
// 宏定义：自动添加文件名和行号参数的包装器
#define ncclCudaMalloc(...) ncclCudaMallocDebug(__VA_ARGS__, __FILE__, __LINE__)

/**
 * ncclCudaCallocDebug - 分配并清零CUDA设备内存（调试版本）
 *
 * 该函数分配GPU设备内存并将内容清零
 * 使用单独的流进行清零操作，避免干扰CUDA图捕获
 *
 * @tparam T: 要分配的数据类型
 * @param ptr: 输出参数，分配的内存指针
 * @param nelem: 要分配的元素个数（总大小 = nelem * sizeof(T)）
 * @param filefunc: 调用此函数的源文件和函数名（用于调试）
 * @param line: 调用此函数的行号（用于调试）
 * @return ncclResult_t: 返回操作状态码，成功返回ncclSuccess
 */
template <typename T>
ncclResult_t ncclCudaCallocDebug(T** ptr, size_t nelem, const char *filefunc, int line) {
  // 初始化返回值为成功
  ncclResult_t result = ncclSuccess;
  // CUDA流捕获模式
  cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed;
  // 初始化输出指针为空指针
  *ptr = nullptr;
  // 交换当前的流捕获模式为Relaxed模式
  CUDACHECK(cudaThreadExchangeStreamCaptureMode(&mode));
  // 只有在元素数量大于0时才进行内存分配
  if (nelem > 0) {
    // Need a side stream so as not to interfere with graph capture.
    // 创建一个非阻塞的辅助流，用于执行清零操作
    // 这样不会干扰用户的CUDA图捕获过程
    cudaStream_t stream;
    CUDACHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    // 检查是否启用了CUMEM支持
    if (ncclCuMemEnable()) {
      // 使用CUMEM API分配内存
      NCCLCHECKGOTO(ncclCuMemAlloc((void **)ptr, NULL, ncclCuMemHandleType, nelem*ncclSizeOfT<T>()), result, finish);
    } else {
      // 使用传统的CUDA运行时API分配内存
      CUDACHECKGOTO(cudaMalloc(ptr, nelem*ncclSizeOfT<T>()), result, finish);
    }
    // 在辅助流上异步将内存清零
    CUDACHECKGOTO(cudaMemsetAsync(*ptr, 0, nelem*ncclSizeOfT<T>(), stream), result, finish);
    // 同步等待清零操作完成
    CUDACHECKGOTO(cudaStreamSynchronize(stream), result, finish);
    // 销毁辅助流
    CUDACHECKGOTO(cudaStreamDestroy(stream), result, finish);
  }
finish:
  // 恢复之前的流捕获模式
  CUDACHECK(cudaThreadExchangeStreamCaptureMode(&mode));
  // 如果分配失败且请求的大小大于0，输出警告
  if (*ptr == nullptr && nelem > 0) WARN("Failed to CUDA calloc %ld bytes", nelem*ncclSizeOfT<T>());
  // 记录分配信息到日志
  INFO(NCCL_ALLOC, "%s:%d Cuda Alloc Size %ld pointer %p", filefunc, line, nelem*ncclSizeOfT<T>(), *ptr);
  return result;
}
// 宏定义：自动添加文件名和行号参数的包装器
#define ncclCudaCalloc(...) ncclCudaCallocDebug(__VA_ARGS__, __FILE__, __LINE__)

/**
 * ncclCudaCallocAsyncDebug - 异步分配并清零CUDA设备内存（调试版本）
 *
 * 该函数分配GPU设备内存并在指定的CUDA流上异步清零
 * 清零操作在提供的流上执行，调用者负责同步
 *
 * @tparam T: 要分配的数据类型
 * @param ptr: 输出参数，分配的内存指针
 * @param nelem: 要分配的元素个数（总大小 = nelem * sizeof(T)）
 * @param stream: 用于执行清零操作的CUDA流
 * @param filefunc: 调用此函数的源文件和函数名（用于调试）
 * @param line: 调用此函数的行号（用于调试）
 * @return ncclResult_t: 返回操作状态码，成功返回ncclSuccess
 */
template <typename T>
ncclResult_t ncclCudaCallocAsyncDebug(T** ptr, size_t nelem, cudaStream_t stream, const char *filefunc, int line) {
  // 初始化返回值为成功
  ncclResult_t result = ncclSuccess;
  // CUDA流捕获模式
  cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed;
  // 初始化输出指针为空指针
  *ptr = nullptr;
  // 交换当前的流捕获模式为Relaxed模式
  CUDACHECK(cudaThreadExchangeStreamCaptureMode(&mode));
  // 只有在元素数量大于0时才进行内存分配
  if (nelem > 0) {
    // 检查是否启用了CUMEM支持
    if (ncclCuMemEnable()) {
      // 使用CUMEM API分配内存
      NCCLCHECKGOTO(ncclCuMemAlloc((void **)ptr, NULL, ncclCuMemHandleType, nelem*ncclSizeOfT<T>()), result, finish);
    } else {
      // 使用传统的CUDA运行时API分配内存
      CUDACHECKGOTO(cudaMalloc(ptr, nelem*ncclSizeOfT<T>()), result, finish);
    }
    // 在用户提供的流上异步将内存清零
    // 调用者负责在需要时同步流
    CUDACHECKGOTO(cudaMemsetAsync(*ptr, 0, nelem*ncclSizeOfT<T>(), stream), result, finish);
  }
finish:
  // 恢复之前的流捕获模式
  CUDACHECK(cudaThreadExchangeStreamCaptureMode(&mode));
  // 如果分配失败且请求的大小大于0，输出警告
  if (*ptr == nullptr && nelem > 0) WARN("Failed to CUDA calloc async %ld bytes", nelem*ncclSizeOfT<T>());
  // 记录分配信息到日志
  INFO(NCCL_ALLOC, "%s:%d Cuda Alloc Size %ld pointer %p", filefunc, line, nelem*ncclSizeOfT<T>(), *ptr);
  return result;
}
// 宏定义：自动添加文件名和行号参数的包装器
#define ncclCudaCallocAsync(...) ncclCudaCallocAsyncDebug(__VA_ARGS__, __FILE__, __LINE__)

/**
 * ncclCudaMemcpy - 同步CUDA内存拷贝
 *
 * 该函数在GPU之间或GPU与主机之间同步拷贝数据
 * 使用辅助流执行拷贝，不干扰CUDA图捕获
 *
 * @tparam T: 数据类型
 * @param dst: 目标内存指针
 * @param src: 源内存指针
 * @param nelem: 要拷贝的元素个数
 * @return ncclResult_t: 返回操作状态码，成功返回ncclSuccess
 */
template <typename T>
ncclResult_t ncclCudaMemcpy(T* dst, T* src, size_t nelem) {
  // 初始化返回值为成功
  ncclResult_t result = ncclSuccess;
  // CUDA流捕获模式
  cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed;
  // 交换当前的流捕获模式为Relaxed模式
  CUDACHECK(cudaThreadExchangeStreamCaptureMode(&mode));
  // Need a side stream so as not to interfere with graph capture.
  // 创建一个非阻塞的辅助流，用于执行拷贝操作
  cudaStream_t stream;
  CUDACHECKGOTO(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), result, finish);
  // 在辅助流上异步执行拷贝
  NCCLCHECKGOTO(ncclCudaMemcpyAsync(dst, src, nelem, stream), result, finish);
  // 同步等待拷贝操作完成
  CUDACHECKGOTO(cudaStreamSynchronize(stream), result, finish);
  // 销毁辅助流
  CUDACHECKGOTO(cudaStreamDestroy(stream), result, finish);
finish:
  // 恢复之前的流捕获模式
  CUDACHECK(cudaThreadExchangeStreamCaptureMode(&mode));
  return result;
}

/**
 * ncclCudaMemcpyAsync - 异步CUDA内存拷贝
 *
 * 该函数在GPU之间或GPU与主机之间异步拷贝数据
 * 拷贝操作在提供的流上执行
 *
 * @tparam T: 数据类型
 * @param dst: 目标内存指针
 * @param src: 源内存指针
 * @param nelem: 要拷贝的元素个数
 * @param stream: 用于执行拷贝操作的CUDA流
 * @return ncclResult_t: 返回操作状态码，成功返回ncclSuccess
 */
template <typename T>
ncclResult_t ncclCudaMemcpyAsync(T* dst, T* src, size_t nelem, cudaStream_t stream) {
  // 初始化返回值为成功
  ncclResult_t result = ncclSuccess;
  // CUDA流捕获模式
  cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed;
  // 交换当前的流捕获模式为Relaxed模式
  CUDACHECK(cudaThreadExchangeStreamCaptureMode(&mode));
  // 使用cudaMemcpyAsync执行异步拷贝
  // cudaMemcpyDefault自动选择拷贝方向（设备到设备、主机到设备等）
  CUDACHECKGOTO(cudaMemcpyAsync(dst, src, nelem*ncclSizeOfT<T>(), cudaMemcpyDefault, stream), result, finish);
finish:
  // 恢复之前的流捕获模式
  CUDACHECK(cudaThreadExchangeStreamCaptureMode(&mode));
  return result;
}

/**
 * ncclCudaFree - 释放CUDA设备内存
 *
 * 该函数释放由ncclCudaMalloc、ncclCudaCalloc等分配的GPU设备内存
 * 自动选择使用CUMEM或传统的cudaFree
 *
 * @tparam T: 数据类型
 * @param ptr: 要释放的内存指针
 * @return ncclResult_t: 返回操作状态码，成功返回ncclSuccess
 */
template <typename T>
ncclResult_t ncclCudaFree(T* ptr) {
  // 初始化返回值为成功
  ncclResult_t result = ncclSuccess;
  // CUDA流捕获模式
  cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed;
  // 记录释放信息到跟踪日志
  TRACE(NCCL_ALLOC, "Cuda Free pointer %p", ptr);
  // 交换当前的流捕获模式为Relaxed模式
  CUDACHECK(cudaThreadExchangeStreamCaptureMode(&mode));
  // 检查是否启用了CUMEM支持
  if (ncclCuMemEnable()) {
    // 使用CUMEM API释放内存
    NCCLCHECKGOTO(ncclCuMemFree((void *)ptr), result, finish);
  } else {
    // 使用传统的CUDA运行时API释放内存
    CUDACHECKGOTO(cudaFree(ptr), result, finish);
  }
finish:
  // 恢复之前的流捕获模式
  CUDACHECK(cudaThreadExchangeStreamCaptureMode(&mode));
  return result;
}

// Allocate memory to be potentially ibv_reg_mr'd. This needs to be
// allocated on separate pages as those pages will be marked DONTFORK
// and if they are shared, that could cause a crash in a child process
/**
 * ncclIbMallocDebug - 分配用于InfiniBand注册的内存（调试版本）
 *
 * 该函数分配专门用于InfiniBand verbs（ibv_reg_mr）的内存
 * 这是一种特殊的主机内存分配，需要满足以下要求：
 * 1. 必须在单独的页上分配，因为这些页会被标记为DONTFORK
 * 2. DONTFORK标志防止这些内存在fork后被复制到子进程
 * 3. 如果这些内存与其他分配共享，可能导致子进程崩溃
 *
 * @param ptr: 输出参数，分配的内存指针
 * @param size: 要分配的内存大小（字节）
 * @param filefunc: 调用此函数的源文件和函数名（用于调试）
 * @param line: 调用此函数的行号（用于调试）
 * @return ncclResult_t: 返回操作状态码，成功返回ncclSuccess
 */
inline ncclResult_t ncclIbMallocDebug(void** ptr, size_t size, const char *filefunc, int line) {
  // 只有在大小大于0时才进行内存分配
  if (size > 0) {
    // 获取系统页面大小（通常是4096字节）
    long page_size = sysconf(_SC_PAGESIZE);
    // 检查页面大小是否获取成功
    if (page_size < 0) return ncclSystemError;
    // 声明指针变量
    void* p;
    // 将size向上对齐到页面大小的倍数
    // 这确保分配的内存总是从页面边界开始
    int size_aligned = ROUNDUP(size, page_size);
    // 使用posix_memalign分配对齐的内存
    // posix_memalign确保分配的内存按指定边界对齐
    int ret = posix_memalign(&p, page_size, size_aligned);
    // 检查内存分配是否成功
    if (ret != 0) return ncclSystemError;
    // 将分配的内存清零（只清零请求的大小，不对齐后多余的部分）
    memset(p, 0, size);
    // 设置输出指针
    *ptr = p;
  } else {
    // 大小为0，设置输出指针为NULL
    *ptr = NULL;
  }
  // 记录分配信息到日志
  INFO(NCCL_ALLOC, "%s:%d Ib Alloc Size %ld pointer %p", filefunc, line, size, *ptr);
  return ncclSuccess;
}
// 宏定义：自动添加文件名和行号参数的包装器
#define ncclIbMalloc(...) ncclIbMallocDebug(__VA_ARGS__, __FILE__, __LINE__)

#endif
