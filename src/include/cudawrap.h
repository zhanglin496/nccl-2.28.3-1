/*************************************************************************
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 * 版权所有 (c) 2022，NVIDIA 公司。保留所有权利。
 *
 * See LICENSE.txt for license information
 * 详见 LICENSE.txt 获取许可证信息
 ************************************************************************/

// 头文件保护宏开始：防止头文件被重复包含
#ifndef NCCL_CUDAWRAP_H_
// 定义头文件保护宏
#define NCCL_CUDAWRAP_H_

// 包含 CUDA 驱动 API 头文件，提供底层 CUDA 驱动功能
#include <cuda.h>
// 包含 CUDA 运行时 API 头文件，提供高层 CUDA 运行时功能
#include <cuda_runtime.h>
// 包含 NCCL 内部检查头文件，提供参数验证和错误检查宏
#include "checks.h"

// 外部函数声明：查询 cuMem API 是否启用
// cuMem API 是 CUDA 的统一内存管理接口，提供跨进程的内存共享功能
// 返回值：非零表示启用，零表示禁用
extern int ncclCuMemEnable();
// 外部函数声明：查询 cuMem Host API 是否启用
// cuMem Host API 允许主机端直接访问 CUDA 统一内存
// 返回值：非零表示启用，零表示禁用
extern int ncclCuMemHostEnable();

// CUDA 运行时版本条件编译：CUDA 11.0 及以上版本包含 Typedef 头文件
// cudaTypedefs.h 提供了 CUDA 类型的类型定义（如 cudaError_t）
#if CUDART_VERSION >= 11030
#include <cudaTypedefs.h>
#endif

// 外部变量声明：cuMemCreate() 使用的内存分配句柄类型
// CUmemAllocationHandleType 是 CUDA 统一内存 API 的句柄类型
// 用于标识不同类型的内存分配（设备内存、主机内存、统一内存等）
extern CUmemAllocationHandleType ncclCuMemHandleType;
// 结束 CUDA 11.0+ 版本的条件编译
#endif

#define CUPFN(symbol) pfn_##symbol

// Check CUDA PFN driver calls
#define CUCHECK(cmd) do {				      \
    CUresult err = pfn_##cmd;				      \
    if( err != CUDA_SUCCESS ) {				      \
      const char *errStr;				      \
      (void) pfn_cuGetErrorString(err, &errStr);	      \
      WARN("Cuda failure %d '%s'", err, errStr);	      \
      return ncclUnhandledCudaError;			      \
    }							      \
} while(false)

#define CUCALL(cmd) do {				      \
    pfn_##cmd;				                \
} while(false)

#define CUCHECKGOTO(cmd, res, label) do {		      \
    CUresult err = pfn_##cmd;				      \
    if( err != CUDA_SUCCESS ) {				      \
      const char *errStr;				      \
      (void) pfn_cuGetErrorString(err, &errStr);	      \
      WARN("Cuda failure %d '%s'", err, errStr);	      \
      res = ncclUnhandledCudaError;			      \
      goto label;					      \
    }							      \
} while(false)

// Report failure but clear error and continue
#define CUCHECKIGNORE(cmd) do {						\
    CUresult err = pfn_##cmd;						\
    if( err != CUDA_SUCCESS ) {						\
      const char *errStr;						\
      (void) pfn_cuGetErrorString(err, &errStr);			\
      INFO(NCCL_ALL,"%s:%d Cuda failure %d '%s'", __FILE__, __LINE__, err, errStr); \
    }									\
} while(false)

#define CUCHECKTHREAD(cmd, args) do {					\
    CUresult err = pfn_##cmd;						\
    if (err != CUDA_SUCCESS) {						\
      INFO(NCCL_INIT,"%s:%d -> %d [Async thread]", __FILE__, __LINE__, err); \
      args->ret = ncclUnhandledCudaError;				\
      return args;							\
    }									\
} while(0)

#define DECLARE_CUDA_PFN_EXTERN(symbol,version) extern PFN_##symbol##_v##version pfn_##symbol

#if CUDART_VERSION >= 11030
/* CUDA Driver functions loaded with cuGetProcAddress for versioning */
DECLARE_CUDA_PFN_EXTERN(cuDeviceGet, 2000);
DECLARE_CUDA_PFN_EXTERN(cuDeviceGetAttribute, 2000);
DECLARE_CUDA_PFN_EXTERN(cuGetErrorString, 6000);
DECLARE_CUDA_PFN_EXTERN(cuGetErrorName, 6000);
DECLARE_CUDA_PFN_EXTERN(cuMemGetAddressRange, 3020);
DECLARE_CUDA_PFN_EXTERN(cuCtxCreate, 11040);
DECLARE_CUDA_PFN_EXTERN(cuCtxDestroy, 4000);
DECLARE_CUDA_PFN_EXTERN(cuCtxGetCurrent, 4000);
DECLARE_CUDA_PFN_EXTERN(cuCtxSetCurrent, 4000);
DECLARE_CUDA_PFN_EXTERN(cuCtxGetDevice, 2000);
DECLARE_CUDA_PFN_EXTERN(cuPointerGetAttribute, 4000);
DECLARE_CUDA_PFN_EXTERN(cuLaunchKernel, 4000);
#if CUDART_VERSION >= 11080
DECLARE_CUDA_PFN_EXTERN(cuLaunchKernelEx, 11060);
#endif
// cuMem API support
DECLARE_CUDA_PFN_EXTERN(cuMemAddressReserve, 10020);
DECLARE_CUDA_PFN_EXTERN(cuMemAddressFree, 10020);
DECLARE_CUDA_PFN_EXTERN(cuMemCreate, 10020);
DECLARE_CUDA_PFN_EXTERN(cuMemGetAllocationGranularity, 10020);
DECLARE_CUDA_PFN_EXTERN(cuMemExportToShareableHandle, 10020);
DECLARE_CUDA_PFN_EXTERN(cuMemImportFromShareableHandle, 10020);
DECLARE_CUDA_PFN_EXTERN(cuMemMap, 10020);
DECLARE_CUDA_PFN_EXTERN(cuMemRelease, 10020);
DECLARE_CUDA_PFN_EXTERN(cuMemRetainAllocationHandle, 11000);
DECLARE_CUDA_PFN_EXTERN(cuMemSetAccess, 10020);
DECLARE_CUDA_PFN_EXTERN(cuMemUnmap, 10020);
DECLARE_CUDA_PFN_EXTERN(cuMemGetAllocationPropertiesFromHandle, 10020);
#if CUDA_VERSION >= 11070
DECLARE_CUDA_PFN_EXTERN(cuMemGetHandleForAddressRange, 11070); // DMA-BUF support
#endif
#if CUDA_VERSION >= 12010
/* NVSwitch Multicast support */
DECLARE_CUDA_PFN_EXTERN(cuMulticastAddDevice, 12010);
DECLARE_CUDA_PFN_EXTERN(cuMulticastBindMem, 12010);
DECLARE_CUDA_PFN_EXTERN(cuMulticastBindAddr, 12010);
DECLARE_CUDA_PFN_EXTERN(cuMulticastCreate, 12010);
DECLARE_CUDA_PFN_EXTERN(cuMulticastGetGranularity, 12010);
DECLARE_CUDA_PFN_EXTERN(cuMulticastUnbind, 12010);
#endif
/* Stream-MemOp support */
DECLARE_CUDA_PFN_EXTERN(cuStreamBatchMemOp, 11070);
DECLARE_CUDA_PFN_EXTERN(cuStreamWaitValue32, 11070);
DECLARE_CUDA_PFN_EXTERN(cuStreamWaitValue64, 11070);
DECLARE_CUDA_PFN_EXTERN(cuStreamWriteValue32, 11070);
DECLARE_CUDA_PFN_EXTERN(cuStreamWriteValue64, 11070);
#endif

ncclResult_t ncclCudaLibraryInit(void);

extern int ncclCudaDriverVersionCache;
extern bool ncclCudaLaunchBlocking; // initialized by ncclCudaLibraryInit()

inline ncclResult_t ncclCudaDriverVersion(int* driver) {
  int version = __atomic_load_n(&ncclCudaDriverVersionCache, __ATOMIC_RELAXED);
  if (version == -1) {
    CUDACHECK(cudaDriverGetVersion(&version));
    __atomic_store_n(&ncclCudaDriverVersionCache, version, __ATOMIC_RELAXED);
  }
  *driver = version;
  return ncclSuccess;
}
#endif
