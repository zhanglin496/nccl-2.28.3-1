/*************************************************************************
 * Copyright (c) 2015-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

/*
 * init.cc - NCCL 通信初始化核心文件
 *
 * 本文件实现了 NCCL 通信器的初始化、创建、销毁等核心功能。
 * 主要包含以下几个模块：
 * 1. 全局初始化（环境变量、GDRCOPY、Bootstrap网络）
 * 2. 通信器分配和释放（commAlloc、commFree）
 * 3. 传输层初始化（initTransportsRank）
 * 4. 拓扑发现和通道建立
 * 5. API函数实现（ncclCommInitRank、ncclCommDestroy等）
 */

#include "nccl.h"                      // NCCL 公共 API 定义
#include "channel.h"                    // 通信通道相关定义
#include "nvmlwrap.h"                   // NVML（NVIDIA 管理库）封装
#include "gdrwrap.h"                    // GDRCOPY（GPU 直接 RDMA）封装
#include "bootstrap.h"                  // Bootstrap 网络初始化（rank 间握手）
#include "transport.h"                  // 传输层抽象（P2P、网络、共享内存）
#include "group.h"                      // NCCL Group 操作（批量提交 collective 调用）
#include "net.h"                        // 网络插件接口
#include "coll_net.h"                   // 集合网络（NVSwitch 硬件加速）
#include "enqueue.h"                    // 操作入队和内核启动
#include "graph.h"                      // CUDA Graph 集成
#include "argcheck.h"                   // 参数校验宏
#include "tuner.h"                      // 性能调优插件
#include "ras.h"                        // 可靠性、可用性、可服务性
#include "profiler.h"                   // 性能分析器
#include "mnnvl.h"                      // Multi-Node NVLink 支持
#include <fcntl.h>
#include <string.h>
#include <errno.h>
#include <assert.h>
#include <dlfcn.h>                      // 动态链接库操作
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/resource.h>               // 资源限制（如栈大小）
#include <unistd.h>
#include "param.h"                      // 环境变量参数定义
#include "nvtx_payload_schemas.h"       // NVTX（NVIDIA 工具扩展）追踪
#include "utils.h"                      // 工具函数
#include <mutex>
#include "ce_coll.h"                    // 集合引擎（Collective Engine）
#include "nvtx.h"                       // NVTX 接口

// 字符串化宏（用于将宏定义转换为字符串）
#define STR2(v) #v
#define STR(v) STR2(v)

// CUDA 版本相关的编译时配置
// CUDA 9.2+ 不需要使用内部 CUDA stream，CUDA 9.0/9.1 需要
#if CUDART_VERSION >= 9020
#define NCCL_GROUP_CUDA_STREAM 0
#else
#define NCCL_GROUP_CUDA_STREAM 1
#endif

// 集合操作名称字符串数组（用于日志和调试）
const char* ncclFuncStr[NCCL_NUM_FUNCTIONS] = { "Broadcast", "Reduce", "AllGather", "ReduceScatter", "AllReduce" };

// 算法名称字符串数组
const char* ncclAlgoStr[NCCL_NUM_ALGORITHMS] = { "Tree", "Ring", "CollNetDirect", "CollNetChain", "NVLS", "NVLSTree", "PAT" };

// 协议名称字符串数组
// LL: Long Jump protocol（优化中等大小消息）
// LL128: 128字节对齐的 LL 变体
// Simple: 简单协议（适合小消息）
const char* ncclProtoStr[NCCL_NUM_PROTOCOLS] = { "LL", "LL128", "Simple" };

// 环境变量参数：是否在 Group 模式下使用内部 CUDA stream
NCCL_PARAM(GroupCudaStream, "GROUP_CUDA_STREAM", NCCL_GROUP_CUDA_STREAM);

// 环境变量参数定义（通过 NCCL_PARAM 宏定义可通过环境变量配置的参数）
NCCL_PARAM(CheckPointers, "CHECK_POINTERS", 0);              // 是否检查指针有效性
NCCL_PARAM(CommBlocking, "COMM_BLOCKING", NCCL_CONFIG_UNDEF_INT);  // 是否使用阻塞模式
NCCL_PARAM(RuntimeConnect, "RUNTIME_CONNECT", 1);            // 是否运行时建立连接
NCCL_PARAM(WinEnable, "WIN_ENABLE", 1);                      // 是否启用 WIN（CUDA Unified Memory）
NCCL_PARAM(CollnetEnable, "COLLNET_ENABLE", NCCL_CONFIG_UNDEF_INT);  // 是否启用 CollNet
NCCL_PARAM(CtaPolicy, "CTA_POLICY", NCCL_CONFIG_UNDEF_INT);  // CTA（线程块）策略
NCCL_PARAM(NvlsChannels, "NVLS_NCHANNELS", NCCL_CONFIG_UNDEF_INT);  // NVLS 通道数
NCCL_PARAM(SetCpuStackSize, "SET_CPU_STACK_SIZE", 1);        // 是否设置 CPU 栈大小

extern int64_t ncclParamSingleProcMemRegEnable();

static ncclResult_t commReclaim(ncclComm_t comm);

// GDRCOPY 支持：默认关闭
// GDRCOPY 允许 CPU 直接访问 GPU 内存，用于某些特定的优化场景
NCCL_PARAM(GdrCopyEnable, "GDRCOPY_ENABLE", 0);

// GDRCOPY 全局句柄
gdr_t ncclGdrCopy = NULL;

// 初始化 GDRCOPY 功能
ncclResult_t initGdrCopy() {
  // 检查是否通过环境变量启用了 GDRCOPY
  if (ncclParamGdrCopyEnable() == 1) {
    ncclGdrCopy = ncclGdrInit();  // 初始化 GDRCOPY
  }
  return ncclSuccess;
}

// 安全的栈大小定义为 8MB
// Linux 默认栈大小是 8MB，但如果设置为 unlimited，GNU libc 会使用 2MB 作为默认值，这不够安全
#define SAFE_STACK_SIZE (8192*1024)

// 设置 CPU 线程栈大小
// NCCL 创建辅助线程时需要确保栈空间足够大，避免栈溢出
static ncclResult_t setCpuStackSize() {
  if (ncclParamSetCpuStackSize() != 0) {
    pthread_attr_t attr;
    size_t stackSize;
    // 初始化线程属性
    PTHREADCHECK(pthread_attr_init(&attr), "pthread_attr_init");
    // 获取当前默认栈大小
    PTHREADCHECK(pthread_attr_getstacksize(&attr, &stackSize), "pthread_attr_getstacksize");

    // 如果栈大小小于安全值（8MB），需要调整
    if (stackSize < SAFE_STACK_SIZE) {
      // GNU libc 通常使用 RLIMIT_STACK 作为默认 pthread 栈大小
      // 除非设置为 unlimited，此时会回退到 2MB
      struct rlimit stackLimit;
      char buf[30];
      // 查询实际的资源限制
      SYSCHECK(getrlimit(RLIMIT_STACK, &stackLimit), "getrlimit");
      if (stackLimit.rlim_cur == RLIM_INFINITY)
        strcpy(buf, "unlimited");
      else
        snprintf(buf, sizeof(buf), "%ldKB", stackLimit.rlim_cur/1024);
      INFO(NCCL_INIT|NCCL_ENV, "Stack size limit (%s) is unsafe; will use %dKB for newly launched threads",
           buf, SAFE_STACK_SIZE/1024);

      // 修改默认 pthread 栈大小（使用非标准但广泛支持的 API）
      PTHREADCHECK(pthread_attr_setstacksize(&attr, SAFE_STACK_SIZE), "pthread_attr_setstacksize");
      PTHREADCHECK(pthread_setattr_default_np(&attr), "pthread_setattr_default_np");
    }

    PTHREADCHECK(pthread_attr_destroy(&attr), "pthread_attr_destroy");
  }

  return ncclSuccess;
}

// 全局初始化结果和标志
static ncclResult_t initResult = ncclSuccess;        // 初始化结果
static std::once_flag initOnceFlag;                  // 用于确保只初始化一次的标志

// 全局初始化函数（只执行一次）
static void initOnceFunc() {
  // 1. 根据配置文件设置环境变量
  initEnv();

  // 2. 设置线程栈大小为 8MB，防止栈溢出
  setCpuStackSize();

  // 3. 初始化 GDRCOPY 支持（如果启用）
  // GDRCOPY 允许 CPU 直接访问 GPU 内存，用于某些优化场景
  initGdrCopy();

  // 4. 为 NCCL 的 Bootstrap 阶段选定一张本机监听网卡 + 地址，并登记到全局变量
  //    Bootstrap 是 NCCL 用于 rank 间握手和交换初始信息的机制
  // Always initialize bootstrap network
  NCCLCHECKGOTO(bootstrapNetInit(), initResult, exit);

  // 5. 在首次规约前把 ncclRedOp_t 的数值→字符串 表注册给 NVTX
  //    使得后续所有 NCCL reduction 的抓痕能直接显示 "Sum/Prod/Max" 等可读枚举名
  //    而非裸数字，对性能零影响，对调试极友好
  initNvtxRegisteredEnums();
exit:;
}

// NCCL 库初始化入口
// 使用 std::call_once 确保在进程生命周期内只执行一次
static ncclResult_t ncclInit() {
  std::call_once(initOnceFlag, initOnceFunc);
  return initResult;
}

// 获取 NCCL 版本号
NCCL_API(ncclResult_t, ncclGetVersion, int* version);
ncclResult_t ncclGetVersion(int* version) {
  if (version == NULL)
    return ncclInvalidArgument;
  // NCCL_VERSION_CODE 在编译时生成，格式为 Major*10000 + Minor*100 + Patch
  *version = NCCL_VERSION_CODE;
  return ncclSuccess;
}

// 获取唯一的通信 ID
// 这个 ID 用于在多个进程/节点间建立通信连接
// root进程调用 ncclGetUniqueId 获取 ID，然后通过其他机制（如 MPI）传递给其他进程
NCCL_API(ncclResult_t, ncclGetUniqueId, ncclUniqueId* out);
ncclResult_t ncclGetUniqueId(ncclUniqueId* out) {
  // 首先确保 NCCL 已经初始化
  NCCLCHECK(ncclInit());
  NCCLCHECK(PtrCheck(out, "GetUniqueId", "out"));

  struct ncclBootstrapHandle handle;
  // 获取一个 IP 监听地址和 magic 到 handle
  // bootstrapGetUniqueId 会在本机上选择一个可用的网络接口
  NCCLCHECK(bootstrapGetUniqueId(&handle));

  // ncclUniqueId 和 bootstrapHandle 的内存布局不同，需要清零避免未定义数据
  memset(out, 0, sizeof(*out));
  // 拷贝 handle 到 out，避免对齐问题
  memcpy(out, &handle, sizeof(handle));

  // 根据 handle 选择的 IP 地址+端口+magic 的值 hash 算出一个 8 字节的 ID
  TRACE_CALL("ncclGetUniqueId(0x%llx)", (unsigned long long)getHash(out->internal, NCCL_UNIQUE_ID_BYTES));
  return ncclSuccess;
}

// 防止编译器优化掉这些操作（用于调试）
#ifdef __clang__
#define NCCL_NO_OPTIMIZE __attribute__((optnone))
#else
#define NCCL_NO_OPTIMIZE __attribute__((optimize("O0")))
#endif
// "毒化" comm 结构体，在释放前将关键字段设为无效值
// 这有助于检测 use-after-free 等内存错误
void NCCL_NO_OPTIMIZE commPoison(ncclComm_t comm) {
  // 注意：不要破坏 intraComm0
  comm->rank = comm->cudaDev = comm->busId = comm->nRanks = -1;
  comm->startMagic = comm->endMagic = 0;
}

#undef NCCL_NO_OPTIMIZE

// ============================================================================
// 析构函数（Destructor）机制
// NCCL 使用析构函数链来管理资源的自动释放
// 当 comm 被销毁时，会按 LIFO 顺序调用所有注册的析构函数
// ============================================================================

// 释放通过 malloc 分配的内存
static ncclResult_t ncclDestructorFnFree(struct ncclDestructor* dtor) {
  free(dtor->obj);
  return ncclSuccess;
}

// 注册一个需要在 comm 销毁时释放的普通内存
void ncclCommPushFree(struct ncclComm* comm, void* obj) {
  // 从 comm 的永久内存栈中分配析构函数结构
  struct ncclDestructor* dtor = ncclMemoryStackAlloc<struct ncclDestructor>(&comm->memPermanent);
  dtor->fn = ncclDestructorFnFree;
  dtor->obj = obj;
  dtor->next = comm->destructorHead;
  comm->destructorHead = dtor;
}

// 释放 CUDA 设备内存
static ncclResult_t ncclDestructorFnCudaFree(struct ncclDestructor* dtor) {
  NCCLCHECK(ncclCudaFree(dtor->obj));
  return ncclSuccess;
}

// 注册一个需要在 comm 销毁时释放的 CUDA 设备内存
void ncclCommPushCudaFree(struct ncclComm* comm, void* obj) {
  struct ncclDestructor* dtor = ncclMemoryStackAlloc<struct ncclDestructor>(&comm->memPermanent);
  dtor->fn = ncclDestructorFnCudaFree;
  dtor->obj = obj;
  dtor->next = comm->destructorHead;
  comm->destructorHead = dtor;
}

// 释放 CUDA 锁页主机内存（cudaHostAlloc 分配的内存）
static ncclResult_t ncclDestructorFnCudaHostFree(struct ncclDestructor* dtor) {
  NCCLCHECK(ncclCudaHostFree(dtor->obj));
  return ncclSuccess;
}

// 注册一个需要在 comm 销毁时释放的 CUDA 锁页主机内存
void ncclCommPushCudaHostFree(struct ncclComm* comm, void* obj) {
  struct ncclDestructor* dtor = ncclMemoryStackAlloc<struct ncclDestructor>(&comm->memPermanent);
  dtor->fn = ncclDestructorFnCudaHostFree;
  dtor->obj = obj;
  dtor->next = comm->destructorHead;
  comm->destructorHead = dtor;
}

// 释放 GDRCOPY 映射的 CUDA 内存
static ncclResult_t ncclDestructorFnCudaGdrFree(struct ncclDestructor* dtor) {
  NCCLCHECK(ncclGdrCudaFree(dtor->obj));
  return ncclSuccess;
}

// 注册一个需要在 comm 销毁时释放的 GDRCOPY 内存
void ncclCommPushCudaGdrFree(struct ncclComm* comm, void* handle) {
  struct ncclDestructor* dtor = ncclMemoryStackAlloc<struct ncclDestructor>(&comm->memPermanent);
  dtor->fn = ncclDestructorFnCudaGdrFree;
  dtor->obj = handle;
  dtor->next = comm->destructorHead;
  comm->destructorHead = dtor;
}

// ============================================================================
// commFree - 释放通信器资源
// ============================================================================
static ncclResult_t commFree(ncclComm_t comm) {
  int abort = 0;
  /* commFree() 不应涉及任何 rank 间的同步操作 */

  if (comm == NULL)
    return ncclSuccess;

  // 完成集合引擎（Collective Engine）的清理
  NCCLCHECK(ncclCeFinalize(comm));

  // 清理对称内核相关资源
  if (comm->symmetricSupport) {
    NCCLCHECK(ncclSymkFinalize(comm));
    NCCLCHECK(ncclDevrFinalize(comm));
  }
  // 清理 RAS（可靠性、可用性、可服务性）资源
  NCCLCHECK(ncclRasCommFini(comm));

  /* 在 commReclaim 中，我们已经保证只有最后一个调用 ncclCommDestroy() 的 rank 会释放所有进程内通信器；
   * 因此我们在 commFree() 中只需要关注本地资源的清理。 */

  // 如果 proxy 线程存在且是原始 owner，则等待 proxy 线程退出
  if (comm->proxyState && comm->proxyRefCountOld == 0 && comm->proxyState->thread) {
    PTHREADCHECK(pthread_join(comm->proxyState->thread, nullptr), "pthread_join");
    if (comm->proxyState->threadUDS) {
      // UDS（Unix Domain Socket）支持
      PTHREADCHECK(pthread_join(comm->proxyState->threadUDS, nullptr), "pthread_join");
    }
  }

  // 销毁 CUDA 内存池
  if (comm->memPool) CUDACHECK(cudaMemPoolDestroy(comm->memPool));

  // 释放用户自定义 reduction 操作的资源
  delete[] comm->userRedOps;

  // 释放连接相关的位图数组
  free(comm->connectSend);
  free(comm->connectRecv);

  // 释放 peer 信息
  free(comm->peerInfo);

  // 释放拓扑信息
  if (comm->topo)
    ncclTopoFree(comm->topo);

  // 释放节点 rank 映射信息
  if (comm->nodeRanks) {
    for (int n=0; n<comm->nNodes; n++) free(comm->nodeRanks[n].localRankToRank);
    free(comm->nodeRanks);
  }
  free(comm->rankToNode);
  free(comm->rankToLocalRank);
  free(comm->collNetHeads);
  free(comm->clique.ranks);

  // 关闭 Bootstrap 连接
  if (comm->bootstrap)
    NCCLCHECK(bootstrapClose(comm->bootstrap));

  // 释放所有通道的资源
  for (int channel=0; channel<MAXCHANNELS; channel++)
    NCCLCHECK(freeChannel(comm->channels+channel, comm->nRanks, 1, comm->localRanks));

  // 释放共享资源
  if (comm->sharedRes) {
    // 使用引用计数，只有最后一个引用者才真正释放
    if (ncclAtomicRefCountDecrement(&comm->sharedRes->refCount) == 0) {
      for (int c=0; c<MAXCHANNELS; c++) {
        if (comm->sharedRes->peers[c]) free(comm->sharedRes->peers[c]);
        if (comm->sharedRes->devPeers[c]) ncclCudaFree(comm->sharedRes->devPeers[c]);
      }
      free(comm->sharedRes->tpRankToLocalRank);
      NCCLCHECK(ncclStrongStreamDestruct(&comm->sharedRes->hostStream));
      NCCLCHECK(ncclStrongStreamDestruct(&comm->sharedRes->deviceStream));
      CUDACHECK(cudaEventDestroy(comm->sharedRes->launchEvent));
      CUDACHECK(cudaEventDestroy(comm->sharedRes->scratchEvent));
      NCCLCHECK(ncclProxyDestroy(comm));
      free(comm->sharedRes);
    }
  }

  // 释放 NVLS（NVLink Sharp）资源
  if (comm->nvlsSupport) NCCLCHECK(ncclNvlsFree(comm));

  // 调用所有注册的析构函数
  struct ncclDestructor* dtor = comm->destructorHead;
  while (dtor != nullptr) {
    NCCLCHECK(dtor->fn(dtor));
    dtor = dtor->next;
  }

  // 销毁内存栈
  ncclMemoryStackDestruct(&comm->memScoped);
  ncclMemoryStackDestruct(&comm->memPermanent);

  // 释放 abort 标志
  abort = *comm->abortFlag;
  if (ncclAtomicRefCountDecrement(comm->abortFlagRefCount) == 0) {
    free(comm->abortFlag);
    NCCLCHECK(ncclCudaHostFree((void*)comm->abortFlagDev));
    free(comm->abortFlagRefCount);
  }
  free((void*)comm->config.netName);

  // 释放其他资源
  free(comm->topParentRanks);
  free(comm->topParentLocalRanks);
  free(comm->gproxyConn);

  // 清理内存注册缓存
  NCCLCHECK(ncclRegCleanup(comm));

  INFO(NCCL_INIT,"comm %p rank %d nranks %d cudaDev %d busId %lx - %s COMPLETE", comm, comm->rank, comm->nRanks, comm->cudaDev, comm->busId, abort ? "Abort" : "Destroy");

  // 在释放前"毒化" comm，防止 comm 被重用
  commPoison(comm);
  NCCLCHECK(ncclProfilerPluginFinalize(comm));
  NCCLCHECK(ncclNetFinalize(comm));
  ncclCudaContextDrop(comm->context);
  free(comm);

  return ncclSuccess;
}

// 环境变量参数定义
NCCL_PARAM(DisableGraphHelper, "GRAPH_HELPER_DISABLE", 0);
// GDRCOPY 支持：FIFO_ENABLE 启用时将 workFifo 定位在 CUDA 内存中
NCCL_PARAM(GdrCopyFifoEnable, "GDRCOPY_FIFO_ENABLE", 1);
#define NCCL_WORK_FIFO_BYTES_DEFAULT (1<<20)  // 默认 1MB
NCCL_PARAM(WorkFifoBytes, "WORK_FIFO_BYTES", NCCL_WORK_FIFO_BYTES_DEFAULT);
NCCL_PARAM(WorkArgsBytes, "WORK_ARGS_BYTES", INT64_MAX);
enum ncclLaunchMode ncclParamLaunchMode;

NCCL_PARAM(DmaBufEnable, "DMABUF_ENABLE", 1);

// ============================================================================
// DMA-BUF 支持检测
// DMA-BUF 是 Linux 内核的一种机制，允许设备间零拷贝共享内存
// ============================================================================
static ncclResult_t dmaBufSupported(struct ncclComm* comm) {
  // 检查是否通过环境变量启用了 DMA-BUF
  if (ncclParamDmaBufEnable() == 0 || comm->ncclNet->regMrDmaBuf == NULL || ncclCudaLibraryInit() != ncclSuccess)
    return ncclInternalError;
#if CUDA_VERSION >= 11070
  int flag = 0;
  CUdevice dev;
  int cudaDriverVersion;
  CUDACHECK(cudaDriverGetVersion(&cudaDriverVersion));
  if (CUPFN(cuDeviceGet) == NULL || cudaDriverVersion < 11070) return ncclInternalError;
  CUCHECK(cuDeviceGet(&dev, comm->cudaDev));
  // 查询设备是否支持 DMA-BUF
  (void) CUPFN(cuDeviceGetAttribute(&flag, CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED, dev));
  if (flag == 0)
    return ncclInternalError;
  INFO(NCCL_INIT, "DMA-BUF is available on GPU device %d", comm->cudaDev);
  return ncclSuccess;
#endif
  return ncclInternalError;
}

// ============================================================================
// ncclCommEnsureReady - 确保通信器已就绪
// 检查通信器是否可以正常使用，如果出现错误则报告
// ============================================================================
ncclResult_t ncclCommEnsureReady(ncclComm_t comm) {
  /* comm 必须就绪，否则会报告错误 */
  ncclResult_t ret = ncclSuccess;
  // 检查 abort 标志
  if (__atomic_load_n(comm->abortFlag, __ATOMIC_ACQUIRE)) {
    ncclGroupJobAbort(comm->groupJob);
  } else {
    // 获取异步错误状态
    NCCLCHECK(ncclCommGetAsyncError(comm, &ret));
    if (ret == ncclInProgress) {
      WARN("Attempt to use communicator before the previous operation returned ncclSuccess");
      ret = ncclInvalidArgument;
      goto exit;
    }
    /* 如果 ret 不是 ncclInProgress，我们保持它 */
  }

exit:
  return ret;
}

// ============================================================================
// commAlloc - 分配并初始化通信器结构
// ============================================================================
//
// 这个函数是 NCCL 通信器创建的第一步，负责初始化基本的通信器结构
//
// 函数调用时机：
// - ncclCommInitRank：创建新的通信器
// - ncclCommSplit：从父通信器分割出子通信器
//
// 主要功能：
// 1. 参数验证（设备数量、rank 编号）
// 2. 初始化内存管理系统
// 3. 加载网络插件
// 4. 获取 CUDA 设备信息
// 5. 设置共享资源（父 comm 复用或创建新的）
// 6. 初始化 CUDA 流和事件
//
// 参数说明：
// - comm：要初始化的通信器结构（调用者已分配内存）
// - parent：父通信器（如果是分割创建的子 comm），NULL 表示根 comm
// - ndev：设备数量（总 rank 数量）
// - rank：当前 rank 的编号（0 到 ndev-1）
//
// ============================================================================
static ncclResult_t commAlloc(struct ncclComm* comm, struct ncclComm* parent, int ndev, int rank) {
  // ============================================================
  // 参数验证
  // ============================================================
  // ndev：设备数量（GPU 数量）
  // rank：当前 rank 的编号（0 到 ndev-1）
  //
  // 这些验证确保参数在合法范围内
  // 避免后续访问数组越界或逻辑错误
  // ============================================================

  if (ndev < 1) {
    WARN("invalid device count (%d) requested", ndev);
    return ncclInvalidArgument;
  }
  if (rank >= ndev || rank < 0) {
    WARN("rank %d exceeds ndev=%d", rank, ndev);
    return ncclInvalidArgument;
  }

  // ============================================================
  // 初始化内存栈（用于管理 comm 的内存分配）
  // ============================================================
  //
  // NCCL 使用内存栈（Memory Stack）来管理动态分配的内存
  //
  // 为什么使用内存栈而不是直接 malloc/free？
  // 1. 批量释放：一次性释放整个栈，无需逐个 free
  // 2. 减少碎片：栈式分配减少内存碎片
  // 3. 性能：栈分配比堆分配更快
  // 4. 简化：无需跟踪每个分配的生命周期
  //
  // memPermanent（永久内存栈）：
  // - 存储 comm 生命周期内一直需要的数据
  // - 例如：通道配置、拓扑信息、rank 映射表等
  // - 在 comm 销毁时释放
  //
  // memScoped（作用域内存栈）：
  // - 存储临时数据，可以在某个操作完成后释放
  // - 例如：临时缓冲区、中间计算结果等
  // - 可以在特定时间点批量释放
  //
  // 内存栈的工作原理：
  //   分配时：栈指针向上移动，返回新地址
  //   释放时：栈指针回退到之前的标记位置
  //   销毁时：释放整个栈占用的内存块
  //
  // ============================================================

  ncclMemoryStackConstruct(&comm->memPermanent);  // 永久内存（comm 生命周期内一直有效）
  ncclMemoryStackConstruct(&comm->memScoped);     // 作用域内存（可以释放）
  comm->destructorHead = nullptr;                 // 析构函数链表头（用于资源清理）

  // ============================================================
  // 设置基本信息
  // ============================================================
  // rank：当前 rank 的全局编号（在整个通信域中唯一）
  // nRanks：通信域中的总 rank 数量
  //
  // 这些信息在后续的拓扑计算、通道配置中都会用到
  // ============================================================

  comm->rank = rank;          // 当前 rank 的编号
  comm->nRanks = ndev;        // 总 rank 数量

  // ============================================================
  // 初始化网络插件
  // ============================================================
  //
  // NCCL 支持网络插件系统，允许第三方提供网络传输实现
  //
  // ncclNetInit() 的工作流程：
  // 1. 尝试加载网络插件（libnccl-net.so）
  // 2. 如果环境变量 NCCL_NET_PLUGIN 指定了插件，优先加载
  // 3. 否则，尝试加载内置的网络实现（Socket、IB 等）
  // 4. 按顺序尝试，第一个成功的插件被选中
  //
  // 设置 comm 中的字段：
  // - ncclNet：网络插件接口函数表
  // - ncclCollNet：集合网络插件（可选，可能为 NULL）
  //
  // 网络插件提供的功能：
  // - 发送/接收数据
  // - 建立连接
  // - 处理 RDMA 操作
  // - 管理 DMA 内存
  //
  // ============================================================

  NCCLCHECK(ncclNetInit(comm));

  // 记录使用的网络类型
  INFO(NCCL_INIT, "Using network %s", comm->ncclNet->name);

  // ============================================================
  // 网络类型一致性检查
  // ============================================================
  //
  // 如果是子通信（从 ncclCommSplit 创建）且共享资源：
  // - 必须使用与父 comm 相同的网络插件
  // - 因为共享资源包括网络连接和状态
  // - 不同的网络插件无法共享这些资源
  //
  // 例如：
  // - 父 comm 使用 IB 插件
  // - 子 comm 也必须使用 IB 插件
  // - 否则无法共享网络连接
  //
  // ============================================================

  if (parent && parent->shareResources) {
    if (parent->ncclNet != comm->ncclNet) {
      WARN("Split shares resources, but parent comm netName %s is different from child comm netName %s", parent->ncclNet->name, comm->ncclNet->name);
      return ncclInvalidUsage;
    }
  }

  // ============================================================
  // 立即创建 CUDA 对象以验证设备状态
  // ============================================================
  //
  // 这一步很关键，尽早验证 CUDA 设备可用性
  //
  // 常见问题 #1：设备被其他进程占用
  // 常见问题 #2：CUDA 驱动版本不匹配
  // 常见问题 #3：GPU 处于异常状态
  //
  // 早点发现这些问题可以：
  // 1. 避免浪费初始化时间
  // 2. 提供更清晰的错误信息
  // 3. 允许应用程序提前失败
  //
  // cudaGetDevice：获取当前 CUDA 设备编号
  // - 这是调用线程当前关联的设备
  // - 必须与 comm->rank 对应的设备一致
  //
  // ncclCudaContextTrack：跟踪 CUDA 上下文
  // - 获取当前 CUDA 上下文
  // - 保存到 comm->context 中
  // - 用于后续的 CUDA 操作
  //
  // ============================================================

  CUDACHECK(cudaGetDevice(&comm->cudaDev));    // 获取当前 CUDA 设备

  NCCLCHECK(ncclCudaContextTrack(&comm->context));  // 获取并跟踪 CUDA 上下文

  // ============================================================
  // 获取 GPU 设备的 bus ID（PCI 地址）
  // ============================================================
  //
  // bus ID 是 GPU 在 PCIe 总线上的地址
  // 格式：Domain:Bus:Device.Function（如 0000:04:00.0）
  //
  // bus ID 的用途：
  // 1. 拓扑发现：确定 GPU 之间的物理连接关系
  // 2. P2P 通信：判断两个 GPU 是否可以直接通信
  // 3. NUMA 亲和性：确定 GPU 与 CPU 的亲和关系
  // 4. NVLink 检测：判断 GPU 之间是否有 NVLink 连接
  //
  // getBusId：将 cudaDev 转换为整数形式的 bus ID
  // - 整数形式便于比较和计算
  // - 存储 64 位整数，足够表示完整的 PCI 地址
  //
  // ============================================================

  NCCLCHECK(getBusId(comm->cudaDev, &comm->busId));

  // ============================================================
  // 获取 NVML 设备句柄和索引
  // ============================================================
  //
  // NVML (NVIDIA Management Library) 是 NVIDIA 的管理库
  //
  // 为什么要使用 NVML？
  // 1. 获取 CUDA API 不提供的信息（如温度、功耗）
  // 2. 获取设备的持久索引（nvmlDev）
  // 3. 设备管理和监控
  //
  // nvmlDev 和 cudaDev 的区别：
  // - cudaDev：CUDA 运行时的设备编号（0, 1, 2, ...）
  // - nvmlDev：NVML 的设备索引，通常是持久的
  //
  // 持久索引的重要性：
  // - cudaDev 可能因 GPU 热插拔而改变
  // - nvmlDev 在系统重启前保持不变
  // - 用于跨进程的设备识别
  //
  // 转换流程：
  // 1. cudaDev → busId（通过 getBusId）
  // 2. busId → 字符串（通过 int64ToBusId）
  // 3. 字符串 → nvmlDev（通过 ncclNvmlDeviceGetHandleByPciBusId）
  // 4. nvmlDev → nvmlDev 索引（通过 ncclNvmlDeviceGetIndex）
  //
  // ============================================================

  nvmlDevice_t nvmlDev;                                    // NVML 设备句柄
  char busId[NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE];          // busId 字符串缓冲区
  // 将 busId 转换为字符串
  NCCLCHECK(int64ToBusId(comm->busId, busId));
  // 通过 bus ID 获取 NVML 设备句柄
  NCCLCHECK(ncclNvmlDeviceGetHandleByPciBusId(busId, &nvmlDev));
  NCCLCHECK(ncclNvmlDeviceGetIndex(nvmlDev, (unsigned int*)&comm->nvmlDev));

  // ============================================================
  // 获取 GPU 计算能力
  // ============================================================
  //
  // 计算能力（Compute Capability）表示 GPU 的架构代数
  //
  // 格式：XY（如 70、75、80、86、90）
  // - X：主版本号（架构代数）
  // - Y：次版本号（架构更新）
  //
  // 常见计算能力：
  // - 70：Volta (V100)
  // - 75：Turing (T4, RTX 2080)
  // - 80：Ampere (A100)
  // - 86：Ampere (RTX 3090)
  // - 90：Hopper (H100)
  //
  // 计算能力的用途：
  // 1. 选择合适的内核实现
  // 2. 确定支持的 CUDA 特性
  // 3. 性能调优和优化
  //
  // TRACE：调试级别的日志输出
  // - 包含 comm 的关键信息
  // - 用于调试和问题诊断
  //
  // ============================================================

  // 获取 GPU 计算能力（如 70、75、80 等）
  comm->compCap = ncclCudaCompCap();
  TRACE(NCCL_INIT,"comm %p rank %d nranks %d cudaDev %d busId %lx compCap %d", comm, rank, ndev, comm->cudaDev, comm->busId, comm->compCap);

  // ============================================================
  // 其他配置
  // ============================================================
  //
  // checkPointers：检查指针的有效性
  // - 环境变量：NCCL_CHECK_POINTERS
  // - 用于调试，检测无效的内存访问
  // - 生产环境通常关闭（性能影响）
  //
  // dmaBufSupport：DMA 缓冲区支持
  // - 检测是否支持 DMA 缓冲区
  // - DMA 可以减少内存拷贝，提高性能
  // - 依赖硬件和驱动支持
  //
  // ============================================================

  comm->checkPointers = ncclParamCheckPointers() == 1 ? true : false;
  comm->dmaBufSupport = (dmaBufSupported(comm) == ncclSuccess) ? true : false;

  // ============================================================
  // 清零 CollNet 支持矩阵
  // ============================================================
  //
  // CollNet (Collective Network) 支持矩阵
  //
  // 这个矩阵记录哪些 GPU 对之间支持 CollNet
  // - 二维矩阵：nRanks × nRanks
  // - collNetSupportMatrix[i][j]：rank i 和 rank j 之间的 CollNet 支持
  //
  // 初始化为 0：
  // - 默认不支持 CollNet
  // - 后续根据硬件检测更新
  //
  // CollNet 是什么？
  // - 使用专用网络硬件的集合通信
  // - 例如：BlueField DPU、NVSwitch
  // - 比传统 Ring/Tree 算法更快
  //
  // ============================================================

  memset(comm->collNetSupportMatrix, 0, sizeof(comm->collNetSupportMatrix));

  // ============================================================
  // 构造内存池（用于内核计划）
  // ============================================================
  //
  // NCCL 使用内存池来管理特定类型的内存分配
  //
  // memPool_ncclKernelPlan：
  // - 用于内核计划（Kernel Planner）的内存池
  // - 存储内核执行计划和相关信息
  //
  // memPool_ncclProxyOp：
  // - 用于 Proxy 操作的内存池
  // - 存储 Proxy 线程的请求和响应
  //
  // 内存池的优势：
  // 1. 减少分配/释放开销
  // 2. 提高内存局部性
  // 3. 简化内存管理
  //
  // ============================================================

  ncclMemoryPoolConstruct(&comm->memPool_ncclKernelPlan);
  ncclMemoryPoolConstruct(&comm->memPool_ncclProxyOp);

  // ============================================================
  // 初始化 group 任务链表
  // ============================================================
  //
  // NCCL 支持组操作（Group Calls）：
  // - 多个通信操作可以一起提交
  // - 例如：ncclGroupStart() ... ncclAllReduce() ... ncclGroupEnd()
  //
  // groupNext 数组：
  // - 每个 ncclGroupTaskType 对应一个链表
  // - 链接所有参与组操作的 comm
  //
  // 特殊值 0x1：
  // - 表示链表头部的未初始化状态
  // - NULL (0x0) 是有效的链表结尾
  // - 0x1 明确表示"未初始化"
  //
  // ncclGroupTaskTypeNum：
  // - 组操作类型的数量
  //
  // preconnectNext：
  // - 预连接链表
  // - 用于跟踪需要预连接的 comm
  //
  // ============================================================

  // 初始化 group 任务链表头节点，设置为特殊值0x1，表示为初始化状态
  for (int i = 0; i < ncclGroupTaskTypeNum; i++) {
    comm->groupNext[i] = reinterpret_cast<struct ncclComm*>(0x1);
  }
  comm->preconnectNext = reinterpret_cast<struct ncclComm*>(0x1);

  // ============================================================
  // 验证位图数组大小是否足够
  // ============================================================
  //
  // connectSend 和 connectRecv 是位图数组
  // - 每个 rank 对应一个 uint64_t（8 字节，64 位）
  // - 每个位表示一个通道的连接状态
  //
  // MAXCHANNELS：
  // - 最大通道数（通常为 64）
  // - 不能超过 uint64_t 的位数
  //
  // static_assert：
  // - 编译时断言
  // - 如果条件不满足，编译失败
  // - 确保在编译时发现配置错误
  //
  // ============================================================

  // 验证位图数组大小是否足够，8个字节，64个bit位
  static_assert(MAXCHANNELS <= sizeof(*comm->connectSend)*8, "comm->connectSend must have enough bits for all channels");
  static_assert(MAXCHANNELS <= sizeof(*comm->connectRecv)*8, "comm->connectRecv must have enough bits for all channels");

  // ============================================================
  // 分配连接位图数组
  // ============================================================
  //
  // connectSend[i]：第 i 个 rank 的发送连接位图
  // - 第 j 位为 1：向 rank j 的第 j 个通道发送已连接
  // - 第 j 位为 0：未连接
  //
  // connectRecv[i]：第 i 个 rank 的接收连接位图
  // - 第 j 位为 1：从 rank j 的第 j 个通道接收已连接
  // - 第 j 位为 0：未连接
  //
  // 位图的使用：
  // - 快速查询某个通道是否已连接
  // - 批量设置连接状态
  // - 减少内存占用（相比布尔数组）
  //
  // 大小：nRanks 个 uint64_t
  // - 每个 rank 一个位图
  // - 每个位图记录与所有其他 rank 的连接状态
  //
  // ============================================================

  // 分配连接位图数组（记录每个 rank 的哪些通道已连接）
  // 大小为 8 字节 * nranks
  NCCLCHECK(ncclCalloc(&comm->connectSend, comm->nRanks));
  NCCLCHECK(ncclCalloc(&comm->connectRecv, comm->nRanks));

  // ============================================================
  // 标记通道为未初始化状态
  // ============================================================
  //
  // NCCL 最多支持 MAXCHANNELS 个通道
  // - 每个通道是一个独立的通信路径
  // - 通道可以并行工作，提高带宽
  //
  // channels[c].id：
  // - 通道编号（0 到 nChannels-1）
  // - -1 表示未初始化
  //
  // 初始化为 -1：
  // - 标记所有通道为未使用状态
  // - 后续根据拓扑需求激活部分通道
  // - 未使用的通道不占用资源
  //
  // ============================================================

  // 标记通道为未初始化状态
  for (int c=0; c < MAXCHANNELS; c++)
    comm->channels[c].id = -1;

  // ============================================================
  // 处理共享资源（用于 ncclCommSplit 等场景）
  // ============================================================
  //
  // 共享资源（Shared Resources）机制：
  //
  // 背景：
  // - ncclCommSplit 可以从父 comm 创建子 comm
  // - 子 comm 可以选择共享父 comm 的某些资源
  // - 共享资源可以提高效率和减少开销
  //
  // 两种模式：
  //
  // 1. 创建新的共享资源（parent == NULL 或 !parent->shareResources）：
  //    - 这是一个独立的 comm，不共享资源
  //    - 需要创建新的 SharedResources 结构
  //    - 分配新的 CUDA 流、事件等资源
  //
  // 2. 复用父 comm 的共享资源（parent && parent->shareResources）：
  //    - 这是一个子 comm，与父 comm 共享资源
  //    - 只需增加引用计数
  //    - 避免重复创建资源
  //
  // 共享的资源包括：
  // - CUDA 流（deviceStream、hostStream）
  // - CUDA 事件（launchEvent、scratchEvent）
  // - Proxy 线程和状态
  // - 内存池
  // - 拓扑信息（tpRankToLocalRank 等）
  //
  // 引用计数（refCount）：
  // - 记录有多少个 comm 在使用这个 SharedResources
  // - 创建时设为 1
  // - 每增加一个使用者，引用计数 +1
  // - comm 销毁时，引用计数 -1
  // - 引用计数为 0 时，释放 SharedResources
  //
  // owner 字段：
  // - 记录是哪个 comm 创建了这个 SharedResources
  // - 用于调试和资源清理
  //
  // ============================================================

  if (parent == NULL || !parent->shareResources) {
    // ============================================================
    // 创建新的共享资源
    // ============================================================

    struct ncclSharedResources* sharedRes = NULL;
    NCCLCHECK(ncclCalloc(&sharedRes, 1));

    /* 大部分属性稍后在 initTransportsRank() 中设置 */
    //记录是那个通信器创建了这个sharedRes
    sharedRes->owner = comm;           // 记录谁拥有这个 comm
    sharedRes->tpNRanks = comm->nRanks;  // 记录总的 ranks 数
    NCCLCHECK(ncclCalloc(&sharedRes->tpRankToLocalRank, comm->nRanks));

    // ============================================================
    // 创建 CUDA 流
    // ============================================================
    //
    // deviceStream：设备流
    // - 用于在设备上执行操作
    // - 例如：内核启动、内存拷贝
    //
    // hostStream：主机流
    // - 用于主机端的操作
    // - 例如：回调、内存管理
    //
    // ncclStrongStream：
    // - NCCL 的强类型流封装
    // - 支持 CUDA Graph
    // - 提供更强的类型安全
    //
    // ============================================================

    // 创建 2 个 CUDA 流（用于内核启动）
    NCCLCHECK(ncclStrongStreamConstruct(&sharedRes->deviceStream));
    NCCLCHECK(ncclStrongStreamConstruct(&sharedRes->hostStream));

    // ============================================================
    // 创建 CUDA 事件
    // ============================================================
    //
    // launchEvent：启动事件
    // - 用于标记内核启动的完成
    //
    // scratchEvent：临时事件
    // - 用于各种临时同步需求
    //
    // cudaEventDisableTiming：
    // - 禁用时间记录
    // - 减少事件创建和记录的开销
    // - NCCL 只需要同步，不需要计时
    //
    // ============================================================

    // 创建 2 个 CUDA event
    CUDACHECK(cudaEventCreateWithFlags(&sharedRes->launchEvent, cudaEventDisableTiming));
    CUDACHECK(cudaEventCreateWithFlags(&sharedRes->scratchEvent, cudaEventDisableTiming));

    comm->sharedRes = sharedRes;
    sharedRes->refCount = 1;  // 引用计数设为 1
  } else {
    // ============================================================
    // 和 parent 共享相同的资源
    // ============================================================
    //
    // 子 comm 复用父 comm 的 SharedResources
    // 只需增加引用计数
    //
    // 原子操作：
    // - 使用原子操作确保线程安全
    // - 多个线程可能同时创建子 comm
    //
    // ============================================================

    // 和 parent 共享相同的资源
    comm->sharedRes = parent->sharedRes;
    ncclAtomicRefCountIncrement(&parent->sharedRes->refCount);
  }

  // ============================================================
  // 初始化 topParentRanks 数组
  // ============================================================
  //
  // topParentRanks：记录每个 rank 在最顶层父 comm 中的 rank 号
  //
  // 这个数组用于：
  // 1. 通信域分割（ncclCommSplit）
  // 2. 嵌套分割（多层分割）
  // 3. 跟踪原始 rank 映射
  //
  // 为什么要记录 top parent ranks？
  // - 子 comm 的 rank 编号会改变
  // - 但有时需要知道原始的 rank 号
  // - 用于资源映射和调试
  //
  // 示例：
  //   顶层 comm：8 个 ranks [0, 1, 2, 3, 4, 5, 6, 7]
  //   分割后：子 comm A 有 ranks [0, 2, 4, 6]
  //   对于子 comm A 的 rank 0（原始 rank 0）：
  //     - comm->rank = 0（子 comm 中的 rank）
  //     - topParentRanks[0] = 0（顶层 comm 中的 rank）
  //   对于子 comm A 的 rank 1（原始 rank 2）：
  //     - comm->rank = 1（子 comm 中的 rank）
  //     - topParentRanks[1] = 2（顶层 comm 中的 rank）
  //
  // 如果 comm->topParentRanks 已经存在：
  // - 说明是从父 comm 继承的
  // - 不需要重新分配
  //
  // ============================================================

  // 初始化 topParentRanks 数组
  if (comm->topParentRanks == NULL) {
    NCCLCHECK(ncclCalloc(&comm->topParentRanks, comm->nRanks));
    // 记录 parent 的 rank 号
    // 对于根 comm，topParentRanks[i] = i（自己就是顶层）
    for (int i = 0; i < comm->nRanks; ++i)
      comm->topParentRanks[i] = i;
  }

  // ============================================================
  // 初始化队列
  // ============================================================
  //
  // NCCL 使用多种无锁队列进行异步操作
  //
  // callbackQueue：回调队列
  // - 多生产者单消费者（MPSC）队列
  // - 存储待执行的回调函数
  // - 用于异步执行清理、通知等操作
  //
  // legacyRegCleanupQueue：旧版注册清理队列
  // - 存储需要清理的内存注册
  // - 用于异步释放 DMA 内存注册
  //
  // ceInitTaskQueue：集合引擎初始化任务队列
  // - 存储集合引擎的初始化任务
  // - 用于异步初始化集合引擎
  //
  // ncclIntruQueue：
  // - 无侵入式队列（Intrusive Queue）
  // - 节点自带链接字段，无需额外分配
  // - 高性能，适合频繁操作
  //
  // ============================================================

  // 初始化队列（用于回调、注册清理等）
  ncclIntruQueueMpscConstruct(&comm->callbackQueue);
  ncclIntruQueueConstruct(&comm->legacyRegCleanupQueue);
  ncclIntruQueueConstruct(&comm->ceInitTaskQueue);

  // ============================================================
  // 获取系统页大小
  // ============================================================
  //
  // 系统页大小（Page Size）：
  // - 虚拟内存管理的最小单位
  // - 通常是 4KB（x86）或 64KB（某些 ARM）
  //
  // 用途：
  // - 内存对齐
  // - DMA 内存分配
  // - 共享内存创建
  //
  // regCache：
  // - 内存注册缓存
  // - pageSize 用于确保内存对齐
  //
  // sysconf(_SC_PAGESIZE)：
  // - POSIX 标准函数
  // - 返回系统的页大小
  //
  // ============================================================

  // 获取系统页大小
  comm->regCache.pageSize = sysconf(_SC_PAGESIZE);

  // ============================================================
  // 创建 CUDA 内存池
  // ============================================================
  //
  // CUDA 内存池（Memory Pool）是 CUDA 11.2+ 引入的优化
  //
  // 为什么使用内存池？
  // 1. 减少分配/释放开销
  // 2. 提高内存重用率
  // 3. 减少内存碎片
  // 4. 支持异步分配
  //
  // cudaMemPoolProps：内存池属性
  // - allocType：分配类型
  //   - cudaMemAllocationTypePinned：锁页内存（不会被换出）
  // - handleTypes：句柄类型
  //   - cudaMemHandleTypeNone：不导出（不需要跨进程共享）
  // - location：内存位置
  //   - cudaMemLocationTypeDevice：设备内存
  //   - id：设备编号
  //
  // cudaMemPoolAttrReleaseThreshold：
  // - 释放阈值
  // - 当内存池的空闲内存超过此阈值时，释放回系统
  // - ~uint64_t(0) = UINT64_MAX：表示不自动释放
  // - 保留所有分配的内存，提高后续分配速度
  //
  // do-while(0) 结构：
  // - 创建一个临时作用域
  // - 避免变量名冲突（如 props、releaseThreshold）
  // - 保证只执行一次
  //
  // ============================================================

  // 创建 CUDA 内存池
  do {
    cudaMemPoolProps props = {};
    props.allocType = cudaMemAllocationTypePinned;      // 锁页内存
    props.handleTypes = cudaMemHandleTypeNone;          // 不使用导出句柄
    props.location.type = cudaMemLocationTypeDevice;    // 设备内存
    props.location.id = comm->cudaDev;                  // 与 comm 关联的 CUDA 设备
    CUDACHECK(cudaMemPoolCreate(&comm->memPool, &props));
    uint64_t releaseThreshold = ~uint64_t(0);           // 设置释放阈值为最大值
    CUDACHECK(cudaMemPoolSetAttribute(comm->memPool, cudaMemPoolAttrReleaseThreshold, &releaseThreshold));
  } while (0);

  // ============================================================
  // 初始化事件回调队列
  // ============================================================
  //
  // eventCallbackQueue：事件回调队列
  // - 存储 CUDA 事件的回调函数
  // - 用于异步处理事件完成通知
  //
  // 应用场景：
  // - 内核执行完成后的回调
  // - 内存拷贝完成后的回调
  // - 用户自定义的异步操作
  //
  // ============================================================

  // 初始化事件回调队列
  ncclIntruQueueConstruct(&comm->eventCallbackQueue);

  return ncclSuccess;
}
// ============================================================================
// devCommSetup - 设备通信设置
// 在 GPU 设备上分配和初始化通信所需的数据结构
// ============================================================================
//
// 这个函数是 NCCL 初始化的关键步骤之一，负责在 GPU 设备上建立通信所需的数据结构
//
// 主要功能：
// 1. 在设备端分配通信器和通道的内存结构
// 2. 将主机端的配置信息拷贝到设备端
// 3. 设置 workFifo（工作队列）用于主机和设备通信
// 4. 配置性能计数器和各种映射表
//
// 数据流：
//   主机端 (comm)           →   设备端 (devCommAndChans)
//   ┌─────────────────┐          ┌──────────────────────┐
//   │ rankToLocalRank  │  拷贝    │ rankToLocalRank      │
//   │ buffSizes        │  ---->   │ buffSizes            │
//   │ channels[]       │  拷贝    │ channels[]           │
//   │ ring.userRanks   │  ---->   │ ring.userRanks       │
//   └─────────────────┘          └──────────────────────┘
//
// ============================================================================
static ncclResult_t devCommSetup(ncclComm_t comm) {
  ncclResult_t ret = ncclSuccess;  // 返回值，默认成功
  int nRanks = comm->nRanks;        // 通信域中的总 rank 数量

  // ============================================================
  // 临时结构说明
  // ============================================================
  // tmpCommAndChans: 主机端临时结构体，用于收集所有要拷贝到设备的数据
  //                  作用：作为主机端数据的"汇聚点"，一次性拷贝到设备
  //
  // devCommAndChans: 设备端指针，指向 GPU 设备上的通信结构
  //                  作用：内核代码通过这个指针访问配置信息
  //
  // 数据拷贝流程：
  //   1. 在主机端填充 tmpCommAndChans
  //   2. 一次性将 tmpCommAndChans 拷贝到 devCommAndChans
  //   3. 设备内核通过 devCommAndChans 访问配置
  // ============================================================

  struct ncclKernelCommAndChannels tmpCommAndChans;   // 临时结构（主机端），用于收集数据
  struct ncclKernelCommAndChannels *devCommAndChans = NULL;  // 设备端指针，指向 GPU 内存

  // CC（Compute Consolidation）状态结构
  // CC 是 NVIDIA 的一种优化技术，允许在单个计算核中合并多个操作
  struct ncclNvmlCCStatus ccStatus;
  bool ccEnable;  // CC 功能是否启用的标志

  // CUDA 流，用于异步内存操作
  // 使用异步流可以提高性能，避免 CPU 等待 GPU 操作完成
  cudaStream_t deviceStream;

  // 清零临时结构体
  // '\0' 和 0 效果相同，这里用 '\0' 强调是字符填充
  // 这一步很重要，确保所有字段都有确定的初始值（避免未初始化内存导致的问题）
  memset(&tmpCommAndChans, '\0', sizeof(tmpCommAndChans));

  // ============================================================
  // 获取设备流
  // ============================================================
  // ncclStrongStreamAcquire: 获取一个强类型的 CUDA 流
  // - ncclCudaGraphNone(): 不使用 CUDA Graph（CUDA 图是一种优化技术）
  // - &comm->sharedRes->deviceStream: 共享的资源池中的设备流
  // - concurrent=false: 不允许并发使用（独占模式）
  // - deviceStream: 输出参数，返回获取到的流
  //
  // 为什么使用共享流？
  // - 多个操作可以共享同一个流，减少流创建开销
  // - 确保内存操作的顺序性（同一流中的操作按顺序执行）
  // ============================================================

  NCCLCHECKGOTO(ncclStrongStreamAcquire(ncclCudaGraphNone(), &comm->sharedRes->deviceStream, /*concurrent=*/false, &deviceStream), ret, fail);

  // ============================================================
  // 在设备上分配通信结构
  // ============================================================
  // ncclCudaCallocAsync: 异步分配并清零设备内存
  // - &devCommAndChans: 输出参数，分配的设备内存地址
  // - 1: 分配 1 个元素（sizeof(struct ncclKernelCommAndChannels) 字节）
  // - deviceStream: 在哪个流上执行分配
  //
  // 这个结构包含：
  // - ncclKernelComm: 通信器信息（rank 数、节点数、缓冲区大小等）
  // - channels[]: 所有通道的配置（Ring、Tree、CollNet、NVLS 等）
  //
  // ncclCommPushCudaFree: 注册析构函数
  // - 当 comm 被销毁时，自动释放这块设备内存
  // - 避免内存泄漏
  // ============================================================

  NCCLCHECKGOTO(ncclCudaCallocAsync(&devCommAndChans, 1, deviceStream), ret, fail);
  ncclCommPushCudaFree(comm, devCommAndChans);  // 注册自动释放

  // ============================================================
  // 分配 rank 到 local rank 的映射表
  // ============================================================
  // 为什么需要这个映射？
  // - 全局 rank: 0 到 nRanks-1，在整个通信域中唯一
  // - local rank: 0 到 localRanks-1，在单个节点内唯一
  // - 设备内核需要通过这个映射快速找到某个 rank 在其节点内的本地编号
  //
  // 例如：4 个 GPU，2 个节点
  //   节点 0: rank 0 (local 0), rank 2 (local 1)
  //   节点 1: rank 1 (local 0), rank 3 (local 1)
  //   rankToLocalRank = [0, 0, 1, 1]
  //
  // 先在设备上分配数组空间，然后拷贝数据
  // ============================================================

  NCCLCHECKGOTO(ncclCudaCallocAsync(&tmpCommAndChans.comm.rankToLocalRank, comm->nRanks, deviceStream), ret, fail);
  ncclCommPushCudaFree(comm, tmpCommAndChans.comm.rankToLocalRank);  // 注册自动释放

  // 将主机端的 rankToLocalRank 拷贝到设备端
  // - 目标: tmpCommAndChans.comm.rankToLocalRank (设备地址)
  // - 源: comm->rankToLocalRank (主机地址)
  // - 大小: comm->nRanks 个 int 元素
  // - 流: deviceStream（异步拷贝）
  NCCLCHECKGOTO(ncclCudaMemcpyAsync(tmpCommAndChans.comm.rankToLocalRank, comm->rankToLocalRank, comm->nRanks, deviceStream), ret, fail);

  // ============================================================
  // 设置设备端通信指针
  // ============================================================
  // comm->devComm 保存设备端通信结构的地址
  // 这样在内核启动时，可以将这个指针作为内核参数传递
  //
  // 内核访问方式：
  //   __global__ void kernel(struct ncclKernelComm* comm) {
  //     int rank = comm->rank;
  //     int* rankToLocalRank = comm->rankToLocalRank;
  //     ...
  //   }
  // ============================================================

  comm->devComm = &devCommAndChans->comm;

  // ============================================================
  // 填充基本通信信息
  // ============================================================
  // 这些信息是设备内核执行时需要的基本配置
  // 它们会被拷贝到设备端，供内核代码访问
  // ============================================================

  tmpCommAndChans.comm.rank = comm->rank;      // 当前 rank 的全局编号
  tmpCommAndChans.comm.nRanks = nRanks;        // 通信域中的总 rank 数
  tmpCommAndChans.comm.node = comm->node;      // 当前 rank 所属的节点编号
  tmpCommAndChans.comm.nNodes = comm->nNodes;  // 总节点数
  tmpCommAndChans.comm.abortFlag = comm->abortFlagDev;  // 设备端的 abort 标志（用于错误处理）

  // isAllNvlink: 是否所有连接都是 NVLink
  // - true: 所有 GPU 之间通过 NVLink 互联（最快）
  // - false: 存在 PCIe 或网络连接
  // 这个标志影响内核的路径选择和优化策略
  tmpCommAndChans.comm.isAllNvlink = comm->isAllNvlink;

  // ============================================================
  // 设置每种协议的缓冲区大小
  // ============================================================
  // NCCL 支持多种通信协议，每种协议有不同的缓冲区大小：
  // - NCCL_PROTO_LL (Long Jump): 中大消息优化协议
  // - NCCL_PROTO_LL128: 128 字节对齐的 LL 变体
  // - NCCL_PROTO_SIMPLE: 小消息简单拷贝协议
  //
  // buffSizes[p] 决定了该协议使用的缓冲区大小
  // 内核根据消息大小选择合适的协议
  // ============================================================

  for (int p=0; p < NCCL_NUM_PROTOCOLS; p++) {
    tmpCommAndChans.comm.buffSizes[p] = comm->buffSizes[p];
  }

  // P2P（Peer-to-Peer）块大小
  // 用于点对点通信（Send/Recv 操作）的块大小
  // 较大的块可以减少通信轮次，但增加延迟
  // 较小的块可以降低延迟，但增加通信轮次
  tmpCommAndChans.comm.p2pChunkSize = comm->p2pChunkSize;

  // ============================================================
  // 设置通道数组指针
  // ============================================================
  // &devCommAndChans->channels[0] 是设备端通道数组的起始地址
  // 每个通道包含 Ring、Tree、CollNet、NVLS 等拓扑的配置
  //
  // 通道是 NCCL 的并行单元：
  // - 数据被分成多个块，通过不同通道并行传输
  // - 最多 MAXCHANNELS（64）个通道
  // - 通道数由拓扑和性能模型决定
  // ============================================================

  tmpCommAndChans.comm.channels = &devCommAndChans->channels[0];

  // ============================================================
  // 计算工作参数大小（用于内核启动）
  // ============================================================
  // workArgsBytes: 内核参数结构体的大小
  //
  // 这个结构体包含：
  // - 操作类型（AllReduce、AllGather 等）
  // - 数据指针
  // - 数据大小
  // - 数据类型
  // - 使用的算法
  // - 使用的协议
  //
  // 限制原因：
  // 1. CUDA 内核参数大小有硬件限制（通常 4KB）
  // 2. 不同 GPU 架构的限制不同
  //
  // 取最小值确保兼容性：
  // - ncclParamWorkArgsBytes(): 用户或调优器设置的大小
  // - ncclMaxKernelArgsSize(): GPU 架构允许的最大大小
  // ============================================================

  comm->workArgsBytes = std::min<size_t>(ncclParamWorkArgsBytes(), ncclMaxKernelArgsSize(comm->cudaArch));

  // ============================================================
  // 检查是否启用 CC（Compute Consolidation）功能
  // ============================================================
  // CC 是 NVIDIA 的计算合并技术
  //
  // CC 的作用：
  // - 将多个独立的 kernel 合并为一个执行
  // - 减少 kernel 启动开销
  // - 提高 GPU 利用率
  //
  // CC 启用的条件（满足任一即可）：
  // 1. CCEnabled: NVML 报告 CC 功能已启用
  // 2. multiGpuProtectedPCIE: 多 GPU PCIe 保护模式
  // 3. multiGpuNVLE: NVLink Envelope 模式
  //
  // CC 的主要影响：
  // - 启用 CC 时，不需要 workFifo（直接在设备上调度）
  // - 禁用 CC 时，需要 workFifo 进行主机-设备通信
  // ============================================================

  memset(&ccStatus, 0, sizeof(ccStatus));  // 清零状态结构
  // 检查 CC 状态并设置 ccEnable 标志
  ccEnable = (ncclSuccess == ncclNvmlGetCCStatus(&ccStatus)) && (ccStatus.CCEnabled || ccStatus.multiGpuProtectedPCIE || ccStatus.multiGpuNVLE);

  if (ccEnable) {
    // ============================================================
    // CC 启用时的配置
    // ============================================================
    // CC 模式下，内核直接在设备上调度，不需要 workFifo
    // workFifoBytes = 0 表示不分配 workFifo 缓冲区
    //
    // CC 模式的优势：
    // - 减少主机-设备通信开销
    // - 更低的延迟
    // - 更高的吞吐量
    //
    // CC 模式的限制：
    // - 需要特定的硬件和驱动支持
    // - 主要用于单节点场景
    // ============================================================

    comm->workFifoBytes = 0;  // 不需要 workFifo
  } else {
    // ============================================================
    // CC 禁用时的配置（需要 workFifo）
    // ============================================================
    // workFifo 是主机和设备之间的通信队列
    //
    // 工作流程：
    // 1. 主机将工作请求写入 workFifo
    // 2. 设备内核从 workFifo 读取工作请求
    // 3. 设备内核执行集体操作
    // 4. 设备内核更新完成状态
    //
    // workFifo 的结构：
    //   ┌─────────────┬─────────────┬─────────────┬─────┐
    //   │  Work Item  │  Work Item  │  Work Item  │ ... │
    //   └─────────────┴─────────────┴─────────────┴─────┘
    //   produced/consumed 索引用于跟踪队列状态
    // ============================================================

    // 从环境变量或默认值获取 workFifo 大小
    comm->workFifoBytes = ncclParamWorkFifoBytes();

    // ============================================================
    // 检查是否是 2 的幂次方
    // ============================================================
    // 为什么 workFifo 大小必须是 2 的幂？
    //
    // 1. 环形队列的实现需要取模运算
    //    - index = (index + 1) % size
    //    - 当 size 是 2 的幂时，可以用位与代替取模：
    //      index = (index + 1) & (size - 1)
    //    - 位与运算比取模运算快得多
    //
    // 2. 检测方法：
    //    - 对于 2 的幂：二进制只有一个 1
    //      例如：16 = 0b10000
    //    - n & (n-1) 会清除最低位的 1
    //      例如：16 & 15 = 0b10000 & 0b01111 = 0
    //    - 如果结果为 0，说明是 2 的幂
    // ============================================================

    if (0 != (comm->workFifoBytes & (comm->workFifoBytes-1))) {
      // 不是 2 的幂，警告并使用默认值
      WARN("NCCL_WORK_FIFO_BYTES=%d is being ignored because it is not a power of 2.", comm->workFifoBytes);
      comm->workFifoBytes = NCCL_WORK_FIFO_BYTES_DEFAULT;  // 使用默认值
    }

    // 限制最大值为 1GB
    // - 过大的 workFifo 会浪费内存
    // - 1GB 对于绝大多数场景已经足够
    comm->workFifoBytes = std::min(comm->workFifoBytes, 1u<<30);
  }

  // 只在 rank 0 打印信息（避免重复输出）
  if (comm->rank == 0) {
    INFO(NCCL_INIT, "CC %s, workFifoBytes %d", ccEnable ? "On" : "Off", comm->workFifoBytes);
  }

  // ============================================================
  // 分配 workFifo 缓冲区
  // ============================================================
  // workFifo 是主机和设备共享的内存区域
  //
  // 两种分配方式：
  // 1. GDRCOPY: 使用 GPUDirect RDMA 技术，直接将内存映射到设备
  //    - 优势：设备可以直接访问，无需拷贝
  //    - 需要 GDRCOPY 库支持
  //
  // 2. cudaHostAlloc: 使用 CUDA 锁页主机内存
  //    - 优势：兼容性好，所有平台都支持
  //    - 设备访问时可能需要经过 PCIe
  // ============================================================

  if (ncclGdrCopy != NULL && ncclParamGdrCopyFifoEnable() == 1) {
    // ============================================================
    // 使用 GDRCOPY 映射的 CUDA 内存
    // ============================================================
    // ncclGdrCudaCalloc: 分配 GDRCOPY 映射的内存
    // - &comm->workFifoBuf: 主机端可访问的地址
    // - &comm->workFifoBufDev: 设备端可访问的地址
    // - comm->workFifoBytes: 分配大小
    // - &comm->workFifoBufGdrHandle: GDRCOPY 句柄（用于释放）
    //
    // GDRCOPY 工作原理：
    //   ┌─────────┐           ┌─────────┐
    //   │  Host   │           │  Device │
    //   └────┬────┘           └────┬────┘
    //        │                     │
    //        └─── Shared Memory ───┘
    //              (GDRCOPY mapped)
    //
    // - 主机和设备可以直接访问同一块物理内存
    // - 无需通过 PCIe 进行内存拷贝
    // ============================================================

    NCCLCHECKGOTO(ncclGdrCudaCalloc(&comm->workFifoBuf, &comm->workFifoBufDev, comm->workFifoBytes, &comm->workFifoBufGdrHandle), ret, fail);
    ncclCommPushCudaGdrFree(comm, comm->workFifoBufGdrHandle);  // 注册自动释放
  } else {
    // ============================================================
    // 使用 cudaHostAlloc 锁页内存（默认方式）
    // ============================================================
    // cudaHostAlloc 分配的主机内存特点：
    // 1. 锁页（pinned）：不会被操作系统换出到磁盘
    // 2. 可被设备访问：设备可以通过 DMA 直接访问
    // 3. 跨平台兼容：所有 CUDA 平台都支持
    //
    // workFifoBufGdrHandle = nullptr: 不使用 GDRCOPY
    // workFifoBuf 和 workFifoBufDev 指向同一地址
    // （因为 cudaHostAlloc 分配的内存主机和设备都用同一地址访问）
    // ============================================================

    comm->workFifoBufGdrHandle = nullptr;
    NCCLCHECKGOTO(ncclCudaHostCalloc(&comm->workFifoBuf, comm->workFifoBytes), ret, fail);
    ncclCommPushCudaHostFree(comm, comm->workFifoBuf);  // 注册自动释放
    comm->workFifoBufDev = comm->workFifoBuf;  // 设备地址 = 主机地址
  }

  // ============================================================
  // 初始化 workFifo 生产者和消费者索引
  // ============================================================
  // workFifo 是一个环形队列，使用两个索引管理：
  //
  // produced: 生产者索引（主机写入）
  // - 指向下一个可写入的位置
  // - 主机每次添加新工作时递增
  //
  // producedLastRecorded: 上次记录的生产者索引
  // - 用于性能统计和调试
  //
  // consumed: 消费者索引（设备读取）
  // - 指向下一个待处理的工作
  // - 设备内核每次完成工作后递增
  //
  // 环形队列状态：
  //   empty: produced == consumed
  //   full: (produced + 1) % size == consumed
  //
  // 内存一致性：
  // - 这些变量可能在主机和设备间共享
  // - 需要使用原子操作或内存屏障确保可见性
  // ============================================================

  comm->workFifoProduced = 0;           // 生产者索引初始为 0
  comm->workFifoProducedLastRecorded = 0;  // 记录索引初始为 0
  comm->workFifoConsumed = 0;           // 消费者索引初始为 0

  // ============================================================
  // 为内核分配性能计数器
  // ============================================================
  // 性能计数器用于测量每个通道的工作执行时间
  //
  // workStarted: 记录每个通道上工作开始的时间戳
  // workCompleted: 记录每个通道上工作完成的时间戳
  //
  // 用途：
  // 1. 性能分析：计算每个操作的延迟
  // 2. 负载均衡：检测通道间的负载差异
  // 3. 调试：识别慢速通道或瓶颈
  //
  // 为什么使用 cudaHostAlloc？
  // - 设备内核写入时间戳
  // - 主机读取时间戳进行统计
  // - 需要主机和设备都能访问
  // ============================================================

  NCCLCHECKGOTO(ncclCudaHostCalloc(&comm->profiler.workStarted, MAXCHANNELS), ret, fail);
  NCCLCHECKGOTO(ncclCudaHostCalloc(&comm->profiler.workCompleted, MAXCHANNELS), ret, fail);

  // 将性能计数器地址保存到临时结构（用于拷贝到设备）
  // 设备内核通过这些指针写入时间戳
  tmpCommAndChans.comm.workStarted = comm->profiler.workStarted;
  tmpCommAndChans.comm.workCompleted = comm->profiler.workCompleted;

  // 注册自动释放（cleanup 时释放）
  ncclCommPushCudaHostFree(comm, comm->profiler.workStarted);
  ncclCommPushCudaHostFree(comm, comm->profiler.workCompleted);

  // ============================================================
  // 如果使用 CollNet，需要拷贝 denseToUserRank 映射
  // ============================================================
  // CollNet（Collective Network）使用特殊的 rank 编号：
  //
  // user rank: 用户看到的 rank（0 到 nRanks-1）
  // dense rank: CollNet 内部使用的连续编号
  //
  // 为什么要使用 dense rank？
  // - CollNet 可能不包含所有 rank（只有部分节点）
  // - dense rank 提供连续的编号空间，便于内部索引
  //
  // 例如：rank 0, 2, 5, 7 参与 CollNet
  //   user rank:  0, 2, 5, 7
  //   dense rank: 0, 1, 2, 3
  //   denseToUserRank = [0, 2, 5, 7]
  //
  // 只有在使用 CollNet 时才需要这个映射
  // ============================================================

  if (comm->collNetDenseToUserRank != nullptr) {
    // 在设备上分配 denseToUserRank 数组
    NCCLCHECKGOTO(ncclCudaCallocAsync(&tmpCommAndChans.comm.collNetDenseToUserRank, nRanks, deviceStream), ret, fail);
    ncclCommPushCudaFree(comm, tmpCommAndChans.comm.collNetDenseToUserRank);  // 注册自动释放
    // 拷贝映射数据到设备
    NCCLCHECKGOTO(ncclCudaMemcpyAsync(tmpCommAndChans.comm.collNetDenseToUserRank, comm->collNetDenseToUserRank, nRanks, deviceStream), ret, fail);
  }

  // ============================================================
  // 为每个通道设置设备端指针
  // ============================================================
  // 遍历所有可能的通道（MAXCHANNELS = 64）
  //
  // 每个通道包含：
  // - peers: 每个 peer 的连接信息（发送/接收缓冲区、连接状态等）
  // - ring: 环形拓扑配置（前驱、后继 rank）
  // - tree: 树形拓扑配置（父节点、子节点）
  // - collnetChain: CollNet 链式拓扑配置
  // - collnetDirect: CollNet 直接拓扑配置
  // - nvls: NVLS（NVLink Switch）拓扑配置
  //
  // 注意：
  // - 这里只设置指针，实际数据在其他地方分配和拷贝
  // - devPeers、devRingUserRanks 等已经在通道初始化时分配
  // ============================================================

  for (int c=0; c < MAXCHANNELS; c++) {
    // peers: 每个 peer 的设备端连接信息数组
    // - devPeers[c][r] 表示通道 c 上与 rank r 的连接
    // - 包含发送/接收缓冲区指针、连接状态等
    tmpCommAndChans.channels[c].peers = comm->channels[c].devPeers;

    // ring: 环形拓扑配置（值拷贝，结构体较小）
    tmpCommAndChans.channels[c].ring = comm->channels[c].ring;
    // ring.userRanks: 环形拓扑的 rank 排序数组（指针）
    // - devRingUserRanks 是设备端的数组地址
    // - 下一步会拷贝实际数据到这个地址
    tmpCommAndChans.channels[c].ring.userRanks = comm->channels[c].devRingUserRanks;

    // tree: 树形拓扑配置（值拷贝）
    tmpCommAndChans.channels[c].tree = comm->channels[c].tree;

    // CollNet 拓扑配置
    tmpCommAndChans.channels[c].collnetChain = comm->channels[c].collnetChain;
    tmpCommAndChans.channels[c].collnetDirect = comm->channels[c].collnetDirect;

    // NVLS 拓扑配置
    tmpCommAndChans.channels[c].nvls = comm->channels[c].nvls;

    // ============================================================
    // 拷贝 ring user ranks 到设备
    // ============================================================
    // ring.userRanks 是环形拓扑的 rank 排序数组
    // - 例如：[2, 3, 0, 1] 表示环形顺序
    // - 设备内核通过这个数组快速查找邻居
    //
    // 只拷贝非空指针的通道（避免无效拷贝）
    // - 某些通道可能未使用（nChannels < MAXCHANNELS）
    // - 这些通道的 userRanks 为 nullptr
    // ============================================================

    if (comm->channels[c].ring.userRanks != nullptr) {
      NCCLCHECKGOTO(ncclCudaMemcpyAsync(tmpCommAndChans.channels[c].ring.userRanks, comm->channels[c].ring.userRanks, nRanks, deviceStream), ret, fail);
    }
  }

  // ============================================================
  // 将临时结构拷贝到设备
  // ============================================================
  // 这是关键的一步：将所有配置数据一次性拷贝到设备
  //
  // 拷贝内容包括：
  // 1. 通信器配置（rank、节点、缓冲区大小等）
  // 2. workFifo 指针
  // 3. 性能计数器指针
  // 4. 所有通道的配置指针
  //
  // 为什么一次性拷贝？
  // - 减少主机-设备传输次数
  // - 确保配置的原子性（要么全部更新，要么都不更新）
  // - 简化错误处理
  //
  // 拷贝完成后，comm->devComm 指向设备端的有效配置
  // 内核可以通过这个指针访问所有必要的配置信息
  // ============================================================

  NCCLCHECKGOTO(ncclCudaMemcpyAsync(devCommAndChans, &tmpCommAndChans, 1, deviceStream), ret, fail);

  // ============================================================
  // 清理和同步
  // ============================================================
  // 执行到这里说明所有操作都成功了
  //
  // exit 标签：
  // - 这是正常的退出路径
  // - 也是 fail 错误处理后的跳转目标
  // ============================================================

exit:
  // 释放设备流
  // - ncclStrongStreamRelease: 释放之前获取的流
  // - 允许其他操作使用这个流
  NCCLCHECK(ncclStrongStreamRelease(ncclCudaGraphNone(), &comm->sharedRes->deviceStream, /*concurrent=*/false));

  // 同步设备流
  // - 确保所有异步操作（内存分配、拷贝）都已完成
  // - 在函数返回前，设备端的数据必须就绪
  // - 这是关键同步点，之后内核可以安全访问这些数据
  NCCLCHECK(ncclStrongStreamSynchronize(&comm->sharedRes->deviceStream));

  return ret;  // 返回操作结果

  // ============================================================
  // 错误处理
  // ============================================================
  // fail 标签：
  // - 当任何 NCCLCHECKGOTO 操作失败时跳转到这里
  // - 使用 goto 而不是直接 return，确保清理代码执行
  //
  // 错误处理策略：
  // 1. 不释放已分配的内存（析构函数链会处理）
  // 2. 跳转到 exit 执行清理和同步
  // 3. 返回错误码
  // ============================================================

fail:
  goto exit;  // 跳转到 exit 标签执行清理
}

// 版本字符串定义（用于 "strings" 命令快速查看版本）
#define VERSION_STRING "NCCL version " STR(NCCL_MAJOR) "." STR(NCCL_MINOR) "." STR(NCCL_PATCH) NCCL_SUFFIX "+cuda" STR(CUDA_MAJOR) "." STR(CUDA_MINOR)

// 显示版本信息
static void showVersion() {
  if (ncclDebugLevel == NCCL_LOG_VERSION || ncclDebugLevel == NCCL_LOG_WARN) {
    VERSION("%s", VERSION_STRING);
  } else {
    INFO(NCCL_ALL,"%s", VERSION_STRING);
  }
}

// MNNVL（Multi-Node NVLink）相关参数
NCCL_PARAM(MNNVLUUID, "MNNVL_UUID", -1);
NCCL_PARAM(MNNVLCliqueId, "MNNVL_CLIQUE_ID", -1);

// ============================================================================
// ============================================================================
// fillInfo - 填充 peer 信息
// 收集当前 rank 的信息，用于后续与其他 rank 交换
// ============================================================================
//
// 这个函数是 NCCL 初始化的第一步（AllGather 1 的准备阶段）
//
// 主要功能：
// 收集当前 rank 的所有相关信息，填充到 ncclPeerInfo 结构体中
// 这些信息将通过 bootstrapAllGather 交换给所有其他 rank
//
// 参数说明：
// - comm: 当前通信器指针
// - info: 输出参数，填充后的 peer 信息结构
// - commHash: 通信域的 hash 值，用于区分不同的通信域
//
// 收集的信息包括：
// 1. 基本信息：rank 号、GPU 设备号、NCCL 版本
// 2. 节点信息：hostHash、pidHash（用于判断是否在同一节点/进程）
// 3. 硬件信息：显存大小、总线 ID、计算能力
// 4. 功能支持：cuMem、GDR、MNNVL
// 5. Fabric 信息：UUID、cliqueId（用于 NVLink 互联）
//
// 数据流：
//   fillInfo()              bootstrapAllGather()          其他 ranks
//      │                            │                          │
//      ├─ 收集本地信息              │                          │
//      ├─ 填充 info 结构 ──────────>│  AllGather  ───────────> │
//      │                            │                          │
//      │                            │ <──────── 交换 ──────────│
//      │                            │                          │
//      │                     分发所有 ranks                  │
//      │                     的 peerInfo                     │
//
// ============================================================================
static ncclResult_t fillInfo(struct ncclComm* comm, struct ncclPeerInfo* info, uint64_t commHash) {
  // ============================================================
  // cudaDeviceProp: CUDA 设备属性结构体
  // ============================================================
  // 用于存储 GPU 设备的详细属性信息
  // 包括：设备名称、计算能力、显存大小、时钟频率等
  // 这里只是声明变量，实际赋值在后面通过 cudaGetDeviceProperties 获取
  // ============================================================

  cudaDeviceProp prop;

  // ============================================================
  // 基本信息
  // ============================================================
  // 这些是识别当前 rank 最基本的信息
  // 所有 rank 必须提供这些信息以便相互识别和验证
  // ============================================================

  // 全局 rank 号
  // - 在通信域中的唯一编号（0 到 nRanks-1）
  // - 用于标识不同的 GPU 进程
  // - 例如：4 个 GPU，每个 rank 有一个唯一编号（0, 1, 2, 3）
  info->rank = comm->rank;

  // GPU 设备号
  // - CUDA 设备 ID，由 CUDA 驱动分配
  // - 在单节点内唯一（0 到 num_gpus-1）
  // - 用于 cudaSetDevice() 选择要使用的 GPU
  // - 注意：不同节点上可能有相同的 cudaDev，但 rank 不同
  info->cudaDev = comm->cudaDev;

  // NVML 设备号
  // - NVML (NVIDIA Management Library) 设备索引
  // - 用于通过 NVML API 查询 GPU 状态和信息
  // - 与 cudaDev 可能不同（NVML 和 CUDA 的编号方式可能有差异）
  // - 主要用于获取 GPU Fabric 信息、温度、功率等
  info->nvmlDev = comm->nvmlDev;

  // NCCL 版本号
  // - 获取当前 NCCL 库的版本
  // - 格式：Major*10000 + Minor*100 + Patch
  // - 例如：2.28.3 -> 22803
  // - 用于版本兼容性检查：
  //   * 所有 rank 必须使用相同版本的 NCCL
  //   * 不同版本可能导致协议不兼容
  NCCLCHECK(ncclGetVersion(&info->version));

  // ============================================================
  // 主机名和 PID 的 hash 值
  // ============================================================
  // 用于识别 rank 之间的拓扑关系：
  //
  // 1. hostHash: 主机的 hash 值
  //    - 相同 hostHash = 同一物理节点（同一台机器）
  //    - 不同 hostHash = 不同物理节点（跨节点通信）
  //
  // 2. pidHash: 进程的 hash 值
  //    - 相同 pidHash = 同一进程（多线程/多 GPU）
  //    - 不同 pidHash = 不同进程（多进程）
  //
  // 3. commHash: 通信域的 hash 值
  //    - 用于区分同一节点上的多个通信域
  //    - 防止不同通信域的 ranks 相互干扰
  //
  // 组合判断：
  // - hostHash 相同 + pidHash 相同 = 同进程多 GPU（最快，使用 NVLink/SHM）
  // - hostHash 相同 + pidHash 不同 = 同节点多进程（快，使用 SHM）
  // - hostHash 不同 = 跨节点（慢，使用网络）
  //
  // 为什么使用 hash 而不是直接使用主机名和 PID？
  // - hash 值固定大小（64 位），便于传输和比较
  // - 避免字符串比较的开销
  // - commHash 的加法确保不同通信域不会冲突
  // ============================================================

  // 计算 hostHash：主机名的 hash 值 + commHash
  // - getHostHash() 内部调用 gethostname() 并计算 hash
  // - 同一节点的所有 ranks 有相同的 hostHash
  info->hostHash = getHostHash() + commHash;

  // 计算 pidHash：进程 ID 的 hash 值 + commHash
  // - getPidHash() 内部使用 getpid() 并计算 hash
  // - 同一进程的所有 ranks 有相同的 pidHash
  info->pidHash = getPidHash() + commHash;

  // ============================================================
  // cuMem 支持情况
  // ============================================================
  // cuMem (CUDA Unified Memory) 是 CUDA 12.2+ 引入的新内存管理 API
  //
  // cuMem 的优势：
  // 1. 统一的内存管理：主机和设备使用同一地址空间
  // 2. 自动迁移：数据在主机和设备间自动迁移
  // 3. 跨进程共享：多个进程可以共享同一块内存
  // 4. NUMA 感知：支持 NUMA 节点的内存分配
  //
  // cuMemSupport 的作用：
  // - 如果所有 rank 都支持 cuMem，可以启用 Runtime Connection 模式
  // - Runtime Connection 延迟连接建立，减少初始化时间
  // - 如果不支持，必须在初始化时建立所有连接
  //
  // 检测条件：
  // - CUDA 版本 >= 12.2
  // - 驱动支持 CUMEM API
  // - 环境变量 NCCL_CUMEM_ENABLE=1（默认启用）
  // ============================================================

  info->cuMemSupport = ncclCuMemEnable();

  // ============================================================
  // 计算全局显存大小
  // ============================================================
  // totalGlobalMem: GPU 的全局显存大小（字节数）
  //
  // 作用：
  // 1. 内存分配决策：决定可以分配多大的缓冲区
  // 2. 一致性检查：确保所有 ranks 有足够的显存
  // 3. 性能调优：根据显存大小调整缓冲区策略
  //
  // 为什么向上取整到 4GB (1L << 32)？
  // - 简化比较和计算
  // - 不同 GPU 的显存大小可能略有不同（如 32768MB vs 32770MB）
  // - 向上取整后，相同档次的 GPU 会有相同的值
  // - 例如：32GB 显存 -> 32GB，48GB 显存 -> 48GB
  //
  // ROUNDUP 宏定义：
  // #define ROUNDUP(x, y) (((x) + ((y) - 1)) & ~((y) - 1))
  // 要求 y 是 2 的幂，这里 y = 1L << 32 = 4GB
  // ============================================================

  // 获取 GPU 设备属性
  // - prop: 输出参数，填充设备属性
  // - comm->cudaDev: 要查询的设备 ID
  CUDACHECK(cudaGetDeviceProperties(&prop, comm->cudaDev));

  // 向上取整到 4GB 并保存
  info->totalGlobalMem = ROUNDUP(prop.totalGlobalMem, (1L << 32));

  // ============================================================
  // 获取 /dev/shm 的设备号
  // ============================================================
  // /dev/shm: Linux 的共享内存文件系统
  //
  // 作用：
  // - 用于同一节点内进程间通信（IPC）
  // - NCCL 使用共享内存实现快速节点内通信
  //
  // shmDev 的作用：
  // - 判断不同 ranks 是否可以使用共享内存通信
  // - st_dev: 文件系统的设备 ID
  // - 相同的 shmDev = 可以通过 /dev/shm 通信
  // - 不同的 shmDev = 可能在容器环境中，/dev/shm 被隔离
  //
  // 容器环境问题：
  // - Docker 容器可能挂载不同的 /dev/shm
  // - 即使在同一节点，不同容器的 shmDev 可能不同
  // - 这种情况下需要使用网络通信，而不是共享内存
  //
  // stat 系统调用：
  // - 获取文件状态信息
  // - st_dev: 设备号（主设备号和次设备号）
  // - st_ino: 文件 inode 号
  // ============================================================

  struct stat statbuf;  // 文件状态缓冲区
  SYSCHECK(stat("/dev/shm", &statbuf), "stat");  // 获取 /dev/shm 的状态
  info->shmDev = statbuf.st_dev;  // 保存设备号

  // ============================================================
  // GPU 的 busId
  // ============================================================
  // busId: GPU 的 PCIe 总线 ID
  //
  // 格式：
  // - 64 位整数，编码了 PCIe 域、总线、设备、功能号
  // - 例如：0x00010000:00:02.0
  //
  // 作用：
  // 1. 唯一标识 GPU：在同一节点内唯一
  // 2. 拓扑发现：用于构建 GPU 互连拓扑
  // 3. 路径计算：计算 GPU 之间的最短路径
  // 4. 重复检测：检测是否有多个 rank 绑定到同一 GPU
  //
  // 示例：
  // - GPU 0: busId = 0x00010000 (域 0, 总线 1, 设备 0, 功能 0)
  // - GPU 1: busId = 0x00020000 (域 0, 总线 2, 设备 0, 功能 0)
  // - 通过 busId 可以判断两个 GPU 是否在同一 PCIe 根复杂体
  // ============================================================

  info->busId = comm->busId;

  // ============================================================
  // 判断是否支持 GDR（GPUDirect RDMA）
  // ============================================================
  // GPUDirect RDMA: 允许网络设备直接访问 GPU 显存
  //
  // 传统模式（无 GDR）：
  //   GPU → 主机内存 → 网卡 → 网络
  //   (需要一次额外的拷贝)
  //
  // GDR 模式：
  //   GPU → 网卡 → 网络
  //   (网卡直接读写 GPU 显存，零拷贝)
  //
  // GDR 的优势：
  // 1. 降低延迟：减少一次内存拷贝
  // 2. 降低 CPU 占用：CPU 不参与数据传输
  // 3. 提高带宽：充分利用 GPU 和网卡之间的带宽
  //
  // GDR 的要求：
  // 1. 硬件支持：
  //    - NVIDIA GPU（支持 GPUDirect RDMA）
  //    - 支持 RDMA 的网卡（如 Mellanox ConnectX）
  //    - GPU 和网卡必须在同一 PCIe 根复杂体
  // 2. 软件支持：
  //    - CUDA 工具包 >= 7.5
  //    - OFED 驱动（RDMA 驱动）
  //    - nvidia_peermem 模块已加载
  //
  // gdrSupport 的作用：
  // - 如果支持，优先使用 RDMA 进行跨节点通信
  // - 如果不支持，使用传统的主机内存中转
  // ============================================================

  NCCLCHECK(ncclGpuGdrSupport(comm, &info->gdrSupport));

  // ============================================================
  // 保存 comm 指针和计算能力
  // ============================================================
  // comm 指针的作用：
  // - 在后续初始化阶段，可以通过 peerInfo 访问对应的 comm
  // - 用于建立进程内 comm 链表（intraNext）
  // - 便于同一进程内的 ranks 之间直接通信
  //
  // cudaCompCap: CUDA 计算能力（Compute Capability）
  // - 例如：70 (Volta), 75 (Turing), 80 (Ampere), 89 (Hopper)
  // - 格式：Major*10 + Minor（如 7.0 -> 70）
  //
  // minCompCap / maxCompCap:
  // - 初始时都设为当前 GPU 的计算能力
  // - AllGather 后，会更新为通信域中的最小/最大值
  // - 用于性能调优：选择合适的内核实现
  // ============================================================

  info->comm = comm;  // 保存 comm 指针
  // 设置初始值（后续会被 AllGather 结果更新）
  info->cudaCompCap = comm->minCompCap = comm->maxCompCap = comm->compCap;

  // ============================================================
  // MNNVL（Multi-Node NVLink）支持信息
  // ============================================================
  // MNNVL: NVIDIA 的跨节点 NVLink 技术
  //
  // 传统 NVLink:
  //   只能在同一节点内的 GPU 之间使用
  //   ┌─────────────────────┐
  //   │ GPU0 ← NVLink → GPU1 │
  //   └─────────────────────┘
  //
  // MNNVL (Multi-Node NVLink):
  //   允许跨节点的 GPU 通过 NVLink 互联
  //   ┌──────────┐           ┌──────────┐
  //   │ 节点 1   │  NVLink   │ 节点 2   │
  //   │ GPU0 ←───────┼────────→ GPU2   │
  //   └──────────┘           └──────────┘
  //
  // MNNVL 的优势：
  // 1. 更低延迟：NVLink 延迟远低于网络
  // 2. 更高带宽：NVLink 带宽远高于网络（300GB/s vs 100Gb/s）
  // 3. 统一编程模型：跨节点和节点内通信使用相同机制
  //
  // MNNVL 的要求：
  // 1. 硬件：支持 NVLink 的 GPU 和交换机（如 NVSwitch）
  // 2. 链路：节点间有 NVLink 物理连接
  // 3. 配置：正确的 Fabric 配置
  //
  // fabricInfo 结构：
  // - state: Fabric 状态
  // - clusterUuid: 集群唯一标识符
  // - cliqueId: 子集群标识符
  // - healthMask: 健康状态掩码
  // ============================================================

  {
    // ============================================================
    // 获取 Fabric UUID 和分区信息
    // ============================================================
    // 步骤：
    // 1. 将 busId 转换为 PCI 总线 ID 字符串格式
    // 2. 通过 PCI 总线 ID 获取 NVML 设备句柄
    // 3. 获取 GPU Fabric 信息
    // ============================================================

    // PCI 总线 ID 字符串缓冲区
    // 格式：如 "0000:00:02.0" (域:总线:设备.功能)
    // NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE 通常是 32 字节
    char busId[NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE];

    // NVML 设备句柄
    // 用于后续调用 NVML API 查询设备信息
    nvmlDevice_t nvmlDev;

    // 将 64 位 busId 整数转换为 PCI 总线 ID 字符串
    // 例如：0x00010000 -> "0000:01:00.0"
    NCCLCHECK(int64ToBusId(info->busId, busId));

    // 通过 PCI 总线 ID 获取 NVML 设备句柄
    // 这是访问 NVML 功能的第一步
    NCCLCHECK(ncclNvmlDeviceGetHandleByPciBusId(busId, &nvmlDev));

    // 初始化 Fabric 状态为"不支持"
    // 如果设备不支持 MNNVL，后续查询会保持这个状态
    info->fabricInfo.state = NVML_GPU_FABRIC_STATE_NOT_SUPPORTED;

    // 获取 GPU Fabric 信息（版本 V）
    // (void) 忽略返回值，因为我们通过 state 判断是否支持
    // 如果支持，state 会被更新为其他值（ACTIVE, NON_COORDINATED 等）
    (void) ncclNvmlDeviceGetGpuFabricInfoV(nvmlDev, &info->fabricInfo);

    // ============================================================
    // 处理 MNNVL 支持的情况
    // ============================================================
    // 只有当 state != NOT_SUPPORTED 时，才进行 MNNVL 配置
    // ============================================================

    if (info->fabricInfo.state != NVML_GPU_FABRIC_STATE_NOT_SUPPORTED) {
      // ============================================================
      // clusterUuid: 集群唯一标识符
      // ============================================================
      // clusterUuid 是 16 字节（128 位）的唯一标识符
      // - 同一 Fabric 中的所有 GPU 有相同的 clusterUuid
      // - 用于判断哪些 GPU 属于同一个 Fabric
      //
      // 为了便于打印和比较，将 16 字节分成两个 64 位整数
      // ============================================================

      unsigned long uuid0 = 0;  // 前 8 字节（64 位）
      unsigned long uuid1 = 0;  // 后 8 字节（64 位）

      // ============================================================
      // 环境变量覆盖：MNNVL_UUID
      // ============================================================
      // 为什么需要环境变量覆盖？
      // 1. 测试：在测试环境中模拟特定的 Fabric 配置
      // 2. 故障排查：强制使用特定的 UUID
      // 3. 兼容性：在某些环境下自动检测失败时可以手动设置
      //
      // NCCL 设置：
      // - export NCCL_MNNVL_UUID=<uuid_value>
      // - -1 表示未设置（使用自动检测的值）
      // ============================================================

      // 如果通过环境变量设置了 UUID，使用环境变量的值
      if (ncclParamMNNVLUUID() != -1) {
        unsigned long temp_uuid0 = (unsigned long)ncclParamMNNVLUUID();
        unsigned long temp_uuid1 = (unsigned long)ncclParamMNNVLUUID();

        // 拷贝到 clusterUuid 数组（16 字节）
        // 前 8 字节
        memcpy(info->fabricInfo.clusterUuid, &temp_uuid0, sizeof(temp_uuid0));
        // 后 8 字节（从偏移 sizeof(temp_uuid0) 开始）
        memcpy(info->fabricInfo.clusterUuid + sizeof(temp_uuid0), &temp_uuid1, sizeof(temp_uuid1));
      }

      // 从 clusterUuid 数组中提取两个 64 位整数（用于日志输出）
      // 这样可以方便地打印 UUID
      memcpy(&uuid0, info->fabricInfo.clusterUuid, sizeof(uuid0));
      memcpy(&uuid1, info->fabricInfo.clusterUuid + sizeof(uuid0), sizeof(uuid1));

      // ============================================================
      // cliqueId: 子集群标识符
      // ============================================================
      // cliqueId 用于将 Fabric 划分为多个子集群（clique）
      //
      // 为什么要划分子集群？
      // 1. 性能优化：同一 clique 内的 GPU 通信更快
      // 2. 故障隔离：某个 clique 的故障不影响其他 clique
      // 3. 灵活配置：支持复杂的拓扑结构
      //
      // cliqueId 的设置方式：
      // 1. -2: 使用机架序列号的 hash（自动检测）
      // 2. -1: 不设置（默认行为）
      // 3. >= 0: 使用指定的值（手动配置）
      // ============================================================

      if (ncclParamMNNVLCliqueId() == -2) {
        // ============================================================
        // 使用机架序列号的 hash 作为 cliqueId
        // ============================================================
        // 这种方式适用于：
        // - 机架式服务器集群
        // - 同一机架内的 GPU 属于同一 clique
        //
        // 平台信息包括：
        // - chassisSerialNumber: 机架序列号
        // - slotNumber: 插槽号
        // - trayIndex: 托盘索引
        // - hostId: 主机 ID
        // - peerType: 对等类型
        // - moduleId: 模块 ID
        // ============================================================

        nvmlPlatformInfo_t platformInfo = { 0 };  // 平台信息结构（初始化为 0）
        NCCLCHECK(ncclNvmlDeviceGetPlatformInfo(nvmlDev, &platformInfo));

        // 打印平台信息（用于调试）
        INFO(NCCL_INIT, "MNNVL rack serial %s slot %d tray %d hostId %d peerType %d moduleId %d",
             platformInfo.chassisSerialNumber, platformInfo.slotNumber, platformInfo.trayIndex,
             platformInfo.hostId, platformInfo.peerType, platformInfo.moduleId);

        // 使用机架序列号的 hash 作为 cliqueId
        // - 同一机架的 GPU 有相同的序列号
        // - hash 确保 cliqueId 在合理范围内
        info->fabricInfo.cliqueId = getHash(platformInfo.chassisSerialNumber, sizeof(platformInfo.chassisSerialNumber));
      } else if (ncclParamMNNVLCliqueId() != -1) {
        // 使用环境变量指定的 cliqueId
        // export NCCL_MNNVL_CLIQUE_ID=<clique_id_value>
        info->fabricInfo.cliqueId = ncclParamMNNVLCliqueId();
      }
      // 如果 ncclParamMNNVLCliqueId() == -1，不设置 cliqueId（使用默认值）

      // ============================================================
      // 打印 MNNVL 信息（用于调试和验证）
      // ============================================================
      // 输出内容：
      // - busId: GPU 的 PCIe 总线 ID
      // - uuid0.uuid1: Fabric 的集群 UUID
      // - cliqueId: 子集群标识符
      // - state: Fabric 状态（ACTIVE, NON_COORDINATED 等）
      // - healthMask: 健康状态掩码（用于故障检测）
      // ============================================================

      INFO(NCCL_INIT, "MNNVL busId 0x%lx fabric UUID %lx.%lx cliqueId 0x%x state %d healthMask 0x%x",
           info->busId, uuid0, uuid1,
           info->fabricInfo.cliqueId, info->fabricInfo.state, info->fabricInfo.healthMask);
    }
    // 如果不支持 MNNVL，不进行任何配置
  }

  return ncclSuccess;
}

// ============================================================================
// ============================================================================
// setupChannel - 设置通信通道
// 初始化环形算法的通道连接
// ============================================================================
//
// 这个函数负责设置单个通道的环形拓扑配置
//
// 参数说明：
// - comm：通信器指针
// - channelId：通道 ID（0 到 nChannels-1）
// - rank：当前 rank 的全局编号
// - nranks：总 rank 数
// - ringRanks：环形拓扑的 rank 排序数组（由 ncclTopoPreset 计算得到）
//
// 环形拓扑示例：
// 假设有 4 个 rank，ringRanks = [0, 1, 2, 3]
// - Rank 0 的前驱是 rank 3，后继是 rank 1
// - Rank 1 的前驱是 rank 0，后继是 rank 2
// - Rank 2 的前驱是 rank 1，后继是 rank 3
// - Rank 3 的前驱是 rank 2，后继是 rank 0
//
// 环形索引的意义：
// 每个通道维护一个 userRanks 数组，存储了环形拓扑的 rank 顺序
// 这样可以快速查找任意 rank 的前驱和后继
//
// ============================================================================
static ncclResult_t setupChannel(struct ncclComm* comm, int channelId, int rank, int nranks, int* ringRanks) {
  TRACE(NCCL_INIT, "rank %d nranks %d", rank, nranks);

  // 初始化通道
  // initChannel 会分配通道所需的内存结构
  NCCLCHECK(initChannel(comm, channelId));

  struct ncclRing* ring = &comm->channels[channelId].ring;

  // ============================================================
  // 环形索引计算
  // ============================================================
  //
  // 目标：计算当前 rank 在环形拓扑中的逻辑位置
  //
  // ixZero：rank 0 在 ringRanks 数组中的索引
  // ixRank：当前 rank 在 ringRanks 数组中的索引
  //
  // ring->index 是当前 rank 的"环索引"，定义为：
  // index = (ixRank - ixZero + nranks) % nranks
  //
  // 这个公式确保：
  // - rank 0 的 index = 0
  // - rank 1 的 index = 1（如果 rank 1 在 rank 0 后面）
  // - rank 3 的 index = 3（如果 rank 3 在 rank 0 前面，则 index = nranks - 1）
  //
  // 示例：ringRanks = [3, 0, 1, 2]
  // - ixZero = 1（rank 0 在索引 1）
  // - 对于 rank 3：ixRank = 0, index = (0-1+4)%4 = 3
  // - 对于 rank 0：ixRank = 1, index = (1-1+4)%4 = 0
  // - 对于 rank 1：ixRank = 2, index = (2-1+4)%4 = 1
  // - 对于 rank 2：ixRank = 3, index = (3-1+4)%4 = 2
  //
  // ============================================================

  // 找到当前 rank 在环中的位置，并重新组织 ranks 使得当前 rank 排在第一位
  int ixZero = 0, ixRank = 0;

  for (int i = 0; i < nranks; i++) {
    if (ringRanks[i] == 0)
        ixZero = i;   // 找到 rank 0 的索引
    if (ringRanks[i] == rank)
        ixRank = i;   // 找到当前 rank 的索引
  }

  // 计算当前 Rank 在环中的逻辑索引
  ring->index = (ixRank - ixZero + nranks) % nranks;

  // ============================================================
  // 环形重排序
  // ============================================================
  //
  // 为什么要重新排列环顺序？
  //
  // 环形算法需要每个 Rank 知道：
  // - 左邻居（接收数据的来源）：userRanks[nranks-1]
  // - 右邻居（发送数据的目标）：userRanks[1]
  //
  // 通过重新排列，使得：
  // - userRanks[0] = 当前 rank
  // - userRanks[1] = 后继 rank（右邻居）
  // - userRanks[nranks-1] = 前驱 rank（左邻居）
  //
  // 这样方便内核代码直接通过索引访问邻居
  //
  // 重排序公式：
  // userRanks[i] = ringRanks[(i + ixRank) % nranks]
  //
  // 示例：ringRanks = [0, 1, 2, 3], 当前 rank = 2 (ixRank = 2)
  // - userRanks[0] = ringRanks[(0+2)%4] = ringRanks[2] = 2（当前 rank）
  // - userRanks[1] = ringRanks[(1+2)%4] = ringRanks[3] = 3（右邻居）
  // - userRanks[2] = ringRanks[(2+2)%4] = ringRanks[0] = 0
  // - userRanks[3] = ringRanks[(3+2)%4] = ringRanks[1] = 1（左邻居）
  //
  // ============================================================

  for (int i = 0; i < nranks; i++) {
    ring->userRanks[i] = ringRanks[(i + ixRank) % nranks];
  }

  return ncclSuccess;
}

// 默认缓冲区大小定义
//8*512*8*16=512KB
#define DEFAULT_LL_BUFFSIZE (NCCL_LL_LINES_PER_THREAD*NCCL_LL_MAX_NTHREADS*NCCL_STEPS*sizeof(union ncclLLFifoLine))
//120*640*8*8=4800KB
#define DEFAULT_LL128_BUFFSIZE (NCCL_LL128_ELEMS_PER_THREAD*NCCL_LL128_MAX_NTHREADS*NCCL_STEPS*sizeof(uint64_t))
#define DEFAULT_BUFFSIZE (1 << 22)  /* 4MiB */

// 缓冲区大小参数（可通过环境变量配置）
NCCL_PARAM(BuffSize, "BUFFSIZE", -2);          // Simple 协议缓冲区大小
NCCL_PARAM(LlBuffSize, "LL_BUFFSIZE", -2);     // LL 协议缓冲区大小
NCCL_PARAM(Ll128BuffSize, "LL128_BUFFSIZE", -2); // LL128 协议缓冲区大小

// P2P chunk size 参数
NCCL_PARAM(P2pNetChunkSize, "P2P_NET_CHUNKSIZE", (1 << 17));  /* 128 kB - 网络传输 */
NCCL_PARAM(P2pPciChunkSize, "P2P_PCI_CHUNKSIZE", (1 << 17));  /* 128 kB - PCIe 传输 */
NCCL_PARAM(P2pNvlChunkSize, "P2P_NVL_CHUNKSIZE", (1 << 19)); /* 512 kB - NVLink 传输 */

// ============================================================================
// computeBuffSizes - 计算各种协议的缓冲区大小
// ============================================================================
static ncclResult_t computeBuffSizes(struct ncclComm* comm) {
  int64_t envs[NCCL_NUM_PROTOCOLS] = { ncclParamLlBuffSize(), ncclParamLl128BuffSize(), ncclParamBuffSize() };
  int defaults[NCCL_NUM_PROTOCOLS] = { DEFAULT_LL_BUFFSIZE, DEFAULT_LL128_BUFFSIZE, DEFAULT_BUFFSIZE };

  // 如果没有设置环境变量，使用默认值
  for (int p = 0; p < NCCL_NUM_PROTOCOLS; p++) {
    comm->buffSizes[p] = envs[p] != -2 ? envs[p] : defaults[p];
  }

  // 根据传输类型设置 P2P chunk size
  if (comm->nNodes > 1)
    comm->p2pChunkSize = ncclParamP2pNetChunkSize();   // 跨节点使用网络 chunk size
  else if (comm->isAllNvlink)
    comm->p2pChunkSize = ncclParamP2pNvlChunkSize();    // 全 NVLink 使用 NVLink chunk size
  else
    comm->p2pChunkSize = ncclParamP2pPciChunkSize();    // 否则使用 PCIe chunk size

  // 确保 P2P chunksize 不大于 coll chunksize
  if (comm->p2pChunkSize * NCCL_STEPS > comm->buffSizes[NCCL_PROTO_SIMPLE])
    comm->p2pChunkSize = comm->buffSizes[NCCL_PROTO_SIMPLE] / NCCL_STEPS;

  // 确保 split comm 的 p2pChunkSize 不超过 shared 的值
  if (comm->sharedRes->owner != comm) {
    comm->p2pChunkSize = std::min(comm->p2pChunkSize, comm->sharedRes->tpP2pChunkSize);
  } else {
    comm->sharedRes->tpP2pChunkSize = comm->p2pChunkSize;
  }

  INFO(NCCL_INIT, "P2P Chunksize set to %d", comm->p2pChunkSize);
  return ncclSuccess;
}

// 更多环境变量参数
NCCL_PARAM(GraphDumpFileRank, "GRAPH_DUMP_FILE_RANK", 0);
NCCL_PARAM(CollNetNodeThreshold, "COLLNET_NODE_THRESHOLD", 2);
NCCL_PARAM(NvbPreconnect, "NVB_PRECONNECT", 1);
NCCL_PARAM(AllocP2pNetLLBuffers, "ALLOC_P2P_NET_LL_BUFFERS", 0);

// MNNVL: 是否启用 Multi-Node NVLink
NCCL_PARAM(MNNVLEnable, "MNNVL_ENABLE", 2);

// 初始化计时器定义
#define TIMER_INIT_TOTAL 0
#define TIMER_INIT_KERNELS 1
#define TIMER_INIT_BOOTSTRAP 2
#define TIMER_INIT_ALLGATHER 3
#define TIMER_INIT_TOPO 4
#define TIMER_INIT_GRAPHS 5
#define TIMER_INIT_CONNECT 6
#define TIMER_INIT_ALLOC 7
#define TIMERS_INIT_COUNT 8

// ============================================================================
// initNvlDomainInfo - 初始化 NVLink 域信息
// ============================================================================
static ncclResult_t initNvlDomainInfo(struct ncclComm* comm) {
  // 初始化 NVLink 域信息
  comm->nvlDomainInfo.nNvlDomains = comm->nNodes;
  comm->nvlDomainInfo.minRanksPerNvlDomain = comm->minLocalRanks;
  comm->nvlDomainInfo.maxRanksPerNvlDomain = comm->maxLocalRanks;

  TRACE(NCCL_INIT, "NVLink domains: %d domains, min ranks per domain: %d, max ranks per domain: %d",
        comm->nNodes, comm->nvlDomainInfo.minRanksPerNvlDomain, comm->nvlDomainInfo.maxRanksPerNvlDomain);

  return ncclSuccess;
}
// ============================================================================
// initTransportsRank - 传输层初始化（核心函数）
// 这是 NCCL 初始化中最复杂的函数，负责：
// 1. 通过 Bootstrap 交换所有 rank 的信息
// 2. 拓扑发现和路径计算
// 3. 构建通信图（Ring、Tree、CollNet、NVLS）
// 4. 建立连接
// 5. 启动 Proxy 服务线程
//
// 该函数采用多次 AllGather 模式来逐步收集和同步所有 rank 的信息：
// - AllGather 1: 交换基本信息（peerInfo、comm指针、计算能力）
// - AllGather 2: 交换拓扑和图信息（nChannels、graphInfo、topoRanks）
// - AllGather 3: 交换完整的图信息（所有算法的详细拓扑数据）
//
// 这种多阶段的设计允许在早期阶段进行快速验证，然后在后续阶段
// 交换更详细的拓扑信息，从而优化整体初始化性能。
// ============================================================================
static ncclResult_t initTransportsRank(struct ncclComm* comm, struct ncclComm* parent, uint64_t timers[TIMERS_INIT_COUNT]) {
  // 我们使用 2 次 AllGather：
  // 1. { peerInfo, comm, compCap } - 交换基本信息
  // 2. { nChannels, graphInfo, topoRanks } - 交换拓扑和图信息
  ncclResult_t ret = ncclSuccess;

  // 当前 rank 在通信域中的编号（0 到 nranks-1）
  int rank = comm->rank;

  // 通信域中的总 rank 数
  int nranks = comm->nRanks;

  // 统计有多少个不同的物理节点（物理服务器）
  // 初始化为 1，后续通过 AllGather 结果统计
  int nNodes = 1;

  // 用于保存和恢复 CPU 亲和性设置
  // 在内存分配等关键操作期间，需要临时改变 CPU 亲和性
  cpu_set_t affinitySave;

  // 图结构指针（用于不同算法）
  // 每种通信算法（Ring、Tree、CollNet等）都有自己的拓扑图
  struct ncclTopoGraph* ringGraph = &comm->graphs[NCCL_ALGO_RING];             // 环形算法拓扑图
  struct ncclTopoGraph* treeGraph = &comm->graphs[NCCL_ALGO_TREE];             // 树形算法拓扑图
  struct ncclTopoGraph* collNetChainGraph = &comm->graphs[NCCL_ALGO_COLLNET_CHAIN]; // CollNet链式拓扑图
  struct ncclTopoGraph* collNetDirectGraph = &comm->graphs[NCCL_ALGO_COLLNET_DIRECT]; // CollNet直接拓扑图
  struct ncclTopoGraph* nvlsGraph = &comm->graphs[NCCL_ALGO_NVLS];             // NVLS（NVLink Switch）拓扑图

  // graphs 数组用于统一访问所有算法的拓扑图
  // 数组顺序与 NCCL_ALGO_* 枚举值对应
  struct ncclTopoGraph* graphs[NCCL_NUM_ALGORITHMS] = { treeGraph, ringGraph, collNetDirectGraph, collNetChainGraph, nvlsGraph, nvlsGraph, treeGraph };

  // 图信息结构（用于 AllGather）
  // 这个结构体包含了每个算法拓扑图的关键信息
  struct graphInfo {
    int pattern;        // 通信模式（Ring、Tree、Balanced Tree 等）
    int nChannels;      // 该算法使用的通道数量
    int sameChannels;   // 是否使用相同的通道（用于优化）
    float bwIntra;      // 节点内带宽（GB/s）
    float bwInter;      // 节点间带宽（GB/s）
    int typeIntra;      // 节点内传输类型（SHM、P2P、NVLink 等）
    int typeInter;      // 节点间传输类型（网络、NVLS 等）
    int crossNic;       // 是否跨网卡（用于负载均衡）
  };

  // AllGather 数据结构
  // 这是第三次 AllGather 的数据结构，包含完整的拓扑信息
  struct allGatherInfo {
    struct graphInfo graphInfo[NCCL_NUM_ALGORITHMS]; // 所有算法的图信息
    struct ncclTopoRanks topoRanks;                     // 拓扑排名信息
    int cpuArch;                                         // CPU 架构（x86、ARM 等）
    int cpuVendor;                                       // CPU 厂商（Intel、AMD 等）
    int localRanks;                                     // 本地节点的 rank 数
  };

  // 保存原始通道数（用于后续比较和验证）
  int nChannelsOrig;

  // 第三次 AllGather 的数据（包含完整的图信息）
  struct allGatherInfo *allGather3Data = NULL;

  // 所有 rank 的拓扑排名信息数组（指针数组）
  struct ncclTopoRanks** allTopoRanks = NULL;

  // 每个节点的第一个 rank（用于节点识别）
  int *nodesFirstRank = NULL, *nodesTreePatterns = NULL;

  // 环形算法的 rank 排序
  int *rings = NULL;

  // NVB（NVLink Bridge）peer 列表
  int* nvbPeers = NULL;

  // Proxy 连接器（用于网络传输）
  struct ncclProxyConnector proxyConn;

  // PXN（Proxy Exchange Network）peer 列表
  int* pxnPeers = NULL;

  // 父通信器的本地 rank 映射（用于通信域分割）
  int *topParentLocalRanks = NULL;

  // P2P 层级（用于确定 P2P 传输策略）
  int p2pLevel = -1;

  // 开始计时第一次 AllGather
  timers[TIMER_INIT_ALLGATHER] = clockNano();

  // ========== AllGather 1 - 交换基本信息 ==========
  // 这次 AllGather 的目的是收集每个 rank 的基本信息，包括：
  // - peerInfo: 包含 rank 号、CUDA 设备号、总线 ID、hostHash、pidHash 等
  // - comm: 指向 ncclComm 结构的指针
  // - compCap: GPU 计算能力（如 70、75、80 等）
  //
  // 这些信息用于：
  // 1. 验证所有 rank 的配置一致性（NCCL 版本、GPU 设备等）
  // 2. 确定节点数量和节点拓扑
  // 3. 检测是否支持 cuMem
  // 4. 建立进程内 rank 之间的通信链表

  // 分配空间，存储所有 rank 的 peerInfo 信息
  // +1 是为了额外的 rank（用于 CollNet root，作为特殊的聚合点）
  NCCLCHECKGOTO(ncclCalloc(&comm->peerInfo, nranks + 1), ret, fail);

  // 填充本 rank 的 peerInfo 信息
  // fillInfo 会将本 rank 的所有相关信息（设备、内存、网络等）填入 peerInfo 结构
  NCCLCHECKGOTO(fillInfo(comm, comm->peerInfo + rank, comm->commHash), ret, fail);

  // 调用 Bootstrap 同步所有 rank 的 peerInfo 信息
  // bootstrapAllGather 会收集所有 rank 的 peerInfo，并分发给每个 rank
  // 这样每个 rank 都能看到通信域中所有其他 rank 的信息
  NCCLCHECKGOTO(bootstrapAllGather(comm->bootstrap, comm->peerInfo, sizeof(struct ncclPeerInfo)), ret, fail);

  // 同步完成，设置 peerInfo 有效标志
  // 使用原子存储确保内存可见性，其他线程可以看到这个更新
  __atomic_store_n(&comm->peerInfoValid, true, __ATOMIC_RELEASE);

  // 先假设支持 cuMem
  // 如果后续发现有 rank 不支持，会将此标志设为 0
  comm->cuMemSupport = 1;

  // 同一个通信组内的所有 rank 都会校验参数有效性
  // 这是确保通信域一致性的关键步骤
  for (int i = 0; i < nranks; i++) {
    // 要求通信组NCCL 版本号必须一致
    // 不同版本可能有协议不兼容的问题
    if (comm->peerInfo[i].version != comm->peerInfo[rank].version) {
      WARN("Mismatched NCCL version detected : rank %d version %d rank %d version %d",
           i, comm->peerInfo[i].version, rank, comm->peerInfo[rank].version);
      ret = ncclInvalidUsage;
      goto fail;
    }

    // 检查节点数量，hostHash 不同表示不在同一物理机上
    // hostHash 是主机名和 PID 的 hash 值
    if (comm->peerInfo[i].hostHash != comm->peerInfo[rank].hostHash)
        nNodes++;

    // 只要有一个 rank 不支持 cuMem，则不使用 cuMem
    // cuMem 是 CUDA 的统一内存管理，需要所有 rank 都支持才能使用
    if (!comm->peerInfo[i].cuMemSupport)
        comm->cuMemSupport = 0;

    // 检查是否有重复的 GPU
    // 同一个通信域内，不同 Rank 必须绑定不同物理 GPU
    // 例如：Rank 0 和 Rank 1 都绑定 GPU 0 是非法的
    // hostHash 相同表示在同一节点，busId 相同表示同一 GPU
    if ((i != rank) &&
        (comm->peerInfo[i].hostHash == comm->peerInfo[rank].hostHash) &&
        (comm->peerInfo[i].busId == comm->peerInfo[rank].busId)) {
      WARN("Duplicate GPU detected : rank %d and rank %d both on CUDA device %lx", rank, i, comm->peerInfo[rank].busId);
      ret = ncclInvalidUsage;
      goto fail;
    }
  }

  // AllGather 1 结束，记录耗时
  timers[TIMER_INIT_ALLGATHER] = clockNano() - timers[TIMER_INIT_ALLGATHER];

  // 检查 MNNVL（Multi-Node NVLink）支持
  // MNNVL 允许跨节点的 NVLink 直接通信，无需经过 PCIe
  NCCLCHECKGOTO(ncclGetUserP2pLevel(&p2pLevel), ret, fail);

  // 如果满足以下任一条件，启用 MNNVL 检查：
  // 1. 多节点环境 + MNNVL_ENABLE 非默认值 + P2P level > 0
  // 2. MNNVL_ENABLE 强制启用（值为 1）
  if ((nNodes > 1 && ncclParamMNNVLEnable() != 0 && p2pLevel != 0)  || ncclParamMNNVLEnable() == 1) {
    NCCLCHECKGOTO(ncclMnnvlCheck(comm), ret, fail);
  }


#if 0
  这段代码主要处理以下场景：
   ┌─────────────┬────────────────────────────────────────────────────────────────┐
   │    场景     │                              描述                              │
   ├─────────────┼────────────────────────────────────────────────────────────────┤
   │ 单进程多GPU │ 一个进程控制多个GPU（如 mpirun -np 1 启动，进程内创建4个comm） │
   ├─────────────┼────────────────────────────────────────────────────────────────┤
   │ 多线程      │ 同一进程的多个线程各自创建NCCL comm                            │
   ├─────────────┼────────────────────────────────────────────────────────────────┤
   │ 进程内同步  │ 通过intraComm0和屏障变量实现进程内同步                         │
   └─────────────┴────────────────────────────────────────────────────────────────┘
#endif
  // ============================================================
  // 进程内 comm 检测和同步设置
  // ============================================================
  //
  // 这段代码处理以下场景：
  // 1. 单进程多 GPU：一个进程控制多个 GPU（如 mpirun -np 1）
  // 2. 多线程：同一进程的多个线程各自创建 NCCL comm
  // 3. 进程内同步：通过 intraComm0 和屏障变量实现进程内同步
  //
  // 为什么需要这个检测？
  // - 同一进程内的多个 comm 需要特殊的同步机制
  // - 进程内 comm 可以通过共享内存直接通信（不需要网络）
  // - 某些功能（如 NVLS 注册）在进程内多 comm 时不支持
  //
  // ============================================================

  // 使用 do-while(0) 结构的作用：
  // 1. 提供一个局部作用域（变量只在内部可见）
  // 2. 可以使用 break 提前退出
  // 3. 不影响后续代码流程（while(0) 只执行一次）
  // 4. 避免变量名冲突（如 i, j 等通用变量名）
  do {
    // ============================================================
    // 进程内 rank 检测变量
    // ============================================================
    //
    // 这三个变量用于识别和定位同一进程内的所有 ranks
    //
    // intraProcRank0: 同一进程中第一个 rank 的全局 rank 号
    // - 例如：rank 0, 2, 4 在同一进程，intraProcRank0 = 0
    // - 用作进程内 comm 的"根"或"代表"
    //
    // intraProcRank: 当前 rank 在进程内的序号（0-based）
    // - 例如：进程内有 ranks [0, 2, 4]，当前 rank=2
    // - intraProcRank = 1（第二个位置）
    //
    // intraProcRanks: 同一进程内的 rank 总数
    // - 例如：进程内包含 ranks [0, 2, 4, 6]
    // - intraProcRanks = 4
    //
    // 初始值设为 -1 表示"未找到"或"无效"
    // ============================================================

    // 同一进程中第一个rank的索引
    int intraProcRank0 = -1;  // 初始值 -1 表示"未找到"
    // 当前rank在进程内的序号（0 到 intraProcRanks-1）
    int intraProcRank = -1;
    // 同一进程内的rank数量（至少为 1，因为至少包含当前 rank）
    int intraProcRanks = 0;

    // ============================================================
    // 统计通信域内 GPU 的最小和最大计算能力
    // ============================================================
    //
    // 计算能力（Compute Capability）格式：
    // - 例如：70 (Volta), 75 (Turing), 80 (Ampere), 89 (Hopper)
    // - 格式：Major*10 + Minor（如 7.0 -> 70）
    //
    // 为什么要统计最小和最大计算能力？
    // 1. 内核兼容性：
    //    - 不同计算能力的 GPU 可能需要不同的内核实现
    //    - 内核必须在最小计算能力的 GPU 上也能运行
    //    - 例如：mix A100 (80) 和 V100 (70)，必须用 V100 的内核
    //
    // 2. 性能调优：
    //    - maxCompCap 决定可以使用的高级特性
    //    - minCompCap 决定内核的最低要求
    //    - tuner 可以根据这个范围选择最优算法
    //
    // 3. 功能检测：
    //    - 某些功能只在特定计算能力上可用
    //    - 例如：Tensor Cores 在 Volta (70) 之后才支持
    //
    // ============================================================

    // 第一遍循环：找最小计算能力
    // std::min: 比较并返回较小值
    // comm->peerInfo[i].cudaCompCap: rank i 的 GPU 计算能力
    // comm->minCompCap: 通信域中的最小计算能力（初始化为当前 rank 的）
    for (int i = 0; i < nranks; i++)
        comm->minCompCap = std::min(comm->minCompCap, comm->peerInfo[i].cudaCompCap);

    // 第二遍循环：找最大计算能力
    // std::max: 比较并返回较大值
    // comm->maxCompCap: 通信域中的最大计算能力（初始化为当前 rank 的）
    for (int i = 0; i < nranks; i++)
        comm->maxCompCap = std::max(comm->maxCompCap, comm->peerInfo[i].cudaCompCap);

    // ============================================================
    // NVLS 注册支持初始化
    // ============================================================
    //
    // NVLS (NVLink Sharp) 注册：
    // - 允许将内存缓冲区注册到 NVLink 网络中
    // - 实现零拷贝的高性能通信
    //
    // 初始值设为 1（假设支持）：
    // - 乐观策略，先假设支持
    // - 后续检测会根据实际情况调整
    // - 如果发现不支持的配置，会设为 0
    //
    // 哪些情况下不支持？
    // 1. 同一进程内启动了多个 comm
    // 2. 启用了 MNNVL（Multi-Node NVLink）
    // 3. 其他硬件或配置限制
    //
    // ============================================================

    comm->nvlsRegSupport = 1;  // 初始值：假设支持

    // ============================================================
    // 检查同一物理节点内的同一进程或多线程中启动了多个 comm
    // ============================================================
    //
    // 这段代码的核心逻辑：
    // 1. 遍历所有 ranks，找出与当前 rank 在同一进程的 ranks
    // 2. 构建进程内 comm 链表（intraNext）
    // 3. 检查 NVLS 注册支持
    //
    // 如何判断在同一进程？
    // - hostHash 相同：在同一物理节点（同一台机器）
    // - pidHash 相同：在同一进程（相同的进程 ID）
    //
    // 示例场景：
    //
    // 场景 1：单进程 4 GPU
    //   ┌─────────────────────────────────┐
    //   │         进程 (PID 1234)          │
    //   │  ┌───┐ ┌───┐ ┌───┐ ┌───┐       │
    //   │  │ 0 │ │ 1 │ │ 2 │ │ 3 │ (GPU) │
    //   │  └───┘ └───┘ └───┘ └───┘       │
    //   │    rank0 rank1 rank2 rank3      │
    //   │    hostHash=A hostHash=A         │
    //   │    pidHash=B pidHash=B           │
    //   └─────────────────────────────────┘
    //   intraProcRank0 = 0
    //   intraProcRanks = 4
    //   rank0: intraProcRank=0, intraNext=1
    //   rank1: intraProcRank=1, intraNext=2
    //   rank2: intraProcRank=2, intraNext=3
    //   rank3: intraProcRank=3, intraNext=NULL
    //
    // 场景 2：两进程，每进程 2 GPU
    //   ┌──────────────────┐  ┌──────────────────┐
    //   │  进程 1 (PID 10)  │  │  进程 2 (PID 20)  │
    //   │  ┌───┐ ┌───┐     │  │  ┌───┐ ┌───┐     │
    //   │  │ 0  │ │ 2 │(GPU)│  │  │ 1 │ │ 3 │(GPU)│
    //   │  └───┘ └───┘     │  │  └───┘ └───┘     │
    //   │   rank0  rank2   │  │   rank1  rank3   │
    //   └──────────────────┘  └──────────────────┘
    //   进程 1: intraProcRank0=0, intraProcRanks=2
    //   进程 2: intraProcRank0=1, intraProcRanks=2
    //
    // ============================================================

    for (int i = 0; i < nranks; i++) {
      // 检查 rank i 是否与当前 rank 在同一进程
      // hostHash 相同：同一物理节点
      // pidHash 相同：同一进程
      if ((comm->peerInfo[i].hostHash == comm->peerInfo[rank].hostHash)
          && (comm->peerInfo[i].pidHash == comm->peerInfo[rank].pidHash)) {

        // ============================================================
        // 找到同一进程内的一个 rank
        // ============================================================

        // 记录同一进程中第一个 rank 的全局 rank 号
        // intraProcRanks == 0：这是找到的第一个同进程 rank
        // intraProcRank0 = i：记录这个 rank 的全局编号
        if (intraProcRanks == 0)
            intraProcRank0 = i;  // 记录同一进程的第一个 rank 号

        // 如果 rank i 就是当前 rank，记录它在进程内的序号
        // intraProcRanks 是当前的计数（从 0 开始）
        // 例如：找到的第 2 个同进程 rank，intraProcRank = 1
        if (i == rank)
            intraProcRank = intraProcRanks;

        // 同一进程内的 Rank 数量递增
        // 每找到一个同进程 rank，计数器加 1
        intraProcRanks++;

        // ============================================================
        // 构建进程内 comm 链表
        // ============================================================
        //
        // intraNext 指针的作用：
        // - 将同一进程内的所有 comm 串联成一个链表
        // - 用于进程内同步和通信
        // - 第一个 comm (intraProcRank0) 是链表的头部
        //
        // 链表结构：
        //   comm0 -> comm1 -> comm2 -> comm3 -> NULL
        //   (进程内第1个)  (进程内第2个) ...
        //
        // 链表插入逻辑（头插法）：
        // - 只有 intraProcRank0（第一个 rank）负责构建链表
        // - 新元素插入到链表头部
        // - 原来的 intraNext 变成新元素的 next
        //
        // 为什么要用头插法？
        // 1. 简单：只需要一个指针操作
        // 2. 高效：O(1) 时间复杂度
        // 3. 可靠：不需要遍历链表
        //
        // 示例：假设 ranks [0, 2, 4] 在同一进程
        // 初始状态：comm0->intraNext = NULL
        //
        // i=2: comm0->intraNext = comm2
        //       comm0 -> comm2 -> NULL
        //
        // i=4: comm0->intraNext = comm4, comm4->intraNext = comm2
        //       comm0 -> comm4 -> comm2 -> NULL
        //
        // 为什么要只在 intraProcRank0 构建链表？
        // - 避免竞争：多个 rank 同时修改链表会出问题
        // - 统一管理：只需要一个 rank 负责构建
        // - 一致性：所有 rank 最终会获得相同的链表结构
        //
        // 条件：intraProcRank0 == rank && rank != i
        // - intraProcRank0 == rank：当前 rank 是进程内第一个
        // - rank != i：不指向自己
        //
        // ============================================================

        if (intraProcRank0 == rank && rank != i) {
          // 将 rank i 的 comm 插入到链表头部
          // comm->peerInfo[i].comm: rank i 的 comm 指针
          // comm->intraNext: 当前的链表头
          // 操作：comm[i]->intraNext = comm->intraNext
          comm->peerInfo[i].comm->intraNext = comm->intraNext;
          // 更新链表头指向 comm[i]
          // comm->intraNext: 现在指向 rank i 的 comm
          comm->intraNext = comm->peerInfo[i].comm;
        }
      }

      // ============================================================
      // 检查是否支持 NVLS 注册
      // ============================================================
      //
      // NVLS 注册：将内存注册到 NVLink 网络中
      //
      // 为什么进程内多 comm 不支持 NVLS 注册？
      // 1. 资源冲突：
      //    - NVLS 注册需要在 GPU 上分配特定资源
      //    - 多个 comm 会竞争这些资源
      //    - 可能导致资源耗尽或冲突
      //
      // 2. 内存管理复杂：
      //    - 多个 comm 可能有不同的内存分配策略
      //    - NVLS 注册需要统一的内存管理
      //    - 进程内多 comm 难以保证一致性
      //
      // 3. 同步问题：
      //    - NVLS 注册需要精确的同步
      //    - 多个 comm 可能有不同的初始化顺序
      //    - 可能导致死锁或竞争
      //
      // 检测逻辑：
      // - 外层循环：遍历所有 ranks (i)
      // - 内层循环：遍历 i 之后的所有 ranks (j)
      // - 如果 rank i 和 rank j 在同一进程
      // - 说明有多个 comm 在同一进程
      // - 将 nvlsRegSupport 设为 0
      //
      // 注意：这是双重循环，时间复杂度 O(n^2)
      // 但因为只执行一次，所以可以接受
      //
      // ============================================================

      // 只有在当前支持时才检查（优化：一旦发现不支持就不再检查）
      if (comm->nvlsRegSupport) {
        // 内层循环：检查 rank i 之后的 ranks
        // j 从 i+1 开始，避免重复检查
        for (int j = i + 1; j < nranks; j++) {
          // 如果 rank i 和 rank j 在同一节点且同一进程
          // hostHash 相同：同一物理节点
          // pidHash 相同：同一进程
          if (comm->peerInfo[i].hostHash == comm->peerInfo[j].hostHash &&
              comm->peerInfo[i].pidHash == comm->peerInfo[j].pidHash) {
            // 发现同一进程内有多个 comm
            // 将 nvlsRegSupport 设为 0（不支持）
            comm->nvlsRegSupport = 0;
            break;  // 跳出内层循环（已经知道不支持，无需继续检查）
          }
        }
      }
      // 如果 nvlsRegSupport 变成 0，后续的迭代会跳过检查（if 条件不满足）
    }

    // ============================================================
    // MNNVL 和单进程内存注册对 NVLS 的影响
    // ============================================================
    //
    // MNNVL (Multi-Node NVLink):
    // - 允许跨节点的 GPU 通过 NVLink 互联
    // - 使用特殊的内存管理机制
    // - 与 NVLS 注册不兼容
    //
    // ncclParamSingleProcMemRegEnable():
    // - 环境变量 NCCL_SINGLE_PROC_MEM_REG_ENABLE
    // - 强制启用单进程内存注册
    // - 可以在某些情况下启用 NVLS 注册
    //
    // 逻辑流程：
    // 1. 如果启用 MNNVL -> 不支持 NVLS 注册
    // 2. 否则，如果启用单进程内存注册 -> 支持 NVLS 注册
    // 3. 否则，保持之前的状态（由双重循环决定）
    //
    // ============================================================

    // MNNVL 不支持 Buffer Registration
    // MNNVL: Multi-Node NVLink 是否启用
    if (comm->MNNVL)
        comm->nvlsRegSupport = 0;  // MNNVL 模式下不支持 NVLS 注册
    else if (ncclParamSingleProcMemRegEnable())
        comm->nvlsRegSupport = 1;  // 强制启用单进程内存注册时支持

    // ============================================================
    // 跟踪日志输出
    // ============================================================
    // 输出进程内 rank 信息，用于调试
    // - pidHash[rank]: 进程的 hash 值
    // - intraProcRank: 当前 rank 在进程内的序号
    // - intraProcRanks: 进程内 rank 总数
    // - intraProcRank0: 进程内第一个 rank 的全局 rank 号
    //
    // TRACE 宏：只在调试模式下输出
    // NCCL_INIT: 日志子系统（初始化相关）
    //
    // 示例输出：
    // TRACE NCCL_INIT pidHash[2] 0x12345678 intraProcRank 1 intraProcRanks 4 intraProcRank0 0
    // 含义：rank 2 在进程内是第 2 个（序号 1），进程内共有 4 个 rank，第一个是 rank 0
    //
    // ============================================================

    TRACE(NCCL_INIT, "pidHash[%d] %lx intraProcRank %d intraProcRanks %d intraProcRank0 %d",
        rank, comm->peerInfo[rank].pidHash, intraProcRank, intraProcRanks, intraProcRank0);

    // ============================================================
    // 错误检查：进程内 rank 检测是否成功
    // ============================================================
    //
    // 检查条件：
    // 1. intraProcRank == -1：未找到当前 rank 在进程内的序号
    //    - 这不应该发生（至少应该找到自己）
    //    - 如果发生，说明 hostHash/pidHash 计算有误
    //
    // 2. intraProcRank0 == -1：未找到进程内第一个 rank
    //    - 这也不应该发生（至少应该有当前 rank）
    //    - 如果发生，说明进程检测逻辑有问题
    //
    // 3. comm->peerInfo[intraProcRank0].comm == NULL：comm 指针无效
    //    - peerInfo 中应该包含有效的 comm 指针
    //    - 如果为 NULL，说明 fillInfo 有问题
    //
    // 错误处理：
    // - 输出警告信息（包含所有相关状态）
    // - 设置返回值为 ncclInternalError
    // - 跳转到 fail 标签（清理资源并返回）
    //
    // ============================================================

    if (intraProcRank == -1 || intraProcRank0 == -1 || comm->peerInfo[intraProcRank0].comm == NULL) {
      // 输出详细的错误信息
      WARN("Failed to determine intra proc ranks rank %d hostHash %lx pidHash %lx intraProcRank %d intraProcRanks %d intraProcRank0 %d",
          rank, comm->peerInfo[rank].hostHash, comm->peerInfo[rank].pidHash,
          intraProcRank, intraProcRanks, intraProcRank0);
      ret = ncclInternalError;  // 内部错误
      goto fail;  // 跳转到 fail 标签
    }

    // ============================================================
    // 获取进程内第一个 comm 的指针
    // ============================================================
    //
    // comm0: 进程内第一个 rank 的 comm 指针
    // - 作为进程内 comm 的"根"或"代表"
    // - 用于进程内同步和协调
    //
    // 示例：
    // - 进程内有 ranks [0, 2, 4]
    // - intraProcRank0 = 0
    // - comm0 = comm->peerInfo[0].comm (rank 0 的 comm)
    //
    // ============================================================

    struct ncclComm* comm0 = comm->peerInfo[intraProcRank0].comm;

    // ============================================================
    // 断言：验证进程内第一个 comm 的正确性
    // ============================================================
    //
    // assert 条件：intraProcRank == 0 ? comm == comm0 : true
    //
    // 含义：
    // - 如果 intraProcRank == 0（当前 rank 是进程内第一个）
    // - 则 comm 应该等于 comm0（就是自己）
    // - 否则（intraProcRank != 0），不需要验证
    //
    // 为什么要这个断言？
    // 1. 验证逻辑正确性：
    //    - 进程内第一个 rank 的 comm 应该就是自己
    //    - 如果不是，说明 intraProcRank0 找错了
    //
    // 2. 早期发现问题：
    //    - 断言在 debug 模式下会触发
    //    - 帮助开发者快速定位问题
    //
    // 注意：
    // - release 模式下断言会被移除
    // - 只用于调试，不影响运行时逻辑
    //
    // ============================================================

    assert(intraProcRank == 0 ? comm == comm0 : true);

    // ============================================================
    // 设置进程内 comm 的相关字段
    // ============================================================
    //
    // 这些字段用于进程内同步和通信
    //
    // 1. intraComm0: 进程内第一个 comm 的指针
    //    - 可能指向自己（单 GPU 进程）
    //    - 可能指向其他 comm（多 GPU 进程）
    //
    // 2. intraRank: 当前 rank 在进程内的序号
    //    - 0 到 intraRanks-1
    //    - 用于进程内数组索引
    //
    // 3. intraRanks: 进程内的 rank 总数
    //    - 至少为 1（至少包含当前 rank）
    //    - 用于分配进程内资源
    //
    // 4. intraBarrierPhase: 进程内屏障的相位
    //    - 0: 初始相位
    //    - 用于实现进程内屏障（bootstrapIntraProcessBarrier）
    //
    // 5. intraBarrierCounter: 进程内屏障的计数器
    //    - 记录有多少个 rank 到达了屏障
    //    - 0: 初始值
    //
    // 6. intraBarrierGate: 进程内屏障的门控
    //    - 用于同步屏障的进入和退出
    //    - 0: 初始值
    //
    // 示例场景：
    //
    // 场景 1：单进程单 GPU
    //   intraComm0 = comm->self (指向自己)
    //   intraRank = 0
    //   intraRanks = 1
    //
    // 场景 2：单进程 4 GPU
    //   intraComm0 = rank0 的 comm
    //   rank0: intraRank = 0, intraRanks = 4
    //   rank1: intraRank = 1, intraRanks = 4
    //   rank2: intraRank = 2, intraRanks = 4
    //   rank3: intraRank = 3, intraRanks = 4
    //
    // 进程内屏障的工作原理：
    //   当一个 rank 到达屏障时：
    //   1. intraBarrierCounter++
    //   2. 如果 intraBarrierCounter == intraRanks
    //      - 所有 rank 都到达
    //      - intraBarrierPhase++ (切换相位)
    //      - 唤醒所有等待的 rank
    //   3. 否则等待
    //
    // ============================================================

    // 这里 intraComm0 可能会指向自己（比如每个 GPU 一个进程的工作模式）
    // 指向进程内第一个 comm
    comm->intraComm0 = comm0;

    // 当前 rank 在进程内的序号（0-based）
    comm->intraRank = intraProcRank;

    // 进程内的 rank 总数
    comm->intraRanks = intraProcRanks;

    // 进程内屏障的初始相位（0）
    comm->intraBarrierPhase = 0;

    // 进程内屏障的计数器（初始为 0）
    comm->intraBarrierCounter = 0;

    // 进程内屏障的门控（初始为 0）
    comm->intraBarrierGate = 0;

  // while(0)：只执行一次，提供局部作用域
  } while(0);

  timers[TIMER_INIT_TOPO] = clockNano();

  // 如果用户请求，转储拓扑 XML 文件
  const char* dumpXmlFile;
  dumpXmlFile = ncclGetEnv("NCCL_TOPO_DUMP_FILE");
  if (dumpXmlFile) {
    // 只 dump 拓扑 XML 文件
    NCCLCHECKGOTO(ncclTopoGetSystem(comm, NULL, dumpXmlFile), ret, fail);
  }

  // ========== 拓扑检测 / 系统图创建 ==========
  // 获取系统的拓扑信息，并存储在 comm 的 topo 成员中
  NCCLCHECKGOTO(ncclTopoGetSystem(comm, &comm->topo), ret, fail);

  // 计算 GPU 和 NIC 之间的路径
  NCCLCHECKGOTO(ncclTopoComputePaths(comm->topo, comm), ret, fail);

  // 移除不可访问的 GPU 和未使用的 NIC
  NCCLCHECKGOTO(ncclTopoTrimSystem(comm->topo, comm), ret, fail);

  // 在移除不可访问的组件后，重新计算路径
  NCCLCHECKGOTO(ncclTopoComputePaths(comm->topo, comm), ret, fail);

  // 初始化拓扑搜索
  NCCLCHECKGOTO(ncclTopoSearchInit(comm->topo), ret, fail);

  // 决定 comm 的 CPU 架构
  NCCLCHECKGOTO(ncclTopoComputeCommCPU(comm), ret, fail);

  // 打印最终的拓扑结构（用于调试或信息展示）
  NCCLCHECKGOTO(ncclTopoPrint(comm->topo), ret, fail);

  timers[TIMER_INIT_TOPO] = clockNano() - timers[TIMER_INIT_TOPO];

  // 设置 CPU 亲和性到本地 GPU，确保分配的所有主机内存在本地
  NCCLCHECKGOTO(ncclTopoGetCpuAffinity(comm->topo, comm->rank, &comm->cpuAffinity), ret, fail);
  if (CPU_COUNT(&comm->cpuAffinity)) {
    sched_getaffinity(0, sizeof(cpu_set_t), &affinitySave);
    sched_setaffinity(0, sizeof(cpu_set_t), &comm->cpuAffinity);
  }

  // 确定本地 CollNet 支持
  // ncclCollNet 为 NULL 表示不支持 CollNet
  if (!collNetSupport(comm)) {
    comm->config.collnetEnable = 0;
  }

  // 确定本地 NVLS 支持
  // 检查是否支持 NVLS Sharp（NVLink Shared Memory for Collectives）
  NCCLCHECK(ncclNvlsInit(comm));

  timers[TIMER_INIT_GRAPHS] = clockNano();

  // ============================================================
  // 开始统计带宽和通道信息 - 拓扑图计算
  // ============================================================
  //
  // 这段代码是 NCCL 初始化的核心部分之一，负责：
  // 1. 为不同的通信算法（Ring、Tree、CollNet、NVLS）计算拓扑图
  // 2. 统计每个算法的带宽和通道信息
  // 3. 为后续的算法选择提供数据支持
  //
  // 拓扑图（Topo Graph）的作用：
  // - 描述通信节点（GPU/NIC）之间的连接关系
  // - 包含带宽、延迟、传输类型等信息
  // - 用于计算最优的通信路径和通道分配
  //
  // 通信算法说明：
  // - Ring：环形拓扑，适合大消息
  // - Tree：树形拓扑，适合中小消息
  // - CollNet：集合网络，需要特殊硬件支持
  // - NVLS：NVLink Sharp，NVLink 优化的集合通信
  //
  // ============================================================

  // ============================================================
  // Ring（环形）拓扑图计算
  // ============================================================
  //
  // Ring 算法的特点：
  // - 每个 rank 只与两个邻居通信（前驱和后继）
  // - 适合大消息传输（高带宽利用率）
  // - 实现：rank (i-1) → rank i → rank (i+1)

  //通信重叠指的是：每个节点在每一步都同时发送和接收，所有节点和所有链路都在并行工作，
  // 为什么 Ring 适合大消息？
  // - 通信可以重叠（流水线）
  // - 带宽利用率高（所有链路同时工作）
  // - 延迟随 rank 数线性增长
  //
  // 参数说明：
  // - minChannels = 1：至少需要 1 个通道
  // - maxChannels = 32（MAXCHANNELS/2）：最多 32 个通道
  //   * Ring 使用双向通信，每个方向占一个通道
  //   * 实际上 Ring 最多使用 32 个通道 = 64/2
  //
  // ============================================================

  // 清零 ringGraph 结构体
  // memset: 将内存设置为全 0
  // sizeof(struct ncclTopoGraph): 结构体的大小
  memset(ringGraph, 0, sizeof(struct ncclTopoGraph));

  // 设置 Ring 图的 ID（0 表示 Ring）
  ringGraph->id = 0;

  // 设置拓扑模式为环形
  // NCCL_TOPO_PATTERN_RING: 环形拓扑枚举值
  ringGraph->pattern = NCCL_TOPO_PATTERN_RING;

  // 设置最小通道数为 1
  ringGraph->minChannels = 1;           // 最小通道数

  // 设置最大通道数为 32（MAXCHANNELS/2）
  ringGraph->maxChannels = MAXCHANNELS / 2;  // 最大通道数（32）

  // ============================================================
  // ncclTopoCompute - 计算拓扑图
  // ============================================================
  //
  // 这个函数是拓扑发现的核心，它会：
  // 1. 遍历所有 GPU 和 NIC
  // 2. 计算 GPU 到 GPU 的路径
  // 3. 计算 GPU 到 NIC 的路径
  // 4. 根据路径信息评估带宽
  // 5. 选择最优的通道分配
  //
  // 输入：
  // - comm->topo: 拓扑结构（GPU、NIC、连接等）
  // - ringGraph: 要计算的图（包含配置参数）
  //
  // 输出：
  // - ringGraph 被填充，包含：
  //   * nChannels: 实际可用的通道数
  //   * bwIntra: 节点内带宽
  //   * bwInter: 节点间带宽
  //   * typeIntra: 节点内传输类型（SHM/P2P/NVLink）
  //   * typeInter: 节点间传输类型（Network）
  //   * sameChannels: 是否所有通道配置相同
  //   * crossNic: 是否跨网卡
  //
  // ============================================================

  // 填入 ringGraph 的带宽和通道信息
  NCCLCHECKGOTO(ncclTopoCompute(comm->topo, ringGraph), ret, fail);

  // 打印 Ring 图信息（用于调试）
  // 输出内容包括：通道数、带宽、传输类型等
  NCCLCHECKGOTO(ncclTopoPrintGraph(comm->topo, ringGraph), ret, fail);

  // ============================================================
  // Tree（树形）拓扑图计算
  // ============================================================
  //
  // Tree 算法的特点：
  // - 分层结构：根节点、中间节点、叶子节点
  // - 适合中小消息传输（低延迟）
  // - 实现：叶子 → 根 → 叶子（归约和广播）
  //
  // 为什么 Tree 适合小消息？
  // - 跳数少（log2(N) 层）
  // - 延迟低
  // - 带宽利用率不如 Ring
  //
  // BALANCED_TREE（平衡树）：
  // - 每个节点的子节点数量大致相同
  // - NCCL 中每个节点最多 3 个子节点
  // - 树的高度为 log2(N)
  //
  // 参数说明：
  // - minChannels = ringGraph->nChannels：使用 Ring 计算出的通道数
  // - maxChannels = ringGraph->nChannels：保持一致
  //   * Tree 使用与 Ring 相同的通道配置
  //   * 确保算法切换时的兼容性
  //
  // 示例（8 个 ranks 的平衡树）：
  //           Rank 0 (根)
  //          /    |    \
  //       Rank 1 Rank 2 Rank 4
  //       /  \    /  \    /  \
  //     3    5  6    7  ...
  //
  // ============================================================

  // 清零 treeGraph 结构体
  memset(treeGraph, 0, sizeof(struct ncclTopoGraph));

  // 设置 Tree 图的 ID（1 表示 Tree）
  treeGraph->id = 1;

  // 设置拓扑模式为平衡树
  // NCCL_TOPO_PATTERN_BALANCED_TREE: 平衡树拓扑枚举值
  treeGraph->pattern = NCCL_TOPO_PATTERN_BALANCED_TREE;

  // 设置最小/最大通道数为 Ring 的通道数
  // Tree 使用与 Ring 相同的通道配置
  treeGraph->minChannels = ringGraph->nChannels;
  treeGraph->maxChannels = ringGraph->nChannels;

  // 计算 Tree 图的带宽和通道信息
  NCCLCHECKGOTO(ncclTopoCompute(comm->topo, treeGraph), ret, fail);

  // 打印 Tree 图信息（用于调试）
  NCCLCHECKGOTO(ncclTopoPrintGraph(comm->topo, treeGraph), ret, fail);

  // ============================================================
  // CollNet Chain（集合网络链式）拓扑图
  // ============================================================
  //
  // CollNet (Collective Network) 说明：
  // - 使用特殊硬件（如 BlueField DPU、NVSwitch）实现集合操作
  // - 硬件加速的集合操作（AllReduce、AllGather 等）
  // - 比纯软件实现更快
  //
  // CollNet Chain 模式：
  // - 多个节点通过硬件网络链式连接
  // - 适合多节点、多 GPU 的场景
  // - 类似于 Tree 结构，但使用硬件加速
  //
  // 参数说明：
  // - id = 2：CollNet Chain 的 ID
  // - pattern = TREE：树形拓扑模式
  // - collNet = 1：标记这是 CollNet 相关的图
  // - min/maxChannels = ringGraph->nChannels：使用 Ring 的通道数
  //
  // 为什么要 collNet = 1 标志？
  // - 区分普通 Tree 和 CollNet Tree
  // - 后续处理时会使用不同的代码路径
  // - CollNet 需要特殊的硬件支持
  //
  // ============================================================

  // CollNet Chain 图
  memset(collNetChainGraph, 0, sizeof(struct ncclTopoGraph));

  // 设置 ID 为 2（CollNet Chain）
  collNetChainGraph->id = 2;

  // 设置拓扑模式为 TREE（树形）
  collNetChainGraph->pattern = NCCL_TOPO_PATTERN_TREE;

  // 标记为 CollNet 图（与普通 Tree 区分）
  collNetChainGraph->collNet = 1;

  // 使用 Ring 的通道数
  collNetChainGraph->minChannels = ringGraph->nChannels;
  collNetChainGraph->maxChannels = ringGraph->nChannels;

  // ============================================================
  // CollNet Direct（集合网络直接）拓扑图
  // ============================================================
  //
  // CollNet Direct 模式：
  // - 所有 GPU 直接连接到 CollNet 硬件
  // - 不需要经过中间节点
  // - 更低的延迟和更高的带宽
  //
  // 与 CollNet Chain 的区别：
  // - Chain：多跳，经过中间节点
  // - Direct：单跳或少数跳，直接连接
  //
  // 参数说明：
  // - id = 4：CollNet Direct 的 ID
  // - pattern = COLLNET_DIRECT：直接连接模式
  // - minChannels = 1：最少 1 个通道
  // - maxChannels = MAXCHANNELS（64）：最多 64 个通道
  //   * Direct 模式可以使用更多通道
  //   * 不受 Ring/Tree 的限制
  //
  // 限制：
  // - 只在节点数量较少时使用（NCCL_MAX_DIRECT_ARITY + 1）
  // - 通常 <= 8 个 GPU（取决于硬件）
  //
  // ============================================================

  // CollNet Direct 图
  memset(collNetDirectGraph, 0, sizeof(struct ncclTopoGraph));

  // 设置 ID 为 4（CollNet Direct）
  collNetDirectGraph->id = 4;

  // 设置拓扑模式为 COLLNET_DIRECT
  collNetDirectGraph->pattern = NCCL_TOPO_PATTERN_COLLNET_DIRECT;

  // 标记为 CollNet 图
  collNetDirectGraph->collNet = 1;

  // 设置通道数范围（1 到 MAXCHANNELS）
  collNetDirectGraph->minChannels = 1;
  collNetDirectGraph->maxChannels = MAXCHANNELS;

  // ============================================================
  // 计算 CollNet 拓扑（如果支持）
  // ============================================================
  //
  // comm->config.collnetEnable：CollNet 是否启用
  // - 0：禁用（硬件不支持或用户禁用）
  // - 1：启用（硬件支持且用户未禁用）
  //
  // 什么时候禁用 CollNet？
  // 1. 硬件不支持（没有 BlueField 或 NVSwitch）
  // 2. 用户通过环境变量禁用（NCCL_COLLNET_ENABLE=0）
  // 3. 节点数量少于阈值（后续检查）
  //
  // 如果启用，计算 CollNet 拓扑
  // - Chain 模式拓扑
  // - Direct 模式拓扑
  //
  // ============================================================

  // 如果支持 CollNet，计算 CollNet 拓扑
  if (comm->config.collnetEnable) {
    // 计算 CollNet Chain 拓扑
    NCCLCHECKGOTO(ncclTopoCompute(comm->topo, collNetChainGraph), ret, fail);
    // 打印 CollNet Chain 图信息
    NCCLCHECKGOTO(ncclTopoPrintGraph(comm->topo, collNetChainGraph), ret, fail);

    // 计算 CollNet Direct 拓扑
    NCCLCHECKGOTO(ncclTopoCompute(comm->topo, collNetDirectGraph), ret, fail);
    // 打印 CollNet Direct 图信息
    NCCLCHECKGOTO(ncclTopoPrintGraph(comm->topo, collNetDirectGraph), ret, fail);
  }

  // ============================================================
  // NVLS（NVLink Sharp）拓扑图
  // ============================================================
  //
  // NVLS (NVLink Sharp) 说明：
  // - 基于 NVLink 的集合操作加速
  // - 需要 NVSwitch 或 NVLink Switch 硬件
  // - 比 CollNet 更快（因为使用 NVLink）
  //
  // NVLS 的优势：
  // - 带宽：NVLink 带宽远高于网络（300GB/s vs 100Gb/s）
  // - 延迟：NVLink 延迟远低于网络
  // - 统一编程模型：与 NCCL 其他算法一致
  //
  // 参数说明：
  // - id = 3：NVLS 的 ID
  // - pattern = NVLS：NVLS 模式
  // - minChannels = 1：最少 1 个通道
  // - maxChannels = MAXCHANNELS（64）：最多 64 个通道
  //
  // comm->nvlsSupport：是否支持 NVLS
  // - 在初始化阶段检测（ncclNvlsInit）
  // - 取决于硬件和驱动支持
  //
  // ============================================================

  // NVLS 图
  memset(nvlsGraph, 0, sizeof(struct ncclTopoGraph));

  // 设置 ID 为 3（NVLS）
  nvlsGraph->id = 3;

  // 设置拓扑模式为 NVLS
  nvlsGraph->pattern = NCCL_TOPO_PATTERN_NVLS;

  // 设置通道数范围
  nvlsGraph->minChannels = 1;
  nvlsGraph->maxChannels = MAXCHANNELS;

  // ============================================================
  // 计算 NVLS 拓扑（如果支持）
  // ============================================================
  //
  // 如果硬件支持 NVLS，计算 NVLS 拓扑
  // - NVLS 需要特殊的硬件支持
  // - 通常需要 NVSwitch 或 NVLink Switch
  //
  // 什么时候支持 NVLS？
  // 1. 有 NVSwitch 或 NVLink Switch 硬件
  // 2. GPU 之间通过 NVLink 全互联
  // 3. 驱动和 CUDA 版本支持
  //
  // ============================================================

  // 是否支持 NVLS Sharp
  if (comm->nvlsSupport) {
    // 计算 NVLS 拓扑
    NCCLCHECKGOTO(ncclTopoCompute(comm->topo, nvlsGraph), ret, fail);
    // 打印 NVLS 图信息
    NCCLCHECKGOTO(ncclTopoPrintGraph(comm->topo, nvlsGraph), ret, fail);
  }

  // ============================================================
  // 记录图计算耗时
  // ============================================================
  // clockNano()：获取当前时间（纳秒）
  // TIMER_INIT_GRAPHS：图计算的起始时间（之前设置的）
  // 差值就是图计算的总耗时
  //
  // 用于性能分析和调试
  // ============================================================

  timers[TIMER_INIT_GRAPHS] = clockNano() - timers[TIMER_INIT_GRAPHS];
  // ========== 统计各个算法通信图带宽流程结束 ==========

  // ============================================================
  // P2P LL 缓冲区分配设置
  // ============================================================
  //
  // P2P LL 缓冲区：
  // - LL (Long Jump) 协议使用的缓冲区
  // - 用于跨节点的 P2P 通信
  // - 通常在 GPU 显存中分配
  //
  // allocP2pNetLLBuffers：
  // - true：分配 P2P LL 网络缓冲区
  // - false：不分配（使用其他机制）
  //
  // 环境变量：NCCL_ALLOC_P2P_NET_LL_BUFFERS
  // - 0：不分配（默认）
  // - 1：分配
  //
  // 什么时候需要分配？
  // - 使用 LL 协议进行跨节点通信
  // - 需要 GPU Direct RDMA 支持
  //
  // ============================================================

  // 初始化此通信器的 P2P LL 缓冲区数量，默认为0
  // ncclParamAllocP2pNetLLBuffers()：读取环境变量 NCCL_ALLOC_P2P_NET_LL_BUFFERS
  // == 1：表示启用，返回 true
  comm->allocP2pNetLLBuffers = ncclParamAllocP2pNetLLBuffers() == 1;

  // ============================================================
  // 转储图信息到文件（用于调试）
  // ============================================================
  //
  // 什么时候转储图信息？
  // - 调试拓扑问题时
  // - 分析通道分配时
  // - 验证算法选择时
  //
  // 环境变量：NCCL_GRAPH_DUMP_FILE_RANK
  // - 指定哪个 rank 负责转储图信息
  // - 通常设为 0（第一个 rank）
  //
  // 为什么要只让一个 rank 转储？
  // - 避免文件冲突（多个 rank 写同一文件）
  // - 减少磁盘 I/O
  // - 所有 rank 的图信息基本相同
  //
  // 转储内容：
  // - Ring、Tree、CollNet Direct、CollNet Chain、NVLS 图
  // - 包含带宽、通道、连接等信息
  //
  // ============================================================

  // 如果需要，转储图信息到文件
  // 只有当前 rank 等于 NCCL_GRAPH_DUMP_FILE_RANK 环境变量时才执行
  if (comm->rank == ncclParamGraphDumpFileRank()) {
    // 创建要转储的图数组（5 个图）
    struct ncclTopoGraph* dumpGraphs[5] = { ringGraph, treeGraph, collNetDirectGraph, collNetChainGraph, nvlsGraph };
    // 转储所有图到文件
    NCCLCHECKGOTO(ncclTopoDumpGraphs(comm->topo, 5, dumpGraphs), ret, fail);
  }

  // ============================================================
  // AllGather 3 - 交换图信息
  // ============================================================
  //
  // 这是第三次也是最后一次 AllGather（之前有两次）
  //
  // 三次 AllGather 的区别：
  //
  // AllGather 1（已完成）：
  // - 交换基本信息（peerInfo）
  // - 包含：rank、cudaDev、busId、hostHash、pidHash 等
  // - 目的：验证配置、确定节点数
  //
  // AllGather 2（已完成）：
  // - 交换拓扑信息（在拓扑计算之前）
  // - 包含：GPU、NIC 等硬件信息
  // - 目的：构建系统拓扑
  //
  // AllGather 3（本次）：
  // - 交换图信息（在拓扑计算之后）
  // - 包含：所有算法的带宽、通道、传输类型等
  // - 目的：对齐所有 rank 的配置，确保一致性
  //
  // 为什么要三次 AllGather？
  // 1. 分阶段交换：逐步收集信息
  // 2. 早期验证：在早期发现配置问题
  // 3. 减少数据量：每次只交换必要的信息
  //
  // ============================================================

  // 因为 timers[TIMER_INIT_ALLGATHER] 已经包含第一次 AllGather 的时间，
  // 我们暂时将第二次 AllGather 的开始时间存储在未使用的 CONNECT 计时器中
  // 注意：这里用 CONNECT 计时器作为临时存储，后续会修正
  timers[TIMER_INIT_CONNECT] = clockNano();

  // ============================================================
  // 分配 allGather3Data 数组
  // ============================================================
  //
  // allGather3Data：存储所有 rank 的图信息
  // - 大小：nranks 个元素
  // - 每个元素包含一个 rank 的完整图信息
  //
  // 数据结构（struct allGatherInfo）：
  // - graphInfo[NCCL_NUM_ALGORITHMS]：每个算法的图信息
  // - topoRanks：拓扑排名信息
  // - cpuArch：CPU 架构
  // - cpuVendor：CPU 厂商
  // - localRanks：本地 rank 数
  //
  // ============================================================

  // 分配 allGather3Data 数组（nranks 个元素）
  NCCLCHECKGOTO(ncclCalloc(&allGather3Data, nranks), ret, fail);

  // ============================================================
  // 填充本 rank 的图信息
  // ============================================================
  //
  // 这段代码将本 rank 计算出的图信息填充到 allGather3Data 中
  // 这些信息将通过 AllGather 发送给其他 ranks
  //
  // graphInfo 结构包含：
  // - pattern：拓扑模式（Ring/Tree/CollNet/NVLS）
  // - nChannels：通道数量
  // - sameChannels：是否所有通道配置相同
  // - bwIntra：节点内带宽
  // - bwInter：节点间带宽
  // - typeIntra：节点内传输类型
  // - typeInter：节点间传输类型
  // - crossNic：是否跨网卡
  //
  // NCCL_NUM_ALGORITHMS：
  // - 算法总数（包括 Ring、Tree、CollNet Chain、CollNet Direct、NVLS 等）
  //
  // ============================================================

  // 填充本 rank 的信息，用于后续发送给其他 rank
  for (int a = 0; a < NCCL_NUM_ALGORITHMS; a++) {
    // 拓扑模式
    allGather3Data[rank].graphInfo[a].pattern = graphs[a]->pattern;
    // 通道数量
    allGather3Data[rank].graphInfo[a].nChannels = graphs[a]->nChannels;
    // 是否所有通道配置相同
    allGather3Data[rank].graphInfo[a].sameChannels = graphs[a]->sameChannels;
    // 节点内带宽（GB/s）
    allGather3Data[rank].graphInfo[a].bwIntra = graphs[a]->bwIntra;
    // 节点间带宽（GB/s）
    allGather3Data[rank].graphInfo[a].bwInter = graphs[a]->bwInter;
    // 节点内传输类型（SHM/P2P/NVLink）
    allGather3Data[rank].graphInfo[a].typeIntra = graphs[a]->typeIntra;
    // 节点间传输类型（Network/NVLS）
    allGather3Data[rank].graphInfo[a].typeInter = graphs[a]->typeInter;
    // 是否跨网卡
    allGather3Data[rank].graphInfo[a].crossNic = graphs[a]->crossNic;
  }

  // ============================================================
  // CPU 架构和厂商信息
  // ============================================================
  //
  // 为什么要交换 CPU 信息？
  // 1. 检测混合 CPU 环境
  // 2. 性能调优（不同 CPU 可能有不同的优化策略）
  // 3. 故障诊断（帮助识别问题）
  //
  // 混合 CPU 的例子：
  // - x86_64 + ARM（不同架构）
  // - Intel + AMD（不同厂商）
  //
  // 影响：
  // - SIMD 指令集不同
  // - 内存访问模式不同
  // - 性能特性不同
  //
  // ============================================================

  // CPU 架构（x86_64, ARM64 等）
  allGather3Data[rank].cpuArch = comm->cpuArch;
  // CPU 厂商（Intel, AMD, ARM 等）
  allGather3Data[rank].cpuVendor = comm->cpuVendor;

  // ============================================================
  // 设置通道数量
  // ============================================================
  //
  // 为什么要取最小值？
  // - 确保所有 rank 使用相同的通道数
  // - 某些 rank 可能因为拓扑限制可用通道较少
  // - 取最小值保证所有 rank 都能工作
  //
  // 例如：
  // - Rank 0-3：可以有 16 个通道
  // - Rank 4-7：只能有 8 个通道（拓扑限制）
  // - 最终使用 8 个通道（min(16, 8)）
  //
  // ============================================================

  // 设置 channels（取最小值以确保所有 rank 一致）
  //由于ringGraph
  comm->nChannels = std::min(treeGraph->nChannels, ringGraph->nChannels);

  // ============================================================
  // 初始化 topoRanks 配置
  // ============================================================
  //
  // ncclTopoPreset 的作用：
  // 1. 设置 Ring、Tree、CollNet 等算法的 rank 排序
  // 2. 填充 topoRanks 结构（包含每个算法的 rank 映射）
  // 3. 准备后续的通道连接
  //
  // topoRanks 包含：
  // - ringRecv：Ring 算法的接收 rank 序列
  // - ringSend：Ring 算法的发送 rank 序列
  // - treeUp/Down：Tree 算法的父/子节点
  // - collNet：CollNet 算法的配置
  //
  // ============================================================

  // 初始化 topoRanks 配置
  NCCLCHECKGOTO(ncclTopoPreset(comm, graphs, &allGather3Data[rank].topoRanks), ret, fail);

  // ============================================================
  // 执行 AllGather 3 - 交换图信息
  // ============================================================
  //
  // bootstrapAllGather：
  // - 收集所有 rank 的 allGather3Data
  // - 分发给所有 rank
  // - 阻塞等待所有 rank 完成
  //
  // AllGather 完成后：
  // - 每个 rank 都有所有 rank 的图信息
  // - 可以进行后续的对齐和验证
  //
  // ============================================================

  // 获取通信组内所有 rank 的 allGatherInfo 信息
  NCCLCHECKGOTO(bootstrapAllGather(comm->bootstrap, allGather3Data, sizeof(*allGather3Data)), ret, fail);

  // ============================================================
  // 确定节点数量、首个 rank 等
  // ============================================================
  //
  // 这段代码的任务：
  // 1. 统计有多少个物理节点
  // 2. 记录每个节点的第一个 rank
  // 3. 建立 rank 到 node 的映射
  //
  // 为什么需要这些信息？
  // - 节点内的 rank 可以使用 SHM/P2P 通信
  // - 节点间需要使用网络通信
  // - 需要区分节点内和节点间的操作
  //
  // 数据结构：
  // - nodesFirstRank：每个节点的第一个 rank
  // - nodesTreePatterns：每个节点的 Tree 模式
  // - rankToNode：rank 到 node 的映射
  //
  // ============================================================

  // 分配 nodesFirstRank 数组
  // 大小：nranks（最多 nranks 个节点）
  NCCLCHECKGOTO(ncclCalloc(&nodesFirstRank, nranks), ret, fail);

  // 分配 nodesTreePatterns 数组
  // 大小：nranks（最多 nranks 个节点）
  NCCLCHECKGOTO(ncclCalloc(&nodesTreePatterns, nranks), ret, fail);

  // rank转换成node
  // 分配 rankToNode 数组
  NCCLCHECKGOTO(ncclCalloc(&comm->rankToNode, comm->nRanks), ret, fail);

  // ============================================================
  // 遍历所有 ranks，统计节点信息
  // ============================================================
  //
  // 算法说明：
  // 1. 对于每个 rank，找到它在 Ring 中的前驱（ringRecv[0]）
  // 2. 这个前驱是该节点在 Ring 中的"代表"
  // 3. 相同前驱的 rank 属于同一节点
  //
  // 为什么要用 ringRecv[0]？
  // - Ring 的设计保证了同一节点的 ranks 在环中是连续的
  // - ringRecv[0] 是环中的前驱
  // - 同一节点的 ranks 有相同的 ringRecv[0]
  //
  // 示例：
  // 节点 0: ranks [0, 2, 4, 6]，ringRecv[0] = 6
  // 节点 1: ranks [1, 3, 5, 7]，ringRecv[0] = 5
  //
  // 对于 rank 0：
  // - firstRank = ringRecv[0] = 6
  // - 找到 nodesFirstRank[0] = 6 的节点（新节点）
  // - comm->rankToNode[0] = 0
  //
  // 对于 rank 2：
  // - firstRank = ringRecv[0] = 6
  // - 找到 nodesFirstRank[0] = 6 的节点（已存在）
  // - comm->rankToNode[2] = 0
  //
  // ============================================================

  for (int r = 0; r < nranks; r++) {
    int node;
    // 获取该 rank 在环中的前驱（ringRecv[0]）
    // 这个前驱是该节点在 Ring 中的"代表"
    int firstRank = allGather3Data[r].topoRanks.ringRecv[0];

    // 找到该 rank 所属的节点
    // 线性搜索 nodesFirstRank 数组
    // 直到找到匹配的 firstRank 或到达 nNodes
    for (node = 0; node < comm->nNodes && nodesFirstRank[node] != firstRank; node++)
        ;  // 空循环体，只用于递增 node

    // ============================================================
    // 判断是否找到节点或需要创建新节点
    // ============================================================
    //
    // 如果 node == comm->nNodes：
    // - 没有找到匹配的节点
    // - 这是一个新节点
    // - 需要添加到 nodesFirstRank 数组
    //
    // ============================================================

    if (node == comm->nNodes) {
      // 发现新节点，增加节点计数
      comm->nNodes++;
      // 记录该节点的第一个 rank
      nodesFirstRank[node] = firstRank;
      // 记录每个节点的 tree 模式（可能因 SM 架构不同而不同）
      // 例如：V100 和 A100 混合，Tree 模式可能不同
      nodesTreePatterns[node] = allGather3Data[r].graphInfo[NCCL_ALGO_TREE].pattern;
    }

    // rank 映射到 node
    comm->rankToNode[r] = node;

    // 检测混合 CPU 架构，每个rank的cpu架构是否一致
    //cpu架构不一致
    if (comm->cpuArch != allGather3Data[r].cpuArch &&
        comm->cpuArch != NCCL_TOPO_CPU_ARCH_MIXED) {
      comm->cpuArch = NCCL_TOPO_CPU_ARCH_MIXED;
    }
    //cpu厂商不一致
    if (comm->cpuVendor != allGather3Data[r].cpuVendor &&
        comm->cpuVendor != NCCL_TOPO_CPU_VENDOR_MIXED) {
      comm->cpuVendor = NCCL_TOPO_CPU_VENDOR_MIXED;
    }
  }

  // 警告用户存在混合 CPU
  if (rank == 0) {
    if (comm->cpuArch == NCCL_TOPO_CPU_ARCH_MIXED) {
      INFO(NCCL_GRAPH, "CPUs with mixed architecture were detected.");
    }
    if (comm->cpuVendor == NCCL_TOPO_CPU_VENDOR_MIXED) {
      INFO(NCCL_GRAPH, "CPUs with mixed vendors were detected.");
    }
  }

  // ============================================================
  // 分配 nodeRanks 并计算 local rank
  // ============================================================
  //
  // 现在我们已经知道了节点数量（nNodes），可以：
  // 1. 分配 nodeRanks 数组（每个节点一个）
  // 2. 分配 rankToLocalRank 数组（每个 rank 一个）
  // 3. 计算每个节点的 local rank 数量
  //
  // nodeRanks 结构：
  // - localRanks：该节点的 local rank 数量
  // - localRankToRank：local rank 到全局 rank 的映射
  //
  // rankToLocalRank 数组：
  // - rankToLocalRank[global_rank] = local_rank
  // - 用于快速查找某个 rank 在其节点内的 local rank
  //
  // 示例（8 个 ranks，2 个节点）：
  //   节点 0: ranks [0, 2, 4, 6]，localRanks = 4
  //   节点 1: ranks [1, 3, 5, 7]，localRanks = 4
  //   rankToLocalRank = [0, 0, 1, 1, 2, 2, 3, 3]
  //   - rank 0 → local 0
  //   - rank 1 → local 0
  //   - rank 2 → local 1
  //   - ...
  //
  // ============================================================

  // 现在知道了 nNodes，分配 nodeRanks 并计算每个节点的 localRanks
  NCCLCHECKGOTO(ncclCalloc(&comm->nodeRanks, comm->nNodes), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&comm->rankToLocalRank, comm->nRanks), ret, fail);

  // ============================================================
  // 第一次遍历：统计每个节点的 local rank 数量
  // ============================================================
  //
  // 这段代码计算每个节点有多少个 local ranks
  // 使用两遍遍历是因为：
  // 1. 第一遍：统计每个节点的 rank 数量
  // 2. 第二遍：填充 localRankToRank 映射
  //
  // 为什么不用一遍遍历？
  // - 需要先知道 localRanks 才能分配 localRankToRank 数组
  // - localRankToRank 数组大小等于 localRanks
  //
  // ============================================================

  // 全局 rank 映射到 local rank
  for (int r = 0; r < comm->nRanks; r++) {
    // 获取该 rank 所属的节点
    // rank 位于那个 node
    int node = comm->rankToNode[r];

    // rank 映射到 local rank，从 0 开始编号
    // 这是该节点内的第几个 rank（计数从 0 开始）
    comm->rankToLocalRank[r] = comm->nodeRanks[node].localRanks;

    // 节点的 local rank 数量加 1
    // 加 1
    comm->nodeRanks[node].localRanks++;
  }

  // ============================================================
  // 计算最大和最小 local rank 数量
  // ============================================================
  //
  // maxLocalRanks：所有节点中，local rank 数最多的节点的 rank 数
  // - 用于分配缓冲区和资源
  // - 确保资源足够支持最大的节点
  //
  // minLocalRanks：所有节点中，local rank 数最少的节点的 rank 数
  // - 用于检测异构集群
  // - 影响某些算法的优化策略
  //
  // 例如：
  // - 节点 0: 4 个 GPU
  // - 节点 1: 2 个 GPU
  // - 节点 2: 4 个 GPU
  // - maxLocalRanks = 4
  // - minLocalRanks = 2
  //
  // 为什么要统计？
  // 1. 资源分配：maxLocalRanks 决定缓冲区大小
  // 2. 负载均衡：minLocalRanks 影响调度策略
  // 3. 性能优化：异构集群需要特殊处理
  //
  // ============================================================

  // 先设置为最大值（用于寻找最小值）
  // INT_MAX 是 int 类型的最大值（通常是 2^31 - 1）
  comm->minLocalRanks = INT_MAX;

  // 计算通信组内一个节点中最大的 rank 数
  // 每个机器上 rank 数量可能不相同（异构集群）
  for (int n = 0; n < comm->nNodes; n++) {
    // 分配 localRankToRank 数组
    // 大小：localRanks（该节点的 local rank 数量）
    NCCLCHECKGOTO(ncclCalloc(&comm->nodeRanks[n].localRankToRank, comm->nodeRanks[n].localRanks), ret, fail);

    // 更新最大 local rank 数量
    // std::max：比较并返回较大值
    comm->maxLocalRanks = std::max(comm->maxLocalRanks, comm->nodeRanks[n].localRanks);

    // 更新最小 local rank 数量
    // std::min：比较并返回较小值
    comm->minLocalRanks = std::min(comm->minLocalRanks, comm->nodeRanks[n].localRanks);

    // 重置 localRanks 为 0（为下一遍遍历做准备）
    // 注意：这里会被重置为 0，然后重新填充
    comm->nodeRanks[n].localRanks = 0;
  }

  // ============================================================
  // 第二次遍历：填充 localRankToRank 映射表
  // ============================================================
  //
  // localRankToRank 映射表：
  // - localRankToRank[local_rank] = global_rank
  // - 用于从 local rank 找到对应的全局 rank
  //
  // 示例：
  // 节点 0: ranks [0, 2, 4, 6]
  // - localRankToRank[0] = 0
  // - localRankToRank[1] = 2
  // - localRankToRank[2] = 4
  // - localRankToRank[3] = 6
  //
  // ============================================================

  // 填充 ranks 数组
  // 设置 localRankToRank 映射表
  for (int r = 0; r < comm->nRanks; r++) {
    // 获取该 rank 所属的节点
    int node = comm->rankToNode[r];
    // 将全局 rank 填入到节点的 localRankToRank 数组中
    // localRanks++：先填充，后递增
    comm->nodeRanks[node].localRankToRank[comm->nodeRanks[node].localRanks++] = r;
  }

  // ============================================================
  // 设置当前 rank 的节点和 local rank 信息
  // ============================================================
  //
  // 这些字段用于快速访问当前 rank 的拓扑信息
  // - 避免每次都通过数组查找
  // - 提高访问效率
  //
  // comm->node：当前 rank 所属的节点号
  // - 用于判断是否与某个 rank 在同一节点
  // - 用于节点内同步
  //
  // comm->localRankToRank：当前节点的 localRankToRank 映射表
  // - 直接指向该节点的映射数组
  // - 避免通过 node 索引查找
  //
  // comm->localRank：当前 rank 在节点内的 local rank 号
  // - 用于节点内通信和同步
  // - 例如：SHM 通信需要 local rank
  //
  // comm->localRanks：当前节点的 local rank 总数
  // - 用于节点内同步
  // - 用于分配节点内资源
  //
  // ============================================================

  // 自己的 node 号
  comm->node = comm->rankToNode[rank];

  // 指向当前节点的 localRankToRank 映射表
  // 这样可以直接访问，不需要通过 node 索引
  comm->localRankToRank = comm->nodeRanks[comm->node].localRankToRank;

  // 自己的 local rank 号
  comm->localRank = comm->rankToLocalRank[rank];

  // 当前 node 上总的 local rank 数量
  comm->localRanks = comm->nodeRanks[comm->node].localRanks;

  // 初始化 NVLS 信息
  NCCLCHECKGOTO(initNvlDomainInfo(comm), ret, fail);

  TRACE(NCCL_INIT, "hostHash[%d] %lx localRank %d localRanks %d localRank0 %d",
        rank, comm->peerInfo[rank].hostHash, comm->localRank, comm->localRanks, comm->localRankToRank[0]);

  // 异常检查
  if (comm->localRank == -1 || comm->localRankToRank[0] == -1 || comm->localRanks == 0) {
    WARN("Failed to determine local ranks rank %d hostHash %lx pidHash %lx localRank %d localRanks %d localRank0 %d",
         rank, comm->peerInfo[rank].hostHash, comm->peerInfo[rank].pidHash,
         comm->localRank, comm->localRanks, comm->localRankToRank[0]);
    ret = ncclInternalError;
    goto fail;
  }

  INFO(NCCL_INIT, "comm %p rank %d nRanks %d nNodes %d localRanks %d localRank %d MNNVL %d",
       comm, rank, comm->nRanks, comm->nNodes, comm->localRanks, comm->localRank, comm->MNNVL);

  nChannelsOrig = comm->nChannels;

  // 分配所有 topoRanks 数组
  NCCLCHECKGOTO(ncclCalloc(&allTopoRanks, comm->nRanks), ret, fail);

  // 对齐所有 rank，确保调优一致
  for (int i = 0; i < nranks; i++) {
    allTopoRanks[i] = &allGather3Data[i].topoRanks;
    for (int a = 0; a < NCCL_NUM_ALGORITHMS; a++) {
      // 取最小值（确保所有 rank 都能支持）
      graphs[a]->nChannels = std::min(allGather3Data[i].graphInfo[a].nChannels, graphs[a]->nChannels);
      graphs[a]->sameChannels = std::min(allGather3Data[i].graphInfo[a].sameChannels, graphs[a]->sameChannels);
      graphs[a]->bwIntra = std::min(allGather3Data[i].graphInfo[a].bwIntra, graphs[a]->bwIntra);
      graphs[a]->bwInter = std::min(allGather3Data[i].graphInfo[a].bwInter, graphs[a]->bwInter);

      // 取最大值（确保性能最优）
      graphs[a]->typeIntra = std::max(allGather3Data[i].graphInfo[a].typeIntra, graphs[a]->typeIntra);
      graphs[a]->typeInter = std::max(allGather3Data[i].graphInfo[a].typeInter, graphs[a]->typeInter);
      graphs[a]->crossNic = std::max(allGather3Data[i].graphInfo[a].crossNic, graphs[a]->crossNic);
    }
    comm->maxTreePattern = std::max(comm->maxTreePattern, allGather3Data[i].graphInfo[NCCL_ALGO_TREE].pattern);
  }

  // 检查是否支持 CollNet 和 NVLS
  if (graphs[NCCL_ALGO_COLLNET_CHAIN]->nChannels == 0)
    comm->config.collnetEnable = 0;
  if (graphs[NCCL_ALGO_NVLS]->nChannels == 0)
    comm->nvlsSupport = comm->nvlsChannels = 0;

  //ringGraph最大channels是32，所以这里最大也是32
  // 再一次更新 channels
  comm->nChannels = treeGraph->nChannels = ringGraph->nChannels = std::min(treeGraph->nChannels, ringGraph->nChannels);
  if (comm->nChannels < nChannelsOrig) {
    // 我们在 Preset() 中已经开始复制通道，所以需要移动复制的通道
    for (int i = 0; i < comm->nChannels; i++)
        memcpy(comm->channels + comm->nChannels + i, comm->channels + nChannelsOrig + i, sizeof(struct ncclChannel));
  }

  // 在 AllGather 后确定 CollNet 支持
  if (comm->config.collnetEnable == 1) {
    int collNetNodeThreshold = ncclParamCollNetNodeThreshold();
    if (comm->nNodes < collNetNodeThreshold) {
      INFO(NCCL_INIT, "Communicator has %d nodes which is less than CollNet node threshold %d, disabling CollNet", comm->nNodes, collNetNodeThreshold);
      comm->config.collnetEnable = 0;
    }
  }

  NCCLCHECK(ncclTopoPathAllNVLink(comm->topo, &comm->isAllNvlink));
  comm->isOneRPN = (comm->maxLocalRanks == 1);

  // 分配 ring 数组（每个 rank 有 MAXCHANNELS 个 channel）
  NCCLCHECKGOTO(ncclCalloc(&rings, nranks * MAXCHANNELS), ret, fail);

  // ============================================================
  // ncclTopoPostset - 拓扑后处理
  // ============================================================
  //
  // 这个函数是拓扑初始化的关键步骤，它完成以下工作：
  //
  // 1. 通道复制和扩展：
  //    - 如果某个算法需要的通道数比 Ring/Tree 多，需要复制通道
  //    - 例如：CollNet 可能需要 32 个通道，而 Ring 只有 16 个
  //    - 这时会复制 Ring 的通道配置给 CollNet 使用
  //
  // 2. Ring 排序计算：
  //    - 计算每个通道的环形拓扑的 rank 顺序
  //    - rings 数组存储了每个通道的 rank 排序
  //    - 这个排序决定了环形算法的通信模式
  //
  // 3. Tree 配置：
  //    - 计算每个通道的树形拓扑的父节点和子节点
  //    - Tree 使用二元树结构，每个节点最多有 3 个子节点和 1 个父节点
  //
  // ============================================================

  // 设置相关 ring 参数等
  NCCLCHECKGOTO(ncclTopoPostset(comm, nodesFirstRank, nodesTreePatterns, allTopoRanks, rings, graphs, parent), ret, fail);

  // AllGather 3 结束
  // 注意：这里使用 += 是因为我们在 AllGather 2 结束时将开始时间存储在了 CONNECT 计时器中
  timers[TIMER_INIT_ALLGATHER] += clockNano() - timers[TIMER_INIT_CONNECT];

  TRACE(NCCL_INIT, "rank %d nranks %d - BUILT %d TREES/RINGS", rank, nranks, comm->nChannels);

  // ============================================================
  // 打印 Tree 和 Ring 信息（用于调试）
  // ============================================================
  //
  // Ring 格式：Ring XX : prev -> rank -> next
  // - prev：环形拓扑中的前驱 rank（从哪个 rank 接收数据）
  // - rank：当前 rank
  // - next：环形拓扑中的后继 rank（向哪个 rank 发送数据）
  //
  // Tree 格式：[channel] down0/down1/down2->rank->up
  // - down0/down1/down2：三个子节点 rank（可能为 -1 表示无子节点）
  // - rank：当前 rank
  // - up：父节点 rank（可能为 -1 表示根节点）
  //
  // 示例：
  // Ring 00 : 7 -> 0 -> 1    表示 rank 0 在环中，从 rank 7 接收，向 rank 1 发送
  // [0] 1/2/3->0->-1        表示 rank 0 是树的根节点，有 3 个子节点（1, 2, 3）
  //
  // ============================================================

  // 打印 Tree 和 Ring 信息
  char line[1024];
  line[0] = '\0';
  for (int c = 0; c < comm->nChannels; c++) {
    struct ncclTree* tree = &comm->channels[c].tree;
    snprintf(line + strlen(line), 1023 - strlen(line), " [%d] %d/%d/%d->%d->%d",
        c, tree->down[0], tree->down[1], tree->down[2], rank, tree->up);
    INFO(NCCL_GRAPH, "Ring %02d : %d -> %d -> %d", c, comm->channels[c].ring.prev, comm->rank, comm->channels[c].ring.next);
  }
  line[1023] = '\0';
  INFO(NCCL_INIT, "Trees%s", line);

  // 计算缓冲区大小
  NCCLCHECKGOTO(computeBuffSizes(comm), ret, fail);

  // 计算 P2P 的每个 peer 的通道数
  NCCLCHECKGOTO(ncclTopoComputeP2pChannels(comm), ret, fail);

  /* 到现在为止，comm 的所有信息都应该已知了。我们可以初始化共享资源并映射 localRanks 到 top parent local ranks。
   * 注意：这个 shareRes 初始化必须放在所有 proxy 操作之前。 */

  // ============================================================
  // 共享资源初始化
  // ============================================================
  //
  // sharedRes 是多个 comm 之间共享的资源池
  //
  // 什么时候共享资源？
  // - 同一节点内的多个 comm 可能需要共享某些资源
  // - 例如：同一进程内的多个 GPU 创建的 comm
  //
  // 共享的资源包括：
  // - Proxy 线程和状态
  // - 设备流（deviceStream）
  // - 内存池
  //
  // owner 标志：
  // - comm->sharedRes->owner == comm 表示这个 comm 是资源的创建者
  // - 只有 owner 负责初始化和清理共享资源
  // - 其他 comm 只是引用这些资源
  //
  // 为什么需要共享资源？
  // 1. 减少资源开销（不需要每个 comm 都创建 proxy 线程）
  // 2. 统一管理（便于协调和同步）
  // 3. 内存效率（共享内存池减少内存占用）
  //
  // ============================================================

  // 如果是 comm 自己创建了 sharedRes
  if (comm->sharedRes->owner == comm) {
    // 保存拓扑信息到共享资源
    comm->sharedRes->tpNLocalRanks = comm->localRanks;        // 节点内 rank 数量
    comm->sharedRes->magic = comm->magic;                     // 魔数（用于验证）
    comm->sharedRes->tpNChannels = comm->nChannels;            // 通道数量
    comm->sharedRes->tpP2pNChannels = comm->p2pnChannels;      // P2P 通道数量
    // 拷贝 rank 到 local rank 的映射表
    memcpy(comm->sharedRes->tpRankToLocalRank, comm->rankToLocalRank, sizeof(int) * comm->nRanks);
  }

  // ============================================================
  // 设置 topParentLocalRanks
  // ============================================================
  //
  // topParentRanks 的概念：
  // - 在通信域分割（ncclCommSplit）时，会创建父子 comm
  // - topParentRanks 记录了每个 rank 在最顶层父 comm 中的 rank 号
  //
  // topParentLocalRanks 的作用：
  // - 将 topParentRanks 转换为 top parent comm 中的 local rank
  // - 用于在分割的 comm 中引用父 comm 的资源
  //
  // 为什么需要这个映射？
  // 1. 子 comm 需要知道父 comm 中的 local rank 来共享资源
  // 2. 某些操作需要在父 comm 的上下文中执行
  // 3. 用于调试和追踪 comm 的层次结构
  //
  // 示例：
  // - 父 comm 有 4 个 rank：[0, 1, 2, 3]
  // - 分割后，子 comm 包含 rank [0, 2]
  // - 对于子 comm 的 rank 0（父 comm 的 rank 0）：
  //   - topParentRanks[0] = 0
  //   - topParentLocalRanks[0] = 0 (父 comm 中的 local rank)
  // - 对于子 comm 的 rank 1（父 comm 的 rank 2）：
  //   - topParentRanks[1] = 2
  //   - topParentLocalRanks[1] = 1 (父 comm 中的 local rank)
  //
  // ============================================================

  // 分配 topParentLocalRanks 数组
  NCCLCHECKGOTO(ncclCalloc(&topParentLocalRanks, comm->localRanks), ret, fail);
  for (int i = 0; i < comm->localRanks; ++i) {
    // 获取当前节点第 i 个 local rank 的全局 rank
    int globalRank = comm->localRankToRank[i];
    // 获取这个全局 rank 在顶层父 comm 中的 rank 号
    int tpRank = comm->topParentRanks[globalRank];
    // 将顶层父 comm 的 rank 号转换为 local rank
    topParentLocalRanks[i] = comm->sharedRes->tpRankToLocalRank[tpRank];
  }
  comm->topParentLocalRanks = topParentLocalRanks;

  // Profiler 插件上下文必须在 proxy 线程之前初始化
  NCCLCHECK(ncclProfilerPluginInit(comm));

  // 检查当前节点的所有 rank GPU 是否支持 P2P
  NCCLCHECKGOTO(ncclTransportCheckP2pType(comm, &comm->isAllDirectP2p, &comm->directMode), ret, fail);

  // 启动 Proxy 服务线程，之后可以使用 proxy 调用
  if (parent && parent->shareResources) {
    comm->proxyState = parent->sharedRes->proxyState;
    ncclAtomicRefCountIncrement(&parent->sharedRes->proxyState->refCount);
  } else {
    // 创建 proxy 线程（线程执行 ncclProxyService 函数）
    //创建uds线程，线程执行函数ncclProxyServiceUDS
    NCCLCHECKGOTO(ncclProxyCreate(comm), ret, fail);
  }

  NCCLCHECKGOTO(ncclCalloc(&comm->gproxyConn, comm->nRanks), ret, fail);

  timers[TIMER_INIT_CONNECT] = clockNano();

  do {
    // ============================================================================
    // 构建 P2P（点对点）调度表
    // ============================================================================
    //
    // P2P 调度决定了每个 rank 在每一轮通信中应该与哪个 peer 通信
    // 这个调度算法使用二次公式 (x*x+x)/2 mod N 来生成通信模式
    //
    // 为什么使用这个特定的公式？
    // 1. 它在 N 是 2 的幂时能生成完美的环形排列
    // 2. 生成的序列具有良好的通信局部性
    // 3. 避免了所有 rank 同时向同一个 peer 发送数据（网络热点问题）
    //
    // 示例：16 个 rank 的序列：0, 1, 3, 6, 10, 15, 5, 12, 4, 13, 7, 2, 14, 11, 9, 8
    // 这个序列确保了每一轮的通信都是均衡的
    //
    // 对于异构集群（节点间 GPU 数量不同），算法会分两层级：
    // - 节点间（inter-node）：使用上述公式枚举目标节点
    // - 节点内（intra-node）：在同一节点内也使用相同的公式枚举目标 rank

    int node = comm->node;           // 当前节点号（0 到 nNodes-1）
    int nNodes = comm->nNodes;        // 总节点数量（物理服务器数）
    int nRanks = comm->nRanks;        // 总 rank 数量（所有 GPU 的总数）
    int local = comm->localRank;     // 本地 rank 号（在当前节点内的编号，0 到 localRanks-1）
    int nLocals = comm->maxLocalRanks;  // 所有 node 中最大的 local rank 数量
    struct ncclNodeRanks* nodeRanks = comm->nodeRanks;  // 节点 rank 信息数组
    bool flat = false;                 // 是否使用扁平化模式（处理异构集群）

    // 检查所有 node 上的 local rank 数量是否相同
    // 如果不同（如节点 0 有 4 GPU，节点 1 有 2 GPU），需要特殊处理
    for (int node = 0; node < nNodes; node++) {
      if (nodeRanks[node].localRanks != nLocals) {
        // 节点间 rank 数量不相同（如节点 0 有 4 GPU，节点 1 有 2 GPU）
        // 这种情况下无法使用分层的节点间/节点内调度，退化为扁平化处理
        // 扁平化意味着忽略节点边界，将所有 rank 视为一个大集合
        flat = true;
        nNodes = 1;              // 虚拟为单节点
        node = 0;                // 虚拟节点号为 0
        nLocals = nRanks;        // 节点内 rank 数等于总 rank 数
        local = rank;           // 本地 rank 等于全局 rank
        break;
      }
    }

    // 计算 2 的幂次向上取整
    // 这是因为二次公式 (x*x+x)/2 mod N 只在 N 是 2 的幂时才有完美的循环性质
    // 例如：nNodes=5 → nNodesPow2=8，nLocals=3 → nLocalsPow2=4
    int nNodesPow2 = pow2Up(nNodes);
    int nLocalsPow2 = pow2Up(nLocals);

    // 分配 P2P 调度表内存（nRanks 个条目，每个条目包含 sendRank 和 recvRank）
    comm->p2pSchedule = ncclMemoryStackAlloc<ncclComm::P2pSchedulePair>(&comm->memPermanent, nRanks);

    // 分配内核规划器的 peer 信息（nRanks 个 peer）
    comm->planner.peers = ncclMemoryStackAlloc<ncclKernelPlanner::Peer>(&comm->memPermanent, nRanks);

    // 节点层的循环变量（用于枚举目标节点）
    uint32_t nodeRound = 0;    // 节点层循环计数器
    uint32_t nodeDelta = 0;     // 节点层 delta 值

    // 总的通信轮数（应该等于 nRanks，每轮每个 rank 与一个不同的 peer 通信）
    int round = 0;

    // ============================================================
    // 二次公式 P2P 调度算法的核心循环
    // ============================================================
    //
    // 算法原理：
    // 使用公式 delta = (delta + round) & (N-1) 来枚举 delta 值
    // 其中 N 是向上取整到 2 的幂，& 是位与操作（相当于 mod 2^k）
    //
    // 这个公式生成的 delta 序列具有以下性质：
    // 1. 生成 0 到 N-1 的所有值（仅当 N 是 2 的幂时）
    // 2. 序列具有良好的分散性，避免热点
    // 3. 可以按需生成任意长度的序列
    //
    // 两层枚举：
    // - 外层：枚举目标节点（inter-node）
    // - 内层：枚举节点内的目标 rank（intra-node）
    //
    // 对于每个 rank，我们计算：
    // - sendRank = f(当前rank, delta) - 本轮应该向谁发送数据
    // - recvRank = f(当前rank, delta) - 本轮应该从谁接收数据
    //
    // ============================================================

    do {
      // 检查节点 delta 是否有效（小于实际节点数）
      // 由于我们使用 pow2Up(nNodes)，可能生成一些无效的大值
      if (nodeDelta < nNodes) { // 过滤无效的节点 delta
        // 计算目标节点（发送方）
        // (node + nodeDelta) % nNodes 确保在节点范围内循环
        int sendNode = (node + nodeDelta) % nNodes;

        // 计算源节点（接收方）
        // (node - nodeDelta + nNodes) % nNodes 确保在节点范围内循环
        // 注意：这里使用减法实现反向遍历
        int recvNode = (node - nodeDelta + nNodes) % nNodes;

        // 节点内层循环变量
        uint32_t localRound = 0;    // 节点内层循环计数器
        uint32_t localDelta = 0;     // 节点内层 delta 值

        do {
          // 检查节点内 delta 是否有效（小于实际本地 rank 数）
          if (localDelta < nLocals) { // 过滤无效的节点内 delta
            // 计算节点内的本地 rank 偏移
            int sendLocal = (local + localDelta) % nLocals;
            int recvLocal = (local - localDelta + nLocals) % nLocals;

            // 根据是否扁平化，设置最终的目标 rank
            if (flat) {
              // 扁平化模式：直接使用本地 rank 作为全局 rank
              comm->p2pSchedule[round].sendRank = sendLocal;
              comm->p2pSchedule[round].recvRank = recvLocal;
            } else {
              // 分层模式：需要先找到目标节点，然后在目标节点内找到目标 rank
              // 发送：当前 rank → 目标节点的 sendLocal rank
              comm->p2pSchedule[round].sendRank = nodeRanks[sendNode].localRankToRank[sendLocal];
              // 接收：目标节点的 recvLocal rank → 当前 rank
              comm->p2Schedule[round].recvRank = nodeRanks[recvNode].localRankToRank[recvLocal];
            }

            // 递增总轮数
            round += 1;
          }

          // 节点内层循环计数器递增
          localRound += 1;

          // 二次公式更新 localDelta
          // (localDelta + localRound) & (nLocalsPow2 - 1)
          // 注意：& 操作符优先级较低，需要括号确保正确的计算顺序
          localDelta = (localDelta + localRound) & (nLocalsPow2 - 1);  // 二次更新

        } while (localRound != nLocalsPow2);  // 节点内层循环：枚举完所有本地 rank 的 delta
      }

      // 节点层循环计数器递增
      nodeRound += 1;

      // 二次公式更新 nodeDelta
      // (nodeDelta + nodeRound) & (nNodesPow2 - 1)
      nodeDelta = (nodeDelta + nodeRound) & (nNodesPow2 - 1);  // 二次更新

    } while (nodeRound != nNodesPow2);  // 节点层循环：枚举完所有节点的 delta

    // 验证：最终生成的轮数应该等于 rank 数量
    // 如果不等于，说明算法有 bug 或实现有问题
    if (round != nRanks) {
      WARN("P2p schedule creation has bugs.");
      ret = ncclInternalError;
      goto fail;
    }
  } while (0);

  // ============================================================
  // 连接模式选择：Runtime Connection vs. Direct Connection
  // ============================================================
  //
  // NCCL 支持两种连接建立模式：
  //
  // 1. Runtime Connection（运行时连接，延迟连接）：
  //    - 仅当 cuMem 支持可用时才启用
  //    - 在初始化阶段只设置通道，不建立实际的通信连接
  //    - 连接延迟到第一次集体操作时才建立
  //    - 优点：减少初始化时间，避免不必要的连接建立
  //    - 缺点：第一次集体操作可能有额外延迟
  //
  // 2. Direct Connection（直接连接，预连接）：
  //    - 在初始化阶段建立所有通信连接
  //    - 确保所有传输通道已就绪
  //    - 优点：第一次集体操作无额外延迟
  //    - 缺点：初始化时间较长
  //
  // ============================================================

  // ncclParamRuntimeConnect 默认为 1
  // cuMemSupport 是 CUDA 12.2+ 的 CUMEM API 支持标志
  comm->runtimeConn = comm->cuMemSupport && ncclParamRuntimeConnect();

  if (comm->runtimeConn) {
    // ============================================================
    // 运行时连接模式（延迟连接）
    // ============================================================
    //
    // 在这种模式下，我们只设置通道的基本配置，不建立实际连接
    // 连接将在第一次使用该通道的集体操作时建立
    //
    // setupChannel 做什么：
    // - 设置通道的 ring 配置（前驱和后继 rank）
    // - 设置通道的 tree 配置（父节点和子节点）
    // - 设置通道的 collnet 配置（如果启用）
    // - 分配通道的内存结构
    //
    // ============================================================

    for (int c = 0; c < comm->nChannels; c++) {
      NCCLCHECKGOTO(setupChannel(comm, c, rank, nranks, rings + c * nranks), ret, fail);
    }
    // 尝试设置 NVLS，可能静默失败并禁用 NVLS
    // NVLS（NVLink Sharp）需要特殊的硬件支持（NVLink Switch）
    NCCLCHECKGOTO(ncclNvlsSetup(comm, parent), ret, fail);
    // 检查是否可以设置 CollNet
    if (comm->config.collnetEnable)
        ncclCollNetSetup(comm, parent, graphs);
  } else {
    // ============================================================
    // 直接连接模式（预连接）
    // ============================================================
    //
    // 在这种模式下，我们在初始化阶段就建立所有通信连接
    // 这包括：
    // 1. Ring 连接：环形拓扑的每个 rank 连接到前驱和后继
    // 2. Tree 连接：树形拓扑的每个 rank 连接到父节点和子节点
    // 3. PAT 连接：单 GPU 节点的 PAT（Peer-Aware Transport）连接
    // 4. NVLS 连接：NVLink Sharp 连接（如果硬件支持）
    // 5. CollNet 连接：集合网络连接（如果启用）
    // 6. Proxy 连接：网络代理连接（用于跨节点通信）
    //
    // ============================================================

    // 直接先建立好连接
    for (int c = 0; c < comm->nChannels; c++) {
      NCCLCHECKGOTO(setupChannel(comm, c, rank, nranks, rings + c * nranks), ret, fail);
    }

    // 连接 Ring
    // ncclTransportRingConnect 为每个通道建立环形连接
    // 每个 rank 连接到 ringRecv[0]（接收数据的来源）和 ringSend[0]（发送数据的目标）
    NCCLCHECKGOTO(ncclTransportRingConnect(comm), ret, fail);

    // 连接 Tree
    // ncclTransportTreeConnect 为每个通道建立树形连接
    // 每个 rank 连接到它的父节点（treeUp）和子节点（treeDown）
    NCCLCHECKGOTO(ncclTransportTreeConnect(comm), ret, fail);

    // 仅对每个节点只有 1 个 GPU 的通信器连接 PAT
    // PAT（Peer-Aware Transport）是一种优化的传输方式
    // 当每个节点只有 1 个 GPU 时，可以使用 PAT 来减少跳数
    if (comm->maxLocalRanks == 1)
        NCCLCHECKGOTO(ncclTransportPatConnect(comm), ret, fail);

    // 尝试设置 NVLS，可能静默失败并禁用 NVLS
    // NVLS（NVLink Sharp）需要特殊的硬件支持（NVLink Switch）
    // 如果硬件不支持，这个调用会静默失败并禁用 NVLS
    NCCLCHECKGOTO(ncclNvlsSetup(comm, parent), ret, fail);
    NCCLCHECKGOTO(ncclNvlsBufferSetup(comm), ret, fail);

    // 如果需要，连接 NVLS Tree
    // NVLS Tree 是 NVLink Sharp 的树形拓扑
    NCCLCHECKGOTO(ncclNvlsTreeConnect(comm), ret, fail);

    // 检查是否可以设置 CollNet
    if (comm->config.collnetEnable) {
      // CollNet（Collective Network）是另一种集合通信拓扑
      // 它使用特殊的网络硬件（如 BlueField、NVSwitch）来优化集体操作
      ncclCollNetSetup(comm, parent, graphs);
      NCCLCHECKGOTO(ncclCollNetChainBufferSetup(comm), ret, fail);
      // CollNet Direct 模式有节点数量限制（最多 NCCL_MAX_DIRECT_ARITY + 1 个 rank）
      if (comm->maxLocalRanks <= NCCL_MAX_DIRECT_ARITY + 1) {
        NCCLCHECKGOTO(ncclCollNetDirectBufferSetup(comm), ret, fail);
      }
    }

    // ============================================================
    // Proxy 连接设置
    // ============================================================
    //
    // Proxy 是 NCCL 的网络代理服务，用于处理跨节点通信
    // 每个 rank 都有一个本地的 proxy 服务
    //
    // Proxy 的作用：
    // 1. 处理网络连接的建立和维护
    // 2. 处理网络数据的发送和接收
    // 3. 提供异步网络操作
    // 4. 处理网络错误和重连
    //
    // 连接到本地 net proxy：
    // - TRANSPORT_NET：使用网络传输
    // - 1：表示连接到本地 proxy（rank == comm->rank）
    // - proxyConn：返回的 proxy 连接句柄
    //
    // ncclProxyCallBlocking 是一个同步的 RPC 调用
    // - ncclProxyMsgSharedInit：初始化共享的 P2P 通道数量
    // - &comm->p2pnChannels：输入参数（本地的 P2P 通道数）
    // - NULL, 0：不需要返回值
    //
    // ============================================================

    // 连接到本地 net proxy
    NCCLCHECKGOTO(ncclProxyConnect(comm, TRANSPORT_NET, 1, comm->rank, &proxyConn), ret, fail);
    NCCLCHECKGOTO(ncclProxyCallBlocking(comm, &proxyConn, ncclProxyMsgSharedInit, &comm->p2pnChannels, sizeof(int), NULL, 0), ret, fail);

    // ============================================================
    // PXN（Proxy Cross Node）设置
    // ============================================================
    //
    // PXN 是 NCCL 的一种特殊通信模式，用于优化跨节点通信
    //
    // 传统跨节点通信的问题：
    // - Rank A → 网卡 → 网络 → 网卡 → Rank B
    // - 需要经过网卡，增加了延迟和带宽瓶颈
    //
    // PXN 的解决方案：
    // - Rank A → NVLink → 中间 GPU → NVLink → Rank B
    // - 完全通过 NVLink 传输，避免网卡瓶颈
    // - 只在必要时才使用网卡（如目标节点没有 NVLink）
    //
    // PXN 的工作原理：
    // 1. 找到可以作为中转的 GPU ranks（pxnPeers）
    // 2. 连接到这些 rank 的 proxy
    // 3. 通过 NVLink 将数据发送到中间 GPU
    // 4. 中间 GPU 再通过 NVLink 转发到目标 rank
    //
    // ============================================================

    // 如果使用 PXN，连接到远程 proxy
    // PXN 禁止借助非本地网卡进行跨节点通信，转而通过 NVLink 经由中间 GPU 中转数据
    if (ncclPxnDisable(comm) == 0) {
      int nranks;
      // 获取可以作为 PXN 中转的 rank 列表
      // pxnPeers 是这些 rank 的数组
      // nranks 是 pxnPeers 数组的大小
      NCCLCHECKGOTO(ncclTopoGetPxnRanks(comm, &pxnPeers, &nranks), ret, fail);
      for (int r = 0; r < nranks; r++) {
        // 连接到远程 rank 的 proxy
        // TRANSPORT_NET：使用网络传输（但实际数据通过 NVLink）
        // 1：表示连接类型（1 = P2P 连接）
        // pxnPeers[r]：目标 rank
        NCCLCHECKGOTO(ncclProxyConnect(comm, TRANSPORT_NET, 1, pxnPeers[r], &proxyConn), ret, fail);
        // 通知远程 proxy 初始化 P2P 通道
        NCCLCHECKGOTO(ncclProxyCallBlocking(comm, &proxyConn, ncclProxyMsgSharedInit, &comm->p2pnChannels, sizeof(int), NULL, 0), ret, fail);
      }
    }

    // ============================================================
    // NVB（NVBridge）预连接设置
    // ============================================================
    //
    // NVB 是 NVIDIA 的桥接技术，用于连接多个 GPU 或多个节点
    //
    // NVB 预连接的作用：
    // - 在实际需要之前预先建立 P2P 连接
    // - 减少第一次集体操作的延迟
    // - 特别适用于通过 NVB 路径连接的 GPU
    //
    // 预连接的工作流程：
    // 1. 找到通过 NVB 路径可达的所有 GPU（nvbPeers）
    // 2. 对于每个 peer，找到对应的 P2P 调度轮次（sendRound/recvRound）
    // 3. 计算该轮次使用的通道范围（sendBase/recvBase）
    // 4. 标记需要建立连接的通道（connectSend/connectRecv）
    // 5. 调用 ncclTransportP2pSetup 实际建立连接
    //
    // ============================================================

    // 默认为 1
    if (ncclParamNvbPreconnect()) {
      // 使用 NVB 路径时连接 P2P
      int nvbNpeers;
      // 获取通过 NVB 路径可达的 GPU 列表
      // nvbNpeers：NVB peer 的数量
      // nvbPeers：NVB peer 的 rank 数组
      NCCLCHECKGOTO(ncclTopoGetNvbGpus(comm->topo, comm->rank, &nvbNpeers, &nvbPeers), ret, fail);
      for (int r = 0; r < nvbNpeers; r++) {
        int peer = nvbPeers[r];
        // 找到这个 peer 对应的发送轮次和接收轮次
        // p2pSchedule 数组存储了每轮的发送和接收 rank
        int sendRound = 0, recvRound = 0;
        while (comm->p2pSchedule[sendRound].sendRank != peer)
            sendRound++;
        while (comm->p2pSchedule[recvRound].recvRank != peer)
            recvRound++;
        // 计算该轮次使用的通道基地址
        // ncclP2pChannelBaseForRound 返回该轮次使用的第一个通道号
        uint8_t sendBase = ncclP2pChannelBaseForRound(comm, sendRound);
        uint8_t recvBase = ncclP2pChannelBaseForRound(comm, recvRound);
        // 对于该 peer 的每个 P2P 通道，检查是否需要建立连接
        for (int c = 0; c < comm->p2pnChannelsPerPeer; c++) {
          int channelId;
          // 计算发送通道 ID
          channelId = ncclP2pChannelForPart(comm->p2pnChannels, sendBase, c);
          // 如果该通道的发送连接尚未建立，标记需要建立
          // send[1] 是第二个发送连接（NCCL 使用两个发送连接以提高吞吐）
          if (comm->channels[channelId].peers[peer]->send[1].connected == 0) {
            comm->connectSend[peer] |= (1UL << channelId);  // 设置位标志
          }
          // 计算接收通道 ID
          channelId = ncclP2pChannelForPart(comm->p2pnChannels, recvBase, c);
          // 如果该通道的接收连接尚未建立，标记需要建立
          if (comm->channels[channelId].peers[peer]->recv[1].connected == 0) {
            comm->connectRecv[peer] |= (1UL << channelId);  // 设置位标志
          }
        }
      }

      // 实际建立 P2P 连接
      // connectSend/connectRecv 位图指定了需要建立哪些连接
      // NULL：表示不使用回调函数
      // 1：表示预连接模式
      NCCLCHECKGOTO(ncclTransportP2pSetup(comm, NULL, 1), ret, fail);
    }
  }

  TRACE(NCCL_INIT, "rank %d nranks %d - CONNECTED %d RINGS AND TREES", rank, nranks, comm->nChannels);

  // 计算算法和协议组合的时间模型
  NCCLCHECKGOTO(ncclTopoInitTunerConstants(comm), ret, fail);

  // 加载性能调优插件
  NCCLCHECKGOTO(ncclTunerPluginLoad(comm), ret, fail);

  // 调用插件初始化函数
  if (comm->tuner) {
    NCCLCHECK(comm->tuner->init(&comm->tunerContext, comm->commHash, comm->nRanks, comm->nNodes, ncclDebugLog, &comm->nvlDomainInfo, &comm->tunerConstants));
  }

  // 调优模型
  NCCLCHECKGOTO(ncclTopoTuneModel(comm, comm->minCompCap, comm->maxCompCap, graphs), ret, fail);

  INFO(NCCL_INIT, "%d coll channels, %d collnet channels, %d nvls channels, %d p2p channels, %d p2p channels per peer", comm->nChannels, comm->nChannels, comm->nvlsChannels, comm->p2pnChannels, comm->p2pnChannelsPerPeer);

  // ============================================================
  // NCCL_LAUNCH_MODE - 内核启动模式
  // ============================================================
  //
  // NCCL 支持两种内核启动模式：
  //
  // 1. PARALLEL 模式（默认）：
  //    - 所有通道的内核并行启动
  //    - 优点：最大化并行度，延迟最低
  //    - 缺点：GPU 资源占用较多
  //
  // 2. GROUP 模式：
  //    - 通道的内核按组启动
  //    - 优点：GPU 资源占用较少
  //    - 缺点：延迟稍高
  //
  // 什么时候使用 GROUP 模式？
  // - GPU 内存有限
  // - 通道数量很多
  // - 需要减少 GPU 资源占用
  //
  // 环境变量设置：
  // - export NCCL_LAUNCH_MODE=PARALLEL
  // - export NCCL_LAUNCH_MODE=GROUP
  //
  // 为什么只用 intraRank == 0 设置？
  // - 同一节点内的所有 ranks 应该使用相同的模式
  // - 只需要一个 rank 读取环境变量并设置
  // - 使用原子交换确保只有一个 rank 生效
  //
  // ============================================================

  // 加载 NCCL_LAUNCH_MODE
  // 只在节点的第一个 local rank（intraRank == 0）执行
  if (comm->intraRank == 0) {
    const char* str = ncclGetEnv("NCCL_LAUNCH_MODE");
    enum ncclLaunchMode mode, modeOld;
    if (str && strcasecmp(str, "GROUP") == 0) {
      mode = ncclLaunchModeGroup;
    } else {
      mode = ncclLaunchModeParallel;  // 默认并行模式
    }
    // 理论上可能与连接到多个 ncclUniqueId 的其他通信器竞争
    // 使用原子交换确保只有一个 comm 的设置生效
    // __ATOMIC_RELAXED：不需要同步，只需要原子性
    modeOld = __atomic_exchange_n(&ncclParamLaunchMode, mode, __ATOMIC_RELAXED);
    // 只有在第一次设置且环境变量非空时才打印信息
    if (modeOld == ncclLaunchModeInvalid && str && str[0] != '\0') {
      INFO(NCCL_ENV, "NCCL_LAUNCH_MODE set by environment to %s", mode == ncclLaunchModeParallel ? "PARALLEL" : "GROUP");
    }
  }

  // ============================================================
  // 对称支持检测
  // ============================================================
  //
  // 对称（Symmetric）通信是 NCCL 的一种优化模式
  //
  // 什么是对称通信？
  // - 所有 GPU 使用相同的内核代码和配置
  // - 消除了因配置差异导致的分支和低效
  //
  // 对称支持的条件（必须同时满足）：
  // 1. comm->isAllDirectP2p：所有 GPU 之间都是直接 P2P 连接
  //    - 没有网络连接，都是 NVLink 或 PCIe
  // 2. comm->nNodes == 1：只有一个节点
  //    - 单节点环境
  // 3. ncclParamWinEnable()：启用 Win 算法
  //    - Win（Wait/Notify）是一种优化的同步机制
  // 4. ncclCuMemEnable()：启用 cuMem
  //    - CUDA 12.2+ 的统一内存管理
  //
  // 对称通信的优势：
  // - 内核代码更简单（无需处理多种情况）
  // - 性能更可预测
  // - 编译器优化空间更大
  // - 减少寄存器压力
  //
  // 对称通信的限制：
  // - 只适用于单节点
  // - 需要所有 GPU 之间直接连接
  // - 需要 cuMem 支持
  //
  // ============================================================

  // 对称支持检测（4 个条件必须同时满足）
  comm->symmetricSupport = comm->isAllDirectP2p && comm->nNodes == 1 && ncclParamWinEnable() && ncclCuMemEnable();

  // devrState：设备端运行时状态
  // bigSize：大消息大小的阈值
  // 初始化为 0，后续根据性能调优结果设置
  comm->devrState.bigSize = 0;

  // 集合引擎（Collective Engine）的统一通信（Unified Communication）对称模式指针
  // baseUCSymReadyPtr：就绪指针（用于同步）
  // baseUCSymComplPtr：完成指针（用于同步）
  // 初始化为 NULL，后续根据是否启用对称模式设置
  comm->ceColl.baseUCSymReadyPtr = NULL;
  comm->ceColl.baseUCSymComplPtr = NULL;

  // ============================================================
  // 设备通信设置（devCommSetup）- 关键同步点
  // ============================================================
  //
  // devCommSetup 是 NCCL 初始化的最后一个关键步骤
  //
  // 为什么要在"最后一个屏障之前"调用？
  // 1. 必须确保所有初始化都已完成
  //    - 拓扑结构已确定
  //    - 通道配置已完成
  //    - 连接已建立（或配置已设置）
  //    - 性能调优已完成
  //
  // 2. 避免死锁
  //    - 如果在其他线程开始启动 NCCL 内核时调用 devCommSetup
  //    - 可能导致资源竞争和死锁
  //    - 这个屏障确保没有线程在前面运行并开始启动 NCCL 内核
  //
  // 3. 设备端数据结构就绪
  //    - devCommSetup 在 GPU 设备上分配通信结构
  //    - 这些结构是内核执行所必需的
  //    - 必须在启动任何内核之前完成
  //
  // devCommSetup 做什么？
  // - 在设备上分配 ncclKernelComm 结构
  // - 拷贝配置信息到设备（rankToLocalRank、buffSizes 等）
  // - 设置 workFifo（主机-设备通信队列）
  // - 设置性能计数器
  // - 拷贝所有通道的配置到设备
  //
  // 为什么是"最后一个屏障"？
  // - 这之后只有一个本地节点内屏障（bootstrapIntraNodeBarrier）
  // - 确保同一节点的所有 ranks 都执行到这里
  // - 然后可以安全地启动集体操作
  //
  // ============================================================

  NCCLCHECKGOTO(devCommSetup(comm), ret, fail);

  timers[TIMER_INIT_CONNECT] = clockNano() - timers[TIMER_INIT_CONNECT];

  // ============================================================
  // 本地节点内屏障（Intra-Node Barrier）
  // ============================================================
  //
  // 这是 initTransportsRank 函数的最后一个同步点
  //
  // 这个调用确保了在同一个节点内的所有 GPU ranks 都执行到了这个同步点
  // 也就是 initTransportsRank 操作都执行完毕了
  //
  // 参数说明：
  // - comm->bootstrap：Bootstrap 上下文
  // - comm->localRankToRank：本地 rank 到全局 rank 的映射数组
  // - comm->localRank：当前 rank 的本地编号
  // - comm->localRanks：本地 rank 的总数
  // - comm->localRankToRank[0]：本地 rank 0 的全局 rank 号（作为根）
  //
  // 为什么需要这个屏障？
  // 1. 确保初始化顺序
  //    - 同一节点的所有 ranks 必须完成初始化
  //    - 才能安全地开始集体操作
  //
  // 2. 资源分配同步
  //    - 某些资源（如共享内存）需要所有 ranks 都准备好
  //    - 避免某个 rank 访问未初始化的资源
  //
  // 3. 错误检测
  //    - 如果某个 rank 初始化失败
  //    - 其他 ranks 会在屏障处超时或检测到错误
  //
  // 4. Proxy 线程同步
  //    - 确保 Proxy 线程已启动并就绪
  //    - 所有 ranks 可以开始通信
  //
  // 实现机制：
  // - 使用共享内存和信号量
  // - 本地 rank 0 作为 coordinator
  // - 其他 ranks 等待 rank 0 的信号
  //
  // ============================================================

  NCCLCHECKGOTO(bootstrapIntraNodeBarrier(comm->bootstrap, comm->localRankToRank, comm->localRank, comm->localRanks, comm->localRankToRank[0]), ret, fail);

  // 我们应该已经分配了所有缓冲区、collective fifos 等，可以恢复亲和性
  TRACE(NCCL_INIT, "rank %d nranks %d - DONE", rank, nranks);

// ============================================================
// 函数退出路径
// ============================================================
//
// exit 标签是正常和异常退出的汇合点
// 所有清理资源的工作都在这里完成
//
// exit:
//   1. 恢复 CPU 亲和性（如果之前设置了）
//   2. 断开 Proxy 共享内存链接（如果不共享资源）
//   3. 释放临时分配的内存
//   4. 返回结果码
//
// fail:
//   发生错误时跳转到 exit 标签
//   确保清理代码被执行
//
// ============================================================

exit:
  // ============================================================
  // 恢复 CPU 亲和性
  // ============================================================
  //
  // 为什么要恢复 CPU 亲和性？
  // 1. 在初始化开始时，我们设置了 CPU 亲和性到本地 GPU
  //    - 确保内存分配在本地 NUMA 节点
  //    - 提高内存访问性能
  //
  // 2. 初始化完成后，需要恢复原来的亲和性
  //    - 避免影响后续应用程序的 CPU 调度
  //    - 允许操作系统自由调度线程到任意 CPU
  //
  // 3. 亲和性保存位置
  //    - affinitySave：在函数开始时保存的原始亲和性
  //    - 在设置 GPU 亲和性之前保存
  //
  // CPU_COUNT(&comm->cpuAffinity)：
  // - 返回 cpuAffinity 中设置的 CPU 数量
  // - 如果为 0，说明没有设置亲和性（某些情况下）
  // - 只有设置了亲和性才需要恢复
  //
  // 注意：即使初始化失败，也要恢复亲和性
  // - 确保进程状态的一致性
  // - 避免资源泄漏
  //
  // ============================================================

  if (CPU_COUNT(&comm->cpuAffinity))
    sched_setaffinity(0, sizeof(cpu_set_t), &affinitySave);

  // ============================================================
  // 断开 Proxy 共享内存链接
  // ============================================================
  //
  // Proxy 共享内存：
  // - Proxy 线程和主进程之间通过共享内存通信
  // - 共享内存通过文件描述符或路径引用
  //
  // 为什么要断开链接？
  // 1. 清理共享资源
  //    - 当不再需要时，释放共享内存
  //    - 避免资源泄漏
  //
  // 2. 多个 comm 共享资源时
  //    - 如果 comm->shareResources == true
  //    - 多个 comm 共享同一个 Proxy
  //    - 不能断开链接（其他 comm 可能还在使用）
  //
  // 3. 单独的 comm
  //    - 如果 comm->sharedRes->owner == comm
  //    - 这个 comm 是资源的所有者
  //    - 负责清理共享资源
  //
  // 4. cuMem 启用时
  //    - 如果启用了 cuMem (ncclCuMemEnable() == true)
  //    - 不需要断开链接（cuMem 管理内存）
  //
  // 什么时候断开链接？
  // - comm 是资源的所有者 (comm->sharedRes->owner == comm)
  // - 不共享资源 (!comm->shareResources)
  // - 初始化成功 (ret == ncclSuccess)
  // - 未启用 cuMem (!ncclCuMemEnable())
  //
  // ncclProxyShmUnlink：
  // - 断开共享内存链接
  // - 删除共享内存文件（如果没有其他引用）
  // - 释放相关资源
  //
  // ============================================================

  /* 如果 split 资源是共享的，我们无法在这里断开 proxy ops pool 的链接，
   * 因为子 comm 可以随时附加父 comm 的 proxy ops pool；
   * 否则，在这里断开链接以确保 pool 被正确清理。 */
  if (comm->sharedRes->owner == comm && !comm->shareResources && ret == ncclSuccess && !ncclCuMemEnable())
    ncclProxyShmUnlink(comm);

  // ============================================================
  // 释放临时分配的内存
  // ============================================================
  //
  // 这些是函数执行过程中临时分配的内存
  // 不再需要后必须释放以避免内存泄漏
  //
  // 释放的内存：
  // - allTopoRanks：所有 rank 的拓扑排名信息数组
  // - nodesTreePatterns：每个节点的树模式数组
  // - nodesFirstRank：每个节点的第一个 rank 数组
  // - allGather3Data：第三次 AllGather 的数据数组
  // - rings：环形拓扑的 rank 排序数组
  // - nvbPeers：NVB peer 的 rank 数组
  // - pxnPeers：PXN peer 的 rank 数组
  //
  // 注意：使用 free() 而不是 ncclMemoryStack
  // - 这些是传统堆分配的内存（通过 ncclCalloc）
  // - 不是永久内存栈的一部分
  // - 必须手动释放
  //
  // 为什么在 exit 之前释放？
  // - 无论成功还是失败都要释放
  // - 确保资源清理的确定性
  // - 避免内存泄漏
  //
  // ============================================================

  free(allTopoRanks);
  free(nodesTreePatterns);
  free(nodesFirstRank);
  free(allGather3Data);
  free(rings);
  free(nvbPeers);
  free(pxnPeers);

  // 返回操作结果
  // - ncclSuccess：初始化成功
  // - 其他值：初始化失败的错误码
  return ret;

// ============================================================
// 错误处理标签
// ============================================================
//
// fail 标签在发生错误时使用
// 使用 goto 跳转到 exit 标签执行清理
//
// 为什么使用 goto 而不是直接 return？
// 1. 确保清理代码执行
//    - 无论是成功还是失败，都需要清理资源
//    - goto fail 跳转到 exit，执行所有清理代码
//
// 2. 简化错误处理
//    - 不需要在每个错误点重复清理代码
//    - 统一的清理路径
//
// 3. 便于维护
//    - 添加新的资源时，只需在 exit 标签添加清理代码
//    - 减少出错的可能性
//
// ============================================================

fail:
  goto exit;  // 跳转到 exit 标签执行清理
}

// 更多参数定义
NCCL_PARAM(SetStackSize, "SET_STACK_SIZE", 0);
NCCL_PARAM(CGAClusterSize, "CGA_CLUSTER_SIZE", NCCL_CONFIG_UNDEF_INT);
NCCL_PARAM(MaxCTAs, "MAX_CTAS", NCCL_CONFIG_UNDEF_INT);
NCCL_PARAM(MinCTAs, "MIN_CTAS", NCCL_CONFIG_UNDEF_INT);
#define NCCL_MAX_CGA_CLUSTER_SIZE 8

NCCL_PARAM(NChannelsPerNetPeer, "NCHANNELS_PER_NET_PEER", NCCL_CONFIG_UNDEF_INT);
NCCL_PARAM(NvlinkUtilCentricSchedEnable, "NVLINK_UTIL_CENTRIC_SCHED_ENABLE", 0);

#define NCCL_COMMINIT_FUNCNAME_LEN 128

// 异步作业结构（用于 ncclCommInitRank）
struct ncclCommInitRankAsyncJob {
  struct ncclAsyncJob base;
  struct ncclComm* comm;
  struct ncclComm** newcomm;
  int cudaDev;
  // 对于 ncclCommInitRank
  int nranks, myrank;
  // root 数量
  int nId;
  // nId 个 ncclUniqueId 数组
  ncclUniqueId* commId;
  // 对于 ncclCommSplit
  struct ncclComm* parent;
  int color, key;
  int splitCount;
  // 对于 Shrink
  int* excludeRanksList;
  int excludeRanksCount;
  // 调用此函数的函数名
  char funcName[NCCL_COMMINIT_FUNCNAME_LEN];
};

// 异步作业结构（用于 ncclCommFinalize）
struct ncclCommFinalizeAsyncJob {
  struct ncclAsyncJob base;
  ncclComm_t comm;
};

NCCL_PARAM(CommSplitShareResources, "COMM_SPLIT_SHARE_RESOURCES", NCCL_CONFIG_UNDEF_INT);
NCCL_PARAM(CommShrinkShareResources, "COMM_SHRINK_SHARE_RESOURCES", NCCL_CONFIG_UNDEF_INT);

// 分割信息结构
typedef struct {
  int key;
  int color;
} commSplitInfo;

// 获取分割信息
static ncclResult_t commGetSplitInfo(struct ncclComm* comm, struct ncclComm* parent, int color, int key, int* nRanksRet, int* myRankRet, int* parentRanksRet) {
  int nRanks = 0, myRank = 0;
  ncclResult_t ret = ncclSuccess;

  commSplitInfo* info = NULL;
  NCCLCHECKGOTO(ncclCalloc(&info, parent->nRanks), ret, fail);

  // 计算 nRanks、my rank 以及前后 ranks
  info[parent->rank].color = color;
  info[parent->rank].key = key;
  NCCLCHECKGOTO(bootstrapAllGather(parent->bootstrap, info, sizeof(commSplitInfo)), ret, fail);

  // 负 color 不创建新 comm，直接返回
  if (color == NCCL_SPLIT_NOCOLOR) goto exit;

  memset(parentRanksRet, 0xff, sizeof(int) * parent->nRanks);
  for (int i = 0; i < parent->nRanks; i++) {
    if (info[i].color != color) continue;
    // 找到插入这个 rank 的位置
    int insert = 0;
    while (insert < nRanks && info[parentRanksRet[insert]].key <= info[i].key) insert++;
    // 在插入位置之后移动 ranks
    for (int r = nRanks; r > insert; r--) parentRanksRet[r] = parentRanksRet[r - 1];
    // 插入我们的 rank
    parentRanksRet[insert] = i;
    nRanks++;
  }

  for (int i = 0; i < nRanks; i++) {
    if (parentRanksRet[i] == parent->rank) myRank = i;
  }

  *nRanksRet = nRanks;
  *myRankRet = myRank;

exit:
  free(info);
  return ret;
fail:
  goto exit;
}

// 获取父 ranks（用于 Shrink 操作）
static ncclResult_t getParentRanks(int parentRanks, int parentRank, int* excludeRanksList, int excludeRanksCount, int* nRanksRet, int* myRankRet, int* parentRanksRet) {
  int count = 0, j = 0;
  for (int i = 0; i < parentRanks; i++) {
    // 假设 excludeRanksList 是已排序的
    if (j < excludeRanksCount && excludeRanksList[j] == i) {
      j++;
      continue;
    }
    if (i == parentRank)
        *myRankRet = count;
    parentRanksRet[count++] = i;
  }
  *nRanksRet = parentRanks - excludeRanksCount;
  return ncclSuccess;
}
// ============================================================================
// ncclCommInitRankFunc - 通信器初始化的核心实现函数
// 这个函数执行实际的通信器初始化工作
// ============================================================================
static ncclResult_t ncclCommInitRankFunc(struct ncclAsyncJob* job_) {
  struct ncclCommInitRankAsyncJob* job = (struct ncclCommInitRankAsyncJob*)job_;
  ncclComm_t comm = job->comm;
  ncclResult_t res = ncclSuccess;
  int archMajor, archMinor;
  size_t maxLocalSizeBytes = 0;
  int cudaDev = job->cudaDev;
  int* parentRanks = NULL;
  int cudaArch;
  int maxSharedMem = 0;
  double sum_timers = 0;
  uint64_t timers[TIMERS_INIT_COUNT] = {0};
  unsigned long long commIdHash;

  timers[TIMER_INIT_TOTAL] = clockNano();

  // 设置 CUDA 设备
  CUDACHECKGOTO(cudaSetDevice(cudaDev), res, fail);

  // 获取 GPU 计算能力
  // 查询每个线程块的最大共享内存
  CUDACHECKGOTO(cudaDeviceGetAttribute(&maxSharedMem, cudaDevAttrMaxSharedMemoryPerBlockOptin, cudaDev), res, fail);
  // 获取设备的计算能力主版本号
  CUDACHECKGOTO(cudaDeviceGetAttribute(&archMajor, cudaDevAttrComputeCapabilityMajor, cudaDev), res, fail);
  // 获取设备的计算能力次版本号
  CUDACHECKGOTO(cudaDeviceGetAttribute(&archMinor, cudaDevAttrComputeCapabilityMinor, cudaDev), res, fail);
  // 将主版本和次版本拼接成一个整数，便于后续判断比较版本号
  cudaArch = 100 * archMajor + 10 * archMinor;

  timers[TIMER_INIT_KERNELS] = clockNano();

  // 初始化 CUDA 内核
  NCCLCHECK(ncclInitKernelsForDevice(cudaArch, maxSharedMem, &maxLocalSizeBytes));

  // 设置最大栈限制
  // 设置所有内核的最大栈大小，避免加载时 CUDA 内存重新配置（参考 NVSHMEM 问题）
  if (maxLocalSizeBytes > 0 && ncclParamSetStackSize() == 1) {
    TRACE(NCCL_INIT, "Setting cudaLimitStackSize to %zu", maxLocalSizeBytes);
    CUDACHECKIGNORE(cudaDeviceSetLimit(cudaLimitStackSize, maxLocalSizeBytes));
  }

  timers[TIMER_INIT_KERNELS] = clockNano() - timers[TIMER_INIT_KERNELS];

  // 处理 ncclCommSplit 的情况
  // 当调用 split 时，parent 指向父 comm
  if (job->parent) {
    NCCLCHECKGOTO(ncclCalloc(&parentRanks, job->parent->nRanks), res, fail);
    if (job->excludeRanksCount) {
      // Shrink 操作
      NCCLCHECKGOTO(getParentRanks(job->parent->nRanks, job->parent->rank, job->excludeRanksList, job->excludeRanksCount, &job->nranks, &job->myrank, parentRanks), res, fail);
    } else {
      // Split 操作
      NCCLCHECKGOTO(commGetSplitInfo(comm, job->parent, job->color, job->key, &job->nranks, &job->myrank, parentRanks), res, fail);
      // 负 color 不创建新 comm 对象。我们需要参与 allgather，但现在完成了。
      if (job->color == NCCL_SPLIT_NOCOLOR) 
        goto exit;
    }
    // 子 hash 从（父 hash，split count，color）获得
    uint64_t hacc[2] = {1, 1};
    eatHash(hacc, &job->parent->commHash);
    eatHash(hacc, &job->splitCount);
    eatHash(hacc, &job->color);
    comm->commHash = digestHash(hacc);

    timers[TIMER_INIT_ALLOC] = clockNano();

    // 分配通信器
    NCCLCHECKGOTO(commAlloc(comm, job->parent, job->nranks, job->myrank), res, fail);
    timers[TIMER_INIT_ALLOC] = clockNano() - timers[TIMER_INIT_ALLOC];

    INFO(NCCL_INIT, "%s comm %p rank %d nranks %d cudaDev %d nvmlDev %d busId %lx parent %p splitCount %d color %d key %d - Init START", job->funcName,
         comm, comm->rank, comm->nRanks, comm->cudaDev, comm->nvmlDev, comm->busId, job->parent, job->splitCount, job->color, job->key);

    timers[TIMER_INIT_BOOTSTRAP] = clockNano();
    // Bootstrap split 初始化
    NCCLCHECKGOTO(bootstrapSplit(comm->commHash, comm, job->parent, job->color, job->key, parentRanks), res, fail);
    timers[TIMER_INIT_BOOTSTRAP] = clockNano() - timers[TIMER_INIT_BOOTSTRAP];

    // debug info，没有使用 commId
    commIdHash = 0;
  } else {
    // 使用第一个 commId 获得唯一的 hash
    // 根据 ncclUniqueId 生成一个 8 字节的 id 号
    comm->commHash = commIdHash = getHash(job->commId->internal, NCCL_UNIQUE_ID_BYTES);

    timers[TIMER_INIT_ALLOC] = clockNano();

    // 继续初始化 comm 结构体
    // 内部调用 ncclNetInit 初始化网络插件
    // 函数实现会加载外部库和 2 个最重要的内部网络插件：ncclNetIb 和 ncclNetSocket
    NCCLCHECKGOTO(commAlloc(comm, NULL, job->nranks, job->myrank), res, fail);
    timers[TIMER_INIT_ALLOC] = clockNano() - timers[TIMER_INIT_ALLOC];

    INFO(NCCL_INIT, "%s comm %p rank %d nranks %d cudaDev %d nvmlDev %d busId %lx commId 0x%llx - Init START", job->funcName,
         comm, comm->rank, comm->nRanks, comm->cudaDev, comm->nvmlDev, comm->busId, commIdHash);

    timers[TIMER_INIT_BOOTSTRAP] = clockNano();
    // 网络初始化，建立 ring 环形连接
    NCCLCHECKGOTO(bootstrapInit(job->nId, (struct ncclBootstrapHandle*)job->commId, comm), res, fail);
    timers[TIMER_INIT_BOOTSTRAP] = clockNano() - timers[TIMER_INIT_BOOTSTRAP];
  }

  // 版本号
  comm->cudaArch = cudaArch;

  // 重要函数：拓扑路径和传输层初始化，核心函数
  NCCLCHECKGOTO(initTransportsRank(comm, job->parent, timers), res, fail);

  // 更新通信器状态
  comm->initState = ncclSuccess;
  timers[TIMER_INIT_TOTAL] = clockNano() - timers[TIMER_INIT_TOTAL];

  // 为 replay 工具追踪此调用
  if (job->parent) {
    /* 解链接子 abort 标志 */
    __atomic_store_n(&job->parent->childAbortFlag, NULL, __ATOMIC_RELEASE);
    TRACE_CALL("ncclCommSplit(%p, %d, %d, %p, %d, %d)", job->parent, job->color, job->key, comm, comm->rank, comm->nRanks);
    INFO(NCCL_INIT, "%s comm %p rank %d nranks %d cudaDev %d nvmlDev %d busId %lx parent %p splitCount %d color %d key %d - Init COMPLETE", job->funcName,
         comm, comm->rank, comm->nRanks, comm->cudaDev, comm->nvmlDev, comm->busId, job->parent, job->splitCount, job->color, job->key);
  } else {
    // replay 工具的名字对所有变体都是 ncclCommInitRank
    TRACE_CALL("ncclCommInitRank(%p, %d, 0x%llx, %d, %d)", comm, comm->nRanks, commIdHash, comm->rank, comm->cudaDev);
    INFO(NCCL_INIT, "%s comm %p rank %d nranks %d cudaDev %d nvmlDev %d busId %lx commId 0x%llx - Init COMPLETE", job->funcName,
         comm, comm->rank, comm->nRanks, comm->cudaDev, comm->nvmlDev, comm->busId, commIdHash);
  }

  // 打印初始化时间统计
  sum_timers = 0.0;
  for (int it = 1; it < TIMERS_INIT_COUNT; ++it)
    sum_timers += (timers[it] / 1e9);
  INFO(NCCL_INIT | NCCL_PROFILE,
       "Init timings - %s: rank %d nranks %d total %.2f (kernels %.2f, alloc %.2f, bootstrap %.2f, allgathers %.2f, topo %.2f, graphs %.2f, "
       "connections %.2f, rest %.2f)",
       job->funcName, comm->rank, comm->nRanks,
       timers[TIMER_INIT_TOTAL] / 1e9, timers[TIMER_INIT_KERNELS] / 1e9, timers[TIMER_INIT_ALLOC] / 1e9,
       timers[TIMER_INIT_BOOTSTRAP] / 1e9, timers[TIMER_INIT_ALLGATHER] / 1e9, timers[TIMER_INIT_TOPO] / 1e9,
       timers[TIMER_INIT_GRAPHS] / 1e9, timers[TIMER_INIT_CONNECT] / 1e9, timers[TIMER_INIT_TOTAL] / 1e9 - sum_timers);

exit:
  //返回新的通信器给调用方
  if (job->newcomm) {
    /* 分配给用户指针 */
    __atomic_store_n(job->newcomm, comm, __ATOMIC_RELEASE);
  }
  free(parentRanks);
  return res;
  
fail:
  comm->initState = res;
  goto exit;
}

// 配置默认值宏
#define NCCL_CONFIG_DEFAULT(config, field, undef, defvalue, fieldStr, format) \
  if (config->field == undef) { \
    config->field = defvalue; \
  } else { \
    INFO(NCCL_ENV, "Comm config " fieldStr " set to " format, config->field); \
  }

// 从环境变量覆盖配置
static ncclResult_t envConfigOverride(ncclComm_t comm) {
  ncclResult_t ret = ncclSuccess;
  const char* tmpNetName = comm->config.netName;
  const char* envNetName;
  int blockingEnv;
  int cgaClusterSizeEnv;
  int minCTAsEnv;
  int maxCTAsEnv;
  int splitShareEnv;
  const char* collnetEnableEnv;
  int ctaPolicyEnv;
  int shrinkShareEnv;
  int nvlsCTAsEnv;
  int nChannelsPerNetPeerEnv;
  int nvlinkUtilCentricSchedEnableEnv;

  /* 用环境变量覆盖配置 */
  blockingEnv = ncclParamCommBlocking();
  if (blockingEnv == 0 || blockingEnv == 1)
    comm->config.blocking = blockingEnv;

  cgaClusterSizeEnv = ncclParamCGAClusterSize();
  if (0 <= cgaClusterSizeEnv && cgaClusterSizeEnv <= NCCL_MAX_CGA_CLUSTER_SIZE) {
    comm->config.cgaClusterSize = cgaClusterSizeEnv;
  } else if (cgaClusterSizeEnv > NCCL_MAX_CGA_CLUSTER_SIZE) {
    INFO(NCCL_ENV, "NCCL_CGA_CLUSTER_SIZE value %d is too big. Limiting value to %d.", cgaClusterSizeEnv, NCCL_MAX_CGA_CLUSTER_SIZE);
    comm->config.cgaClusterSize = NCCL_MAX_CGA_CLUSTER_SIZE;
  }

  minCTAsEnv = ncclParamMinCTAs();
  if (minCTAsEnv != NCCL_CONFIG_UNDEF_INT) {
    if (minCTAsEnv <= 0)
      INFO(NCCL_ENV, "NCCL_MIN_CTAS %d is too low, leaving it set at %d", minCTAsEnv, comm->config.minCTAs);
    else
      comm->config.minCTAs = minCTAsEnv;
  }

  maxCTAsEnv = ncclParamMaxCTAs();
  if (maxCTAsEnv != NCCL_CONFIG_UNDEF_INT) {
    if (maxCTAsEnv <= 0)
      INFO(NCCL_ENV, "NCCL_MAX_CTAS %d is too low, leaving it set at %d", maxCTAsEnv, comm->config.maxCTAs);
    else
      comm->config.maxCTAs = maxCTAsEnv;
  }

  /* 用环境变量覆盖配置 */
  nChannelsPerNetPeerEnv = ncclParamNChannelsPerNetPeer();
  if (nChannelsPerNetPeerEnv != NCCL_CONFIG_UNDEF_INT) {
    if (nChannelsPerNetPeerEnv <= 0)
      INFO(NCCL_ENV, "NCCL_NCHANNELS_PER_NET_PEER %d is too low, leaving it set at %d", nChannelsPerNetPeerEnv, comm->config.nChannelsPerNetPeer);
    else
      comm->config.nChannelsPerNetPeer = nChannelsPerNetPeerEnv;
  }

  nvlinkUtilCentricSchedEnableEnv = ncclParamNvlinkUtilCentricSchedEnable();
  if (nvlinkUtilCentricSchedEnableEnv != NCCL_CONFIG_UNDEF_INT) {
    if (nvlinkUtilCentricSchedEnableEnv != 0 && nvlinkUtilCentricSchedEnableEnv != 1)
      INFO(NCCL_ENV, "NCCL_NVLINK_UTIL_CENTRIC_SCHED_ENABLE %d is not valid, leaving it set at %d", nvlinkUtilCentricSchedEnableEnv, comm->config.nvlinkCentricSched);
    else
      comm->config.nvlinkCentricSched = nvlinkUtilCentricSchedEnableEnv;
  }

  envNetName = ncclGetEnv("NCCL_NET");
  if (envNetName)
    tmpNetName = envNetName;
  if (tmpNetName != NULL) {
    int netNameLen = strlen(tmpNetName) + 1;
    comm->config.netName = (char*)malloc(netNameLen);
    memcpy((void*)comm->config.netName, tmpNetName, netNameLen);
  } else {
    comm->config.netName = NULL;
  }

  splitShareEnv = ncclParamCommSplitShareResources();
  if (splitShareEnv != NCCL_CONFIG_UNDEF_INT) {
    comm->config.splitShare = splitShareEnv;
  }
  shrinkShareEnv = ncclParamCommShrinkShareResources();
  if (shrinkShareEnv != NCCL_CONFIG_UNDEF_INT) {
    comm->config.shrinkShare = shrinkShareEnv;
  }

  // NCCL_COLLNET_ENABLE 需要在每次 comm init 时重新加载
  // 因为用户可能会动态更改环境变量来启用/禁用 collnet
  collnetEnableEnv = ncclGetEnv("NCCL_COLLNET_ENABLE");
  if (collnetEnableEnv != NULL) {
    int collnetEnableInt = (int)strtol(collnetEnableEnv, NULL, 0);
    if (collnetEnableInt != NCCL_CONFIG_UNDEF_INT) {
      comm->config.collnetEnable = collnetEnableInt;
      INFO(NCCL_ENV, "NCCL_COLLNET_ENABLE set by environment to %d.", collnetEnableInt);
    }
  }

  ctaPolicyEnv = ncclParamCtaPolicy();
  if (ctaPolicyEnv != NCCL_CONFIG_UNDEF_INT) {
    comm->config.CTAPolicy = ctaPolicyEnv;
  }

  nvlsCTAsEnv = ncclParamNvlsChannels();
  if (nvlsCTAsEnv != NCCL_CONFIG_UNDEF_INT) {
    comm->config.nvlsCTAs = nvlsCTAsEnv;
  }

  /* 如果需要，限制通道数 */
  if (comm->config.minCTAs > MAXCHANNELS) {
    INFO(NCCL_ENV, "minCTAs %d is larger than #channels upper limit %d, cap it to %d", comm->config.minCTAs, MAXCHANNELS, MAXCHANNELS);
    comm->config.minCTAs = MAXCHANNELS;
  }

  if (comm->config.maxCTAs > MAXCHANNELS) {
    INFO(NCCL_ENV, "maxCTAs %d is larger than #channels upper limit %d, cap it to %d", comm->config.maxCTAs, MAXCHANNELS, MAXCHANNELS);
    comm->config.maxCTAs = MAXCHANNELS;
  }

  if (comm->config.minCTAs > comm->config.maxCTAs) {
    INFO(NCCL_ENV, "minCTAs %d is larger than maxCTAs %d, set both to %d", comm->config.minCTAs, comm->config.maxCTAs, comm->config.maxCTAs);
    comm->config.minCTAs = comm->config.maxCTAs;
  }

  if (comm->config.splitShare != 1 && comm->config.splitShare != 0) {
    INFO(NCCL_ENV, "splitShare %d is not a valid value 0/1, set it to 0", comm->config.splitShare);
    comm->config.splitShare = 0;
  }

  if (comm->config.collnetEnable != 1 && comm->config.collnetEnable != 0) {
    INFO(NCCL_ENV, "collnetEnable %d is not a valid value 0/1, set it to 0", comm->config.collnetEnable);
    comm->config.collnetEnable = 0;
  }

  if (comm->config.CTAPolicy < NCCL_CTA_POLICY_DEFAULT || comm->config.CTAPolicy > NCCL_CTA_POLICY_ZERO) {
    INFO(NCCL_ENV, "CTAPolicy %d is not a valid value, set it to %d", comm->config.CTAPolicy, NCCL_CTA_POLICY_DEFAULT);
    comm->config.CTAPolicy = NCCL_CTA_POLICY_DEFAULT;
  }

  if (comm->config.nvlsCTAs != NCCL_CONFIG_UNDEF_INT && comm->config.nvlsCTAs <= 0) {
    INFO(NCCL_ENV, "nvlsCTAs %d is not a valid value, NCCL will decide the default value automatically", comm->config.nvlsCTAs);
    comm->config.nvlsCTAs = NCCL_CONFIG_UNDEF_INT;
  }

  return ret;
}

// 从父 comm 复制配置
static ncclResult_t copyCommConfig(ncclComm_t childComm, ncclComm_t parent) {
  memcpy(&childComm->config, &parent->config, sizeof(ncclConfig_t));
  NCCLCHECK(envConfigOverride(childComm));
  return ncclSuccess;
}

// 解析通信器配置
static ncclResult_t parseCommConfig(ncclComm_t comm, ncclConfig_t *config) {
  ncclResult_t ret = ncclSuccess;
  /* config 在此函数中不能为 NULL */
  // 默认配置
  ncclConfig_t defaultConfig = NCCL_CONFIG_INITIALIZER;
  ncclConfig_t internalConfig = NCCL_CONFIG_INITIALIZER;
  ncclConfig_t *internalConfigPtr;
  size_t realSize;

  internalConfig.magic = 0;
  internalConfigPtr = &internalConfig;
  if (config) {
    memcpy((void*)&realSize, (void*)config, sizeof(size_t));
    realSize = realSize > sizeof(ncclConfig_t) ? sizeof(ncclConfig_t) : realSize;
    memcpy((void*)internalConfigPtr, (void*)config, realSize);
    if (internalConfigPtr->magic != 0xcafebeef) {
      WARN("ncclConfig_t argument not initialized via NCCL_CONFIG_INITIALIZER");
      ret = ncclInvalidArgument;
      goto fail;
    }

    /* 检查版本 */
    if (internalConfigPtr->version < NCCL_VERSION(2, 14, 0)) {
      internalConfigPtr->blocking = defaultConfig.blocking;
    }

    if (internalConfigPtr->version < NCCL_VERSION(2, 17, 0)) {
      internalConfigPtr->cgaClusterSize = defaultConfig.cgaClusterSize;
      internalConfigPtr->minCTAs = defaultConfig.minCTAs;
      internalConfigPtr->maxCTAs = defaultConfig.maxCTAs;
      internalConfigPtr->netName = defaultConfig.netName;
    }

    if (internalConfigPtr->version < NCCL_VERSION(2, 25, 0)) {
      internalConfigPtr->trafficClass = defaultConfig.trafficClass;
    }

    if (internalConfigPtr->version < NCCL_VERSION(2, 27, 0)) {
      internalConfigPtr->collnetEnable = defaultConfig.collnetEnable;
      internalConfigPtr->CTAPolicy = defaultConfig.CTAPolicy;
      internalConfigPtr->shrinkShare = defaultConfig.shrinkShare;
      internalConfigPtr->nvlsCTAs = defaultConfig.nvlsCTAs;
    }
    if (internalConfigPtr->version < NCCL_VERSION(2, 28, 0)) {
      internalConfigPtr->nChannelsPerNetPeer = defaultConfig.nChannelsPerNetPeer;
      internalConfigPtr->nvlinkCentricSched = defaultConfig.nvlinkCentricSched;
    }
  }

  /* 检查输入配置属性，-1 表示用户未定义，应使用 NCCL 的默认值 */
  if (internalConfigPtr->blocking != NCCL_CONFIG_UNDEF_INT && internalConfigPtr->blocking != 0 && internalConfigPtr->blocking != 1) {
    WARN("Invalid config blocking attribute value %d", internalConfigPtr->blocking);
    ret = ncclInvalidArgument;
    goto fail;
  }

  if (internalConfigPtr->cgaClusterSize != NCCL_CONFIG_UNDEF_INT && internalConfigPtr->cgaClusterSize < 0) {
    WARN("Invalid config cgaClusterSize attribute value %d", internalConfigPtr->cgaClusterSize);
    ret = ncclInvalidArgument;
    goto fail;
  }

  if ((internalConfigPtr->minCTAs != NCCL_CONFIG_UNDEF_INT &&
    internalConfigPtr->minCTAs <= 0) ||
    (internalConfigPtr->maxCTAs != NCCL_CONFIG_UNDEF_INT &&
      internalConfigPtr->maxCTAs <= 0) ||
      (internalConfigPtr->minCTAs > internalConfigPtr->maxCTAs)) {
    WARN("Invalid config min/max channels attribute value %d/%d", internalConfigPtr->minCTAs, internalConfigPtr->maxCTAs);
    ret = ncclInvalidArgument;
    goto fail;
  }

  if (internalConfigPtr->splitShare != NCCL_CONFIG_UNDEF_INT && internalConfigPtr->splitShare != 0 && internalConfigPtr->splitShare != 1) {
    WARN("Invalid config splitShare attribute value %d", internalConfigPtr->splitShare);
    ret = ncclInvalidArgument;
    goto fail;
  }

  if (internalConfigPtr->collnetEnable != NCCL_CONFIG_UNDEF_INT && (internalConfigPtr->collnetEnable < 0 || internalConfigPtr->collnetEnable > 1)) {
    WARN("Invalid config collnetEnable attribute value %d", internalConfigPtr->collnetEnable);
    ret = ncclInvalidArgument;
    goto fail;
  }

  if (internalConfigPtr->CTAPolicy != NCCL_CONFIG_UNDEF_INT && (internalConfigPtr->CTAPolicy < NCCL_CTA_POLICY_DEFAULT ||
    internalConfigPtr->CTAPolicy > NCCL_CTA_POLICY_ZERO)) {
    WARN("Invalid config policy attribute value %d", internalConfigPtr->CTAPolicy);
    ret = ncclInvalidArgument;
    goto fail;
  }

  if (internalConfigPtr->shrinkShare != NCCL_CONFIG_UNDEF_INT && internalConfigPtr->shrinkShare != 0 && internalConfigPtr->shrinkShare != 1) {
    WARN("Invalid config shrinkShare attribute value %d", internalConfigPtr->shrinkShare);
    ret = ncclInvalidArgument;
    goto fail;
  }

  if (internalConfigPtr->nvlsCTAs != NCCL_CONFIG_UNDEF_INT && internalConfigPtr->nvlsCTAs <= 0) {
    WARN("Invalid config nvlsCTAs attribute value %d", internalConfigPtr->nvlsCTAs);
    ret = ncclInvalidArgument;
    goto fail;
  }

  if (internalConfigPtr->nChannelsPerNetPeer != NCCL_CONFIG_UNDEF_INT && (internalConfigPtr->nChannelsPerNetPeer <= 0 || internalConfigPtr->nChannelsPerNetPeer > MAXCHANNELS)) {
    WARN("Invalid config nChannelsPerNetPeer attribute value %d", internalConfigPtr->nChannelsPerNetPeer);
    ret = ncclInvalidArgument;
    goto fail;
  }

  if (internalConfigPtr->nvlinkCentricSched != NCCL_CONFIG_UNDEF_INT && internalConfigPtr->nvlinkCentricSched != 0 && internalConfigPtr->nvlinkCentricSched != 1) {
    WARN("Invalid config nvlinkCentricSched attribute value %d", internalConfigPtr->nvlinkCentricSched);
    ret = ncclInvalidArgument;
    goto fail;
  }

  /* 默认配置值可以在不同平台上调整 */
  NCCL_CONFIG_DEFAULT(internalConfigPtr, blocking, NCCL_CONFIG_UNDEF_INT, 1, "Blocking", "%d");
  NCCL_CONFIG_DEFAULT(internalConfigPtr, cgaClusterSize, NCCL_CONFIG_UNDEF_INT, 4, "CGA cluster size", "%d");
  NCCL_CONFIG_DEFAULT(internalConfigPtr, minCTAs, NCCL_CONFIG_UNDEF_INT, 1, "Min CTAs", "%d");
  NCCL_CONFIG_DEFAULT(internalConfigPtr, maxCTAs, NCCL_CONFIG_UNDEF_INT, MAXCHANNELS, "Max CTAs", "%d");
  NCCL_CONFIG_DEFAULT(internalConfigPtr, netName, NCCL_CONFIG_UNDEF_PTR, NULL, "Net name", "%s");
  NCCL_CONFIG_DEFAULT(internalConfigPtr, splitShare, NCCL_CONFIG_UNDEF_INT, 0, "Split share", "%d");
  NCCL_CONFIG_DEFAULT(internalConfigPtr, trafficClass, NCCL_CONFIG_UNDEF_INT, NCCL_CONFIG_UNDEF_INT, "Traffic class", "%d");
  NCCL_CONFIG_DEFAULT(internalConfigPtr, commName, NCCL_CONFIG_UNDEF_PTR, NULL, "Comm name", "%s");
  NCCL_CONFIG_DEFAULT(internalConfigPtr, collnetEnable, NCCL_CONFIG_UNDEF_INT, 0, "Collnet enable", "%d");
  NCCL_CONFIG_DEFAULT(internalConfigPtr, CTAPolicy, NCCL_CONFIG_UNDEF_INT, NCCL_CTA_POLICY_DEFAULT, "CTA policy flags", "%d");
  NCCL_CONFIG_DEFAULT(internalConfigPtr, shrinkShare, NCCL_CONFIG_UNDEF_INT, 0, "shrinkShare", "%d");
  NCCL_CONFIG_DEFAULT(internalConfigPtr, nvlsCTAs, NCCL_CONFIG_UNDEF_INT, NCCL_CONFIG_UNDEF_INT, "nvlsCTAs", "%d");
  NCCL_CONFIG_DEFAULT(internalConfigPtr, nChannelsPerNetPeer, NCCL_CONFIG_UNDEF_INT,
                      NCCL_CONFIG_UNDEF_INT, "nChannelsPerNetPeer", "%d");
  NCCL_CONFIG_DEFAULT(internalConfigPtr, nvlinkCentricSched, NCCL_CONFIG_UNDEF_INT, 0, "nvlinkCentricSched", "%d");

  /* 将配置分配给通信器 */
  comm->config.blocking = internalConfigPtr->blocking;
  comm->config.cgaClusterSize = internalConfigPtr->cgaClusterSize;
  comm->config.minCTAs = internalConfigPtr->minCTAs;
  comm->config.maxCTAs = internalConfigPtr->maxCTAs;
  comm->config.netName = internalConfigPtr->netName;
  comm->config.splitShare = internalConfigPtr->splitShare;
  comm->config.trafficClass = internalConfigPtr->trafficClass;
  comm->config.commName = internalConfigPtr->commName;
  comm->config.collnetEnable = internalConfigPtr->collnetEnable;
  comm->config.CTAPolicy = internalConfigPtr->CTAPolicy;
  comm->config.shrinkShare = internalConfigPtr->shrinkShare;
  comm->config.nvlsCTAs = internalConfigPtr->nvlsCTAs;
  comm->config.nChannelsPerNetPeer = internalConfigPtr->nChannelsPerNetPeer;
  comm->config.nvlinkCentricSched = internalConfigPtr->nvlinkCentricSched;
  NCCLCHECKGOTO(envConfigOverride(comm), ret, fail);

exit:
  return ret;
fail:
  goto exit;
}

// 释放异步作业
static void ncclCommInitJobFree(void* _job) {
  struct ncclCommInitRankAsyncJob* job = (struct ncclCommInitRankAsyncJob*)_job;
  free(job->commId);
  free(_job);
}

// ncclCommInitRank 的设备端实现
static ncclResult_t ncclCommInitRankDev(ncclComm_t* newcomm, int nranks, int nId, ncclUniqueId* commId, int myrank, int cudaDev, ncclConfig_t *config, const char funcName[]) {
  if (nId <= 0 || nId > nranks) {
    WARN("improper usage of ncclCommInitRank: nId = %d, nranks=%d", nId, nranks);
    return ncclInvalidArgument;
  }
  ncclResult_t res = ncclSuccess;
  const char* commIdEnv = NULL;
  ncclComm_t comm = NULL;
  struct ncclCommInitRankAsyncJob* job = NULL;
  bool launchedJob = false;

  // 首先调用 ncclInit，设置环境
  // 初始化环境变量，这个 root rank 已经调用过了不再执行
  // 其他 rank 会执行，获取地址到 bootstrapNetIfAddr
  NCCLCHECKGOTO(ncclInit(), res, fail);

  // 在特定条件下显示版本信息
  if (ncclDebugLevel > NCCL_LOG_WARN || (ncclDebugLevel != NCCL_LOG_NONE && myrank == 0)) {
    static std::once_flag once;
    std::call_once(once, showVersion);
  }

  // 确保 CUDA 运行时已初始化
  CUDACHECKGOTO(cudaFree(NULL), res, fail);

  NCCLCHECKGOTO(PtrCheck(newcomm, "CommInitRank", "newcomm"), res, fail);
  NCCLCHECKGOTO(PtrCheck(config, "CommInitRank", "config"), res, fail);

  // 参数有效性检查
  if (nranks < 1 || myrank < 0 || myrank >= nranks) {
    WARN("Invalid rank requested : %d/%d", myrank, nranks);
    res = ncclInvalidArgument;
    goto fail;
  }

  // 分配 comm 通信器内存
  NCCLCHECKGOTO(ncclCalloc(&comm, 1), res, fail);

  // 分配 abortFlag 等相关标志
  NCCLCHECKGOTO(ncclCalloc(&comm->abortFlag, 1), res, fail);
  NCCLCHECKGOTO(ncclCudaHostCalloc(&comm->abortFlagDev, 1), res, fail);
  NCCLCHECKGOTO(ncclCalloc(&comm->abortFlagRefCount, 1), res, fail);

  comm->startMagic = comm->endMagic = NCCL_MAGIC;  // 用于检测 comm 损坏
  *comm->abortFlagRefCount = 1;

  // 解析配置，保存到 comm->config 中
  NCCLCHECKGOTO(parseCommConfig(comm, config), res, fail);

  /* 从 ncclInProgress 开始，如果初始化成功将改为 ncclSuccess */
  comm->initState = ncclInProgress;

  // 把指针返回给调用方
  *newcomm = comm;

  // 分配一个 job 任务
  NCCLCHECKGOTO(ncclCalloc(&job, 1), res, fail);
  job->nId = nId;
  job->comm = comm;
  job->nranks = nranks;     // 有多少个进程节点
  job->myrank = myrank;    // 本机的节点号
  job->cudaDev = cudaDev;  // 当前进程绑定的 GPU 设备号

  snprintf(job->funcName, NCCL_COMMINIT_FUNCNAME_LEN, "%s", funcName);

  // 需要拷贝 commIds 以允许异步 commInit 并避免从 ncclUniqueId 到 ncclBootstrapHandle 的对齐问题
  // ncclUniqueId 和 ncclBootstrapHandle 的对齐要求不同
  // 因此来自用户的 ID 数组可能没有正确对齐以转换为 ncclBootstrapHandle
  // 拷贝到分配的内存保证内存对任何对象都正确对齐，消除了这个问题
  // 可能有多个 root，拷贝 nId 个 commId
  NCCLCHECKGOTO(ncclCalloc(&job->commId, nId), res, fail);
  memcpy(job->commId, commId, nId * NCCL_UNIQUE_ID_BYTES);

  commIdEnv = ncclGetEnv("NCCL_COMM_ID");
  if (commIdEnv && myrank == 0) {
    INFO(NCCL_ENV, "NCCL_COMM_ID set by environment to %s", commIdEnv);
    // 如果设置了 NCCL_COMM_ID，强制设置 nId 为 1，并且只能使用 rank 0 来创建 root 线程
    if (nId > 1) {
      INFO(NCCL_INIT | NCCL_ENV, "NCCL_COMM_ID cannot be used with more than one ncclUniqueId");
      job->nId = 1;
    }
    // root 节点创建监听套接字
    // 在 bootstrapping 之前启动 bootstrap root，只使用第一个 handle
    // 如果有多个 root，只使用第一个 root
    NCCLCHECKGOTO(bootstrapCreateRoot((struct ncclBootstrapHandle*)&job->commId[0], true), res, fail);
  }

  launchedJob = true;
  // 启动任务，调用 ncclCommInitRankFunc 函数，会调用 ncclNetInit 函数
  NCCLCHECKGOTO(ncclAsyncLaunch((struct ncclAsyncJob*)job, ncclCommInitRankFunc, NULL, ncclCommInitJobFree, comm), res, fail);

exit:
  return ncclGroupErrCheck(res);
fail:
  if (job && !launchedJob) ncclCommInitJobFree(job);
  if (comm) {
    free(comm->abortFlag);
    if (comm->abortFlagDev) (void)ncclCudaHostFree((void*)comm->abortFlagDev);
    free(comm->abortFlagRefCount);
    free(comm);
  }
  if (newcomm) *newcomm = NULL;
  goto exit;
}

// ============================================================================
// NCCL API 函数实现
// ============================================================================

// ncclCommInitRank - 初始化指定 rank 的通信
NCCL_API(ncclResult_t, ncclCommInitRank, ncclComm_t* newcomm, int nranks, ncclUniqueId commId, int myrank);
ncclResult_t ncclCommInitRank(ncclComm_t* newcomm, int nranks, ncclUniqueId commId, int myrank) {
  NVTX3_RANGE(NcclNvtxParamsCommInitRank)

  // 初始化 CUDA 库
  (void)ncclCudaLibraryInit();

  int cudaDev;
  // 初始化默认配置
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;

  // 获取当前进程绑定的 GPU 设备
  // 在调用 ncclCommInitRank 之前，需要调用 cudaSetDevice(localRank) 绑定到那个 GPU 设备
  CUDACHECK(cudaGetDevice(&cudaDev));

  // 默认只有一个 root
  NCCLCHECK(ncclCommInitRankDev(newcomm, nranks, /* commId 数量 */1, &commId, myrank, cudaDev, &config, __func__));

  NVTX3_RANGE_ADD_PAYLOAD(CommInitRank, NcclNvtxParamsCommInitRankSchema,
    NVTX3_PAYLOAD((*newcomm)->commHash, nranks, myrank, cudaDev));

  return ncclSuccess;
}

// ncclCommInitAll - 初始化所有设备的通信
NCCL_API(ncclResult_t, ncclCommInitAll, ncclComm_t* comms, int ndev, const int* devlist);
ncclResult_t ncclCommInitAll(ncclComm_t* comms, int ndev, const int* devlist) {
  ncclResult_t ret = ncclSuccess;
  int totalnDev;
  int *gpuFlags = NULL;
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  int oldDev = 0;

  NVTX3_RANGE(NcclNvtxParamsCommInitAll);

  // 初始化 CUDA 库
  (void)ncclCudaLibraryInit();

  CUDACHECK(cudaGetDevice(&oldDev));
  NCCLCHECKGOTO(PtrCheck(comms, "CommInitAll", "comms"), ret, fail);

  if (ndev < 0) {
    WARN("Invalid device count requested : %d", ndev);
    ret = ncclInvalidArgument;
    goto fail;
  }

  // 获取系统中 GPU 的数量
  CUDACHECKGOTO(cudaGetDeviceCount(&totalnDev), ret, fail);

  if (devlist) {
    NCCLCHECKGOTO(ncclCalloc(&gpuFlags, totalnDev), ret, fail);
    for (int i = 0; i < ndev; ++i) {
      /* 无效设备检查 */
      // 设备编号检查，必须在区间 0～totalnDev-1
      if (devlist[i] < 0 || devlist[i] >= totalnDev) {
        WARN("Invalid device %d (totalnDev=%d)", devlist[i], totalnDev);
        ret = ncclInvalidArgument;
        goto fail;
      }

      /* 重复设备检查 */
      // devlist 中指定了多个相同的 GPU 编号
      if (gpuFlags[devlist[i]] != 0) {
        ret = ncclInvalidUsage;
        goto fail;
      }

      gpuFlags[devlist[i]] = 1;
    }
    free(gpuFlags);
    gpuFlags = nullptr;
  }

  ncclUniqueId uniqueId;

  NCCLCHECKGOTO(ncclGetUniqueId(&uniqueId), ret, fail);
  NCCLCHECKGOTO(ncclGroupStartInternal(), ret, fail);
  for (int i = 0; i < ndev; i++) {
    // 忽略返回码..我们需要调用 ncclGroupEnd 来清理
    // 如果没有指定 devlist，从 0 开始按顺序使用 GPU 设备
    // 否则使用 devlist 配置的 GPU 设备
    int dev = devlist ? devlist[i] : i;
    CUDACHECKGOTO(cudaSetDevice(dev), ret, fail);
    ncclCommInitRankDev(comms + i, ndev, 1, &uniqueId, i, dev, &config, __func__);
  }
  NCCLCHECKGOTO(ncclGroupEndInternal(), ret, fail);

  NVTX3_RANGE_ADD_PAYLOAD(CommInitAll, NcclNvtxParamsCommInitAllSchema,
    NVTX3_PAYLOAD(comms[0]->commHash, ndev));

exit:
  (void)cudaSetDevice(oldDev);
  free(gpuFlags);
  return ret;
fail:
  goto exit;
}

// 设置异步错误
ncclResult_t ncclCommSetAsyncError(ncclComm_t comm, ncclResult_t nextState) {
  if (nextState < 0 || nextState >= ncclNumResults || comm == NULL) {
    WARN("ncclCommSetAsyncError: error comm %p sets state %d", comm, nextState);
    return ncclInvalidArgument;
  }

  __atomic_store_n(&comm->asyncResult, nextState, __ATOMIC_RELEASE);
  return ncclSuccess;
}

// ncclCommInitRankConfig - 使用配置初始化通信
NCCL_API(ncclResult_t, ncclCommInitRankConfig, ncclComm_t* comm, int nranks, ncclUniqueId commId, int myrank, ncclConfig_t *config);
ncclResult_t ncclCommInitRankConfig(ncclComm_t *newcomm, int nranks, ncclUniqueId commId, int myrank, ncclConfig_t *config) {
  int cudaDev;
  ncclResult_t ret = ncclSuccess;
  ncclConfig_t internalConfig = NCCL_CONFIG_INITIALIZER;
  ncclConfig_t *internalConfigPtr = NULL;

  NVTX3_RANGE(NcclNvtxParamsCommInitRankConfig);

  NCCLCHECK(ncclGroupStartInternal());

  (void)ncclCudaLibraryInit();
  CUDACHECK(cudaGetDevice(&cudaDev));

  if (config == NULL)
    internalConfigPtr = &internalConfig;
  else
    internalConfigPtr = config;
  NCCLCHECKGOTO(ncclCommInitRankDev(newcomm, nranks, 1, &commId, myrank, cudaDev, internalConfigPtr, __func__), ret, fail);

exit:
  ncclGroupErrCheck(ret);
  NCCLCHECK(ncclGroupEndInternal());
  if (newcomm && *newcomm) {
    if (!(*newcomm)->config.blocking) {
      (void) ncclCommGetAsyncError(*newcomm, &ret);
    }
    NVTX3_RANGE_ADD_PAYLOAD(CommInitRankConfig, NcclNvtxParamsCommInitRankSchema,
      NVTX3_PAYLOAD((*newcomm)->commHash, nranks, myrank, cudaDev));
  }
  return ret;
fail:
  if (newcomm && *newcomm && !(*newcomm)->config.blocking) (void) ncclCommSetAsyncError(*newcomm, ret);
  goto exit;
}

// ncclCommInitRankScalable - 可扩展的初始化（支持多个 root）
NCCL_API(ncclResult_t, ncclCommInitRankScalable, ncclComm_t* newcomm, int nranks, int myrank, int nId, ncclUniqueId* commId, ncclConfig_t* config);
ncclResult_t ncclCommInitRankScalable(ncclComm_t* newcomm, int nranks, int myrank, int nId, ncclUniqueId* commId, ncclConfig_t* config) {
  NVTX3_RANGE(NcclNvtxParamsCommInitRankScalable);

  int cudaDev;
  ncclResult_t ret = ncclSuccess;
  ncclConfig_t internalConfig = NCCL_CONFIG_INITIALIZER;
  ncclConfig_t *internalConfigPtr = NULL;
  NCCLCHECK(ncclGroupStartInternal());

  (void)ncclCudaLibraryInit();
  CUDACHECK(cudaGetDevice(&cudaDev));

  if (config == NULL)
    internalConfigPtr = &internalConfig;
  else
    internalConfigPtr = config;
  NCCLCHECKGOTO(ncclCommInitRankDev(newcomm, nranks, nId, commId, myrank, cudaDev, internalConfigPtr, __func__), ret, fail);

exit:
  ncclGroupErrCheck(ret);
  NCCLCHECK(ncclGroupEndInternal());
  if (newcomm && *newcomm) {
    if (!(*newcomm)->config.blocking) {
      (void) ncclCommGetAsyncError(*newcomm, &ret);
    }
    NVTX3_RANGE_ADD_PAYLOAD(CommInitRankScalable, NcclNvtxParamsCommInitRankSchema,
      NVTX3_PAYLOAD((*newcomm)->commHash, nranks, myrank, cudaDev));
  }
  return ret;
fail:
  if (newcomm && *newcomm && !(*newcomm)->config.blocking) (void) ncclCommSetAsyncError(*newcomm, ret);
  goto exit;
}
// 第五部分：通信器销毁、分割等 API

// ============================================================================
// commDestroySync - 同步销毁通信器
// ============================================================================
static ncclResult_t commDestroySync(struct ncclAsyncJob* job_) {
  struct ncclCommFinalizeAsyncJob* job = (struct ncclCommFinalizeAsyncJob*) job_;
  ncclComm_t comm = job->comm;
  ncclResult_t ret = ncclSuccess;

  CUDACHECKGOTO(cudaSetDevice(comm->cudaDev), ret, fail);

  TRACE(NCCL_INIT, "Destroying comm %p rank %d abortFlag %d asyncResult %d", comm, comm->rank, *comm->abortFlag, comm->asyncResult);

  if (comm->initState == ncclSuccess) {
    // 同步主机流
    if ((ret = ncclStrongStreamSynchronize(&comm->sharedRes->hostStream)) != ncclSuccess) {
      WARN("commDestroySync: comm %p rank %d sync hostStream error %d\n", comm, comm->rank, ret);
    }
    // 同步设备流
    if ((ret = ncclStrongStreamSynchronize(&comm->sharedRes->deviceStream)) != ncclSuccess) {
      WARN("commDestroySync: comm %p rank %d sync deviceStream error %d\n", comm, comm->rank, ret);
    }

    // 轮询事件回调
    NCCLCHECKGOTO(ncclCommPollEventCallbacks(comm, true), ret, fail);
    NCCLCHECKGOTO(ncclCommPollCallbacks(comm, false), ret, fail);

    // 持续轮询直到所有引用此 comm 的 graph 都销毁
    while (comm->localPersistentRefs != 0) {
      NCCLCHECKGOTO(ncclCommPollCallbacks(comm, /*waitSome=*/true), ret, fail);
    }

    // 处理遗留的 IPC 清理队列
    while (!ncclIntruQueueEmpty(&comm->legacyRegCleanupQueue)) {
      struct ncclCommCallback* cb = ncclIntruQueueDequeue(&comm->legacyRegCleanupQueue);
      if (cb->fn(comm, cb) != ncclSuccess) {
        WARN("Legacy IPC cleanup callback failed comm %p (rank = %d) cb %p", comm, comm->rank, cb);
      }
    }
  }

  // 停止 proxy
  if ((ret = ncclProxyStop(comm)) != ncclSuccess) {
    WARN("ncclProxyStop: comm %p (rank = %d) destroys proxy resource error %d", comm, comm->rank, ret);
  }

exit:
  return ret;
fail:
  goto exit;
}

// ============================================================================
// commCleanup - 清理通信器资源
// ============================================================================
static ncclResult_t commCleanup(ncclComm_t comm) {
  CUDACHECK(cudaSetDevice(comm->cudaDev));

  // 最终化 tuner
  if (comm->tuner != NULL) {
    NCCLCHECK(comm->tuner->finalize(comm->tunerContext));
    NCCLCHECK(ncclTunerPluginUnload(comm));
  }

  // 释放 comm 资源
  NCCLCHECK(commFree(comm));
  return ncclSuccess;
}

// ============================================================================
// ncclCommFinalize - 完成通信器（显式完成初始化）
// 用户可以显式调用此函数来完成初始化，但通常不需要
// ============================================================================
NCCL_API(ncclResult_t, ncclCommFinalize, ncclComm_t comm);
ncclResult_t ncclCommFinalize(ncclComm_t comm) {
  NVTX3_RANGE(NcclNvtxParamsCommFinalize);

  ncclResult_t ret = ncclSuccess;
  struct ncclCommFinalizeAsyncJob *job = NULL;

  NCCLCHECK(ncclGroupStartInternal());
  if (comm == NULL) goto exit;

  /* 在 finalize 之前等待 comm 就绪 */
  NCCLCHECKGOTO(ncclCommEnsureReady(comm), ret, fail);

  /* 防止重复 finalize */
  if (comm->finalizeCalled) {
    ret = ncclInvalidArgument;
    goto fail;
  }

  comm->finalizeCalled = true;

  /* 启动异步线程来 finalize comm */
  NCCLCHECKGOTO(ncclCalloc(&job, 1), ret, fail);
  job->comm = comm;
  NCCLCHECKGOTO(ncclAsyncLaunch((struct ncclAsyncJob*)job, commDestroySync, NULL, free, comm), ret, fail);

exit:
  ncclGroupErrCheck(ret);
  NCCLCHECK(ncclGroupEndInternal());
  if (comm) {
    if (!comm->config.blocking) {
      NCCLCHECK(ncclCommGetAsyncError(comm, &ret));
    }
    NVTX3_RANGE_ADD_PAYLOAD(CommFinalize, NcclNvtxParamsCommFinalizeSchema,
      NVTX3_PAYLOAD(comm->commHash));
  }
  return ret;
fail:
  if (comm && !comm->config.blocking) (void) ncclCommSetAsyncError(comm, ret);
  goto exit;
}

// ============================================================================
// commReclaim - 回收通信器资源
// 这是最后一个调用 ncclCommDestroy/Abort 的 rank 执行的清理工作
// ============================================================================
static ncclResult_t commReclaim(struct ncclAsyncJob* job_) {
  struct ncclCommFinalizeAsyncJob* job = (struct ncclCommFinalizeAsyncJob*) job_;
  ncclComm_t comm = job->comm;
  ncclResult_t ret = ncclSuccess;

  if (comm->intraComm0 != NULL) {
    int curRankCnt;
    int curRank; /* Debug info */
    int intraRanks = comm->intraRanks;
    ncclComm_t intracomm0 = comm->intraComm0;
    int *finalizeRankCnt = &intracomm0->finalizeRankCnt;

    assert(intracomm0 != NULL && finalizeRankCnt != NULL);
    curRankCnt = __atomic_add_fetch(finalizeRankCnt, 1, __ATOMIC_ACQ_REL);

    if (curRankCnt == intraRanks) {
      // 这是最后一次调用 ncclCommDestroy/Abort
      // 我们需要确保进程中的所有 comm 都已 finalized，然后才能释放本地资源
      ncclComm_t curIntraComm;
      ncclComm_t nextIntraComm = intracomm0;

      /* 确保所有 comm 已 finalized */
      while (nextIntraComm) {
        curIntraComm = nextIntraComm;
        curRank = curIntraComm->rank;
        nextIntraComm = nextIntraComm->intraNext;

        if (curIntraComm->finalizeCalled == false) {
          struct ncclCommFinalizeAsyncJob job;
          job.comm = curIntraComm;
          /* 每个 comm 都 abort，commDestroySync 不应被阻塞 */
          if ((ret = commDestroySync((struct ncclAsyncJob*) &job)) != ncclSuccess)
            WARN("commReclaim: comm %p (rank = %d) in commDestroySync, error %d", curIntraComm, curRank, ret);
        }
      }

      /* 释放本地资源 */
      nextIntraComm = intracomm0;
      while (nextIntraComm) {
        curIntraComm = nextIntraComm;
        curRank = curIntraComm->rank;
        nextIntraComm = nextIntraComm->intraNext;

        if ((ret = commCleanup(curIntraComm)) != ncclSuccess) {
          WARN("commReclaim: cleanup comm %p rank %d failed in destroy/abort, error %d", curIntraComm, curRank, ret);
        }
      }
    }
  }

  return ncclSuccess;
}

// ============================================================================
// ncclCommDestroy - 销毁通信器
// 用户调用此 API 来销毁通信器并释放资源
// ============================================================================
NCCL_API(ncclResult_t, ncclCommDestroy, ncclComm_t comm);
ncclResult_t ncclCommDestroy(ncclComm_t comm) {
  if (comm == NULL) {
    NCCL_NVTX3_FUNC_RANGE;
    return ncclSuccess;
  }

  int rank = comm->rank, nranks = comm->nRanks, cudaDev = comm->cudaDev;
  struct ncclCommFinalizeAsyncJob *job = NULL;
  ncclResult_t res = ncclSuccess;

  NVTX3_FUNC_WITH_PARAMS(CommDestroy, NcclNvtxParamsCommInitRank,
    NVTX3_PAYLOAD(comm->commHash, nranks, rank, cudaDev));

  TRACE(NCCL_INIT, "comm %p rank %d nRanks %d cudaDev %d busId %lx", comm, rank, nranks, cudaDev, comm->busId);
  NCCLCHECK(ncclGroupStartInternal());

  // 防止重复释放 comm 结构（用户错误）
  if (comm->rank == -1 || comm->nRanks == -1 || comm->cudaDev == -1 || comm->busId == -1) {
    WARN("comm %p has already been destroyed", comm);
    return ncclInvalidArgument;
  }

  comm->destroyFlag = 1;

  /* init 线程必须在销毁 comm 之前被 join */
  NCCLCHECK(ncclCommEnsureReady(comm));

  NCCLCHECKGOTO(ncclCalloc(&job, 1), res, fail);
  job->comm = comm;
  NCCLCHECKGOTO(ncclAsyncLaunch((struct ncclAsyncJob*)job, commReclaim, NULL, free, comm), res, fail);

exit:
  ncclGroupErrCheck(res);
  NCCLCHECK(ncclGroupEndInternal());
  return res;
fail:
  goto exit;
}

// ============================================================================
// setCommAbortFlags - 设置 abort 标志
// ============================================================================
static ncclResult_t setCommAbortFlags(ncclComm_t comm, int value) {
  // 设置 abort 标志
  if (comm->childAbortFlag != nullptr) {
    __atomic_store_n(comm->childAbortFlag, value, __ATOMIC_RELEASE);
    __atomic_store_n(comm->childAbortFlagDev, value, __ATOMIC_RELEASE);
  }
  __atomic_store_n(comm->abortFlag, value, __ATOMIC_RELEASE);
  __atomic_store_n(comm->abortFlagDev, value, __ATOMIC_RELEASE);
  return ncclSuccess;
}

// ============================================================================
// ncclCommAbort - 中止通信器
// 紧急中止通信，用于错误恢复
// ============================================================================
NCCL_API(ncclResult_t, ncclCommAbort, ncclComm_t comm);
ncclResult_t ncclCommAbort(ncclComm_t comm) {
  NVTX3_RANGE(NcclNvtxParamsCommAbort);

  if (comm == NULL) {
    return ncclSuccess;
  }

  INFO(NCCL_INIT, "comm %p rank %d nRanks %d cudaDev %d busId %lx - Abort START",
      comm->rank, comm->nRanks, comm->cudaDev, comm->busId);

  NCCLCHECK(ncclGroupStartInternal());

  // 要求可能仍在设备上运行的任何操作退出
  NCCLCHECK(setCommAbortFlags(comm, 1));
  comm->destroyFlag = 1;

  /* 在销毁 comm 之前，init 线程必须被 join
   * 这里我们应该忽略 init 错误 */
  (void)ncclCommEnsureReady(comm);

  // 一旦 comm 就绪，我们可以访问 ranks 等
  int rank = comm->rank, nranks = comm->nRanks, cudaDev = comm->cudaDev;
  struct ncclCommFinalizeAsyncJob *job = NULL;
  ncclResult_t res = ncclSuccess;

  NVTX3_RANGE_ADD_PAYLOAD(CommAbort, NcclNvtxParamsCommInitRankSchema,
    NVTX3_PAYLOAD(comm->commHash, nranks, rank, cudaDev));

  TRACE(NCCL_INIT, "comm %p rank %d nRanks %d cudaDev %d busId %lx", comm, rank, nranks, cudaDev, comm->busId);

  NCCLCHECKGOTO(ncclCalloc(&job, 1), res, fail);
  job->comm = comm;
  NCCLCHECKGOTO(ncclAsyncLaunch((struct ncclAsyncJob*)job, commReclaim, NULL, free, comm), res, fail);

exit:
  ncclGroupErrCheck(res);
  NCCLCHECK(ncclGroupEndInternal());
  return res;
fail:
  goto exit;
}

// ============================================================================
// 子 comm 清理作业
// ============================================================================
static void childCommCleanupJob(void* job) {
  struct ncclCommInitRankAsyncJob* initJob = (struct ncclCommInitRankAsyncJob*)job;
  if (initJob->excludeRanksList) free(initJob->excludeRanksList);
  free(job);
}

// ============================================================================
// ncclCommInitChildComm - 初始化子通信器（用于 split 和 shrink）
// ============================================================================
static ncclResult_t ncclCommInitChildComm(ncclComm_t comm, ncclComm_t* newcomm, bool isShrink, int flags, int color, int key, int* excludeRanksList, int excludeRanksCount,
                                          ncclConfig_t* config, const char* caller) {
  struct ncclCommInitRankAsyncJob *job = NULL;
  struct ncclComm* childComm = NCCL_COMM_NULL;
  ncclResult_t res = ncclSuccess;

  int oldDev;
  CUDACHECK(cudaGetDevice(&oldDev));
  NCCLCHECKGOTO(CommCheck(comm, caller, "comm"), res, exit);
  NCCLCHECKGOTO(PtrCheck(newcomm, caller, "newcomm"), res, exit);

  if (isShrink) {
    // Shrink 模式检查
    NCCLCHECKGOTO(PtrCheck(excludeRanksList, caller, "excludeRanksList"), res, exit);
    NCCLCHECKGOTO(excludeRanksCount > 0 ? ncclSuccess : ncclInvalidArgument, res, exit);
    // excludeRanksList 可能未排序，需要排序
    qsort(excludeRanksList, excludeRanksCount, sizeof(int), compareInts);
    // excludeRanksList 中的 ranks 不应调用此函数
    NCCLCHECKGOTO(bsearch(&comm->rank, excludeRanksList, excludeRanksCount, sizeof(int), compareInts) ? ncclInvalidArgument : ncclSuccess, res, exit);
  }

  NCCLCHECKGOTO(ncclCommEnsureReady(comm), res, exit);
  CUDACHECKGOTO(cudaSetDevice(comm->cudaDev), res, exit);

  /* *newcomm 在 comm split 完全完成之前应该是 NCCL_COMM_NULL */
  *newcomm = NCCL_COMM_NULL;

  if (!isShrink && color == NCCL_SPLIT_NOCOLOR) {
    INFO(NCCL_INIT, "Rank %d has color with NCCL_SPLIT_NOCOLOR, not creating a new communicator", comm->rank);
  } else {
    NCCLCHECKGOTO(ncclCalloc(&childComm, 1), res, fail);
    childComm->startMagic = childComm->endMagic = NCCL_MAGIC;

    // 设置 shareResource 字段，这在整个初始化中使用，每次都必须重置
    // 如果我们 shrink，只有在默认模式下 shrink 时才重用资源
    comm->shareResources = isShrink ? (!(flags & NCCL_SHRINK_ABORT) && comm->config.shrinkShare) : comm->config.splitShare;

    if (comm->shareResources) {
      // 共享父 comm 的资源
      childComm->abortFlag = comm->abortFlag;
      childComm->abortFlagDev = comm->abortFlagDev;
      childComm->abortFlagRefCount = comm->abortFlagRefCount;
      comm->childAbortFlag = NULL;
      ncclAtomicRefCountIncrement(comm->abortFlagRefCount);
    } else {
      // 创建新的 abort 标志
      NCCLCHECKGOTO(ncclCalloc(&childComm->abortFlag, 1), res, fail);
      NCCLCHECKGOTO(ncclCudaHostCalloc(&childComm->abortFlagDev, 1), res, fail);
      NCCLCHECKGOTO(ncclCalloc(&childComm->abortFlagRefCount, 1), res, fail);
      /* 临时用于在子 comm 初始化期间 abort 所有操作 */
      comm->childAbortFlag = childComm->abortFlag;
      comm->childAbortFlagDev = childComm->abortFlagDev;
      *childComm->abortFlagRefCount = 1;
    }

    if (config == NULL) {
      NCCLCHECKGOTO(copyCommConfig(childComm, comm), res, fail);
    } else {
      NCCLCHECKGOTO(parseCommConfig(childComm, config), res, fail);
    }

    /* 从 ncclInternalError 开始，如果初始化成功将改为 ncclSuccess */
    childComm->initState = ncclInternalError;
  }

  NCCLCHECKGOTO(ncclCalloc(&job, 1), res, fail);
  //指向childComm
  job->comm = childComm;
  job->newcomm = newcomm;
  //指向原始的主comm
  job->parent = comm;
  job->color = color;
  job->key = key;

  if (excludeRanksList) {
    // 需要拷贝排除的 ranks 列表，因为 job 是异步的
    job->excludeRanksCount = excludeRanksCount;
    NCCLCHECKGOTO(ncclCalloc(&job->excludeRanksList, excludeRanksCount), res, fail);
    memcpy(job->excludeRanksList, excludeRanksList, excludeRanksCount * sizeof(int));
  } else {
    // 每个 split 必须导致唯一的 comm，所以递增 splitCount
    job->splitCount = ++comm->splitCount;
    job->excludeRanksList = NULL;
  }

  job->cudaDev = comm->cudaDev;
  snprintf(job->funcName, NCCL_COMMINIT_FUNCNAME_LEN, "%s", caller);
  NCCLCHECKGOTO(ncclAsyncLaunch((struct ncclAsyncJob*)job, ncclCommInitRankFunc, /*undo=*/NULL, /*destructor=*/childCommCleanupJob, comm), res, fail);

exit:
  (void)cudaSetDevice(oldDev);
  return res;

fail:
  if (childComm) {
    if (!comm->shareResources) {
      if (childComm->abortFlag) 
        free(childComm->abortFlag);
      if (childComm->abortFlagDev) 
        ncclCudaHostFree(childComm->abortFlagDev);
      if (childComm->abortFlagRefCount) 
        free(childComm->abortFlagRefCount);
    }
    free(childComm);
  }
  
  if (newcomm)
    *newcomm = NULL;
  
  goto exit;
}

// ============================================================================
// ncclCommShrink - 缩减通信器规模
// 从通信器中排除某些 ranks，创建一个新的更小的通信器
// ============================================================================
NCCL_API(ncclResult_t, ncclCommShrink, ncclComm_t comm, int* excludeRanksList, int excludeRanksCount, ncclComm_t* newcomm, ncclConfig_t* config, int shrinkFlags);
ncclResult_t ncclCommShrink(ncclComm_t comm, int* excludeRanksList, int excludeRanksCount, ncclComm_t *newcomm, ncclConfig_t *config, int shrinkFlags) {
  NVTX3_RANGE(NcclNvtxParamsCommShrink)
  ncclResult_t res = ncclSuccess;
  NCCLCHECK(ncclGroupStartInternal());

  // 通过设置 abort 标志并等待内核完成来处理错误模式
  if (shrinkFlags & NCCL_SHRINK_ABORT) {
    NCCLCHECKGOTO(setCommAbortFlags(comm, 1), res, exit);
    NCCLCHECKGOTO(ncclStrongStreamSynchronize(&comm->sharedRes->deviceStream), res, exit);
    NCCLCHECKGOTO(setCommAbortFlags(comm, 0), res, exit);
  }

  NCCLCHECKGOTO(ncclCommInitChildComm(comm, newcomm, /*isShrink=*/true, shrinkFlags, /*color=*/0, /*key=*/comm->rank, excludeRanksList, excludeRanksCount, config, __func__), res, exit);

  if (*newcomm) NVTX3_RANGE_ADD_PAYLOAD(CommShrink, NcclNvtxParamsCommShrinkSchema, NVTX3_PAYLOAD(comm->commHash, comm->nRanks, comm->rank, comm->cudaDev, excludeRanksCount));

exit:
  (void)ncclGroupErrCheck(res);
  NCCLCHECK(ncclGroupEndInternal());
  return res;
}

// ============================================================================
// ncclCommSplit - 分割通信器
// 根据颜色和键值将通信器分割成多个子通信器
// color: 相同 color 的 ranks 分到同一组
// key: 在同一 color 内，按 key 排序确定新 rank号，内部新 rank号还是从0开始编号
//split后原始通信器 comm 完全不变
//    - comm->rank 保持原值
//    - comm->nRanks 保持原值
//newcomm包哈新的rank成员，rank号还是从0开始重新编号
// ============================================================================
NCCL_API(ncclResult_t, ncclCommSplit, ncclComm_t comm, int color, int key, ncclComm_t *newcomm, ncclConfig_t *config);
ncclResult_t ncclCommSplit(ncclComm_t comm, int color, int key, ncclComm_t *newcomm, ncclConfig_t *config) {
  NVTX3_RANGE(NcclNvtxParamsCommSplit)

  ncclResult_t res = ncclSuccess;
  NCCLCHECK(ncclGroupStartInternal());
  NCCLCHECKGOTO(ncclCommInitChildComm(comm, newcomm, /*isShrink=*/false, /*shrink mode=*/NCCL_SHRINK_DEFAULT, color, key, NULL, 0, config, __func__), res, exit);

  if (*newcomm)
    NVTX3_RANGE_ADD_PAYLOAD(CommSplit, NcclNvtxParamsCommSplitSchema, NVTX3_PAYLOAD((*newcomm)->commHash, comm->commHash, comm->nRanks, comm->rank, comm->cudaDev, color, key));

exit:
  (void)ncclGroupErrCheck(res);
  NCCLCHECK(ncclGroupEndInternal());
  return res;
}

// ============================================================================
// 辅助 API 函数
// ============================================================================

// 获取错误字符串
NCCL_API(const char*, ncclGetErrorString, ncclResult_t code);
const char* ncclGetErrorString(ncclResult_t code) {
  switch (code) {
    case ncclSuccess                : return "no error";
    case ncclUnhandledCudaError     : return "unhandled cuda error (run with NCCL_DEBUG=INFO for details)";
    case ncclSystemError            : return "unhandled system error (run with NCCL_DEBUG=INFO for details)";
    case ncclInternalError          : return "internal error - please report this issue to the NCCL developers";
    case ncclInvalidArgument        : return "invalid argument (run with NCCL_DEBUG=WARN for details)";
    case ncclInvalidUsage           : return "invalid usage (run with NCCL_DEBUG=WARN for details)";
    case ncclRemoteError            : return "remote process exited or there was a network error";
    case ncclInProgress             : return "NCCL operation in progress";
    default                         : return "unknown result code";
  }
}

/* 返回发生的最后一个错误的可读消息
 * comm 当前未使用，可以设置为 NULL
 */
NCCL_API(const char*, ncclGetLastError, const ncclComm_t comm);
const char* ncclGetLastError(ncclComm_t comm) {
  return ncclLastError;
}

// 获取异步错误
NCCL_API(ncclResult_t, ncclCommGetAsyncError, ncclComm_t comm, ncclResult_t *asyncError);
ncclResult_t ncclCommGetAsyncError(ncclComm_t comm, ncclResult_t *asyncError) {
  NCCLCHECK(CommCheck(comm, "ncclGetAsyncError", "comm"));
  NCCLCHECK(PtrCheck(asyncError, "ncclGetAsyncError", "asyncError"));

  *asyncError = __atomic_load_n(&comm->asyncResult, __ATOMIC_ACQUIRE);
  if (*asyncError == ncclSuccess && comm->proxyState) *asyncError = __atomic_load_n(&comm->proxyState->asyncResult, __ATOMIC_ACQUIRE);

  /* 如果有链接的 group job，我们应该完成它 */
  if (*asyncError == ncclSuccess && comm->groupJob) {
    NCCLCHECK(ncclGroupJobComplete(comm->groupJob));
    comm->groupJob = NULL;
  }
  return ncclSuccess;
}

// 获取通信器中的 rank 数量
NCCL_API(ncclResult_t, ncclCommCount, const ncclComm_t comm, int* count);
ncclResult_t ncclCommCount(const ncclComm_t comm, int* count) {
  NCCL_NVTX3_FUNC_RANGE;

  NCCLCHECK(CommCheck(comm, "CommCount", "comm"));
  NCCLCHECK(PtrCheck(count, "CommCount", "count"));

  /* 在访问 comm 的属性之前，init 线程必须被 join */
  NCCLCHECK(ncclCommEnsureReady(comm));

  *count = comm->nRanks;
  return ncclSuccess;
}

// 获取通信器绑定的 CUDA 设备 ID
NCCL_API(ncclResult_t, ncclCommCuDevice, const ncclComm_t comm, int* devid);
ncclResult_t ncclCommCuDevice(const ncclComm_t comm, int* devid) {
  NCCL_NVTX3_FUNC_RANGE;

  NCCLCHECK(CommCheck(comm, "CommCuDevice", "comm"));
  NCCLCHECK(PtrCheck(devid, "CommCuDevice", "devid"));

  NCCLCHECK(ncclCommEnsureReady(comm));

  *devid = comm->cudaDev;
  return ncclSuccess;
}

// 获取通信器的用户 rank
NCCL_API(ncclResult_t, ncclCommUserRank, const ncclComm_t comm, int* rank);
ncclResult_t ncclCommUserRank(const ncclComm_t comm, int* rank) {
  NCCL_NVTX3_FUNC_RANGE;

  NCCLCHECK(CommCheck(comm, "CommUserRank", "comm"));
  NCCLCHECK(PtrCheck(rank, "CommUserRank", "rank"));

  NCCLCHECK(ncclCommEnsureReady(comm));

  *rank = comm->rank;
  return ncclSuccess;
}
