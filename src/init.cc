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
static ncclResult_t commAlloc(struct ncclComm* comm, struct ncclComm* parent, int ndev, int rank) {
  // 参数验证
  if (ndev < 1) {
    WARN("invalid device count (%d) requested", ndev);
    return ncclInvalidArgument;
  }
  if (rank >= ndev || rank < 0) {
    WARN("rank %d exceeds ndev=%d", rank, ndev);
    return ncclInvalidArgument;
  }

  // 初始化内存栈（用于管理 comm 的内存分配）
  ncclMemoryStackConstruct(&comm->memPermanent);  // 永久内存（comm 生命周期内一直有效）
  ncclMemoryStackConstruct(&comm->memScoped);     // 作用域内存（可以释放）
  comm->destructorHead = nullptr;

  // 设置基本信息
  comm->rank = rank;          // 当前 rank 的编号
  comm->nRanks = ndev;        // 总 rank 数量

  // 初始化网络插件
  // ncclNetInit() 按顺序尝试加载插件，一旦某个插件分配成功，循环立即 break
  // 设置 comm 中的 ncclNet 和 ncclCollNet，ncclCollNet 可能为 NULL
  NCCLCHECK(ncclNetInit(comm));

  // 记录使用的网络类型
  INFO(NCCL_INIT, "Using network %s", comm->ncclNet->name);

  // 如果是子通信且共享资源，检查网络类型是否一致
  if (parent && parent->shareResources) {
    if (parent->ncclNet != comm->ncclNet) {
      WARN("Split shares resources, but parent comm netName %s is different from child comm netName %s", parent->ncclNet->name, comm->ncclNet->name);
      return ncclInvalidUsage;
    }
  }

  // 立即创建 CUDA 对象以验证设备状态
  // 如果设备有问题（最常见的原因 #1），最好早点知道
  CUDACHECK(cudaGetDevice(&comm->cudaDev));    // 获取当前 CUDA 设备

  NCCLCHECK(ncclCudaContextTrack(&comm->context));  // 获取并跟踪 CUDA 上下文

  // 获取 GPU 设备的 bus ID（PCI 地址）
  NCCLCHECK(getBusId(comm->cudaDev, &comm->busId));

  // 获取 NVML 设备句柄
  nvmlDevice_t nvmlDev;
  char busId[NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE];
  // 将 busId 转换为字符串
  NCCLCHECK(int64ToBusId(comm->busId, busId));
  // 通过 bus ID 获取 NVML 设备句柄
  NCCLCHECK(ncclNvmlDeviceGetHandleByPciBusId(busId, &nvmlDev));
  NCCLCHECK(ncclNvmlDeviceGetIndex(nvmlDev, (unsigned int*)&comm->nvmlDev));

  // 获取 GPU 计算能力（如 70、75、80 等）
  comm->compCap = ncclCudaCompCap();
  TRACE(NCCL_INIT,"comm %p rank %d nranks %d cudaDev %d busId %lx compCap %d", comm, rank, ndev, comm->cudaDev, comm->busId, comm->compCap);

  // 其他配置
  comm->checkPointers = ncclParamCheckPointers() == 1 ? true : false;
  comm->dmaBufSupport = (dmaBufSupported(comm) == ncclSuccess) ? true : false;

  // 清零 CollNet 支持矩阵
  memset(comm->collNetSupportMatrix, 0, sizeof(comm->collNetSupportMatrix));

  // 构造内存池（用于内核计划）
  ncclMemoryPoolConstruct(&comm->memPool_ncclKernelPlan);
  ncclMemoryPoolConstruct(&comm->memPool_ncclProxyOp);

  // 初始化 group 任务链表头节点
  for (int i = 0; i < ncclGroupTaskTypeNum; i++) {
    comm->groupNext[i] = reinterpret_cast<struct ncclComm*>(0x1);
  }
  comm->preconnectNext = reinterpret_cast<struct ncclComm*>(0x1);

  // 验证位图数组大小是否足够
  static_assert(MAXCHANNELS <= sizeof(*comm->connectSend)*8, "comm->connectSend must have enough bits for all channels");
  static_assert(MAXCHANNELS <= sizeof(*comm->connectRecv)*8, "comm->connectRecv must have enough bits for all channels");

  // 分配连接位图数组（记录每个 rank 的哪些通道已连接）
  // 大小为 8 字节 * nranks
  NCCLCHECK(ncclCalloc(&comm->connectSend, comm->nRanks));
  NCCLCHECK(ncclCalloc(&comm->connectRecv, comm->nRanks));

  // 标记通道为未初始化状态
  for (int c=0; c < MAXCHANNELS; c++)
    comm->channels[c].id = -1;

  // 处理共享资源（用于 split 等场景）
  if (parent == NULL || !parent->shareResources) {
    // 创建新的共享资源
    struct ncclSharedResources* sharedRes = NULL;
    NCCLCHECK(ncclCalloc(&sharedRes, 1));
  
    /* 大部分属性稍后在 initTransportsRank() 中设置 */
    //记录是那个通信器创建了这个sharedRes
    sharedRes->owner = comm;           // 记录谁拥有这个 comm
    sharedRes->tpNRanks = comm->nRanks;  // 记录总的 ranks 数
    NCCLCHECK(ncclCalloc(&sharedRes->tpRankToLocalRank, comm->nRanks));

    // 创建 2 个 CUDA 流（用于内核启动）
    NCCLCHECK(ncclStrongStreamConstruct(&sharedRes->deviceStream));
    NCCLCHECK(ncclStrongStreamConstruct(&sharedRes->hostStream));

    // 创建 2 个 CUDA event
    CUDACHECK(cudaEventCreateWithFlags(&sharedRes->launchEvent, cudaEventDisableTiming));
    CUDACHECK(cudaEventCreateWithFlags(&sharedRes->scratchEvent, cudaEventDisableTiming));

    comm->sharedRes = sharedRes;
    sharedRes->refCount = 1;  // 引用计数设为 1
  } else {
    // 和 parent 共享相同的资源
    comm->sharedRes = parent->sharedRes;
    ncclAtomicRefCountIncrement(&parent->sharedRes->refCount);
  }

  // 初始化 topParentRanks 数组
  if (comm->topParentRanks == NULL) {
    NCCLCHECK(ncclCalloc(&comm->topParentRanks, comm->nRanks));
    // 记录 parent 的 rank 号
    for (int i = 0; i < comm->nRanks; ++i)
      comm->topParentRanks[i] = i;
  }

  // 初始化队列（用于回调、注册清理等）
  ncclIntruQueueMpscConstruct(&comm->callbackQueue);
  ncclIntruQueueConstruct(&comm->legacyRegCleanupQueue);
  ncclIntruQueueConstruct(&comm->ceInitTaskQueue);

  // 获取系统页大小
  comm->regCache.pageSize = sysconf(_SC_PAGESIZE);

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

  // 初始化事件回调队列
  ncclIntruQueueConstruct(&comm->eventCallbackQueue);

  return ncclSuccess;
}
// ============================================================================
// devCommSetup - 设备通信设置
// 在 GPU 设备上分配和初始化通信所需的数据结构
// ============================================================================
static ncclResult_t devCommSetup(ncclComm_t comm) {
  ncclResult_t ret = ncclSuccess;
  int nRanks = comm->nRanks;
  struct ncclKernelCommAndChannels tmpCommAndChans;   // 临时结构（主机端）
  struct ncclKernelCommAndChannels *devCommAndChans = NULL;  // 设备端指针
  struct ncclNvmlCCStatus ccStatus;
  bool ccEnable;
  cudaStream_t deviceStream;

  memset(&tmpCommAndChans, '\0', sizeof(tmpCommAndChans));

  // 获取设备流（用于异步内存分配和拷贝）
  NCCLCHECKGOTO(ncclStrongStreamAcquire(ncclCudaGraphNone(), &comm->sharedRes->deviceStream, /*concurrent=*/false, &deviceStream), ret, fail);

  // 在设备上分配通信结构
  NCCLCHECKGOTO(ncclCudaCallocAsync(&devCommAndChans, 1, deviceStream), ret, fail);
  ncclCommPushCudaFree(comm, devCommAndChans);

  // 在设备上分配 rank 到 local rank 的映射表
  NCCLCHECKGOTO(ncclCudaCallocAsync(&tmpCommAndChans.comm.rankToLocalRank, comm->nRanks, deviceStream), ret, fail);
  ncclCommPushCudaFree(comm, tmpCommAndChans.comm.rankToLocalRank);

  // 将主机端的 rankToLocalRank 拷贝到设备端
  NCCLCHECKGOTO(ncclCudaMemcpyAsync(tmpCommAndChans.comm.rankToLocalRank, comm->rankToLocalRank, comm->nRanks, deviceStream), ret, fail);

  // 设置设备端通信指针
  comm->devComm = &devCommAndChans->comm;

  // 填充基本通信信息
  tmpCommAndChans.comm.rank = comm->rank;
  tmpCommAndChans.comm.nRanks = nRanks;
  tmpCommAndChans.comm.node = comm->node;
  tmpCommAndChans.comm.nNodes = comm->nNodes;
  tmpCommAndChans.comm.abortFlag = comm->abortFlagDev;  // 设备端的 abort 标志
  tmpCommAndChans.comm.isAllNvlink = comm->isAllNvlink;

  // 设置每种协议的缓冲区大小
  for (int p=0; p < NCCL_NUM_PROTOCOLS; p++) {
    tmpCommAndChans.comm.buffSizes[p] = comm->buffSizes[p];
  }
  tmpCommAndChans.comm.p2pChunkSize = comm->p2pChunkSize;
  tmpCommAndChans.comm.channels = &devCommAndChans->channels[0];

  // 计算工作参数大小（用于内核启动）
  comm->workArgsBytes = std::min<size_t>(ncclParamWorkArgsBytes(), ncclMaxKernelArgsSize(comm->cudaArch));

  // 检查是否启用 CC（Compute Consolidation）功能
  memset(&ccStatus, 0, sizeof(ccStatus));
  ccEnable = (ncclSuccess == ncclNvmlGetCCStatus(&ccStatus)) && (ccStatus.CCEnabled || ccStatus.multiGpuProtectedPCIE || ccStatus.multiGpuNVLE);

  if (ccEnable) {
    // CC 启用时，不需要 workFifo
    comm->workFifoBytes = 0;
  } else {
    // 设置 workFifo 大小（用于内核和主机通信）
    comm->workFifoBytes = ncclParamWorkFifoBytes();
    // 检查是否是 2 的幂次方
    if (0 != (comm->workFifoBytes & (comm->workFifoBytes-1))) {
      WARN("NCCL_WORK_FIFO_BYTES=%d is being ignored because it is not a power of 2.", comm->workFifoBytes);
      comm->workFifoBytes = NCCL_WORK_FIFO_BYTES_DEFAULT;
    }
    // 限制最大值为 1GB
    comm->workFifoBytes = std::min(comm->workFifoBytes, 1u<<30);
  }

  if (comm->rank == 0) {
    INFO(NCCL_INIT, "CC %s, workFifoBytes %d", ccEnable ? "On" : "Off", comm->workFifoBytes);
  }

  // 分配 workFifo 缓冲区
  if (ncclGdrCopy != NULL && ncclParamGdrCopyFifoEnable() == 1) {
    // 使用 GDRCOPY 映射的 CUDA 内存
    NCCLCHECKGOTO(ncclGdrCudaCalloc(&comm->workFifoBuf, &comm->workFifoBufDev, comm->workFifoBytes, &comm->workFifoBufGdrHandle), ret, fail);
    ncclCommPushCudaGdrFree(comm, comm->workFifoBufGdrHandle);
  } else {
    // 使用 cudaHost 内存（锁页内存）
    comm->workFifoBufGdrHandle = nullptr;
    NCCLCHECKGOTO(ncclCudaHostCalloc(&comm->workFifoBuf, comm->workFifoBytes), ret, fail);
    ncclCommPushCudaHostFree(comm, comm->workFifoBuf);
    comm->workFifoBufDev = comm->workFifoBuf;
  }

  // 初始化 workFifo 生产者和消费者索引
  comm->workFifoProduced = 0;
  comm->workFifoProducedLastRecorded = 0;
  comm->workFifoConsumed = 0;

  // 为内核分配性能计数器
  NCCLCHECKGOTO(ncclCudaHostCalloc(&comm->profiler.workStarted, MAXCHANNELS), ret, fail);
  NCCLCHECKGOTO(ncclCudaHostCalloc(&comm->profiler.workCompleted, MAXCHANNELS), ret, fail);
  tmpCommAndChans.comm.workStarted = comm->profiler.workStarted;
  tmpCommAndChans.comm.workCompleted = comm->profiler.workCompleted;
  ncclCommPushCudaHostFree(comm, comm->profiler.workStarted);
  ncclCommPushCudaHostFree(comm, comm->profiler.workCompleted);

  // 如果使用 CollNet，需要拷贝 denseToUserRank 映射
  if (comm->collNetDenseToUserRank != nullptr) {
    NCCLCHECKGOTO(ncclCudaCallocAsync(&tmpCommAndChans.comm.collNetDenseToUserRank, nRanks, deviceStream), ret, fail);
    ncclCommPushCudaFree(comm, tmpCommAndChans.comm.collNetDenseToUserRank);
    NCCLCHECKGOTO(ncclCudaMemcpyAsync(tmpCommAndChans.comm.collNetDenseToUserRank, comm->collNetDenseToUserRank, nRanks, deviceStream), ret, fail);
  }

  // 为每个通道设置设备端指针
  for (int c=0; c < MAXCHANNELS; c++) {
    tmpCommAndChans.channels[c].peers = comm->channels[c].devPeers;
    tmpCommAndChans.channels[c].ring = comm->channels[c].ring;
    tmpCommAndChans.channels[c].ring.userRanks = comm->channels[c].devRingUserRanks;
    tmpCommAndChans.channels[c].tree = comm->channels[c].tree;
    tmpCommAndChans.channels[c].collnetChain = comm->channels[c].collnetChain;
    tmpCommAndChans.channels[c].collnetDirect = comm->channels[c].collnetDirect;
    tmpCommAndChans.channels[c].nvls = comm->channels[c].nvls;

    // 拷贝 ring user ranks 到设备
    if (comm->channels[c].ring.userRanks != nullptr) {
      NCCLCHECKGOTO(ncclCudaMemcpyAsync(tmpCommAndChans.channels[c].ring.userRanks, comm->channels[c].ring.userRanks, nRanks, deviceStream), ret, fail);
    }
  }

  // 将临时结构拷贝到设备
  NCCLCHECKGOTO(ncclCudaMemcpyAsync(devCommAndChans, &tmpCommAndChans, 1, deviceStream), ret, fail);

exit:
  NCCLCHECK(ncclStrongStreamRelease(ncclCudaGraphNone(), &comm->sharedRes->deviceStream, /*concurrent=*/false));
  NCCLCHECK(ncclStrongStreamSynchronize(&comm->sharedRes->deviceStream));
  return ret;
fail:
  goto exit;
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
// fillInfo - 填充 peer 信息
// 收集当前 rank 的信息，用于后续与其他 rank 交换
// ============================================================================
static ncclResult_t fillInfo(struct ncclComm* comm, struct ncclPeerInfo* info, uint64_t commHash) {
  cudaDeviceProp prop;

  // 基本信息
  info->rank = comm->rank;           // 全局 rank 号
  info->cudaDev = comm->cudaDev;     // GPU 设备号
  info->nvmlDev = comm->nvmlDev;
  NCCLCHECK(ncclGetVersion(&info->version));

  // 主机名和 PID 的 hash 值（用于识别是否在同一节点/进程）
  info->hostHash = getHostHash() + commHash;
  info->pidHash = getPidHash() + commHash;

  // cuMem 支持情况
  info->cuMemSupport = ncclCuMemEnable();

  // 计算全局显存大小
  CUDACHECK(cudaGetDeviceProperties(&prop, comm->cudaDev));
  info->totalGlobalMem = ROUNDUP(prop.totalGlobalMem, (1L << 32));

  // 获取 /dev/shm 的设备号
  // 用于判断在容器环境中是否可以使用共享内存进行进程间通信
  struct stat statbuf;
  SYSCHECK(stat("/dev/shm", &statbuf), "stat");
  info->shmDev = statbuf.st_dev;

  // GPU 的 busId
  info->busId = comm->busId;

  // 判断是否支持 GDR（GPUDirect RDMA）
  NCCLCHECK(ncclGpuGdrSupport(comm, &info->gdrSupport));

  info->comm = comm;
  info->cudaCompCap = comm->minCompCap = comm->maxCompCap = comm->compCap;

  // MNNVL（Multi-Node NVLink）支持信息
  {
    // 获取 Fabric UUID 和分区信息
    char busId[NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE];
    nvmlDevice_t nvmlDev;
    NCCLCHECK(int64ToBusId(info->busId, busId));
    NCCLCHECK(ncclNvmlDeviceGetHandleByPciBusId(busId, &nvmlDev));
    info->fabricInfo.state = NVML_GPU_FABRIC_STATE_NOT_SUPPORTED;
    (void) ncclNvmlDeviceGetGpuFabricInfoV(nvmlDev, &info->fabricInfo);

    if (info->fabricInfo.state != NVML_GPU_FABRIC_STATE_NOT_SUPPORTED) {
      unsigned long uuid0 = 0;
      unsigned long uuid1 = 0;

      // 如果通过环境变量设置了 UUID，使用环境变量的值
      if (ncclParamMNNVLUUID() != -1) {
        unsigned long temp_uuid0 = (unsigned long)ncclParamMNNVLUUID();
        unsigned long temp_uuid1 = (unsigned long)ncclParamMNNVLUUID();
        memcpy(info->fabricInfo.clusterUuid, &temp_uuid0, sizeof(temp_uuid0));
        memcpy(info->fabricInfo.clusterUuid + sizeof(temp_uuid0), &temp_uuid1, sizeof(temp_uuid1));
      }

      memcpy(&uuid0, info->fabricInfo.clusterUuid, sizeof(uuid0));
      memcpy(&uuid1, info->fabricInfo.clusterUuid + sizeof(uuid0), sizeof(uuid1));

      // 设置 cliqueId
      if (ncclParamMNNVLCliqueId() == -2) {
        // 使用机架序列号的 hash 作为 cliqueId
        nvmlPlatformInfo_t platformInfo = { 0 };
        NCCLCHECK(ncclNvmlDeviceGetPlatformInfo(nvmlDev, &platformInfo));
        INFO(NCCL_INIT, "MNNVL rack serial %s slot %d tray %d hostId %d peerType %d moduleId %d",
             platformInfo.chassisSerialNumber, platformInfo.slotNumber, platformInfo.trayIndex,
             platformInfo.hostId, platformInfo.peerType, platformInfo.moduleId);
        info->fabricInfo.cliqueId = getHash(platformInfo.chassisSerialNumber, sizeof(platformInfo.chassisSerialNumber));
      } else if (ncclParamMNNVLCliqueId() != -1) {
        info->fabricInfo.cliqueId = ncclParamMNNVLCliqueId();
      }

      INFO(NCCL_INIT, "MNNVL busId 0x%lx fabric UUID %lx.%lx cliqueId 0x%x state %d healthMask 0x%x",
           info->busId, uuid0, uuid1,
           info->fabricInfo.cliqueId, info->fabricInfo.state, info->fabricInfo.healthMask);
    }
  }

  return ncclSuccess;
}

// ============================================================================
// setupChannel - 设置通信通道
// 初始化环形算法的通道连接
// ============================================================================
static ncclResult_t setupChannel(struct ncclComm* comm, int channelId, int rank, int nranks, int* ringRanks) {
  TRACE(NCCL_INIT, "rank %d nranks %d", rank, nranks);

  // 初始化通道
  NCCLCHECK(initChannel(comm, channelId));

  struct ncclRing* ring = &comm->channels[channelId].ring;

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

  // 为什么要重新排列环顺序？
  // 环形算法需要每个 Rank 知道：
  // - 左邻居（接收数据的来源）
  // - 右邻居（发送数据的目标）
  // 通过重新排列，使得当前 rank 在索引 0，方便计算邻居
  for (int i = 0; i < nranks; i++) {
    ring->userRanks[i] = ringRanks[(i + ixRank) % nranks];
  }

  return ncclSuccess;
}

// 默认缓冲区大小定义
#define DEFAULT_LL_BUFFSIZE (NCCL_LL_LINES_PER_THREAD*NCCL_LL_MAX_NTHREADS*NCCL_STEPS*sizeof(union ncclLLFifoLine))
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
// ============================================================================
static ncclResult_t initTransportsRank(struct ncclComm* comm, struct ncclComm* parent, uint64_t timers[TIMERS_INIT_COUNT]) {
  // 我们使用 2 次 AllGather：
  // 1. { peerInfo, comm, compCap } - 交换基本信息
  // 2. { nChannels, graphInfo, topoRanks } - 交换拓扑和图信息
  ncclResult_t ret = ncclSuccess;
  int rank = comm->rank;
  int nranks = comm->nRanks;
  int nNodes = 1;  // 统计有多少个不同的节点
  cpu_set_t affinitySave;

  // 图结构指针（用于不同算法）
  struct ncclTopoGraph* ringGraph = &comm->graphs[NCCL_ALGO_RING];
  struct ncclTopoGraph* treeGraph = &comm->graphs[NCCL_ALGO_TREE];
  struct ncclTopoGraph* collNetChainGraph = &comm->graphs[NCCL_ALGO_COLLNET_CHAIN];
  struct ncclTopoGraph* collNetDirectGraph = &comm->graphs[NCCL_ALGO_COLLNET_DIRECT];
  struct ncclTopoGraph* nvlsGraph = &comm->graphs[NCCL_ALGO_NVLS];
  struct ncclTopoGraph* graphs[NCCL_NUM_ALGORITHMS] = { treeGraph, ringGraph, collNetDirectGraph, collNetChainGraph, nvlsGraph, nvlsGraph, treeGraph };

  // 图信息结构（用于 AllGather）
  struct graphInfo {
    int pattern;        // 通信模式（Ring、Tree 等）
    int nChannels;      // 通道数量
    int sameChannels;   // 是否使用相同的通道
    float bwIntra;      // 节点内带宽
    float bwInter;      // 节点间带宽
    int typeIntra;      // 节点内传输类型
    int typeInter;      // 节点间传输类型
    int crossNic;       // 是否跨网卡
  };

  // AllGather 数据结构
  struct allGatherInfo {
    struct graphInfo graphInfo[NCCL_NUM_ALGORITHMS];
    struct ncclTopoRanks topoRanks;
    int cpuArch;
    int cpuVendor;
    int localRanks;
  };

  int nChannelsOrig;
  struct allGatherInfo *allGather3Data = NULL;
  struct ncclTopoRanks** allTopoRanks = NULL;
  int *nodesFirstRank = NULL, *nodesTreePatterns = NULL;
  int *rings = NULL;
  int* nvbPeers = NULL;
  struct ncclProxyConnector proxyConn;
  int* pxnPeers = NULL;
  int *topParentLocalRanks = NULL;
  int p2pLevel = -1;

  timers[TIMER_INIT_ALLGATHER] = clockNano();

  // ========== AllGather 1 - 交换基本信息 ==========
  // 分配空间，存储所有 rank 的 peerInfo 信息
  // +1 是为了额外的 rank（用于 CollNet root）
  NCCLCHECKGOTO(ncclCalloc(&comm->peerInfo, nranks + 1), ret, fail);

  // 填充本 rank 的 peerInfo 信息
  NCCLCHECKGOTO(fillInfo(comm, comm->peerInfo + rank, comm->commHash), ret, fail);

  // 调用 Bootstrap 同步所有 rank 的 peerInfo 信息
  // 包含重要的 cudaDev 和 rank 号
  NCCLCHECKGOTO(bootstrapAllGather(comm->bootstrap, comm->peerInfo, sizeof(struct ncclPeerInfo)), ret, fail);

  // 同步完成，设置 peerInfo 有效标志
  __atomic_store_n(&comm->peerInfoValid, true, __ATOMIC_RELEASE);

  // 先假设支持 cuMem
  comm->cuMemSupport = 1;

  // 同一个通信组内的所有 rank 都会校验参数有效性
  for (int i = 0; i < nranks; i++) {
    // 要求 NCCL 版本号必须一致
    if (comm->peerInfo[i].version != comm->peerInfo[rank].version) {
      WARN("Mismatched NCCL version detected : rank %d version %d rank %d version %d",
           i, comm->peerInfo[i].version, rank, comm->peerInfo[rank].version);
      ret = ncclInvalidUsage;
      goto fail;
    }

    // 检查节点数量，hostHash 不同表示不在同一物理机上
    if (comm->peerInfo[i].hostHash != comm->peerInfo[rank].hostHash)
        nNodes++;

    // 只要有一个 rank 不支持 cuMem，则不使用 cuMem
    if (!comm->peerInfo[i].cuMemSupport)
        comm->cuMemSupport = 0;

    // 检查是否有重复的 GPU
    // 同一个通信域内，不同 Rank 必须绑定不同物理 GPU
    // 例如：Rank 0 和 Rank 1 都绑定 GPU 0 是非法的
    if ((i != rank) &&
        (comm->peerInfo[i].hostHash == comm->peerInfo[rank].hostHash) &&
        (comm->peerInfo[i].busId == comm->peerInfo[rank].busId)) {
      WARN("Duplicate GPU detected : rank %d and rank %d both on CUDA device %lx", rank, i, comm->peerInfo[rank].busId);
      ret = ncclInvalidUsage;
      goto fail;
    }
  }

  // AllGather 1 结束
  timers[TIMER_INIT_ALLGATHER] = clockNano() - timers[TIMER_INIT_ALLGATHER];

  // 检查 MNNVL 支持
  NCCLCHECKGOTO(ncclGetUserP2pLevel(&p2pLevel), ret, fail);
  if ((nNodes > 1 && ncclParamMNNVLEnable() != 0 && p2pLevel != 0) || ncclParamMNNVLEnable() == 1) {
    NCCLCHECKGOTO(ncclMnnvlCheck(comm), ret, fail);
  }

  do {
    // 计算进程内 ranks
    int intraProcRank0 = -1, intraProcRank = -1, intraProcRanks = 0;

    // 统计组内 GPU 最小和最大计算能力
    for (int i = 0; i < nranks; i++)
        comm->minCompCap = std::min(comm->minCompCap, comm->peerInfo[i].cudaCompCap);
    for (int i = 0; i < nranks; i++)
        comm->maxCompCap = std::max(comm->maxCompCap, comm->peerInfo[i].cudaCompCap);

    comm->nvlsRegSupport = 1;

    // 检查同一物理节点内的同一进程或多线程中启动了多个 comm
    for (int i = 0; i < nranks; i++) {
      if ((comm->peerInfo[i].hostHash == comm->peerInfo[rank].hostHash)
          && (comm->peerInfo[i].pidHash == comm->peerInfo[rank].pidHash)) {
        // Rank 在同一进程中
        if (intraProcRanks == 0)
            intraProcRank0 = i;  // 记录同一进程的第一个 rank 号

        if (i == rank)
            intraProcRank = intraProcRanks;

        intraProcRanks++;  // 同一进程内的 Rank 数量++

        // 构建进程内 comm 链表
        if (intraProcRank0 == rank && rank != i) {
          comm->peerInfo[i].comm->intraNext = comm->intraNext;
          comm->intraNext = comm->peerInfo[i].comm;
        }
      }

      // 检查是否支持 NVLS 注册
      // 同一进程内启动了多个 comm，不支持 NVLinkSharp 注册
      if (comm->nvlsRegSupport) {
        for (int j = i + 1; j < nranks; j++) {
          if (comm->peerInfo[i].hostHash == comm->peerInfo[j].hostHash &&
              comm->peerInfo[i].pidHash == comm->peerInfo[j].pidHash) {
            comm->nvlsRegSupport = 0;
            break;
          }
        }
      }
    }

    // MNNVL 不支持 Buffer Registration
    if (comm->MNNVL)
        comm->nvlsRegSupport = 0;
    else if (ncclParamSingleProcMemRegEnable())
        comm->nvlsRegSupport = 1;

    TRACE(NCCL_INIT, "pidHash[%d] %lx intraProcRank %d intraProcRanks %d intraProcRank0 %d",
        rank, comm->peerInfo[rank].pidHash, intraProcRank, intraProcRanks, intraProcRank0);

    if (intraProcRank == -1 || intraProcRank0 == -1 || comm->peerInfo[intraProcRank0].comm == NULL) {
      WARN("Failed to determine intra proc ranks rank %d hostHash %lx pidHash %lx intraProcRank %d intraProcRanks %d intraProcRank0 %d",
          rank, comm->peerInfo[rank].hostHash, comm->peerInfo[rank].pidHash,
          intraProcRank, intraProcRanks, intraProcRank0);
      ret = ncclInternalError;
      goto fail;
    }

    struct ncclComm* comm0 = comm->peerInfo[intraProcRank0].comm;
    assert(intraProcRank == 0 ? comm == comm0 : true);

    // 这里 intraComm0 可能会指向自己（比如每个 GPU 一个进程的工作模式）
    comm->intraComm0 = comm0;
    comm->intraRank = intraProcRank;
    comm->intraRanks = intraProcRanks;
    comm->intraBarrierPhase = 0;
    comm->intraBarrierCounter = 0;
    comm->intraBarrierGate = 0;
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

  // ========== 开始统计带宽和通道信息 ==========
  // 获取 Ring 和 Tree 图
  memset(ringGraph, 0, sizeof(struct ncclTopoGraph));
  ringGraph->id = 0;
  ringGraph->pattern = NCCL_TOPO_PATTERN_RING;
  ringGraph->minChannels = 1;           // 最小通道数
  ringGraph->maxChannels = MAXCHANNELS / 2;  // 最大通道数（32）
  // 填入 ringGraph 的带宽和通道信息
  NCCLCHECKGOTO(ncclTopoCompute(comm->topo, ringGraph), ret, fail);
  NCCLCHECKGOTO(ncclTopoPrintGraph(comm->topo, ringGraph), ret, fail);

  memset(treeGraph, 0, sizeof(struct ncclTopoGraph));
  treeGraph->id = 1;
  treeGraph->pattern = NCCL_TOPO_PATTERN_BALANCED_TREE;
  treeGraph->minChannels = ringGraph->nChannels;
  treeGraph->maxChannels = ringGraph->nChannels;
  NCCLCHECKGOTO(ncclTopoCompute(comm->topo, treeGraph), ret, fail);
  NCCLCHECKGOTO(ncclTopoPrintGraph(comm->topo, treeGraph), ret, fail);

  // CollNet Chain 图
  memset(collNetChainGraph, 0, sizeof(struct ncclTopoGraph));
  collNetChainGraph->id = 2;
  collNetChainGraph->pattern = NCCL_TOPO_PATTERN_TREE;
  collNetChainGraph->collNet = 1;
  collNetChainGraph->minChannels = ringGraph->nChannels;
  collNetChainGraph->maxChannels = ringGraph->nChannels;

  // CollNet Direct 图
  memset(collNetDirectGraph, 0, sizeof(struct ncclTopoGraph));
  collNetDirectGraph->id = 4;
  collNetDirectGraph->pattern = NCCL_TOPO_PATTERN_COLLNET_DIRECT;
  collNetDirectGraph->collNet = 1;
  collNetDirectGraph->minChannels = 1;
  collNetDirectGraph->maxChannels = MAXCHANNELS;

  // 如果支持 CollNet，计算 CollNet 拓扑
  if (comm->config.collnetEnable) {
    NCCLCHECKGOTO(ncclTopoCompute(comm->topo, collNetChainGraph), ret, fail);
    NCCLCHECKGOTO(ncclTopoPrintGraph(comm->topo, collNetChainGraph), ret, fail);
    NCCLCHECKGOTO(ncclTopoCompute(comm->topo, collNetDirectGraph), ret, fail);
    NCCLCHECKGOTO(ncclTopoPrintGraph(comm->topo, collNetDirectGraph), ret, fail);
  }

  // NVLS 图
  memset(nvlsGraph, 0, sizeof(struct ncclTopoGraph));
  nvlsGraph->id = 3;
  nvlsGraph->pattern = NCCL_TOPO_PATTERN_NVLS;
  nvlsGraph->minChannels = 1;
  nvlsGraph->maxChannels = MAXCHANNELS;

  // 是否支持 NVLS Sharp
  if (comm->nvlsSupport) {
    NCCLCHECKGOTO(ncclTopoCompute(comm->topo, nvlsGraph), ret, fail);
    NCCLCHECKGOTO(ncclTopoPrintGraph(comm->topo, nvlsGraph), ret, fail);
  }

  timers[TIMER_INIT_GRAPHS] = clockNano() - timers[TIMER_INIT_GRAPHS];
  // ========== 统计各个算法通信图带宽流程结束 ==========

  // 初始化此通信器的 P2P LL 缓冲区数量
  comm->allocP2pNetLLBuffers = ncclParamAllocP2pNetLLBuffers() == 1;

  // 如果需要，转储图信息到文件
  if (comm->rank == ncclParamGraphDumpFileRank()) {
    struct ncclTopoGraph* dumpGraphs[5] = { ringGraph, treeGraph, collNetDirectGraph, collNetChainGraph, nvlsGraph };
    NCCLCHECKGOTO(ncclTopoDumpGraphs(comm->topo, 5, dumpGraphs), ret, fail);
  }

  // 因为 timers[TIMER_INIT_ALLGATHER] 已经包含第一次 AllGather 的时间，
  // 我们暂时将第二次 AllGather 的开始时间存储在未使用的 CONNECT 计时器中
  timers[TIMER_INIT_CONNECT] = clockNano();

  // ========== AllGather 3 - 交换图信息 ==========
  NCCLCHECKGOTO(ncclCalloc(&allGather3Data, nranks), ret, fail);

  // 填充本 rank 的信息，用于后续发送给其他 rank
  for (int a = 0; a < NCCL_NUM_ALGORITHMS; a++) {
    allGather3Data[rank].graphInfo[a].pattern = graphs[a]->pattern;
    allGather3Data[rank].graphInfo[a].nChannels = graphs[a]->nChannels;
    allGather3Data[rank].graphInfo[a].sameChannels = graphs[a]->sameChannels;
    allGather3Data[rank].graphInfo[a].bwIntra = graphs[a]->bwIntra;
    allGather3Data[rank].graphInfo[a].bwInter = graphs[a]->bwInter;
    allGather3Data[rank].graphInfo[a].typeIntra = graphs[a]->typeIntra;
    allGather3Data[rank].graphInfo[a].typeInter = graphs[a]->typeInter;
    allGather3Data[rank].graphInfo[a].crossNic = graphs[a]->crossNic;
  }

  allGather3Data[rank].cpuArch = comm->cpuArch;
  allGather3Data[rank].cpuVendor = comm->cpuVendor;

  // 设置 channels（取最小值以确保所有 rank 一致）
  comm->nChannels = std::min(treeGraph->nChannels, ringGraph->nChannels);

  // 初始化 topoRanks 配置
  NCCLCHECKGOTO(ncclTopoPreset(comm, graphs, &allGather3Data[rank].topoRanks), ret, fail);

  // 获取通信组内所有 rank 的 allGatherInfo 信息
  NCCLCHECKGOTO(bootstrapAllGather(comm->bootstrap, allGather3Data, sizeof(*allGather3Data)), ret, fail);

  // 确定节点数量、首个 rank 等
  NCCLCHECKGOTO(ncclCalloc(&nodesFirstRank, nranks), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&nodesTreePatterns, nranks), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&comm->rankToNode, comm->nRanks), ret, fail);

  for (int r = 0; r < nranks; r++) {
    int node;
    int firstRank = allGather3Data[r].topoRanks.ringRecv[0];
    // 找到该 rank 所属的节点
    for (node = 0; node < comm->nNodes && nodesFirstRank[node] != firstRank; node++)
        ;

    if (node == comm->nNodes) {
      comm->nNodes++;
      nodesFirstRank[node] = firstRank;
      // 记录每个节点的 tree 模式（可能因 SM 架构不同而不同）
      nodesTreePatterns[node] = allGather3Data[r].graphInfo[NCCL_ALGO_TREE].pattern;
    }
    // rank 映射到 node
    comm->rankToNode[r] = node;

    // 检测混合 CPU 架构
    if (comm->cpuArch != allGather3Data[r].cpuArch &&
        comm->cpuArch != NCCL_TOPO_CPU_ARCH_MIXED) {
      comm->cpuArch = NCCL_TOPO_CPU_ARCH_MIXED;
    }
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

  // 现在知道了 nNodes，分配 nodeRanks 并计算每个节点的 localRanks
  NCCLCHECKGOTO(ncclCalloc(&comm->nodeRanks, comm->nNodes), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&comm->rankToLocalRank, comm->nRanks), ret, fail);

  // 全局 rank 映射到 local rank
  for (int r = 0; r < comm->nRanks; r++) {
    int node = comm->rankToNode[r];
    comm->rankToLocalRank[r] = comm->nodeRanks[node].localRanks;
    comm->nodeRanks[node].localRanks++;
  }

  // 先设置为最大值
  comm->minLocalRanks = INT_MAX;

  // 计算通信组内一个节点中最大的 rank 数
  for (int n = 0; n < comm->nNodes; n++) {
    NCCLCHECKGOTO(ncclCalloc(&comm->nodeRanks[n].localRankToRank, comm->nodeRanks[n].localRanks), ret, fail);
    comm->maxLocalRanks = std::max(comm->maxLocalRanks, comm->nodeRanks[n].localRanks);
    comm->minLocalRanks = std::min(comm->minLocalRanks, comm->nodeRanks[n].localRanks);
    comm->nodeRanks[n].localRanks = 0;
  }

  // 填充 ranks 数组
  // 设置 localRankToRank 映射表
  for (int r = 0; r < comm->nRanks; r++) {
    int node = comm->rankToNode[r];
    comm->nodeRanks[node].localRankToRank[comm->nodeRanks[node].localRanks++] = r;
  }

  // 自己的 node 号
  comm->node = comm->rankToNode[rank];
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

  // 设置相关 ring 参数等
  NCCLCHECKGOTO(ncclTopoPostset(comm, nodesFirstRank, nodesTreePatterns, allTopoRanks, rings, graphs, parent), ret, fail);

  // AllGather 3 结束
  timers[TIMER_INIT_ALLGATHER] += clockNano() - timers[TIMER_INIT_CONNECT];

  TRACE(NCCL_INIT, "rank %d nranks %d - BUILT %d TREES/RINGS", rank, nranks, comm->nChannels);

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

  // 如果 comm 自己创建了 sharedRes
  if (comm->sharedRes->owner == comm) {
    comm->sharedRes->tpNLocalRanks = comm->localRanks;
    comm->sharedRes->magic = comm->magic;
    comm->sharedRes->tpNChannels = comm->nChannels;
    comm->sharedRes->tpP2pNChannels = comm->p2pnChannels;
    memcpy(comm->sharedRes->tpRankToLocalRank, comm->rankToLocalRank, sizeof(int) * comm->nRanks);
  }

  // 设置 topParentLocalRanks
  NCCLCHECKGOTO(ncclCalloc(&topParentLocalRanks, comm->localRanks), ret, fail);
  for (int i = 0; i < comm->localRanks; ++i) {
    int tpRank = comm->topParentRanks[comm->localRankToRank[i]];
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
    NCCLCHECKGOTO(ncclProxyCreate(comm), ret, fail);
  }

  NCCLCHECKGOTO(ncclCalloc(&comm->gproxyConn, comm->nRanks), ret, fail);

  timers[TIMER_INIT_CONNECT] = clockNano();

  do { // 构建 P2P 调度
    int node = comm->node;           // 当前节点号
    int nNodes = comm->nNodes;        // 总节点数量
    int nRanks = comm->nRanks;        // 总 rank 数量
    int local = comm->localRank;     // 本地 rank 号
    int nLocals = comm->maxLocalRanks;  // 所有 node 中最大的 local rank 数量
    struct ncclNodeRanks* nodeRanks = comm->nodeRanks;
    bool flat = false;

    // 检查所有 node 上的 local rank 数量是否相同
    for (int node = 0; node < nNodes; node++) {
      if (nodeRanks[node].localRanks != nLocals) {
        // 节点间 rank 数量不相同（如节点 0 有 4 GPU，节点 1 有 2 GPU）
        // 退化为扁平化处理
        flat = true;
        nNodes = 1;
        node = 0;
        nLocals = nRanks;
        local = rank;
        break;
      }
    }

    int nNodesPow2 = pow2Up(nNodes);
    int nLocalsPow2 = pow2Up(nLocals);
    comm->p2pSchedule = ncclMemoryStackAlloc<ncclComm::P2pSchedulePair>(&comm->memPermanent, nRanks);
    comm->planner.peers = ncclMemoryStackAlloc<ncclKernelPlanner::Peer>(&comm->memPermanent, nRanks);
    uint32_t nodeRound = 0;
    uint32_t nodeDelta = 0;
    int round = 0;

    // 使用二次公式 (x*x+x)/2 mod N 枚举 peer delta
    // 因为这个公式只在 N 是 2 的幂时产生有效排列
    // 我们让 N = pow2Up(n) 并过滤掉 >= n 的结果
    // 16 个 rank 的示例序列：0, 1, 3, 6, 10, 15, 5, 12, 4, 13, 7, 2, 14, 11, 9, 8
    do {
      if (nodeDelta < nNodes) { // 过滤无效的节点 delta
        int sendNode = (node + nodeDelta) % nNodes;
        int recvNode = (node - nodeDelta + nNodes) % nNodes;
        uint32_t localRound = 0;
        uint32_t localDelta = 0;
        do {
          if (localDelta < nLocals) { // 过滤无效的节点内 delta
            int sendLocal = (local + localDelta) % nLocals;
            int recvLocal = (local - localDelta + nLocals) % nLocals;
            comm->p2pSchedule[round].sendRank = flat ? sendLocal : nodeRanks[sendNode].localRankToRank[sendLocal];
            comm->p2pSchedule[round].recvRank = flat ? recvLocal : nodeRanks[recvNode].localRankToRank[recvLocal];
            round += 1;
          }
          localRound += 1;
          localDelta = (localDelta + localRound) & (nLocalsPow2 - 1);  // 二次更新
        } while (localRound != nLocalsPow2);
      }

      nodeRound += 1;
      nodeDelta = (nodeDelta + nodeRound) & (nNodesPow2 - 1);  // 二次更新
    } while (nodeRound != nNodesPow2);

    if (round != nRanks) {
      WARN("P2p schedule creation has bugs.");
      ret = ncclInternalError;
      goto fail;
    }
  } while (0);

  // ncclParamRuntimeConnect 默认为 1
  comm->runtimeConn = comm->cuMemSupport && ncclParamRuntimeConnect();

  if (comm->runtimeConn) {
    // 运行时建立连接（延迟连接）
    for (int c = 0; c < comm->nChannels; c++) {
      NCCLCHECKGOTO(setupChannel(comm, c, rank, nranks, rings + c * nranks), ret, fail);
    }
    // 尝试设置 NVLS，可能静默失败并禁用 NVLS
    NCCLCHECKGOTO(ncclNvlsSetup(comm, parent), ret, fail);
    // 检查是否可以设置 CollNet
    if (comm->config.collnetEnable)
        ncclCollNetSetup(comm, parent, graphs);
  } else {
    // 直接先建立好连接
    for (int c = 0; c < comm->nChannels; c++) {
      NCCLCHECKGOTO(setupChannel(comm, c, rank, nranks, rings + c * nranks), ret, fail);
    }

    // 连接 Ring
    NCCLCHECKGOTO(ncclTransportRingConnect(comm), ret, fail);

    // 连接 Tree
    NCCLCHECKGOTO(ncclTransportTreeConnect(comm), ret, fail);

    // 仅对每个节点只有 1 个 GPU 的通信器连接 PAT
    if (comm->maxLocalRanks == 1)
        NCCLCHECKGOTO(ncclTransportPatConnect(comm), ret, fail);

    // 尝试设置 NVLS，可能静默失败并禁用 NVLS
    NCCLCHECKGOTO(ncclNvlsSetup(comm, parent), ret, fail);
    NCCLCHECKGOTO(ncclNvlsBufferSetup(comm), ret, fail);

    // 如果需要，连接 NVLS Tree
    NCCLCHECKGOTO(ncclNvlsTreeConnect(comm), ret, fail);

    // 检查是否可以设置 CollNet
    if (comm->config.collnetEnable) {
      ncclCollNetSetup(comm, parent, graphs);
      NCCLCHECKGOTO(ncclCollNetChainBufferSetup(comm), ret, fail);
      if (comm->maxLocalRanks <= NCCL_MAX_DIRECT_ARITY + 1) {
        NCCLCHECKGOTO(ncclCollNetDirectBufferSetup(comm), ret, fail);
      }
    }

    // 连接到本地 net proxy
    NCCLCHECKGOTO(ncclProxyConnect(comm, TRANSPORT_NET, 1, comm->rank, &proxyConn), ret, fail);
    NCCLCHECKGOTO(ncclProxyCallBlocking(comm, &proxyConn, ncclProxyMsgSharedInit, &comm->p2pnChannels, sizeof(int), NULL, 0), ret, fail);

    // 如果使用 PXN，连接到远程 proxy
    // PXN 禁止借助非本地网卡进行跨节点通信，转而通过 NVLink 经由中间 GPU 中转数据
    if (ncclPxnDisable(comm) == 0) {
      int nranks;
      NCCLCHECKGOTO(ncclTopoGetPxnRanks(comm, &pxnPeers, &nranks), ret, fail);
      for (int r = 0; r < nranks; r++) {
        NCCLCHECKGOTO(ncclProxyConnect(comm, TRANSPORT_NET, 1, pxnPeers[r], &proxyConn), ret, fail);
        NCCLCHECKGOTO(ncclProxyCallBlocking(comm, &proxyConn, ncclProxyMsgSharedInit, &comm->p2pnChannels, sizeof(int), NULL, 0), ret, fail);
      }
    }

    // 默认为 1
    if (ncclParamNvbPreconnect()) {
      // 使用 NVB 路径时连接 P2P
      int nvbNpeers;
      NCCLCHECKGOTO(ncclTopoGetNvbGpus(comm->topo, comm->rank, &nvbNpeers, &nvbPeers), ret, fail);
      for (int r = 0; r < nvbNpeers; r++) {
        int peer = nvbPeers[r];
        int sendRound = 0, recvRound = 0;
        while (comm->p2pSchedule[sendRound].sendRank != peer)
            sendRound++;
        while (comm->p2pSchedule[recvRound].recvRank != peer)
            recvRound++;
        uint8_t sendBase = ncclP2pChannelBaseForRound(comm, sendRound);
        uint8_t recvBase = ncclP2pChannelBaseForRound(comm, recvRound);
        for (int c = 0; c < comm->p2pnChannelsPerPeer; c++) {
          int channelId;
          channelId = ncclP2pChannelForPart(comm->p2pnChannels, sendBase, c);
          if (comm->channels[channelId].peers[peer]->send[1].connected == 0) {
            comm->connectSend[peer] |= (1UL << channelId);
          }
          channelId = ncclP2pChannelForPart(comm->p2pnChannels, recvBase, c);
          if (comm->channels[channelId].peers[peer]->recv[1].connected == 0) {
            comm->connectRecv[peer] |= (1UL << channelId);
          }
        }
      }

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

  // 加载 NCCL_LAUNCH_MODE
  if (comm->intraRank == 0) {
    const char* str = ncclGetEnv("NCCL_LAUNCH_MODE");
    enum ncclLaunchMode mode, modeOld;
    if (str && strcasecmp(str, "GROUP") == 0) {
      mode = ncclLaunchModeGroup;
    } else {
      mode = ncclLaunchModeParallel;  // 默认并行模式
    }
    // 理论上可能与连接到多个 ncclUniqueId 的其他通信器竞争
    modeOld = __atomic_exchange_n(&ncclParamLaunchMode, mode, __ATOMIC_RELAXED);
    if (modeOld == ncclLaunchModeInvalid && str && str[0] != '\0') {
      INFO(NCCL_ENV, "NCCL_LAUNCH_MODE set by environment to %s", mode == ncclLaunchModeParallel ? "PARALLEL" : "GROUP");
    }
  }

  // 对称支持检测
  comm->symmetricSupport = comm->isAllDirectP2p && comm->nNodes == 1 && ncclParamWinEnable() && ncclCuMemEnable();
  comm->devrState.bigSize = 0;

  comm->ceColl.baseUCSymReadyPtr = NULL;
  comm->ceColl.baseUCSymComplPtr = NULL;

  // 在最后一个屏障之前调用 devCommSetup，确保没有线程在前面运行并开始启动 NCCL 内核
  // 这可能导致死锁
  NCCLCHECKGOTO(devCommSetup(comm), ret, fail);

  timers[TIMER_INIT_CONNECT] = clockNano() - timers[TIMER_INIT_CONNECT];

  /* 本地节点内屏障 */
  // 这个调用确保了在同一个节点内的所有 GPU ranks 都执行到了这个同步点
  // 也就是 initTransportsRank 操作都执行完毕了
  NCCLCHECKGOTO(bootstrapIntraNodeBarrier(comm->bootstrap, comm->localRankToRank, comm->localRank, comm->localRanks, comm->localRankToRank[0]), ret, fail);

  // 我们应该已经分配了所有缓冲区、collective fifos 等，可以恢复亲和性
  TRACE(NCCL_INIT, "rank %d nranks %d - DONE", rank, nranks);

exit:
  if (CPU_COUNT(&comm->cpuAffinity))
    sched_setaffinity(0, sizeof(cpu_set_t), &affinitySave);

  /* 如果 split 资源是共享的，我们无法在这里断开 proxy ops pool 的链接，
   * 因为子 comm 可以随时附加父 comm 的 proxy ops pool；
   * 否则，在这里断开链接以确保 pool 被正确清理。 */
  if (comm->sharedRes->owner == comm && !comm->shareResources && ret == ncclSuccess && !ncclCuMemEnable())
    ncclProxyShmUnlink(comm);

  free(allTopoRanks);
  free(nodesTreePatterns);
  free(nodesFirstRank);
  free(allGather3Data);
  free(rings);
  free(nvbPeers);
  free(pxnPeers);
  return ret;
fail:
  goto exit;
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
  NCCLCHECK(ncclCommInitRankDev(newcomm, nranks, /* root 数量 */1, &commId, myrank, cudaDev, &config, __func__));

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
