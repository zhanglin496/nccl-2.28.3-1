/*************************************************************************
 * Copyright (c) 2016-2023, NVIDIA CORPORATION. All rights reserved.
 * 版权所有 (c) 2016-2023，NVIDIA 公司。保留所有权利。
 *
 * See LICENSE.txt for license information
 * 详见 LICENSE.txt 获取许可证信息
 ************************************************************************/

// Implementation of the NVLink SHARP (NVLS) transport
// NVLink SHARP (NVLS) 传输层的实现

// 包含通信上下文相关的头文件，定义了 ncclComm 结构体和通信相关操作
#include "comm.h"
// 包含 CUDA Graph 相关的头文件，用于图捕获和执行
#include "graph.h"
// 包含工具函数头文件，提供通用工具函数
#include "utils.h"
// 包含代理相关头文件，用于代理操作
#include "proxy.h"
// 包含操作排队相关头文件，用于操作入队管理
#include "enqueue.h"
// 包含内存注册相关头文件，用于内存注册管理
#include "register.h"
// 包含传输层相关头文件，定义传输层接口
#include "transport.h"
// 包含内联内存注册头文件，提供内存注册的内联函数
#include "register_inline.h"

// 如果 CUDA 运行时版本大于等于 12.1（CUDA 12.1），则编译以下代码
#if CUDART_VERSION >= 12010

// CUDA Graph 注册数据结构，用于在 CUDA Graph 中记录内存注册信息
struct graphRegData {
  uintptr_t offset;  // 偏移量，表示在已注册内存区域中的偏移位置
  size_t size;       // 大小，表示注册的内存区域大小
};

// 本地注册数据结构，用于存储本地内存的注册信息
struct localRegData {
  struct ncclReg reg;  // NCCL 内存注册记录结构体
  intptr_t offset;     // 在注册内存区域中的偏移量
};

// NVLS 传输层是否可以连接的判断函数
// 参数说明：
//   ret: 输出参数，返回是否可以连接（0 表示不可以连接）
//   comm: NCCL 通信上下文
//   graph: 拓扑图
//   info1, info2: 对等节点的信息
ncclResult_t nvlsCanConnect(int* ret, struct ncclComm* comm, struct ncclTopoGraph* graph, struct ncclPeerInfo* info1, struct ncclPeerInfo* info2) {
  // This transport cannot be used for p2p
  // 此传输层不能用于点对点（P2P）通信
  *ret = 0;  // 设置返回值为 0，表示不能使用此传输层进行连接
  return ncclSuccess;  // 返回成功状态
}

// NVLS 发送端释放资源函数
// 参数说明：
//   send: 连接器指针，指向发送端连接器
ncclResult_t nvlsSendFree(struct ncclConnector* send) {
  return ncclSuccess;  // 直接返回成功，NVLS 发送端无需特殊释放操作
}

// NVLS 接收端释放资源函数
// 参数说明：
//   recv: 连接器指针，指向接收端连接器
ncclResult_t nvlsRecvFree(struct ncclConnector* recv) {
  return ncclSuccess;  // 直接返回成功，NVLS 接收端无需特殊释放操作
}

// NVLS 传输层结构体，实现了 NCCL 传输层接口
struct ncclTransport nvlsTransport = {
  "NVLS",                    // 传输层名称：NVLink SHARP
  nvlsCanConnect,            // 连接判断函数指针
  { NULL, NULL, nvlsSendFree, NULL, NULL, NULL, NULL, NULL },  // 发送端操作函数指针表
  { NULL, NULL, nvlsRecvFree, NULL, NULL, NULL, NULL, NULL }   // 接收端操作函数指针表
};

// NVLS 组创建函数，用于创建多播组
// 参数说明：
//   comm: NCCL 通信上下文
//   prop: 多播对象属性指针
//   rank: 当前 rank 的编号
//   nranks: 总 rank 数量
//   mcHandle: 输出参数，返回多播句柄
//   shareableHandle: 输出参数，返回可共享的句柄（用于其他 rank 连接）
ncclResult_t ncclNvlsGroupCreate(struct ncclComm *comm, CUmulticastObjectProp *prop, int rank, unsigned int nranks, CUmemGenericAllocationHandle *mcHandle, char *shareableHandle) {
  CUmemAllocationHandleType type = ncclCuMemHandleType;  // 获取内存句柄类型（取决于系统配置）
  size_t size = prop->size;  // 获取请求的多播组大小

  // Create a Multicast group
  // 创建一个多播组

  INFO(NCCL_NVLS, "NVLS Creating Multicast group nranks %d size %zu on rank %d", nranks, size, rank);  // 记录日志：正在创建多播组
  CUCHECK(cuMulticastCreate(mcHandle, prop));  // 调用 CUDA API 创建多播组，并检查返回值

  if (type == CU_MEM_HANDLE_TYPE_FABRIC) {  // 如果句柄类型是 FABRIC 类型
    // Get a handle to pass to other ranks
    // 获取一个句柄以传递给其他 rank
    CUCHECK(cuMemExportToShareableHandle(shareableHandle, *mcHandle, ncclCuMemHandleType, 0));  // 将多播句柄导出为可共享句柄
  }
  else {  // 对于其他句柄类型（如 POSIX 文件描述符）
    memcpy(shareableHandle, mcHandle, sizeof(CUmemGenericAllocationHandle));  // 直接复制多播句柄到共享句柄缓冲区
  }

  INFO(NCCL_NVLS, "NVLS Created Multicast group %llx nranks %d size %zu on rank %d", *mcHandle, nranks, size, rank);  // 记录日志：多播组创建成功

  return ncclSuccess;  // 返回成功状态
}

// NVLS 组连接函数，用于连接到其他 rank 创建的多播组
// 参数说明：
//   comm: NCCL 通信上下文
//   shareableHandle: 共享句柄，由 rank 0 创建并广播
//   rank: 创建组的 rank 的编号（通常是 rank 0）
//   mcHandle: 输出参数，返回导入的多播句柄
ncclResult_t ncclNvlsGroupConnect(struct ncclComm *comm, char *shareableHandle, int rank, CUmemGenericAllocationHandle *mcHandle) {
  CUmemAllocationHandleType type = ncclCuMemHandleType;  // 获取内存句柄类型
  int fd = -1;  // 文件描述符，初始化为 -1（无效值）
  ncclResult_t ret = ncclSuccess;  // 返回值，初始化为成功
  INFO(NCCL_NVLS, "NVLS importing shareableHandle %p from rank %d", shareableHandle, rank);  // 记录日志：正在导入共享句柄

  // Import and map the remote memory descriptor to the local GPU
  // 导入并映射远程内存描述符到本地 GPU

  if (type == CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) {  // 如果句柄类型是 POSIX 文件描述符
    // cuMem UDS support
    // cuMem Unix Domain Socket 支持
    TRACE(NCCL_NVLS, "NVLS rank %d Importing shareable handle %p from rank %d", comm->localRank, shareableHandle, rank);  // 追踪日志：正在导入共享句柄
    TRACE(NCCL_NVLS, "NVLS rank %d request conversion of handle 0x%lx from rank %d", comm->localRank, *(uint64_t*)shareableHandle, rank);  // 追踪日志：请求转换句柄
    NCCLCHECKGOTO(ncclProxyClientGetFdBlocking(comm, rank, shareableHandle, &fd), ret, fail);  // 通过代理客户端阻塞式获取文件描述符
    TRACE(NCCL_NVLS, "NVLS rank %d received converted fd %d from rank %d", comm->localRank, fd, rank);  // 追踪日志：已收到转换后的文件描述符
    CUCHECKGOTO(cuMemImportFromShareableHandle(mcHandle, (void *)(uintptr_t)fd, type), ret, fail);  // 从文件描述符导入共享句柄并检查错误
    SYSCHECK(close(fd), "close");  // 关闭文件描述符
  } else {  // 对于其他句柄类型
    if (type == CU_MEM_HANDLE_TYPE_FABRIC) {  // 如果是 FABRIC 类型
      CUCHECKGOTO(cuMemImportFromShareableHandle(mcHandle, (void *)shareableHandle, type), ret, fail);  // 从共享句柄导入
    } else {  // 对于其他类型
      memcpy(mcHandle, shareableHandle, sizeof(CUmemGenericAllocationHandle));  // 直接复制句柄
    }
  }
exit:  // 正常退出标签
  return ret;  // 返回结果
fail:  // 失败处理标签
  if (fd != -1) close(fd);  // 如果文件描述符有效，关闭它
  goto exit;  // 跳转到退出标签
}

// NVLS 组解绑函数，从多播组解绑物理内存
// 参数说明：
//   comm: NCCL 通信上下文
//   size: 要解绑的内存大小
//   mcHandle: 多播句柄
ncclResult_t nvlsGroupUnbind(struct ncclComm *comm, size_t size, CUmemGenericAllocationHandle* mcHandle) {
  int dev = comm->cudaDev;  // 获取 CUDA 设备号
  INFO(NCCL_NVLS, "NVLS Unbind MC handle %llx size %zu dev %d", *mcHandle, size, dev);  // 记录日志：解绑多播句柄

  // Unbind physical memory from group for the given device
  // 从组中解绑给定设备的物理内存

  if (size) CUCHECK(cuMulticastUnbind(*mcHandle, dev, 0/*mcOffset*/, size));  // 如果 size > 0，则解绑多播内存

  return ncclSuccess;  // 返回成功状态
}

// NVLS 注销缓冲区函数，释放已注册的 NVLS 缓冲区
// 参数说明：
//   comm: NCCL 通信上下文
//   mcHandler: 多播句柄
//   ptr: 设备内存指针
//   dev: 设备号
//   ucsize: UC (Unicast) 内存大小
//   mcsize: MC (Multicast) 内存大小
ncclResult_t ncclNvlsDeregBuffer(struct ncclComm* comm, CUmemGenericAllocationHandle *mcHandler, CUdeviceptr ptr, int dev, size_t ucsize, size_t mcsize) {
  // unbind can trigger RM error if buffer is freed already by users
  // however, it is safe to ignore the error, and unbind will succeed anyway
  // 如果缓冲区已被用户释放，解绑操作可能触发资源管理器（RM）错误
  // 然而，忽略这个错误是安全的，解绑操作无论如何会成功

  CUCALL(cuMulticastUnbind(*mcHandler, dev, 0/*mcOffset*/, ucsize));  // 调用解绑操作（使用 CUCALL 忽略错误）
  CUCHECK(cuMemUnmap(ptr, mcsize));  // 取消映射内存
  CUCHECK(cuMemAddressFree(ptr, mcsize));  // 释放虚拟地址范围
  CUCHECK(cuMemRelease(*mcHandler));  // 释放多播句柄
  INFO(NCCL_NVLS, "rank %d - NVLS deregistered buffer %p on device %d ucsize %ld mcsize %ld", comm->rank, (void*)ptr, dev, ucsize, mcsize);  // 记录日志：已注销缓冲区
  return ncclSuccess;  // 返回成功状态
}

// NVLS 组取消映射内存函数，释放 UC 和 MC 内存
// 参数说明：
//   comm: NCCL 通信上下文
//   ucsize: UC 内存大小
//   ucptr: UC 内存指针
//   ucHandle: UC 句柄
//   mcsize: MC 内存大小
//   mcptr: MC 内存指针
//   mcHandle: MC 句柄
ncclResult_t nvlsGroupUnmapMem(struct ncclComm *comm, size_t ucsize, void* ucptr, CUmemGenericAllocationHandle* ucHandle, size_t mcsize, void* mcptr, CUmemGenericAllocationHandle* mcHandle) {
  INFO(NCCL_NVLS, "NVLS Unmap mem UC handle 0x%llx(%p) ucsize %zu MC handle 0x%llx(%p) mcsize %zd", *ucHandle, ucptr, ucsize, *mcHandle, mcptr, mcsize);  // 记录日志：取消映射内存

  // Release the UC memory and mapping
  // 释放 UC（单播）内存和映射

  if (ucptr) {  // 如果 UC 指针有效
    CUCHECK(cuMemUnmap((CUdeviceptr)ucptr, ucsize));  // 取消映射 UC 内存
    CUCHECK(cuMemAddressFree((CUdeviceptr)ucptr, ucsize));  // 释放 UC 虚拟地址范围
    CUCHECK(cuMemRelease(*ucHandle));  // 释放 UC 句柄
  }

  // Release the MC memory and mapping
  // 释放 MC（多播）内存和映射

  if (mcptr) {  // 如果 MC 指针有效
    CUCHECK(cuMemUnmap((CUdeviceptr)mcptr, mcsize));  // 取消映射 MC 内存
    CUCHECK(cuMemAddressFree((CUdeviceptr)mcptr, mcsize));  // 释放 MC 虚拟地址范围
    CUCHECK(cuMemRelease(*mcHandle));  // 释放 MC 句柄
  }

  return ncclSuccess;  // 返回成功状态
}

// 包含引导程序头文件，用于节点内通信和同步
#include "bootstrap.h"
// 包含通道头文件，定义了通道相关的数据结构和操作
#include "channel.h"

// NVLS 内存对齐大小定义为 2MB (1 << 21)
#define NVLS_MEM_ALIGN_SIZE (1 << 21)
// SM90 架构的 NVLS 通道数量为 16
#define NVLS_NCHANNELS_SM90 16
// SM100 架构的 NVLS 通道数量为 32
#define NVLS_NCHANNELS_SM100 32
// SM100 NVL 架构的 NVLS 通道数量为 24（用于单节点场景）
#define NVLS_NCHANNELS_SM100_NVL 24

// NVLS 启用参数，可通过环境变量 NCCL_NVLS_ENABLE 配置
NCCL_PARAM(NvlsEnable, "NVLS_ENABLE", 2);
// NVLS 块大小参数，可通过环境变量 NCCL_NVLS_CHUNKSIZE 配置，默认 128KB
NCCL_PARAM(NvlsChunkSize, "NVLS_CHUNKSIZE", 128*1024);

// NVLS 初始化函数，检测并初始化 NVLS 支持
// 参数说明：
//   comm: NCCL 通信上下文
ncclResult_t ncclNvlsInit(struct ncclComm* comm) {
  comm->nvlsSupport = 0;  // 初始化 NVLS 支持标志为 0（不支持）
  comm->nvlsChannels = 0;  // 初始化 NVLS 通道数为 0

  int gpuCount;  // GPU 数量变量
  NCCLCHECK(ncclTopoGetGpuCount(comm->topo, &gpuCount));  // 获取拓扑中的 GPU 数量
  if (!ncclParamNvlsEnable() || gpuCount <= 2) 
    return ncclSuccess;  // 如果 NVLS 未启用或 GPU 数量 <= 2，直接返回

  CUdevice dev;  // CUDA 设备变量
  int driverVersion;  // 驱动版本变量

  if (CUPFN(cuDeviceGet) == NULL) 
    return ncclSuccess;  // 如果 CUDA API 不可用，直接返回
    
  CUCHECK(cuCtxGetDevice(&dev));  // 获取当前 CUDA 设备
  CUDACHECK(cudaDriverGetVersion(&driverVersion));  // 获取驱动版本
  
  if (ncclParamNvlsEnable() == 2) {  // 如果 NVLS_ENABLE == 2（自动检测模式）
    // NVLS Multicast support requires CUDA12.1 UMD + KMD
    // NVLS 多播支持需要 CUDA 12.1 用户模式驱动 + 内核模式驱动

    if (CUPFN(cuMulticastCreate) != NULL /*&& driverVersion >= 12010 */) {  // 如果多播创建 API 可用
      CUCHECK(cuDeviceGetAttribute(&comm->nvlsSupport, CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED, dev));  // 检查设备是否支持多播
    }
  } else {  // 如果 NVLS_ENABLE == 1（强制启用模式）
    comm->nvlsSupport = 1;  // 强制设置 NVLS 支持标志
  }

  if (comm->nvlsSupport) {  // 如果 NVLS 被支持
    int channels;  // 通道数变量
    if (comm->compCap >= 100) {  // 如果计算能力 >= 100（Blackwell 架构）
      // Use a reduced number of channels for single node/MNNVL domain on Blackwell.
      // 在 Blackwell 上对单节点/MNNVL 域使用减少的通道数。

      // comm->nNodes is not yet initialized at this point so we need to use other data.
      // comm->nNodes 此时还未初始化，所以我们需要使用其他数据。

      bool multiNode;  // 多节点标志
      if (comm->MNNVL) {  // 如果启用了 MNNVL（多节点 NVLink）
        multiNode = (comm->clique.size < comm->nRanks);  // 如果派系大小小于总 rank 数，则为多节点
      } else {  // 如果未启用 MNNVL
        int i;  // 循环变量
        for (i = 1; i < comm->nRanks; i++) {  // 遍历所有 rank
          if (comm->peerInfo[i].hostHash != comm->peerInfo[0].hostHash)  // 如果发现不同主机的 hash
            break;  // 退出循环
        }
        multiNode = (i < comm->nRanks);  // 如果 i < nRanks，说明有多节点
      }
      channels = (multiNode ? NVLS_NCHANNELS_SM100 : NVLS_NCHANNELS_SM100_NVL);  // 根据是否多节点选择通道数
    } else {  // 对于其他架构（Hopper 等）
      channels = NVLS_NCHANNELS_SM90;  // 使用 SM90 的通道数
    }
    
    if (comm->config.nvlsCTAs != NCCL_CONFIG_UNDEF_INT)  // 如果用户配置了 NVLS CTA 数量
        channels = comm->config.nvlsCTAs;  // 使用用户配置的通道数
    comm->nvlsChannels = std::max(comm->config.minCTAs, std::min(comm->config.maxCTAs, channels));  // 将通道数限制在配置的 min 和 max 之间
  }
  INFO(NCCL_INIT, "NVLS multicast support is %savailable on dev %d (NVLS_NCHANNELS %d)",
       comm->nvlsSupport ? "" : "not ", dev, comm->nvlsChannels);  // 记录日志：NVLS 多播支持状态
  return ncclSuccess;  // 返回成功状态
}

// NVLS 树连接函数，用于多节点场景下的 NVLS 树连接
// 参数说明：
//   comm: NCCL 通信上下文
ncclResult_t ncclNvlsTreeConnect(struct ncclComm* comm) {
  ncclResult_t ret = ncclSuccess;  // 返回值初始化
  if (comm && comm->nvlsSupport && comm->nNodes > 1) {  // 如果通信上下文有效、支持 NVLS 且是多节点场景
    for (int c = 0; c < comm->nvlsChannels; c++) {  // 遍历所有 NVLS 通道
      struct ncclChannel* channel = comm->channels + c;  // 获取当前通道指针
      // 建立下行连接（从本节点到子节点），最多支持 NCCL_MAX_NVLS_TREE_ARITY 个子节点
      NCCLCHECKGOTO(ncclTransportP2pConnect(comm, c, NCCL_MAX_NVLS_TREE_ARITY, channel->nvls.treeDown, 1, &channel->nvls.treeUp, 0), ret, fail);
      // 建立上行连接（从本节点到父节点），只有 1 个父节点
      NCCLCHECKGOTO(ncclTransportP2pConnect(comm, c, 1, &channel->nvls.treeUp, NCCL_MAX_NVLS_TREE_ARITY, channel->nvls.treeDown, 0), ret, fail);
    }
    NCCLCHECKGOTO(ncclTransportP2pSetup(comm, &comm->graphs[NCCL_ALGO_NVLS], 0), ret, fail);  // 设置 P2P 传输
    INFO(NCCL_INIT, "Connected NVLS tree");  // 记录日志：NVLS 树已连接
  }
exit:  // 正常退出标签
  return ret;  // 返回结果
fail:  // 失败处理标签
  goto exit;  // 跳转到退出标签
}

// NVLS 内存分配静态函数，分配 UC 和 MC 内存
// 参数说明：
//   comm: NCCL 通信上下文
//   desc: 内存访问描述符
//   size: 请求的内存大小
//   ucHandle: 输出参数，UC 内存句柄
//   mcHandle: 输出参数，MC 内存句柄
//   ucptr: 输出参数，UC 内存指针
//   mcptr: 输出参数，MC 内存指针
//   ucsizePtr: 输出参数，UC 内存实际大小
//   mcsizePtr: 输出参数，MC 内存实际大小
static ncclResult_t nvlsAllocateMem(struct ncclComm* comm, const CUmemAccessDesc* desc, size_t size, CUmemGenericAllocationHandle* ucHandle, CUmemGenericAllocationHandle* mcHandle, void** ucptr, void** mcptr, size_t* ucsizePtr, size_t* mcsizePtr) {
  char shareableHandle[NVLS_HANDLE_SIZE];  // 共享句柄缓冲区，用于在 rank 间传递
  CUmulticastObjectProp mcprop;  // 多播对象属性
  CUmemAllocationProp ucprop;  // 单播内存分配属性
  ncclResult_t ret = ncclSuccess;  // 返回值初始化
  size_t mcsize;  // MC 内存大小
  size_t ucsize;  // UC 内存大小
  size_t ucgran, mcgran;  // UC 和 MC 的内存粒度
  int allocMcHandle = 0;  // 标志：是否分配了 MC 句柄

  mcsize = ucsize = size;  // 初始化 MC 和 UC 大小为请求的大小
  *ucptr = *mcptr = NULL;  // 初始化指针为 NULL
  memset(shareableHandle, '\0', sizeof(shareableHandle));  // 清零共享句柄缓冲区
  memset(&mcprop, 0, sizeof(CUmulticastObjectProp));  // 清零多播属性结构体
  mcprop.numDevices = comm->localRanks;  // 设置本地 rank 数量
  mcprop.handleTypes = ncclCuMemHandleType;  // 设置句柄类型
  mcprop.flags = 0;  // 设置标志为 0
  mcprop.size = size;  // 设置请求的大小
  CUCHECKGOTO(cuMulticastGetGranularity(&mcgran, &mcprop, CU_MULTICAST_GRANULARITY_RECOMMENDED), ret, fail);  // 获取推荐的 MC 内存粒度
  ALIGN_SIZE(mcsize, mcgran);  // 将 mcsize 对齐到 MC 粒度
  mcprop.size = mcsize;  // 更新 MC 属性中的大小

  if (comm->localRank == 0) {  // 如果是本地 rank 0
    NCCLCHECKGOTO(ncclNvlsGroupCreate(comm, &mcprop, comm->localRank, comm->localRanks, mcHandle, shareableHandle), ret, fail);  // 创建多播组
    allocMcHandle = 1;  // 标记已分配 MC 句柄
    NCCLCHECKGOTO(bootstrapIntraNodeBroadcast(comm->bootstrap, comm->localRankToRank, comm->localRank, comm->localRanks, 0, shareableHandle, NVLS_HANDLE_SIZE), ret, fail);  // 广播共享句柄到所有本地 rank
  } else {  // 如果不是本地 rank 0
    NCCLCHECKGOTO(bootstrapIntraNodeBroadcast(comm->bootstrap, comm->localRankToRank, comm->localRank, comm->localRanks, 0, shareableHandle, NVLS_HANDLE_SIZE), ret, fail);  // 接收广播的共享句柄
    NCCLCHECKGOTO(ncclNvlsGroupConnect(comm, shareableHandle, comm->localRankToRank[0], mcHandle), ret, fail);  // 连接到多播组
    allocMcHandle = 1;  // 标记已分配 MC 句柄
  }

  CUCHECKGOTO(cuMulticastAddDevice(*mcHandle, comm->cudaDev), ret, fail);  // 将当前设备添加到多播组

  memset(&ucprop, 0, sizeof(CUmemAllocationProp));  // 清零 UC 分配属性
  ucprop.type = CU_MEM_ALLOCATION_TYPE_PINNED;  // 设置类型为固定内存
  ucprop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;  // 设置位置类型为设备
  ucprop.location.id = comm->cudaDev;  // 设置设备 ID
  ucprop.requestedHandleTypes = ncclCuMemHandleType;  // 设置请求的句柄类型
  CUCHECKGOTO(cuMemGetAllocationGranularity(&ucgran, &ucprop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED), ret, fail);  // 获取推荐的 UC 内存粒度
  ALIGN_SIZE(ucsize, ucgran);  // 将 ucsize 对齐到 UC 粒度
  // Map a VA for UC memory with MC alignment and size
  // 为 UC 内存映射一个虚拟地址，使用 MC 的对齐和大小

  CUCHECKGOTO(cuMemAddressReserve((CUdeviceptr*)ucptr, ucsize, ucgran, 0U, 0), ret, fail);  // 保留 UC 虚拟地址空间

  // Alloc local physical mem for this NVLS group
  // 为此 NVLS 组分配本地物理内存

  CUCHECKGOTO(cuMemCreate(ucHandle, ucsize, &ucprop, 0), ret, fail1);  // 创建 UC 内存
  CUCHECKGOTO(cuMemMap((CUdeviceptr)*ucptr, ucsize, 0, *ucHandle, 0), ret, fail2);  // 映射 UC 内存到虚拟地址
  CUCHECKGOTO(cuMemSetAccess((CUdeviceptr)*ucptr, ucsize, desc, 1), ret, fail3);  // 设置 UC 内存访问权限
  CUDACHECKGOTO(cudaMemset(*ucptr, 0, ucsize), ret, fail3);  // 将 UC 内存清零

  // intra-node barrier to mitigate the possible hang in cuMulticastBindMem during abort
  // 节点内屏障，以缓解中止期间 cuMulticastBindMem 可能出现的挂起

  NCCLCHECKGOTO(bootstrapIntraNodeBarrier(comm->bootstrap, comm->localRankToRank, comm->localRank, comm->localRanks, comm->localRankToRank[0]), ret, fail3);  // 节点内屏障同步
  // Bind physical memory to the Multicast group
  // 将物理内存绑定到多播组

  // NB: It will block until all ranks have been added to the Group
  // 注意：此调用会阻塞，直到所有 rank 都添加到组中

  // This is where we normally see issues if the system NVLS/Multicast support is broken
  // 如果系统的 NVLS/多播支持有问题，通常在这里会出现错误

  {
    CUresult err = CUPFN(cuMulticastBindMem(*mcHandle, 0/*mcOffset*/, *ucHandle, 0/*memOffset*/, ucsize, 0/*flags*/));  // 绑定 UC 内存到多播组
    if (err != CUDA_SUCCESS) {  // 如果绑定失败
      const char *errStr;  // 错误字符串
      (void) pfn_cuGetErrorString(err, &errStr);  // 获取错误描述
      if (ncclParamNvlsEnable() == 1) {  // 如果是强制启用模式
        // Fail the job as NVLS support is not available
        // NVLS 支持不可用，使任务失败

        WARN("Failed to bind NVLink SHARP (NVLS) Multicast memory of size %ld : CUDA error %d '%s'.\nThis is usually caused by a system or configuration error in the Fabric Manager or NVSwitches.\nDo not force-enable NVLS (NCCL_NVLS_ENABLE=1) if you wish to avoid this error in the future.", ucsize, err, errStr );  // 输出警告信息
        ret = ncclUnhandledCudaError;  // 设置返回值为未处理的 CUDA 错误
      } else {  // 如果是自动检测模式
        // Continue without NVLS support (returns ncclSuccess)
        // 在没有 NVLS 支持的情况下继续（返回 ncclSuccess）

        INFO(NCCL_INIT|NCCL_NVLS, "Failed to bind NVLink SHARP (NVLS) Multicast memory of size %ld : CUDA error %d '%s'. Proceeding without NVLS support.", ucsize, err, errStr);  // 记录信息日志
      }
      comm->nvlsSupport = comm->nvlsChannels = 0;  // 关闭 NVLS 支持
      goto fail3;  // 跳转到失败处理
   }
  }

  // Map mc virtual address
  // 映射 MC 虚拟地址

  CUCHECKGOTO(cuMemAddressReserve((CUdeviceptr*)mcptr, mcsize, mcgran, 0U, 0), ret, fail);  // 保留 MC 虚拟地址空间
  CUCHECKGOTO(cuMemMap((CUdeviceptr)*mcptr, mcsize, 0, *mcHandle, 0), ret, fail);  // 映射 MC 内存到虚拟地址
  CUCHECKGOTO(cuMemSetAccess((CUdeviceptr)*mcptr, mcsize, desc, 1), ret, fail);  // 设置 MC 内存访问权限
  *ucsizePtr = ucsize;  // 返回 UC 实际大小
  *mcsizePtr = mcsize;  // 返回 MC 实际大小
  INFO(NCCL_NVLS, "NVLS rank %d (dev %d) alloc done, ucptr %p ucgran %ld mcptr %p mcgran %ld ucsize %ld mcsize %ld (inputsize %ld)", comm->rank, comm->cudaDev, *ucptr, ucgran, *mcptr, mcgran, ucsize, mcsize, size);  // 记录分配成功的日志

exit:  // 正常退出标签
  return ret;  // 返回结果
fail3:  // 失败处理标签 3
  CUCHECK(cuMemUnmap((CUdeviceptr)*ucptr, ucsize));  // 取消映射 UC 内存
fail2:  // 失败处理标签 2
  CUCHECK(cuMemRelease(*ucHandle));  // 释放 UC 句柄
fail1:  // 失败处理标签 1
  CUCHECK(cuMemAddressFree((CUdeviceptr)*ucptr, ucsize));  // 释放 UC 虚拟地址
fail:  // 失败处理标签
  if (allocMcHandle && *mcptr == NULL && *ucptr == NULL) CUCHECK(cuMemRelease(*mcHandle));  // 如果分配了 MC 句柄但未成功映射，释放它
  goto exit;  // 跳转到退出标签
}

// NVLS 缓冲区设置函数，设置 NVLS 通信所需的缓冲区
// 参数说明：
//   comm: NCCL 通信上下文
ncclResult_t ncclNvlsBufferSetup(struct ncclComm* comm) {
  int nHeads = -1;  // 头节点数量，初始化为 -1
  int headRank = -1;  // 头节点 rank，初始化为 -1
  ncclResult_t res = ncclSuccess;  // 返回值初始化
  int nvlsStepSize = -1;  // NVLS 步长大小，初始化为 -1
  size_t buffSize = 0;  // 缓冲区大小，初始化为 0
  size_t nvlsPerRankSize = 0;  // 每个 rank 的 NVLS 大小，初始化为 0
  size_t nvlsTotalSize = 0;  // NVLS 总大小，初始化为 0
  struct ncclNvlsSharedRes* resources = NULL;  // NVLS 共享资源指针，初始化为 NULL
  int nChannels = -1;  // 通道数量，初始化为 -1
  cudaStream_t deviceStream, hostStream;  // 设备流和主机流

  if (comm->nvlsSupport == 0 || comm->nvlsResources->inited) return ncclSuccess;  // 如果不支持 NVLS 或已初始化，直接返回
  // initialize after checking comm->nvlsSupport
  // 在检查 nvlsSupport 之后初始化

  nHeads = comm->channels[0].nvls.nHeads;  // 获取头节点数量
  headRank = comm->channels[0].nvls.headRank;  // 获取头节点 rank
  resources = comm->nvlsResources;  // 获取 NVLS 资源
  nChannels = comm->nvlsChannels;  // 获取 NVLS 通道数
  nvlsStepSize = comm->nvlsChunkSize;  // 获取 NVLS 块大小
  buffSize = nvlsStepSize * NCCL_STEPS;  // 计算缓冲区大小（块大小 * 步数）
  nvlsPerRankSize = nChannels * 2 * buffSize;  // 计算每个 rank 的 NVLS 大小（通道数 * 2 * 缓冲区大小）
  nvlsTotalSize = nvlsPerRankSize * nHeads;  // 计算总 NVLS 大小

  INFO(NCCL_INIT | NCCL_NVLS, "NVLS comm %p headRank %d nHeads %d nvlsRanks %d buffSize %zu nvlsPerRankSize %zu nvlsTotalSize %zu",
       comm, headRank, nHeads, comm->localRanks, buffSize, nvlsPerRankSize, nvlsTotalSize);  // 记录 NVLS 缓冲区设置信息

  NCCLCHECKGOTO(nvlsAllocateMem(comm, &resources->accessDesc, nvlsTotalSize, &resources->ucBuffHandle, &resources->mcBuffHandle, (void**)&resources->ucBuff, (void**)&resources->mcBuff, &resources->buffUCSize, &resources->buffMCSize), res, fail);  // 分配 NVLS 内存

  NCCLCHECKGOTO(ncclStrongStreamAcquire(ncclCudaGraphNone(), &comm->sharedRes->hostStream, /*concurrent=*/false, &hostStream), res, fail);  // 获取主机流
  NCCLCHECKGOTO(ncclStrongStreamAcquire(ncclCudaGraphNone(), &comm->sharedRes->deviceStream, /*concurrent=*/false, &deviceStream), res, fail);  // 获取设备流
  for (int h = 0; h < nHeads; h++) {  // 遍历所有头节点
    int nvlsPeer = comm->nRanks + 1 + h;  // 计算 NVLS 对等节点索引（在 nRanks 之后）
    for (int c = 0; c < nChannels; c++) {  // 遍历所有通道
      struct ncclChannel* channel = comm->channels + c;  // 获取当前通道
      struct ncclChannelPeer* peer = channel->peers[nvlsPeer];  // 获取 NVLS 对等节点

      // Reduce UC -> MC
      // 归约操作：从 UC（单播）到 MC（多播）

      peer->send[1].conn.buffs[NCCL_PROTO_SIMPLE] = resources->ucBuff + (h * 2 * nChannels + c) * buffSize;  // 设置发送缓冲区（方向 1，归约阶段）
      peer->recv[0].conn.buffs[NCCL_PROTO_SIMPLE] = resources->mcBuff + (h * 2 * nChannels + c) * buffSize;  // 设置接收缓冲区（方向 0，归约阶段）

      // Broadcast MC -> UC
      // 广播操作：从 MC（多播）到 UC（单播）

      peer->recv[1].conn.buffs[NCCL_PROTO_SIMPLE] = resources->ucBuff + ((h * 2 + 1) * nChannels + c) * buffSize;  // 设置接收缓冲区（方向 1，广播阶段）
      peer->send[0].conn.buffs[NCCL_PROTO_SIMPLE] = resources->mcBuff + ((h * 2 + 1) * nChannels + c) * buffSize;  // 设置发送缓冲区（方向 0，广播阶段）

      // 将连接信息从主机复制到设备（send[0]）
      CUDACHECKGOTO(cudaMemcpyAsync(&comm->channels[c].devPeersHostPtr[nvlsPeer]->send[0], &peer->send[0].conn, sizeof(struct ncclConnInfo), cudaMemcpyHostToDevice, hostStream), res, fail);
      // 将连接信息从主机复制到设备（recv[0]）
      CUDACHECKGOTO(cudaMemcpyAsync(&comm->channels[c].devPeersHostPtr[nvlsPeer]->recv[0], &peer->recv[0].conn, sizeof(struct ncclConnInfo), cudaMemcpyHostToDevice, hostStream), res, fail);
      // 将连接信息从主机复制到设备（send[1]）
      CUDACHECKGOTO(cudaMemcpyAsync(&comm->channels[c].devPeersHostPtr[nvlsPeer]->send[1], &peer->send[1].conn, sizeof(struct ncclConnInfo), cudaMemcpyHostToDevice, hostStream), res, fail);
      // 将连接信息从主机复制到设备（recv[1]）
      CUDACHECKGOTO(cudaMemcpyAsync(&comm->channels[c].devPeersHostPtr[nvlsPeer]->recv[1], &peer->recv[1].conn, sizeof(struct ncclConnInfo), cudaMemcpyHostToDevice, hostStream), res, fail);
    }
  }

  NCCLCHECKGOTO(ncclStreamWaitStream(deviceStream, hostStream, comm->sharedRes->scratchEvent), res, fail);  // 设备流等待主机流完成
  NCCLCHECKGOTO(ncclStrongStreamRelease(ncclCudaGraphNone(), &comm->sharedRes->deviceStream, /*concurrent=*/false), res, fail);  // 释放设备流
  NCCLCHECKGOTO(ncclStrongStreamRelease(ncclCudaGraphNone(), &comm->sharedRes->hostStream, /*concurrent=*/false), res, fail);  // 释放主机流
  // For now, the barrier is a must that guarantees all buffers are mc-mapped before accessing peer's buffer
  // 目前，屏障是必须的，保证在访问对等节点的缓冲区之前所有缓冲区都已 MC 映射

  NCCLCHECKGOTO(bootstrapIntraNodeBarrier(comm->bootstrap, comm->localRankToRank, comm->localRank, comm->localRanks, comm->localRankToRank[0]), res, fail);  // 节点内屏障同步
  comm->nvlsResources->inited = true;  // 标记 NVLS 资源已初始化

exit:  // 正常退出标签
  return res;  // 返回结果
fail:  // 失败处理标签
  comm->nvlsResources->inited = false;  // 标记 NVLS 资源未初始化
  goto exit;  // 跳转到退出标签
}

// NVLS 设置函数，设置 NVLS 通信所需的资源
// 参数说明：
//   comm: NCCL 通信上下文
//   parent: 父通信上下文（用于资源共享）
ncclResult_t ncclNvlsSetup(struct ncclComm* comm, struct ncclComm* parent) {
  ncclResult_t res = ncclSuccess;  // 返回值初始化
  size_t typeSize;  // 类型大小
  char shmPath[sizeof("/dev/shm/nccl-XXXXXX")];  // 共享内存路径
  uintptr_t *nvlsShmem = NULL;  // NVLS 共享内存指针
  bool nvlsShare = parent && parent->nvlsSupport && parent->shareResources && parent->localRanks == comm->localRanks;  // 判断是否可以共享 NVLS 资源

  if (comm->nvlsSupport == 0 || comm->nvlsChannels == 0) return ncclSuccess;  // 如果不支持 NVLS 或通道数为 0，直接返回

  comm->nvlsChunkSize = ncclParamNvlsChunkSize();  // 获取 NVLS 块大小参数
  if (nvlsShare) {  // 如果可以共享 NVLS 资源
    /* reuse NVLS resources */
    /* 重用 NVLS 资源 */

    comm->nvlsChannels = std::min(comm->nvlsChannels, parent->nvlsResources->nChannels);  // 通道数取最小值
    for (int c = 0; c < comm->nvlsChannels; c++) {  // 遍历所有通道
      NCCLCHECKGOTO(initNvlsChannel(comm, c, parent, true), res, fail);  // 初始化 NVLS 通道（从父通信上下文继承）
    }

    comm->nvlsResources = parent->nvlsResources;  // 共享 NVLS 资源
    ncclAtomicRefCountIncrement(&parent->nvlsResources->refCount);  // 增加资源引用计数
  } else {  // 如果不能共享 NVLS 资源
    struct ncclNvlsSharedRes* resources = NULL;  // NVLS 共享资源指针
    int nHeads = comm->channels[0].nvls.nHeads;  // 头节点数量
    int nChannels = comm->nvlsChannels;  // 通道数量
    size_t memSize = 64;  // 内存大小（用于 credit）
    size_t creditSize = nChannels * 2 * memSize * nHeads;  // credit 总大小（通道数 * 2 方向 * 每个方向的大小 * 头节点数）
    int nvlsStepSize = comm->nvlsChunkSize;  // NVLS 步长大小
    cudaStream_t hostStream, deviceStream;  // 主机流和设备流

    NCCLCHECKGOTO(ncclCalloc(&comm->nvlsResources, 1), res, fail);  // 分配 NVLS 资源结构体
    comm->nvlsResources->inited = false;  // 标记未初始化
    comm->nvlsResources->refCount = 1;  // 设置引用计数为 1
    comm->nvlsResources->nChannels = comm->nvlsChannels;  // 设置通道数
    comm->nvlsResources->nHeads = nHeads;  // 设置头节点数
    resources = comm->nvlsResources;  // 保存资源指针

    if (parent && parent->nvlsSupport && parent->shareResources) {  // 如果有父通信上下文且支持资源共享
      /* ranks on other nodes might share the NVLS resources, we need to cap nvlsChannels
       * to make sure nvlsChannels match for each rank. */
      /* 其他节点上的 rank 可能共享 NVLS 资源，我们需要限制 nvlsChannels
       * 以确保每个 rank 的 nvlsChannels 匹配。 */

      comm->nvlsChannels = std::min(comm->nvlsChannels, parent->nvlsResources->nChannels);  // 通道数取最小值
    }
    comm->nvlsResources->nChannels = comm->nvlsChannels;  // 更新通道数

    for (int c = 0; c < nChannels; c++) {  // 遍历所有通道
      NCCLCHECKGOTO(initNvlsChannel(comm, c, NULL, false), res, fail);  // 初始化 NVLS 通道（新建）
    }

    memset(&resources->accessDesc, 0, sizeof(resources->accessDesc));  // 清零访问描述符
    resources->accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;  // 设置读写权限
    resources->accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;  // 设置位置类型为设备
    resources->accessDesc.location.id = comm->cudaDev;  // 设置设备 ID
    resources->dev = comm->cudaDev;  // 保存设备号

    NCCLCHECKGOTO(nvlsAllocateMem(comm, &resources->accessDesc, creditSize, &resources->ucCreditHandle, &resources->mcCreditHandle, (void**)&resources->ucCredit, (void**)&resources->mcCredit, &resources->creditUCSize, &resources->creditMCSize), res, fail);  // 分配 credit 内存

    // Set up head and tail only for now
    // 目前只设置 head 和 tail

    NCCLCHECKGOTO(ncclStrongStreamAcquire(ncclCudaGraphNone(), &comm->sharedRes->hostStream, /*concurrent=*/false, &hostStream), res, fail);  // 获取主机流
    NCCLCHECKGOTO(ncclStrongStreamAcquire(ncclCudaGraphNone(), &comm->sharedRes->deviceStream, /*concurrent=*/false, &deviceStream), res, fail);  // 获取设备流
    for (int h = 0; h < nHeads; h++) {  // 遍历所有头节点
      int nvlsPeer = comm->nRanks + 1 + h;  // 计算 NVLS 对等节点索引
      for (int c = 0; c < nChannels; c++) {  // 遍历所有通道
        struct ncclChannel* channel = comm->channels + c;  // 获取当前通道
        char* mem = NULL;  // 内存指针
        struct ncclChannelPeer* peer = channel->peers[nvlsPeer];  // 获取 NVLS 对等节点

        // Reduce UC -> MC
        // 归约操作：从 UC（单播）到 MC（多播）

        mem = resources->ucCredit + (h * 2 * nChannels + c) * memSize;  // 计算 UC credit 内存位置
        peer->send[1].transportComm = &nvlsTransport.send;  // 设置传输通信
        peer->send[1].conn.buffs[NCCL_PROTO_SIMPLE] = NULL;  // 设置缓冲区为 NULL（稍后设置）
        peer->send[1].conn.head = (uint64_t*)mem;  // 设置 head 指针
        peer->send[1].conn.tail = (uint64_t*)(mem + memSize / 2);  // 设置 tail 指针（后半部分）
        peer->send[1].conn.stepSize = nvlsStepSize;  // 设置步长大小
        mem = resources->mcCredit + (h * 2 * nChannels + c) * memSize;  // 计算 MC credit 内存位置
        peer->recv[0].transportComm = &nvlsTransport.recv;  // 设置传输通信
        peer->recv[0].conn.buffs[NCCL_PROTO_SIMPLE] = NULL;  // 设置缓冲区为 NULL
        peer->recv[0].conn.head = (uint64_t*)mem;  // 设置 head 指针
        peer->recv[0].conn.tail = (uint64_t*)(mem + memSize / 2);  // 设置 tail 指针
        peer->recv[0].conn.stepSize = nvlsStepSize;  // 设置步长大小
        peer->recv[0].conn.flags |= NCCL_NVLS_MIN_POLL;  // 设置最小轮询标志

        // Broadcast MC -> UC
        // 广播操作：从 MC（多播）到 UC（单播）

        mem = resources->ucCredit + ((h * 2 + 1) * nChannels + c) * memSize;  // 计算 UC credit 内存位置
        peer->recv[1].transportComm = &nvlsTransport.recv;  // 设置传输通信
        peer->recv[1].conn.buffs[NCCL_PROTO_SIMPLE] = NULL;  // 设置缓冲区为 NULL
        peer->recv[1].conn.head = (uint64_t*)mem;  // 设置 head 指针
        peer->recv[1].conn.tail = (uint64_t*)(mem + memSize / 2);  // 设置 tail 指针
        peer->recv[1].conn.stepSize = nvlsStepSize;  // 设置步长大小
        mem = resources->mcCredit + ((h * 2 + 1) * nChannels + c) * memSize;  // 计算 MC credit 内存位置
        peer->send[0].transportComm = &nvlsTransport.send;  // 设置传输通信
        peer->send[0].conn.buffs[NCCL_PROTO_SIMPLE] = NULL;  // 设置缓冲区为 NULL
        peer->send[0].conn.head = (uint64_t*)mem;  // 设置 head 指针
        peer->send[0].conn.tail = (uint64_t*)(mem + memSize / 2);  // 设置 tail 指针
        peer->send[0].conn.stepSize = nvlsStepSize;  // 设置步长大小
        peer->send[0].conn.flags |= NCCL_NVLS_MIN_POLL;  // 设置最小轮询标志

        // 将连接信息从主机复制到设备（send[0]）
        CUDACHECKGOTO(cudaMemcpyAsync(&comm->channels[c].devPeersHostPtr[nvlsPeer]->send[0], &peer->send[0].conn, sizeof(struct ncclConnInfo), cudaMemcpyHostToDevice, hostStream), res, fail);
        // 将连接信息从主机复制到设备（recv[0]）
        CUDACHECKGOTO(cudaMemcpyAsync(&comm->channels[c].devPeersHostPtr[nvlsPeer]->recv[0], &peer->recv[0].conn, sizeof(struct ncclConnInfo), cudaMemcpyHostToDevice, hostStream), res, fail);
        // 将连接信息从主机复制到设备（send[1]）
        CUDACHECKGOTO(cudaMemcpyAsync(&comm->channels[c].devPeersHostPtr[nvlsPeer]->send[1], &peer->send[1].conn, sizeof(struct ncclConnInfo), cudaMemcpyHostToDevice, hostStream), res, fail);
        // 将连接信息从主机复制到设备（recv[1]）
        CUDACHECKGOTO(cudaMemcpyAsync(&comm->channels[c].devPeersHostPtr[nvlsPeer]->recv[1], &peer->recv[1].conn, sizeof(struct ncclConnInfo), cudaMemcpyHostToDevice, hostStream), res, fail);
      }
    }
    NCCLCHECKGOTO(ncclStreamWaitStream(deviceStream, hostStream, comm->sharedRes->scratchEvent), res, fail);  // 设备流等待主机流完成
    NCCLCHECKGOTO(ncclStrongStreamRelease(ncclCudaGraphNone(), &comm->sharedRes->hostStream, /*concurrent=*/false), res, fail);  // 释放主机流
    NCCLCHECKGOTO(ncclStrongStreamRelease(ncclCudaGraphNone(), &comm->sharedRes->deviceStream, /*concurrent=*/false), res, fail);  // 释放设备流
  }

  // MNNVL does not support NVLS buffer registration
  // MNNVL 不支持 NVLS 缓冲区注册

  if (!comm->MNNVL && comm->nvlsResources->nvlsShmemHandle == NULL) {  // 如果不是 MNNVL 且共享内存句柄为空
    /* create shared memory for fast NVLS buffer registration */
    /* 创建共享内存用于快速 NVLS 缓冲区注册 */

    typeSize = DIVUP(sizeof(struct localRegData) << 1, CACHE_LINE_SIZE) * CACHE_LINE_SIZE;  // 计算对齐后的类型大小

    if (comm->localRank == 0) {  // 如果是本地 rank 0
      shmPath[0] = '\0';  // 清空共享内存路径
      NCCLCHECKGOTO(ncclShmOpen(shmPath, sizeof(shmPath), (CACHE_LINE_SIZE * comm->localRanks + typeSize * comm->localRanks) * 2, (void**)&nvlsShmem, NULL, comm->localRanks - 1, &comm->nvlsResources->nvlsShmemHandle), res, fail);  // 创建共享内存
      NCCLCHECKGOTO(bootstrapIntraNodeBroadcast(comm->bootstrap, comm->localRankToRank, comm->localRank, comm->localRanks, 0, shmPath, sizeof(shmPath)), res, fail);  // 广播共享内存路径
    } else {  // 如果不是本地 rank 0
      NCCLCHECKGOTO(bootstrapIntraNodeBroadcast(comm->bootstrap, comm->localRankToRank, comm->localRank, comm->localRanks, 0, shmPath, sizeof(shmPath)), res, fail);  // 接收广播的共享内存路径
      NCCLCHECKGOTO(ncclShmOpen(shmPath, sizeof(shmPath), (CACHE_LINE_SIZE * comm->localRanks + typeSize * comm->localRanks) * 2, (void**)&nvlsShmem, NULL, -1, &comm->nvlsResources->nvlsShmemHandle), res, fail);  // 打开共享内存
    }
    /* need 2 pools and a shared counter for shmem-based collectives */
    /* 需要 2 个内存池和一个共享计数器用于基于共享内存的集合操作 */

    comm->nvlsResources->nvlsShmem.cnt[0] = (size_t*)nvlsShmem;  // 设置池 0 的计数器
    comm->nvlsResources->nvlsShmem.ptr[0] = (void*)((char*)comm->nvlsResources->nvlsShmem.cnt[0] + CACHE_LINE_SIZE * comm->localRanks);  // 设置池 0 的指针
    comm->nvlsResources->nvlsShmem.cnt[1] = (size_t*)((char*)comm->nvlsResources->nvlsShmem.ptr[0] + typeSize * comm->localRanks);  // 设置池 1 的计数器
    comm->nvlsResources->nvlsShmem.ptr[1] = (void*)((char*)comm->nvlsResources->nvlsShmem.cnt[1] + CACHE_LINE_SIZE * comm->localRanks);  // 设置池 1 的指针
    comm->nvlsResources->nvlsShmem.round = 0;  // 设置轮次为 0
    comm->nvlsResources->nvlsShmem.maxTypeSize = typeSize;  // 设置最大类型大小
  }

exit:  // 正常退出标签
  return res;  // 返回结果
fail:  // 失败处理标签
  comm->nvlsSupport = 0;  // 关闭 NVLS 支持
  goto exit;  // 跳转到退出标签
}

// NVLS 释放函数，释放 NVLS 相关资源
// 参数说明：
//   comm: NCCL 通信上下文
ncclResult_t ncclNvlsFree(struct ncclComm* comm) {
  struct ncclNvlsSharedRes* resources = (struct ncclNvlsSharedRes*)comm->nvlsResources;  // 获取 NVLS 资源
  if (resources == NULL) return ncclSuccess;  // 如果资源为空，直接返回

  if (ncclAtomicRefCountDecrement(&resources->refCount) == 0) {  // 如果引用计数减为 0
    if (!comm->MNNVL && resources->nvlsShmemHandle)  // 如果不是 MNNVL 且共享内存句柄有效
      NCCLCHECK(ncclShmClose(resources->nvlsShmemHandle));  // 关闭共享内存

    if (resources->ucCredit || resources->mcCredit) {  // 如果 credit 内存有效
      NCCLCHECK(nvlsGroupUnbind(comm, resources->creditUCSize, &resources->mcCreditHandle));  // 解绑 credit MC 内存
      NCCLCHECK(nvlsGroupUnmapMem(comm, resources->creditUCSize, resources->ucCredit, &resources->ucCreditHandle, resources->creditMCSize, resources->mcCredit, &resources->mcCreditHandle));  // 取消映射 credit 内存
    }

    if (comm->nvlsResources->inited) {  // 如果 NVLS 资源已初始化
      NCCLCHECK(nvlsGroupUnbind(comm, resources->buffUCSize, &resources->mcBuffHandle));  // 解绑缓冲区 MC 内存
      NCCLCHECK(nvlsGroupUnmapMem(comm, resources->buffUCSize, resources->ucBuff, &resources->ucBuffHandle, resources->buffMCSize, resources->mcBuff, &resources->mcBuffHandle));  // 取消映射缓冲区内存
    }
    free(resources);  // 释放资源结构体
    comm->nvlsResources = NULL;  // 设置资源指针为 NULL
  }
  return ncclSuccess;  // 返回成功状态
}

// 尝试注册缓冲区到 NVLS 的函数
// 参数说明：
//   comm: NCCL 通信上下文
//   userBuff: 用户缓冲区地址
//   buffSize: 缓冲区大小
//   regAddr: 输出参数，返回注册后的地址
//   regUsed: 输出参数，返回是否使用了注册的缓冲区
ncclResult_t tryRegisterBuffer(struct ncclComm *comm, uintptr_t userBuff, size_t buffSize, CUdeviceptr *regAddr, int *regUsed) {
  ncclResult_t ret = ncclSuccess;  // 返回值初始化
  struct ncclReg *regRecord = NULL;  // 注册记录指针
  CUdeviceptr regPtr = 0;  // 注册后的指针
  CUmulticastObjectProp mcprop;  // 多播对象属性
  CUmemAllocationProp ucprop;  // 单播内存分配属性
  char shareableHandle[NVLS_HANDLE_SIZE];  // 共享句柄缓冲区
  CUmemGenericAllocationHandle mcHandle;  // 多播句柄
  size_t minSize = SIZE_MAX;  // 最小大小，初始化为最大值
  struct localRegData* regData = NULL;  // 本地注册数据数组
  cudaPointerAttributes attr;  // CUDA 指针属性
  size_t ucgran, mcgran, ucsize, mcsize;  // UC 和 MC 的粒度和大小

  NCCLCHECKGOTO(ncclCalloc(&regData, comm->localRanks), ret, fail);  // 分配本地注册数据数组

  if (userBuff) {  // 如果用户缓冲区有效
    NCCLCHECKGOTO(ncclRegFind(comm, (void*)userBuff, buffSize, &regRecord), ret, fail);  // 查找已有的注册记录
    if (regRecord) {  // 如果找到注册记录
      CUDACHECKGOTO(cudaPointerGetAttributes(&attr, (void*)regRecord->begAddr), ret, fail);  // 获取指针属性
      if (attr.type == cudaMemoryTypeDevice) {  // 如果是设备内存
        size_t regSize = regRecord->endAddr - regRecord->begAddr;  // 计算已注册内存大小
        memset(&mcprop, 0, sizeof(CUmulticastObjectProp));  // 清零多播属性
        mcprop.numDevices = comm->localRanks;  // 设置设备数
        mcprop.handleTypes = ncclCuMemHandleType;  // 设置句柄类型
        mcprop.flags = 0;  // 设置标志
        mcprop.size = regSize;  // 设置大小
        CUCHECKGOTO(cuMulticastGetGranularity(&mcgran, &mcprop, CU_MULTICAST_GRANULARITY_RECOMMENDED), ret, fail);  // 获取 MC 粒度

        memset(&ucprop, 0, sizeof(CUmemAllocationProp));  // 清零 UC 分配属性
        ucprop.type = CU_MEM_ALLOCATION_TYPE_PINNED;  // 设置类型为固定内存
        ucprop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;  // 设置位置类型为设备
        ucprop.location.id = comm->cudaDev;  // 设置设备 ID
        ucprop.requestedHandleTypes = ncclCuMemHandleType;  // 设置请求的句柄类型
        CUCHECKGOTO(cuMemGetAllocationGranularity(&ucgran, &ucprop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED), ret, fail);  // 获取 UC 粒度

        if (regRecord->begAddr % ucgran == 0) {  // 如果起始地址对齐到 UC 粒度
          if (regSize % ucgran != 0) {  // 如果大小未对齐到 UC 粒度
            regRecord->regUCSize = ALIGN_SIZE(regSize, ucgran);  // 对齐 UC 大小
          } else {  // 如果大小已对齐
            regRecord->regUCSize = regSize;  // 直接使用原始大小
          }
          regRecord->state |= NVLS_REG_POSSIBLE;  // 标记可以注册到 NVLS
          memcpy(&regData[comm->localRank].reg, regRecord, sizeof(struct ncclReg));  // 复制注册记录
          regData[comm->localRank].offset = userBuff - regRecord->begAddr;  // 计算偏移量
        }
      }

      if ((regRecord->state & NVLS_REG_POSSIBLE) == 0) {  // 如果不能注册到 NVLS
        regRecord->state |= NVLS_REG_NO_SUPPORT;  // 标记不支持 NVLS 注册
      }
    }
  }

  NCCLCHECKGOTO(ncclShmemAllgather(comm, &comm->nvlsResources->nvlsShmem, regData + comm->localRank, regData, sizeof(struct localRegData)), ret, fail);  // 收集所有本地 rank 的注册数据

  for (int i = 0; i < comm->localRanks; ++i) {  // 遍历所有本地 rank
    if ((regData[i].reg.state & NVLS_REG_POSSIBLE) == 0) {  // 如果某个 rank 不能注册
      goto fail;  // 跳转到失败处理
    }
    /* get minimal reg size of nvls buffers */
    /* 获取 NVLS 缓冲区的最小注册大小 */

    if (minSize > regData[i].reg.regUCSize)  // 如果当前 rank 的 UC 大小更小
      minSize = regData[i].reg.regUCSize;  // 更新最小大小
  }

  /* start registration */
  /* 开始注册 */

  mcsize = ucsize = minSize;  // 设置 MC 和 UC 大小为最小大小
  mcprop.size = minSize;  // 设置多播属性的大小
  CUCHECKGOTO(cuMulticastGetGranularity(&mcgran, &mcprop, CU_MULTICAST_GRANULARITY_RECOMMENDED), ret, fail);  // 获取 MC 粒度
  ALIGN_SIZE(mcsize, mcgran);  // 对齐 MC 大小
  mcprop.size = mcsize;  // 更新多播属性的大小

  if (comm->localRank == 0) {  // 如果是本地 rank 0
    NCCLCHECKGOTO(ncclNvlsGroupCreate(comm, &mcprop, comm->localRank, comm->localRanks, &mcHandle, shareableHandle), ret, fail);  // 创建多播组
    NCCLCHECKGOTO(bootstrapIntraNodeBroadcast(comm->bootstrap, comm->localRankToRank, comm->localRank, comm->localRanks, 0, shareableHandle, NVLS_HANDLE_SIZE), ret, fail);  // 广播共享句柄
  } else {  // 如果不是本地 rank 0
    NCCLCHECKGOTO(bootstrapIntraNodeBroadcast(comm->bootstrap, comm->localRankToRank, comm->localRank, comm->localRanks, 0, shareableHandle, NVLS_HANDLE_SIZE), ret, fail);  // 接收广播的共享句柄
    NCCLCHECKGOTO(ncclNvlsGroupConnect(comm, shareableHandle, comm->localRankToRank[0], &mcHandle), ret, fail);  // 连接到多播组
  }

  CUCHECKGOTO(cuMulticastAddDevice(mcHandle, comm->nvlsResources->dev), ret, fail);  // 添加设备到多播组
  // Coverity complains that regRecord could be NULL.  That won't in practice be the case because we've already checked
  // (regData[i].reg.state & NVLS_REG_POSSIBLE) of all local ranks, which would catch it and bail out.
  // Coverity 抱怨 regRecord 可能为 NULL。实际上不会发生这种情况，因为我们已经检查了
  // 所有本地 rank 的 (regData[i].reg.state & NVLS_REG_POSSIBLE)，这会捕获并退出。

  // coverity[var_deref_op]
  CUCHECKGOTO(cuMulticastBindAddr(mcHandle, 0, (CUdeviceptr)regRecord->begAddr, ucsize, 0), ret, fail);  // 绑定地址到多播组

  // Create a VA for the NVLS
  // 为 NVLS 创建虚拟地址

  CUCHECKGOTO(cuMemAddressReserve(&regPtr, mcsize, mcgran, 0U, 0), ret, fail);  // 保留 MC 虚拟地址空间
  // Map the VA locally
  // 本地映射虚拟地址

  CUCHECKGOTO(cuMemMap(regPtr, mcsize, 0, mcHandle, 0), ret, fail);  // 映射 MC 内存
  CUCHECKGOTO(cuMemSetAccess(regPtr, mcsize, &comm->nvlsResources->accessDesc, 1), ret, fail);  // 设置 MC 内存访问权限

  regRecord->regAddr = regPtr;  // 保存注册后的地址
  regRecord->regUCSize = ucsize;  // 保存 UC 大小
  regRecord->regMCSize = mcsize;  // 保存 MC 大小
  regRecord->dev = comm->nvlsResources->dev;  // 保存设备号
  regRecord->mcHandle = mcHandle;  // 保存多播句柄
  regRecord->state |= NVLS_REG_COMPLETE;  // 标记注册完成
  /* get all buffer addresses */
  /* 获取所有缓冲区地址 */

  regRecord->caddrs[comm->localRank] = regRecord->begAddr;  // 保存本地 rank 的地址
  NCCLCHECKGOTO(ncclShmemAllgather(comm, &comm->nvlsResources->nvlsShmem, regRecord->caddrs + comm->localRank, regRecord->caddrs, sizeof(uintptr_t)), ret, fail);  // 收集所有 rank 的地址

  /* Although registration is done, we still need to check whether the offsets are same among ranks. */
  /* 虽然注册已完成，我们仍需检查各 rank 间的偏移量是否相同。 */

  for (int i = 0; i < comm->localRanks - 1; ++i) {  // 遍历所有本地 rank（除最后一个）
    if (regData[i].offset != regData[i + 1].offset) {  // 如果偏移量不同
      goto fail;  // 跳转到失败处理
    }
  }

  *regAddr = (uintptr_t)regPtr + regData[comm->localRank].offset;  // 计算并返回注册后的地址
  *regUsed = 1;  // 标记使用了注册的缓冲区
exit:  // 正常退出标签
  free(regData);  // 释放注册数据数组
  return ret;  // 返回结果
fail:  // 失败处理标签
  *regUsed = 0;  // 标记未使用注册的缓冲区
  goto exit;  // 跳转到退出标签
}

// NVLS 注册缓冲区静态函数
// 参数说明：
//   comm: NCCL 通信上下文
//   sendbuff: 发送缓冲区
//   recvbuff: 接收缓冲区
//   sendbuffSize: 发送缓冲区大小
//   recvbuffSize: 接收缓冲区大小
//   sendRegRecord: 发送端注册记录
//   recvRegRecord: 接收端注册记录
//   outRegBufUsed: 输出参数，返回是否使用了注册的缓冲区
//   outRegBufSend: 输出参数，返回注册后的发送缓冲区地址
//   outRegBufRecv: 输出参数，返回注册后的接收缓冲区地址
static ncclResult_t nvlsRegisterBuffer(struct ncclComm *comm, const void *sendbuff, void *recvbuff, size_t sendbuffSize, size_t recvbuffSize, struct ncclReg *sendRegRecord, struct ncclReg *recvRegRecord, int *outRegBufUsed, void **outRegBufSend, void **outRegBufRecv) {
  ncclResult_t ret = ncclSuccess;  // 返回值初始化
  int regBufUsed = 0;  // 是否使用注册缓冲区的标志
  struct localRegData *regData = NULL;  // 本地注册数据数组
  bool sendNeedReg = false, recvNeedReg = false;  // 发送和接收端是否需要注册的标志
  CUdeviceptr regSendPtr = 0;  // 注册后的发送指针
  CUdeviceptr regRecvPtr = 0;  // 注册后的接收指针

  NCCLCHECKGOTO(ncclCalloc(&regData, comm->localRanks * 2), ret, fail);  // 分配本地注册数据数组（发送和接收各一份）

  if (sendRegRecord) {  // 如果发送端有注册记录
    memcpy(&regData[comm->localRank * 2].reg, sendRegRecord, sizeof(struct ncclReg));  // 复制发送端注册记录
    regData[comm->localRank * 2].offset = (uintptr_t)sendbuff - sendRegRecord->begAddr;  // 计算发送端偏移量
  }

  if (recvRegRecord) {  // 如果接收端有注册记录
    memcpy(&regData[comm->localRank * 2 + 1].reg, recvRegRecord, sizeof(struct ncclReg));  // 复制接收端注册记录
    regData[comm->localRank * 2 + 1].offset = (uintptr_t)recvbuff - recvRegRecord->begAddr;  // 计算接收端偏移量
  }

  NCCLCHECKGOTO(ncclShmemAllgather(comm, &comm->nvlsResources->nvlsShmem, regData + comm->localRank * 2, regData, sizeof(struct localRegData) * 2), ret, fail);  // 收集所有本地 rank 的注册数据

  /* first check whether all local ranks find their registered buffer */
  /* 首先检查所有本地 rank 是否找到它们的注册缓冲区 */

  for (int i = 0; i < comm->localRanks; ++i) {  // 遍历所有本地 rank
    if ((regData[i * 2].reg.state & NVLS_REG_COMPLETE) == 0 || regData[comm->localRank * 2].reg.caddrs[i] != regData[i * 2].reg.begAddr) {  // 如果发送端未完成注册或地址不匹配
      sendNeedReg = true;  // 标记发送端需要注册
    }

    if ((regData[i * 2 + 1].reg.state & NVLS_REG_COMPLETE) == 0 || regData[comm->localRank * 2 + 1].reg.caddrs[i] != regData[i * 2 + 1].reg.begAddr) {  // 如果接收端未完成注册或地址不匹配
      recvNeedReg = true;  // 标记接收端需要注册
    }

    if ((regData[i * 2].reg.state & NVLS_REG_NO_SUPPORT) || (regData[i * 2 + 1].reg.state & NVLS_REG_NO_SUPPORT)) {  // 如果发送或接收端不支持注册
      goto fail;  // 跳转到失败处理
    }
  }

  if (sendNeedReg == false) {  // 如果发送端不需要注册
    for (int i = 0; i < comm->localRanks - 1; ++i) {  // 遍历所有本地 rank（除最后一个）
      if (regData[i * 2].offset != regData[(i + 1) * 2].offset) {  // 如果偏移量不同
        /* offset are different, we cannot apply user buffer registration */
        /* 偏移量不同，我们无法应用用户缓冲区注册 */

        goto fail;  // 跳转到失败处理
      }
    }

    /* reuse previous registered buffer if possible */
    /* 如果可能，重用之前注册的缓冲区 */

    if (!sendNeedReg)  // 如果发送端不需要注册
      regSendPtr = (CUdeviceptr)((uintptr_t)sendRegRecord->regAddr + regData[comm->localRank * 2].offset);  // 重用已注册的发送缓冲区
  }

  if (recvNeedReg == false) {  // 如果接收端不需要注册
    for (int i = 0; i < comm->localRanks - 1; ++i) {  // 遍历所有本地 rank（除最后一个）
      if (regData[i * 2 + 1].offset != regData[(i + 1) * 2 + 1].offset) {  // 如果偏移量不同
        goto fail;  // 跳转到失败处理
      }
    }

    if (!recvNeedReg)  // 如果接收端不需要注册
      regRecvPtr = (CUdeviceptr)((uintptr_t)recvRegRecord->regAddr + regData[comm->localRank * 2 + 1].offset);  // 重用已注册的接收缓冲区
  }

  if ((!sendNeedReg || sendbuff == NULL) && (!recvNeedReg || recvbuff == NULL)) {  // 如果发送和接收端都不需要注册或缓冲区为空
    regBufUsed = 1;  // 标记使用了注册的缓冲区
    INFO(NCCL_REG, "rank %d reuse registered NVLS sendbuff %p, recvbuff %p, sendbuff size %ld, recvbuff size %ld, reg sendbuff %p, reg recvbuff %p", comm->rank, sendbuff, recvbuff, sendbuffSize, recvbuffSize, (void*)regSendPtr, (void*)regRecvPtr);  // 记录日志
    goto exit;  // 跳转到退出标签
  }

  /* Start Registration. Not found registered buffers, then check whether both send and recv buffer locate
   * in register request cache. */
  /* 开始注册。未找到注册的缓冲区，然后检查发送和接收缓冲区是否位于
   * 注册请求缓存中。 */

  if (sendNeedReg && sendbuff && sendbuffSize > 0) {  // 如果发送端需要注册且缓冲区有效
    tryRegisterBuffer(comm, (uintptr_t)sendbuff, sendbuffSize, &regSendPtr, &regBufUsed);  // 尝试注册发送缓冲区
    if (regBufUsed == 0) goto fail;  // 如果注册失败，跳转到失败处理
  }

  if (recvNeedReg && recvbuff && recvbuffSize > 0) {  // 如果接收端需要注册且缓冲区有效
    tryRegisterBuffer(comm, (uintptr_t)recvbuff, recvbuffSize, &regRecvPtr, &regBufUsed);  // 尝试注册接收缓冲区
    if (regBufUsed == 0) goto fail;  // 如果注册失败，跳转到失败处理
  }

  INFO(NCCL_REG, "rank %d successfully registered NVLS sendbuff %p, recvbuff %p, sendbuff size %ld, recvbuff size %ld, reg sendbuff %p, reg recvbuff %p", comm->rank, sendbuff, recvbuff, sendbuffSize, recvbuffSize, (void*)regSendPtr, (void*)regRecvPtr);  // 记录注册成功的日志

exit:  // 正常退出标签
  *outRegBufSend = (void*)regSendPtr;  // 返回注册后的发送缓冲区地址
  *outRegBufRecv = (void*)regRecvPtr;  // 返回注册后的接收缓冲区地址
  *outRegBufUsed = regBufUsed;  // 返回是否使用了注册的缓冲区
  free(regData);  // 释放注册数据数组
  return ncclSuccess;  // 返回成功状态
fail:  // 失败处理标签
  regBufUsed = 0;  // 标记未使用注册的缓冲区
  INFO(NCCL_REG, "rank %d failed to NVLS register sendbuff %p sendbuffSize %ld recvbuff %p recvbuffSize %ld", comm->rank, sendbuff, sendbuffSize, recvbuff, recvbuffSize);  // 记录失败的日志
  goto exit;  // 跳转到退出标签
}

// NVLS 本地注册缓冲区函数
// 参数说明：
//   comm: NCCL 通信上下文
//   sendbuff: 发送缓冲区
//   recvbuff: 接收缓冲区
//   sendbuffSize: 发送缓冲区大小
//   recvbuffSize: 接收缓冲区大小
//   outRegBufUsed: 输出参数，返回是否使用了注册的缓冲区
//   outRegBufSend: 输出参数，返回注册后的发送缓冲区地址
//   outRegBufRecv: 输出参数，返回注册后的接收缓冲区地址
ncclResult_t ncclNvlsLocalRegisterBuffer(struct ncclComm *comm, const void *sendbuff, void *recvbuff, size_t sendbuffSize, size_t recvbuffSize, int *outRegBufUsed, void **outRegBufSend, void **outRegBufRecv) {
  struct ncclReg *sendRegRecord = NULL;  // 发送端注册记录
  struct ncclReg *recvRegRecord = NULL;  // 接收端注册记录
  bool sendIsValid = false;  // 发送端是否有效的标志
  bool recvIsValid = false;  // 接收端是否有效的标志
  void *baseSend = NULL;  // 发送端基础地址
  void *baseRecv = NULL;  // 接收端基础地址
  size_t baseSendSize = 0;  // 发送端基础大小
  size_t baseRecvSize = 0;  // 接收端基础大小

  *outRegBufUsed = 0;  // 初始化为未使用注册缓冲区
  if (sendbuff) {  // 如果发送缓冲区有效
    NCCLCHECK(ncclRegFind(comm, sendbuff, sendbuffSize, &sendRegRecord));  // 查找发送端注册记录
    NCCLCHECK(ncclRegLocalIsValid(sendRegRecord, &sendIsValid));  // 检查发送端记录是否有效
    if (sendIsValid) {  // 如果发送端有效
      CUCHECK(cuMemGetAddressRange((CUdeviceptr *)&baseSend, &baseSendSize, (CUdeviceptr)sendbuff));  // 获取发送缓冲区的地址范围
      if ((uint64_t)baseSend + baseSendSize < (uint64_t)sendbuff + sendbuffSize) {  // 如果缓冲区跨越多个物理内存区域
        // the virtual address is backed by multiple physical memory regions, just fall back to non-UB path
        // 虚拟地址由多个物理内存区域支持，回退到非 UB（用户缓冲区）路径

        goto exit;  // 跳转到退出标签
      }
    }
  } else {  // 如果发送缓冲区为空
    sendIsValid = true;  // 标记为有效
  }

  if (recvbuff) {  // 如果接收缓冲区有效
    NCCLCHECK(ncclRegFind(comm, recvbuff, recvbuffSize, &recvRegRecord));  // 查找接收端注册记录
    NCCLCHECK(ncclRegLocalIsValid(recvRegRecord, &recvIsValid));  // 检查接收端记录是否有效
    if (recvIsValid) {  // 如果接收端有效
      CUCHECK(cuMemGetAddressRange((CUdeviceptr *)&baseRecv, &baseRecvSize, (CUdeviceptr)recvbuff));  // 获取接收缓冲区的地址范围
      if ((uint64_t)baseRecv + baseRecvSize < (uint64_t)recvbuff + recvbuffSize) {  // 如果缓冲区跨越多个物理内存区域
        // the virtual address is backed by multiple physical memory regions, just fall back to non-UB path
        // 虚拟地址由多个物理内存区域支持，回退到非 UB 路径

        goto exit;  // 跳转到退出标签
      }
    }
  } else {  // 如果接收缓冲区为空
    recvIsValid = true;  // 标记为有效
  }

  if (sendIsValid && recvIsValid)  // 如果发送和接收端都有效
    NCCLCHECK(nvlsRegisterBuffer(comm, sendbuff, recvbuff, sendbuffSize, recvbuffSize, sendRegRecord, recvRegRecord, outRegBufUsed, outRegBufSend, outRegBufRecv));  // 调用 NVLS 注册缓冲区函数

exit:  // 正常退出标签
  return ncclSuccess;  // 返回成功状态
}

// NVLS 清理回调结构体
struct ncclNvlsCleanupCallback {
  struct ncclCommCallback base;  // 基础回调结构体
  struct ncclReg *reg;  // 注册记录
  struct ncclComm *comm;  // 通信上下文
};

// NVLS 清理函数
// 参数说明：
//   comm: NCCL 通信上下文
//   cb: 回调对象
static ncclResult_t cleanupNvls(struct ncclComm* comm, struct ncclCommCallback* cb) {
  struct ncclNvlsCleanupCallback* obj = (struct ncclNvlsCleanupCallback*)cb;  // 转换回调对象类型
  NCCLCHECK(ncclCommGraphDeregister(obj->comm, obj->reg));  // 注销图注册
  free(obj);  // 释放回调对象
  return ncclSuccess;  // 返回成功状态
}

// NVLS 图注册缓冲区函数，用于 CUDA Graph 模式下的缓冲区注册
// 参数说明：
//   comm: NCCL 通信上下文
//   sendbuff: 发送缓冲区
//   recvbuff: 接收缓冲区
//   sendbuffSize: 发送缓冲区大小
//   recvbuffSize: 接收缓冲区大小
//   outRegBufUsed: 输出参数，返回是否使用了注册的缓冲区
//   outRegBufSend: 输出参数，返回注册后的发送缓冲区地址
//   outRegBufRecv: 输出参数，返回注册后的接收缓冲区地址
//   cleanupQueue: 清理队列
//   nCleanupQueueEltsAdded: 输出参数，返回添加的清理队列元素数量
ncclResult_t ncclNvlsGraphRegisterBuffer(
    struct ncclComm *comm, const void *sendbuff, void *recvbuff, size_t sendbuffSize, size_t recvbuffSize,
    int *outRegBufUsed, void **outRegBufSend, void **outRegBufRecv,
    struct ncclIntruQueue<struct ncclCommCallback, &ncclCommCallback::next>* cleanupQueue, int* nCleanupQueueEltsAdded
  ) {
  struct ncclNvlsCleanupCallback* sendRecord = NULL;  // 发送端清理回调
  struct ncclNvlsCleanupCallback* recvRecord = NULL;  // 接收端清理回调
  void *baseSend = NULL;  // 发送端基础地址
  void *baseRecv = NULL;  // 接收端基础地址
  size_t baseSendSize = 0;  // 发送端基础大小
  size_t baseRecvSize = 0;  // 接收端基础大小
  struct ncclReg *sendRegRecord = NULL;  // 发送端注册记录
  struct ncclReg *recvRegRecord = NULL;  // 接收端注册记录

  *outRegBufUsed = 0;  // 初始化为未使用注册缓冲区
  if (sendbuff) {  // 如果发送缓冲区有效
    CUCHECK(cuMemGetAddressRange((CUdeviceptr *)&baseSend, &baseSendSize, (CUdeviceptr)sendbuff));  // 获取发送缓冲区的地址范围
    if ((uint64_t)baseSend + baseSendSize < (uint64_t)sendbuff + sendbuffSize) {  // 如果缓冲区跨越多个物理内存区域
      // the virtual address is backed by multiple physical memory regions, just fall back to non-UB path
      // 虚拟地址由多个物理内存区域支持，回退到非 UB 路径

      goto exit;  // 跳转到退出标签
    }
    NCCLCHECK(ncclCommGraphRegister(comm, baseSend, baseSendSize, (void**)&sendRegRecord));  // 注册发送缓冲区到图
  }

  if (recvbuff) {  // 如果接收缓冲区有效
    CUCHECK(cuMemGetAddressRange((CUdeviceptr *)&baseRecv, &baseRecvSize, (CUdeviceptr)recvbuff));  // 获取接收缓冲区的地址范围
    if ((uint64_t)baseRecv + baseRecvSize < (uint64_t)recvbuff + recvbuffSize) {  // 如果缓冲区跨越多个物理内存区域
      // the virtual address is backed by multiple physical memory regions, just fall back to non-UB path
      // 虚拟地址由多个物理内存区域支持，回退到非 UB 路径

      goto exit;  // 跳转到退出标签
    }
    NCCLCHECK(ncclCommGraphRegister(comm, baseRecv, baseRecvSize, (void**)&recvRegRecord));  // 注册接收缓冲区到图
  }

  NCCLCHECK(nvlsRegisterBuffer(comm, sendbuff, recvbuff, sendbuffSize, recvbuffSize, sendRegRecord, recvRegRecord, outRegBufUsed, outRegBufSend, outRegBufRecv));  // 调用 NVLS 注册缓冲区函数

  if (*outRegBufUsed) {  // 如果使用了注册的缓冲区
    if (sendRegRecord) {  // 如果发送端注册记录有效
      sendRecord = (struct ncclNvlsCleanupCallback*)malloc(sizeof(struct ncclNvlsCleanupCallback));  // 分配发送端清理回调
      sendRecord->base.fn = cleanupNvls;  // 设置清理函数
      sendRecord->reg = sendRegRecord;  // 保存注册记录
      sendRecord->comm = comm;  // 保存通信上下文
      ncclIntruQueueEnqueue(cleanupQueue, (struct ncclCommCallback*)sendRecord);  // 将清理回调加入队列
      *nCleanupQueueEltsAdded += 1;  // 增加清理队列元素计数
    }

    if (recvRegRecord) {  // 如果接收端注册记录有效
      recvRecord = (struct ncclNvlsCleanupCallback*)malloc(sizeof(struct ncclNvlsCleanupCallback));  // 分配接收端清理回调
      recvRecord->base.fn = cleanupNvls;  // 设置清理函数
      recvRecord->reg = recvRegRecord;  // 保存注册记录
      recvRecord->comm = comm;  // 保存通信上下文
      ncclIntruQueueEnqueue(cleanupQueue, (struct ncclCommCallback*)recvRecord);  // 将清理回调加入队列
      *nCleanupQueueEltsAdded += 1;  // 增加清理队列元素计数
    }
  } else {  // 如果未使用注册的缓冲区
    if (sendbuff) NCCLCHECK(ncclCommGraphDeregister(comm, sendRegRecord));  // 注销发送端图注册
    if (recvbuff) NCCLCHECK(ncclCommGraphDeregister(comm, recvRegRecord));  // 注销接收端图注册
  }

exit:  // 正常退出标签
  return ncclSuccess;  // 返回成功状态
}

// NVLS 注册资源查询函数，用于查询推荐的通道数
// 参数说明：
//   comm: NCCL 通信上下文
//   info: 集合任务信息
//   recChannels: 输出参数，返回推荐的通道数
ncclResult_t ncclNvlsRegResourcesQuery(struct ncclComm* comm, struct ncclTaskColl* info, int* recChannels) {
  int factor;  // 因子，用于计算通道数
  ncclResult_t ret = ncclSuccess;  // 返回值初始化
  if (comm->nNodes == 1) {  // 如果是单节点场景
    if (info->func == ncclFuncReduceScatter) {  // 如果是 ReduceScatter 操作
      factor = (comm->compCap >= 100 ? 6 : 5) * 8;  // 根据计算能力设置因子（Blackwell: 6*8, 其他: 5*8）
      *recChannels = std::max(comm->config.minCTAs, std::min(comm->config.maxCTAs, DIVUP(factor, comm->nvlsResources->nHeads)));  // 计算推荐的通道数
    } else if (info->func == ncclFuncAllGather) {  // 如果是 AllGather 操作
      factor = 4 * 8;  // 设置因子为 4*8
      *recChannels = std::max(comm->config.minCTAs, std::min(comm->config.maxCTAs, DIVUP(factor, comm->nvlsResources->nHeads)));  // 计算推荐的通道数
    } else if (info->func == ncclFuncAllReduce) {  // 如果是 AllReduce 操作
      if (comm->compCap >= 100) {  // 如果计算能力 >= 100（Blackwell）
        factor = 8 * 8;  // 设置因子为 8*8
      } else {  // 对于其他架构
        factor = 4 * 8;  // 设置因子为 4*8
      }
      *recChannels = std::max(comm->config.minCTAs, std::min(comm->config.maxCTAs, DIVUP(factor, comm->nvlsResources->nHeads)));  // 计算推荐的通道数
    } else {  // 对于其他操作类型
      goto fail;  // 跳转到失败处理
    }
  } else {  // 如果是多节点场景
    // Further tweaks for Blackwell with NVLS registered buffers
    // 针对 Blackwell 架构使用 NVLS 注册缓冲区的进一步调整

    if (info->func == ncclFuncReduceScatter) {  // 如果是 ReduceScatter 操作
      factor = (comm->bandwidths[ncclFuncReduceScatter][NCCL_ALGO_NVLS][NCCL_PROTO_SIMPLE] > 400 ? 7 : 6) * 8;  // 根据带宽设置因子
      *recChannels = std::max(comm->config.minCTAs, std::min(comm->config.maxCTAs, DIVUP(factor, comm->nvlsResources->nHeads)));  // 计算推荐的通道数
    } else if (info->func == ncclFuncAllGather) {  // 如果是 AllGather 操作
      factor = 6 * 8;  // 设置因子为 6*8
      *recChannels = std::max(comm->config.minCTAs, std::min(comm->config.maxCTAs, DIVUP(factor, comm->nvlsResources->nHeads)));  // 计算推荐的通道数
    } else if (info->func == ncclFuncAllReduce) {  // 如果是 AllReduce 操作
      factor = (comm->compCap >= 100 ? 7 : 6) * 8;  // 根据计算能力设置因子
      *recChannels = std::max(comm->config.minCTAs, std::min(comm->config.maxCTAs, DIVUP(factor, comm->nvlsResources->nHeads)));  // 计算推荐的通道数
    } else {  // 对于其他操作类型
      goto fail;  // 跳转到失败处理
    }
  }

exit:  // 正常退出标签
  return ret;  // 返回结果
fail:  // 失败处理标签
  ret = ncclInvalidArgument;  // 设置返回值为无效参数错误
  goto exit;  // 跳转到退出标签
}

// 如果 CUDA 版本 < 12.1，使用以下存根函数
#else

/*
 * Pre CUDA 12.1 stubs
 * CUDA 12.1 之前的存根函数
 */

ncclResult_t ncclNvlsInit(struct ncclComm* comm) {
  comm->nvlsChannels = 0;  // 设置 NVLS 通道数为 0（不支持）
  return ncclSuccess;  // 返回成功状态
}

ncclResult_t ncclNvlsBufferSetup(struct ncclComm* comm) {
  return ncclSuccess;  // 直接返回成功（不执行任何操作）
}

ncclResult_t ncclNvlsSetup(struct ncclComm* comm, struct ncclComm* parent) {
  return ncclSuccess;  // 直接返回成功（不执行任何操作）
}

ncclResult_t ncclNvlsFree(struct ncclComm* comm) {
  return ncclSuccess;  // 直接返回成功（不执行任何操作）
}

ncclResult_t ncclNvlsTreeConnect(struct ncclComm* comm) {
  return ncclSuccess;  // 直接返回成功（不执行任何操作）
}

ncclResult_t ncclNvlsGraphRegisterBuffer(
    struct ncclComm *comm, const void *sendbuff, void *recvbuff, size_t sendbuffSize, size_t recvbuffSize,
    int *outRegBufUsed, void **outRegBufSend, void **outRegBufRecv,
    struct ncclIntruQueue<struct ncclCommCallback, &ncclCommCallback::next>* cleanupQueue, int* nCleanupQueueEltsAdded
  ) {
  *outRegBufUsed = false;  // 设置为未使用注册缓冲区
  return ncclSuccess;  // 返回成功状态
}

ncclResult_t ncclNvlsLocalRegisterBuffer(struct ncclComm *comm, const void *sendbuff, void *recvbuff, size_t sendbuffSize, size_t recvbuffSize, int *outRegBufUsed, void **outRegBufSend, void **outRegBufRecv) {
  *outRegBufUsed = false;  // 设置为未使用注册缓冲区
  return ncclSuccess;  // 返回成功状态
}

ncclResult_t ncclNvlsDeregBuffer(struct ncclComm* comm, CUmemGenericAllocationHandle *mcHandler, CUdeviceptr ptr, int dev, size_t ucsize, size_t mcsize) {
  return ncclSuccess;  // 直接返回成功（不执行任何操作）
}

ncclResult_t ncclNvlsSymmetricInit(struct ncclComm* comm) {
  return ncclSuccess;  // 直接返回成功（不执行任何操作）
}

ncclResult_t ncclNvlsSymmetricMap(struct ncclComm* comm, size_t offset, size_t ucsize, void* ucaddr) {
  return ncclSuccess;  // 直接返回成功（不执行任何操作）
}

ncclResult_t ncclNvlsSymmetricFree(struct ncclComm* comm, size_t ucsize, void* ucaddr) {
  return ncclSuccess;  // 直接返回成功（不执行任何操作）
}

ncclResult_t ncclNvlsSymmetricFinalize(struct ncclComm* comm) {
  return ncclSuccess;  // 直接返回成功（不执行任何操作）
}

ncclResult_t ncclNvlsRegResourcesQuery(struct ncclComm* comm, struct ncclTaskColl* info, int* recChannels) {
  *recChannels = 0;  // 设置推荐通道数为 0
  return ncclSuccess;  // 返回成功状态
}

#endif /* CUDA_VERSION >= 12010 */
