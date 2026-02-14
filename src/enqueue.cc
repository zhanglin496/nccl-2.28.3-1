/*************************************************************************
 * Copyright (c) 2017-2022, NVIDIA CORPORATION. All rights reserved.
 * 版权所有 (c) 2017-2022, NVIDIA CORPORATION. 保留所有权利。
 *
 * See LICENSE.txt for license information
 * 请参阅 LICENSE.txt 获取许可证信息
 ************************************************************************/

// 引入操作入队头文件，包含操作入队和内核规划相关的声明
#include "enqueue.h"
// 引入参数检查头文件，包含参数验证相关的函数
#include "argcheck.h"
// 引入集合网络头文件，包含 CollNet 网络支持
#include "coll_net.h"
// 引入 GPU Direct RDMA 包装头文件，提供 GPU Direct 相关功能
#include "gdrwrap.h"
// 引入引导程序头文件，包含连接引导和管理功能
#include "bootstrap.h"
// 引入通道头文件，包含通道管理相关的结构和函数
#include "channel.h"
// 引入 CUDA 包装头文件，提供 CUDA 相关功能的封装
#include "cudawrap.h"
// 引入性能分析器头文件，提供性能分析支持
#include "profiler.h"
// 引入传输层头文件，包含各种传输方式的实现
#include "transport.h"
// 引入内存注册内联头文件，提供内存注册相关的内联函数
#include "register_inline.h"
// 引入 CE（Compute Engine）集合通信头文件
#include "ce_coll.h"
// 引入 NVIDIA Tools Extension 头文件，提供性能分析工具支持
#include "nvtx.h"
// 引入调度器头文件，包含操作调度相关的逻辑
#include "scheduler.h"

// 引入 C++ 标准库字符串头文件
// 提供 std::memcpy 等内存操作函数
#include <cstring> // std::memcpy
// 引入 C 标准库整型类型头文件
// 提供 PRIx64 等格式化宏
#include <cinttypes> // PRIx64
// 引入 C 标准库断言头文件
// 提供 assert 宏用于断言检查
#include <cassert>

// 定义 L1 共享内存预留量的参数
// 参数名称："L1_SHARED_MEMORY_CARVEOUT"，默认值为 0
// 此参数用于配置 CUDA 内核优先从 L1 缓存的 carveout 区域分配共享内存
NCCL_PARAM(L1SharedMemoryCarveout, "L1_SHARED_MEMORY_CARVEOUT", 0);

// Returns maximum kernel stack size of all CUDA kernels
// 返回所有 CUDA 内核的最大内核栈大小
// 此函数遍历所有 NCCL 设备内核，计算它们使用的最大本地内存（栈）大小，
// 并配置内核的共享内存属性（carveout 和动态共享内存大小）
// 参数 cudaArch: CUDA 设备的计算能力版本（如 70、75、80 等）
// 参数 maxSharedMem: 设备支持的最大共享内存大小
// 参数 maxStackSize: 输出参数，返回所有内核中的最大栈大小
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
ncclResult_t ncclInitKernelsForDevice(int cudaArch, int maxSharedMem, size_t* maxStackSize) {
  ncclResult_t result = ncclSuccess;                   // 初始化结果为成功

  if (maxStackSize)                                     // 如果提供了输出参数指针
    *maxStackSize = 0;                                  // 初始化最大栈大小为 0

  // 获取 NCCL 配置的 L1 共享内存预留量（如保留 L1 的 10% 给 NCCL）
  // carveout 是一个百分比，指定 L1 缓存中有多少比例应该优先用于共享内存
  int carveout = ncclParamL1SharedMemoryCarveout();
  // 计算 NCCL 所需的动态共享内存大小（基于计算能力）
  // 不同架构的 GPU 有不同的共享内存需求
  int ncclMaxSharedMem = ncclShmemDynamicSize(cudaArch);

  // 遍历对称和非对称内核类型
  // sym=0: 常规设备内核；sym=1: 对称内核（Symmetric kernel）
  for (int sym=0; sym <= 1; sym++) {
    // 根据内核类型获取对应的内核数量
    int kcount = sym==0 ? ncclDevKernelCount : ncclSymkKernelCount;
    // 根据内核类型获取对应的内核函数指针数组
    void* const* kptrs = sym==0 ? ncclDevKernelList : ncclSymkKernelList;

    // 遍历所有核函数
    for (int k=0; k < kcount; k++) {
      void* fn = kptrs[k];                              // 获取当前内核函数指针
      cudaFuncAttributes attr = {0};                    // 初始化内核属性结构体
      // 如果内核函数指针为空，跳过此内核
      if (fn == nullptr)
        continue;

           // 获取内核函数的属性信息
           // cudaFuncGetAttributes 获取内核的寄存器使用、共享内存、本地内存等信息
      cudaError_t errcode = cudaFuncGetAttributes(&attr, fn);
      if (errcode != cudaSuccess)                       // 如果获取属性失败
        continue; // Silently ignore failures（静默忽略失败）

      // 统计最大栈大小
      // localSizeBytes 表示内核使用的本地内存（栈）大小
      if (maxStackSize) {                               // 如果需要统计最大栈大小
        if (attr.localSizeBytes > *maxStackSize)        // 如果当前内核的栈大小更大
            *maxStackSize = attr.localSizeBytes;        // 更新最大栈大小
      }

      // 告诉 CUDA 优先从 L1 共享内存的 carveout 区域分配
      // 这可以优化内核性能，确保共享内存从 L1 缓存的特定区域分配
      if (carveout) {                                   // 如果设置了 carveout
        // 设置内核的首选共享内存预留属性
        CUDACHECKGOTO(cudaFuncSetAttribute(fn,
          cudaFuncAttributePreferredSharedMemoryCarveout, carveout),
          result, ignore1);                             // 如果失败，跳转到 ignore1 标签
      ignore1:;                                         // 空标签，用于忽略错误继续执行
      }

      // 如果 NCCL 需要动态共享内存
      if (ncclMaxSharedMem != 0) {
        int sharedMemSize = ncclMaxSharedMem;           // 设置共享内存大小
        // NCCL 所需动态内存 + 核函数本身的共享内存 > 设备最大共享内存，则报错返回
        // attr.sharedSizeBytes 是内核静态声明的共享内存大小
        if (sharedMemSize > (maxSharedMem-attr.sharedSizeBytes)) {
          WARN("cudaArch %d ncclMaxSharedMem %d exceeds device/fn maxSharedMem %zu",
               cudaArch, sharedMemSize, maxSharedMem-attr.sharedSizeBytes);
          return ncclSystemError;                       // 返回系统错误
        }

        // 设置内核的最大动态共享内存大小
        CUDACHECKGOTO(cudaFuncSetAttribute(fn,
          cudaFuncAttributeMaxDynamicSharedMemorySize, sharedMemSize),
          result, next_kernel);                         // 如果失败，跳转到 next_kernel
      }
    next_kernel:;                                       // 继续下一个内核
    }
  }
  return result;                                        // 返回操作结果
}

////////////////////////////////////////////////////////////////////////////////
// Data movement metrics.
// 数据移动度量

// 内联函数：计算每个字节的通信流量
// 此函数根据不同的集合通信操作类型，计算每个字节在网络中实际传输的次数
// 这对于算法选择和性能估算非常重要
// 参数 func: 集合通信操作类型（AllReduce、AllGather 等）
// 参数 nRanks: 通信域中的 rank 总数
// 返回值：每个字节需要传输的次数
static inline int ncclFuncTrafficPerByte(ncclFunc_t func, int nRanks) {
  switch (func) {                                       // 根据操作类型判断
  case ncclFuncAllReduce: return 2;                     // AllReduce: 每个字节需要传输 2 次（Reduce-Scatter + AllGather）
  case ncclFuncAllGather: return nRanks;                // AllGather: 每个 rank 的数据需要发送到所有其他 rank，所以是 nRanks 次
  case ncclFuncReduceScatter: return nRanks;            // ReduceScatter: 先 AllReduce 再 Scatter，所以是 nRanks 次
  default: return 1;                                    // 其他操作默认为 1 次传输
  }
}

/*****************************************************************************/
/*       Launch system : synchronization and CUDA kernel launch              */
/*****************************************************************************/
/*       启动系统：同步和 CUDA 内核启动                                        */

// 静态函数：如果需要，将代理操作添加到计划中
// 此函数检查是否需要通过代理线程执行某个操作，如果需要则将其加入代理操作队列
// 代理操作用于在后台线程中处理通信操作，避免阻塞主线程
// 参数 comm: NCCL 通信器指针
// 参数 plan: 内核计划指针
// 参数 op: 代理操作指针
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
static ncclResult_t addProxyOpIfNeeded(struct ncclComm* comm, struct ncclKernelPlan* plan, struct ncclProxyOp* op) {
  bool needed = true;                                   // 标记是否需要代理操作
  NCCLCHECK(ncclProxySaveOp(comm, op, &needed));        // 检查并保存代理操作，更新 needed 标志
  if (needed) {                                         // 如果需要代理操作
    // 从内存池中分配一个代理操作结构
    // 使用 memPool_ncclProxyOp 内存池，使用永久内存生命周期
    struct ncclProxyOp* q = ncclMemoryPoolAlloc<struct ncclProxyOp>(&comm->memPool_ncclProxyOp, &comm->memPermanent);
    *q = *op; // C++ struct assignment（C++ 结构体赋值，复制整个结构体）
    // 将代理操作加入对应通道的代理操作队列
    // op->channelId 指定了该操作属于哪个通道
    ncclIntruQueueEnqueue(&comm->planner.wipPlan.channels[op->channelId].proxyOpQueue, q);
  }
  return ncclSuccess;                                   // 返回成功状态
}

// 静态函数：向计划中添加工作批次
// 此函数将工作项添加到内核计划的批次队列中，管理批次的创建、扩展和合并
// 批次是设备端执行的基本单位，多个工作可以合并到一个批次中以提高效率
// 参数 comm: NCCL 通信器指针
// 参数 plan: 内核计划指针
// 参数 channelId: 通道 ID
// 参数 workType: 工作类型（集合通信或 P2P）
// 参数 devFuncId: 设备函数 ID
// 参数 workOffset: 工作项在 FIFO 中的偏移量
// 参数 p2pRound: P2P 通信轮次（默认为 -1，表示非 P2P 操作）
static void addWorkBatchToPlan(
    struct ncclComm* comm, struct ncclKernelPlan* plan, int channelId,
    enum ncclDevWorkType workType, int devFuncId, uint32_t workOffset,
    int p2pRound = -1
  ) {
  // 获取正在进行工作的计划中对应通道的指针
  ncclKernelPlanner::WipPlan::Channel* chan = &comm->planner.wipPlan.channels[channelId];
  size_t workSize = ncclDevWorkSize(workType);           // 获取工作项的大小
  // 判断是否需要创建一个新的空批次的条件
  // 如果批次队列的尾节点为空，说明队列为空，需要创建新批次
  bool newBatch = (chan->workBatchQueue.tail == nullptr);
  struct ncclDevWorkBatch* batch = nullptr;              // 批次指针
  if (!newBatch) {                                       // 如果不需要创建新批次
    batch = &chan->workBatchQueue.tail->batch;           // 获取当前批次
    // 所有阻止我们将工作追加到当前批次的条件
    // 如果工作类型不同，必须创建新批次
    newBatch |= batch->workType != (uint8_t)workType;
    // 如果函数 ID 不同，必须创建新批次
    newBatch |= batch->funcId != devFuncId;
    // 以下确保设备可以处理这么大的批次。它们必须考虑所有扩展批次被融合在一起，
    // 这就是为什么 wipBatch.workBytes 和 wipBatch.nP2ps 在新的扩展批次中不会重置为 0
    // 如果批次大小超过最大值，必须创建新批次
    newBatch |= NCCL_MAX_DEV_WORK_BATCH_BYTES < chan->wipBatch.workBytes + workSize;
    if (workType == ncclDevWorkTypeP2p) {                // 如果是 P2P 工作类型
      // 如果 P2P 操作数量已达上限，必须创建新批次
      newBatch |= chan->wipBatch.nP2ps == NCCL_MAX_DEV_WORK_P2P_PER_BATCH;
      // 检查是否已经有相同轮次的 P2P 操作
      // 单个批次不能有多个相同轮次的 P2P，因为它们会使用相同的连接
      for (int i=0; i < chan->wipBatch.nP2ps; i++) {
        newBatch |= p2pRound == chan->wipBatch.p2pRounds[i];
      }
    }
  }
  // 判断是否需要创建扩展批次（prev->nextExtends=1）
  // 扩展批次与前一批次在设备端融合执行，但具有不同的偏移基准
  uint32_t offset = newBatch ? 0 : (workOffset - batch->offsetBase);
  // 如果偏移量超过 63 倍的工作大小，需要扩展批次
  bool extendBatch = 63*workSize < offset;
  // 如果偏移量不是工作大小的整数倍，需要扩展批次
  extendBatch |= 0 != offset%workSize;
  if (newBatch || extendBatch) {                         // 如果需要新批次或扩展批次
    if (!newBatch) 
        batch->nextExtends = extendBatch;     // 如果是扩展批次，设置标志
    // 从内存栈中分配批次节点（作用域内存，计划完成后自动释放）
    struct ncclWorkBatchList* batchNode = ncclMemoryStackAlloc<ncclWorkBatchList>(&comm->memScoped);
    // Coverity 认为 ncclIntruQueueEnqueue 会访问 chan->workBatchQueue->tail，可能为 NULL
    // 但该代码由 chan->workBatchQueue->head 不为 NULL 保护，这种情况下 tail 也不会为 NULL
    // coverity[var_deref_model:FALSE]
    ncclIntruQueueEnqueue(&chan->workBatchQueue, batchNode); // 将批次节点加入队列
    batch = &batchNode->batch;                           // 获取批次结构
    batch->nextExtends = 0;                              // 初始化扩展标志
    batch->workType = (uint32_t)workType;                // 设置工作类型
    batch->funcId = devFuncId;                           // 设置函数 ID
    batch->offsetBase = workOffset;                      // 设置偏移基准
    batch->offsetBitset = 0;                             // 初始化偏移位集
    offset = 0;                                          // 重置偏移量
    if (newBatch) {                                      // 如果是新批次
      // 由于扩展批次在设备端融合在一起，而这些值考虑了融合批次的约束，
      // 我们只在新批次时重置这些值
      chan->wipBatch.workBytes = 0;                      // 重置工作字节计数
      chan->wipBatch.nP2ps = 0;                          // 重置 P2P 计数
      // 我们不计算扩展批次，因为这用于推导 proxyOpCount，
      // 我们希望所有融合在一起的操作具有相同的值
      chan->nWorkBatchesP2p += (workType == ncclDevWorkTypeP2p ? 1 : 0);
    }
    plan->nWorkBatches += 1;                             // 增加计划中的批次计数
  }
  // 设置偏移位集中对应的位，标记此偏移量有工作项
  batch->offsetBitset |= 1ull<<(offset/workSize);
  // 累加正在进行工作的批次的工作字节
  chan->wipBatch.workBytes += workSize;
  if (workType == ncclDevWorkTypeP2p) {                 // 如果是 P2P 工作类型
    // 我们需要确保单个批次没有多个相同轮次的 P2P
    // 因为它们会使用相同的连接
    chan->wipBatch.p2pRounds[chan->wipBatch.nP2ps++] = p2pRound; // 记录 P2P 轮次
  }
}

// 静态函数：完成计划构建
// 此函数在所有工作项添加到计划后调用，完成计划的最终配置
// 包括：内核参数结构分配、工作存储类型选择、批次和代理操作的排序
// 参数 comm: NCCL 通信器指针
// 参数 plan: 内核计划指针
static void finishPlan(struct ncclComm* comm, struct ncclKernelPlan* plan) {
  // 获取正在进行工作的计划中的通道数组
  ncclKernelPlanner::WipPlan::Channel* wipChannels = comm->planner.wipPlan.channels;
  size_t workBytes = plan->workBytes;                    // 获取工作数据的总字节数
  size_t batchBytes = plan->nWorkBatches*sizeof(struct ncclDevWorkBatch); // 计算批次数据的总字节数

  if (plan->isSymColl) return;                           // 如果是对称集合，直接返回
  // 确保每个块的线程数不小于最小值
  plan->threadPerBlock = std::max(plan->threadPerBlock, NCCL_MIN_NTHREADS);

  // 如果我们可以将所有内容放入内核参数中，我们就这样做
  // 这样可以避免使用额外的 FIFO 或持久缓冲区，提高性能
  if (sizeof(ncclDevKernelArgs) + batchBytes + workBytes <= comm->workArgsBytes) {
    plan->workStorageType = ncclDevWorkStorageTypeArgs;  // 使用内核参数存储工作数据
  }
  // 计算内核参数结构的大小
  plan->kernelArgsSize = sizeof(struct ncclDevKernelArgs) + batchBytes;
  // 如果使用内核参数存储，需要加上工作数据的大小
  plan->kernelArgsSize += (plan->workStorageType == ncclDevWorkStorageTypeArgs) ? workBytes : 0;
  // 对齐到 16 字节边界（CUDA 设备代码的对齐要求）
  plan->kernelArgsSize = alignUp(plan->kernelArgsSize, 16);
  // 从内存栈中分配内核参数结构（作用域内存，计划完成后自动释放）
  plan->kernelArgs = (struct ncclDevKernelArgs*)ncclMemoryStackAlloc(&comm->memScoped, plan->kernelArgsSize, /*align=*/16);
  // 初始化内核参数的基本字段
  plan->kernelArgs->comm = comm->devComm;                // 设置设备端通信器指针
  plan->kernelArgs->channelMask = plan->channelMask;     // 设置通道掩码（哪些通道参与此计划）
  plan->kernelArgs->workStorageType = plan->workStorageType; // 设置工作存储类型

  // 将批次放入内核参数中。每个通道的第一个批次必须位于 batchZero[blockIdx.x]
  // 为了实现这一点，我们按升序轮询遍历通道，直到它们耗尽
  // 这种轮询方式确保设备端代码可以通过 blockIdx.x 快速找到对应通道的批次
  uint64_t hasBatchMask = plan->channelMask;            // 标记哪些通道有批次
  struct ncclDevWorkBatch* batchPrev[MAXCHANNELS] = {}; // {0...} 存储每个通道的上一个批次
  struct ncclDevWorkBatch* batchZero = (struct ncclDevWorkBatch*)(plan->kernelArgs+1); // 批次数组的起始位置
  int batchIx = 0;                                       // 当前批次索引
  while (hasBatchMask != 0) {                            // 当还有通道有批次时
    uint64_t tmpMask = hasBatchMask;                     // 此轮有批次的通道
    do {
      int c = popFirstOneBit(&tmpMask);                 // 弹出第一个置位，获取通道 ID
      if (!ncclIntruQueueEmpty(&wipChannels[c].workBatchQueue)) { // 如果此通道还有批次
        // 从通道的批次队列中取出一个批次节点
        struct ncclWorkBatchList* batchNode = ncclIntruQueueDequeue(&wipChannels[c].workBatchQueue);
        if (batchPrev[c] != nullptr) {                   // 如果此通道有上一个批次
          // 设置上一个批次的 nextJump，指向当前批次的相对位置
          // 这样设备端代码可以快速跳转到下一个批次
          batchPrev[c]->nextJump = int(&batchZero[batchIx] - batchPrev[c]);
        }
        batchPrev[c] = &batchZero[batchIx];              // 记录当前批次为此通道的最后一个批次
        batchZero[batchIx++] = batchNode->batch;         // 将批次数据复制到批次数组中
      }
      // 如果此通道的批次队列已空，从掩码中清除该通道
      if (ncclIntruQueueEmpty(&wipChannels[c].workBatchQueue)) {
        hasBatchMask ^= 1ull<<c;
      }
    } while (tmpMask != 0);                              // 继续处理下一轮
  }

  // 在将每个通道的代理操作列表合并到 plan->proxyOpQueue 时，按 opCount 进行归并排序
  // 这确保代理操作按正确的顺序执行，这对于同步和正确性至关重要
  // 阶段 1：扫描每个通道的第一个操作，将 opCount 存储在 headIds[c] 中
  uint64_t headIds[MAXCHANNELS];                         // 存储每个通道队列头的 opCount
  int nHeads = 0;                                        // 非空通道的数量
  int channelUbound = 0;                                 // 最高的非空通道 ID + 1
  for (int c=0; c < MAXCHANNELS; c++) {                  // 遍历所有通道
    struct ncclProxyOp* op = ncclIntruQueueHead(&wipChannels[c].proxyOpQueue); // 获取队列头
    headIds[c] = op ? op->opCount : uint64_t(-1);       // 存储 opCount，空队列为 -1
    if (op) nHeads += 1;                                 // 统计非空通道数
    if (op) plan->hasProxyOps = true;                    // 标记计划有代理操作
    if (op) channelUbound = c+1;                         // 更新最高通道 ID
  }
  // 阶段 2：从 planner->channels[c] 出队，按合并顺序入队到 plan
  while (nHeads != 0) {                                  // 当还有非空通道时
    int c = -1;                                          // 最小 opCount 的通道
    uint64_t minId = uint64_t(-1);                       // 最小的 opCount
    // 找到具有最小代理操作 ID 的通道。我们将 heads[c]->opCount 存储在
    // headIds[c] 中，以从此循环中移除间接加载
    for (int c1=0; c1 < channelUbound; c1++) {           // 遍历所有可能非空的通道
      uint64_t id = headIds[c1];                         // 获取该通道的 opCount
      id = (id>>1 | id<<63); // 将标记位移到最低位，使集合操作在 P2P 之前排序
      if (id < minId) { c = c1; minId = id; }            // 找到最小 ID 的通道
    }
    // 从最小 ID 的通道出队一个操作
    struct ncclProxyOp* op = ncclIntruQueueDequeue(&wipChannels[c].proxyOpQueue);
    // 获取该通道的下一个操作
    struct ncclProxyOp* opNext = ncclIntruQueueHead(&wipChannels[c].proxyOpQueue);
    headIds[c] = opNext ? opNext->opCount : uint64_t(-1); // 更新该通道的 headId
    nHeads -= opNext ? 0 : 1;                            // 如果队列空了，减少非空通道数
    ncclIntruQueueEnqueue(&plan->proxyOpQueue, op);      // 将操作加入计划的代理操作队列
  }
}

// 定义 CUDA 图内存注册参数
// 参数名称："GRAPH_REGISTER"，默认值为 1
// 控制在 CUDA 图捕获期间是否注册内存缓冲区
NCCL_PARAM(GraphRegister, "GRAPH_REGISTER", 1);

// 静态函数声明：获取 CollNet（集合网络）支持状态
// 此函数检查当前配置是否支持 CollNet 加速
static ncclResult_t getCollNetSupport(struct ncclComm* comm, struct ncclTaskColl* task, int* collNetSupport);
// 静态函数声明：获取算法信息
// 此函数根据任务特性选择最优的算法和协议组合
static ncclResult_t getAlgoInfo(
  struct ncclComm* comm, struct ncclTaskColl* task,
  int collNetSupport, int nvlsSupport, int numPipeOps, ncclSimInfo_t* simInfo = NULL
);
// 静态函数声明：计算集合通信的分块大小
// 此函数计算如何在多个通道间分割数据和选择合适的块大小
static ncclResult_t calcCollChunking(
  struct ncclComm* comm, struct ncclTaskColl* task, int nChannels, size_t nBytes,
  /*outputs*/uint32_t* outChunkSize, uint32_t* outDirectFlags, struct ncclProxyOp* proxyOp
);

// 内核计划预算结构体
// 此结构体用于跟踪内核计划的内存预算，确保不超过限制
struct ncclKernelPlanBudget {
  ssize_t inArgsBytes;  // 内核参数结构内可用的空间（用于直接传递给内核的数据）
  ssize_t outArgsBytes; // 参数结构外可用的空间（FIFO 或持久缓冲区）
};

// 静态函数：测试预算是否足够
// 此函数检查计划的批次和工作数据是否在预算范围内
// 参数 budget: 预算指针
// 参数 nWorkBatches: 批次数量
// 参数 workBytes: 工作数据字节数
// 返回值：如果预算足够返回 true，否则返回 false
static bool testBudget(
    struct ncclKernelPlanBudget* budget, int nWorkBatches, ssize_t workBytes
  ) {
  ssize_t batchBytes = nWorkBatches*sizeof(struct ncclDevWorkBatch); // 计算批次数据的总字节数
  bool ok = false;                                     // 初始化为不满足
  // 检查是否可以将所有数据放入内核参数中
  ok |= (batchBytes + workBytes <= budget->inArgsBytes);
  // 检查是否可以批次放入参数，工作数据放入外部缓冲区
  ok |= (batchBytes <= budget->inArgsBytes) && (workBytes <= budget->outArgsBytes);
  return ok;                                           // 返回预算检查结果
}

// 函数实现：注册内存并入队任务
// 此函数为每个集合通信任务注册内存缓冲区，并构建设备端工作结构
// 参数 comm: NCCL 通信器指针
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
ncclResult_t ncclTasksRegAndEnqueue(struct ncclComm* comm) {
  struct ncclKernelPlanner* planner = &comm->planner; // 获取内核规划器指针
  struct ncclTaskColl *task;                            // 任务指针
  task = ncclIntruQueueHead(&planner->collTaskQueue);  // 获取集合任务队列的头节点
  while (task != nullptr) {                             // 遍历所有任务
    // 为每个任务构建一个 ncclDevWorkColl[Reg?] 结构
    // 此结构将被复制到 GPU 内存，供设备端内核使用
    void* regBufSend[NCCL_MAX_LOCAL_RANKS];            // 已注册的发送缓冲区数组
    void* regBufRecv[NCCL_MAX_LOCAL_RANKS];            // 已注册的接收缓冲区数组
    bool regNeedConnect = true;                         // 是否需要建立连接
    struct ncclWorkList* workNode = NULL;               // 工作节点指针
    struct ncclDevWorkColl devWork = {};                // 设备端工作结构（初始化为全零）

    // NVLS 算法不需要在这里处理，直接使用临时队列中的节点
    if (task->algorithm == NCCL_ALGO_NVLS_TREE || task->algorithm == NCCL_ALGO_NVLS) {
      workNode = ncclIntruQueueDequeue(&planner->tmpCollWorkQueue); // 从临时队列中取出工作节点
      goto next;                                        // 跳转到入队步骤
    }
    // 注册集合通信缓冲区
    // 这将缓冲区注册到传输层，以启用 GPU Direct RDMA 等优化
    ncclRegisterCollBuffers(comm, task, regBufSend, regBufRecv, &planner->collCleanupQueue, &regNeedConnect);

    // 从任务中复制基本信息到设备工作结构
    devWork.sendbuff = (void*)task->sendbuff;          // 发送缓冲区地址
    devWork.recvbuff = (void*)task->recvbuff;          // 接收缓冲区地址
    devWork.sendbuffOffset = task->sendbuffOffset;     // 发送缓冲区偏移量
    devWork.recvbuffOffset = task->recvbuffOffset;     // 接收缓冲区偏移量
    devWork.sendbuffRmtAddrs = task->sendbuffRmtAddrs; // 远程发送缓冲区地址
    devWork.recvbuffRmtAddrs = task->recvbuffRmtAddrs; // 远程接收缓冲区地址
    devWork.root = task->root;                         // 根节点 rank（用于 Broadcast、Reduce 等操作）
    devWork.nWarps = task->nWarps;                     // 每个 CTa 的 warp 数量
    devWork.redOpArg = task->opDev.scalarArg;          // 规约操作的标量参数
    devWork.redOpArgIsPtr = task->opDev.scalarArgIsPtr; // 规约参数是否为指针
    devWork.oneNode = (comm->nNodes == 1);             // 是否为单节点通信
    devWork.isOneRPN = comm->isOneRPN;                 // 是否为每个节点一个 rank
    devWork.netRegUsed = devWork.regUsed = 0;          // 初始化注册标志
    // 检查是否启用了性能分析器
    devWork.profilerEnabled = ncclProfilerPluginLoaded() && (task->eActivationMask & ncclProfileKernelCh);
    // 检查是否使用了网络注册缓冲区（GPU Direct RDMA）
    if (task->regBufType & NCCL_NET_REG_BUFFER)
      devWork.netRegUsed = 1;                           // 标记使用了网络注册
    // 检查是否使用了 IPC 或 NVLS 注册缓冲区
    if (task->regBufType & (NCCL_IPC_REG_BUFFER | NCCL_NVLS_REG_BUFFER))
      devWork.regUsed = 1;                              // 标记使用了 IPC/NVLS 注册

    // 如果使用了 NVLS 注册缓冲区，需要使用带注册信息的工作结构
    if (task->regBufType & NCCL_NVLS_REG_BUFFER) {
      struct ncclDevWorkCollReg workReg = {};          // 带注册信息的集合工作结构
      workReg.coll = devWork; // C++ struct assignment（复制基本集合工作信息）
      /* NVLS 只有一个发送和接收缓冲区被注册 */
      workReg.dnInputs[0] = regBufSend[0];              // NVLS 输入缓冲区
      workReg.dnOutputs[0] = regBufRecv[0];             // NVLS 输出缓冲区
      // 分配工作节点（带注册信息）
      workNode = ncclMemoryStackAllocInlineArray<ncclWorkList, ncclDevWorkCollReg>(&comm->memScoped, 1);
      workNode->workType = ncclDevWorkTypeCollReg;      // 设置工作类型为带注册的集合通信
      workNode->size = sizeof(struct ncclDevWorkCollReg); // 设置工作结构大小
      memcpy((void*)(workNode+1), (void*)&workReg, workNode->size); // 复制工作结构数据
    } else {
      // 分配工作节点（不带注册信息）
      workNode = ncclMemoryStackAllocInlineArray<ncclWorkList, ncclDevWorkColl>(&comm->memScoped, 1);
      workNode->workType = ncclDevWorkTypeColl;         // 设置工作类型为普通集合通信
      workNode->size = sizeof(struct ncclDevWorkColl);  // 设置工作结构大小
      memcpy((void*)(workNode+1), (void*)&devWork, workNode->size); // 复制工作结构数据
    }
next:
    ncclIntruQueueEnqueue(&planner->collWorkQueue, workNode); // 将工作节点加入集合工作队列
    task = task->next;                                  // 移动到下一个任务
  }
  // 确保临时工作队列为空（所有 NVLS 任务都应该被处理）
  assert(ncclIntruQueueEmpty(&planner->tmpCollWorkQueue));
  return ncclSuccess;                                   // 返回成功状态
}

// 在每次 ncclGroupEnd 时调用一次，用于组织用户提交的任务到 comm->planner 中
// 以便它们可以被分配到各个计划中
// 参数 comm: NCCL 通信器指针
// 参数 algoNeedConnect: 输出数组，标记哪些算法需要建立连接
// 参数 needConnect: 输出标志，是否需要建立连接
// 参数 simInfo: 模拟信息指针（用于性能预测）
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
ncclResult_t ncclPrepareTasks(struct ncclComm* comm, bool* algoNeedConnect, bool* needConnect, ncclSimInfo_t* simInfo) {
  struct ncclKernelPlanner* planner = &comm->planner;     // 获取内核规划器指针
  // 检查是否正在捕获 CUDA 图（持久化模式）
  planner->persistent = ncclCudaGraphValid(planner->capturingGraph);

  // 从排序器中取出所有任务，按大小降序排列
  // 排序器确保大消息优先处理，有助于优化整体性能
  struct ncclTaskColl* task = ncclTaskCollSorterDequeueAll(&planner->collSorter);
  // 任务按 (函数, 操作, 数据类型) 分类组装，大小升序
  // tasksByFnOpTy 数组的索引由 (func * numOps + op) * numTypes + type 计算
  struct ncclTaskColl* tasksByFnOpTy[ncclNumFuncs*ncclNumDevRedOps*ncclNumTypes];
  memset(tasksByFnOpTy, 0, sizeof(tasksByFnOpTy));        // 初始化数组为全零
  int fnOpTyIndices[ncclNumFuncs*ncclNumDevRedOps*ncclNumTypes]; // 存储实际使用的索引
  int fnOpTyCount = 0;                                    // 实际使用的 (函数, 操作, 类型) 组合数量

  // 如果支持对称操作，构建对称任务列表
  if (comm->symmetricSupport) {
    NCCLCHECK(ncclMakeSymmetricTaskList(comm, task, &planner->collSymTaskQueue, &task));
  }

  // 遍历按大小排序的任务，按 (函数, 操作, 类型) 分组
  // 相同 (fn, op, ty) 的任务将使用相同的算法和协议
  while (task != nullptr) {
    struct ncclTaskColl* next = task->next;              // 保存下一个任务指针
    // 计算任务在 tasksByFnOpTy 数组中的索引
    int index = ((int)task->func*ncclNumDevRedOps + (int)task->opDev.op)*ncclNumTypes + (int)task->datatype;
    // 第一次出现时，将索引添加到索引集合
    if (tasksByFnOpTy[index] == nullptr) fnOpTyIndices[fnOpTyCount++] = index;
    // 将任务添加到此 (fn,op,ty) 的 LIFO 链表头部
    task->next = tasksByFnOpTy[index];
    tasksByFnOpTy[index] = task;
    // 移动到下一个任务
    task = next;
  }

  // 遍历 (fn,op,ty) 分组，计算算法和协议等，然后按调度约束分组
  // collBins[isCollnet][isNvls] 四种组合：[0][0] 标准, [0][1] NVLS, [1][0] CollNet, [1][1] CollNet+NVLS
  struct ncclIntruQueue<struct ncclTaskColl, &ncclTaskColl::next> collBins[2][2] = {};
  for (int cursor=0; cursor < fnOpTyCount; cursor++) {    // 遍历所有 (fn,op,ty) 组合
    struct ncclTaskColl* aggBeg = tasksByFnOpTy[fnOpTyIndices[cursor]]; // 获取此组合的任务链表
    int collNetSupport = 0;                               // CollNet 支持标志
        // 检查是否支持 CollNet
    NCCLCHECK(getCollNetSupport(comm, aggBeg, &collNetSupport));
   // 检查是否支持 NVLS
    int nvlsSupport = comm->nvlsSupport && (ncclNvlsSupported(aggBeg->opDev.op, aggBeg->datatype) || aggBeg->func == ncclFuncAllGather);

    // 粗略估计每个通道的任务数。这对 NVLS 算法使用了错误的通道数，
    // 但知道算法需要此值，所以要么粗略估计要么迭代直到固定点，我们选择前者。
    // 每个 nChannels 分配多少个任务
    int nTasksPerChannel = divUp(comm->planner.nTasksColl, comm->nChannels);

    do {
      struct ncclTaskColl* aggEnd = aggBeg->next;         // 聚合结束位置
      struct ncclTaskColl agg = *aggBeg;                  // 聚合任务结构
      // 我们聚合大小在 4 倍以内的操作
      // 这样可以在算法选择时将相似大小的任务一起考虑
      while (aggEnd != nullptr && aggEnd->trafficBytes < 4*aggBeg->trafficBytes) {
        agg.count += aggEnd->count;                       // 累加元素数量
        agg.trafficBytes += aggEnd->trafficBytes;         // 累加流量字节数
        aggEnd = aggEnd->next;                            // 移动到下一个
      }

      // 获取算法信息（选择最优算法和协议）
      NCCLCHECK(getAlgoInfo(comm, &agg, collNetSupport, nvlsSupport, nTasksPerChannel, simInfo));
      // 计算设备函数 ID（用于调用正确的内核）
      agg.devFuncId = ncclDevFuncId(agg.func, agg.opDev.op, agg.datatype, agg.algorithm, agg.protocol);

      int isCollnet=0, isNvls=0;                          // 算法类型标志
      switch (agg.algorithm) {
      case NCCL_ALGO_NVLS:                                // NVLS 算法
      case NCCL_ALGO_NVLS_TREE:                           // NVLS 树算法
        isNvls = 1;                                       // 标记为 NVLS
        // 多节点 NVLS 同时使用 CollNet
        isCollnet = agg.algorithm == NCCL_ALGO_NVLS && comm->nNodes > 1;
        break;
      case NCCL_ALGO_COLLNET_CHAIN:                       // CollNet 链算法
      case NCCL_ALGO_COLLNET_DIRECT:                      // CollNet 直连算法
        isCollnet = 1;                                    // 标记为 CollNet
        break;
      }
      // 用计算出的值更新聚合任务
      do {
        struct ncclTaskColl* next = aggBeg->next;        // 保存下一个任务
        aggBeg->algorithm = agg.algorithm;                // 设置算法
        aggBeg->protocol = agg.protocol;                  // 设置协议
        // LL 协议实际传输量是数据的 4 倍（由于编码开销）
        if (aggBeg->protocol == NCCL_PROTO_LL)
            aggBeg->trafficBytes *= 4;
        aggBeg->nMaxChannels = agg.nMaxChannels;          // 设置最大通道数
        aggBeg->nWarps = agg.nWarps;                      // 设置 warp 数量
        aggBeg->devFuncId = agg.devFuncId;                // 设置设备函数 ID
        aggBeg->isCollnet = isCollnet;                    // 设置 CollNet 标志
        aggBeg->isNvls = isNvls;                          // 设置 NVLS 标志
        // 将任务加入对应的分组队列
        ncclIntruQueueEnqueue(&collBins[isCollnet][isNvls], aggBeg);
        aggBeg = next;                                    // 移动到下一个任务
      } while (aggBeg != aggEnd);
    } while (aggBeg != nullptr);
  }

  // 将 `collBins[*][*]` 连接成最终列表 `planner->collTaskQueue`
  // CollNet 是外层维度，因为它影响我们如何在通道间分配
  for (int isCollnet=0; isCollnet <= 1; isCollnet++) {
    for (int isNvls=0; isNvls <= 1; isNvls++) {
      // 将分组的任务队列转移到最终的集合任务队列
      ncclIntruQueueTransfer(&planner->collTaskQueue, &collBins[isCollnet][isNvls]);
    }
  }

  // 再次遍历任务以：
  // 1. 可能注册缓冲区
  // 2. 构建 ncclDevWorkColl 结构
  // 3. 根据它们可能分配到的有效通道数量对工作结构进行分组 {collnet, nvls, standard}
  task = ncclIntruQueueHead(&planner->collTaskQueue);
  while (task != nullptr) {
    // 为每个任务构建一个 ncclDevWorkColl[Reg?] 结构
    void* regBufSend[NCCL_MAX_LOCAL_RANKS];              // 已注册的发送缓冲区数组
    void* regBufRecv[NCCL_MAX_LOCAL_RANKS];              // 已注册的接收缓冲区数组
    bool regNeedConnect = true;                          // 是否需要建立连接
    // 注册 NVLS 集合通信缓冲区
    ncclRegisterCollNvlsBuffers(comm, task, regBufSend, regBufRecv, &planner->collCleanupQueue, &regNeedConnect);

    // 检查是否需要为算法建立连接
    if (comm->runtimeConn && comm->initAlgoChannels[task->algorithm] == false) {
      // NVLS_TREE 算法也需要初始化 NVLS 连接
      if (task->algorithm == NCCL_ALGO_NVLS_TREE && comm->initAlgoChannels[NCCL_ALGO_NVLS] == false && regNeedConnect == true) {
        comm->initAlgoChannels[NCCL_ALGO_NVLS] = true;
        algoNeedConnect[NCCL_ALGO_NVLS] = true;
      }
      // 标记此算法需要建立连接
      if (task->algorithm != NCCL_ALGO_NVLS || regNeedConnect == true) {
        comm->initAlgoChannels[task->algorithm] = true;
        algoNeedConnect[task->algorithm] = true;
        *needConnect = true;
      }
    }

    // NVLS 任务需要特殊处理
    if (task->algorithm == NCCL_ALGO_NVLS_TREE || task->algorithm == NCCL_ALGO_NVLS) {
      struct ncclDevWorkColl devWork = {};                // 设备工作结构
      devWork.sendbuff = (void*)task->sendbuff;          // 发送缓冲区
      devWork.recvbuff = (void*)task->recvbuff;          // 接收缓冲区
      devWork.sendbuffOffset = task->sendbuffOffset;     // 发送缓冲区偏移
      devWork.recvbuffOffset = task->recvbuffOffset;     // 接收缓冲区偏移
      devWork.sendbuffRmtAddrs = task->sendbuffRmtAddrs; // 远程发送地址
      devWork.recvbuffRmtAddrs = task->recvbuffRmtAddrs; // 远程接收地址
      devWork.root = task->root;                         // 根节点 rank
      devWork.nWarps = task->nWarps;                     // Warp 数量
      devWork.redOpArg = task->opDev.scalarArg;          // 规约操作参数
      devWork.redOpArgIsPtr = task->opDev.scalarArgIsPtr; // 参数是否为指针
      devWork.oneNode = (comm->nNodes == 1);             // 是否为单节点
      devWork.netRegUsed = devWork.regUsed = 0;          // 初始化注册标志
      // 检查是否启用性能分析
      devWork.profilerEnabled = ncclProfilerPluginLoaded() && (task->eActivationMask & ncclProfileKernelCh);
      if (task->regBufType & NCCL_NET_REG_BUFFER)
        devWork.netRegUsed = 1;                           // 网络注册标志
      if (task->regBufType & (NCCL_IPC_REG_BUFFER | NCCL_NVLS_REG_BUFFER))
        devWork.regUsed = 1;                              // IPC/NVLS 注册标志

      struct ncclWorkList* workNode;                     // 工作节点
      if (task->regBufType & NCCL_NVLS_REG_BUFFER) {     // 如果使用 NVLS 注册
        struct ncclDevWorkCollReg workReg = {};          // 带注册的工作结构
        workReg.coll = devWork; // C++ struct assignment
        /* NVLS 只有一个发送和接收缓冲区被注册 */
        workReg.dnInputs[0] = regBufSend[0];              // NVLS 输入缓冲区
        workReg.dnOutputs[0] = regBufRecv[0];             // NVLS 输出缓冲区
        // 分配带注册信息的工作节点
        workNode = ncclMemoryStackAllocInlineArray<ncclWorkList, ncclDevWorkCollReg>(&comm->memScoped, 1);
        workNode->workType = ncclDevWorkTypeCollReg;      // 工作类型：带注册的集合通信
        workNode->size = sizeof(struct ncclDevWorkCollReg); // 工作结构大小
        memcpy((void*)(workNode + 1), (void*)&workReg, workNode->size); // 复制工作结构数据
      } else {
        // 分配不带注册信息的工作节点
        workNode = ncclMemoryStackAllocInlineArray<ncclWorkList, ncclDevWorkColl>(&comm->memScoped, 1);
        workNode->workType = ncclDevWorkTypeColl;         // 工作类型：普通集合通信
        workNode->size = sizeof(struct ncclDevWorkColl);  // 工作结构大小
        memcpy((void*)(workNode + 1), (void*)&devWork, workNode->size); // 复制工作结构数据
      }

      // 将工作节点加入临时集合工作队列（NVLS 任务将在 ncclTasksRegAndEnqueue 中处理）
      ncclIntruQueueEnqueue(&planner->tmpCollWorkQueue, workNode);
    }
    task = task->next;                                    // 移动到下一个任务
  }

  return ncclSuccess;                                   // 返回成功状态
}

// 静态函数：如果需要，添加性能分析器代理操作
// 此函数创建一个特殊的代理操作用于性能分析，记录内核事件
// 参数 comm: NCCL 通信器指针
// 参数 plan: 内核计划指针
// 参数 op: 原始代理操作指针
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
static ncclResult_t addProfilerProxyOpIfNeeded(struct ncclComm* comm, struct ncclKernelPlan* plan, struct ncclProxyOp* op) {
  int tmp = op->pattern;                                // 保存原始操作模式
  op->pattern = ncclPatternProfiler;                    // 设置为性能分析器模式
  ncclResult_t ret = addProxyOpIfNeeded(comm, plan, op); // 添加代理操作
  op->pattern = tmp;                                    // 恢复原始操作模式
  return ret;                                           // 返回操作结果
}

// 静态函数：将集合任务调度到计划中
// 此函数将集合任务从队列中取出，分配通道，并添加到内核计划中
// 它负责在预算范围内优化任务分配，并处理 CollNet 和非 CollNet 任务的不同调度策略
// 参数 comm: NCCL 通信器指针
// 参数 plan: 内核计划指针
// 参数 budget: 预算指针
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
static ncclResult_t scheduleCollTasksToPlan(
    struct ncclComm* comm, struct ncclKernelPlan* plan, struct ncclKernelPlanBudget* budget
  ) {
  struct ncclKernelPlanner* planner = &comm->planner;   // 获取内核规划器指针
  // 估计适合此计划的任务数量
  int nPlanColls = 0;                                    // 计划中的集合任务数量
  size_t trafficBytes[2*2] = {0, 0, 0, 0};              // [collnet][nvls] 流量字节数组
  int nChannels[2*2] = {0, 0, 0, 0};                    // [collnet][nvls] 通道数数组
  // 最大通道数数组：[0][0]=标准, [0][1]=NVLS, [1][0]=CollNet, [1][1]=CollNet+NVLS
  int const nMaxChannels[2*2] = {comm->nChannels, comm->nvlsChannels, // [collnet][nvls]
                                 comm->nChannels, std::min(comm->nChannels, comm->nvlsChannels)};
  constexpr size_t MinTrafficPerChannel = 16 << 10;      // 每个通道的最小流量：16KB
  // 第一遍：估计可以放入此计划的任务数量
  do {
    size_t workBytes = 0;                                // 工作数据字节数
    struct ncclTaskColl* task = ncclIntruQueueHead(&planner->collTaskQueue); // 获取任务队列头
    struct ncclWorkList* workNode = ncclIntruQueueHead(&planner->collWorkQueue); // 获取工作队列头
    while (task != nullptr) {                            // 遍历所有任务
      int nBatches = divUp(nPlanColls, 4);               // 粗略估计：每批次 4 个集合操作
      // 检查预算是否足够
      if (!testBudget(budget, nBatches, workBytes + workNode->size)) 
        goto plan_full;

      nPlanColls += 1;                                   // 增加计划任务计数
      workBytes += workNode->size;                       // 累加工作数据大小
      int kind = 2*task->isCollnet + task->isNvls;       // 计算任务类型索引
      // 累加流量字节数，确保至少为最小流量
      trafficBytes[kind] += std::max(MinTrafficPerChannel, task->trafficBytes);
      nChannels[kind] += task->nMaxChannels;             // 累加通道数
      nChannels[kind] = std::min(nChannels[kind], nMaxChannels[kind]); // 限制在最大通道数
      task = task->next;                                 // 移动到下一个任务
      workNode = workNode->next;                         // 移动到下一个工作节点
    }
  plan_full:;                                            // 计划已满的标签
  } while (0);

  // 第二遍：实际将任务分配到通道
  int kindPrev = -1;                                     // 上一个任务的类型
  size_t trafficPerChannel = 0;                          // 每个通道的流量
  int channelId = 0;                                     // 当前通道 ID
  size_t currentTraffic = 0;                             // 当前通道已分配的流量
  while (nPlanColls!=0 && !ncclIntruQueueEmpty(&planner->collTaskQueue)) { // 当还有任务时
    struct ncclTaskColl* task = ncclIntruQueueHead(&planner->collTaskQueue); // 获取任务
    struct ncclWorkList* workNode = ncclIntruQueueHead(&planner->collWorkQueue); // 获取工作节点
    struct ncclDevWorkColl* devWork = (struct ncclDevWorkColl*)(workNode+1); // 获取设备工作结构
    size_t elementSize = ncclTypeSize(task->datatype);   // 获取数据类型大小

    int kind = 2*task->isCollnet + task->isNvls;         // 计算任务类型索引
    if (kind != kindPrev) {                              // 如果任务类型改变
      // 计算每个通道的平均流量
      trafficPerChannel = std::max<size_t>(MinTrafficPerChannel, trafficBytes[kind]/nChannels[kind]);
      kindPrev = kind;                                   // 更新类型记录
      channelId = 0;                                     // 重置通道 ID
      currentTraffic = 0;                                // 重置当前流量
    }

    // 处理 CollNet 任务
    if (task->isCollnet) {
      int nChannels = task->nMaxChannels;                // 获取最大通道数
      // 确保最坏情况下每个通道一个新批次的空间
      if (!testBudget(budget, plan->nWorkBatches + nChannels, plan->workBytes + workNode->size)) {
        return ncclSuccess;                              // 预算不足，返回成功（当前计划已满）
      }

      // 计算每个元素的全局字节数（考虑通信模式）
      size_t globalBytesPerElement = elementSize*ncclFuncMaxSendRecvCount(task->func, comm->nRanks, 1);
      struct ncclProxyOp proxyOp;                        // 代理操作结构
      uint32_t chunkSize, directFlags=0;                 // 块大小和直接标志
      // 计算集合通信的分块
      NCCLCHECK(calcCollChunking(comm, task, nChannels, globalBytesPerElement*task->count, &chunkSize, &directFlags, &proxyOp));
      devWork->channelLo = 0;                            // 第一个通道 ID
      devWork->channelHi = nChannels-1;                  // 最后一个通道 ID
      devWork->collnet.count = task->count;              // 元素数量
      devWork->collnet.chunkCount = chunkSize/ncclTypeSize(task->datatype); // 每块的元素数
      devWork->direct = directFlags;                     // 直接标志

      // 生成代理操作 ID（集合操作使用偶数 ID）
      uint64_t proxyOpId = uint64_t(plan->collOpCount++)<<1 | 0;
      // 为每个通道添加代理操作
      for (int c=devWork->channelLo; c <= (int)devWork->channelHi; c++) {
        proxyOp.channelId = c;                            // 设置通道 ID
        proxyOp.opCount = proxyOpId;                      // 设置操作计数
        proxyOp.task.coll = task;                         // 关联任务
        proxyOp.rank = comm->rank;                        // 设置 rank
        proxyOp.eActivationMask = task->eActivationMask;  // 设置事件激活掩码
        proxyOp.incWorkCounter = true;                    // 增加工作计数器
        addWorkBatchToPlan(comm, plan, c, workNode->workType, task->devFuncId, plan->workBytes);
        // 设置模式为性能分析器，为内核事件添加代理分析器
        NCCLCHECK(addProxyOpIfNeeded(comm, plan, &proxyOp));
        NCCLCHECK(addProfilerProxyOpIfNeeded(comm, plan, &proxyOp));
      }
    } else { // 非 CollNet 任务（Ring、Tree 等标准算法）
      int trafficPerByte = ncclFuncTrafficPerByte(task->func, comm->nRanks); // 每字节的流量
      if (task->protocol == NCCL_PROTO_LL) trafficPerByte *= 4; // LL 协议的流量是 4 倍
      // 计算单元格大小（对齐到 16 字节）
      size_t cellSize = divUp(divUp(MinTrafficPerChannel, (size_t)trafficPerByte), 16) * 16;
      int elementsPerCell = cellSize/elementSize;         // 每个单元格的元素数
      size_t cells = divUp(task->count*elementSize, cellSize); // 总单元格数
      size_t trafficPerElement = elementSize*trafficPerByte; // 每个元素的流量
      size_t trafficPerCell = cellSize*trafficPerByte;   // 每个单元格的流量
      // 计算每个通道的单元格数
      size_t cellsPerChannel = std::min(cells, divUp(trafficPerChannel, trafficPerCell));
      size_t cellsLo;                                     // 第一个通道的单元格数
      if (channelId+1 == nMaxChannels[kind]) {           // 如果是最后一个通道，所有数据都分配给 "lo"
        cellsLo = cells;
      } else {
        // 计算第一个通道分配的单元格数
        cellsLo = std::min(cells, divUp((trafficPerChannel-currentTraffic),trafficPerCell));
      }
      // 计算中间通道的数量（完整通道）
      int nMidChannels = (cells-cellsLo)/cellsPerChannel;
      // 计算最后一个通道的单元格数
      size_t cellsHi = (cells-cellsLo)%cellsPerChannel;
      // 计算实际使用的通道数（lo + mid + hi）
      int nChannels = (cellsLo!=0 ? 1 : 0) + nMidChannels + (cellsHi!=0 ? 1 : 0);
      // 检查是否溢出可用通道数
      if (nMaxChannels[kind] < channelId + nChannels) { // 溢出可用通道
        // 调整中间通道数量
        nMidChannels = nMaxChannels[kind] - channelId - 2;
        // 重新计算每个通道的单元格数
        cellsPerChannel = (cells-cellsLo)/(nMidChannels+1);
        // 重新计算最后一个通道的单元格数
        cellsHi = cellsPerChannel + (cells-cellsLo)%(nMidChannels+1);
      }
      // 如果 hi 为 0 但有中间通道，将一个中间通道转为 hi
      if (cellsHi == 0 && nMidChannels != 0) {
        cellsHi = cellsPerChannel;                        // 将一个中间通道移到 hi
        nMidChannels -= 1;                                // 减少中间通道数
      }
      // 如果 lo 为 0，跳过最少通道
      if (cellsLo == 0) { // 最少通道被跳过，使下一个通道成为新的最少通道
        channelId += 1;                                   // 移动到下一个通道
        if (nMidChannels == 0) { cellsLo = cellsHi; cellsHi = 0; } // 没有中间通道，将 hi 移到 lo
        else { cellsLo = cellsPerChannel; nMidChannels -= 1; } // 将一个中间通道移到 lo
      }
      // 计算每个通道的元素数量
      size_t countMid = nMidChannels!=0 ? cellsPerChannel*elementsPerCell : 0;
      size_t countLo = cellsLo*elementsPerCell;
      size_t countHi = cellsHi*elementsPerCell;
      // 调整最后一个通道的元素数量以匹配实际任务元素数（四舍五入误差修正）
      (countHi != 0 ? countHi : countLo) -= cells*elementsPerCell - task->count;

      // 重新计算实际使用的通道数
      nChannels = (countLo!=0 ? 1 : 0) + nMidChannels + (cellsHi!=0 ? 1 : 0);

      // 更新传播到性能分析器的通道数
      task->nChannels = (uint8_t)nChannels;

      // 确保最坏情况下每个通道一个新批次的空间
      if (!testBudget(budget, plan->nWorkBatches + nChannels, plan->workBytes + workNode->size)) {
        return ncclSuccess;                              // 预算不足，返回成功
      }

      // 设置设备工作结构的通道范围
      devWork->channelLo = channelId;                    // 第一个通道 ID
      devWork->channelHi = channelId + nChannels-1;      // 最后一个通道 ID
      // 设置每个通道的元素数量
      devWork->cbd.countLo = countLo;                    // 第一个通道的元素数
      devWork->cbd.countMid = countMid;                  // 中间通道的元素数
      devWork->cbd.countHi = countHi;                    // 最后一个通道的元素数

      // calcCollChunking() 使用全局字节而不是流量，不同之处在于
      // AllReduce 不会被乘以 2（流量 vs 实际数据量）
      size_t globalBytesPerElement = elementSize*ncclFuncMaxSendRecvCount(task->func, comm->nRanks, 1);
      struct ncclProxyOp proxyOpLo, proxyOpMid, proxyOpHi; // 三个通道的代理操作

      uint32_t chunkSize, directFlags=0;                 // 块大小和直接标志
      size_t grainSize = ncclProtoGrainSize(task->protocol); // 获取协议的粒度大小
      // 为第一个通道计算分块
      if (countLo != 0) {
        NCCLCHECK(calcCollChunking(comm, task, /*nChannels=*/1, globalBytesPerElement*countLo, &chunkSize, &directFlags, &proxyOpLo));
        devWork->cbd.chunkGrainsLo = chunkSize/grainSize; // 存储块粒度数
      }
      // 为最后一个通道计算分块
      if (countHi != 0) {
        NCCLCHECK(calcCollChunking(comm, task, /*nChannels=*/1, globalBytesPerElement*countHi, &chunkSize, &directFlags, &proxyOpHi));
        devWork->cbd.chunkGrainsHi = chunkSize/grainSize;
      }
      // 为中间通道计算分块
      if (nMidChannels != 0) {
        NCCLCHECK(calcCollChunking(comm, task, /*nChannels=*/1, globalBytesPerElement*countMid, &chunkSize, &directFlags, &proxyOpMid));
        devWork->cbd.chunkGrainsMid = chunkSize/grainSize;
      }
      devWork->direct = directFlags;                     // 设置直接标志

      // 更新当前通道和剩余流量预算
      if (countHi != 0) {                                // 如果有最后一个通道
        channelId += nChannels-1;                         // 移动到最后一个通道
        currentTraffic = cellsHi*elementsPerCell*trafficPerElement; // 计算流量
      } else if (nMidChannels != 0) {                    // 如果有中间通道
        channelId += nChannels;                           // 移动到下一个可用通道
        currentTraffic = 0;                               // 重置流量
      } else {                                            // 只有第一个通道
        currentTraffic += cellsLo*elementsPerCell*trafficPerElement;
      }

      // 如果当前通道流量已满且不是最后一个可用通道，移动到下一个通道
      if (currentTraffic >= trafficPerChannel && channelId+1 != nMaxChannels[kind]) {
        channelId += 1;                                   // 移动到下一个通道
        currentTraffic = 0;                               // 重置流量
      }

      // 生成代理操作 ID
      uint64_t proxyOpId = uint64_t(plan->collOpCount++)<<1 | 0;
      for (int c=devWork->channelLo; c <= (int)devWork->channelHi; c++) {
        struct ncclProxyOp* proxyOp;
        if (c == (int)devWork->channelLo) {
          proxyOp = &proxyOpLo;
          proxyOp->loopOffset = 0;
          proxyOp->channelSize = countLo * elementSize;
        } else if (c == (int)devWork->channelHi) {
          proxyOp = &proxyOpHi;
          proxyOp->loopOffset = (countLo + nMidChannels * countMid) * elementSize;
          proxyOp->channelSize = countHi * elementSize;
        } else {
          proxyOp = &proxyOpMid;
          proxyOp->loopOffset = (countLo + (c - devWork->channelLo - 1) * countMid) * elementSize;
          proxyOp->channelSize = countMid * elementSize;
        }
        // 为每个通道添加代理操作
        proxyOp->channelId = c;                           // 设置通道 ID
        proxyOp->opCount = proxyOpId;                     // 设置操作计数
        proxyOp->task.coll = task;                        // 关联任务
        proxyOp->rank = comm->rank;                       // 设置 rank
        proxyOp->ringAlgo = NULL;                         // 初始化环形算法指针
        // 如果使用了内存注册且是环形算法且有网络句柄，创建环形算法对象
        if (proxyOp->reg && task->algorithm == NCCL_ALGO_RING && (task->recvNetHandles[c] || task->sendNetHandles[c])) {
          // 根据不同的集合通信类型创建相应的环形算法对象
          if (task->func == ncclFuncAllGather) {
            // 创建 AllGather 环形算法对象
            proxyOp->ringAlgo = new RingAGAlgorithm(task->sendbuff, task->recvbuff, comm->nRanks, comm->channels[c].ring.userRanks, proxyOp->chunkSteps, proxyOp->sliceSteps, proxyOp->chunkSize, proxyOp->sliceSize, proxyOp->loopOffset, proxyOp->channelSize, elementSize, task->count * elementSize, task->sendNetHandles[c], task->recvNetHandles[c], task->srecvNetHandles[c]);
          } else if (task->func == ncclFuncAllReduce) {
            // 创建 AllReduce 环形算法对象
            proxyOp->ringAlgo = new RingARAlgorithm(task->sendbuff, task->recvbuff, comm->nRanks, comm->channels[c].ring.index, proxyOp->chunkSteps, proxyOp->sliceSteps, proxyOp->chunkSize, proxyOp->sliceSize, proxyOp->loopOffset, proxyOp->channelSize, elementSize, task->sendNetHandles[c], task->recvNetHandles[c], task->srecvNetHandles[c]);
          } else if (task->func == ncclFuncBroadcast) {
            // 创建 Broadcast 环形算法对象
            proxyOp->ringAlgo = new RingBCAlgorithm(task->sendbuff, task->recvbuff, comm->rank, task->root, comm->nRanks, comm->channels[c].ring.userRanks, proxyOp->chunkSteps, proxyOp->sliceSteps, proxyOp->chunkSize, proxyOp->sliceSize, proxyOp->loopOffset, proxyOp->channelSize, task->sendNetHandles[c], task->recvNetHandles[c], task->srecvNetHandles[c]);
          }
          proxyOp->ringAlgo->incRefCount();               // 增加引用计数
        }
        proxyOp->eActivationMask = task->eActivationMask; // 设置事件激活掩码
        proxyOp->incWorkCounter = true;                    // 增加工作计数器
        proxyOp->nChannels = nChannels;                    // 设置通道数
        addWorkBatchToPlan(comm, plan, c, workNode->workType, task->devFuncId, plan->workBytes);
        // Coverity 报告 "proxyOp->connection" 可能未初始化。很难确定
        // 这是否真的正确，但也不清楚这是否会是一个问题。
        // coverity[uninit_use_in_call:FALSE]
        NCCLCHECK(addProxyOpIfNeeded(comm, plan, proxyOp)); // 添加代理操作
        NCCLCHECK(addProfilerProxyOpIfNeeded(comm, plan, proxyOp)); // 添加性能分析器代理操作
      }
    }

    // 更新计划的通道掩码（标记使用的通道范围）
    plan->channelMask |= (2ull<<devWork->channelHi) - (1ull<<devWork->channelLo);
    // 更新每个块的线程数（取最大值）
    plan->threadPerBlock = std::max(plan->threadPerBlock, task->nWarps*WARP_SIZE);
    // 如果内核尚未特化，设置内核函数
    if (!plan->kernelSpecialized) {
      plan->kernelFn = ncclDevKernelForFunc[task->devFuncId]; // 获取内核函数指针
      plan->kernelSpecialized = ncclDevKernelForFuncIsSpecialized[task->devFuncId]; // 检查是否特化
    }
    // 性能分析器
    plan->groupApiEventHandle = task->groupApiEventHandle; // 保存组 API 事件句柄

    // 在 rank 0 上输出调优信息
    if (comm->rank == 0) {
      INFO(NCCL_TUNING, "%s: %ld Bytes -> Algo %s proto %s channel{Lo..Hi}={%d..%d}",
        ncclFuncToString(task->func), task->count * ncclTypeSize(task->datatype), ncclAlgoToString(task->algorithm),
        ncclProtoToString(task->protocol), devWork->channelLo, devWork->channelHi);

      // 输出详细的集合通信跟踪信息
      if (task->isCollnet) {
        TRACE(NCCL_COLL, "Collective %s(%s, %s, %s, %s) count=%ld devFuncId=%d channel{Lo..Hi}={%d..%d} count=%ld chunkCount=%d",
          ncclFuncToString(task->func), ncclDevRedOpToString(task->opDev.op),
          ncclDatatypeToString(task->datatype), ncclAlgoToString(task->algorithm),
          ncclProtoToString(task->protocol),
          (long)task->count, task->devFuncId, devWork->channelLo, devWork->channelHi,
          (long)devWork->collnet.count, devWork->collnet.chunkCount);
      } else {
        TRACE(NCCL_COLL, "Collective %s(%s, %s, %s, %s) count=%ld devFuncId=%d channel{Lo..Hi}={%d..%d} count{Lo,Mid,Hi}={%ld,%ld,%ld} chunkBytes{Lo,Mid,Hi}={%d,%d,%d}",
          ncclFuncToString(task->func), ncclDevRedOpToString(task->opDev.op),
          ncclDatatypeToString(task->datatype), ncclAlgoToString(task->algorithm),
          ncclProtoToString(task->protocol),
          (long)task->count, task->devFuncId, devWork->channelLo, devWork->channelHi,
          (long)devWork->cbd.countLo, (long)devWork->cbd.countMid, (long)devWork->cbd.countHi,
          int(devWork->cbd.chunkGrainsLo*ncclProtoGrainSize(task->protocol)),
          int(devWork->cbd.chunkGrainsMid*ncclProtoGrainSize(task->protocol)),
          int(devWork->cbd.chunkGrainsHi*ncclProtoGrainSize(task->protocol)));
      }
    }

    // 将清理队列元素转移到计划的清理队列
    for (int i=0; i < task->nCleanupQueueElts; i++) {
      ncclIntruQueueEnqueue(&plan->cleanupQueue, ncclIntruQueueDequeue(&planner->collCleanupQueue));
    }
    // 从规划器队列中移除已处理的任务
    ncclIntruQueueDequeue(&planner->collTaskQueue);
    ncclIntruQueueDequeue(&planner->collWorkQueue);
    nPlanColls -= 1;                                      // 减少计划集合计数
    planner->nTasksColl -= 1;                             // 减少规划器任务计数
    // 将任务和工作节点加入计划的队列
    ncclIntruQueueEnqueue(&plan->collTaskQueue, task);
    ncclIntruQueueEnqueue(&plan->workQueue, workNode);
    plan->workBytes += workNode->size;                    // 累加工作字节
  }
  return ncclSuccess;                                     // 返回成功状态
}

// 定义 P2P LL 协议阈值参数
// 参数名称："P2P_LL_THRESHOLD"，默认值为 16384 字节
// 小于此阈值的 P2P 操作将使用 LL 协议
NCCL_PARAM(P2pLLThreshold, "P2P_LL_THRESHOLD", 16384);
// 定义块大小参数
// 参数名称："CHUNK_SIZE"，默认值为 0（自动选择）
// 用户可以通过此参数覆盖默认的块大小选择
NCCL_PARAM(ChunkSize, "CHUNK_SIZE", 0);

// 将 P2P 操作添加到计划中，假设批次预算中有 sizeof(ncclDevWorkBatch)
// 工作预算中有 sizeof(ncclDevWorkP2p)。"sendRank" 和 "recvRank" 必须
// 与此轮 P2P 调度的相应值匹配（没有 -1）。
// 空操作用 -1 大小编码。
// 参数 comm: NCCL 通信器指针
// 参数 plan: 内核计划指针
// 参数 nChannelsMin: 最小通道数
// 参数 nChannelsMax: 最大通道数
// 参数 p2pRound: P2P 调度轮次
// 参数 sendRank: 发送方 rank
// 参数 sendAddr: 发送缓冲区地址
// 参数 sendBytes: 发送字节数
// 参数 recvRank: 接收方 rank
// 参数 recvAddr: 接收缓冲区地址
// 参数 recvBytes: 接收字节数
// 参数 planTotalTasks: 计划中的任务总数数组
// 参数 p2pTasks: P2P 任务指针数组
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
static ncclResult_t addP2pToPlan(
    struct ncclComm* comm, struct ncclKernelPlan* plan,
    int nChannelsMin, int nChannelsMax, int p2pRound,
    int sendRank, void* sendAddr, ssize_t sendBytes,
    int recvRank, void* recvAddr, ssize_t recvBytes,
    const int planTotalTasks[], struct ncclTaskP2p** p2pTasks
  ) {
  ncclResult_t ret = ncclSuccess;                         // 初始化返回结果
  constexpr int connIndex = 1;                            // 连接索引（P2P 使用连接 1）
  bool selfSend = (sendRank == comm->rank);              // 检查是否为自己发送
  // recv: dir=0, send: dir=1
  // 使用数组索引来区分发送和接收方向
  void* addrs[2] = {recvAddr, sendAddr};                 // 地址数组：[接收, 发送]
  ssize_t bytes[2] = {recvBytes, sendBytes};             // 字节数组：[接收, 发送]
  bool protoLL[2] = {!selfSend, !selfSend};              // LL 协议标志：[接收, 发送]
  bool network[2] = {false, false};                       // 网络标志：[接收, 发送]
  bool proxySameProcess[2] = {true, true};               // 代理同进程标志：[接收, 发送]
  void** handles[2] = {NULL, NULL};                      // 句柄数组：[接收, 发送]
  uint8_t base = ncclP2pChannelBaseForRound(comm, p2pRound); // 计算 P2P 通道的基准 ID
  struct ncclProxyOp proxyOps[2] = {};                    // 代理操作数组：[接收, 发送]
  int nProxyOps = selfSend ? 0 : 2;                       // 代理操作数量（自己发送则为 0）
  if (!selfSend) {                                        // 如果不是自己发送
    // 遍历所有通道部分，确定传输属性
    for (int part=0; part < nChannelsMax; part++) {
      // 计算此部分的通道 ID
      int channelId = ncclP2pChannelForPart(comm->p2pnChannels, base, part);
      struct ncclChannelPeer** channelPeers = comm->channels[channelId].peers; // 获取通道的对等节点
      for (int dir=0; dir <= 1; dir++) {                  // 遍历接收和发送方向
        int peerRank = dir ? sendRank : recvRank;         // 获取对等节点 rank
        // 获取连接器指针（发送或接收）
        struct ncclConnector* conn = dir ? &channelPeers[peerRank]->send[connIndex]
                                         : &channelPeers[peerRank]->recv[connIndex];
        // 检查是否支持 LL 协议（所有通道都必须支持）
        protoLL[dir] &= conn->conn.buffs[NCCL_PROTO_LL] != nullptr;
        // 检查是否使用网络传输
        network[dir] |= conn->transportComm == (dir ? &netTransport.send : &netTransport.recv);
        // 检查代理是否在同一进程
        proxySameProcess[dir] &= conn->proxyConn.sameProcess;
      }
    }
  }

  // 计算 LL 协议的阈值（通道数 × 单通道阈值）
  ssize_t thresholdLL = nChannelsMax*ncclParamP2pLLThreshold();
  // 获取用户指定的块大小参数
  ssize_t paramChunkSize = ncclParamChunkSize();
  // 按方向索引的数组，其中 recv=0, send=1：
  int nChannels[2];                                       // 每个方向的通道数
  int protocol[2];                                        // 每个方向的协议
  int stepSize[2];                                        // 每个方向的步长
  int chunkSize[2];                                       // 每个方向的块大小
  int chunkDataSize[2];                                   // 每个方向的数据块大小
  int chunkDataSize_u32fp8[2];                            // 编码后的数据块大小（u32fp8 格式）
  bool netRegistered[2] = {false, false};                // 网络注册标志：[接收, 发送]
  bool ipcRegistered[2] = {false, false};                // IPC 注册标志：[接收, 发送]

  for (int dir=0; dir < 2; dir++) {                      // 0=recv, 1=send（遍历接收和发送方向）
    // 如果字节数不是 -1（有效操作）且超过阈值，则不使用 LL 协议
    if (bytes[dir] != -1) 
        protoLL[dir] &= bytes[dir] <= thresholdLL;
    // 根据标志选择协议（LL 或 SIMPLE）
    protocol[dir] = protoLL[dir] ? NCCL_PROTO_LL : NCCL_PROTO_SIMPLE;

    // 计算步长（每个步骤传输的数据量）
    stepSize[dir] = comm->buffSizes[protocol[dir]]/NCCL_STEPS;
    if (protocol[dir] == NCCL_PROTO_SIMPLE) stepSize[dir] = comm->p2pChunkSize;
    chunkSize[dir] = stepSize[dir];                       // 初始块大小等于步长
    // 如果用户指定了块大小，使用用户指定的值
    if (paramChunkSize != 0) {
      chunkSize[dir] = paramChunkSize;
    } else if (network[dir]) {                           // 如果使用网络传输
      // 为网络调整块大小
      // 对于小消息和 SIMPLE 协议，减小块大小以提高效率
      if (protocol[dir] == NCCL_PROTO_SIMPLE && bytes[dir] < stepSize[dir]) chunkSize[dir] /= 4;
      else if (bytes[dir] < 8*stepSize[dir]) chunkSize[dir] /= 2;
    }

    // 计算实际的数据块大小（LL 协议有编码开销）
    chunkDataSize[dir] = chunkSize[dir];
    if (protocol[dir] == NCCL_PROTO_LL) chunkDataSize[dir] /= 2; // LL 协议数据是块大小的一半
    chunkDataSize_u32fp8[dir] = u32fp8Encode(chunkDataSize[dir]); // 编码为 u32fp8 格式
    chunkDataSize[dir] = u32fp8Decode(chunkDataSize_u32fp8[dir]); // 解码验证
    chunkSize[dir] = chunkDataSize[dir];                  // 更新块大小
    if (protocol[dir] == NCCL_PROTO_LL) chunkSize[dir] *= 2; // LL 协议块大小是数据的 2 倍

    // 如果使用网络传输
    if (network[dir]) {
      // 检查是否使用 PXN（Peer eXchange Network）
      bool pxnUsed = !ncclPxnDisable(comm) && comm->isAllNvlink && comm->maxLocalRanks > 1;
      // 如果满足注册条件：有效数据、同进程代理、SIMPLE 协议、不使用 PXN
      if (bytes[dir] > 0 && proxySameProcess[dir] && protocol[dir] == NCCL_PROTO_SIMPLE && (!pxnUsed)) {
        int regFlag = 0;                                  // 注册标志
        // 分配句柄数组
        NCCLCHECKGOTO(ncclCalloc(&handles[dir], nChannelsMax), ret, cleanup);
        // 遍历所有通道部分进行网络缓冲区注册
        for (int part = 0; part < nChannelsMax; part++) {
          int channelId = ncclP2pChannelForPart(comm->p2pnChannels, base, part); // 计算通道 ID
          struct ncclChannelPeer** channelPeers = comm->channels[channelId].peers; // 获取通道对等节点
          int peerRank = dir ? sendRank : recvRank;       // 获取对等节点 rank
          // 获取连接器指针
          struct ncclConnector* conn = dir ? &channelPeers[peerRank]->send[connIndex]
            : &channelPeers[peerRank]->recv[connIndex];
          // 如果是直接 NIC 连接，注册网络缓冲区以启用 GPU Direct RDMA
          if (conn->conn.flags & NCCL_DIRECT_NIC)
            ncclRegisterP2pNetBuffer(comm, addrs[dir], bytes[dir], conn, &regFlag, &handles[dir][part], &plan->cleanupQueue);
          if (!regFlag) break;                           // 如果注册失败，退出循环
        }
        netRegistered[dir] = regFlag ? true : false;     // 设置网络注册标志
      }
    } else if (bytes[dir] > 0 && addrs[dir] && protocol[dir] == NCCL_PROTO_SIMPLE && !selfSend) {
      // P2P 传输（不是网络），检查是否需要 IPC 注册
      int peerRank = dir ? sendRank : recvRank;           // 获取对等节点 rank
      int regFlag = 0;                                    // 注册标志
      // 只检查第一个通道（P2P 使用相同的连接配置）
      int channelId = ncclP2pChannelForPart(comm->p2pnChannels, base, 0);
      struct ncclChannelPeer** channelPeers = comm->channels[channelId].peers; // 获取通道对等节点
      // 获取连接器指针
      struct ncclConnector* conn = dir ? &channelPeers[peerRank]->send[connIndex]
        : &channelPeers[peerRank]->recv[connIndex];
      void* regAddr = NULL;                               // 注册后的地址
      // 如果支持 P2P 写或读，注册 IPC 缓冲区
      if (conn->conn.flags & (NCCL_P2P_WRITE | NCCL_P2P_READ)) {
        // 我们要求用户在两侧都注册缓冲区
        NCCLCHECKGOTO(ncclRegisterP2pIpcBuffer(comm, addrs[dir], bytes[dir], peerRank, &regFlag, &regAddr, &plan->cleanupQueue), ret, cleanup);
        if (regFlag) {                                    // 如果注册成功
          // 更新地址为注册后的地址
          if (dir == 0 && (conn->conn.flags & NCCL_P2P_WRITE)) recvAddr = regAddr; // 接收方向
          else if (dir == 1 && (conn->conn.flags & NCCL_P2P_READ)) sendAddr = regAddr; // 发送方向
        }
      }
      ipcRegistered[dir] = regFlag ? true : false;       // 设置 IPC 注册标志
    }

    // 计算使用的通道数
    if (bytes[dir] == -1) 
        nChannels[dir] = 0;            // 空操作，不使用通道
    else if (bytes[dir] == 0) 
        nChannels[dir] = 1;         // 零字节操作，使用 1 个通道
    else {
      // 计算最小和最大分区大小（根据是否为多节点）
      ssize_t minPartSize = comm->nNodes > 1 ? stepSize[dir]/2 : stepSize[dir]/8;
      ssize_t maxPartSize = comm->nNodes > 1 ? stepSize[dir]   : stepSize[dir]*32;
      // 从最小通道数开始
      nChannels[dir] = std::min<int>(nChannelsMin, divUp(bytes[dir], minPartSize));
      // 计算实际的分区大小
      size_t partSize = std::max(minPartSize, divUp(bytes[dir], nChannels[dir]));
      // 如果分区太大，增加通道数（最多增加到最大通道数的一半）
      while (partSize > maxPartSize && nChannels[dir] <= nChannelsMax/2) {
        nChannels[dir] *= 2;                             // 通道数翻倍
        partSize = divUp(bytes[dir], nChannels[dir]);    // 重新计算分区大小
      }
    }
    // 更新传播到性能分析器的通道数
    if (p2pTasks[dir])
        p2pTasks[dir]->nChannels = nChannels[dir];
  }

  // 创建 P2P 工作节点
  struct ncclWorkList* workNode;
  // 从内存栈中分配 P2P 工作节点
  workNode = ncclMemoryStackAllocInlineArray<ncclWorkList, ncclDevWorkP2p>(&comm->memScoped, 1);
  workNode->workType = ncclDevWorkTypeP2p;               // 设置工作类型为 P2P
  workNode->size = sizeof(struct ncclDevWorkP2p);        // 设置工作结构大小
  ncclIntruQueueEnqueue(&plan->workQueue, workNode);      // 将工作节点加入计划的工作队列
  uint32_t workOffset;                                   // 工作偏移量
  workOffset = plan->workBytes;                         // 记录当前工作字节偏移
  plan->workBytes += sizeof(struct ncclDevWorkP2p);      // 增加计划的工作字节

  // 初始化 P2P 工作结构
  struct ncclDevWorkP2p* work;
  work = (struct ncclDevWorkP2p*)(workNode+1);           // 获取工作结构指针
  work->nP2pChannels = comm->p2pnChannels;              // P2P 通道总数
  work->channelBase = base;                             // 通道基准 ID
  // 发送方向配置
  work->nSendChannels = nChannels[1];                   // 发送通道数
  work->sendProtoLL = protoLL[1];                        // 发送是否使用 LL 协议
  work->sendNetReg = netRegistered[1];                   // 发送是否使用网络注册
  work->sendIpcReg = ipcRegistered[1];                   // 发送是否使用 IPC 注册
  work->sendChunkSize_u32fp8 = chunkDataSize_u32fp8[1];  // 发送块大小（编码）
  work->sendRank = sendRank;                            // 发送方 rank
  work->sendAddr = sendAddr;                            // 发送缓冲区地址
  work->sendBytes = sendBytes==-1 ? 0 : sendBytes;      // 发送字节数（-1 转为 0）
  // 接收方向配置
  work->nRecvChannels = nChannels[0];                   // 接收通道数
  work->recvProtoLL = protoLL[0];                        // 接收是否使用 LL 协议
  work->recvNetReg = netRegistered[0];                   // 接收是否使用网络注册
  work->recvIpcReg = ipcRegistered[0];                   // 接收是否使用 IPC 注册
  work->recvChunkSize_u32fp8 = chunkDataSize_u32fp8[0];  // 接收块大小（编码）
  work->recvRank = recvRank;                            // 接收方 rank
  work->recvAddr = recvAddr;                            // 接收缓冲区地址
  work->recvBytes = recvBytes==-1 ? 0 : recvBytes;      // 接收字节数（-1 转为 0）
  // 检查是否启用性能分析器
  work->profilerEnabled = ncclProfilerPluginLoaded() && ((p2pTasks[0] ? p2pTasks[0] : p2pTasks[1])->eActivationMask & ncclProfileKernelCh);

  // 为每个代理操作初始化参数
  for (int dir=0; dir < nProxyOps; dir++) {              // 遍历接收和发送方向
    struct ncclProxyOp* op = &proxyOps[dir];            // 获取代理操作指针
    op->root = dir ? sendRank : recvRank;               // 根节点（对等节点 rank）
    op->sliceSteps = 1;                                  // 切片步数（P2P 不使用切片）
    op->chunkSteps = 1;                                  // 块步数（P2P 不使用块步）
    op->dtype = ncclInt8;                                // 数据类型（P2P 使用字节流）
    op->redOp = ncclSum;                                 // 规约操作（P2P 不使用）
    op->protocol = protocol[dir];                        // 协议类型
    op->pattern = dir ? ncclPatternSend : ncclPatternRecv; // 通信模式
    op->chunkSize = chunkSize[dir];                      // 块大小
    op->reg = netRegistered[dir];                        // 注册标志
    op->coll = p2pTasks[dir] ? p2pTasks[dir]->func : 0; // 关联的集合操作（如果有）
    op->collAPI = p2pTasks[dir] ? p2pTasks[dir]->collAPI : 0; // 集合操作 API（如果有）
    op->task.p2p = p2pTasks[dir];                       // 关联 P2P 任务
    op->rank = comm->rank;                              // 当前 rank
    op->eActivationMask = p2pTasks[dir] ? p2pTasks[dir]->eActivationMask : 0; // 事件激活掩码
    // 以下将在每个通道部分的 addWorkToChannels() 中修改：
    // op->buffer, op->nbytes, op->nsteps = ...;
  }

  // 计算实际使用的最大通道数
  nChannelsMax = std::max(nChannels[0], nChannels[1]);
  // 确定此计划将同时针对多少个对等节点。做一个
  // 简化的假设，即每个任务针对不同的对等节点。
  // 每个任务在 'p2pnChannels' 通道中的 'nChannelsMax' 上进行条带化。
  // 每个通道最多同时运行 NCCL_MAX_DEV_WORK_P2P_PER_BATCH 个任务。
  int maxConcurrent;                                     // 最大并发任务数
  int concurrentTasks[2];                               // 每个方向的并发任务数
  // 计算最大并发任务数：通道数 / 每任务通道数 × 每批次最大 P2P 数
  maxConcurrent = comm->p2pnChannels / nChannelsMax * NCCL_MAX_DEV_WORK_P2P_PER_BATCH;
  // 限制并发任务数不超过计划中的任务总数
  concurrentTasks[0] = std::min(planTotalTasks[0], maxConcurrent); // 接收方向
  concurrentTasks[1] = std::min(planTotalTasks[1], maxConcurrent); // 发送方向
  // 遍历所有通道部分
  for (int part=0; part < nChannelsMax; part++) {
    int incWorkCounter = -1;                             // 工作计数器增加标志（-1 表示未设置）
    // 计算此部分的通道 ID
    int channelId = ncclP2pChannelForPart(comm->p2pnChannels, base, part);
    plan->channelMask |= uint64_t(1)<<channelId;        // 标记此通道被使用
    // 首先添加批次。
    addWorkBatchToPlan(comm, plan, channelId, ncclDevWorkTypeP2p, ncclDevFuncId_P2p(), workOffset, p2pRound);
    for (int dir=0; dir < nProxyOps; dir++) {           // 遍历接收和发送方向
      // 在通道间分区步骤。
      int nParts = dir ? work->nSendChannels : work->nRecvChannels; // 此方向的通道数
      void* addr = dir ? work->sendAddr : work->recvAddr; // 缓冲区地址
      size_t bytes = dir ? work->sendBytes : work->recvBytes; // 字节数

      proxyOps[dir].recvbuff = nullptr;                 // 初始化接收缓冲区指针
      if (nParts <= part) {                              // 如果此部分超出通道数
        proxyOps[dir].nsteps = 0;                        // 无步骤（空操作）
      } else if (bytes == 0) {                           // 如果是零字节操作
        proxyOps[dir].nsteps = 1;                        // 1 个步骤
        proxyOps[dir].nbytes = 0;                        // 0 字节
      } else {                                            // 正常数据传输
        // 解码块大小
        size_t chunkDataSize = u32fp8Decode(dir ? work->sendChunkSize_u32fp8 : work->recvChunkSize_u32fp8);
        size_t partBeg, partEnd;                         // 此部分的起始和结束偏移
        // 计算此部分的数据边界
        ncclP2pPartBounds(nParts, part, bytes, &partBeg, &partEnd);
        if (proxyOps[dir].reg) {                          // 如果使用了内存注册
          // 设置缓冲区指针为注册后的地址加上偏移
          (dir ? proxyOps[dir].sendbuff : proxyOps[dir].recvbuff) = (uint8_t*)addr + partBeg;
          // 设置内存句柄
          (dir ? proxyOps[dir].sendMhandle : proxyOps[dir].recvMhandle) = handles[dir][part];
          proxyOps[dir].nbytes = partEnd - partBeg;       // 此部分的字节数
          // 网络传输的步骤数（每个网络大小限制一个步骤）
          proxyOps[dir].nsteps = DIVUP(proxyOps[dir].nbytes, NCCL_MAX_NET_SIZE);
        } else {                                          // 未使用内存注册
          // 步骤数 = 数据量 / 块大小
          proxyOps[dir].nsteps = divUp(partEnd-partBeg, chunkDataSize);
          // 每步的字节数 = min(剩余数据, 块大小)
          proxyOps[dir].nbytes = std::min(partEnd-partBeg, chunkDataSize);
        }
        // LL 协议需要调整字节数（编码开销）
        if (proxyOps[dir].protocol == NCCL_PROTO_LL) {
          proxyOps[dir].nbytes *= 2;                      // LL 协议字节数翻倍
          // 对齐到 FIFO 行大小
          proxyOps[dir].nbytes = roundUp(proxyOps[dir].nbytes, sizeof(union ncclLLFifoLine));
        }
      }

      // 为 <发送, 接收> 对增加工作计数器，而不是单个 P2P
      // 这样确保发送-接收对作为一个整体被计数
      if (proxyOps[dir].nsteps && incWorkCounter < 0) {
        proxyOps[dir].incWorkCounter = true;              // 标记需要增加工作计数器
        incWorkCounter = dir;                             // 记录哪个方向增加了计数器
      }

      if (proxyOps[dir].nsteps != 0) {                   // 如果有步骤需要执行
        // 在添加批次后计算 opCount，因为此时批次计数将
        // 等于此 P2P 所在的批次索引加 1。
        // P2P 操作使用奇数 ID（LSB = 1）
        proxyOps[dir].channelId = channelId;              // 设置通道 ID
        // 计算操作计数（P2P 使用奇数 ID）
        proxyOps[dir].opCount = uint64_t(comm->planner.wipPlan.channels[channelId].nWorkBatchesP2p)<<1 | 1;
        proxyOps[dir].nChannels = nChannels[dir];         // 设置通道数
        proxyOps[dir].nPeers = concurrentTasks[dir];       // 设置并发对等节点数
        // 添加代理操作
        NCCLCHECKGOTO(addProxyOpIfNeeded(comm, plan, &proxyOps[dir]), ret, cleanup);
        // 添加性能分析器代理操作
        NCCLCHECKGOTO(addProfilerProxyOpIfNeeded(comm, plan, &proxyOps[dir]), ret, cleanup);
      }
    }
  }
cleanup:                                                 // 清理标签
  free(handles[0]);                                      // 释放接收句柄数组
  free(handles[1]);                                      // 释放发送句柄数组
  return ret;                                            // 返回操作结果
}

// 静态函数：计算 P2P 通道数
// 此函数根据总数据大小和最小/最大约束，计算最优的通道数
// 参数 totalSize: 总数据大小
// 参数 minChannels: 最小通道数
// 参数 maxChannels: 最大通道数
// 参数 minSize: 每个通道的最小数据大小
// 参数 maxSize: 每个通道的最大数据大小
// 返回值：计算得到的通道数
static int calcP2pChannelCount(size_t totalSize, int minChannels, int maxChannels, size_t minSize, size_t maxSize) {
  // 初始大小 = max(最小大小, 总大小 / 最小通道数)
  size_t size = std::max(minSize, divUp(totalSize, minChannels));
  int nChannels = minChannels;                          // 从最小通道数开始
  // 如果大小超过最大值且通道数可以翻倍，则增加通道数
  while (size > maxSize && nChannels <= maxChannels/2) {
    nChannels *= 2;                                     // 通道数翻倍
    size = divUp(totalSize, nChannels);                 // 重新计算每个通道的大小
  }
  return nChannels;                                     // 返回计算得到的通道数
}

// 静态函数：将 P2P 任务调度到计划中
// 此函数处理 P2P Send/Recv 任务的调度，将它们分配到通道并添加到内核计划
// 参数 comm: NCCL 通信器指针
// 参数 plan: 内核计划指针
// 参数 budget: 预算指针
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
static ncclResult_t scheduleP2pTasksToPlan(
    struct ncclComm* comm, struct ncclKernelPlan* plan, struct ncclKernelPlanBudget* budget
  ) {
  int nRanks = comm->nRanks;                            // 通信域中的 rank 总数
  struct ncclKernelPlanner::Peer* peers = comm->planner.peers; // 获取对等节点数组

  // 更新计划的每块线程数（P2P 使用最大线程数）
  plan->threadPerBlock = std::max(plan->threadPerBlock, NCCL_MAX_NTHREADS);
  // 如果内核尚未特化，设置 P2P 内核函数
  if (!plan->kernelSpecialized) {
    plan->kernelFn = ncclDevKernelForFunc[ncclDevFuncId_P2p()]; // P2P 内核函数
    plan->kernelSpecialized = ncclDevKernelForFuncIsSpecialized[ncclDevFuncId_P2p()]; // 特化标志
  }

  // 计算如何分割操作
  // 尝试使用所有通道
  int nChannelsMax = comm->p2pnChannelsPerPeer;         // 每个 P2P 对的最大通道数
  int nChannelsMin = nChannelsMax;                       // 初始最小通道数等于最大通道数
  // 尝试使用所有通道，但每个操作使用一个通道
  // 如果最小通道数 × rank 数超过总 P2P 通道数，则减少最小通道数
  while (nChannelsMin*nRanks > comm->p2pnChannels && nChannelsMin > 1) 
    nChannelsMin /= 2;

  // Save the total count of send/recv tasks in the plan
  // 保存计划中发送/接收任务的总数
  int planTotalTasks[2] = {comm->planner.nTasksP2pRecv, comm->planner.nTasksP2pSend};
  while (comm->planner.nTasksP2p != 0) {
    for (int round=0; round < nRanks; round++) {
      int sendRank = comm->p2pSchedule[round].sendRank;
      int recvRank = comm->p2pSchedule[round].recvRank;
      struct ncclTaskP2p* send = ncclIntruQueueHead(&peers[sendRank].sendQueue);
      struct ncclTaskP2p* recv = ncclIntruQueueHead(&peers[recvRank].recvQueue);
      if (send == nullptr && recv == nullptr) continue;

      if (sendRank == comm->rank) {                         // 如果当前 rank 是发送方
        if (send != nullptr && recv == nullptr) {            // 有发送但没有接收
          WARN("Trying to send to self without a matching recv"); // 警告：发送给自己但没有匹配的接收
          return ncclInvalidUsage;                           // 返回无效使用错误
        }
        if (send == nullptr && recv != nullptr) {            // 有接收但没有发送
          WARN("Trying to recv to self without a matching send"); // 警告：从自己接收但没有匹配的发送
          return ncclInvalidUsage;                           // 返回无效使用错误
        }
      }
      ssize_t sendBytes = send ? send->bytes : -1;          // 获取发送字节数，空操作为 -1
      ssize_t recvBytes = recv ? recv->bytes : -1;          // 获取接收字节数，空操作为 -1
      void* sendBuff = send ? send->buff : nullptr;          // 获取发送缓冲区指针
      void* recvBuff = recv ? recv->buff : nullptr;          // 获取接收缓冲区指针

      if (sendRank == comm->rank && send->buff == recv->buff) { // 如果发送给自己且缓冲区相同
        // Skip send to self in-place (we don't need to support this).
        // 跳过原地发送给自己（我们不需要支持这种情况）
        ncclIntruQueueDequeue(&peers[sendRank].sendQueue);  // 从发送队列移除
        ncclIntruQueueDequeue(&peers[recvRank].recvQueue);  // 从接收队列移除
        ncclMemoryPoolFree(&comm->memPool_ncclTaskP2p, send); // 释放发送任务内存
        ncclMemoryPoolFree(&comm->memPool_ncclTaskP2p, recv); // 释放接收任务内存
        comm->planner.nTasksP2p -= 2;                        // 减少 P2P 任务计数（发送+接收）
        comm->planner.nTasksP2pSend -= 1;                    // 减少发送任务计数
        comm->planner.nTasksP2pRecv -= 1;                    // 减少接收任务计数
      } else {                                                // 正常的 P2P 操作
        // Ensure room for worst case of one new batch per channel.
        // 确保每个通道最坏情况下一个新批次的空间
        if (!testBudget(budget, plan->nWorkBatches+nChannelsMax, plan->workBytes + sizeof(struct ncclDevWorkP2p))) {
          return ncclSuccess;                                // 预算不足，返回成功（当前计划已满）
        }
        struct ncclTaskP2p* p2pTasks[2] = { recv, send };  // P2P 任务数组：[接收, 发送]
        NCCLCHECK(addP2pToPlan(comm, plan, nChannelsMin, nChannelsMax, round, sendRank, sendBuff, sendBytes, recvRank, recvBuff, recvBytes, planTotalTasks, p2pTasks)); // 添加 P2P 到计划
        if (send != nullptr) {                                // 如果有发送任务
          ncclIntruQueueDequeue(&peers[sendRank].sendQueue); // 从发送队列移除
          // Profiler - We can overwrite groupAPI event handles here since all operations here belong to the same group
          // 性能分析器 - 我们可以覆盖组 API 事件句柄，因为这里的所有操作都属于同一组
          plan->groupApiEventHandle = send->groupApiEventHandle; // 设置组 API 事件句柄
          ncclIntruQueueEnqueue(&plan->p2pTaskQueue, send);   // 将发送任务加入计划的 P2P 任务队列
          comm->planner.nTasksP2p -= 1;                      // 减少 P2P 任务计数
          comm->planner.nTasksP2pSend -= 1;                   // 减少发送任务计数
        }
        if (recv != nullptr) {                                // 如果有接收任务
          ncclIntruQueueDequeue(&peers[recvRank].recvQueue); // 从接收队列移除
          // Profiler - We can overwrite groupAPI event handles here since all operations here belong to the same group
          // 性能分析器 - 我们可以覆盖组 API 事件句柄，因为这里的所有操作都属于同一组
          plan->groupApiEventHandle = recv->groupApiEventHandle; // 设置组 API 事件句柄
          ncclIntruQueueEnqueue(&plan->p2pTaskQueue, recv);   // 将接收任务加入计划的 P2P 任务队列
          comm->planner.nTasksP2p -= 1;                      // 减少 P2P 任务计数
          comm->planner.nTasksP2pRecv -= 1;                   // 减少接收任务计数
        }
      }
    }
  }
  return ncclSuccess;                                       // 返回成功状态
}

// Spin until its safe to increase comm->workFifoProduced to desiredProduced.
// 循环等待直到可以安全地将 comm->workFifoProduced 增加到 desiredProduced
// 此函数确保工作 FIFO 有足够的空间，避免溢出
// 参数 comm: 通信器指针
// 参数 desiredProduced: 期望的生产位置（FIFO 写入位置）
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
static ncclResult_t waitWorkFifoAvailable(struct ncclComm* comm, uint32_t desiredProduced) {
  // 计算是否有足够的空间：生产位置 - 消费位置 <= FIFO 大小
  bool hasRoom = (desiredProduced - comm->workFifoConsumed) <= comm->workFifoBytes;
  if (!hasRoom) {                                        // 如果空间不足
    while (true) {                                       // 循环等待空间可用
      NCCLCHECK(ncclCommPollEventCallbacks(comm, /*waitSome=*/true)); // 轮询事件回调，处理已完成的回调
      hasRoom = (desiredProduced - comm->workFifoConsumed) <= comm->workFifoBytes; // 重新检查是否有足够空间
      if (hasRoom) break;                                // 如果有空间了，退出循环
      sched_yield();                                     // 让出 CPU，避免忙等待
    }
  }
  return ncclSuccess;                                   // 返回成功状态
}

namespace {
  // 上传工作清理结构体
  // 此结构体用于在异步拷贝完成后清理资源
  struct uploadWork_cleanup_t {
    struct ncclCommEventCallback base;                  // 基础事件回调结构体
    void *hostBuf;                                       // 主机端缓冲区指针，需要释放
  };
  // 上传工作清理函数
  // 此函数在 CUDA event 完成后调用，释放相关的资源
  // 参数 comm: 通信器指针
  // 参数 cb: 回调结构体指针
  // 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
  ncclResult_t uploadWork_cleanup_fn(
      struct ncclComm* comm, struct ncclCommEventCallback* cb
    ) {
    struct uploadWork_cleanup_t* me = (struct uploadWork_cleanup_t*)cb; // 转换为派生结构体指针
    free(me->hostBuf);                                   // 释放主机端缓冲区
    CUDACHECK(cudaEventDestroy(me->base.event));        // 销毁 CUDA event
    free(me);                                            // 释放回调结构体本身
    return ncclSuccess;                                 // 返回成功状态
  }
}

// 上传工作数据到设备
// 此函数根据不同的存储类型，将工作数据从主机端复制到设备端
// 参数 comm: 通信器指针
// 参数 plan: 内核计划指针
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
static ncclResult_t uploadWork(struct ncclComm* comm, struct ncclKernelPlan* plan) {
  // 对称集合和 CE 集合不需要上传工作数据
  if (plan->isSymColl || plan->isCeColl) return ncclSuccess;

  size_t workBytes = plan->workBytes;                    // 工作数据的总字节数
  size_t batchBytes = plan->nWorkBatches*sizeof(struct ncclDevWorkBatch); // 批次数据的总字节数
  void* fifoBufHost;                                     // 主机端 FIFO 缓冲区指针
  uint32_t fifoCursor, fifoMask;                        // FIFO 游标和掩码

  // 根据工作存储类型设置缓冲区参数
  switch (plan->workStorageType) {
  case ncclDevWorkStorageTypeArgs:                      // 使用内核参数存储
    plan->kernelArgs->workBuf = nullptr;                // 工作缓冲区设为空（数据在参数中）
    fifoBufHost = (void*)plan->kernelArgs;              // 使用内核参数作为缓冲区
    fifoCursor = sizeof(ncclDevKernelArgs) + batchBytes; // 游标从批次数据之后开始
    fifoMask = ~0u;                                     // 掩码为全 1（不使用环形缓冲区）
    break;
  case ncclDevWorkStorageTypeFifo:                      // 使用 FIFO 环形缓冲区
    fifoBufHost = comm->workFifoBuf;                    // 使用通信器的 FIFO 缓冲区
    fifoCursor = comm->workFifoProduced;                // 当前生产位置
    fifoMask = comm->workFifoBytes-1;                   // 掩码（用于环形缓冲区取模）
    NCCLCHECK(waitWorkFifoAvailable(comm, fifoCursor + workBytes)); // 等待 FIFO 有足够空间
    plan->kernelArgs->workBuf = comm->workFifoBufDev;   // 设置设备端 FIFO 缓冲区指针
    break;
  case ncclDevWorkStorageTypePersistent:                // 使用持久化缓冲区
    // We rely on 16-byte alignment
    // 我们依赖 16 字节对齐
    #if __cplusplus >= 201103L                           // C++11 及以上版本
    fifoBufHost = aligned_alloc(16, ROUNDUP(workBytes, 16)); // 使用对齐分配
    #else                                              // C++98 版本
    static_assert(16 <= alignof(max_align_t), "We rely on 16-byte alignment."); // 验证对齐要求
    fifoBufHost = malloc(workBytes);                    // 使用标准 malloc（假设已对齐）
    #endif
    fifoCursor = 0;                                     // 游标从 0 开始
    fifoMask = ~0u;                                     // 掩码为全 1
    break;
  default:
    return ncclInternalError;                           // 未知存储类型，返回内部错误
  }
  
  plan->kernelArgs->workMask = fifoMask;                // 设置 FIFO 掩码到内核参数

  // Batches were placed after kernelArgs by finishPlan(). Only thing left to
  // do is translate the work offset from zero based (in plan) to:
  //  ncclDevWorkStorageTypeArgs: offset from beginning of kernel args
  //  ncclDevWorkStorageTypeFifo: offset from base of fifo
  //  ncclDevWorkStorageTypePersistent: no translation since our dedicated buffer will also begin at zero.
  // finishPlan() 已经将批次放置在 kernelArgs 之后。剩下的唯一事情是
  // 将工作偏移量从基于零（在计划中）转换为：
  //  ncclDevWorkStorageTypeArgs: 从内核参数开始的偏移
  //  ncclDevWorkStorageTypeFifo: 从 FIFO 基准的偏移
  //  ncclDevWorkStorageTypePersistent: 不需要转换，因为我们的专用缓冲区也将从零开始
  struct ncclDevWorkBatch* batchZero = (struct ncclDevWorkBatch*)(plan->kernelArgs+1); // 获取批次数组起始位置
  for (int b=0; b < plan->nWorkBatches; b++) {           // 遍历所有批次
    batchZero[b].offsetBase += fifoCursor;               // 调整批次的偏移基准（加上当前游标）
  }

  // Write the channel-shared work structs.
  // 写入通道共享的工作结构
  //这里把ncclDevWorkColl host传入到kernel的参数写入到dst
  struct ncclWorkList* workNode = ncclIntruQueueHead(&plan->workQueue); // 获取工作队列头
  while (workNode != nullptr) {                          // 遍历所有工作节点
    char* dst = (char*)fifoBufHost;                      // 目标地址（FIFO 或参数缓冲区）
    char* src = (char*)(workNode+1);                     // 源地址（工作数据）
    //workNode->size通过函数ncclDevWorkSize计算
    for (int n = workNode->size; n != 0; n -= 16) {     // 每次复制 16 字节
      memcpy(
        __builtin_assume_aligned(dst + (fifoCursor & fifoMask), 16), // 目标地址 16 字节对齐
        __builtin_assume_aligned(src, 16),              // 源地址 16 字节对齐
        16                                              // 复制 16 字节
      );
      fifoCursor += 16;                                 // 移动游标
      src += 16;                                        // 移动源指针
    }
    workNode = workNode->next;                           // 移动到下一个工作节点
  }

  // 根据存储类型完成后续处理
  switch (plan->workStorageType) {
  case ncclDevWorkStorageTypeFifo:                      // FIFO 存储类型
    comm->workFifoProduced = fifoCursor;                // 更新 FIFO 生产位置
    if (comm->workFifoBufGdrHandle != nullptr)           // 如果使用 GPU Direct RDMA
      wc_store_fence();                                 // 写入内存屏障，确保写入完成
    break;
  case ncclDevWorkStorageTypePersistent:                // 持久化存储类型
    { ncclResult_t result = ncclSuccess;                // 结果状态
      struct uploadWork_cleanup_t* cleanup = nullptr;    // 清理回调结构体
      cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed; // 流捕获模式
      void* fifoBufDev = nullptr;                        // 设备端缓冲区指针
      cudaStream_t deviceStream;

      CUDACHECKGOTO(cudaThreadExchangeStreamCaptureMode(&mode), result, fail);

      // Acquire deviceStream. Since the user's graph will be launched later and it also
      // acquires the deviceStream, it will observe this upload.
      NCCLCHECKGOTO(ncclStrongStreamAcquire(ncclCudaGraphNone(), &comm->sharedRes->deviceStream, /*concurrent=*/false, &deviceStream), result, fail);

      // 在设备上异步分配内存，用于持久化的工作缓冲区
      // 使用 comm->memPool 作为内存池，在 deviceStream 上执行分配
      CUDACHECKGOTO(cudaMallocAsync(&fifoBufDev, workBytes, comm->memPool, deviceStream), result, fail);
      // 保存持久化缓冲区指针，用于后续释放
      plan->workBufPersistent = fifoBufDev;
      // 设置内核参数的工作缓冲区指针
      plan->kernelArgs->workBuf = fifoBufDev;

      // coverity[uninit_use_in_call:FALSE] => fifoBufHost is never NULL
      // 将主机端的工作数据异步拷贝到设备端
      CUDACHECKGOTO(cudaMemcpyAsync(fifoBufDev, fifoBufHost, workBytes, cudaMemcpyDefault, deviceStream), result, fail);
      // 创建一个 CUDA event 用于标记拷贝完成
      cudaEvent_t memcpyDone;
      CUDACHECKGOTO(cudaEventCreateWithFlags(&memcpyDone, cudaEventDisableTiming), result, fail);
      // 在 deviceStream 上记录 event，表示拷贝操作的完成点
      CUDACHECKGOTO(cudaEventRecord(memcpyDone, deviceStream), result, fail);

      // 分配清理回调结构体
      NCCLCHECKGOTO(ncclCalloc(&cleanup, 1), result, fail);
      // 设置清理函数指针
      cleanup->base.fn = uploadWork_cleanup_fn;
      // 设置清理函数的 event（拷贝完成时触发）
      cleanup->base.event = memcpyDone;
      // 保存主机端缓冲区指针，用于后续释放
      cleanup->hostBuf = fifoBufHost;
      // 将清理回调加入到通信器的 event 回调队列中
      ncclIntruQueueEnqueue(&comm->eventCallbackQueue, (struct ncclCommEventCallback *)cleanup);

      // 释放设备流（允许其他操作使用）
      NCCLCHECKGOTO(ncclStrongStreamRelease(ncclCudaGraphNone(), &comm->sharedRes->deviceStream, /*concurrent=*/false), result, fail);
      // 轮询事件回调，处理已完成的回调
      NCCLCHECKGOTO(ncclCommPollEventCallbacks(comm, /*waitSome=*/false), result, fail);

    finish_scope:
      // 恢复 CUDA 流捕获模式（如果之前修改过）
      if (mode != cudaStreamCaptureModeRelaxed) 
        (void)cudaThreadExchangeStreamCaptureMode(&mode);
      return result;
    fail:
      // 失败时，如果没有设置清理回调，直接释放主机端缓冲区
      if (!cleanup) free(fifoBufHost);
      // 跳转到清理作用域
      goto finish_scope;
    } 
    break;
  default:
    break;
  }
  
  return ncclSuccess;
}

// 上传代理操作到网络插件
// 参数 comm: 通信器指针
// 参数 plan: 内核计划指针
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
static ncclResult_t uploadProxyOps(struct ncclComm* comm, struct ncclKernelPlan* plan) {
  // 获取当前通信域的集合操作计数（作为基准）
  uint64_t collOpCount = comm->sharedRes->collOpCount;
  // 记录每个通道的 P2P 操作增量
  uint64_t p2pOpBump[MAXCHANNELS] = {/*0...*/};
  // Advance comm's collOpCount by number of colls in this plan.
  // 根据此计划中的集合操作数量增加 comm 的 collOpCount
  // 标记是否存在 P2P 操作
  int hasp2p = 0;
  // 增加共享资源和本地通信器的集合操作计数
  comm->sharedRes->collOpCount += plan->collOpCount;
  comm->collOpCount += plan->collOpCount;

  // 遍历计划中的所有代理操作
  struct ncclProxyOp* op = ncclIntruQueueHead(&plan->proxyOpQueue);
  while (op != nullptr) {
    // 设置性能分析器上下文
    op->profilerContext = comm->profilerContext;
    // 设置引擎激活掩码（根据操作类型从集合或 P2P 任务中获取）
    op->eActivationMask = op->coll <= ncclFuncAllReduce ? op->task.coll->eActivationMask : op->task.p2p->eActivationMask;
    // 设置任务事件句柄（用于性能分析和同步）
    op->taskEventHandle = op->coll <= ncclFuncAllReduce ? op->task.coll->eventHandle : op->task.p2p->eventHandle;
    // 将进程 ID 添加到代理操作中（用于性能分析）
    ncclProfilerAddPidToProxyOp(op);

    // 保存原始的操作 ID
    uint64_t oldId = op->opCount;
    // Ignoring the bottom tag bit, opCount's are zero-based within plan so
    // translate them to the tip of the comm's history.
    // 忽略最低的标签位，操作计数在计划内是从零开始的，所以需要将它们转换为通信器历史记录的顶端
    if (oldId & 1) { // p2p 操作（最低位为 1）
      // opCount is monotonic increasing within a plan's channel so just
      // remember last value to compute max.
      // 操作计数在计划的通道内是单调递增的，所以只需记住最后一个值来计算最大值
      // 记录该通道的 P2P 操作增量（+1 确保下一个计划不会冲突）
      p2pOpBump[op->channelId] = (oldId>>1) + 1; // +1 to ensure next plan doesn't collide
      // 将操作 ID 转换为全局 ID（左移 1 位并加上当前通道的基准计数）
      op->opCount = (comm->sharedRes->p2pOpCount[op->channelId]<<1) + oldId;
      hasp2p = 1;
    } else { // coll 操作（最低位为 0）
      // 将集合操作 ID 转换为全局 ID
      op->opCount = (collOpCount<<1) + oldId;
    }

    // 保存代理操作到网络插件
    NCCLCHECK(ncclProxySaveOp(comm, op, nullptr));
    // 恢复原始操作 ID（为了下次 uploadProxyOps 调用）
    op->opCount = oldId; // Restore for next uploadProxyOps()
    // 移动到下一个代理操作
    op = op->enqNext;
  }

  // 如果存在 P2P 操作，更新每个通道的 P2P 操作计数
  if (hasp2p) {
    for (int c=0; c < MAXCHANNELS; c++) {
      // Advance channel's p2pOpCount by number of p2p's in this plan channel.
      // 增加该通道的 P2P 操作计数（加上本计划中的 P2P 操作数）
      comm->sharedRes->p2pOpCount[c] += p2pOpBump[c];
    }
  }
  return ncclSuccess;
}

// 在主机流上执行计划任务（包括性能分析和代理操作）
// 参数 comm: 通信器指针
// 参数 plan: 内核计划指针
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
static ncclResult_t hostStreamPlanTask(struct ncclComm* comm, struct ncclKernelPlan* plan) {
  // 启动性能分析器的组事件（记录整个计划组的开始）
  NCCLCHECK(ncclProfilerStartGroupEvent(plan));
  // 启动性能分析器的任务事件（记录计划中各个任务的开始）
  NCCLCHECK(ncclProfilerStartTaskEvents(plan));
  // 如果计划中有代理操作，需要上传并启动代理操作
  if (ncclIntruQueueHead(&plan->proxyOpQueue)) {
    // 上传代理操作到网络插件
    NCCLCHECK(uploadProxyOps(comm, plan));
    // 启动代理操作（通知网络插件开始执行）
    NCCLCHECK(ncclProxyStart(comm));
  }
  // 停止性能分析器的任务事件（记录各个任务的结束）
  NCCLCHECK(ncclProfilerStopTaskEvents(plan));
  // 停止性能分析器的组事件（记录整个计划组的结束）
  NCCLCHECK(ncclProfilerStopGroupEvent(plan));
  // 如果计划不是持久化的，需要通知主线程进行回收
  if (!plan->persistent) {
    // Notify main thread of our reclaiming. This will reclaim plan concurrently.
    // 将回收器加入到回调队列，通知主线程可以并发回收该计划
    ncclIntruQueueMpscEnqueue(&comm->callbackQueue, &plan->reclaimer);
  }
  return ncclSuccess;
}

// 主机流计划回调函数（在 CUDA 主机流上执行的回调）
// 参数 plan_: 内核计划指针（作为 void* 传递）
// 返回值：无
static void CUDART_CB hostStreamPlanCallback(void *plan_) {
  // NVTX（NVIDIA Tools Extension）性能分析范围标记
  NCCL_NVTX3_FUNC_RANGE;
  // 将 void* 转换回内核计划指针
  struct ncclKernelPlan* plan = (struct ncclKernelPlan*)plan_;
  // 执行主机流计划任务
  ncclResult_t result = hostStreamPlanTask(plan->comm, plan);
  // 如果任务执行失败，输出警告信息
  if (result != ncclSuccess) {
    WARN("hostStreamPlanCallback() failed : %s", ncclGetErrorString(result));
  }
  return;
}

// 回收内核计划的资源
// 参数 comm: 通信器指针
// 参数 me: 回调结构体指针（实际指向 ncclKernelPlan）
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
static ncclResult_t reclaimPlan(struct ncclComm* comm, struct ncclCommCallback* me) {
  // 将回调结构体指针转换为内核计划指针（reclaimer 是第一个成员）
  struct ncclKernelPlan* plan = (struct ncclKernelPlan*)me; // cast from first member `reclaim`
  // 如果是持久化计划，需要减少持久化引用计数
  if (plan->persistent) {
    // 减少共享资源和本地通信器的持久化引用计数
    comm->sharedRes->persistentRefs -= 1;
    comm->localPersistentRefs -= 1;
    // 如果工作存储类型是持久化的，需要释放设备端内存
    if (plan->workStorageType == ncclDevWorkStorageTypePersistent) {
      // 临时切换 CUDA 流捕获模式，以便在 CUDA 图中安全释放内存
      cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed;
      // 交换流捕获模式（临时禁用捕获）
      CUDACHECK(cudaThreadExchangeStreamCaptureMode(&mode));
      // 释放持久化的工作缓冲区
      CUDACHECK(cudaFree(plan->workBufPersistent));
      // 恢复原来的流捕获模式
      CUDACHECK(cudaThreadExchangeStreamCaptureMode(&mode));
    }
  }
  // 如果是对称集合操作，释放对称内核参数
  if (plan->isSymColl) {
    free(plan->kernelSymArgs);
  }
  // Free coll tasks
  // 释放所有集合任务
  struct ncclTaskColl* ct = ncclIntruQueueHead(&plan->collTaskQueue);
  while (ct != nullptr) {
    struct ncclTaskColl* ct1 = ct->next;
    // 释放网络句柄（发送、接收、辅助接收）
    free(ct->sendNetHandles);
    free(ct->recvNetHandles);
    free(ct->srecvNetHandles);
    // 将集合任务归还到内存池
    ncclMemoryPoolFree(&comm->memPool_ncclTaskColl, ct);
    ct = ct1;
  }
  // Free p2p tasks
  // 释放所有 P2P 任务
  struct ncclTaskP2p* pt = ncclIntruQueueHead(&plan->p2pTaskQueue);
  while (pt != nullptr) {
    struct ncclTaskP2p* pt1 = pt->next;
    // 将 P2P 任务归还到内存池
    ncclMemoryPoolFree(&comm->memPool_ncclTaskP2p, pt);
    pt = pt1;
  }
  // Free proxy ops
  // 释放所有代理操作
  struct ncclProxyOp* q = ncclIntruQueueHead(&plan->proxyOpQueue);
  while (q != nullptr) {
    struct ncclProxyOp* q1 = q->enqNext;
    // 如果有环形算法对象，减少其引用计数，如果归零则删除
    if (q->ringAlgo && q->ringAlgo->decRefCount() == 0) delete q->ringAlgo;
    // 将代理操作归还到内存池
    ncclMemoryPoolFree(&comm->memPool_ncclProxyOp, q);
    q = q1;
  }
  // Run other free callbacks
  // 执行其他清理回调
  ncclResult_t result = ncclSuccess;
  while (!ncclIntruQueueEmpty(&plan->cleanupQueue)) {
    struct ncclCommCallback* cb = ncclIntruQueueDequeue(&plan->cleanupQueue);
    // 调用清理函数（期望回收 cb 的内存）
    ncclResult_t res1 = cb->fn(comm, cb); // Expect to reclaim memory of cb
    if (res1 != ncclSuccess) result = res1;
  }
  // 检查清理过程中的错误
  NCCLCHECK(result);
  // Free plan struct
  // 将内核计划结构体归还到内存池
  ncclMemoryPoolFree(&comm->memPool_ncclKernelPlan, plan);
  return ncclSuccess;
}

// 持久化计划的析构函数（在 CUDA 图销毁时调用）
// 参数 plans_: 内核计划链表的头指针
// 返回值：无
static void persistentDestructor(void* plans_) {
  // 转换为内核计划指针
  struct ncclKernelPlan* plan = (struct ncclKernelPlan*)plans_;
  // 获取通信器指针
  struct ncclComm* comm = plan->comm;
  // 遍历所有持久化计划
  while (plan != nullptr) {
    struct ncclKernelPlan* next = plan->next;
    // 将每个计划的回收器加入到通信器的回调队列
    // 这样可以并发地回收计划资源
    ncclIntruQueueMpscEnqueue(&comm->callbackQueue, &plan->reclaimer);
    plan = next;
  }
}

// 定义启动顺序隐式控制的环境变量参数
// LAUNCH_ORDER_IMPLICIT: 控制是否使用隐式启动顺序（0=禁用，1=启用）
NCCL_PARAM(LaunchOrderImplicit, "LAUNCH_ORDER_IMPLICIT", 0);

namespace {
  // 隐式顺序模式枚举
  enum ncclImplicitOrder {
    ncclImplicitOrderNone,      // 不使用隐式顺序
    ncclImplicitOrderSerial,    // 使用串行顺序（通过事件同步）
    ncclImplicitOrderLaunch     // 使用启动顺序（通过 CUDA 12.3+ 的启动完成事件）
  };
}

// 获取隐式顺序模式
// 参数 mode: 输出参数，返回隐式顺序模式
// 参数 capturing: 是否正在捕获 CUDA 图
// 参数 driver: CUDA 驱动版本（-1 表示自动检测）
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
static ncclResult_t getImplicitOrder(enum ncclImplicitOrder *mode, bool capturing, int driver=-1) {
  // 如果启用了启动顺序隐式控制参数
  if (ncclParamLaunchOrderImplicit()) {
    // 如果驱动版本未指定，自动检测驱动版本
    if (driver < 0) { NCCLCHECK(ncclCudaDriverVersion(&driver)); }
    // 如果正在捕获 CUDA 图且驱动版本低于 12.0.9，使用串行顺序
    if (capturing && driver < 12090) { *mode = ncclImplicitOrderSerial; return ncclSuccess; }
    // 如果运行时和驱动版本都 >= 12.0.3，使用启动顺序；否则使用串行顺序
    *mode = 12030 <= std::min<int>(CUDART_VERSION, driver) ? ncclImplicitOrderLaunch : ncclImplicitOrderSerial;
    return ncclSuccess;
  }
  // 默认不使用隐式顺序
  *mode = ncclImplicitOrderNone;
  return ncclSuccess;
}

// 准备启动内核（将任务调度到内核计划中）
// 这是 NCCL 内核启动流程的核心函数，负责：
// 1. 将集合操作和 P2P 任务调度到内核计划中
// 2. 设置流同步和事件依赖
// 3. 启动主机回调（用于代理操作）
// 参数 comm: 通信器指针
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
ncclResult_t ncclLaunchPrepare(struct ncclComm* comm) {
  ncclResult_t result = ncclSuccess;
  // 获取内核规划器
  struct ncclKernelPlanner* planner = &comm->planner;
  // 检查是否正在捕获 CUDA 图（持久化模式）
  bool persistent = ncclCudaGraphValid(planner->capturingGraph);
  // 设置持久化标志
  planner->persistent = persistent;
  // 计划计数器
  int nPlans = 0;

  // 如果有待处理的任务（集合操作、P2P、对称集合、CE 集合）
  if (planner->nTasksColl + planner->nTasksP2p != 0 ||
      !ncclIntruQueueEmpty(&planner->collSymTaskQueue) ||
      !ncclIntruQueueEmpty(&planner->collCeTaskQueue)) {
    // 循环处理所有任务，直到所有任务都被调度
    do {
      // 清空进行中的计划
      memset(&planner->wipPlan, 0, sizeof(planner->wipPlan));

      // 从内存池分配一个新的内核计划
      struct ncclKernelPlan* plan = ncclMemoryPoolAlloc<struct ncclKernelPlan>(&comm->memPool_ncclKernelPlan, &comm->memPermanent);
      // 设置计划的通信器
      plan->comm = comm;
      // 设置计划的回收函数
      plan->reclaimer.fn = reclaimPlan;
      // 设置持久化标志
      plan->persistent = persistent;
      // finishPlan() promotes ncclDevWorkStorageType[Fifo|Persistent]->Args if the work can fit.
      // 设置工作存储类型：
      // - 持久化模式：使用 Persistent（在设备上分配专用内存）
      // - 非持久化模式：使用 Fifo（使用环形缓冲区）
      plan->workStorageType = persistent ? ncclDevWorkStorageTypePersistent
                                         : ncclDevWorkStorageTypeFifo;

      // 优先处理 CE（Collective Engine）集合操作任务
      if (!ncclIntruQueueEmpty(&planner->collCeTaskQueue)) {
        // 获取 CE 集合任务
        struct ncclTaskColl* task = ncclIntruQueueHead(&planner->collCeTaskQueue);
        // 标记为 CE 集合操作
        plan->isCeColl = true;
        // 在栈内存中分配 CE 集合参数
        plan->ceCollArgs = ncclMemoryStackAlloc<struct ncclCeCollArgs>(&comm->memScoped);
        // 设置 CE 集合参数（从任务中复制）
        plan->ceCollArgs->rootRank = task->root;           // 根 rank
        plan->ceCollArgs->nElts = task->count;             // 元素数量
        plan->ceCollArgs->eltSize = ncclTypeSize(task->datatype);  // 元素大小
        plan->ceCollArgs->sendBuff = (uint8_t*)task->sendbuff;    // 发送缓冲区
        plan->ceCollArgs->recvBuff = (uint8_t*)task->recvbuff;    // 接收缓冲区
        plan->ceCollArgs->func = task->func;               // 集合操作函数
        plan->ceCollArgs->sendWin = task->sendWin;         // 发送窗口
        plan->ceCollArgs->recvWin = task->recvWin;         // 接收窗口

        // 将计划加入到计划队列
        ncclIntruQueueEnqueue(&planner->planQueue, plan);
        // 从 CE 任务队列中移除任务
        ncclIntruQueueDequeue(&planner->collCeTaskQueue);
        // 释放任务内存
        ncclMemoryPoolFree(&comm->memPool_ncclTaskColl, task);
        // 增加计划计数
        nPlans += 1;
      } else {
        // 如果没有 CE 任务，检查对称集合任务
        if (!ncclIntruQueueEmpty(&planner->collSymTaskQueue)) {
          // 调度对称集合任务
          NCCLCHECKGOTO(ncclSymmetricTaskScheduler(comm, &planner->collSymTaskQueue, plan), result, failure);
        }
        else {
          // 没有特殊任务，处理常规的集合和 P2P 任务
          struct ncclKernelPlanBudget budget;
          // 计算内核参数的可用字节数（总字节数减去内核参数结构体大小）
          budget.inArgsBytes = comm->workArgsBytes - sizeof(struct ncclDevKernelArgs);
          // Non-persistent kernels fill up at most half of our fifo per kernel.
          // 非持久化内核最多使用 FIFO 的一半空间（避免 FIFO 溢出）
          // 持久化内核可以使用 1GB 空间（大值表示无限制）
          budget.outArgsBytes = plan->persistent ? (1<<30) : comm->workFifoBytes/2;

          // Drain coll tasks first. This is essential since we partition tasks based
          // on the work budget and p2p work isn't collective. If we were to drain p2p
          // first, the place where we cut the kernel could vary by rank which would
          // cause the "shortest channel first" channel picker to have divergent results.
          // 首先处理集合任务。这很重要，因为：
          // 1. 我们基于工作预算来分割任务
          // 2. P2P 工作不是集合性的
          // 3. 如果先处理 P2P，内核的分割位置可能因 rank 而异，导致"最短通道优先"的通道选择器产生分歧结果
          if (planner->nTasksColl != 0) {
            // 将集合任务调度到计划中
            NCCLCHECKGOTO(scheduleCollTasksToPlan(comm, plan, &budget), result, failure);
          }
          // And only drain p2p tasks once colls are depleted.
          // 只有在集合任务处理完毕后才处理 P2P 任务
          if (planner->nTasksColl == 0 && planner->nTasksP2p != 0) {
            // 将 P2P 任务调度到计划中
            NCCLCHECKGOTO(scheduleP2pTasksToPlan(comm, plan, &budget), result, failure);
          }
        }

        // 完成计划（计算工作字节数、优化存储类型等）
        finishPlan(comm, plan);
        // 如果计划有工作内容（workBytes != 0），将其加入计划队列
        if (plan->workBytes != 0) {
          ncclIntruQueueEnqueue(&planner->planQueue, plan);
          nPlans += 1;
        }
      }
    // 循环继续，直到所有任务都被处理完毕
    } while (planner->nTasksColl + planner->nTasksP2p != 0 ||
             !ncclIntruQueueEmpty(&planner->collSymTaskQueue) ||
             !ncclIntruQueueEmpty(&planner->collCeTaskQueue));

    // 获取计划队列的头（第一个计划）
    struct ncclKernelPlan* planHead = ncclIntruQueueHead(&planner->planQueue);
    // 保存未启动计划的头指针（用于后续启动）
    planner->unlaunchedPlansHead = planHead;

    // 如果没有计划，直接返回成功
    if (nPlans == 0)
        return ncclSuccess;

    // 获取启动流（第一个用户流）
    cudaStream_t launchStream = planner->streams->stream;
    cudaStream_t deviceStream, launchOrder;
    // 获取设备流（非并发访问）
    NCCLCHECKGOTO(ncclStrongStreamAcquire(planner->capturingGraph, &comm->sharedRes->deviceStream, /*concurrent=*/false, &deviceStream), result, failure);

    // userStream[0] waits on each userStream[i]...
    // 让启动流（userStream[0]）等待所有其他用户流
    // 这样确保所有用户流的工作在启动流之前完成
    for (struct ncclCudaStreamList* l=planner->streams->next; l != nullptr; l = l->next) {
      // 在每个用户流上记录事件
      CUDACHECKGOTO(cudaEventRecord(comm->sharedRes->scratchEvent, l->stream), result, failure);
      // 让启动流等待该事件
      CUDACHECKGOTO(cudaStreamWaitEvent(launchStream, comm->sharedRes->scratchEvent, 0), result, failure);
    }
    // userStream[0] waits on deviceStream
    // 让启动流等待设备流（确保设备流的工作完成）
    NCCLCHECKGOTO(ncclStreamWaitStream(launchStream, deviceStream, comm->sharedRes->scratchEvent), result, failure);

    // 检查是否正在捕获 CUDA 图
    bool capturing = ncclCudaGraphValid(planner->capturingGraph);
    enum ncclImplicitOrder implicitOrder;
    cudaError_t status = cudaSuccess;
    // 获取隐式顺序模式
    NCCLCHECKGOTO(getImplicitOrder(&implicitOrder, capturing), result, failure);

    // 如果需要使用隐式顺序
    if (implicitOrder != ncclImplicitOrderNone) {
      // userStream[0] waits on per-device (context) launchOrder. Concurrent strong stream access is
      // required if this is a graph capture, non-captured cannot be concurrent because that would violate
      // deterministic program order of launches.
      // 让启动流等待每个设备（上下文）的启动顺序流
      // 如果正在捕获图，需要并发访问强流；非捕获时不能并发，因为这会违反启动的确定性程序顺序
      bool concurrent = capturing;
      // 获取启动顺序流
      NCCLCHECKGOTO(ncclStrongStreamAcquire(planner->capturingGraph, &comm->context->launchOrder, concurrent, &launchOrder), result, failure);
      // 让启动流等待启动顺序流
      NCCLCHECKGOTO(ncclStreamWaitStream(launchStream, launchOrder, comm->sharedRes->scratchEvent), result, failure);
    }

    // 检查是否需要使用主机流（用于代理操作）
    if (!persistent && comm->sharedRes->persistentRefs) status = cudaEventQuery(comm->sharedRes->hostStream.serialEvent);
    // 如果是持久化模式、CUDA 启动阻塞、或持久化引用未完成，需要启动主机任务
    if (persistent || ncclCudaLaunchBlocking || status == cudaErrorNotReady) {
      // We have to launch host tasks to push proxy args. We are careful to only
      // do this if necessary since host tasks impose a high performance cost in CUDA.
      // 我们必须启动主机任务来推送代理参数。但要谨慎，只在必要时这样做，因为主机任务在 CUDA 中有很高的性能开销
      bool acquired = false;
      cudaStream_t hostStream;
      // 遍历所有计划，查找有代理操作的计划
      for (struct ncclKernelPlan* plan=planHead; plan != nullptr; plan = plan->next) {
        if (plan->hasProxyOps) {
          // 如果这是第一个有代理操作的计划，获取主机流
          if (!acquired) {
            acquired = true;
            // 获取主机流（非并发访问）
            NCCLCHECKGOTO(ncclStrongStreamAcquire(planner->capturingGraph, &comm->sharedRes->hostStream, /*concurrent=*/false, &hostStream), result, failure);
          }
          // 标记计划已加入主机回调
          plan->isHostCbEnq = true;
          // 在主机流上启动主机函数回调（用于处理代理操作）
          CUDACHECKGOTO(cudaLaunchHostFunc(hostStream, hostStreamPlanCallback, plan), result, failure);
        }
      }
      // 如果获取了主机流
      if (acquired) {
        // Make to-be-launched kernels dependent on just-launched host stream tasks.
        // 让待启动的内核依赖刚刚启动的主机流任务
        // 这样确保代理操作在内核启动前完成
        NCCLCHECKGOTO(ncclStreamWaitStream(launchStream, hostStream, comm->sharedRes->scratchEvent), result, failure);
        // 释放主机流
        NCCLCHECKGOTO(ncclStrongStreamRelease(planner->capturingGraph, &comm->sharedRes->hostStream, /*concurrent=*/false), result, failure);
      }
    }

    // 如果是持久化模式（正在捕获 CUDA 图）
    if (persistent) {
      // 增加持久化引用计数（共享资源和本地通信器）
      comm->sharedRes->persistentRefs += nPlans;
      comm->localPersistentRefs += nPlans;
      // 将持久化析构函数添加到 CUDA 图中（在图销毁时调用）
      NCCLCHECKGOTO(ncclCudaGraphAddDestructor(planner->capturingGraph, persistentDestructor, (void*)planHead), result, failure);
    }
  }
failure:
  return result;
}

// 启动内核之前的工作（在进程内屏障之后，内核启动之前调用）
// 注意：此函数不允许调用 CUDA（除非内核启动被捕获到图中）
// 参数 comm: 通信器指针
// 参数 plan: 内核计划指针
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
ncclResult_t ncclLaunchKernelBefore_NoUncapturedCuda(struct ncclComm* comm, struct ncclKernelPlan* plan) {
  // This code is called after we've checked in to the intra-process barrier
  // but before launching the kernel. We are not allowed to call CUDA unless the
  // kernel launch is captured.
  // 这段代码在进入进程内屏障之后、启动内核之前调用。除非内核启动被捕获，否则不允许调用 CUDA。
  // 上传工作数据到设备（如果需要）
  NCCLCHECK(uploadWork(comm, plan));
  return ncclSuccess;
}

#if CUDART_VERSION >= 12000
// 定义内存同步域参数（CUDA 12.0+）
// NCCL uses the "Remote" Mem Sync domain by default
// NCCL 默认使用"远程"内存同步域
NCCL_PARAM(MemSyncDomain, "MEM_SYNC_DOMAIN", cudaLaunchMemSyncDomainRemote);
#endif

// 启动 NCCL 内核
// 参数 comm: 通信器指针
// 参数 plan: 内核计划指针
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
ncclResult_t ncclLaunchKernel(struct ncclComm* comm, struct ncclKernelPlan* plan) {
  ncclResult_t ret = ncclSuccess;
  // 获取内核规划器
  struct ncclKernelPlanner* planner = &comm->planner;
  // 计算使用的通道数（通过统计 channelMask 中置位的数量）
  int nChannels = countOneBits(plan->channelMask);
  // 获取内核函数符号
  void* sym = plan->kernelFn;
  // 设置 CUDA 网格维度（每个通道一个线程块）
  dim3 grid = {(unsigned)nChannels, 1, 1};
  // 设置 CUDA 线程块维度
  dim3 block = {(unsigned)plan->threadPerBlock, 1, 1};
  // 计算动态共享内存大小
  int smem = ncclShmemDynamicSize(comm->cudaArch);
  // 获取启动流
  cudaStream_t launchStream = planner->streams->stream;

  // 启动性能分析器的内核启动事件
  NCCLCHECK(ncclProfilerStartKernelLaunchEvent(plan, launchStream));

  // 准备内核启动的额外参数（使用 CUDA 驱动 API 的参数缓冲区机制）
  void* extra[] = {
    CU_LAUNCH_PARAM_BUFFER_POINTER, plan->kernelArgs,      // 参数缓冲区指针
    CU_LAUNCH_PARAM_BUFFER_SIZE, &plan->kernelArgsSize,     // 参数缓冲区大小
    CU_LAUNCH_PARAM_END                                      // 参数结束标记
  };

  // 获取 CUDA 驱动版本
  int driverVersion;
  NCCLCHECKGOTO(ncclCudaDriverVersion(&driverVersion), ret, do_return);

  // 获取 CUDA 函数句柄（从符号）
  CUfunction fn;
  CUDACHECKGOTO(cudaGetFuncBySymbol(&fn, sym), ret, do_return);

  // 如果 CUDA 运行时和驱动版本都 >= 11.8，使用扩展启动配置
  if (CUDART_VERSION >= 11080 && driverVersion >= 11080) {
  #if CUDART_VERSION >= 11080
    // 获取计算能力
    int compCap = comm->compCap;
    // 如果计算能力 >= 90（Hopper 架构），获取线程块簇大小
    unsigned int clusterSize = (compCap >= 90) ? comm->config.cgaClusterSize : 0;

    // 初始化启动配置
    CUlaunchConfig launchConfig = {0};
    // 初始化启动属性数组（最多 6 个属性）
    CUlaunchAttribute launchAttrs[6] = {};
    int attrs = 0;
    /* Cooperative Group Array (CGA)
     * On sm90 and later we have an extra level of hierarchy where we
     * can group together several blocks within the Grid, called
     * Thread Block Clusters.
     * Clusters enable multiple thread blocks running concurrently
     * across multiple SMs to synchronize and collaboratively fetch
     * and exchange data. A cluster of blocks are guaranteed to be
     * concurrently scheduled onto a group of SMs.
     * The maximum value is 8 and it must be divisible into the grid dimensions
     */
    /* 协作组数组（CGA）
     * 在 sm90 及更高版本上，我们有额外的层次结构，可以在网格中将多个线程块组合在一起，
     * 称为线程块簇。
     * 簇使多个线程块能够跨多个 SM 并发运行，以同步和协作地获取和交换数据。
     * 线程块簇保证被并发调度到一组 SM 上。
     * 最大值为 8，且必须能整除网格维度
     */
    if (clusterSize) {
      // Grid dimension must be divisible by clusterSize
      // 网格维度必须能被簇大小整除
      if (grid.x % clusterSize) clusterSize = 1;
      // 设置簇维度属性
      launchAttrs[attrs].id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
      launchAttrs[attrs++].value.clusterDim = {clusterSize, 1, 1};
      // 设置簇调度策略（SPREAD 表示分散到不同 SM）
      launchAttrs[attrs].id = CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE;
      launchAttrs[attrs++].value.clusterSchedulingPolicyPreference = CU_CLUSTER_SCHEDULING_POLICY_SPREAD;
    }
    #if CUDART_VERSION >= 12000
    // CUDA 12.0+ 支持，设置内存同步域（sm90 及以上）
    if (compCap >= 90 && driverVersion >= 12000) {
      // Set the NCCL Mem Sync domain on CUDA 12.0 and later (sm90)
      // 在 CUDA 12.0 及更高版本（sm90）上设置 NCCL 内存同步域
      launchAttrs[attrs].id = CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN;
      launchAttrs[attrs++].value.memSyncDomain = (CUlaunchMemSyncDomain) ncclParamMemSyncDomain();
    }
    #endif
    #if CUDART_VERSION >= 12030
    // CUDA 12.3+ 支持，使用启动完成事件
    enum ncclImplicitOrder implicitOrder;
    NCCLCHECKGOTO(getImplicitOrder(&implicitOrder, plan->persistent, driverVersion), ret, do_return);
    if (implicitOrder == ncclImplicitOrderLaunch) {
      // 设置启动完成事件属性
      launchAttrs[attrs].id = CU_LAUNCH_ATTRIBUTE_LAUNCH_COMPLETION_EVENT;
      launchAttrs[attrs].value.launchCompletionEvent.event = comm->sharedRes->launchEvent;
      launchAttrs[attrs].value.launchCompletionEvent.flags = 0;
      attrs++;
    }
    // 如果是对称集合操作且 CUDA 版本 >= 12.3，启用程序化流序列化
    if (plan->isSymColl && compCap >= 90 && driverVersion >= 12030) {
      // 设置程序化流序列化属性（允许 CUDA 优化流执行顺序）
      launchAttrs[attrs].id = CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION;
      launchAttrs[attrs].value.programmaticStreamSerializationAllowed = 1;
      attrs++;
    }
    #endif
    #if CUDART_VERSION >= 13000
    // CUDA 13.0+ 支持，设置 NVLink 利用率中心调度（计算能力 >= 100）
    if (compCap >= 100 && driverVersion >= 13000) {
      // 设置 NVLink 利用率中心调度属性
      launchAttrs[attrs].id = CU_LAUNCH_ATTRIBUTE_NVLINK_UTIL_CENTRIC_SCHEDULING;
      launchAttrs[attrs].value.nvlinkUtilCentricScheduling = comm->config.nvlinkCentricSched;
      attrs++;
    }
    #endif
    // 设置网格维度
    launchConfig.gridDimX = grid.x;
    launchConfig.gridDimY = grid.y;
    launchConfig.gridDimZ = grid.z;
    // 设置线程块维度
    launchConfig.blockDimX = block.x;
    launchConfig.blockDimY = block.y;
    launchConfig.blockDimZ = block.z;
    // 设置动态共享内存大小
    launchConfig.sharedMemBytes = smem;
    // 设置启动属性数组
    launchConfig.attrs = launchAttrs;
    // 设置启动属性数量
    launchConfig.numAttrs = attrs;
    // 设置启动流
    launchConfig.hStream = launchStream;
    // 使用扩展启动接口启动内核（cuLaunchKernelEx）
    CUCHECKGOTO(cuLaunchKernelEx(&launchConfig, fn, nullptr, extra), ret, do_return);
  #endif
  } else {
    // Standard kernel launch
    // 使用标准内核启动接口（旧版本 CUDA 或驱动）
    CUCHECKGOTO(cuLaunchKernel(fn, grid.x, grid.y, grid.z, block.x, block.y, block.z, smem, launchStream, nullptr, extra), ret, do_return);
  }

do_return:
  // 停止性能分析器的内核启动事件
  NCCLCHECK(ncclProfilerStopKernelLaunchEvent(plan));
  return ret;
}

// 启动内核之后的工作（不调用 CUDA API）
// 参数 comm: 通信器指针
// 参数 plan: 内核计划指针
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
ncclResult_t ncclLaunchKernelAfter_NoCuda(struct ncclComm* comm, struct ncclKernelPlan* plan) {
  // 如果计划没有使用主机回调（即没有使用主机流处理代理操作和回收提交）
  if (!plan->isHostCbEnq) {
    // we are not using the host stream for proxy ops and reclaimation submission, call
    // hostStreamPlanTask directly
    // 我们没有使用主机流来处理代理操作和回收提交，直接调用 hostStreamPlanTask
    NCCLCHECK(hostStreamPlanTask(comm, plan));
  }
  return ncclSuccess;
}

namespace {
  // 内核完成回调结构体
  struct KernelFinishCallback {
    struct ncclCommEventCallback base;  // 基础回调结构体
    uint32_t workFifoConsumed;          // 工作_fifo 消耗的位置
  };
  // 内核完成回调函数（更新 workFifoConsumed 并释放资源）
  // 参数 comm: 通信器指针
  // 参数 cb: 回调结构体指针
  // 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
  ncclResult_t KernelFinishCallback_fn(
      struct ncclComm* comm, struct ncclCommEventCallback* cb
    ) {
    // 转换为内核完成回调结构体
    struct KernelFinishCallback* me = (struct KernelFinishCallback*)cb;
    // 更新工作 FIFO 的消耗位置（允许重用 FIFO 空间）
    comm->workFifoConsumed = me->workFifoConsumed;
    // 销毁 CUDA event
    CUDACHECK(cudaEventDestroy(me->base.event));
    // 释放回调结构体内存
    free(me);
    return ncclSuccess;
  }
}

// 完成内核启动（设置流同步和事件依赖）
// 参数 comm: 通信器指针
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
ncclResult_t ncclLaunchFinish(struct ncclComm* comm) {
  // 获取内核规划器
  struct ncclKernelPlanner* planner = &comm->planner;
  // 如果计划队列不为空
  if (!ncclIntruQueueEmpty(&planner->planQueue)) {
    // Reset queue to empty without destroying plans since those will be sent
    // back to us for reclaiming via callbackQueue.
    // 重置队列为空，但不销毁计划（因为它们会通过 callbackQueue 发回给我们进行回收）
    ncclIntruQueueConstruct(&planner->planQueue);

    // 获取启动流（第一个用户流）
    cudaStream_t launchStream = planner->streams->stream; // First user stream gets launch
    cudaStream_t deviceStream, launchOrder;
    // 使用共享的临时 event
    cudaEvent_t finishedEvent = comm->sharedRes->scratchEvent;
    // 在启动流上记录完成事件（标记所有内核启动已完成）
    CUDACHECK(cudaEventRecord(finishedEvent, launchStream));

    // 如果工作 FIFO 的生产量超过了上次记录量的 1/8，注册完成回调
    if (comm->workFifoProduced - comm->workFifoProducedLastRecorded > comm->workFifoBytes/8) {
      // 更新上次记录的生产位置
      comm->workFifoProducedLastRecorded = comm->workFifoProduced;
      // 分配内核完成回调结构体
      struct KernelFinishCallback* cb;
      NCCLCHECK(ncclCalloc(&cb, 1));
      // 设置回调的 event（使用 finishedEvent）
      cb->base.event = finishedEvent;
      // 设置回调函数指针
      cb->base.fn = KernelFinishCallback_fn;
      // 记录当前的工作 FIFO 生产位置（回调时会设置为消耗位置）
      cb->workFifoConsumed = comm->workFifoProduced;
      // 将回调加入到事件回调队列
      ncclIntruQueueEnqueue(&comm->eventCallbackQueue, &cb->base);
      // We just stole scratchEvent so must create a new one.
      // 我们刚刚占用了 scratchEvent，所以必须创建一个新的
      CUDACHECK(cudaEventCreateWithFlags(&comm->sharedRes->scratchEvent, cudaEventDisableTiming));
    }

    // deviceStream waits on userStream[0]
    // 设备流等待启动流（确保设备流的工作在内核启动之后执行）
    // 注意：这里使用 AcquiredWorkStream 表示设备流已经有工作在进行
    NCCLCHECK(ncclStrongStreamAcquiredWorkStream(planner->capturingGraph, &comm->sharedRes->deviceStream, /*concurrent=*/false, &deviceStream));

    // We know that deviceStream is strictly behind the launchStream because launchStream
    // synced with it before kernel launch. This allows us to to see deviceStream waiting
    // on launchStream as a fast-forward. When building CUDA graphs fast forwards should
    // be handled specially so as not to create graphs with a blowup in the number of edges.
    // So we could do this:
    //   CUDACHECK(cudaStreamWaitEvent(deviceStream, finishedEvent, 0));
    // But instead we do:
    // 我们知道 deviceStream 严格位于 launchStream 之后，因为 launchStream 在内核启动之前与它同步了。
    // 这允许我们将 deviceStream 等待 launchStream 视为"快进"。
    // 在构建 CUDA 图时，应该特殊处理快进，以避免创建边数爆炸的图。
    // 所以我们可以这样做：
    //   CUDACHECK(cudaStreamWaitEvent(deviceStream, finishedEvent, 0));
    // 但我们实际这样做：
    NCCLCHECK(ncclStreamAdvanceToEvent(planner->capturingGraph, deviceStream, finishedEvent));

    // Each userStream[i] waits on userStream[0]
    // 让每个用户流等待启动流（确保所有用户流在启动流之后执行）
    for (struct ncclCudaStreamList* l=planner->streams->next; l != nullptr; l = l->next) {
      CUDACHECK(cudaStreamWaitEvent(l->stream, finishedEvent, 0));
    }
    // 检查是否正在捕获 CUDA 图
    bool capturing = ncclCudaGraphValid(planner->capturingGraph);
    enum ncclImplicitOrder implicitOrder;
    // 获取隐式顺序模式
    NCCLCHECK(getImplicitOrder(&implicitOrder, capturing));
    // 如果需要使用隐式顺序
    if (implicitOrder != ncclImplicitOrderNone) {
      // As in ncclLaunchPrepare, strong stream can be non-concurrent when non-captured.
      // 与 ncclLaunchPrepare 中一样，非捕获时强流不能并发
      bool concurrent = capturing;
      // Incorporate launch event into per-device (context) launch order.
      // 将启动事件加入到每个设备（上下文）的启动顺序中
      NCCLCHECK(ncclStrongStreamAcquiredWorkStream(planner->capturingGraph, &comm->context->launchOrder, concurrent, &launchOrder));
      // If we don't have launch events (requires CUDA 12.3) then just use completion event (serialize execution).
      // 如果没有启动事件（需要 CUDA 12.3），则使用完成事件（序列化执行）
      CUDACHECK(cudaStreamWaitEvent(launchOrder, implicitOrder == ncclImplicitOrderLaunch ? comm->sharedRes->launchEvent : finishedEvent));
      // Release launchOrder as acquired in ncclLaunchPrepare()
      // 释放启动顺序流（在 ncclLaunchPrepare 中获取的）
      NCCLCHECK(ncclStrongStreamRelease(planner->capturingGraph, &comm->context->launchOrder, concurrent));
    }
    // Release deviceStream as acquired in ncclLaunchPrepare()
    // 释放设备流（在 ncclLaunchPrepare 中获取的）
    NCCLCHECK(ncclStrongStreamRelease(planner->capturingGraph, &comm->sharedRes->deviceStream, /*concurrent=*/false));
  }
  return ncclSuccess;
}

/*****************************************************************************/
/* Enqueueing system : computation of kernel and proxy operations parameters */
/*****************************************************************************/

/*****************************************************************************/
/* Enqueueing system : computation of kernel and proxy operations parameters */
/*****************************************************************************/

// 获取集合网络（CollNet）支持情况
// 参数 comm: 通信器指针
// 参数 info: 集合任务信息指针
// 参数 collNetSupport: 输出参数，返回 CollNet 支持级别
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
static inline ncclResult_t getCollNetSupport(
    struct ncclComm* comm, struct ncclTaskColl* info, int* collNetSupport
  ) {
  // Translate ncclAvg and PreMulSum
  // 转换 ncclAvg 和 PreMulSum 操作为 netOp（CollNet 支持的操作）
  ncclRedOp_t netOp = info->opHost;
  // 如果是预乘求和或后除求和，转换为普通求和（CollNet 可能不支持这些特殊操作）
  if (info->opDev.op == ncclDevPreMulSum || info->opDev.op == ncclDevSumPostDiv) {
    netOp = ncclSum;
  }
  // 初始设置为配置的 CollNet 启用状态
  *collNetSupport = comm->config.collnetEnable;
  // 根据集合操作类型检查 CollNet 支持
  switch (info->func) {
  case ncclFuncAllReduce:
  case ncclFuncReduce:
  case ncclFuncReduceScatter:
    // 对于这些归约操作，需要检查特定操作和数据类型的 CollNet 支持矩阵
    *collNetSupport &= comm->collNetSupportMatrix[netOp][info->datatype];
    break;
  default:
    break;
  }
  return ncclSuccess;
}

// 初始化集合操作成本表（将所有算法-协议组合标记为忽略）
// 参数 collCostTable: 二维成本表指针 [算法][协议]
// 返回值：无
static void initCollCostTable(float** collCostTable) {
  // 将 collCostTable 转换为二维数组指针 [算法][协议]
  float (*table)[NCCL_NUM_PROTOCOLS] = (float (*)[NCCL_NUM_PROTOCOLS])collCostTable;
  // 遍历所有算法
  for (int a = 0; a < NCCL_NUM_ALGORITHMS; a++) {
    // 遍历所有协议
    for (int p = 0; p < NCCL_NUM_PROTOCOLS; p++) {
      // 将所有算法-协议组合的成本设置为"忽略"（表示不可用）
      table[a][p] = NCCL_ALGO_PROTO_IGNORE;
    }
  }
}

// 更新集合操作成本表（计算每个算法-协议组合的执行时间）
// 参数 comm: 通信器指针
// 参数 info: 集合任务信息指针
// 参数 nBytes: 数据字节数
// 参数 collNetSupport: CollNet 支持级别
// 参数 nvlsSupport: NVLS 支持标志
// 参数 numPipeOps: 流水线操作数量（聚合模式下可能 >1，用于调整延迟）
// 参数 collCostTable: 二维成本表指针 [算法][协议]
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
// numPipeOps: number of pipelined ops. Can be greater than 1 in aggregation mode. Used to adjust latency.
static ncclResult_t updateCollCostTable(
    struct ncclComm* comm, struct ncclTaskColl* info, size_t nBytes,
    int collNetSupport, int nvlsSupport, int numPipeOps,
    float** collCostTable) {
  // 将 collCostTable 转换为二维数组指针 [算法][协议]
  float (*table)[NCCL_NUM_PROTOCOLS] = (float (*)[NCCL_NUM_PROTOCOLS])collCostTable;

  //单rank。直接使用ring和simple
  // 特殊情况：只有一个 rank（单 GPU）
  if (comm->nRanks == 1) {
    //设为最小值
    // 直接使用 RING 算法和 SIMPLE 协议，成本设为 0（最快）
    table[NCCL_ALGO_RING][NCCL_PROTO_SIMPLE] = 0.0;
    return ncclSuccess;
  }

  // 遍历所有算法，计算每个算法在各个协议下的成本
  for (int a=0; a<NCCL_NUM_ALGORITHMS; a++) {
    // 如果是 CollNet 算法但不支持 CollNet，跳过
    if ((a == NCCL_ALGO_COLLNET_DIRECT || a == NCCL_ALGO_COLLNET_CHAIN) && collNetSupport != 1)
        continue;

    // CollNetDirect is only supported for up to 8 local GPUs
    //大于8个GPU，不支持
    // CollNetDirect 仅支持最多 8 个本地 GPU
    if (a == NCCL_ALGO_COLLNET_DIRECT && comm->maxLocalRanks > NCCL_MAX_DIRECT_ARITY+1) continue;
    // Disable CollNet Chain for more than 8 local GPUs
    // 禁用超过 8 个本地 GPU 的 CollNet Chain

    if (a == NCCL_ALGO_COLLNET_CHAIN && comm->maxLocalRanks > NCCL_MAX_DIRECT_ARITY+1)
        continue;

   //是否支持nvls
    // 如果是 NVLS 算法，检查 NVLS 支持情况
    if ((a == NCCL_ALGO_NVLS || a == NCCL_ALGO_NVLS_TREE) && (!nvlsSupport || (info->func != ncclFuncAllReduce && comm->localRanks > NCCL_MAX_NVLS_ARITY)))
        continue;

    //不支持CollNet的多节点环境中不使用NVLS算法
    // 在不支持 CollNet 的多节点环境中不使用 NVLS 算法
    if (a == NCCL_ALGO_NVLS && collNetSupport != 1 && comm->nNodes > 1)
        continue;

    /* Tree reduceScatter doesn't support scaling yet */
    // PAT（Tree）算法的 ReduceScatter 不支持缩放操作（PreMulSum/SumPostDiv）
    if (a == NCCL_ALGO_PAT && info->func == ncclFuncReduceScatter
        && (info->opDev.op == ncclDevPreMulSum || info->opDev.op == ncclDevSumPostDiv))
        continue;

    //计算选择的算法在每个协议中需要的时间
    // 遍历所有协议，计算该算法在每个协议下的执行时间
    for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
      //计算延迟时间
      // 调用拓扑层获取算法执行时间
      NCCLCHECK(ncclTopoGetAlgoTime(comm, info->func, a, p, nBytes, numPipeOps, &table[a][p]));
      // Relegate fp8 reduction trees of sufficient depth that they incur precision loss
      // to be least preferred.
      // 对于 FP8 数据类型，如果环形算法的深度会导致精度损失，降低其优先级
      if (info->datatype == ncclFloat8e4m3 || info->datatype == ncclFloat8e5m2) {
        // 如果 rank 数 > 8，环形算法会有累积精度损失
        if (a == NCCL_ALGO_RING && comm->nRanks > 8) {
          // 乘以一个大因子，使其成为最不优先的选择
          table[a][p] *= 1024.0; // Any factor large enough to act as a partition between lossy and non-lossy algos.
        }
      }
    }
  }

  return ncclSuccess;
}

// 从拓扑获取算法信息（选择最优算法、协议、通道数和线程数）
// 参数 comm: 通信器指针
// 参数 info: 集合任务信息指针（输出算法和协议选择）
// 参数 nBytes: 数据字节数
// 参数 collCostTable: 成本表指针
// 参数 simInfo: 模拟信息指针（可为 NULL）
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
static ncclResult_t topoGetAlgoInfo(
    struct ncclComm* comm, struct ncclTaskColl* info, size_t nBytes,
    float** collCostTable, ncclSimInfo_t* simInfo
  ) {
  // 将 collCostTable 转换为二维数组指针 [算法][协议]
  float (*table)[NCCL_NUM_PROTOCOLS] = (float (*)[NCCL_NUM_PROTOCOLS])collCostTable;

  // 初始化最小时间（设置为一个很大的值）
  float minTime = 3600000000.0;
  // 初始化算法和协议为未定义
  int algorithm = info->algorithm = NCCL_ALGO_UNDEF;
  int protocol = info->protocol = NCCL_PROTO_UNDEF;

  //选择一个算法和协议组合，那个组合花费时间最少
  // 遍历所有算法和协议组合，找到成本最低的
  for (int a=0; a<NCCL_NUM_ALGORITHMS; a++) {
    for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
        //=-1，跳过
      // 如果该组合被忽略（不可用），跳过
      if (table[a][p] == NCCL_ALGO_PROTO_IGNORE)
        continue;
      //选择时间最小的组合
      // 如果找到更小的成本，更新最优算法和协议
      if (table[a][p] >= 0.0 && table[a][p] < minTime) {
        algorithm = a;
        protocol = p;
        minTime = table[a][p];
      }
    }
  }

  //设置使用的算法和协议
  info->algorithm = algorithm;
  info->protocol = protocol;
  float time = minTime;

  // Yes, we are first assigning and then testing if protocol is sane, but that's OK in this case.
  // coverity[check_after_sink]
  // 检查是否找到了有效的算法和协议
  if (info->algorithm == NCCL_ALGO_UNDEF || info->protocol == NCCL_PROTO_UNDEF) {
    // 构建错误信息（包含环境变量设置）
    char ncclAlgoEnvStr[1024] = "";
    char ncclProtoEnvStr[1024] = "";
    // 检查是否设置了 NCCL_ALGO 环境变量
    const char* algoEnv = ncclGetEnv("NCCL_ALGO");
    if (algoEnv) {
      snprintf(ncclAlgoEnvStr, 1023, " NCCL_ALGO was set to %s.", algoEnv);
    }
    // 检查是否设置了 NCCL_PROTO 环境变量
    const char* protoEnv = ncclGetEnv("NCCL_PROTO");
    if (protoEnv) {
      snprintf(ncclProtoEnvStr, 1023, " NCCL_PROTO was set to %s.", protoEnv);
    }
    // 输出警告信息
    WARN("Error : no algorithm/protocol available for function %s with datatype %s.%s%s", ncclFuncToString(info->func), ncclDatatypeToString(info->datatype), ncclAlgoEnvStr, ncclProtoEnvStr);
    // 如果用户设置了环境变量，返回无效使用错误；否则返回内部错误
    return (algoEnv || protoEnv) ? ncclInvalidUsage : ncclInternalError;
  }
  // 如果提供了模拟信息，记录估计时间
  if (simInfo) simInfo->estimatedTime = time;
  // 输出跟踪信息
  TRACE(NCCL_COLL, "%ld Bytes -> Algo %d proto %d time %f", nBytes, info->algorithm, info->protocol, time);

  // 获取通道数和线程数
  int nc = comm->nChannels;
  int nt = comm->maxThreads[info->algorithm][info->protocol];
  int threadThreshold = comm->threadThresholds[info->algorithm][info->protocol];
  // 根据算法类型调整通道数
  if (info->algorithm == NCCL_ALGO_COLLNET_DIRECT) {
    // CollNet channel tuning
    // CollNet 通道调优（使用分级策略）
    int ncSwitch = 16;
    bool flag = true;
    while (ncSwitch >= 1 && flag) {
      // 如果数据量小，减少通道数
      while ((flag = nBytes < nc*nt*comm->channels[0].collnetDirect.nHeads*threadThreshold) && nc > ncSwitch) {
        // 在特定阈值点减少线程阈值
        if (nc == ncSwitch+ncSwitch/2) threadThreshold /= 2;
        nc--;
      }
      ncSwitch /= 2;
    }
  } else if (info->algorithm == NCCL_ALGO_NVLS || info->algorithm == NCCL_ALGO_NVLS_TREE) {
    // NVLS should not need more than 16 channels to get peak BW.
    // NVLS 不需要超过 16 个通道即可获得峰值带宽
    if (comm->nNodes > 1 && info->algorithm == NCCL_ALGO_NVLS) {
      // 多节点环境下，取 nvlsChannels 和 nChannels 的最小值
      nc = std::min(comm->nvlsChannels, comm->nChannels);
    } else {
      // 单节点环境，直接使用 nvlsChannels
      nc = comm->nvlsChannels;
    }
  } else {
    // Ring/Tree channel tuning
    // Ring/Tree 算法的通道调优（根据数据量减少通道数）
    while (nBytes < nc * nt * threadThreshold) {
      if (nc >= 2) nc--;
      else break;
    }
  }

  // 对于非 NVLS 和 CollNet 算法，调整线程数
  if (info->algorithm != NCCL_ALGO_NVLS && info->algorithm != NCCL_ALGO_NVLS_TREE &&
    info->algorithm != NCCL_ALGO_COLLNET_DIRECT) {
    // 如果数据量小，减少线程数（以 128 为单位）
    while (nBytes < nc * nt * threadThreshold) {
      if (nt % 128 == 0) nt /= 2;
      else break;
    }
  }
  // SIMPLE 协议需要额外的同步线程
  if (info->protocol == NCCL_PROTO_SIMPLE) {
    // Ring 算法需要额外一个 warp 用于同步
    if (info->algorithm == NCCL_ALGO_RING) nt += WARP_SIZE; // Extra warp for sync
    // More threads or sync warps needed due to split thread model
    // Tree 算法由于分割线程模型需要更多同步 warp
    if (info->algorithm == NCCL_ALGO_TREE) nt += 4*WARP_SIZE;
  }
  // 确保至少有 3 个 warp（96 个线程）
  nt = nt/WARP_SIZE < 3 ? 3*WARP_SIZE : nt;
  // Tree 算法始终使用最大线程数
  if (info->algorithm == NCCL_ALGO_TREE)
    nt = NCCL_MAX_NTHREADS; // Tree now uses all threads always.
  // PAT 算法也使用最大线程数
  if (info->algorithm == NCCL_ALGO_PAT)
    nt = NCCL_MAX_NTHREADS;

  // 设置最大通道数和 warp 数
  info->nMaxChannels = nc;
  info->nWarps = nt/WARP_SIZE;
  return ncclSuccess;
}

// 获取算法信息（使用调优插件或默认的基于拓扑的调优器）
// 参数 comm: 通信器指针
// 参数 info: 集合任务信息指针（输出算法和协议选择）
// 参数 collNetSupport: CollNet 支持级别
// 参数 nvlsSupport: NVLS 支持标志
// 参数 numPipeOps: 流水线操作数量
// 参数 simInfo: 模拟信息指针（可为 NULL）
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
// Use the default topo-based tuner if tuner plugin is not successful.
// Call the plugin first. Let it set algo+proto, and/or nChannels.
// Then, topoGetAlgoInfo will set algo/proto if not set, then nChannels and nThreads based on algo/proto.
// Finally, nChannels will be overriden by the plugin setting.
// 如果调优插件不成功，使用默认的基于拓扑的调优器。
// 首先调用插件，让插件设置算法+协议和/或通道数。
// 然后，topoGetAlgoInfo 将设置算法/协议（如果未设置），然后根据算法/协议设置通道数和线程数。
// 最后，nChannels 将被插件设置覆盖。
static ncclResult_t getAlgoInfo(
    struct ncclComm* comm, struct ncclTaskColl* info,
    int collNetSupport, int nvlsSupport, int numPipeOps, ncclSimInfo_t* simInfo/* = NULL*/
  ) {
  // 计算元素大小
  size_t elementSize = ncclTypeSize(info->datatype);
  // 计算总字节数（最大发送或接收计数）
  size_t nBytes = elementSize * ncclFuncMaxSendRecvCount(info->func, comm->nRanks, info->count);
  // 发送和接收缓冲区的注册信息
  struct ncclReg* regSendBuf = NULL;
  struct ncclReg* regRecvBuf = NULL;
  int regBuff;
  bool isSendValid, isRecvValid;
  // 计算发送和接收缓冲区大小
  size_t sendbuffSize = elementSize * ncclFuncSendCount(info->func, comm->nRanks, info->count);
  size_t recvbuffSize = elementSize * ncclFuncRecvCount(info->func, comm->nRanks, info->count);
  // 初始化算法和协议为未定义
  info->algorithm = NCCL_ALGO_UNDEF;
  info->protocol = NCCL_PROTO_UNDEF;
  // 最大通道数（由插件设置）
  int nMaxChannels = 0;
  // 成本表（算法 x 协议）
  float collCostTable[NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS];
  //初始化collCostTable，全部写为NCCL_ALGO_PROTO_IGNORE，-1
  // 初始化成本表（所有组合标记为忽略）
  initCollCostTable((float **)collCostTable);

  // 更新成本表（计算每个算法-协议组合的成本）
  NCCLCHECK(updateCollCostTable(comm, info, nBytes, collNetSupport, nvlsSupport, numPipeOps, (float **)collCostTable));

  // 如果配置了调优插件
  if (comm->tuner != NULL) {
    // 查找发送和接收缓冲区的注册信息
    NCCLCHECK(ncclRegFind(comm, info->sendbuff, sendbuffSize, &regSendBuf));
    NCCLCHECK(ncclRegFind(comm, info->recvbuff, recvbuffSize, &regRecvBuf));
    // 检查注册是否有效
    NCCLCHECK(ncclRegLocalIsValid(regSendBuf, &isSendValid));
    NCCLCHECK(ncclRegLocalIsValid(regRecvBuf, &isRecvValid));
    // 确定缓冲区是否已注册（或正在捕获图且要求图注册）
    regBuff = (regSendBuf && regRecvBuf && isSendValid && isRecvValid) || (ncclCudaGraphValid(comm->planner.capturingGraph) && ncclParamGraphRegister());
    // 调用插件的 getCollInfo 方法获取调优建议
    NCCLCHECK(comm->tuner->getCollInfo(
          comm->tunerContext, info->func, nBytes,
          numPipeOps, (float **)collCostTable, NCCL_NUM_ALGORITHMS, NCCL_NUM_PROTOCOLS,
          regBuff, &nMaxChannels));
    //获取算法信息，比如选择使用哪种算法和协议
    // 使用拓扑信息选择算法（如果插件未设置）
    NCCLCHECK(topoGetAlgoInfo(comm, info, nBytes, (float **)collCostTable, simInfo));
  } else {
    // 没有插件，直接使用拓扑信息选择算法
    NCCLCHECK(topoGetAlgoInfo(comm, info, nBytes, (float **)collCostTable, simInfo));
    // NCCL_CTA_POLICY_EFFICIENCY requires user (non-symmetric) buffer registration (currently unsupported with MNNVL)
    // CTA 效率策略需要用户（非对称）缓冲区注册（目前 MNNVL 不支持）
    if (comm->config.CTAPolicy == NCCL_CTA_POLICY_EFFICIENCY && ncclGetEnv("NCCL_ALGO") == NULL && ncclGetEnv("NCCL_PROTO") == NULL && !comm->MNNVL) {
      // make algorithm selection based on buffer registration
      // there can be other specialized policies for algorithms and protocols pickup in the future
      // 基于缓冲区注册进行算法选择（未来可能有其他专门的算法和协议选择策略）
      NCCLCHECK(ncclRegFind(comm, info->sendbuff, sendbuffSize, &regSendBuf));
      NCCLCHECK(ncclRegFind(comm, info->recvbuff, recvbuffSize, &regRecvBuf));
      NCCLCHECK(ncclRegLocalIsValid(regSendBuf, &isSendValid));
      NCCLCHECK(ncclRegLocalIsValid(regRecvBuf, &isRecvValid));
      regBuff = (regSendBuf && regRecvBuf && isSendValid && isRecvValid) || (ncclCudaGraphValid(comm->planner.capturingGraph) && ncclParamGraphRegister());
      // 如果缓冲区已注册且是 AllGather 或 ReduceScatter 操作
      if (regBuff && (info->func == ncclFuncAllGather || info->func == ncclFuncReduceScatter)) {
        // 如果支持 NVLS（单节点或多节点且支持 CollNet）
        if ((comm->nNodes > 1 && collNetSupport && nvlsSupport) || (comm->nNodes == 1 && nvlsSupport)) {
          // 查询 NVLS 注册资源推荐的通道数
          int recChannels;
          NCCLCHECK(ncclNvlsRegResourcesQuery(comm, info, &recChannels));
          // 如果推荐的通道数 <= 最大通道数，使用 NVLS 算法
          if (recChannels <= info->nMaxChannels) {
            info->algorithm = NCCL_ALGO_NVLS;
            info->protocol = NCCL_PROTO_SIMPLE;
            info->nMaxChannels = recChannels;
            info->nWarps = comm->maxThreads[info->algorithm][info->protocol] / WARP_SIZE;
          }
        }
      }
    }
  }

  // 如果插件设置了最大通道数，使用插件的值；否则使用拓扑计算的值
  info->nMaxChannels = nMaxChannels == 0 ? info->nMaxChannels : nMaxChannels;
  return ncclSuccess;
}

// 定义 NVLS Tree 最大块大小参数
// NVLS Tree 是 NVLink Switch Tree 算法，这个参数控制其最大块大小
NCCL_PARAM(NvlsTreeMaxChunkSize, "NVLSTREE_MAX_CHUNKSIZE", -2);

// 计算集合操作的分块大小和步数
// 这个函数是 NCCL 性能调优的核心，决定了如何将大数据分成小块进行处理
// 分块策略直接影响内存使用、网络带宽利用和延迟
// 参数 comm: 通信器指针，包含拓扑和通道信息
// 参数 info: 集合任务信息指针，包含操作类型、算法、协议等
// 参数 nChannels: 使用的通道数（并行度）
// 参数 nBytes: 数据总字节数
// 输出参数 outChunkSize: 计算出的块大小（对齐后）
// 输出参数 outDirectFlags: 直接传输标志（用于 CollNet，表示读/写模式）
// 输出参数 proxyOp: 代理操作指针，填充所有分块和步数信息
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
static ncclResult_t calcCollChunking(
    struct ncclComm* comm, struct ncclTaskColl* info, int nChannels, size_t nBytes,
    /*outputs*/uint32_t* outChunkSize, uint32_t* outDirectFlags, struct ncclProxyOp* proxyOp
  ) {
  // 通信模式（Ring、Tree、NVLS 等）
  // pattern 决定了数据在节点间的流动方式
  ncclPattern_t pattern;                                // 通信模式变量

  // 协议的粒度大小（对齐要求）
  // 不同协议有不同的对齐要求：SIMPLE=16字节，LL=16字节，LL128=128字节
  size_t grainSize = ncclProtoGrainSize(info->protocol); // 获取协议的对齐粒度

  // 根据集合操作类型和算法选择通信模式
  // 不同的操作和算法组合使用不同的通信模式
  switch (info->func) {                                 // 根据集合操作类型判断
  case ncclFuncBroadcast:                              // Broadcast：根 rank 广播数据到所有其他 rank
    // 如果是 Tree 算法，使用树形向下模式；否则使用流水线模式
    pattern = info->algorithm == NCCL_ALGO_TREE ? ncclPatternTreeDown : ncclPatternPipelineFrom; // 选择通信模式
    break;
  case ncclFuncReduce:                                 // Reduce：所有 rank 的数据规约到根 rank
    // 如果是 Tree 算法，使用树形向上模式；否则使用流水线模式
    pattern = info->algorithm == NCCL_ALGO_TREE ? ncclPatternTreeUp : ncclPatternPipelineTo; // 选择通信模式
    break;
  case ncclFuncReduceScatter:                          // ReduceScatter：先规约再分散
    // PAT 算法使用 PAT 向上模式，NVLS 使用 NVLS 模式，CollNet Direct 使用 CollNet Direct，其他使用环形模式
    pattern =                                           // 根据算法选择模式
      info->algorithm == NCCL_ALGO_PAT ? ncclPatternPatUp :      // PAT 算法
      info->algorithm == NCCL_ALGO_NVLS ? ncclPatternNvls :      // NVLS 算法
      info->algorithm == NCCL_ALGO_COLLNET_DIRECT ? ncclPatternCollnetDirect : // CollNet Direct 算法
      ncclPatternRing;                                                // 其他算法使用环形
    break;
  case ncclFuncAllGather:                              // AllGather：所有 rank 的数据收集到每个 rank
    // PAT 算法使用 PAT 向下模式，NVLS 使用 NVLS 模式，CollNet Direct 使用 CollNet Direct，其他使用环形模式
    pattern =                                           // 根据算法选择模式
      info->algorithm == NCCL_ALGO_PAT ? ncclPatternPatDown :    // PAT 算法
      info->algorithm == NCCL_ALGO_NVLS ? ncclPatternNvls :      // NVLS 算法
      info->algorithm == NCCL_ALGO_COLLNET_DIRECT ? ncclPatternCollnetDirect : // CollNet Direct 算法
      ncclPatternRing;                                                // 其他算法使用环形
    break;
  case ncclFuncAllReduce:                              // AllReduce：所有 rank 的数据规约并广播到所有 rank
    // NVLS 使用 NVLS 模式，NVLS_TREE 使用 NVLS Tree，CollNet Direct/Chain 使用对应模式，Tree 使用 Tree 双向，其他使用环形两次
    pattern =                                           // 根据算法选择模式
      info->algorithm == NCCL_ALGO_NVLS ? ncclPatternNvls :           // NVLS 算法
      info->algorithm == NCCL_ALGO_NVLS_TREE ? ncclPatternNvlsTree :  // NVLS Tree 算法
      info->algorithm == NCCL_ALGO_COLLNET_DIRECT ? ncclPatternCollnetDirect : // CollNet Direct 算法
      info->algorithm == NCCL_ALGO_COLLNET_CHAIN ? ncclPatternCollnetChain :  // CollNet Chain 算法
      info->algorithm == NCCL_ALGO_TREE ? ncclPatternTreeUpDown :     // Tree 算法（双向）
      ncclPatternRingTwice;                                             // 其他算法使用环形两次
    break;
  default:                                             // 未知的集合操作
    WARN("Unknown pattern for collective %d algorithm %d", info->func, info->algorithm); // 输出警告
    return ncclInternalError;                         // 返回内部错误
  }

  // 初始化分块和步数相关变量
  int nstepsPerLoop, nchunksPerLoop;                   // 每个循环的步数和块数
  size_t loopOffset = 0;                               // 循环偏移量（用于 NVLS/CollNet）

  // 基本步长大小（每个通道的缓冲区大小除以步数）
  int stepSize   = comm->buffSizes[info->protocol]/NCCL_STEPS; // 计算基本步长（考虑协议的缓冲区大小）

  // 块步数：每个块包含的步数（仅在 SIMPLE 协议的 Ring 算法中使用）
  // 这允许将多个步合并为一个块，提高大消息的性能
  int chunkSteps = (info->protocol == NCCL_PROTO_SIMPLE && info->algorithm == NCCL_ALGO_RING) ? info->chunkSteps : 1; // 条件判断

  // 切片步数：每个切片包含的块步数（仅在 SIMPLE 协议的 Ring 算法中使用）
  // 提供额外的层次来进一步优化性能
  int sliceSteps = (info->protocol == NCCL_PROTO_SIMPLE && info->algorithm == NCCL_ALGO_RING) ? info->sliceSteps : 1; // 条件判断

  // 块大小 = 步长 × 块步数
  int chunkSize = stepSize*chunkSteps;                 // 计算初始块大小

  // LL 协议调整：LL 协议使用编码，实际有效负载减半
  if (info->protocol == NCCL_PROTO_LL)                 // 如果是 LL 协议
    chunkSize /= 2;                                    // 块大小减半（考虑编码开销）

  // LL128 协议调整：LL128 协议使用 128 字节缓存行对齐
  if (info->protocol == NCCL_PROTO_LL128) {            // 如果是 LL128 协议
    // 重新计算块大小以匹配 LL128 的对齐要求
    // NCCL_LL128_LINEELEMS 是每个缓存行的元素数，NCCL_LL128_DATAELEMS 是数据元素数
    chunkSize = (chunkSize / NCCL_LL128_LINEELEMS) * NCCL_LL128_DATAELEMS; // 按 128 字节缓存行对齐
  }

  // CollNet Direct 算法的特殊分块优化
  // CollNet Direct 使用专用的硬件网络，需要特殊的块大小调优
  if (info->algorithm == NCCL_ALGO_COLLNET_DIRECT) {    // CollNet Direct 算法
    // Optimize chunkSize / nSteps
    // 优化块大小和步数，确保充分利用 CollNet Direct 的并行度
    // 根据数据量和深度动态调整块大小

    // 第一级优化：如果块太大导致每个块的并行度不足 64 倍深度，减少块大小
    // nBytes / (nChannels * nHeads * chunkSize) 是块的数量
    // 块数量太少意味着没有充分利用硬件并行度
    while (nBytes / (nChannels * comm->channels[0].collnetDirect.nHeads * chunkSize) < comm->channels[0].collnetDirect.depth * 64 && chunkSize > 131072) chunkSize /= 2; // 减半块大小

    // 第二级优化：块数量至少要是深度的 8 倍
    while (nBytes / (nChannels * comm->channels[0].collnetDirect.nHeads * chunkSize) < comm->channels[0].collnetDirect.depth * 8 && chunkSize > 65536) chunkSize /= 2; // 减半块大小

    // 第三级优化：块数量至少要等于深度
    while (nBytes / (nChannels * comm->channels[0].collnetDirect.nHeads * chunkSize) < comm->channels[0].collnetDirect.depth * 8 && chunkSize > 32768) chunkSize /= 2; // 减半块大小
  } else if (info->algorithm == NCCL_ALGO_COLLNET_CHAIN) { // CollNet Chain 算法
    // CollNet Chain 使用链式连接，需要不同的优化策略
    stepSize = comm->buffSizes[NCCL_PROTO_SIMPLE] / NCCL_STEPS; // 使用 SIMPLE 协议的步长
    chunkSize = std::min(256 * 1024, stepSize * chunkSteps);    // 块大小上限为 256KB

    // 根据深度调整块大小，确保足够的并行度
    while (nBytes / (nChannels * chunkSize) < comm->channels[0].collnetChain.depth * 64 && chunkSize > 131072) chunkSize /= 2; // 64 倍深度
    while (nBytes / (nChannels * chunkSize) < comm->channels[0].collnetChain.depth * 8 && chunkSize > 65536) chunkSize /= 2; // 8 倍深度
    while (nBytes / (nChannels * chunkSize) < comm->channels[0].collnetChain.depth && chunkSize > 32768) chunkSize /= 2; // 等于深度
  } else if (info->algorithm == NCCL_ALGO_NVLS) {         // NVLS（NVLink Switch）算法
    // NVLS 使用 NVLink 交换网络，需要根据注册缓冲区和带宽调整块大小
    if ((info->regBufType & NCCL_NVLS_REG_BUFFER) && (info->func == ncclFuncAllGather || info->func == ncclFuncReduceScatter)) {
      // 如果使用 NVLS 注册缓冲区且是 AllGather 或 ReduceScatter
      chunkSize = comm->buffSizes[NCCL_PROTO_SIMPLE] / NCCL_STEPS; // 使用 SIMPLE 协议的步长
    } else {                                              // 不使用注册缓冲区或其他操作
      int maxChunkSize = comm->nvlsChunkSize;            // NVLS 默认最大块大小

      // 多节点且带宽受限时，降低最大块大小
      if (comm->nNodes > 1 && comm->bandwidths[ncclFuncAllReduce][NCCL_ALGO_NVLS][NCCL_PROTO_SIMPLE] < 150) maxChunkSize = 32768; // 低带宽时限制为 32KB

      if (chunkSize > maxChunkSize) chunkSize = maxChunkSize; // 限制块大小不超过最大值

      // Use uint64_t so that concurrentOps*chunkSize*X does not overflow.
      // However, nChannels * comm->channels[0].nvls.nHeads should easily fit in 32 bits.
      // 使用 uint64_t 防止并发操作数 × 块大小 × 系数溢出
      // 虽然通道数 × 头数应该能轻松放入 32 位，但为安全起见使用 64 位
      // coverity[overflow_before_widen]
      uint64_t concurrentOps = nChannels * comm->channels[0].nvls.nHeads; // 计算并发操作数（通道数 × 头数）

      // 根据数据量和并发操作数调整块大小
      // 数据量相对较小时，减小块大小以避免块内空闲
      if ((nBytes < (64 * (concurrentOps * chunkSize))) && (chunkSize > 65536)) chunkSize = 65536; // 降至 64KB
      if ((nBytes < (8 * (concurrentOps * chunkSize))) && (chunkSize > 32768)) chunkSize = 32768; // 降至 32KB
      if ((nBytes < (2 * (concurrentOps * chunkSize))) && (chunkSize > 16384)) chunkSize = 16384; // 降至 16KB
    }
  } else if (info->algorithm == NCCL_ALGO_NVLS_TREE) {   // NVLS Tree 算法
    // Use uint64_t so that concurrentOps*chunkSize*X does not overflow.
    // However, nChannels * comm->channels[0].nvls.nHeads should easily fit in 32 bits.
    // 使用 uint64_t 防止溢出
    // coverity[overflow_before_widen]
    uint64_t concurrentOps = nChannels * comm->channels[0].nvls.nHeads; // 计算并发操作数

    chunkSize = comm->nvlsChunkSize;                    // 获取 NVLS 默认块大小
    int maxChunkSize = (int)ncclParamNvlsTreeMaxChunkSize(); // 获取用户设置的最大块大小参数

    // 如果参数未设置（-2），根据节点数选择默认值
    if (maxChunkSize == -2) maxChunkSize = comm->nNodes >= 4 ? 65536 : chunkSize; // 4+ 节点用 64KB，其他用默认

    chunkSize = std::min(chunkSize, maxChunkSize);       // 使用块大小和最大值的较小者

    // 根据数据量和并发操作数调整块大小（NVLS Tree 的特殊调优）
    if ((nBytes < (32 * (concurrentOps * chunkSize))) && (chunkSize > 262144)) chunkSize = 262144; // 降至 256KB
    if ((nBytes < (16 * (concurrentOps * chunkSize))) && (chunkSize > 131072)) chunkSize = 131072; // 降至 128KB
    if ((nBytes < (4 * (concurrentOps * chunkSize))) && (chunkSize > 65536)) chunkSize = 65536; // 降至 64KB
    if ((nBytes < (1 * (concurrentOps * chunkSize))) && (chunkSize > 32768)) chunkSize = 32768; // 降至 32KB
  } else if (info->algorithm == NCCL_ALGO_TREE && info->protocol == NCCL_PROTO_LL128) { // Tree 算法 + LL128 协议
    // Tree 算法配合 LL128 协议的特殊优化
    int nNodes = comm->nNodes;                          // 节点数
    float ppn = comm->nRanks / (float)nNodes;           // 每节点平均 rank 数

    // 估算 LL128 协议的步数（考虑节点数和每节点 rank 数）
    float nstepsLL128 = 1+log2i(nNodes) + 0.1*ppn;      // LL128 步数估算公式

    // Yes, we are OK with the division on the left side of the < operand being integer.
    // 我们接受 < 操作符左侧的整数除法
    // coverity[integer_division]
    // 根据估算的步数调整块大小，确保足够的并行度
    while (nBytes / (nChannels*chunkSize) < nstepsLL128*64/ppn && chunkSize > 131072) chunkSize /= 2; // 降至一半

    // coverity[integer_division]
    while (nBytes / (nChannels*chunkSize) < nstepsLL128*16/ppn && chunkSize > 32768) chunkSize /= 2; // 降至一半
  } else if (info->func == ncclFuncAllGather && info->algorithm == NCCL_ALGO_PAT) { // AllGather + PAT 算法
    // PAT 算法的 AllGather 优化：确保块大小不会太大导致块数量不足
    while (chunkSize*nChannels*32 > nBytes && chunkSize > 65536) chunkSize /= 2; // 降至一半
  } else if (info->func == ncclFuncReduceScatter && info->algorithm == NCCL_ALGO_PAT) { // ReduceScatter + PAT 算法
    // PAT 算法的 ReduceScatter 优化
    while (chunkSize*nChannels*16 > nBytes && chunkSize > 65536) chunkSize /= 2; // 降至一半
  }

  // Compute directFlags of work struct.
  // 计算工作结构的直接传输标志
  // directFlags 控制数据传输方式（读或写），影响性能和内存使用
  if (info->algorithm == NCCL_ALGO_COLLNET_DIRECT) {    // CollNet Direct 算法
    // Set direct direction for broadcast-gather (read or write)
    // 为 broadcast-gather 设置直接方向（读或写）
    // 小数据使用读模式，大数据使用写模式（优化性能）
    *outDirectFlags = (nBytes/nChannels <= 1024 * 4) ? NCCL_P2P_READ : NCCL_P2P_WRITE; // 每通道 ≤4KB 用读，否则用写
  } else {                                              // 其他算法
    *outDirectFlags = 0;                               // 不使用直接传输标志
  }

  // Compute nSteps for proxies
  // 计算代理操作的步数
  // 首先对块大小进行对齐，确保符合协议要求
  chunkSize = chunkSize / grainSize * grainSize;       // 对齐块大小到粒度大小的倍数

  // 根据通信模式设置每个循环的步数和块数
  // 这些参数决定了数据如何在通道间分布和处理
  switch (pattern) {                                   // 根据通信模式判断
  case ncclPatternTreeUp:                              // 树形向上（Reduce）
  case ncclPatternTreeDown:                            // 树形向下（Broadcast）
  case ncclPatternTreeUpDown:                          // 树形双向（AllReduce）
  case ncclPatternPatUp:                               // PAT 向上（ReduceScatter）
  case ncclPatternPatDown:                             // PAT 向下（AllGather）
  case ncclPatternPipelineFrom:                        // 流水线从（Broadcast）
  case ncclPatternPipelineTo:                          // 流水线到（Reduce）
  case ncclPatternCollnetChain:                        // CollNet 链式
    // 这些模式都是线性流水线，每循环只有 1 步和 1 块
    nstepsPerLoop = nchunksPerLoop = 1;                // 单步单块模式
    break;
  case ncclPatternNvls:                                // NVLS 模式
    // NVLS 使用多个头并行，每个循环有 1 步，但块数等于头数
    nstepsPerLoop = 1; nchunksPerLoop = comm->channels[0].nvls.nHeads; // 块数 = NVLS 头数
    loopOffset = nChannels * chunkSize * comm->channels[0].nvls.headRank; // 计算 NVLS 循环偏移
    break;
  case ncclPatternCollnetDirect:                       // CollNet Direct 模式
    // CollNet Direct 使用多个头并行，每个循环有 1 步，但块数等于头数
    nstepsPerLoop = 1; nchunksPerLoop = comm->channels[0].collnetDirect.nHeads; // 块数 = CollNet 头数
    loopOffset = nChannels * chunkSize * comm->channels[0].collnetDirect.headRank; // 计算 CollNet 循环偏移
    break;
  case ncclPatternRing:                                // 环形模式（ReduceScatter、AllGather）
    // 环形模式：每个循环需要 (nRanks-1) 步，有 nRanks 个块
    nstepsPerLoop = comm->nRanks-1; nchunksPerLoop = comm->nRanks; // 环形特征：步数 = rank-1，块数 = rank
    break;
  case ncclPatternRingTwice:                           // 双环形模式（AllReduce）
    // 双环形模式：每个循环需要 2*(nRanks-1) 步，有 nRanks 个块
    nstepsPerLoop = 2*(comm->nRanks-1); nchunksPerLoop = comm->nRanks; // AllReduce 的环形实现需要两次遍历
    break;
  case ncclPatternNvlsTree:                            // NVLS Tree 模式
    // NVLS Tree 使用多个头并行，每个循环有 1 步，块数等于头数
    nstepsPerLoop = 1; nchunksPerLoop = comm->channels[0].nvls.nHeads; // 块数 = NVLS 头数
    break;
  default:                                             // 未知模式
    WARN("Unknown pattern %d", pattern);                // 输出警告
    return ncclInternalError;                         // 返回内部错误
  }

  // Compute nSteps for proxies
  // 计算代理操作的总步数和相关参数
  // 计算循环大小（每个循环处理的总数据量）
  size_t loopSize = size_t(nChannels)*nchunksPerLoop*chunkSize; // 循环大小 = 通道数 × 每循环块数 × 块大小

  // 计算需要多少个循环才能处理完所有数据
  int nLoops = (int)DIVUP(nBytes, loopSize);           // 向上整除：循环数 = ceil(总字节 / 循环大小)

  // 初始化代理操作结构体（清零所有字段）
  memset(proxyOp, 0, sizeof(*proxyOp));                // 清零代理操作结构体

  // 填充代理操作的基本参数
  proxyOp->nsteps = nstepsPerLoop * nLoops * chunkSteps; // 总步数 = 每循环步数 × 循环数 × 块步数
  proxyOp->sliceSteps = sliceSteps;                     // 切片步数（用于 SIMPLE 协议的 Ring 算法）
  proxyOp->chunkSteps = chunkSteps;                     // 块步数（用于 SIMPLE 协议的 Ring 算法）
  proxyOp->chunkSize = chunkSize;                       // 块大小（对齐后的值）
  proxyOp->sliceSize = chunkSize / chunkSteps * sliceSteps; // 切片大小 = (块大小 / 块步数) × 切片步数
  proxyOp->loopSize = loopSize;                         // 循环大小
  proxyOp->loopOffset = loopOffset;                     // 循环偏移（用于 NVLS/CollNet 的头偏移）
  proxyOp->protocol = info->protocol;                   // 协议类型（LL、LL128、SIMPLE）
  proxyOp->dtype = info->datatype;                      // 数据类型
  proxyOp->algorithm = info->algorithm;                 // 算法类型（Ring、Tree、NVLS 等）

  // 处理规约操作类型（特别是平均值操作）
  if (info->opDev.op == ncclDevPreMulSum || info->opDev.op == ncclDevSumPostDiv) { // 预乘求和或后除求和
    proxyOp->redOp = ncclSum;                            // 网络层看到的平均值为求和（预处理或后处理在设备端完成）
  } else {                                              // 其他规约操作
    proxyOp->redOp = info->opHost;                      // 使用主机端的规约操作类型
  }

  proxyOp->pattern = pattern;                           // 通信模式
  proxyOp->coll = info->func;                           // 集合操作类型
  proxyOp->collAPI = info->func;                        // 集合 API 类型（通常与 func 相同）
  proxyOp->root = info->root;                           // 根 rank（用于 Broadcast、Reduce 等操作）
  proxyOp->isOneRPN = comm->isOneRPN;                   // 是否每节点只有一个 rank（影响缓冲区管理）

  // This is used by P2P to reduce the receive buffer size. We don't use it in collectives
  // because some protocols need to transmit more than the total size, plus they sometimes
  // round up
  // 这用于 P2P 以减少接收缓冲区大小。我们在集合操作中不使用它，
  // 因为某些协议需要传输比总大小更多的数据，而且有时会向上取整
  proxyOp->nbytes = stepSize*sliceSteps;                // 设置字节大小（用于接收缓冲区）

  // 网络注册缓冲区的特殊处理
  // 如果使用网络注册缓冲区（内存由网络插件管理），需要特殊设置
  if (info->regBufType & NCCL_NET_REG_BUFFER) {         // 如果使用网络注册缓冲区
    proxyOp->reg = 1;                                   // 标记为注册模式

    // CollNet Direct、NVLS 和 CollNet Chain 的注册缓冲区处理
    if (info->algorithm == NCCL_ALGO_COLLNET_DIRECT || info->algorithm == NCCL_ALGO_NVLS || info->algorithm == NCCL_ALGO_COLLNET_CHAIN) {
      // 每节点单 rank（1RPN）模式：直接使用用户缓冲区
      if (proxyOp->isOneRPN) {                          // 如果每节点只有一个 rank
        proxyOp->nsteps = 1;                            // 步数设为 1（直接传输）
        proxyOp->loopOffset = 0;                         // 偏移设为 0
        proxyOp->sendbuff = (uint8_t*)info->sendbuff;   // 设置发送缓冲区
        proxyOp->sendMhandle = info->sendMhandle;       // 设置发送内存句柄
      } else {                                          // 多 rank 模式：需要根据操作类型调整
        if (info->func == ncclFuncAllGather || info->func == ncclFuncReduceScatter) { // AllGather 或 ReduceScatter
          proxyOp->nbytes = nBytes / nchunksPerLoop;     // 每个头的字节数（分散数据）
          proxyOp->loopSize = proxyOp->loopSize / nchunksPerLoop; // 调整循环大小
          proxyOp->loopOffset = 0;                       // 偏移设为 0
          if (info->func == ncclFuncAllGather) {         // AllGather 需要发送缓冲区
            proxyOp->sendbuff = (uint8_t*)info->sendbuff; // 设置发送缓冲区
            proxyOp->sendMhandle = info->sendMhandle;   // 设置发送内存句柄
          }
        } else {                                          // 其他操作（AllReduce、Broadcast 等）
          proxyOp->sendbuff = (uint8_t*)info->recvbuff;  // 使用接收缓冲区作为发送缓冲区（就地操作）
          proxyOp->sendMhandle = info->recvMhandle;      // 使用接收内存句柄
        }
      }
    } else if (info->algorithm == NCCL_ALGO_RING) {     // Ring 算法的注册缓冲区处理
      // Ring 算法的 AllGather 在 1RPN 模式下的特殊处理
      if (proxyOp->isOneRPN && info->func == ncclFuncAllGather) { // 每节点单 rank + AllGather
        // 使用网络最大传输大小作为块大小
        proxyOp->chunkSize = NCCL_MAX_NET_SIZE;         // 设置块大小为网络最大值
        proxyOp->sliceSize = NCCL_MAX_NET_SIZE;         // 设置切片大小
        proxyOp->chunkSteps = 1;                         // 块步数设为 1
        proxyOp->sliceSteps = 1;                         // 切片步数设为 1
        proxyOp->loopSize = size_t(nChannels) * nchunksPerLoop * proxyOp->chunkSize; // 重新计算循环大小
        proxyOp->nsteps = DIVUP(nBytes, proxyOp->loopSize) * nstepsPerLoop; // 重新计算步数
        proxyOp->loopOffset = 0;                         // 偏移设为 0
      }
    } else {                                            // 不支持的算法
      WARN("Net registration invalid algorithm %s", ncclAlgoToString(info->algorithm)); // 输出警告
      return ncclInternalError;                       // 返回内部错误
    }

    // 设置接收端参数（所有算法通用）
    proxyOp->recvMhandle = info->recvMhandle;           // 设置接收内存句柄
    proxyOp->recvbuff = (uint8_t*)info->recvbuff;       // 设置接收缓冲区
    proxyOp->nbytes = nBytes;                           // 设置总字节数
  } else {                                              // 不使用网络注册缓冲区
    proxyOp->reg = 0;                                   // 标记为非注册模式
  }

  // CollNet Direct 和 NVLS 的特殊参数设置
  if (pattern == ncclPatternCollnetDirect || pattern == ncclPatternNvls) { // CollNet Direct 或 NVLS
    // 设置 CollNet 特定的参数
    proxyOp->specifics.collnetDirect.nNodes = comm->nNodes; // 节点总数
    proxyOp->specifics.collnetDirect.node = comm->node;     // 当前节点 ID

    // AllGather 和 ReduceScatter 需要知道每个 rank 的大小
    if (info->func == ncclFuncAllGather || info->func == ncclFuncReduceScatter) { // AllGather 或 ReduceScatter
      proxyOp->specifics.collnetDirect.sizePerRank = info->count*ncclTypeSize(info->datatype); // 每个 rank 的数据量
    }
  }

  // PAT 算法的特殊字节大小调整
  if (pattern == ncclPatternPatUp || pattern == ncclPatternPatDown) { // PAT 向上或向下
    // PAT 算法将数据分散到所有通道，每个通道处理总数据的一部分
    proxyOp->nbytes = DIVUP(nBytes, nChannels);         // 每通道的字节数（向上取整）
  }

  // Set peer count hints used by network plugin
  // 设置网络插件使用的对等节点数量提示
  // 这些提示帮助网络插件优化资源分配和连接管理
  switch (proxyOp->pattern) {                           // 根据通信模式判断
  case ncclPatternRing:                                 // 环形模式
  case ncclPatternRingTwice:                            // 双环形模式
  case ncclPatternPipelineFrom:                         // 流水线从
  case ncclPatternPipelineTo:                           // 流水线到
  case ncclPatternPatUp:                                // PAT 向上
  case ncclPatternPatDown:                              // PAT 向下
    // 这些模式每个 rank 只与 1 个对等节点通信
    proxyOp->nPeers = 1;                                // 对等节点数为 1
    break;
  case ncclPatternTreeUp:                               // 树形向上
  case ncclPatternTreeDown:                             // 树形向下
  case ncclPatternTreeUpDown:                           // 树形双向
  case ncclPatternNvlsTree:                             // NVLS Tree
    // Tree 模式每个 rank 可能与多个对等节点通信（父节点和子节点）
    // NCCL_MAX_TREE_ARITY 是树的最大分支数，-1 是因为不包括自己，×2 是包括上行和下行
    proxyOp->nPeers = (NCCL_MAX_TREE_ARITY - 1) * 2;   // 最大对等节点数 = (最大分支数 - 1) × 2
    break;
  case ncclPatternCollnetChain:                         // CollNet Chain
  case ncclPatternCollnetDirect:                        // CollNet Direct
  case ncclPatternNvls:                                 // NVLS
  case ncclPatternProfiler:                             // 性能分析器
    // Peer count hints unused
    // 对等节点数量提示未使用（这些算法有自己的连接管理方式）
    break;
  case ncclPatternSend:                                 // 发送
  case ncclPatternRecv:                                 // 接收
  default:                                             // 未知模式
    WARN("Unknown pattern %d", pattern);                // 输出警告
    return ncclInternalError;                         // 返回内部错误
  }

  // 输出计算结果
  *outChunkSize = proxyOp->chunkSize;                   // 设置输出参数：计算出的块大小
  return ncclSuccess;                                   // 返回成功状态
}

// 主机到设备规约操作转换函数
// 此函数将主机端的规约操作符转换为设备端可以使用的格式
// 对于某些操作（如求平均值），需要转换为设备端的特殊操作（如预乘求和或后除求和）
// 参数 opFull: 输出参数，设备端规约操作的完整描述结构体
// 参数 op: 主机端规约操作符（如 ncclSum、ncclProd、ncclAvg 等）
// 参数 datatype: 数据类型（如 ncclInt32、ncclFloat32 等）
// 参数 comm: 通信器指针
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
static ncclResult_t hostToDevRedOp(
    ncclDevRedOpFull *opFull, ncclRedOp_t op, ncclDataType_t datatype, ncclComm *comm) {
  // 使用联合体来处理不同的数据类型
  // 这样可以安全地在不同的数据类型间进行转换
  union {
    int8_t   i8; uint8_t   u8;                        // 8 位整型
    int32_t i32; uint32_t u32;                      // 32 位整型
    int64_t i64; uint64_t u64;                      // 64 位整型
    __half f16; float f32; double f64;               // 浮点型
    #if defined(__CUDA_BF16_TYPES_EXIST__)         // 如果支持 BFloat16
      __nv_bfloat16 bf16;
    #endif
    #if defined(__CUDA_FP8_TYPES_EXIST__)          // 如果支持 FP8
      __nv_fp8_storage_t f8;
    #endif
    void *ptr;                                      // 指针类型
  };
  u64 = 0;                                          // 初始化联合体为 0
  opFull->scalarArgIsPtr = false;                    // 标记标量参数不是指针
  opFull->proxyOp = op;                              // 保存原始操作符

  int nbits = 8*ncclTypeSize(datatype);              // 计算数据类型的位数（8、16、32、64）
  if (nbits <= 0)                                   // 验证数据类型有效
    return ncclInvalidArgument;

  uint64_t allBits = uint64_t(-1)>>(64-nbits);      // 生成全 1 的掩码（根据数据位数）
  uint64_t signBit = allBits^(allBits>>1);           // 提取符号位（最高有效位）
  bool datatype_signed = false;                      // 标记数据类型是否有符号

  // 根据规约操作类型进行转换
  switch (int(op)) {
  case ncclSum:                                      // 求和操作
    opFull->op = ncclDevSum;                         // 设备端求和操作
    break;
  case ncclProd:                                     // 求积操作
    opFull->op = ncclDevProd;                        // 设备端求积操作
    break;
  case ncclMin:                                      // 求最小值操作
  case ncclMax:                                      // 求最大值操作
    opFull->op = ncclDevMinMax;                      // 设备端最小/最大值操作
    opFull->scalarArg = 0;                           // 初始化标量参数
    // The xormask used by ncclFuncMinMax<[u]int> is the XOR of the sign bit
    // for signed (opposed to unsigned) types and all the bits for max (opposed to min).
    // ncclFuncMinMax<[u]int> 使用的 XOR 掩码是符号位的 XOR，
    // 对于有符号（相对于无符号）类型，max（相对于 min）使用所有位。
    if (datatype==ncclInt8 || datatype==ncclInt32 || datatype==ncclInt64) { // 有符号整型
      opFull->scalarArg ^= signBit;                 // XOR 符号位以处理有符号数
    }
    opFull->scalarArg ^= (op == ncclMax) ? allBits : 0; // 如果是 max，XOR 所有位
    break;
  case ncclAvg:                                      // 求平均值操作
    // 整型平均值的处理：使用后除求和（sum + divide）
    switch ((int)datatype) {
    case ncclInt8:                                    // 8 位有符号整型
    case ncclInt32:                                  // 32 位有符号整型
    case ncclInt64:                                  // 64 位有符号整型
      datatype_signed = true;                        // 标记为有符号类型
      // no break, we want to fall through...
      // 不中断，继续执行下面的代码（fall through 到无符号分支）
    case ncclUint8:                                   // 8 位无符号整型
    case ncclUint32:                                  // 32 位无符号整型
    case ncclUint64:                                  // 64 位无符号整型
      opFull->op = ncclDevSumPostDiv;                 // 设备端后除求和操作
      u64 = comm->nRanks<<1 | datatype_signed;       // 编码：rank 数和符号标志
      break;
    #if defined(__CUDA_FP8_TYPES_EXIST__)            // FP8 数据类型支持
    case ncclFloat8e4m3:                             // FP8 E4M3 格式
      opFull->op = ncclDevPreMulSum;                 // 设备端预乘求和操作
      f8 = __nv_cvt_float_to_fp8(float(1.0/comm->nRanks), __NV_SATFINITE, __NV_E4M3); // 转换 1/nRanks 到 FP8
      break;
    case ncclFloat8e5m2:                             // FP8 E5M2 格式
      opFull->op = ncclDevPreMulSum;                 // 设备端预乘求和操作
      f8 = __nv_cvt_float_to_fp8(float(1.0/comm->nRanks), __NV_SATFINITE, __NV_E5M2); // 转换 1/nRanks 到 FP8
      break;
    #endif
    case ncclFloat16:                                 // 半精度浮点（FP16）
      opFull->op = ncclDevPreMulSum;                 // 设备端预乘求和操作
      f16 = __float2half(float(1.0/comm->nRanks));   // 转换 1/nRanks 到 half 精度
      break;
    #if defined(__CUDA_BF16_TYPES_EXIST__)         // BFloat16 数据类型
    case ncclBfloat16:
      opFull->op = ncclDevPreMulSum;
      bf16 = __float2bfloat16(float(1.0/comm->nRanks));
      break;
    #endif
    case ncclFloat32:
      opFull->op = ncclDevPreMulSum;
      f32 = float(1.0/comm->nRanks);
      break;
    case ncclFloat64:
      opFull->op = ncclDevPreMulSum;
      f64 = 1.0/comm->nRanks;
      break;
    }
    opFull->scalarArgIsPtr = false;
    opFull->scalarArg = u64;
    break;
  default: // user created
    int ix = int(ncclUserRedOpMangle(comm, op)) - int(ncclNumOps);
    ncclUserRedOp *user = &comm->userRedOps[ix];
    if (datatype != user->datatype) {
      WARN("Data type supplied to user-created ncclRedOp_t does not match type "
           "given to reduction operation");
      return ncclInvalidArgument;
    }
    *opFull = user->opFull;
    break;
  }
  return ncclSuccess;
}

// 设置 CUDA 图捕获状态并跟踪使用的流列表
// 此函数管理 NCCL 在 CUDA 图捕获模式下的状态
// 它跟踪在 NCCL 组操作中使用的所有 CUDA 流，并确保它们具有一致的捕获状态
// 要么所有流都未被捕获，要么所有流都被同一个图捕获
// 参数 comm: NCCL 通信器指针，包含内核规划器和状态信息
// 参数 info: 操作信息结构体指针，包含当前操作的 CUDA 流
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
static ncclResult_t ncclPlannerSetCapturingGraph(struct ncclComm* comm, struct ncclInfo* info) {
  struct ncclKernelPlanner *planner = &comm->planner;  // 获取内核规划器指针，包含流跟踪信息

  // 检查是否需要更新捕获状态
  // 条件1：当前流与最近使用的流不同（说明切换了流）
  // 条件2：流列表为空（第一次调用）
  if (info->stream != planner->streamRecent || planner->streams == nullptr) { // 如果流发生了变化或流列表未初始化
    planner->streamRecent = info->stream;                // 更新最近使用的流为当前流

    // 遍历流列表，查找或添加当前流
    struct ncclCudaStreamList* l = planner->streams;      // 获取流列表头指针（链表结构）
    while (true) {                                       // 循环遍历流列表
      if (l == nullptr) {                               // 如果到达链表末尾
        // Got to the end, this must be a new stream.
        // 到达列表末尾，说明这是一个新流，需要添加到列表中

        struct ncclCudaGraph graph;                     // CUDA 图结构体
        // 查询当前流的捕获状态（是否正在捕获 CUDA 图）
        NCCLCHECK(ncclCudaGetCapturingGraph(&graph, info->stream)); // 获取流的捕获图信息

        // 验证捕获状态的一致性
        // 如果流列表不为空，且新流的捕获状态与现有图不同，则报错
        if (planner->streams != nullptr && !ncclCudaGraphSame(planner->capturingGraph, graph)) { // 如果图不一致
          // 输出错误信息：NCCL 组中的流必须要么全部未捕获，要么全部被同一个图捕获
          WARN("Streams given to a communicator within a NCCL group must either be all uncaptured or all captured by the same graph.");
          return ncclInvalidUsage;                       // 返回无效使用错误
        }

        // 保存捕获图状态
        planner->capturingGraph = graph;                 // 更新规划器的捕获图状态（C++ 结构体赋值）

        // Add stream to list
        // 将新流添加到流列表的头部
        l = ncclMemoryStackAlloc<struct ncclCudaStreamList>(&comm->memScoped); // 从作用域内存栈分配流节点
        l->stream = info->stream;                         // 设置流的 CUDA 流指针
        l->next = planner->streams;                       // 将新节点的 next 指向当前列表头
        planner->streams = l;                             // 更新列表头为新节点（插入到链表头部）

        break;                                            // 新流已添加，退出循环
      }
      // 检查是否找到了当前流（流已存在于列表中）
      if (l->stream == info->stream)                      // 如果找到了匹配的流
        break;                                            // Already seen stream. 流已经在列表中，无需重复添加，退出循环

      l = l->next;                                       // 移动到链表的下一个节点，继续查找
    }
  }

  return ncclSuccess;                                   // 返回成功状态
}

// P2P 任务追加函数
// 此函数将点对点（Send/Recv）操作添加到通信器的任务队列中
// P2P 操作用于 AlltoAll、Gather、Scatter 等集合通信的底层实现
// 参数 comm: NCCL 通信器指针，包含通信状态和资源
// 参数 info: 操作信息结构体指针，包含流、缓冲区等基本信息
// 参数 coll: 集合操作类型（ncclFuncSend 或 ncclFuncRecv）
// 参数 collAPI: 原始集合 API 类型（如 ncclFuncAlltoAll），用于性能分析
// 参数 buff: 缓冲区指针（发送或接收缓冲区）
// 参数 count: 数据元素个数
// 参数 datatype: 数据类型（如 ncclInt32、ncclFloat32 等）
// 参数 peer: 对等节点的 rank ID（目标 rank）
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
static ncclResult_t p2pTaskAppend(
    struct ncclComm* comm,
    struct ncclInfo* info,
    ncclFunc_t coll,
    ncclFunc_t collAPI,
    void* buff,
    size_t count,
    ncclDataType_t datatype,
    int peer) {
    
  struct ncclKernelPlanner *planner = &comm->planner;  // 获取内核规划器指针，用于管理任务队列

  // Determine peer and basic parameters.
  // 确定对等节点和基本参数
  ssize_t nBytes = count*ncclTypeSize(datatype);        // 计算传输数据的总字节数（元素个数 × 每个元素的大小）
  bool isSendNotRecv = coll == ncclFuncSend;           // 判断是发送操作还是接收操作（true=发送，false=接收）

  // Must be in thread local group before tasks can be alloc'd in `comm->memScoped`.
  // 在可以从 `comm->memScoped` 分配任务之前，必须在线程本地组中
  // 这确保内存分配在正确的作用域内，便于后续批量处理和清理
  ncclGroupCommJoin(comm, ncclGroupTaskTypeCollective); // 将通信器加入当前线程的本地组，标记为集合操作类型
  info->coll = coll;                                   // 设置 info 结构体中的操作类型

  // Set capturing graph. Called here so that profiler can emit a group API event with this information
  // 设置捕获图。在这里调用，以便性能分析器可以发出带有此信息的组 API 事件
  // CUDA 图捕获允许将一系列 CUDA 操作记录为图，以便后续重复执行，减少启动开销
  NCCLCHECK(ncclPlannerSetCapturingGraph(comm, info));  // 设置并验证当前流是否在捕获 CUDA 图
  bool isGraphCaptured = ncclCudaGraphValid(planner->capturingGraph); // 检查 CUDA 图是否有效（正在捕获）

  // 性能分析：启动组 API 事件，用于记录操作的开始
  NCCLCHECK(ncclProfilerStartGroupApiEvent(info, isGraphCaptured)); // 启动组 API 性能分析事件
  NCCLCHECK(ncclProfilerRecordGroupApiEventState(ncclProfilerGroupStartApiStop)); // 记录组 API 事件状态为停止

  // 性能分析：启动 P2P API 事件，用于记录点对点操作的开始
  NCCLCHECK(ncclProfilerStartP2pApiEvent(info, isGraphCaptured)); // 启动 P2P API 性能分析事件

  // 从内存池中分配一个 P2P 任务结构体
  // 使用永久内存生命周期，确保任务在执行期间一直有效
  struct ncclTaskP2p* p2p = ncclMemoryPoolAlloc<struct ncclTaskP2p>(&comm->memPool_ncclTaskP2p, &comm->memPermanent);

  // 初始化 P2P 任务结构体的各个字段
  p2p->func = coll;                                   // 设置操作类型（Send 或 Recv）
  p2p->collAPI = collAPI;                            // 设置原始集合 API 类型（用于性能分析和调试）
  p2p->buff = buff;                                  // 设置缓冲区指针
  p2p->count = count;                                // 设置数据元素个数
  p2p->datatype = datatype;                          // 设置数据类型
  p2p->root = peer;                                  // 设置对等节点的 rank ID（root 字段在这里表示目标 peer）
  p2p->bytes = nBytes;                               // 设置传输字节数
  p2p->eActivationMask = ncclProfilerApiState.eActivationMask; // 记录性能分析器激活掩码（哪些事件被启用）
  p2p->groupApiEventHandle = ncclProfilerApiState.groupApiEventHandle; // 记录组 API 事件句柄
  p2p->p2pApiEventHandle = ncclProfilerApiState.p2pApiEventHandle;     // 记录 P2P API 事件句柄

  // 将 P2P 任务加入对应 peer 的发送或接收队列的尾部
  // 使用无锁 intrusive 队列实现高效的任务入队
  ncclIntruQueueEnqueue(
    isSendNotRecv ? &planner->peers[peer].sendQueue : &planner->peers[peer].recvQueue, // 根据操作类型选择队列
    p2p);                                            // 要入队的任务

  // 更新规划器中的任务计数器
  planner->nTasksP2p += 1;                           // 增加 P2P 任务总数（发送+接收）
  if (isSendNotRecv)                                  // 如果是发送操作
    planner->nTasksP2pSend += 1;                     // 增加发送任务计数
  else                                               // 如果是接收操作
    planner->nTasksP2pRecv += 1;                     // 增加接收任务计数

  // Mark channels that need pre-connect
  // 标记需要预连接的通道
  // 预连接是在实际数据传输之前建立连接，以减少延迟
  if (comm->rank != peer) {                          // 只有与不同的 peer 通信才需要建立连接
    // 检查此 peer 的发送或接收操作是否已经被见过
    // sendSeen/recvSeen 用于避免重复设置连接
    if (!(isSendNotRecv ? planner->peers[peer].sendSeen : planner->peers[peer].recvSeen)) {
      // planner->peers[peer].send/recvSeen is private to each comm, so we need to set it anyway.
      // planner->peers[peer].send/recvSeen 是每个 comm 私有的，所以我们需要设置它
      (isSendNotRecv ? planner->peers[peer].sendSeen : planner->peers[peer].recvSeen) = true; // 标记已见过

      // 查找此 peer 在 P2P 调度表中的轮次
      // P2P 调度表定义了不同 round 中哪些 rank 之间进行通信
      int round = 0;                                  // 初始化轮次为 0
      // 遍历调度表，找到包含此 peer 的 round
      while (peer != (isSendNotRecv ? comm->p2pSchedule[round].sendRank
                                    : comm->p2pSchedule[round].recvRank)) {
        round += 1;                                   // 移动到下一轮
      }

      // 计算此 round 使用的通道基础索引
      // P2P 通道按 round 分组，以避免不同 round 之间的冲突
      uint8_t base = ncclP2pChannelBaseForRound(comm, round); // 获取 round 的基础通道 ID

      // 遍历分配给此 peer 的所有通道
      for (int c=0; c < comm->p2pnChannelsPerPeer; c++) { // 每个 peer 可以使用多个通道并行传输
        // 计算具体的通道 ID
        // 通道 ID 计算考虑了总的 P2P 通道数、基础索引和部分索引
        int channelId = ncclP2pChannelForPart(comm->p2pnChannels, base, c); // 获取实际的通道 ID

        if (isSendNotRecv) {                          // 处理发送操作
          // 检查此通道的发送连接是否已经设置过
          // P2P 只使用 1 个连接器（索引 1），索引 0 用于集合通信
          if (comm->channels[channelId].peers[peer]->send[1].hasSeen == 0) { // P2P uses only 1 connector
            // the send/recv connector is shared among split shared comms. We need to set hasSeen to
            // 1 in order to avoid duplicate connection setup if user group sendrecv ops with split
            // shared comms together.
            // 发送/接收连接器在分割的共享通信器之间共享。
            // 如果用户将分割共享通信器的 sendrecv 操作组合在一起，
            // 我们需要将 hasSeen 设置为 1，以避免重复的连接设置。
            comm->channels[channelId].peers[peer]->send[1].hasSeen = 1; // 标记已见过此发送连接
            comm->channels[channelId].peers[peer]->send[1].p2pOnly = 1; // 标记此连接仅用于 P2P（不用于集合通信）
            //记录peer这个rank，使用那个通道channelId
            comm->connectSend[peer] |= (1UL<<channelId); // 在连接掩码中设置此通道位，表示需要建立发送连接
            ncclGroupCommPreconnect(comm);              // 触发组通信预连接（实际建立连接）
          }
        } else {                                      // 处理接收操作
          // 检查此通道的接收连接是否已经设置过
          if (comm->channels[channelId].peers[peer]->recv[1].hasSeen == 0) { // P2P uses only 1 connector
            comm->channels[channelId].peers[peer]->recv[1].hasSeen = 1; // 标记已见过此接收连接
            comm->channels[channelId].peers[peer]->recv[1].p2pOnly = 1; // 标记此连接仅用于 P2P（不用于集合通信）
            comm->connectRecv[peer] |= (1UL<<channelId); // 在连接掩码中设置此通道位，表示需要建立接收连接
            ncclGroupCommPreconnect(comm);              // 触发组通信预连接（实际建立连接）
          }
        }
      }
    }
  }

  // 性能分析：停止 P2P API 事件，记录操作完成
  ncclProfilerStopP2pApiEvent();                     // 停止 P2P API 性能分析事件

  return ncclSuccess;                                // 返回成功状态
}

// 集合通信任务追加函数
// 此函数将集合通信操作（如 AllReduce、AllGather、Broadcast 等）添加到通信器的任务队列中
// 集合通信任务会按照流量大小进行排序，大消息优先调度，以优化整体性能
// 参数 comm: NCCL 通信器指针，包含通信状态和资源
// 参数 info: 操作信息结构体指针，包含操作类型、缓冲区、数据类型等信息
// 参数 opDev: 设备端规约操作的完整描述结构体，包含规约操作类型和参数
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
static ncclResult_t collTaskAppend(
    struct ncclComm* comm,
    struct ncclInfo* info,
    struct ncclDevRedOpFull opDev) {
  struct ncclKernelPlanner *planner = &comm->planner;  // 获取内核规划器指针，用于管理任务队列和调度

  // Must be in thread local group before tasks can be alloc'd in `comm->memScoped`
  // Add comm to this thread's group，等待后续触发ncclGroupEndInternal调用
  // 在可以从 `comm->memScoped` 分配任务之前，必须在线程本地组中
  // 将通信器加入当前线程的本地组，等待后续触发 ncclGroupEndInternal 调用
  // 这确保内存分配在正确的作用域内，便于后续批量处理和清理
  ncclGroupCommJoin(info->comm, ncclGroupTaskTypeCollective); // 将通信器加入当前线程的本地组，标记为集合操作类型

  // Set capturing graph. Called here so that profiler can emit a group API event with this information
  // 设置捕获图。在这里调用，以便性能分析器可以发出带有此信息的组 API 事件
  // CUDA 图捕获允许将一系列 CUDA 操作记录为图，以便后续重复执行，减少启动开销
  NCCLCHECK(ncclPlannerSetCapturingGraph(comm, info));  // 设置并验证当前流是否在捕获 CUDA 图
  bool isGraphCaptured = ncclCudaGraphValid(planner->capturingGraph); // 检查 CUDA 图是否有效（正在捕获）

  // 性能分析器事件：记录集合通信操作的开始
  NCCLCHECK(ncclProfilerStartGroupApiEvent(info, isGraphCaptured)); // 启动组 API 性能分析事件
  NCCLCHECK(ncclProfilerRecordGroupApiEventState(ncclProfilerGroupStartApiStop)); // 记录组 API 事件状态为停止
  NCCLCHECK(ncclProfilerStartCollApiEvent(info, isGraphCaptured)); // 启动集合通信 API 性能分析事件

  //分配一个ncclTaskColl
  // 从内存池中分配一个集合通信任务结构体
  // 使用永久内存生命周期，确保任务在执行期间一直有效
  struct ncclTaskColl* t = ncclMemoryPoolAlloc<struct ncclTaskColl>(&comm->memPool_ncclTaskColl, &comm->memPermanent);

  // 初始化集合任务的基本信息
  t->func = info->coll;                                // 设置集合操作类型（AllReduce、AllGather、Broadcast 等）
  //保存缓冲区地址
  // 保存发送和接收缓冲区的地址
  t->sendbuff = info->sendbuff;                        // 设置发送缓冲区指针（源数据）
  t->recvbuff = info->recvbuff;                        // 设置接收缓冲区指针（目标数据）
  //多少个元素
  // 设置数据元素个数
  t->count = info->count;                              // 设置数据元素个数
  t->root = info->root;                                // 设置根节点 rank（用于 Broadcast、Reduce 等操作）
  t->datatype = info->datatype;                        // 设置数据类型（如 ncclInt32、ncclFloat32 等）

  //每个数据类型的大小
  // 获取每个数据类型的大小（字节数）
  size_t elementSize = ncclTypeSize(t->datatype);       // 查询数据类型的大小（如 int32 为 4 字节）

  // 特殊处理 AllGather 和 Broadcast 操作
  // 这两个操作需要处理任意字节长度的数据，因此转换为以字节为单位
  if (t->func == ncclFuncAllGather || t->func == ncclFuncBroadcast) { // 如果是 AllGather 或 Broadcast 操作
    //转换为数据总大小
    // 将元素个数转换为总字节数，将数据类型改为 Int8（字节类型）
    // 这样可以处理非标准数据类型大小的数据传输
    t->count *= elementSize;                           // 将元素个数转换为总字节数
    t->datatype = ncclInt8;                            // 将数据类型改为 Int8（字节类型）
    elementSize = 1;                                   // 元素大小设为 1 字节
  }

  //计算发送的总字节数
  // 计算网络流量字节数，考虑不同集合操作的通信模式
  // 例如 AllReduce 需要每个字节传输 2 次（Reduce-Scatter + AllGather）
  t->trafficBytes = t->count*elementSize*ncclFuncTrafficPerByte(t->func, comm->nRanks); // 计算流量字节数 = 数据量 × 每字节传输次数

  // 设置规约操作相关信息
  t->opHost = info->op;                                // 保存主机端规约操作（如 ncclSum、ncclProd 等）
  t->opDev = opDev;                                    // 保存设备端规约操作的完整描述（C++ 结构体赋值）

  // 设置分块和切片步数
  // 这些参数用于将大数据分成多个小块并行处理
  t->chunkSteps = info->chunkSteps;                   // 设置块步数（每个块包含的切片数）
  t->sliceSteps = info->sliceSteps;                   // 设置切片步数（每个切片包含的操作数）

  // 设置性能分析相关字段
  t->eActivationMask = ncclProfilerApiState.eActivationMask; // 记录性能分析器激活掩码（哪些事件被启用）
  t->groupApiEventHandle = ncclProfilerApiState.groupApiEventHandle; // 记录组 API 事件句柄
  t->collApiEventHandle = ncclProfilerApiState.collApiEventHandle;   // 记录集合通信 API 事件句柄

  // 更新规划器中的集合任务计数器
  planner->nTasksColl += 1;                            // 增加集合任务总数

  //按消息大小插入队列，等待调度。同组内先调度大消息
  // 将任务插入排序器中，按照流量字节大小进行排序
  // 使用大消息优先的策略，可以优化整体带宽利用率
  ncclTaskCollSorterInsert(&planner->collSorter, t, t->trafficBytes); // 将任务插入排序器，按流量大小排序

  // 性能分析器事件：记录集合通信操作的完成
  ncclProfilerStopCollApiEvent();                     // 停止集合通信 API 性能分析事件

  return ncclSuccess;                                 // 返回成功状态
}

// CE（Collect Engine）集合通信任务追加函数
// 此函数将使用 CE 引擎的集合通信操作添加到通信器的任务队列中
// CE 引擎是一种优化的执行模型，使用对称窗口和设备端运行时来提高性能
// 它需要特殊的初始化和窗口管理，与常规的内核执行模式不同
// 参数 comm: NCCL 通信器指针，包含通信状态和资源
// 参数 info: 操作信息结构体指针，包含操作类型、缓冲区、数据类型等信息
// 参数 sendWin: 发送对称窗口指针，用于 CE 引擎的发送端缓冲区管理
// 参数 recvWin: 接收对称窗口指针，用于 CE 引擎的接收端缓冲区管理
// 参数 opDev: 设备端规约操作的完整描述结构体，包含规约操作类型和参数
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
static ncclResult_t ceCollTaskAppend(
    struct ncclComm* comm,
    struct ncclInfo* info,
    struct ncclDevrWindow* sendWin,
    struct ncclDevrWindow* recvWin,
    struct ncclDevRedOpFull opDev) {
  struct ncclKernelPlanner *planner = &comm->planner;  // 获取内核规划器指针，用于管理任务队列和调度

  // Check if CE needs initialization
  // 检查 CE（Collect Engine）是否需要初始化
  // CE 引擎是 NCCL 的设备端运行时，需要在第一次使用前进行特殊初始化
  if (comm->ceColl.baseUCSymReadyPtr == NULL && ncclIntruQueueEmpty(&comm->ceInitTaskQueue)) { // 如果 CE 未初始化且初始化队列为空
    struct ncclCeInitTask* ceTask;                       // CE 初始化任务指针
    NCCLCHECK(ncclCalloc(&ceTask, 1));                   // 分配并清零 CE 初始化任务结构体
    ceTask->comm = comm;                                  // 设置初始化任务的通信器指针
    ncclIntruQueueEnqueue(&comm->ceInitTaskQueue, ceTask); // 将初始化任务加入 CE 初始化队列
    // 将通信器加入线程本地组，标记为对称注册类型
    // 这确保 CE 初始化在正确的时机执行，与其他操作同步
    ncclGroupCommJoin(comm, ncclGroupTaskTypeSymRegister); // 加入对称注册任务类型
  }

  // Must be in thread local group before tasks can be alloc'd in `comm->memScoped`.
  // 在可以从 `comm->memScoped` 分配任务之前，必须在线程本地组中
  // 这确保内存分配在正确的作用域内，便于后续批量处理和清理
  ncclGroupCommJoin(info->comm, ncclGroupTaskTypeCollective); // 将通信器加入当前线程的本地组，标记为集合操作类型
  NCCLCHECK(ncclPlannerSetCapturingGraph(comm, info));      // 设置并验证当前流是否在捕获 CUDA 图

  // 分配集合任务结构体
  // 从内存池中分配一个集合通信任务结构体
  // 使用永久内存生命周期，确保任务在执行期间一直有效
  struct ncclTaskColl* t = ncclMemoryPoolAlloc<struct ncclTaskColl>(&comm->memPool_ncclTaskColl, &comm->memPermanent);

  // 初始化集合任务的基本字段
  t->func = info->coll;                                // 设置集合操作类型（AllReduce、AllGather、Broadcast 等）
  t->sendbuff = info->sendbuff;                        // 设置发送缓冲区指针（源数据）
  t->recvbuff = info->recvbuff;                        // 设置接收缓冲区指针（目标数据）
  t->count = info->count;                              // 设置数据元素个数
  t->root = info->root;                                // 设置根节点 rank（用于 Broadcast、Reduce 等操作）
  t->datatype = info->datatype;                        // 设置数据类型（如 ncclInt32、ncclFloat32 等）

  // 处理数据类型大小和特殊操作类型
  size_t elementSize = ncclTypeSize(t->datatype);       // 获取每个数据类型的大小（字节数）
  if (t->func == ncclFuncAllGather || t->func == ncclFuncBroadcast) { // 如果是 AllGather 或 Broadcast 操作
    // 将元素个数转换为总字节数，将数据类型改为 Int8（字节类型）
    // 这样可以处理非标准数据类型大小的数据传输
    t->count *= elementSize;                           // 将元素个数转换为总字节数
    t->datatype = ncclInt8;                            // 将数据类型改为 Int8（字节类型）
    elementSize = 1;                                   // 元素大小设为 1 字节
  }

  // 计算网络流量字节数，考虑不同集合操作的通信模式
  // 例如 AllReduce 需要每个字节传输 2 次（Reduce-Scatter + AllGather）
  t->trafficBytes = t->count*elementSize*ncclFuncTrafficPerByte(t->func, comm->nRanks); // 计算流量字节数 = 数据量 × 每字节传输次数

  // 设置规约操作相关信息
  t->opHost = info->op;                                // 保存主机端规约操作（如 ncclSum、ncclProd 等）
  t->opDev = opDev;                                    // 保存设备端规约操作的完整描述（C++ 结构体赋值）

  // 设置分块和切片步数
  // 这些参数用于将大数据分成多个小块并行处理
  t->chunkSteps = info->chunkSteps;                   // 设置块步数（每个块包含的切片数）
  t->sliceSteps = info->sliceSteps;                   // 设置切片步数（每个切片包含的操作数）

  // 设置性能分析相关字段
  // 使用原子加载获取性能分析器事件掩码，确保线程安全
  t->eActivationMask = __atomic_load_n(&ncclProfilerEventMask, __ATOMIC_RELAXED); // 原子加载性能分析器事件掩码（Relaxed 内存序）

  // 设置 CE 引擎特有的对称窗口字段
  // 对称窗口是 CE 引擎用于管理设备端缓冲区的机制
  t->sendWin = sendWin;                                // 设置发送对称窗口指针（CE 引擎使用）
  t->recvWin = recvWin;                                // 设置接收对称窗口指针（CE 引擎使用）

  // 将 CE 集合任务加入专门的 CE 任务队列
  // CE 任务使用单独的队列，因为它们的处理方式与常规集合任务不同
  ncclIntruQueueEnqueue(&planner->collCeTaskQueue, t);  // 将任务加入 CE 集合任务队列

  return ncclSuccess;                                 // 返回成功状态
}

// Converts `info` to a task and adds it to `comm->planner`. The exception is with
// single rank communicators, collectives are issued as `ncclMemcpyAsync`s and
// thus don't need a task.
// 将 `info` 转换为任务并添加到 `comm->planner` 中。
// 特殊情况：对于单 rank 通信器，集合操作直接作为 `ncclMemcpyAsync` 发出，
// 因此不需要创建任务。
// 此函数是任务追加的统一入口，根据操作类型和配置选择最优的执行路径
// 参数 comm: NCCL 通信器指针，包含通信状态和资源
// 参数 info: 操作信息结构体指针，包含操作类型、缓冲区、数据类型等所有信息
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
static ncclResult_t taskAppend(struct ncclComm* comm, struct ncclInfo* info) {
  ncclFunc_t collAPI = info->coll;                       // 保存原始集合 API 类型（用于性能分析和调试）

  // 第一类操作：点对点操作（Send/Recv）
  // 这两个操作虽然是集合通信 API，但底层实现为 P2P 模式
  if (info->coll == ncclFuncSend || info->coll == ncclFuncRecv) { // 如果是 Send 或 Recv 操作
    // 直接调用 P2P 任务追加函数
    // 注意：Send 操作的缓冲区在 recvbuff 字段（这是 info 结构体的设计约定）
    NCCLCHECK(p2pTaskAppend(comm, info, info->coll, collAPI, (void*)info->recvbuff, info->count, info->datatype, info->root));
  } else {                                                // 第二类操作：真正的集合通信操作
    // Empty collectives can be discarded.
    // 空集合操作可以直接丢弃，无需处理
    // count 为 0 表示没有数据需要传输，直接返回成功
    if (info->count == 0)                                // 如果数据元素个数为 0
        return ncclSuccess;                             // 直接返回成功，无需创建任务

    // FP8 数据类型检查（8 位浮点数）
    // FP8 是一种新的低精度浮点格式，需要特定的硬件支持
    if (info->datatype == ncclFloat8e4m3 || info->datatype == ncclFloat8e5m2) { // 如果是 FP8 E4M3 或 E5M2 格式
      // FP8 归约操作需要 sm90（Hopper）及更高版本的 GPU
      // 但某些操作（AllGather、Broadcast、AlltoAll、Scatter、Gather）不需要归约，可以在旧硬件上运行
      if (comm->minCompCap < 90 && info->coll != ncclFuncAllGather && info->coll != ncclFuncBroadcast && info->coll != ncclFuncAlltoAll && info->coll != ncclFuncScatter && info->coll != ncclFuncGather) {
        WARN("FP8 reduction support begins with sm90 capable devices."); // 输出警告：FP8 归约需要 sm90 设备
        return ncclInvalidArgument;                       // 返回无效参数错误
      }
    }

    // Copy reduction op state from op handle into info struct here since the
    // op handle may be destroyed before ncclGroupEnd().
    // 从操作句柄复制规约操作状态到本地结构体
    // 因为操作句柄可能在 ncclGroupEnd() 之前被销毁，所以需要提前保存
    struct ncclDevRedOpFull opDev;                       // 设备端规约操作的完整描述结构体
    //把主机的规约操作转化为设备规约操作值
    // 将主机端的规约操作（如 ncclSum）转换为设备端可以直接使用的格式
    NCCLCHECK(hostToDevRedOp(&opDev, info->op, info->datatype, comm)); // 执行转换操作

    // 特殊情况处理：单 rank 通信器
    // 单 rank 的情况下，集合操作退化为简单的内存复制
    if (comm->nRanks == 1) {                              // 如果通信器中只有 1 个 rank
      // 直接执行单 rank 操作（内部使用 cudaMemcpyAsync）
      NCCLCHECK(ncclLaunchOneRank(info->recvbuff, info->sendbuff, info->count, opDev, info->datatype, info->stream)); // 发送=接收，直接复制
      return ncclSuccess;                                 // 返回成功，无需创建任务
    } else {                                              // 多 rank 通信器，需要真正的集合通信
      // 查找对称窗口（用于 CE 引擎）
      // 对称窗口是 CE 引擎管理设备端缓冲区的机制
      struct ncclDevrWindow* sendWin;                    // 发送对称窗口指针
      struct ncclDevrWindow* recvWin;                    // 接收对称窗口指针
      ncclDevrFindWindow(comm, info->sendbuff, &sendWin); // 查找发送缓冲区对应的对称窗口
      ncclDevrFindWindow(comm, info->recvbuff, &recvWin); // 查找接收缓冲区对应的对称窗口

      // 检查 CE 引擎是否实现了此操作
      //CE只支持指定类型的集合操作
      bool ceImplemented = ncclCeImplemented(info->coll, info->op, info->datatype); // 查询 CE 实现状态

      // Append CE collective task if CE is supported and requested by user
      // 如果 CE 引擎支持且用户请求，则追加 CE 集合任务
      // CE 引擎需要满足以下所有条件才能使用：
      // 1. 对称操作支持（symmetricSupport）
      // 2. 单节点环境（nNodes == 1）
      // 3. 发送和接收窗口都存在
      // 4. 窗口标记为对称集合（NCCL_WIN_COLL_SYMMETRIC）
      // 5. CTA 策略为 ZERO（NCCL_CTA_POLICY_ZERO）
      // 6. CE 引擎实现了此操作
      if (comm->symmetricSupport && comm->nNodes == 1 && sendWin && recvWin && (sendWin->winFlags & recvWin->winFlags & NCCL_WIN_COLL_SYMMETRIC) && comm->config.CTAPolicy == NCCL_CTA_POLICY_ZERO && ceImplemented) {
        // 追加 CE 集合任务（使用设备端运行时）
        NCCLCHECK(ceCollTaskAppend(comm, info, sendWin, recvWin, opDev)); // CE 引擎路径
      }
      // Append kernel-based collective
      // 否则，追加基于内核的集合任务
      // 这是传统的执行路径，使用 CUDA 内核在 GPU 上执行集合通信
      else {                                                // 常规内核执行路径
        //根据操作类型选择不同的函数调用
        // 不同的集合操作有不同的最优实现方式
        if (info->coll == ncclFuncAlltoAll) {            // AlltoAll：全收集到全（每个 rank 向所有其他 rank 发送数据）
          // AlltoAll 实现为 nRanks 个 Send 和 nRanks 个Recv 操作
          // 对于每个 rank，发送当前 rank 的数据，接收该 rank 的数据
          for (int r=0; r<comm->nRanks; r++) {            // 遍历所有 rank
            // 发送当前 rank 的数据给 rank r
            // 计算发送缓冲区偏移：r * count * elementSize
            NCCLCHECK(p2pTaskAppend(comm, info, ncclFuncSend, collAPI, (void*)((char*)info->sendbuff+r*info->count*ncclTypeSize(info->datatype)), info->count, info->datatype, r));
            // 从 rank r 接收数据
            // 计算接收缓冲区偏移：r * count * elementSize
            NCCLCHECK(p2pTaskAppend(comm, info, ncclFuncRecv, collAPI, (void*)((char*)info->recvbuff+r*info->count*ncclTypeSize(info->datatype)), info->count, info->datatype, r));
          }
        } else if (info->coll == ncclFuncGather){         // Gather：根 rank 从所有 rank 收集数据
          size_t offset = 0;                               // 接收缓冲区的偏移量
          // 所有 rank 都要发送数据给根 rank
          NCCLCHECK(p2pTaskAppend(comm, info, ncclFuncSend, collAPI, (void*)info->sendbuff, info->count, info->datatype, info->root)); // 发送到根 rank
          // 只有根 rank 需要接收所有其他 rank 的数据
          if (comm->rank == info->root) {                  // 如果当前 rank 是根 rank
            for (int r=0; r<comm->nRanks; r++) {          // 遍历所有 rank
              void* buff = (void*)((char*)info->recvbuff + offset); // 计算当前 rank 的接收缓冲区位置
              NCCLCHECK(p2pTaskAppend(comm, info, ncclFuncRecv, collAPI, buff, info->count, info->datatype, r)); // 从 rank r 接收数据
              offset += info->count * ncclTypeSize(info->datatype); // 更新偏移量，为下一个 rank 准备
            }
          }
        } else if (info->coll == ncclFuncScatter) {        // Scatter：根 rank 分发数据给所有 rank
          size_t offset = 0;                               // 发送缓冲区的偏移量
          // 根 rank 要给所有其他 rank 发送数据
          if (comm->rank == info->root) {                  // 如果当前 rank 是根 rank
            for (int r = 0; r < comm->nRanks; r++) {       // 遍历所有 rank
              void* buff = (void*)((char*)info->sendbuff + offset); // 计算当前 rank 的发送缓冲区位置
              NCCLCHECK(p2pTaskAppend(comm, info, ncclFuncSend, collAPI, buff, info->count, info->datatype, r)); // 发送数据给 rank r
              offset += info->count * ncclTypeSize(info->datatype); // 更新偏移量，为下一个 rank 准备
            }
          }
          // 所有 rank（包括根 rank）都要从根 rank 接收数据
          NCCLCHECK(p2pTaskAppend(comm, info, ncclFuncRecv, collAPI, (void*)info->recvbuff, info->count, info->datatype, info->root)); // 从根 rank 接收数据
        } else {                                          
            // 其他集合操作（AllReduce、Broadcast、ReduceScatter、Reduce 等）
          // 使用常规的集合通信任务追加路径
          // 这些操作使用专门的内核实现，支持多种算法和协议
          NCCLCHECK(collTaskAppend(comm, info, opDev));   // 追加常规集合任务
        }
      }
    }
  }

  return ncclSuccess;                                   // 返回成功状态
}

// NCCL 集合操作的入队检查函数
// 这是所有 NCCL 集合操作 API（AllReduce、AllGather 等）的入口点
// 负责参数验证、通信器检查、设备管理和任务入队
// 参数 info: 包含操作所有信息的结构体指针
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
ncclResult_t ncclEnqueueCheck(struct ncclInfo* info) {
  // Profiler - If a group API event has already started, update the profilerGroupDepth so that the depth
  // updates correctly for implicit ncclGroupStartInternal and ncclGroupEndInternal calls
  // 性能分析器 - 如果组 API 事件已经开始，更新 profilerGroupDepth
  // 这样对于隐式的 ncclGroupStartInternal 和 ncclGroupEndInternal 调用，深度可以正确更新
  if (ncclProfilerApiState.profilerGroupDepth > 0) {
    ncclProfilerApiState.profilerGroupDepth++;
  }
  // 开始内部组操作（支持 NCCL 组 API）
  NCCLCHECK(ncclGroupStartInternal());
  ncclResult_t ret = ncclSuccess;
  // 保存旧的设备 ID（用于恢复）
  int devOld = -1;

//参数合法性检查
  // 检查通信器是否有效
  NCCLCHECKGOTO(CommCheck(info->comm, info->opName, "comm"), ret, fail);

  // Check whether communicator is ready to communicate
  //检查通信器是否准备好了
  // 确保通信器已初始化并准备好通信
  NCCLCHECKGOTO(ncclCommEnsureReady(info->comm), ret, fail);

  // 如果配置要求检查指针，切换到正确的设备
  if (info->comm->checkPointers) {
    // 获取当前设备
    CUDACHECKGOTO(cudaGetDevice(&devOld), ret, fail);
    // 切换到通信器的设备
    CUDACHECKGOTO(cudaSetDevice(info->comm->cudaDev), ret, fail);
  }
  //参数合法性检查
  // 检查操作参数（缓冲区、计数、数据类型等）
  NCCLCHECKGOTO(ArgsCheck(info), ret, fail);

  // 输出操作信息（日志）
  INFO(NCCL_COLL,"%s: opCount %lx sendbuff %p recvbuff %p count %zu datatype %d op %d root %d comm %p [nranks=%d] stream %p",
        info->opName, info->comm->opCount, info->sendbuff, info->recvbuff, info->count,
        info->datatype, info->op, info->root, info->comm, info->comm->nRanks, info->stream);
  // 输出跟踪信息
  TRACE_CALL("nccl%s(%" PRIx64 ",%" PRIx64 ",%zu,%d,%d,%d,%p,%p)", info->opName, reinterpret_cast<int64_t>(info->sendbuff), reinterpret_cast<int64_t>(info->recvbuff), info->count, info->datatype, info->op, info->root, info->comm, info->stream);

  // 将任务追加到通信器的队列中，这里只是任务入队，并不触发真正的内核操作
  NCCLCHECKGOTO(taskAppend(info->comm, info), ret, fail);

exit:
  // 恢复原来的设备（如果之前切换过）
  if (devOld != -1)
    CUDACHECK(cudaSetDevice(devOld));
  // 检查组操作的错误
  ncclGroupErrCheck(ret);
  // 结束内部组操作，触发kernel函数执行数据通信  操作，触发真正的内核启动流程
  //groupLaunch → doLaunches → ncclLaunchKernel：层层调用，最终启动 CUDA 内核
  NCCLCHECK(ncclGroupEndInternal());

  /* if depth is 1, ncclGroupEndInternal() will trigger group ops. The state can change
   * so we have to check state here. */
  /* 如果深度为 1，ncclGroupEndInternal() 将触发组操作。状态可能会改变，所以这里必须检查状态。 */
  // 如果是非阻塞模式，获取异步错误状态
  if (info->comm && !info->comm->config.blocking) { 
    NCCLCHECK(ncclCommGetAsyncError(info->comm, &ret));
  }
  return ret;
fail:
  // 失败时，如果是非阻塞模式，设置异步错误
  if (info->comm && !info->comm->config.blocking) 
    (void) ncclCommSetAsyncError(info->comm, ret);
  goto exit;
}

// NCCL API 声明：创建预乘求和归约操作
NCCL_API(ncclResult_t, ncclRedOpCreatePreMulSum, ncclRedOp_t *op, void *scalar, ncclDataType_t datatype, ncclScalarResidence_t residence, ncclComm_t comm);
// 创建预乘求和归约操作
// 这个函数创建一个特殊的归约操作，在求和之前先乘以一个标量
// 参数 op: 输出参数，返回创建的操作句柄
// 参数 scalar: 标量值指针（用于预乘）
// 参数 datatype: 标量和数据的数据类型
// 参数 residence: 标量驻留位置（主机或设备）
// 参数 comm: 通信器
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
ncclResult_t ncclRedOpCreatePreMulSum(ncclRedOp_t *op, void *scalar, ncclDataType_t datatype, ncclScalarResidence_t residence, ncclComm_t comm) {
  // 检查通信器是否有效
  NCCLCHECK(CommCheck(comm, "ncclRedOpCreatePreMulSum", "comm"));
  /* join init thread before creating PreMulSum op. */
  /* 在创建 PreMulSum 操作之前，等待初始化线程完成 */
  NCCLCHECK(ncclCommEnsureReady(comm));

  // 如果空闲列表为空，需要扩展容量
  if (comm->userRedOpFreeHead == comm->userRedOpCapacity) {
    // double capacity and resize
    // 容量翻倍并调整大小
    int cap = 2*comm->userRedOpCapacity;
    // 最小容量为 4
    if (cap < 4) 
        cap = 4;
    // 分配新的操作数组
    ncclUserRedOp *ops = new ncclUserRedOp[cap];
    // 如果有旧数据，复制到新数组
    if (comm->userRedOpCapacity > 0)
      std::memcpy(ops, comm->userRedOps, comm->userRedOpCapacity*sizeof(ncclUserRedOp));
    // 初始化新分配的槽位为空闲链表
    for(int ix=comm->userRedOpCapacity; ix < cap; ix++)
      ops[ix].freeNext = ix + 1;
    // 释放旧数组
    delete[] comm->userRedOps;
    comm->userRedOps = ops;
    comm->userRedOpCapacity = cap;
  }
  // pop from free list
  // 从空闲列表弹出一个槽位
  int ix = comm->userRedOpFreeHead;
  ncclUserRedOp *user = &comm->userRedOps[ix];
  // 更新空闲列表头
  comm->userRedOpFreeHead = user->freeNext;

  // 标记为已分配
  user->freeNext = -1; // allocated
  // 设置数据类型
  user->datatype = datatype;
  // 设置操作类型为预乘求和
  user->opFull.op = ncclDevPreMulSum;
  // 根据标量驻留类型处理标量参数
  if (residence == ncclScalarHostImmediate) {
    // 标量在主机内存中（立即值）
    int size = ncclTypeSize(datatype);
    if (size < 1) 
        return ncclInternalError;
    // 标量不是指针，是立即值
    user->opFull.scalarArgIsPtr = false;
    // 复制标量值到操作结构体
    std::memcpy(&user->opFull.scalarArg, scalar, size);
  } else {
    // 标量在设备内存中
    user->opFull.scalarArgIsPtr = true;
    // 保存标量指针
    user->opFull.scalarArg = reinterpret_cast<uint64_t>(scalar);
  }
  // 计算操作句柄（基础操作数 + 索引）
  *op = ncclRedOp_t(int(ncclNumOps) + ix);
  // 对句柄进行混淆（防止用户句柄冲突）
  *op = ncclUserRedOpMangle(comm, *op);
  // 输出跟踪信息
  TRACE_CALL("ncclRedOpCreatePreMulSum(%d,%p,%d,%d,%p)", *op, scalar, datatype, residence, comm);
  return ncclSuccess;
}

// NCCL API 声明：销毁归约操作
NCCL_API(ncclResult_t, ncclRedOpDestroy, ncclRedOp_t op, ncclComm_t comm);
// 销毁用户创建的归约操作（释放资源）
// 参数 op: 要销毁的操作句柄
// 参数 comm: 通信器
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
ncclResult_t ncclRedOpDestroy(ncclRedOp_t op, ncclComm_t comm) {
  // 检查是否是 NCCL 内置操作（不能销毁）
  if (0 <= int(op) && int(op) < int(ncclNumOps)) {
    WARN("ncclRedOpDestroy : operator is a NCCL builtin.");
    return ncclInvalidArgument;
  }
  // int(ncclMaxRedOp) < int(op) will always be false due to the sizes of
  // the datatypes involved, and that's by design.  We keep the check though
  // just as a reminder.
  // coverity[result_independent_of_operands]
  // 检查操作是否是垃圾值（超出范围）
  if (int(op) < 0 || int(ncclMaxRedOp) < int(op)) {
    WARN("ncclRedOpDestroy :  operator is garbage.");
    return ncclInvalidArgument;
  }
  // 检查通信器是否有效
  if (comm == NULL) {
    WARN("ncclRedOpDestroy : invalid communicator passed.");
    return ncclInvalidArgument;
  }

  // 反混淆操作句柄，获取索引
  int ix = int(ncclUserRedOpMangle(comm, op)) - int(ncclNumOps);
  // 检查操作是否属于此通信器且未被销毁
  if (comm->userRedOpCapacity <= ix || comm->userRedOps[ix].freeNext != -1) {
    WARN("ncclRedOpDestroy : operator unknown to this communicator.");
    return ncclInvalidArgument;
  }
  // push to free list
  // 将操作槽位推回空闲列表（标记为可用）
  comm->userRedOps[ix].freeNext = comm->userRedOpFreeHead;
  comm->userRedOpFreeHead = ix;
  // 输出跟踪信息
  TRACE_CALL("ncclRedOpDestroy(%d,%p)", op, comm);
  return ncclSuccess;
}
