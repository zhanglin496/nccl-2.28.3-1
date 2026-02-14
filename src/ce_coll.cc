/*************************************************************************
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

/**
 * @file ce_coll.cc
 * @brief NCCL Copy Engine (CE) Collective Operations Implementation
 *
 * 关键概念说明：
 * ================
 *
 * 1. Copy Engine (CE)
 *    - CE (Copy Engine) 是 NVIDIA GPU 中的专用 DMA 引擎
 *    - 用于在 GPU 内存之间直接复制数据，无需 SM (Streaming Multiprocessor) 参与
 *    - 优势：
 *      * 释放 SM 资源用于计算任务
 *      * 降低延迟（DMA 引擎专门针对内存复制优化）
 *      * 支持更细粒度的流水线操作
 *
 * 2. CE vs 传统 NCCL 通信
 *    - 传统方式：SM 执行内核，通过共享内存和网络进行数据传输
 *    - CE 方式：使用 DMA 引擎直接进行 GPU 间内存复制
 *    - 适用场景：
 *      * 纯数据移动操作（AllGather, AlltoAll, Scatter, Gather）
 *      * 不适用于需要计算的操作（如 AllReduce 需要规约计算）
 *
 * 3. 对称内存 (Symmetric Memory)
 *    - 所有 GPU 上相同虚拟地址映射到相同的物理内存偏移
 *    - 使用 LSA (Local Symmetric Address) 空间进行寻址
 *    - 支持 P2P 和 NVLS (NVLink SHARP) 两种访问方式
 *
 * 4. 同步机制
 *    - Ready 指针：表示 rank 已准备好开始操作
 *    - Complete 指针：表示 rank 已完成操作
 *    - 序列号 (ceSeqNum)：用于区分不同操作，避免乱序问题
 *    - 双缓冲：通过 useCompletePtr 切换使用不同的同步指针
 *
 * 5. MC (Multicast) vs UC (Unicast)
 *    - MC：使用 NVLS 多播，一次写入所有 GPU（更快，需要 NVLS 支持）
 *    - UC：单播，逐个写入每个 GPU
 *
 * 6. 批量操作 (Batch Operations)
 *    - 将多个内存复制操作合并为一个批次执行
 *    - CUDA 12.8+ 支持 cudaMemcpyBatchAsync API
 *    - 提高内存复制效率
 *
 * 7. Intra-Batch 同步
 *    - 在大批量操作中定期插入同步点
 *    - 防止内存缓冲区溢出和流水线停顿
 *    - 由 intraBatchSyncFreq 和 intraBatchSyncMsgThreshold 控制
 */

#include "comm.h"
#include "register_inline.h"
#include <cuda.h>
#include "cudawrap.h"
#include "ce_coll.h"
#include "alloc.h"

// Static constant for graph synchronization
// CUDA Graph 捕获期间使用的同步常量值
// 由于 CUDA Graph 需要可重放性，使用固定值而非递增的序列号
static const uint32_t GRAPH_SYNC_VALUE = 1;

// Static constants for intra-batch synchronization to improve CE collective performance with large scale
// 大规模 CE 集合通信的性能优化参数：批次内同步
// 批次内同步频率：每隔 N 个操作执行一次同步
static const uint32_t CE_COLL_INTRA_BATCH_SYNC_FREQ = 8;
// 批次内同步消息阈值：当总消息大小超过此值时启用批次内同步
static const uint64_t CE_COLL_INTRA_BATCH_SYNC_MSG_THRESHOLD = 512*1024*1024; // 512MB

/**
 * @brief 初始化 Copy Engine 集合通信子系统
 * @param comm NCCL 通信器指针
 * @return ncclResult_t 操作结果状态码
 *
 * 功能说明：
 * 1. 初始化对称内存运行时 (DevR)
 * 2. 分配并注册同步内存窗口
 *    - Ready 指针数组：所有 rank 标记自己已准备好
 *    - Complete 指针数组：所有 rank 标记自己已完成
 * 3. 设置同步指针和序列号
 *
 * 内存布局：
 * ┌─────────────────────────────────────┐
 * │ Ready 指针数组 (nRanks * uint32_t)  │
 * ├─────────────────────────────────────┤
 * │ Complete 指针数组 (nRanks * uint32_t)│
 * └─────────────────────────────────────┘
 * 每个数组都对齐到 16 字节边界以优化访问
 */
ncclResult_t ncclCeInit(struct ncclComm* comm) {
  ncclResult_t ret = ncclSuccess;

  // CE 设备端内存基地址（包含 ready 和 complete 两个同步数组）
  uint8_t* ceDevBase;
  // 计算所需的内存大小：
  // - 每个数组包含 nRanks 个 uint32_t 元素
  // - 对齐到 16 字节边界（缓存行对齐，优化访问性能）
  // - 乘以 2 是因为需要 ready 和 complete 两个数组
  size_t ceDevBaseSize = alignUp(comm->nRanks*sizeof(uint32_t), 16) * 2;
  // 设备端窗口指针（用于对称内存注册）
  ncclWindow_vidmem* ceWinDev;
  // 主机端窗口指针（用于访问和管理设备端窗口）
  ncclWindow_vidmem* ceWinDevHost;

  // Ensure symmetric memory runtime is initialized
  // 确保对称内存运行时已初始化（DevR 是 Device Runtime 的缩写）
  NCCLCHECKGOTO(ncclDevrInitOnce(comm), ret, fail);
  // Allocate and register memory for the symmetric memory
  // 分配对称内存，并在组内注册为集合通信窗口
  // NCCL_WIN_COLL_SYMMETRIC 标志表示这是一个用于集合通信的对称窗口
  NCCLCHECKGOTO(ncclMemAlloc((void**)&ceDevBase, ceDevBaseSize), ret, fail);
  NCCLCHECKGOTO(ncclDevrWindowRegisterInGroup(comm, ceDevBase, ceDevBaseSize, NCCL_WIN_COLL_SYMMETRIC, &ceWinDev), ret, fail);
  // 将设备端窗口指针映射到主机端，以便主机代码可以访问
  NCCLCHECKGOTO(ncclShadowPoolToHost(&comm->devrState.shadows, ceWinDev, &ceWinDevHost), ret, fail);
  // Get the ncclDevrWindow from the winHost field
  // 从主机端窗口结构中提取 ncclDevrWindow 指针
  // winHost 字段指向实际的设备运行时窗口结构
  comm->ceColl.ceSyncWin = (struct ncclDevrWindow*)ceWinDevHost->winHost;

  // 初始化 Ready 指针数组的偏移量（从基地址开始）
  comm->ceColl.baseUCSymReadyOffset = 0;
  // 初始化 Complete 指针数组的偏移量（紧跟在 Ready 数组之后）
  // alignUp 确保数组起始地址对齐到 16 字节边界
  comm->ceColl.baseUCSymComplOffset = alignUp(comm->nRanks*sizeof(uint32_t), 16);
  // 计算 Ready 指针数组的起始地址（用户空间指针）
  comm->ceColl.baseUCSymReadyPtr = (uint8_t*)comm->ceColl.ceSyncWin->userPtr + comm->ceColl.baseUCSymReadyOffset;
  // 计算 Complete 指针数组的起始地址
  comm->ceColl.baseUCSymComplPtr = (uint8_t*)comm->ceColl.ceSyncWin->userPtr + comm->ceColl.baseUCSymComplOffset;
  // 初始化序列号为 0（每次同步操作会递增此值）
  comm->ceColl.ceSeqNum = 0;
  // 初始化为使用 Complete 指针（双缓冲机制，每次同步后切换）
  comm->ceColl.useCompletePtr = false;
  // 设置批次内同步参数
  comm->ceColl.intraBatchSyncFreq = CE_COLL_INTRA_BATCH_SYNC_FREQ;
  comm->ceColl.intraBatchSyncMsgThreshold = CE_COLL_INTRA_BATCH_SYNC_MSG_THRESHOLD;
  // 输出初始化信息日志（包含指针地址和序列号，便于调试）
  INFO(NCCL_INIT, "Init CE, rank %d baseUCSymReadyPtr %p, baseUCSymComplPtr %p, seq num %d", comm->rank, comm->ceColl.baseUCSymReadyPtr, comm->ceColl.baseUCSymComplPtr, comm->ceColl.ceSeqNum);

exit:
  return ret;
fail:
  goto exit;
}

/**
 * @brief 清理 Copy Engine 集合通信资源
 * @param comm NCCL 通信器指针
 * @return ncclResult_t 操作结果状态码
 *
 * 功能说明：
 * 1. 清理 CE 初始化任务队列
 * 2. 注销同步内存窗口
 * 3. 释放分配的内存
 */
ncclResult_t ncclCeFinalize(struct ncclComm* comm) {
  ncclResult_t ret = ncclSuccess;

  // Clean up ceInitTaskQueue
  // 清空并释放 CE 初始化任务队列中的所有任务
  // ceInitTaskQueue 用于延迟某些初始化操作
  while (!ncclIntruQueueEmpty(&comm->ceInitTaskQueue)) {
    struct ncclCeInitTask* task = ncclIntruQueueDequeue(&comm->ceInitTaskQueue);
    free(task);
  }

  // Clean up CE resources
  // 清理 CE 相关资源（同步内存窗口）
  if (comm->ceColl.baseUCSymReadyPtr != NULL) {
    // 检查同步窗口是否存在且包含有效的视频内存
    if (comm->ceColl.ceSyncWin && comm->ceColl.ceSyncWin->vidmem) {
      // 注销窗口（解除内存注册）
      NCCLCHECKGOTO(ncclCommWindowDeregister(comm, comm->ceColl.ceSyncWin->vidmem), ret, fail);
      // 释放内存
      NCCLCHECKGOTO(ncclMemFree(comm->ceColl.baseUCSymReadyPtr), ret, fail);
    }
    // 清空指针，避免悬垂指针
    comm->ceColl.baseUCSymReadyPtr = NULL;
    comm->ceColl.baseUCSymComplPtr = NULL;
    comm->ceColl.ceSyncWin = NULL;
  }

exit:
  return ret;
fail:
  goto exit;
}

/**
 * @brief 检查指定的集合操作是否支持 Copy Engine 实现
 * @param coll 集合操作类型（AllGather, AlltoAll, Scatter, Gather 等）
 * @param red 归约操作类型（暂未使用）
 * @param ty 数据类型
 * @return true 如果 CE 支持该操作，false 否则
 *
 * 功能说明：
 * - CE 仅支持纯数据移动操作，不支持需要计算的操作（如 AllReduce）
 * - 需要 CUDA 12.5 或更高版本
 * - 支持的操作：AllGather, AlltoAll, Scatter, Gather
 */
bool ncclCeImplemented(ncclFunc_t coll, int/*ncclDevRedOp_t*/ red, ncclDataType_t ty) {
  int driverVersion;
  // 尝试获取 CUDA 驱动版本，失败则返回不支持
  if (ncclCudaDriverVersion(&driverVersion) != ncclSuccess) 
    return false;

  // CE is supported in CUDA 12.5 and later
  // CE 功能需要 CUDA 12.5 或更高版本
  if (driverVersion >= 12050) {
    switch (coll) {
    // 支持的集合操作类型
    case ncclFuncAllGather:   // AllGather: 每个 rank 的数据被收集到所有 rank
    case ncclFuncAlltoAll:    // AlltoAll: 每个 rank 向所有其他 rank 发送不同的数据
    case ncclFuncScatter:     // Scatter: root rank 将数据分发到所有 rank
    case ncclFuncGather:      // Gather: 所有 rank 的数据收集到 root rank
      return true;
    // 不支持的操作（如 AllReduce, Broadcast, ReduceScatter 等）
    default:
      return false;
    }
  }
  return false;
}

/**
 * @brief 准备多播（MC）同步操作
 * @param comm NCCL 通信器指针
 * @param isComplete 是否使用 Complete 指针（true）或 Ready 指针（false）
 * @param batchParams 批量内存操作参数数组
 * @param opIdx 当前操作索引（输入输出参数）
 * @param stream CUDA 流
 * @return ncclResult_t 操作结果状态码
 *
 * 功能说明：
 * - MC (Multicast) 同步使用 NVLS 多播功能
 * - 一次写入操作可以同时到达所有 GPU
 * - 比单播方式更高效，但需要 NVLS 支持
 *
 * 同步流程：
 * 1. 当前 rank 将自己的同步标志写入多播地址（所有 GPU 同时可见）
 * 2. 等待其他所有 rank 的同步标志
 */
ncclResult_t ncclPrepMCSync(struct ncclComm* comm, bool isComplete, CUstreamBatchMemOpParams* batchParams, size_t* opIdx, cudaStream_t stream) {
  ncclResult_t ret = ncclSuccess;

  // 获取 Ready 和 Complete 指针数组（转换为 uint32_t*）
  uint32_t* readyPtrs    = (uint32_t*)comm->ceColl.baseUCSymReadyPtr;
  uint32_t* completePtrs = (uint32_t*)comm->ceColl.baseUCSymComplPtr;

  // 检查是否正在捕获 CUDA Graph
  bool capturing = ncclCudaGraphValid(comm->planner.capturingGraph);
  // 获取当前序列号并递增（用于区分不同的同步操作）
  uint32_t currentSeq = ++comm->ceColl.ceSeqNum;

  // Source pointer is either the constant graph sync value or the sequence number
  // 源指针：CUDA Graph 捕获时使用常量值，否则使用当前序列号
  void* srcPtr = capturing ? (void*)&GRAPH_SYNC_VALUE : (void*)&currentSeq;
  // Wait value is either the constant graph sync value or the sequence number
  // 等待值：与源指针相同，用于等待操作
  uint32_t waitValue = capturing ? GRAPH_SYNC_VALUE : currentSeq;

  // Use multi-cast address as destination pointer
  // 准备使用多播地址作为目标指针
  void* mcDstPtr;
  // 计算当前 rank 的同步指针地址（Ready 或 Complete）
  void* dstPtr = isComplete ? (void*)&completePtrs[comm->rank] : (void*)&readyPtrs[comm->rank];
  // 计算相对于窗口起始地址的偏移量
  size_t offset = (uint8_t*)dstPtr - (uint8_t*)comm->ceColl.ceSyncWin->userPtr;
  // 获取 LSA 团队的多播地址（所有 GPU 通过 NVLS 同时接收）
  // ncclTeamLsa(comm) 获取当前通信器的 LSA 团队标识
  NCCLCHECKGOTO(ncclDevrGetLsaTeamPtrMC(comm, comm->ceColl.ceSyncWin, offset, ncclTeamLsa(comm), &mcDstPtr), ret, fail);

  // Write our own ready/complete flag to the multi-cast address
  // 将当前 rank 的同步标志写入多播地址（所有 GPU 同时可见）
  // cudaMemcpyHostToDevice：从主机内存（srcPtr）复制到设备内存（mcDstPtr）
  CUDACHECKGOTO(cudaMemcpyAsync(
    mcDstPtr,              // 目标地址（多播地址）
    srcPtr,                // 源地址（序列号值）
    sizeof(uint32_t),      // 复制大小
    cudaMemcpyHostToDevice, // 复制方向
    stream), ret, fail);

  // Add local wait operations for every other rank
  // 为每个其他 rank 添加本地等待操作
  // 这确保当前 rank 等待所有其他 rank 都准备好/完成
  for (int r = 0; r < comm->nRanks; ++r) {
    if (r == comm->rank) continue;  // 跳过自己
    // 初始化批量操作参数
    batchParams[*opIdx] = {};
    // 设置为等待值 32 位操作
    batchParams[*opIdx].waitValue.operation = CU_STREAM_MEM_OP_WAIT_VALUE_32;
    // 等待地址：目标 rank 的同步指针
    batchParams[*opIdx].waitValue.address = (CUdeviceptr)(isComplete ? (void*)&completePtrs[r] : (void*)&readyPtrs[r]);
    // 等待值：序列号
    batchParams[*opIdx].waitValue.value = waitValue;
    // 等待条件：相等时继续
    batchParams[*opIdx].waitValue.flags = CU_STREAM_WAIT_VALUE_EQ;
    // 递增操作索引
    (*opIdx)++;
  }

exit:
  return ret;
fail:
  goto exit;
}

/**
 * @brief 准备单播（UC）同步操作
 * @param comm NCCL 通信器指针
 * @param isComplete 是否使用 Complete 指针
 * @param batchParams 批量内存操作参数数组
 * @param opIdx 当前操作索引
 * @return ncclResult_t 操作结果状态码
 *
 * 功能说明：
 * - UC (Unicast) 同步不使用 NVLS 多播
 * - 逐个向每个其他 rank 写入同步标志
 * - 适用于没有 NVLS 支持的系统
 *
 * 与 MC 同步的区别：
 * - MC：一次多播写入到达所有 GPU
 * - UC：N-1 次单播写入（N 是 rank 数量）
 */
ncclResult_t ncclPrepUCSync(struct ncclComm* comm, bool isComplete,
                               CUstreamBatchMemOpParams* batchParams,
                               size_t* opIdx) {
  ncclResult_t ret = ncclSuccess;

  // 获取 Ready 和 Complete 指针数组
  uint32_t* readyPtrs    = (uint32_t*)comm->ceColl.baseUCSymReadyPtr;
  uint32_t* completePtrs = (uint32_t*)comm->ceColl.baseUCSymComplPtr;

  // 检查是否正在捕获 CUDA Graph
  bool capturing = ncclCudaGraphValid(comm->planner.capturingGraph);
  // 获取当前序列号并递增
  uint32_t currentSeq = ++comm->ceColl.ceSeqNum;

  // Write our own ready/complete flag to remote ranks
  // 向每个远程 rank 写入当前 rank 的同步标志
  uint32_t waitValue = capturing ? GRAPH_SYNC_VALUE : currentSeq;
  for (int r = 0; r < comm->nRanks; ++r) {
    if (r == comm->rank) 
        continue;  // 跳过自己
    // 获取目标 rank 的同步指针地址（通过 LSA 映射）
    void * peerDstPtr;
    // 当前 rank 的同步指针（本地地址）
    void* dstPtr = isComplete ? (void*)&completePtrs[comm->rank] : (void*)&readyPtrs[comm->rank];
    // 计算偏移量
    size_t offset = (uint8_t*)dstPtr - (uint8_t*)comm->ceColl.ceSyncWin->userPtr;
    // 获取远程 rank 的 LSA 指针（通过对称内存映射）
    // r 是目标 rank 的索引
    NCCLCHECKGOTO(ncclDevrGetLsaRankPtr(comm, comm->ceColl.ceSyncWin, offset, r, &peerDstPtr), ret, fail);
    // 设置写入操作参数
    batchParams[*opIdx] = {};
    // 写入值 32 位操作
    batchParams[*opIdx].writeValue.operation = CU_STREAM_MEM_OP_WRITE_VALUE_32;
    // 目标地址：远程 rank 的同步指针
    batchParams[*opIdx].writeValue.address  = (CUdeviceptr)peerDstPtr;
    // 写入值：序列号
    batchParams[*opIdx].writeValue.value = waitValue;
    // 标志：使用默认写入属性
    batchParams[*opIdx].writeValue.flags = CU_STREAM_WRITE_VALUE_DEFAULT;
    (*opIdx)++;
  }

  // Add local wait operations for every other rank
  // 为每个其他 rank 添加本地等待操作
  // 这确保当前 rank 等待所有其他 rank 都准备好/完成
  for (int r = 0; r < comm->nRanks; ++r) {
    if (r == comm->rank) continue;  // 跳过自己
    batchParams[*opIdx] = {};
    // 等待值 32 位操作
    batchParams[*opIdx].waitValue.operation = CU_STREAM_MEM_OP_WAIT_VALUE_32;
    // 等待地址：其他 rank 的同步指针
    batchParams[*opIdx].waitValue.address  = (CUdeviceptr)(isComplete ? (void*)&completePtrs[r] : (void*)&readyPtrs[r]);
    // 等待值：序列号
    batchParams[*opIdx].waitValue.value = waitValue;
    // 等待条件：相等时继续
    batchParams[*opIdx].waitValue.flags = CU_STREAM_WAIT_VALUE_EQ;
    (*opIdx)++;
  }

exit:
  return ret;
fail:
  goto exit;
}


/**
 * @brief 执行内存操作同步
 * @param comm NCCL 通信器指针
 * @param stream CUDA 流
 * @return ncclResult_t 操作结果状态码
 *
 * 功能说明：
 * 1. 根据 NVLS 支持情况选择 MC 或 UC 同步方式
 * 2. 准备批量内存操作参数
 * 3. 如果正在捕获 CUDA Graph，添加重置操作
 * 4. 执行批量内存操作
 * 5. 切换同步指针（Ready/Complete 双缓冲）
 *
 * 同步机制：
 * - 使用双缓冲机制：交替使用 Ready 和 Complete 指针
 * - 序列号机制：确保不会混淆不同操作的同步信号
 * - CUDA Graph 支持：使用固定值而非递增序列号
 */
ncclResult_t ncclMemOpSync(struct ncclComm* comm, cudaStream_t stream) {
  ncclResult_t ret = ncclSuccess;

  // Get pointers to the ready and complete synchronization arrays
  // 获取 Ready 和 Complete 同步指针数组（转换为 uint32_t*）
  uint32_t* readyPtrs = (uint32_t*)comm->ceColl.baseUCSymReadyPtr;
  uint32_t* completePtrs = (uint32_t*)comm->ceColl.baseUCSymComplPtr;

  // Allocate enough slots for all possible ops
  // 计算所需的批量操作数量
  // MC 同步：每个 rank 需要 2 个操作（1 个多播写入 + N-1 个等待）
  // UC 同步：每个 rank 需要 3 个操作（N-1 个单播写入 + N-1 个等待）
  size_t batchSize = (comm->nvlsSupport ? NCCL_CE_SYNC_OPS_PER_RANK_MC : NCCL_CE_SYNC_OPS_PER_RANK_UC) * comm->nRanks;
  size_t opIdx = 0;  // 当前操作索引

  // Prepare batch memory operations for synchronization
  // 分配批量内存操作参数数组
  CUstreamBatchMemOpParams* batchParams = nullptr;
  NCCLCHECKGOTO(ncclCalloc(&batchParams, batchSize), ret, fail);

  // 根据系统支持选择同步方式
  if (comm->nvlsSupport) {
    // 使用多播（MC）同步（NVLS 支持）
    NCCLCHECKGOTO(ncclPrepMCSync(comm, comm->ceColl.useCompletePtr, batchParams, &opIdx, stream), ret, fail);
  } else {
    // 使用单播（UC）同步（无 NVLS 支持）
    NCCLCHECKGOTO(ncclPrepUCSync(comm, comm->ceColl.useCompletePtr, batchParams, &opIdx), ret, fail);
  }

  // For CUDA graph capture, add reset operation
  // 如果正在捕获 CUDA Graph，添加重置操作
  // 原因：CUDA Graph 需要可重放性，每次执行后需要重置同步值
  if (ncclCudaGraphValid(comm->planner.capturingGraph)) {
    for (int i = 0; i < comm->nRanks; i++) {
      batchParams[opIdx] = {};
      // 写入值 32 位操作
      batchParams[opIdx].writeValue.operation = CU_STREAM_MEM_OP_WRITE_VALUE_32;
      // 重置地址：Ready 或 Complete 指针
      batchParams[opIdx].writeValue.address = (CUdeviceptr)(comm->ceColl.useCompletePtr ? (void*)&completePtrs[i] : (void*)&readyPtrs[i]);
      // 重置值：0（初始状态）
      batchParams[opIdx].writeValue.value = 0;
      // 标志：使用默认写入属性
      batchParams[opIdx].writeValue.flags = CU_STREAM_WRITE_VALUE_DEFAULT;
      opIdx++;
    }
  }

  // Execute all memory operations in a single batch
  // 在单个批次中执行所有内存操作
  // cuStreamBatchMemOp：CUDA Driver API，批量执行内存操作
  CUCHECKGOTO(cuStreamBatchMemOp(stream, opIdx, batchParams, 0), ret, fail);

  // Toggle the flag for next call
  // 切换同步指针标志（双缓冲机制）
  // 下次调用将使用另一个指针数组
  comm->ceColl.useCompletePtr = !comm->ceColl.useCompletePtr;

exit:
  if (batchParams) free(batchParams);  // 释放批量操作参数数组
  return ret;
fail:
  goto exit;
}

/**
 * @brief 初始化批量操作参数结构
 * @param params 批量操作参数指针
 * @param nRanks rank 数量（决定数组大小）
 * @return ncclResult_t 操作结果状态码
 *
 * 功能说明：
 * - 初始化批量操作所需的源地址、目标地址、大小数组
 * - CUDA 12.8+ 还支持属性数组
 */
ncclResult_t ncclCeInitBatchOpsParams(struct ncclCeBatchOpsParams* params, int nRanks) {
  ncclResult_t ret = ncclSuccess;

  // 初始化所有指针为空
  params->srcs = nullptr;
  params->dsts = nullptr;
  params->sizes = nullptr;
  params->numOps = 0;
  params->intraBatchSync = false;  // 默认不使用批次内同步
#if CUDART_VERSION >= 12080
  // CUDA 12.8+ 支持额外的复制属性
  params->attrs = nullptr;
  params->attrIdxs = nullptr;
  params->numAttrs = 0;
#endif

  // 分配源地址数组（每个操作一个源地址）
  NCCLCHECKGOTO(ncclCalloc(&params->srcs, nRanks), ret, fail);
  // 分配目标地址数组
  NCCLCHECKGOTO(ncclCalloc(&params->dsts, nRanks), ret, fail);
  // 分配大小数组
  NCCLCHECKGOTO(ncclCalloc(&params->sizes, nRanks), ret, fail);
#if CUDART_VERSION >= 12080
  // CUDA 12.8+ 分配属性数组
  NCCLCHECKGOTO(ncclCalloc(&params->attrs, nRanks), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&params->attrIdxs, nRanks), ret, fail);
#endif
exit:
  return ret;
fail:
  goto exit;
}

/**
 * @brief 释放批量操作参数结构
 * @param params 批量操作参数指针
 *
 * 功能说明：
 * - 释放批量操作参数分配的所有内存
 */
void ncclCeFreeBatchOpsParams(struct ncclCeBatchOpsParams* params) {
  if (params->srcs) free(params->srcs);
  if (params->dsts) free(params->dsts);
  if (params->sizes) free(params->sizes);
#if CUDART_VERSION >= 12080
  if (params->attrs) free(params->attrs);
  if (params->attrIdxs) free(params->attrIdxs);
#endif
}

/**
 * @brief 启动批量内存复制操作
 * @param comm NCCL 通信器指针
 * @param params 批量操作参数
 * @param stream CUDA 流
 * @return ncclResult_t 操作结果状态码
 *
 * 功能说明：
 * 1. 检查是否为 CUDA Graph 捕获模式
 * 2. CUDA 12.8+：使用 cudaMemcpyBatchAsync（批量复制 API）
 * 3. 旧版本：使用循环调用 cudaMemcpyAsync
 * 4. 支持批次内同步（根据 intraBatchSyncFreq 和 intraBatchSyncMsgThreshold）
 *
 * 性能优化：
 * - 批量复制减少 API 调用开销
 * - 批次内同步防止流水线停顿
 * - 属性控制复制行为（如源访问顺序、与计算重叠）
 */
ncclResult_t ncclCeLaunchBatchOps(struct ncclComm* comm, struct ncclCeBatchOpsParams* params, cudaStream_t stream) {
  ncclResult_t ret = ncclSuccess;

  // Check if there are any operations to perform
  // 如果没有操作要执行，直接返回
  if (params->numOps == 0) {
    return ncclSuccess;
  }

  // Check if we are in a CUDA graph capture
  // 检查是否正在捕获 CUDA Graph
  bool capturing = ncclCudaGraphValid(comm->planner.capturingGraph);

  int driverVersion;
  // 获取 CUDA 驱动版本
  NCCLCHECKGOTO(ncclCudaDriverVersion(&driverVersion), ret, fail);

  //--------------Graph capture--------------
  // cudaMemcpyBatchAsync is not supported during CUDA graph capture
  // CUDA Graph 捕获模式：批量复制 API 不支持，使用循环调用
  if (capturing) {
    for (int i =0; i < params->numOps; i++) {
      // 异步内存复制（设备到设备）
      CUDACHECKGOTO(cudaMemcpyAsync(
        (void*)params->dsts[i],        // 目标地址
        (void*)params->srcs[i],        // 源地址
        params->sizes[i],              // 复制大小
        cudaMemcpyDeviceToDevice,      // 设备到设备复制
        stream), ret, fail);

      // 如果启用批次内同步，每隔 intraBatchSyncFreq 个操作插入一次同步
      // 条件：(i+1) 是同步频率的倍数 且 不是最后一个操作
      if (params->intraBatchSync && ((i+1) % comm->ceColl.intraBatchSyncFreq == 0) && ((i+1) < params->numOps)) {
        NCCLCHECKGOTO(ncclMemOpSync(comm, stream), ret, fail);
      }
    }
  }
  //--------------No graph capture--------------
  else {
    // 检查 CUDA 版本是否支持批量复制 API（12.8+）
    if (CUDART_VERSION >= 12080 && driverVersion >= 12080) {
#if CUDART_VERSION >= 12080
    // For CUDA 12.8+, use batch memory copy for better performance
    // 设置批量复制属性
    params->attrs[0] = {};
    // 源访问顺序：流顺序（按照流中提交的顺序访问源内存）
    params->attrs[0].srcAccessOrder = cudaMemcpySrcAccessOrderStream;
    // 标志：优先与计算重叠（提高 GPU 利用率）
    params->attrs[0].flags = cudaMemcpyFlagPreferOverlapWithCompute;
    // 属性索引：0（使用第一个属性）
    params->attrIdxs[0] = 0;
    // 属性数量：1
    params->numAttrs = 1;

    if (params->intraBatchSync) {
      // Break into multiple batches with sync between them
      // 分批次执行：将操作分成多个小批次，批次之间插入同步
      int batchSize = comm->ceColl.intraBatchSyncFreq;  // 每批次操作数量
      for (int i = 0; i < params->numOps; i += batchSize) {
        // 计算当前批次的大小（最后一批可能小于 batchSize）
        int currentBatchSize = (i + batchSize <= params->numOps) ? batchSize : params->numOps - i;

        #if CUDART_VERSION >= 13000
        // CUDA 13.0+ 版本的批量复制 API（不需要额外的 context 参数）
        CUDACHECKGOTO(cudaMemcpyBatchAsync(
          &params->dsts[i], &params->srcs[i], &params->sizes[i], currentBatchSize,
          params->attrs, params->attrIdxs, params->numAttrs, stream), ret, fail);
        #else
        // CUDA 12.8 版本的批量复制 API（context 参数为 nullptr）
        CUDACHECKGOTO(cudaMemcpyBatchAsync(
          &params->dsts[i], &params->srcs[i], &params->sizes[i], currentBatchSize,
          params->attrs, params->attrIdxs, params->numAttrs, nullptr, stream), ret, fail);
        #endif

        // Sync after each batch
        // 每批次后插入同步（除了最后一批）
        if (i + batchSize < params->numOps) {
          NCCLCHECKGOTO(ncclMemOpSync(comm, stream), ret, fail);
        }
      }
    } else {
      // Use single batch for all operations
      // 不使用批次内同步：一次性执行所有操作
      #if CUDART_VERSION >= 13000
      CUDACHECKGOTO(cudaMemcpyBatchAsync(
        params->dsts, params->srcs, params->sizes, params->numOps,
        params->attrs, params->attrIdxs, params->numAttrs, stream), ret, fail);
      #else
      CUDACHECKGOTO(cudaMemcpyBatchAsync(
        params->dsts, params->srcs, params->sizes, params->numOps,
        params->attrs, params->attrIdxs, params->numAttrs, nullptr, stream), ret, fail);
      #endif
    }
#endif
    } else {
      // For older CUDA versions, fall back to individual transfers
      // 旧版 CUDA：使用循环调用 cudaMemcpyAsync
      for (int i = 0; i < params->numOps; i++) {
        CUDACHECKGOTO(cudaMemcpyAsync(
          (void*)params->dsts[i],
          (void*)params->srcs[i],
          params->sizes[i],
          cudaMemcpyDeviceToDevice,
          stream), ret, fail);

        // 批次内同步
        if (params->intraBatchSync && ((i+1) % comm->ceColl.intraBatchSyncFreq == 0) && ((i+1) < params->numOps)) {
          NCCLCHECKGOTO(ncclMemOpSync(comm, stream), ret, fail);
        }
      }
    }
  }

exit:
  return ret;
fail:
  goto exit;
}


/**
 * @brief CE 实现的 AllGather 操作
 * @param comm NCCL 通信器指针
 * @param args CE 集合操作参数
 * @param stream CUDA 流
 * @return ncclResult_t 操作结果状态码
 *
 * 功能说明：
 * AllGather 操作：每个 rank 的数据被收集到所有 rank
 *
 * 示例（3 个 rank，每个 rank 有 4 个元素）：
 * 输入：
 *   Rank 0: [0, 1, 2, 3]
 *   Rank 1: [4, 5, 6, 7]
 *   Rank 2: [8, 9, 10, 11]
 *
 * 输出（每个 rank 都有相同的结果）：
 *   [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
 *
 * 实现方式：
 * 1. 等待所有 rank 准备好
 * 2. 如果是非原位操作，复制自己的数据
 * 3. 将自己的数据发送到所有其他 rank
 * 4. 等待所有传输完成
 */
ncclResult_t ncclCeAllGather(struct ncclComm* comm, struct ncclCeCollArgs* args, cudaStream_t stream) {
  ncclResult_t ret = ncclSuccess;

  // Calculate the size of each rank's data chunk
  // 每个 rank 的数据块大小（字节数）
  const size_t chunkBytes = args->nElts * args->eltSize;
  // 发送缓冲区指针（当前 rank 的数据）
  uint8_t* mySendBuff = (uint8_t*)args->sendBuff;
  // 接收缓冲区指针（当前 rank 在最终输出中的位置）
  // 输出布局：[Rank0数据][Rank1数据]...[RankN-1数据]
  uint8_t* myRecvBuff = (uint8_t*)args->recvBuff + comm->rank * chunkBytes;
  // 接收缓冲区指针（用于远程访问）
  void* peerRecvBuff;
  // 偏移量（用于计算 LSA 地址）
  size_t offset;

  // 初始化批量操作参数
  struct ncclCeBatchOpsParams batchOpsParams = {};
  NCCLCHECKGOTO(ncclCeInitBatchOpsParams(&batchOpsParams, comm->nRanks), ret, fail);

  // Ensure all ranks are ready before starting transfers
  // 等待所有 rank 准备好开始传输
  NCCLCHECKGOTO(ncclMemOpSync(comm, stream), ret, fail);

  // Copy own data to receive buffer if operation is out-of-place
  // 如果是非原位操作（发送和接收缓冲区不同），需要复制自己的数据
  if (myRecvBuff != mySendBuff) {
    batchOpsParams.srcs[batchOpsParams.numOps] = (void*)mySendBuff;
    batchOpsParams.dsts[batchOpsParams.numOps] = (void*)myRecvBuff;
    batchOpsParams.sizes[batchOpsParams.numOps] = chunkBytes;
    batchOpsParams.numOps++;
  }

  // Copy data to other ranks
  // 将当前 rank 的数据复制到所有其他 rank 的接收缓冲区
  // r 从 1 开始，因为 r=0 是自己（已经在上面处理）
  for (int r = 1; r < comm->nRanks; r++) {
    // 计算目标 rank（环形顺序）
    // 例如：rank 0 的目标顺序是 rank 1, rank 2, ..., rank N-1
    int targetRank = (comm->rank + r) % comm->nRanks;
    // 计算接收缓冲区相对于窗口的偏移量
    offset = myRecvBuff - (uint8_t*)args->recvWin->userPtr;
    // 获取目标 rank 的 LSA 指针（用于远程写入）
    // targetRank 将在其接收缓冲区的 comm->rank * chunkBytes 位置接收数据
    NCCLCHECKGOTO(ncclDevrGetLsaRankPtr(comm, args->recvWin, offset, targetRank, &peerRecvBuff), ret, fail);
    // 添加复制操作：当前 rank 的数据 -> 目标 rank 的接收缓冲区
    batchOpsParams.srcs[batchOpsParams.numOps] = (void*)mySendBuff;
    batchOpsParams.dsts[batchOpsParams.numOps] = (void*)peerRecvBuff;
    batchOpsParams.sizes[batchOpsParams.numOps] = chunkBytes;
    batchOpsParams.numOps++;
  }

  // Check if we need to perform intra-batch synchronization
  // 检查是否需要批次内同步
  // 条件：操作数量超过阈值 且 总数据量超过阈值
  batchOpsParams.intraBatchSync = (batchOpsParams.numOps > comm->ceColl.intraBatchSyncFreq && chunkBytes*batchOpsParams.numOps >= comm->ceColl.intraBatchSyncMsgThreshold);

  // Launch the batch operations
  // 执行批量复制操作
  NCCLCHECKGOTO(ncclCeLaunchBatchOps(comm, &batchOpsParams, stream), ret, fail);

  // Ensure all transfers are complete across all ranks
  // 等待所有传输完成
  NCCLCHECKGOTO(ncclMemOpSync(comm, stream), ret, fail);

exit:
  // 释放批量操作参数
  ncclCeFreeBatchOpsParams(&batchOpsParams);
  return ret;
fail:
  goto exit;
}

/**
 * @brief CE 实现的 AlltoAll 操作
 * @param comm NCCL 通信器指针
 * @param args CE 集合操作参数
 * @param stream CUDA 流
 * @return ncclResult_t 操作结果状态码
 *
 * 功能说明：
 * AlltoAll 操作：每个 rank 向所有其他 rank 发送不同的数据
 *
 * 示例（2 个 rank，每个 rank 发送 4 个元素）：
 * 输入：
 *   Rank 0: [0, 1, 2, 3]  (前2个给rank0，后2个给rank1)
 *   Rank 1: [4, 5, 6, 7]  (前2个给rank0，后2个给rank1)
 *
 * 输出：
 *   Rank 0: [0, 1, 4, 5]  (接收来自 rank0 的 [0,1] 和 rank1 的 [4,5])
 *   Rank 1: [2, 3, 6, 7]  (接收来自 rank0 的 [2,3] 和 rank1 的 [6,7])
 *
 * 内存布局：
 * 发送缓冲区：每个 rank 的数据按目标 rank 排列
 *   [数据给Rank0][数据给Rank1]...[数据给RankN-1]
 * 接收缓冲区：每个 rank 的数据按源 rank 排列
 *   [来自Rank0的数据][来自Rank1的数据]...[来自RankN-1的数据]
 */
ncclResult_t ncclCeAlltoAll(struct ncclComm* comm, struct ncclCeCollArgs* args, cudaStream_t stream) {
  ncclResult_t ret = ncclSuccess;

  // Calculate the size of data each rank sends to every other rank
  // 每个 rank 向每个其他 rank 发送的数据块大小
  const size_t chunkBytes = args->nElts * args->eltSize;
  // 发送缓冲区指针
  uint8_t* mySendBuff = (uint8_t*)args->sendBuff;
  // 接收缓冲区指针
  uint8_t* myRecvBuff = (uint8_t*)args->recvBuff;
  // 接收缓冲区指针（用于远程访问）
  void* peerRecvBuff;
  // 偏移量
  size_t offset;

  // 初始化批量操作参数
  // 需要 nRanks * nRanks 个操作（每个 rank 向每个其他 rank 发送数据）
  struct ncclCeBatchOpsParams batchOpsParams = {};
  NCCLCHECKGOTO(ncclCeInitBatchOpsParams(&batchOpsParams, comm->nRanks * comm->nRanks), ret, fail);

  // Ensure all ranks are ready before starting transfers
  NCCLCHECKGOTO(ncclMemOpSync(comm, stream), ret, fail);

  // Copy data to other ranks: send data chunk for each destination rank
  // 遍历所有目标 rank
  for (int r = 0; r < comm->nRanks; r++) {
    // 计算目标 rank（环形顺序）
    int dstRank = (comm->rank + r) % comm->nRanks;
    // 源指针：发送缓冲区中目标 rank 的数据
    // 布局：[给Rank0][给Rank1]...[给RankN-1]
    uint8_t* srcPtr = mySendBuff + dstRank * chunkBytes;
    // 目标指针：接收缓冲区中当前 rank 应该放入的位置
    // 布局：[来自Rank0][来自Rank1]...[来自RankN-1]
    uint8_t* dstPtr = myRecvBuff + comm->rank * chunkBytes;

    if (dstRank == comm->rank) {
      // Local copy for own data
      // 本地复制：发送给自己的数据
      batchOpsParams.srcs[batchOpsParams.numOps] = (void*)srcPtr;
      batchOpsParams.dsts[batchOpsParams.numOps] = (void*)dstPtr;
      batchOpsParams.sizes[batchOpsParams.numOps] = chunkBytes;
      batchOpsParams.numOps++;
    } else {
      // Remote copy to other ranks: send to rank dstRank's receive buffer at position comm->rank
      // 远程复制：发送给其他 rank
      // 目标：dstRank 的接收缓冲区中 comm->rank 的位置
      offset = dstPtr - (uint8_t*)args->recvWin->userPtr;
      NCCLCHECKGOTO(ncclDevrGetLsaRankPtr(comm, args->recvWin, offset, dstRank, &peerRecvBuff), ret, fail);
      batchOpsParams.srcs[batchOpsParams.numOps] = (void*)srcPtr;
      batchOpsParams.dsts[batchOpsParams.numOps] = (void*)peerRecvBuff;
      batchOpsParams.sizes[batchOpsParams.numOps] = chunkBytes;
      batchOpsParams.numOps++;
    }
  }

  // Check if we need to perform intra-batch synchronization
  batchOpsParams.intraBatchSync = (batchOpsParams.numOps > comm->ceColl.intraBatchSyncFreq && chunkBytes*batchOpsParams.numOps >= comm->ceColl.intraBatchSyncMsgThreshold);

  // Launch the batch operations
  NCCLCHECKGOTO(ncclCeLaunchBatchOps(comm, &batchOpsParams, stream), ret, fail);

  // Ensure all transfers are complete across all ranks
  NCCLCHECKGOTO(ncclMemOpSync(comm, stream), ret, fail);

exit:
  ncclCeFreeBatchOpsParams(&batchOpsParams);
  return ret;
fail:
  goto exit;
}

/**
 * @brief CE 实现的 Scatter 操作
 * @param comm NCCL 通信器指针
 * @param args CE 集合操作参数
 * @param stream CUDA 流
 * @return ncclResult_t 操作结果状态码
 *
 * 功能说明：
 * Scatter 操作：root rank 将数据分发到所有 rank
 *
 * 示例（3 个 rank，root = 0）：
 * 输入（Rank 0）：
 *   [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
 *   (每个 rank 分配 4 个元素)
 *
 * 输出：
 *   Rank 0: [0, 1, 2, 3]
 *   Rank 1: [4, 5, 6, 7]
 *   Rank 2: [8, 9, 10, 11]
 *
 * 内存布局：
 * 发送缓冲区（root）：[Rank0数据][Rank1数据]...[RankN-1数据]
 * 接收缓冲区（每个 rank）：只接收自己的那部分数据
 */
ncclResult_t ncclCeScatter(struct ncclComm* comm, struct ncclCeCollArgs* args, cudaStream_t stream) {
  ncclResult_t ret = ncclSuccess;

  // Calculate the size of data root sends to each rank
  // root 发送给每个 rank 的数据块大小
  const size_t chunkBytes = args->nElts * args->eltSize;
  // 发送缓冲区指针
  uint8_t* mySendBuff = (uint8_t*)args->sendBuff;
  // 接收缓冲区指针
  uint8_t* myRecvBuff = (uint8_t*)args->recvBuff;
  // root rank 索引
  int rootRank = args->rootRank;
  // 目标指针（用于远程访问）
  void* peerDstPtr;
  // 偏移量
  size_t offset;

  // 初始化批量操作参数
  struct ncclCeBatchOpsParams batchOpsParams = {};
  NCCLCHECKGOTO(ncclCeInitBatchOpsParams(&batchOpsParams, comm->nRanks), ret, fail);

  // Ensure all ranks are ready before starting transfers
  NCCLCHECKGOTO(ncclMemOpSync(comm, stream), ret, fail);

  if (comm->rank == rootRank) {
    // Check if this is an in-place scatter operation
    // 检查是否为原位操作
    // 原位条件：接收缓冲区 == 发送缓冲区中 root rank 数据的起始位置
    bool isInPlace = (myRecvBuff == mySendBuff + comm->rank * chunkBytes);

    // Copy root's own data first if not in-place
    // 如果不是原位操作，root rank 需要复制自己的数据
    if (!isInPlace) {
      uint8_t* srcPtr = mySendBuff + comm->rank * chunkBytes;  // root rank 在发送缓冲区中的数据
      uint8_t* dstPtr = myRecvBuff;                            // root rank 的接收缓冲区
      batchOpsParams.srcs[batchOpsParams.numOps] = (void*)srcPtr;
      batchOpsParams.dsts[batchOpsParams.numOps] = (void*)dstPtr;
      batchOpsParams.sizes[batchOpsParams.numOps] = chunkBytes;
      batchOpsParams.numOps++;
    }

    // Root rank distributes data to other ranks
    // root rank 向其他 rank 分发数据
    for (int r = 1; r < comm->nRanks; r++) {
      // 计算目标 rank（环形顺序）
      int dstRank = (comm->rank + r) % comm->nRanks;
      // 源指针：发送缓冲区中目标 rank 的数据
      uint8_t* srcPtr = mySendBuff + dstRank * chunkBytes;
      // 目标指针：
      // - 原位操作：目标数据在发送缓冲区的位置
      // - 非原位操作：目标 rank 的接收缓冲区
      uint8_t* dstPtr = isInPlace ? myRecvBuff + dstRank * chunkBytes : myRecvBuff;

      // 计算目标 rank 的 LSA 指针
      offset = dstPtr - (uint8_t*)args->recvWin->userPtr;
      NCCLCHECKGOTO(ncclDevrGetLsaRankPtr(comm, args->recvWin, offset, dstRank, &peerDstPtr), ret, fail);
      // 添加复制操作
      batchOpsParams.srcs[batchOpsParams.numOps] = (void*)srcPtr;
      batchOpsParams.dsts[batchOpsParams.numOps] = (void*)peerDstPtr;
      batchOpsParams.sizes[batchOpsParams.numOps] = chunkBytes;
      batchOpsParams.numOps++;
    }
  }
  // Non-root ranks don't need to perform any copy operations
  // 非 root rank 不需要执行任何复制操作（只等待接收）

  // Launch the batch operations
  NCCLCHECKGOTO(ncclCeLaunchBatchOps(comm, &batchOpsParams, stream), ret, fail);

  // Ensure all transfers are complete across all ranks
  NCCLCHECKGOTO(ncclMemOpSync(comm, stream), ret, fail);

exit:
  ncclCeFreeBatchOpsParams(&batchOpsParams);
  return ret;
fail:
  goto exit;
}

/**
 * @brief CE 实现的 Gather 操作
 * @param comm NCCL 通信器指针
 * @param args CE 集合操作参数
 * @param stream CUDA 流
 * @return ncclResult_t 操作结果状态码
 *
 * 功能说明：
 * Gather 操作：所有 rank 的数据收集到 root rank
 *
 * 示例（3 个 rank，root = 0）：
 * 输入：
 *   Rank 0: [0, 1, 2, 3]
 *   Rank 1: [4, 5, 6, 7]
 *   Rank 2: [8, 9, 10, 11]
 *
 * 输出（Rank 0）：
 *   [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
 *   (按 rank 顺序排列)
 *
 * 内存布局：
 * 发送缓冲区（每个 rank）：只发送自己的数据
 * 接收缓冲区（root）：[Rank0数据][Rank1数据]...[RankN-1数据]
 */
ncclResult_t ncclCeGather(struct ncclComm* comm, struct ncclCeCollArgs* args, cudaStream_t stream) {
  ncclResult_t ret = ncclSuccess;

  // Calculate the size of data each rank sends to root
  // 每个 rank 发送给 root 的数据块大小
  const size_t chunkBytes = args->nElts * args->eltSize;
  // 发送缓冲区指针
  uint8_t* mySendBuff = (uint8_t*)args->sendBuff;
  // 接收缓冲区指针
  uint8_t* myRecvBuff = (uint8_t*)args->recvBuff;
  // root rank 索引
  int rootRank = args->rootRank;
  // 远程接收缓冲区指针（用于远程写入）
  void* peerRecvBuff;
  // 偏移量
  size_t offset;

  // 初始化批量操作参数
  // 只需要 1 个操作（每个 rank 只发送一次数据）
  struct ncclCeBatchOpsParams batchOpsParams = {};
  NCCLCHECKGOTO(ncclCeInitBatchOpsParams(&batchOpsParams, 1), ret, fail);

  // Ensure all ranks are ready before starting transfers
  NCCLCHECKGOTO(ncclMemOpSync(comm, stream), ret, fail);

  if (comm->rank == rootRank) {
    // Root rank copies its own data to the correct position in receive buffer
    // root rank 将自己的数据复制到接收缓冲区的正确位置
    uint8_t* dstPtr = myRecvBuff + comm->rank * chunkBytes;
    // 如果发送缓冲区和目标位置不同，需要复制
    if (mySendBuff != dstPtr) {
      batchOpsParams.srcs[batchOpsParams.numOps] = (void*)mySendBuff;
      batchOpsParams.dsts[batchOpsParams.numOps] = (void*)dstPtr;
      batchOpsParams.sizes[batchOpsParams.numOps] = chunkBytes;
      batchOpsParams.numOps++;
    }
  } else {
    // Non-root ranks send their data to root's receive buffer
    // 非 root rank 将数据发送到 root 的接收缓冲区
    // 计算 root 接收缓冲区中当前 rank 数据应该放置的位置
    uint8_t* rootRecvPtr = (uint8_t*)args->recvBuff + comm->rank * chunkBytes;
    // 计算偏移量
    offset = rootRecvPtr - (uint8_t*)args->recvWin->userPtr;
    // 获取 root 的 LSA 指针（远程写入目标）
    NCCLCHECKGOTO(ncclDevrGetLsaRankPtr(comm, args->recvWin, offset, rootRank, &peerRecvBuff), ret, fail);
    // 添加复制操作：当前 rank 的数据 -> root 的接收缓冲区
    batchOpsParams.srcs[batchOpsParams.numOps] = (void*)mySendBuff;
    batchOpsParams.dsts[batchOpsParams.numOps] = (void*)peerRecvBuff;
    batchOpsParams.sizes[batchOpsParams.numOps] = chunkBytes;
    batchOpsParams.numOps++;
  }

  // Launch the batch operations
  NCCLCHECKGOTO(ncclCeLaunchBatchOps(comm, &batchOpsParams, stream), ret, fail);

  // Ensure all transfers are complete across all ranks
  NCCLCHECKGOTO(ncclMemOpSync(comm, stream), ret, fail);

exit:
  ncclCeFreeBatchOpsParams(&batchOpsParams);
  return ret;
fail:
  goto exit;
}

/**
 * @brief 启动 CE 集合操作
 * @param comm NCCL 通信器指针
 * @param plan 内核计划（包含 CE 集合操作参数）
 * @return ncclResult_t 操作结果状态码
 *
 * 功能说明：
 * 根据 CE 集合操作类型调度到具体的实现函数
 *
 * 支持的操作：
 * - AllGather: 收集所有 rank 的数据到每个 rank
 * - AlltoAll: 每个 rank 向所有其他 rank 发送不同的数据
 * - Scatter: root rank 将数据分发到所有 rank
 * - Gather: 收集所有 rank 的数据到 root rank
 */
ncclResult_t ncclLaunchCeColl(struct ncclComm* comm, struct ncclKernelPlan* plan) {
  ncclResult_t ret = ncclSuccess;
  // 获取 CUDA 流
  cudaStream_t stream = comm->planner.streams->stream;
  // 获取 CE 集合操作参数
  struct ncclCeCollArgs* args = plan->ceCollArgs;

  // 根据操作类型分发到具体实现
  switch (args->func) {
    case ncclFuncAllGather:
      // AllGather 操作
      NCCLCHECKGOTO(ncclCeAllGather(comm, args, stream), ret, fail);
      break;
    case ncclFuncAlltoAll:
      // AlltoAll 操作
      NCCLCHECKGOTO(ncclCeAlltoAll(comm, args, stream), ret, fail);
      break;
    case ncclFuncScatter:
      // Scatter 操作
      NCCLCHECKGOTO(ncclCeScatter(comm, args, stream), ret, fail);
      break;
    case ncclFuncGather:
      // Gather 操作
      NCCLCHECKGOTO(ncclCeGather(comm, args, stream), ret, fail);
      break;
    default:
      // 不支持的操作类型
      ret = ncclInvalidUsage;
  }

exit:
  return ret;
fail:
  goto exit;
}
