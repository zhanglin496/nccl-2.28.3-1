/*************************************************************************
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// 设备端运行时实现头文件
// 包含设备端运行时的核心定义和接口
#include "dev_runtime.h"
// 通信器头文件，包含 ncclComm 结构体定义
#include "comm.h"
// 设备端相关操作和内核定义
#include "device.h"
// 传输层接口定义（网络、共享内存等）
#include "transport.h"
// 组操作相关定义（用于协调多个集合通信操作）
#include "group.h"
// NCCL 设备端公共定义
#include "nccl_device.h"

// 窗口步长参数：定义对称地址空间中每个窗口的跨度
// -1 表示自动选择（默认为最大 GPU 内存）
NCCL_PARAM(WinStride, "WIN_STRIDE", -1);

// 从 src/include/dev_runtime.h 完整的类型定义
// 设备端运行时内存结构：表示一个对称内存分配
struct ncclDevrMemory {
  int refCount;                                    // 引用计数，用于跟踪有多少窗口使用此内存
  struct ncclDevrMemory* next;                     // 链表下一个节点，用于管理所有内存分配
  CUmemGenericAllocationHandle memHandle;          // CUDA 内存分配句柄，用于跨进程共享
  size_t size;                                     // 内存分配大小（字节）
  size_t bigOffset;                                // 在大地址空间（big VA space）中的偏移量
};

// 排序后的窗口结构：用于快速地址查找
// 按用户地址排序的窗口列表，支持二分查找
struct ncclDevrWindowSorted {
  uintptr_t userAddr;                              // 用户空间地址
  size_t size;                                     // 窗口大小
  struct ncclDevrWindow* win;                      // 指向实际的窗口结构
};

// 设备端运行时团队结构：表示一个通信团队
// 团队是参与对称内存操作的 rank 子集
struct ncclDevrTeam {
  struct ncclDevrTeam* next;                       // 链表下一个节点
  struct ncclTeam team;                            // 团队描述信息（rank数、stride、rank索引等）
  CUmemGenericAllocationHandle mcHandle;           // 多播（multicast）内存句柄（用于 NVLS）
  void* mcBasePtr;                                 // 多播内存基地址指针
  int worldRankList[];                             // 灵活数组成员，存储团队中各成员的世界 rank 列表
};

////////////////////////////////////////////////////////////////////////////////
// 辅助函数前向声明（实现在文件底部）:

// 查找最小索引使得 `arg < sorted[i].key`（最小上界，Least Upper Bound）
// 这是一个二分查找函数，用于在已排序数组中快速定位插入位置
template<typename Obj, typename Key>
static int listFindSortedLub(Key Obj::*key, Obj* sorted, int count, Key arg);

// 向动态数组中插入元素
// 如果容量不足会自动扩容（2倍增长策略）
template<typename Obj>
static void listInsert(Obj** list, int* capacity, int* count, int index, Obj val);

// 从动态数组中移除指定索引的元素
// 移除后会将后续元素前移
template<typename Obj>
static void listRemove(Obj* list, int* count, int index);

////////////////////////////////////////////////////////////////////////////////

// 设备端运行时一次性初始化函数
// 功能：初始化对称内存管理所需的数据结构和资源
// 参数：comm - NCCL 通信器
// 返回：ncclResult_t - 操作结果状态码
ncclResult_t ncclDevrInitOnce(struct ncclComm* comm) {
  ncclResult_t ret = ncclSuccess;                      // 初始化返回值为成功
  struct ncclDevrState* devr = &comm->devrState;       // 获取设备端运行时状态指针
  if (devr->bigSize != 0) 
    return ncclSuccess;          // 如果已经初始化过，直接返回（幂等操作）

  // 检查本地 rank 是否连续（用于确定 LSA 团队）
  // LSA（Local Symmetric Area）是对称内存的本地访问区域
  bool lsaIsLocal = true;                              // 假设本地 rank 连续
  for (int i=0; i < comm->localRanks; i++) {
    // 检查相邻的 local rank 是否对应相邻的 world rank
    // 如果所有 local rank 的 world rank 连续，则可以形成对称视图
    lsaIsLocal &= comm->localRankToRank[i] == comm->localRankToRank[0] + i;
  }
  
  // 设置 LSA 团队的本地 rank 索引
  devr->lsaSelf = lsaIsLocal ? comm->localRank : 0;   // 如果连续则使用本地 rank，否则使用 0
  // 设置 LSA 团队的大小
  devr->lsaSize = lsaIsLocal ? comm->localRanks : 1;   // 如果连续则使用本地 rank 数，否则只有 1 个
  // 分配 LSA rank 列表内存
  devr->lsaRankList = (int*)malloc(devr->lsaSize*sizeof(int));
  // 构建 LSA rank 列表：存储每个 LSA rank 对应的世界 rank
  for (int i=0; i < devr->lsaSize; i++) {
    // 计算当前 rank 在 LSA 团队中的偏移，然后转换为世界 rank
    devr->lsaRankList[i] = comm->rank + (i - devr->lsaSelf);
  }

  // 配置 CUDA 内存分配属性
  CUmemAllocationProp memProp = {};                    // 初始化内存属性结构体
  memProp.type = CU_MEM_ALLOCATION_TYPE_PINNED;        // 使用固定内存类型（页锁定）
  memProp.location.type = CU_MEM_LOCATION_TYPE_DEVICE; // 分配在设备内存
  memProp.requestedHandleTypes = ncclCuMemHandleType;  // 设置句柄类型（POSIX fd 或 Fabric handle）
  memProp.location.id = comm->cudaDev;                 // 指定 GPU 设备 ID
  // 获取内存分配粒度（用于对齐，提高性能）
  CUCHECKGOTO(cuMemGetAllocationGranularity(&devr->granularity, &memProp, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED), ret, fail_lsaRankList);

  // 确定对称地址空间大小（bigSize）
  devr->bigSize = ncclParamWinStride();                // 从环境变量或默认值获取窗口步长
  if (-devr->bigSize <= 1) {                           // 如果值为负数或 -1（自动模式）
    devr->bigSize = 1;                                 // 从最小值开始
    // 遍历所有 rank，找到最大的 GPU 内存容量
    for (int r=0; r < comm->nRanks; ++r) {
      devr->bigSize = std::max<size_t>(devr->bigSize, comm->peerInfo[r].totalGlobalMem);
    }
  }
  // 将 bigSize 向上对齐到 4GB 边界（2^32 字节）
  // 这样可以确保每个 rank 的对称地址空间在 4GB 对齐的边界上
  devr->bigSize = alignUp(devr->bigSize, size_t(1)<<32);
  INFO(NCCL_INIT, "Symmetric VA size=%ldGB", (long)devr->bigSize>>30); // 输出对称地址空间大小（GB）

  // 构造大地址空间管理器（用于分配偏移量）
  ncclSpaceConstruct(&devr->bigSpace);
  // 构造影子内存池（用于主机和设备间数据结构同步）
  ncclShadowPoolConstruct(&devr->shadows);
  return ncclSuccess;                                  // 初始化成功

fail_lsaRankList:                                      // 错误处理：释放 LSA rank 列表
  free(devr->lsaRankList);
  return ret;                                          // 返回错误码
}

// 前向声明：销毁所有对称团队（实现在文件后面）
static void symTeamDestroyAll(struct ncclComm* comm); // Further down

// 设备端运行时清理函数
// 功能：释放对称内存管理相关的所有资源
// 参数：comm - NCCL 通信器
// 返回：ncclResult_t - 操作结果状态码
ncclResult_t ncclDevrFinalize(struct ncclComm* comm) {
  struct ncclDevrState* devr = &comm->devrState;       // 获取设备端运行时状态指针
  if (devr->bigSize == 0) return ncclSuccess;          // 如果未初始化，直接返回

  // 清空注册任务队列（释放所有待处理的窗口注册任务）
  while (!ncclIntruQueueEmpty(&devr->regTaskQueue)) {
    struct ncclDevrRegTask* task = ncclIntruQueueDequeue(&devr->regTaskQueue);
    free(task);                                        // 释放任务结构体内存
  }

  // 销毁所有对称团队（包括多播资源）
  symTeamDestroyAll(comm);

  { // 删除窗口表（windowTable）
    cudaStream_t stream;                               // 创建 CUDA 流用于异步操作
    if (cudaSuccess == cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking)) {
      struct ncclDevCommWindowTable* tableDev = devr->windowTable; // 获取设备端窗口表
      while (tableDev != nullptr) {                    // 遍历窗口表链表
        struct ncclDevCommWindowTable* tableHost;      // 主机端窗口表
        // 将设备端窗口表同步到主机端
        if (ncclSuccess != ncclShadowPoolToHost(&devr->shadows, tableDev, &tableHost)) break;
        struct ncclDevCommWindowTable* next = tableHost->next; // 保存下一个节点
        // 释放设备端窗口表内存
        ncclShadowPoolFree(&devr->shadows, tableDev, stream);
        tableDev = next;                               // 继续处理下一个节点
      }
      cudaStreamSynchronize(stream);                   // 等待所有异步操作完成
      cudaStreamDestroy(stream);                       // 销毁 CUDA 流
    }
  }
  // 解除映射 LSA 平面地址空间
  CUdeviceptr flatAddr = reinterpret_cast<CUdeviceptr>(devr->lsaFlatBase);
  CUCHECKIGNORE(cuMemUnmap(flatAddr, devr->lsaSize*devr->bigSize)); // 取消内存映射
  CUCHECKIGNORE(cuMemAddressFree(flatAddr, devr->lsaSize*devr->bigSize)); // 释放地址空间预留
  // 销毁影子内存池
  ncclShadowPoolDestruct(&devr->shadows);
  // 销毁大地址空间管理器
  ncclSpaceDestruct(&devr->bigSpace);
  // 释放 LSA rank 列表
  free(devr->lsaRankList);
  // 释放排序后的窗口列表
  free(devr->winSorted);
  return ncclSuccess;                                  // 清理完成
}

////////////////////////////////////////////////////////////////////////////////

// 将内存映射到 LSA（本地对称区域）团队
// 功能：将一个内存句柄映射到所有 LSA rank 的对称地址空间中
// 参数：
//   comm      - NCCL 通信器
//   memHandle - CUDA 内存分配句柄
//   size      - 要映射的内存大小
//   bigOffset - 在大地址空间中的偏移量
// 返回：ncclResult_t - 操作结果状态码
static ncclResult_t symMemoryMapLsaTeam(
    struct ncclComm* comm, CUmemGenericAllocationHandle memHandle, size_t size, size_t bigOffset
  ) {
  ncclResult_t ret = ncclSuccess;                      // 初始化返回值为成功
  struct ncclDevrState* devr = &comm->devrState;       // 获取设备端运行时状态
  CUmemAccessDesc accessDesc = {};                     // 内存访问描述符（用于设置访问权限）
  union Message {                                      // 消息联合体，用于交换内存句柄
    CUmemGenericAllocationHandle memHandle;            // POSIX 文件描述符类型的句柄
    CUmemFabricHandle fabricHandle;                    // Fabric 网络句柄类型
  };

  // 分配消息数组，用于 AllGather 交换所有 rank 的内存句柄
  Message* messages = (Message*)calloc(devr->lsaSize, sizeof(Message));
  if (ncclCuMemHandleType == CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) {
    // POSIX 文件描述符类型：直接使用句柄
    messages[devr->lsaSelf].memHandle = memHandle;
  } else {
    // Fabric 句柄类型：需要导出为可共享的句柄
    CUCHECKGOTO(cuMemExportToShareableHandle(&messages[devr->lsaSelf].fabricHandle, memHandle, ncclCuMemHandleType, 0), ret, fail);
  }

  // 在 LSA 团队内收集所有 rank 的内存句柄
  // 每个 rank 都会获得其他所有 rank 的内存句柄
  NCCLCHECKGOTO(bootstrapIntraNodeAllGather(comm->bootstrap, devr->lsaRankList, devr->lsaSelf, devr->lsaSize, messages, sizeof(Message)), ret, fail);

  // 首次需要时创建平面地址空间（Flat VA Space）
  if (devr->lsaFlatBase == nullptr) { // Create on first need.
    CUdeviceptr addr;                                   // 保留的虚拟地址
    // 保留足够的地址空间（lsaSize * bigSize），按页对齐
    CUCHECKGOTO(cuMemAddressReserve(&addr, devr->lsaSize*devr->bigSize, NCCL_MAX_PAGE_SIZE, 0, 0), ret, fail);
    devr->lsaFlatBase = reinterpret_cast<void*>(addr); // 保存平面地址基地址
  }
  // 配置访问权限：本设备可读写
  accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE; // 位置类型：设备内存
  accessDesc.location.id = comm->cudaDev;                 // 设备 ID
  accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;  // 权限：可读可写

  // 遍历 LSA 团队中的所有 rank，映射每个 rank 的内存
  for (int r = 0; r < devr->lsaSize; r++) {
    CUmemGenericAllocationHandle impHandle;             // 导入的内存句柄
    if (r == devr->lsaSelf) {
      // 当前 rank 自己的内存：直接使用原句柄
      impHandle = memHandle;
    } else {
      // 其他 rank 的内存：需要导入共享句柄
      if (ncclCuMemHandleType == CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) {
        // POSIX 文件描述符类型：通过代理获取文件描述符
        int fd = -1;
        // 从代理客户端获取文件描述符（阻塞等待）
        NCCLCHECKGOTO(ncclProxyClientGetFdBlocking(comm, devr->lsaRankList[r], &messages[r], &fd), ret, fail);
        // 从文件描述符导入内存句柄
        CUCHECKGOTO(cuMemImportFromShareableHandle(&impHandle, reinterpret_cast<void*>((uintptr_t)fd), ncclCuMemHandleType), ret, fail);
        // 关闭文件描述符（句柄已导入，不再需要 fd）
        SYSCHECKGOTO(close(fd), "close", ret, fail);
      } else {
        // Fabric 句柄类型：直接从 Fabric 句柄导入
        CUCHECKGOTO(cuMemImportFromShareableHandle(&impHandle, (void*)&messages[r].fabricHandle, ncclCuMemHandleType), ret, fail);
      }
    }
    // 计算目标地址：平面基地址 + rank偏移 + bigOffset
    CUdeviceptr addr = reinterpret_cast<uintptr_t>((char*)devr->lsaFlatBase + r*devr->bigSize + bigOffset);
    // 将内存映射到目标地址
    CUCHECKGOTO(cuMemMap(addr, size, 0, impHandle, 0), ret, fail);
    // 设置内存访问权限
    CUCHECKGOTO(cuMemSetAccess(addr, size, &accessDesc, 1), ret, fail);
    if (r != devr->lsaSelf) {
      // 释放其他 rank 的导入句柄（映射完成后不再需要）
      CUCHECKGOTO(cuMemRelease(impHandle), ret, fail);
    }
  }
  // Ensure everyone has imported my mem handle.
  // 确保所有 rank 都已导入我的内存句柄
  // 这个 barrier 确保在继续执行前，所有 rank 都已完成内存映射
  NCCLCHECKGOTO(bootstrapIntraNodeBarrier(comm->bootstrap, devr->lsaRankList, devr->lsaSelf, devr->lsaSize, 0xbeef), ret, fail);
leave:                                                   // 正常退出标签
  free(messages);                                       // 释放消息数组
  return ret;                                          // 返回结果
fail:                                                    // 错误处理标签
  goto leave;                                          // 跳转到清理代码
}

// 将内存绑定到团队的多播区域
// 功能：将对称内存绑定到 NVLS 多播组，实现一对多的硬件加速传输
// 参数：
//   comm - NCCL 通信器
//   tm   - 团队结构（包含多播句柄）
//   mem  - 要绑定的内存
// 返回：ncclResult_t - 操作结果状态码
static ncclResult_t symBindTeamMemory(
    struct ncclComm* comm, struct ncclDevrTeam* tm, struct ncclDevrMemory* mem
  ) {
  // 只有在支持 NVLS 且团队已初始化多播时才执行绑定
  if (comm->nvlsSupport && tm->mcBasePtr != nullptr) {
  #if CUDART_VERSION >= 12010                      // CUDA 12.1+ 才支持多播 API
    INFO(NCCL_NVLS, "Binding multicast memory at big=%lx to team {%d x %d}", mem->bigOffset, tm->team.nRanks, tm->team.stride);
    // 将内存绑定到多播句柄，允许硬件多播传输
    // 参数：mcHandle=多播句柄, bigOffset=在大地址空间的偏移, memHandle=内存句柄, size=大小
    CUCHECK(cuMulticastBindMem(tm->mcHandle, mem->bigOffset, mem->memHandle, 0, mem->size, 0));
  #endif
  }
  return ncclSuccess;
}

// 将内存从团队的多播区域解绑
// 功能：解除内存与 NVLS 多播组的绑定
// 参数：
//   comm - NCCL 通信器
//   tm   - 团队结构（包含多播句柄）
//   mem  - 要解绑的内存
// 返回：ncclResult_t - 操作结果状态码
static ncclResult_t symUnbindTeamMemory(
    struct ncclComm* comm, struct ncclDevrTeam* tm, struct ncclDevrMemory* mem
  ) {
  // 只有在支持 NVLS 且团队已初始化多播时才执行解绑
  if (comm->nvlsSupport && tm->mcBasePtr != nullptr) {
  #if CUDART_VERSION >= 12010                      // CUDA 12.1+ 才支持多播 API
    // 从多播句柄解绑内存
    // 参数：mcHandle=多播句柄, cudaDev=设备ID, bigOffset=偏移, size=大小
    CUCHECK(cuMulticastUnbind(tm->mcHandle, comm->cudaDev, mem->bigOffset, mem->size));
  #endif
  }
  return ncclSuccess;
}

// 获取或创建一个团队
// 注意：调用者需要在返回后对团队执行 barrier 操作
// 参数：
//   comm     - NCCL 通信器
//   team     - 团队描述（rank数、stride、rank索引）
//   multimem - 是否启用多播支持（NVLS）
//   outTeam  - 输出：返回的团队指针
// 返回：ncclResult_t - 操作结果状态码
static ncclResult_t symTeamObtain(
    struct ncclComm* comm, struct ncclTeam team, bool multimem,
    struct ncclDevrTeam** outTeam
  ) {
  ncclResult_t ret = ncclSuccess;                      // 初始化返回值为成功
  struct ncclDevrState* devr = &comm->devrState;       // 获取设备端运行时状态
  struct ncclDevrTeam* t = devr->teamHead;             // 从团队链表头开始查找
  bool teamIsNew = false;                              // 标记团队是否为新创建

  // 遍历团队链表，查找匹配的团队或创建新团队
  while (true) {
    if (t == nullptr) {                                // 链表末尾，未找到匹配的团队
      teamIsNew = true;                                // 标记为新团队
      // 分配团队结构体（包含灵活数组成员 worldRankList）
      t = (struct ncclDevrTeam*)malloc(sizeof(struct ncclDevrTeam) + team.nRanks*sizeof(int));
      t->team = team;                                  // 保存团队描述信息
      t->mcHandle = 0x0;                               // 多播句柄初始化为空
      t->mcBasePtr = nullptr;                          // 多播基地址初始化为空
      // 构建世界 rank 列表：计算团队中每个成员对应的世界 rank
      for (int i=0; i < team.nRanks; i++) {
        t->worldRankList[i] = comm->rank + (i - team.rank)*team.stride;
      }
      break;                                           // 创建完成，退出循环
    } else if (t->team.rank == team.rank && t->team.nRanks == team.nRanks && t->team.stride == team.stride) {
      // 找到匹配的团队（rank、nRanks、stride 都相同）
      if (!multimem || t->mcBasePtr != nullptr) {
        // 如果不需要多播，或者多播已初始化，则当前团队满足需求
        // Matching team is sufficient
        if (outTeam) *outTeam = t;                    // 返回找到的团队
        return ncclSuccess;                            // 直接返回成功
      }
      break;                                           // Need to enable multimem（需要启用多播）
    }
    // 继续遍历链表下一个节点
    t = t->next;
  }

  // 如果需要启用多播支持
  if (multimem) {
    if (!comm->nvlsSupport) {                          // 检查系统是否支持 NVLS
      WARN("Multicast support requested for team but none available on system.");
      ret = ncclInvalidArgument;
      goto fail;
    } else {
    #if CUDART_VERSION >= 12010                      // CUDA 12.1+ 才支持多播 API
      CUmemGenericAllocationHandle mcHandle = 0;       // 多播内存句柄
      CUdeviceptr mcAddr = 0;                          // 多播地址
      CUmulticastObjectProp mcProp = {};               // 多播对象属性
      char shareableHandle[NVLS_HANDLE_SIZE];          // 可共享的句柄（用于跨进程）

      // 配置多播属性
      mcProp.numDevices = team.nRanks;                 // 团队中的设备数量
      mcProp.handleTypes = ncclCuMemHandleType;        // 句柄类型
      mcProp.flags = 0;                                // 标志位（保留）
      mcProp.size = devr->bigSize;                     // 多播区域大小（与 bigSize 相同）

      // 创建多播组：rank 0 负责创建，其他 rank 连接
      if (team.rank == 0) {
        // rank 0：创建多播组并获取可共享句柄
        NCCLCHECKGOTO(ncclNvlsGroupCreate(comm, &mcProp, team.rank, team.nRanks, &mcHandle, shareableHandle), ret, fail);
        // 广播可共享句柄给团队中的其他 rank
        NCCLCHECKGOTO(bootstrapIntraNodeBroadcast(comm->bootstrap, t->worldRankList, team.rank, team.nRanks, 0, shareableHandle, NVLS_HANDLE_SIZE), ret, fail_mcHandle);
      } else {
        // 其他 rank：接收可共享句柄并连接到多播组
        NCCLCHECKGOTO(bootstrapIntraNodeBroadcast(comm->bootstrap, t->worldRankList, team.rank, team.nRanks, 0, shareableHandle, NVLS_HANDLE_SIZE), ret, fail);
        NCCLCHECKGOTO(ncclNvlsGroupConnect(comm, shareableHandle, t->worldRankList[0], &mcHandle), ret, fail);
      }

      // 将当前设备添加到多播组
      CUCHECKGOTO(cuMulticastAddDevice(mcHandle, comm->cudaDev), ret, fail_mcHandle);
      // 保留虚拟地址空间（用于映射多播内存）
      CUCHECKGOTO(cuMemAddressReserve(&mcAddr, devr->bigSize, NCCL_MAX_PAGE_SIZE, 0, 0), ret, fail_mcHandle);
      // 将多播内存映射到保留的地址空间
      CUCHECKGOTO(cuMemMap(mcAddr, devr->bigSize, 0, mcHandle, 0), ret, fail_mcHandle_mcAddr);
      // 设置访问权限
      { CUmemAccessDesc accessDesc = {};
        accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        accessDesc.location.id = comm->cudaDev;
        accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        CUCHECKGOTO(cuMemSetAccess(mcAddr, devr->bigSize, &accessDesc, 1), ret, fail_mcHandle_mcAddr_unmap);
      }
      // 保存多播句柄和基地址到团队结构
      t->mcHandle = mcHandle;
      t->mcBasePtr = reinterpret_cast<void*>(mcAddr);

      // Bind new team with all existing memories.
      // 将新团队与所有现有内存绑定（使这些内存可通过多播访问）
      for (struct ncclDevrMemory* mem = devr->memHead; mem != nullptr; mem = mem->next) {
        NCCLCHECKGOTO(symBindTeamMemory(comm, t, mem), ret, fail_mcHandle_mcAddr_unmap_mems);
      }

      // 错误处理标签（用于资源清理）
      if (false) { // Error labels:
      fail_mcHandle_mcAddr_unmap_mems:                 // 解绑所有内存
        for (struct ncclDevrMemory* mem = devr->memHead; mem != nullptr; mem = mem->next) {
          symUnbindTeamMemory(comm, t, mem);
        }
      fail_mcHandle_mcAddr_unmap:                      // 取消内存映射
        CUCHECKIGNORE(cuMemUnmap(mcAddr, devr->bigSize));
        goto fail_mcHandle_mcAddr; // silence unused label warning
      fail_mcHandle_mcAddr:                            // 释放地址空间预留
        CUCHECKIGNORE(cuMemAddressFree(mcAddr, devr->bigSize));
        goto fail_mcHandle; // silence unused label warning
      fail_mcHandle:                                   // 释放多播句柄
        CUCHECKIGNORE(cuMemRelease(mcHandle));
        goto fail; // silence unused label warning
      }
    #else
      goto fail; // silence unused label warning
    #endif
    }
  }

  // 如果是新团队，添加到团队链表
  if (teamIsNew) {
     // Add to list
    t->next = devr->teamHead;                          // 插入到链表头部
    devr->teamHead = t;                                // 更新链表头
  }
  if (outTeam) *outTeam = t;                           // 返回团队指针
  return ret;                                          // 返回结果

fail:                                                    // 错误处理
  if (teamIsNew) free(t);                              // 释放新分配的团队结构
  return ret;                                          // 返回错误码
}

// 销毁所有对称团队
// 功能：清理所有团队的多播资源和内存绑定
// 参数：comm - NCCL 通信器
static void symTeamDestroyAll(struct ncclComm* comm) {
  struct ncclDevrState* devr = &comm->devrState;       // 获取设备端运行时状态
  // 遍历团队链表，逐个销毁
  while (devr->teamHead != nullptr) {
    struct ncclDevrTeam* t = devr->teamHead;           // 获取当前团队
    devr->teamHead = t->next;                          // 移动到下一个团队
    if (t->mcBasePtr != nullptr) {                     // 如果已初始化多播
      // 解绑该团队与所有内存的绑定
      for (struct ncclDevrMemory* m = devr->memHead; m != nullptr; m = m->next) {
        symUnbindTeamMemory(comm, t, m);
      }
      // 释放多播内存资源
      CUdeviceptr mcAddr = reinterpret_cast<CUdeviceptr>(t->mcBasePtr);
      CUCHECKIGNORE(cuMemUnmap(mcAddr, devr->bigSize));        // 取消内存映射
      CUCHECKIGNORE(cuMemAddressFree(mcAddr, devr->bigSize));  // 释放地址空间
      CUCHECKIGNORE(cuMemRelease(t->mcHandle));                // 释放多播句柄
    }
    free(t);                                            // 释放团队结构体
  }
}

// 获取或创建对称内存对象
// 注意：成功时我们会接管调用者的 memHandle 引用
// 由于需要为每个已存在的团队绑定多播，此函数要求调用者在返回用户前执行全局 barrier
// 参数：
//   comm      - NCCL 通信器
//   memHandle - CUDA 内存分配句柄
//   size      - 内存大小
//   outMem    - 输出：返回的内存对象
// 返回：ncclResult_t - 操作结果状态码
static ncclResult_t symMemoryObtain(
    struct ncclComm* comm, CUmemGenericAllocationHandle memHandle, size_t size,
    struct ncclDevrMemory** outMem
  ) {
  ncclResult_t ret = ncclSuccess;                      // 初始化返回值为成功
  struct ncclDevrState* devr = &comm->devrState;       // 获取设备端运行时状态
  int64_t bigOffset = 0;                               // 在大地址空间中的偏移量

  // 搜索是否已存在相同句柄的内存对象
  struct ncclDevrMemory* mem = devr->memHead;          // 从内存链表头开始
  while (mem != nullptr) {
    if (mem->memHandle == memHandle) {                 // 找到匹配的句柄
      CUCHECKIGNORE(cuMemRelease(memHandle));          // 释放调用者的句柄（我们使用现有的）
      goto leave;                                      // 跳转到引用计数增加
    }
    mem = mem->next;                                   // 继续遍历
  }
  // New memory. 未找到，创建新内存对象
  mem = (struct ncclDevrMemory*)malloc(sizeof(struct ncclDevrMemory));
  mem->refCount = 0;                                   // 引用计数初始化为 0
  mem->memHandle = memHandle;                          // 保存内存句柄
  mem->size = size;                                    // 保存内存大小

  // Grab offset in the big space.
  // 在大地址空间中分配偏移量（用于对称地址映射）
  NCCLCHECKGOTO(ncclSpaceAlloc(&devr->bigSpace, devr->bigSize, size, devr->granularity, &bigOffset), ret, fail_mem);
  mem->bigOffset = bigOffset;                          // 保存偏移量

  // Map unicast addresses into flat VA space for lsa team.
  // 将单播地址映射到 LSA 团队的平面地址空间
  NCCLCHECKGOTO(symMemoryMapLsaTeam(comm, memHandle, size, bigOffset), ret, fail_mem_space);

  // Bind new memory with each existing team.
  // 将新内存与每个已存在的团队绑定（启用多播访问）
  for (struct ncclDevrTeam* t = devr->teamHead; t != nullptr; t = t->next) {
    NCCLCHECKGOTO(symBindTeamMemory(comm, t, mem), ret, fail_mem_space_teams);
  }
  // Add to list of mems. 将新内存添加到内存链表
  mem->next = devr->memHead;                           // 插入到链表头部
  devr->memHead = mem;                                 // 更新链表头

leave:                                                   // 正常退出
  mem->refCount += 1;                                  // 增加引用计数
  *outMem = mem;                                       // 返回内存对象
  return ret;                                          // 返回结果

fail_mem_space_teams:                                  // 错误处理：解绑所有团队
  for (struct ncclDevrTeam* t = devr->teamHead; t != nullptr; t = t->next) {
    symUnbindTeamMemory(comm, t, mem);
  }
fail_mem_space:                                        // 错误处理：释放地址空间偏移
  ncclSpaceFree(&devr->bigSpace, bigOffset, size);
fail_mem:                                               // 错误处理：释放内存对象
  free(mem);
//fail:
  return ret;                                          // 返回错误码
}

// 释放对称内存对象的引用
// 功能：减少引用计数，当计数归零时释放所有相关资源
// 参数：
//   comm - NCCL 通信器
//   mem  - 要释放的内存对象
static void symMemoryDropRef(
    struct ncclComm* comm, struct ncclDevrMemory* mem
  ) {
  if (mem != nullptr && 0 == --mem->refCount) {        // 引用计数归零时释放资源
    struct ncclDevrState* devr = &comm->devrState;     // 获取设备端运行时状态
    // 解绑该内存与所有团队的绑定
    for (struct ncclDevrTeam* t = devr->teamHead; t != nullptr; t = t->next) {
      symUnbindTeamMemory(comm, t, mem);
    }
    // 取消 LSA 团队中所有 rank 的内存映射
    for (int r = 0; r < devr->lsaSize; r++) {
      CUdeviceptr addr = reinterpret_cast<uintptr_t>((char*)devr->lsaFlatBase + r*devr->bigSize + mem->bigOffset);
      CUCHECKIGNORE(cuMemUnmap(addr, mem->size));      // 取消映射
    }
    // 释放在大地址空间中的偏移量
    ncclSpaceFree(&devr->bigSpace, mem->bigOffset, mem->size);
    // 释放 CUDA 内存句柄
    CUCHECKIGNORE(cuMemRelease(mem->memHandle));

    // 从内存链表中移除
    struct ncclDevrMemory** ptr = &devr->memHead;      // 链表指针
    while (*ptr != mem) ptr = &(*ptr)->next;           // 找到要移除的节点
    *ptr = mem->next;                                  // Remove from list. 从链表中移除

    free(mem);                                         // 释放内存对象结构体
  }
}

// 一次性初始化窗口表
// 功能：在首次需要时创建窗口表（用于存储窗口地址到设备结构的映射）
// 参数：
//   comm   - NCCL 通信器
//   stream - CUDA 流（用于异步操作）
// 返回：ncclResult_t - 操作结果状态码
static ncclResult_t symWindowTableInitOnce(struct ncclComm* comm, cudaStream_t stream) {
  struct ncclDevrState* devr = &comm->devrState;       // 获取设备端运行时状态
  struct ncclDevCommWindowTable* tableDev = devr->windowTable; // 获取设备端窗口表
  if (tableDev == nullptr) { // Create on first need.
    // 从影子内存池分配窗口表（同时分配主机和设备端内存）
    NCCLCHECK(ncclShadowPoolAlloc<ncclDevCommWindowTable>(&devr->shadows, &tableDev, nullptr, stream));
    devr->windowTable = tableDev;                      // 保存窗口表指针
  }
  return ncclSuccess;                                  // 初始化完成
}

// 创建对称窗口
// 注意：成功时我们会接管调用者的 mem 引用
// 参数：
//   comm      - NCCL 通信器
//   mem       - 对称内存对象
//   memOffset - 内存内的偏移量
//   userPtr   - 用户空间指针（可为 nullptr）
//   userSize  - 窗口大小
//   winFlags  - 窗口标志位
//   localReg  - 本地注册句柄
//   outWinDev - 输出：设备端窗口结构
//   outWin    - 输出：主机端窗口结构
//   stream    - CUDA 流
// 返回：ncclResult_t - 操作结果状态码
static ncclResult_t symWindowCreate(
    struct ncclComm* comm, struct ncclDevrMemory* mem,
    size_t memOffset, void* userPtr, size_t userSize, int winFlags, void* localReg,
    struct ncclWindow_vidmem** outWinDev, struct ncclDevrWindow** outWin,
    cudaStream_t stream
  ) {
  uintptr_t userAddr = reinterpret_cast<uintptr_t>(userPtr); // 用户地址（整数形式）
  struct ncclDevrState* devr = &comm->devrState;              // 获取设备端运行时状态
  struct ncclDevrWindow* win;                                  // 主机端窗口结构

  // 分配并初始化主机端窗口结构
  win = (struct ncclDevrWindow*)malloc(sizeof(struct ncclDevrWindow));
  memset(win, 0, sizeof(*win));                               // 清零结构体
  win->memory = mem;                                          // 关联的对称内存对象
  win->size = userSize;                                       // 窗口大小
  win->bigOffset = mem->bigOffset + memOffset;                // 在大地址空间中的偏移
  win->winFlags = winFlags;                                   // 窗口标志位
  win->localRegHandle = localReg;                             // 本地注册句柄
  if (userPtr == nullptr) {
    // Null means caller has no VA and will use the lsa team flat VA address.
    // 用户指针为空，使用 LSA 团队的平面地址空间地址
    win->userPtr = (char*)devr->lsaFlatBase + (devr->lsaSelf*devr->bigSize) + mem->bigOffset;
  } else {
    // 使用用户提供的地址
    win->userPtr = userPtr;
  }

  // 分配设备端窗口结构（同时分配主机和设备端内存）
  struct ncclWindow_vidmem* winDev;                           // 设备端窗口结构指针
  struct ncclWindow_vidmem* winDevHost;                       // 主机端窗口结构指针
  NCCLCHECK(ncclShadowPoolAlloc(&devr->shadows, &winDev, &winDevHost, stream));
  win->vidmem = winDev;                                       // 保存设备端指针

  // 初始化设备端窗口结构（主机端副本）
  winDevHost->lsaFlatBase = (char*)devr->lsaFlatBase + win->bigOffset; // LSA 平面基地址 + 窗口偏移
  winDevHost->mcOffset4K = win->bigOffset>>12;                // 多播偏移（4KB 为单位，右移 12 位）
  winDevHost->stride4G = devr->bigSize>>32;                   // 每个 rank 的跨度（4GB 为单位，右移 32 位）
  winDevHost->lsaRank = devr->lsaSelf;                        // 在 LSA 团队中的 rank
  winDevHost->worldRank = comm->rank;                         // 世界 rank
  winDevHost->winHost = (void*)win;                           // 指向主机端窗口结构的指针
  // 将主机端结构复制到设备端
  CUDACHECK(cudaMemcpyAsync(winDev, winDevHost, sizeof(struct ncclWindow_vidmem), cudaMemcpyHostToDevice, stream));

  // NCCLCHECK(symWindowTableInitOnce(comm, stream)); // ensure devr->windowTable exists
  // 确保窗口表已初始化
  NCCLCHECK(symWindowTableInitOnce(comm, stream));
  struct ncclDevCommWindowTable* tableDev = devr->windowTable; // 设备端窗口表
  struct ncclDevCommWindowTable* tableHost;                     // 主机端窗口表
  NCCLCHECK(ncclShadowPoolToHost(&devr->shadows, tableDev, &tableHost)); // 同步到主机

  // 在窗口表中查找空闲槽位并插入窗口（每张表有 32 个槽位）
  while (true) {
    int i = 0;
    // 查找第一个空闲槽位
    while (i < 32 && tableHost->entries[i].window != nullptr) i += 1;
    if (i < 32) {
      // 找到空闲槽位，填充窗口信息
      tableHost->entries[i].base = userAddr;                  // 窗口基地址
      tableHost->entries[i].size = userAddr + userSize;       // 窗口结束地址（用于边界检查）
      tableHost->entries[i].window = winDev;                  // 设备端窗口结构指针
      // 将槽位信息复制到设备端
      CUDACHECK(cudaMemcpyAsync(&tableDev->entries[i], &tableHost->entries[i], sizeof(tableHost->entries[i]), cudaMemcpyHostToDevice, stream));
      break;                                                   // 插入完成
    }
    // 当前表已满，需要分配新表
    if (tableHost->next == nullptr) {
      // 分配新的窗口表节点
      NCCLCHECK(ncclShadowPoolAlloc<ncclDevCommWindowTable>(&devr->shadows, &tableHost->next, nullptr, stream));
      // 更新设备端链表指针
      CUDACHECK(cudaMemcpyAsync(&tableDev->next, &tableHost->next, sizeof(tableHost->next), cudaMemcpyHostToDevice, stream));
    }
    // 移动到下一张表
    tableDev = tableHost->next;
    NCCLCHECK(ncclShadowPoolToHost(&devr->shadows, tableHost->next, &tableHost));
  }

  { // insert into winSorted[] 插入到排序后的窗口数组（用于快速查找）
    int i = listFindSortedLub(&ncclDevrWindowSorted::userAddr, devr->winSorted, devr->winSortedCount, userAddr);
    // 创建排序窗口条目
    struct ncclDevrWindowSorted winSort;
    winSort.userAddr = userAddr;                              // 用户地址
    winSort.size = userSize;                                   // 窗口大小
    winSort.win = win;                                         // 窗口指针
    // 插入到排序数组
    listInsert(&devr->winSorted, &devr->winSortedCapacity, &devr->winSortedCount, i, winSort);
  }

  // 返回输出参数
  if (outWinDev) *outWinDev = winDev;                         // 设备端窗口结构
  if (outWin) *outWin = win;                                  // 主机端窗口结构
  return ncclSuccess;                                         // 返回成功
}

// 销毁对称窗口
// 功能：释放窗口相关的所有资源
// 参数：
//   comm   - NCCL 通信器
//   winDev - 设备端窗口结构
//   stream - CUDA 流
// 返回：ncclResult_t - 操作结果状态码
static ncclResult_t symWindowDestroy(struct ncclComm* comm, struct ncclWindow_vidmem* winDev, cudaStream_t stream) {
  ncclResult_t ret = ncclSuccess;                        // 初始化返回值为成功
  struct ncclDevrState* devr = &comm->devrState;         // 获取设备端运行时状态
  struct ncclWindow_vidmem* winDevHost;                  // 主机端窗口结构（设备版本的副本）
  struct ncclDevrWindow* winHost;                        // 主机端窗口结构

  // 将设备端窗口结构同步到主机端
  NCCLCHECKGOTO(ncclShadowPoolToHost(&devr->shadows, winDev, &winDevHost), ret, fail);
  winHost = (struct ncclDevrWindow*)winDevHost->winHost; // 获取主机端窗口结构

  // 释放对称内存对象的引用
  symMemoryDropRef(comm, winHost->memory);

  // 从窗口表中移除窗口
  { struct ncclDevCommWindowTable* tableDev = devr->windowTable; // 设备端窗口表
    struct ncclDevCommWindowTable* tableHost;                     // 主机端窗口表
    NCCLCHECKGOTO(ncclShadowPoolToHost(&devr->shadows, tableDev, &tableHost), ret, remove_winSorted);
    // 遍历窗口表链表，查找要删除的窗口
    while (true) {
      int i = 0;
      // 在当前表中查找窗口
      while (i < 32 && tableHost->entries[i].window != winDev) i += 1;
      if (i < 32) {
        // 找到窗口，清空槽位
        memset(&tableHost->entries[i], 0, sizeof(tableHost->entries[i]));
        // 清空设备端槽位
        CUDACHECKGOTO(cudaMemsetAsync(&tableDev->entries[i], 0, sizeof(tableDev->entries[i]), stream), ret, remove_winSorted);
        break;                                           // 移除完成
      }
      if (tableHost->next == nullptr) break; // Error didn't find window in table
      // 未找到，移动到下一张表
      tableDev = tableHost->next;
      NCCLCHECKGOTO(ncclShadowPoolToHost(&devr->shadows, tableHost->next, &tableHost), ret, remove_winSorted);
    }
  }
  // 释放设备端窗口结构的内存
  NCCLCHECKGOTO(ncclShadowPoolFree(&devr->shadows, winDev, stream), ret, remove_winSorted);

  // 取消本地内存注册
  NCCLCHECKGOTO(ncclCommDeregister(comm, winHost->localRegHandle), ret, remove_winSorted);

remove_winSorted:                                        // 从排序窗口数组中移除
  { int i = listFindSortedLub(&ncclDevrWindowSorted::userAddr, devr->winSorted, devr->winSortedCount, reinterpret_cast<uintptr_t>(winHost->userPtr));
    i -= 1; // least upper bound is just after ours.（最小上界在我们的窗口之后）
    listRemove(devr->winSorted, &devr->winSortedCount, i); // 从数组中移除
  }
  free(winHost);                                         // 释放主机端窗口结构
fail:                                                    // 错误处理
  return ret;                                            // 返回结果
}

// 在组内注册窗口
// 功能：注册用户内存窗口到对称内存管理系统
// 参数：
//   comm     - NCCL 通信器
//   userPtr  - 用户空间指针
//   userSize - 窗口大小
//   winFlags - 窗口标志位
//   outWinDev - 输出：设备端窗口结构
// 返回：ncclResult_t - 操作结果状态码
ncclResult_t ncclDevrWindowRegisterInGroup(
    struct ncclComm* comm,
    void* userPtr, size_t userSize, int winFlags, ncclWindow_t* outWinDev
  ) {
  ncclResult_t ret = ncclSuccess;                        // 初始化返回值为成功
  CUdeviceptr memAddr = 0;                               // 内存基地址
  size_t memSize = 0;                                    // 内存总大小
  CUmemGenericAllocationHandle memHandle = 0x0;          // CUDA 内存句柄
  size_t memOffset;                                      // 在内存分配中的偏移
  struct ncclDevrMemory* mem = nullptr;                  // 对称内存对象
  cudaStream_t stream = nullptr;                         // CUDA 流
  void* localRegHandle = nullptr;                        // 本地注册句柄

  // 注册本地内存（用于 DMA 访问）
  NCCLCHECKGOTO(ncclCommRegister(comm, userPtr, userSize, &localRegHandle), ret, fail);

  // 如果不支持对称内存，直接返回本地注册句柄
  if (!comm->symmetricSupport) {
    // We just return the local registration handle directly in this case, as there's no reason to allocate the
    // ncclWindow_vidmem structure on the device, etc.
    // 直接返回本地注册句柄（转换为窗口指针类型）
    *outWinDev = reinterpret_cast<struct ncclWindow_vidmem*>(localRegHandle);
    return ncclSuccess;
  }
  // 如果请求对称集合通信，初始化对称内核
  if (winFlags & NCCL_WIN_COLL_SYMMETRIC) {
    // Defer symmetric kernel init until at least one window with that flag exists.
    // 延迟对称内核初始化，直到至少有一个带有该标志的窗口存在
    NCCLCHECKGOTO(ncclSymkInitOnce(comm), ret, fail);
  }

  // Get underlying cumem handle:
  // 获取底层的 CUDA 内存句柄
  CUCHECKGOTO(cuMemGetAddressRange(&memAddr, &memSize, reinterpret_cast<CUdeviceptr>(userPtr)), ret, fail_locReg);
  // 计算用户指针在内存分配中的偏移
  memOffset = reinterpret_cast<CUdeviceptr>(userPtr) - memAddr;
  // 检查对齐是否满足要求
  if (memOffset%NCCL_WIN_REQUIRED_ALIGNMENT != 0) {
    WARN("Window address must be suitably aligned.");
    ret = ncclInvalidArgument;
    goto fail;
  }

  // 获取内存分配句柄（增加引用计数）
  CUCHECKGOTO(cuMemRetainAllocationHandle(&memHandle, reinterpret_cast<void*>(memAddr)), ret, fail_locReg);

  // Trade cumem handle for ncclDevrMemory*
  // 用 CUDA 内存句柄交换 NCCL 对称内存对象
  NCCLCHECKGOTO(symMemoryObtain(comm, memHandle, memSize, &mem), ret, fail_locReg_memHandle);
  memHandle = 0x0; // symMemoryObtain took our reference（symMemoryObtain 接管了引用）

  // 创建非阻塞 CUDA 流用于异步操作
  CUDACHECKGOTO(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), ret, fail);

  // 创建对称窗口
  NCCLCHECKGOTO(symWindowCreate(
      comm, mem, memOffset, userPtr, userSize, winFlags, localRegHandle, outWinDev, nullptr, stream
    ), ret, fail_locReg_memHandle_mem_stream);
  mem = nullptr; // symWindowCreate took our reference（symWindowCreate 接管了引用）

  // 等待所有异步操作完成
  CUDACHECKGOTO(cudaStreamSynchronize(stream), ret, fail_locReg_memHandle_mem_stream_win);

  // symWindowCreate needs barrier.
  // symWindowCreate 需要 barrier（确保所有 rank 都完成了窗口创建）
  NCCLCHECKGOTO(bootstrapBarrier(comm->bootstrap, comm->rank, comm->nRanks, 0xbeef), ret, fail_locReg_memHandle_mem_stream_win);

  // 销毁 CUDA 流
  cudaStreamDestroy(stream);
  return ret;                                            // 返回成功

  // 错误处理标签（按资源获取的逆序释放）
fail_locReg_memHandle_mem_stream_win:                    // 销毁窗口
  symWindowDestroy(comm, *outWinDev, stream);
  *outWinDev = nullptr;
  cudaStreamSynchronize(stream);
fail_locReg_memHandle_mem_stream:                        // 销毁流并释放内存引用
  cudaStreamDestroy(stream);
  symMemoryDropRef(comm, mem);
fail_locReg_memHandle:                                   // 释放内存句柄
  if (memHandle != 0x0) { CUCHECKIGNORE(cuMemRelease(memHandle)); }
fail_locReg:                                             // 取消本地注册
  ncclCommDeregister(comm, localRegHandle);
fail:                                                    // 最终错误处理
  *outWinDev = nullptr;
  return ret;
}

// 深拷贝设备端通信器需求
// 功能：完整复制通信器需求结构，包括所有链表成员
// 参数：
//   src - 源需求结构
//   dst - 输出：新分配的目标需求结构
// 返回：ncclResult_t - 操作结果状态码
static ncclResult_t deepCopyDevCommRequirements(
    struct ncclDevCommRequirements const* src,
    struct ncclDevCommRequirements** dst
) {
  ncclResult_t ret = ncclSuccess;                        // 初始化返回值为成功
  struct ncclDevResourceRequirements **dstRes;           // 目标资源需求链表指针
  struct ncclTeamRequirements **dstTeam;                 // 目标团队需求链表指针

  // 分配并清零目标结构体
  NCCLCHECK(ncclCalloc(dst, 1));

  /* copy the entire struct now and update linked lists later */
  // 先复制整个结构体，然后更新链表成员
  **dst = *src;

  // 复制资源需求链表
  dstRes = &(*dst)->resourceRequirementsList;             // 资源需求链表头
  for (struct ncclDevResourceRequirements* rr = src->resourceRequirementsList; rr != nullptr; rr = rr->next) {
    NCCLCHECKGOTO(ncclCalloc(dstRes, 1), ret, fail);     // 分配新节点
    (*dstRes)->bufferSize = rr->bufferSize;               // 缓冲区大小
    (*dstRes)->bufferAlign = rr->bufferAlign;             // 缓冲区对齐
    (*dstRes)->outBufferHandle = rr->outBufferHandle;     // 输出缓冲区句柄
    dstRes = &(*dstRes)->next;                            // 移动到下一个节点
  }

  // 复制团队需求链表
  dstTeam = &(*dst)->teamRequirementsList;                // 团队需求链表头
  for (struct ncclTeamRequirements* tr = src->teamRequirementsList; tr != nullptr; tr = tr->next) {
    NCCLCHECKGOTO(ncclCalloc(dstTeam, 1), ret, fail);    // 分配新节点
    (*dstTeam)->team = tr->team;                          // 团队描述
    (*dstTeam)->multimem = tr->multimem;                  // 多播支持标志
    (*dstTeam)->outMultimemHandle = tr->outMultimemHandle; // 输出多播句柄
    dstTeam = &(*dstTeam)->next;                          // 移动到下一个节点
  }

exit:                                                    // 正常退出
  return ret;
fail:                                                    // 错误处理
  freeDevCommRequirements(*dst);                          // 释放已分配的资源
  *dst = nullptr;
  goto exit;
}

// 释放设备端通信器需求
// 功能：释放通信器需求结构及其所有链表成员
// 参数：reqs - 要释放的需求结构
void freeDevCommRequirements(
    struct ncclDevCommRequirements* reqs
) {
  if (reqs) {                                            // 如果指针非空
    // 释放资源需求链表
    while (reqs->resourceRequirementsList) {
      struct ncclDevResourceRequirements* rr_next = reqs->resourceRequirementsList->next;
      free(reqs->resourceRequirementsList);              // 释放当前节点
      reqs->resourceRequirementsList = rr_next;           // 移动到下一个节点
    }

    // 释放团队需求链表
    while (reqs->teamRequirementsList) {
      struct ncclTeamRequirements* tr_next = reqs->teamRequirementsList->next;
      free(reqs->teamRequirementsList);                  // 释放当前节点
      reqs->teamRequirementsList = tr_next;              // 移动到下一个节点
    }

    free(reqs);                                          // 释放主结构体
  }
}

// 内部创建设备端通信器
// 功能：根据需求创建设备端通信器，包括资源分配和团队初始化
// 参数：
//   comm      - NCCL 通信器
//   reqs      - 通信器需求（资源、团队等）
//   outDevComm - 输出：设备端通信器结构
// 返回：ncclResult_t - 操作结果状态码
ncclResult_t ncclDevrCommCreateInternal(
    struct ncclComm* comm,
    struct ncclDevCommRequirements const* reqs, struct ncclDevComm* outDevComm
  ) {
  ncclResult_t ret = ncclSuccess;                        // 初始化返回值为成功
  struct ncclDevrState* devr = &comm->devrState;         // 获取设备端运行时状态
  struct ncclTeam world = ncclTeamWorld(comm);           // 世界团队（所有 rank）
  struct ncclTeam lsa = ncclTeamInnerFactor(world, devr->lsaSize); // LSA 团队
  struct ncclDevrTeam* tmLsa;                            // LSA 团队指针
  size_t bufSizeTotal;                                   // 总缓冲区大小
  struct ncclDevResourceRequirements* resReqsHead;       // 资源需求链表头
  struct ncclDevResourceRequirements lsaBarReq;          // LSA barrier 需求
  cudaStream_t stream = nullptr;                         // CUDA 流
  CUmemGenericAllocationHandle memHandle = 0x0;          // CUDA 内存句柄
  struct ncclDevrMemory* mem = nullptr;                  // 对称内存对象
  struct ncclDevrWindow* win = nullptr;                  // 窗口结构
  struct ncclWindow_vidmem* winHost = nullptr;           // 窗口主机端副本

  // 初始化输出结构体
  memset(outDevComm, 0, sizeof(*outDevComm));
  outDevComm->rank = comm->rank;                         // 当前 rank
  outDevComm->nRanks = comm->nRanks;                     // 总 rank 数
  outDevComm->nRanks_rcp32 = idivRcp32(comm->nRanks);   // 总 rank 数的倒数（32 位定点）
  outDevComm->lsaRank = devr->lsaSelf;                   // LSA 团队中的 rank
  outDevComm->lsaSize = devr->lsaSize;                   // LSA 团队大小
  outDevComm->lsaSize_rcp32 = idivRcp32(devr->lsaSize); // LSA 团队大小的倒数（32 位定点）

  // 获取或创建 LSA 团队（支持多播）
  NCCLCHECKGOTO(symTeamObtain(comm, lsa, /*multicast=*/reqs->lsaMultimem, &tmLsa), ret, fail);
  outDevComm->lsaMultimem.mcBasePtr = tmLsa->mcBasePtr; // 保存多播基地址

  // 处理用户指定的团队需求
  { struct ncclTeamRequirements* tr = reqs->teamRequirementsList;
    while (tr != nullptr) {
      if (tr->multimem) {                               // 如果需要多播支持
        struct ncclDevrTeam* tm;
        // 获取或创建带多播的团队
        NCCLCHECKGOTO(symTeamObtain(comm, tr->team, tr->multimem, &tm), ret, fail);
        // 返回多播基地址给用户
        if (tr->outMultimemHandle != nullptr) tr->outMultimemHandle->mcBasePtr = tm->mcBasePtr;
      }
      tr = tr->next;                                     // 继续处理下一个团队
    }
  }

  // 资源需求链表头
  resReqsHead = reqs->resourceRequirementsList;

  // 创建 LSA barrier 需求并插入到资源需求链表头部
  ncclLsaBarrierCreateRequirement(lsa, reqs->lsaBarrierCount, &outDevComm->lsaBarrier, &lsaBarReq);
  lsaBarReq.next = resReqsHead;                          // 插入到链表头部
  resReqsHead = &lsaBarReq;

  // 计算总缓冲区大小并分配缓冲区句柄
  { struct ncclDevResourceRequirements* rr = resReqsHead;
    bufSizeTotal = 0;                                    // 初始化总大小
    // 遍历所有资源需求，计算总大小
    while (rr != nullptr) {
      // 对齐到 128 字节或用户指定的对齐要求
      bufSizeTotal = alignUp(bufSizeTotal, std::max<size_t>(128, rr->bufferAlign));
      if (rr->outBufferHandle != nullptr) *rr->outBufferHandle = bufSizeTotal/128; // 返回缓冲区句柄
      bufSizeTotal += rr->bufferSize;                    // 累加缓冲区大小
      rr = rr->next;                                     // 继续下一个需求
    }
    // 最后对齐到内存粒度
    bufSizeTotal = alignUp(bufSizeTotal, devr->granularity);
  }

  // 创建非阻塞 CUDA 流
  CUDACHECKGOTO(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), ret, fail);

  // NCCLCHECKGOTO(symWindowTableInitOnce(comm, stream), ret, fail); // ensure devr->windowTable exists
  // 确保窗口表已初始化
  NCCLCHECKGOTO(symWindowTableInitOnce(comm, stream), ret, fail);
  outDevComm->windowTable = comm->devrState.windowTable; // 保存窗口表指针

  // 分配资源窗口
  if (bufSizeTotal == 0) {
    // 如果不需要资源缓冲区
    outDevComm->resourceWindow = nullptr;                // 无资源窗口
    outDevComm->resourceWindow_inlined = {};             // 清空内联窗口
  } else {
    // 配置 CUDA 内存分配属性
    CUmemAllocationProp memProp = {};
    memProp.type = CU_MEM_ALLOCATION_TYPE_PINNED;        // 固定内存类型
    memProp.location.type = CU_MEM_LOCATION_TYPE_DEVICE; // 设备内存
    memProp.requestedHandleTypes = ncclCuMemHandleType;  // 句柄类型
    memProp.location.id = comm->cudaDev;                 // 设备 ID

    // 创建 CUDA 内存分配
    CUCHECKGOTO(cuMemCreate(&memHandle, bufSizeTotal, &memProp, 0), ret, fail);

    // 获取或创建对称内存对象
    NCCLCHECKGOTO(symMemoryObtain(comm, memHandle, bufSizeTotal, &mem), ret, fail);
    memHandle = 0x0; // Reference given to symMemoryObtain（引用已转交）

    // NCCLCHECKGOTO(symWindowCreate( // Requires world barrier afterward.
    // 创建资源窗口（之后需要全局 barrier）
    NCCLCHECKGOTO(symWindowCreate(
      comm, mem, /*memOffset=*/0, nullptr, bufSizeTotal, /*winFlags=*/0,
      /*localReg=*/nullptr, &outDevComm->resourceWindow, &win,
      stream), ret, fail);
    mem = nullptr; // Reference given to symWindowCreate（引用已转交）
    // 同步窗口结构到主机端（用于内联副本）
    NCCLCHECKGOTO(ncclShadowPoolToHost(&comm->devrState.shadows, win->vidmem, &winHost), ret, fail);
    outDevComm->resourceWindow_inlined = *winHost;       // 保存内联副本

    // 清零资源窗口内存
    CUDACHECKGOTO(cudaMemsetAsync(win->userPtr, 0, bufSizeTotal, stream), ret, fail);
  }

  // 等待所有异步操作完成
  CUDACHECKGOTO(cudaStreamSynchronize(stream), ret, fail);

  // 全局 barrier（确保所有 rank 都完成了资源窗口创建）
  NCCLCHECKGOTO(bootstrapBarrier(comm->bootstrap, comm->rank, comm->nRanks, 0xbeef), ret, fail);

  // 销毁 CUDA 流
  cudaStreamDestroy(stream);
  return ret;                                            // 返回成功

fail:                                                    // 错误处理
  if (win != nullptr) {                                  // 销毁窗口
    symWindowDestroy(comm, win->vidmem, stream);
    cudaStreamSynchronize(stream);
  }
  if (mem != nullptr) {                                  // 释放内存引用
    symMemoryDropRef(comm, mem);
  }
  if (memHandle != 0x0) {                               // 释放内存句柄
    CUCHECKIGNORE(cuMemRelease(memHandle));
  }
  if (stream != nullptr) {                               // 销毁流
    cudaStreamDestroy(stream);
  }
  return ret;
}

////////////////////////////////////////////////////////////////////////////////

// NCCL 公共 API：注册窗口
NCCL_API(ncclResult_t, ncclCommWindowRegister, ncclComm_t comm, void* ptr, size_t size, ncclWindow_t* win, int winFlags);
// 注册对称内存窗口
// 功能：将用户内存注册为对称窗口，支持设备端直接访问
// 参数：
//   comm     - NCCL 通信器
//   userPtr  - 用户空间指针
//   userSize - 窗口大小
//   outWinDev - 输出：设备端窗口结构
//   winFlags - 窗口标志位
// 返回：ncclResult_t - 操作结果状态码
ncclResult_t ncclCommWindowRegister(
    struct ncclComm* comm, void* userPtr, size_t userSize,
    struct ncclWindow_vidmem** outWinDev, int winFlags
  ) {
  ncclResult_t ret = ncclSuccess;                        // 初始化返回值为成功
  int saveDev;                                           // 保存当前 CUDA 设备
  struct ncclDevrRegTask* task;                          // 注册任务

  CUDACHECK(cudaGetDevice(&saveDev));                    // 获取当前设备
  NCCLCHECK(ncclGroupStartInternal());                   // 开始组操作

  // 如果参数无效，且不支持对称内存也不启用本地注册，则直接退出
  if (userPtr == nullptr || userSize == 0 || !(comm->symmetricSupport || ncclParamLocalRegister())) goto exit;

  NCCLCHECKGOTO(ncclCommEnsureReady(comm), ret, fail);   // 确保通信器已就绪
  CUDACHECKGOTO(cudaSetDevice(comm->cudaDev), ret, fail); // 设置为通信器的设备

  NCCLCHECKGOTO(ncclDevrInitOnce(comm), ret, fail);      // 初始化设备端运行时

  // 创建注册任务并加入队列（延迟到组结束时执行）
  NCCLCHECKGOTO(ncclCalloc(&task, 1), ret, fail);
  task->userPtr = userPtr;                               // 用户指针
  task->userSize = userSize;                             // 用户大小
  task->winFlags = winFlags;                             // 窗口标志
  task->outWinDev = outWinDev;                           // 输出指针
  ncclIntruQueueEnqueue(&comm->devrState.regTaskQueue, task); // 加入任务队列
  ncclGroupCommJoin(comm, ncclGroupTaskTypeSymRegister); // 加入组操作

exit:                                                    // 正常退出
  ncclGroupErrCheck(ret);                                // 检查组错误
  NCCLCHECK(ncclGroupEndInternal());                     // 结束组操作
  cudaSetDevice(saveDev);                                // 恢复原始设备
  return ret;
fail:                                                    // 错误处理
  goto exit;
}

// NCCL 公共 API：注销窗口
NCCL_API(ncclResult_t, ncclCommWindowDeregister, ncclComm_t comm, ncclWindow_t win);
// 注销对称内存窗口
// 功能：释放之前注册的窗口及其相关资源
// 参数：
//   comm   - NCCL 通信器
//   winDev - 设备端窗口结构
// 返回：ncclResult_t - 操作结果状态码
ncclResult_t ncclCommWindowDeregister(struct ncclComm* comm, struct ncclWindow_vidmem* winDev) {
  ncclResult_t ret = ncclSuccess;                        // 初始化返回值为成功
  int saveDev;                                           // 保存当前 CUDA 设备
  cudaStream_t stream;                                   // CUDA 流

  if (winDev == nullptr) goto exit;                      // 空指针直接返回

  // 如果不支持对称内存，直接注销本地注册
  if (!comm->symmetricSupport) {
    NCCLCHECKGOTO(ncclCommDeregister(comm, winDev), ret, fail);
    goto exit;
  }
  CUDACHECKGOTO(cudaGetDevice(&saveDev), ret, fail);     // 获取当前设备
  CUDACHECKGOTO(cudaSetDevice(comm->cudaDev), ret, fail); // 设置为通信器的设备
  CUDACHECKGOTO(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), ret, fail_dev); // 创建流
  NCCLCHECKGOTO(symWindowDestroy(comm, winDev, stream), ret, fail_dev_stream); // 销毁窗口
fail_dev_stream:                                         // 错误处理：同步并销毁流
  cudaStreamSynchronize(stream);
  cudaStreamDestroy(stream);
fail_dev:                                                // 错误处理：恢复设备
  cudaSetDevice(saveDev);
fail:                                                    // 错误处理
exit:                                                    // 正常退出
  return ret;
}

// 根据用户指针查找窗口
// 功能：在已注册的窗口中查找包含指定地址的窗口
// 参数：
//   comm    - NCCL 通信器
//   userPtr - 用户空间指针
//   outWin  - 输出：找到的窗口（如果存在）
// 返回：ncclResult_t - 操作结果状态码
ncclResult_t ncclDevrFindWindow(
    struct ncclComm* comm, void const* userPtr, struct ncclDevrWindow** outWin
  ) {
  struct ncclDevrState* devr = &comm->devrState;         // 获取设备端运行时状态
  uintptr_t userAddr = reinterpret_cast<uintptr_t>(userPtr); // 转换为整数地址
  // 在排序的窗口数组中查找最小上界（第一个大于 userAddr 的窗口）
  int i = listFindSortedLub(&ncclDevrWindowSorted::userAddr, devr->winSorted, devr->winSortedCount, userAddr);
  if (0 < i && (userAddr - devr->winSorted[i-1].userAddr < devr->winSorted[i-1].size)) {
    // 如果前一个窗口包含该地址，返回该窗口
    *outWin = devr->winSorted[i-1].win;
  } else {
    // 未找到包含该地址的窗口
    *outWin = nullptr;
  }
  return ncclSuccess;
}

// NCCL 公共 API：创建设备端通信器
NCCL_API(ncclResult_t, ncclDevCommCreate, ncclComm_t comm, ncclDevCommRequirements_t const* reqs, ncclDevComm_t* outDevComm);
// 创建设备端通信器
// 功能：根据需求创建设备端通信器，用于设备端发起的集合通信
// 参数：
//   comm      - NCCL 通信器
//   reqs      - 通信器需求（资源、团队等）
//   outDevComm - 输出：设备端通信器结构
// 返回：ncclResult_t - 操作结果状态码
ncclResult_t ncclDevCommCreate(
    ncclComm_t comm, struct ncclDevCommRequirements const* reqs,
    struct ncclDevComm* outDevComm
  ) {
  ncclResult_t ret = ncclSuccess;                        // 初始化返回值为成功
  int saveDev;                                           // 保存当前 CUDA 设备
  struct ncclDevrCommCreateTask* task = nullptr;         // 通信器创建任务

  CUDACHECK(cudaGetDevice(&saveDev));                    // 获取当前设备
  NCCLCHECK(ncclGroupStartInternal());                   // 开始组操作

  // 检查是否支持对称内存
  if (!comm->symmetricSupport) {
    WARN("Communicator does not support symmetric memory!");
    ret = ncclInvalidUsage;
    goto fail;
  }

  NCCLCHECKGOTO(ncclCommEnsureReady(comm), ret, fail);   // 确保通信器已就绪
  CUDACHECKGOTO(cudaSetDevice(comm->cudaDev), ret, fail); // 设置为通信器的设备

  NCCLCHECKGOTO(ncclDevrInitOnce(comm), ret, fail);      // 初始化设备端运行时

  // 创建通信器创建任务并加入队列（延迟到组结束时执行）
  NCCLCHECKGOTO(ncclCalloc(&task, 1), ret, fail);
  // reqs must be deep copied to the task so background threads can safely access it
  // 需求必须深拷贝到任务中，以便后台线程可以安全访问
  NCCLCHECKGOTO(deepCopyDevCommRequirements(reqs, &task->reqs), ret, fail);
  task->outDevComm = outDevComm;                         // 输出指针
  ncclIntruQueueEnqueue(&comm->devrState.commCreateTaskQueue, task); // 加入任务队列
  ncclGroupCommJoin(comm, ncclGroupTaskTypeSymRegister); // 加入组操作

exit:                                                    // 正常退出
  ncclGroupErrCheck(ret);                                // 检查组错误
  NCCLCHECK(ncclGroupEndInternal());                     // 结束组操作
  cudaSetDevice(saveDev);                                // 恢复原始设备
  return ret;
fail:                                                    // 错误处理
  free(task);                                            // 释放任务
  goto exit;
}

// NCCL 公共 API：销毁设备端通信器
NCCL_API(ncclResult_t, ncclDevCommDestroy, ncclComm_t comm, ncclDevComm_t const* devComm);
// 销毁设备端通信器
// 功能：释放设备端通信器及其相关资源
// 参数：
//   comm    - NCCL 通信器
//   devComm - 设备端通信器
// 返回：ncclResult_t - 操作结果状态码
ncclResult_t ncclDevCommDestroy(
    struct ncclComm* comm, struct ncclDevComm const* devComm
  ) {
  //struct ncclDevrState* devr = &comm->devrState;
  // 注销资源窗口（如果存在）
  if (devComm->resourceWindow != nullptr) {
    NCCLCHECK(ncclCommWindowDeregister(comm, devComm->resourceWindow));
  }
  return ncclSuccess;
}


// Get the corresponding pointer in another lsa rank's symmetric memory window
// 获取另一个 LSA rank 的对称内存窗口中的对应指针
ncclResult_t ncclDevrGetLsaRankPtr(struct ncclComm* comm, struct ncclDevrWindow* winHost, size_t offset, int lsaRank, void** outPtr) {
  if (winHost == nullptr || outPtr == nullptr) {
    return ncclInvalidArgument;                          // 参数无效
  }

  struct ncclDevrState* devr = &comm->devrState;         // 获取设备端运行时状态

  // Validate lsaRank is within bounds
  // 验证 lsaRank 是否在有效范围内
  if (lsaRank < 0 || lsaRank >= devr->lsaSize) {
    return ncclInvalidArgument;
  }

  // Validate offset is within bounds
  // 验证偏移量是否在有效范围内
  if (offset < 0 || offset >= winHost->size) {
    return ncclInvalidArgument;
  }

  // Calculate the address with offset for the specified lsa rank
  // 计算指定 lsa rank 的带偏移地址
  *outPtr = (void*)((uintptr_t)devr->lsaFlatBase + lsaRank * devr->bigSize + winHost->bigOffset + offset);
  return ncclSuccess;
}

// Get the multicast address for a given team
// 获取给定团队的多播地址
ncclResult_t ncclDevrGetLsaTeamPtrMC(struct ncclComm* comm, struct ncclDevrWindow* winHost, size_t offset, struct ncclTeam lsaTeam, void** outPtr){
  if (winHost == nullptr || outPtr == nullptr) {
    return ncclInvalidArgument;                          // 参数无效
  }

  if (!comm->nvlsSupport) {
    return ncclInvalidUsage;                             // 不支持 NVLS
  }

  bool multimem = true;                                  // 启用多播
  struct ncclDevrTeam* tm;                               // 团队指针
  NCCLCHECK(symTeamObtain(comm, lsaTeam, multimem, &tm)); // 获取或创建团队

  // Return the base multicast address for this team with offset
  // 返回该团队的带偏移多播基地址
  *outPtr = (void*)((uintptr_t)tm->mcBasePtr + winHost->bigOffset + offset);
  return ncclSuccess;
}

////////////////////////////////////////////////////////////////////////////////

// Find the least index strictly greater than arg.
// 查找最小索引使得 arr[index] > arg（最小上界，Least Upper Bound）
// 使用混合算法：大范围用二分查找，小范围用线性扫描
template<typename Obj, typename Key>
static int listFindSortedLub(Key Obj::*key, Obj* sorted, int count, Key arg) {
  int lo = 0, hi = count;                                // 搜索范围 [lo, hi)
  // 二分查找阶段（当范围大于 16 时使用）
  while (lo + 16 < hi) {
    int i = (lo + hi)/2;                                // 中点
    if (sorted[i].*key <= arg) lo = i+1;                 // 如果中点 <= arg，搜索右半部分
    else hi = i;                                         // 否则搜索左半部分
  }
  // 线性扫描阶段（处理剩余的小范围）
  int i = lo;
  while (i < hi && sorted[i].*key <= arg) i++;           // 找到第一个 > arg 的元素
  return i;                                              // 返回索引
}

// 向动态数组中插入元素（带自动扩容）
// 功能：在指定索引处插入元素，如果容量不足则自动扩容（2倍增长策略）
// 参数：
//   list    - 动态数组指针的指针（可能需要重新分配）
//   capacity - 当前容量指针（可能需要更新）
//   count   - 当前元素数量指针（插入后会增加）
//   index   - 插入位置索引
//   val     - 要插入的值
template<typename Obj>
static void listInsert(Obj** list, int* capacity, int* count, int index, Obj val) {
  // 检查是否需要扩容
  if (*capacity < *count + 1) {
    *capacity *= 2;                                      // 容量翻倍
    if (*capacity == 0) *capacity = 16;                  // 初始容量为 16
    // 重新分配内存（保留原有数据）
    *list = (Obj*)realloc(*list, (*capacity)*sizeof(Obj));
  }
  // 将插入位置之后的元素后移一位（从后向前移动，避免覆盖）
  for (int j = *count; j != index; j--) {
    (*list)[j] = (*list)[j-1];
  }
  // 在插入位置放置新元素
  (*list)[index] = val;
  // 增加元素计数
  *count += 1;
}

// 从动态数组中移除指定索引的元素
// 功能：移除指定索引的元素，将后续元素前移
// 参数：
//   list  - 动态数组指针
//   count - 当前元素数量指针（移除后会减少）
//   index - 要移除的元素索引
template<typename Obj>
static void listRemove(Obj* list, int* count, int index) {
  // 将移除位置之后的元素前移一位（覆盖要移除的元素）
  for (int i = index; i+1 < *count; i++) {
    list[i] = list[i+1];
  }
  // 减少元素计数
  *count -= 1;
}

