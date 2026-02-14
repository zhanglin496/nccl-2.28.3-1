/*************************************************************************
 * Copyright (c) 2015-2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// 通信器结构，包含设备、拓扑等信息
#include "comm.h"
// 传输层接口，网络、共享内存等
#include "transport.h"
// 组操作相关定义
#include "group.h"
// NVTX 性能分析工具
#include "nvtx.h"

// NCCL 公共 API：内存分配函数声明
NCCL_API(ncclResult_t, ncclMemAlloc, void **ptr, size_t size);
// NCCL 内存分配函数
// 功能：分配设备内存，优先使用 CUDA 12.0+ 的 cuMem API（支持虚拟地址管理）
// 参数：
//   ptr  - 输出：分配的内存指针
//   size - 要分配的字节数
// 返回：ncclResult_t - 操作结果状态码
ncclResult_t ncclMemAlloc(void **ptr, size_t size) {
  NCCL_NVTX3_FUNC_RANGE;                               // NVTX 性能分析范围标记
  ncclResult_t ret = ncclSuccess;                      // 初始化返回值为成功

#if CUDART_VERSION >= 12010                            // CUDA 12.0+ 支持 cuMem API
  size_t memGran = 0;                                  // 内存分配粒度（字节对齐单位）
  CUdevice currentDev;                                // 当前 CUDA 设备
  CUmemAllocationProp memprop = {};                   // 内存分配属性
  CUmemAccessDesc accessDesc = {};                    // 内存访问描述符（用于设置访问权限）
  CUmemGenericAllocationHandle handle = (CUmemGenericAllocationHandle)-1; // 内存分配句柄
  int cudaDev;                                        // CUDA 设备 ID
  int flag;                                           // 标志位（用于查询设备属性）
  int dcnt;                                           // 设备总数

  // 检查参数有效性
  if (ptr == NULL || size == 0) goto fallback;        // 如果指针为空或大小为0，回退到传统方式

  // 初始化 CUDA 库（加载 cuMem 相关函数）
  if (ncclCudaLibraryInit() != ncclSuccess) goto fallback; // 如果初始化失败，回退

  CUDACHECK(cudaGetDevice(&cudaDev));                  // 获取当前 CUDA 设备
  CUCHECK(cuDeviceGet(&currentDev, cudaDev));        // 获取 CU 设备句柄

  //实际调用cuMem api
  // 如果启用了 cuMem 支持（环境变量 NCCL_CUMEM_ENABLE=1）
  if (ncclCuMemEnable()) {
    size_t handleSize = size;                          // 内存分配大小（对齐后）
    int requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR; // 请求的句柄类型（POSIX 文件描述符）
    // Query device to see if FABRIC handle support is available
    // 查询设备是否支持 FABRIC 句柄类型（用于 NVLink 互连）
    flag = 0;
    (void) CUPFN(cuDeviceGetAttribute(&flag, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED, currentDev));
    if (flag) requestedHandleTypes |= CU_MEM_HANDLE_TYPE_FABRIC; // 如果支持，添加 FABRIC 句柄类型
    memprop.type = CU_MEM_ALLOCATION_TYPE_PINNED;        // 固定内存类型（页锁定）
    memprop.location.type = CU_MEM_LOCATION_TYPE_DEVICE; // 设备内存位置
    memprop.requestedHandleTypes = (CUmemAllocationHandleType) requestedHandleTypes; // 句柄类型
    memprop.location.id = currentDev;                   // 设备 ID
    // Query device to see if RDMA support is available
    // 查询设备是否支持 GPU Direct RDMA（用于 NVLink 等高速互连）
    flag = 0;
    CUCHECK(cuDeviceGetAttribute(&flag, CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED, currentDev));
    if (flag) memprop.allocFlags.gpuDirectRDMACapable = 1; // 设置 RDMA 支持标志
    // 获取推荐的内存分配粒度（用于对齐，提高性能）
    CUCHECK(cuMemGetAllocationGranularity(&memGran, &memprop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
    CUDACHECK(cudaGetDeviceCount(&dcnt));               // 获取设备总数
    ALIGN_SIZE(handleSize, memGran);                    // 按粒度对齐大小

    // 优先尝试使用 FABRIC 句柄类型（如果支持）
    if (requestedHandleTypes & CU_MEM_HANDLE_TYPE_FABRIC) {
      /* First try cuMemCreate() with FABRIC handle support and then remove if it fails */
      /* 首先尝试使用 FABRIC 句柄支持创建内存，如果失败则移除该类型 */
      CUresult err = CUPFN(cuMemCreate(&handle, handleSize, &memprop, 0));
      if (err == CUDA_ERROR_NOT_PERMITTED || err == CUDA_ERROR_NOT_SUPPORTED) {
        // 如果权限不足或不支持，回退到 POSIX 文件描述符类型
        requestedHandleTypes &= ~CU_MEM_HANDLE_TYPE_FABRIC;
        memprop.requestedHandleTypes = (CUmemAllocationHandleType) requestedHandleTypes;
        /* Allocate the physical memory on the device */
        /* 在设备上分配物理内存 */
        CUCHECK(cuMemCreate(&handle, handleSize, &memprop, 0));
      } else if (err != CUDA_SUCCESS) {
        // Catch and report any error from above
        // 捕获并报告上述错误
        CUCHECK(cuMemCreate(&handle, handleSize, &memprop, 0));
      }
    } else {
      /* Allocate the physical memory on the device */
      /* 在设备上分配物理内存 */
      CUCHECK(cuMemCreate(&handle, handleSize, &memprop, 0));
    }
    /* Reserve a virtual address range */
    /* 保留一段虚拟地址空间（不分配实际物理内存） */
    //分配连续虚拟地址空间
    CUCHECK(cuMemAddressReserve((CUdeviceptr*)ptr, handleSize, memGran, 0, 0));
    /* Map the virtual address range to the physical allocation */
    /* 将虚拟地址空间映射到物理内存分配 */
    //虚拟地址空间映射到物理地址空间
    CUCHECK(cuMemMap((CUdeviceptr)*ptr, handleSize, 0, handle, 0));
    /* Now allow RW access to the newly mapped memory */
    /* 为新映射的内存设置读写访问权限 */
    //设置地址空间的访问权限
    for (int i = 0; i < dcnt; ++i) {                  // 遍历所有 GPU 设备
      int p2p = 0;                                     // P2P 访问支持标志
      // 如果是当前设备或支持 P2P 访问，则设置访问权限
      if (i == cudaDev || ((cudaDeviceCanAccessPeer(&p2p, i, cudaDev) == cudaSuccess) && p2p)) {
        accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE; // 位置类型：设备
        accessDesc.location.id = i;                     // 设备 ID
        accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE; // 访问权限：可读写
        CUCHECK(cuMemSetAccess((CUdeviceptr)*ptr, handleSize, &accessDesc, 1)); // 设置访问权限
      }
      if (0 == p2p && i != cudaDev)                     // 如果不支持 P2P 且不是当前设备
        INFO(NCCL_ALLOC, "P2P not supported between GPU%d and GPU%d", cudaDev, i); // 输出信息日志
    }
    goto exit;                                        // 跳转到退出（成功）
  }

  // 回退路径：使用传统的 cudaMalloc
fallback:
#endif
  // Coverity is right to complain that we may pass a NULL ptr to cudaMalloc.  That's deliberate though:
  // we want CUDA to return an error to the caller.
  // coverity[var_deref_model]
  // 使用传统的 cudaMalloc 分配内存
  CUDACHECKGOTO(cudaMalloc(ptr, size), ret, fail);   // 调用 CUDA 分配内存

exit:                                                    // 正常退出
  return ret;                                         // 返回结果
fail:                                                    // 错误处理
  goto exit;                                          // 跳转到退出
}

// NCCL 公共 API：内存释放函数声明
NCCL_API(ncclResult_t, ncclMemFree, void *ptr);
// NCCL 内存释放函数
// 功能：释放之前分配的内存，支持 cuMem API 和传统 cudaFree
// 参数：ptr - 要释放的内存指针
// 返回：ncclResult_t - 操作结果状态码
ncclResult_t  ncclMemFree(void *ptr) {
  NCCL_NVTX3_FUNC_RANGE;                               // NVTX 性能分析范围标记
  ncclResult_t ret = ncclSuccess;                      // 初始化返回值为成功
  int saveDevice;                                     // 保存当前设备

  CUDACHECK(cudaGetDevice(&saveDevice));               // 获取并保存当前 CUDA 设备

#if CUDART_VERSION >= 12010                            // CUDA 12.0+ 支持 cuMem API
  CUdevice ptrDev = 0;                                 // 指针所属的设备 ID

  if (ptr == NULL) goto fallback;                       // 如果指针为空，直接回退
  if (ncclCudaLibraryInit() != ncclSuccess) goto fallback; // 如果初始化失败，回退

  // 获取指针所属的设备
  CUCHECKGOTO(cuPointerGetAttribute((void*)&ptrDev, CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL, (CUdeviceptr)ptr), ret, fail);
  // 切换到指针所属的设备
  CUDACHECKGOTO(cudaSetDevice((int)ptrDev), ret, fail);
  // 如果启用了 cuMem 支持，使用 cuMem 释放
  if (ncclCuMemEnable()) {
    NCCLCHECKGOTO(ncclCuMemFree(ptr), ret, fail);     // 释放 cuMem 分配的内存
    goto exit;                                        // 跳转到退出（成功）
  }

  // 回退路径：使用传统的 cudaFree
fallback:
#endif
  // 使用传统的 cudaFree 释放内存
  CUDACHECKGOTO(cudaFree(ptr), ret, fail);           // 调用 CUDA 释放内存

exit:                                                    // 正常退出
  CUDACHECK(cudaSetDevice(saveDevice));                // 恢复原始设备
  return ret;                                         // 返回结果
fail:                                                    // 错误处理
  goto exit;                                          // 跳转到退出
}

////////////////////////////////////////////////////////////////////////////////
// ncclSpace: 地址空间管理器
//
// This datastructure "cuts" the line of non-negative integers into segments
// which alternate between "full" (allocated) and "empty" (not allocated). The
// cuts are sorted ascending. The segment after the last cut must be empty
// (the unallocated frontier). Knwoing this we can deduce whether the segment
// ending at cut[i] is full or empty with this formula:
//   isFull(i) = (i%2 != ncuts%2)
//
// 这个数据结构使用"切割点"将非负整数线段分割为交替的"满"（已分配）和"空"（未分配）段。
// 切割点按升序排列。最后一个切割点之后的段必须是空的（未分配的前沿）。
// 据此我们可以通过以下公式判断结束于 cut[i] 的段是满还是空：
//   isFull(i) = (i%2 != ncuts%2)

// 构造地址空间管理器
// 功能：初始化 ncclSpace 结构
// 参数：a - 地址空间管理器指针
void ncclSpaceConstruct(struct ncclSpace* a) {
  memset(a, 0, sizeof(*a));                            // 清零结构体（初始化为空状态）
}

// 销毁地址空间管理器
// 功能：释放地址空间管理器占用的资源
// 参数：a - 地址空间管理器指针
void ncclSpaceDestruct(struct ncclSpace* a) {
  free(a->cuts);                                      // 释放切割点数组
}

// 插入段（内部辅助函数）
// 功能：在指定索引处插入一个新段，并合并相邻的空段
// 参数：
//   a     - 地址空间管理器
//   index - 插入位置索引
//   lo    - 段的起始位置
//   hi    - 段的结束位置
static void insertSegment(struct ncclSpace* a, int index, int64_t lo, int64_t hi) {
  // Insert space for two cuts in `a->cuts[]` before `index`.
  // 在 a->cuts[] 中的 index 之前插入两个切割点的空间
  // 如果容量不足，需要扩容
  if (a->count + 2 > a->capacity) {
    a->capacity *= 2;                                // 容量翻倍
    if (a->capacity == 0) a->capacity = 16;          // 初始容量为 16
    // 分配新数组并复制数据
    int64_t* cuts1 = (int64_t*)malloc(a->capacity*sizeof(int64_t));
    for (int i=0; i < index; i++) 
        cuts1[i] = a->cuts[i];  // 复制 index 之前的数据
    for (int i=index; i < a->count; i++) 
        cuts1[i+2] = a->cuts[i]; // 复制 index 及之后的数据（偏移 2）
    free(a->cuts);                                   // 释放旧数组
    a->cuts = cuts1;                                  // 更新指针
  } else {
    // 原地移动数据（从后向前，避免覆盖）
    for (int i=a->count-1; index <= i; i--) 
        a->cuts[i+2] = a->cuts[i];
  }
  // 设置新切割点
  a->cuts[index+0] = lo;                              // 段的起始位置
  a->cuts[index+1] = hi;                              // 段的结束位置
  a->count += 2;                                     // 切割点数量加 2

  // Filter pairs of adjacent repeated values from cuts[]. Since these mark
  // boundaries where segments transition between full<->empty, dropping such a
  // pair fuses two adjacent segments together. Examples:
  // 过滤 cuts[] 中相邻的重复值对。由于这些标记满<->空转换的边界，
  // 删除这样的一对会融合两个相邻的段。示例：
  //   [1,2,3,3,4] -> [1,2,4]              （两个 3 融合为空段）
  //   [1,2,3,3,3,4] -> [1,2,3,4]          // 保留一个 3，因为是满<->空转换
  //   [1,2,3,3,3,3,4] -> [1,2,4]           （三个 3，中间的与相邻的融合）
  // Leading zeros don't have to be in pairs, they are always dropped:
  // 前导零不必成对出现，总是被删除：
  //   [0,1,2] -> [1,2]                   （删除前导零）
  //   [0,0,1,2] -> [1,2]                   （删除所有前导零）
  int r = index, w = index; // Read and write cursors. 读写游标
  int64_t prev = r==0 ? 0 : a->cuts[r-1];            // 前一个值（用于检测重复）
  while (r < a->count) {
    int64_t cur = a->cuts[r++];                      // 当前值（r 是读游标）
    a->cuts[w++] = cur;                               // 写入值（w 是写游标）
    if (prev == cur) {                                // 重复值表示空段，可以删除
      // Erase last two cuts or just one if we're at the start.
      // 删除最后两个切割点，如果在开始位置则只删除一个
      w -= w==1 ? 1 : 2;
      // Zeros can only occur at the beginning (due to being sorted). We want to
      // drop any number of zeros, but only even numbers of other repeated values.
      // 零只能出现在开头（因为已排序）。我们希望删除任意数量的零，
      // 但其他重复值只能成对删除。
      // So set to zero here, which will make prev=0, thus if next value is zero
      // 因此这里设置为零，使得 prev=0，这样如果下一个值是零会被删除，
      // 但如果不是零则需要开始新的一对才能被删除。
      cur = 0;
    }
    prev = cur;                                       // 更新前一个值
  }
  a->count = w;                                       // 更新切割点数量
}

// 从地址空间管理器中分配内存
// 功能：在地址空间中查找合适的空闲段并分配
// 参数：
//   a         - 地址空间管理器
//   limit     - 地址空间上限（字节）
//   size      - 要分配的大小（字节）
//   align     - 对齐要求（字节）
//   outOffset - 输出：分配的偏移量
// 返回：ncclResult_t - 操作结果状态码
ncclResult_t ncclSpaceAlloc(
    struct ncclSpace* a, int64_t limit, int64_t size, int align,
    int64_t* outOffset
  ) {
  // When allocating we try to locate the first empty segment which can hold
  // the allocation and move its lower cut upward.
  // 分配时，我们尝试定位第一个能容纳分配的空闲段，并将其下界向上移动。
  int i = a->count%2; // First empty segment ends at cuts[i]
  // 第一个空闲段结束于 cuts[i]
  // （由于段是满<->空交替的，如果切割点数量为偶数，第一个段是空的）
  size_t off;
  while (i <= a->count) {                             // 遍历所有空闲段
    size_t lo = i == 0 ? 0 : a->cuts[i-1];            // 段的下界（0 或前一个切割点）
    size_t hi = i == a->count ? limit : a->cuts[i];    // 段的上界（上限或当前切割点）
    off = alignUp(lo, align);                         // 对齐下界
    if (off + size <= hi) {                             // 检查段是否足够大
      *outOffset = off;                               // 返回分配的偏移量
      if (i == 0 || off + size == hi) { // Slow path required.
        // 慢速路径：需要插入新切割点（分配在段的开头或填满整个段）
        insertSegment(a, i, off, off+size);           // 插入新段
      } else { // We can just append to the end of a full segment.
        // 快速路径：只需将满段的上界向上移动
        a->cuts[i-1] = off + size;                   // 移动前一个切割点（满段的结束位置）
      }
      return ncclSuccess;                             // 分配成功
    }
    i += 2; // Next empty segment                      // 移动到下一个空闲段
  }
  // 未找到合适的空间
  WARN("Allocation failed. No suitable space found to accommodate size=0x%lx within limit=0x%lx", (long)size, (long)limit);
  return ncclInternalError;                           // 返回内部错误
}

// 释放地址空间管理器中的内存
// 功能：将已分配的段标记为空闲
// 参数：
//   a      - 地址空间管理器
//   offset - 要释放的偏移量
//   size   - 要释放的大小
// 返回：ncclResult_t - 操作结果状态码
ncclResult_t ncclSpaceFree(struct ncclSpace* a, int64_t offset, int64_t size) {
  // 检查是否有任何分配
  if (a->count == 0 || a->cuts[a->count-1] <= offset) {
    WARN("No allocation found at offset=0x%lx", (long)offset);
    return ncclInternalError;
  }

  // This could be binary search, but since allocate is linear there's no point.
  // 这里可以使用二分查找，但由于分配是线性的，所以没有必要。
  int i = 1 - a->count%2; // First full segment ends at cuts[i]
  // 第一个满段结束于 cuts[i]
  // （如果切割点数量为奇数，第一个段是满的）
  while (a->cuts[i] <= offset) i += 2;                 // 线性查找包含 offset 的满段

  int64_t lo = i==0 ? 0 : a->cuts[i-1];                 // 段的下界
  int64_t hi = a->cuts[i];                                 // 段的上界

  // 验证偏移和大小是否在段范围内
  if (offset < lo || hi < offset + size) {
    WARN("Given size=0x%lx extends beyond allocation.", (long)size);
    return ncclInternalError;
  }

  // First try the two fast cases which just shrink a segment from one side.
  // 首先尝试两种快速情况，只需从一侧收缩段。
  if (i != 0 && lo == offset && offset + size != hi) {
    // 快速情况 1：从底部收缩（释放段的前部）
    a->cuts[i-1] = offset + size; // Bring bottom up. // 将下界向上移动
  } else if (lo != offset && offset + size == hi) {
    // 快速情况 2：从顶部收缩（释放段的尾部）
    a->cuts[i] = offset; // Bring top down.             // 将上界向下移动
  } else { // Slow path.                                // 慢速路径：需要插入新切割点
    insertSegment(a, i, offset, offset+size);         // 插入新段（将段分割）
  }
  return ncclSuccess;                                  // 释放成功
}

////////////////////////////////////////////////////////////////////////////////
// ncclShadowPool: 影子内存池
//
// 影子内存池用于管理主机端和设备端的数据结构同步
// 它维护设备对象的哈希表，以及用于批量分配的页系统

// 影子页结构：一个连续的内存块（最多 64 个对象）
struct ncclShadowPage { // A contiguous block of (at most) 64 objects
  struct ncclShadowPage* next;                       // 链表下一个节点
  int objSize;                                      // 每个对象的大小（字节）
  uint64_t freeMask;                                  // 空闲掩码（64 位，每位代表一个槽位）
  void* devObjs;                                     // 设备端对象数组指针
};
// 影子对象结构
struct ncclShadowObject {
  struct ncclShadowObject* next;                   // 哈表下一个节点（用于哈希冲突链表）
  void* devObj;                                     // 设备端对象指针
  void* hostObj;                                    // 主机端对象指针（内存在结构体末尾）
  struct ncclShadowPage* page;                     // 所属页（null 表示直接在 CUDA 内存池中分配）
};

// 构造影子内存池
// 功能：初始化影子内存池
// 参数：pool - 影子内存池指针
void ncclShadowPoolConstruct(struct ncclShadowPool* pool) {
  pool->hbits = 0;                                   // 哈希表位数（初始化为 0，表示未初始化）
  pool->count = 0;                                   // 对象数量
  pool->table = nullptr;                              // 哈希表指针
  pool->pages = nullptr;                             // 页链表头指针
}

// 销毁影子内存池
// 功能：释放影子内存池中的所有资源，包括哈希表、页链表和内存池
// 参数：pool - 影子内存池指针
// 返回：ncclSuccess 成功，其他值表示失败
ncclResult_t ncclShadowPoolDestruct(struct ncclShadowPool* pool) {
  if (pool->hbits != 0) {                             // 检查哈希表是否已初始化（hbits != 0 表示已初始化）
    cudaStream_t stream;                              // CUDA 流，用于异步内存释放操作
    CUDACHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));  // 创建非阻塞 CUDA 流

    if (pool->count != 0) {                           // 检查是否有已分配的对象
      for (int i=0; i < 1<<pool->hbits; i++) {        // 遍历哈希表的所有桶
        struct ncclShadowObject* obj = pool->table[i];  // 获取当前桶的对象链表头
        while (obj != nullptr) {                      // 遍历对象链表
          struct ncclShadowPage* page = obj->page;    // 获取对象所属的页
          if (page != nullptr) {                      // 如果对象来自页分配
            if (page->freeMask == 0) {                // 检查页是否已满（freeMask == 0 表示没有空闲槽位）
              // 将满页重新放回页链表，以便后续重用
              page->freeMask = 1;                     // 设置 freeMask 为 1，表示有一个槽位可用
              page->next = pool->pages;               // 将页插入到页链表头部
              pool->pages = page;
            }
          } else {                                    // 对象不是从页分配的（单独分配）
            cudaFreeAsync(obj->devObj, stream);       // 异步释放设备内存
          }
          struct ncclShadowObject* next = obj->next;  // 保存下一个对象指针
          free(obj);                                   // 释放主机端影子对象
          obj = next;                                  // 移动到下一个对象
        }
      }
    }
    free(pool->table);                                // 释放哈希表内存

    while (pool->pages != nullptr) {                  // 遍历并释放所有页
      cudaFreeAsync(pool->pages->devObjs, stream);    // 异步释放页中的设备内存
      struct ncclShadowPage* next = pool->pages->next;  // 保存下一页指针
      free(pool->pages);                              // 释放页结构体
      pool->pages = next;                             // 移动到下一页
    }

    cudaStreamSynchronize(stream);                    // 等待所有异步操作完成
    cudaStreamDestroy(stream);                        // 销毁 CUDA 流
    cudaMemPoolDestroy(pool->memPool);                // 销毁 CUDA 内存池
  }
  return ncclSuccess;                                 // 返回成功
}

// 计算哈希桶索引
// 功能：使用乘法哈希将设备对象指针映射到哈希表桶索引
// 参数：
//   hbits - 哈希表的位数（桶数为 2^hbits）
//   devObj - 设备对象指针
// 返回：哈希桶索引
// 哈希算法：使用乘法哈希（multiplicative hashing）
//   0x9e3779b97f4a7c13 是 2^64 / φ（黄金比例），常用于哈希函数
static int hashBucket(int hbits, void* devObj) {
  uintptr_t h = reinterpret_cast<uintptr_t>(devObj);  // 将设备指针转换为整数
  h ^= h>>32;                                          // 折叠 64 位指针的高低位
  h *= 0x9e3779b97f4a7c13;                             // 乘以黄金比例常数，混合比特位
  return (uint64_t)h >> (64-hbits);                    // 取高位作为哈希桶索引
}

// 将对象插入哈希表
// 功能：将影子对象插入到哈希表的适当桶中
// 参数：
//   pool - 影子内存池指针
//   obj - 要插入的影子对象
static void hashInsert(struct ncclShadowPool* pool, struct ncclShadowObject* obj) {
  int b = hashBucket(pool->hbits, obj->devObj);        // 计算对象应插入的哈希桶索引
  obj->next = pool->table[b];                          // 将对象的 next 指向当前桶的链表头
  pool->table[b] = obj;                                // 将对象插入到桶的链表头部（头插法）
}

// 从影子内存池分配对象
// 功能：分配设备内存和对应的主机端影子对象，可选择使用页分配或直接分配
// 参数：
//   pool - 影子内存池指针
//   size - 要分配的大小（字节）
//   outDevObj - 输出参数，返回分配的设备对象指针
//   outHostObj - 输出参数，返回对应的主机端影子对象指针
//   stream - CUDA 流，用于异步内存操作
// 返回：ncclSuccess 成功，其他值表示失败
// 分配策略：
//   - 小对象（64KB 能容纳 >= 3 个）：使用页分配，批量分配 64KB 页
//   - 大对象：直接从内存池分配
ncclResult_t ncclShadowPoolAlloc(
    struct ncclShadowPool* pool, size_t size, void** outDevObj, void** outHostObj,
    cudaStream_t stream
  ) {
  if (size == 0) {                                    // 特殊情况：分配大小为 0
    if (outDevObj) *outDevObj = nullptr;              // 设备对象指针设为 nullptr
    if (outHostObj) *outHostObj = nullptr;            // 主机对象指针设为 nullptr
    return ncclSuccess;                               // 返回成功
  }

  int hbits = pool->hbits;                            // 获取当前哈希表位数
  if (hbits == 0) {                                   // 首次使用时初始化
    cudaMemPoolProps props = {};                      // 内存池属性（零初始化）
    props.allocType = cudaMemAllocationTypePinned;    // 分配类型：固定内存（pinned memory）
    props.handleTypes = cudaMemHandleTypeNone;        // 不使用导出句柄
    props.location.type = cudaMemLocationTypeDevice;  // 位置类型：设备内存
    cudaGetDevice(&props.location.id);                // 获取当前设备 ID
    CUDACHECK(cudaMemPoolCreate(&pool->memPool, &props));  // 创建 CUDA 内存池

    pool->hbits = hbits = 4;                          // 初始哈希表位数为 4（16 个桶）
    pool->table = (struct ncclShadowObject**)malloc(sizeof(struct ncclShadowObject*)<<hbits);  // 分配哈希表
    for (int i=0; i < 1<<hbits; i++) pool->table[i] = nullptr;  // 初始化所有桶为 nullptr
  }

  // 检查是否需要扩展哈希表（在插入之前检查）
  // 维持 2:1 的对象与桶比例，以保持哈希表效率
  if (pool->count+1 > 2<<hbits) {
    struct ncclShadowObject** table0 = pool->table;   // 保存旧哈希表指针
    struct ncclShadowObject** table1 = (struct ncclShadowObject**)malloc(sizeof(struct ncclShadowObject*)<<(hbits+1));  // 分配新的更大的哈希表（大小翻倍）
    pool->table = table1;                              // 更新池的哈希表指针
    pool->hbits = hbits+1;                             // 更新哈希表位数（增加 1 位）
    for (int i1=0; i1 < 2<<hbits; i1++) table1[i1] = nullptr;  // 初始化新哈希表的所有桶为 nullptr
    for (int i0=0; i0 < 1<<hbits; i0++) {             // 遍历旧哈希表
      struct ncclShadowObject* obj = table0[i0];      // 获取旧哈希表当前桶的对象链表
      while (obj) {                                    // 遍历对象链表
        struct ncclShadowObject* next = obj->next;    // 保存下一个对象指针
        hashInsert(pool, obj);                         // 将对象重新插入到新哈希表
        obj = next;                                    // 移动到下一个对象
      }
    }
    hbits += 1;                                        // 同步更新局部 hbits 变量，匹配 pool->hbits
    free(table0);                                      // 释放旧哈希表内存
  }

  struct ncclShadowPage* page;                         // 指向分配页的指针
  void *devObj;                                        // 设备对象指针
  if ((64<<10)/size >= 3) {                            // 判断是否使用页分配（64KB 能容纳 >= 3 个对象）
    // 使用页分配策略（适合小对象）
    // 计算页内对象大小的对齐方式（向上舍入到 2 的幂次）
    int shift = std::max<int>(0, (int)log2Down(size) + 1 - 4);  // 计算 shift 值（使对象大小对齐）
    int pageObjSize = ((size + (1<<shift)-1)>>shift)<<shift;     // 页内对象大小（对齐后）
    struct ncclShadowPage** pagePtr = &pool->pages;    // 指向页链表指针的指针（用于修改链表）
    while (true) {                                     // 查找合适的页
      page = *pagePtr;                                 // 获取当前页
      if (page == nullptr) {                           // 没有找到合适的页，需要创建新页
        size_t pageSize = std::min<size_t>(64<<10, 64*pageObjSize);  // 计算页大小（最大 64KB 或 64 个对象）
        page = (struct ncclShadowPage*)malloc(sizeof(struct ncclShadowPage));  // 分配页结构体
        page->objSize = pageObjSize;                   // 设置页内对象大小
        page->freeMask = uint64_t(-1)>>(64 - pageSize/pageObjSize);  // 初始化空闲掩码（所有位都为 1，表示全部可用）
        page->next = pool->pages;                      // 将新页插入到页链表头部
        pool->pages = page;
        CUDACHECK(cudaMallocFromPoolAsync(&page->devObjs, pageSize, pool->memPool, stream));  // 从内存池异步分配页的设备内存
        CUDACHECK(cudaMemsetAsync(page->devObjs, 0, pageSize, stream));  // 异步将页内存初始化为 0
        // fall through...                              // 继续执行后续代码
      }
      if (page->objSize == pageObjSize) {              // 找到对象大小匹配的页
        int slot = popFirstOneBit(&page->freeMask);    // 从 freeMask 中取出第一个 1 的位置（找到空闲槽位）
        devObj = (char*)page->devObjs + slot*pageObjSize;  // 计算设备对象地址（页基地址 + 槽位偏移）
        if (page->freeMask == 0) *pagePtr = page->next;  // 如果页已满，从页链表中移除
        break;                                         // 找到合适的槽位，退出循环
      }
      pagePtr = &page->next;                           // 移动到下一页
    }
  } else {                                             // 使用直接分配策略（适合大对象）
    page = nullptr;                                    // 不使用页分配
    CUDACHECK(cudaMallocFromPoolAsync(&devObj, size, pool->memPool, stream));  // 直接从内存池分配设备内存
    CUDACHECK(cudaMemsetAsync(devObj, 0, size, stream));  // 异步将内存初始化为 0
  }

  // 分配主机端影子对象
  struct ncclShadowObject* obj = (struct ncclShadowObject*)malloc(
    sizeof(struct ncclShadowObject) + /*padding=*/alignof(max_align_t)-1 + size  // 分配影子对象结构体 + 填充 + 主机端存储
  );
  obj->page = page;                                    // 设置所属页（可能为 nullptr）
  obj->devObj = devObj;                                // 设置设备对象指针
  obj->hostObj = alignUp((char*)(obj+1), alignof(max_align_t));  // 计算主机端存储地址（对齐到 max_align_t）
  memset(obj->hostObj, 0, size);                       // 将主机端存储初始化为 0
  hashInsert(pool, obj);                               // 将对象插入到哈希表
  pool->count += 1;                                    // 增加对象计数
  if (outDevObj) *outDevObj = devObj;                  // 返回设备对象指针
  if (outHostObj) *outHostObj = obj->hostObj;          // 返回主机端对象指针
  return ncclSuccess;                                  // 返回成功
}

// 释放影子内存池中的对象
// 功能：释放设备对象并回收对应的主机端影子对象，页分配的对象会回收槽位
// 参数：
//   pool - 影子内存池指针
//   devObj - 要释放的设备对象指针
//   stream - CUDA 流，用于异步内存操作
// 返回：ncclSuccess 成功，其他值表示失败
ncclResult_t ncclShadowPoolFree(struct ncclShadowPool* pool, void* devObj, cudaStream_t stream) {
  if (devObj == nullptr) return ncclSuccess;            // 特殊情况：释放 nullptr，直接返回成功

  int b = hashBucket(pool->hbits, devObj);              // 计算设备对象对应的哈希桶索引
  struct ncclShadowObject** pobj = &pool->table[b];     // 获取哈希桶的指针（用于修改链表）
  while (true) {                                        // 在哈希桶的链表中查找对象
    if (*pobj == nullptr) {                             // 链表遍历结束仍未找到
      WARN("Device object does not exist in shadow pool.");  // 输出警告信息
      return ncclInternalError;                         // 返回内部错误
    }
    if ((*pobj)->devObj == devObj) break;               // 找到匹配的设备对象，退出循环
    pobj = &(*pobj)->next;                              // 移动到链表下一个节点
  }
  struct ncclShadowObject* obj = *pobj;                 // 保存找到的对象指针
  *pobj = obj->next;                                    // 从哈希表链表中移除对象
  if (obj->page != nullptr) {                           // 对象来自页分配
    if (obj->page->freeMask == 0) {                     // 如果页之前已满
      obj->page->next = pool->pages;                    // 将页重新插入到页链表（使其可供分配）
      pool->pages = obj->page;
    }
    int slot = ((char*)obj->devObj - (char*)obj->page->devObjs)/obj->page->objSize;  // 计算对象在页中的槽位索引
    obj->page->freeMask |= uint64_t(1)<<slot;           // 设置对应的 freeMask 位，标记槽位为空闲
  } else {                                              // 对象不是从页分配的（单独分配）
    CUDACHECK(cudaFreeAsync(devObj, stream));           // 异步释放设备内存
  }
  free(obj);                                            // 释放主机端影子对象结构体
  pool->count -= 1;                                     // 减少对象计数
  return ncclSuccess;                                   // 返回成功
}

// 根据设备对象指针查找对应的主机端影子对象
// 功能：在影子内存池中查找设备对象，并返回对应的主机端指针
// 参数：
//   pool - 影子内存池指针
//   devObj - 设备对象指针
//   hostObj - 输出参数，返回对应的主机端对象指针
// 返回：ncclSuccess 成功，其他值表示失败
ncclResult_t ncclShadowPoolToHost(struct ncclShadowPool* pool, void* devObj, void** hostObj) {
  if (devObj == nullptr) {                             // 特殊情况：设备对象指针为 nullptr
    *hostObj = nullptr;                                // 主机对象指针也设为 nullptr
    return ncclSuccess;                                // 返回成功
  }

  int b = hashBucket(pool->hbits, devObj);             // 计算设备对象对应的哈希桶索引
  struct ncclShadowObject* obj = pool->table[b];       // 获取哈希桶的链表头
  while (true) {                                       // 在哈希桶的链表中查找对象
    if (obj == nullptr) {                              // 链表遍历结束仍未找到
      WARN("Device object does not exist in shadow pool.");  // 输出警告信息
      return ncclInternalError;                        // 返回内部错误
    }
    if (obj->devObj == devObj) break;                  // 找到匹配的设备对象，退出循环
    obj = obj->next;                                   // 移动到链表下一个节点
  }
  *hostObj = obj->hostObj;                             // 返回主机端对象指针
  return ncclSuccess;                                  // 返回成功
}
