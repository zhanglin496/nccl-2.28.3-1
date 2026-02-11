/*************************************************************************
 * Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
 * 版权所有 (c) 2023，NVIDIA 公司。保留所有权利。
 *
 * See LICENSE.txt for license information
 * 有关许可信息，请参见 LICENSE.txt 文件
 ************************************************************************/

// 包含参数检查头文件，提供参数验证宏和函数
// Need some checks here since we access comm
// 这里需要进行一些检查，因为我们要访问 comm 对象
#include "argcheck.h"
// 包含 NCCL 公共 API 头文件，定义了用户可见的接口
#include "nccl.h"
// 包含通信域内部定义，描述 ncclComm 结构体
#include "comm.h"
// 包含网络相关接口和定义
#include "net.h"
// 包含内存注册相关接口
#include "register.h"
// 包含传输层接口
#include "transport.h"
// 包含组通信相关定义
#include "group.h"

// 定义本地注册参数
// NCCL_PARAM: 参数定义宏，创建可通过环境变量配置的参数
// LocalRegister: 参数名称
// "LOCAL_REGISTER": 环境变量名称
// 1: 默认值，表示启用本地内存注册
NCCL_PARAM(LocalRegister, "LOCAL_REGISTER", 1);

// 函数功能：检查本地注册是否有效
// 通过检查本地引用计数来判断注册是否仍然有效
// 参数说明：
//   - reg: 注册记录指针，包含内存注册的信息
//   - isValid: 输出参数，返回注册是否有效
// 返回值：ncclSuccess 表示成功
ncclResult_t ncclRegLocalIsValid(struct ncclReg *reg, bool *isValid) {
  // 检查 reg 指针和 isValid 指针是否有效
  if (reg && isValid) {
    // 检查本地引用计数是否大于 0
    // localRefs: 本地引用计数，表示有多少个本地使用者引用此注册
    if (reg->localRefs)
      // 引用计数大于 0，注册有效
      *isValid = true;
    else
      // 引用计数等于 0，注册无效
      *isValid = false;
  }
  // 返回成功状态码
  return ncclSuccess;
}

// 函数功能：注册内存缓冲区
// 这是一个内部函数，被 ncclCommRegister 和 ncclCommGraphRegister 调用
// 参数说明：
//   - comm: 通信上下文指针
//   - data: 要注册的内存缓冲区地址
//   - size: 缓冲区的大小（字节数）
//   - isGraph: 是否为图注册（true=CUDA Graph 模式，false=运行时模式）
//   - handle: 输出参数，返回注册句柄，用于后续的注销操作
// 返回值：ncclSuccess 表示成功
ncclResult_t ncclRegister(struct ncclComm* comm, void* data, size_t size, bool isGraph, void** handle) {
  // 检查 comm 指针是否有效
  // CommCheck: 参数检查宏，验证 comm 指针并返回错误码
  NCCLCHECK(CommCheck(comm, "ncclCommRegister", "comm"));
  // 获取注册缓存指针
  // regCache: 注册缓存，管理所有已注册的内存区域
  struct ncclRegCache* cache = &comm->regCache;
  // 获取页大小（用于对齐）
  // pageSize: 内存页的大小，通常是 4KB 或 64KB
  uintptr_t pageSize = cache->pageSize;
  // 计算起始地址对齐后的值
  // (uintptr_t)data: 将 data 指针转换为整数类型
  // & -pageSize: 位运算技巧，将地址向下对齐到页边界
  // 例如：pageSize=4096，data=0x1234 => begAddr=0x1000
  uintptr_t begAddr = (uintptr_t)data & -pageSize;
  // 计算结束地址对齐后的值
  // (uintptr_t)data + size: 缓冲区结束地址
  // + pageSize-1: 向上取整的偏移量
  // & -pageSize: 对齐到页边界
  uintptr_t endAddr = ((uintptr_t)data + size + pageSize-1) & -pageSize;

  // 如果启用了指针检查，验证 CUDA 指针的有效性
  // checkPointers: 配置标志，是否检查指针有效性
  if (comm->checkPointers) NCCLCHECK(CudaPtrCheck(data, comm, "buff", "ncclCommRegister"));
  // 记录注册信息到日志
  // INFO: 信息日志宏
  INFO(NCCL_REG, "register comm %p buffer %p size %zi", comm, data, size);

  // 遍历缓存槽位，查找合适的插入位置或已存在的注册
  // slot: 当前检查的槽位索引
  for (int slot=0; /*true*/; slot++) {
    // 检查是否到达缓存末尾或找到插入位置
    // 插入条件：到达末尾 或 当前地址小于槽位地址
    if ((slot == cache->population) || (begAddr < cache->slots[slot]->begAddr)) {
      // 检查缓存是否需要扩容
      if (cache->population == cache->capacity) { // must grow cache
        // 扩容策略：如果容量小于 32，直接设置为 32；否则容量翻倍
        cache->capacity = cache->capacity < 32 ? 32 : 2*cache->capacity;
        // 重新分配槽位数组
        // ncclRealloc: 内存重新分配函数，保留原有数据
        NCCLCHECK(ncclRealloc(&cache->slots, cache->population, cache->capacity));
      }
      // 移动现有槽位，为新槽位腾出空间
      // memmove: 内存移动函数，从 source 移动到 destination
      memmove(cache->slots+slot+1, cache->slots+slot, (cache->population-slot)*sizeof(struct ncclReg*));
      // 为新槽位分配内存并初始化为 0
      NCCLCHECK(ncclCalloc(cache->slots+slot, 1));
      // 获取新分配的注册槽位指针
      struct ncclReg* regSlot = cache->slots[slot];
      // 设置注册的起始地址
      regSlot->begAddr = begAddr;
      // 设置注册的结束地址
      regSlot->endAddr = endAddr;
      // 根据注册类型增加相应的引用计数
      if (isGraph) regSlot->graphRefs = 1;
      // 图注册引用计数
      else regSlot->localRefs = 1;
      // 本地注册引用计数
      // 增加缓存人口数量
      cache->population += 1;
      // 返回注册句柄
      *handle = regSlot;
      // 跳转到退出标签
      goto exit;
    // 检查是否找到完全匹配的已有注册
    // 条件：起始地址包含在内 且 结束地址包含在内
    } else if ((cache->slots[slot]->begAddr <= begAddr) &&
               (cache->slots[slot]->endAddr >= endAddr)) {
      // 找到匹配的注册，增加相应的引用计数
      if (isGraph) cache->slots[slot]->graphRefs++;
      else cache->slots[slot]->localRefs++;
      // 返回已存在的注册句柄
      *handle = cache->slots[slot];
      // 跳转到退出标签
      goto exit;
    }
  }

// 退出标签
exit:
  // 返回成功状态码
  return ncclSuccess;
}

// 静态函数：清理注册资源（内部函数）
// 释放网络、NVLS、CollNet 和 IPC 相关的注册资源
// 参数说明：
//   - comm: 通信上下文指针
//   - reg: 注册记录指针
// 返回值：ncclSuccess 表示成功
static ncclResult_t regCleanup(struct ncclComm* comm, struct ncclReg* reg) {
  // 检查网络注册是否完成
  // NET_REG_COMPLETE: 状态标志，表示网络注册已完成
  if (reg->state & NET_REG_COMPLETE) {
    // 获取网络句柄链表头
    struct ncclRegNetHandles* netHandle = reg->netHandleHead;
    // 声明前驱节点指针，用于释放链表
    struct ncclRegNetHandles* netHandlePrev;
    // 遍历网络句柄链表
    while(netHandle) {
      // 注销网络缓冲区
      // ncclNetDeregBuffer: 网络缓冲区注销函数
      // proxyConn: 代理连接，用于实际的网络传输
      // handle: 内存句柄
      if (ncclNetDeregBuffer(comm, netHandle->proxyConn, netHandle->handle) != ncclSuccess) {
        // 注销失败，记录警告日志
        WARN("rank %d deregister NET buffer handle %p proxy rank %d failed\n", comm->rank, netHandle->handle, netHandle->proxyConn->rank);
      }
      // 保存当前节点指针
      netHandlePrev = netHandle;
      // 移动到下一个节点
      netHandle = netHandle->next;
      // 释放当前节点内存
      free(netHandlePrev);
    }
  }
  // 检查 NVLS 注册是否完成
  // NVLS_REG_COMPLETE: 状态标志，表示 NVLS 注册已完成
  if (reg->state & NVLS_REG_COMPLETE) {
    // 注销 NVLS 缓冲区
    // ncclNvlsDeregBuffer: NVLS 缓冲区注销函数
    // &reg->mcHandle: 内存一致性句柄
    // reg->regAddr: 注册的地址
    // reg->dev: GPU 设备编号
    // reg->regUCSize: 统一缓存大小
    // reg->regMCSize: 内存一致性缓存大小
    if (ncclNvlsDeregBuffer(comm, &reg->mcHandle, reg->regAddr, reg->dev, reg->regUCSize, reg->regMCSize) != ncclSuccess) {
      // 注销失败，记录警告日志
      WARN("rank %d deregister NVLS buffer %p dev %d ucsize %ld mcsize %ld failed", comm->rank, (void*)reg->regAddr, reg->dev, reg->regUCSize, reg->regMCSize);
    }
    // 将注册地址设置为 NULL
    reg->regAddr = (CUdeviceptr)NULL;
  }
  // 检查 CollNet 注册是否完成
  // COLLNET_REG_COMPLETE: 状态标志，表示 CollNet 注册已完成
  if (reg->state & COLLNET_REG_COMPLETE) {
    // 注销 CollNet 缓冲区
    if (ncclCollnetDeregBuffer(comm, reg->collnetProxyconn, reg->collnetHandle) != ncclSuccess) {
      // 注销失败，记录警告日志
      WARN("rank %d deregister COLLNET buffer handle %p proxy rank %d failed", comm->rank, reg->collnetHandle, reg->collnetProxyconn->rank);
    }
  }
  // 检查 IPC 注册是否完成
  // IPC_REG_COMPLETE: 状态标志，表示 IPC 注册已完成
  if (reg->state & IPC_REG_COMPLETE) {
    // 遍历所有本地 rank 的 IPC 信息
    for (int i = 0; i < NCCL_MAX_LOCAL_RANKS; ++i)
      // 检查该 rank 的 IPC 信息是否存在
      if (reg->ipcInfos[i]) {
        // 注销 IPC 缓冲区
        if (ncclIpcDeregBuffer(comm, reg->ipcInfos[i]) != ncclSuccess) {
          // 注销失败，记录警告日志
          WARN("rank %d deregister IPC buffer %p peerRank %d failed", comm->rank, reg->ipcInfos[i]->baseAddr, reg->ipcInfos[i]->peerRank);
        }
        // 释放 IPC 信息结构体
        free(reg->ipcInfos[i]);
      }
    // 释放主机端的远程地址数组
    if (reg->regIpcAddrs.hostPeerRmtAddrs) free(reg->regIpcAddrs.hostPeerRmtAddrs);
    // 释放设备端的远程地址数组
    if (reg->regIpcAddrs.devPeerRmtAddrs) NCCLCHECK(ncclCudaFree(reg->regIpcAddrs.devPeerRmtAddrs));
  }
  // 返回成功状态码
  return ncclSuccess;
}

// 函数功能：清理通信域的所有注册资源
// 在销毁通信域时调用，释放所有已注册的内存
// 参数说明：
//   - comm: 通信上下文指针
// 返回值：ncclSuccess 表示成功
ncclResult_t ncclRegCleanup(struct ncclComm* comm) {
  // 获取注册缓存指针
  struct ncclRegCache* cache = &comm->regCache;
  // 遍历所有已注册的槽位
  for (int i = 0; i < cache->population; i++) {
    // 获取当前槽位的注册记录
    struct ncclReg* reg = cache->slots[i];
    // 记录清理信息到日志
    INFO(NCCL_INIT, "Cleanup buffer %p pages %lx", (void*)reg->begAddr, (reg->endAddr-reg->begAddr)/cache->pageSize);
    // 清理该注册的所有资源
    NCCLCHECK(regCleanup(comm, reg));
    // 释放注册记录结构体
    free(reg);
  }
  // 释放槽位数组
  free(cache->slots);
  // 返回成功状态码
  return ncclSuccess;
}

// NCCL API 声明宏
// 定义 ncclCommRegister 函数为公开的 NCCL API
NCCL_API(ncclResult_t, ncclCommRegister, const ncclComm_t comm, void* buff, size_t size, void** handle);
// 函数功能：注册内存缓冲区（公开 API）
// 用户调用此函数来注册内存，以便 NCCL 可以直接访问
// 参数说明：
//   - comm: 通信句柄
//   - buff: 要注册的缓冲区地址
//   - size: 缓冲区大小
//   - handle: 输出参数，返回注册句柄
// 返回值：ncclSuccess 表示成功
ncclResult_t ncclCommRegister(const ncclComm_t comm, void* buff, size_t size, void** handle) {
  // 检查是否启用了本地注册参数
  if (!ncclParamLocalRegister())
    // 未启用注册，返回 NULL 句柄
    *handle = NULL;
  else
    // 启用注册，调用内部注册函数
    // false: 表示非图注册（本地注册）
    NCCLCHECK(ncclRegister(comm, buff, size, false, handle));
  // 返回成功状态码
  return ncclSuccess;
}

// 函数功能：在 CUDA Graph 模式下注册内存缓冲区
// 与 ncclCommRegister 类似，但用于 CUDA Graph 捕获
// 参数说明：
//   - comm: 通信句柄
//   - buff: 要注册的缓冲区地址
//   - size: 缓冲区大小
//   - handle: 输出参数，返回注册句柄
// 返回值：ncclSuccess 表示成功
ncclResult_t ncclCommGraphRegister(const ncclComm_t comm, void* buff, size_t size, void** handle) {
  // 调用内部注册函数，isGraph=true 表示图注册
  NCCLCHECK(ncclRegister(comm, buff, size, true, handle));
  // 返回成功状态码
  return ncclSuccess;
}

// 静态函数：注销内存缓冲区（内部函数）
// 参数说明：
//   - comm: 通信上下文指针
//   - isGraph: 是否为图注册
//   - reg: 注册记录指针（要注销的句柄）
// 返回值：ncclSuccess 表示成功
static ncclResult_t commDeregister(struct ncclComm *comm, bool isGraph, struct ncclReg* reg) {
  // 检查 comm 指针是否有效
  NCCLCHECK(CommCheck(comm, "ncclCommRegister", "comm"));
  // 获取注册缓存指针
  struct ncclRegCache* cache = &comm->regCache;
  // 声明槽位索引变量
  int slot;
  // 声明保存当前设备的变量
  int saveDev;
  // 如果句柄为 NULL，直接退出
  if (reg == NULL) goto exit;
  // 获取当前 CUDA 设备
  CUDACHECK(cudaGetDevice(&saveDev));
  // 切换到 comm 关联的 CUDA 设备
  CUDACHECK(cudaSetDevice(comm->cudaDev));
  // 在缓存中查找注册记录
  // 从头开始遍历，直到找到匹配的记录或到达末尾
  for (slot = 0; slot < cache->population && cache->slots[slot] != reg; slot++);
  // 检查是否找到了注册记录
  if (slot == cache->population) {
    // 未找到注册记录
    WARN("Deregister: Could not find handle");
    // 返回无效使用错误
    return ncclInvalidUsage;
  }
  // 根据注册类型减少相应的引用计数
  if (isGraph) --reg->graphRefs;
  // 图注册引用计数减 1
  else --reg->localRefs;
  // 本地注册引用计数减 1
  // 检查是否还有引用
  // 如果还有引用（本地或图），则不释放资源
  if (reg->localRefs || reg->graphRefs) return ncclSuccess;
  // 没有引用了，清理所有注册资源
  NCCLCHECK(regCleanup(comm, reg));
  // 释放注册记录结构体
  free(reg);
  // 从缓存中移除该槽位
  // memmove: 将后面的槽位向前移动，填补空缺
  memmove(cache->slots + slot, cache->slots + slot + 1, (cache->population - slot - 1) * sizeof(struct ncclReg*));
  // 减少缓存人口数量
  cache->population -= 1;
  // 恢复原来的 CUDA 设备
  CUDACHECK(cudaSetDevice(saveDev));
// 退出标签
exit:
  // 返回成功状态码
  return ncclSuccess;
}

// NCCL API 声明宏
// 定义 ncclCommDeregister 函数为公开的 NCCL API
NCCL_API(ncclResult_t, ncclCommDeregister, const ncclComm_t comm, void* handle);
// 函数功能：注销内存缓冲区（公开 API）
// 用户调用此函数来释放之前注册的内存
// 参数说明：
//   - comm: 通信句柄
//   - handle: 注册句柄（由 ncclCommRegister 返回）
// 返回值：ncclSuccess 表示成功
ncclResult_t ncclCommDeregister(const ncclComm_t comm, void *handle) {
  // 调用内部注销函数，isGraph=false 表示本地注销
  NCCLCHECK(commDeregister(comm, false, (struct ncclReg*)handle));
  // 返回成功状态码
  return ncclSuccess;
}

// 函数功能：在 CUDA Graph 模式下注销内存缓冲区
// 与 ncclCommDeregister 类似，但用于 CUDA Graph
// 参数说明：
//   - comm: 通信句柄
//   - handle: 注册句柄（由 ncclCommGraphRegister 返回）
// 返回值：ncclSuccess 表示成功
ncclResult_t ncclCommGraphDeregister(const ncclComm_t comm, struct ncclReg *handle) {
  // 调用内部注销函数，isGraph=true 表示图注销
  NCCLCHECK(commDeregister(comm, true, handle));
  // 返回成功状态码
  return ncclSuccess;
}
