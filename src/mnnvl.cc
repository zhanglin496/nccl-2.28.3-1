/*************************************************************************
 * Copyright (c) 2015-2024, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// MNNVL (Multi-Node NVLink) 功能检测和初始化
// MNNVL 允许跨节点的 GPU 通过 NVLink 网络进行直接通信，提供更高的带宽和更低的延迟

#include "mnnvl.h"           // MNNVL 相关定义
#include "transport.h"        // 传输层接口定义，包含 peerInfo 等结构
#include <cuda.h>             // CUDA 驱动 API
#include "cudawrap.h"         // CUDA API 封装和 cuMem 相关函数

// 检测系统是否支持 MNNVL (Multi-Node NVLink) 功能
// 输入参数: comm - NCCL 通信域结构体指针
// 返回值: ncclSuccess 成功, ncclSystemError 系统错误
ncclResult_t ncclMnnvlCheck(struct ncclComm* comm) {
  // ========== 第一步: 检查 cuMem 功能是否启用 ==========
  // MNNVL 依赖于 CUDA 的 cuMem API (CUDA Unified Memory Management)
  // 如果 cuMem 未启用，则无法使用 MNNVL，直接返回成功（MNNVL 不会被使用）
  if (!ncclCuMemEnable()) 
    return ncclSuccess;

  // ========== 第二步: 检查设备是否支持 FABRIC 句柄类型 ==========
  // FABRIC 句柄是 CUDA 用于跨节点内存共享的机制
  int cudaDev;               // CUDA 设备 ID
  int flag = 0;              // 用于存储设备属性查询结果
  CUdevice currentDev;       // CUdevice 设备句柄

  // 获取当前 CUDA 设备 ID
  CUDACHECK(cudaGetDevice(&cudaDev));

  // 根据 CUDA 设备 ID 获取 CUdevice 句柄
  CUCHECK(cuDeviceGet(&currentDev, cudaDev));

  // 查询设备是否支持 FABRIC 句柄类型
  // CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED: 查询是否支持 FABRIC 内存句柄
  // CUPFN 宏用于调用动态加载的 CUDA 驱动函数
  // 注意: 如果驱动不支持此属性，函数会失败，这里用 (void) 忽略错误
  (void) CUPFN(cuDeviceGetAttribute(&flag, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED, currentDev));

  // 如果设备不支持 FABRIC 句柄，MNNVL 无法使用，直接返回成功
  if (!flag) 
    return ncclSuccess;

  // ========== 第三步: 检查所有 rank 的 Fabric 状态是否已完成 ==========
  // MNNVL 要求所有通信的 GPU 都已完成 Fabric 初始化
  // Fabric 状态包括:
  //   - NVML_GPU_FABRIC_STATE_NOT_SUPPORTED: 不支持
  //   - NVML_GPU_FABRIC_STATE_NOT_STARTED: 未启动
  //   - NVML_GPU_FABRIC_STATE_IN_PROGRESS: 初始化中
  //   - NVML_GPU_FABRIC_STATE_COMPLETED: 已完成
  for (int i = 0; i < comm->nRanks; i++) {
    // peerInfo[i] 包含第 i 个 rank 的 GPU 信息
    // fabricInfo.state 存储 Fabric 初始化状态
    if (comm->peerInfo[i].fabricInfo.state != NVML_GPU_FABRIC_STATE_COMPLETED) 
        return ncclSuccess;
  }

  // ========== 第四步: 确定 MNNVL 域(clique/派系) ==========
  // Clique 是指通过 NVLink Fabric 互连的一组 GPU
  // 只有在同一个 clique 内的 GPU 才能使用 MNNVL 进行通信

  // 分配内存用于存储属于同一 clique 的所有 rank 编号
  // clique.ranks 数组将存储所有在同一 clique 中的全局 rank ID
  NCCLCHECK(ncclCalloc(&comm->clique.ranks, comm->nRanks));

  // 获取当前 rank 所在 clique 的 ID
  // cliqueId 标识一个特定的 NVLink Fabric 域
  comm->clique.id = comm->peerInfo[comm->rank].fabricInfo.cliqueId;

  // 遍历所有 rank，找出属于同一 clique 的 rank
  for (int i = 0; i < comm->nRanks; i++) {
    // 获取当前 rank 和第 i 个 rank 的 Fabric 信息
    nvmlGpuFabricInfoV_t *fabricInfo1 = &comm->peerInfo[comm->rank].fabricInfo;
    nvmlGpuFabricInfoV_t *fabricInfo2 = &comm->peerInfo[i].fabricInfo;

    // ========== 检查 cluster UUID 和 cliqueId 是否匹配 ==========
    // clusterUuid 是 16 字节的 UUID，标识一个 GPU 集群
    // 如果 UUID 全为 0，说明没有有效的 Fabric 信息

    unsigned long uuid0 = 0;  // UUID 的前 8 字节
    unsigned long uuid1 = 0;  // UUID 的后 8 字节

    // 从 clusterUuid 的前 8 字节提取 uuid0
    memcpy(&uuid0, fabricInfo2->clusterUuid, sizeof(uuid0));

    // 从 clusterUuid 的后 8 字节提取 uuid0
    memcpy(&uuid1, fabricInfo2->clusterUuid + sizeof(uuid0), sizeof(uuid1));

    // 如果 UUID 全为 0，说明没有 MNNVL fabric 信息，禁用 MNNVL
    if ((uuid0 | uuid1) == 0) 
        return ncclSuccess;

    // 比较两个 GPU 的 cluster UUID 和 clique ID
    // memcmp 返回 0 表示两个 UUID 相同
    // clusterUuid 相同: 表示在同一个物理集群中
    // cliqueId 相同: 表示在同一个 NVLink 域中
    if ((memcmp(fabricInfo1->clusterUuid, fabricInfo2->clusterUuid, NVML_GPU_FABRIC_UUID_LEN) == 0) &&
        (fabricInfo1->cliqueId == fabricInfo2->cliqueId)) {

      // 如果是当前 rank 自己，记录在 clique 中的本地 rank
      if (i == comm->rank) {
        // cliqueRank 是当前 rank 在 clique 中的本地编号
        comm->cliqueRank = comm->clique.size;
      }

      // 将该 rank 添加到 clique 的 ranks 数组中
      comm->clique.ranks[comm->clique.size++] = i;
    }
  }

  // ========== 第五步: 检查 clique 大小 ==========
  // 如果 clique 中只有 1 个或更少的 GPU，MNNVL 没有意义
  // MNNVL 需要至少 2 个 GPU 才能进行通信
  if (comm->clique.size <= 1) 
    return ncclSuccess;

  // ========== 第六步: 验证 FABRIC 句柄的导入/导出功能 ==========
  // 通过实际分配和测试 FABRIC 内存，验证 IMEX (Import/Export) 通道是否正常工作
  // IMEX 通道配置在 /dev/nvidia-caps-imex-channels
  {
    void *ptr = NULL;                    // 分配的内存指针
    CUmemGenericAllocationHandle handle; // CUDA 内存分配句柄
    ncclCuDesc cuDesc;                   // CUDA 描述符，用于跨进程共享
    CUresult err;                        // CUDA 错误码

    // 分配 FABRIC 句柄兼容的内存
    // CU_MEM_HANDLE_TYPE_FABRIC: 使用 FABRIC 句柄类型
    // CUDA_IPC_MIN: 最小 IPC 内存大小 (2MB)
    ncclResult_t ret = ncclCuMemAlloc(&ptr, &handle, CU_MEM_HANDLE_TYPE_FABRIC, CUDA_IPC_MIN);

    if (ret != ncclSuccess) {
      // 如果分配失败，说明系统虽然是 MNNVL capable，但 FABRIC 句柄不支持
      // 这通常是由于 IMEX 通道配置问题
      WARN("MNNVL (cliqueSize %d) is available but not working on this system. Check the IMEX channel configuration (/dev/nvidia-caps-imex-channels). Set NCCL_MNNVL_ENABLE=0 to ignore this issue.",
           comm->clique.size);
      return ncclSystemError;
    }

    // 测试导出 FABRIC 句柄到可共享的描述符
    // cuMemExportToShareableHandle: 将内存句柄导出为可跨进程共享的形式
    err = CUPFN(cuMemExportToShareableHandle(&cuDesc, handle, CU_MEM_HANDLE_TYPE_FABRIC, 0));

    // 测试从共享描述符导入 FABRIC 句柄
    // 这是验证 IMEX 通道是否正常工作的关键测试
    if (err != CUDA_SUCCESS ||
        (err = CUPFN(cuMemImportFromShareableHandle(&handle, &cuDesc, CU_MEM_HANDLE_TYPE_FABRIC))) != CUDA_SUCCESS) {

      const char *errStr;
      // 获取 CUDA 错误字符串
      (void) pfn_cuGetErrorString(err, &errStr);

      // 释放测试分配的内存
      NCCLCHECK(ncclCuMemFree(ptr));

      // 导入/导出失败，说明 IMEX 配置有问题
      // 可能需要使用 nvidia-imex-ctl 工具检查配置
      WARN("MNNVL (cliqueSize %d) is available but not working on this system. Check the IMEX configuration (nvidia-imex-ctl -N). Set NCCL_MNNVL_ENABLE=0 to ignore this issue.",
          comm->clique.size);
      return ncclSystemError;
    }

    // 释放测试分配的内存
    NCCLCHECK(ncclCuMemFree(ptr));

    // ========== 第七步: 启用 MNNVL ==========
    // 强制使用 FABRIC 句柄类型作为 cuMem 分配的默认类型
    // ncclCuMemHandleType 是全局变量，控制后续所有 cuMem 分配使用的句柄类型
    ncclCuMemHandleType = CU_MEM_HANDLE_TYPE_FABRIC;

    // 标记该通信域启用了 MNNVL
    comm->MNNVL = 1;

    // 输出 MNNVL 初始化信息日志
    // MNNVL: 是否启用 (1)
    // cliqueId: clique ID (十六进制)
    // cliqueSize: clique 中的 GPU 数量
    // cliqueRank: 当前 rank 在 clique 中的编号
    INFO(NCCL_INIT, "MNNVL %d cliqueId %x cliqueSize %d cliqueRank %d",
        comm->MNNVL, comm->clique.id, comm->clique.size, comm->cliqueRank);
  }

  // 成功完成 MNNVL 检测和初始化
  return ncclSuccess;
}
