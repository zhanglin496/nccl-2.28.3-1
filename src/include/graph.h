/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 * 版权所有 (c) 2016-2022, NVIDIA CORPORATION. 保留所有权利。
 *
 * See LICENSE.txt for license information
 * 请参阅 LICENSE.txt 获取许可证信息
 ************************************************************************/

// 防止头文件重复包含的保护宏开始
// 如果未定义 NCCL_GRAPH_H_ 宏，则定义它
#ifndef NCCL_GRAPH_H_
#define NCCL_GRAPH_H_

// 引入 NCCL 核心头文件，包含基本的类型定义和 API 接口
#include "nccl.h"
// 引入设备相关头文件，包含 GPU 设备操作的接口定义
#include "device.h"
// 引入标准库头文件，定义了一些系统限制常量（如 PATH_MAX 等）
#include <limits.h>
// 引入标准库头文件，提供内存分配、进程控制等通用工具函数
#include <stdlib.h>
// 引入字符处理头文件，提供字符分类和转换函数（如 isdigit, toupper 等）
#include <ctype.h>
// 引入标准输入输出头文件，提供文件操作和格式化 I/O 函数
#include <stdio.h>
// 引入 CPU 调度相关头文件，提供 CPU 亲和性设置（cpu_set_t）和调度函数
#include <sched.h>

// 函数声明：获取指定 CUDA 设备的系统路径
// 参数：
//   cudaDev - CUDA 设备编号
//   path - 输出参数，返回设备路径字符串指针的指针
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
ncclResult_t ncclTopoCudaPath(int cudaDev, char** path);

// 前向声明：ncclTopoSystem 结构体
// 这个结构体用于表示整个系统的拓扑信息，包括 GPU、网卡、交换机等设备及其连接关系
// 具体定义在其他源文件中，这里只做前向声明以支持指针类型
struct ncclTopoSystem;

// Build the topology
// 构建系统拓扑
// 函数声明：获取并构建系统拓扑结构
// 参数：
//   comm - NCCL 通信上下文指针
//   system - 输出参数，返回构建好的拓扑系统结构指针
//   dumpXmlFile - 可选参数，指定将拓扑信息导出的 XML 文件路径，默认为 NULL 不导出
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
ncclResult_t ncclTopoGetSystem(struct ncclComm* comm, struct ncclTopoSystem** system, const char* dumpXmlFile=NULL);

// 函数声明：对系统拓扑进行排序
// 此函数用于对拓扑中的节点进行排序优化，以便后续的路径计算和通信模式选择
// 参数：
//   system - 拓扑系统结构指针
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
ncclResult_t ncclTopoSortSystem(struct ncclTopoSystem* system);

// 函数声明：打印系统拓扑信息
// 用于调试和日志输出，将拓扑结构以可读格式打印出来
// 参数：
//   system - 拓扑系统结构指针
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
ncclResult_t ncclTopoPrint(struct ncclTopoSystem* system);

// 函数声明：计算系统拓扑中所有节点之间的通信路径
// 此函数是拓扑分析的核心，用于确定最优的通信路径
// 参数：
//   system - 拓扑系统结构指针
//   comm - NCCL 通信上下文指针
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
ncclResult_t ncclTopoComputePaths(struct ncclTopoSystem* system, struct ncclComm* comm);

// 函数声明：释放拓扑系统结构占用的内存
// 用于清理和释放 ncclTopoGetSystem 分配的所有资源
// 参数：
//   system - 要释放的拓扑系统结构指针
// 返回值：无（void 类型函数）
void ncclTopoFree(struct ncclTopoSystem* system);

// 函数声明：裁剪系统拓扑，移除与当前通信无关的节点
// 根据通信上下文中的 rank 信息，只保留相关的拓扑节点，优化后续计算
// 参数：
//   system - 拓扑系统结构指针
//   comm - NCCL 通信上下文指针
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
ncclResult_t ncclTopoTrimSystem(struct ncclTopoSystem* system, struct ncclComm* comm);

// 函数声明：计算点对点（P2P）通信的通道信息
// 为 GPU 之间的直接通信建立必要的通道连接
// 参数：
//   comm - NCCL 通信上下文指针
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
ncclResult_t ncclTopoComputeP2pChannels(struct ncclComm* comm);

// 函数声明：获取指定 rank 的 NVLink 连接的 GPU 列表
// NVLink 是 NVIDIA 的高速互连技术，此函数查找通过 NVLink 连接的所有 GPU
// 参数：
//   system - 拓扑系统结构指针
//   rank - 要查询的进程 rank 编号
//   nranks - 输出参数，返回找到的 rank 数量
//   ranks - 输出参数，返回分配的 rank 数组指针，需要调用者释放
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
ncclResult_t ncclTopoGetNvbGpus(struct ncclTopoSystem* system, int rank, int* nranks, int** ranks);

// 函数声明：检查所有 GPU 之间的路径是否都使用 NVLink
// 用于判断系统是否支持全 NVLink 连接，这对通信性能有重要影响
// 参数：
//   system - 拓扑系统结构指针
//   allNvLink - 输出参数，返回布尔值（非零表示所有路径都是 NVLink，零表示存在非 NVLink 路径）
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
ncclResult_t ncclTopoPathAllNVLink(struct ncclTopoSystem* system, int* allNvLink);

// 函数声明：计算通信域中各 rank 对应的 CPU 信息
// 用于确定每个 rank 应该绑定到哪个 CPU 核心以优化性能
// 参数：
//   comm - NCCL 通信上下文指针
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
ncclResult_t ncclTopoComputeCommCPU(struct ncclComm* comm);

// Query topology
// 查询拓扑信息
// 函数声明：获取指定 rank 和通道的网络设备信息
// 用于查找某个 rank 在特定通道上使用的网络设备
// 参数：
//   comm - NCCL 通信上下文指针
//   rank - 要查询的 rank 编号
//   graph - 拓扑图结构指针
//   channelId - 通道编号
//   peerRank - 对端 rank 编号
//   id - 输出参数，返回网络设备的 ID（如 IB 设备的 GUID）
//   dev - 输出参数，返回网络设备索引
//   proxyRank - 输出参数，返回代理 rank 编号（如果需要代理转发）
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
ncclResult_t ncclTopoGetNetDev(struct ncclComm* comm, int rank, struct ncclTopoGraph* graph, int channelId, int peerRank, int64_t* id, int* dev, int* proxyRank);

// 函数声明：检查两个 rank 之间是否支持点对点（P2P）通信
// P2P 通信指 GPU 之间直接通过 PCIe 或 NVLink 进行数据传输，无需通过主机内存
// 参数：
//   comm - NCCL 通信上下文指针
//   system - 拓扑系统结构指针
//   rank1 - 第一个 rank 编号
//   rank2 - 第二个 rank 编号
//   p2p - 输出参数，返回是否支持 P2P（1=支持，0=不支持）
//   read - 输出参数，返回是否支持 P2P 读取（1=支持，0=不支持）
//   intermediateRank - 输出参数，如果需要中继，返回中继 rank 的编号
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
ncclResult_t ncclTopoCheckP2p(struct ncclComm* comm, struct ncclTopoSystem* system, int rank1, int rank2, int* p2p, int *read, int* intermediateRank);

// 函数声明：检查是否支持多节点 NVLink（Multi-Node NVLink）连接
// MNNVL 是跨节点的 NVLink 互连技术，需要特定硬件支持
// 参数：
//   system - 拓扑系统结构指针
//   info1 - 第一个节点的对等信息结构指针
//   info2 - 第二个节点的对等信息结构指针
//   ret - 输出参数，返回是否支持 MNNVL（1=支持，0=不支持）
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
ncclResult_t ncclTopoCheckMNNVL(struct ncclTopoSystem* system, struct ncclPeerInfo* info1, struct ncclPeerInfo* info2, int* ret);
// 枚举类型：GPU Direct RDMA（GPUDirect RDMA）模式
// GPU Direct RDMA 允许网络设备直接访问 GPU 内存，无需通过主机内存中转
enum ncclTopoGdrMode {
  ncclTopoGdrModeDisable = 0,    // 禁用 GPU Direct RDMA 模式
  ncclTopoGdrModeDefault = 1,    // 默认模式，由 NCCL 自动判断是否启用
  ncclTopoGdrModePci = 2,        // PCI 模式，强制使用 PCI 方式的 GPU Direct RDMA
  ncclTopoGdrModeNum = 3         // 模式数量标记，用于边界检查
};

// 函数声明：检查是否支持 GPU Direct RDMA（GPUDirect）
// GPU Direct 允许网卡直接读写 GPU 内存，显著提升性能
// 参数：
//   topo - 拓扑系统结构指针
//   rank - 要检查的 rank 编号
//   netId - 网络设备 ID
//   read - 是否检查读操作（非零表示检查读，零表示检查写）
//   gdrMode - 输出参数，返回 GPU Direct 模式
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
ncclResult_t ncclTopoCheckGdr(struct ncclTopoSystem* topo, int rank, int64_t netId, int read, enum ncclTopoGdrMode* gdrMode);

// 函数声明：检查是否需要执行 cache flush 操作
// 某些网络设备需要刷新 GPU cache 以确保数据一致性
// 参数：
//   comm - NCCL 通信上下文指针
//   netId - 网络设备 ID
//   netDev - 网络设备索引
//   rank - rank 编号
//   flush - 输出参数，返回是否需要 flush（非零表示需要，零表示不需要）
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
ncclResult_t ncclTopoNeedFlush(struct ncclComm* comm, int64_t netId, int netDev, int rank, int* flush);

// 函数声明：检查 GPU Direct RDMA 是否可用
// 参数：
//   system - 拓扑系统结构指针
//   rank - 要检查的 rank 编号
//   avail - 输出参数，返回 GPU Direct 是否可用（true=可用，false=不可用）
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
ncclResult_t ncclTopoIsGdrAvail(struct ncclTopoSystem* system, int rank, bool *avail);

// 函数声明：检查两个 rank 之间是否需要通过网络设备通信
// 参数：
//   system - 拓扑系统结构指针
//   rank1 - 第一个 rank 编号
//   rank2 - 第二个 rank 编号
//   net - 输出参数，返回是否需要网络通信（非零表示需要，零表示不需要）
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
ncclResult_t ncclTopoCheckNet(struct ncclTopoSystem* system, int rank1, int rank2, int* net);

// 函数声明：检查是否应该禁用 PXN（Proxy eXtension for NVLink）功能
// PXN 是一种通过代理节点扩展 NVLink 连接的技术
// 参数：
//   comm - NCCL 通信上下文指针
// 返回值：非零表示应该禁用 PXN，零表示不禁用
int ncclPxnDisable(struct ncclComm* comm);

// 函数声明：获取 PXN 模式下的中间代理 rank 列表
// 参数：
//   comm - NCCL 通信上下文指针
//   intermediateRanks - 输出参数，返回分配的代理 rank 数组指针，需要调用者释放
//   nranks - 输出参数，返回代理 rank 的数量
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
ncclResult_t ncclTopoGetPxnRanks(struct ncclComm* comm, int** intermediateRanks, int* nranks);

// 函数声明：获取指定 GPU 关联的本地 CPU 编号
// 用于 CPU 亲和性设置，将线程绑定到正确的 CPU 核心
// 参数：
//   system - 拓扑系统结构指针
//   gpu - GPU 设备编号
//   retCpu - 输出参数，返回对应的 CPU 编号
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
ncclResult_t ncclGetLocalCpu(struct ncclTopoSystem* system, int gpu, int* retCpu);

// 函数声明：获取用户设置的 P2P（点对点）级别
// 用户可以通过环境变量控制 P2P 通信的级别，影响通信策略选择
// 参数：
//   level - 输出参数，返回 P2P 级别值
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
ncclResult_t ncclGetUserP2pLevel(int* level);

// Find CPU affinity
// 查找 CPU 亲和性信息
// 函数声明：获取指定 rank 的 CPU 亲和性设置
// CPU 亲和性指进程/线程应该在哪些 CPU 核心上运行，以优化性能
// 参数：
//   system - 拓扑系统结构指针
//   rank - rank 编号
//   affinity - 输出参数，返回 CPU 亲和性掩码（cpu_set_t 类型，使用位图表示可用的 CPU 核心）
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
ncclResult_t ncclTopoGetCpuAffinity(struct ncclTopoSystem* system, int rank, cpu_set_t* affinity);

// CPU 架构类型宏定义：x86 架构（Intel 和 AMD 处理器）
#define NCCL_TOPO_CPU_ARCH_X86 1
// CPU 架构类型宏定义：POWER 架构（IBM Power 处理器）
#define NCCL_TOPO_CPU_ARCH_POWER 2
// CPU 架构类型宏定义：ARM 架构（ARM 处理器）
#define NCCL_TOPO_CPU_ARCH_ARM 3
// CPU 架构类型宏定义：混合架构（系统中存在多种架构）
#define NCCL_TOPO_CPU_ARCH_MIXED 4

// CPU 厂商标识宏定义：Intel 厂商
#define NCCL_TOPO_CPU_VENDOR_INTEL 1
// CPU 厂商标识宏定义：AMD 厂商
#define NCCL_TOPO_CPU_VENDOR_AMD 2
// CPU 厂商标识宏定义：Zhaoxin（兆芯）厂商
#define NCCL_TOPO_CPU_VENDOR_ZHAOXIN 3
// CPU 厂商标识宏定义：混合厂商（系统中存在多种厂商）
#define NCCL_TOPO_CPU_VENDOR_MIXED 4

// Intel CPU 型号宏定义：Broadwell 微架构（第 5 代 Core 处理器）
#define NCCL_TOPO_CPU_MODEL_INTEL_BDW 1
// Intel CPU 型号宏定义：Skylake 微架构（第 6-9 代 Core 处理器）
#define NCCL_TOPO_CPU_MODEL_INTEL_SKL 2
// Intel CPU 型号宏定义：Sapphire Rapids（第 4 代 Xeon 可扩展处理器）
#define NCCL_TOPO_CPU_MODEL_INTEL_SRP 3
// Intel CPU 型号宏定义：Emerald Rapids（第 5 代 Xeon 可扩展处理器）
#define NCCL_TOPO_CPU_MODEL_INTEL_ERP 4
// CPU 型号宏定义：Yongfeng（海光处理器型号）
#define NCCL_TOPO_CPU_MODEL_YONGFENG 1

// 函数声明：获取系统的 CPU 类型信息（架构、厂商、型号）
// 参数：
//   system - 拓扑系统结构指针
//   arch - 输出参数，返回 CPU 架构类型（使用上述 NCCL_TOPO_CPU_ARCH_* 宏）
//   vendor - 输出参数，返回 CPU 厂商（使用上述 NCCL_TOPO_CPU_VENDOR_* 宏）
//   model - 输出参数，返回 CPU 型号（使用上述 NCCL_TOPO_CPU_MODEL_* 宏）
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
ncclResult_t ncclTopoCpuType(struct ncclTopoSystem* system, int* arch, int* vendor, int* model);

// 函数声明：获取系统中 GPU 的总数
// 参数：
//   system - 拓扑系统结构指针
//   count - 输出参数，返回 GPU 数量
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
ncclResult_t ncclTopoGetGpuCount(struct ncclTopoSystem* system, int* count);

// 函数声明：获取系统中网络设备（网卡）的总数
// 参数：
//   system - 拓扑系统结构指针
//   count - 输出参数，返回网络设备数量
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
ncclResult_t ncclTopoGetNetCount(struct ncclTopoSystem* system, int* count);

// 函数声明：获取系统中 NVSwitch 设备的数量
// NVSwitch 是 NVIDIA 的高速交换芯片，用于连接多个 GPU
// 参数：
//   system - 拓扑系统结构指针
//   count - 输出参数，返回 NVSwitch 设备数量
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
ncclResult_t ncclTopoGetNvsCount(struct ncclTopoSystem* system, int* count);

// 函数声明：获取指定 rank 和通道的本地网络设备信息
// 参数：
//   system - 拓扑系统结构指针
//   rank - rank 编号
//   channelId - 通道编号
//   id - 输出参数，返回网络设备 ID
//   dev - 输出参数，返回网络设备索引
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
ncclResult_t ncclTopoGetLocalNet(struct ncclTopoSystem* system, int rank, int channelId, int64_t* id, int* dev);

// 函数声明：根据网络设备 ID 查找关联的本地 GPU
// 参数：
//   system - 拓扑系统结构指针
//   netId - 网络设备 ID
//   gpuIndex - 输出参数，返回 GPU 索引
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
ncclResult_t ncclTopoGetLocalGpu(struct ncclTopoSystem* system, int64_t netId, int* gpuIndex);

// 函数声明：根据带宽获取本地网络设备数量
// 返回具有足够带宽的网络设备数量，用于通信路径选择
// 参数：
//   system - 拓扑系统结构指针
//   gpu - GPU 设备编号
//   count - 输出参数，返回符合条件的网络设备数量
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
ncclResult_t getLocalNetCountByBw(struct ncclTopoSystem* system, int gpu, int *count);

// Allows for up to 32 NICs per node on GB200-NVL72
// 允许在 GB200-NVL72 系统中每个节点最多支持 32 个网卡（NIC）
// 最大节点数计算：18 个 GPU 节点 × 32 NIC = 576，再加上交换机等设备
#define NCCL_TOPO_MAX_NODES 576

// 函数声明：获取本地拓扑节点信息
// 这是一个通用的拓扑查询函数，可以获取不同类型的本地节点信息
// 参数：
//   system - 拓扑系统结构指针
//   type - 要查询的节点类型（如 GPU、NET、NIC 等）
//   index - 节点索引
//   resultType - 结果类型，指定返回信息的格式
//   locals - 输出参数，返回本地节点数组，最多 NCCL_TOPO_MAX_NODES 个
//   localCount - 输出参数，返回找到的本地节点数量
//   pathType - 输出参数，返回路径类型（如 P2P、NET、SHM 等）
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
ncclResult_t ncclTopoGetLocal(struct ncclTopoSystem* system, int type, int index, int resultType, int locals[NCCL_TOPO_MAX_NODES], int* localCount, int* pathType);

// Init search. Needs to be done before calling ncclTopoCompute
// 初始化搜索功能。在调用 ncclTopoCompute 之前必须完成此初始化
// 此函数为拓扑计算搜索算法准备必要的数据结构和状态
// 参数：
//   system - 拓扑系统结构指针
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
ncclResult_t ncclTopoSearchInit(struct ncclTopoSystem* system);

// 拓扑模式枚举：平衡树模式（BALANCED TREE）
// Spread NIC traffic between two GPUs (Tree parent + one child on first GPU, second child on second GPU)
// 将网卡流量分散到两个 GPU（树的父节点和一个子节点在第一个 GPU，另一个子节点在第二个 GPU）
// 这种模式可以平衡网络负载，避免单个 GPU 成为瓶颈
#define NCCL_TOPO_PATTERN_BALANCED_TREE 1

// 拓扑模式枚举：分裂树模式（SPLIT TREE）
// Spread NIC traffic between two GPUs (Tree parent on first GPU, tree children on the second GPU)
// 将网卡流量分散到两个 GPU（树的父节点在第一个 GPU，所有子节点在第二个 GPU）
// 这种模式进一步分离父节点和子节点的流量
#define NCCL_TOPO_PATTERN_SPLIT_TREE 2

// 拓扑模式枚举：树模式（TREE）
// All NIC traffic going to/from the same GPU
// 所有网卡流量都进出同一个 GPU
// 这种模式简单但可能导致单个 GPU 成为网络瓶颈
#define NCCL_TOPO_PATTERN_TREE 3

// 拓扑模式枚举：环形模式（RING）
// Ring
// 环形通信模式，每个节点只与相邻的两个节点通信
// 适合大规模集群，具有良好的可扩展性
#define NCCL_TOPO_PATTERN_RING 4

// 拓扑模式枚举：NVLS 模式（NVLink fabric + SHARP/Tree）
// NVLS+SHARP and NVLS+Tree
// NVLS（NVLink fabric）结合 SHARP 或 Tree 的混合模式
// 适合使用 NVLink Switch 的系统，提供高带宽低延迟通信
#define NCCL_TOPO_PATTERN_NVLS 5

// 拓扑模式枚举：CollNet 直连模式（COLLNET DIRECT）
// Collnet Direct
// 直接使用集合网络（CollNet）进行通信，绕过 GPU 之间的直接连接
// 适合有专用集合网络硬件的系统
#define NCCL_TOPO_PATTERN_COLLNET_DIRECT 6
// 拓扑图结构体：描述 NCCL 通信的拓扑连接和性能参数
// 此结构体包含了通信路径的完整信息，用于算法选择和性能优化
struct ncclTopoGraph {
  // Input / output
  // 输入/输出参数标记

  // 拓扑图 ID，标识使用的通信算法类型
  // ring : 0 - 环形算法
  // tree : 1 - 树形算法
  // collnet : 2 - 集合网络算法
  // nvls : 3 - NVLink fabric 算法
  // collnetDirect : 4 - 集合网络直连算法
  int id;

  // 拓扑模式，使用上述 NCCL_TOPO_PATTERN_* 宏定义的值
  // 指定通信图的连接模式（如平衡树、分裂树、环形、NVLS 等）
  int pattern;

  // 是否跨网卡（cross-NIC）的标记
  // 非零表示允许使用多个网卡来分散网络流量
  // 这可以提高带宽利用率，但可能增加延迟
  int crossNic;

  // 是否使用集合网络（CollNet）的标记
  // 非零表示使用专用的集合网络硬件进行通信
  // 集合网络可以卸载部分通信操作到网络设备
  int collNet;

  // 最小通道数
  // 指定使用的最小通信通道数量，影响并行度
  // 更多通道可以提高带宽利用率，但也增加开销
  int minChannels;

  // 最大通道数
  // 指定允许使用的最大通信通道数量
  // 实际使用的通道数在 minChannels 和 maxChannels 之间选择
  int maxChannels;

  // Output
  // 输出参数，以下字段由拓扑计算函数填充

  // 实际使用的通道数量
  // 在 minChannels 和 maxChannels 之间，根据系统拓扑和性能要求确定
  int nChannels;

  // 节点内（Intra-node）带宽（GB/s）
  // 同一节点内 GPU 之间通信的可用带宽
  float bwIntra;

  // 节点间（Inter-node）带宽（GB/s）
  // 不同节点之间 GPU 通信的可用带宽
  float bwInter;

  // 节点间通信延迟（微秒）
  // 跨节点通信的延迟开销，用于小消息性能预测
  float latencyInter;

  // 节点内通信类型
  // 表示节点内使用的传输类型（如 SHM、P2P、NVLink 等）
  int typeIntra;

  // 节点间通信类型
  // 表示节点间使用的传输类型（如 IB、Network、NVLS 等）
  int typeInter;

  // 是否所有通道使用相同的通信路径
  // 非零表示所有通道使用相同的拓扑结构，零表示通道可能使用不同的路径
  int sameChannels;

  // 通信跳数（hop count）
  // 数据传输需要经过的中转次数，较少的跳数通常意味着更低的延迟
  int nHops;

  // 节点内通信路径数组
  // 存储每个通道在节点内的通信路径信息
  // 数组大小：MAXCHANNELS * NCCL_TOPO_MAX_NODES
  // 每个元素表示通道的目的节点或中继节点
  int intra[MAXCHANNELS*NCCL_TOPO_MAX_NODES];

  // 节点间通信路径数组
  // 存储每个通道在节点间的通信路径信息
  // 数组大小：MAXCHANNELS * 2（每个通道可能有两个方向）
  // 使用 64 位整数以支持更大的设备 ID 范围（如 InfiniBand 的 64 位 GUID）
  int64_t inter[MAXCHANNELS*2];
};
// 函数声明：计算最优拓扑图
// 这是 NCCL 拓扑系统的核心函数，根据系统硬件拓扑计算最优的通信路径
// 参数：
//   system - 拓扑系统结构指针，包含硬件拓扑信息
//   graph - 输入/输出参数，输入包含约束条件（如 minChannels, maxChannels），输出包含计算结果
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
ncclResult_t ncclTopoCompute(struct ncclTopoSystem* system, struct ncclTopoGraph* graph);

// 函数声明：打印拓扑图信息
// 用于调试和日志输出，以可读格式打印拓扑图的详细信息
// 参数：
//   system - 拓扑系统结构指针
//   graph - 要打印的拓扑图结构指针
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
ncclResult_t ncclTopoPrintGraph(struct ncclTopoSystem* system, struct ncclTopoGraph* graph);

// 函数声明：导出多个拓扑图到文件
// 用于保存拓扑计算结果，便于后续分析和调试
// 参数：
//   system - 拓扑系统结构指针
//   ngraphs - 拓扑图数量
//   graphs - 拓扑图数组指针
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
ncclResult_t ncclTopoDumpGraphs(struct ncclTopoSystem* system, int ngraphs, struct ncclTopoGraph** graphs);

// 拓扑 rank 关系结构体：存储每个 rank 在不同通信模式下的连接关系
// 此结构体描述了每个 rank 在环形、树形、NVLS 等通信模式中的对等节点信息
struct ncclTopoRanks {
  // 跨网卡环形标记
  // 非零表示环形通信跨越多个网卡以分散负载
  int crossNicRing;

  // 环形接收方向的 rank 数组
  // ringRecv[i] 表示在通道 i 上，当前 rank 从哪个 rank 接收数据
  // 环形通信中，每个 rank 从前驱节点接收数据
  int ringRecv[MAXCHANNELS];

  // 环形发送方向的 rank 数组
  // ringSend[i] 表示在通道 i 上，当前 rank 向哪个 rank 发送数据
  // 环形通信中，每个 rank 向后继节点发送数据
  int ringSend[MAXCHANNELS];

  // 环形前驱节点 rank 数组
  // ringPrev[i] 表示在通道 i 上，当前 rank 的前驱节点（接收来源）
  // 与 ringRecv 类似，但在语义上更明确表示前驱关系
  int ringPrev[MAXCHANNELS];

  // 环形后继节点 rank 数组
  // ringNext[i] 表示在通道 i 上，当前 rank 的后继节点（发送目标）
  // 与 ringSend 类似，但在语义上更明确表示后继关系
  int ringNext[MAXCHANNELS];

  // 树形父节点 rank 数组
  // treeToParent[i] 表示在通道 i 上，当前 rank 的父节点 rank
  // 在树形算法中，非根节点需要向父节点发送数据（reduce）或从父节点接收数据（broadcast）
  int treeToParent[MAXCHANNELS];

  // 树形第一个子节点 rank 数组
  // treeToChild0[i] 表示在通道 i 上，当前 rank 的第一个子节点 rank
  // 在树形算法中，父节点需要与子节点通信
  int treeToChild0[MAXCHANNELS];

  // 树形第二个子节点 rank 数组
  // treeToChild1[i] 表示在通道 i 上，当前 rank 的第二个子节点 rank
  // 二叉树中每个节点最多有两个子节点
  int treeToChild1[MAXCHANNELS];

  // NVLS 头节点 rank 数组
  // nvlsHeads[i] 表示在通道 i 上，NVLS 通信的头节点 rank
  // NVLS（NVLink fabric）模式中，头节点负责协调通信
  int nvlsHeads[MAXCHANNELS];

  // NVLS 头节点数量
  // 表示系统中 NVLS 头节点的总数
  int nvlsHeadNum;
};

// 函数声明：拓扑预设（Preset）处理
// 在实际通信之前设置拓扑相关的配置和连接关系
// 参数：
//   comm - NCCL 通信上下文指针
//   graphs - 拓扑图数组指针，包含多个候选拓扑图
//   topoRanks - 输出参数，返回计算得到的 rank 关系结构指针
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
ncclResult_t ncclTopoPreset(struct ncclComm* comm, struct ncclTopoGraph** graphs, struct ncclTopoRanks* topoRanks);

// 函数声明：拓扑后置（Postset）处理
// 在拓扑预设完成后的后续处理，包括建立连接、分配资源等
// 参数：
//   comm - NCCL 通信上下文指针
//   firstRanks - 每个节点的第一个 rank 数组
//   treePatterns - 树形模式数组，描述每个通道使用的树形模式
//   allTopoRanks - 所有 rank 的拓扑关系数组
//   rings - 环形模式相关的信息
//   graphs - 拓扑图数组指针
//   parent - 父通信上下文指针（用于嵌套通信场景）
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
ncclResult_t ncclTopoPostset(struct ncclComm* comm, int* firstRanks, int* treePatterns,
    struct ncclTopoRanks** allTopoRanks, int* rings, struct ncclTopoGraph** graphs, struct ncclComm* parent);

// 函数声明：初始化调优器常量
// 为性能调优模型初始化必要的常量参数
// 参数：
//   comm - NCCL 通信上下文指针
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
ncclResult_t ncclTopoInitTunerConstants(struct ncclComm* comm);

// 函数声明：调优性能模型
// 根据实际系统性能调优拓扑图的性能模型参数
// 参数：
//   comm - NCCL 通信上下文指针
//   minCompCap - 最小计算能力值
//   maxCompCap - 最大计算能力值
//   graphs - 拓扑图数组指针，将被调优
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
ncclResult_t ncclTopoTuneModel(struct ncclComm* comm, int minCompCap, int maxCompCap, struct ncclTopoGraph** graphs);

// 函数声明：获取指定算法和协议的预计执行时间
// 用于性能预测和算法选择，返回在给定数据量和操作数下的预计时间
// 参数：
//   comm - NCCL 通信上下文指针
//   coll - 集合操作类型（如 AllReduce、Broadcast 等）
//   algorithm - 算法类型（如 Tree、Ring 等）
//   protocol - 协议类型（如 LL、LL128、Simple 等）
//   nBytes - 数据大小（字节数）
//   numPipeOps - 流水线操作数量
//   time - 输出参数，返回预计的执行时间（微秒）
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
ncclResult_t ncclTopoGetAlgoTime(struct ncclComm* comm, int coll, int algorithm, int protocol, size_t nBytes, int numPipeOps, float* time);

// 头文件保护结束宏
// 与开头的 #ifndef NCCL_GRAPH_H_ 配对，防止头文件重复包含
#endif
