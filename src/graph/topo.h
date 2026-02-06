/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// 防止头文件重复包含的宏保护
#ifndef NCCL_TOPO_H_
#define NCCL_TOPO_H_

// 引入 NCCL 图相关的公共接口声明
#include "graph.h"
// 引入 NCCL 核心数据结构和定义
#include "core.h"
// 引入 XML 解析相关的数据结构和函数
#include "xml.h"
// 引入网络层相关的接口和定义
#include "net.h"

// ============================================================================
// 带宽常量定义 (单位: GB/s)
// ============================================================================

// 本地互连带宽 (用于同一 PCIe 交换机内部的连接)
#define LOC_BW 5000.0

// NVLink 带宽定义，根据 GPU 计算能力 (SM 版本) 不同而不同
#define SM60_NVLINK_BW 18.0    // Pascal P100: 18 GB/s
#define SM70_NVLINK_BW 20.0    // Volta V100: 20 GB/s
#define SM80_NVLINK_BW 20.0    // Ampere A100: 20 GB/s
#define SM90_NVLINK_BW 20.6    // Hopper H100: 20.6 GB/s
#define SM86_NVLINK_BW 12.0    // Ampere A30/A40: 12 GB/s
#define SM100_NVLINK_BW 40.1   // Blackwell B100/GB200: 40.1 GB/s

// PCIe 带宽定义 (Gen3 x16 的理论带宽)
#define PCI_BW 12.0            // PCI Gen3 x16

// CPU 间互连带宽定义
#define AMD_BW 16.0            // AMD CPU Infinity Fabric 带宽
#define BDW_QPI_BW 6.0         // Intel Broadwell QPI 带宽
#define SKL_QPI_BW 10.0        // Intel Skylake QPI 带宽
#define SRP_QPI_BW 22.0        // Intel Skylake Refresh QPI 带宽
#define ERP_QPI_BW 40.0        // Intel Ice Lake UPI 带宽
#define ZPI_BW 6.0             // Zhaoxin CPU 互连带宽
#define YONGFENG_ZPI_BW 9.0    // Zhaoxin Yongfeng CPU 互连带宽
#define P9_BW 32.0             // IBM POWER9 NX 带宽
#define ARM_BW 6.0             // ARM CPU CCIX/CAPI 带宽
#define NET_BW 12.0            // 网络带宽基准值 (100Gbit ≈ 12GB/s)

// Intel CPU 会将 GPU P2P 流量转换为 64B PCI TLP 格式，导致 GPU-GPU 通信消耗更多带宽
// 这个宏计算 Intel CPU 的 P2P 额外开销 (带宽 × 1.2)
#define INTEL_P2P_OVERHEAD(bw) (bw*6/5)

// ============================================================================
// 节点类型定义
// ============================================================================

// 拓扑图中的节点类型总数
#define NCCL_TOPO_NODE_TYPES 6

// 各节点类型的枚举定义
#define GPU 0                 // GPU 设备节点
#define PCI 1                 // PCIe 交换芯片/桥接器节点
#define NVS 2                 // NVLink 交换机节点
#define CPU 3                 // CPU 节点 (实际表示 NUMA 域)
#define NIC 4                 // 网卡节点 (本节点的物理网卡)
#define NET 5                 // 网络节点 (跨机器的远程网络设备)

// 节点类型名称字符串数组 (在 topo.cc 中定义)
extern const char* topoNodeTypeStr[];

// ============================================================================
// 链接类型定义
// ============================================================================

// 链接类型与路径类型尽可能保持一致，以便于路径计算和比较
#define LINK_LOC 0            // 本地连接 (同一设备内部)
#define LINK_NVL 1            // NVLink 连接
// 跳过 2，对应 PATH_NVB
#define LINK_C2C 3            // Chip-to-Chip 连接
#define LINK_PCI 4            // PCIe 连接
// 跳过 5-8，对应 PATH_PXB, PATH_PXN, PATH_P2C, PATH_PHB
#define LINK_SYS 9            // 系统互连 (QPI/UPI/NUMA)
#define LINK_NET 10           // 网络连接

// 链接类型名称字符串数组 (在 topo.cc 中定义)
extern const char* topoLinkTypeStr[];

// ============================================================================
// 路径类型定义 (按性能从高到低排序，用于选择最优通信路径)
// ============================================================================

// 路径类型按性能递减排序，用于选择最优通信路径
// Local (myself)
// 本地通信 (自己到自己)
#define PATH_LOC 0            // 本地路径 (同一设备)

// 通过nvlink
// Connection traversing NVLink
#define PATH_NVL 1            // NVLink 路径 (直接 NVLink 互连)

// 通过NVLink经中间GPU
// Connection through NVLink using an intermediate GPU
#define PATH_NVB 2            // NVBridge 路径 (通过中间 GPU 转发)

// 通过Chip-to-Chip
// Connection through C2C
#define PATH_C2C 3            // C2C 路径 (Chip-to-Chip 互连)

// 通过单个PCIe桥
// Connection traversing at most a single PCIe bridge
#define PATH_PIX 4            // PIX 路径 (仅通过一个 PCIe 桥)

// 通过多个PCIe桥
// Connection traversing multiple PCIe bridges (without traversing the PCIe Host Bridge)
#define PATH_PXB 5            // PXB 路径 (通过多个 PCIe 桥，但不经过 Host Bridge)

// GPU通过C2C到CPU再到NIC
// Connection between a GPU and a NIC using the C2C connection to the CPU and the PCIe connection to the NIC
#define PATH_P2C 6            // P2C 路径 (GPU → C2C → CPU → NIC)

// 通过中间GPU到NIC
// Connection between a GPU and a NIC using an intermediate GPU. Used to enable rail-local, aggregated network send/recv operations.
#define PATH_PXN 7            // PXN 路径 (GPU → 中间 GPU → NIC，用于 rail-local 聚合网络通信)

// 通过PCIe Host Bridge(CPU)
// Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)
#define PATH_PHB 8            // PHB 路径 (经过 PCIe Host Bridge，即 CPU)

// 通过NUMA互连(QPI/UPI)
// Connection traversing PCIe as well as the SMP interconnect between NUMA nodes (e.g., QPI/UPI)
#define PATH_SYS 9            // SYS 路径 (经过 NUMA 互连，如 QPI/UPI)

// 通过网络连接
// Connection through the network
#define PATH_NET 10           // NET 路径 (通过网络设备)

// 新的路径类型别名，应该优先于 PATH_PIX
#define PATH_PORT PATH_NVL     // PORT 路径 (等同于 NVLink，用于端口级别的路径选择)

// 断开连接/不可达
#define PATH_DIS 11           // DIS 路径 (断开连接，无路径)

// 路径类型名称字符串数组 (在 topo.cc 中定义)
extern const char* topoPathTypeStr[];

// 声明 PXN C2C 参数获取函数
extern int64_t ncclParamPxnC2c();

// ============================================================================
// 数据结构定义
// ============================================================================

// 前向声明，用于在 ncclTopoLink 中引用 ncclTopoNode
struct ncclTopoNode;

// 单个链接结构 (用于邻接表，记录直接相连的节点)
struct ncclTopoLink {
  int type;                   // 链接类型 (LINK_LOC/LINK_NVL/LINK_PCI 等)
  float bw;                   // 链接带宽 (GB/s)
  struct ncclTopoNode* remNode;  // 指向远程节点的指针 (链路另一端的节点)
};

// 链接和节点的最大数量定义
// GB200-NVL72 系统中，单节点最多可以有 32 个 NIC，因此需要大量链接槽位
#define NCCL_TOPO_MAX_LINKS 576                 // 每个节点最大链接数
#define NCCL_TOPO_MAX_HOPS (NCCL_TOPO_MAX_NODES*NCCL_TOPO_NODE_TYPES)  // 最大跳数 (576×6=3456)

// 路径列表结构 (用于缓存从源节点到目标节点的完整路径)
struct ncclTopoLinkList {
  struct ncclTopoLink* list[NCCL_TOPO_MAX_HOPS];  // 路径上每一步的链接指针数组
  int count;                   // 路径跳数 (经过多少个节点)
  float bw;                    // 路径总带宽 (瓶颈链路的带宽)
  int type;                    // 路径类型 (路径上最差的链路类型)
};

// 未定义的标记值
#define NCCL_TOPO_UNDEF (-1)

// ============================================================================
// 节点 ID 编码/解码宏
// ============================================================================
// 节点 ID 是一个 64 位整数，编码格式:
// [63:56] = systemId (8 bits, 最多支持 256 个节点)
// [55:0]  = localId  (56 bits, 节点在系统内的本地标识，如 busId)

#define NCCL_TOPO_ID_LOCAL_ID_MASK 0x00ffffffffffffff  // 本地 ID 的掩码 (低 56 位)
#define NCCL_TOPO_ID_SYSTEM_ID(id) (id >> 56)          // 从节点 ID 中提取 systemId
#define NCCL_TOPO_ID_LOCAL_ID(id) (id & NCCL_TOPO_ID_LOCAL_ID_MASK)  // 从节点 ID 中提取 localId
#define NCCL_TOPO_LOCAL_NIC_ID(numaid, busid) (((int64_t)numaid << 56) + busid)  // 编码 NIC ID (numaid 在高 8 位)
#define NCCL_TOPO_ID(systemid, localid) (((int64_t)systemid << 56) + (localid & NCCL_TOPO_ID_LOCAL_ID_MASK))  // 编码完整节点 ID

// 拓扑节点结构 (表示图中的一个顶点)
struct ncclTopoNode {
 //节点类型,gpu,cpu等
  int type;                   // 节点类型 (GPU/PCI/NVS/CPU/NIC/NET)
  int64_t id;                 // 节点唯一标识 (编码了 systemId 和 localId)

  // Type specific data
  //根据type区分数据
  // 使用联合体存储不同类型节点的特有属性，节省内存空间
  union {
    // GPU 节点特有属性
    struct {
      int dev;               // NVML 设备号 (物理 GPU 编号)
      int rank;              // NCCL rank 号 (通信域中的逻辑编号)
      int cudaCompCap;       // CUDA 计算能力 (如 80 表示 SM 8.0)
      int gdrSupport;        // 是否支持 GPUDirect RDMA
    }gpu;

    // 网络 (NIC/NET) 节点特有属性
    struct {
      int dev;               // 插件设备号
      uint64_t asic;         // ASIC ID (网卡的硬件标识)
      int port;              // 端口号
      float bw;              // 网络带宽 (GB/s)
      float latency;         // 网络延迟 (微秒)
      int gdrSupport;        // 是否支持 GPUDirect RDMA
      int collSupport;       // 是否支持集合通信网络
      int maxChannels;       // 最大通道数
      int localGpu;          // 关联的本地 GPU (-1 表示无关联)
    }net;

    // CPU 节点特有属性
    struct {
      int arch;              // CPU 架构 (x86_64/arm64/ppc64)
      int vendor;            // CPU 厂商 (Intel/AMD/...)
      int model;             // CPU 型号
      cpu_set_t affinity;    // CPU 亲和性掩码 (标识哪些 CPU 核属于此 NUMA 节点)
    }cpu;

    // PCIe 节点特有属性
    struct {
      uint64_t device;       // PCIe 设备 ID (BDF 格式编码)
    }pci;
  };

  //链路数量
  int nlinks;                // 该节点的直接链接数量 (邻接表大小)

  //links[] 按带宽降序排列
  struct ncclTopoLink links[NCCL_TOPO_MAX_LINKS];  // 邻接表数组，记录所有直接相连的节点

  // Pre-computed paths to GPUs and NICs
  //存储到其他NODE类型的路径
  // paths[] 是一个指针数组，每个元素指向一个 ncclTopoLinkList 数组
  // paths[type] 指向该节点到所有 type 类型节点的路径缓存
  struct ncclTopoLinkList* paths[NCCL_TOPO_NODE_TYPES];

  // Used during search
  uint64_t used;             // 标记位 (在图搜索算法中标记节点是否已访问)
};

// 拓扑节点集合结构 (管理同一类型的所有节点)
struct ncclTopoNodeSet {
    //当前nodes节点数量
  int count;                 // 该类型节点的实际数量
    //每个节点下有多少个node
  struct ncclTopoNode nodes[NCCL_TOPO_MAX_NODES];  // 节点数组 (预分配最大空间)
};

//包含系统拓扑所有信息，比如路径等
// 拓扑系统结构 (整个集群拓扑的根结构)
struct ncclTopoSystem {
   //系统id
  int systemId;              // 当前节点在多节点系统中的 ID (用于标识自己是哪个节点)
   //存储主机的hash值，最大576个
  uint64_t hostHashes[NCCL_TOPO_MAX_NODES];  // 所有节点的 host hash (用于识别不同物理节点)
   //主机的实际数量
  int nHosts;                // 系统中的物理节点总数

  //节点类型，值为6
  struct ncclTopoNodeSet nodes[NCCL_TOPO_NODE_TYPES];  // 6 种类型的节点集合 (GPU/PCI/NVS/CPU/NIC/NET)

  //统计gpu的最大带宽
  float maxBw;               // 系统中 GPU 的最大单链路带宽
  //统计gpu总带宽，nvlink是累加
  float totalBw;             // 系统中 GPU 的总带宽 (所有 NVLink 带宽累加)
};

// ============================================================================
// 函数声明
// ============================================================================

// 获取指定类型和 ID 的节点
ncclResult_t ncclTopoGetNode(struct ncclTopoSystem* system, struct ncclTopoNode** node, int type, uint64_t id);

// 创建新节点
ncclResult_t ncclTopoCreateNode(struct ncclTopoSystem* system, struct ncclTopoNode** node, int type, uint64_t id);

// 移除节点
ncclResult_t ncclTopoRemoveNode(struct ncclTopoSystem* system, int type, int id);

// 连接两个节点 (建立双向链接)
ncclResult_t ncclTopoConnectNodes(struct ncclTopoNode* node, struct ncclTopoNode* remNode, int type, float bw);

// 打印路径信息 (用于调试)
ncclResult_t ncclTopoPrintPaths(struct ncclTopoSystem* system);

// 从 XML 文件加载系统拓扑
ncclResult_t ncclTopoLoadSystem(const char* xmlTopoFile, struct ncclTopoSystem* system);

// 获取中间 rank (用于 PXN 路径)
ncclResult_t ncclTopoGetIntermediateRank(struct ncclTopoSystem* system, int rank, int64_t netId, int* intermediateRank);

// 获取 GPU 到指定类型节点的最小路径类型
ncclResult_t ncclTopoGetGpuMinPath(struct ncclTopoSystem* system, int type, int* min);

// 获取 GPU 到指定类型节点的最大路径类型
ncclResult_t ncclTopoGetGpuMaxPath(struct ncclTopoSystem* system, int type, int* max);

// 分割 NVLink (用于负载均衡)
ncclResult_t ncclTopoSplitNvLink(struct ncclTopoSystem* system, int* splitNvLink);

// 网络信息结构 (传递网络插件参数)
struct ncclTopoNetInfo {
  bool coll;                 // 是否为集合通信网络
  // communicator-specific information
  int netPluginIndex;        // 网络插件索引
  bool dmaBufSupport;        // 是否支持 DMA buffer
  // NIC fusion
  int mergeLevel;            // NIC 融合级别
  const char* forceMerge;    // 强制融合策略
  // dev count tracking functions (not part of ncclNet)
  ncclResult_t (*getDevCount)(int, int*, int*);  // 获取设备数量的函数指针
  ncclResult_t (*setVirtDevCount)(int, int);      // 设置虚拟设备数量的函数指针
  // ncclNet API functions
  const char* name;          // 网络名称
  ncclResult_t (*getProperties)(int, ncclNetProperties_t*);  // 获取网络属性的函数指针
  ncclResult_t (*makeVDevice)(int*, ncclNetVDeviceProps_t*); // 创建虚拟设备的函数指针
  ncclResult_t (*devices)(int*);  // 获取设备列表的函数指针
};

// 处理网络设备 (添加 NIC 到 XML 拓扑)
ncclResult_t ncclTopoProcessNet(ncclXml* xml, const char* dumpXmlFile, struct ncclTopoNetInfo* net);

// 获取 NIC 融合环境变量
ncclResult_t ncclTopoGetFusionEnv(int* mergeLevel, const char** forceMerge);

// XML 和 Graph 的最大节点数定义
#define NCCL_TOPO_XML_MAX_NODES 256     // XML 拓扑的最大节点数
#define NCCL_GRAPH_XML_MAX_NODES 4096   // Graph XML 的最大节点数

// 从 XML 构建系统拓扑
ncclResult_t ncclTopoGetSystemFromXml(struct ncclXml* xml, struct ncclTopoSystem** topoSystem, uint64_t localHostHash);

// 从 XML 构建通信图
ncclResult_t ncclTopoGetGraphFromXml(struct ncclXmlNode *xmlGraphs, struct ncclTopoSystem* system, struct ncclTopoGraph* graph, int* nChannels);

// 从 Graphs 生成 XML
ncclResult_t ncclTopoGetXmlFromGraphs(int ngraphs, struct ncclTopoGraph** graphs, struct ncclTopoSystem* system, struct ncclXml *xml);

// 获取计算能力范围
ncclResult_t ncclTopoGetCompCap(struct ncclTopoSystem* system, int* ccMin, int* ccMax);

// ============================================================================
// 内联辅助函数
// ============================================================================

// 将节点 ID 转换为节点索引 (在 nodes[type].nodes 数组中的下标)
static ncclResult_t ncclTopoIdToIndex(struct ncclTopoSystem* system, int type, int64_t id, int* index) {
  *index = -1;  // 初始化为 -1 (表示未找到)
  // 遍历指定类型的所有节点
  for (int i=0; i<system->nodes[type].count; i++) {
    // 比较 ID 是否匹配
    if (system->nodes[type].nodes[i].id == id) {
      *index = i;  // 找到匹配的节点，记录索引
      return ncclSuccess;
    }
  }
  // 遍历完未找到，返回错误
  return ncclInternalError;
}

// 将 rank 号转换为节点索引 (仅用于 GPU 类型)
static ncclResult_t ncclTopoRankToIndex(struct ncclTopoSystem* system, int rank, int* index, bool showWarn) {
  *index = -1;  // 初始化为 -1 (表示未找到)
  //遍历gpu
  for (int i=0; i<system->nodes[GPU].count; i++) {
    //比较rank
    if (system->nodes[GPU].nodes[i].gpu.rank == rank) {
        //返回index
      *index = i;  // 找到匹配的 GPU，记录索引
      return ncclSuccess;
    }
  }

  // 如果未找到且需要显示警告，则打印警告信息
  if (showWarn)
    WARN("ncclTopoRankToIndex could not find rank %d", rank);
  return ncclInternalError;
}

// 将设备号转换为 rank 号 (仅用于 GPU 类型)
static ncclResult_t ncclTopoDevToRank(struct ncclTopoSystem* system, int dev, int* rank) {
  *rank = -1;  // 初始化为 -1 (表示未找到)
  // 遍历所有 GPU
  for (int i=0; i<system->nodes[GPU].count; i++) {
    // 只考虑当前节点的 GPU (通过 systemId 过滤)
    if (NCCL_TOPO_ID_SYSTEM_ID(system->nodes[GPU].nodes[i].id) != system->systemId) continue; // Only consider GPUs on our node
    // 比较 NVML 设备号
    if (system->nodes[GPU].nodes[i].gpu.dev == dev) {
      *rank = system->nodes[GPU].nodes[i].gpu.rank;  // 找到匹配，返回 rank
      return ncclSuccess;
    }
  }
  // 未找到设备
  return ncclInternalError;
}

// 声明 NIC 路径类型字典 (用于环境变量解析)
extern struct kvDict nicPathKvList[];

// 将网络节点 ID 转换为网络设备号
static ncclResult_t ncclTopoIdToNetDev(struct ncclTopoSystem* system, int64_t id, int* netDev) {
  *netDev = -1;  // 初始化为 -1 (表示未找到)
  // 遍历所有 NET 类型节点
  for (int i=0; i<system->nodes[NET].count; i++) {
    // 比较 ID
    if (system->nodes[NET].nodes[i].id == id) {
      *netDev = system->nodes[NET].nodes[i].net.dev;  // 找到匹配，返回设备号
      return ncclSuccess;
    }
  }
  // 未找到，打印警告
  WARN("Could not find NET with id %lx", id);
  return ncclInternalError;
}

// Returns NVLink bw in GB/s
// 根据 CUDA 计算能力返回 NVLink 带宽
static float ncclTopoNVLinkBw(int cudaCompCap) {
  // 根据计算能力 (SM 版本) 返回对应的 NVLink 带宽
  return
    cudaCompCap >= 100 ? SM100_NVLINK_BW :  // Blackwell (B100/GB200)
    cudaCompCap >= 90  ? SM90_NVLINK_BW  :  // Hopper (H100)
    cudaCompCap == 86 ? SM86_NVLINK_BW  :  // Ampere (A30/A40)
    cudaCompCap >= 80 ? SM80_NVLINK_BW  :  // Ampere (A100)
    cudaCompCap >= 70 ? SM70_NVLINK_BW  :  // Volta (V100)
    cudaCompCap >= 60 ? SM60_NVLINK_BW  :  // Pascal (P100)
    SM80_NVLINK_BW;                        // 默认值 (A100 带宽)
}

// Mirror bits
// 位镜像辅助函数 (用于 NVLink 拓扑优化)
// 检查一个数是否为 2 的幂次方
static bool isPow2(int val) {
  // 利用位运算特性：2 的幂次方二进制表示只有一个 1
  // 例如: 8 (1000) & 7 (0111) = 0
  return (val & (val-1)) == 0;
}

// 镜像位函数 (用于优化 NVLink 连接的拓扑映射)
static int mirrorBits(int val, int pow2) {
  int mirror = 0;  // 镜像后的值
  // 从低位到高位遍历，将位顺序反转
  // 例如: pow2=8 时，val=5 (00000101) → mirror=160 (10100000)
  for (int b=1, mb=(pow2>>1); b<pow2; b<<=1, mb>>=1)
    if (val & b) mirror |= mb;
  return mirror;
}

// 结束头文件保护宏
#endif
