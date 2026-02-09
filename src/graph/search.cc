/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 * 版权所有 (c) 2016-2022, NVIDIA CORPORATION. 保留所有权利。
 *
 * See LICENSE.txt for license information
 * 查看 LICENSE.txt 获取许可证信息
 ************************************************************************/

// 引入通信相关定义和结构体
#include "comm.h"
// 引入核心功能和常量定义
#include "core.h"
// 引入图拓扑相关结构体和函数
#include "graph.h"
// 引入拓扑发现和管理相关功能
#include "topo.h"
// 引入传输层接口定义
#include "transport.h"
// 引入XML配置导入导出功能
#include "xml.h"
// 引入数学库函数
#include <math.h>

// 定义跨网卡(CROSS_NIC)参数，默认值为2
// 用于控制是否使用多个网卡进行跨节点通信
NCCL_PARAM(CrossNic, "CROSS_NIC", 2);

// Initialize system->maxBw. This is the per-channel (i.e. per-SM)
// max bw.
// 初始化 system->maxBw。这是每个通道（即每个SM）的最大带宽
// 获取最大带宽函数
// 参数：
//   system: 拓扑系统结构体指针
//   gpu: GPU节点指针
//   type: 节点类型（GPU或NET）
// 返回：该GPU到指定类型节点的最大路径带宽
static float getMaxBw(struct ncclTopoSystem* system, struct ncclTopoNode* gpu, int type) {
  // 初始化最大带宽为0
  float maxBw = 0.0;
  // 遍历系统中所有指定类型的节点
  for (int i=0; i<system->nodes[type].count; i++) {
    // 获取从当前GPU到第i个类型节点的路径列表
    struct ncclTopoLinkList* path = gpu->paths[type]+i;
    // 获取该路径的带宽
    float bw = path->bw;
    // 如果路径长度为0（无路径），跳过
    if (path->count == 0)
        continue;
    // 更新最大带宽
    maxBw = std::max(maxBw, bw);
  }
  // 返回找到的最大带宽
  return maxBw;
}
// 获取总带宽函数
// 计算GPU节点的总带宽，取NVLink带宽和PCI带宽的最大值
// 参数：
//   system: 拓扑系统结构体指针
//   gpu: GPU节点指针
// 返回：GPU的总带宽
static float getTotalBw(struct ncclTopoSystem* system, struct ncclTopoNode* gpu) {
  // 初始化NVLink和PCI带宽
  float nvlinkBw = 0.0, pciBw = 0.0;
  // 遍历GPU节点的所有链路
  for (int l=0; l<gpu->nlinks; l++) {
    // 获取第l条链路
    struct ncclTopoLink* link = gpu->links+l;
    // 如果是NVLink链路，累加NVLink带宽
    if (link->type == LINK_NVL)
        nvlinkBw += link->bw;
    // 如果是PCI链路，记录PCI带宽（只取一条，因为是最大值）
    if (link->type == LINK_PCI)
        pciBw = link->bw;
  }
  // 返回NVLink总带宽和PCI带宽的最大值
  return std::max(pciBw, nvlinkBw);
}

// 拓扑搜索初始化函数
// 计算并设置拓扑系统的最大带宽和总带宽
// 参数：
//   system: 拓扑系统结构体指针
// 返回：操作结果状态码
ncclResult_t ncclTopoSearchInit(struct ncclTopoSystem* system) {
  // 初始化最大带宽为0
  system->maxBw = 0.0;
  // 初始化总带宽为0
  system->totalBw = 0.0;
  // 获取网络节点数量（用于判断是否跨节点通信）
  int inter = system->nodes[NET].count;
  //不是跨节点，只有1个gpu卡
  // 如果没有网络节点（单节点）且只有1个GPU
  if (inter == 0 && system->nodes[GPU].count == 1) {
    // 使用本地总线带宽作为最大带宽
    system->maxBw = LOC_BW;
    // 使用本地总线带宽作为总带宽
    system->totalBw = LOC_BW;
    // 返回成功
    return ncclSuccess;
  }

  // 遍历所有GPU节点
  for (int g=0; g<system->nodes[GPU].count; g++) {
    // 获取第g个GPU节点
    struct ncclTopoNode* gpu = system->nodes[GPU].nodes+g;
    //取最大值
    //如果有跨节点，统计网卡
    // 如果有跨节点（inter>0），则计算到网络节点的最大带宽；否则计算到其他GPU的最大带宽
    system->maxBw = std::max(system->maxBw, getMaxBw(system, gpu, inter ? NET : GPU));
    // 更新所有GPU中的最大总带宽
    system->totalBw = std::max(system->totalBw, getTotalBw(system, gpu));
  }
  // 返回成功
  return ncclSuccess;
}

// 计算通信器的CPU架构和厂商信息
// 参数：
//   comm: 通信器结构体指针
// 返回：操作结果状态码
ncclResult_t ncclTopoComputeCommCPU(struct ncclComm* comm) {
  // We assume there is at least one CPU and that the CPUs have the same
  // architecture and vendor.
  // 我们假设至少有一个CPU，并且所有CPU具有相同的架构和厂商
  // 获取CPU节点集合
  const struct ncclTopoNodeSet* cpus = &comm->topo->nodes[CPU];
  // 设置通信器的CPU架构为第一个CPU的架构
  comm->cpuArch = cpus->nodes[0].cpu.arch;
  // 设置通信器的CPU厂商为第一个CPU的厂商
  comm->cpuVendor = cpus->nodes[0].cpu.vendor;
  // 返回成功
  return ncclSuccess;
}

// 查找反向链路函数
// 在node2的所有链路中查找指向node1的指定类型的反向链路
// 参数：
//   node1: 源节点指针
//   node2: 目标节点指针
//   type: 链路类型
//   revLink: 输出参数，用于返回找到的反向链路指针
// 返回：操作结果状态码
static ncclResult_t findRevLink(struct ncclTopoNode* node1, struct ncclTopoNode* node2, int type, struct ncclTopoLink** revLink) {
  // 遍历node2的所有链路
  for (int l=0; l<node2->nlinks; l++) {
    // 获取第l条链路
    struct ncclTopoLink* link = node2->links+l;
    // 如果该链路的远程节点是node1且类型匹配
    if (link->remNode == node1 && link->type == type) {
      // 返回找到的反向链路
      *revLink = link;
      // 返回成功
      return ncclSuccess;
    }
  }
  // 未找到反向链路，输出警告信息
  WARN("Could not find rev link for %d/%ld -> %d/%ld", node1->type, node1->id, node2->type, node2->id);
  // 返回内部错误
  return ncclInternalError;
}

// This is unfortunately needed since manipulating floats often results in rounding errors.
// 这是必需的，因为浮点数运算经常会产生舍入误差
// 定义减法舍入宏，将结果舍入到3位小数以避免浮点误差
#define SUB_ROUND(a, b) (a = roundf((a-b)*1000)/1000)

// 沿路径跟踪并分配/释放带宽的内部函数
// 参数：
//   path: 链路路径列表
//   start: 起始节点
//   maxSteps: 最大跟踪步数
//   bw: 要分配的带宽（正数为分配，负数为释放）
//   steps: 输出参数，返回实际跟踪的步数
// 返回：操作结果状态码
static ncclResult_t followPath(struct ncclTopoLinkList* path, struct ncclTopoNode* start, int maxSteps, float bw, int* steps) {
  // 初始化PCI带宽为输入带宽
  float pciBw = bw;
  // 第一遍遍历：计算PCI带宽调整（针对Intel CPU的P2P低效问题）
  for (int step=0; step<path->count; step++) {
    // 获取当前步骤链路的远程节点
    struct ncclTopoNode* node = path->list[step]->remNode;
    // 如果远程节点是CPU
    if (node->type == CPU) {
      // Account for P2P inefficiency through Intel CPU RC
      // 考虑通过Intel CPU RC（根联合体）进行P2P通信的低效性
      // 如果路径类型是PHB（PCI Host Bridge），起始节点是GPU
      // 且CPU架构是x86，厂商是Intel
      if (path->type == PATH_PHB && start->type == GPU &&
          node->cpu.arch == NCCL_TOPO_CPU_ARCH_X86 &&
          node->cpu.vendor == NCCL_TOPO_CPU_VENDOR_INTEL) {
        // 应用Intel P2P开销系数降低带宽
        pciBw = INTEL_P2P_OVERHEAD(bw);
      }
    }
  }

  // 从起始节点开始
  struct ncclTopoNode* node = start;
  // 第二遍遍历：实际分配带宽并检查是否有足够带宽
  for (int step=0; step<maxSteps; step++) {
    // 获取当前步骤的链路
    struct ncclTopoLink* link = path->list[step];
    // 初始化反向链路为NULL
    struct ncclTopoLink* revLink = NULL;
    // 计算前向带宽：如果是PCI链路使用调整后的pciBw，否则使用原始bw
    float fwBw = link->type == LINK_PCI ? pciBw : bw;
    // 初始化反向带宽为0
    float revBw = 0;
    // 如果远程节点是计算能力小于8.0的GPU（旧架构），且起始节点不是GPU
    // 则需要考虑反向带宽开销（旧GPU的P2P反向通信有额外开销）
    if (link->remNode->type == GPU && link->remNode->gpu.cudaCompCap < 80 && start->type != GPU) {
      // 查找反向链路
      if (revLink == NULL) NCCLCHECK(findRevLink(node, link->remNode, link->type, &revLink));
      // 反向带宽为前向带宽的1/8
      revBw += fwBw/8;
    }
    // 如果远程节点是Power架构CPU且链路类型是NVLink
    // Power架构的NVLink需要考虑反向带宽
    if (link->remNode->type == CPU && link->remNode->cpu.arch == NCCL_TOPO_CPU_ARCH_POWER && link->type == LINK_NVL) {
      // 查找反向链路
      if (revLink == NULL) NCCLCHECK(findRevLink(node, link->remNode, link->type, &revLink));
      // 反向带宽等于前向带宽
      revBw += fwBw;
    }
    // Coverity thinks that revLink could be NULL below.  However, we access it only if revBw is non-0, and the
    // logic of the code is that revBw can become non-0 only if revLink is non-NULL (see the "if" statement right above).
    // coverity[var_deref_op]
    // Coverity认为revLink可能为NULL。但我们只在revBw非零时访问它，
    // 而代码逻辑是只有当revLink非NULL时revBw才会变为非零（见上面的if语句）
    // 如果前向带宽不足，或反向带宽不足（且有反向带宽要求），则提前返回
    if (link->bw < fwBw || (revBw && revLink->bw < revBw)) { *steps = step; return ncclSuccess; }
    // 从链路带宽中减去已使用的前向带宽
    SUB_ROUND(link->bw, fwBw);
    // 如果有反向带宽需求，从反向链路带宽中减去
    if (revBw) SUB_ROUND(revLink->bw, revBw);
    // 移动到下一个节点
    node = link->remNode;
  }
  // 返回实际跟踪的步数
  *steps = maxSteps;
  // 返回成功
  return ncclSuccess;
}

// Try to go from node type1/index1 to no type2/index2. mult indicates whether we are counting the bandwidth (1) or undoing (-1).
// 尝试从节点type1/index1到节点type2/index2。mult指示是分配带宽(1)还是释放带宽(-1)
// 参数：
//   system: 拓扑系统结构体指针
//   graph: 图结构指针
//   type1: 源节点类型（-1表示无源节点）
//   index1: 源节点索引
//   type2: 目标节点类型
//   index2: 目标节点索引
//   mult: 带宽乘数（1=分配，-1=释放）
//   node: 输出参数，返回到达的目标节点
// 返回：操作结果状态码
static ncclResult_t ncclTopoFollowPath(struct ncclTopoSystem* system, struct ncclTopoGraph* graph, int type1, int index1, int type2, int index2, float mult, struct ncclTopoNode** node) {
  // First handle easy cases
  // 首先处理简单情况
  // 默认设置目标节点
  *node = system->nodes[type2].nodes+index2;
  // 如果没有源节点（type1==-1），直接返回成功
  if (type1 == -1) return ncclSuccess;
  // 获取源节点
  struct ncclTopoNode* node1 = system->nodes[type1].nodes+index1;
  // 获取从源节点到目标节点的路径
  struct ncclTopoLinkList* path = node1->paths[type2]+index2;
  // 获取目标节点
  struct ncclTopoNode* node2 = system->nodes[type2].nodes+index2;
  // 获取反向路径（从目标节点回到源节点）
  struct ncclTopoLinkList* revPath = node2->paths[type1]+index1;

  // 检查路径是否存在
  if (path == NULL) {
    // 路径不存在，输出警告
    WARN("No path computed to go from %s/%d to %s/%d", topoNodeTypeStr[type1], index1, topoNodeTypeStr[type2], index2);
    // 返回内部错误
    return ncclInternalError;
  }

  // Now check link type
  // 现在检查链路类型
  // 重置目标节点为NULL（将在成功后设置）
  *node = NULL;
  // 判断是否是节点内通信（GPU或NVS之间的通信）
  int intra = (type1 == GPU || type1 == NVS) && (type2 == GPU || type2 == NVS);
  // 根据是否节点内通信选择对应的带宽
  float bw = intra ? graph->bwIntra : graph->bwInter;
  // 根据是否节点内通信选择对应的路径类型
  int type = intra ? graph->typeIntra : graph->typeInter;

  // 如果路径类型是断开连接或更差，直接返回成功
  if (path->type >= PATH_DIS) return ncclSuccess;
  // 如果正在分配带宽且路径类型超过了允许的类型，直接返回
  if (mult == 1 && (path->type > type)) return ncclSuccess;
  // 对于树形拓扑模式，需要同时检查正向和反向路径类型
  if (mult == 1 && (graph->pattern == NCCL_TOPO_PATTERN_BALANCED_TREE ||
        graph->pattern == NCCL_TOPO_PATTERN_TREE ||
        graph->pattern == NCCL_TOPO_PATTERN_SPLIT_TREE) &&
      (revPath->type > type)) return ncclSuccess;

  // 根据乘数调整带宽（正数=分配，负数=释放）
  bw *= mult;

  // Check there is enough bandwidth on paths.
  // 检查路径上是否有足够的带宽
  int step = 0;
  // 跟踪路径并尝试分配带宽
  NCCLCHECK(followPath(path, node1, path->count, bw, &step));
  // 如果步数小于路径长度，说明带宽不足
  if (step < path->count) goto rewind;

  // Enough bandwidth : return destination node.
  // 带宽足够：返回目标节点
  // 增加跳数计数（用于评估路径长度）
  graph->nHops += mult*path->count;
  // 设置返回的目标节点
  *node = system->nodes[type2].nodes+index2;
  // 返回成功
  return ncclSuccess;

rewind:
  // Not enough bandwidth : rewind and exit.
  // 带宽不足：回滚已分配的带宽并退出
  NCCLCHECK(followPath(path, node1, step, -bw, &step));
  return ncclSuccess;
}

// 获取GPU的PCI带宽
// 遍历GPU的所有PCI链路，找到并返回PCI链路的双向最小带宽
// 参数：
//   gpu: GPU节点指针
// 返回：PCI链路的最小带宽，如果没有PCI链路则返回-1
static int gpuPciBw(struct ncclTopoNode* gpu) {
  // 遍历GPU的所有链路
  for (int l=0; l<gpu->nlinks; l++) {
    // 获取第l条链路
    struct ncclTopoLink* gpuLink = gpu->links+l;
    // 如果不是PCI链路，跳过
    if (gpuLink->type != LINK_PCI) continue;
    // 获取PCI节点
    struct ncclTopoNode* pci = gpuLink->remNode;
    // 遍历PCI节点的所有链路
    for (int l=0; l<pci->nlinks; l++) {
      // 获取PCI节点的第l条链路
      struct ncclTopoLink* pciLink = pci->links+l;
      // 如果该链路不指向原GPU，跳过
      if (pciLink->remNode != gpu) continue;
      // 返回GPU到PCI和PCI到GPU链路带宽的最小值
      return std::min(gpuLink->bw, pciLink->bw);
    }
  }
  // 没有找到PCI链路，返回-1
  return -1;
}

/* Choose the order in which we try next GPUs. This is critical for the search
   to quickly converge to the best solution even if it eventually times out. */
/* 选择尝试下一个GPU的顺序。这对于搜索快速收敛到最优解至关重要，
   即使搜索最终超时也能找到较好的解 */
// GPU评分结构体，用于对GPU进行排序以选择最优连接顺序
struct ncclGpuScore {
  int g;             // GPU索引
  int startIndex;    // 起始索引（最不重要）
  int intraNhops;    // 节点内跳数（越少越好）
  int intraBw;       // 节点内带宽（越多越好）
  int interNhops;    // 节点间跳数（越少越好）
  int interPciBw;    // 节点间PCI带宽（越多越好）
  int interBw;       // 节点间带宽（最多越好，最重要）
};

// GPU评分比较函数，用于qsort排序
// 按重要性顺序比较：节点间带宽 > 节点间PCI带宽 > 节点间跳数 > 节点内带宽 > 节点内跳数 > 起始索引
static int cmpScore(const void * g1, const void * g2) {
   // 将指针转换为ncclGpuScore结构体指针
   struct ncclGpuScore *s1 = (struct ncclGpuScore*)g1;
   struct ncclGpuScore *s2 = (struct ncclGpuScore*)g2;
   int d;
   // 首先比较节点间带宽（越大的越好）
   if ((d = (s2->interBw - s1->interBw))) return d;
   // 其次比较节点间PCI带宽
   if ((d = (s2->interPciBw - s1->interPciBw))) return d;
   // 然后比较节点间跳数（越少的越好）
   if ((d = (s1->interNhops - s2->interNhops))) return d;
   // 再比较节点内带宽
   if ((d = (s2->intraBw - s1->intraBw))) return d;
   // 最后比较节点内跳数
   if ((d = (s1->intraNhops - s2->intraNhops))) return d;
   // 如果都相同，按起始索引排序
   return s1->startIndex - s2->startIndex;
}

// 比较所有GPU的节点内评分是否相同
// 参数：
//   scores: GPU评分数组
//   count: GPU数量
// 返回：0表示所有节点内评分相同，1表示不同
static int cmpIntraScores(struct ncclGpuScore* scores, int count) {
  // 获取第一个GPU的节点内带宽
  int intraBw = scores[0].intraBw;
  // 获取第一个GPU的节点内跳数
  int intraNhops = scores[0].intraNhops;
  // 遍历其余GPU
  for (int i=1; i<count; i++) {
    // 如果有GPU的节点内带宽或跳数不同，返回1
    if (scores[i].intraBw != intraBw || scores[i].intraNhops != intraNhops) return 1;
  }
  // 所有GPU的节点内评分都相同，返回0
  return 0;
}

// 根据rank号获取GPU索引
// 参数：
//   system: 拓扑系统结构体指针
//   rank: GPU的rank号
//   index: 输出参数，返回GPU索引
// 返回：操作结果状态码
static ncclResult_t getGpuIndex(struct ncclTopoSystem* system, int rank, int* index) {
  // 遍历所有GPU节点
  for (int g=0; g<system->nodes[GPU].count; g++) {
    // 如果找到rank匹配的GPU
    if (system->nodes[GPU].nodes[g].gpu.rank == rank) {
      // 返回GPU索引
      *index = g;
      // 返回成功
      return ncclSuccess;
    }
  }
  // 未找到，输出警告
  WARN("Could not find gpu rank %d", rank);
  // 返回内部错误
  return ncclInternalError;
}

// 根据ID获取网络节点索引
// 参数：
//   system: 拓扑系统结构体指针
//   id: 网络节点ID
//   index: 输出参数，返回网络节点索引
// 返回：操作结果状态码
static ncclResult_t getNetIndex(struct ncclTopoSystem* system, int64_t id, int* index) {
  // 遍历所有网络节点
  for (int n=0; n<system->nodes[NET].count; n++) {
    // 如果找到ID匹配的网络节点
    if (system->nodes[NET].nodes[n].id == id) {
      // 返回网络节点索引
      *index = n;
      // 返回成功
      return ncclSuccess;
    }
  }
  // 未找到，输出警告
  WARN("Could not find net id %lx", id);
  // 返回内部错误
  return ncclInternalError;
}

// 获取网络路径
// 从图结构中提取网络节点ID，并获取该网络节点到所有GPU的路径
// 参数：
//   system: 拓扑系统结构体指针
//   graph: 图结构指针
//   netPaths: 输出参数，返回网络到GPU的路径数组
// 返回：操作结果状态码
static ncclResult_t getNetPaths(struct ncclTopoSystem* system, struct ncclTopoGraph* graph, struct ncclTopoLinkList** netPaths) {
  // 从图的inter数组中获取当前通道的网络ID
  int64_t netId = graph->inter[graph->nChannels*2];
  int n;
  // 根据网络ID获取网络节点索引
  NCCLCHECK(getNetIndex(system, netId, &n));
  // 返回该网络节点到所有GPU的路径
  *netPaths=system->nodes[NET].nodes[n].paths[GPU];
  // 返回成功
  return ncclSuccess;
}

// 搜索并排序下一个可连接的GPU
// 根据带宽和跳数对候选GPU进行排序，优先选择带宽高、跳数少的GPU
// 参数：
//   system: 拓扑系统结构体指针
//   graph: 图结构指针
//   gpu: 当前GPU节点指针
//   next: 输出参数，返回排序后的GPU索引数组
//   countPtr: 输出参数，返回可用GPU数量
//   sortNet: 是否考虑网络路径排序（-1表示反向排序，0表示不排序，1表示排序）
// 返回：操作结果状态码
ncclResult_t ncclTopoSearchNextGpuSort(struct ncclTopoSystem* system, struct ncclTopoGraph* graph, struct ncclTopoNode* gpu, int* next, int* countPtr, int sortNet) {
  // 构造当前通道的标志位，用于检查GPU是否已被使用
  const uint64_t flag = 1ULL<<(graph->nChannels);
  // 获取GPU总数
  int ngpus = system->nodes[GPU].count;
  // 获取当前GPU到所有其他GPU的路径
  struct ncclTopoLinkList* paths = gpu->paths[GPU];
  // 初始化网络路径为NULL
  struct ncclTopoLinkList* netPaths = NULL;
  // 如果需要考虑网络路径排序，获取网络路径
  if (sortNet) NCCLCHECK(getNetPaths(system, graph, &netPaths));

  // 定义GPU评分数组
  struct ncclGpuScore scores[NCCL_TOPO_MAX_NODES];
  // 清零评分数组
  memset(scores, 0, ngpus*sizeof(struct ncclGpuScore));
  // 计算当前GPU在GPU数组中的索引
  int start = gpu-system->nodes[GPU].nodes;
  // 初始化可用GPU计数
  int count = 0;
  // 遍历所有其他GPU
  for (int i=1; i<ngpus; i++) {
    // 计算环形索引，从当前GPU的下一个开始
    int g = (start+i)%ngpus;
    // 如果没有路径到该GPU，跳过
    // There is no path to that GPU
    if (paths[g].count == 0) continue;
    // 如果该GPU已被当前通道使用，跳过
    if (system->nodes[GPU].nodes[g].used & flag) continue;
    // 记录GPU索引
    scores[count].g = g;
    // 记录起始索引
    scores[count].startIndex = i;
    // 记录节点内跳数
    scores[count].intraNhops = paths[g].count;
    // 记录节点内带宽
    scores[count].intraBw = paths[g].bw;
    // 如果有网络路径，记录节点间相关指标
    if (netPaths) {
      // 记录节点间跳数
      scores[count].interNhops = netPaths[g].count;
      // 记录节点间PCI带宽
      scores[count].interPciBw = gpuPciBw(system->nodes[GPU].nodes+g);
      // 记录节点间带宽
      scores[count].interBw = netPaths[g].bw;
    }
    // 增加计数
    count++;
  }

  // Sort GPUs
  // 对GPU进行排序
  qsort(scores, count, sizeof(struct ncclGpuScore), cmpScore);

  // Check if all have the same intra-node score in which case we go reverse for sortNet = -1
  // 检查是否所有GPU的节点内评分相同，如果是且sortNet=-1则反转顺序
  if (sortNet == -1 && cmpIntraScores(scores, count) == 0) {
    // 反转顺序
    for (int i=0; i<count; i++) next[i] = scores[count-1-i].g;
  } else {
    // 正常顺序
    for (int i=0; i<count; i++) next[i] = scores[i].g;
  }

  // 返回可用GPU数量
  *countPtr = count;

  // 如果系统中有NVSwitch节点
  if (system->nodes[NVS].count) {
    // NVSwitches prefer when we talk to a limited set of peers. Try to use neighbors first.
    // NVSwitch更喜欢与有限的对等节点通信。优先尝试使用相邻GPU。
    // 获取当前GPU的索引
    int index = gpu-system->nodes[GPU].nodes;
    int i;
    // 计算前一个GPU索引（环形）
    int prevGpu = (index-1+ngpus)%ngpus;
    // 计算后一个GPU索引（环形）
    int nextGpu = (index+1)%ngpus;
    // 优先GPU数组（最多2个）
    int firstGpus[2];
    // 优先GPU数量
    int firstGpuCount = 0;
    // 根据拓扑模式确定优先GPU
    if (graph->pattern == NCCL_TOPO_PATTERN_RING) {
      // 环形模式：优先下一个和上一个
      firstGpus[0] = nextGpu; firstGpus[1] = prevGpu; firstGpuCount = 2;
    } else if (graph->pattern == NCCL_TOPO_PATTERN_SPLIT_TREE ||
        graph->pattern == NCCL_TOPO_PATTERN_BALANCED_TREE) {
      // 分裂树或平衡树模式：优先上一个和下一个
      firstGpus[0] = prevGpu; firstGpus[1] = nextGpu; firstGpuCount = 2;
    } else {
      // 其他模式：优先下一个
      firstGpus[0] = nextGpu; firstGpuCount = 1;
    }
    // 如果只有2个GPU（nextGpu和prevGpu相同），只保留1个
    if (nextGpu == prevGpu && firstGpuCount == 2) firstGpuCount = 1;
    // 实际找到的优先GPU数量
    int firstGpuRealCount = 0;
    // 将优先GPU移到数组前面
    for (int g=0; g<firstGpuCount; g++) {
      // 查找优先GPU在next数组中的位置
      for (i=0; i<count && next[i] != firstGpus[g]; i++);
      // 如果找到了
      if (i<count) {
        // 将该GPU移到数组开头
        for (; i>0; i--) next[i] = next[i-1];
        next[0] = firstGpus[g];
        // 增加实际找到的优先GPU计数
        firstGpuRealCount++;
      }
    }
    // 优先只尝试优先GPU
    *countPtr = firstGpuRealCount;
  }
  // 返回成功
  return ncclSuccess;
}

// 递归搜索函数的前向声明
ncclResult_t ncclTopoSearchRec(struct ncclTopoSystem* system, struct ncclTopoGraph* graph, struct ncclTopoGraph* saveGraph, int* time);

// Try to keep all searchs within one second
// 尝试将所有搜索保持在一秒内完成
// 定义全局搜索超时阈值（约0.5秒）
#define NCCL_SEARCH_GLOBAL_TIMEOUT (1ULL<<19)
// 定义搜索超时阈值
#define NCCL_SEARCH_TIMEOUT (1<<14)
// 定义树形搜索超时阈值
#define NCCL_SEARCH_TIMEOUT_TREE (1<<14)
// 定义相同通道数时的搜索超时阈值
#define NCCL_SEARCH_TIMEOUT_SAMECHANNELS (1<<8)

// 定义强制顺序类型：PCI顺序
#define FORCED_ORDER_PCI 1
// 定义强制顺序类型：回放顺序
#define FORCED_ORDER_REPLAY 2

// 从重放图中获取下一个GPU
// 参数：
//   system: 拓扑系统结构体指针
//   graph: 图结构指针
//   step: 当前步骤
//   g: 输出参数，返回GPU索引
// 返回：操作结果状态码
ncclResult_t ncclTopoReplayGetGpu(struct ncclTopoSystem* system, struct ncclTopoGraph* graph, int step, int* g) {
  // 初始化返回值为-1（无效索引）
  *g = -1;
  // 如果通道数为0，返回错误
  if (graph->nChannels == 0) return ncclInternalError;
  // 获取GPU总数
  int ngpus = system->nodes[GPU].count;
  // 从重放图中获取下一个rank
  int nextRank = graph->intra[(graph->nChannels-1)*ngpus+step+1];
  // 查找该rank对应的GPU索引
  for (int i=0; i<ngpus; i++) if (system->nodes[GPU].nodes[i].gpu.rank == nextRank) {
    // 找到，返回GPU索引
    *g = i;
    // 返回成功
    return ncclSuccess;
  }
  // 未找到，返回错误
  return ncclInternalError;
}

// 递归搜索GPU函数的前向声明
ncclResult_t ncclTopoSearchRecGpu(struct ncclTopoSystem* system, struct ncclTopoGraph* graph, struct ncclTopoGraph* saveGraph, struct ncclTopoNode* gpu, int step, int backToNet, int backToFirstRank, int forcedOrder, int *time);

// 尝试连接到指定GPU
// 参数：
//   system: 拓扑系统结构体指针
//   graph: 图结构指针
//   saveGraph: 保存图结构指针
//   step: 当前步骤
//   backToNet: 是否返回网络节点
//   backToFirstRank: 是否返回第一个rank
//   forcedOrder: 强制顺序类型
//   time: 时间指针
//   type: 源节点类型
//   index: 源节点索引
//   g: 目标GPU索引
// 返回：操作结果状态码
ncclResult_t ncclTopoSearchTryGpu(struct ncclTopoSystem* system, struct ncclTopoGraph* graph, struct ncclTopoGraph* saveGraph, int step, int backToNet, int backToFirstRank, int forcedOrder, int *time, int type, int index, int g) {
  // 构造当前通道的标志位
  const uint64_t flag = 1ULL<<(graph->nChannels);
  // 目标GPU节点指针
  struct ncclTopoNode* gpu;
  // 尝试分配带宽并连接到目标GPU
  NCCLCHECK(ncclTopoFollowPath(system, graph, type, index, GPU, g, 1, &gpu));
  // 如果成功连接
  if (gpu) {
    // 标记该GPU已使用
    gpu->used ^= flag;
    // 递归搜索下一个GPU
    NCCLCHECK(ncclTopoSearchRecGpu(system, graph, saveGraph, gpu, step, backToNet, backToFirstRank, forcedOrder, time));
    // 取消标记该GPU已使用
    gpu->used ^= flag;
    // 释放带宽
    NCCLCHECK(ncclTopoFollowPath(system, graph, type, index, GPU, g, -1, &gpu));
  }
  // 返回成功
  return ncclSuccess;
}

// 尝试CollNet直接模式
// 在这种模式下，所有GPU直接通信，不需要中继节点
// 参数：
//   system: 拓扑系统结构体指针
//   graph: 图结构指针
//   saveGraph: 保存图结构指针
//   g: 当前GPU索引
//   ngpus: GPU总数
//   time: 时间指针
// 返回：操作结果状态码
ncclResult_t ncclTopoSearchTryCollnetDirect(struct ncclTopoSystem* system, struct ncclTopoGraph* graph, struct ncclTopoGraph* saveGraph, int g, int ngpus, int *time) {
  // 前向GPU索引
  int fwdg = 0;
  // 反向GPU索引
  int bwdg = 0;
  // GPU节点指针
  struct ncclTopoNode* gpu = NULL;
  // 计算带宽倍数（平均分配带宽）
  float mul = 1.0 / (float)(system->nodes[GPU].count - 1);
  // 尝试前向连接所有GPU
  do {
    NCCLCHECK(ncclTopoFollowPath(system, graph, GPU, g, GPU, fwdg, mul, &gpu));
  } while (gpu && ++fwdg < system->nodes[GPU].count);

  // 如果前向连接全部成功
  if (gpu != NULL) {
    // 尝试反向连接所有GPU
    do {
      NCCLCHECK(ncclTopoFollowPath(system, graph, GPU, bwdg, GPU, g, mul, &gpu));
    } while (gpu && ++bwdg < system->nodes[GPU].count);
    // 如果反向连接也全部成功
    if (gpu != NULL) {
      // Both directions worked. Now we already have head, so pop the all other intra ranks.
      // 两个方向都成功了。现在已经有了head，所以填充所有其他节点内rank
      int step = 1;
      // 遍历所有GPU
      for (int index = 0; index < ngpus; ++index) {
        // 跳过当前GPU
        if (index != g) {
          // 记录该GPU的rank到intra数组
          graph->intra[graph->nChannels * ngpus + step] = system->nodes[GPU].nodes[index].gpu.rank;
          // 增加步骤计数
          step++;
        }
      }
      // 继续递归搜索
      NCCLCHECK(ncclTopoSearchRecGpu(system, graph, saveGraph, NULL, ngpus, -1, -1, 0, time));
    }
    // 释放反向连接的带宽
    while (bwdg) {
      bwdg--;
      NCCLCHECK(ncclTopoFollowPath(system, graph, GPU, bwdg, GPU, g, -mul, &gpu));
    }
  }
  // 释放前向连接的带宽
  while (fwdg) {
    fwdg--;
    NCCLCHECK(ncclTopoFollowPath(system, graph, GPU, g, GPU, fwdg, -mul, &gpu));
  }
  // 返回成功
  return ncclSuccess;
}

// 尝试NVLS（NVLink Switch）模式
// 参数：
//   system: 拓扑系统结构体指针
//   graph: 图结构指针
//   saveGraph: 保存图结构指针
//   g: 当前GPU索引
//   ngpus: GPU总数
//   time: 时间指针
// 返回：操作结果状态码
ncclResult_t ncclTopoSearchTryNvls(struct ncclTopoSystem* system, struct ncclTopoGraph* graph, struct ncclTopoGraph* saveGraph, int g, int ngpus, int *time) {
  // NVSwitch节点指针
  struct ncclTopoNode* nvs;
  // GPU节点指针
  struct ncclTopoNode* gpu;
  int d0=0; // See if there is enough bandwidth for NVS->GPU traffic
  // d0: 检查NVS到GPU方向的带宽是否足够
  do {
    // NVS到GPU：当前GPU使用2倍带宽（发送+接收），其他GPU使用1倍带宽
    NCCLCHECK(ncclTopoFollowPath(system, graph, NVS, 0, GPU, d0, d0 == g ? 2 : 1, &gpu));
    d0++;
  } while (gpu && d0 < system->nodes[GPU].count);
  // 如果前向连接失败
  if (gpu == NULL) {
    // 回退一步
    d0--;
  } else {
    int d1=0; // See if there is enough bandwidth for GPU->NVS traffic
    // d1: 检查GPU到NVS方向的带宽是否足够
    do {
      // GPU到NVS：当前GPU使用2倍带宽，其他GPU使用1倍带宽
      NCCLCHECK(ncclTopoFollowPath(system, graph, GPU, d1, NVS, 0, d1 == g ? 2 : 1, &nvs));
      d1++;
    } while (nvs && d1 < system->nodes[GPU].count);
    // 如果反向连接失败
    if (nvs == NULL) {
      // 回退一步
      d1--;
    } else { // Both directions worked. Move on to the next path.
      // 两个方向都成功，继续下一步
      NCCLCHECK(ncclTopoSearchRecGpu(system, graph, saveGraph, NULL, ngpus, -1, -1, 0, time));
    }
    while (d1) {
      d1--;
      NCCLCHECK(ncclTopoFollowPath(system, graph, GPU, d1, NVS, 0, d1 == g ? -2 : -1, &nvs));
    }
  }
  while (d0) {
    d0--;
    NCCLCHECK(ncclTopoFollowPath(system, graph, NVS, 0, GPU, d0, d0 == g ? -2 : -1, &gpu));
  }
  return ncclSuccess;
}

// 比较两个图结构，判断是否需要用新图替换参考图
// 参数：
//   system: 拓扑系统结构体指针
//   graph: 新图结构指针
//   refGraph: 参考图结构指针
//   copy: 输出参数，1表示需要复制新图，0表示不需要
// 返回：操作结果状态码
ncclResult_t ncclTopoCompareGraphs(struct ncclTopoSystem* system, struct ncclTopoGraph* graph, struct ncclTopoGraph* refGraph, int* copy) {
  // 1. Try to get the same nChannels between Rings and Trees
  // 1. 尝试在环形和树形拓扑之间获得相同的通道数
  // 如果通道数小于最小通道数，不复制
  if (graph->nChannels < graph->minChannels) return ncclSuccess;

  // 如果是NVLS模式
  if (graph->pattern == NCCL_TOPO_PATTERN_NVLS) { // NVLS channels correspond to GPUs pulling from NVLS. So the more the better.
    // NVLS通道对应于从NVLS拉取数据的GPU，通道越多越好
    // 如果新图通道数更多且不超过GPU数量，复制
    if (graph->nChannels > refGraph->nChannels && graph->nChannels <= system->nodes[GPU].count) *copy = 1;
    // 如果新图的总带宽更高，复制
    if (graph->nChannels*graph->bwInter > refGraph->nChannels*refGraph->bwInter) *copy = 1;
    // 返回成功
    return ncclSuccess;
  }
  // 2. Try to get better bandwidth
  // 2. 尝试获得更好的带宽
  // 如果新图的节点内总带宽更高，复制
  if (graph->nChannels*graph->bwIntra > refGraph->nChannels*refGraph->bwIntra) {
    *copy = 1;
    // 返回成功
    return ncclSuccess;
  }
  // 如果新图的节点内总带宽更低，不复制
  if (graph->nChannels*graph->bwIntra < refGraph->nChannels*refGraph->bwIntra) return ncclSuccess;

  // 3. Less hops
  // 3. 更少的跳数
  // 如果模式相同、跨网卡设置相同，且新图跳数更少，复制
  if (graph->pattern == refGraph->pattern && graph->crossNic == refGraph->crossNic && graph->nHops < refGraph->nHops) *copy = 1;
  // 返回成功
  return ncclSuccess;
}

// Add the preferred NICs ordered by GPU first
// 按GPU优先的顺序添加首选网卡
// 参数：
//   system: 拓扑系统结构体指针
//   gpu: GPU索引（-1表示所有GPU）
//   nets: 输出参数，返回网卡索引数组
//   netCount: 输出参数，返回网卡数量
// 返回：操作结果状态码
static ncclResult_t ncclTopoPrefNetsGpuFirst(struct ncclTopoSystem* system, int gpu, int nets[NCCL_TOPO_MAX_NODES], int* netCount) {
  // 确定要处理的GPU数量
  const int nGpus = (gpu == -1) ? system->nodes[GPU].count : 1;
  // 初始化GPU计数
  int gpuCount = nGpus;
  // GPU ID数组，初始化为指定的GPU或全部GPU
  int gpuIds[NCCL_TOPO_MAX_NODES] = {gpu};
  // 记录每个GPU的首个网卡
  int firstNets[NCCL_TOPO_MAX_NODES];
  // 如果gpu为-1，填充所有GPU的ID
  if (gpu == -1)
    for (int g = 0; g < nGpus; g++) gpuIds[g] = g;

  // 遍历所有通道
  for (int c = 0; c < MAXCHANNELS; c++) {
    // 遍历所有GPU
    for (int g = 0; g < nGpus; g++) {
      // 如果该GPU已完成，跳过
      if (gpuIds[g] == -1) continue;
      // 本地网卡索引
      int localNet;
      // 网卡ID
      int64_t netId;
      // 获取GPU节点指针
      struct ncclTopoNode* gpu = system->nodes[GPU].nodes + gpuIds[g];
      // 获取该GPU在通道c上的本地网卡ID
      NCCLCHECK(ncclTopoGetLocalNet(system, gpu->gpu.rank, c, &netId, NULL));
      // 将网卡ID转换为索引
      NCCLCHECK(ncclTopoIdToIndex(system, NET, netId, &localNet));
      // store the first net found for each GPU in case of duplicates
      // 存储每个GPU找到的第一个网卡，以防重复
      if(c == 0) firstNets[g] = localNet;
      // if the NET has already been returned for channel 0, that GPU is done
      // 如果该网卡已经在通道0返回过，则该GPU已完成
      if (c > 0 && firstNets[g] == localNet) {
        // 标记该GPU已完成
        gpuIds[g] = -1;
        // 减少GPU计数
        gpuCount--;
        continue;
      }
      // only add it to the list if it doesn't already exist
      int found = 0;
      while (found < (*netCount) && nets[found] != localNet) found++;
      if (found == (*netCount)) nets[(*netCount)++] = localNet;
    }
    if (gpuCount == 0) break;
  }
  return ncclSuccess;
}

// Add the preferred NICs ordered by channels first
// 按通道优先的顺序添加首选网卡
// 参数：
//   system: 拓扑系统结构体指针
//   gpu: GPU索引（-1表示所有GPU）
//   nets: 输出参数，返回网卡索引数组
//   netCount: 输出参数，返回网卡数量
// 返回：操作结果状态码
static ncclResult_t ncclTopoPrefNetsChannelFirst(struct ncclTopoSystem* system, int gpu, int nets[NCCL_TOPO_MAX_NODES], int* netCount) {
  // 遍历所有GPU
  for (int g = 0; g < system->nodes[GPU].count; g++) {
    // 如果指定了GPU且不是当前GPU，跳过
    if (gpu != -1 && gpu != g) continue;
    // 本地网卡计数和数组
    int localNetCount = 0, localNets[MAXCHANNELS];
    // 获取GPU节点指针
    struct ncclTopoNode* gpu = system->nodes[GPU].nodes + g;
    // 遍历所有通道，收集该GPU的网卡
    for (int c = 0; c < MAXCHANNELS; c++) {
      // 网卡ID
      int64_t netId;
      // 获取该GPU在通道c上的本地网卡ID
      NCCLCHECK(ncclTopoGetLocalNet(system, gpu->gpu.rank, c, &netId, NULL));
      // 将网卡ID转换为索引
      NCCLCHECK(ncclTopoIdToIndex(system, NET, netId, localNets + localNetCount));
      // 如果遇到重复的网卡（与第一个相同），停止
      if (localNetCount > 0 && localNets[localNetCount] == localNets[0]) break;
      // 增加本地网卡计数
      localNetCount++;
    }
    // Append NICs to list
    // 将网卡添加到列表
    for (int i = 0; i < localNetCount; i++) {
      // 获取网卡索引
      int n = localNets[i];
      // 查找是否已存在
      int found = 0;
      while (found < (*netCount) && nets[found] != n) found++;
      // 如果不存在，添加到列表
      if (found == (*netCount)) nets[(*netCount)++] = n;
    }
  }
  // 返回成功
  return ncclSuccess;
}

// Build a sorted list of the NETs to try.
// 构建一个排序的网卡列表以供尝试
//
// "gpu" can be set to -1 to build a list suitable for all GPUs (search start) or to a given gpu
// "gpu"可以设置为-1来构建适合所有GPU的列表（搜索开始）或特定GPU索引
//  index when trying to get back to the NIC.
//  当尝试返回网卡时使用
//
// The list is built the following way:
// 列表按以下方式构建：
// 1. Select NETs starting with those close to GPU(s), based on paths[n].type.
// 1. 首先选择靠近GPU的网卡，基于paths[n].type
// 2. add other NETs satisfying typeInter but not already in the list.
// 2. 添加满足typeInter但不在列表中的其他网卡
// 定义是否启用分散网卡的参数
NCCL_PARAM(ScatterEnable, "MNNVL_SCATTER_NETS_ENABLE", 1);
// 选择网卡函数
// 参数：
//   system: 拓扑系统结构体指针
//   typeInter: 节点间链路类型
//   gpu: GPU索引（-1表示所有GPU）
//   nets: 输出参数，返回网卡索引数组
//   netCountRet: 输出参数，返回网卡数量
// 返回：操作结果状态码
ncclResult_t ncclTopoSelectNets(struct ncclTopoSystem* system, int typeInter, int gpu, int nets[NCCL_TOPO_MAX_NODES], int* netCountRet) {
  // 初始化返回值为成功
  ncclResult_t ret = ncclSuccess;
  // 初始化网卡计数
  int netCount = 0;

  // First add the preferred NETs.
  // 首先添加首选网卡
  // 如果是多主机节点且启用了分散模式
  if (system->nHosts > 1 && ncclParamScatterEnable()) {
    // For MNNVL systems, we sort the devices by GPU first, then by channel
    // 对于MNNVL系统，我们按GPU优先，然后按通道排序设备
    NCCLCHECK(ncclTopoPrefNetsGpuFirst(system, gpu, nets, &netCount));
  } else {
    // For other systems, we sort the devices by channel first, then by GPU
    // 对于其他系统，我们按通道优先，然后按GPU排序设备
    NCCLCHECK(ncclTopoPrefNetsChannelFirst(system, gpu, nets, &netCount));
  }

  // Then add others satisfying typeInter
  // 然后添加满足typeInter的其他网卡
  for (int t=0; t <= typeInter; t++) {
    // 遍历所有GPU
    for (int g = 0; g < system->nodes[GPU].count; g++) {
      // 如果指定了GPU且不是当前GPU，跳过
      if (gpu != -1 && gpu != g) continue;
      // 本地网卡计数和数组
      int localNetCount = 0, localNets[MAXCHANNELS];
      // 获取GPU节点指针
      struct ncclTopoNode* gpu = system->nodes[GPU].nodes+g;
      // 获取该GPU到所有网卡的路径
      struct ncclTopoLinkList* paths = gpu->paths[NET];
      // 遍历所有网卡
      for (int n=0; n<system->nodes[NET].count && n<MAXCHANNELS; n++) {
        // 如果路径类型匹配，添加到本地网卡列表
        if (paths[n].type == t) localNets[localNetCount++] = n;
      }
      // Append NICs to list
      // 将网卡添加到列表
      for (int i=0; i<localNetCount; i++) {
        // 获取网卡索引
        int n = localNets[i];
        // 查找是否已存在
        int found = 0;
        while (found<netCount && nets[found] != n) found++;
        // 如果不存在，添加到列表
        if (found == netCount) nets[netCount++] = n;
      }
    }
  }

  // 返回网卡数量
  *netCountRet = netCount;
  // 返回结果
  return ret;
}

// 递归搜索GPU连接的函数
// 这是拓扑搜索的核心递归函数，尝试不同的GPU连接方式
// 参数：
//   system: 拓扑系统结构体指针
//   graph: 当前图结构指针
//   saveGraph: 保存的最佳图结构指针
//   gpu: 当前GPU节点指针
//   step: 当前步骤（已连接的GPU数量）
//   backToNet: 返回网卡的步骤（-1表示不返回）
//   backToFirstRank: 返回第一个rank的步骤
//   forcedOrder: 强制顺序类型
//   time: 剩余搜索时间
// 返回：操作结果状态码
ncclResult_t ncclTopoSearchRecGpu(struct ncclTopoSystem* system, struct ncclTopoGraph* graph, struct ncclTopoGraph* saveGraph, struct ncclTopoNode* gpu, int step, int backToNet, int backToFirstRank, int forcedOrder, int *time) {
  // 检查超时
  if ((*time) <= 0) return ncclSuccess;
  // 减少剩余时间
  (*time)--;

  // 获取GPU总数
  int ngpus = system->nodes[GPU].count;
  // 如果已经连接完所有GPU
  if (step == ngpus) {
    // Determine whether we found a better solution or not
    // 判断是否找到了更好的解
    // 初始化复制标志
    int copy = 0;
    // 增加通道数
    graph->nChannels++;
    // 比较当前图与保存的图
    NCCLCHECK(ncclTopoCompareGraphs(system, graph, saveGraph, &copy));
    // 如果当前图更好
    if (copy) {
      // 复制当前图到保存图
      memcpy(saveGraph, graph, sizeof(struct ncclTopoGraph));
      // 如果达到最大通道数，设置时间标志
      if (graph->nChannels == graph->maxChannels) *time = -1;
    }
    // 如果未达到最大通道数，继续搜索
    if (graph->nChannels < graph->maxChannels) {
      NCCLCHECK(ncclTopoSearchRec(system, graph, saveGraph, time));
    }
    // 恢复通道数
    graph->nChannels--;
    // 返回成功
    return ncclSuccess;
  }
  // 记录当前GPU的rank到intra数组
  graph->intra[graph->nChannels*ngpus+step] = gpu->gpu.rank;
  // 计算当前GPU的索引
  int g = gpu - system->nodes[GPU].nodes;
  // 网卡数组
  int nets[NCCL_TOPO_MAX_NODES];
  // 如果需要返回网卡
  if (step == backToNet) {
    // first get back to NIC
    // 首先返回到网卡
    // 如果存在网络节点
    if (system->nodes[NET].count) {
      // 获取起始网卡索引
      int startNetIndex;
      NCCLCHECK(getNetIndex(system, graph->inter[graph->nChannels*2], &startNetIndex));
      // 获取起始网卡节点
      struct ncclTopoNode* startNet = system->nodes[NET].nodes+startNetIndex;
      // 网卡数量
      int netCount;
      // 选择可用网卡
      NCCLCHECK(ncclTopoSelectNets(system, graph->typeInter, g, nets, &netCount));
      // 遍历所有网卡
      for (int i=0; i<netCount; i++) {
        // 获取网卡索引
        int n = nets[i];
        // 获取网卡节点
        struct ncclTopoNode* net = system->nodes[NET].nodes+n;
        // 树形拓扑必须使用相同的网卡（对称）
        if (graph->pattern == NCCL_TOPO_PATTERN_TREE && net->id != startNet->id) continue; // Trees are symmetric
        // 环形拓扑跨网卡模式下的特殊处理
        if (graph->pattern == NCCL_TOPO_PATTERN_RING && graph->crossNic == 2) {
          // 奇数通道必须使用相同的网卡
          if (graph->nChannels & 1 && net->id != graph->inter[(graph->nChannels-1)*2]) continue;
        } else {
          // 不跨网卡时，必须使用相同ASIC和端口
          if (graph->crossNic == 0 && (net->net.asic != startNet->net.asic || net->net.port != startNet->net.port)) continue;
        }

        // Balanced Tree : count half of the bandwidth on first two GPUs
        // 平衡树：前两个GPU只计算一半带宽
        // 下一个返回网卡的步骤
        int nextBackToNet = -1;
        // 保存节点间带宽
        float bwInterSave = graph->bwInter;
        // 平衡树特殊处理
        if (graph->pattern == NCCL_TOPO_PATTERN_BALANCED_TREE) {
          // Count half of the bandwidth on each of the first two GPUs
          // 前两个GPU各计算一半带宽
          if (step == 0) nextBackToNet = 1;
          // 第二个GPU必须使用第二个网卡
          else if (net->id != graph->inter[graph->nChannels*2+1]) continue;
          // 带宽减半
          graph->bwInter /= 2;
        }

        // 尝试连接到网卡
        NCCLCHECK(ncclTopoFollowPath(system, graph, GPU, g, NET, n, 1, &net));
        // 恢复带宽
        graph->bwInter = bwInterSave;
        // 如果连接成功
        if (net) {
          // 记录第二个网卡ID
          graph->inter[graph->nChannels*2+1] = net->id;
          // 递归搜索
          NCCLCHECK(ncclTopoSearchRecGpu(system, graph, saveGraph, gpu, step, nextBackToNet, backToFirstRank, forcedOrder, time));

          // 平衡树需要再次减半带宽
          if (graph->pattern == NCCL_TOPO_PATTERN_BALANCED_TREE) graph->bwInter /= 2;
          // 释放带宽
          NCCLCHECK(ncclTopoFollowPath(system, graph, GPU, g, NET, n, -1, &net));
          // 恢复带宽
          graph->bwInter = bwInterSave;
        }
      }
    }
  } else if (graph->pattern == NCCL_TOPO_PATTERN_NVLS) {
    // NVLS模式：尝试NVLS连接
    NCCLCHECK(ncclTopoSearchTryNvls(system, graph, saveGraph, g, ngpus, time));
  } else if (graph->pattern == NCCL_TOPO_PATTERN_COLLNET_DIRECT) {
    // CollNet直接模式
    NCCLCHECK(ncclTopoSearchTryCollnetDirect(system, graph, saveGraph, g, ngpus, time));
  } else if (step < system->nodes[GPU].count-1) {
    // Go to next GPU
    // 连接到下一个GPU
    // 下一个GPU候选数组
    int next[NCCL_TOPO_MAX_NODES];
    // 候选数量
    int count;
    // 根据强制顺序选择搜索策略
    if (forcedOrder == FORCED_ORDER_PCI) { // Try the PCI order
      // 尝试PCI顺序
      next[0] = step+1;
      count = 1;
    } else if (forcedOrder == FORCED_ORDER_REPLAY) { // Try last channel order
      // 尝试上一个通道的顺序
      NCCLCHECK(ncclTopoReplayGetGpu(system, graph, step, next));
      count = 1;
    } else { // Normal search
      // 正常搜索：排序候选GPU
      NCCLCHECK(ncclTopoSearchNextGpuSort(system, graph, gpu, next, &count, backToNet == -1 ? 0 : backToNet == step+1 ? 1 : -1 ));
    }
    // 尝试所有候选GPU
    for (int i=0; i<count; i++) {
      NCCLCHECK(ncclTopoSearchTryGpu(system, graph, saveGraph, step+1, backToNet, backToFirstRank, forcedOrder, time, GPU, g, next[i]));
    }
  } else if (step == backToFirstRank) {
    // Find first GPU and loop back to it
    // 找到第一个GPU并回环连接
    // 第一个GPU的索引
    int p;
    // 获取第一个GPU的索引
    NCCLCHECK(getGpuIndex(system, graph->intra[graph->nChannels*ngpus], &p));
    // 第一个GPU节点
    struct ncclTopoNode* firstGpu;
    // 尝试连接到第一个GPU
    NCCLCHECK(ncclTopoFollowPath(system, graph, GPU, g, GPU, p, 1, &firstGpu));
    // 如果连接成功
    if (firstGpu) {
      // 递归搜索（完成环形）
      NCCLCHECK(ncclTopoSearchRecGpu(system, graph, saveGraph, firstGpu, step+1, backToNet, -1, forcedOrder, time));
      // 释放带宽
      NCCLCHECK(ncclTopoFollowPath(system, graph, GPU, g, GPU, p, -1, &firstGpu));
    }
  } else {
    // Next path
    // 进入下一通道搜索
    NCCLCHECK(ncclTopoSearchRecGpu(system, graph, saveGraph, gpu, ngpus, -1, -1, forcedOrder, time));
  }
  // 返回成功
  return ncclSuccess;
}

// 递归搜索网络连接的函数
// 尝试不同的网卡作为起点进行拓扑搜索
// 参数：
//   system: 拓扑系统结构体指针
//   graph: 当前图结构指针
//   saveGraph: 保存的最佳图结构指针
//   backToNet: 返回网卡的步骤
//   backToFirstRank: 返回第一个rank的步骤
//   time: 剩余搜索时间
// 返回：操作结果状态码
ncclResult_t ncclTopoSearchRecNet(struct ncclTopoSystem* system, struct ncclTopoGraph* graph, struct ncclTopoGraph* saveGraph, int backToNet, int backToFirstRank, int* time) {
  // 保存节点间带宽
  const int bw = graph->bwInter;
  // 网卡数组
  int nets[NCCL_TOPO_MAX_NODES];
  // 网卡数量
  int netCount;
  // 图找到标志（用于NVLS/CollNet）
  int graphFound = 0;
  // 选择所有可用的网卡
  NCCLCHECK(ncclTopoSelectNets(system, graph->typeInter, -1, nets, &netCount));
  // 遍历所有网卡（从当前通道开始循环）
  for (int i=0; i<netCount; i++) {
    // 如果是NVLS或CollNet模式且已找到图，停止
    if ((graph->pattern == NCCL_TOPO_PATTERN_NVLS || graph->pattern == NCCL_TOPO_PATTERN_COLLNET_DIRECT) && graphFound) break;
    // 循环获取网卡索引
    int n = nets[(graph->nChannels+i)%netCount];
    // 获取网卡节点
    struct ncclTopoNode* net = system->nodes[NET].nodes+n;
    // 如果是CollNet且该网卡不支持CollNet，跳过
    if (graph->collNet && net->net.collSupport == 0) continue;
    // 如果网卡带宽不足，跳过
    if (net->net.bw < bw) continue;
    // 环形拓扑跨网卡模式：奇数通道必须使用相同的网卡
    if (graph->pattern == NCCL_TOPO_PATTERN_RING && graph->crossNic == 2
        && (graph->nChannels & 1) && net->id != graph->inter[(graph->nChannels-1)*2+1]) continue;

    // 记录网卡ID到inter数组
    graph->inter[graph->nChannels*2] = net->id;
    // 记录节点间延迟
    graph->latencyInter = net->net.latency;

    // 减少同一ASIC和端口的所有网卡带宽
    for (int i=0; i<system->nodes[NET].count; i++) {
      if ((system->nodes[NET].nodes[i].net.asic == net->net.asic) &&
          (system->nodes[NET].nodes[i].net.port == net->net.port)) {
        system->nodes[NET].nodes[i].net.bw -= bw;
      }
    }

    // NVLS或CollNet直接模式特殊处理
    if (graph->pattern == NCCL_TOPO_PATTERN_NVLS || graph->pattern == NCCL_TOPO_PATTERN_COLLNET_DIRECT) {
      // NVLS search only tries to find NIC:GPU combinations to compute the heads.
      // NVLS搜索只尝试找到网卡:GPU组合来计算头部
      // 如果通道数少于网卡数
      if (graph->nChannels < netCount) {
        // 获取该网卡的本地GPU
        int gpu = net->net.localGpu;
        // 如果有本地GPU
        if (gpu != -1) {
          // 重复标志
          int duplicate = 0;
          // check whether there is duplicate head when one GPU connects with multiple NICs
          // 检查当一个GPU连接多个网卡时是否有重复的头部
          for (int gc = 0; gc < graph->nChannels; gc++) {
            if (graph->intra[gc * system->nodes[GPU].count] == system->nodes[GPU].nodes[gpu].gpu.rank) {
              duplicate = 1;
              break;
            }
          }
          // 如果没有重复
          if (!duplicate) {
            // 尝试使用该GPU
            NCCLCHECK(ncclTopoSearchTryGpu(system, graph, saveGraph, 0, backToNet, backToFirstRank, 0, time, NET, n, gpu));
            // 标记已找到图
            graphFound = 1;
          }
        }
      }
    } else {
      // 非NVLS/CollNet模式
      // 如果需要重放上一次通道
      if (graph->nChannels > 0 && graph->sameChannels == 1) {
        // Try to replay the last channel
        // 尝试重放最后一个通道
        int g;
        NCCLCHECK(ncclTopoReplayGetGpu(system, graph, -1, &g));
        NCCLCHECK(ncclTopoSearchTryGpu(system, graph, saveGraph, 0, backToNet, backToFirstRank, FORCED_ORDER_REPLAY, time, NET, n, g));
      } else {
        // 如果是第一个通道且没有NVSwitch
        if (graph->nChannels == 0 && system->nodes[NVS].count == 0) {
          // Always try the PCI order first to set a reference, but don't count in the timeout nor let it run for long
          // 总是先尝试PCI顺序以设置参考，但不计入超时也不让它运行太久
          int t = 1 << 10;
          NCCLCHECK(ncclTopoSearchTryGpu(system, graph, saveGraph, 0, backToNet, backToFirstRank, FORCED_ORDER_PCI, &t, NET, n, 0));
          // 如果找到最大通道数，设置时间标志
          if (t == -1) *time = -1;
        }

        // Then try the most local GPUs
        // 然后尝试最接近的GPU
        // 获取该网卡的本地GPU
        int localGpu = net->net.localGpu;
        if (localGpu != -1) {
          NCCLCHECK(ncclTopoSearchTryGpu(system, graph, saveGraph, 0, backToNet, backToFirstRank, 0, time, NET, n, localGpu));
        }
        // 获取所有本地GPU
        int localGpus[NCCL_TOPO_MAX_NODES], localGpuCount, pathType;
        NCCLCHECK(ncclTopoGetLocal(system, NET, n, GPU, localGpus, &localGpuCount, &pathType));
        // if no GPUs are connected, skip this net
        // 如果没有GPU连接，跳过该网卡
        if (pathType == PATH_DIS) continue;
        // 尝试所有本地GPU
        for (int g = 0; g < localGpuCount; ++g) {
          // 跳过已经尝试过的GPU
          if (localGpus[g] == localGpu) continue; // We already tried this one
          NCCLCHECK(ncclTopoSearchTryGpu(system, graph, saveGraph, 0, backToNet, backToFirstRank, 0, time, NET, n, localGpus[g]));
        }
      }
    }

    // 恢复同一ASIC和端口的所有网卡带宽
    for (int i=0; i<system->nodes[NET].count; i++) {
      if ((system->nodes[NET].nodes[i].net.asic == net->net.asic) &&
          (system->nodes[NET].nodes[i].net.port == net->net.port)) {
        system->nodes[NET].nodes[i].net.bw += bw;
      }
    }
  }
  // 返回成功
  return ncclSuccess;
}

/* Search Patterns
 * 搜索模式说明
 *
 *     Intra-node (节点内)
 * Ring            : GPU a -> GPU b -> .. -> GPU x -> GPU a
 *                  环形模式：所有GPU形成一个环
 * (=Split Tree Loop)
 * Tree            : GPU a -> GPU b -> .. -> GPU x
 *                  树形模式：GPU形成树形结构
 * (=Split Tree)
 *
 *     Inter-node (跨节点)
 * Ring            : NET n -> GPU a -> GPU b -> .. -> GPU x -> NET n (or m if crossNic)
 *                  环形模式：从网卡出发，经过所有GPU，返回到同一网卡(或跨网卡返回)
 * Tree            : NET n -> GPU a -> GPU b -> .. -> GPU x
 *                              `--> NET n (or m if crossNic)
 *                  树形模式：从网卡出发，经过所有GPU，返回到同一网卡(或跨网卡返回)
 * Split Tree      : NET n -> GPU a -> GPU b -> .. -> GPU x
 *                                       `--> NET n (or m if crossNic)
 *                  分裂树模式：类似于Tree，但返回路径从中间某处分裂
 * Split Tree Loop : NET n -> GPU a -> GPU b -> .. -> GPU x -> GPU a
 *                                       `--> NET n (or m if crossNic)
 *                  分裂树环模式：Tree和Ring的混合体
 */

// 根据系统拓扑和模式获取搜索参数
// 参数说明：
// - system: 拓扑系统对象，包含所有节点信息
// - pattern: 通信模式（RING/TREE/SPLIT_TREE等）
// - backToNet: 输出参数，表示返回到网卡的步数
// - backToFirstRank: 输出参数，表示返回到第一个rank的步数
ncclResult_t ncclTopoSearchParams(struct ncclTopoSystem* system, int pattern, int* backToNet, int* backToFirstRank) {
  // 检查系统中是否存在网络节点（网卡）
  if (system->nodes[NET].count) {
    // 如果有网络设备（多节点场景）
    // 对于RING模式，最后一个GPU需要连接回网卡，所以backToNet = GPU数量-1
    if (pattern == NCCL_TOPO_PATTERN_RING) *backToNet = system->nodes[GPU].count-1;
    // 对于SPLIT_TREE模式，只有一个GPU连接回网卡
    else if (pattern == NCCL_TOPO_PATTERN_SPLIT_TREE) *backToNet = 1;
    // 其他模式（如TREE），所有GPU的路径都结束在网卡，不需要额外的回环步骤
    else *backToNet = 0;
    // 多节点场景不需要返回到第一个rank
    *backToFirstRank = -1;
  } else {
    // 如果没有网络设备（单节点场景）
    // 单节点场景不需要返回到网卡
    *backToNet = -1;
    // 对于RING模式，最后一个GPU需要连接回第一个GPU
    if (pattern == NCCL_TOPO_PATTERN_RING) *backToFirstRank = system->nodes[GPU].count-1;
    // 其他模式不需要返回到第一个rank
    else *backToFirstRank = -1;
  }
  // 返回成功
  return ncclSuccess;
}

// 递归搜索通信拓扑图
// 参数说明：
// - system: 拓扑系统对象
// - graph: 当前正在搜索的图（临时图）
// - saveGraph: 用于保存找到的最佳图的指针
// - time: 搜索超时时间（微秒），如果为-1表示找到完美解，0表示超时
ncclResult_t ncclTopoSearchRec(struct ncclTopoSystem* system, struct ncclTopoGraph* graph, struct ncclTopoGraph* saveGraph, int* time) {
  // 获取搜索参数：回环到网卡的步数和回环到第一个rank的步数
  int backToNet, backToFirstRank;
  NCCLCHECK(ncclTopoSearchParams(system, graph->pattern, &backToNet, &backToFirstRank));

  // 判断是否有网络设备（多节点场景）
  if (system->nodes[NET].count) {
    // 从网络设备开始搜索（多节点场景）
    // Start from NET
    ncclTopoSearchRecNet(system, graph, saveGraph, backToNet, backToFirstRank, time);
  } else {
    // 只有节点内设备（单节点场景）
    // Intra-node only.

    // 特殊处理NVLS模式
    if (graph->pattern == NCCL_TOPO_PATTERN_NVLS) {
      // NVLS模式直接尝试从GPU 0开始，使用所有可能的通道
      NCCLCHECK(ncclTopoSearchTryGpu(system, graph, saveGraph, 0, backToNet, backToFirstRank, 0, time, -1, -1, graph->nChannels));
      return ncclSuccess;
    } else if (graph->nChannels == 0) {
      // 这是第一次搜索，还没有找到任何通道
      // 首先尝试按照PCI顺序排列GPU（通常是最优的）
      // Try PCI order first
      NCCLCHECK(ncclTopoSearchTryGpu(system, graph, saveGraph, 0, backToNet, backToFirstRank, FORCED_ORDER_PCI, time, -1, -1, 0));
    } else {
      // 不是第一次搜索，尝试重放之前的通道配置
      // Also try to replay previous channel
      int g;
      // 获取之前搜索中使用的GPU起始位置
      NCCLCHECK(ncclTopoReplayGetGpu(system, graph, -1, &g));
      // 使用重放模式尝试搜索
      NCCLCHECK(ncclTopoSearchTryGpu(system, graph, saveGraph, 0, backToNet, backToFirstRank, FORCED_ORDER_REPLAY, time, -1, -1, g));
    }

    // 如果不需要相同的通道或者还没有找到任何通道，尝试所有其他可能性
    if (graph->sameChannels == 0 || graph->nChannels == 0) {
      // 遍历所有可能的GPU起始位置
      // Finally, try all other possibilities unless we are forced to use the same channels
      for (int g=0; g<system->nodes[GPU].count; g++) {
        // 尝试从每个GPU开始构建通信路径
        NCCLCHECK(ncclTopoSearchTryGpu(system, graph, saveGraph, 0, backToNet, backToFirstRank, 0, time, -1, -1, g));
      }
    }
  }
  // 返回成功
  return ncclSuccess;
}

/************************************/
/* User defined graph from XML file */
/* 从XML文件导入用户定义的拓扑图   */
/************************************/

// 链路类型字典：用于XML导入导出时转换链路类型字符串和枚举值
struct kvDict kvDictLinkType[] = {
  { "LOC", PATH_LOC },   // LOC: 本地路径
  { "NVL", PATH_NVL },   // NVL: NVLink
  { "NVB", PATH_NVB },   // NVB: NVBridge
  { "PIX", PATH_PIX },   // PIX: PCIe内部路径
  { "PXB", PATH_PXB },   // PXB: PCIe桥接路径
  { "P2C", PATH_P2C },   // P2C: P2PCache
  { "PXN", PATH_PXN },   // PXN: PCIe网络路径
  { "PHB", PATH_PHB },   // PHB: PCIe主机桥
  { "SYS", PATH_SYS },   // SYS: 系统路径
  { NULL, 0 }            // 数组结束标记
};

// 从XML节点中读取单个通信通道的信息
// 参数说明：
// - xmlChannel: XML通道节点，包含该通道的所有设备信息
// - c: 通道索引
// - system: 拓扑系统对象
// - graph: 要填充的拓扑图对象
ncclResult_t ncclTopoGetChannelFromXml(struct ncclXmlNode *xmlChannel, int c, struct ncclTopoSystem* system, struct ncclTopoGraph* graph) {
  // 获取GPU数量
  int ngpus = system->nodes[GPU].count;

  // 获取节点间连接数组（网络设备）的指针，每个通道有2个网络设备（起点和终点）
  int64_t* inter = graph->inter+2*c;

  // 获取节点内连接数组（GPU）的指针，每个通道有ngpus个GPU
  int* intra = graph->intra+ngpus*c;

  // n: 网络设备计数器, g: GPU计数器
  int n=0, g=0;

  // 遍历XML通道节点的所有子节点
  for (int s=0; s<xmlChannel->nSubs; s++) {
    // 获取当前子节点
    struct ncclXmlNode* sub = xmlChannel->subs[s];

    // 用于存储设备ID
    int64_t dev;
    // 用于存储从XML读取的字符串
    const char* str;

    // 从XML属性中读取"dev"字段（设备ID，十六进制格式）
    NCCLCHECK(xmlGetAttrStr(sub, "dev", &str));

    // 将十六进制字符串转换为64位整数
    dev = strtol(str, NULL, 16);

    // 判断子节点类型
    if (strcmp(sub->name, "net") == 0) {
      // 如果是网络设备节点
      // 将设备ID存入inter数组
      inter[n++] = dev;
    } else if (strcmp(sub->name, "gpu") == 0) {
      // 如果是GPU节点
      // 初始化rank为-1（表示未找到）
      int rank = -1;

      // 在系统中查找该设备对应的rank
      for (int g=0; g<ngpus; g++) {
        // 从拓扑ID中提取系统ID
        int systemId = NCCL_TOPO_ID_SYSTEM_ID(system->nodes[GPU].nodes[g].id);

        // 比较设备ID是否匹配
        if (NCCL_TOPO_ID(systemId, system->nodes[GPU].nodes[g].gpu.dev) == dev)
          // 找到匹配的GPU，记录其rank
          rank = system->nodes[GPU].nodes[g].gpu.rank;
      }

      // 如果没有找到对应的GPU
      if (rank == -1) {
        // 输出警告信息
        WARN("XML Import Channel : dev %ld not found.", dev);
        // 返回系统错误
        return ncclSystemError;
      }

      // 将GPU的rank存入intra数组
      intra[g++] = rank;
    }
  }

  // 返回成功
  return ncclSuccess;
}
// 从XML图节点中读取拓扑图信息
// 参数说明：
// - xmlGraph: XML图节点，包含图的属性和通道信息
// - system: 拓扑系统对象
// - graph: 要填充的拓扑图对象
// - nChannels: 输出参数，返回从XML读取的通道数量
ncclResult_t ncclTopoGetGraphFromXmlSub(struct ncclXmlNode *xmlGraph, struct ncclTopoSystem* system, struct ncclTopoGraph* graph, int* nChannels) {
  // 用于存储图ID
  int id;

  // 从XML属性中读取图ID
  NCCLCHECK(xmlGetAttrInt(xmlGraph, "id", &id));

  // 如果XML中的图ID与当前图ID不匹配，直接返回成功（跳过此图）
  if (graph->id != id) return ncclSuccess;

  // 用于存储跨网卡标志
  int crossNic;

  // 从XML属性中读取crossNic标志（是否使用跨网卡通信）
  NCCLCHECK(xmlGetAttrInt(xmlGraph, "crossnic", &crossNic));

  // 如果系统参数禁用了跨网卡，但XML中启用了跨网卡，则跳过此图
  if (ncclParamCrossNic() == 0 && crossNic == 1) return ncclSuccess;

  // 设置图的跨网卡标志
  graph->crossNic = crossNic;

  // 从XML属性中读取通信模式（RING/TREE/SPLIT_TREE等）
  NCCLCHECK(xmlGetAttrInt(xmlGraph, "pattern", &graph->pattern));

  // 从XML属性中读取通道数量
  NCCLCHECK(xmlGetAttrInt(xmlGraph, "nchannels", &graph->nChannels));

  // 从XML属性中读取节点内带宽（单位：GB/s）
  NCCLCHECK(xmlGetAttrFloat(xmlGraph, "speedintra", &graph->bwIntra));

  // 从XML属性中读取节点间带宽（单位：GB/s）
  NCCLCHECK(xmlGetAttrFloat(xmlGraph, "speedinter", &graph->bwInter));

  // 用于存储从XML读取的字符串
  const char* str;

  // 尝试从XML属性中读取节点间延迟（单位：微秒）
  NCCLCHECK(xmlGetAttr(xmlGraph, "latencyinter", &str));

  // 如果没有找到延迟属性，输出信息并使用默认值0.0
  if (!str) INFO(NCCL_GRAPH, "latencyinter not found in graph, using 0.0");

  // 将延迟字符串转换为浮点数，如果没有则使用默认值0.0
  graph->latencyInter = str ? strtof(str, NULL) : 0.0;

  // 从XML属性中读取节点内链路类型
  NCCLCHECK(xmlGetAttr(xmlGraph, "typeintra", &str));

  // 将链路类型字符串转换为枚举值
  NCCLCHECK(kvConvertToInt(str, &graph->typeIntra, kvDictLinkType));

  // 从XML属性中读取节点间链路类型
  NCCLCHECK(xmlGetAttr(xmlGraph, "typeinter", &str));

  // 将链路类型字符串转换为枚举值
  NCCLCHECK(kvConvertToInt(str, &graph->typeInter, kvDictLinkType));

  // 从XML属性中读取sameChannels标志（是否所有通道使用相同的拓扑）
  NCCLCHECK(xmlGetAttrInt(xmlGraph, "samechannels", &graph->sameChannels));

  // 遍历XML图的所有子节点（通道）
  for (int s=0; s<xmlGraph->nSubs; s++) {
    // 从XML通道节点中读取通道信息
    NCCLCHECK(ncclTopoGetChannelFromXml(xmlGraph->subs[s], s, system, graph));
  }

  // 设置输出参数：通道数量等于XML子节点数量
  *nChannels = xmlGraph->nSubs;

  // 返回成功
  return ncclSuccess;
}
// 从XML节点集合中读取拓扑图信息（顶层函数）
// 参数说明：
// - xmlGraphs: XML图集合节点，包含多个图的定义
// - system: 拓扑系统对象
// - graph: 要填充的拓扑图对象
// - nChannels: 输出参数，返回从XML读取的通道数量
ncclResult_t ncclTopoGetGraphFromXml(struct ncclXmlNode *xmlGraphs, struct ncclTopoSystem* system, struct ncclTopoGraph* graph, int* nChannels) {
  // 遍历XML图集合的所有子节点（每个子节点代表一个图）
  for (int s=0; s<xmlGraphs->nSubs; s++) {
    // 从每个XML图子节点中读取图信息
    // 注意：这个函数会根据graph->id来匹配正确的图
    NCCLCHECK(ncclTopoGetGraphFromXmlSub(xmlGraphs->subs[s], system, graph, nChannels));
  }

  // 返回成功
  return ncclSuccess;
}

/* And the reverse : graph->xml */
/* 反向操作：将拓扑图导出为XML格式 */

// 将单个通道信息导出为XML节点
// 参数说明：
// - graph: 拓扑图对象
// - c: 通道索引
// - system: 拓扑系统对象
// - xml: XML对象
// - parent: 父XML节点（通常是graph节点）
ncclResult_t ncclTopoGetXmlFromChannel(struct ncclTopoGraph* graph, int c, struct ncclTopoSystem* system, struct ncclXml *xml, struct ncclXmlNode* parent) {
  // XML通道节点指针
  struct ncclXmlNode* xmlChannel;

  // 获取GPU数量
  int ngpus = system->nodes[GPU].count;

  // 获取该通道的网络设备数组指针（2个网络设备：起点和终点）
  int64_t* inter = graph->inter+2*c;

  // 获取该通道的GPU数组指针（ngpus个GPU）
  int* intra = graph->intra+ngpus*c;

  // 在XML中添加一个"channel"节点
  NCCLCHECK(xmlAddNode(xml, parent, "channel", &xmlChannel));

  // XML节点指针（用于添加网络和GPU节点）
  struct ncclXmlNode* node;

  // 如果系统中有网络设备
  if (system->nodes[NET].count) {
    // 添加第一个网络设备节点（起点）
    NCCLCHECK(xmlAddNode(xml, xmlChannel, "net", &node));

    // 设置网络设备的设备ID属性（十六进制格式）
    NCCLCHECK(xmlSetAttrLong(node, "dev", inter[0]));
  }

  // 遍历该通道的所有GPU
  for (int g=0; g<ngpus; g++) {
    // 添加GPU节点
    NCCLCHECK(xmlAddNode(xml, xmlChannel, "gpu", &node));

    // 初始化设备ID为-1（表示未找到）
    int64_t dev = -1;

    // 在系统中查找该rank对应的GPU设备ID
    for (int i=0; i<ngpus; i++) {
      // 如果找到了匹配的rank
      if (system->nodes[GPU].nodes[i].gpu.rank == intra[g]) {
        // 提取系统ID
        int systemId = NCCL_TOPO_ID_SYSTEM_ID(system->nodes[GPU].nodes[i].id);

        // 构造完整的设备ID（系统ID + 设备ID）
        dev = NCCL_TOPO_ID(systemId, system->nodes[GPU].nodes[i].gpu.dev);
      }
    }

    // 如果没有找到对应的设备
    if (dev == -1) {
      // 输出警告信息
      WARN("XML Export Channel : rank %d not found.", intra[g]);

      // 返回内部错误
      return ncclInternalError;
    }

    // 设置GPU的设备ID属性
    NCCLCHECK(xmlSetAttrLong(node, "dev", dev));

    // 如果是NVLS图（id==3），只使用第一个GPU
    // NVLS graphs only use the first GPU
    if (graph->id == 3) break;
  }

  // 如果系统中有网络设备
  if (system->nodes[NET].count) {
    // 添加第二个网络设备节点（终点）
    NCCLCHECK(xmlAddNode(xml, xmlChannel, "net", &node));

    // 设置网络设备的设备ID属性
    NCCLCHECK(xmlSetAttrLong(node, "dev", inter[1]));
  }

  // 返回成功
  return ncclSuccess;
}
// 将拓扑图导出为XML节点
// 参数说明：
// - graph: 要导出的拓扑图对象
// - system: 拓扑系统对象
// - xml: XML对象
// - parent: 父XML节点（通常是graphs节点）
ncclResult_t ncclTopoGetXmlFromGraph(struct ncclTopoGraph* graph, struct ncclTopoSystem* system, struct ncclXml *xml, struct ncclXmlNode* parent) {
  // XML图节点指针
  struct ncclXmlNode* xmlGraph;

  // 在XML中添加一个"graph"节点
  NCCLCHECK(xmlAddNode(xml, parent, "graph", &xmlGraph));

  // 设置图ID属性
  NCCLCHECK(xmlSetAttrInt(xmlGraph, "id", graph->id));

  // 设置通信模式属性
  NCCLCHECK(xmlSetAttrInt(xmlGraph, "pattern", graph->pattern));

  // 设置跨网卡标志属性
  NCCLCHECK(xmlSetAttrInt(xmlGraph, "crossnic", graph->crossNic));

  // 设置通道数量属性
  NCCLCHECK(xmlSetAttrInt(xmlGraph, "nchannels", graph->nChannels));

  // 设置节点内带宽属性（单位：GB/s）
  NCCLCHECK(xmlSetAttrFloat(xmlGraph, "speedintra", graph->bwIntra));

  // 设置节点间带宽属性（单位：GB/s）
  NCCLCHECK(xmlSetAttrFloat(xmlGraph, "speedinter", graph->bwInter));

  // 设置节点间延迟属性（单位：微秒）
  NCCLCHECK(xmlSetAttrFloat(xmlGraph, "latencyinter", graph->latencyInter));

  // 用于存储转换后的字符串
  const char* str;

  // 将节点内链路类型枚举值转换为字符串
  NCCLCHECK(kvConvertToStr(graph->typeIntra, &str, kvDictLinkType));

  // 设置节点内链路类型属性
  NCCLCHECK(xmlSetAttr(xmlGraph, "typeintra", str));

  // 将节点间链路类型枚举值转换为字符串
  NCCLCHECK(kvConvertToStr(graph->typeInter, &str, kvDictLinkType));

  // 设置节点间链路类型属性
  NCCLCHECK(xmlSetAttr(xmlGraph, "typeinter", str));

  // 设置sameChannels标志属性
  NCCLCHECK(xmlSetAttrInt(xmlGraph, "samechannels", graph->sameChannels));

  // 遍历图的所有通道
  for (int c=0; c<graph->nChannels; c++) {
    // 将每个通道导出为XML子节点
    NCCLCHECK(ncclTopoGetXmlFromChannel(graph, c, system, xml, xmlGraph));
  }

  // 返回成功
  return ncclSuccess;
}
// 将多个拓扑图导出为XML格式（顶层函数）
// 参数说明：
// - ngraphs: 图的数量
// - graphs: 图指针数组
// - system: 拓扑系统对象
// - xml: XML对象（输出）
ncclResult_t ncclTopoGetXmlFromGraphs(int ngraphs, struct ncclTopoGraph** graphs, struct ncclTopoSystem* system, struct ncclXml *xml) {
  // 重置XML的最大索引
  xml->maxIndex = 0;

  // XML图集合节点指针
  struct ncclXmlNode* xmlGraphs;

  // 在XML中添加一个"graphs"根节点（parent为NULL表示根节点）
  NCCLCHECK(xmlAddNode(xml, NULL, "graphs", &xmlGraphs));

  // 设置XML版本号属性
  NCCLCHECK(xmlSetAttrInt(xmlGraphs, "version", NCCL_GRAPH_XML_VERSION));

  // 遍历所有图
  for (int g=0; g<ngraphs; g++) {
    // 将每个图导出为XML子节点
    NCCLCHECK(ncclTopoGetXmlFromGraph(graphs[g], system, xml, xmlGraphs));
  }

  // 返回成功
  return ncclSuccess;
}

// 复制通道以增加并行度
// 通过复制现有通道来增加通道数量，从而提高通信带宽
// 参数说明：
// - graph: 拓扑图对象
// - ccMin: 最小计算能力（Compute Capability）
// - ngpus: GPU数量
ncclResult_t ncclTopoDupChannels(struct ncclTopoGraph* graph, int ccMin, int ngpus) {
  // 如果没有通道，直接返回
  if (graph->nChannels == 0) return ncclSuccess;

  // NVLS模式不需要复制通道
  if (graph->pattern == NCCL_TOPO_PATTERN_NVLS) return ncclSuccess;

  // 如果节点内带宽太低（<25 GB/s），不复制通道
  if (graph->bwIntra < 25.0) return ncclSuccess;

  // 对于计算能力>80的GPU，如果带宽适中且通道数>4，不复制通道
  if (ccMin > 80 && graph->bwIntra < 50.0 && graph->nChannels > 4) return ncclSuccess;

  // 计算复制后的通道数量：取（当前通道数*2）和最大通道数的较小值
  int dupChannels = std::min(graph->nChannels*2, graph->maxChannels);

  // 复制节点内连接数组（GPU rank信息）
  // 从graph->nChannels*ngpus位置开始，复制(dupChannels-graph->nChannels)*ngpus个int
  memcpy(graph->intra+graph->nChannels*ngpus, graph->intra, (dupChannels-graph->nChannels)*ngpus*sizeof(int));

  // 复制节点间连接数组（网络设备信息）
  // 从graph->nChannels*2位置开始，复制(dupChannels-graph->nChannels)*2个int64_t
  memcpy(graph->inter+graph->nChannels*2,graph->inter, (dupChannels-graph->nChannels)*2*sizeof(int64_t));

  // 调整节点内带宽：除以复制倍数（向上取整）
  // 因为通道增加，每个通道分到的带宽会减少
  graph->bwIntra /= DIVUP(dupChannels, graph->nChannels);

  // 调整节点间带宽：除以复制倍数（向上取整）
  graph->bwInter /= DIVUP(dupChannels, graph->nChannels);

  // 更新通道数量
  graph->nChannels = dupChannels;

  // 返回成功
  return ncclSuccess;
}

// 节点内带宽搜索数组（单位：GB/s）
// 用于不同GPU架构的节点内带宽搜索，从高到低排列
// 这些值代表了可能的PCIe/NVLink带宽配置
float speedArrayIntra[] = { 40.0, 30.0, 20.0, 18.0, 15.0, 12.0, 10.0, 9.0, 7.0, 6.0, 5.0, 4.0, 3.0 };

// 节点间带宽搜索数组（单位：GB/s）
// 用于不同GPU架构的节点间带宽搜索，从高到低排列
// 这些值代表了可能的网络（InfiniBand/RoCE）带宽配置
float speedArrayInter[] = { 48.0, 30.0, 28.0, 24.0, 20.0, 18.0, 15.0, 12.0, 10.0, 9.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.4, 1.2, 0.24, 0.12 };

// 计算节点内带宽数组的元素数量
#define NSPEEDSINTRA (sizeof(speedArrayIntra)/sizeof(float))

// 计算节点间带宽数组的元素数量
#define NSPEEDSINTER (sizeof(speedArrayInter)/sizeof(float))

// SM90架构（Hopper H100）的节点内带宽搜索数组
// SM90支持更高的带宽，最高可达60 GB/s
float sm90SpeedArrayIntra[] = { 60.0, 50.0, 40.0, 30.0, 24.0, 20.0, 15.0, 12.0, 11.0, 6.0, 3.0 };

// SM90架构的节点间带宽搜索数组
// SM90支持更高的网络带宽
float sm90SpeedArrayInter[] = { 48.0, 45.0, 42.0, 40.0, 30.0, 24.0, 22.0, 20.0, 17.5, 15.0, 12.0, 6.0, 3.0, 2.4, 1.2, 0.24, 0.12 };

// 计算SM90节点内带宽数组的元素数量
#define NSPEEDSINTRA_SM90 (sizeof(sm90SpeedArrayIntra)/sizeof(float))

// 计算SM90节点间带宽数组的元素数量
#define NSPEEDSINTER_SM90 (sizeof(sm90SpeedArrayInter)/sizeof(float))

// SM100架构（Blackwell B100/GB200）的节点内带宽搜索数组
// SM100支持更高的带宽，最高可达90 GB/s
float sm100SpeedArrayIntra[] = { 90.0, 80.0, 70.0, 60.0, 50.0, 40.0, 30.0, 24.0, 20.0, 19.0, 18.0 };

// SM100架构的节点间带宽搜索数组
// SM100支持更高的网络带宽，最高可达96 GB/s
float sm100SpeedArrayInter[] = { 96.0, 48.0, 45.1, 42.0, 40.0, 30.0, 24.0, 22.0, 20.0, 17.5, 15.0, 12.0, 6.0, 3.0, 2.4, 1.2, 0.24, 0.12 };

// 计算SM100节点内带宽数组的元素数量
#define NSPEEDSINTRA_SM100 (sizeof(sm100SpeedArrayIntra)/sizeof(float))

// 计算SM100节点间带宽数组的元素数量
#define NSPEEDSINTER_SM100 (sizeof(sm100SpeedArrayInter)/sizeof(float))

// 计算并搜索最优通信拓扑图
// 这是拓扑搜索的主函数，负责搜索最优的通信路径配置
// 参数说明：
// - system: 拓扑系统对象，包含所有硬件信息
// - graph: 要填充的拓扑图对象（输入和输出参数）
ncclResult_t ncclTopoCompute(ncclTopoSystem* system, struct ncclTopoGraph* graph) {
  // 获取GPU数量
  int ngpus = system->nodes[GPU].count;

  // 确定是否使用跨网卡通信
  // 条件：系统有多个网卡 AND 通信模式是RING/BALANCED_TREE/SPLIT_TREE AND 环境变量允许跨网卡
  int crossNic = (system->nodes[NET].count > 1) &&
	 (graph->pattern == NCCL_TOPO_PATTERN_RING ||
          graph->pattern == NCCL_TOPO_PATTERN_BALANCED_TREE ||
          graph->pattern == NCCL_TOPO_PATTERN_SPLIT_TREE) ? ncclParamCrossNic() : 0;

  // 设置图的跨网卡标志（只取0或1）
  graph->crossNic = crossNic == 1 ? 1 : 0;

  // 初始化带宽为0（搜索时会设置）
  graph->bwIntra = graph->bwInter = 0;

  // 初始化节点间延迟为0
  graph->latencyInter = 0;

  // 初始化最小和最大链路类型
  // PATH_LOC: 最快（本地），PATH_PIX: PCIe内部，PATH_SYS: 最慢（系统总线）
  int minTypeIntra = PATH_LOC, minTypeInter = PATH_PIX;
  int maxTypeIntra = PATH_SYS, maxTypeInter = PATH_SYS;

  // 如果有多个GPU，获取GPU到GPU的最小和最大路径类型
  if (ngpus > 1) {
    // 获取GPU之间最短的路径类型（最快的连接）
    NCCLCHECK(ncclTopoGetGpuMinPath(system, GPU, &minTypeIntra));

    // 获取GPU之间最长的路径类型（最慢的连接）
    NCCLCHECK(ncclTopoGetGpuMaxPath(system, GPU, &maxTypeIntra));
  }

  // 如果有网络设备，获取GPU到网络设备的最小和最大路径类型
  if (system->nodes[NET].count > 0) {
    // 获取GPU到网卡的最短路径类型
    NCCLCHECK(ncclTopoGetGpuMinPath(system, NET, &minTypeInter));

    // 获取GPU到网卡的最长路径类型
    NCCLCHECK(ncclTopoGetGpuMaxPath(system, NET, &maxTypeInter));

    // 节点内最大路径类型不能超过节点间路径类型
    maxTypeIntra = maxTypeInter;
  }

  // 设置初始路径类型为最小值（最快的路径）
  graph->typeIntra = minTypeIntra;
  graph->typeInter = minTypeInter;

  // 初始化通道数量为0
  graph->nChannels = 0;

  // 确定是否尝试使用相同的通道配置
  // NVLS模式不需要相同通道，其他模式默认尝试相同通道
  int trySameChannels = graph->pattern == NCCL_TOPO_PATTERN_NVLS ? 0 : 1;

  // 设置sameChannels标志
  graph->sameChannels = trySameChannels;

  // 获取CPU架构、厂商和型号信息
  int cpuArch, cpuVendor, cpuModel;
  NCCLCHECK(ncclTopoCpuType(system, &cpuArch, &cpuVendor, &cpuModel));

  // 检查是否设置了环境变量NCCL_GRAPH_FILE（用于从XML文件加载预定义的拓扑图）
  const char* str = ncclGetEnv("NCCL_GRAPH_FILE");

  // 如果设置了NCCL_GRAPH_FILE环境变量
  if (str) {
    // 输出环境变量信息
    INFO(NCCL_ENV, "NCCL_GRAPH_FILE set by environment to %s", str);

    // XML对象指针
    struct ncclXml* xml;

    // 分配XML对象内存
    NCCLCHECK(xmlAlloc(&xml, NCCL_GRAPH_XML_MAX_NODES));

    // 从文件读取XML图
    NCCLCHECK(ncclTopoGetXmlGraphFromFile(str, xml));

    // 用于存储从XML读取的通道数量
    int nChannels;

    // 从XML节点中解析图信息
    NCCLCHECK(ncclTopoGetGraphFromXml(xml->nodes, system, graph, &nChannels));

    // 输出加载信息
    INFO(NCCL_GRAPH, "Search %d : %d channels loaded from XML graph", graph->id, nChannels);

    // 释放XML对象内存
    free(xml);

    // 如果成功加载了通道，直接返回成功（跳过搜索）
    if (graph->nChannels > 0)
        return ncclSuccess;
  }

  // 获取GPU的最小计算能力（Compute Capability）
  int ccMin;
  NCCLCHECK(ncclTopoGetCompCap(system, &ccMin, NULL));

  // 检查NVLS模式的可行性
  // NVLS需要：1) 有NVS设备 OR 2) 计算能力>=90（Hopper或更新架构）
  // 如果不满足条件，直接返回成功（不搜索NVLS模式）
  if (graph->pattern == NCCL_TOPO_PATTERN_NVLS && (system->nodes[NVS].count == 0 || ccMin < 90))
    return ncclSuccess;

  // NVLS和COLLNET_DIRECT模式的最大通道数不能超过GPU数量
  // NVLS and COLLNET_DIRECT search must have ngpus heads at most.

  // 对于NVLS模式，最大通道数取NVLS最大arity和GPU数量的较小值
  if (graph->pattern == NCCL_TOPO_PATTERN_NVLS)
    graph->maxChannels = std::min(NCCL_MAX_NVLS_ARITY, system->nodes[GPU].count);

  // 对于COLLNET_DIRECT模式，最大通道数取DIRECT最大arity+1和GPU数量的较小值
  if (graph->pattern == NCCL_TOPO_PATTERN_COLLNET_DIRECT)
    graph->maxChannels = std::min(NCCL_MAX_DIRECT_ARITY+1, system->nodes[GPU].count);

  // 如果只有一个GPU且不是RING模式，强制使用TREE模式
  // 单GPU场景下，TREE模式更简单高效
  if (ngpus == 1) if (graph->pattern != NCCL_TOPO_PATTERN_RING)
    graph->pattern = NCCL_TOPO_PATTERN_TREE;

  // 如果是单节点NVLS模式（没有网络设备）
  if (system->nodes[NET].count == 0 && graph->pattern == NCCL_TOPO_PATTERN_NVLS) {
    // 强制节点内NVLS算法从所有GPU均匀拉取数据
    // 设置最小通道数等于最大通道数，确保使用所有可能的通道
    // Force intra-node NVLS algorithm to pull evenly from all GPUs.
    graph->minChannels = graph->maxChannels;
  }

  // 检查是否是分裂的NVLink拓扑（两个socket之间有NVLink，但中间有较慢的链路如QPI）
  int splitNvLink;
  NCCLCHECK(ncclTopoSplitNvLink(system, &splitNvLink));

  // 如果是RING模式且有分裂的NVLink拓扑
  if (graph->pattern == NCCL_TOPO_PATTERN_RING && splitNvLink) {
    // 我们有两个socket，通过NVLink连接，但中间有较慢的链路（通常是QPI）
    // TREE模式在这种场景下效果更好，但它至少需要2个通道
    // 由于TREE需要和RING使用相同数量的通道，所以也强制RING使用2个通道
    // We have two sockets with NVLink and a slower link in between (typically QPI).
    // Tree is likely going to work better but it needs at least 2 channels.
    // Since Tree needs to have the same number of channels as Ring, also force Ring to use 2 channels.

    // 如果最大通道数>=2且最小通道数==1，则将最小通道数设置为2
    if (graph->maxChannels >= 2 && graph->minChannels == 1)
        graph->minChannels = 2;
  }

// 创建临时图对象，用于搜索过程中的试探性配置
  struct ncclTopoGraph tmpGraph;

  // 复制原始图到临时图
  memcpy(&tmpGraph, graph, sizeof(struct ncclTopoGraph));

  // 搜索策略：
  // 第一阶段：首先尝试跨网卡，然后降低带宽，最后增加节点内带宽
  // 第二阶段：从找到的解开始，尝试提高带宽
  // First try crossnic, then decrease bw and finally increase bwIntra.

  // 带宽数组的元素数量
  int nspeeds = 0;

  // 带宽数组指针
  float* speedArray = NULL;

  // 根据是否有网络设备和GPU架构选择合适的带宽数组
  if (system->nodes[NET].count == 0) {
    // 单节点场景：使用节点内带宽数组
    // 根据计算能力选择对应架构的带宽数组
    nspeeds = ccMin >= 100 ? NSPEEDSINTRA_SM100 : (ccMin >= 90 ? NSPEEDSINTRA_SM90 : NSPEEDSINTRA);
    speedArray = ccMin >= 100 ? sm100SpeedArrayIntra : (ccMin >= 90 ? sm90SpeedArrayIntra : speedArrayIntra);
  } else {
    // 多节点场景：使用节点间带宽数组
    // 根据计算能力选择对应架构的带宽数组
    nspeeds = ccMin >= 100 ? NSPEEDSINTER_SM100 : (ccMin >= 90 ? NSPEEDSINTER_SM90 : NSPEEDSINTER);
    speedArray = ccMin >= 100 ? sm100SpeedArrayInter : (ccMin >= 90 ? sm90SpeedArrayInter : speedArrayInter);
  }

  // 搜索阶段标志：1=第一阶段（降低带宽找解），2=第二阶段（提高带宽优化）
  int pass = 1;

  // 带宽数组的索引（从高到低尝试）
  int speedIndex = 0;

  // 系统的最大单链路带宽
  float maxBw = system->maxBw;

  // 系统的总带宽
  float totalBw = system->totalBw;

  // 对于非RING模式，调整总带宽计算
  // RING模式每个节点只参与一条路径，其他模式可能需要更多路径
  if (ngpus > 1 && graph->pattern != NCCL_TOPO_PATTERN_RING)
    totalBw *= ngpus*1.0/(ngpus-1);

  // 调整初始带宽索引：
  // 跳过那些超过最大单链路带宽 或 超过总带宽（考虑最小通道数）的带宽值
  while ((speedArray[speedIndex] > maxBw || speedArray[speedIndex]*graph->minChannels > totalBw) && speedIndex < nspeeds-1)
    speedIndex++;

  // 设置临时图的初始带宽（从调整后的索引开始）
  tmpGraph.bwIntra = tmpGraph.bwInter = speedArray[speedIndex];

  // 全局搜索超时时间（微秒）
  int64_t globalTimeout = NCCL_SEARCH_GLOBAL_TIMEOUT;

// 搜索标签：用于多次搜索尝试的跳转点
search:
  // 根据图类型和sameChannels标志确定本次搜索的超时时间
  // sameChannels模式需要更长的超时时间，因为搜索空间更大
  // TREE模式使用较短的超时时间
  int time = tmpGraph.sameChannels ? NCCL_SEARCH_TIMEOUT_SAMECHANNELS :
    tmpGraph.pattern == NCCL_TOPO_PATTERN_TREE ? NCCL_SEARCH_TIMEOUT_TREE : NCCL_SEARCH_TIMEOUT;

  // 重置临时图的通道数量为0（搜索开始前清空）
  tmpGraph.nChannels = 0;

  // 从全局超时时间中减去本次搜索的超时时间
  globalTimeout -= time;

  // 执行递归搜索
  // 输入：tmpGraph（搜索参数）
  // 输出：graph（最佳结果），time（剩余时间，-1表示找到完美解，0表示超时）
  NCCLCHECK(ncclTopoSearchRec(system, &tmpGraph, graph, &time));
// 调试输出代码块（已禁用）
// 如果需要启用搜索过程的详细输出，将#if 0改为#if 1
#if 0
  // 打印搜索参数和结果
  printf("Id %d Pattern %d, crossNic %d, Bw %g/%g, type %d/%d, channels %d-%d sameChannels %d -> nChannels %dx%g/%g %s\n", tmpGraph.id, tmpGraph.pattern, tmpGraph.crossNic, tmpGraph.bwInter, tmpGraph.bwIntra, tmpGraph.typeInter, tmpGraph.typeIntra, tmpGraph.minChannels, tmpGraph.maxChannels, tmpGraph.sameChannels, graph->nChannels, graph->bwInter, graph->bwIntra, time == 0 ? "TIMEOUT" : time == -1 ? "PERFECT" : "");

  // 打印每个通道的详细信息
  for (int c=0; c<graph->nChannels; c++) {
    // 打印通道编号
    printf("%2d : ", c);

    // 打印该通道中所有GPU的rank
    for (int g=0; g<ngpus; g++) {
      printf("%d ", graph->intra[c*ngpus+g]);
    }

    // 打印该通道的两个网络设备ID（十六进制）
    printf("[%lx %lx]", graph->inter[c*2+0], graph->inter[c*2+1]);

    // 换行
    printf("\n");
  }
#endif

  // 检查是否找到最优解
  // time == -1 表示找到完美解（满足所有条件，无法进一步优化）
  // Optimal solution, stop here
  if (time == -1)
    goto done;

  // 检查是否已经达到系统总带宽
  // 如果 通道数*节点间带宽 >= 系统总带宽，说明已经充分利用了硬件，无需继续搜索
  if (graph->nChannels*graph->bwInter >= system->totalBw)
    goto done;

// 第一阶段搜索：降低带宽和放宽约束以找到可行解
  if (pass == 1) {
    // 第一阶段：还没有找到完美解；尝试其他选项
    // First pass, we don't have a solution yet ; try other options

    // 尝试使用不同的通道配置（除了AMD CPU + PATH_SYS的情况）
    // AMD CPU的PATH_SYS路径性能较差，不适合使用不同通道
    // Try having different channels (except when going through AMD CPUs)
    if (tmpGraph.sameChannels == 1 &&
        !(cpuArch == NCCL_TOPO_CPU_ARCH_X86 && cpuVendor == NCCL_TOPO_CPU_VENDOR_AMD && tmpGraph.typeIntra == PATH_SYS)) {
      // 设置sameChannels为0，允许使用不同的通道配置
      tmpGraph.sameChannels = 0;

      // 跳转到search标签，重新搜索
      goto search;
    }

    // 恢复sameChannels为原始值
    tmpGraph.sameChannels = trySameChannels;

    // 调整全局超时时间
    // 如果没有超时，将剩余时间加回全局超时
    // 如果已经找到完美解（time==-1），重置全局超时
    if (time != -1)
        globalTimeout += time;
    else
        globalTimeout = NCCL_SEARCH_GLOBAL_TIMEOUT;

    // 如果全局超时时间耗尽且已经找到一些通道，停止搜索
    if (globalTimeout < 0 && graph->nChannels)
        goto done;

    // 尝试更简单的树形结构
    // 对于计算能力>=90的GPU，如果当前是BALANCED_TREE，尝试普通的TREE
    // Try a simpler tree
    if (ccMin >= 90 && tmpGraph.pattern == NCCL_TOPO_PATTERN_BALANCED_TREE) {
      // 切换到TREE模式
      tmpGraph.pattern = NCCL_TOPO_PATTERN_TREE;

      // 重新搜索
      goto search;
    }

    // 恢复原始模式
    tmpGraph.pattern = graph->pattern;

    // 尝试增加节点内路径类型（使用更慢但可能更可行的路径）
    // 确定最大节点内路径类型
    int maxIntra = system->nodes[NET].count > 0 ? tmpGraph.typeInter : maxTypeIntra;

    // 如果当前节点内路径类型小于最大值 且 （没有找到通道 或 当前类型小于之前找到的类型）
    if (tmpGraph.typeIntra < maxIntra && (graph->nChannels == 0 || tmpGraph.typeIntra < graph->typeIntra)) {
      // 增加路径类型（使用更慢的路径）
      tmpGraph.typeIntra += 1;

      // 如果路径类型仍然小于PATH_DIS（断开的路径），尝试搜索
      if (tmpGraph.typeIntra < PATH_DIS) goto search;
    }

    // 恢复节点内路径类型为最小值
    tmpGraph.typeIntra = minTypeIntra;

    // 尝试增加节点间路径类型（使用更慢但可能更可行的路径）
    // 条件：有网络设备 AND 当前类型小于最大值 AND （没有找到通道 或 可以优化）
    if (system->nodes[NET].count > 0 && tmpGraph.typeInter < maxTypeInter && (graph->nChannels == 0 || tmpGraph.typeInter < graph->typeInter || tmpGraph.typeInter < PATH_PXN)) {
      // 增加节点间路径类型
      tmpGraph.typeInter += 1;

      // 如果路径类型仍然小于PATH_DIS，尝试搜索
      if (tmpGraph.typeInter < PATH_DIS) goto search;
    }

    // 恢复节点间路径类型为最小值
    tmpGraph.typeInter = minTypeInter;

    // 尝试启用跨网卡通信
    // 条件：允许跨网卡（crossNic==2）AND 当前未启用 AND 模式是RING或BALANCED_TREE
    if (crossNic == 2 && tmpGraph.crossNic == 0
        && (graph->pattern == NCCL_TOPO_PATTERN_RING || graph->pattern == NCCL_TOPO_PATTERN_BALANCED_TREE)) {
      // 启用跨网卡通信
      // Try again with crossNic if permitted
      tmpGraph.crossNic = 2;

      // 重新搜索
      goto search;
    }

    // 恢复跨网卡标志
    tmpGraph.crossNic = crossNic == 1 ? 1 : 0;

    // 降低带宽直到找到解
    // 条件：还有更低的带宽可尝试 AND （没有找到通道 或 下一个带宽与当前带宽比值>0.49）
    // Decrease bw until we find a solution
    if ((speedIndex < nspeeds-1) && (graph->nChannels == 0 || (speedArray[speedIndex+1]/graph->bwInter > .49))) {
      // 移动到下一个更低的带宽
      tmpGraph.bwInter = tmpGraph.bwIntra = speedArray[++speedIndex];

      // 重新搜索
      goto search;
    }

    // 重置带宽索引到最大可行带宽
    speedIndex = 0;

    // 跳过那些超过最大单链路带宽的值
    while (speedArray[speedIndex] > maxBw && speedIndex < nspeeds-1)
        speedIndex++;

    // 设置带宽为调整后的值
    tmpGraph.bwIntra = tmpGraph.bwInter = speedArray[speedIndex];

  }

// 完成标签：已经找到解，准备进入第二阶段优化
done:
  // 如果在第一阶段，已经找到了一个解，现在进入第二阶段（提高带宽优化）
  // We have a solution. Start from that solution and move to pass 2.
  if (pass == 1) {
    // 设置time为-1（表示已经找到解）
    time = -1;

    // 尝试复制通道以增加并行度
    NCCLCHECK(ncclTopoDupChannels(graph, ccMin, ngpus));

    // 将找到的图复制到临时图，作为第二阶段优化的起点
    memcpy(&tmpGraph, graph, sizeof(tmpGraph));

    // 重置带宽索引到最高带宽
    speedIndex = 0;

    // 跳过那些超过当前图带宽的值，找到对应的起始点
    while (speedArray[speedIndex] > graph->bwInter && speedIndex < nspeeds-1)
        speedIndex++;

    // 设置临时图的带宽（从最高可行带宽开始）
    tmpGraph.bwIntra = tmpGraph.bwInter = speedArray[speedIndex];

    // 设置最小通道数为当前找到的通道数（第二阶段不会减少通道数）
    tmpGraph.minChannels = graph->nChannels;

    // 进入第二阶段
    pass = 2;
  }

  // 第二阶段：从找到的解开始，尝试提高带宽
  if (pass == 2) {
    // 检查是否可以提高带宽
    // 条件：没有超时 且 还有更高的带宽可尝试
    // See if we can increase bw
    if (time != 0 && speedIndex > 0) {
      // 对于RING模式，同时提高节点内和节点间带宽
      if (graph->pattern == NCCL_TOPO_PATTERN_RING) {
        // 移动到前一个更高的带宽值
        // increase bw for Ring
        tmpGraph.bwIntra = tmpGraph.bwInter = speedArray[--speedIndex];

        // 重新搜索
        goto search;
      } else if (graph->pattern == NCCL_TOPO_PATTERN_NVLS && tmpGraph.bwInter == graph->bwInter && tmpGraph.bwInter < tmpGraph.bwIntra*2) {
        // 对于NVLS模式，只提高节点间带宽（保持节点内带宽不变）
        // 条件：当前节点间带宽等于图的节点间带宽 且 节点间带宽小于节点内带宽的2倍

        // 固定通道数为当前值
        tmpGraph.minChannels = tmpGraph.maxChannels = graph->nChannels;

        // 只提高节点间带宽
        tmpGraph.bwInter = speedArray[--speedIndex];

        // 重新搜索
        goto search;
      } else if (tmpGraph.bwIntra == graph->bwIntra && tmpGraph.bwIntra < tmpGraph.bwInter*2) {
        // 对于其他模式（主要是TREE），只提高节点内带宽
        // 条件：当前节点内带宽等于图的节点内带宽 且 节点内带宽小于节点间带宽的2倍
        // 这通常用于2节点或COLLNET场景

        // 只提高节点内带宽
        // increase bwIntra for trees (2 nodes or collnet)
        tmpGraph.bwIntra = speedArray[--speedIndex];

        // 重新搜索
        goto search;
      }
    }

    // 无法继续提高带宽，设置time为-1表示搜索完成
    time = -1;

    // 将最终结果复制回临时图
    memcpy(&tmpGraph, graph, sizeof(tmpGraph));
  }

  // 如果没有找到任何通道 且 不是COLLNET模式 且 不是NVLS模式
  // 使用简单的顺序配置作为后备方案
  if (graph->nChannels == 0 && graph->collNet == 0 && graph->pattern != NCCL_TOPO_PATTERN_NVLS) {
    // 输出警告信息：无法为该模式找到路径，使用简单顺序
    INFO(NCCL_GRAPH, "Could not find a path for pattern %d, falling back to simple order", graph->pattern);

    // 按照GPU在系统中的顺序设置rank
    for (int i=0; i<ngpus; i++)
        graph->intra[i] = system->nodes[GPU].nodes[i].gpu.rank;

    // 设置网络设备为0（无效值，表示没有网络设备）
    graph->inter[0] = graph->inter[1] = 0;

    // 设置较低的带宽（0.1 GB/s，表示这是后备方案）
    graph->bwIntra = graph->bwInter = 0.1;

    // 设置路径类型为系统总线（最慢的路径）
    graph->typeIntra = graph->typeInter = PATH_SYS;

    // 设置通道数为1（最简单的配置）
    graph->nChannels = 1;
  }

  // 返回成功
  return ncclSuccess;
}

// 打印拓扑图的详细信息（用于调试）
// 参数说明：
// - system: 拓扑系统对象
// - graph: 要打印的拓扑图对象
ncclResult_t ncclTopoPrintGraph(struct ncclTopoSystem* system, struct ncclTopoGraph* graph) {
  // 打印图的基本信息：模式、跨网卡标志、通道数、带宽、链路类型、sameChannels标志
  INFO(NCCL_GRAPH, "Pattern %d, crossNic %d, nChannels %d, bw %f/%f, type %s/%s, sameChannels %d", graph->pattern, graph->crossNic, graph->nChannels, graph->bwIntra, graph->bwInter, topoPathTypeStr[graph->typeIntra], topoPathTypeStr[graph->typeInter], graph->sameChannels);

  // 获取GPU数量
  int ngpus = system->nodes[GPU].count;

  // 字符行缓冲区，用于构建单行输出
  char line[1024];

  // 遍历所有通道
  for (int c=0; c<graph->nChannels; c++) {
    // 首先打印通道编号
    sprintf(line, "%2d :", c);

    // 获取当前行的长度（用于后续追加）
    int offset = strlen(line);

    // 如果有网络设备，打印第一个网络设备（通道起点）
    if (system->nodes[NET].count > 0) {
      // 格式： " NET/系统ID-本地ID"
      sprintf(line+offset, " %s/%lx-%lx", topoNodeTypeStr[NET], NCCL_TOPO_ID_SYSTEM_ID(graph->inter[2*c]), NCCL_TOPO_ID_LOCAL_ID(graph->inter[2*c]));

      // 更新偏移量
      offset = strlen(line);
    }

    // 遍历该通道的所有GPU
    for (int i=0; i<ngpus; i++) {
      // GPU索引
      int g;

      // 将rank转换为索引
      ncclTopoRankToIndex(system, graph->intra[ngpus * c + i], &g, true);

      // 获取GPU的拓扑ID
      int64_t topoId = system->nodes[GPU].nodes[g].id;

      // 打印GPU信息：格式 " GPU/系统ID-本地ID"
      sprintf(line + offset, " %s/%lx-%lx", topoNodeTypeStr[GPU], NCCL_TOPO_ID_SYSTEM_ID(topoId), NCCL_TOPO_ID_LOCAL_ID(topoId));

      // 更新偏移量
      offset = strlen(line);

      // 如果是NVLS图（id==3），只使用第一个GPU
      // NVLS graphs only use the first GPU
      if (graph->id == 3) break;
    }

    // 如果有网络设备，打印第二个网络设备（通道终点）
    if (system->nodes[NET].count > 0) {
      // 格式： " NET/系统ID-本地ID"
      sprintf(line+offset, " %s/%lx-%lx", topoNodeTypeStr[NET], NCCL_TOPO_ID_SYSTEM_ID(graph->inter[2*c+1]), NCCL_TOPO_ID_LOCAL_ID(graph->inter[2*c+1]));

      // 更新偏移量
      offset = strlen(line);
    }

    // 打印该通道的完整信息
    INFO(NCCL_GRAPH, "%s", line);
  }

  // 返回成功
  return ncclSuccess;
}

// 将多个拓扑图导出到XML文件（用于调试）
// 检查环境变量NCCL_GRAPH_DUMP_FILE，如果设置了，将图导出到指定文件
// 参数说明：
// - system: 拓扑系统对象
// - ngraphs: 图的数量
// - graphs: 图指针数组
ncclResult_t ncclTopoDumpGraphs(struct ncclTopoSystem* system, int ngraphs, struct ncclTopoGraph** graphs) {
  // 返回值
  ncclResult_t ret = ncclSuccess;

  // 获取环境变量NCCL_GRAPH_DUMP_FILE的值
  const char* str = ncclGetEnv("NCCL_GRAPH_DUMP_FILE");

  // XML对象指针（初始化为NULL）
  struct ncclXml* xml = NULL;

  // 如果设置了环境变量
  if (str) {
    // 输出环境变量信息
    INFO(NCCL_ENV, "NCCL_GRAPH_DUMP_FILE set by environment to %s", str);

    // 分配XML对象内存
    NCCLCHECK(xmlAlloc(&xml, NCCL_GRAPH_XML_MAX_NODES));

    // 将图转换为XML格式
    // 如果失败，跳转到fail标签
    NCCLCHECKGOTO(ncclTopoGetXmlFromGraphs(ngraphs, graphs, system, xml), ret, fail);

    // 将XML写入文件
    // 如果失败，跳转到fail标签
    NCCLCHECKGOTO(ncclTopoDumpXmlToFile(str, xml), ret, fail);
  }

// 正常退出标签
exit:
  // 释放XML对象内存
  if (xml) free(xml);

  // 返回结果
  return ret;

// 失败处理标签
fail:
  // 跳转到exit标签进行清理
  goto exit;
}

// 包含通信器相关的定义
#include "comm.h"

// 获取NVLS模式下对应rank的网卡设备
// NVLS通道不是计算通道，需要找到对应rank作为head时使用的NIC
// 参数说明：
// - comm: 通信器对象
// - graph: 拓扑图对象
// - channelId: 通道ID
// - netId: 输出参数，返回网卡ID
// NVLS channels aren't compute channels. Find which NIC corresponds to our rank being the head
ncclResult_t getNvlsNetDev(struct ncclComm* comm, struct ncclTopoGraph* graph, int channelId, int64_t* netId) {
  // 返回值
  ncclResult_t ret = ncclSuccess;

  // 获取本地rank数量
  int localRanks = comm->topo->nodes[GPU].count;

  // 找到的网络设备数量
  int netNum = 0;

  // 网络设备ID数组（存储所有该rank作为head的通道的网络设备）
  int64_t net[MAXCHANNELS];

  // 遍历图的所有通道
  for (int c = 0; c < graph->nChannels; c++) {
    // 检查该通道的第一个GPU（head）是否是当前rank
    if (graph->intra[c * localRanks] == comm->rank) {
      // 记录该通道的网络设备ID
      net[netNum++] = graph->inter[c * 2];
    }
  }

  // 如果找到了至少一个网络设备
  if (netNum) {
    // 根据channelId选择一个网络设备（使用取模运算循环使用）
    *netId = net[channelId % netNum];
  } else {
    // 没有找到网络设备，返回内部错误
    ret = ncclInternalError;

    // 跳转到失败处理
    goto fail;
  }

// 正常退出标签
exit:
  // 返回结果
  return ret;

// 失败处理标签
fail:
  // 输出警告信息
  WARN("Could not find NIC for rank %d in NVLS graph", comm->rank);

  // 跳转到exit标签
  goto exit;
}

// P2P通信的PXN（Proxy eXtension for Network）级别参数
// 0: 不在P2P中使用PXN
// 1: 必要时使用PXN
// 2: 尽可能使用PXN以最大化聚合
// 0: don't use PXN for P2P, 1: use PXN if needed, 2: use PXN as much as possible to maximize aggregation
NCCL_PARAM(P2pPxnLevel, "P2P_PXN_LEVEL", 2);

// 获取网络设备信息（用于P2P通信）
// 参数说明：
// - comm: 通信器对象
// - rank: 当前rank
// - graph: 拓扑图对象（可以为NULL）
// - channelId: 通道ID
// - peerRank: 对端rank（-1表示没有对端）
// - id: 输出参数，返回网络拓扑ID
// - dev: 输出参数，返回网络设备编号
// - proxyRank: 输出参数，返回代理rank（用于PXN）
ncclResult_t ncclTopoGetNetDev(struct ncclComm* comm, int rank, struct ncclTopoGraph* graph, int channelId, int peerRank, int64_t* id, int* dev, int* proxyRank) {
  // 网络拓扑ID（初始化为-1表示无效）
  int64_t netId = -1;

  // 网络设备编号（初始化为-1表示无效）
  int netDev = -1;

  // 如果有图对象，从图中获取网络设备信息
  if (graph) {
    // 使用图中定义的网络设备
    // Honor the net device in the graph

    // 计算实际使用的通道（使用取模运算循环使用通道）
    int channel = channelId%graph->nChannels;

    // 获取GPU数量
    int ngpus = comm->topo->nodes[GPU].count;

    // 确定使用哪个网络设备：
    // 如果当前rank是通道的第一个GPU，使用索引0的网络设备
    // 否则使用索引1的网络设备
    int index = graph->intra[channel*ngpus] == rank ? 0 : 1;

    // 根据通信模式获取网络设备ID
    if (graph->pattern != NCCL_TOPO_PATTERN_NVLS) {
      // 非NVLS模式：直接从图中获取
      netId = graph->inter[channel*2+index];
    } else {
      // NVLS模式：调用特殊函数获取网络设备
      NCCLCHECK(getNvlsNetDev(comm, graph, channelId, &netId));
    }

    // 将拓扑ID转换为设备编号
    NCCLCHECK(ncclTopoIdToNetDev(comm->topo, netId, &netDev));

    // 如果调用者请求设备编号，设置输出参数
    if (dev) *dev = netDev;

    // 如果调用者请求拓扑ID，设置输出参数
    if (id) *id = netId;

    // 获取代理rank（用于PXN通信）
    NCCLCHECK(ncclTopoGetIntermediateRank(comm->topo, rank, netId, proxyRank));
  } else if (peerRank == -1) {
    // 如果没有图对象且没有对端rank，返回内部错误
    return ncclInternalError;
  } else {
    // 没有图对象，需要根据拓扑和对端rank信息确定网络设备
    // Start with our local NIC and local Rank

    // 获取本地rank对应的网络设备
    NCCLCHECK(ncclTopoGetLocalNet(comm->topo, rank, channelId, &netId, &netDev));

    // 如果调用者请求设备编号，设置输出参数
    if (dev) *dev = netDev;

    // 如果调用者请求拓扑ID，设置输出参数
    if (id) *id = netId;

    // 初始化代理rank为当前rank（表示不使用代理）
    *proxyRank = rank;

    // 获取PXN级别
    // 如果PXN被禁用，级别为0；否则使用参数值
    int pxnLevel = ncclPxnDisable(comm) == 1 ? 0 : ncclParamP2pPxnLevel();

    // 检查是否可以使用对端rank的首选设备
    // 条件：禁用跨网卡 或 PXN级别不为0
    // See whether we can use the remote rank preferred device.
    if (ncclParamCrossNic() == 0 || (pxnLevel != 0)) {
      // 查找靠近对端NVML设备的本地网卡编号
      // Find local NIC number close to local nvmlDev

      // 获取对端的NVML设备号
      int nvmlDev = comm->peerInfo[peerRank].nvmlDev;

      // 将NVML设备号转换为本地rank
      int localRank;
      if (ncclTopoDevToRank(comm->topo, nvmlDev, &localRank) != ncclSuccess) return ncclSuccess;

      // 获取该本地rank对应的网络设备
      NCCLCHECK(ncclTopoGetLocalNet(comm->topo, localRank, channelId, &netId, &netDev));

      // 检查设备是否存在于当前节点
      // Check that device exists on our node
      if (ncclParamCrossNic() == 0) {
        // 不允许跨网卡：使用找到的设备
        if (dev) *dev = netDev;
        if (id) *id = netId;
      }

      // PXN级别1：必要时使用PXN
      if (pxnLevel == 1) {
        // GPU和网络设备索引
        int g, n;

        // 将当前rank转换为GPU索引
        NCCLCHECK(ncclTopoRankToIndex(comm->topo, rank, &g, /*showWarn=*/true));

        // 将网络拓扑ID转换为网络设备索引
        NCCLCHECK(ncclTopoIdToIndex(comm->topo, NET, netId, &n));

        // 获取当前GPU节点
        struct ncclTopoNode* gpu = comm->topo->nodes[GPU].nodes+g;

        // 如果GPU到该网络设备的路径类型<=PATH_PXN，可以使用PXN
        if (gpu->paths[NET][n].type <= PATH_PXN) {
          // 使用该网络设备和PXN代理
          if (dev) *dev = netDev;
          if (id) *id = netId;
          NCCLCHECK(ncclTopoGetIntermediateRank(comm->topo, rank, *dev, proxyRank));
        }
      } else if (pxnLevel == 2) {
        // PXN级别2：尽可能使用PXN以最大化聚合
        // 检查哪个本地GPU对应该网卡，看是否可以使用PXN
        // Check which local GPU corresponds to that NIC and see if we can use PXN.

        // 索引变量
        int n, g1, g2;

        // 将网络拓扑ID转换为网络设备索引
        NCCLCHECK(ncclTopoIdToIndex(comm->topo, NET, netId, &n));

        // 将当前rank转换为GPU索引
        NCCLCHECK(ncclTopoRankToIndex(comm->topo, rank, &g1, /*showWarn=*/true));

        // 获取对应该网卡的本地GPU索引
        NCCLCHECK(ncclTopoGetLocalGpu(comm->topo, netId, &g2));

        // 如果找到了对应的GPU
        if (g2 != -1) {
          // 获取对端GPU节点
          struct ncclTopoNode* peerGpu = comm->topo->nodes[GPU].nodes+g2;

          // 确定PXN类型：如果启用C2C，使用P2C；否则使用PXB
          int pxnType = ncclParamPxnC2c() ? PATH_P2C : PATH_PXB;

          // 检查路径条件：
          // 1. 对端GPU到当前GPU的路径类型<=PATH_NVL（有NVLink连接）
          // 2. 对端GPU到网卡的路径类型<=PXN类型
          if (peerGpu->paths[GPU][g1].type <= PATH_NVL && peerGpu->paths[NET][n].type <= pxnType) {
            // 使用对端GPU作为代理
            *proxyRank = peerGpu->gpu.rank;

            // 设置网络设备信息
            if (dev) *dev = netDev;
            if (id) *id = netId;

            // 返回成功
            return ncclSuccess;
          }
        }
      }
    }
  }

  // 返回成功
  return ncclSuccess;
}
