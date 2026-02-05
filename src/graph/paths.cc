/*************************************************************************
 * Copyright (c) 2018-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

/*
 * paths.cc - NCCL 拓扑路径计算模块
 *
 * 本文件负责计算和管理系统中各节点之间的路径信息，包括：
 * 1. GPU 到 GPU 的路径（P2P 通信）
 * 2. GPU 到 NIC 的路径（网络通信）
 * 3. GPU 到 CPU 的路径（亲和性绑定）
 * 4. 路径类型判定（NVLink/PCIe/NET 等）
 * 5. GPUDirect RDMA 支持检测
 * 6. P2P 能力检查
 * 7. 通道数量计算
 */

#include "core.h"
#include "graph.h"
#include "topo.h"
#include "comm.h"
#include "net.h"
#include "channel.h"
#include "transport.h"
#include "device.h"

// Pre-compute GPU->NIC, GPU->GPU and NIC->GPU paths
// 预计算 GPU→NIC、GPU→GPU 和 NIC→GPU 的路径
// 目的：避免在后续通信图构建时重复计算路径，提高性能

// ============================================================================
// ncclTopoNodeList - 节点列表结构（用于 BFS 遍历）
// ============================================================================
struct ncclTopoNodeList {
  struct ncclTopoNode* list[NCCL_TOPO_MAX_NODES];  // 节点指针数组
  int count;                                         // 节点数量
};

// ============================================================================
// getPath - 获取节点到指定类型/ID 节点的路径
// ============================================================================
// 参数：
//   system - 拓扑系统
//   node   - 源节点
//   t      - 目标节点类型
//   id     - 目标节点ID
//   path   - 输出：路径指针
//
// 返回：ncclSuccess 或 ncclInternalError
//
// 说明：在 node->paths[t] 数组中查找 ID 为 id 的节点路径
static ncclResult_t getPath(struct ncclTopoSystem* system, struct ncclTopoNode* node, int t, int64_t id, struct ncclTopoLinkList** path) {
  // 遍历系统中所有类型为 t 的节点
  for (int i=0; i<system->nodes[t].count; i++) {
    if (system->nodes[t].nodes[i].id == id) {
      // 找到目标节点，返回其路径
      *path = node->paths[t]+i;
      return ncclSuccess;
    }
  }
  WARN("Could not find node of type %d id %lx", t, id);
  return ncclInternalError;
}

// ============================================================================
// 环境变量参数定义
// ============================================================================
// NVB_DISABLE: 是否禁用 NVBridge（通过 GPU 转发）
NCCL_PARAM(NvbDisable, "NVB_DISABLE", 0);

// ============================================================================
// ncclTopoSetPaths - 使用 BFS 计算从 baseNode 到所有其他节点的路径
// ============================================================================
// 参数：
//   baseNode - 起始节点（计算从该节点出发的所有路径）
//   system   - 拓扑系统
//
// 返回：ncclSuccess 或错误码
//
// 算法：广度优先搜索（BFS）
// - 从 baseNode 开始，逐层扩展
// - 计算到同类型所有节点（GPU/GPU, GPU/NIC 等）的路径
// - 记录路径的带宽、类型、跳数
static ncclResult_t ncclTopoSetPaths(struct ncclTopoNode* baseNode, struct ncclTopoSystem* system) {
  // ========================================================================
  // 1. 初始化路径数组
  // ========================================================================
  // 为 baseNode->paths[baseNode->type] 分配内存
  // 例如：baseNode 是 GPU，则分配 paths[GPU][同类型节点数量]
  if (baseNode->paths[baseNode->type] == NULL) {
    NCCLCHECK(ncclCalloc(baseNode->paths+baseNode->type, system->nodes[baseNode->type].count));
    // 初始化所有路径为 PATH_DIS（断开/不可达）
    for (int i=0; i<system->nodes[baseNode->type].count; i++) baseNode->paths[baseNode->type][i].type = PATH_DIS;
  }

  // breadth-first search to set all paths to that node in the system
  // 使用广度优先搜索设置系统中所有到 baseNode 的路径
  struct ncclTopoNodeList nodeList;          // 当前层的节点列表
  struct ncclTopoNodeList nextNodeList = { { 0 }, 0 };  // 下一层的节点列表
  nodeList.count = 1; nodeList.list[0] = baseNode;   // BFS 从 baseNode 开始

  // 获取 baseNode 到自己的路径
  struct ncclTopoLinkList* basePath;
  NCCLCHECK(getPath(system, baseNode, baseNode->type, baseNode->id, &basePath));
  basePath->count = 0;         // 0 跳（到自己）
  basePath->bw = LOC_BW;       // 5000 GB/s（本地访问带宽）
  basePath->type = PATH_LOC;   // 路径类型：本地

  // ========================================================================
  // 2. BFS 逐层遍历
  // ========================================================================
  while (nodeList.count) {
    nextNodeList.count = 0;

    // 遍历当前层的所有节点
    for (int n=0; n<nodeList.count; n++) {
      struct ncclTopoNode* node = nodeList.list[n];

      // 获取 node 到 baseNode 的路径
      struct ncclTopoLinkList* path;
      NCCLCHECK(getPath(system, node, baseNode->type, baseNode->id, &path));

      // 遍历 node 的所有连接
      for (int l=0; l<node->nlinks; l++) {
        struct ncclTopoLink* link = node->links+l;              // 当前连接
        struct ncclTopoNode* remNode = link->remNode;         // 连接的远端节点

        // 为 remNode 分配路径数组（如果尚未分配）
        if (remNode->paths[baseNode->type] == NULL) {
          NCCLCHECK(ncclCalloc(remNode->paths+baseNode->type, system->nodes[baseNode->type].count));
          for (int i=0; i<system->nodes[baseNode->type].count; i++)
            remNode->paths[baseNode->type][i].type = PATH_DIS;
        }

        // 获取 remNode 到 baseNode 的路径
        struct ncclTopoLinkList* remPath;
        NCCLCHECK(getPath(system, remNode, baseNode->type, baseNode->id, &remPath));

        // 计算路径带宽：取当前路径带宽和链路带宽的最小值
        // 瓶颈原理：路径的总带宽取决于最慢的那一段
        float bw = std::min(path->bw, link->bw);

        // allow routing through a GPU only as 1 hop
        // 允许通过 GPU 转发，但只能转发 1 跳（避免多跳延迟累积）
        if (node != baseNode && node->type == GPU &&
            (ncclParamNvbDisable() || link->type != LINK_NVL || remNode->type != GPU || path->count > 1))
          continue;  // 不允许通过 GPU 转发（满足以下任一条件即禁止）：
                     // - NVB 被禁用
                     // - 不是 NVLink 连接
                     // - 远端不是 GPU
                     // - 已经超过 1 跳

        // ======================================================================
        // 更新路径（如果找到更优路径）
        // ======================================================================
        // 条件：
        // 1. remPath->bw == 0：尚未找到路径
        // 2. remPath->count > path->count：当前路径跳数更少
        // 3. remPath->bw < bw：当前路径带宽更大
        if ((remPath->bw == 0 || remPath->count > path->count) && remPath->bw < bw) {
          // Find reverse link
          // 找到反向连接（从 remNode 指向 node 的连接）
          for (int l=0; l<remNode->nlinks; l++) {
            if (remNode->links[l].remNode == node && remNode->links[l].type == link->type) {
              remPath->list[0] = remNode->links+l;  // 路径第一步：反向连接
              break;
            }
          }

          // 检查是否找到反向连接
          if (remPath->list[0] == NULL) {
            WARN("Failed to find reverse path from remNode %d/%lx nlinks %d to node %d/%lx",
                 remNode->type, remNode->id, remNode->nlinks, node->type, node->id);
            return ncclInternalError;
          }

          // Copy the rest of the path
          // 复制剩余路径（从 node 到 baseNode）
          for (int i=0; i<path->count; i++) remPath->list[i+1] = path->list[i];

          // 更新路径信息
          remPath->count = path->count + 1;      // 跳数 +1
          remPath->bw = bw;                      // 更新带宽

          // ========================================================================
          // 计算路径类型
          // ========================================================================
          // Start with path type = link type. PATH and LINK types are supposed to match.
          // Don't consider LINK_NET as we only care about the NIC->GPU path.
          int type = link->type == LINK_NET ? LINK_LOC : link->type;

          // Differentiate between one and multiple PCI switches
          // 区分单个和多个 PCIe 桥
          if (node->type == PCI && remNode->type == PCI)
            type = PATH_PXB;  // 多个 PCIe 桥

          // Consider a path going through the CPU as PATH_PHB
          // 经过 CPU 的路径标记为 PHB（PCI Host Bridge）
          if (link->type == LINK_PCI && (node->type == CPU || link->remNode->type == CPU))
            type = PATH_PHB;

          // Set 1 hop NVLink as NVB
          // 1 跳 NVLink 转发标记为 NVB（NVBridge）
          if (node->type == GPU && path->type == PATH_NVL && type == PATH_NVL && remPath->count > 1)
            type = PATH_NVB;

          // 路径类型 = max(当前路径类型, 新链路类型)
          // 取较差的类型（数字越大，路径越差）
          remPath->type = std::max(path->type, type);

          // Add to the list for the next iteration if not already in the list
          // 将 remNode 加入下一层列表（如果尚未加入）
          int i;
          for (i=0; i<nextNodeList.count; i++) if (nextNodeList.list[i] == remNode) break;
          if (i == nextNodeList.count) nextNodeList.list[nextNodeList.count++] = remNode;
        }
      }
    }
    // 下一层成为当前层
    memcpy(&nodeList, &nextNodeList, sizeof(nodeList));
  }
  return ncclSuccess;
}

// ============================================================================
// printNodePaths - 打印节点的所有路径
// ============================================================================
// 参数：
//   system - 拓扑系统
//   node   - 要打印的节点
//
// 输出格式：
//   ENABLE_TRACE 模式：详细显示每一步路径
//   普通模式：压缩显示（count/bw/type）
static void printNodePaths(struct ncclTopoSystem* system, struct ncclTopoNode* node) {
  const int linesize = 1024;
  char line[linesize];

#ifdef ENABLE_TRACE
  // 详细模式：显示路径标题
  INFO(NCCL_GRAPH, "Paths from %s/%lx-%lx :", topoNodeTypeStr[node->type], NCCL_TOPO_ID_SYSTEM_ID(node->id), NCCL_TOPO_ID_LOCAL_ID(node->id));
#else
  // 普通模式：构建路径字符串
  snprintf(line, linesize, "%s/%lx-%lx :", topoNodeTypeStr[node->type], NCCL_TOPO_ID_SYSTEM_ID(node->id), NCCL_TOPO_ID_LOCAL_ID(node->id));
  int offset = strlen(line);
#endif

  // 遍历所有节点类型
  for (int t=0; t<NCCL_TOPO_NODE_TYPES; t++) {
    if (node->paths[t] == NULL) continue;

    // 遍历该类型的所有节点
    for (int n = 0; n<system->nodes[t].count; n++) {
#ifdef ENABLE_TRACE
      // 详细模式：展开显示每一步
      line[0] = 0;
      int offset = 0;
      for (int i=0; i<node->paths[t][n].count; i++) {
        struct ncclTopoLink* link = node->paths[t][n].list[i];
        struct ncclTopoNode* remNode = link->remNode;
        snprintf(line+offset, linesize-offset, "--%s(%g)->%s/%lx-%lx",
                 topoLinkTypeStr[link->type], link->bw,
                 topoNodeTypeStr[remNode->type],
                 NCCL_TOPO_ID_SYSTEM_ID(remNode->id),
                 NCCL_TOPO_ID_LOCAL_ID(remNode->id));
        offset = strlen(line);
      }
      INFO(NCCL_GRAPH, "%s (%f)", line, node->paths[t][n].bw);
#else
      // 普通模式：压缩显示
      // 格式：节点类型/ID (跳数/带宽/路径类型)
      snprintf(line+offset, linesize-offset, "%s/%lx-%lx (%d/%.1f/%s) ",
               topoNodeTypeStr[t],
               NCCL_TOPO_ID_SYSTEM_ID(system->nodes[t].nodes[n].id),
               NCCL_TOPO_ID_LOCAL_ID(system->nodes[t].nodes[n].id),
               node->paths[t][n].count,    // 跳数
               node->paths[t][n].bw,      // 带宽
               topoPathTypeStr[node->paths[t][n].type]);  // 路径类型
      offset = strlen(line);
#endif
    }
  }
#ifndef ENABLE_TRACE
  INFO(NCCL_GRAPH, "%s", line);
#endif
}

// ============================================================================
// ncclTopoPrintPaths - 打印系统中所有 GPU 和 NET 的路径
// ============================================================================
// 参数：
//   system - 拓扑系统
//
// 输出：每个 GPU/NIC 的路径信息到日志
ncclResult_t ncclTopoPrintPaths(struct ncclTopoSystem* system) {
  // 打印所有 GPU 的路径
  for (int i=0; i<system->nodes[GPU].count; i++) {
    printNodePaths(system, system->nodes[GPU].nodes+i);
  }
  // 打印所有 NET 的路径
  for (int i=0; i<system->nodes[NET].count; i++) {
    printNodePaths(system, system->nodes[NET].nodes+i);
  }
  return ncclSuccess;
}

// ============================================================================
// ncclGetLocalCpu - 获取距离指定 GPU 最近的 CPU
// ============================================================================
// 参数：
//   system - 拓扑系统
//   gpu    - GPU 索引
//   retCpu - 输出：最近的 CPU 索引
//
// 返回：ncclSuccess 或 ncclInternalError
//
// 用途：CPU 亲和性绑定，将进程绑定到最近的 CPU
ncclResult_t ncclGetLocalCpu(struct ncclTopoSystem* system, int gpu, int* retCpu) {
  // Find the closest CPU to a GPU
  // 找到距离 GPU 最近的 CPU
  int minHops = 0;      // 最小跳数
  int localCpu = -1;    // 最近的 CPU 索引
  struct ncclTopoLinkList* paths = system->nodes[GPU].nodes[gpu].paths[CPU];

  // 遍历所有 CPU，找跳数最少的
  for (int c=0; c<system->nodes[CPU].count; c++) {
    int hops = paths[c].count;
    if (hops > 0 && (minHops == 0 || hops < minHops)) {
      localCpu = c;
      minHops = hops;
    }
  }

  if (localCpu == -1) {
    WARN("Error : could not find CPU close to GPU %d", gpu);
    return ncclInternalError;
  }
  *retCpu = localCpu;
  return ncclSuccess;
}

// ============================================================================
// mergePathType - 合并两个路径类型
// ============================================================================
// 参数：
//   type0, type1 - 两个路径类型
//
// 返回：合并后的路径类型
//
// 规则：
// - 一般取最大值（较差的类型）
// - 特殊情况：PHB + C2C = P2C
static int mergePathType(int type0, int type1){
  int max = std::max(type0,type1);
  int min = std::min(type0,type1);
  // 特殊处理：PHB（PCI Host Bridge） + C2C（Chip-to-Chip） = P2C
  if(max == PATH_PHB && min == PATH_C2C) return PATH_P2C;
  else return max;
}

// ============================================================================
// addInterStep - 添加中间节点到路径
// ============================================================================
// 参数：
//   system - 拓扑系统
//   tx, ix - 中间节点的类型和索引
//   t1, i1 - 源节点的类型和索引
//   t2, i2 - 目标节点的类型和索引
//
// 返回：ncclSuccess
//
// 功能：在 t1→t2 的路径中插入中间节点 tx
// 例如：GPU0 → CPU → GPU1（CPU 是中间节点）
static ncclResult_t addInterStep(struct ncclTopoSystem* system, int tx, int ix, int t1, int i1, int t2, int i2) {
  struct ncclTopoNode* cpuNode = system->nodes[tx].nodes+ix;    // 中间节点（如 CPU）
  struct ncclTopoNode* srcNode = system->nodes[t1].nodes+i1;  // 源节点（如 GPU0）

  int l=0;
  // Node 1 -> CPU
  // 复制源节点到中间节点的路径
  for (int i=0; i<srcNode->paths[tx][ix].count; i++)
    srcNode->paths[t2][i2].list[l++] = srcNode->paths[tx][ix].list[i];

  // CPU -> Node 2
  // 复制中间节点到目标节点的路径
  for (int i=0; i<cpuNode->paths[t2][i2].count; i++)
    srcNode->paths[t2][i2].list[l++] = cpuNode->paths[t2][i2].list[i];

  // Update path characteristics
  // 更新路径特性
  srcNode->paths[t2][i2].count = l;  // 总跳数
  srcNode->paths[t2][i2].type = mergePathType(srcNode->paths[tx][ix].type, cpuNode->paths[t2][i2].type);  // 合并路径类型

  // 如果通过 GPU 转发，标记为 PXN
  if (tx == GPU) srcNode->paths[t2][i2].type = PATH_PXN;

  // 带宽 = 两段路径带宽的最小值
  srcNode->paths[t2][i2].bw = std::min(srcNode->paths[tx][ix].bw, cpuNode->paths[t2][i2].bw);
  return ncclSuccess;
}

// ============================================================================
// ncclTopoRemovePaths - 释放/移除所有路径
// ============================================================================
// 参数：
//   system - 拓扑系统
//
// 功能：释放所有节点分配的路径内存，用于重新计算或清理
static void ncclTopoRemovePaths(struct ncclTopoSystem* system) {
  // 遍历所有节点类型
  for (int t1=0; t1<NCCL_TOPO_NODE_TYPES; t1++) {
    // 遍历该类型的所有节点
    for (int n=0; n<system->nodes[t1].count; n++) {
      struct ncclTopoNode* node = system->nodes[t1].nodes+n;
      // 遍历所有目标类型
      for (int t2=0; t2<NCCL_TOPO_NODE_TYPES; t2++) {
        if (node->paths[t2]) free(node->paths[t2]);  // 释放路径数组
        node->paths[t2] = NULL;
      }
    }
  }
}

// ============================================================================
// 旧版本路径类型到新版本的映射表
// ============================================================================
static const int levelsOldToNew[] = {
  PATH_LOC,   // 旧版本 0 → LOC
  PATH_PIX,   // 旧版本 1 → PIX
  PATH_PXB,   // 旧版本 2 → PXB
  PATH_PHB,   // 旧版本 3 → PHB
  PATH_SYS,   // 旧版本 4 → SYS
  PATH_SYS    // 旧版本 5+ → SYS
};

// ============================================================================
// ncclGetLevel - 从环境变量获取路径级别配置
// ============================================================================
// 参数：
//   level       - 输出：路径级别
//   disableEnv - 禁用环境变量名（如 "NCCL_P2P_DISABLE"）
//   levelEnv   - 级别环境变量名（如 "NCCL_P2P_LEVEL"）
//
// 返回：ncclSuccess
//
// 环境变量格式：
// - NCCL_P2P_DISABLE=1  → 仅允许本地（PATH_LOC）
// - NCCL_P2P_LEVEL=NVL → 允许 NVLink 及更好路径
ncclResult_t ncclGetLevel(int* level, const char* disableEnv, const char* levelEnv) {
  if (*level == -1) {  // 尚未初始化
    int l = -1;

    // 先检查禁用环境变量
    if (disableEnv) {
      const char* str = ncclGetEnv(disableEnv);
      if (str) {
        int disable = strtol(str, NULL, 0);
        if (disable == 1)
          l = PATH_LOC;  // 禁用时只允许本地
        if (l >= 0)
          INFO(NCCL_ALL, "%s set by environment to %d", disableEnv, disable);
      }
    }

    // 再检查级别环境变量
    if (l == -1) {
      const char* str = ncclGetEnv(levelEnv);
      if (str) {
        // 尝试匹配路径类型字符串（如 "NVL", "PIX"）
        for (int i=0; i<=PATH_SYS; i++) {
          if (strcmp(str, topoPathTypeStr[i]) == 0) {
            l = i;
            break;
          }
        }
        // Old style numbering
        // 支持旧版本数字格式（0, 1, 2...）
        // levelsOldToNew 数组将旧级别映射到新级别
        if (l == -1 && str[0] >= '0' && str[0] <= '9') {
          int oldLevel = strtol(str, NULL, 0);
          const int maxOldLevel = sizeof(levelsOldToNew)/sizeof(int) - 1;
          if (oldLevel > maxOldLevel) oldLevel = maxOldLevel;
          l = levelsOldToNew[oldLevel];
        }
        if (l >= 0)
          INFO(NCCL_ALL, "%s set by environment to %s", levelEnv, topoPathTypeStr[l]);
      }
    }
    *level = l >= 0 ? l : -2;  // -2 表示未设置
  }
  return ncclSuccess;
}

// ============================================================================
// 环境变量参数定义
// ============================================================================
NCCL_PARAM(IgnoreDisabledP2p, "IGNORE_DISABLED_P2P", 0);

// 用户配置的 P2P 级别（-1: 未初始化, -2: 未设置）
static int ncclTopoUserP2pLevel = -1;

// ============================================================================
// ncclGetUserP2pLevel - 获取用户配置的 P2P 级别
// ============================================================================
// 参数：
//   level - 输出：P2P 级别
//
// 返回：ncclSuccess
//
// 环境变量：
// - NCCL_P2P_DISABLE=1  → 仅本地
// - NCCL_P2P_LEVEL=SYS  → 允许系统级别（NUMA 互连）
ncclResult_t ncclGetUserP2pLevel(int* level) {
  if (ncclTopoUserP2pLevel == -1)
    NCCLCHECK(ncclGetLevel(&ncclTopoUserP2pLevel, "NCCL_P2P_DISABLE", "NCCL_P2P_LEVEL"));
  if (ncclTopoUserP2pLevel != -2)
    *level = ncclTopoUserP2pLevel;
  return ncclSuccess;
}

// ============================================================================
// ncclTopoCheckP2p - 检查两个 rank 之间是否支持 P2P 通信
// ============================================================================
// 参数：
//   comm             - 通信域
//   system           - 拓扑系统
//   rank1, rank2     - 要检查的两个 rank
//   p2p              - 输出：是否支持 P2P（1: 支持, 0: 不支持）
//   read             - 输出：是否支持 P2P Read（可选）
//   intermediateRank - 输出：中间 GPU rank（如果需要转发）
//
// 返回：ncclSuccess
//
// P2P 支持条件：
// 1. 同一节点或同一 MNNVL clique
// 2. 路径类型 <= p2pLevel
// 3. NVML P2P 状态检查
ncclResult_t ncclTopoCheckP2p(struct ncclComm* comm, struct ncclTopoSystem* system, int rank1, int rank2,
                              int* p2p, int *read, int* intermediateRank) {
  //multi node nvlink
  int mnnvl = 0;
  struct ncclPeerInfo* info1 = NULL;
  struct ncclPeerInfo* info2 = NULL;

  //初始化，默认不支持p2p
  *p2p = 0;
  if (read)
    *read = 0;
  if (intermediateRank)
    *intermediateRank = -1;

  // Rule out different nodes / isolated containers
  // 排除不同节点或隔离的容器
  if (comm) {
    info1 = comm->peerInfo+rank1;
    info2 = comm->peerInfo+rank2;

    //是否在同一个node上
    if (info1->hostHash != info2->hostHash) {
        //不同的节点上，支持多节点nvlink
        //Multi-Node NVLink
        //MNNVL 通过以下方式实现跨节点的高效通信：
        //Fabric 集群：多个节点通过 NVLink 互连，形成一个 Fabric 集群。
        //Clique：在 Fabric 集群中，一组节点可以形成一个 clique，这些节点之间可以直接通信
      if (comm->MNNVL) {
        // 检查是否在同一个 MNNVL clique 中
        NCCLCHECK(ncclTopoCheckMNNVL(comm->topo, info1, info2, &mnnvl));
        if (!mnnvl)
          return ncclSuccess;  // 不在同一 clique，不支持 p2p
      } else {
        //不支持mnvl，意味着不支持跨节点p2p
        return ncclSuccess;
      }
    } else if (info1->shmDev != info2->shmDev) {
    //在不同的容器内
    //共享内存设备不同, 表示在不同的容器中？？？
      return ncclSuccess;
    }
  }

  // Get GPUs from topology
  int g1, g2;
  //获取rank1在拓扑中的索引号
  NCCLCHECK(ncclTopoRankToIndex(system, rank1, &g1, /*showWarn=*/true));
  //获取GPU1节点的拓扑结构
  struct ncclTopoNode* gpu1 = system->nodes[GPU].nodes+g1;

  //获取rank2在拓扑中的索引号
  if (ncclTopoRankToIndex(system, rank2, &g2, /*showWarn=*/false) == ncclInternalError) {
    // GPU not found, we can't use p2p.
    return ncclSuccess;
  }

  int intermediateIndex = -1;
  // Set intermediate GPU rank, if routing through an intermediate GPU.
  // 如果需要通过中间 GPU 路由，设置中间 GPU rank
  //获取gpu1到gpu2的通信路径
  struct ncclTopoLinkList* path = gpu1->paths[GPU]+g2;

  //如果存在中间gpu，记录中间gpu的rank号
  //特别是在多 GPU 环境中，某些 GPU 可能需要通过中间 GPU 来通信
  if (path->count == 2) {
    struct ncclTopoNode* intermediateNode = path->list[0]->remNode;
    if (intermediateNode->type == GPU) {
      intermediateIndex = intermediateNode - system->nodes[GPU].nodes;
      if (intermediateRank)
        *intermediateRank = intermediateNode->gpu.rank;
    }
  }

  // By default don't use P2P across CPU Host Bridges and further apart
  //默认p2p级别：允许跨 PCIe 桥，但不允许跨 CPU Host Bridge
  int p2pLevel = PATH_PXB;

  int arch, vendor, model;
  //获取cpu架构
  NCCLCHECK(ncclTopoCpuType(system, &arch, &vendor, &model));

  // Allow P2P between pairs of GPUs on AMD systems
  // AMD 系统上允许 GPU 对之间的 P2P（包括 NUMA 互连）
  if ((arch == NCCL_TOPO_CPU_ARCH_X86 && vendor == NCCL_TOPO_CPU_VENDOR_AMD) && system->nodes[GPU].count <= 2)
    p2pLevel = PATH_SYS;

  // User override
  //获取用户设置的p2p级别（优先级高于默认值）
  NCCLCHECK(ncclGetUserP2pLevel(&p2pLevel));

  // Compute the PCI distance and compare with the p2pLevel.
  //路径类型 <= p2pLevel 才有可能启用p2p
  // 路径类型数值越小，路径越好
  if (path->type <= p2pLevel)
    *p2p = 1;

  if (*p2p == 1) {
    // NCCL_IGNORE_DISABLED_P2P=2 is used by unit tests that don't want to
    // validate against NVML at all since they are pretending to be on other hw.
    //g1 != g2表示不是同一个gpu
    //并且两个GPU在同一主机
    //ncclParamIgnoreDisabledP2p() != 2），且不是单元测试模式
    if (g1 != g2 && (comm == NULL || (info1->hostHash == comm->peerInfo[comm->rank].hostHash &&
                                      info1->hostHash == info2->hostHash)) && ncclParamIgnoreDisabledP2p() != 2) {
      int indexes[3] = {-1,-1,-1};
      int verticeN = 0;
      NCCLCHECK(ncclNvmlEnsureInitialized());

      // 收集路径上的 GPU 设备号
      indexes[verticeN++] = system->nodes[GPU].nodes[g1].gpu.dev;
      if (intermediateIndex != -1)
        indexes[verticeN++] = system->nodes[GPU].nodes[intermediateIndex].gpu.dev;
      indexes[verticeN++] = system->nodes[GPU].nodes[g2].gpu.dev;

      // 检查每对 GPU 之间的 P2P 状态
      for (int i=1; i < verticeN; i++) {
        nvmlGpuP2PStatus_t status;
        //测试是否支持p2p读写
        status = ncclNvmlDevicePairs[indexes[i-1]][indexes[i-0]].p2pStatusRead;
        bool good = status == NVML_P2P_STATUS_OK;
        status = ncclNvmlDevicePairs[indexes[i-1]][indexes[i-0]].p2pStatusWrite;
        good &= status == NVML_P2P_STATUS_OK;

        //如果不支持p2p
        if (!good) {
          if (!ncclParamIgnoreDisabledP2p()) {
            // NVLink 连接但 P2P 禁用，可能是硬件问题
            if (path->type <= PATH_NVB) {
              WARN("P2P is disabled between NVLINK connected GPUs %d and %d. This should not be the case given their connectivity, and is probably due to a hardware issue. If you still want to proceed, you can set NCCL_IGNORE_DISABLED_P2P=1.", indexes[i-1], indexes[i-0]);
              return ncclUnhandledCudaError;
            } else if (path->type < PATH_SYS) {
              INFO(NCCL_INIT, "P2P is disabled between connected GPUs %d and %d. You can repress this message with NCCL_IGNORE_DISABLED_P2P=1.", indexes[i-1], indexes[i-0]);
            }
          }
          *p2p = 0;  // 标记为不支持 P2P
        }
      }
    }
  }

//如果路径类型是 NVLink（PATH_NVL）
  if (path->type == PATH_NVL) {
    struct ncclTopoNode* gpu2 = system->nodes[GPU].nodes+g2;
    // Enable P2P Read for Ampere/NVLink only
    // 仅在 Ampere 架构 + NVLink 时启用 P2P Read
    if (read && (gpu1->gpu.cudaCompCap == gpu2->gpu.cudaCompCap) && (gpu1->gpu.cudaCompCap == 80))
        *read = 1;
  }

  return ncclSuccess;
}

// ============================================================================
// ncclTopoCheckMNNVL - 检查两个 peer 是否在同一个 MNNVL fabric cluster 和 clique 中
// ============================================================================
// 参数：
//   system - 拓扑系统
//   info1, info2 - 两个 peer 的信息
//   ret     - 输出：是否在同一个 clique（1: 是, 0: 否）
//
// MNNVL (Multi-Node NVLink)：跨节点 NVLink 通信
// - Fabric UUID：标识一个 NVLink Fabric 集群
// - Clique ID：标识 Fabric 中的一个通信组
ncclResult_t ncclTopoCheckMNNVL(struct ncclTopoSystem* system, struct ncclPeerInfo* info1, struct ncclPeerInfo* info2, int* ret) {
  *ret = 0;

  nvmlGpuFabricInfoV_t *fabricInfo1 = &info1->fabricInfo;
  nvmlGpuFabricInfoV_t *fabricInfo2 = &info2->fabricInfo;

  // A zero UUID means we don't have MNNVL fabric info
  // UUID 为 0 表示没有 MNNVL fabric 信息
  unsigned long uuid0 = 0;
  unsigned long uuid1 = 0;
  memcpy(&uuid0, fabricInfo2->clusterUuid, sizeof(uuid0));
  memcpy(&uuid1, fabricInfo2->clusterUuid + sizeof(uuid0), sizeof(uuid1));
  if ((uuid0 | uuid1) == 0)
    return ncclSuccess;

  // 检查是否在同一个 cluster 和 clique 中
  if ((memcmp(fabricInfo1->clusterUuid, fabricInfo2->clusterUuid, NVML_GPU_FABRIC_UUID_LEN) == 0) &&
      (fabricInfo1->cliqueId == fabricInfo2->cliqueId)) {
    TRACE(NCCL_NET, "MNNVL matching peer 0x%lx UUID %lx.%lx cliqueId 0x%x",
         info2->busId, uuid0, uuid1, fabricInfo2->cliqueId);
    *ret = 1;  // 在同一个 clique 中
  }
  return ncclSuccess;
}

// ============================================================================
// GPUDirect RDMA 相关参数和变量
// ============================================================================
NCCL_PARAM(NetGdrRead, "NET_GDR_READ", -2);  // GDR Read 配置
int ncclTopoUserGdrLevel = -1;                 // 用户配置的 GDR 级别
const char* ncclTopoGdrModeStr[ncclTopoGdrModeNum] = { "Disabled", "Default", "PCI" };

// On C2C platforms use GDRDMA on NICs which are connected to the CPUs
// 在 C2C 平台上，对连接到 CPU 的 NIC 使用 GPUDirect RDMA
NCCL_PARAM(NetGdrC2c, "NET_GDR_C2C", 1);

// ============================================================================
// ncclTopoCheckGdr - 检查是否支持 GPUDirect RDMA
// ============================================================================
// 参数：
//   system  - 拓扑系统
//   rank    - GPU rank
//   netId   - 网络 ID
//   read    - 是否检查 GDR Read（发送方向）
//   gdrMode - 输出：GDR 模式
//
// 返回：ncclSuccess
//
// GDR (GPUDirect RDMA)：网卡直接访问 GPU 内存，绕过 CPU
// 支持条件：
// 1. 网卡支持 gdr
// 2. GPU 支持 gdr
// 3. 距离足够近（<= netGdrLevel）
// 4. 某些架构下 Read 有特殊要求
ncclResult_t ncclTopoCheckGdr(struct ncclTopoSystem* system, int rank, int64_t netId, int read, enum ncclTopoGdrMode* gdrMode) {
  *gdrMode = ncclTopoGdrModeDisable;  // 默认禁用

  // Get GPU and NET
  int n, g;
  NCCLCHECK(ncclTopoIdToIndex(system, NET, netId, &n));
  struct ncclTopoNode* net = system->nodes[NET].nodes+n;
  NCCLCHECK(ncclTopoRankToIndex(system, rank, &g, /*showWarn=*/true));
  struct ncclTopoNode* gpu = system->nodes[GPU].nodes+g;

  // Check that both the NIC and GPUs support it
  // 检查网卡和 GPU 是否都支持 GDR
  if (net->net.gdrSupport == 0) return ncclSuccess;
  if (gpu->gpu.gdrSupport == 0) return ncclSuccess;

  // 对于 Read（发送方向）的特殊检查
  if (read) {  // For reads (sends) only enable under certain conditions
    int gdrReadParam = ncclParamNetGdrRead();
    if (gdrReadParam == 0) return ncclSuccess;  // 明确禁用

    // Disable GDR Reads pre-Ampere when we have other PCI flows
    // Ampere 之前的架构，如果有其他 PCI 流则禁用 GDR Read
    if (gdrReadParam < 0 && gpu->gpu.cudaCompCap < 80) {
      int nvlink = 0;
      // Since we don't know whether there are other communicators,
      // it's better to keep things local if we have a single GPU.
      if (system->nodes[GPU].count == 1) nvlink = 1;
      for (int i=0; i<system->nodes[GPU].count; i++) {
        if (i == g) continue;
        if (gpu->paths[GPU][i].type == PATH_NVL) {
          nvlink = 1;
          break;
        }
      }
      if (!nvlink) return ncclSuccess;  // 没有 NVLink，禁用 GDR Read
    }
  }

  // Check if we are close enough that it makes sense to enable GDR
  // 检查距离是否足够近，启用 GDR 才有意义
  int netGdrLevel = PATH_PXB;  // 默认允许 PCIe 桥
  NCCLCHECK(ncclGetLevel(&ncclTopoUserGdrLevel, NULL, "NCCL_NET_GDR_LEVEL"));
  if (ncclTopoUserGdrLevel != -2) netGdrLevel = ncclTopoUserGdrLevel;

  int distance = gpu->paths[NET][n].type;

  if (distance == PATH_PXN) {
    // In case of PXN, use the intermediate GPU distance instead
    // PXN 情况下，使用中间 GPU 的距离
    int proxyRank;
    NCCLCHECK(ncclTopoGetIntermediateRank(system, gpu->gpu.rank, netId, &proxyRank));
    NCCLCHECK(ncclTopoRankToIndex(system, proxyRank, &g, /*showWarn=*/true));
    gpu = system->nodes[GPU].nodes+g;
    distance = gpu->paths[NET][n].type;
  }

  // On C2C platforms we can still use GDRDMA on NICs connected to the CPUs
  // 在 C2C 平台上，连接到 CPU 的 NIC 仍可使用 GDRDMA
  if (ncclParamNetGdrC2c() && distance == PATH_P2C) {
    INFO(NCCL_GRAPH | NCCL_NET, "GPU %d / HCA %lx connected via C2C link", rank, netId);
    distance = PATH_C2C;
  }

  // 距离超过阈值，禁用 GDR
  if (distance > netGdrLevel) {
    INFO(NCCL_GRAPH|NCCL_NET,"GPU Direct RDMA Disabled for GPU %d / HCA %lx (distance %d > %d)", rank, netId, distance, netGdrLevel);
    return ncclSuccess;
  }

  // Force PCIe mapping if path goes through PCI on a C2C system
  // 在 C2C 系统上，如果路径经过 PCIe，强制使用 PCIe 映射
  int c;
  NCCLCHECK(ncclGetLocalCpu(system, g, &c));
  if (gpu->paths[CPU][c].type == PATH_C2C && distance != PATH_C2C)
    *gdrMode = ncclTopoGdrModePci;  // 强制 PCIe 模式
  else
    *gdrMode = ncclTopoGdrModeDefault;  // 默认模式

  INFO(NCCL_GRAPH|NCCL_NET,"GPU Direct RDMA Enabled for GPU %d / HCA %lx (distance %d <= %d), read %d mode %s", rank, netId, distance, netGdrLevel, read, ncclTopoGdrModeStr[*gdrMode]);
  return ncclSuccess;
}

// ============================================================================
// ncclTopoIsGdrAvail - 检查指定 rank 是否有可用的 GDR
// ============================================================================
// 参数：
//   system - 拓扑系统
//   rank   - GPU rank
//   avail  - 输出：是否有可用的 GDR
//
// 返回：ncclSuccess
ncclResult_t ncclTopoIsGdrAvail(struct ncclTopoSystem* system, int rank, bool *avail) {
  int netNum = system->nodes[NET].count;
  enum ncclTopoGdrMode useGdr = ncclTopoGdrModeDisable;
  *avail = false;

  // 遍历所有网卡，检查是否有任一支持 GDR
  for (int n = 0; n < netNum; n++) {
    int64_t netId = system->nodes[NET].nodes[n].id;

    // 先检查 Read（发送）
    NCCLCHECK(ncclTopoCheckGdr(system, rank, netId, 1, &useGdr));
    if (useGdr) {
      *avail = true;
      break;
    }

    // 再检查 Write（接收）
    NCCLCHECK(ncclTopoCheckGdr(system, rank, netId, 0, &useGdr));
    if (useGdr) {
      *avail = true;
      break;
    }
  }
  return ncclSuccess;
}

// ============================================================================
// GPUDirect 相关参数
// ============================================================================
// Set to 0 to disable the flush on Hopper when using GDR
NCCL_PARAM(NetForceFlush, "NET_FORCE_FLUSH", 0);

// ============================================================================
// ncclTopoNeedFlush - 确定是否需要刷新 GDR 接收缓冲区
// ============================================================================
// 参数：
//   comm    - 通信域
//   netId   - 网络 ID
//   netDev  - 网络设备号
//   rank    - GPU rank
//   flush   - 输出：是否需要刷新
//
// 返回：ncclSuccess
//
// Flush 需求：
// - Ampere 及更早架构：需要 flush
// - Hopper：不需要 flush
// - C2C 平台：数据可能走 PCIe，但完成标志走 C2C，需要 flush
ncclResult_t ncclTopoNeedFlush(struct ncclComm* comm, int64_t netId, int netDev, int rank, int* flush) {
  *flush = 1;  // 默认需要 flush

  ncclNetProperties_t props;
  NCCLCHECK(comm->ncclNet->getProperties(netDev, &props));

  // 网卡插件强制要求 flush
  if (props.forceFlush == 1 || ncclParamNetForceFlush())
    return ncclSuccess;

  int g;
  struct ncclTopoSystem* system = comm->topo;
  NCCLCHECK(ncclTopoRankToIndex(system, rank, &g, /*showWarn=*/true));
  struct ncclTopoNode* gpu = system->nodes[GPU].nodes+g;

  // Flush is required on Ampere and earlier
  // Ampere（8.0）及更早架构需要 flush，Hopper（9.0）不需要
  if (gpu->gpu.cudaCompCap >= 90) *flush = 0;

  // On C2C platforms, data could go through a PCI switch while completions and
  // flags would go through C2C. In that case, force a flush.
  // C2C 平台上，数据可能经过 PCIe 交换机，但完成信号通过 C2C
  // 这种情况下需要强制 flush
  int c, n;
  NCCLCHECK(ncclGetLocalCpu(system, g, &c));
  NCCLCHECK(ncclTopoIdToIndex(system, NET, netId, &n));
  if (gpu->paths[NET][n].type <= PATH_PXB && gpu->paths[CPU][c].type == PATH_C2C) {
    *flush = 1;
  }
  return ncclSuccess;
}

// ============================================================================
// 网络相关参数
// ============================================================================
NCCL_PARAM(NetDisableIntra, "NET_DISABLE_INTRA", 0);

// ============================================================================
// ncclTopoCheckNet - 检查是否应该通过网络而不是 P2P/SHM
// ============================================================================
// 参数：
//   system - 拓扑系统
//   rank1, rank2 - 两个 rank
//   net - 输出：是否应该通过网络（1: 是, 0: 否）
//
// 返回：ncclSuccess
//
// 场景：某些情况下，网络可能比本地 P2P/SHM 更快
ncclResult_t ncclTopoCheckNet(struct ncclTopoSystem* system, int rank1, int rank2, int* net) {
//默认关闭网络优先
  if (ncclParamNetDisableIntra() == 1) {
    *net = 0;
    return ncclSuccess;
  }

  *net = 1;  // 默认使用网络

  // First check the current GPU-to-GPU speed.
  // 首先检查当前 GPU 到 GPU 的速度
  int g1, g2;
  if (ncclTopoRankToIndex(system, rank1, &g1, /*showWarn=*/false) != ncclSuccess ||
      ncclTopoRankToIndex(system, rank2, &g2, /*showWarn=*/false) != ncclSuccess) {
    return ncclSuccess;
  }

  struct ncclTopoNode* gpu1 = system->nodes[GPU].nodes+g1;
  struct ncclTopoNode* gpu2 = system->nodes[GPU].nodes+g2;
  float speed = gpu1->paths[GPU][g2].bw;  // GPU-GPU 带宽

  // Now check the speed each GPU can access the network through PXB or better
  // 检查每个 GPU 通过网络访问的带宽（距离 <= PXB）
  float netSpeed1 = 0, netSpeed2 = 0;
  for (int n=0; n<system->nodes[NET].count; n++) {
    struct ncclTopoLinkList* path = gpu1->paths[NET]+n;
    if (path->type <= PATH_PXB && path->bw > netSpeed1)
      netSpeed1 = path->bw;
    path = gpu2->paths[NET]+n;
    if (path->type <= PATH_PXB && path->bw > netSpeed2)
      netSpeed2 = path->bw;
  }

  // 如果网络带宽大于 GPU-GPU 带宽，使用网络
  if (netSpeed1 > speed && netSpeed2 > speed)
    return ncclSuccess;

  *net = 0;  // 不使用网络
  return ncclSuccess;
}

// ============================================================================
// ncclTopoGetIntermediateRank - 获取 PXN 路径中的中间 GPU rank
// ============================================================================
// 参数：
//   system          - 拓扑系统
//   rank            - 源 GPU rank
//   netId           - 网络 ID
//   intermediateRank - 输出：中间 GPU rank
//
// 返回：ncclSuccess 或 ncclInternalError
//
// PXN 路径：GPU0 --(NVLink)--> GPU1 --(PCIe)--> NIC
// 此时中间 GPU rank = GPU1 的 rank
ncclResult_t ncclTopoGetIntermediateRank(struct ncclTopoSystem* system, int rank, int64_t netId, int* intermediateRank) {
  // Get GPU and NET
  int n, g;
  NCCLCHECK(ncclTopoIdToIndex(system, NET, netId, &n));
  NCCLCHECK(ncclTopoRankToIndex(system, rank, &g, /*showWarn=*/true));
  struct ncclTopoNode* gpu = system->nodes[GPU].nodes+g;
  struct ncclTopoLinkList* path = gpu->paths[NET]+n;

  if (path->type == PATH_PXN) {
    // PXN 路径，查找中间 GPU
    struct ncclTopoNode* node;
    int type = NVS;
    // 遍历路径，跳过 NVSwitch
    for (int i=0; i<path->count && type == NVS; i++) {
      node = path->list[i]->remNode;
      type = node->type;
    }
    // 应该找到 GPU 类型的中间节点
    if (type != GPU) {
      WARN("Could not find intermediate GPU between GPU rank %d and NIC %lx", rank, netId);
      return ncclInternalError;
    }
    *intermediateRank = node->gpu.rank;
  } else {
    *intermediateRank = rank;  // 非 PXN 路径，中间 GPU 就是自己
  }
  return ncclSuccess;
}

// ============================================================================
// PXN 相关参数和函数
// ============================================================================
NCCL_PARAM(PxnDisable, "PXN_DISABLE", 0);

// Net v4 plugins don't have non-blocking connect/accept. We can't therefore use
// remote proxies without risking deadlocks
// Net v4 插件没有非阻塞 connect/accept，使用远程代理可能导致死锁
int ncclPxnDisable(struct ncclComm* comm) {
  static int pxnDisable = -1;  // 缓存禁用状态
  if (pxnDisable == -1) {
    if (comm && comm->ncclNetVer == 4) {
      INFO(NCCL_INIT, "PXN Disabled as plugin is v4");
      pxnDisable = 1;
    } else {
      pxnDisable = ncclParamPxnDisable();
    }
  }
  return pxnDisable;
}

// ============================================================================
// ncclTopoGetPxnRanks - 获取 PXN 中间 ranks 列表
// ============================================================================
// 参数：
//   comm             - 通信域
//   intermediateRanks - 输出：中间 ranks 数组
//   nranks           - 输出：中间 ranks 数量
//
// 返回：ncclSuccess
ncclResult_t ncclTopoGetPxnRanks(struct ncclComm* comm, int** intermediateRanks, int* nranks) {
  struct ncclTopoSystem* system = comm->topo;
  *nranks = 0;
  *intermediateRanks = NULL;
  if (system->nodes[NET].count == 0) return ncclSuccess;

  int nr = 0;
  int* ranks = NULL;

  // 遍历所有 rank，找出可以作为 PXN 中介的 ranks
  for (int rank=0; rank<comm->nRanks; rank++) {
    int64_t netId;
    int proxyRank;
    NCCLCHECK(ncclTopoGetNetDev(comm, comm->rank, NULL, 0, rank, &netId, NULL, &proxyRank));
    if (proxyRank == comm->rank) continue;  // 跳过自己

    enum ncclTopoGdrMode useGdr;
    NCCLCHECK(ncclTopoCheckGdr(comm->topo, comm->rank, netId, 1, &useGdr));
    if (useGdr == ncclTopoGdrModeDisable) continue;  // 需要 GDR 支持

    // 检查是否已添加
    int found = 0;
    for (int r=0; r<nr; r++) {
      if (ranks[r] == proxyRank) found = 1;
    }
    if (!found) {
      NCCLCHECK(ncclRealloc(&ranks, nr, nr+1));
      ranks[nr++] = proxyRank;
    }
  }
  *nranks = nr;
  *intermediateRanks = ranks;
  return ncclSuccess;
}

// ============================================================================
// PXN C2C 参数
// ============================================================================
NCCL_PARAM(PxnC2c, "PXN_C2C", 1);

// ============================================================================
// ncclTopoComputePaths - 计算系统中所有路径的主函数
// ============================================================================
// 参数：
//   system - 拓扑系统
//   comm   - 通信域
//
// 返回：ncclSuccess
//
// 流程：
// 1. 移除旧路径
// 2. 为 CPU/GPU/NET/NVS 计算 BFS 路径
// 3. 修正不支持的 P2P 路径（改为通过 CPU 中转）
// 4. 标记不可访问的 GPU
// 5. 处理 C2C + PHB 场景
// 6. 处理 PXN（通过 GPU 转发到 NIC）
ncclResult_t ncclTopoComputePaths(struct ncclTopoSystem* system, struct ncclComm* comm) {
  // Precompute paths between GPUs/NICs.
  // 预计算 GPU/NIC 之间的路径

  // Remove everything in case we're re-computing
  // 移除所有旧路径（防止重新计算时残留）
  ncclTopoRemovePaths(system);

  // Set direct paths to CPUs. We need them in many cases.
  // 为 CPU 计算 BFS 路径（很多场景需要 CPU 路径）
  for (int c=0; c<system->nodes[CPU].count; c++) {
    NCCLCHECK(ncclTopoSetPaths(system->nodes[CPU].nodes+c, system));
  }

  // Set direct paths to GPUs.
  // 为 GPU 计算 BFS 路径
  for (int g=0; g<system->nodes[GPU].count; g++) {
    NCCLCHECK(ncclTopoSetPaths(system->nodes[GPU].nodes+g, system));
  }

  // Set direct paths to NICs.
  // 为 NIC 计算 BFS 路径
  for (int n=0; n<system->nodes[NET].count; n++) {
    NCCLCHECK(ncclTopoSetPaths(system->nodes[NET].nodes+n, system));
  }

  // Set direct paths to NVSwitches.
  // 为 NVSwitch 计算 BFS 路径
  for (int n=0; n<system->nodes[NVS].count; n++) {
    NCCLCHECK(ncclTopoSetPaths(system->nodes[NVS].nodes+n, system));
  }

  // Update path for GPUs when we don't want to / can't use GPU Direct P2P
  // 对于不支持或不想使用 P2P 的 GPU 对，更新路径
  for (int g=0; g<system->nodes[GPU].count; g++) {
    for (int p=0; p<system->nodes[GPU].count; p++) {
      int p2p;
      NCCLCHECK(ncclTopoCheckP2p(comm, system, system->nodes[GPU].nodes[p].gpu.rank,
                                 system->nodes[GPU].nodes[g].gpu.rank, &p2p, NULL, NULL));
      if (p2p == 0) {
        // Divert all traffic through the CPU
        // 将所有流量重定向到 CPU
        int cpu;
        NCCLCHECK(ncclGetLocalCpu(system, g, &cpu));
        NCCLCHECK(addInterStep(system, CPU, cpu, GPU, p, GPU, g));
      }
    }

    if (comm == NULL)
      continue;

    // Remove GPUs we can't (or don't want to) communicate with through P2P or SHM
    // 移除无法或不想通过 P2P/SHM 通信的 GPU
    struct ncclPeerInfo* dstInfo = comm->peerInfo+system->nodes[GPU].nodes[g].gpu.rank;
    for (int p=0; p<system->nodes[GPU].count; p++) {
      if (p == g) continue;  // 跳过自己
      struct ncclPeerInfo* srcInfo = comm->peerInfo+system->nodes[GPU].nodes[p].gpu.rank;

      int p2p;
      NCCLCHECK(ncclTransports[TRANSPORT_P2P]->canConnect(&p2p, comm, NULL, srcInfo, dstInfo));
      if (p2p == 0) {
        int shm;
        NCCLCHECK(ncclTransports[TRANSPORT_SHM]->canConnect(&shm, comm, NULL, srcInfo, dstInfo));
        if (shm == 0) {
          // Mark this peer as inaccessible. We'll trim it later.
          // 标记为不可访问，稍后会被裁剪掉
          system->nodes[GPU].nodes[p].paths[GPU][g].type = PATH_NET;
        }
      }
    }
  }

  // update the GPU -> NIC path in the case of C2C + PHB
  // 更新 C2C + PHB 场景下的 GPU→NIC 路径
  for (int n = 0; n < system->nodes[NET].count; n++) {
    struct ncclTopoNode* netNode = system->nodes[NET].nodes + n;
    for (int g = 0; g < system->nodes[GPU].count; g++) {
      struct ncclTopoNode* gpuNode = system->nodes[GPU].nodes + g;
      int c;
      NCCLCHECK(ncclGetLocalCpu(system, g, &c));
      if (c == -1) continue;

      // 如果 GPU→CPU 是 C2C，CPU→NIC 是 PHB，合并为 P2C
      if (mergePathType(gpuNode->paths[CPU][c].type, netNode->paths[CPU][c].type) == PATH_P2C) {
        gpuNode->paths[NET][n].type = std::min(PATH_P2C, gpuNode->paths[NET][n].type);
        netNode->paths[GPU][g].type = std::min(PATH_P2C, netNode->paths[GPU][g].type);
      }
    }
  }

  // Update paths for NICs (no GPU Direct, PXN, ...)
  // 更新 NIC 相关路径（无 GDR、PXN 等）
  for (int n=0; n<system->nodes[NET].count; n++) {
    struct ncclTopoNode* netNode = system->nodes[NET].nodes+n;

    for (int g=0; g<system->nodes[GPU].count; g++) {
      // Check whether we can access the NIC through another NVLink-connected GPU (PXN)
      // 检查是否可以通过另一个 NVLink 连接的 GPU 访问 NIC（PXN）
      struct ncclTopoNode* gpu = system->nodes[GPU].nodes+g;

      if (ncclPxnDisable(comm) != 1) {
        int localGpuIndex;
        NCCLCHECK(ncclTopoGetLocalGpu(system, netNode->id, &localGpuIndex));

        // 如果本地 GPU 不是当前 GPU
        if (localGpuIndex != g && localGpuIndex != -1) {
          // PXN = PCI + NVLink.
          struct ncclTopoNode* peerNode = system->nodes[GPU].nodes+localGpuIndex;

          // Only use PXN for NIC n if remote GPU p ...
          // 仅在以下情况使用 PXN：
          int pxnType = ncclParamPxnC2c() ? PATH_P2C : PATH_PXB;
          if (/* (1) peerNode 通过 PXN 类型连接到 NIC */
              peerNode->paths[NET][n].type <= pxnType &&
              /* and (2) peerNode 通过 NVLink 连接到当前 GPU */
              peerNode->paths[GPU][g].type <= PATH_NVL &&
              /* and (3) peerNode 与当前 GPU 在同一节点 */
              NCCL_TOPO_ID_SYSTEM_ID(peerNode->id) == NCCL_TOPO_ID_SYSTEM_ID(gpu->id) &&
              /* and (4) peerNode 到 NIC 的带宽更大，或避免通过 CPU */
              (peerNode->paths[NET][n].bw > gpu->paths[NET][n].bw || gpu->paths[NET][n].type > PATH_PXN))
            // We can use that GPU as relay to communicate with that NIC.
            // 可以使用 peerNode 作为中继与 NIC 通信
            // Only enabling it in the GPU->NIC direction for now to favor
            // receiving locally and sending remotely (consistent with net.cc)
            NCCLCHECK(addInterStep(system, GPU, localGpuIndex, GPU, g, NET, n));
        }
      }

      // 如果不支持 GPUDirect RDMA，通过 CPU 中转
      if (gpu->paths[NET][n].type < PATH_PHB) {
        // Update path when we dont want to / can't use GPU Direct RDMA.
        enum ncclTopoGdrMode gdr;
        NCCLCHECK(ncclTopoCheckGdr(system, system->nodes[GPU].nodes[g].gpu.rank, netNode->id, 0, &gdr));
        if (gdr == 0) {
          // We cannot use GPU Direct RDMA, divert all traffic through the CPU local to the GPU
          int localCpu;
          NCCLCHECK(ncclGetLocalCpu(system, g, &localCpu));
          NCCLCHECK(addInterStep(system, CPU, localCpu, NET, n, GPU, g));
          NCCLCHECK(addInterStep(system, CPU, localCpu, GPU, g, NET, n));
        }
      }
    }
  }

  // Pre-compute NET local gpus to accelerate search
  // 预计算每个网卡对应的本地 GPU，加速后续搜索
  for (int n=0; n<system->nodes[NET].count; n++) {
    struct ncclTopoNode* net = system->nodes[NET].nodes+n;
    NCCLCHECK(ncclTopoGetLocalGpu(system, net->id, &net->net.localGpu));
  }
  return ncclSuccess;
}

// ============================================================================
// ncclTopoTrimSystem - 裁剪拓扑系统，移除不可访问的 GPU
// ============================================================================
// 参数：
//   system - 拓扑系统
//   comm   - 通信域
//
// 返回：ncclSuccess 或错误码
//
// 功能：
// - 移除与当前 rank 不同域的 GPU
// - 如果所有 GPU 都在通信域中，移除所有网卡
// - 域定义：通过 P2P/SHM 可达的 GPU 集合
ncclResult_t ncclTopoTrimSystem(struct ncclTopoSystem* system, struct ncclComm* comm) {
  ncclResult_t ret = ncclSuccess;
  int *domains;           // 每个 GPU 所属的域
  int64_t *ids = NULL;     // 每个 GPU 的 ID
  int myDomain = 0;        // 当前 rank 的域
  int ngpus = system->nodes[GPU].count;

  NCCLCHECK(ncclCalloc(&domains, system->nodes[GPU].count));
  NCCLCHECKGOTO(ncclCalloc(&ids, system->nodes[GPU].count), ret, fail);

  // 计算每个 GPU 的域（通过路径可达性）
  for (int g=0; g<system->nodes[GPU].count; g++) {
    struct ncclTopoNode* gpu = system->nodes[GPU].nodes+g;
    domains[g] = g;  // 初始：每个 GPU 是自己的域
    ids[g] = gpu->id;

    // 如果 GPU[g] 可以通过 P2P/SHM 访问 GPU[p]（路径类型 < PATH_NET），则合并域
    for (int p=0; p<g; p++) {
      if (gpu->paths[GPU][p].type < PATH_NET) {
        domains[g] = std::min(domains[g], domains[p]);
      }
    }

    // 记录当前 rank 的域
    if (gpu->gpu.rank == comm->rank)
      myDomain = domains[g];
  }

  // 移除不在当前域的 GPU
  for (int i=0; i<ngpus; i++) {
    if (domains[i] == myDomain) continue;  // 同一域，保留

    // 查找并移除 GPU
    struct ncclTopoNode* gpu = NULL;
    int g;
    for (g=0; g<system->nodes[GPU].count /* This one varies over the loops */; g++) {
      gpu = system->nodes[GPU].nodes+g;
      if (gpu->id == ids[i]) break; else gpu=NULL;
    }
    if (gpu == NULL) {
      WARN("Could not find id %lx", ids[i]);
      ret = ncclInternalError;
      goto fail;
    }
    NCCLCHECKGOTO(ncclTopoRemoveNode(system, GPU, g), ret, fail);
  }

  // 如果所有 GPU 都在通信域中，移除所有网卡
  // 因为不需要网络中转
  if (system->nodes[GPU].count == comm->nRanks) {
    for (int n=system->nodes[NET].count-1; n>=0; n--)
      NCCLCHECKGOTO(ncclTopoRemoveNode(system, NET, n), ret, fail);
  }

exit:
  free(domains);
  if (ids) free(ids);
  return ret;
fail:
  goto exit;
}

// ============================================================================
// ncclTopoFree - 释放拓扑系统及其所有路径
// ============================================================================
// 参数：
//   system - 拓扑系统
//
// 功能：释放所有路径内存和系统本身
void ncclTopoFree(struct ncclTopoSystem* system) {
  ncclTopoRemovePaths(system);  // 释放所有路径
  free(system);                  // 释放系统结构
}

// ============================================================================
// ncclTopoGetNchannels - 获取与某个 peer 通信的通道数量
// ============================================================================
// 参数：
//   comm     - 通信域
//   g        - 本地 GPU 索引
//   peerRank - peer rank
//   nChannels - 输出：通道数量
//
// 返回：ncclSuccess
//
// 通道数量计算：
// - 本地 rank：-1（特殊标记）
// - NVLink：根据带宽计算（2 * (path.bw / 单条NVLink带宽)）
// - 其他：2 个通道
// - 远程 rank：nChannelsPerNetPeer
static ncclResult_t ncclTopoGetNchannels(struct ncclComm* comm, int g /*local gpu index*/, int peerRank, int* nChannels) {
  int peer;
  struct ncclTopoSystem* system = comm->topo;
  struct ncclTopoLinkList* path = NULL;

  if (ncclTopoRankToIndex(system, peerRank, &peer, /*showWarn=*/false) == ncclSuccess) {
    // Same rank
    if (g == peer) {
      *nChannels = -1;  // 自己，特殊标记
      return ncclSuccess;
    }

    // Local rank
    path = system->nodes[GPU].nodes[peer].paths[GPU]+g;
    if (path->type == PATH_NVL) {
      // NVLink：根据带宽计算通道数
      float nvlBw = ncclTopoNVLinkBw(system->nodes[GPU].nodes[g].gpu.cudaCompCap);
      *nChannels = 2*std::max(1, (int)(path->bw / nvlBw));
    } else {
      // 其他路径：2 个通道
      *nChannels = 2;
    }
  } else {
    // Remote rank, use network
    // 远程 rank：使用网络通信
    int nNetChannels = comm->config.nChannelsPerNetPeer;
    if (nNetChannels == NCCL_CONFIG_UNDEF_INT) {
       //start from 2 channels per NIC and reduce with scale
       nNetChannels = 2;

       // check if we need to use more than one NIC, hence more than one channel
       int netCountByBw = 1, nChannelsMax = nNetChannels;
       NCCLCHECK(getLocalNetCountByBw(system, g, &netCountByBw));

       // Avoid overloading channels with 8+ operations as we loose the sync warp, hence a bit of bandwidth.
       // 避免通道过载（8+ 操作时失去 sync warp）
       while (nChannelsMax*comm->nRanks > comm->p2pnChannels*4 && nChannelsMax > 1)
         nChannelsMax /= 2;

       //allow upto channels requires to drive the NICs
       nNetChannels = std::max(netCountByBw, nChannelsMax);
    }
    *nChannels = nNetChannels;
  }
  return ncclSuccess;
}

// ============================================================================
// 通道数量相关参数
// ============================================================================
NCCL_PARAM(MinP2pNChannels, "MIN_P2P_NCHANNELS", 1);         // 最小 P2P 通道数
NCCL_PARAM(MaxP2pNChannels, "MAX_P2P_NCHANNELS", MAXCHANNELS); // 最大 P2P 通道数
extern int64_t ncclParamWorkArgsBytes();

// ============================================================================
// ncclTopoComputeP2pChannels - 计算 P2P 通道数量
// ============================================================================
// 参数：
//   comm - 通信域
//
// 返回：ncclSuccess
//
// 功能：
// - 计算每个 rank 的 p2p 通道总数
// - 计算每个 peer 的 p2p 通道数
// - 初始化新增的通道
ncclResult_t ncclTopoComputeP2pChannels(struct ncclComm* comm) {
  /* here we already honor comm->max/minCTAs for p2pnChannels. */
  // 如果是共享资源，使用 shared 通道数
  if (comm->sharedRes->owner != comm) {
    comm->p2pnChannels = std::min(comm->nChannels, (int)ncclParamMaxP2pNChannels());
    comm->p2pnChannels = std::min(std::max(comm->p2pnChannels, (int)ncclParamMinP2pNChannels()), comm->sharedRes->tpP2pNChannels);
  } else {
    comm->p2pnChannels = std::min(comm->nChannels, (int)ncclParamMaxP2pNChannels());
    comm->p2pnChannels = std::max(comm->p2pnChannels, (int)ncclParamMinP2pNChannels());
  }

  int minChannels = comm->p2pnChannels;

  // We need to loop through all local GPUs to have a global picture
  // 遍历所有本地 GPU，找到最小通道数
  for (int g=0; g<comm->topo->nodes[GPU].count; g++) {
    for (int r=0; r<comm->nRanks; r++) {
      int nChannels;
      NCCLCHECK(ncclTopoGetNchannels(comm, g, r, &nChannels));
      if (nChannels >= 0)
        minChannels = std::min(minChannels, nChannels);
    }
  }

  // Make nChannelsPerPeer and nChannels powers of 2. This is relied on when
  // mapping p2p peers to channels.
  // 将通道数转换为 2 的幂次方（方便映射）
  comm->p2pnChannelsPerPeer = pow2Up(minChannels);
  comm->p2pnChannels = pow2Up(comm->p2pnChannels);

  // 确保不超过设备参数限制
  comm->p2pnChannels = std::min(comm->p2pnChannels, pow2Down(ncclDevMaxChannelsForArgsBytes(ncclParamWorkArgsBytes())));
  comm->p2pnChannelsPerPeer = std::min(comm->p2pnChannelsPerPeer, comm->p2pnChannels);

  // Init channels that weren't used so far
  // 初始化新增的通道
  for (int c=comm->nChannels; c<comm->p2pnChannels; c++)
    NCCLCHECK(initChannel(comm, c));

  return ncclSuccess;
}

// ============================================================================
// ncclTopoGetNvbGpus - 获取通过 NVBridge 连接的 GPUs
// ============================================================================
// 参数：
//   system - 拓扑系统
//   rank   - 当前 rank
//   nranks - 输出：NVB GPU 数量
//   ranks  - 输出：NVB GPU ranks 数组
//
// 返回：ncclSuccess
//
// NVB：通过中间 GPU 转发的 NVLink 连接（1 跳转发）
ncclResult_t ncclTopoGetNvbGpus(struct ncclTopoSystem* system, int rank, int* nranks, int** ranks) {
  int ngpus = system->nodes[GPU].count;
  NCCLCHECK(ncclCalloc(ranks, ngpus));
  int nvbGpus = 0;

  // 查找当前 rank 的所有 NVB 连接
  for (int g=0; g<ngpus; g++) {
    struct ncclTopoNode* gpu = system->nodes[GPU].nodes+g;
    if (gpu->gpu.rank != rank) continue;

    for (int p=0; p<ngpus; p++) {
      if (gpu->paths[GPU][p].type == PATH_NVB) {
        (*ranks)[nvbGpus++] = system->nodes[GPU].nodes[p].gpu.rank;
      }
    }
  }
  *nranks = nvbGpus;
  return ncclSuccess;
}

// ============================================================================
// ncclTopoGetGpuMinPath - 获取 GPU 到指定类型节点的最小（最好）路径类型
// ============================================================================
// 参数：
//   system - 拓扑系统
//   type   - 目标节点类型
//   min    - 输出：最小路径类型
//
// 返回：ncclSuccess
ncclResult_t ncclTopoGetGpuMinPath(struct ncclTopoSystem* system, int type, int* min) {
  int minPath = PATH_SYS;  // 初始化为最差路径

  // 遍历所有 GPU 的路径
  for (int i=0; i<system->nodes[GPU].count; i++) {
    struct ncclTopoLinkList* paths = system->nodes[GPU].nodes[i].paths[type];
    if (paths == NULL)
      continue;
    for (int j=0; j<system->nodes[type].count; j++) {
      if (type == GPU && i == j)
        continue;  // 跳过自己
      minPath = std::min(minPath, paths[j].type);
    }
  }
  *min = minPath;
  return ncclSuccess;
}

// ============================================================================
// ncclTopoGetGpuMaxPath - 获取 GPU 到指定类型节点的最大（最差）路径类型
// ============================================================================
// 参数：
//   system - 拓扑系统
//   type   - 目标节点类型
//   max    - 输出：最大路径类型
//
// 返回：ncclSuccess
ncclResult_t ncclTopoGetGpuMaxPath(struct ncclTopoSystem* system, int type, int* max) {
  int maxPath = PATH_LOC;  // 初始化为最好路径
  for (int i=0; i<system->nodes[GPU].count; i++) {
    struct ncclTopoLinkList* paths = system->nodes[GPU].nodes[i].paths[type];
    if (paths == NULL) continue;
    for (int j=0; j<system->nodes[type].count; j++) {
      if (type == GPU && i == j) continue;  // 跳过自己
      maxPath = std::max(maxPath, paths[j].type);
    }
  }
  *max = maxPath;
  return ncclSuccess;
}

// ============================================================================
// ncclTopoPathAllNVLink - 检查所有 GPU-GPU 路径是否都通过 NVLink
// ============================================================================
// 参数：
//   system   - 拓扑系统
//   allNvLink - 输出：是否全部 NVLink（1: 是, 0: 否）
//
// 返回：ncclSuccess
//
// 逻辑：如果最大路径类型 < PATH_PIX，说明全部通过 NVLink（或更好）
ncclResult_t ncclTopoPathAllNVLink(struct ncclTopoSystem* system, int* allNvLink) {
  int maxPath;
  NCCLCHECK(ncclTopoGetGpuMaxPath(system, GPU, &maxPath));
  *allNvLink = maxPath >= PATH_PIX ? 0 : 1;  // >= PIX 表示有 PCIe
  return ncclSuccess;
}

// ============================================================================
// ncclTopoSplitNvLink - 检查是否存在分裂的 NVLink 域
// ============================================================================
// 参数：
//   system     - 拓扑系统
//   splitNvLink - 输出：是否分裂（1: 是, 0: 否）
//
// 返回：ncclSuccess
//
// 分裂 NVLink 场景：
// - 两个 NVLink 域，不通过 NVLink 互连
// - 通过 QPI/NUMA 互连（带宽较低）
//
// Check whether we are in a split NVLink situation, with two NVLink domains, not
// connected through NVLink (e.g. QPI).
ncclResult_t ncclTopoSplitNvLink(struct ncclTopoSystem* system, int* splitNvLink) {
  ncclResult_t res = ncclSuccess;
  int nvlDomains = 0;       // NVLink 域数量
  int *nvlDomain = NULL;    // 每个 GPU 所属的 NVLink 域
  int *nvlDomainCount = NULL;  // 每个 NVLink 域的 GPU 数量

  // Compute NVLink domains
  // 计算 NVLink 域
  NCCLCHECKGOTO(ncclCalloc(&nvlDomain, system->nodes[GPU].count), res, exit);
  for (int g=0; g<system->nodes[GPU].count; g++) nvlDomain[g] = g;  // 初始：每个 GPU 是自己的域

  // 合并通过 NVLink 连接的 GPU 域
  for (int g=0; g<system->nodes[GPU].count; g++) {
    struct ncclTopoNode* gpu = system->nodes[GPU].nodes+g;
    int domain = nvlDomain[g];
    for (int p=g+1; p<system->nodes[GPU].count; p++) {
      if (gpu->paths[GPU][p].type == PATH_NVL) {
        nvlDomain[p] = domain;  // 合并域
      }
    }
  }

  // Compute number of GPUs per NVLink domain.
  // 计算每个 NVLink 域的 GPU 数量
  NCCLCHECKGOTO(ncclCalloc(&nvlDomainCount, system->nodes[GPU].count), res, exit);
  for (int g=0; g<system->nodes[GPU].count; g++) {
    nvlDomainCount[nvlDomain[g]]++;
  }

  // Count the number of NVLink domains
  // 统计 NVLink 域数量
  for (int g=0; g<system->nodes[GPU].count; g++) {
    if (nvlDomainCount[g] > 1) nvlDomains++;
  }

  *splitNvLink = nvlDomains == 2 ? 1 : 0;  // 恰好 2 个域才算分裂

exit:
  if(nvlDomain) free(nvlDomain);
  if(nvlDomainCount) free(nvlDomainCount);
  return res;
}
