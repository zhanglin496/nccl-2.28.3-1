/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// 引入NCCL通信相关的头文件，包含通信器、设备、图、传输层等核心组件的定义
#include "comm.h"      // 通信器结构体和相关函数定义
#include "device.h"    // 设备相关操作和结构体定义
#include "graph.h"     // 拓扑图相关结构体和函数定义
#include "transport.h" // 传输层接口定义（网络、共享内存等）
#include "trees.h"     // 树形算法相关结构和辅助函数
#include "rings.h"     // 环形算法相关结构和辅助函数
#include "topo.h"      // 拓扑检测和分析相关函数

/******************************************************************/
/********************* Internode connection ***********************/
// 节点间连接：负责建立不同计算节点之间的通信连接关系
/******************************************************************/

/**
 * ncclTopoPreset - 预设拓扑连接关系
 *
 * 该函数负责初始化通信器的拓扑连接信息，包括：
 * 1. 从预生成的拓扑图中提取各种算法（环形、树形、CollNet）的连接关系
 * 2. 初始化每个通信通道的环、树、CollNet等连接参数
 * 3. 复制通道以支持双向通信
 * 4. 收集NVLS（NVLink Switch）的头部节点信息
 *
 * @param comm: 通信器指针，包含全局通信状态和配置信息
 * @param graphs: 拓扑图数组指针，包含不同算法（RING、TREE、COLLNET等）的拓扑连接信息
 *               graphs[NCCL_ALGO_RING] - 环形算法拓扑图
 *               graphs[NCCL_ALGO_TREE] - 树形算法拓扑图
 *               graphs[NCCL_ALGO_COLLNET_CHAIN] - CollNet链式拓扑图
 *               graphs[NCCL_ALGO_NVLS] - NVLS拓扑图
 * @param topoRanks: 拓扑排名信息结构体，用于输出收集到的连接关系数据
 * @return ncclResult_t: 返回操作状态码，成功返回ncclSuccess
 */
ncclResult_t ncclTopoPreset(struct ncclComm* comm, struct ncclTopoGraph** graphs, struct ncclTopoRanks* topoRanks) {
  // 获取当前进程在通信域中的全局排名（0到nRanks-1）
  int rank = comm->rank;
  // 获取本地节点（同一物理服务器）上的GPU数量
  // comm->topo->nodes[GPU].count 存储了检测到的本地GPU个数
  int localRanks = comm->topo->nodes[GPU].count;
  // 获取当前通信器配置的通道数量，通道是NCCL并行通信的基本单元
  int nChannels = comm->nChannels;

  // 从环形算法的拓扑图中获取跨网卡（crossNic）标志
  // crossNic=2 表示环形拓扑跨越了多个网卡（Rail），可用于负载均衡
  topoRanks->crossNicRing = graphs[NCCL_ALGO_RING]->crossNic;
  // 初始化NVLS头部节点数量为0，后续会遍历NVLS拓扑图来统计
  topoRanks->nvlsHeadNum = 0;

  // 遍历所有通信通道，初始化每个通道的连接信息
  for (int c=0; c<nChannels; c++) {
    // 获取第c个通道的指针，comm->channels是通道数组的起始地址
    struct ncclChannel* channel = comm->channels+c;

    // 初始化环形连接的前驱和后继节点为-1（表示未连接）
    // 环形算法中每个rank有一个前驱节点（prev）和后继节点（next）
    channel->ring.prev = channel->ring.next = -1;

    // 初始化树形连接的父节点为-1（表示根节点或未连接）
    channel->tree.up = -1;

    // 初始化CollNet链式连接的父节点为-1
    // CollNet是集合网络（Collective Network）的缩写，用于优化多节点通信
    channel->collnetChain.up = -1;

    // 初始化树形连接的所有子节点为-1
    // NCCL_MAX_TREE_ARITY定义了树的最大分支数（通常是3）
    for (int i=0; i<NCCL_MAX_TREE_ARITY; i++)
        channel->tree.down[i] = -1;

    // 初始化CollNet链式连接的所有子节点为-1
    for (int i=0; i<NCCL_MAX_TREE_ARITY; i++)
        channel->collnetChain.down[i] = -1;

    // 初始化CollNet直接连接模式的输出节点为-1
    // CollNet有两种模式：CHAIN（链式）和DIRECT（直接连接）
    channel->collnetDirect.out = -1;

    // 初始化CollNet直接连接模式的头部节点索引为-1
    // headRank表示当前rank是否是某个集合网络的头部节点
    channel->collnetDirect.headRank = -1;

    // 初始化CollNet直接连接模式的头部节点数量为0
    channel->collnetDirect.nHeads = 0;

    // 初始化CollNet直接连接模式的偏移量为0
    // shift用于在CUDA kernel中计算数据偏移，实现负载均衡
    channel->collnetDirect.shift = 0;

    // 初始化CollNet直接连接模式的所有头部节点为-1
    // NCCL_MAX_DIRECT_ARITY定义了直接连接模式的最大分支数
    for (int i=0; i<NCCL_MAX_DIRECT_ARITY+1; i++)
        channel->collnetDirect.heads[i] = -1;

    // 初始化CollNet直接连接模式的所有上游连接为-1
    for (int i=0; i<NCCL_MAX_DIRECT_ARITY; i++)
        channel->collnetDirect.up[i] = -1;

    // 初始化CollNet直接连接模式的所有下游连接为-1
    for (int i=0; i<NCCL_MAX_DIRECT_ARITY; i++)
        channel->collnetDirect.down[i] = -1;

    // 获取第c个通道在环形拓扑中的本地节点排名数组
    // intra数组存储了同一节点内各GPU在环形拓扑中的顺序
    int* ringIntra = graphs[NCCL_ALGO_RING]->intra+c*localRanks;

    // 获取第c个通道在树形拓扑中的本地节点排名数组
    int* treeIntra = graphs[NCCL_ALGO_TREE]->intra+c*localRanks;

    // 获取第c个通道在CollNet链式拓扑中的本地节点排名数组
    int* collNetIntra = graphs[NCCL_ALGO_COLLNET_CHAIN]->intra+c*localRanks;

    // 遍历本地节点上的所有GPU，查找当前rank在拓扑中的位置
    for (int i=0; i<localRanks; i++) {
      // 如果在环形拓扑数组中找到当前rank
      if (ringIntra[i] == rank) {
        // 设置环形接收节点：数组第一个元素是接收数据的源节点
        topoRanks->ringRecv[c] = ringIntra[0];
        // 设置环形发送节点：数组最后一个元素是发送数据的目标节点
        topoRanks->ringSend[c] = ringIntra[localRanks-1];
        // 设置环形前驱节点：如果是第一个元素则前驱为-1，否则为前一个元素
        topoRanks->ringPrev[c] = (i == 0) ? -1 : ringIntra[i-1];
        // 设置环形后继节点：如果是最后一个元素则后继为-1，否则为后一个元素
        topoRanks->ringNext[c] = (i == localRanks-1) ? -1 : ringIntra[i+1];
      }

      // 如果在树形拓扑数组中找到当前rank
      if (treeIntra[i] == rank) {
        // 父节点索引默认为0（数组第一个位置是父节点或当前rank）
        int parentIndex = 0;
        // 第一个子节点索引：普通树模式为0，分裂树模式为1
        // NCCL_TOPO_PATTERN_TREE: 普通树形，rank0是根，rank1是第一个子节点
        // NCCL_TOPO_PATTERN_SPLIT_TREE: 分裂树，根节点有两个子树
        int child0Index = graphs[NCCL_ALGO_TREE]->pattern == NCCL_TOPO_PATTERN_TREE ? 0 : 1;
        // 第二个子节点索引：分裂树模式为1，普通树模式为0
        int child1Index = graphs[NCCL_ALGO_TREE]->pattern == NCCL_TOPO_PATTERN_SPLIT_TREE ? 1 : 0;

        // 记录当前rank到父节点的连接
        topoRanks->treeToParent[c] = treeIntra[parentIndex];
        // 记录当前rank到第一个子节点的连接
        topoRanks->treeToChild0[c] = treeIntra[child0Index];
        // 记录当前rank到第二个子节点的连接
        topoRanks->treeToChild1[c] = treeIntra[child1Index];

        // 设置树形结构中当前rank的父节点
        // 如果当前rank是数组第一个元素（i==0），则没有父节点（-1）
        // 否则父节点是数组中的前一个元素
        channel->tree.up         = i == 0 ? -1 : treeIntra[i-1];
        // 设置树形结构中当前rank的第一个子节点
        // 如果当前rank是数组最后一个元素，则没有子节点（-1）
        // 否则第一个子节点是数组中的后一个元素
        channel->tree.down[0]    = i == localRanks-1 ? -1 : treeIntra[i+1];
      }

      // 如果在CollNet链式拓扑数组中找到当前rank
      if (collNetIntra[i] == rank) {
        // 设置CollNet链式结构的父节点
        // 如果当前rank是数组第一个元素（i==0），父节点设为nRanks（表示连接到网络）
        // 否则父节点是数组中的前一个元素
        channel->collnetChain.up      = i == 0 ? comm->nRanks : collNetIntra[i-1];
        // 设置CollNet链式结构的子节点
        // 如果当前rank是数组最后一个元素，则没有子节点（-1）
        // 否则子节点是数组中的后一个元素
        channel->collnetChain.down[0] = i == localRanks-1 ? -1 : collNetIntra[i+1];
      }
    }
  }

  // 复制通道树：将前nChannels个通道的结构完整复制到后nChannels个位置
  // 这是因为NCCL使用双通道机制（channel0和channel1），用于不同的数据传输方向或算法
  // Duplicate channels trees
  struct ncclChannel* channel0 = comm->channels;        // 前半部分通道的起始地址
  struct ncclChannel* channel1 = channel0+nChannels;    // 后半部分通道的起始地址
  memcpy(channel1, channel0, nChannels*sizeof(struct ncclChannel)); // 内存拷贝复制通道结构

  // 收集NVLS（NVLink Switch）头部节点信息，并去重
  // NVLS是NVIDIA的NVLink Switch互连技术，用于跨节点的高速GPU通信
  // 每个NVLS通道有一个头部节点（head），负责协调该通道的通信
  // Get nvls heads and the number of heads. Duplicate head is not allowed.
  for (int c = 0; c < graphs[NCCL_ALGO_NVLS]->nChannels; ++c) {
    // 标志位：当前头部节点是否应该添加到列表中（默认为true）
    bool addHead = true;
    // 获取第c个NVLS通道在本地节点内的排名数组
    int* nvlsIntra = graphs[NCCL_ALGO_NVLS]->intra + c * localRanks;

    // 检查当前NVLS通道的头部节点是否已经存在于列表中（去重）
    for (int dup = 0; dup < topoRanks->nvlsHeadNum; dup++) {
      if (topoRanks->nvlsHeads[dup] == nvlsIntra[0]) {
        // 如果已存在，标记为不添加并跳出循环
        addHead = false;
        break;
      }
    }
    // 如果头部节点不存在于列表中，则添加到列表末尾
    if (addHead) {
      topoRanks->nvlsHeads[topoRanks->nvlsHeadNum++] = nvlsIntra[0];
    }
  }
  // 将去重后的NVLS头部节点列表复制到通信器结构中
  memcpy(comm->nvlsHeads, topoRanks->nvlsHeads, sizeof(int) * topoRanks->nvlsHeadNum);

  // 返回成功状态码
  return ncclSuccess;
}

/**
 * connectRings - 连接环形拓扑中的各个节点
 *
 * 该函数根据节点间的收发关系，构建完整的环形连接
 * 对于每个通道，它会遍历所有节点，建立每个rank的前驱和后继关系
 *
 * @param comm: 通信器指针，包含通道数量和节点数量信息
 * @param ringRecv: 接收节点数组，每个通道存储每个节点的接收源
 * @param ringSend: 发送节点数组，每个通道存储每个节点的发送目标
 * @param ringPrev: 前驱节点数组，输出参数，存储每个rank的前驱节点
 * @param ringNext: 后继节点数组，输出参数，存储每个rank的后继节点
 * @return ncclResult_t: 返回操作状态码，成功返回ncclSuccess
 */
static ncclResult_t connectRings(struct ncclComm* comm, int* ringRecv, int* ringSend, int* ringPrev, int* ringNext) {
  // 获取通信器的通道数量
  int nChannels = comm->nChannels;
  // 获取通信器的节点数量（物理服务器数量）
  int nNodes = comm->nNodes;

  // 遍历每个通信通道
  for (int c=0; c<nChannels; c++) {
    // 获取第c个通道的接收节点数组起始位置
    // ringRecv是二维数组的一维展开，大小为nChannels*nNodes
    int* recv = ringRecv+c*comm->nNodes;
    // 获取第c个通道的发送节点数组起始位置
    int* send = ringSend+c*comm->nNodes;
    // 获取第c个通道的前驱节点数组起始位置
    // ringPrev存储每个rank的前驱节点，大小为nChannels*nRanks
    int* prev = ringPrev+c*comm->nRanks;
    // 获取第c个通道的后继节点数组起始位置
    int* next = ringNext+c*comm->nRanks;

    // 遍历所有节点，构建环形连接关系
    for (int n=0; n<nNodes; n++) {
      // 获取当前节点n在环形中的接收rank
      // recv[n]表示节点n负责从哪个rank接收数据
      int recvRank = recv[n];
      // 计算前一个节点的发送rank（环形结构中的前驱）
      // (n-1+nNodes)%nNodes实现环形回绕，当n=0时回到最后一个节点
      int prevSendRank = send[(n-1+nNodes)%nNodes];
      // 设置接收rank的前驱节点为前一个节点的发送rank
      // 这样就形成了环形连接：... -> prevSendRank -> recvRank -> ...
      prev[recvRank] = prevSendRank;

      // 获取当前节点n在环形中的发送rank
      // send[n]表示节点n负责向哪个rank发送数据
      int sendRank = send[n];
      // 计算后一个节点的接收rank（环形结构中的后继）
      // (n+1)%nNodes实现环形回绕，当n=nNodes-1时回到第一个节点
      int nextRecvRank = recv[(n+1)%nNodes];
      // 设置发送rank的后继节点为后一个节点的接收rank
      // 这样就形成了环形连接：... -> sendRank -> nextRecvRank -> ...
      next[sendRank] = nextRecvRank;
    }
  }
  return ncclSuccess;
}

/**
 * getIndexes - 获取节点索引数组
 *
 * 该函数是一个简单的辅助函数，将ranks数组复制到indexes数组
 * 主要用于统一接口，方便后续处理
 *
 * @param ranks: 输入的排名数组
 * @param indexes: 输出的索引数组，将与ranks内容相同
 * @param nNodes: 节点数量
 * @return ncclResult_t: 返回操作状态码，成功返回ncclSuccess
 */
static ncclResult_t getIndexes(int* ranks, int* indexes, int nNodes) {
 // 遍历所有节点，将ranks数组的值复制到indexes数组
 for (int n=0; n<nNodes; n++) indexes[n] = ranks[n];
 return ncclSuccess;
}

/**
 * setTreeUp - 设置树形结构的父节点
 *
 * 该函数用于设置树的向上连接（父节点）
 * 如果u为-1表示没有父节点（即为根节点），直接返回成功
 *
 * @param tree: 树结构指针，要设置父节点的树
 * @param indexes: 索引数组，用于查找实际的rank值
 * @param u: 父节点在indexes数组中的索引，-1表示无父节点
 * @return ncclResult_t: 返回操作状态码，成功返回ncclSuccess
 */
static ncclResult_t setTreeUp(struct ncclTree* tree, int* indexes, int u) {
  // 如果u为-1，表示没有父节点（根节点），直接返回成功
  if (u == -1) return ncclSuccess;
  // 通过索引u从indexes数组中获取实际的父节点rank值
  tree->up = indexes[u];
  return ncclSuccess;
}

/**
 * setTreeDown - 设置树形结构的子节点
 *
 * 该函数用于设置树的向下连接（子节点）
 * NCCL的树最多有NCCL_MAX_TREE_ARITY个子节点（通常为3）
 * 函数会自动找到第一个空闲的子节点位置进行设置
 *
 * @param tree: 树结构指针，要添加子节点的树
 * @param indexes: 索引数组，用于查找实际的rank值
 * @param d: 子节点在indexes数组中的索引，-1表示无子节点
 * @return ncclResult_t: 返回操作状态码，成功返回ncclSuccess，失败返回ncclInternalError
 */
static ncclResult_t setTreeDown(struct ncclTree* tree, int* indexes, int d) {
  // 如果d为-1，表示没有子节点，直接返回成功
  if (d == -1) return ncclSuccess;
  // 查找第一个空闲的子节点位置
  int x = 0;
  // 遍历子节点数组，找到第一个值为负数（空闲）的位置
  // NCCL_MAX_TREE_ARITY定义了树的最大子节点数
  while (x < NCCL_MAX_TREE_ARITY && tree->down[x] >= 0) x++;

  // 如果所有子节点位置都已被占用，触发错误
  if (x == NCCL_MAX_TREE_ARITY) {
    // 输出警告信息，显示树已经满了（所有子节点位置都有值）
    WARN("Internal error : tree already has %d children (%d %d %d)", x, tree->down[0], tree->down[1], tree->down[2]);
    return ncclInternalError;
  }
  // 在找到的空闲位置设置子节点的rank值
  tree->down[x] = indexes[d];
  return ncclSuccess;
}

/**
 * connectTrees - 连接树形拓扑中的各个节点
 *
 * 该函数负责建立树形拓扑的连接关系。NCCL使用双树机制（tree0和tree1），
 * 通过调用ncclGetDtree获取树的父节点和子节点信息，然后为每个通道设置树连接
 *
 * @param comm: 通信器指针，包含通道、节点、rank等信息
 * @param treeToParent: 每个通道每个节点到父节点的映射数组
 * @param treeToChild0: 每个通道每个节点到第一个子节点的映射数组
 * @param treeToChild1: 每个通道每个节点到第二个子节点的映射数组
 * @param treePatterns: 树形模式数组（保留参数，当前未使用）
 * @return ncclResult_t: 返回操作状态码，成功返回ncclSuccess
 */
static ncclResult_t connectTrees(struct ncclComm* comm, int* treeToParent, int* treeToChild0, int* treeToChild1, int* treePatterns) {
  // 使用const声明确保这些变量不会被修改
  // nChannels: 通道数量
  // nNodes: 节点数量（物理服务器数）
  // node: 当前节点ID
  const int nChannels = comm->nChannels, nNodes = comm->nNodes, node = comm->node;

  // 计算树的深度
  // 深度 = (每个节点的平均rank数 - 1) + log2(节点数)
  // 这是一个近似值，表示从叶子节点到根节点的跳数
  // nRanks/nNodes 是每个节点的GPU数量
  // log2i(nNodes) 是节点间树的高度
  // Compute tree depth. Not an exact value but a good approximation in most
  // cases
  int depth = comm->nRanks/nNodes - 1 + log2i(nNodes);

  // 声明双树（tree0和tree1）的父节点和子节点索引变量
  // t0前缀表示tree0（第一个树），t1前缀表示tree1（第二个树）
  // u表示父节点（up），d0/d1表示两个子节点（down），childType表示子节点类型
  int t0u, t0d0, t0d1, t0ChildType, t1u, t1d0, t1d1, t1ChildType;

  // 声明指向树连接数组的指针
  // ttp: 指向tree to parent数组
  // ttc0: 指向tree to child 0数组
  // ttc1: 指向tree to child 1数组
  int* ttp, *ttc0, *ttc1;

  // 获取分布式树（dtree）的连接信息
  // ncclGetDtree会根据节点数和当前节点ID，计算出该节点在双树中的位置
  // 包括父节点索引、两个子节点索引以及子节点类型
  NCCLCHECK(ncclGetDtree(nNodes, node, &t0u, &t0d0, &t0d1, &t0ChildType, &t1u, &t1d0, &t1d1, &t1ChildType));

  // 遍历每个通信通道
  for (int c=0; c<nChannels; c++) {
     // 获取第c个通道的指针（channel0是前半部分通道）
     struct ncclChannel* channel0 = comm->channels+c;
     // 获取第c+nChannels个通道的指针（channel1是后半部分通道，对应channel0的副本）
     struct ncclChannel* channel1 = channel0+nChannels;

     // 设置指向第c个通道树连接数组的指针
     // 这些数组存储了该通道上每个节点在树中的连接关系
     ttp = treeToParent+c*comm->nNodes;  // 第c个通道的父节点数组
     ttc0 = treeToChild0+c*comm->nNodes; // 第c个通道的第一个子节点数组
     ttc1 = treeToChild1+c*comm->nNodes; // 第c个通道的第二个子节点数组

     // 如果当前rank是该节点的父节点rank（即当前rank负责该节点的父连接）
     if (comm->rank == ttp[node]) {
       // 设置channel0的树父节点
       // 根据childType选择使用哪个子节点数组作为父连接的目标
       // t0ChildType == 0 表示使用ttc0，否则使用ttc1
       NCCLCHECK(setTreeUp(&channel0->tree, t0ChildType == 0 ? ttc0 : ttc1, t0u));
       // 同样设置channel1的树父节点
       NCCLCHECK(setTreeUp(&channel1->tree, t1ChildType == 0 ? ttc0 : ttc1, t1u));
     }

     // 如果当前rank是该节点的第一个子节点rank
     if (comm->rank == ttc0[node]) {
       // 设置channel0的第一个子节点连接
       // ttp数组作为索引源，t0d0是要连接的子节点索引
       NCCLCHECK(setTreeDown(&channel0->tree, ttp, t0d0));
       // 设置channel1的第一个子节点连接
       NCCLCHECK(setTreeDown(&channel1->tree, ttp, t1d0));
     }

     // 如果当前rank是该节点的第二个子节点rank
     if (comm->rank == ttc1[node]) {
       // 设置channel0的第二个子节点连接
       NCCLCHECK(setTreeDown(&channel0->tree, ttp, t0d1));
       // 设置channel1的第二个子节点连接
       NCCLCHECK(setTreeDown(&channel1->tree, ttp, t1d1));
     }

     // 如果当前rank参与了树的连接（作为父节点或子节点）
     // 则输出日志信息显示树的连接关系
     if (comm->rank == ttp[node] ||
         comm->rank == ttc0[node] ||
         comm->rank == ttc1[node]) {
       // 输出channel0的树连接关系：父节点 -> 当前rank -> 子节点0/子节点1/子节点2
       INFO(NCCL_GRAPH, "Tree %d : %d -> %d -> %d/%d/%d", c,           channel0->tree.up, comm->rank, channel0->tree.down[0], channel0->tree.down[1], channel0->tree.down[2]);
       // 输出channel1的树连接关系
       INFO(NCCL_GRAPH, "Tree %d : %d -> %d -> %d/%d/%d", c+nChannels, channel1->tree.up, comm->rank, channel1->tree.down[0], channel1->tree.down[1], channel1->tree.down[2]);
     }

     // 设置树的深度（用于kernel中的同步和性能优化）
     channel0->tree.depth = channel1->tree.depth = depth;
  }
  return ncclSuccess;
}

/**
 * connectCollNet - 连接集合网络（CollNet）拓扑
 *
 * CollNet（Collective Network）是NCCL的一种优化算法，用于多节点通信
 * 该函数建立CollNet的直接连接模式，其中每个节点有一个"head"（头部节点）
 * 头部节点负责聚合来自本节点其他rank的数据，并与其他节点的头部通信
 *
 * @param comm: 通信器指针，包含rank、通道数等信息
 * @param collNetGraph: CollNet拓扑图，包含节点间的连接关系
 * @return ncclResult_t: 返回操作状态码，成功返回ncclSuccess
 */
static ncclResult_t connectCollNet(struct ncclComm* comm, struct ncclTopoGraph* collNetGraph) {
  // 获取当前rank的全局排名
  int rank = comm->rank;
  // 获取本地节点上的rank数量
  int localRanks = comm->localRanks;
  // 初始化头部节点数量为0
  int nHeads = 0;
  // 声明头部节点数组，用于存储所有头部rank
  int *heads;

  // 为heads数组分配内存并初始化为0
  NCCLCHECK(ncclCalloc(&heads, localRanks));

  // 查找所有头部rank
  // 在CollNet拓扑中，每个通道的头部节点是intra数组的第一个元素（索引0）
  // Head index is always 0
  for (int c=0; c<collNetGraph->nChannels; c++) {
    // 获取第c个通道在本地节点内的排名数组
    int* collNetIntra = collNetGraph->intra+c*localRanks;
    // 头部节点是该数组的第一个元素
    int head = collNetIntra[0];
    // 检查这个head是否已经在heads数组中（去重）
    for (int h=0; h<nHeads; h++) if (heads[h] == head) head = -1;
    // 如果head没有重复，添加到heads数组
    if (head != -1) heads[nHeads++] = collNetIntra[0];
  }

  // 遍历所有通信通道，设置CollNet连接
  // For all channels
  for (int c=0; c<comm->nChannels; c++) {
    // 获取第c个通道的指针
    struct ncclChannel* channel = comm->channels+c;
    // 声明日志缓冲区，用于输出调试信息
    char line[1024];
    // 初始化日志字符串，包含通道号和当前rank
    sprintf(line, "CollNetDirect channel %d rank %d ", c, rank);
    // 初始化下游连接计数器
    int nDown = 0;

    // 遍历所有头部节点，检查当前rank是否是某个头部节点
    for (int i=0; i<nHeads; i++) {
      if (rank == heads[i]) { // is head
        // 记录当前rank是第i个头部节点
        // 这个索引用于在CUDA kernel中决定数据偏移量
        channel->collnetDirect.headRank = i; // Mark the index for deciding offset in the CUDA kernel
        // 设置CollNet直接连接的输出节点为nRanks（表示连接到网络根）
        // Set root of collnetDirect to id nranks
        channel->collnetDirect.out = comm->nRanks;
        // 获取该头部节点对应的本地节点排名数组
        int* collNetIntra = collNetGraph->intra+i*localRanks;
        sprintf(line+strlen(line), "down ");
        // 遍历本地节点上的所有rank，建立下游连接
        for (int r=0; r<localRanks; r++) {
          // 跳过当前rank自己
          if (collNetIntra[r] == rank) continue;
          // 将其他rank添加到下游连接列表
          // 头部节点需要连接到本节点的所有其他rank
          channel->collnetDirect.down[nDown++] = collNetIntra[r];  // connect to all peers
          sprintf(line+strlen(line), " %d ", collNetIntra[r]);
        }
        sprintf(line+strlen(line), "nDown %d ", nDown);
        // 既然找到了当前rank对应的头部节点，就可以跳出循环
        break;
      }
    }

    // 连接到所有头部节点（上游连接）
    // Connect to all heads
    int nUp = 0;
    sprintf(line+strlen(line), "up ");
    for (int h=0; h<nHeads; h++) {
      // 跳过当前rank自己（如果它是头部节点）
      if (rank == heads[h]) continue;
      // 将其他头部节点添加到上游连接列表
      channel->collnetDirect.up[nUp++] = heads[h];
      sprintf(line+strlen(line), " %d ", heads[h]);
    }

    sprintf(line+strlen(line), "heads ");
    // 构建头部节点列表，按照从当前rank开始的顺序排列
    // 这样做是为了让每个rank的head列表从自己开始，实现负载均衡
    { // heads[] is the list of heads ordered in head order startubg with self
      // 确定起始索引：如果是头部节点则从自己开始，否则从第一个头部节点开始
      int h0 = (channel->collnetDirect.headRank == -1) ? 0 : channel->collnetDirect.headRank;
      for (int h1=0; h1 < nHeads; h1++) {
        // 使用模运算实现循环访问heads数组
        int h = (h0+h1)%nHeads;
        // 将排序后的头部节点存入channel结构
        channel->collnetDirect.heads[h1] = heads[h];
        sprintf(line+strlen(line), " %d ", heads[h]);
      }
    }

    // 记录头部节点总数
    channel->collnetDirect.nHeads = nHeads;

    // 计算偏移量：用于在CUDA kernel中实现负载均衡
    // (rank % localRanks) 获得rank在本地节点内的索引
    // 再对nHeads取模，使得同一节点的不同rank发送到不同的头部节点
    // 这样可以避免所有叶子节点同时向同一个头部节点发送数据
    // nHeads should always be greater than 0.
    // coverity[divide_by_zero]
    channel->collnetDirect.shift = (rank%localRanks)%nHeads; // Shift by intraRank so that leaves don't send to same head simultaneously

    // 设置CollNet的深度
    // 如果既没有上游也没有下游连接，深度为1（纯头部或纯叶子）
    // 否则深度为2（既有上游又有下游，即中间节点）
    channel->collnetDirect.depth = (nUp == 0 && nDown == 0) ? 1 : 2;

    sprintf(line+strlen(line), "nUp %d nHeads %d ", nUp, nHeads);
    sprintf(line+strlen(line), "headRank %d out %d shift %d", channel->collnetDirect.headRank, channel->collnetDirect.out, channel->collnetDirect.shift);
    // 输出CollNet连接的调试信息
    INFO(NCCL_GRAPH, "%s", line);
  }

  // 释放临时分配的heads数组
  free(heads);
  return ncclSuccess;
}

/**
 * connectNvls - 连接NVLS（NVLink Switch）拓扑
 *
 * NVLS是NVIDIA的NVLink Switch互连技术，提供跨节点的高速GPU通信
 * 该函数建立NVLS的树形连接结构，支持NVLS compute和search通道
 *
 * @param comm: 通信器指针，包含NVLS通道数、节点数等信息
 * @param nvlsHeads: NVLS头部节点数组，每个头部节点对应一个NVLS集合
 * @param nHeads: NVLS头部节点数量
 * @return ncclResult_t: 返回操作状态码，成功返回ncclSuccess
 */
static ncclResult_t connectNvls(struct ncclComm* comm, int* nvlsHeads, int nHeads) {
  // 初始化当前rank作为头部节点的索引，-1表示当前rank不是头部节点
  int headRank = -1;

  // 如果没有NVLS头部节点，禁用NVLS通道并直接返回
  if (nHeads == 0) {
    comm->nvlsChannels = 0;
    return ncclSuccess;
  }

  // 遍历所有NVLS头部节点，查找当前rank是否是某个头部节点
  for (int h = 0; h < nHeads; h++) {
    // nvlsHeads是一个二维数组，每行存储一个头部节点在所有节点上的rank映射
    // h * comm->nNodes + comm->node 计算当前节点在第h个头部节点集合中的位置
    if (nvlsHeads[h * comm->nNodes + comm->node] == comm->rank) headRank = h;
  }

  // 遍历所有NVLS通道，初始化NVLS连接信息
  for (int c=0; c<comm->nvlsChannels; c++) {
    // 获取第c个通道的指针
    struct ncclChannel* channel = comm->channels+c;
    // 记录NVLS头部节点的总数
    channel->nvls.nHeads = nHeads;

    // 设置上游连接：NVLS中每个rank连接到所有头部节点
    // comm->nRanks+1+h 是虚拟rank值，用于标识NVLS头部节点
    // +1是为了避免与正常rank冲突（正常rank从0到nRanks-1）
    for (int h=0; h<nHeads; h++) channel->nvls.up[h] = comm->nRanks+1+h;
    // 将剩余的上游连接位置初始化为-1（无效连接）
    for (int h=nHeads; h<NCCL_MAX_NVLS_ARITY; h++) channel->nvls.up[h] = -1;

    // 设置下游连接：指向当前rank对应的头部节点
    // 如果当前rank是头部节点，down指向自己；否则指向负责该rank的头部节点
    channel->nvls.down = comm->nRanks+1+headRank;

    // 设置输出节点：-1表示没有输出（NVLS+SHARP功能尚未实现）
    // NVLS+SHARP not yet implemented.
    channel->nvls.out = -1;

    // 记录当前rank在头部节点列表中的索引
    channel->nvls.headRank = headRank;

    // 初始化NVLS树形连接的上游和下游节点为-1
    channel->nvls.treeUp = channel->nvls.treeDown[0] = channel->nvls.treeDown[1] = channel->nvls.treeDown[2] = -1;

    // 如果启用了CollNet且当前rank是头部节点，设置输出为nRanks
    // 这样可以让头部节点参与CollNet集合操作
    if (comm->config.collnetEnable && channel->nvls.headRank != -1) channel->nvls.out = comm->nRanks;
  }

  // 如果只有一个节点（单机情况），不需要建立跨节点连接，直接返回
  if (comm->nNodes == 1) return ncclSuccess;

  // 建立NVLS树形连接
  // Connect Trees
  // 声明双树（tree0和tree1）的父节点和子节点索引
  int tree0Parent, tree0Child0, tree0Child1, tree1Parent, tree1Child0, tree1Child1;
  // 声明子节点类型变量（被忽略）
  int pc0, pc1; // ignored

  // 获取节点间的分布式树连接信息
  // ncclGetDtree返回当前节点在双树结构中的位置
  NCCLCHECK(ncclGetDtree(comm->nNodes, comm->node,
        &tree0Parent, &tree0Child0, &tree0Child1, &pc0,
        &tree1Parent, &tree1Child0, &tree1Child1, &pc1));

  // 声明头部节点数组和树连接信息
  // treeUp[2]: 双树的上游节点
  // treeDown0[2]: 双树的第一个下游节点
  // treeDown1[2]: 双树的第二个下游节点
  int* heads = NULL;
  int treeUp[2] = { -1, -1 };
  int treeDown0[2] = { -1, -1 };
  int treeDown1[2] = { -1, -1 };

  // 如果当前节点是节点0（主节点），输出NVLS头部节点信息
  if (comm->node == 0) {
    for (int h=0; h<nHeads; h++) {
      // 声明日志缓冲区
      char line[1024];
      sprintf(line, "NVLS Head %2d:", h);
      // 获取第h个头部节点在所有节点上的rank映射
      heads = nvlsHeads+h*comm->nNodes;
      // 输出前20个节点的头部rank信息（避免日志过长）
      for (int n=0; n<comm->nNodes && n<20; n++) {
        sprintf(line+strlen(line), " %2d", heads[n]);
      }
      INFO(NCCL_INIT, "%s", line);
    }
  }

  // 查找当前rank作为头部节点的位置，并保留该头部的树连接信息
  // Find the heads where I'm the head rank and retain tree up/down
  for (int h=0; h<nHeads; h++) {
    // 获取第h个头部节点在所有节点上的rank映射
    heads = nvlsHeads+h*comm->nNodes;
    // 检查当前节点的头部rank是否是当前rank
    if (heads[comm->node] == comm->rank) {
      // 设置tree0的上游节点：如果父节点为-1则设为-1，否则设为父节点的头部rank
      treeUp[0] = tree0Parent == -1 ? -1: heads[tree0Parent];
      // 设置tree0的第一个下游节点
      treeDown0[0] = tree0Child0 == -1 ? -1 : heads[tree0Child0];
      // 设置tree0的第二个下游节点
      treeDown1[0] = tree0Child1 == -1 ? -1 : heads[tree0Child1];

      // 设置tree1的上游节点
      treeUp[1] = tree1Parent == -1 ? -1 : heads[tree1Parent];
      // 设置tree1的第一个下游节点
      treeDown0[1] = tree1Child0 == -1 ? -1 : heads[tree1Child0];
      // 设置tree1的第二个下游节点
      treeDown1[1] = tree1Child1 == -1 ? -1 : heads[tree1Child1];

      // 找到当前rank对应的头部节点后，跳出循环
      break;
    }
  }

  // 在所有NVLS通道中设置树形连接的前驱和后继关系
  // NVLS compute通道与NVLS search通道正交工作
  // Set prev/next in all channels (NVLS compute channels work
  // orthogonally to NVLS search channels).
  for (int c=0; c<comm->nvlsChannels; c++) {
    // 获取第c个通道的指针
    struct ncclChannel* channel = comm->channels+c;
    // 使用c%2在tree0和tree1之间交替，设置上游节点
    channel->nvls.treeUp = treeUp[c%2];

    // 设置第一个下游节点为当前rank的down连接
    channel->nvls.treeDown[0] = channel->nvls.down;

    // 从索引1开始添加其他下游节点
    int ix = 1;
    // 如果treeDown0有有效值，添加为第二个下游节点
    if (treeDown0[c%2] != -1) channel->nvls.treeDown[ix++] = treeDown0[c%2];
    // 如果treeDown1有有效值，添加为第三个下游节点
    if (treeDown1[c%2] != -1) channel->nvls.treeDown[ix] = treeDown1[c%2];
  }

  // 获取前两个NVLS通道的指针，用于输出调试信息
  struct ncclNvls* nvls0 = &comm->channels[0].nvls;
  struct ncclNvls* nvls1 = &comm->channels[1].nvls;

  // 输出NVLS树形连接的调试信息
  // 格式：子节点2/子节点1/子节点0->当前rank->父节点
  INFO(NCCL_GRAPH, "NVLS Trees : %d/%d/%d->%d->%d %d/%d/%d->%d->%d",
      nvls0->treeDown[0], nvls0->treeDown[1], nvls0->treeDown[2], comm->rank, nvls0->treeUp,
      nvls1->treeDown[0], nvls1->treeDown[1], nvls1->treeDown[2], comm->rank, nvls1->treeUp);

  return ncclSuccess;
}

// 旧的环境变量命名（保留以兼容旧版本）
// Legacy naming
NCCL_PARAM(MinNrings, "MIN_NRINGS", -2);
NCCL_PARAM(MaxNrings, "MAX_NRINGS", -2);
// 新的环境变量命名
// New naming
NCCL_PARAM(MinNchannels, "MIN_NCHANNELS", -2);
NCCL_PARAM(MaxNchannels, "MAX_NCHANNELS", -2);

/**
 * ncclMinNchannels - 获取用户配置的最小通道数
 *
 * 该函数从环境变量中读取用户配置的最小通道数
 * 支持两种命名方式：MIN_NRINGS（旧）和MIN_NCHANNELS（新）
 * 新命名优先级更高
 *
 * @return: 返回最小通道数（0到MAXCHANNELS之间）
 */
int ncclMinNchannels() {
  // 初始化最小通道数为0
  int minNchannels = 0;
  // 如果设置了旧的环境变量MIN_NRINGS（值不为-2表示已设置），使用旧值
  if (ncclParamMinNrings() != -2) minNchannels = ncclParamMinNrings();
  // 如果设置了新的环境变量MIN_NCHANNELS，新值覆盖旧值
  if (ncclParamMinNchannels() != -2) minNchannels = ncclParamMinNchannels();

  // 如果请求的最小通道数超过系统最大值，限制为MAXCHANNELS
  if (minNchannels > MAXCHANNELS) {
    INFO(NCCL_GRAPH|NCCL_ENV, "User asked for a minimum of %d channels, limiting to %d", minNchannels, MAXCHANNELS);
    minNchannels = MAXCHANNELS;
  }
  // 确保最小值不为负
  if (minNchannels < 0) minNchannels = 0;
  return minNchannels;
}

// 声明外部函数，获取工作参数字节数
extern int64_t ncclParamWorkArgsBytes();

/**
 * ncclMaxNchannels - 获取用户配置的最大通道数
 *
 * 该函数从环境变量中读取用户配置的最大通道数
 * 最大通道数受限于：1. 系统MAXCHANNELS硬限制 2. 设备参数字节数限制
 *
 * @return: 返回最大通道数（1到MAXCHANNELS之间）
 */
int ncclMaxNchannels() {
  // 初始化为系统最大通道数（硬限制，通常是64）
  int maxNchannels = MAXCHANNELS;
  // 如果设置了旧的环境变量MAX_NRINGS，使用旧值
  if (ncclParamMaxNrings() != -2)
    maxNchannels = ncclParamMaxNrings();
  // 如果设置了新的环境变量MAX_NCHANNELS，新值覆盖旧值
  if (ncclParamMaxNchannels() != -2)
    maxNchannels = ncclParamMaxNchannels();

  // 根据设备工作参数字节数限制最大通道数
  // 通道数越多，kernel参数越多，可能超过参数空间限制
  maxNchannels = std::min(maxNchannels, ncclDevMaxChannelsForArgsBytes(ncclParamWorkArgsBytes()));

  // 硬限制：通道数不能超过MAXCHANNELS（64）
   //硬限制最大64
  if (maxNchannels > MAXCHANNELS)
    maxNchannels = MAXCHANNELS;

  // 确保至少有1个通道
  if (maxNchannels < 1) {
    INFO(NCCL_GRAPH|NCCL_ENV, "User asked for a maximum of %d channels, setting it to 1", maxNchannels);
    maxNchannels = 1;
  }
  return maxNchannels;
}

/**
 * copyChannels - 复制通道配置以扩展通道数量
 *
 * 该函数将[start, end)范围内的通道配置从现有通道复制过来
 * 这是NCCL扩展通道数的常用方法：通过复制现有通道的配置来增加通道数
 *
 * @param comm: 通信器指针
 * @param start: 起始通道索引
 * @param end: 结束通道索引（不包含）
 * @param ringPrev: 环形前驱节点数组
 * @param ringNext: 环形后继节点数组
 * @return: 返回最后一个复制的通道索引
 */
static int copyChannels(struct ncclComm* comm, int start, int end, int* ringPrev, int* ringNext) {
  // 获取总rank数
  int nranks = comm->nRanks;
  int c;
  // 从start到end-1，逐个复制通道配置
  for (c=start; c<end; c++) {
    // 复制环形前驱节点配置：从(c-start)位置的配置复制到c位置
    memcpy(ringPrev+c*nranks, ringPrev+(c-start)*nranks, nranks*sizeof(int));
    // 复制环形后继节点配置
    memcpy(ringNext+c*nranks, ringNext+(c-start)*nranks, nranks*sizeof(int));
    // 复制整个通道结构（包括树、CollNet等配置）
    memcpy(comm->channels+c, comm->channels+c-start, sizeof(struct ncclChannel));
  }
  return c;
}

/**
 * exchangeValues - 交换两个整数的值
 *
 * 简单的值交换函数，用于环形通道的负载均衡
 * 当使用多个网卡（rail）时，交换相邻通道的连接可以避免跨rail冲突
 *
 * @param v0: 第一个整数的指针
 * @param v1: 第二个整数的指针
 */
void exchangeValues(int* v0, int* v1) {
  int tmp = *v1;
  *v1 = *v0;
  *v0 = tmp;
}

// 环境变量：是否在使用unpack网络时自动加倍通道数（默认启用）
// unpack网络是NVIDIA的一种网络传输方式
NCCL_PARAM(UnpackDoubleNChannels, "UNPACK_DOUBLE_NCHANNELS", 1);

/**
 * ncclTopoPostset - 完成拓扑连接的后续设置
 *
 * 该函数是NCCL拓扑设置的最后一个主要步骤，负责：
 * 1. 收集所有rank的拓扑信息
 * 2. 处理跨网卡（crossNic）的通道交换
 * 3. 建立环形和树形连接
 * 4. 复制通道以支持双通道模式
 * 5. 设置CollNet连接
 * 6. 根据配置调整通道数量
 * 7. 连接NVLS（如果启用）
 *
 * @param comm: 当前通信器指针
 * @param firstRanks: 每个节点的第一个rank数组
 * @param treePatterns: 树形模式数组（保留参数）
 * @param allTopoRanks: 所有rank的拓扑信息数组指针
 * @param rings: 环形索引数组（输出参数）
 * @param graphs: 拓扑图数组，包含各种算法的拓扑信息
 * @param parent: 父通信器（用于通信域分割的情况）
 * @return ncclResult_t: 返回操作状态码
 */
ncclResult_t ncclTopoPostset(struct ncclComm* comm, int* firstRanks, int* treePatterns, struct ncclTopoRanks** allTopoRanks, int* rings, struct ncclTopoGraph** graphs, struct ncclComm* parent) {
  // 初始化返回值为成功
  // Gather data from all ranks
  ncclResult_t ret = ncclSuccess;
  // 声明所有临时数组指针，初始化为NULL
  // 这些数组用于收集所有rank的拓扑连接信息
  int *ringRecv = NULL, *ringSend = NULL, *ringPrev = NULL, *ringNext = NULL, *treeToParent = NULL, *treeToChild0 = NULL, *treeToChild1 = NULL, *nvlsHeads = NULL;

  // 获取通信器的基本参数
  int nranks = comm->nRanks;      // 总rank数
  int nNodes = comm->nNodes;      // 节点数（物理服务器数）
  int nChannels = comm->nChannels;// 当前通道数
  int minHeadNum = INT_MAX;       // 最小NVLS头部节点数，初始化为最大值
  // 判断是否与父通信器共享NVLS资源
  int shared = parent && parent->nvlsSupport  && parent->shareResources;

  // 为所有拓扑数组分配内存，大小为节点数/通道数乘以MAXCHANNELS
  // 使用ncclCalloc确保内存初始化为0
  NCCLCHECK(ncclCalloc(&ringRecv, nNodes*MAXCHANNELS));
  NCCLCHECKGOTO(ncclCalloc(&ringSend, nNodes*MAXCHANNELS), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&ringPrev, nranks*MAXCHANNELS), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&ringNext, nranks*MAXCHANNELS), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&treeToParent, nNodes*MAXCHANNELS), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&treeToChild0, nNodes*MAXCHANNELS), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&treeToChild1, nNodes*MAXCHANNELS), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&nvlsHeads, nNodes*MAXCHANNELS), ret, fail);

  // 交换环形通道以避免跨rail冲突
  // 当系统有多个网卡（rail）时，通过交换相邻通道的连接可以分散负载
  // CrossNic值取决于网卡数量和NVLink带宽，不同节点可能不同
  // 因此只在某个rank获得crossNic=2的解时才执行交换
  // Alternate rings to avoid crossing rails.
  // CrossNic values could be not the same on all nodes as it depends on the number of net devs and the NVLink bandwidth.
  // Therefore, it's only done if the rank obtained a solution with crossNic=2.
  for (int r = 0; r < comm->nRanks; r++) {
    // 检查是否需要交换：crossNicRing==2表示跨双rail，通道数为偶数，节点ID为奇数
    if (allTopoRanks[r]->crossNicRing == 2 && (nChannels % 2) == 0 && (comm->rankToNode[r] % 2) == 1) {
      // 交换相邻通道的环形连接
      // c^1实现通道0/1交换，2/3交换，4/5交换等
      // Exchange rings
      for (int c=0; c<nChannels; c+=2) {
        // 交换接收rank
        exchangeValues(allTopoRanks[r]->ringRecv+c, allTopoRanks[r]->ringRecv+(c^1));
        // 交换发送rank
        exchangeValues(allTopoRanks[r]->ringSend+c, allTopoRanks[r]->ringSend+(c^1));
        // 交换前驱rank
        exchangeValues(allTopoRanks[r]->ringPrev+c, allTopoRanks[r]->ringPrev+(c^1));
        // 交换后继rank
        exchangeValues(allTopoRanks[r]->ringNext+c, allTopoRanks[r]->ringNext+(c^1));
      }
    }
  }

  // 收集所有节点的环形和树形连接信息
  // 遍历每个通道
  for (int c=0; c<nChannels;c++) {
    // 遍历每个节点，收集节点级连接信息
    for (int n=0; n<nNodes; n++) {
      // 获取节点n的第一个rank
      int r = firstRanks[n];
      // 从该rank的拓扑信息中提取环形接收rank
      ringRecv[c*nNodes+n] = allTopoRanks[r]->ringRecv[c];
      // 从该rank的拓扑信息中提取环形发送rank
      ringSend[c*nNodes+n] = allTopoRanks[r]->ringSend[c];
      // 从该rank的拓扑信息中提取树形父节点
      treeToParent[c*nNodes+n] = allTopoRanks[r]->treeToParent[c];
      // 从该rank的拓扑信息中提取树形第一个子节点
      treeToChild0[c*nNodes+n] = allTopoRanks[r]->treeToChild0[c];
      // 从该rank的拓扑信息中提取树形第二个子节点
      treeToChild1[c*nNodes+n] = allTopoRanks[r]->treeToChild1[c];
    }
    // 遍历每个rank，收集rank级连接信息
    for (int r=0; r<nranks; r++) {
      // 从每个rank的拓扑信息中提取环形前驱rank
      ringPrev[c*nranks+r] = allTopoRanks[r]->ringPrev[c];
      // 从每个rank的拓扑信息中提取环形后继rank
      ringNext[c*nranks+r] = allTopoRanks[r]->ringNext[c];
    }
  }

  // 找出所有节点中最小的NVLS头部节点数量
  // 这是为了确保所有节点使用相同数量的NVLS头部节点
  for (int n = 0; n < nNodes; n++) {
    // 获取节点n的第一个rank
    int r = firstRanks[n];
    // 更新最小头部节点数量
    if (minHeadNum > allTopoRanks[r]->nvlsHeadNum)
      minHeadNum = allTopoRanks[r]->nvlsHeadNum;
  }

  // 收集所有节点的NVLS头部节点信息
  for (int c = 0; c < minHeadNum; c++) {
    for (int n = 0; n < nNodes; n++) {
      // 获取节点n的第一个rank
      int r = firstRanks[n];
      // 收集第c个NVLS头部节点在节点n上的rank
      nvlsHeads[c * nNodes + n] = allTopoRanks[r]->nvlsHeads[c];
    }
  }

  // 连接环形和树形拓扑，同时也会复制通道
  // connectRings会建立完整的环形连接关系
  // Connect rings and trees. This should also duplicate the channels.
  NCCLCHECKGOTO(connectRings(comm, ringRecv, ringSend, ringPrev, ringNext), ret, fail);
  // connectTrees会建立树形连接关系
  NCCLCHECKGOTO(connectTrees(comm, treeToParent, treeToChild0, treeToChild1, treePatterns), ret, fail);

  // 复制ringPrev/ringNext数组以供ncclBuildRing使用
  // ncclBuildRing需要完整的环形信息来构建CUDA kernel使用的环形索引
  // Duplicate ringPrev/ringNext for ncclBuildRing
  memcpy(ringPrev+nChannels*nranks, ringPrev, nChannels*nranks*sizeof(int));
  memcpy(ringNext+nChannels*nranks, ringNext, nChannels*nranks*sizeof(int));

  // 为当前rank设置环形的前驱和后继节点
  // Set ring prev/next for my rank
  for (int c=0; c<nChannels; c++) {
    // 获取channel0和channel1的指针
    struct ncclChannel* channel0 = comm->channels+c;
    struct ncclChannel* channel1 = channel0+nChannels;
    // 设置两个通道的前驱节点（相同）
    channel0->ring.prev = channel1->ring.prev = ringPrev[c*nranks+comm->rank];
    // 设置两个通道的后继节点（相同）
    channel0->ring.next = channel1->ring.next = ringNext[c*nranks+comm->rank];
  }

  // 通道复制完成，更新通道数量
  // 原始通道被复制了一份，所以总数乘以2
  // 但不能超过MAXCHANNELS的硬限制
  // Duplication should be complete now
  nChannels = comm->nChannels = std::min(MAXCHANNELS,nChannels*2);

  // 设置CollNet（集合网络）连接
  // Setup CollNet
  if (comm->config.collnetEnable) {
    // 获取CollNet链式拓扑图
    struct ncclTopoGraph* collNetChainGraph = graphs[NCCL_ALGO_COLLNET_CHAIN];

    // 如果节点内带宽大于节点间带宽，且不是每节点1个GPU的情况
    // 则增加更多通道以充分利用节点内带宽
    // Add more channels to saturate intra-node bandwidth, except the 1 PPN case
    if (collNetChainGraph->bwIntra > collNetChainGraph->bwInter && comm->nRanks > comm->nNodes) {
      // 计算新的通道数：当前通道数加上一半，但不超过MAXCHANNELS
      int collNetNchannels = std::min(MAXCHANNELS, nChannels+nChannels/2);
      // 复制通道配置以扩展通道数
      nChannels = comm->nChannels = copyChannels(comm, nChannels, collNetNchannels, ringPrev, ringNext);
    }

    // 设置CollNet链式结构的深度
    // 深度等于每个节点的平均rank数
    for (int c = 0; c < comm->nChannels; c++) {
      comm->channels[c].collnetChain.depth = comm->nRanks/comm->nNodes;
    }

    // 如果本地节点rank数不超过CollNet直接连接模式的最大分支数+1
    // 则建立CollNet直接连接
    if (comm->maxLocalRanks <= NCCL_MAX_DIRECT_ARITY+1) {
      NCCLCHECKGOTO(connectCollNet(comm, graphs[NCCL_ALGO_COLLNET_DIRECT]), ret, fail);
    }
  }

  // 在PPN（每节点GPU数）小于8的情况下，使用4个计算通道对应1个搜索通道
  // 这样可以达到峰值带宽
  // minCompCap >= 90 表示计算能力足够高，可以使用更多通道
  // Use 4 compute channels per search channel to reach peak BW on <8 PPN
  if (comm->minCompCap >= 90 && comm->nNodes > 1 && graphs[NCCL_ALGO_RING]->bwIntra > 45.0 && nChannels < 16) {
     // 将通道数翻倍
     nChannels = comm->nChannels = copyChannels(comm, nChannels, 2*nChannels, ringPrev, ringNext);
  }

  // 当使用unpack网络且有多个节点时，自动将通道数翻倍
  // unpack是NVIDIA的一种网络传输模式
  // 自动翻倍只到16通道，用户可以手动指定32通道
  // Double the number of channels when using unpack networking (greater than 1 node)
  // We won't automatically double past 16 channels, users can specify 32 if they want
  if (comm->netDeviceType == NCCL_NET_DEVICE_UNPACK && comm->nNodes > 1 && nChannels < 16 && ncclParamUnpackDoubleNChannels()) {
     // 将通道数翻倍
     nChannels = comm->nChannels = copyChannels(comm, nChannels, 2*nChannels, ringPrev, ringNext);
  }

  // 应用NCCL_MIN_NRINGS/NCCL_MAX_NRINGS环境变量设置
  // 先应用最大值限制，再应用最小值限制（通过复制前面的通道）
  // 这样可以只使用前几个通道，然后复制它们以满足最小通道数要求
  // Honor NCCL_MIN_NRINGS/NCCL_MAX_NRINGS.
  // We permit combining max, then min, to only use the first channels, then duplicate them.
  if (comm->sharedRes->owner != comm) {
    // 如果当前通信器不是共享资源的所有者（子通信器）
    /* child comm #channels cannot exceed top parent #channels. */
    // 子通信器的通道数不能超过父通信器的通道数
    // 取以下值的最小值：用户最大通道数、当前通道数、配置的CTA最大数、父通信器的通道数
    nChannels = comm->nChannels = std::min(std::min(std::min(ncclMaxNchannels(), nChannels), comm->config.maxCTAs), comm->sharedRes->tpNChannels);
    // 然后应用最小通道数要求，通过复制通道来满足
    nChannels = comm->nChannels = copyChannels(comm, nChannels, std::min(std::max(ncclMinNchannels(), comm->config.minCTAs), comm->sharedRes->tpNChannels), ringPrev, ringNext);
  } else {
    // 如果当前通信器是共享资源的所有者（主通信器）
    // 应用最大通道数限制
    nChannels = comm->nChannels = std::min(std::min(ncclMaxNchannels(), nChannels), comm->config.maxCTAs);
    // 然后应用最小通道数要求，通过复制通道来满足
    nChannels = comm->nChannels = copyChannels(comm, nChannels, std::max(ncclMinNchannels(), comm->config.minCTAs), ringPrev, ringNext);
  }

  // 设置集合操作使用的通道数
  comm->collChannels = comm->nChannels;

#if CUDART_VERSION >= 12010
  // CUDA运行时版本 >= 12.1 时支持NVLS资源的聚合使用
  // Support maximal channel usage for aggregation
  if (shared && comm->nvlsChannels > parent->nvlsResources->nChannels) {
    // 如果子通信器的NVLS通道数超过父通信器的NVLS资源限制
    // 则限制子通信器的NVLS通道数为父通信器的资源数
    comm->nvlsChannels = parent->nvlsResources->nChannels;
  }
  // 连接NVLS拓扑，建立NVLS头部节点间的连接关系
  NCCLCHECKGOTO(connectNvls(comm, nvlsHeads, minHeadNum), ret, fail);
#endif

  // 如果与父通信器共享资源，检查通道数是否超过父通信器的限制
  if (shared && comm->nChannels > parent->sharedRes->tpNChannels) {
    // 限制当前通信器的通道数为父通信器的通道数
    nChannels = comm->nChannels = parent->sharedRes->tpNChannels;
    // 集合通道数也不能超过总通道数
    comm->collChannels = std::min(comm->collChannels, comm->nChannels);
  }

  // 创建环形数组并验证所有连接都正确
  // ncclBuildRings会生成CUDA kernel使用的环形索引表
  // Create rings array and check all is fine
  NCCLCHECKGOTO(ncclBuildRings(nChannels, rings, comm->rank, comm->nRanks, ringPrev, ringNext), ret, fail);

// 正常退出标签，释放所有临时分配的内存
exit:
  if (ringRecv) free(ringRecv);
  if (ringSend) free(ringSend);
  if (ringPrev) free(ringPrev);
  if (ringNext) free(ringNext);
  if (treeToParent) free(treeToParent);
  if (treeToChild0) free(treeToChild0);
  if (treeToChild1) free(treeToChild1);
  if (nvlsHeads) free(nvlsHeads);
  return ret;
// 错误处理标签，跳转到exit进行清理
fail:
  goto exit;
}
