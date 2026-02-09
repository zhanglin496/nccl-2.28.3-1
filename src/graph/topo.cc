/*************************************************************************
 * Copyright (c) 2016-2024, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// 包含NCCL核心定义头文件，提供基础数据结构和宏定义
#include "core.h"
// 包含图结构相关头文件，处理拓扑图的构建和操作
#include "graph.h"
// 包含拓扑相关头文件，定义拓扑系统的数据结构
#include "topo.h"
// 包含通信域相关头文件，提供ncclComm等核心结构
#include "comm.h"
// 包含NCCL公共头文件，定义API接口
#include "nccl.h"
// 包含NVML包装层头文件，用于GPU设备管理
#include "nvmlwrap.h"
// 包含集合网络相关头文件，支持collnet功能
#include "coll_net.h"
// 包含传输层相关头文件，管理各种通信传输方式
#include "transport.h"
// 包含系统状态头文件，用于文件操作
#include <sys/stat.h>
// 包含文件控制头文件，用于文件描述符操作
#include <fcntl.h>
// 包含CPU集合头文件，用于CPU亲和性管理
#include "cpuset.h"
// 包含引导启动相关头文件，用于进程间引导和初始化
#include "bootstrap.h"
// 包含互斥锁头文件，用于线程同步
#include <mutex>

// 定义PCI总线ID字符串的大小，格式为"0000:00:00.0"（12个字符+结束符）
#define BUSID_SIZE (sizeof("0000:00:00.0"))
// 定义简化版PCI总线ID字符串的大小，格式为"0000:00"（7个字符+结束符）
#define BUSID_REDUCED_SIZE (sizeof("0000:00"))

// 拓扑节点类型字符串数组，用于调试输出和日志记录
// 对应枚举类型：GPU=0, PCI=1, NVS=2, CPU=3, NIC=4, NET=5
const char* topoNodeTypeStr[] = { "GPU", "PCI", "NVS", "CPU", "NIC", "NET" };
// 拓扑链路类型字符串数组，定义节点间的连接类型
// LOC=本地, NVL=NVLink, C2C=Cache-Coherent Interconnect, PCI=PCI总线, SYS=系统互联, NET=网络
const char* topoLinkTypeStr[] = { "LOC", "NVL", "",    "C2C", "PCI",    "",    "",    "",    "", "SYS", "NET" };
// 拓扑路径类型字符串数组，定义节点间通信路径的类型
// LOC=本地, NVL=NVLink, NVB=NVLink Bridge, C2C=C2C链路, PIX=PCI内部交换, PXB=PCI外部交换
// P2C=PCI到CPU, PXN=PCI到网络, PHB=PCI Host桥, SYS=系统级, NET=网络, DIS=断开
const char* topoPathTypeStr[] = { "LOC", "NVL", "NVB", "C2C", "PIX", "PXB", "P2C", "PXN", "PHB", "SYS", "NET", "DIS" };

/******************************************************************/
/******************* Graph Creation Functions *********************/
/******************************************************************/

// Get an int64 from a PCI path. For example, sys/class/pci0000:00/0000:00:02.0/0000:02:00.0/ will return 0x000002000.
// 从PCI路径中提取64位整数ID
// 例如：sys/class/pci0000:00/0000:00:02.0/0000:02:00.0/ 会返回 0x000002000
ncclResult_t pciPathToInt64(char* path, int offset, int minOffset, int64_t* id) {
  // 将指针移动到指定的偏移位置
  char* str = path+offset;
  // Remove trailing "/" - 移除末尾的斜杠（如果存在）
  if (*str == '/') str--;
  // Find next / - 向前查找下一个斜杠，定位到PCI设备名的起始位置
  while (*str != '/') str--;
  // 移动到斜杠后的第一个字符，即PCI设备ID的起始位置
  str++;
  // 定义临时变量存储转换后的数字ID
  int64_t numid;
  // 将PCI总线ID字符串转换为64位整数
  NCCLCHECK(busIdToInt64(str, &numid));
  // Ignore subdevice because those should use the same PCI link so we want to merge nodes.
  // 忽略子设备号（最低4位），因为同一PCI链路上的子设备应该合并为一个节点
  // 例如：0000:02:00.0 和 0000:02:00.1 会被合并为 0000:02:00.0
  numid -= numid & 0xf;
  // 将结果通过输出参数返回
  *id = numid;
  // 返回成功状态
  return ncclSuccess;
}

// 静态辅助函数：从给定节点开始，向上遍历PCI树查找本地CPU节点
// 参数：
//   node - 起始节点
//   cpu - 输出参数，返回找到的CPU节点指针
//   from - 来源节点，用于避免回溯，防止无限循环
static ncclResult_t findLocalCpu(struct ncclTopoNode* node, struct ncclTopoNode** cpu, struct ncclTopoNode* from) {
  // 初始化输出参数为NULL
  *cpu = NULL;
  // 如果当前节点本身就是CPU类型，直接返回
  if (node->type == CPU) {
    *cpu = node;
    return ncclSuccess;
  }
  // 遍历当前节点的所有链路
  for (int l=0; l<node->nlinks; l++) {
    // Go up the PCI tree to find the CPU. Follow only PCI switches.
    // 检查是否为PCI链路，且不是回退到来源节点，且目标节点是PCI或CPU类型
    if (node->links[l].type == LINK_PCI
	&& node->links[l].remNode != from
	&& (node->links[l].remNode->type == PCI
	    || node->links[l].remNode->type == CPU)) {
      // 递归查找，沿着PCI树向上搜索CPU节点
      NCCLCHECK(findLocalCpu(node->links[l].remNode, cpu, node));
    }
    // 如果已经找到CPU节点，直接返回成功
    if (*cpu != NULL) return ncclSuccess;
  }
  // 未找到CPU节点，也返回成功（cpu保持为NULL）
  return ncclSuccess;
}

// CPU间互联带宽全局变量（当前未使用）
int interCpuBw = 0;
// CPU到PCI带宽全局变量（当前未使用）
int cpuPciBw = 0;

// 静态函数：获取CPU间的互联带宽
// 不同CPU架构和型号有不同的互联带宽（如QPI、UPI、ZPI等）
// 参数：
//   cpu - CPU节点指针
//   bw - 输出参数，返回带宽值（GB/s）
static ncclResult_t ncclTopoGetInterCpuBw(struct ncclTopoNode* cpu, float* bw) {
  // 默认使用本地带宽值（12.0 GB/s）
  *bw = LOC_BW;
  // IBM Power架构CPU的带宽设置
  if (cpu->cpu.arch == NCCL_TOPO_CPU_ARCH_POWER) {
    // Power9 CPU使用特殊的带宽值（32.0 GB/s）
    *bw = P9_BW;
    return ncclSuccess;
  }
  // ARM架构CPU的带宽设置
  if (cpu->cpu.arch == NCCL_TOPO_CPU_ARCH_ARM) {
    // ARM CPU使用ARM特定的带宽值
    *bw = ARM_BW;
    return ncclSuccess;
  }
  // Intel x86架构CPU的带宽设置（根据型号区分）
  if (cpu->cpu.arch == NCCL_TOPO_CPU_ARCH_X86 && cpu->cpu.vendor == NCCL_TOPO_CPU_VENDOR_INTEL) {
    // 根据Intel CPU型号设置不同的QPI/UPI带宽：
    // ERP (Emerald Rapids) - 最新款，带宽最高
    // SRP (Sapphire Rapids) - 次新款
    // SKL (Skylake) - 较旧款
    // BDW (Broadwell) - 最旧款，默认值
    *bw =
      cpu->cpu.model == NCCL_TOPO_CPU_MODEL_INTEL_ERP ? ERP_QPI_BW :
      cpu->cpu.model == NCCL_TOPO_CPU_MODEL_INTEL_SRP ? SRP_QPI_BW :
      cpu->cpu.model == NCCL_TOPO_CPU_MODEL_INTEL_SKL ? SKL_QPI_BW :
      BDW_QPI_BW;
  }
  // AMD x86架构CPU的带宽设置
  if (cpu->cpu.arch == NCCL_TOPO_CPU_ARCH_X86 && cpu->cpu.vendor == NCCL_TOPO_CPU_VENDOR_AMD) {
    // AMD CPU使用AMD特定的带宽值（基于Infinity Fabric或类似技术）
    *bw = AMD_BW;
  }
  // Zhaoxin（兆芯）x86架构CPU的带宽设置
  if (cpu->cpu.arch == NCCL_TOPO_CPU_ARCH_X86 && cpu->cpu.vendor == NCCL_TOPO_CPU_VENDOR_ZHAOXIN) {
    // Zhaoxin CPU根据型号使用不同的ZPI带宽
    *bw = cpu->cpu.model ==  NCCL_TOPO_CPU_MODEL_YONGFENG ? YONGFENG_ZPI_BW : ZPI_BW;
  }
  // 返回成功状态
  return ncclSuccess;
}

// NVLink设备类型枚举，用于标识通过NVLink连接的设备类型
enum ncclNvLinkDeviceType {
  ncclNvLinkDeviceUnknown,  // 未知设备类型
  ncclNvLinkDeviceGpu,      // GPU设备
  ncclNvLinkDeviceSwitch,   // NVLink交换机（如NVSwitch）
  ncclNvLinkDeviceBridge,   // NVLink网桥（IBM/Power平台的NVBridge设备ID 04ea）
};

// 在拓扑系统中查找指定类型和ID的节点
// 参数：
//   system - 拓扑系统指针
//   node - 输出参数，返回找到的节点指针
//   type - 节点类型（GPU/PCI/NVS/CPU/NIC/NET）
//   id - 节点的唯一标识符
ncclResult_t ncclTopoGetNode(struct ncclTopoSystem* system, struct ncclTopoNode** node, int type, uint64_t id) {
  // 遍历指定类型的所有节点
  for (int i=0; i<system->nodes[type].count; i++) {
    // 如果找到匹配的ID，返回该节点的指针
    if (system->nodes[type].nodes[i].id == id) {
      *node = system->nodes[type].nodes+i;
      return ncclSuccess;
    }
  }
  // 未找到节点也返回成功（node保持为NULL）
  return ncclSuccess;
}

// 在拓扑系统中创建一个新的节点
// 参数：
//   system - 拓扑系统指针
//   node - 输出参数，返回创建的节点指针
//   type - 要创建的节点类型
//   id - 新节点的唯一标识符
ncclResult_t ncclTopoCreateNode(struct ncclTopoSystem* system, struct ncclTopoNode** node, int type, uint64_t id) {
  // 检查是否超过该类型节点的最大数量限制
  if (system->nodes[type].count == NCCL_TOPO_MAX_NODES) {
    WARN("Error : tried to create too many nodes of type %d", type);
    return ncclInternalError;
  }
  // 在节点数组末尾获取新节点的位置
  struct ncclTopoNode* n = system->nodes[type].nodes+system->nodes[type].count;
  // 增加该类型节点的计数
  system->nodes[type].count++;
  // 设置节点类型
  n->type = type;
  // 设置节点ID
  n->id = id;
  // 根据节点类型初始化特定的字段
  if (type == GPU) {
    // GPU节点：初始化设备号、rank号和计算能力为未定义值
    n->gpu.dev = NCCL_TOPO_UNDEF;
    n->gpu.rank = NCCL_TOPO_UNDEF;
    n->gpu.cudaCompCap = NCCL_TOPO_UNDEF;
  } else if (type == CPU) {
    // CPU节点：初始化架构、厂商和型号为未定义值
    n->cpu.arch = NCCL_TOPO_UNDEF;
    n->cpu.vendor = NCCL_TOPO_UNDEF;
    n->cpu.model = NCCL_TOPO_UNDEF;
  } else if (type == NET) {
    // 网络节点：初始化ASIC ID、端口号、带宽和延迟
    n->net.asic = 0ULL;
    n->net.port = NCCL_TOPO_UNDEF;
    n->net.bw = 0.0;
    n->net.latency = 0.0;
  }
  // 返回创建的节点指针
  *node = n;
  return ncclSuccess;
}

// 从拓扑系统中删除指定类型的节点
// 参数：
//   system - 拓扑系统指针
//   type - 要删除的节点类型
//   index - 要删除的节点在该类型数组中的索引
ncclResult_t ncclTopoRemoveNode(struct ncclTopoSystem* system, int type, int index) {
  // 获取要删除的节点指针
  struct ncclTopoNode* delNode = system->nodes[type].nodes+index;
  // 遍历所有节点类型，清理与被删除节点相关的引用
  for (int t=0; t<NCCL_TOPO_NODE_TYPES; t++) {
    // 释放该节点的路径信息
    free(delNode->paths[t]);
    // 遍历该类型的所有节点
    for (int n=0; n<system->nodes[t].count; n++) {
      // 获取当前节点的指针
      struct ncclTopoNode* node = system->nodes[t].nodes+n;
      // 跳过被删除的节点本身
      if (node == delNode) continue;
      // 遍历当前节点的所有链路
      for (int l=0; l<node->nlinks; l++) {
        // 删除所有指向被删除节点的链路
        while (l<node->nlinks && node->links[l].remNode == delNode) {
          // 通过memmove移动链路数组，覆盖被删除的链路
          memmove(node->links+l, node->links+l+1, (node->nlinks-l-1)*sizeof(struct ncclTopoLink));
          // 减少链路计数
          node->nlinks--;
        }
        // 如果链路指向同类型的节点，且节点指针在被删除节点之后，需要调整指针
        if (l<node->nlinks && node->links[l].remNode->type == type && node->links[l].remNode >= delNode) {
          // 由于节点被删除，数组前移，指针需要减1
          node->links[l].remNode--;
        }
      }
    }
  }
  // 将被删除节点之后的所有节点向前移动一位，填补空缺
  memmove(delNode, delNode+1, (system->nodes[type].count-index-1)*sizeof(struct ncclTopoNode));
  // 减少该类型的节点计数
  system->nodes[type].count--;
  return ncclSuccess;
}

// 连接两个拓扑节点，创建一条有向链路
// 参数：
//   node - 源节点
//   remNode - 远端（目标）节点
//   type - 链路类型（LINK_PCI/LINK_NVL/LINK_SYS等）
//   bw - 链路带宽（GB/s）
ncclResult_t ncclTopoConnectNodes(struct ncclTopoNode* node, struct ncclTopoNode* remNode, int type, float bw) {
  // Aggregate links into higher bw for NVLink
  // 对于NVLink，如果同一对节点之间有多条链路，需要聚合带宽
  struct ncclTopoLink* link;
  // 查找是否已存在相同类型、连接到相同目标节点的链路
  for (link = node->links; link - node->links != NCCL_TOPO_MAX_LINKS && link->remNode; link++) {
    if (link->remNode == remNode && link->type == type) break;
  }
  // 检查是否超过最大链路数限制
  if (link - node->links == NCCL_TOPO_MAX_LINKS) {
    WARN("Error : too many Topo links (max %d)", NCCL_TOPO_MAX_LINKS);
    return ncclInternalError;
  }
  // 如果是新链路（remNode为NULL），增加链路计数
  if (link->remNode == NULL) node->nlinks++;
  // 设置链路类型
  link->type = type;
  // 设置远端节点
  link->remNode = remNode;
  // 累加带宽（对于NVLink，多条链路的带宽会累加）
  link->bw += bw;

  // Sort links in BW descending order
  // 将链路按带宽降序排序，这样高带宽链路在数组前面，便于后续快速选择最优路径
  struct ncclTopoLink linkSave;
  // 保存当前链路的内容
  memcpy(&linkSave, link, sizeof(struct ncclTopoLink));
  // 向前查找插入位置，保持带宽降序
  while (link != node->links) {
    // 如果前一个链路的带宽大于等于当前链路，找到正确位置
    if ((link-1)->bw >= linkSave.bw) break;
    // 将前一个链路向后移动一位
    memcpy(link, link-1, sizeof(struct ncclTopoLink));
    // 继续向前比较
    link--;
  }
  // 将当前链路放到正确的位置
  memcpy(link, &linkSave, sizeof(struct ncclTopoLink));
  return ncclSuccess;
}

// BCM Gen4 Switches present themselves as a two-level hierarchical switch
// even though they're supposed to sustain full BW across all ports.
// Flatten the switch as this extra level can break the search and make
// NCCL take wrong topology decisions.
// BCM Gen4交换机表现为两层分层结构，但实际上所有端口应该能维持全带宽
// 需要扁平化交换机，因为额外的层级可能破坏搜索算法，导致NCCL做出错误的拓扑决策
// 判断BCM交换机代数的辅助函数
// 参数：
//   id - PCI设备ID
//   level - 层级（0=父交换机，1=子交换机）
// 返回值：4表示Gen4，5表示Gen5，0表示非BCM交换机
int getBcmGen(uint64_t id, int level) {
  // 检查是否为BCM Gen4交换机（特定设备ID掩码匹配）
  if ((id & 0xfffffffffffff000) == 0x1000c0101000a000) return 4;
  // 检查是否为BCM Gen5交换机（设备ID随层级变化）
  if ((id & 0xfffffffffffff000) == (0x1000c03010000000 | level*0x1000)) return 5;
  // 不是BCM交换机
  return 0;
}

// 扁平化BCM交换机结构，将两层结构合并为单层
// 参数：
//   system - 拓扑系统指针
ncclResult_t ncclTopoFlattenBcmSwitches(struct ncclTopoSystem* system) {
  // 初始化返回值为成功
  ncclResult_t ret = ncclSuccess;
  // 遍历所有PCI节点
  for (int s=0; s<system->nodes[PCI].count; s++) {
    // 获取当前PCI交换机节点
    struct ncclTopoNode* pciSwitch = system->nodes[PCI].nodes+s;
    // 检查是否为BCM交换机
    int gen = getBcmGen(pciSwitch->pci.device, 0);
    // Flatten Gen4 PEX switches in base mode
    // 如果是Gen4或Gen5交换机，需要扁平化处理
    if (gen) {
      // Find sub switches with the same device ID.
      // 分配数组存储子交换机的ID
      int64_t* subSwIds;
      NCCLCHECK(ncclCalloc(&subSwIds, pciSwitch->nlinks));
      // 计数器：记录找到的子交换机数量
      int subs = 0;
      // 遍历父交换机的所有链路，查找子交换机
      for (int l=0; l<pciSwitch->nlinks; l++) {
        // 获取链路连接的远端节点
        struct ncclTopoNode* sub = pciSwitch->links[l].remNode;
        // Only fuse sub switches with the same device ID.
        // 只合并相同代数的子交换机
        if (sub->type != PCI || getBcmGen(sub->pci.device, 1) != gen) continue;
        // Save sub switch for later
        // 保存子交换机ID，供后续处理
        subSwIds[subs++] = sub->id;
        // Remove link to that sub switch
        // 从父交换机中删除指向子交换机的链路
        memmove(pciSwitch->links+l, pciSwitch->links+l+1, (pciSwitch->nlinks-l-1)*(sizeof(struct ncclTopoLink)));
        // 减少父交换机的链路计数
        pciSwitch->nlinks--;
        // Don't increase l for the next iteration as we just shifted all links by one.
        // 不增加l，因为链路数组已经前移了一位
        l--;
      }

      // 处理所有找到的子交换机
      for (int s=0; s<subs; s++) {
        // Find sub switch (system->nodes[PCI].nodes is changing every time we remove a node)
        // 通过ID查找子交换机在数组中的索引（数组在删除操作后会变化）
        int index;
        NCCLCHECKGOTO(ncclTopoIdToIndex(system, PCI, subSwIds[s], &index), ret, fail);
        // 获取子交换机节点指针
        struct ncclTopoNode* sub = system->nodes[PCI].nodes+index;
        // Connect all sub PCI devices to the parent switch
        // 将子交换机连接的所有PCI设备重新连接到父交换机
        for (int l=0; l<sub->nlinks; l++) {
          // 获取子交换机连接的远端节点
          struct ncclTopoNode* remNode = sub->links[l].remNode;
          // 跳过指回父交换机的链路
          if (remNode == pciSwitch) continue;
          // Add link from parent PCI switch -> PCI device
          // 检查父交换机的链路数是否已满
          if (pciSwitch->nlinks == NCCL_TOPO_MAX_LINKS) {
            WARN("Error : too many Topo links (max %d)", NCCL_TOPO_MAX_LINKS);
            ret = ncclInternalError;
            goto fail;
          }
          // 将子交换机的链路复制到父交换机
          memcpy(pciSwitch->links+pciSwitch->nlinks, sub->links+l, sizeof(struct ncclTopoLink));
          // 增加父交换机的链路计数
          pciSwitch->nlinks++;
          // Update link from PCI device -> parent PCI switch
          // 更新远端节点的反向链路，使其指向父交换机而不是子交换机
          for (int rl=0; rl<remNode->nlinks; rl++) {
            if (remNode->links[rl].remNode == sub) {
              remNode->links[rl].remNode = pciSwitch;
              break;
            }
          }
        }
        // 删除子交换机节点
        NCCLCHECKGOTO(ncclTopoRemoveNode(system, PCI, index), ret, fail);
      }
      // Set subdevice to 0xffff to make sure we don't merge this switch again.
      // 设置子设备字段为0xffff，确保不会再次合并这个交换机
      pciSwitch->pci.device |= 0xffff;
      // 释放子交换机ID数组
      free(subSwIds);
      // Restart, as system->nodes[PCI].nodes has changed.
      // 由于删除节点后数组结构改变，需要重新开始遍历
      s = -1;  // Will be incremented to 0 in the next loop iteration
      continue;
fail:
      // 错误处理：释放内存并返回
      free(subSwIds);
      return ret;
    }
  }
  // 返回处理结果
  return ret;
}

// 连接拓扑系统中的所有CPU节点，创建CPU间的互联链路
// 参数：
//   system - 拓扑系统指针
ncclResult_t ncclTopoConnectCpus(struct ncclTopoSystem* system) {
  // And connect all CPU nodes together
  // 双重循环遍历所有CPU节点对
  for (int n=0; n<system->nodes[CPU].count; n++) {
    // 获取第一个CPU节点
    struct ncclTopoNode* cpu1 = system->nodes[CPU].nodes+n;
    // 遍历所有CPU节点
    for (int p=0; p<system->nodes[CPU].count; p++) {
      // 获取第二个CPU节点
      struct ncclTopoNode* cpu2 = system->nodes[CPU].nodes+p;
      // 跳过同一个CPU节点，以及不同系统（节点）的CPU
      if (n == p || (NCCL_TOPO_ID_SYSTEM_ID(cpu1->id) != NCCL_TOPO_ID_SYSTEM_ID(cpu2->id))) continue;
      // 获取CPU间互联带宽
      float bw;
      NCCLCHECK(ncclTopoGetInterCpuBw(cpu1, &bw));
      // 创建系统级链路（LINK_SYS）连接两个CPU
      NCCLCHECK(ncclTopoConnectNodes(cpu1, cpu2, LINK_SYS, bw));
    }
  }
  return ncclSuccess;
}

// 递归打印拓扑结构的辅助函数
// 参数：
//   node - 当前要打印的节点
//   prevNode - 前一个节点（用于避免回溯打印）
//   line - 打印缓冲区
//   offset - 缓冲区中的偏移量（用于缩进）
static ncclResult_t ncclTopoPrintRec(struct ncclTopoNode* node, struct ncclTopoNode* prevNode, char* line, int offset) {
  // 根据节点类型格式化节点信息
  if (node->type == GPU) {
    // GPU节点：打印类型、系统ID、本地ID和rank
    sprintf(line+offset, "%s/%lx-%lx (%d)", topoNodeTypeStr[node->type], NCCL_TOPO_ID_SYSTEM_ID(node->id), NCCL_TOPO_ID_LOCAL_ID(node->id), node->gpu.rank);
  } else if (node->type == CPU) {
    // CPU节点：打印类型、系统ID、本地ID、架构、厂商和型号
    sprintf(line+offset, "%s/%lx-%lx (%d/%d/%d)", topoNodeTypeStr[node->type], NCCL_TOPO_ID_SYSTEM_ID(node->id), NCCL_TOPO_ID_LOCAL_ID(node->id), node->cpu.arch, node->cpu.vendor, node->cpu.model);
  } else if (node->type == PCI) {
    // PCI节点：打印类型、系统ID、本地ID和设备ID
    sprintf(line+offset, "%s/%lx-%lx (%lx)", topoNodeTypeStr[node->type], NCCL_TOPO_ID_SYSTEM_ID(node->id), NCCL_TOPO_ID_LOCAL_ID(node->id), node->pci.device);
  } else {
    // 其他节点类型：只打印类型、系统ID和本地ID
    sprintf(line+offset, "%s/%lx-%lx", topoNodeTypeStr[node->type], NCCL_TOPO_ID_SYSTEM_ID(node->id), NCCL_TOPO_ID_LOCAL_ID(node->id));
  }
  // 输出当前节点的信息
  INFO(NCCL_GRAPH, "%s", line);
  // 清空缩进区域（用空格填充）
  for (int i=0; i<offset; i++) line[i] = ' ';

  // 遍历当前节点的所有链路
  for (int l=0; l<node->nlinks; l++) {
    // 获取链路指针
    struct ncclTopoLink* link = node->links+l;
    // 处理本地链路（LINK_LOC）
    if (link->type == LINK_LOC) {
      sprintf(line+offset, "+ %s[%2.1f] - %s/%lx-%lx", topoLinkTypeStr[link->type], link->bw, topoNodeTypeStr[link->remNode->type], NCCL_TOPO_ID_SYSTEM_ID(link->remNode->id), NCCL_TOPO_ID_LOCAL_ID(link->remNode->id));
      INFO(NCCL_GRAPH, "%s", line);
    // 处理非PCI链路，或者不是回退到前一个节点的PCI链路
    } else if (link->type != LINK_PCI || link->remNode != prevNode) {
      // 打印链路类型和带宽
      sprintf(line+offset, "+ %s[%2.1f] - ", topoLinkTypeStr[link->type], link->bw);
      // 计算下一个输出的位置
      int nextOffset = strlen(line);
      // 如果是PCI链路，递归打印子节点
      if (link->type == LINK_PCI) {
        NCCLCHECK(ncclTopoPrintRec(link->remNode, node, line, nextOffset));
      } else {
        // 对于其他类型的链路，直接打印远端节点信息
        if (link->remNode->type == NET) {
          // 网络节点：打印额外的网络属性
          sprintf(line+nextOffset, "%s/%lx-%lx (%d/%lx/%d/%f)", topoNodeTypeStr[link->remNode->type], NCCL_TOPO_ID_SYSTEM_ID(link->remNode->id), NCCL_TOPO_ID_LOCAL_ID(link->remNode->id), link->remNode->net.collSupport, link->remNode->net.asic, link->remNode->net.port, link->remNode->net.bw);
        } else {
          // 其他节点类型：只打印基本信息
          sprintf(line+nextOffset, "%s/%lx-%lx", topoNodeTypeStr[link->remNode->type], NCCL_TOPO_ID_SYSTEM_ID(link->remNode->id), NCCL_TOPO_ID_LOCAL_ID(link->remNode->id));
        }
        // 输出链路和远端节点信息
        INFO(NCCL_GRAPH, "%s", line);
      }
    }
  }
  return ncclSuccess;
}

// 打印整个拓扑系统的结构
// 参数：
//   s - 拓扑系统指针
ncclResult_t ncclTopoPrint(struct ncclTopoSystem* s) {
  // 打印系统标题，包含最大带宽和总带宽
  INFO(NCCL_GRAPH, "=== System : maxBw %2.1f totalBw %2.1f ===", s->maxBw, s->totalBw);
  // 准备打印缓冲区
  char line[1024];
  // 从每个CPU节点开始递归打印拓扑结构
  for (int n=0; n<s->nodes[CPU].count; n++) NCCLCHECK(ncclTopoPrintRec(s->nodes[CPU].nodes+n, NULL, line, 0));
  // 打印系统结束分隔线
  INFO(NCCL_GRAPH, "==========================================");
  // 打印所有路径信息
  NCCLCHECK(ncclTopoPrintPaths(s));
  return ncclSuccess;
}

// 递归排序拓扑节点的链路，优化遍历顺序
// 参数：
//   node - 当前要排序的节点
//   upNode - 上层节点（用于识别PCI上行链路）
static ncclResult_t ncclTopoSort(struct ncclTopoNode* node, struct ncclTopoNode* upNode) {
  // Shift all links to have upLink as last link
  // 将指向上层节点的链路移到最后
  if (upNode) {
    // 查找指向上层节点的链路索引
    int l=0;
    while (node->links[l].remNode != upNode) l++;
    // 保存上层链路
    struct ncclTopoLink upLink;
    memcpy(&upLink, node->links+l, sizeof(struct ncclTopoLink));
    // 将该链路之后的所有链路向前移动一位
    while (node->links[l+1].remNode) {
      memcpy(node->links+l, node->links+l+1, sizeof(struct ncclTopoLink));
      l++;
    }
    // 将上层链路放到最后位置
    memcpy(node->links+l, &upLink, sizeof(struct ncclTopoLink));
  }

  // Recursively sort the PCI tree
  // 递归排序PCI树结构中的所有子节点
  for (int l=0; l<node->nlinks; l++) {
    // 获取当前链路
    struct ncclTopoLink* link = node->links+l;
    // 对PCI下行链路进行递归排序（跳过上行链路）
    if (link->type == LINK_PCI && link->remNode != upNode)
        NCCLCHECK(ncclTopoSort(link->remNode, node));
  }
  return ncclSuccess;
}

// We want the graph to be organized to ease/accelerate traversal :
// 我们希望图结构有序排列以简化/加速遍历：
// 1. NVLinks (already the case) - NVLink链路（已经有序）
// 2. PCI down - PCI下行链路
// 3. PCI up - PCI上行链路
// 4. SYS (already the case) - 系统级链路（已经有序）
// 对整个拓扑系统进行排序
// 参数：
//   system - 拓扑系统指针
ncclResult_t ncclTopoSortSystem(struct ncclTopoSystem* system) {
  // 从每个CPU节点开始排序
  for (int n=0; n<system->nodes[CPU].count; n++)
    NCCLCHECK(ncclTopoSort(system->nodes[CPU].nodes+n, NULL));
  return ncclSuccess;
}

// 从XML节点添加网络设备到拓扑系统
// 参数：
//   xmlNet - XML网络节点
//   system - 拓扑系统指针
//   nic - NIC节点指针
//   systemId - 系统（节点）ID
ncclResult_t ncclTopoAddNet(struct ncclXmlNode* xmlNet, struct ncclTopoSystem* system, struct ncclTopoNode* nic, int systemId) {
  // 获取网络设备索引
  int dev;
  NCCLCHECK(xmlGetAttrInt(xmlNet, "dev", &dev));

  // 创建网络节点
  struct ncclTopoNode* net;
  NCCLCHECK(ncclTopoCreateNode(system, &net, NET, NCCL_TOPO_ID(systemId, dev)));
  // 设置网络设备索引
  net->net.dev = dev;
  // 获取ASIC GUID（全局唯一标识符）
  const char* str;
  NCCLCHECK(xmlGetAttr(xmlNet, "guid", &str));
  if (str) sscanf(str, "0x%lx", &net->net.asic);
  else net->net.asic = dev;

  // 禁用图形子系统的警告（因为我们可能获取不到某些属性）
  ncclDebugNoWarn = NCCL_GRAPH;
  // 获取网络速度（Mbps）
  int mbps;
  NCCLCHECK(xmlGetAttrIntDefault(xmlNet, "speed", &mbps, 0));
  // Some NICs define speed = -1 - 某些网卡将速度定义为-1，使用默认值10Gbps
  if (mbps <= 0) mbps = 10000;
  // 将Mbps转换为GB/s（除以8000，因为使用的是Base-1000）
  net->net.bw = mbps / 8000.0;
  // 获取延迟（如果没有则使用默认值0）
  if (xmlGetAttrFloat(xmlNet, "latency", &net->net.latency) != ncclSuccess) net->net.latency = 0;
  // 获取端口号（默认为0）
  NCCLCHECK(xmlGetAttrIntDefault(xmlNet, "port", &net->net.port, 0));
  // 获取GPU Direct RDMA支持标志（默认为0）
  NCCLCHECK(xmlGetAttrIntDefault(xmlNet, "gdr", &net->net.gdrSupport, 0));
  // 获取最大连接数（默认为MAXCHANNELS，即64）
  NCCLCHECK(xmlGetAttrIntDefault(xmlNet, "maxconn", &net->net.maxChannels, MAXCHANNELS));
  // 获取集合网络支持标志（默认为0）
  NCCLCHECK(xmlGetAttrIntDefault(xmlNet, "coll", &net->net.collSupport, 0));
  // 恢复调试警告设置
  ncclDebugNoWarn = 0;

  // 在NIC和网络节点之间创建双向链路
  NCCLCHECK(ncclTopoConnectNodes(nic, net, LINK_NET, net->net.bw));
  NCCLCHECK(ncclTopoConnectNodes(net, nic, LINK_NET, net->net.bw));
  return ncclSuccess;
}

// 从XML NIC节点添加所有子网络节点到拓扑系统
// 参数：
//   xmlNic - XML NIC节点
//   system - 拓扑系统指针
//   nic - NIC节点指针
//   systemId - 系统（节点）ID
ncclResult_t ncclTopoAddNic(struct ncclXmlNode* xmlNic, struct ncclTopoSystem* system, struct ncclTopoNode* nic, int systemId) {
  // 遍历XML NIC节点的所有子节点
  for (int s=0; s<xmlNic->nSubs; s++) {
    // 获取子节点
    struct ncclXmlNode* xmlNet = xmlNic->subs[s];
    // 只处理名为"net"的子节点
    if (strcmp(xmlNet->name, "net") != 0) continue;
    // 检查是否有"dev"属性
    int index;
    NCCLCHECK(xmlGetAttrIndex(xmlNet, "dev", &index));
    // This means that the "dev" attribute wasn't set on this net xml node. That means it should not be added to the system topology graph
    // 如果没有"dev"属性，则跳过该节点（不应添加到系统拓扑图）
    if (index == -1) continue;
    // 添加网络节点到拓扑系统
    NCCLCHECK(ncclTopoAddNet(xmlNet, system, nic, systemId));
  }
  return ncclSuccess;
}

// 从XML GPU节点填充GPU节点的属性
// 参数：
//   xmlGpu - XML GPU节点
//   system - 拓扑系统指针
//   gpu - GPU节点指针
ncclResult_t ncclTopoAddGpu(struct ncclXmlNode* xmlGpu, struct ncclTopoSystem* system, struct ncclTopoNode* gpu) {
  // 获取并设置GPU的计算能力（SM版本，如70、80、90等）
  NCCLCHECK(xmlGetAttrInt(xmlGpu, "sm", &gpu->gpu.cudaCompCap));
  // 获取并设置GPU在通信域中的rank号
  NCCLCHECK(xmlGetAttrInt(xmlGpu, "rank", &gpu->gpu.rank));
  // 获取并设置GPU设备号
  NCCLCHECK(xmlGetAttrInt(xmlGpu, "dev", &gpu->gpu.dev));
  // 获取并设置GPU Direct RDMA支持标志
  NCCLCHECK(xmlGetAttrInt(xmlGpu, "gdr", &gpu->gpu.gdrSupport));
  // Do not go any further, nvlinks will be added in a second pass
  // 不继续处理子节点，NVLink将在第二轮遍历中添加
  return ncclSuccess;
}

// PCI桥设备类代码定义，用于识别PCI桥接器
#define PCI_BRIDGE_DEVICE_CLASS "0x060400"

// PCI设备类到节点类型的转换字典
// 键值对格式：{"类代码字符串", 节点类型枚举}
struct kvDict kvDictPciClass[] = {
  { PCI_BRIDGE_DEVICE_CLASS, PCI },           // PCI桥接器
  {"0x080100", /*CX8 data direct*/PCI},       // CX8数据直接设备
  { "0x068000", NVS },                        // NVSwitch设备
  { "0x068001", CPU },                        // CPU设备
  { "0x03", GPU },                            // GPU设备（0x03xx表示显示控制器类）
  { "0x02", NIC },                            // 网络控制器（NIC）
  { NULL, PCI /* Default fallback value */ }  // 默认值：普通PCI设备
};

// PCI代数到带宽的转换字典
// 键值对格式：{"速度字符串", 带宽值}，带宽单位为100 Mbps/通道
// 最终需要除以80转换为GB/s（8 bits/byte，10进制转2进制约数）
struct kvDict kvDictPciGen[] = {
  // Kernel 5.6 and earlier - 早期内核格式
  { "2.5 GT/s", 15 },      // PCIe Gen1: 2.5 GT/s → 1.5 GB/s (15*100/80)
  { "5 GT/s", 30 },        // PCIe Gen2: 5 GT/s → 3.75 GB/s
  { "8 GT/s", 60 },        // PCIe Gen3: 8 GT/s → 6 GB/s
  { "16 GT/s", 120 },      // PCIe Gen4: 16 GT/s → 12 GB/s
  { "32 GT/s", 240 },      // PCIe Gen5: 32 GT/s → 24 GB/s
  // 新格式（带"PCIe"后缀）
  { "2.5 GT/s PCIe", 15 },
  { "5.0 GT/s PCIe", 30 },
  { "8.0 GT/s PCIe", 60 },
  { "16.0 GT/s PCIe", 120 },
  { "32.0 GT/s PCIe", 240 },
  { "64.0 GT/s PCIe", 480 },  // PCIe Gen6: 64 GT/s → 48 GB/s
  { NULL, 60 /* Default fallback */ }  // 默认值：6 GB/s (Gen3)
}; // x100 Mbps per lane
// 从XML PCI节点添加PCI设备到拓扑系统
// 这是一个递归函数，可以处理PCI树结构
// 参数：
//   xmlPci - XML PCI节点
//   system - 拓扑系统指针
//   parent - 父节点指针
//   systemId - 系统（节点）ID
//   numaId - NUMA节点ID
ncclResult_t ncclTopoAddPci(struct ncclXmlNode* xmlPci, struct ncclTopoSystem* system, struct ncclTopoNode* parent, int systemId, int numaId) {
  // 字符串指针，用于存储从XML获取的属性值
  const char* str;

  // PCI设备类型变量
  int type;
  // 从XML获取class属性并转换为设备类型
  NCCLCHECK(xmlGetAttrStr(xmlPci, "class", &str));
  NCCLCHECK(kvConvertToInt(str, &type, kvDictPciClass));

  // PCI总线ID变量
  int64_t busId;
  // 从XML获取busid属性并转换为64位整数
  NCCLCHECK(xmlGetAttrStr(xmlPci, "busid", &str));
  NCCLCHECK(busIdToInt64(str, &busId));

  // 节点指针初始化为NULL
  struct ncclTopoNode* node = NULL;
  // 查找GPU子节点
  struct ncclXmlNode* xmlGpu = NULL;
  NCCLCHECK(xmlGetSub(xmlPci, "gpu", &xmlGpu));
  if (xmlGpu != NULL) {
    // 如果找到GPU子节点，强制设置类型为GPU
    type = GPU;
    // 检查GPU是否有rank属性
    int index;
    NCCLCHECK(xmlGetAttrIndex(xmlGpu, "rank", &index));
    // 如果没有rank属性，跳过该GPU
    if (index == -1) return ncclSuccess;
    // 创建GPU节点
    NCCLCHECK(ncclTopoCreateNode(system, &node, type, NCCL_TOPO_ID(systemId, busId)));
    // 填充GPU节点属性
    NCCLCHECK(ncclTopoAddGpu(xmlGpu, system, node));
  }
  // 查找NIC子节点
  struct ncclXmlNode* xmlNic = NULL;
  NCCLCHECK(xmlGetSub(xmlPci, "nic", &xmlNic));
  if (xmlNic != NULL) {
    // 如果找到NIC子节点，设置类型为NIC
    type = NIC;
    // Ignore sub device ID and merge multi-port NICs into one PCI device.
    // 忽略子设备ID，将多端口网卡合并为一个PCI设备
    struct ncclTopoNode* nicNode = NULL;
    // 生成本地NIC ID（包含NUMA节点信息）
    int64_t localNicId = NCCL_TOPO_LOCAL_NIC_ID(numaId, busId);
    // 生成完整的NIC ID
    int64_t id = NCCL_TOPO_ID(systemId, localNicId);
    // 尝试获取已存在的NIC节点
    NCCLCHECK(ncclTopoGetNode(system, &nicNode, type, id));
    if (nicNode == NULL) {
      // 如果不存在，创建新的NIC节点
      NCCLCHECK(ncclTopoCreateNode(system, &nicNode, type, id));
      // 保存节点指针，稍后连接到父节点
      node = nicNode; // Connect it to parent later on
    }
    // 添加NIC的网络设备
    NCCLCHECK(ncclTopoAddNic(xmlNic, system, nicNode, systemId));
  } else if (type == PCI) {
    // 如果是普通PCI设备，创建PCI节点
    NCCLCHECK(ncclTopoCreateNode(system, &node, type, NCCL_TOPO_ID(systemId, busId)));
    // 获取并设置供应商ID（位于高16位）
    NCCLCHECK(xmlGetAttr(xmlPci, "vendor", &str));
    if (str) node->pci.device += strtol(str, NULL, 0) << 48;
    // 获取并设置设备ID（位于次高16位）
    NCCLCHECK(xmlGetAttr(xmlPci, "device", &str));
    if (str) node->pci.device += strtol(str, NULL, 0) << 32;
    // 获取并设置子系统供应商ID（位于中16位）
    NCCLCHECK(xmlGetAttr(xmlPci, "subsystem_vendor", &str));
    if (str) node->pci.device += strtol(str, NULL, 0) << 16;
    // 获取并设置子系统设备ID（位于低16位）
    NCCLCHECK(xmlGetAttr(xmlPci, "subsystem_device", &str));
    if (str) node->pci.device += strtol(str, NULL, 0);

    // 递归处理所有子PCI节点
    for (int s=0; s<xmlPci->nSubs; s++) {
      // 获取子节点
      struct ncclXmlNode* xmlSubPci = xmlPci->subs[s];
      // PCI links will be added later - PCI链路将在后续添加，跳过pcilink节点
      if (strcmp(xmlSubPci->name, "pcilink") != 0) {
        NCCLCHECK(ncclTopoAddPci(xmlSubPci, system, node, systemId, numaId));
      }
    }
  }

  // 如果创建了节点，连接到父节点
  if (node) {
    // 链路宽度和速度变量
    int width, speed;
    // 获取链路宽度（通道数，如x1, x4, x8, x16）
    NCCLCHECK(xmlGetAttrInt(xmlPci, "link_width", &width));
    // 获取链路速度字符串
    NCCLCHECK(xmlGetAttrStr(xmlPci, "link_speed", &str));

    // Manage cases where speed was not indicated in /sys
    // 如果/sys中没有指定链路宽度，使用默认值16
    if (width == 0) width = 16;
    // 将速度字符串转换为数值（单位：100 Mbps/通道）
    NCCLCHECK(kvConvertToInt(str, &speed, kvDictPciGen)); // Values in 100Mbps, per lane (we want GB/s in the end)

    // 创建双向PCI链路，带宽 = 宽度 × 速度 / 80（转换为GB/s）
    NCCLCHECK(ncclTopoConnectNodes(node, parent, LINK_PCI, width*speed/80.0));
    NCCLCHECK(ncclTopoConnectNodes(parent, node, LINK_PCI, width*speed/80.0));
  }
  return ncclSuccess;
}

// CPU架构字符串到枚举的转换字典
struct kvDict kvDictCpuArch[] = {
  { "x86_64", NCCL_TOPO_CPU_ARCH_X86 },      // x86-64架构（Intel/AMD/兆芯）
  { "arm64", NCCL_TOPO_CPU_ARCH_ARM },       // ARM64架构
  { "ppc64", NCCL_TOPO_CPU_ARCH_POWER },     // IBM PowerPC64架构
  { NULL, 0 }                                // 结束标记
};

// CPU厂商字符串到枚举的转换字典
struct kvDict kvDictCpuVendor[] = {
  { "GenuineIntel", NCCL_TOPO_CPU_VENDOR_INTEL },          // Intel厂商ID
  { "AuthenticAMD", NCCL_TOPO_CPU_VENDOR_AMD },            // AMD厂商ID
  { "CentaurHauls", NCCL_TOPO_CPU_VENDOR_ZHAOXIN },        // Centaur（兆芯前身）
  { "  Shanghai  ", NCCL_TOPO_CPU_VENDOR_ZHAOXIN },        // 上海兆芯（注意前后有空格）
  { NULL, 0 }                                              // 结束标记
};

// 获取或分配系统ID（用于多节点环境）
// 每个物理节点（主机）有一个唯一的系统ID
// 参数：
//   system - 拓扑系统指针
//   xmlCpu - XML CPU节点（包含host_hash属性）
//   systemIdPtr - 输出参数，返回系统ID
ncclResult_t ncclGetSystemId(struct ncclTopoSystem* system, struct ncclXmlNode* xmlCpu, int* systemIdPtr) {
  // 主机哈希字符串（用于唯一标识一个物理节点）
  const char* hostHashStr;
  NCCLCHECK(xmlGetAttr(xmlCpu, "host_hash", &hostHashStr));
  // 将十六进制字符串转换为64位整数
  uint64_t hostHash = hostHashStr ? strtoull(hostHashStr, NULL, 16) : 0;
  // 系统ID变量
  int systemId;

  // 在已知的主机哈希列表中查找是否已存在该主机
  for (systemId=0; systemId<system->nHosts; systemId++)
    if (system->hostHashes[systemId] == hostHash)
        break;

  // 如果是新主机，添加到主机哈希列表
  if (systemId == system->nHosts)
    system->hostHashes[system->nHosts++] = hostHash;

  // 返回系统ID
  *systemIdPtr = systemId;
  return ncclSuccess;
}


// 从XML CPU节点添加CPU到拓扑系统
// 参数：
//   xmlCpu - XML CPU节点
//   system - 拓扑系统指针
ncclResult_t ncclTopoAddCpu(struct ncclXmlNode* xmlCpu, struct ncclTopoSystem* system) {
  // NUMA节点ID
  int numaId;
  NCCLCHECK(xmlGetAttrInt(xmlCpu, "numaid", &numaId));
  // 系统（节点）ID
  int systemId;
  NCCLCHECK(ncclGetSystemId(system, xmlCpu, &systemId));
  // 创建CPU节点
  struct ncclTopoNode* cpu;
  NCCLCHECK(ncclTopoCreateNode(system, &cpu, CPU, NCCL_TOPO_ID(systemId, numaId)));
  // 字符串指针
  const char* str;
  // 获取CPU亲和性（CPU核心集合）
  NCCLCHECK(xmlGetAttr(xmlCpu, "affinity", &str));
  if (str != NULL) {
    // 将字符串转换为cpu_set_t结构
    NCCLCHECK(ncclStrToCpuset(str, &cpu->cpu.affinity));
  }

  // 获取并设置CPU架构
  NCCLCHECK(xmlGetAttrStr(xmlCpu, "arch", &str));
  NCCLCHECK(kvConvertToInt(str, &cpu->cpu.arch, kvDictCpuArch));
  // 如果是x86架构，需要进一步获取厂商和型号信息
  if (cpu->cpu.arch == NCCL_TOPO_CPU_ARCH_X86) {
    // 获取CPU厂商
    NCCLCHECK(xmlGetAttrStr(xmlCpu, "vendor", &str));
    NCCLCHECK(kvConvertToInt(str, &cpu->cpu.vendor, kvDictCpuVendor));
    // Intel CPU的特殊处理：根据family和model ID确定具体型号
    if (cpu->cpu.vendor == NCCL_TOPO_CPU_VENDOR_INTEL) {
      // 获取CPU family ID和model ID
      int familyId, modelId;
      NCCLCHECK(xmlGetAttrInt(xmlCpu, "familyid", &familyId));
      NCCLCHECK(xmlGetAttrInt(xmlCpu, "modelid", &modelId));
      // 根据family和model确定Intel CPU型号（从新到旧判断）：
      // ERP (Emerald Rapids) - model >= 0xCF
      // SRP (Sapphire Rapids) - model >= 0x8F
      // SKL (Skylake) - model >= 0x55
      // BDW (Broadwell) - 其他情况（默认）
      cpu->cpu.model =
        (familyId == 6 && modelId >= 0xCF) ? NCCL_TOPO_CPU_MODEL_INTEL_ERP :
        (familyId == 6 && modelId >= 0x8F) ? NCCL_TOPO_CPU_MODEL_INTEL_SRP :
        (familyId == 6 && modelId >= 0x55) ? NCCL_TOPO_CPU_MODEL_INTEL_SKL :
        NCCL_TOPO_CPU_MODEL_INTEL_BDW;
    // Zhaoxin（兆芯）CPU的特殊处理
    } else if (cpu->cpu.vendor == NCCL_TOPO_CPU_VENDOR_ZHAOXIN) {
      // 获取CPU family ID和model ID
      int familyId, modelId;
      NCCLCHECK(xmlGetAttrInt(xmlCpu, "familyid", &familyId));
      NCCLCHECK(xmlGetAttrInt(xmlCpu, "modelid", &modelId));
      // 检查是否为 YongFeng（兆芯的特定型号）
      if (familyId == 7 && modelId == 0x5B) cpu->cpu.model = NCCL_TOPO_CPU_MODEL_YONGFENG;
    }
  }
  // 遍历CPU节点的所有子节点
  for (int s=0; s<xmlCpu->nSubs; s++) {
    // 获取子节点
    struct ncclXmlNode* node = xmlCpu->subs[s];
    // 处理PCI子节点（递归添加PCI设备）
    if (strcmp(node->name, "pci") == 0) NCCLCHECK(ncclTopoAddPci(node, system, cpu, systemId, numaId));
    // 处理NIC子节点（直接连接到CPU的网卡）
    if (strcmp(node->name, "nic") == 0) {
      // 查找或创建NIC节点
      struct ncclTopoNode* nic = NULL;
      // 生成本地NIC ID（busId为0表示直接连接到CPU）
      int64_t localNicId = NCCL_TOPO_LOCAL_NIC_ID(numaId, 0);
      // 生成完整的NIC ID
      int64_t id = NCCL_TOPO_ID(systemId, localNicId);
      // 尝试获取已存在的NIC节点
      NCCLCHECK(ncclTopoGetNode(system, &nic, NIC, id));
      if (nic == NULL) {
        // 如果不存在，创建新的NIC节点并连接到CPU
        NCCLCHECK(ncclTopoCreateNode(system, &nic, NIC, id));
        // 创建CPU到NIC的双向链路（使用本地带宽）
        NCCLCHECK(ncclTopoConnectNodes(cpu, nic, LINK_PCI, LOC_BW));
        NCCLCHECK(ncclTopoConnectNodes(nic, cpu, LINK_PCI, LOC_BW));
      }
      // 添加NIC的网络设备
      NCCLCHECK(ncclTopoAddNic(node, system, nic, systemId));
    }
  }
  return ncclSuccess;
}

// 从XML节点递归添加NVLink连接到拓扑系统
// 这是一个递归函数，遍历XML树查找nvlink节点
// 参数：
//   node - XML节点
//   system - 拓扑系统指针
//   parentBusId - 父节点的PCI总线ID字符串
//   systemId - 系统（节点）ID
ncclResult_t ncclTopoAddNvLinks(struct ncclXmlNode* node, struct ncclTopoSystem* system, const char* parentBusId, int systemId) {
  // 如果当前节点是nvlink节点，处理NVLink连接
  if (strcmp(node->name, "nvlink") == 0) {
    // GPU节点指针
    struct ncclTopoNode* gpu = NULL;
    // 父GPU的PCI总线ID
    int64_t pBusId;
    NCCLCHECK(busIdToInt64(parentBusId, &pBusId));
    // 生成完整的节点ID
    pBusId = NCCL_TOPO_ID(systemId, pBusId);
    // 查找源GPU节点
    NCCLCHECK(ncclTopoGetNode(system, &gpu, GPU, pBusId));
    if (gpu == NULL) {
      WARN("Add NVLink error : could not find GPU %lx", pBusId);
      return ncclInternalError;
    }
    // NVLink数量（可能有多条NVLink）
    int count;
    NCCLCHECK(xmlGetAttrInt(node, "count", &count));
    // 目标设备类字符串
    const char* targetClass;
    NCCLCHECK(xmlGetAttrStr(node, "tclass", &targetClass));
    // 目标设备类型
    int targetType;
    NCCLCHECK(kvConvertToInt(targetClass, &targetType, kvDictPciClass));
    // 远端节点指针
    struct ncclTopoNode* remote = NULL;
    if (targetType == GPU) {
      // NVL P2P connection to another GPU
      // 目标是另一个GPU，建立GPU到GPU的NVLink连接
      const char* target;
      NCCLCHECK(xmlGetAttrStr(node, "target", &target));
      // 转换目标GPU的PCI总线ID
      int64_t busId;
      NCCLCHECK(busIdToInt64(target, &busId));
      // 查找目标GPU节点
      NCCLCHECK(ncclTopoGetNode(system, &remote, GPU, NCCL_TOPO_ID(systemId, busId)));
    } else if (targetType == CPU) {
      // NVL connection to the local CPU
      // 目标是本地CPU，通过PCI树查找CPU
      NCCLCHECK(findLocalCpu(gpu, &remote, NULL));
    } else {
      // 目标是NVSwitch或其他设备
      if (system->nodes[NVS].count == 0) {
        // 如果NVS节点不存在，创建一个新的
        NCCLCHECK(ncclTopoCreateNode(system, &remote, NVS, 0));
      } else {
        // 使用已存在的NVS节点
        remote = system->nodes[NVS].nodes;
      }
    }
    // 如果找到了远端节点，创建NVLink连接
    if (remote) {
      // 根据GPU计算能力获取单条NVLink的带宽
      float nvlBw = ncclTopoNVLinkBw(gpu->gpu.cudaCompCap);
      // 创建GPU到远端节点的NVLink链路（带宽 = NVLink数量 × 单条带宽）
      NCCLCHECK(ncclTopoConnectNodes(gpu, remote, LINK_NVL, count*nvlBw));
      // 如果远端不是GPU，需要创建反向链路
      // GPU到GPU的连接是双向的，已经在两个GPU上都创建了链路
      if (remote->type != GPU) {
        NCCLCHECK(ncclTopoConnectNodes(remote, gpu, LINK_NVL, count*nvlBw));
      }
    }
  } else {
    // 如果不是nvlink节点，递归遍历子节点
    if (strcmp(node->name, "cpu") == 0) {
      // 如果是CPU节点，获取/更新系统ID
      NCCLCHECK(ncclGetSystemId(system, node, &systemId));
    }
    // 获取当前节点的总线ID
    const char* busId;
    NCCLCHECK(xmlGetAttr(node, "busid", &busId));
    // 递归处理所有子节点，传递总线ID（如果没有则使用父节点的）
    for (int s=0; s<node->nSubs; s++) {
      NCCLCHECK(ncclTopoAddNvLinks(node->subs[s], system, busId ? busId : parentBusId, systemId));
    }
  }
  return ncclSuccess;
}

// 从XML节点递归添加PCI链路连接到拓扑系统
// 这是一个递归函数，遍历XML树查找pcilink节点
// PCI链路用于连接同一层级的PCI交换机
// 参数：
//   node - XML节点
//   system - 拓扑系统指针
//   parentBusId - 父节点的PCI总线ID字符串
//   systemId - 系统（节点）ID
ncclResult_t ncclTopoAddPciLinks(struct ncclXmlNode* node, struct ncclTopoSystem* system, const char* parentBusId, int systemId) {
  // 如果当前节点是pcilink节点，处理PCI链路连接
  if (strcmp(node->name, "pcilink") == 0) {
    // PCI交换机节点指针
    struct ncclTopoNode* pci = NULL;
    // 父PCI交换机的总线ID
    int64_t pBusId;
    NCCLCHECK(busIdToInt64(parentBusId, &pBusId));
    // 生成完整的节点ID
    pBusId = NCCL_TOPO_ID(systemId, pBusId);
    // 查找源PCI交换机节点
    NCCLCHECK(ncclTopoGetNode(system, &pci, PCI, pBusId));
    if (pci == NULL) {
      WARN("Add PCI Link error : could not find PCI SW %lx", pBusId);
      return ncclInternalError;
    }
    // 远端PCI节点指针
    struct ncclTopoNode* remote = NULL;
    // 获取目标设备的PCI总线ID字符串
    const char* target;
    NCCLCHECK(xmlGetAttrStr(node, "target", &target));
    // 转换目标PCI总线ID
    int64_t busId;
    NCCLCHECK(busIdToInt64(target, &busId));
    // 查找目标PCI节点
    NCCLCHECK(ncclTopoGetNode(system, &remote, PCI, NCCL_TOPO_ID(systemId, busId)));
    // 如果找到目标PCI节点，创建本地链路（LINK_LOC）
    // 使用本地带宽，表示同一层级的PCI交换机之间的连接
    if (remote) NCCLCHECK(ncclTopoConnectNodes(pci, remote, LINK_LOC, LOC_BW));
  } else {
    // 如果不是pcilink节点，递归遍历子节点
    if (strcmp(node->name, "cpu") == 0) {
      // 如果是CPU节点，获取/更新系统ID
      NCCLCHECK(ncclGetSystemId(system, node, &systemId));
    }
    // 获取当前节点的总线ID
    const char* busId;
    NCCLCHECK(xmlGetAttr(node, "busid", &busId));
    // 递归处理所有子节点，传递总线ID（如果没有则使用父节点的）
    for (int s=0; s<node->nSubs; s++) {
      NCCLCHECK(ncclTopoAddPciLinks(node->subs[s], system, busId ? busId : parentBusId, systemId));
    }
  }
  return ncclSuccess;
}


// 从XML节点递归添加C2C（Cache-Coherent Interconnect）连接到拓扑系统
// C2C是NVIDIA Grace CPU和GPU之间的高速互连技术
// 这是一个递归函数，遍历XML树查找c2c节点
// 参数：
//   node - XML节点
//   system - 拓扑系统指针
//   parentBusId - 父节点的PCI总线ID字符串
//   systemId - 系统（节点）ID
ncclResult_t ncclTopoAddC2c(struct ncclXmlNode* node, struct ncclTopoSystem* system, const char* parentBusId, int systemId) {
  // 如果当前节点是c2c节点，处理C2C连接
  if (strcmp(node->name, "c2c") == 0) {
    // GPU节点指针
    struct ncclTopoNode* gpu = NULL;
    // 父GPU的PCI总线ID
    int64_t pBusId;
    NCCLCHECK(busIdToInt64(parentBusId, &pBusId));
    // 生成完整的节点ID
    pBusId = NCCL_TOPO_ID(systemId, pBusId);
    // 查找GPU节点
    NCCLCHECK(ncclTopoGetNode(system, &gpu, GPU, pBusId));
    if (gpu == NULL) {
      WARN("Add NVLink error : could not find GPU %lx", pBusId);
      return ncclInternalError;
    }
    // C2C链路数量
    int count = 0;
    NCCLCHECK(xmlGetAttrInt(node, "count", &count));
    // 单条C2C链路的带宽（MBps）
    int bw = 0;
    NCCLCHECK(xmlGetAttrInt(node, "bw", &bw));
    // 计算总带宽并转换为GB/s（MBps × 数量 / 1000）
    double c2cBw = (bw*count)/1000.0;
    // 本地CPU节点指针
    struct ncclTopoNode* cpu = NULL;
    // 查找与GPU相连的本地CPU
    NCCLCHECK(findLocalCpu(gpu, &cpu, NULL));
    // 如果找不到CPU，返回成功（跳过该连接）
    if (cpu == NULL) return ncclSuccess;
    // 创建GPU到CPU的C2C双向链路
    NCCLCHECK(ncclTopoConnectNodes(gpu, cpu, LINK_C2C, c2cBw));
    NCCLCHECK(ncclTopoConnectNodes(cpu, gpu, LINK_C2C, c2cBw));
  } else {
    // 如果不是c2c节点，递归遍历子节点
    if (strcmp(node->name, "cpu") == 0) {
      // 如果是CPU节点，获取/更新系统ID
      NCCLCHECK(ncclGetSystemId(system, node, &systemId));
    }
    // 获取当前节点的总线ID
    const char* busId;
    NCCLCHECK(xmlGetAttr(node, "busid", &busId));
    // 递归处理所有子节点，传递总线ID（如果没有则使用父节点的）
    for (int s=0; s<node->nSubs; s++) {
      NCCLCHECK(ncclTopoAddC2c(node->subs[s], system, busId ? busId : parentBusId, systemId));
    }
  }
  return ncclSuccess;
}

// 从XML结构构建拓扑系统
// 这是NCCL拓扑发现的核心函数，将XML描述的硬件拓扑转换为内部数据结构
// 参数：
//   xml - XML结构指针
//   topoSystem - 输出参数，返回创建的拓扑系统
//   localHostHash - 本地主机的哈希值，用于标识当前节点
ncclResult_t ncclTopoGetSystemFromXml(struct ncclXml* xml, struct ncclTopoSystem** topoSystem, const uint64_t localHostHash) {
  // 分配并初始化拓扑系统结构
  NCCLCHECK(ncclCalloc(topoSystem, 1));

  // 获取拓扑系统指针的局部引用
  struct ncclTopoSystem* system = *topoSystem;
  // XML顶层节点指针（system节点）
  struct ncclXmlNode* topNode;
  // 在XML中查找system标签
  NCCLCHECK(xmlFindTag(xml, "system", &topNode));
  // 遍历system节点的所有子节点
  for (int s=0; s<topNode->nSubs; s++) {
    // 获取子节点
    struct ncclXmlNode* node = topNode->subs[s];
    // 处理CPU节点（添加CPU及其下的PCI设备）
    if (strcmp(node->name, "cpu") == 0)
        NCCLCHECK(ncclTopoAddCpu(node, *topoSystem));
  }

  // 系统ID变量
  int systemId = 0;
  //hostHashes中找到等于localHostHash的索引
  // 在主机哈希列表中查找本地主机的索引
  while (systemId < system->nHosts && system->hostHashes[systemId] != localHostHash)
    systemId++;

  // 设置系统ID（当前节点在整个多节点系统中的索引）
  system->systemId = systemId;
  // 检查是否找到了本地主机
  if(systemId == system->nHosts){
    WARN("localHostHash = 0x%lx not found in the list of system hostHashes",localHostHash);
    return ncclInvalidArgument;
  }

  // 添加NVLink连接（第二轮处理，此时所有GPU节点已创建）
  NCCLCHECK(ncclTopoAddNvLinks(topNode, *topoSystem, NULL, 0));
  // 添加C2C连接（Grace CPU和GPU之间的互连）
  NCCLCHECK(ncclTopoAddC2c(topNode, *topoSystem, NULL, 0));
  // 添加PCI链路（同级PCI交换机之间的连接）
  NCCLCHECK(ncclTopoAddPciLinks(topNode, *topoSystem, NULL, 0));

  // 扁平化BCM交换机结构（将两层结构合并为单层）
  NCCLCHECK(ncclTopoFlattenBcmSwitches(*topoSystem));
  // 连接所有CPU节点（创建CPU间互联）
  NCCLCHECK(ncclTopoConnectCpus(*topoSystem));
  // 排序拓扑系统（优化链路顺序以加速遍历）
  NCCLCHECK(ncclTopoSortSystem(*topoSystem));

  return ncclSuccess;
}

// 定义环境变量参数：哪个rank负责转储拓扑文件
NCCL_PARAM(TopoDumpFileRank, "TOPO_DUMP_FILE_RANK", 0);

// Only set values if not already set
// XML属性初始化辅助函数：仅在属性不存在时设置整数值
// 参数：
//   node - XML节点指针
//   attrName - 属性名称
//   value - 要设置的整数值
static ncclResult_t xmlInitAttrInt(struct ncclXmlNode* node, const char* attrName, const int value) {
  // 查找属性索引
  int index;
  NCCLCHECK(xmlGetAttrIndex(node, attrName, &index));
  // 如果属性不存在，则添加
  if (index == -1) {
    // 增加属性计数并获取新属性的索引
    index = node->nAttrs++;
    // 复制属性名称
    strncpy(node->attrs[index].key, attrName, MAX_STR_LEN);
    // 确保字符串以null结尾
    node->attrs[index].key[MAX_STR_LEN] = '\0';
    // 将整数值转换为字符串
    snprintf(node->attrs[index].value, MAX_STR_LEN, "%d", value);
  }
  return ncclSuccess;
}

// XML属性初始化辅助函数：仅在属性不存在时设置64位无符号整数值
// 参数：
//   node - XML节点指针
//   attrName - 属性名称
//   value - 要设置的64位无符号整数值
static ncclResult_t xmlInitAttrUint64(struct ncclXmlNode* node, const char* attrName, const uint64_t value) {
  // 查找属性索引
  int index;
  NCCLCHECK(xmlGetAttrIndex(node, attrName, &index));
  // 如果属性不存在，则添加
  if (index == -1) {
    // 增加属性计数并获取新属性的索引
    index = node->nAttrs++;
    // 复制属性名称
    strncpy(node->attrs[index].key, attrName, MAX_STR_LEN);
    // 确保字符串以null结尾
    node->attrs[index].key[MAX_STR_LEN] = '\0';
    // 将64位整数值转换为十六进制字符串
    snprintf(node->attrs[index].value, MAX_STR_LEN, "0x%lx", value);
  }
  return ncclSuccess;
}

// XML属性初始化辅助函数：仅在属性不存在时设置浮点数值
// 参数：
//   node - XML节点指针
//   attrName - 属性名称
//   value - 要设置的浮点数值
static ncclResult_t xmlInitAttrFloat(struct ncclXmlNode* node, const char* attrName, const float value) {
  // 查找属性索引
  int index;
  NCCLCHECK(xmlGetAttrIndex(node, attrName, &index));
  // 如果属性不存在，则添加
  if (index == -1) {
    // 增加属性计数并获取新属性的索引
    index = node->nAttrs++;
    // 复制属性名称
    strncpy(node->attrs[index].key, attrName, MAX_STR_LEN);
    // 确保字符串以null结尾
    node->attrs[index].key[MAX_STR_LEN] = '\0';
    // 将浮点数值转换为字符串
    snprintf(node->attrs[index].value, MAX_STR_LEN, "%f", value);
  }
  return ncclSuccess;
}

// 刷新BCM P2P链路信息
// 通过读取内核文件触发PCI交换机拓扑刷新
ncclResult_t ncclTopoRefreshBcmP2pLinks(void) {
  //refresh the switch topology by reading the link below
  // 打开BCM交换机内核接口文件（注意：文件名中"toplogy"是拼写错误，应为"topology"）
  FILE *fp = fopen("/sys/kernel/pci_switch_link/refresh_switch_toplogy", "r");
  if (fp != NULL) {
    // 临时变量
    int tmp;
    // 读取文件内容（触发光拓扑刷新）
    size_t r = fread(&tmp, sizeof(tmp), 1, fp);
    // 检查读取是否成功
    if (r != 1)
      INFO(NCCL_GRAPH, "Failed to read refresh_switch_toplogy");
    // 关闭文件
    fclose(fp);
  }
  return ncclSuccess;
}

// This is just checking for direct descendence
// 检查多个节点是否都直接从同一个PCI交换机（父节点） descend
// 用于判断是否为PIX（PCI Internal Xbar）路径类型
// 参数：
//   common - 公共父节点
//   nodes - 节点数组
//   nNodes - 节点数量
// 返回值：1表示是PIX，0表示不是
int ncclTopoCheckPix(ncclXmlNode* common, ncclXmlNode** nodes, int nNodes) {
  // 临时变量：存储PCI总线ID
  const char* tempBusId;
  // If the common parent isn't a pci switch, then this isn't PIX
  // 如果公共父节点不是PCI交换机（没有busid属性），则不是PIX
  NCCLCHECK(xmlGetAttrStr(common, "busid", &tempBusId));
  if (tempBusId == NULL) return 0;
  TRACE(NCCL_GRAPH, "Checking pix for busid=%s", tempBusId);

  // All the nodes must have a "nic" which is a parent, and then a pci node (busid) which must be a child of the "common"
  // 所有节点必须满足：net -> nic -> pci，且pci的父节点是common
  for (int i = 0; i < nNodes; i++) {
    // 获取当前节点
    ncclXmlNode* node = nodes[i];
    // 检查是否为网络设备节点
    if (strcmp(node->name, "net") == 0) {
      // 向上移动到父节点（NIC）
      node = node->parent;
      if (node == NULL) return 0;
      // 检查是否为NIC节点
      if (strcmp(node->name, "nic") == 0) {
        // 向上移动到父节点（PCI）
        node = node->parent;
        if (node == NULL) return 0;
        // All nodes must descend from the same first level pci switch
        // 检查是否为PCI节点且其父节点是common
        if (strcmp(node->name, "pci") == 0) {
          TRACE(NCCL_GRAPH, "Comparing parent of node=%p to common=%p", node->parent, common);
          // 如果父节点不是common，则不是PIX
          if (node->parent != common) return 0;
        }
      }
    }
  }

  // 所有节点都满足条件，返回1（是PIX）
  return 1;
}

// 定义XML树的最大深度
#define NCCL_TOPO_XML_DEPTH_MAX 256
// XML节点栈结构，用于深度优先遍历XML树
typedef struct xmlNodeStack {
  // 节点指针数组
  ncclXmlNode* elems[NCCL_TOPO_XML_DEPTH_MAX];
  // 栈顶指针（指向下一个空位）
  int tail;

  // 获取栈顶元素（不弹出）
  ncclXmlNode* top() {
    if (!empty()) {
      // 返回栈顶元素
      return elems[tail - 1];
    } else {
      // 栈为空，返回NULL
      return NULL;
    }
  }

  // 弹出栈顶元素
  ncclXmlNode* pop() {
    // 获取栈顶元素
    ncclXmlNode* node = top();
    if (node) {
      // 移动栈顶指针
      tail--;
    }
    // 返回弹出的元素
    return node;
  }

  // 将元素压入栈中
  void push(ncclXmlNode* node) {
    // 检查栈是否已满
    if (tail < NCCL_TOPO_XML_DEPTH_MAX) {
      // 压入节点并移动栈顶指针
      elems[tail++] = node;
    }
  }

  // 检查栈是否为空
  bool empty() {
    // 栈顶指针为0表示栈为空
    return tail == 0;
  }

} xmlNodeStack;

// 查找第一个PCI父节点
// 向上遍历XML树，找到第一个名为"pci"的节点
// 参数：
//   parent - 输入输出参数，输入为起始节点，输出为找到的PCI节点
ncclResult_t ncclFindFirstPciParent(ncclXmlNode** parent) {
  // 获取父节点的副本
  ncclXmlNode* newParent = *parent;
  // 向上遍历直到找到PCI节点
  while (strcmp(newParent->name, "pci") != 0) {
    // 移动到父节点
    newParent = newParent->parent;
    // 检查是否到达顶层
    if (newParent == nullptr) return ncclSuccess;
    // 检查是否到达system节点（XML根节点）
    if (strcmp(newParent->name, "system") == 0) return ncclSuccess;
  }
  // 返回找到的PCI节点
  *parent = newParent;
  return ncclSuccess;
}

// 1. Find the common parent xmlNode between the given set of nodes
// 找到一组节点之间的公共父节点，并确定它们之间的路径类型
// 参数：
//   nodes - 节点数组
//   nNodes - 节点数量
//   path - 输出参数，返回路径类型
//   parent - 输出参数，返回公共父节点
ncclResult_t ncclTopoGetPath(ncclXmlNode** nodes, int nNodes, int* path, ncclXmlNode** parent) {
  // Track a stack of parents per-net node being merged
  // 为每个节点分配一个栈，用于存储从节点到根的路径
  xmlNodeStack* parents;
  NCCLCHECK(ncclCalloc(&parents, nNodes));
  // Find the common parent
  // 公共父节点指针
  ncclXmlNode* common = NULL;

  // 特殊情况：只有一个节点，路径类型为本地（PATH_LOC）
  if (nNodes == 1) {
    common = nodes[0];
    *path = PATH_LOC;
    goto out;
  }

  // 为每个节点建立从自身到根节点的路径栈
  for (int i = 0; i < nNodes; i++) {
    // 临时变量
    ncclXmlNode* temp;
    // 从当前节点开始
    temp = nodes[i];
    // 向上遍历到根节点
    while (temp) {
      // 将当前节点压入栈
      parents[i].push(temp);
      // 如果到达system节点则停止，否则继续向上
      temp = strcmp(temp->name, "system") == 0 ? NULL : temp->parent;
    }
  }

  // 初始化公共父节点
  common = NULL;
  // 匹配标志
  int c;
  c = 1;
  // 从栈顶开始比较，找到最后一个公共节点
  while (c && !parents[0].empty()) {
    // 获取第一个节点的栈顶元素
    ncclXmlNode* temp = parents[0].top();
    // 与其他节点的栈顶元素比较
    for (int i = 1; i < nNodes; i++) {
      // 检查栈是否为空
      if (!parents[i].empty()) {
        // 比较栈顶元素是否相同
        c &= (temp == parents[i].top());
      } else {
        // 栈为空，不匹配
        c = 0;
        break;
      }
    }

    // 如果所有节点的栈顶都相同
    if (c) {
      // 记录公共节点
      common = temp;
      if (common == NULL) TRACE(NCCL_GRAPH, "COMMON IS NULL");
      // 所有栈都弹出栈顶，继续查找
      for (int i = 0; i < nNodes; i++) {
        parents[i].pop();
      }
    // Check multi-port while we still have the mismatched parents
    // For multi-port to be true, all parents (peers) must have the busId attribute with all but the last character matching
    // 检查是否为多端口网卡（同一NIC的不同端口）
    } else {
      // 多端口标志
      int multiPort = 1;
      // 第一个节点的总线ID
      const char* tempBusId;

      // 获取第一个节点的总线ID
      NCCLCHECK(xmlGetAttr(temp, "busid", &tempBusId));
      if (tempBusId) {
        // 检查其他节点的总线ID
        for (int i = 1; i < nNodes; i++) {
          if (!parents[i].empty()) {
            // 当前节点的总线ID
            const char* busId;
            NCCLCHECK(xmlGetAttr(parents[i].top(), "busid", &busId));
            if (busId) {
              // 比较长度
              if (strlen(busId) != strlen(tempBusId)) {
                multiPort = 0;
                break;
              }
              // 比较除最后一个字符外的所有字符（多端口差异在最后一位）
              if (strncmp(busId, tempBusId, strlen(busId)-1) != 0) {
                multiPort = 0;
                break;
              }
            } else {
              multiPort = 0;
              break;
            }
          }
        }
      } else {
        multiPort = 0;
      }

      // 如果是多端口，设置路径类型并返回
      if (multiPort) {
        *path = PATH_PORT;
        goto out;
      }
    }
  }

  // 根据公共父节点的类型确定路径类型
  if (common == NULL) {
    // 没有公共父节点，路径断开
    *path = PATH_DIS;
  } else if (strcmp(common->name,"system") == 0) {
    // 公共父节点是system，跨系统（多节点）
    *path = PATH_SYS;
  } else if (strcmp(common->name, "cpu") == 0) {
    // 公共父节点是CPU，PCI Host Bridge
    *path = PATH_PHB;
  } else if (strcmp(common->name, "nic") == 0) {
    // 公共父节点是NIC，多端口
    *path = PATH_PORT;
  } else if (strcmp(common->name, "net") == 0) {
    // 公共父节点是网络设备
    *path = PATH_PORT;
  } else if (ncclTopoCheckPix(common, nodes, nNodes)) {
    // 检查是否为PIX（PCI内部交换）
    *path = PATH_PIX;
  } else {
    // 默认：PCI外部交换（PXB）
    *path = PATH_PXB;
  }

out:
  // 找到第一个PCI父节点
  ncclFindFirstPciParent(&common);
  // 返回公共父节点
  *parent = common;
  // 释放栈数组
  free(parents);
  return ncclSuccess;
}

// 生成唯一的PCI总线ID
// 在合并多个网卡时，需要创建虚拟PCI节点，需要生成唯一的总线ID
// 参数：
//   xml - XML结构指针
//   busId - 输入输出参数，输入为种子ID，输出为生成的唯一ID
//   pciNode - 输出参数，返回创建的PCI节点
//   parent - 父节点指针
ncclResult_t ncclTopoMakeUniqueBusId(struct ncclXml* xml, char* busId, struct ncclXmlNode** pciNode, struct ncclXmlNode* parent) {
  // 尝试计数器
  int i = 0;
  // 整数形式的总线ID
  int64_t rBusId;
  // 将字符串转换为整数
  NCCLCHECK(busIdToInt64(busId, &rBusId));
  // Try to find an unused busid - NCCL expects leaf busid to be unique
  // 尝试最多100次找到一个未使用的总线ID
  while (i < 100) {
    // 递增总线ID
    rBusId++;
    TRACE(NCCL_GRAPH, "Trying to make new busId %lx", rBusId);
    // 将整数转换回字符串
    int64ToBusId(rBusId, busId);
    // 检查该总线ID是否已存在
    struct ncclXmlNode* temp = NULL;
    NCCLCHECK(xmlFindTagKv(xml, "pci", &temp, "busid", busId));
    if (temp == NULL) {
      // 总线ID未被使用，创建新的PCI节点
      NCCLCHECK(xmlAddNode(xml, parent, "pci", pciNode));
      NCCLCHECK(xmlSetAttr(*pciNode, "busid", busId));
      TRACE(NCCL_GRAPH, "Made new busId %lx", rBusId);
      return ncclSuccess;
    }
    // 总线ID已被使用，继续尝试
    TRACE(NCCL_GRAPH, "Conflicting busId %lx", rBusId);
    i++;
  }

  // 尝试100次后仍无法生成唯一ID
  WARN("TOPO/NET : Couldn't generate unique busId after %d tries", i);
  return ncclInternalError;
}

// 为虚拟网卡创建PCI父节点
// 当合并多个物理网卡时，需要创建一个虚拟PCI节点作为它们的父节点
// 参数：
//   xml - XML结构指针
//   parent - 输入输出参数，输入为原始父节点，输出为新创建的PCI父节点
//   physNetNode - 第一个物理网络节点（用于获取参考PCI信息）
ncclResult_t ncclTopoMakePciParent(struct ncclXml* xml, struct ncclXmlNode** parent, struct ncclXmlNode* physNetNode) {
  // 新PCI节点指针
  struct ncclXmlNode* newBusId = NULL;
  // 获取物理网络节点的父节点（NIC）
  struct ncclXmlNode* pci = physNetNode->parent;
  if (pci) {
    // 向上移动到PCI节点
    pci = pci->parent;
    if (pci) {
      // 检查是否为PCI节点
      if (strcmp(pci->name, "pci") == 0) {
        // 总线ID缓冲区
        char busId[NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE];
        memset(busId, 0, sizeof(busId));
        // 原始总线ID
        const char* originalBusId;
        // Seed busId with the current NIC 0's busId to make discovering a unique hash quicker
        // 使用NIC 0的总线ID作为种子，以便更快地发现唯一哈希
        NCCLCHECK(xmlGetAttrStr(pci, "busid", &originalBusId));
        snprintf(busId, sizeof(busId), "%s", originalBusId);
        // 基于种子生成唯一的总线ID并创建PCI节点
        NCCLCHECK(ncclTopoMakeUniqueBusId(xml, busId, &newBusId, *parent));
        // 复制原PCI节点的所有属性到新节点
        for (int i = 0; i < pci->nAttrs; i++) {
          NCCLCHECK(xmlSetAttr(newBusId, pci->attrs[i].key, pci->attrs[i].value));
        }
        // 设置新的总线ID
        NCCLCHECK(xmlSetAttr(newBusId, "busid", busId));
        // 更新父节点指针
        *parent = newBusId;
      }
    }
  }

  // 检查是否成功创建了新PCI节点
  if (newBusId == NULL) {
    const char* name;
    NCCLCHECK(xmlGetAttr(physNetNode, "name", &name));
    WARN("TOPO/NET : Can't find busId of child 0 %s", name);
    return ncclInternalError;
  }

  return ncclSuccess;
}

// 创建虚拟网卡（vNic）以合并多个物理网卡
// 虚拟网卡可以提供更高的聚合带宽和更好的负载均衡
// 参数：
//   xml - XML结构指针
//   netInfo - 网络信息结构指针
//   vProps - 虚拟设备属性指针（包含要合并的物理设备列表）
//   physNetNodes - 物理网络节点数组
ncclResult_t ncclTopoMakeVnic(struct ncclXml* xml, struct ncclTopoNetInfo* netInfo, ncclNetVDeviceProps_t* vProps, struct ncclXmlNode** physNetNodes) {
  // 检查要合并的设备数量是否超过最大值
  if (vProps->ndevs > NCCL_NET_MAX_DEVS_PER_NIC) {
    WARN("TOPO/NET : Tried to merge too many NICs. %d > %d", vProps->ndevs, NCCL_NET_MAX_DEVS_PER_NIC);
    return ncclInternalError;
  }

  // Don't make vNics of size 1
  // 如果只有一个设备，不需要创建虚拟网卡
  if (vProps->ndevs == 1) {
    TRACE(NCCL_GRAPH, "TOPO/NET : Skipping vNic of size 1");
    return ncclSuccess;
  }

  // Trigger the merge, then get the new device's properties
  // 调用网络插件的makeVDevice函数创建虚拟设备
  int vDevIndex = 0;
  ncclResult_t ret = netInfo->makeVDevice(&vDevIndex, vProps);
  if (ret != ncclSuccess) {
    // 合并失败，输出详细信息并返回错误
    INFO(NCCL_GRAPH|NCCL_INIT|NCCL_NET, "TOPO/NET : Tried merging multiple devices together and failed. vProps={ndevs=%d, devs=[%d %d %d %d]}. Set NCCL_NET_MERGE_LEVEL=LOC to disable NIC fusion.",
      vProps->ndevs, vProps->devs[0], vProps->devs[1], vProps->devs[2], vProps->devs[3]);
    return ret;
  }

  // Mark original NICs as keep="0" in the topology
  // 标记原始物理网卡为keep="0"，表示它们将被虚拟网卡替代
  for (int i = 0; i < vProps->ndevs; i++) {
    // 获取物理设备索引
    int dev = vProps->devs[i];
    // 获取对应的XML节点
    struct ncclXmlNode* netNode = physNetNodes[dev];
    // 设置keep属性为0（在拓扑清理时将被移除）
    NCCLCHECK(xmlSetAttrInt(netNode, "keep", 0));
  }

  // 输出成功创建虚拟网卡的信息
  INFO(NCCL_GRAPH, "TOPO/NET : Made vNic %d", vDevIndex);
  return ncclSuccess;
}

ncclResult_t ncclTopoForceMerge(struct ncclXml* xml, struct ncclTopoNetInfo* netInfo, int* placedDevs, ncclNetProperties_t* propsList, struct ncclXmlNode** physNetNodes, int nPhysDevs) {
  ncclResult_t ret = ncclSuccess;
  const char* str = netInfo->forceMerge;
  INFO(NCCL_ENV | NCCL_NET, "TOPO/NET : Force-fusing NICs using NCCL_NET_FORCE_MERGE=%s", str);
  char* ncStr;
  NCCLCHECK(ncclCalloc(&ncStr, strlen(str)+1));
  strcpy(ncStr, str);
  char* semi_token;
  char* semi = strtok_r(ncStr, ";", &semi_token);
  while (semi) {
    TRACE(NCCL_NET, "Fusing %s", semi);
    struct netIf userIfs[NCCL_NET_MAX_DEVS_PER_NIC];
    int nUserIfs = parseStringList(semi, userIfs, NCCL_NET_MAX_DEVS_PER_NIC);
    if (nUserIfs == 0) {
      INFO(NCCL_NET, "NET/IB : Invalid NCCL_NET_FORCE_MERGE specified %s. Couldn't parse substring %s. Please provide a semicolon-delimited list of comma-delimited NIC groups.",
        ncStr, semi);
      continue;
    }

    ncclNetVDeviceProps_t vProps = {0};
    for (int d = 0; d < nPhysDevs; d++) {
      if (matchIfList(propsList[d].name, propsList[d].port, userIfs, nUserIfs, 1)) {
        vProps.devs[vProps.ndevs++] = d;
      }
    }

    if (vProps.ndevs != nUserIfs) {
      WARN("TOPO/NET : Only matched %d devices, %d requested from %s",
        vProps.ndevs, nUserIfs, semi);
      ret = ncclInvalidUsage;
      goto fail;
    }

    if (vProps.ndevs > NCCL_NET_MAX_DEVS_PER_NIC) {
      WARN("Specified fused NIC %s which has too many devices (%d). Max %d", semi, vProps.ndevs, NCCL_NET_MAX_DEVS_PER_NIC);
      ret = ncclInvalidUsage;
      goto fail;
    }

    ret = ncclTopoMakeVnic(xml, netInfo, &vProps, physNetNodes);
    if (ret == ncclSuccess) {
      // Only set that a device is "placed" after successfully making a vNic (it's possible to exit before this)
      for (int i = 0; i < vProps.ndevs; i++) {
        placedDevs[vProps.devs[i]] = 1;
      }
    } else {
      WARN("TOPO/NET : Could not force merge NICs %s. Please specify a valid NCCL_NET_FORCE_MERGE string.", semi);
      ret = ncclInvalidUsage;
      goto fail;
    }

    semi = strtok_r(NULL, ";", &semi_token);;
  }

exit:
  free(ncStr);
  return ret;
fail:
  goto exit;
}

ncclResult_t ncclTopoAutoMerge(struct ncclXml* xml, struct ncclTopoNetInfo* netInfo, int* placedDevs, ncclNetProperties_t* propsList, struct ncclXmlNode** physNetNodes, int nPhysDevs) {
  // Compute the path type between each device
  int* paths = NULL;
  ncclResult_t res = ncclSuccess;
  ncclCalloc(&paths, nPhysDevs*nPhysDevs);
  TRACE(NCCL_GRAPH, "Allocated %d paths", nPhysDevs*nPhysDevs);
  for (int i = 0; i < nPhysDevs; i++) {
    for (int j = 0; j < nPhysDevs; j++) {
      struct ncclXmlNode* nodes[2];
      nodes[0] = physNetNodes[i];
      nodes[1] = physNetNodes[j];
      struct ncclXmlNode* parent;
      NCCLCHECKGOTO(ncclTopoGetPath(nodes, 2, &paths[i*nPhysDevs + j], &parent), res, out);
    }
  }

  // Place all remaining physical devices into a virtual device given the mergeLevel criteria
  for (int i = 0; i < nPhysDevs; i++) {
    // Select the first unplaced device "i" as the root
    if (placedDevs[i] == 0) {
      // Init a new vDevice
      ncclNetVDeviceProps_t vProps;
      vProps = {0};
      vProps.devs[vProps.ndevs++] = i;
      placedDevs[i] = 1;
      TRACE(NCCL_GRAPH, "Placed dev %d", i);

      // Select each unplaced device "j" which is at most "mergeLevel" distance from "i", but not equal to "i"
      // (Don't merge the same device with itself)
      for (int j = 0; j < nPhysDevs; j++) {
        if (paths[i*nPhysDevs + j] <= netInfo->mergeLevel &&
        placedDevs[j] == 0 && j != i) {
          vProps.devs[vProps.ndevs++] = j;
          placedDevs[j] = 1;
          TRACE(NCCL_GRAPH, "Placed dev %d path=%d", j, paths[i*nPhysDevs + j] );
        }
        if (vProps.ndevs == NCCL_NET_MAX_DEVS_PER_NIC) break;
      }

      if (vProps.ndevs > NCCL_NET_MAX_DEVS_PER_NIC) {
        WARN("TOPO/NET : Tried to merge too many NICs. %d > %d", vProps.ndevs, NCCL_NET_MAX_DEVS_PER_NIC);
        return ncclInternalError;
      }

      ncclResult_t ret = ncclTopoMakeVnic(xml, netInfo, &vProps, physNetNodes);

      // Merging failed.
      // Mark all as unplaced and increase their distance to disconnected (PATH_DIS)
      // Set i to 0 to restart the automatic merging process and ensure all are placed
      if (ret != ncclSuccess) {
        INFO(NCCL_GRAPH|NCCL_INIT|NCCL_NET, "Marking physical devices as unplaced, increasing distance and restarting search.");
        placedDevs[i] = 0;
        TRACE(NCCL_GRAPH, "Setting dev %d as unplaced, keeping distance -> self as PATH_LOC", i);
        for (int k = 1; k < vProps.ndevs; k++) {
          int dev = vProps.devs[k];
          placedDevs[dev] = 0;
          paths[i*nPhysDevs + dev] = PATH_DIS;
          paths[dev*nPhysDevs + i] = PATH_DIS;
          TRACE(NCCL_GRAPH, "Setting dev %d as unplaced, setting distance -> %d as PATH_DIS", dev, i);
        }
        i = 0;
      }
    }
  }

out:
  free(paths);
  return res;
}

struct kvDict nicPathKvList[] = {
  { "LOC",  PATH_LOC },
  { "PORT", PATH_PORT },
  { "PIX",  PATH_PIX },
  { "PXB",  PATH_PXB },
  { "P2C",  PATH_P2C },
  { "PXN",  PATH_PXN },
  { "PHB",  PATH_PHB },
  { "SYS",  PATH_SYS },
  { NULL, 0 }
};


ncclResult_t ncclTopoFindLinkWidthRec(ncclXmlNode* node, ncclXmlNode** physNetNodes, int ndevs, int* foundPhysNet, int* linkWidth) {
  int myLinkWidth = 0;
  if (strcmp(node->name, "pci") == 0) {
    NCCLCHECK(xmlGetAttrInt(node, "link_width", &myLinkWidth));
#ifdef ENABLE_TRACE
    const char *busidAttr, *linkAttr;
    NCCLCHECK(xmlGetAttrStr(node, "busid", &busidAttr));
    NCCLCHECK(xmlGetAttr(node, "link_width", &linkAttr));
    TRACE(NCCL_GRAPH, "Found link_width (%s)=%d for busid=%s", linkAttr, myLinkWidth, busidAttr);
#endif
  }

  *foundPhysNet = 0;
  // Detect if a physical child is found. This information will be propagated up the stack.
  int devId = 0;
  while (devId < ndevs && !(*foundPhysNet)) *foundPhysNet = (node == physNetNodes[devId++]);

  int totalChildLinkWidth = 0;
  for (int i = 0; i < node->nSubs; i++) {
    ncclXmlNode* child = node->subs[i];
    int found = 0;
    int tempLinkWidth = 0;
    NCCLCHECK(ncclTopoFindLinkWidthRec(child, physNetNodes, ndevs, &found, &tempLinkWidth));
    if (found) {
      *foundPhysNet = 1;
      totalChildLinkWidth += tempLinkWidth;
    }
  }

  if (*foundPhysNet == 0) {
    // No child NICs were found, do not accrue any detected link_width
    *linkWidth = 0;
    INFO(NCCL_GRAPH, "Did not find child net device. Returning link_width=%d totalChildLinkWidth=%d", *linkWidth, totalChildLinkWidth);
  } else if (totalChildLinkWidth == 0) {
    // If A child NIC was found but no link_width was detected among children, assign the link_width to mine (I am the first pci node right above the physNetNode).
    *linkWidth = myLinkWidth;
    INFO(NCCL_GRAPH, "Found child net device for %s. Returning link_width=%d totalChildLinkWidth=%d", node->name, *linkWidth, totalChildLinkWidth);
  } else {
  // Standard recursive accrual of link_width. The link_width is either the bottleneck of this PCI node's width or the sum of its children's width.
    *linkWidth = myLinkWidth > 0 ? std::min(myLinkWidth, totalChildLinkWidth) : totalChildLinkWidth;
    INFO(NCCL_GRAPH, "Found child net device for %s. Returning link_width=%d totalChildLinkWidth=%d", node->name, *linkWidth, totalChildLinkWidth);
  }

  return ncclSuccess;
}

// DFS over nodes under common parent
// Exclude link widths of non-physNetNodes chains
ncclResult_t ncclTopoFindLinkWidth(ncclXmlNode* parent, ncclXmlNode** physNetNodes, int ndevs, int* linkWidth) {
  *linkWidth = 0;
  for (int i = 0; i < parent->nSubs; i++) {
    ncclXmlNode* child = parent->subs[i];
    int foundPhysNet = 0;
    int childLinkWidth = 0;
    NCCLCHECK(ncclTopoFindLinkWidthRec(child, physNetNodes, ndevs, &foundPhysNet, &childLinkWidth));
    if (foundPhysNet) {
      *linkWidth += childLinkWidth;
    }
  }

  return ncclSuccess;
}

ncclResult_t ncclTopoWidenLinks(ncclXmlNode** physNetNodes, int ndevs, ncclXmlNode* parent) {
  int sumLinkWidth = 0;
  NCCLCHECK(ncclTopoFindLinkWidth(parent, physNetNodes, ndevs, &sumLinkWidth));
  for (int i = 0; i < ndevs; i++) {
    ncclXmlNode* temp = physNetNodes[i];
    while (temp != parent) {
      if (strcmp(temp->name, "pci") == 0) {
        NCCLCHECK(xmlSetAttrInt(temp, "link_width", sumLinkWidth));
        TRACE(NCCL_GRAPH, "Set link_width to %d for node %s", sumLinkWidth, temp->name);
      }
      temp = temp->parent;
    }
  }

  if (strcmp(parent->name, "pci") == 0) {
    NCCLCHECK(xmlSetAttrInt(parent, "link_width", sumLinkWidth));
    TRACE(NCCL_GRAPH, "Set link_width to %d for node %s", sumLinkWidth, parent->name);
  }

  return ncclSuccess;
}

ncclResult_t ncclTopoGetVNicParent(struct ncclXml* xml, ncclResult_t (*getProperties)(int, ncclNetProperties_t*), ncclNetVDeviceProps_t* vProps, ncclXmlNode** parent) {
  ncclNetProperties_t props[NCCL_NET_MAX_DEVS_PER_NIC];
  ncclXmlNode* physNetNodes[NCCL_NET_MAX_DEVS_PER_NIC];
  for (int i = 0; i < vProps->ndevs; i++) {
    NCCLCHECK(getProperties(vProps->devs[i], props + i));
    struct ncclXmlNode* physNetNode;
    NCCLCHECK(xmlFindTagKv(xml, "net", &physNetNode, "name", props[i].name));
    physNetNodes[i] = physNetNode;
    TRACE(NCCL_GRAPH, "Re-found physical ncclNet node %d %s", i,  props[i].name);
  }

  int path = PATH_LOC;
  NCCLCHECK(ncclTopoGetPath(physNetNodes, vProps->ndevs, &path, parent));
  if (path == PATH_PHB || path == PATH_PXB || path == PATH_PIX) {
    INFO(NCCL_GRAPH, "Widening links");
    NCCLCHECK(ncclTopoWidenLinks(physNetNodes, vProps->ndevs, *parent));
  }

  if (*parent) {
    if (strcmp((*parent)->name, "pci") == 0) {
      // Compare PCI class here to avoid NCCL WARN when the "class" attribute doesn't exist
      const char* c;
      NCCLCHECK(xmlGetAttrStr(*parent, "class", &c));
      if (c && strcmp(c, PCI_BRIDGE_DEVICE_CLASS) == 0) {
        // If the common parent is a PCI switch, we must reparent the new NIC under a made up pci device with a unique busid
        NCCLCHECK(ncclTopoMakePciParent(xml, parent, physNetNodes[0]));
      }
    } else if (strcmp((*parent)->name, "cpu") == 0) {
      // If the common parent is a PCI switch, we must reparent the new NIC under a made up pci device with a unique busid
      NCCLCHECK(ncclTopoMakePciParent(xml, parent, physNetNodes[0]));
    }
  }

  TRACE(NCCL_GRAPH, "Selected parent %s with path %d", (*parent)->name, path);
  return ncclSuccess;
}

ncclResult_t ncclTopoMakeVNics(struct ncclXml* xml, struct ncclTopoNetInfo* netInfo, int physicalDevs) {
  int* placedDevs = NULL;
  struct ncclXmlNode** physNetNodes = NULL;
  ncclNetProperties_t* props = NULL;
  ncclResult_t res = ncclSuccess;
  if (physicalDevs == 0) return ncclSuccess;

  NCCLCHECK(ncclCalloc(&physNetNodes, physicalDevs));
  NCCLCHECK(ncclCalloc(&placedDevs, physicalDevs));
  NCCLCHECK(ncclCalloc(&props, physicalDevs));
  for (int i = 0; i < physicalDevs; i++) {
    NCCLCHECKGOTO(netInfo->getProperties(i, props + i), res, out);
    struct ncclXmlNode* physNetNode;
    NCCLCHECKGOTO(xmlFindTagKv(xml, "net", &physNetNode, "name", props[i].name), res, out);
    physNetNodes[i] = physNetNode;
    TRACE(NCCL_GRAPH, "Found physical ncclNet node %d %s", i,  props[i].name);
  }

  if (netInfo->forceMerge) NCCLCHECKGOTO(ncclTopoForceMerge(xml, netInfo, placedDevs, props, physNetNodes, physicalDevs), res, out);
  NCCLCHECKGOTO(ncclTopoAutoMerge(xml, netInfo, placedDevs, props, physNetNodes, physicalDevs), res, out);

out:
  free(physNetNodes);
  free(props);
  if (placedDevs) free(placedDevs);
  return res;
}

// 填充网卡信息到XML拓扑结构
// 参数：
//   xml - XML结构指针
//   startIndex - 起始设备索引
//   endIndex - 结束设备索引
//   netInfo - 网络信息结构指针
//   virtualNics - 是否为虚拟网卡标志
static ncclResult_t ncclTopoPopulateNics(ncclXml* xml, int startIndex, int endIndex, struct ncclTopoNetInfo* netInfo, int virtualNics) {
  // 遍历指定范围的网卡
  for (int n = startIndex; n < endIndex; n++) {
    // 获取网卡属性
    ncclNetProperties_t props;
    NCCLCHECK(netInfo->getProperties(n, &props));
    // 网络节点和父节点指针
    struct ncclXmlNode* netNode = NULL;
    struct ncclXmlNode* parent = NULL;
    // 如果是虚拟网卡，需要特殊处理
    if (virtualNics) {
      // 查找是否已存在该网络节点
      struct ncclXmlNode* net = NULL;
      NCCLCHECK(xmlFindTagKv(xml, "net", &net, "name", props.name));
      // In the event of multithreaded use case, we need to re-discover the shared parent of the given devices for this vNIC
      // Only run this if the net doesn't exist locally - this may alter the XML state
      // 在多线程使用情况下，需要重新发现虚拟网卡的共享父节点
      // 只有在本地不存在网络节点时才运行此操作（可能改变XML状态）
      if (net == NULL) NCCLCHECK(ncclTopoGetVNicParent(xml, netInfo->getProperties, &props.vProps, &parent));
    }

    // 填充网络节点到XML
    NCCLCHECK(ncclTopoFillNet(xml, props.pciPath, props.name, &netNode, parent));

    // 获取集合网络属性（用于调试）
    const char* colAttr;
    NCCLCHECK(xmlGetAttr(netNode, "coll", &colAttr));

    // 设置keep属性为1（保留该节点）
    NCCLCHECK(xmlSetAttrInt(netNode, "keep", 1));
    // 检查设备索引是否正确
    int dev;
    xmlGetAttrIntDefault(netNode, "dev", &dev, -1);
    if (dev != -1 && dev != n) INFO(NCCL_GRAPH, "TOPO/NET : Changing %s dev index from %d to %d", netInfo->name, dev, n);
    // 设置设备索引
    NCCLCHECK(xmlSetAttrInt(netNode, "dev", n));
    // 设置延迟
    NCCLCHECK(xmlInitAttrInt(netNode, "latency", props.latency));
    // 设置速度（Mbps）
    NCCLCHECK(xmlInitAttrInt(netNode, "speed", props.speed));
    // 设置端口号
    NCCLCHECK(xmlInitAttrInt(netNode, "port", props.port));
    // 设置GUID
    NCCLCHECK(xmlInitAttrUint64(netNode, "guid", props.guid));
    // 设置最大连接数
    NCCLCHECK(xmlInitAttrInt(netNode, "maxconn", props.maxComms));
    //检查网卡是否支持gdr
    // GPU Direct RDMA支持检查：支持CUDA指针或支持DMA-BUF
    bool gdrSupport = (props.ptrSupport & NCCL_PTR_CUDA) || (netInfo->dmaBufSupport && (props.ptrSupport & NCCL_PTR_DMABUF));
    INFO(NCCL_NET,"NET/%s : GPU Direct RDMA %s for HCA %d '%s'", netInfo->name, gdrSupport ? "Enabled" : "Disabled", n, props.name);
    NCCLCHECK(xmlInitAttrInt(netNode, "gdr", gdrSupport));

    // Only set coll if it's not 0
    // 如果支持集合网络，设置coll属性
    if (netInfo->coll)
        NCCLCHECK(xmlInitAttrInt(netNode, "coll", netInfo->coll));

    // 输出调试信息
    const char* keepAttr;
    NCCLCHECK(xmlGetAttr(netNode, "coll", &colAttr));
    NCCLCHECK(xmlGetAttr(netNode, "keep", &keepAttr));
    INFO(NCCL_GRAPH, "ncclTopoPopulateNics : Filled %s in topo with pciPath=%s keep=%s coll=%s",
      props.name, props.pciPath, keepAttr, colAttr);
  }

  return ncclSuccess;
}

// Calls to network plugin APIs should be protected. This function should be called inside a per-process lock.
// 处理网络插件，将网络设备添加到拓扑系统
// 对网络插件API的调用应该被保护，此函数应该在进程级锁内调用
// 参数：
//   xml - XML结构指针
//   dumpXmlFile - XML转储文件路径（如果非NULL，则不创建虚拟设备）
//   net - 网络信息结构指针
ncclResult_t ncclTopoProcessNet(ncclXml* xml, const char* dumpXmlFile, struct ncclTopoNetInfo* net) {
  // 确定是否使用物理设备（转储模式或插件不支持虚拟设备）
  bool usePhysicalDevices = (dumpXmlFile || net->makeVDevice == NULL);
  // 物理和虚拟网卡数量
  int nPhysicalNics, nVirtualNics;
  NCCLCHECK(net->getDevCount(net->netPluginIndex, &nPhysicalNics, &nVirtualNics));
  // List the physical devices in the topo
  //添加nic信息到xml中
  // 首先添加所有物理网卡
  NCCLCHECK(ncclTopoPopulateNics(xml, 0, nPhysicalNics, net, /*virtual=*/false));

  if (!usePhysicalDevices) {
    // Virtual devices are only created once per network
    // 每个网络只创建一次虚拟设备
    if (nVirtualNics == NCCL_UNDEF_DEV_COUNT) {
      NCCLCHECK(ncclTopoMakeVNics(xml, net, nPhysicalNics));
      // Update the number of virtual devices both locally and in the state tracking the plugin.
      // Note: 0 is a valid number of virtual devices
      int nDevs;
      NCCLCHECK(net->devices(&nDevs));
      nVirtualNics = nDevs - nPhysicalNics;
      NCCLCHECK(net->setVirtDevCount(net->netPluginIndex, nVirtualNics));
    }
    // populate the virtual devices if any
    if (nVirtualNics > 0) {
      NCCLCHECK(ncclTopoPopulateNics(xml, nPhysicalNics, nPhysicalNics + nVirtualNics, net, /*virtual=*/true));
    }
  }

  return ncclSuccess;
}

// 获取网卡融合相关的环境变量
// 参数：
//   mergeLevel - 输出参数，返回合并级别（路径类型）
//   forceMerge - 输出参数，返回强制合并字符串
ncclResult_t ncclTopoGetFusionEnv(int* mergeLevel, const char** forceMerge) {
  // 获取强制合并环境变量（NCCL_NET_FORCE_MERGE）
  if (forceMerge) *forceMerge = ncclGetEnv("NCCL_NET_FORCE_MERGE");
  // 获取合并级别环境变量（NCCL_NET_MERGE_LEVEL）
  const char* mergeLevelEnv = ncclGetEnv("NCCL_NET_MERGE_LEVEL");
  if (mergeLevelEnv) {
    // 将环境变量字符串转换为路径类型
    kvConvertToInt(mergeLevelEnv, mergeLevel, nicPathKvList);
  } else {
    // 默认合并级别为PORT（多端口）
    *mergeLevel = PATH_PORT;
  }
  return ncclSuccess;
}

// 网络互斥锁，保护网络插件API调用
static std::mutex netMutex;

// 获取拓扑系统
// 这是NCCL拓扑发现的主入口函数，负责构建整个系统拓扑
// 参数：
//   comm - 通信域指针
//   system - 输出参数，返回创建的拓扑系统
//   dumpXmlFile - XML转储文件路径（用于调试）
ncclResult_t ncclTopoGetSystem(struct ncclComm* comm, struct ncclTopoSystem** system, const char* dumpXmlFile) {
  // 返回值
  ncclResult_t ret = ncclSuccess;
  // XML结构指针
  struct ncclXml* xml;
  // 内存缓冲区（用于AllGather）
  char* mem = NULL;
  // 本地rank数组
  int* localRanks = NULL;
  // rank的XML结构
  struct ncclXml* rankXml;
  // 本地rank和本地rank数量
  int localRank = -1, nLocalRanks = 0;
  // 网络信息结构
  struct ncclTopoNetInfo netInfo = {0};
  //分配xml结构
  NCCLCHECK(xmlAlloc(&xml, NCCL_TOPO_XML_MAX_NODES));

  // 获取拓扑文件环境变量
  const char* xmlTopoFile = ncclGetEnv("NCCL_TOPO_FILE");
  if (xmlTopoFile) {
    //如果定义了拓扑xml文件，则从文件构建拓扑
    INFO(NCCL_ENV, "NCCL_TOPO_FILE set by environment to %s", xmlTopoFile);
    NCCLCHECKGOTO(ncclTopoGetXmlFromFile(xmlTopoFile, xml, 1), ret, fail);
  } else {
    // Try default XML topology location
    // 尝试从默认位置读取拓扑文件（不强制要求）
    NCCLCHECKGOTO(ncclTopoGetXmlFromFile("/var/run/nvidia-topologyd/virtualTopology.xml", xml, 0), ret, fail);
  }

  // Fixup the cpu's host_hashes.
  // 更新CPU节点的host_hash属性（这些属性不应从文件中保留）
  struct ncclXmlNode* node;
  // Update every cpu node's host_hash attribute since those are not
  // intended to be preserved from the XML files that have been read.
  NCCLCHECKGOTO(xmlFindTag(xml, "cpu", &node), ret, fail);
  //先不考虑通过加载文件的方式生成xml数据
  //则node为NULL
  while (node != nullptr) {
    NCCLCHECKGOTO(xmlSetAttrLong(node, "host_hash", getHostHash()), ret, fail);
    NCCLCHECKGOTO(xmlFindNextTag(xml, "cpu", node, &node), ret, fail);
  }

  //默认为0
  if (xml->maxIndex == 0) {
    // Create top tag
    struct ncclXmlNode* top;
    //添加一个system node
    //<system version="1">
    NCCLCHECKGOTO(xmlAddNode(xml, NULL, "system", &top), ret, fail);
    NCCLCHECKGOTO(xmlSetAttrInt(top, "version", NCCL_TOPO_XML_VERSION), ret, fail);
  }

  NCCLCHECKGOTO(ncclTopoRefreshBcmP2pLinks(), ret, fail);

  // Detect only the GPU managed by this process.  We'll get any others through XML fusion.
  char busId[NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE];
  //把自己gpu的busid转换为字符串
  NCCLCHECKGOTO(int64ToBusId(comm->peerInfo[comm->rank].busId, busId), ret, fail);
  //填充gpu信息到xml树中，比如pci busid等
  //比如：<pci busid="0000:28:00.0" class="0x030200" vendor="0x10de" device="0x2335" 
  //subsystem_vendor="0x10de" subsystem_device="0x18be" link_speed="32.0 GT/s PCIe" link_width="16">
  NCCLCHECKGOTO(ncclTopoFillGpu(xml, busId, &node), ret, fail);
  //<gpu dev="2" sm="90" rank="2" gdr="1">
  if (node) {
    //往gpunode补充信息，本gpu信息要暴力
    NCCLCHECKGOTO(xmlSetAttrInt(node, "keep", 1), ret, fail);
    //自己的rank号
    NCCLCHECKGOTO(xmlSetAttrInt(node, "rank", comm->rank), ret, fail);
    NCCLCHECKGOTO(xmlInitAttrInt(node, "gdr", comm->peerInfo[comm->rank].gdrSupport), ret, fail);
  }

  // Auto-detect NICs if needed. net/collnet share the same xml/graph nodes,
  // so we start with collnet so that it has precedence.
  {
      std::lock_guard<std::mutex> lock(netMutex);
      INFO(NCCL_GRAPH, "TOPO/NET : Importing network plugins to topology");
      if (collNetSupport(comm)) {
        netInfo.coll = 1;
        netInfo.netPluginIndex = comm->netPluginIndex;
        netInfo.dmaBufSupport = comm->dmaBufSupport;
        netInfo.getDevCount = ncclCollNetGetDevCount;
        netInfo.setVirtDevCount = ncclCollNetSetVirtDevCount;
        netInfo.name = comm->ncclCollNet->name;
        netInfo.getProperties = comm->ncclCollNet->getProperties;
        netInfo.makeVDevice = comm->ncclCollNet->makeVDevice;
        netInfo.devices = comm->ncclCollNet->devices;
        NCCLCHECK(ncclTopoGetFusionEnv(&netInfo.mergeLevel, &netInfo.forceMerge));
        NCCLCHECKGOTO(ncclTopoProcessNet(xml, dumpXmlFile, &netInfo), ret, fail);
      }

      netInfo.coll = 0;
      netInfo.netPluginIndex = comm->netPluginIndex;
      netInfo.dmaBufSupport = comm->dmaBufSupport;
      netInfo.getDevCount = ncclNetGetDevCount;
      netInfo.setVirtDevCount = ncclNetSetVirtDevCount;
      netInfo.name = comm->ncclNet->name;
      netInfo.getProperties = comm->ncclNet->getProperties;
      netInfo.makeVDevice = comm->ncclNet->makeVDevice;
      netInfo.devices = comm->ncclNet->devices;
      NCCLCHECK(ncclTopoGetFusionEnv(&netInfo.mergeLevel, &netInfo.forceMerge));
      //添加nic信息到xml中
      // <nic>
      //  <net name="mlx5_1" dev="0" latency="0" speed="100000" port="1" guid="0xac6bba0003739de0" maxconn="131072" gdr="1"/>
      //</nic>
      NCCLCHECKGOTO(ncclTopoProcessNet(xml, dumpXmlFile, &netInfo), ret, fail);
  }

  // Remove XML branches which don't have a node with keep="1" (typically when importing a topology)
  //移除keep不为1的node
  NCCLCHECKGOTO(ncclTopoTrimXml(xml), ret, fail);

  // XML topo fusion.
  if (comm->MNNVL) {
    // MNNVL clique support
    nLocalRanks = comm->clique.size;
    localRank = comm->cliqueRank;
    localRanks = comm->clique.ranks;
  } else {
    // Intra-node fusion.  Much of the comm is not initialized yet at this point so we need to do our own calculations.
    //记录当前节点内的localrank信息
    NCCLCHECKGOTO(ncclCalloc(&localRanks, comm->nRanks), ret, fail);
    for (int i = 0; i < comm->nRanks; i++) {
        //在同一个节点内，nLocalRanks统计当前节点的rank数量
      if (comm->peerInfo[i].hostHash == comm->peerInfo[comm->rank].hostHash) {
        if (i == comm->rank)
          localRank = nLocalRanks;
        localRanks[nLocalRanks++] = i;
      }
    }
  }
 
  NCCLCHECKGOTO(ncclCalloc(&mem, nLocalRanks * xmlMemSize(NCCL_TOPO_XML_MAX_NODES)), ret, fail);
  rankXml = (struct ncclXml*)(mem+xmlMemSize(NCCL_TOPO_XML_MAX_NODES)*localRank);
  //拷贝当前节点的xml信息到rankXml，同步给节点内的rank
  memcpy(rankXml, xml, xmlMemSize(NCCL_TOPO_XML_MAX_NODES));
  //把指针转换为偏移值，因为指针值在其他rank上是无效的
  NCCLCHECKGOTO(ncclTopoConvertXml(rankXml, (uintptr_t)xml->nodes, 1), ret, fail);
  
  // nLocalRanks can't actually be 0, or we wouldn't be running at all...
  // coverity[divide_by_zero]
  //聚合同一个节点内的ncclXml信息
  //  在一个 8 GPU 的节点上，每个进程只负责一个 GPU。通过融合：
  ///- Rank 0 看到完整的 GPU0 信息
  //- Rank 1 看到完整的 GPU1 信息
 // - 融合后，所有 rank 都能看到节点内全部 8 个 GPU 的拓扑信息
  NCCLCHECKGOTO(bootstrapIntraNodeAllGather(comm->bootstrap, localRanks, localRank, nLocalRanks, mem, xmlMemSize(NCCL_TOPO_XML_MAX_NODES)), ret, fail);
  if (comm->MNNVL) {
    // Ensure that we have enough room when fusing topos from multiple nodes.
    free(xml);
    xml = NULL;
    NCCLCHECKGOTO(xmlAlloc(&xml, nLocalRanks*NCCL_TOPO_XML_MAX_NODES), ret, fail);
  } else {
    // In the intra-node case there's no need to enlarge the topo xml.
    xml->maxIndex = 0;
  }

  //把偏移值还原成指针值
  for (int i = 0; i < nLocalRanks; i++) {
    struct ncclXml* peerXml = (struct ncclXml*)(mem+xmlMemSize(NCCL_TOPO_XML_MAX_NODES)*i);
    NCCLCHECKGOTO(ncclTopoConvertXml(peerXml, (uintptr_t)peerXml->nodes, 0), ret, fail);
    NCCLCHECKGOTO(ncclTopoFuseXml(xml, peerXml), ret, fail);
  }

  if (dumpXmlFile && comm->rank == ncclParamTopoDumpFileRank()) {
    INFO(NCCL_ENV, "NCCL_TOPO_DUMP_FILE set by environment to %s", dumpXmlFile);
    NCCLCHECKGOTO(ncclTopoDumpXmlToFile(dumpXmlFile, xml), ret, fail);
  }

  // Only update our topo tracking structure if we aren't dumping (separate steps)
  //不是dump模式，把 XML 拓扑转化加载到内存system结构里
  if (dumpXmlFile == NULL) 
    NCCLCHECKGOTO(ncclTopoGetSystemFromXml(xml, system, getHostHash()), ret, fail);

exit:
  if (!comm->MNNVL && localRanks) 
    free(localRanks);
  if (mem) 
    free(mem);
  free(xml);
  return ret;
fail:
  goto exit;
}

// 获取本地节点列表（具有最佳路径的节点）
// 根据带宽和路径类型选择最优的本地节点
// 参数：
//   system - 拓扑系统指针
//   type - 源节点类型
//   index - 源节点在该类型数组中的索引
//   resultType - 目标节点类型
//   locals - 输出参数，返回找到的本地节点索引数组
//   localCount - 输出参数，返回找到的节点数量
//   pathType - 输出参数，返回路径类型
ncclResult_t ncclTopoGetLocal(struct ncclTopoSystem* system, int type, int index, int resultType,
                                     int locals[NCCL_TOPO_MAX_NODES], int* localCount, int* pathType) {
  // 最小路径类型（越大距离越远）
  int minType = PATH_DIS;
  // 最大带宽
  float maxBw = 0;
  // 找到的节点计数
  int count = 0;
  // 获取源节点到所有目标类型节点的路径
  struct ncclTopoLinkList* paths = system->nodes[type].nodes[index].paths[resultType];
  // 如果路径不存在，返回0
  if (paths == NULL) { *localCount = 0; return ncclSuccess; }
  // 遍历所有目标节点
  for (int i=0; i<system->nodes[resultType].count; i++) {
    // 如果找到更高带宽的路径，或者相同带宽但路径类型更优（距离更近）
    if (paths[i].bw > maxBw || (paths[i].bw == maxBw && paths[i].type < minType)) {
      // 更新最大带宽和最小路径类型
      maxBw = paths[i].bw;
      minType = paths[i].type;
      // 返回路径类型
      if (pathType) *pathType = minType;
      // 重置计数
      count = 0;
    }
    // 如果路径带宽和类型都与最优值匹配，添加到结果列表
    if (paths[i].bw == maxBw && paths[i].type == minType) {
      // 检查结果数组是否已满
      if (count == NCCL_TOPO_MAX_NODES) {
        WARN("Error : ran out of room to store found nodes in ncclTopoGetLocal."
             " Filled %d of type %d, starting from index %d of type %d.",
             NCCL_TOPO_MAX_NODES, resultType, index, type);
        return ncclInternalError;
      }
      // 添加节点索引到结果列表
      locals[count++] = i;
    }
  }
  // 返回找到的节点数量
  *localCount = count;
  return ncclSuccess;
}

// 根据带宽获取本地网卡数量
// 计算需要多少个网卡才能达到GPU的带宽
// 参数：
//   system - 拓扑系统指针
//   gpu - GPU节点索引
//   count - 输出参数，返回需要的网卡数量
ncclResult_t getLocalNetCountByBw(struct ncclTopoSystem* system, int gpu, int *count) {
  // 本地网卡总数和按带宽计算的网卡数
  int localNetCount = 0, netCountByBw = 0;
  // 本地网卡索引数组
  int localNets[NCCL_TOPO_MAX_NODES];
  // 总网卡带宽和GPU带宽
  float totalNetBw = 0, gpuBw = 0;

  // 遍历GPU的所有链路，找到GPU到CPU的带宽
  for (int l=0; l<system->nodes[GPU].nodes[gpu].nlinks; l++) {
    //assuming BW to CPU reflects the GPU bandwidth via P2P or C2C
    // 假定到CPU的带宽反映了GPU通过P2P或C2C的带宽
    //caveat, this could be wrong if there is a PCIe switch,
    //and a narrower link to the CPU
    // 注意：如果存在PCI交换机且到CPU的链路更窄，这可能不准确
    if (system->nodes[GPU].nodes[gpu].links[l].remNode->type == CPU) {
       // 获取GPU带宽
       gpuBw = system->nodes[GPU].nodes[gpu].links[l].bw;
    }
  }

  // 获取本地网卡列表
  NCCLCHECK(ncclTopoGetLocal(system, GPU, gpu, NET, localNets, &localNetCount, NULL));
  // 计算需要多少个网卡才能达到GPU带宽
  for (int l=0; (l < localNetCount) && (totalNetBw < gpuBw); l++, netCountByBw++) {
     // 累加网卡带宽
     totalNetBw += system->nodes[GPU].nodes[gpu].paths[NET][localNets[l]].bw;
  }
  // 返回需要的网卡数量
  *count = netCountByBw;

  return ncclSuccess;
}

// 网络设备选择策略枚举
enum netDevsPolicy {
  NETDEVS_POLICY_AUTO = 0x0,       // 自动策略（根据拓扑选择）
  NETDEVS_POLICY_ALL = 0x1,        // 使用所有可用网卡
  NETDEVS_POLICY_MAX = 0x2,        // 使用指定数量的网卡
  NETDEVS_POLICY_UNDEF = 0xffffffff // 未定义
};

// 网络设备选择策略（全局变量）
static enum netDevsPolicy netDevsPolicy = NETDEVS_POLICY_UNDEF;
// MAX策略的最大设备数
static int netDevsPolicyNum = -1;

// 一次性初始化网络设备选择策略（从环境变量读取）
static void getNetDevsPolicyOnce() {
  // 获取环境变量NCCL_NETDEVS_POLICY
  const char* envStr = ncclGetEnv("NCCL_NETDEVS_POLICY");
  if (envStr) {
    // 解析策略字符串
    if (strcasecmp(envStr, "AUTO") == 0) {
      // 自动策略
      netDevsPolicy = NETDEVS_POLICY_AUTO;
    } else if (strcasecmp(envStr, "ALL") == 0) {
      // 使用所有网卡
      netDevsPolicy = NETDEVS_POLICY_ALL;
    } else if (strncasecmp(envStr, "MAX:", strlen("MAX:")) == 0) {
      // 使用指定数量的网卡（格式：MAX:n）
      int envNum = atoi(envStr + strlen("MAX:"));
      if (envNum > 0) {
        netDevsPolicy = NETDEVS_POLICY_MAX;
        netDevsPolicyNum = envNum;
      }
    }
    // 检查是否成功识别策略
    if (netDevsPolicy == NETDEVS_POLICY_UNDEF)
      INFO(NCCL_ENV, "Unable to recognize NCCL_NETDEVS_POLICY=%s, using NCCL_NETDEVS_POLICY_AUTO instead.", envStr);
    else
      INFO(NCCL_ENV, "NCCL_NETDEVS_POLICY set by environment to %s", envStr);
  }
  // 如果未设置环境变量，使用默认的AUTO策略
  if (netDevsPolicy == NETDEVS_POLICY_UNDEF) netDevsPolicy = NETDEVS_POLICY_AUTO;
}

// 获取指定rank的本地网络设备
// 根据策略选择每个GPU使用的网卡
// 参数：
//   system - 拓扑系统指针
//   rank - GPU的rank号
//   channelId - 通道ID（用于选择不同的网卡）
//   id - 输出参数，返回网卡节点ID
//   dev - 输出参数，返回网卡设备索引
ncclResult_t ncclTopoGetLocalNet(struct ncclTopoSystem* system, int rank, int channelId, int64_t* id, int* dev) {
  // 将rank转换为GPU索引
  int gpu;
  NCCLCHECK(ncclTopoRankToIndex(system, rank, &gpu, /*showWarn=*/true));

  // 本地网卡索引数组
  int localNets[NCCL_TOPO_MAX_NODES];
  // 本地网卡数量
  int localNetCount;
  // 获取本地网卡列表
  NCCLCHECK(ncclTopoGetLocal(system, GPU, gpu, NET, localNets, &localNetCount, NULL));
  // 检查是否找到本地网卡
  if (localNetCount==0) {
    WARN("Could not find any local path from gpu %d to net.", gpu);
    return ncclInternalError;
  }

  // 确保策略初始化只执行一次
  static pthread_once_t once = PTHREAD_ONCE_INIT;
  pthread_once(&once,getNetDevsPolicyOnce);
  // 每个GPU分配的网卡数量
  int netsPerGpu = 0;
  if (netDevsPolicy == NETDEVS_POLICY_AUTO) {
    // AUTO策略：根据本地GPU和网卡数量自动分配
    int localGpus[NCCL_TOPO_MAX_NODES];
    int localGpuCount;
    // 获取从第一个网卡可访问的GPU列表
    NCCLCHECK(ncclTopoGetLocal(system, NET, localNets[0], GPU, localGpus, &localGpuCount, NULL));
    // 平均分配：网卡数 / GPU数（向上取整）
    netsPerGpu = DIVUP(localNetCount, localGpuCount);
  } else if (netDevsPolicy == NETDEVS_POLICY_ALL) {
    // ALL策略：每个GPU使用所有网卡
    netsPerGpu = localNetCount;
  } else if (netDevsPolicy == NETDEVS_POLICY_MAX) {
    // MAX策略：使用指定数量的网卡
    if (netDevsPolicyNum <= 0) {
      WARN("Invalid number of network devices = %d for policy MAX", netDevsPolicyNum);
      return ncclInternalError;
    }
    // 取最小值（不超过可用网卡数）
    netsPerGpu = std::min(netDevsPolicyNum, localNetCount);
  } else {
    // 未知策略
    WARN("Unknown netDevs policy");
    return ncclInternalError;
  }

  // 计算网卡索引
  // 使用GPU设备号作为基础
  int net = system->nodes[GPU].nodes[gpu].gpu.dev;
  // 如果网卡数量是2的幂，使用位反转以获得更好的分布
  if (isPow2(localNetCount)) net = mirrorBits(net, localNetCount);
  // 加上通道偏移，实现轮询分配
  net += channelId%(netsPerGpu);
  // 返回网卡ID和设备索引
  if (id) *id = system->nodes[NET].nodes[localNets[net%localNetCount]].id;
  if (dev) *dev = system->nodes[NET].nodes[localNets[net%localNetCount]].net.dev;
  return ncclSuccess;
}

// 根据网卡ID查找本地GPU
// 参数：
//   system - 拓扑系统指针
//   netId - 网卡节点ID
//   gpuIndex - 输出参数，返回GPU索引
ncclResult_t ncclTopoGetLocalGpu(struct ncclTopoSystem* system, int64_t netId, int* gpuIndex) {
  // 返回值
  ncclResult_t ret = ncclSuccess;
  // 网卡节点索引
  int netIndex;
  // 将网卡ID转换为索引
  NCCLCHECK(ncclTopoIdToIndex(system, NET, netId, &netIndex));

  // 本地GPU索引数组
  // 本地GPU索引数组
  int localGpus[NCCL_TOPO_MAX_NODES];
  // 本地GPU数量
  int localGpuCount;
  // 获取从该网卡可访问的GPU列表
  NCCLCHECK(ncclTopoGetLocal(system, NET, netIndex, GPU, localGpus, &localGpuCount, NULL));

  // 找到的GPU索引
  int foundGpu = -1;
  // 遍历所有通道，查找使用该网卡的GPU
  for (int c=0; c<MAXCHANNELS; c++) {
    // 遍历所有本地GPU
    for (int lg=0; lg<localGpuCount; lg++) {
      // 获取GPU索引
      int g = localGpus[lg];
      // 获取GPU节点
      struct ncclTopoNode* gpu = system->nodes[GPU].nodes+g;
      // 网卡ID变量
      int64_t id;
      // 获取该GPU在该通道使用的网卡
      NCCLCHECK(ncclTopoGetLocalNet(system, gpu->gpu.rank, c, &id, NULL));
      // 如果匹配，找到目标GPU
      if (netId == id) {
        foundGpu = g;
        goto exit;
      }
    }
  }
exit:
  // 返回找到的GPU索引（-1表示未找到）
  *gpuIndex = foundGpu;
  return ret;
}

/****************************/
/* External query functions */
/****************************/

// 获取CPU类型信息
// 参数：
//   system - 拓扑系统指针
//   arch - 输出参数，返回CPU架构
//   vendor - 输出参数，返回CPU厂商
//   model - 输出参数，返回CPU型号
ncclResult_t ncclTopoCpuType(struct ncclTopoSystem* system, int* arch, int* vendor, int* model) {
  // 获取第一个CPU节点的信息
  *arch = system->nodes[CPU].nodes[0].cpu.arch;
  *vendor = system->nodes[CPU].nodes[0].cpu.vendor;
  *model = system->nodes[CPU].nodes[0].cpu.model;
  return ncclSuccess;
}

// 环境变量参数：是否忽略CPU亲和性
NCCL_PARAM(IgnoreCpuAffinity, "IGNORE_CPU_AFFINITY", 0);

// 获取CPU亲和性
// 计算进程应该绑定到哪些CPU核心上，以获得最佳性能
// 参数：
//   system - 拓扑系统指针
//   rank - GPU的rank号
//   affinity - 输出参数，返回CPU亲和性掩码
ncclResult_t ncclTopoGetCpuAffinity(struct ncclTopoSystem* system, int rank, cpu_set_t* affinity) {
  // CPU和GPU节点指针
  struct ncclTopoNode* cpu = NULL, *gpu = NULL;
  // GPU和CPU索引
  int gpuIndex, cpuIndex;
  // 将rank转换为GPU索引
  NCCLCHECK(ncclTopoRankToIndex(system, rank, &gpuIndex, /*showWarn=*/true));
  // 获取本地CPU节点
  NCCLCHECK(ncclGetLocalCpu(system, gpuIndex, &cpuIndex));
  // 获取GPU和CPU节点
  gpu = system->nodes[GPU].nodes+gpuIndex;
  cpu = system->nodes[CPU].nodes+cpuIndex;

  // Query the CPU affinity set we were provided
  // 查询当前进程的CPU亲和性
  cpu_set_t mask;
  SYSCHECK(sched_getaffinity(0, sizeof(cpu_set_t), &mask), "sched_getaffinity");

  // Get the affinity of the CPU close to our GPU.
  // 获取靠近GPU的CPU的亲和性
  cpu_set_t cpuMask = cpu->cpu.affinity;

  // Get the final affinity
  // 计算最终的亲和性掩码
  cpu_set_t finalMask;
  if (ncclParamIgnoreCpuAffinity())
    // Ignore the CPU affinity set and use the GPU one instead
    // 忽略CPU亲和性设置，使用GPU的亲和性
    finalMask = cpuMask;
  else
    // Use a subset of the GPU affinity set
    // 使用进程亲和性和GPU亲和性的交集
    CPU_AND(&finalMask, &mask, &cpuMask);

  // 复制最终亲和性到输出参数
  memcpy(affinity, &finalMask, sizeof(cpu_set_t));

  // display the final affinity
  // 输出调试信息，显示最终的亲和性设置
  char msg[1024] = "";
  snprintf(msg + strlen(msg), sizeof(msg) - strlen(msg), "Affinity for GPU %d is ", gpu->gpu.dev);
  if (CPU_COUNT(&finalMask)) {
    (void)ncclCpusetToRangeStr(&finalMask, msg + strlen(msg), sizeof(msg) - strlen(msg));
  } else {
    snprintf(msg + strlen(msg), sizeof(msg) - strlen(msg), "empty, ignoring");
  }
  snprintf(msg + strlen(msg), sizeof(msg) - strlen(msg), ". (GPU affinity = ");
  (void)ncclCpusetToRangeStr(&cpuMask, msg + strlen(msg), sizeof(msg) - strlen(msg));
  if (!ncclParamIgnoreCpuAffinity()) {
    snprintf(msg + strlen(msg), sizeof(msg) - strlen(msg), " ; CPU affinity = ");
    (void)ncclCpusetToRangeStr(&mask, msg + strlen(msg), sizeof(msg) - strlen(msg));
  }
  snprintf(msg + strlen(msg), sizeof(msg) - strlen(msg), ").");
  INFO(NCCL_INIT, "%s: %s", __func__, msg);
  return ncclSuccess;
}

// 获取GPU数量
// 参数：
//   system - 拓扑系统指针
//   count - 输出参数，返回GPU数量
ncclResult_t ncclTopoGetGpuCount(struct ncclTopoSystem* system, int* count) {
  *count = system->nodes[GPU].count;
  return ncclSuccess;
}

// 获取网络设备数量
// 参数：
//   system - 拓扑系统指针
//   count - 输出参数，返回网络设备数量
ncclResult_t ncclTopoGetNetCount(struct ncclTopoSystem* system, int* count) {
  *count = system->nodes[NET].count;
  return ncclSuccess;
}

// 获取NVSwitch数量
// 参数：
//   system - 拓扑系统指针
//   count - 输出参数，返回NVSwitch数量
ncclResult_t ncclTopoGetNvsCount(struct ncclTopoSystem* system, int* count) {
  *count = system->nodes[NVS].count;
  return ncclSuccess;
}

// 获取GPU计算能力的范围
// 参数：
//   system - 拓扑系统指针
//   ccMin - 输出参数，返回最小计算能力
//   ccMax - 输出参数，返回最大计算能力
ncclResult_t ncclTopoGetCompCap(struct ncclTopoSystem* system, int* ccMin, int* ccMax) {
  // 检查是否有GPU
  if (system->nodes[GPU].count == 0) return ncclInternalError;
  // 最小和最大计算能力
  int min, max;
  // 初始化为第一个GPU的计算能力
  min = max = system->nodes[GPU].nodes[0].gpu.cudaCompCap;
  // 遍历所有GPU，找出最小和最大计算能力
  for (int g=1; g<system->nodes[GPU].count; g++) {
    min = std::min(min, system->nodes[GPU].nodes[g].gpu.cudaCompCap);
    max = std::max(max, system->nodes[GPU].nodes[g].gpu.cudaCompCap);
  }
  // 返回结果（如果指针不为NULL）
  if (ccMin)
    *ccMin = min;
  if (ccMax)
    *ccMax = max;
  return ncclSuccess;
}
