/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

//================================================================================
// 文件名称: net_ib.cc
// 功能描述: NCCL InfiniBand/RoCE 网络传输层实现（带详细中文注释）
//
// 本文件实现了 NCCL 的 IB (InfiniBand) 网络传输后端，支持：
// 1. InfiniBand (IB) 协议栈 - 通过 verbs API 进行高性能 RDMA 通信
// 2. RoCE (RDMA over Converged Ethernet) - RoCE v1 和 v2
// 3. GPU Direct RDMA (GDR) - 支持 nvidia_peermem 和 DMA-BUF 两种模式
// 4. 多 QP (Queue Pair) 支持 - 可配置每连接的 QP 数量
// 5. 多设备虚拟化 - 将多个物理 NIC 合并为一个虚拟设备
// 6. 自适应路由 - 根据 IB 网络拓扑动态优化路由
//
// 主要数据结构：
// - ncclIbDev: 单个 IB 设备的完整状态
// - ncclIbSendComm: 发送通信端点
// - ncclIbRecvComm: 接收通信端点
// - ncclIbRequest: 异步操作请求（发送/接收/刷新）
// - ncclIbQp: Queue Pair 封装
//
// 关键概念：
// - QP (Queue Pair): IB 通信的基本单元，包含发送队列(SQ)和接收队列(RQ)
// - CQ (Completion Queue): 完成队列，用于接收操作完成通知
// - MR (Memory Region): 内存区域，需要注册后才能用于 RDMA 操作
// - RDMA: 远程直接内存访问，零拷贝网络通信
// - FIFO: 先入先出队列，用于接收端通知发送端已准备好接收
//================================================================================

#include "nccl.h"
#include "core.h"
#include "socket.h"
#include "net.h"
#include "graph.h"
#include "utils.h"
#include "param.h"
#include "profiler/net_ib.h"

#include <assert.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <poll.h>
#include <sys/types.h>
#include <unistd.h>
#include <mutex>
#define ENABLE_TIMER 0
#include "timer.h"

#include "ibvwrap.h"
#include "mlx5/mlx5dvwrap.h"

//================================================================================
// 常量定义和全局变量
//================================================================================

// 最大接口名称后缀长度（用于多端口设备）
#define MAXSUFFIXSIZE 16
// 最大设备名称长度
#define MAXNAMESIZE (64 + MAXSUFFIXSIZE)

// IB 接口名称和地址（用于带外通信）
static char ncclIbIfName[MAX_IF_NAME_SIZE+1];
static union ncclSocketAddress ncclIbIfAddr;

// IB 通信配置上下文
static ncclNetCommConfig_t ibContext;

//================================================================================
// 内存注册缓存结构
//================================================================================

// 单个内存注册条目
struct ncclIbMr {
  uintptr_t addr;      // 内存区域起始地址（页对齐）
  size_t pages;        // 内存区域大小（以页为单位）
  int refs;            // 引用计数（用于共享内存区域）
  ibv_mr *mr;          // IB verbs 内存区域句柄
};

// 内存注册缓存（每个设备一个）
struct ncclIbMrCache {
  struct ncclIbMr *slots;  // 缓存槽位数组
  int capacity;            // 缓存容量
  int population;          // 当前条目数量
};

//================================================================================
// 设备相关全局变量和结构
//================================================================================

// 实际检测到的 IB 设备数量
static int ncclNIbDevs = -1;

// 每个 NIC 最多支持的设备数（用于虚拟设备）
#define NCCL_IB_MAX_DEVS_PER_NIC 4
// 合并设备名称的最大长度
#define MAX_MERGED_DEV_NAME (MAXNAMESIZE*NCCL_IB_MAX_DEVS_PER_NIC)+NCCL_IB_MAX_DEVS_PER_NIC

// 合并的虚拟设备结构
// 将多个物理 IB 设备合并为一个逻辑设备，以提高带宽和容错能力
struct alignas(64) ncclIbMergedDev {
  ncclNetVDeviceProps_t vProps;  // 虚拟设备属性
  int speed;                      // 总速度（所有子设备速度之和）
  char devName[MAX_MERGED_DEV_NAME]; // 设备名称（如 "mlx5_0+mlx5_1"）
};

//================================================================================
// 统计和错误处理结构
//================================================================================

// IB 设备统计信息
struct ncclIbStats {
  int fatalErrorCount;  // 致命错误计数器（原子操作）
};

//================================================================================
// IB 提供商枚举
//================================================================================

// IB 设备提供商类型
enum ncclIbProvider {
  IB_PROVIDER_NONE = 0,   // 不支持特殊功能
  IB_PROVIDER_MLX5 = 1,   // Mellanox ConnectX/BlueField 系列
  IB_PROVIDER_MAX = 2,
};

// 提供商名称字符串
const char* ibProviderName[] = {
  "None",
  "Mlx5",
};

//================================================================================
// IB 设备结构
//================================================================================

// 最大 IB 设备数
#define MAX_IB_DEVS  32
// 最大虚拟设备数
#define MAX_IB_VDEVS MAX_IB_DEVS*8

// 合并设备数组（虚拟设备）
struct ncclIbMergedDev ncclIbMergedDevs[MAX_IB_VDEVS];

// 物理设备数组
struct ncclIbDev ncclIbDevs[MAX_IB_DEVS];

// 全局互斥锁（保护设备初始化和访问）
static std::mutex ncclIbMutex;

// 是否启用宽松排序（Relaxed Ordering）优化
static int ncclIbRelaxedOrderingEnabled = 0;

// 网络引用计数（支持多次初始化）
// 使用 ncclNet_v11_t 时，NCCL 核心会为每个通信器初始化网络插件
// 但内部实现仍假设插件只初始化一次。引用计数确保内部只初始化一次
static int netRefCount;

// 链路层类型转字符串宏
#define NCCL_IB_LLSTR(ll) (((ll) == IBV_LINK_LAYER_INFINIBAND) ? "IB" : (((ll) == IBV_LINK_LAYER_ETHERNET) ? "RoCE" : "UNSPECIFIED"))

// 默认 Service Level (SL) 和 Traffic Class (TC)
#define NCCL_IB_SL_DEFAULT 0
#define NCCL_IB_TC_DEFAULT 0

//================================================================================
// 环境变量和参数定义
// 通过 NCCL_PARAM 宏定义可配置的环境变量
// 格式: NCCL_PARAM(名称, 环境变量名, 默认值)
//================================================================================

NCCL_PARAM(IbGidIndex, "IB_GID_INDEX", -1);  // GID 索引（RoCE 地址选择）
NCCL_PARAM(IbRoutableFlidIbGidIndex, "IB_ROUTABLE_FLID_GID_INDEX", 1);  // 可路由 FLID 的 GID 索引
NCCL_PARAM(IbRoceVersionNum, "IB_ROCE_VERSION_NUM", 2);  // RoCE 版本（1 或 2）
NCCL_PARAM(IbTimeout, "IB_TIMEOUT", 20);  // QP 超时值（单位：4.096us）
NCCL_PARAM(IbRetryCnt, "IB_RETRY_CNT", 7);  // 重试计数
NCCL_PARAM(IbPkey, "IB_PKEY", 0);  // Partition Key
NCCL_PARAM(IbUseInline, "IB_USE_INLINE", 0);  // 是否使用内联发送（小消息优化）
NCCL_PARAM(IbSl, "IB_SL", -1);  // Service Level（服务质量）
NCCL_PARAM(IbTc, "IB_TC", -1);  // Traffic Class（流量类别）
NCCL_PARAM(IbArThreshold, "IB_AR_THRESHOLD", 8192);  // 自适应路由阈值（字节）
NCCL_PARAM(IbPciRelaxedOrdering, "IB_PCI_RELAXED_ORDERING", 2);  // PCI 宽松排序模式
NCCL_PARAM(IbAdaptiveRouting, "IB_ADAPTIVE_ROUTING", -2);  // 自适应路由开关
NCCL_PARAM(IbFifoTc, "IB_FIFO_TC", -1);  // FIFO 流量类别
NCCL_PARAM(IbAsyncEvents,"IB_RETURN_ASYNC_EVENTS",1);  // 是否返回异步事件
NCCL_PARAM(IbEceEnable,"IB_ECE_ENABLE",1);  // 是否启用增强连接建立 (ECE)
NCCL_PARAM(IbDataDirect,"IB_DATA_DIRECT",1);  // 是否启用 Data Direct DMA

//================================================================================
// 统计信息函数
//================================================================================

// 初始化统计信息
static ncclResult_t ncclIbStatsInit(struct ncclIbStats* stat) {
  // 原子存储：将致命错误计数初始化为 0
  __atomic_store_n(&stat->fatalErrorCount, 0, __ATOMIC_RELAXED);
  return ncclSuccess;
}

// 记录致命错误（原子递增）
static void ncclIbStatsFatalError(struct ncclIbStats* stat){
  __atomic_fetch_add(&stat->fatalErrorCount, 1, __ATOMIC_RELAXED);
}

// 检查是否有致命错误发生
static ncclResult_t ncclIbStatsCheckFatalCount(struct ncclIbStats* stat, const char* funcName) {
  // 如果启用了异步事件且检测到致命错误
  if (ncclParamIbAsyncEvents() && __atomic_load_n(&stat->fatalErrorCount, __ATOMIC_RELAXED)) {
    WARN("communicator encountered a fatal error (detected in %s)\n", funcName);
    return ncclSystemError;
  }
  return ncclSuccess;
}

// QP 致命错误处理（从 qp_context 获取统计结构）
static void ncclIbQpFatalError(struct ibv_qp* qp) {
  ncclIbStatsFatalError((struct ncclIbStats*)qp->qp_context);
}

// CQ 致命错误处理（从 cq_context 获取统计结构）
static void ncclIbCqFatalError(struct ibv_cq* cq) {
  ncclIbStatsFatalError((struct ncclIbStats*)cq->cq_context);
}

// 设备致命错误处理
static void ncclIbDevFatalError(struct ncclIbDev* dev) {
  ncclIbStatsFatalError(&dev->stats);
}

//================================================================================
// 异步事件处理线程
//================================================================================

// 异步事件线程句柄
pthread_t ncclIbAsyncThread;

// 异步事件处理线程主函数
// 功能：监听 IB 设备的异步事件（如链路故障、QP 错误等）
static void* ncclIbAsyncThreadMain(void* args) {

  //是那个设备
  struct ncclIbDev* dev = (struct ncclIbDev*)args;
  
  while (1) {
    struct ibv_async_event event;
    
    // 阻塞调用，等待异步事件
    // 通过 ibv_get_async_event 从指定设备获取异步事件
    if (ncclSuccess != wrap_ibv_get_async_event(dev->context, &event)) { 
        break;  // 出错或设备关闭，退出线程
    }
    
    char *str;
    // 提取事件中的对象指针（仅在特定事件类型时有效）
    struct ibv_cq* cq = event.element.cq;    // 仅在 CQ 错误时有效
    struct ibv_qp* qp = event.element.qp;    // 仅在 QP 错误时有效
    struct ibv_srq* srq = event.element.srq; // 仅在 SRQ 错误时有效
    
    // 将事件类型转换为可读字符串
    if (ncclSuccess != wrap_ibv_event_type_str(&str, event.event_type)) {
        break; 
    }
    
    // 根据事件类型进行处理
    switch (event.event_type) {
    case IBV_EVENT_DEVICE_FATAL:
      // 设备致命错误（整个设备不可用）
      WARN("NET/IB : %s:%d async fatal event: %s", dev->devName, dev->portNum, str);
      ncclIbDevFatalError(dev);
      break;
      
    case IBV_EVENT_CQ_ERR:
      // 完成队列错误
      WARN("NET/IB : %s:%d async fatal event on CQ (%p): %s", dev->devName, dev->portNum, cq, str);
      ncclIbCqFatalError(cq);
      break;
      
    case IBV_EVENT_QP_FATAL:
    case IBV_EVENT_QP_REQ_ERR:
    case IBV_EVENT_QP_ACCESS_ERR:
      // Queue Pair 错误（传输层致命错误）
      WARN("NET/IB : %s:%d async fatal event on QP (%p): %s", dev->devName, dev->portNum, qp, str);
      ncclIbQpFatalError(qp);
      break;
      
    case IBV_EVENT_SRQ_ERR:
      // 共享接收队列错误（NCCL 目前不使用 SRQ）
      WARN("NET/IB : %s:%d async fatal event on SRQ, unused for now (%p): %s", dev->devName, dev->portNum, srq, str);
      break;
      
    case IBV_EVENT_GID_CHANGE:
      // GID 表变化（RoCE 地址改变）
      WARN("NET/IB : %s:%d GID table changed", dev->devName, dev->portNum);
      break;
      
    case IBV_EVENT_PATH_MIG_ERR:
    case IBV_EVENT_PORT_ERR:
    case IBV_EVENT_PATH_MIG:
    case IBV_EVENT_PORT_ACTIVE:
    case IBV_EVENT_SQ_DRAINED:
    case IBV_EVENT_LID_CHANGE:
    case IBV_EVENT_PKEY_CHANGE:
    case IBV_EVENT_SM_CHANGE:
    case IBV_EVENT_QP_LAST_WQE_REACHED:
    case IBV_EVENT_CLIENT_REREGISTER:
    case IBV_EVENT_SRQ_LIMIT_REACHED:
      // 非致命事件（仅记录警告）
      WARN("NET/IB : %s:%d Got async error event: %s", dev->devName, dev->portNum, str);
      break;
      
    case IBV_EVENT_COMM_EST:
      // 连接建立事件（通常忽略，因为使用自定义握手）
      break;
      
    default:
      WARN("NET/IB : %s:%d unknown event type (%d)", dev->devName, dev->portNum, event.event_type);
      break;
    }
    
    // 确认事件处理完成（必须在最后调用，避免 use-after-free）
    // 发送 ack 给 libibverbs，释放事件资源
    if (ncclSuccess != wrap_ibv_ack_async_event(&event)) {
        break;
    }
  }
  
  return NULL;
}

//================================================================================
// GID (Global Identifier) 和地址族辅助函数
//================================================================================

// 从环境变量获取地址族（IPv4 或 IPv6）
static sa_family_t envIbAddrFamily(void) {
  sa_family_t family = AF_INET;  // 默认 IPv4
  const char* env = ncclGetEnv("NCCL_IB_ADDR_FAMILY");
  
  if (env == NULL || strlen(env) == 0) {
    return family;
  }

  INFO(NCCL_ENV, "NCCL_IB_ADDR_FAMILY set by environment to %s", env);

//使用ipv4还是ipv6
  if (strcmp(env, "AF_INET") == 0) {
    family = AF_INET;
  } else if (strcmp(env, "AF_INET6") == 0) {
    family = AF_INET6;
  }

  return family;
}

// 从环境变量获取地址范围前缀（用于子网过滤）
// 返回：前缀地址指针，mask 为前缀长度
static void* envIbAddrRange(sa_family_t af, int* mask) {
  *mask = 0;
  static struct in_addr addr;     // IPv4 地址
  static struct in6_addr addr6;   // IPv6 地址
  void *ret = (af == AF_INET) ? (void *)&addr : (void *)&addr6;

  const char* env = ncclGetEnv("NCCL_IB_ADDR_RANGE");
  if (NULL == env || strlen(env) == 0) {
    return NULL;
  }

  INFO(NCCL_ENV, "NCCL_IB_ADDR_RANGE set by environment to %s", env);

//fe80::ccde:f7ff:fe26:bf0b/64 子网掩码格式
  // 解析 "addr/prefix" 格式
  char addrString[128] = { 0 };
  snprintf(addrString, 128, "%s", env);
  char *addrStrPtr = addrString;
  char *maskStrPtr = strstr(addrString, "/");
  if (NULL == maskStrPtr) {
    return NULL;
  }
  
  *(maskStrPtr++) = '\0';  // 分割地址和掩码

  // 将字符串转换为网络地址
  if (inet_pton(af, addrStrPtr, ret) == 0) {
    INFO(NCCL_INIT|NCCL_NET, "NET/IB: Ip address '%s' is invalid for family %s, ignoring address", addrStrPtr, (af == AF_INET) ? "AF_INET" : "AF_INET6");
    return NULL;
  }

  // 解析掩码长度
  *mask = (int)strtol(maskStrPtr, NULL, 10);
  
  // 验证掩码长度是否合法
  if (af == AF_INET && *mask > 32) {
    INFO(NCCL_INIT|NCCL_NET, "NET/IB: Ip address mask '%d' is invalid for family %s, ignoring mask", *mask, (af == AF_INET) ? "AF_INET" : "AF_INET6");
    *mask = 0;
    ret = NULL;
  } else if (af == AF_INET6 && *mask > 128) {
    INFO(NCCL_INIT|NCCL_NET, "NET/IB: Ip address mask '%d' is invalid for family %s, ignoring mask", *mask, (af == AF_INET) ? "AF_INET" : "AF_INET6");
    *mask = 0;
    ret = NULL;
  }

  return ret;
}

// 从 GID 获取地址族（IPv4 或 IPv6）
// RoCE v2 使用 IPv4 映射的 IPv6 地址格式
static sa_family_t getGidAddrFamily(union ibv_gid* gid) {
  const struct in6_addr *a = (struct in6_addr *)gid->raw;
  
  // 检查是否为 IPv4 映射的 IPv6 地址
  // 格式：::ffff:w.x.y.z 或 ::ffff:0:w.x.y.z（多播）
  bool isIpV4Mapped = ((a->s6_addr32[0] | a->s6_addr32[1]) | (a->s6_addr32[2] ^ htonl(0x0000ffff))) == 0UL;
  bool isIpV4MappedMulticast = (a->s6_addr32[0] == htonl(0xff0e0000) && ((a->s6_addr32[1] | (a->s6_addr32[2] ^ htonl(0x0000ffff))) == 0UL));
  
  return (isIpV4Mapped || isIpV4MappedMulticast) ? AF_INET : AF_INET6;
}

// 检查 GID 是否匹配指定的地址前缀
// 用于过滤 GID，选择特定子网的地址
static bool matchGidAddrPrefix(sa_family_t af, void* prefix, int prefixlen, union ibv_gid* gid) {
  struct in_addr *base = NULL;
  struct in6_addr *base6 = NULL;
  struct in6_addr *addr6 = NULL;;
  
  if (af == AF_INET) {
    base = (struct in_addr *)prefix;
  } else {
    base6 = (struct in6_addr *)prefix;
  }
  addr6 = (struct in6_addr *)gid->raw;

  // 网络掩码宏
#define NETMASK(bits) (htonl(0xffffffff ^ ((1 << (32 - bits)) - 1)))

  int i = 0;
  while (prefixlen > 0 && i < 4) {
    if (af == AF_INET) {
      // IPv4：比较低 32 位
      int mask = NETMASK(prefixlen);
      if ((base->s_addr & mask) ^ (addr6->s6_addr32[3] & mask)) {
        break;
      }
      prefixlen = 0;
      break;
    } else {
      // IPv6：逐个 32 位比较
      if (prefixlen >= 32) {
        if (base6->s6_addr32[i] ^ addr6->s6_addr32[i]) {
          break;
        }
        prefixlen -= 32;
        ++i;
      } else {
        int mask = NETMASK(prefixlen);
        if ((base6->s6_addr32[i] & mask) ^ (addr6->s6_addr32[i] & mask)) {
          break;
        }
        prefixlen = 0;
      }
    }
  }

  return (prefixlen == 0) ? true : false;
}

// 检查 GID 是否已配置（非零且非 link-local）
static bool configuredGid(union ibv_gid* gid) {
  const struct in6_addr *a = (struct in6_addr *)gid->raw;
  int trailer = (a->s6_addr32[1] | a->s6_addr32[2] | a->s6_addr32[3]);
  
  // 全零或 link-local 地址（fe80::）视为未配置
  if (((a->s6_addr32[0] | trailer) == 0UL) || ((a->s6_addr32[0] == htonl(0xfe800000)) && (trailer == 0UL))) {
    return false;
  }
  return true;
}

// 检查是否为 link-local 地址（fe80::/10）
static bool linkLocalGid(union ibv_gid* gid) {
  const struct in6_addr *a = (struct in6_addr *)gid->raw;
  if (a->s6_addr32[0] == htonl(0xfe800000) && a->s6_addr32[1] == 0UL) {
    return true;
  }
  return false;
}

// 检查 GID 是否有效（已配置且非 link-local）
static bool validGid(union ibv_gid* gid) {
  return (configuredGid(gid) && !linkLocalGid(gid));
}

// 从 sysfs 获取 RoCE 版本号
// RoCE v1 使用以太网链路层，RoCE v2 使用 UDP/IPv4
static ncclResult_t ncclIbRoceGetVersionNum(const char* deviceName, int portNum, int gidIndex, int* version) {
  char gidRoceVerStr[16] = { 0 };
  char roceTypePath[PATH_MAX] = { 0 };
  
  // 构建 sysfs 路径
  //cat  /sys/class/infiniband/mlx5_0/ports/1/gid_attrs/types/0
  snprintf(roceTypePath, sizeof(roceTypePath), "/sys/class/infiniband/%s/ports/%d/gid_attrs/types/%d", deviceName, portNum, gidIndex);

  int fd = open(roceTypePath, O_RDONLY);
  if (fd == -1) {
    WARN("NET/IB: open failed in ncclIbRoceGetVersionNum: %s", strerror(errno));
    return ncclSystemError;
  }
  
  int ret = read(fd, gidRoceVerStr, 15);
  close(fd);

  if (ret == -1) {
    // 容器环境中，如果 GID 索引未映射到容器的 sysfs，read 可能返回 EINVAL
    if (errno == EINVAL) return ncclSuccess;
    WARN("NET/IB: read failed in ncclIbRoceGetVersionNum: %s", strerror(errno));
    return ncclSystemError;
  }

  // 解析版本字符串
  if (strlen(gidRoceVerStr)) {
    if (strncmp(gidRoceVerStr, "IB/RoCE v1", strlen("IB/RoCE v1")) == 0 || strncmp(gidRoceVerStr, "RoCE v1", strlen("RoCE v1")) == 0) {
      *version = 1;
    } else if (strncmp(gidRoceVerStr, "RoCE v2", strlen("RoCE v2")) == 0) {
      *version = 2;
    }
  }

  return ncclSuccess;
}

// 更新 GID 索引（根据地址族、RoCE 版本等条件选择最佳 GID）
static ncclResult_t ncclUpdateGidIndex(struct ibv_context* context, uint8_t portNum, sa_family_t af, void* prefix, int prefixlen, int roceVer, int gidIndexCandidate, int* gidIndex) {
  union ibv_gid gid, gidCandidate;
  NCCLCHECK(wrap_ibv_query_gid(context, portNum, *gidIndex, &gid));
  NCCLCHECK(wrap_ibv_query_gid(context, portNum, gidIndexCandidate, &gidCandidate));

  sa_family_t usrFam = af;
  sa_family_t gidFam = getGidAddrFamily(&gid);
  sa_family_t gidCandidateFam = getGidAddrFamily(&gidCandidate);
  bool gidCandidateMatchSubnet = matchGidAddrPrefix(usrFam, prefix, prefixlen, &gidCandidate);

  // 优先选择匹配用户指定地址族的 GID
  if (gidCandidateFam != gidFam && gidCandidateFam == usrFam && gidCandidateMatchSubnet) {
    *gidIndex = gidIndexCandidate;
  } else {
    // 检查候选 GID 是否有效
    if (gidCandidateFam != usrFam || !validGid(&gidCandidate) || !gidCandidateMatchSubnet) {
      return ncclSuccess;
    }
    
    // 比较 RoCE 版本
    int usrRoceVer = roceVer;
    int gidRoceVerNum, gidRoceVerNumCandidate = -1;
    const char* deviceName = wrap_ibv_get_device_name(context->device);
    NCCLCHECK(ncclIbRoceGetVersionNum(deviceName, portNum, *gidIndex, &gidRoceVerNum));
    NCCLCHECK(ncclIbRoceGetVersionNum(deviceName, portNum, gidIndexCandidate, &gidRoceVerNumCandidate));
    
    // 如果候选 GID 的 RoCE 版本匹配用户指定版本，则更新
    if ((gidRoceVerNum != gidRoceVerNumCandidate || !validGid(&gid)) && gidRoceVerNumCandidate == usrRoceVer) {
      *gidIndex = gidIndexCandidate;
    }
  }

  return ncclSuccess;
}

//================================================================================
// GID 格式和 FLID (Forwardable LID) 提取
//================================================================================

// GID 格式说明：
// Global: | 64-bit subnet-prefix | 64-bit EUI (Extended Unique Identifier) |
// Raw:    | 10-bit fixed | 22-bit 0 | 16-bit FLID | 16-bit subnet-prefix | 64-bit EUI |

// 从子网前缀中提取本地子网前缀（低 16 位）
static uint16_t ncclIbExtractLocalSubnetPrefix(uint64_t subnet_prefix)
{
  return (be64toh(subnet_prefix) & 0xffff);
}

// 从 GID 中提取 FLID（Forwardable LID）
// FLID 用于跨子网路由（需要配置路由管理器）
static int ncclIbExtractFlid (union ibv_gid *gid)
{
  return ntohs(*((uint16_t*)((uintptr_t)(gid->raw) + 4)));
}

// 获取最佳 GID 索引
// 根据 IB/RoCE 类型、地址族、子网前缀等条件选择合适的 GID
static ncclResult_t ncclIbGetGidIndex(struct ibv_context *context, uint8_t portNum, struct ibv_port_attr* portAttr, int *gidIndex) {
  int gidTblLen = portAttr->gid_tbl_len;

  // IB 网络：优先选择具有可路由 FLID 的 GID 索引
  if (portAttr->link_layer == IBV_LINK_LAYER_INFINIBAND) {
    union ibv_gid gid;
    int routableGidIndex = ncclParamIbRoutableFlidIbGidIndex();
    
    if (routableGidIndex < gidTblLen) {
      NCCLCHECK(wrap_ibv_query_gid(context, portNum, routableGidIndex, &gid));
      // 如果该索引有非零 FLID，则使用它
      if (ncclIbExtractFlid(&gid) != 0) {
        *gidIndex = routableGidIndex;
        return ncclSuccess;
      }
    }
    *gidIndex = 0;
    return ncclSuccess;
  }

  // RoCE 网络
  *gidIndex = ncclParamIbGidIndex();
  if (*gidIndex >= 0) {
    return ncclSuccess;
  }

  // 自动选择最佳 GID
  sa_family_t userAddrFamily = envIbAddrFamily();
  int userRoceVersion = ncclParamIbRoceVersionNum();
  int prefixlen;
  void *prefix = envIbAddrRange(userAddrFamily, &prefixlen);

  *gidIndex = 0;
  // 遍历所有 GID，寻找最佳匹配
  for (int gidIndexNext = 1; gidIndexNext < gidTblLen; ++gidIndexNext) {
    NCCLCHECK(ncclUpdateGidIndex(context, portNum, userAddrFamily, prefix, prefixlen, userRoceVersion, gidIndexNext, gidIndex));
  }

  return ncclSuccess;
}

//================================================================================
// 设备初始化参数
//================================================================================

NCCL_PARAM(IbDisable, "IB_DISABLE", 0);  // 禁用 IB 传输
NCCL_PARAM(IbMergeVfs, "IB_MERGE_VFS", 1);  // 合并虚拟功能（VF）
NCCL_PARAM(IbMergeNics, "IB_MERGE_NICS", 1);  // 合并多端口 NIC

// 检查两个路径是否属于同一物理设备的 VF
// 返回 0 表示同一设备，非 0 表示不同设备
static int ncclIbMatchVfPath(char* path1, char* path2) {
  // 将多端口 NIC 合并到同一 PCI 设备
  if (ncclParamIbMergeVfs()) {
    // 比较到倒数第 4 个字符（虚拟函数编号）
    return strncmp(path1, path2, strlen(path1)-4) == 0;
  } else {
    // 比较到倒数第 1 个字符（端口号）
    return strncmp(path1, path2, strlen(path1)-1) == 0;
  }
}

// 获取设备的 PCI 路径和真实端口号
// 用于识别多端口 NIC 和虚拟功能（VF）
static ncclResult_t ncclIbGetPciPath(char* devName, char** path, int* realPort) {
  char devicePath[PATH_MAX];
  snprintf(devicePath, PATH_MAX, "/sys/class/infiniband/%s/device", devName);
  
  // 解析真实路径
  char* p = realpath(devicePath, NULL);
  if (p == NULL) {
    WARN("Could not find real path of %s (%s)", devName, devicePath);
  } else {
    // 将端口号替换为 0（合并多端口）
    p[strlen(p)-1] = '0';
    
    // 合并虚拟功能（SR-IOV）
    if (ncclParamIbMergeVfs()) 
        p[strlen(p)-3] = p[strlen(p)-4] = '0';
    
    // 计算真实端口号（基于已存在的设备数量）
    *realPort = 0;
    for (int d=0; d<ncclNIbDevs; d++) {
      if (ncclIbMatchVfPath(p, ncclIbDevs[d].pciPath)) 
        (*realPort)++;
    }
  }
  *path = p;
  
  return ncclSuccess;
}

// IB 带宽数组（索引对应 active_width 的位位置）
static int ibvWidths[] = { 1, 4, 8, 12, 2 };  // 1x, 4x, 8x, 12x, 2x

// IB 速度数组（索引对应 active_speed 的位位置，单位：Mbps）
static int ibvSpeeds[] = {
  2500,  /* SDR (2.5 Gbps) */
  5000,  /* DDR (5 Gbps) */
  10000, /* QDR (10 Gbps) */
  10000, /* QDR (10 Gbps) */
  14000, /* FDR (14 Gbps) */
  25000, /* EDR (25 Gbps) */
  50000, /* HDR (50 Gbps) */
  100000, /* NDR (100 Gbps) */
  200000  /* XDR (200 Gbps) */
};

// 查找第一个设置的位
static int firstBitSet(int val, int max) {
  int i = 0;
  while (i<max && ((val & (1<<i)) == 0))
    i++;
  
  return i;
}

// 根据 active_width 获取带宽倍数
static int ncclIbWidth(int width) {
  return ibvWidths[firstBitSet(width, sizeof(ibvWidths)/sizeof(int)-1)];
}

// 根据 active_speed 获取速度（Mbps）
static int ncclIbSpeed(int speed) {
  return ibvSpeeds[firstBitSet(speed, sizeof(ibvSpeeds)/sizeof(int)-1)];
}

// 检测是否支持宽松排序（Relaxed Ordering）
// 宽松排序可以提升性能，但需要硬件和驱动支持
static int ncclIbRelaxedOrderingCapable(void) {
  int roMode = ncclParamIbPciRelaxedOrdering();
  ncclResult_t r = ncclInternalError;
  
  if (roMode == 1 || roMode == 2) {
    // 查询 IBVERBS_1.8 API - 需要 IBV_ACCESS_RELAXED_ORDERING 支持
    r = wrap_ibv_reg_mr_iova2(NULL, NULL, NULL, 0, 0, 0);
  }
  return r == ncclInternalError ? 0 : 1;
}

// 检测是否支持 DMA-BUF（GPU Direct RDMA via DMA-BUF）
// 这是较新的 GDR 实现，不需要 nvidia_peermem 模块
static bool ncclMlx5dvDmaBufCapable(ibv_context *context){
  ncclResult_t res;
  int dev_fail = 0;

  struct ibv_pd* pd;
  NCCLCHECKGOTO(wrap_ibv_alloc_pd(&pd, context), res, failure);
  
  // 使用虚拟调用测试内核 DMA-BUF 支持（fd=-1）
  (void)wrap_direct_ibv_reg_dmabuf_mr(pd, 0ULL /*offset*/, 0ULL /*len*/, 0ULL /*iova*/, -1 /*fd*/, 0 /*flags*/);
  // ibv_reg_dmabuf_mr() 如果不支持会返回 EOPNOTSUPP/EPROTONOSUPPORT（否则为 EBADF）
  (void)wrap_direct_mlx5dv_reg_dmabuf_mr(pd, 0ULL /*offset*/, 0ULL /*len*/, 0ULL /*iova*/, -1 /*fd*/, 0 /*flags*/, 0 /* mlx5 flags*/);
  // mlx5dv_reg_dmabuf_mr() 如果不支持会返回 EOPNOTSUPP/EPROTONOSUPPORT（否则为 EBADF）
  dev_fail |= (errno == EOPNOTSUPP) || (errno == EPROTONOSUPPORT);
  
  NCCLCHECKGOTO(wrap_ibv_dealloc_pd(pd), res, failure);
  
  if (dev_fail) 
    goto failure;
  return true;
  
failure:
  return false;
}

//================================================================================
// 虚拟设备创建
//================================================================================

// 内部函数：创建虚拟设备（需要持有锁）
ncclResult_t ncclIbMakeVDeviceInternal(int* d, ncclNetVDeviceProps_t* props) {
  // 检查是否禁用 NIC 合并
  if (ncclParamIbMergeNics() == 0 && props->ndevs > 1) {
    INFO(NCCL_NET, "NET/IB : Skipping makeVDevice, NCCL_IB_MERGE_NICS=0");
    return ncclInvalidUsage;
  }

  if (props->ndevs == 0) {
      WARN("NET/IB : Can't make virtual NIC with 0 devices");
      return ncclInvalidUsage;
  }

  if (ncclNMergedIbDevs == MAX_IB_VDEVS) {
    WARN("NET/IB : Cannot allocate any more virtual devices (%d)", MAX_IB_VDEVS);
    return ncclInvalidUsage;
  }

  // 虚拟设备：总是递增合并设备数量
  ncclIbMergedDev* mDev = ncclIbMergedDevs + ncclNMergedIbDevs;
  mDev->vProps.ndevs = 0;
  mDev->speed = 0;

  // 遍历所有要合并的物理设备
  for (int i = 0; i < props->ndevs; i++) {
    ncclIbDev* dev = ncclIbDevs + props->devs[i];
    
    if (mDev->vProps.ndevs == NCCL_IB_MAX_DEVS_PER_NIC)
        return ncclInvalidUsage;
    
    mDev->vProps.devs[mDev->vProps.ndevs++] = props->devs[i];
    mDev->speed += dev->speed;  // 累加速率
    
    // 构建设备名称（如 "mlx5_0+mlx5_1"）
    if (mDev->vProps.ndevs > 1) {
      snprintf(mDev->devName + strlen(mDev->devName), sizeof(mDev->devName) - strlen(mDev->devName), "+%s", dev->devName);
    } else {
      strncpy(mDev->devName, dev->devName, MAXNAMESIZE);
    }
  }

  // 检查链路层一致性（所有设备必须是 IB 或 RoCE）
  ncclIbDev* dev0 = ncclIbDevs + props->devs[0];
  for (int i = 1; i < props->ndevs; i++) {
    if (props->devs[i] >= ncclNIbDevs) {
      WARN("NET/IB : Cannot use physical device %d, max %d", props->devs[i], ncclNIbDevs);
      return ncclInvalidUsage;
    }
    ncclIbDev* dev = ncclIbDevs + props->devs[i];
    if (dev->link != dev0->link) {
      WARN("NET/IB : Attempted to merge incompatible devices: [%d]%s:%d/%s and [%d]%s:%d/%s. Try selecting NICs of only one link type using NCCL_IB_HCA",
        props->devs[0], dev0->devName, dev0->portNum, NCCL_IB_LLSTR(dev0->link), props->devs[i], dev->devName, dev->portNum, NCCL_IB_LLSTR(dev->link));
      return ncclInvalidUsage;
    }
  }

  *d = ncclNMergedIbDevs++;
  INFO(NCCL_NET, "NET/IB : Made virtual device [%d] name=%s speed=%d ndevs=%d", *d, mDev->devName, mDev->speed, mDev->vProps.ndevs);
  return ncclSuccess;
}

// 公共接口：创建虚拟设备（带锁）
ncclResult_t ncclIbMakeVDevice(int* d, ncclNetVDeviceProps_t* props) {
  std::lock_guard<std::mutex> lock(ncclIbMutex);
  ncclResult_t res = ncclIbMakeVDeviceInternal(d, props);
  return res;
}

// 设置网络属性（预留接口）
ncclResult_t ncclIbSetNetAttr(void *ctx, ncclNetAttr_t *netAttr) {
  (void)ctx;
  (void)netAttr;
  return ncclSuccess;
}

// 性能分析器回调函数指针
static ncclProfilerCallback_t ncclProfilerFunction;

//================================================================================
// IB 设备结构定义（完整版）
//================================================================================

// 单个 IB 设备的完整状态
struct alignas(64) ncclIbDev {
  std::mutex mutex;        // 保护设备状态的互斥锁
  int device;              // 设备索引（在 devices 列表中的位置）
  uint64_t guid;           // 全局唯一标识符
  uint8_t portNum;         // 端口号（1-based）
  uint8_t link;            // 链路层类型（IB/RoCE）
  int speed;               // 链路速度（Mbps）
  ibv_context* context;    // IB verbs 上下文
  int pdRefs;              // Protection Domain 引用计数
  ibv_pd* pd;              // Protection Domain（内存保护域）
  char devName[MAXNAMESIZE];  // 设备名称（如 "mlx5_0"）
  char* pciPath;           // PCI 设备路径
  char* virtualPciPath;    // 虚拟 PCI 路径（用于 VF）
  int realPort;            // 真实端口号（多端口设备）
  int maxQp;               // 最大 Queue Pair 数量
  float latency;           // 延迟（未使用）
  struct ncclIbMrCache mrCache;  // 内存注册缓存
  int ar;                  // 是否启用自适应路由（ADAPTIVE_ROUTING）
  struct ibv_port_attr portAttr; // 端口属性
  struct ncclIbStats stats;      // 统计信息
  int dmaBufSupported;      // 是否支持 DMA-BUF
  enum ncclIbProvider ibProvider; // 提供商类型
  union {
    struct {
      int dataDirect;      // 是否支持 Data Direct DMA（CX-8 特性）
    } mlx5;
  } capsProvider;          // 提供商特定能力
};

//================================================================================
// 初始化和设备发现
//================================================================================

// IB 网络初始化
// 功能：探测可用的 IB 设备，初始化 verbs 上下文
ncclResult_t ncclIbInit(void** ctx, uint64_t commId, ncclNetCommConfig_t* config, ncclDebugLogger_t logFunction, ncclProfilerCallback_t profFunction) {
  // 引用计数：支持多次初始化（但只初始化一次）
  if (netRefCount++)
    return ncclSuccess;
  
  ncclResult_t ret = ncclSuccess;
  ncclProfilerFunction = profFunction;
  
  // 检查是否禁用 IB
  if (ncclParamIbDisable()) 
    return ncclInternalError;
  
  static int shownIbHcaEnv = 0;
  
  // 动态加载 libibverbs.so，将函数指针加载到 ibvSymbols
  if(wrap_ibv_symbols() != ncclSuccess) { 
    return ncclInternalError; 
  }

  // 动态加载 libmlx5.so，将函数指针加载到 mlx5dvSymbols
  // 用于高级特性如 CX-8 Direct-NIC
  if(wrap_mlx5dv_symbols() != ncclSuccess) { 
    INFO(NCCL_NET, "NET/IB : Failed to open mlx5dv symbols. Advance features like CX-8 Direct-NIC will be disabled."); 
  }

  // 仅首次初始化时探测设备
  if (ncclNIbDevs == -1) {
    std::lock_guard<std::mutex> lock(ncclIbMutex);
    
    // 初始化 fork 处理（libibverbs 要求）
    wrap_ibv_fork_init();
    
    if (ncclNIbDevs == -1) {
      int nIpIfs = 0;
      ncclNIbDevs = 0;
      ncclNMergedIbDevs = 0;
      
      // 查找网络接口（用于带外通信）
      NCCLCHECK(ncclFindInterfaces(ncclIbIfName, &ncclIbIfAddr, MAX_IF_NAME_SIZE, 1, &nIpIfs));
      if (nIpIfs != 1) {
        WARN("NET/IB : No IP interface found.");
        ret = ncclInternalError;
        goto fail;
      }

      // 检测 IB 卡
      int nIbDevs;
      struct ibv_device** devices;

      // 检查用户是否指定了要使用的 IB 设备：端口
      // 格式示例：mlx5_0:1,mlx5_1:1 表示使用 mlx5_0 和 mlx5_1 的端口 1
      const char* userIbEnv = ncclGetEnv("NCCL_IB_HCA");
      if (userIbEnv != NULL && shownIbHcaEnv++ == 0) 
        INFO(NCCL_NET|NCCL_ENV, "NCCL_IB_HCA set to %s", userIbEnv);
      
      struct netIf userIfs[MAX_IB_DEVS];
      bool searchNot = userIbEnv && userIbEnv[0] == '^';  // ^ 前缀表示排除模式
      if (searchNot) 
        userIbEnv++;
      bool searchExact = userIbEnv && userIbEnv[0] == '=';  // = 前缀表示精确匹配
      if (searchExact) 
        userIbEnv++;

      // 解析用户指定的设备列表到 userIfs
      int nUserIfs = parseStringList(userIbEnv, userIfs, MAX_IB_DEVS);

      // 获取系统当前可用的 RDMA 设备列表
      if (ncclSuccess != wrap_ibv_get_device_list(&devices, &nIbDevs)) {
        ret = ncclInternalError; 
        goto fail;
      }

      // 遍历所有设备
      for (int d=0; d<nIbDevs && ncclNIbDevs<MAX_IB_DEVS; d++) {
        struct ibv_context * context = NULL;
        
        // 打开设备（获取 verbs 上下文）
        if (ncclSuccess != wrap_ibv_open_device(&context, devices[d]) || context == NULL) {
          WARN("NET/IB : Unable to open device %s", devices[d]->name);
          continue;
        }
        
        char dataDirectDevicePath[PATH_MAX] = "/sys";
        int devCount = -1, devOffset = 0;
        
        // 检测设备提供商（是否为 Mellanox 设备）
        enum ncclIbProvider ibProvider = wrap_mlx5dv_is_supported(devices[d]) ? IB_PROVIDER_MLX5 : IB_PROVIDER_NONE;

        int nPorts = 0;
        struct ibv_device_attr devAttr;
        memset(&devAttr, 0, sizeof(devAttr));
        
        // 查询设备属性
        if (ncclSuccess != wrap_ibv_query_device(context, &devAttr)) {
          WARN("NET/IB : Unable to query device %s", devices[d]->name);
          if (ncclSuccess != wrap_ibv_close_device(context)) { 
            ret = ncclInternalError; 
            goto fail; 
          }
          continue;
        }

        // 遍历所有端口
        for (int port_num = 1; port_num <= devAttr.phys_port_cnt; port_num++) {
            struct ibv_port_attr portAttr;
            
            // 查询端口属性
            if (ncclSuccess != wrap_ibv_query_port(context, port_num, &portAttr)) {
              WARN("NET/IB : Unable to query port_num %d", port_num);
              continue;
            }

            // 仅使用 ACTIVE 状态的端口
            if (portAttr.state != IBV_PORT_ACTIVE) 
                continue;
           
            // 仅支持 IB 或 RoCE 链路层
            if (portAttr.link_layer != IBV_LINK_LAYER_INFINIBAND && portAttr.link_layer != IBV_LINK_LAYER_ETHERNET) 
                continue;

            // 根据用户指定的 HCAs/端口进行过滤
            if (! (matchIfList(devices[d]->name, port_num, userIfs, nUserIfs, searchExact) ^ searchNot)) {
                continue;
            }

            // 检查 mlx5 Data Direct 支持（仅检查一次）
            if (devCount == -1) {
              devCount = 1;
              devOffset = 0;
              
              if (ncclParamIbDataDirect() > 0 && ibProvider == IB_PROVIDER_MLX5 && ncclMlx5dvDmaBufCapable(context)) {
                int pathLen = strlen(dataDirectDevicePath);
                ncclResult_t res = wrap_mlx5dv_get_data_direct_sysfs_path(context, dataDirectDevicePath + pathLen, sizeof(dataDirectDevicePath) - pathLen);
                
                if (res == ncclSuccess) {
                  // Data Direct 设备暴露两次：C2C+PCIe 链路和 Data Direct 链路
                  devCount = 2;
                  // 默认只暴露 Data Direct NIC（devOffset = 1），除非用户设置为 2
                  devOffset = (ncclParamIbDataDirect() == 2) ? 0 : 1;
                  INFO(NCCL_INIT | NCCL_NET, "NET/IB: Data Direct DMA Interface is detected for device %s", devices[d]->name);
                } else if (res == ncclInvalidArgument) {
                  TRACE(NCCL_NET, "NET/IB: Device %s does not support Data Direct DMA.", devices[d]->name);
                } else {
                  WARN("NET/IB: Error in mlx5dv_get_data_direct_sysfs_path with device %s", devices[d]->name);
                  return res;
                }
              }
            }
            
            // 可能需要创建两个设备（普通 + Data Direct）
            for (int dev = devOffset; dev < devCount; ++dev) {
              ncclIbDevs[ncclNIbDevs].device = d;
              ncclIbDevs[ncclNIbDevs].ibProvider = ibProvider;
              ncclIbDevs[ncclNIbDevs].guid = devAttr.sys_image_guid;
              ncclIbDevs[ncclNIbDevs].portAttr = portAttr;
              ncclIbDevs[ncclNIbDevs].portNum = port_num;
              ncclIbDevs[ncclNIbDevs].link = portAttr.link_layer;
              
              // 计算链路速度 = 速度 * 带宽
              if (portAttr.active_speed_ex) {
                // 非 zero active_speed_ex 表示 XDR 速率（0x100）或更高
                ncclIbDevs[ncclNIbDevs].speed = ncclIbSpeed(portAttr.active_speed_ex) * ncclIbWidth(portAttr.active_width);
              } else {
                ncclIbDevs[ncclNIbDevs].speed = ncclIbSpeed(portAttr.active_speed) * ncclIbWidth(portAttr.active_width);
              }
              
              ncclIbDevs[ncclNIbDevs].context = context;
              ncclIbDevs[ncclNIbDevs].pdRefs = 0;
              ncclIbDevs[ncclNIbDevs].pd = NULL;
              
              if (dev == 0) {
                // 普通设备
                strncpy(ncclIbDevs[ncclNIbDevs].devName, devices[d]->name, MAXNAMESIZE);
                NCCLCHECKGOTO(ncclIbGetPciPath(ncclIbDevs[ncclNIbDevs].devName, &ncclIbDevs[ncclNIbDevs].pciPath, &ncclIbDevs[ncclNIbDevs].realPort), ret, fail);
              } else {
                // Data Direct 设备
                snprintf(ncclIbDevs[ncclNIbDevs].devName, MAXNAMESIZE, "%s_dma", devices[d]->name);
                NCCLCHECK(ncclCalloc(&ncclIbDevs[ncclNIbDevs].pciPath, PATH_MAX));
                strncpy(ncclIbDevs[ncclNIbDevs].pciPath, dataDirectDevicePath, PATH_MAX);
                ncclIbDevs[ncclNIbDevs].capsProvider.mlx5.dataDirect = 1;
              }
              
              ncclIbDevs[ncclNIbDevs].maxQp = devAttr.max_qp;
              ncclIbDevs[ncclNIbDevs].mrCache.capacity = 0;
              ncclIbDevs[ncclNIbDevs].mrCache.population = 0;
              ncclIbDevs[ncclNIbDevs].mrCache.slots = NULL;
              NCCLCHECK(ncclIbStatsInit(&ncclIbDevs[ncclNIbDevs].stats));

              // 默认在 IB 网络上启用自适应路由，但允许环境变量覆盖
              ncclIbDevs[ncclNIbDevs].ar = (portAttr.link_layer == IBV_LINK_LAYER_INFINIBAND) ? 1 : 0;
              if (ncclParamIbAdaptiveRouting() != -2) ncclIbDevs[ncclNIbDevs].ar = ncclParamIbAdaptiveRouting();

              INFO(NCCL_NET, "NET/IB: [%d] %s:%s:%d/%s provider=%s speed=%d context=%p pciPath=%s ar=%d", d, devices[d]->name, devices[d]->dev_name,
                   ncclIbDevs[ncclNIbDevs].portNum, NCCL_IB_LLSTR(portAttr.link_layer), ibProviderName[ncclIbDevs[ncclNIbDevs].ibProvider], ncclIbDevs[ncclNIbDevs].speed, context,
                   ncclIbDevs[ncclNIbDevs].pciPath, ncclIbDevs[ncclNIbDevs].ar);

              // 创建异步事件处理线程
              PTHREADCHECKGOTO(pthread_create(&ncclIbAsyncThread, NULL, ncclIbAsyncThreadMain, ncclIbDevs + ncclNIbDevs), "pthread_create", ret, fail);
              ncclSetThreadName(ncclIbAsyncThread, "NCCL IbAsync %2d", ncclNIbDevs);
              PTHREADCHECKGOTO(pthread_detach(ncclIbAsyncThread), "pthread_detach", ret, fail); // 不会 pthread_join()

              // 将此物理设备添加到虚拟设备列表
              int vDev;
              ncclNetVDeviceProps_t vProps = {0};
              vProps.ndevs = 1;
              vProps.devs[0] = ncclNIbDevs;
              NCCLCHECK(ncclIbMakeVDeviceInternal(&vDev, &vProps));

              ncclNIbDevs++;
              nPorts++;
            }
        }
        
        // 如果没有有效端口，关闭设备
        if (nPorts == 0 && ncclSuccess != wrap_ibv_close_device(context)) { 
            ret = ncclInternalError; 
            goto fail;
        }
      }

      // 释放设备列表
      if (nIbDevs && (ncclSuccess != wrap_ibv_free_device_list(devices))) { 
          ret = ncclInternalError; 
          goto fail; 
      }
    }
    
    // 检查是否找到设备
    if (ncclNIbDevs == 0) {
      INFO(NCCL_INIT|NCCL_NET, "NET/IB : No device found.");
    }

    // 打印所有网络设备（格式与之前兼容）
    char line[2048];
    line[0] = '\0';
    
    // 确定是否启用宽松排序
    ncclIbRelaxedOrderingEnabled = ncclIbRelaxedOrderingCapable();
    
    for (int d = 0; d < ncclNIbDevs; d++) {
        snprintf(line+strlen(line), sizeof(line)-strlen(line), " [%d]%s:%d/%s", d, ncclIbDevs[d].devName,
          ncclIbDevs[d].portNum, NCCL_IB_LLSTR(ncclIbDevs[d].link));
    }
    char addrline[SOCKET_NAME_MAXLEN+1];
    INFO(NCCL_INIT|NCCL_NET, "NET/IB : Using%s %s; OOB %s:%s", line, ncclIbRelaxedOrderingEnabled ? "[RO]" : "",
          ncclIbIfName, ncclSocketToString(&ncclIbIfAddr, addrline));

  }
exit:
  ibContext.trafficClass = config->trafficClass;
  *ctx = &ibContext;
  return ret;
fail:
  goto exit;
}

// 获取设备数量（虚拟设备）
ncclResult_t ncclIbDevices(int* ndev) {
  *ndev = ncclNMergedIbDevs;
  return ncclSuccess;
}

//================================================================================
// GPU Direct RDMA (GDR) 支持检测
//================================================================================

// 检查内核模块是否已加载
#define KNL_MODULE_LOADED(a) ((access(a, F_OK) == -1) ? 0 : 1)

// GDR 模块加载状态（1=已加载, 0=未加载）
static int ncclIbGdrModuleLoaded = 0;

// 一次性初始化：检查 nvidia_peermem 模块是否已加载
// nvidia_peermem 是传统 GDR 实现方式
static void ibGdrSupportInitOnce() {
  // 要求加载 nvidia_peermem.ko 内核模块
  // 检查多个可能的模块路径（不同发行版位置不同）
  ncclIbGdrModuleLoaded = KNL_MODULE_LOADED("/sys/kernel/mm/memory_peers/nv_mem/version") ||
                          KNL_MODULE_LOADED("/sys/kernel/mm/memory_peers/nv_mem_nc/version") ||
                          KNL_MODULE_LOADED("/sys/module/nvidia_peermem/version");
}

// 检查是否支持 GDR（通过 nvidia_peermem 模块）
ncclResult_t ncclIbGdrSupport() {
  static std::once_flag once;
  std::call_once(once, ibGdrSupportInitOnce);
  
  if (!ncclIbGdrModuleLoaded)
    return ncclSystemError;
  return ncclSuccess;
}

//================================================================================
// DMA-BUF 支持检测
//================================================================================

// 哪个设备需要初始化（线程局部变量）
static __thread int ibDmaSupportInitDev;

// 一次性初始化：检查是否支持 DMA-BUF
// DMA-BUF 是较新的 GDR 实现，不需要 nvidia_peermem 模块
static void ibDmaBufSupportInitOnce(){
  ncclResult_t res;
  int dev_fail = 0;

  // 这是物理设备，不是虚拟设备，从 ibDevs 选择
  ncclIbMergedDev* mergedDev = ncclIbMergedDevs + ibDmaSupportInitDev;
  ncclIbDev* ibDev = ncclIbDevs + mergedDev->vProps.devs[0];
  struct ibv_pd* pd;
  struct ibv_context* ctx = ibDev->context;
  
  NCCLCHECKGOTO(wrap_ibv_alloc_pd(&pd, ctx), res, failure);
  
  // 使用虚拟调用测试内核 DMA-BUF 支持（fd=-1）
  (void)wrap_direct_ibv_reg_dmabuf_mr(pd, 0ULL /*offset*/, 0ULL /*len*/, 0ULL /*iova*/, -1 /*fd*/, 0 /*flags*/);
  // 调用 IB 的接口看 DMA buf 能否注册成功
  // 如果能，表示支持 DMA-BUF
  // ibv_reg_dmabuf_mr() 如果不支持会返回 EOPNOTSUPP/EPROTONOSUPPORT（否则为 EBADF）
  dev_fail |= (errno == EOPNOTSUPP) || (errno == EPROTONOSUPPORT);
  
  NCCLCHECKGOTO(wrap_ibv_dealloc_pd(pd), res, failure);
  
  if (dev_fail) 
    goto failure;
  
  ibDev->dmaBufSupported = 1;
  return;
  
failure:
  ibDev->dmaBufSupported = -1;
  return;
}

// 检查是否支持 DMA-BUF
ncclResult_t ncclIbDmaBufSupport(int dev) {
  static std::once_flag onces[MAX_IB_DEVS];
  
  // 仅初始化设备一次
  ibDmaSupportInitDev = dev;
  std::call_once(onces[dev], ibDmaBufSupportInitOnce);
  
  ncclIbMergedDev* mergedDev = ncclIbMergedDevs + ibDmaSupportInitDev;
  ncclIbDev* ibDev = ncclIbDevs + mergedDev->vProps.devs[0];
  int dmaBufSupported = ibDev->dmaBufSupported;
  
  if (dmaBufSupported == 1) 
    return ncclSuccess;
  return ncclSystemError;
}

//================================================================================
// 常量定义
//================================================================================

// 每个连接最大支持的同时接收操作数
#define NCCL_NET_IB_MAX_RECVS 8

// 获取物理设备属性
ncclResult_t ncclIbGetPhysProperties(int dev, ncclNetProperties_t* props) {
  struct ncclIbDev* ibDev = ncclIbDevs + dev;
  std::lock_guard<std::mutex> lock(ibDev->mutex);
  
  props->name = ibDev->devName;
  props->speed = ibDev->speed;
  props->pciPath = ibDev->pciPath;
  props->guid = ibDev->guid;
  props->ptrSupport = NCCL_PTR_HOST;  // 支持主机内存
  
  // 检查 GPU Direct RDMA 支持
  if (ncclIbGdrSupport() == ncclSuccess) {
    props->ptrSupport |= NCCL_PTR_CUDA;  // GDR via nvidia_peermem
  }
  
  props->regIsGlobal = 1;  // 内存注册是全局的
  
  // 检查 DMA-BUF GDR 支持
  if (ncclIbDmaBufSupport(dev) == ncclSuccess) {
    props->ptrSupport |= NCCL_PTR_DMABUF;  // GDR via DMA-BUF
  }
  
  // Data Direct 设备需要强制 flush
  props->forceFlush = 0;
  if (ibDev->capsProvider.mlx5.dataDirect) {
    props->forceFlush = 1;
  }
  
  props->latency = 0;  // 未设置
  props->port = ibDev->portNum + ibDev->realPort;  // 端口号（考虑多端口）
  props->maxComms = ibDev->maxQp;  // 最大通信数
  props->maxRecvs = NCCL_NET_IB_MAX_RECVS;
  props->netDeviceType    = NCCL_NET_DEVICE_HOST;  // 主机端设备
  props->netDeviceVersion = NCCL_NET_DEVICE_INVALID_VERSION;
  props->maxP2pBytes = NCCL_MAX_NET_SIZE_BYTES;
  props->maxCollBytes = MAX_COLLNET_SIZE;
  props->maxMultiRequestSize = 1;
  return ncclSuccess;
}

// 获取虚拟设备属性
ncclResult_t ncclIbGetProperties(int dev, ncclNetProperties_t* props) {
  if (dev >= ncclNMergedIbDevs) {
    WARN("NET/IB : Requested properties for vNic %d, only %d vNics have been created", dev, ncclNMergedIbDevs);
    return ncclInvalidUsage;
  }
  
  struct ncclIbMergedDev* mergedDev = ncclIbMergedDevs + dev;
  // 从任意子设备获取其余属性（应该相同）
  NCCLCHECK(ncclIbGetPhysProperties(mergedDev->vProps.devs[0], props));
  
  props->name = mergedDev->devName;
  props->speed = mergedDev->speed;  // 虚拟设备速度是所有子设备之和
  memcpy(&props->vProps, &mergedDev->vProps, sizeof(ncclNetVDeviceProps_t));
  return ncclSuccess;
}

//================================================================================
// 连接和通信相关结构定义
//================================================================================

// 我们需要为每个并发接收操作支持 NCCL_NET_MAX_REQUESTS 个请求
#define MAX_REQUESTS (NCCL_NET_MAX_REQUESTS*NCCL_NET_IB_MAX_RECVS)
// 静态断言：确保请求 ID 可以编码在 wr_id 中（每个完成最多 8 个请求）
static_assert(MAX_REQUESTS <= 256, "request id are encoded in wr_id and we need up to 8 requests ids per completion");

// 每个 QP 的最大数量
#define NCCL_IB_MAX_QPS 128

//================================================================================
// Per-QP 连接元数据
//================================================================================

// 每个 QP 的连接信息
struct ncclIbQpInfo {
  uint32_t qpn;  // Queue Pair Number

  // ECE (Enhanced Connection Establishment) 相关字段
  struct ibv_ece ece;     // ECE 配置
  int ece_supported;      // 是否支持 ECE
  int devIndex;           // 设备索引
};

//================================================================================
// Per-Dev 连接元数据
//================================================================================

// 每个设备的连接信息
struct ncclIbDevInfo {
  uint32_t lid;           // Local Identifier (IB)
  uint8_t ib_port;        // IB 端口号
  enum ibv_mtu mtu;       // 最大传输单元
  uint8_t link_layer;     // 链路层类型（IB/RoCE）

  // RoCE 和 IB 路由器
  union ibv_gid gid;      // Global Identifier

  // FIFO RDMA 信息
  uint32_t fifoRkey;      // FIFO 的远程密钥

  // 远程设备信息
  union ibv_gid remoteGid;  // 远程 GID
};

//================================================================================
// 连接元数据（包含建立连接所需的所有信息）
//================================================================================

// 连接元数据结构
struct ncclIbConnectionMetadata {
  struct ncclIbQpInfo qpInfo[NCCL_IB_MAX_QPS];      // 每个 QP 的信息
  struct ncclIbDevInfo devs[NCCL_IB_MAX_DEVS_PER_NIC];  // 每个设备的信息
  char devName[MAX_MERGED_DEV_NAME];  // 设备名称
  uint64_t fifoAddr;       // FIFO 地址
  int ndevs;               // 设备数量
  int tc;                  // Traffic Class
  int sl;                  // Service Level
};

//================================================================================
// 连接状态机
//================================================================================

// 连接状态枚举
enum ncclIbCommState {
  ncclIbCommStateStart = 0,         // 初始状态
  ncclIbCommStateConnect = 1,       // 连接中（检查连接）
  ncclIbCommStateAccept = 3,        // 接受连接
  ncclIbCommStateSend = 4,          // 发送数据
  ncclIbCommStateRecv = 5,          // 接收数据
  ncclIbCommStateConnecting = 6,    // 正在建立 QP 连接
  ncclIbCommStateConnected = 7,     // 已连接
  ncclIbCommStatePendingReady = 8,  // 等待 ready 信号
  ncclIbCommStateSendDevList = 9,   // 发送设备列表
  ncclIbCommStateRecvDevList = 10,  // 接收设备列表
};

// 连接阶段（用于状态机的临时存储）
struct ncclIbCommStage {
  enum ncclIbCommState state;  // 当前状态
  int offset;                   // 传输偏移量（用于分步传输）
  void* buffer;                 // 临时缓冲区
  void* comm;                   // 通信对象指针
};

// 连接句柄（用于带外通信交换连接信息）
struct ncclIbHandle {
  union ncclSocketAddress connectAddr;  // 填充目标地址
  uint64_t magic;  // 魔数（随机数，用于调试）
  struct ncclIbCommStage stage;  // 对方连接时使用
};

//================================================================================
// 本地 GID 信息（用于错误日志）
//================================================================================

// GID 信息
struct ncclIbGidInfo {
  uint8_t link_layer;        // 链路层类型
  union ibv_gid localGid;    // 本地 GID
  int32_t localGidIndex;     // 本地 GID 索引
};

//================================================================================
// 请求类型定义
//================================================================================

#define NCCL_NET_IB_REQ_UNUSED 0  // 未使用
#define NCCL_NET_IB_REQ_SEND 1    // 发送请求
#define NCCL_NET_IB_REQ_RECV 2    // 接收请求
#define NCCL_NET_IB_REQ_FLUSH 3   // 刷新请求（GDR）

// 请求类型字符串（用于调试）
const char* reqTypeStr[] = { "Unused", "Send", "Recv", "Flush" };

// 每个 QP 的最大请求数
#define MAX_QPS_PER_REQ 8

//================================================================================
// 性能分析信息结构
//================================================================================

#ifdef NCCL_ENABLE_NET_PROFILING
// 性能分析器信息
struct ncclProfilerInfo {
  void* qpEventHandles[MAX_QPS_PER_REQ];  // QP 事件句柄
  int qpIndex[MAX_QPS_PER_REQ];            // QP 索引
  int nEventHandles;                       // 事件句柄数量
  ncclProfilerNetIbDescr_v1_t data;        // 描述符数据
  void* pHandle;                           // 性能分析器句柄
};
#endif

//================================================================================
// 请求结构
//================================================================================

// IB 网络请求（异步操作）
struct ncclIbRequest {
  struct ncclIbNetCommBase* base;  // 基础通信结构
  int type;                         // 请求类型（SEND/RECV/FLUSH）
  struct ncclSocket* sock;          // 套接字（用于错误日志）
  int events[NCCL_IB_MAX_DEVS_PER_NIC];  // 每个设备的待处理事件数
  struct ncclIbNetCommDevBase* devBases[NCCL_IB_MAX_DEVS_PER_NIC];  // 设备基础结构指针
#ifdef NCCL_ENABLE_NET_PROFILING
  struct ncclProfilerInfo pInfo[NCCL_NET_IB_MAX_RECVS];  // 性能分析信息
#endif
  int nreqs;  // 子请求数量（用于多接收）
  union {
    struct {
      int size;                           // 数据大小
      void* data;                         // 数据指针
      uint32_t lkeys[NCCL_IB_MAX_DEVS_PER_NIC];  // 本地密钥
      int offset;                         // 传输偏移量
    } send;
    struct {
      int* sizes;  // 接收到的数据大小数组
    } recv;
  };
};

//================================================================================
// 通信基础结构
//================================================================================

// 每个 QP 的设备基础结构
struct ncclIbNetCommDevBase {
  int ibDevN;                    // IB 设备索引
  struct ibv_pd* pd;             // Protection Domain
  struct ibv_cq* cq;             // Completion Queue
  uint64_t pad[2];               // 填充（对齐）
  struct ncclIbGidInfo gidInfo;  // GID 信息
};

// 监听通信结构
struct ncclIbListenComm {
  int dev;                       // 设备索引
  struct ncclSocket sock;        // 套接字
  struct ncclIbCommStage stage;  // 连接阶段
};

//================================================================================
// FIFO 结构（用于接收通知）
//================================================================================

// 发送 FIFO 元素（本地）
struct ncclIbSendFifo {
  uint64_t addr;                              // 远程接收缓冲区地址
  uint64_t size;                              // 远程接收缓冲区大小
  uint32_t rkeys[NCCL_IB_MAX_DEVS_PER_NIC];   // 远程密钥（每个设备一个）
  uint32_t nreqs;                             // 请求数量（多接收）
  uint32_t tag;                               // 标签（用于匹配）
  uint64_t idx;                               // FIFO 索引
  char padding[16];                           // 填充到 32 字节
};

// Queue Pair 封装
struct ncclIbQp {
  struct ibv_qp* qp;  // Queue Pair 指针
  int devIndex;       // 设备索引
  int remDevIdx;      // 远程设备索引
};

// 远程大小 FIFO（用于多接收）
struct ncclIbRemSizesFifo {
  int elems[MAX_REQUESTS][NCCL_NET_IB_MAX_RECVS];  // 大小数组
  uint64_t fifoTail;                                // FIFO 尾指针
  uint64_t addr;                                    // FIFO 地址
  uint32_t rkeys[NCCL_IB_MAX_DEVS_PER_NIC];         // 远程密钥
  uint32_t flags;                                   // 标志（如 INLINE）
  struct ibv_mr* mrs[NCCL_IB_MAX_DEVS_PER_NIC];     // 内存注册句柄
  struct ibv_sge sge;                               // 散射/聚集条目
};

// 发送通信的每个设备结构
struct alignas(8) ncclIbSendCommDev {
  struct ncclIbNetCommDevBase base;  // 基础结构
  struct ibv_mr* fifoMr;              // FIFO 内存注册句柄
};

//================================================================================
// 内存注册句柄包装
//================================================================================

// 内存注册句柄（每个设备一个 MR）
struct ncclIbMrHandle {
  ibv_mr* mrs[NCCL_IB_MAX_DEVS_PER_NIC];  // 每个设备的 MR
};

//================================================================================
// 通信基础结构（发送和接收共用）
//================================================================================

// 网络通信基础结构（32 字节对齐）
struct alignas(32) ncclIbNetCommBase {
  ncclNetVDeviceProps_t vProps;  // 虚拟设备属性
  bool isSend;                   // 是否为发送端
  struct ncclIbRequest reqs[MAX_REQUESTS];  // 请求池
  struct ncclIbQp qps[NCCL_IB_MAX_QPS];     // Queue Pairs
  int nqps;                      // QP 总数
  int qpIndex;                   // 当前 QP 索引
  int devIndex;                  // 当前设备索引
  struct ncclSocket sock;        // 套接字
  int ready;                     // 是否就绪
  // 跟踪必要的远程设备信息
  int nRemDevs;                  // 远程设备数量
  int nDataQps;                  // 数据 QP 数量
  struct ncclIbDevInfo remDevs[NCCL_IB_MAX_DEVS_PER_NIC];  // 远程设备信息
  // 通信统计
  struct ncclIbStats stats;      // 统计信息
};

//================================================================================
// 发送通信结构
//================================================================================

// 发送通信结构
struct ncclIbSendComm {
  struct ncclIbNetCommBase base;
  // 以 fifo 和 ibv 结构开始（有对齐限制）
  struct ncclIbSendFifo fifo[MAX_REQUESTS][NCCL_NET_IB_MAX_RECVS];  // 本地 FIFO
  struct ibv_sge sges[NCCL_NET_IB_MAX_RECVS];  // 散射/聚集条目
  struct ibv_send_wr wrs[NCCL_NET_IB_MAX_RECVS + 1];  // 工作请求
  // 每个 dev 对应一个 mergedIbDev
  struct ncclIbSendCommDev devs[NCCL_IB_MAX_DEVS_PER_NIC];  // 设备特定数据
  struct ncclIbRequest* fifoReqs[MAX_REQUESTS][NCCL_NET_IB_MAX_RECVS];  // FIFO 请求指针
  struct ncclIbRemSizesFifo remSizesFifo;  // 远程大小 FIFO
  uint64_t fifoHead;                       // FIFO 头指针
  int ar;  // 当所有合并设备都启用时使用自适应路由
};

// 静态断言：确保对齐要求（32 字节）
// SendFifo 需要 32 字节对齐，每个元素需要是 32 字节的倍数
// 这样当启用 IB 宽松排序时，条目不会被拆分且不会乱序写入
static_assert((sizeof(struct ncclIbNetCommBase) % 32) == 0, "ncclIbNetCommBase size must be 32-byte multiple to ensure fifo is at proper offset");
static_assert((offsetof(struct ncclIbSendComm, fifo) % 32) == 0, "ncclIbSendComm fifo must be 32-byte aligned");
static_assert((sizeof(struct ncclIbSendFifo) % 32) == 0, "ncclIbSendFifo element size must be 32-byte multiples");
static_assert((offsetof(struct ncclIbSendComm, sges) % 32) == 0, "sges must be 32-byte aligned");
static_assert((offsetof(struct ncclIbSendComm, wrs) % 32) == 0, "wrs must be 32-byte aligned");

//================================================================================
// GPU Flush 结构（GDR）
//================================================================================

// GPU Direct RDMA 刷新结构
struct ncclIbGpuFlush {
  struct ibv_mr* hostMr;      // 主机内存 MR（用于刷新）
  struct ibv_sge sge;         // 散射/聚集条目
  struct ncclIbQp qp;         // 专用 QP（用于刷新）
};

// 远程 FIFO（接收端）
struct ncclIbRemFifo {
  struct ncclIbSendFifo elems[MAX_REQUESTS][NCCL_NET_IB_MAX_RECVS];  // FIFO 元素
  uint64_t fifoTail;           // FIFO 尾指针
  uint64_t addr;               // FIFO 地址
  uint32_t flags;              // 标志
};

//================================================================================
// 接收通信结构
//================================================================================

// 接收通信的每个设备结构（16 字节对齐）
struct alignas(16) ncclIbRecvCommDev {
  struct ncclIbNetCommDevBase base;  // 基础结构
  struct ncclIbGpuFlush gpuFlush;    // GPU 刷新
  struct ibv_mr* fifoMr;             // FIFO 内存注册
  struct ibv_sge fifoSge;            // FIFO 散射/聚集条目
  struct ibv_mr* sizesFifoMr;        // 大小 FIFO 内存注册
};

// 接收通信结构
struct ncclIbRecvComm {
  struct ncclIbNetCommBase base;
  struct ncclIbRecvCommDev devs[NCCL_IB_MAX_DEVS_PER_NIC];  // 设备特定数据
  struct ncclIbRemFifo remFifo;  // 远程 FIFO
  int sizesFifo[MAX_REQUESTS][NCCL_NET_IB_MAX_RECVS];  // 大小 FIFO
  int gpuFlushHostMem;  // GPU 刷新主机内存
  int flushEnabled;     // 是否启用刷新
};

// 静态断言：确保 FIFO 对齐
static_assert((offsetof(struct ncclIbRecvComm, remFifo) % 32) == 0, "ncclIbRecvComm fifo must be 32-byte aligned");

//================================================================================
// 参数定义
//================================================================================

// 每个连接的 QP 数量
NCCL_PARAM(IbQpsPerConn, "IB_QPS_PER_CONNECTION", 1);

//================================================================================
// 辅助函数
//================================================================================

// 添加事件到请求（跟踪待完成事件）
static void ncclIbAddEvent(struct ncclIbRequest* req, int devIndex, struct ncclIbNetCommDevBase* base) {
  req->events[devIndex]++;
  req->devBases[devIndex] = base;
}

// 初始化通信设备基础结构
// 功能：分配 Protection Domain 和 Completion Queue
ncclResult_t ncclIbInitCommDevBase(int ibDevN, struct ncclIbNetCommDevBase* base, void* cq_context) {
  base->ibDevN = ibDevN;
  ncclIbDev* ibDev = ncclIbDevs + ibDevN;
  
  {
    std::lock_guard<std::mutex> lock(ibDev->mutex);
    // 引用计数：首次使用时分配 PD
    if (0 == ibDev->pdRefs++) {
      NCCLCHECK(wrap_ibv_alloc_pd(&ibDev->pd, ibDev->context));
    }
    base->pd = ibDev->pd;
  }

  // 创建 Completion Queue
  // 接收请求可能生成 2 个完成（一个用于 FIFO，一个用于接收）
  NCCLCHECK(wrap_ibv_create_cq(&base->cq, ibDev->context, 2*MAX_REQUESTS*ncclParamIbQpsPerConn(), cq_context, NULL, 0));

  return ncclSuccess;
}

// 销毁基础结构
// 功能：销毁 CQ，递减 PD 引用计数
ncclResult_t ncclIbDestroyBase(struct ncclIbNetCommDevBase* base) {
  NCCLCHECK(wrap_ibv_destroy_cq(base->cq));

  std::lock_guard<std::mutex> lock(ncclIbDevs[base->ibDevN].mutex);
  if (0 == --ncclIbDevs[base->ibDevN].pdRefs) {
    NCCLCHECK(wrap_ibv_dealloc_pd(ncclIbDevs[base->ibDevN].pd));
  }
  return ncclSuccess;
}

// 创建 Queue Pair
// 功能：分配并初始化 QP，设置初始状态为 INIT
ncclResult_t ncclIbCreateQp(uint8_t ib_port, struct ncclIbNetCommDevBase* base, int access_flags, void* qp_context, struct ncclIbQp* qp) {
  struct ibv_qp_init_attr qpInitAttr;
  memset(&qpInitAttr, 0, sizeof(struct ibv_qp_init_attr));
  
  qpInitAttr.qp_context = qp_context;
  qpInitAttr.send_cq = base->cq;
  qpInitAttr.recv_cq = base->cq;
  qpInitAttr.qp_type = IBV_QPT_RC;  // Reliable Connected
  
  // 我们可能每次发送发送 2 条消息（RDMA 和 RDMA_WITH_IMM）
  qpInitAttr.cap.max_send_wr = 2*MAX_REQUESTS;
  qpInitAttr.cap.max_recv_wr = MAX_REQUESTS;
  qpInitAttr.cap.max_send_sge = 1;
  qpInitAttr.cap.max_recv_sge = 1;
  qpInitAttr.cap.max_inline_data = ncclParamIbUseInline() ? sizeof(struct ncclIbSendFifo) : 0;
  
  NCCLCHECK(wrap_ibv_create_qp(&qp->qp, base->pd, &qpInitAttr));
  
  // 将 QP 转换为 INIT 状态
  struct ibv_qp_attr qpAttr;
  memset(&qpAttr, 0, sizeof(struct ibv_qp_attr));
  qpAttr.qp_state = IBV_QPS_INIT;
  qpAttr.pkey_index = ncclParamIbPkey();
  qpAttr.port_num = ib_port;
  qpAttr.qp_access_flags = access_flags;
  NCCLCHECK(wrap_ibv_modify_qp(qp->qp, &qpAttr, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS));
  
  TRACE(NCCL_NET, "NET/IB : ncclIbCreateQp port=%d dev=%d devName=%s ndevs=%d nmdevs=%d qpn=%u pkey=%u pd=%p",
    ib_port, base->ibDevN, ncclIbDevs[base->ibDevN].devName, ncclNIbDevs, ncclNMergedIbDevs, qp->qp->qp_num, qpAttr.pkey_index, base->pd);
  
  return ncclSuccess;
}

// 将 QP 转换为 RTR (Ready to Receive) 状态
// 功能：配置远程 QP 信息，准备接收数据
ncclResult_t ncclIbRtrQp(struct ibv_qp* qp, struct ncclIbGidInfo* sGidInfo, uint32_t dest_qp_num, struct ncclIbDevInfo* info, bool fifoTc, int tc, int sl) {
  struct ibv_qp_attr qpAttr;
  memset(&qpAttr, 0, sizeof(struct ibv_qp_attr));
  
  qpAttr.qp_state = IBV_QPS_RTR;
  qpAttr.path_mtu = info->mtu;
  qpAttr.dest_qp_num = dest_qp_num;
  qpAttr.rq_psn = 0;
  qpAttr.max_dest_rd_atomic = 1;
  qpAttr.min_rnr_timer = 12;
  
  if (info->link_layer == IBV_LINK_LAYER_ETHERNET) {
    // RoCE：使用全局路由（GRH）
    qpAttr.ah_attr.is_global = 1;
    qpAttr.ah_attr.grh.dgid.global.subnet_prefix = info->gid.global.subnet_prefix;
    qpAttr.ah_attr.grh.dgid.global.interface_id = info->gid.global.interface_id;
    qpAttr.ah_attr.grh.flow_label = 0;
    qpAttr.ah_attr.grh.sgid_index = sGidInfo->localGidIndex;
    qpAttr.ah_attr.grh.hop_limit = 255;
    // 为 FIFO 使用不同的 TC（如果配置）
    qpAttr.ah_attr.grh.traffic_class = fifoTc && ncclParamIbFifoTc() != -1 ? ncclParamIbFifoTc() : tc;
  } else {
    // InfiniBand：
    // 如果子网前缀相同，使用 LID；否则使用 FLID（需要路由器）
    if (ncclIbExtractLocalSubnetPrefix(sGidInfo->localGid.global.subnet_prefix) ==
		    ncclIbExtractLocalSubnetPrefix(info->gid.global.subnet_prefix)) {
        // 同一子网：使用 LID
        qpAttr.ah_attr.is_global = 0;
        qpAttr.ah_attr.dlid = info->lid;
    } else {
        // 跨子网：使用 FLID（需要路由管理器配置）
        uint16_t flid = ncclIbExtractFlid(&info->gid);
        if (flid == 0) {
          WARN("Warning: remote FLID configured as zero even when endpoints are on different subnets, using dlid as fallback");
          qpAttr.ah_attr.dlid = info->lid;
	} else {
          qpAttr.ah_attr.dlid = flid;
	}
        qpAttr.ah_attr.is_global = 1;
        qpAttr.ah_attr.grh.dgid.global.subnet_prefix = info->gid.global.subnet_prefix;
        qpAttr.ah_attr.grh.dgid.global.interface_id = info->gid.global.interface_id;
        qpAttr.ah_attr.grh.sgid_index = sGidInfo->localGidIndex;
        qpAttr.ah_attr.grh.hop_limit = 255;
    }
  }
  
  qpAttr.ah_attr.sl = sl;  // Service Level
  qpAttr.ah_attr.src_path_bits = 0;
  qpAttr.ah_attr.port_num = info->ib_port;
  
  TRACE(NCCL_NET, "NET/IB : ncclIbRtrQp qpn=%u mtu=%d dst=%u ll=%u port=%u sl: %d tc: %d", qp->qp_num, info->mtu, dest_qp_num, info->link_layer, info->ib_port, qpAttr.ah_attr.sl, qpAttr.ah_attr.grh.traffic_class);
  
  NCCLCHECK(wrap_ibv_modify_qp(qp, &qpAttr, IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER));
  
  return ncclSuccess;
}

// 将 QP 转换为 RTS (Ready to Send) 状态
// 功能：配置超时和重试参数，准备发送数据
ncclResult_t ncclIbRtsQp(struct ibv_qp* qp) {
  struct ibv_qp_attr qpAttr;
  memset(&qpAttr, 0, sizeof(struct ibv_qp_attr));
  
  qpAttr.qp_state = IBV_QPS_RTS;
  qpAttr.timeout = ncclParamIbTimeout();  // 超时（单位：4.096us）
  qpAttr.retry_cnt = ncclParamIbRetryCnt();  // 重试计数
  qpAttr.rnr_retry = 7;  // 接收端未就绪重试
  qpAttr.sq_psn = 0;
  qpAttr.max_rd_atomic = 1;
  
  NCCLCHECK(wrap_ibv_modify_qp(qp, &qpAttr, IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC));
  
  return ncclSuccess;
}

//================================================================================
// 连接管理：Listen, Connect, Accept
//================================================================================

// 监听连接请求
// 功能：创建监听套接字，等待远程连接
ncclResult_t ncclIbListen(void* ctx, int dev, void* opaqueHandle, void** listenComm) {
  ncclResult_t ret = ncclSuccess;
  struct ncclIbListenComm* comm;
  
  NCCLCHECK(ncclCalloc(&comm, 1));
  struct ncclIbHandle* handle = (struct ncclIbHandle*) opaqueHandle;
  
  // 静态断言：确保句柄大小足够
  static_assert(sizeof(struct ncclIbHandle) < NCCL_NET_HANDLE_MAXSIZE, "ncclIbHandle size too large");
  memset(handle, 0, sizeof(struct ncclIbHandle));
  
  comm->dev = dev;
  handle->magic = NCCL_SOCKET_MAGIC;
  
  // 初始化套接字（用于带外通信）
  NCCLCHECKGOTO(ncclSocketInit(&comm->sock, &ncclIbIfAddr, handle->magic, ncclSocketTypeNetIb, NULL, 1), ret, fail);
  
  // 开始监听
  NCCLCHECKGOTO(ncclSocketListen(&comm->sock), ret, fail);
  
  // 获取监听地址（填充到 handle）
  NCCLCHECKGOTO(ncclSocketGetAddr(&comm->sock, &handle->connectAddr), ret, fail);
  
  *listenComm = comm;
  
exit:
  return ret;
  
fail:
  (void)ncclSocketClose(&comm->sock);
  free(comm);
  goto exit;
}

// 连接到远程监听器（主动连接）
// 功能：建立到远程节点的连接，交换连接元数据，创建 QP
// 这是一个状态机实现，通过多次调用完成连接
ncclResult_t ncclIbConnect(void* ctx, int dev, void* opaqueHandle, void** sendComm, ncclNetDeviceHandle_t** /*sendDevComm*/) {
  ncclResult_t ret = ncclSuccess;
  struct ncclIbHandle* handle = (struct ncclIbHandle*) opaqueHandle;
  struct ncclIbCommStage* stage = &handle->stage;
  struct ncclIbSendComm* comm = (struct ncclIbSendComm*)stage->comm;
  int ready;
  uint8_t link_layer = IBV_LINK_LAYER_UNSPECIFIED;
  *sendComm = NULL;

  // 状态机：跳转到相应状态
  if (stage->state == ncclIbCommStateConnect)      
    goto ib_connect_check;
  if (stage->state == ncclIbCommStateSendDevList)  
    goto ib_send_dev_list;
  if (stage->state == ncclIbCommStateRecvDevList)  
    goto ib_recv_dev_list;
  if (stage->state == ncclIbCommStateSend)         
    goto ib_send;
  if (stage->state == ncclIbCommStateConnecting)   
    goto ib_connect;
  if (stage->state == ncclIbCommStateConnected)    
    goto ib_send_ready;
  if (stage->state != ncclIbCommStateStart) {
    WARN("Error: trying to connect already connected sendComm");
    return ncclInternalError;
  }
  
  // 初始化
  stage->buffer = NULL;

  NCCLCHECK(ncclIbMalloc((void**)&comm, sizeof(struct ncclIbSendComm)));
  NCCLCHECKGOTO(ncclIbStatsInit(&comm->base.stats), ret, fail);
  
  // 初始化套接字并连接到远程地址
  NCCLCHECKGOTO(ncclSocketInit(&comm->base.sock, &handle->connectAddr, handle->magic, ncclSocketTypeNetIb, NULL, 1), ret, fail);
  stage->comm = comm;
  stage->state = ncclIbCommStateConnect;
  NCCLCHECKGOTO(ncclSocketConnect(&comm->base.sock), ret, fail);

ib_connect_check:
  // 检查套接字连接是否完成（异步操作）
  NCCLCHECKGOTO(ncclSocketReady(&comm->base.sock, &ready), ret, fail);
  if (!ready) return ncclSuccess;  // 连接未完成，稍后再试

  // IB 设置
  struct ncclIbMergedDev* mergedDev;
  if (dev >= ncclNMergedIbDevs) {
    WARN("NET/IB : Trying to use non-existent virtual device %d", dev);
    return ncclInternalError;
  }

  mergedDev = ncclIbMergedDevs + dev;
  comm->base.vProps = mergedDev->vProps;
  comm->base.isSend = true;
  
  stage->state = ncclIbCommStateSendDevList;
  stage->offset = 0;
  
  struct ncclIbConnectionMetadata meta;
  NCCLCHECKGOTO(ncclIbMalloc((void**)&stage->buffer, sizeof(meta)), ret, fail);
  
  // 发送本地虚拟设备属性
  memcpy(stage->buffer, &mergedDev->vProps, sizeof(ncclNetVDeviceProps_t));

// 在设备数量不匹配的情况下，确保连接双方有相同数量的 RC QP
ib_send_dev_list:
  NCCLCHECK(ncclSocketProgress(NCCL_SOCKET_SEND, &comm->base.sock, stage->buffer, sizeof(ncclNetVDeviceProps_t), &stage->offset));
  if (stage->offset != sizeof(ncclNetVDeviceProps_t)) return ncclSuccess;  // 发送未完成

  stage->state = ncclIbCommStateRecvDevList;
  stage->offset = 0;

ib_recv_dev_list:
  // 接收远程虚拟设备属性
  NCCLCHECK(ncclSocketProgress(NCCL_SOCKET_RECV, &comm->base.sock, stage->buffer, sizeof(ncclNetVDeviceProps_t), &stage->offset));
  if (stage->offset != sizeof(ncclNetVDeviceProps_t)) return ncclSuccess;  // 接收未完成
  
  stage->offset = 0;
  ncclNetVDeviceProps_t remoteVProps;
  ncclNetCommConfig_t* config;
  memcpy(&remoteVProps, stage->buffer, sizeof(ncclNetVDeviceProps_t));
  
  mergedDev = ncclIbMergedDevs + dev;
  comm->base.vProps = mergedDev->vProps;
  
  // 确定本地和远程的 QP 数量，取最大值
  int localNqps, remoteNqps;
  localNqps  = ncclParamIbQpsPerConn() * comm->base.vProps.ndevs;  // 每个设备至少 1 个 QP
  remoteNqps = ncclParamIbQpsPerConn() * remoteVProps.ndevs;
  comm->base.nqps = remoteNqps > localNqps ? remoteNqps : localNqps;  // 选择最大值

  // 为每个 IB 设备初始化 PD 和 CQ
  comm->ar = 1;  // 假设启用自适应路由
  for (int i = 0; i < comm->base.vProps.ndevs; i++) {
    int ibDevN = comm->base.vProps.devs[i];
    NCCLCHECKGOTO(ncclIbInitCommDevBase(ibDevN, &comm->devs[i].base, &comm->base.stats), ret, fail);
    comm->ar = comm->ar && ncclIbDevs[ibDevN].ar;  // 仅当所有设备都启用时才使用
  }

  // 准备连接元数据
  memset(&meta, 0, sizeof(meta));
  meta.ndevs = comm->base.vProps.ndevs;

  // 在设备间交替分配 QP（负载均衡）
  int devIndex;
  devIndex = 0;
  for (int q = 0; q < comm->base.nqps; q++) {
    ncclIbSendCommDev* commDev = comm->devs + devIndex;
    ncclIbDev* ibDev = ncclIbDevs + commDev->base.ibDevN;
    
    // 创建 QP
    NCCLCHECKGOTO(ncclIbCreateQp(ibDev->portNum, &commDev->base, IBV_ACCESS_REMOTE_WRITE, &comm->base.stats, comm->base.qps + q), ret, fail);
    comm->base.qps[q].devIndex = devIndex;
    meta.qpInfo[q].qpn      = comm->base.qps[q].qp->qp_num;
    meta.qpInfo[q].devIndex = comm->base.qps[q].devIndex;

    // 查询 ECE 能力（增强连接建立）
    if (ncclParamIbEceEnable()) {
      NCCLCHECKGOTO(wrap_ibv_query_ece(comm->base.qps[q].qp, &meta.qpInfo[q].ece, &meta.qpInfo[q].ece_supported), ret, fail);
    } else {
      meta.qpInfo[q].ece_supported = 0;
    }
    
    devIndex = (devIndex + 1) % comm->base.vProps.ndevs;
  }

  // 为每个设备准备元数据
  for (int i = 0; i < comm->base.vProps.ndevs; i++) {
    ncclIbSendCommDev* commDev = comm->devs + i;
    ncclIbDev* ibDev = ncclIbDevs + commDev->base.ibDevN;

    // 写入元数据结构的指针
    ncclIbDevInfo* devInfo = meta.devs + i;
    devInfo->ib_port       = ibDev->portNum;
    devInfo->mtu           = ibDev->portAttr.active_mtu;
    devInfo->lid           = ibDev->portAttr.lid;

    // 准备本地 FIFO（用于接收远程接收通知）
    NCCLCHECKGOTO(wrap_ibv_reg_mr(&commDev->fifoMr, commDev->base.pd, comm->fifo, sizeof(struct ncclIbSendFifo)*MAX_REQUESTS*NCCL_NET_IB_MAX_RECVS, IBV_ACCESS_LOCAL_WRITE|IBV_ACCESS_REMOTE_WRITE|IBV_ACCESS_REMOTE_READ), ret, fail);
    devInfo->fifoRkey = commDev->fifoMr->rkey;

    // 打包本地 GID 信息
    devInfo->link_layer = commDev->base.gidInfo.link_layer = ibDev->portAttr.link_layer;
    NCCLCHECKGOTO(ncclIbGetGidIndex(ibDev->context, ibDev->portNum, &ibDev->portAttr, &commDev->base.gidInfo.localGidIndex), ret, fail);
    NCCLCHECKGOTO(wrap_ibv_query_gid(ibDev->context, ibDev->portNum, commDev->base.gidInfo.localGidIndex, &commDev->base.gidInfo.localGid), ret, fail);
    devInfo->gid.global.subnet_prefix = commDev->base.gidInfo.localGid.global.subnet_prefix;
    devInfo->gid.global.interface_id = commDev->base.gidInfo.localGid.global.interface_id;

    // 信息日志（记录每个 QP 的详细信息）
    for (int q = 0; q < comm->base.nqps; q++) {
      // 仅打印此设备的 QP
      if (comm->base.qps[q].devIndex == i) {
        if (devInfo->link_layer == IBV_LINK_LAYER_INFINIBAND) {  // IB
          INFO(NCCL_NET,"NET/IB: %s %d IbDev %d Port %d qpn %d mtu %d LID %d subnet-prefix %lu  FLID %d fifoRkey=0x%x fifoLkey=0x%x",
               comm->base.vProps.ndevs > 2 ? "NCCL MergedDev" : "NCCL Dev",
               dev, commDev->base.ibDevN, ibDev->portNum, meta.qpInfo[q].qpn, devInfo->mtu, devInfo->lid,
               devInfo->gid.global.subnet_prefix, ncclIbExtractFlid(&devInfo->gid), devInfo->fifoRkey, commDev->fifoMr->lkey);
        } else {  // RoCE
          INFO(NCCL_NET,"NET/IB: %s %d IbDev %d Port %d qpn %d mtu %d GID %ld (%lX/%lX) fifoRkey=0x%x fifoLkey=0x%x",
               comm->base.vProps.ndevs > 2 ? "NCCL MergedDev" : "NCCL Dev", dev,
               commDev->base.ibDevN, ibDev->portNum, meta.qpInfo[q].qpn, devInfo->mtu,
               (int64_t)commDev->base.gidInfo.localGidIndex,
               devInfo->gid.global.subnet_prefix, devInfo->gid.global.interface_id, devInfo->fifoRkey, commDev->fifoMr->lkey);
        }
        // 记录 ECE 信息
        if (meta.qpInfo[q].ece_supported) {
          INFO(NCCL_NET,"NET/IB: IbDev %d Port %d qpn %d query_ece={supported=%d, vendor_id=0x%x, options=0x%x, comp_mask=0x%x}",
               commDev->base.ibDevN, ibDev->portNum, meta.qpInfo[q].qpn,
               meta.qpInfo[q].ece_supported, meta.qpInfo[q].ece.vendor_id, meta.qpInfo[q].ece.options, meta.qpInfo[q].ece.comp_mask);
        }
      }
    }
    
    // 检查链路层一致性
    if (link_layer == IBV_LINK_LAYER_UNSPECIFIED) link_layer = devInfo->link_layer;
    if (link_layer != devInfo->link_layer) {
      int ibDev0 = comm->devs[0].base.ibDevN;
      WARN("NET/IB : Attempted to connect incompatible devices: [%d]%s:%d/%s and [%d]%s:%d/%s. Try selecting NICs of only one link type using NCCL_IB_HCA",
           commDev->base.ibDevN, ibDev->devName, ibDev->portNum, NCCL_IB_LLSTR(ibDev->portAttr.link_layer), ibDev0, ncclIbDevs[ibDev0].devName, ncclIbDevs[ibDev0].portNum, NCCL_IB_LLSTR(link_layer));
      return ncclInternalError;
    }
  }
  
  config = (ncclNetCommConfig_t*)ctx;
  meta.fifoAddr = (uint64_t)comm->fifo;
  meta.sl = (ncclParamIbSl() != -1) ? ncclParamIbSl() : (config && config->trafficClass != NCCL_NET_TRAFFIC_CLASS_UNDEF) ? config->trafficClass : NCCL_IB_SL_DEFAULT;
  meta.tc = (ncclParamIbTc() != -1) ? ncclParamIbTc() : (config && config->trafficClass != NCCL_NET_TRAFFIC_CLASS_UNDEF) ? config->trafficClass : NCCL_IB_TC_DEFAULT;
  strncpy(meta.devName, mergedDev->devName, MAX_MERGED_DEV_NAME);

  stage->state = ncclIbCommStateSend;
  stage->offset = 0;

  memcpy(stage->buffer, &meta, sizeof(meta));

ib_send:
  // 发送本地连接元数据
  NCCLCHECKGOTO(ncclSocketProgress(NCCL_SOCKET_SEND, &comm->base.sock, stage->buffer, sizeof(meta), &stage->offset), ret, fail);
  if (stage->offset != sizeof(meta)) return ncclSuccess;  // 发送未完成

  stage->state = ncclIbCommStateConnecting;
  stage->offset = 0;
  // 清空临时缓冲区以重用
  memset(stage->buffer, 0, sizeof(meta));

ib_connect:
  // 接收远程连接元数据
  struct ncclIbConnectionMetadata remMeta;
  NCCLCHECKGOTO(ncclSocketProgress(NCCL_SOCKET_RECV, &comm->base.sock, stage->buffer, sizeof(ncclIbConnectionMetadata), &stage->offset), ret, fail);
  if (stage->offset != sizeof(remMeta)) return ncclSuccess;  // 接收未完成

  memcpy(&remMeta, stage->buffer, sizeof(ncclIbConnectionMetadata));

  comm->base.nRemDevs = remMeta.ndevs;

  // 确保远程设备与本地设备使用相同的链路层
  if (comm->base.vProps.ndevs > 0) {
    int ibDev0 = comm->devs[0].base.ibDevN;
    link_layer = ncclIbDevs[ibDev0].portAttr.link_layer;
    for (int i = 0; i < remMeta.ndevs; i++) {
      if (remMeta.devs[i].link_layer != link_layer) {
        WARN("NET/IB : Remote %s device is incompatible with the local [%d]%s:%d/%s. Try selecting NICs of only one link type using NCCL_IB_HCA",
             NCCL_IB_LLSTR(remMeta.devs[i].link_layer), ibDev0, ncclIbDevs[ibDev0].devName, ncclIbDevs[ibDev0].portNum, NCCL_IB_LLSTR(link_layer));
        return ncclInternalError;
      }
    }
  }

  // 复制远程设备信息（用于 RDMA 操作）
  for (int i = 0; i < remMeta.ndevs; i++) {
    comm->base.remDevs[i] = remMeta.devs[i];
    comm->base.remDevs[i].remoteGid.global.interface_id = comm->base.remDevs[i].gid.global.interface_id;
    comm->base.remDevs[i].remoteGid.global.subnet_prefix = comm->base.remDevs[i].gid.global.subnet_prefix;

    // 保留远程大小 FIFO 信息并准备 RDMA 操作
    comm->remSizesFifo.rkeys[i] = remMeta.devs[i].fifoRkey;
    comm->remSizesFifo.addr = remMeta.fifoAddr;
  }

  // 为远程大小 FIFO 注册内存
  for (int i=0; i < comm->base.vProps.ndevs; i++) {
    NCCLCHECKGOTO(wrap_ibv_reg_mr(comm->remSizesFifo.mrs+i, comm->devs[i].base.pd, &comm->remSizesFifo.elems, sizeof(int)*MAX_REQUESTS*NCCL_NET_IB_MAX_RECVS, IBV_ACCESS_REMOTE_WRITE|IBV_ACCESS_LOCAL_WRITE|IBV_ACCESS_REMOTE_READ), ret, fail);
  }
  comm->base.nRemDevs = remMeta.ndevs;

  // 将所有 QP 转换为 RTR 和 RTS 状态
  for (int q = 0; q < comm->base.nqps; q++) {
    struct ncclIbQpInfo* remQpInfo   = remMeta.qpInfo + q;
    struct ncclIbDevInfo* remDevInfo = remMeta.devs + remQpInfo->devIndex;

    // 分配每个 QP 的远程设备
    comm->base.qps[q].remDevIdx = remQpInfo->devIndex;
    int devIndex = comm->base.qps[q].devIndex;
    ncclIbSendCommDev* commDev = comm->devs + devIndex;

    struct ibv_qp* qp = comm->base.qps[q].qp;
    
    // 设置 ECE（增强连接建立）
    if (remQpInfo->ece_supported) {
      struct ncclIbQp* nqp = comm->base.qps + q;
      int ibDevN = comm->devs[nqp->devIndex].base.ibDevN;
      struct ncclIbDev* ibDev = ncclIbDevs + ibDevN;
      INFO(NCCL_NET,"NET/IB: IbDev %d Port %d qpn %d set_ece={supported=%d, vendor_id=0x%x, options=0x%x, comp_mask=0x%x}",
        ibDevN, ibDev->portNum, qp->qp_num, remMeta.qpInfo[q].ece_supported, remMeta.qpInfo[q].ece.vendor_id, remMeta.qpInfo[q].ece.options, remMeta.qpInfo[q].ece.comp_mask);
      NCCLCHECKGOTO(wrap_ibv_set_ece(qp, &remQpInfo->ece, &remQpInfo->ece_supported), ret, fail);
    }

    ncclIbDev* ibDev = ncclIbDevs + commDev->base.ibDevN;
    remDevInfo->mtu = std::min(remDevInfo->mtu, ibDev->portAttr.active_mtu);
    NCCLCHECKGOTO(ncclIbRtrQp(qp, &commDev->base.gidInfo, remQpInfo->qpn, remDevInfo, false, remMeta.tc, remMeta.sl), ret, fail);
    NCCLCHECKGOTO(ncclIbRtsQp(qp), ret, fail);
  }

  comm->base.nDataQps = std::max(comm->base.vProps.ndevs, comm->base.nRemDevs);

  comm->base.ready = 1;
  stage->state = ncclIbCommStateConnected;
  stage->offset = 0;

ib_send_ready:
  // 发送 ready 信号
  NCCLCHECKGOTO(ncclSocketProgress(NCCL_SOCKET_SEND, &comm->base.sock, &comm->base.ready, sizeof(int), &stage->offset), ret, fail);
  if (stage->offset != sizeof(int)) return ncclSuccess;

  *sendComm = comm;
  
exit:
  if (stage->buffer) free(stage->buffer);
  stage->state = ncclIbCommStateStart;
  return ret;
  
fail:
  free(comm);
  goto exit;
}

// 参数：是否警告本地 rail
NCCL_PARAM(IbWarnRailLocal, "IB_WARN_RAIL_LOCAL", 0);

// 检查虚拟设备属性
// 功能：验证两个虚拟设备列表的兼容性，找到交集
ncclResult_t ncclIbCheckVProps(ncclNetVDeviceProps_t* vProps1, ncclNetVDeviceProps_t* vProps2) {
  ncclNetVDeviceProps_t  outVProps = {0};
  ncclNetVDeviceProps_t* minVProps = vProps2;
  ncclNetVDeviceProps_t* maxVProps = vProps1;
  
  if (vProps2->ndevs > vProps1->ndevs) {
    minVProps = vProps1;
    maxVProps = vProps2;
  }

  // 找到设备的交集
  for (int i = 0; i < minVProps->ndevs; i++) {
    int dev = minVProps->devs[i];
    for (int j = 0; j < maxVProps->ndevs; j++) {
      // 找到匹配
      if (maxVProps->devs[j] == dev) {
        outVProps.devs[outVProps.ndevs++] = dev;
      }
    }
  }

  // 如果至少一方有融合 NIC 但没有匹配的物理 NIC，检查用户是否想要这样
  if (ncclParamIbWarnRailLocal() && outVProps.ndevs < maxVProps->ndevs) {
    char local[128];
    int cursor = 1;
    snprintf(local, sizeof(local), "%d", vProps1->devs[0]);
    for (int i = 1; i < vProps1->ndevs; i++) {
      snprintf(local+cursor, sizeof(local)-cursor, ",%d", vProps1->devs[i]);
      cursor += 2;
    }
    char remote[128];
    snprintf(remote, sizeof(remote), "%d", vProps2->devs[0]);
    cursor = 1;
    for (int i = 1; i < vProps2->ndevs; i++) {
      snprintf(remote+cursor, sizeof(remote)-cursor, ",%d", vProps2->devs[i]);
      cursor += 2;
    }
    INFO(NCCL_NET, "NET/IB : There are mismatched physical devices between local (%s) and remote (%s). To disable this warning, set NCCL_IB_WARN_RAIL_LOCAL=0", local, remote);
  }

  return ncclSuccess;
}

// 参数：是否禁用 GDR flush
NCCL_PARAM(IbGdrFlushDisable, "GDR_FLUSH_DISABLE", 0);

// 接受连接（被动连接）
// 功能：接受远程连接请求，交换元数据，创建 QP
ncclResult_t ncclIbAccept(void* listenComm, void** recvComm, ncclNetDeviceHandle_t** /*recvDevComm*/) {
  ncclResult_t ret = ncclSuccess;
  struct ncclIbListenComm* lComm = (struct ncclIbListenComm*)listenComm;
  struct ncclIbCommStage* stage = &lComm->stage;
  struct ncclIbRecvComm* rComm = (struct ncclIbRecvComm*)stage->comm;
  int ready;
  int link_layer = IBV_LINK_LAYER_UNSPECIFIED;
  *recvComm = NULL;

  // 状态机：跳转到相应状态
  if (stage->state == ncclIbCommStateAccept)   
    goto ib_accept_check;
  if (stage->state == ncclIbCommStateRecvDevList) 
    goto ib_recv_dev_list;
  if (stage->state == ncclIbCommStateSendDevList) 
    goto ib_send_dev_list;
  if (stage->state == ncclIbCommStateRecv) 
    goto ib_recv;
  if (stage->state == ncclIbCommStateSend) 
    goto ib_send;
  if (stage->state == ncclIbCommStatePendingReady) 
    goto ib_recv_ready;
  if (stage->state != ncclIbCommStateStart) {
    WARN("Listencomm in unknown state %d", stage->state);
    return ncclInternalError;
  }

  // 初始化
  NCCLCHECK(ncclIbMalloc((void**)&rComm, sizeof(struct ncclIbRecvComm)));
  NCCLCHECKGOTO(ncclIbStatsInit(&rComm->base.stats), ret, fail);
  stage->comm = rComm;
  stage->state = ncclIbCommStateAccept;
  
  NCCLCHECKGOTO(ncclSocketInit(&rComm->base.sock), ret, fail);
  NCCLCHECKGOTO(ncclSocketAccept(&rComm->base.sock, &lComm->sock), ret, fail);

  // 分配 stage->buffer 用于后续所有步骤
  struct ncclIbConnectionMetadata remMeta;
  stage->offset = 0;
  NCCLCHECK(ncclIbMalloc((void**)&stage->buffer, sizeof(remMeta)));

ib_accept_check:
  // 检查套接字是否就绪
  NCCLCHECKGOTO(ncclSocketReady(&rComm->base.sock, &ready), ret, fail);
  if (!ready) return ncclSuccess;  // 未就绪，稍后再试
  
  stage->state = ncclIbCommStateRecvDevList;
  stage->offset = 0;

// 在设备数量不匹配的情况下，确保连接双方有相同数量的 RC QP
ib_recv_dev_list:
  // 接收远程虚拟设备属性
  NCCLCHECK(ncclSocketProgress(NCCL_SOCKET_RECV, &rComm->base.sock, stage->buffer, sizeof(ncclNetVDeviceProps_t), &stage->offset));
  if (stage->offset != sizeof(ncclNetVDeviceProps_t)) return ncclSuccess;  // 接收未完成
  
  ncclNetVDeviceProps_t remoteVProps;
  memcpy(&remoteVProps, stage->buffer, sizeof(ncclNetVDeviceProps_t));
  
  if (lComm->dev >= ncclNMergedIbDevs) {
    WARN("NET/IB : Trying to use non-existent virtual device %d", lComm->dev);
    return ncclInternalError;
  }

  // 减少物理设备列表并存储在连接基础结构中
  struct ncclIbMergedDev* mergedDev;
  mergedDev = ncclIbMergedDevs + lComm->dev;
  NCCLCHECK(ncclIbCheckVProps(&mergedDev->vProps, &remoteVProps));
  rComm->base.vProps = mergedDev->vProps;
  memcpy(stage->buffer, &rComm->base.vProps, sizeof(ncclNetVDeviceProps_t));
  rComm->base.isSend = false;
  
  // 确定本地和远程的 QP 数量
  int localNqps, remoteNqps;
  localNqps  = ncclParamIbQpsPerConn() * rComm->base.vProps.ndevs;
  remoteNqps = ncclParamIbQpsPerConn() * remoteVProps.ndevs;
  rComm->base.nqps = remoteNqps > localNqps ? remoteNqps : localNqps;

  stage->offset = 0;
  stage->state = ncclIbCommStateSendDevList;

ib_send_dev_list:
  // 发送本地虚拟设备属性
  NCCLCHECKGOTO(ncclSocketProgress(NCCL_SOCKET_SEND, &rComm->base.sock, stage->buffer, sizeof(ncclNetVDeviceProps_t), &stage->offset), ret, fail);
  if (stage->offset != sizeof(ncclNetVDeviceProps_t)) return ncclSuccess;  // 发送未完成

  stage->offset = 0;
  stage->state = ncclIbCommStateRecv;

ib_recv:
  // 接收远程连接元数据
  NCCLCHECKGOTO(ncclSocketProgress(NCCL_SOCKET_RECV, &rComm->base.sock, stage->buffer, sizeof(remMeta), &stage->offset), ret, fail);
  if (stage->offset != sizeof(remMeta)) return ncclSuccess;  // 接收未完成

  // 复制接收到的信息
  memcpy(&remMeta, stage->buffer, sizeof(struct ncclIbConnectionMetadata));

  // IB 设置
  struct ncclIbDev* ibDev;
  int ibDevN;
  struct ncclIbRecvCommDev* rCommDev;
  struct ncclIbDevInfo* remDevInfo;
  struct ncclIbQp* qp;

  mergedDev = ncclIbMergedDevs + lComm->dev;
  rComm->base.nRemDevs = remMeta.ndevs;
  
  if (rComm->base.nRemDevs != rComm->base.vProps.ndevs) {
    INFO(NCCL_NET, "NET/IB : Local mergedDev %s has a different number of devices=%d as remote %s %d",
      mergedDev->devName, rComm->base.vProps.ndevs, remMeta.devName, rComm->base.nRemDevs);
  }

  // 发送回请求者的元数据
  struct ncclIbConnectionMetadata meta;
  memset(&meta, 0, sizeof(meta));
  
  for (int i = 0; i < rComm->base.vProps.ndevs; i++) {
    rCommDev = rComm->devs + i;
    ibDevN = rComm->base.vProps.devs[i];
    NCCLCHECKGOTO(ncclIbInitCommDevBase(ibDevN, &rCommDev->base, &rComm->base.stats), ret, fail);
    ibDev = ncclIbDevs + ibDevN;
    NCCLCHECKGOTO(ncclIbGetGidIndex(ibDev->context, ibDev->portNum, &ibDev->portAttr, &rCommDev->base.gidInfo.localGidIndex), ret, fail);
    NCCLCHECKGOTO(wrap_ibv_query_gid(ibDev->context, ibDev->portNum, rCommDev->base.gidInfo.localGidIndex, &rCommDev->base.gidInfo.localGid), ret, fail);
    
    if (link_layer == IBV_LINK_LAYER_UNSPECIFIED) link_layer = ibDev->portAttr.link_layer;
    if (link_layer != ibDev->portAttr.link_layer) {
      int ibDev0 = rComm->devs[0].base.ibDevN;
      WARN("NET/IB : Attempted to connect incompatible devices: [%d]%s:%d/%s and [%d]%s:%d/%s. Try selecting NICs of only one link type using NCCL_IB_HCA",
           ibDevN, ibDev->devName, ibDev->portNum, NCCL_IB_LLSTR(ibDev->portAttr.link_layer), ibDev0, ncclIbDevs[ibDev0].devName, ncclIbDevs[ibDev0].portNum, NCCL_IB_LLSTR(link_layer));
      return ncclInternalError;
    }
  }

  // 复制远程设备信息
  for (int i = 0; i < remMeta.ndevs; i++) {
    rComm->base.remDevs[i] = remMeta.devs[i];
    rComm->base.remDevs[i].remoteGid.global.interface_id  = rComm->base.remDevs[i].gid.global.interface_id;
    rComm->base.remDevs[i].remoteGid.global.subnet_prefix = rComm->base.remDevs[i].gid.global.subnet_prefix;
    
    if (remMeta.devs[i].link_layer != link_layer) {
      int ibDev0 = rComm->devs[0].base.ibDevN;
      WARN("NET/IB : Remote %s device is incompatible with the local [%d]%s:%d/%s. Try selecting NICs of only one link type using NCCL_IB_HCA",
           NCCL_IB_LLSTR(remMeta.devs[i].link_layer), ibDev0, ncclIbDevs[ibDev0].devName, ncclIbDevs[ibDev0].portNum, NCCL_IB_LLSTR(link_layer));
      return ncclInternalError;
    }
  }

  // 在合并设备间条带化创建 QP
  // 确保获取正确的远程对等设备和 QP 信息
  int remDevIndex;
  int devIndex;
  devIndex = 0;
  
  for (int q = 0; q < rComm->base.nqps; q++) {
    remDevIndex = remMeta.qpInfo[q].devIndex;
    remDevInfo = remMeta.devs + remDevIndex;
    qp = rComm->base.qps+q;
    rCommDev = rComm->devs + devIndex;
    qp->remDevIdx = remDevIndex;

    // 本地 ibDevN
    ibDevN = rComm->devs[devIndex].base.ibDevN;
    ibDev = ncclIbDevs + ibDevN;
    NCCLCHECKGOTO(ncclIbCreateQp(ibDev->portNum, &rCommDev->base, IBV_ACCESS_REMOTE_WRITE, &rComm->base.stats, qp), ret, fail);
    qp->devIndex = devIndex;
    devIndex = (devIndex + 1) % rComm->base.vProps.ndevs;

    // 在 RTR 之前在此 QP 上设置 ECE
    if (remMeta.qpInfo[q].ece_supported) {
      // Coverity 怀疑下面的复制粘贴错误，因为 remMeta 和 mix 出现在不同参数中
      // 然而，这已确认为有意为之
      NCCLCHECKGOTO(wrap_ibv_set_ece(qp->qp, &remMeta.qpInfo[q].ece, &meta.qpInfo[q].ece_supported), ret, fail);
    } else {
      meta.qpInfo[q].ece_supported = 0;
    }

    // 转换为 RTR 和 RTS 状态
    NCCLCHECKGOTO(ncclIbRtrQp(qp->qp, &rCommDev->base.gidInfo, remMeta.qpInfo[q].qpn, remDevInfo, true, remMeta.tc, remMeta.sl), ret, fail);
    NCCLCHECKGOTO(ncclIbRtsQp(qp->qp), ret, fail);

    // 查询此 QP 的减少 ECE（请求者和响应者之间的匹配增强）
    // 存储在我们自己的 qpInfo 中以返回给请求者
    if (remMeta.qpInfo[q].ece_supported && meta.qpInfo[q].ece_supported) {
      NCCLCHECKGOTO(wrap_ibv_query_ece(qp->qp, &meta.qpInfo[q].ece, &meta.qpInfo[q].ece_supported), ret, fail);
    }
  }

  // 确定是否启用 GPU Direct RDMA flush
  rComm->flushEnabled = ((ncclIbGdrSupport() == ncclSuccess || ncclIbDmaBufSupport(lComm->dev) == ncclSuccess)
                            && (ncclParamIbGdrFlushDisable() == 0)) ? 1 : 0;

  for (int i = 0; i < rComm->base.vProps.ndevs; i++) {
    rCommDev = rComm->devs + i;
    ibDev = ncclIbDevs + rCommDev->base.ibDevN;

    // 保留远程 FIFO 信息并准备本地 RDMA 操作
    rComm->remFifo.addr = remMeta.fifoAddr;
    NCCLCHECKGOTO(wrap_ibv_reg_mr(&rCommDev->fifoMr, rCommDev->base.pd, &rComm->remFifo.elems, sizeof(struct ncclIbSendFifo)*MAX_REQUESTS*NCCL_NET_IB_MAX_RECVS, IBV_ACCESS_REMOTE_WRITE|IBV_ACCESS_LOCAL_WRITE|IBV_ACCESS_REMOTE_READ), ret, fail);
    rCommDev->fifoSge.lkey = rCommDev->fifoMr->lkey;
    if (ncclParamIbUseInline()) rComm->remFifo.flags = IBV_SEND_INLINE;

    // 为 GPU Direct RDMA 分配 Flush 虚拟缓冲区
    if (rComm->flushEnabled) {
      NCCLCHECKGOTO(wrap_ibv_reg_mr(&rCommDev->gpuFlush.hostMr, rCommDev->base.pd, &rComm->gpuFlushHostMem, sizeof(int), IBV_ACCESS_LOCAL_WRITE), ret, fail);
      rCommDev->gpuFlush.sge.addr = (uint64_t)&rComm->gpuFlushHostMem;
      rCommDev->gpuFlush.sge.length = 1;
      rCommDev->gpuFlush.sge.lkey = rCommDev->gpuFlush.hostMr->lkey;
      
      // 为 flush 创建专用 QP
      NCCLCHECKGOTO(ncclIbCreateQp(ibDev->portNum, &rCommDev->base, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ, &rComm->base.stats, &rCommDev->gpuFlush.qp), ret, fail);
      
      struct ncclIbDevInfo devInfo;
      devInfo.lid         = ibDev->portAttr.lid;
      devInfo.link_layer  = ibDev->portAttr.link_layer;
      devInfo.ib_port     = ibDev->portNum;
      devInfo.gid.global.subnet_prefix        = rCommDev->base.gidInfo.localGid.global.subnet_prefix;
      devInfo.gid.global.interface_id         = rCommDev->base.gidInfo.localGid.global.interface_id;
      devInfo.mtu         = ibDev->portAttr.active_mtu;
      NCCLCHECKGOTO(ncclIbRtrQp(rCommDev->gpuFlush.qp.qp, &rCommDev->base.gidInfo, rCommDev->gpuFlush.qp.qp->qp_num, &devInfo, false, remMeta.tc, remMeta.sl), ret, fail);
      NCCLCHECKGOTO(ncclIbRtsQp(rCommDev->gpuFlush.qp.qp), ret, fail);
    }

    // 填充句柄
    meta.devs[i].lid                            = ibDev->portAttr.lid;
    meta.devs[i].link_layer                     = rCommDev->base.gidInfo.link_layer = ibDev->portAttr.link_layer;
    meta.devs[i].ib_port                        = ibDev->portNum;
    meta.devs[i].gid.global.subnet_prefix       = rCommDev->base.gidInfo.localGid.global.subnet_prefix;
    meta.devs[i].gid.global.interface_id        = rCommDev->base.gidInfo.localGid.global.interface_id;
    meta.devs[i].mtu                            = ibDev->portAttr.active_mtu;

    // 准备大小 FIFO
    NCCLCHECKGOTO(wrap_ibv_reg_mr(&rComm->devs[i].sizesFifoMr, rComm->devs[i].base.pd, rComm->sizesFifo, sizeof(int)*MAX_REQUESTS*NCCL_NET_IB_MAX_RECVS, IBV_ACCESS_LOCAL_WRITE|IBV_ACCESS_REMOTE_WRITE|IBV_ACCESS_REMOTE_READ), ret, fail);
    meta.devs[i].fifoRkey = rComm->devs[i].sizesFifoMr->rkey;
  }
  
  meta.fifoAddr = (uint64_t)rComm->sizesFifo;
  meta.sl = remMeta.sl;
  meta.tc = remMeta.tc;

  for (int q = 0; q < rComm->base.nqps; q++) {
    meta.qpInfo[q].qpn      = rComm->base.qps[q].qp->qp_num;
    meta.qpInfo[q].devIndex = rComm->base.qps[q].devIndex;
  }
  meta.ndevs = rComm->base.vProps.ndevs;
  strncpy(meta.devName, mergedDev->devName, MAX_MERGED_DEV_NAME);
  rComm->base.nDataQps = std::max(rComm->base.vProps.ndevs, rComm->base.nRemDevs);

  stage->state = ncclIbCommStateSend;
  stage->offset = 0;
  if (stage->buffer) {
    free(stage->buffer);
    stage->buffer = NULL;
  }
  NCCLCHECKGOTO(ncclIbMalloc((void**)&stage->buffer, sizeof(struct ncclIbConnectionMetadata)), ret, fail);
  memcpy(stage->buffer, &meta, sizeof(struct ncclIbConnectionMetadata));

ib_send:
  // 发送本地连接元数据
  NCCLCHECKGOTO(ncclSocketProgress(NCCL_SOCKET_SEND, &rComm->base.sock, stage->buffer, sizeof(struct ncclIbConnectionMetadata), &stage->offset), ret, fail);
  if (stage->offset < sizeof(struct ncclIbConnectionMetadata)) return ncclSuccess;  // 发送未完成

  stage->offset = 0;
  stage->state = ncclIbCommStatePendingReady;

ib_recv_ready:
  // 接收 ready 信号
  NCCLCHECKGOTO(ncclSocketProgress(NCCL_SOCKET_RECV,  &rComm->base.sock, &rComm->base.ready, sizeof(int), &stage->offset), ret, fail);
  if (stage->offset != sizeof(int)) return ncclSuccess;  // 接收未完成

  *recvComm = rComm;
  
exit:
  // 重置 lComm stage
  if (stage->buffer) free(stage->buffer);
  stage->state = ncclIbCommStateStart;
  stage->offset = 0;
  stage->comm = NULL;
  stage->buffer = NULL;
  return ret;
  
fail:
  free(rComm);
  goto exit;
}

//================================================================================
// 请求管理
//================================================================================

// 获取空闲请求
ncclResult_t ncclIbGetRequest(struct ncclIbNetCommBase* base, struct ncclIbRequest** req) {
  for (int i=0; i<MAX_REQUESTS; i++) {
    struct ncclIbRequest* r = base->reqs+i;
    if (r->type == NCCL_NET_IB_REQ_UNUSED) {
      r->base = base;
      r->sock = NULL;
      memset(r->devBases, 0, sizeof(r->devBases));
      memset(r->events, 0, sizeof(r->events));
      *req = r;
      return ncclSuccess;
    }
  }
  WARN("NET/IB : unable to allocate requests");
  *req = NULL;
  return ncclInternalError;
}

// 释放请求（标记为未使用）
ncclResult_t ncclIbFreeRequest(struct ncclIbRequest* r) {
  r->type = NCCL_NET_IB_REQ_UNUSED;
  return ncclSuccess;
}

// 前向声明
ncclResult_t ncclIbTest(void* request, int* done, int* size);

//================================================================================
// 内存注册（MR）管理
//================================================================================

// 内存注册（内部函数）
// 功能：注册内存区域用于 RDMA，使用缓存优化
ncclResult_t ncclIbRegMrDmaBufInternal(ncclIbNetCommDevBase* base, void* data, size_t size, int type, uint64_t offset, int fd, ibv_mr** mhandle) {
  static __thread uintptr_t pageSize = 0;
  if (pageSize == 0) pageSize = sysconf(_SC_PAGESIZE);
  
  struct ncclIbMrCache* cache = &ncclIbDevs[base->ibDevN].mrCache;
  uintptr_t addr = (uintptr_t)data & -pageSize;  // 页对齐
  size_t pages = ((uintptr_t)data + size - addr + pageSize-1)/pageSize;
  
  std::lock_guard<std::mutex> lock(ncclIbDevs[base->ibDevN].mutex);
  
  // 在缓存中查找或创建新条目
  for (int slot=0; /*true*/; slot++) {
    if (slot == cache->population || addr < cache->slots[slot].addr) {  // 未在缓存中找到
      // 必须增长缓存
      if (cache->population == cache->capacity) { 
        cache->capacity = cache->capacity < 32 ? 32 : 2*cache->capacity;
        NCCLCHECK(ncclRealloc(&cache->slots, cache->population, cache->capacity));
      }
      
      // 注册内存
      struct ibv_mr* mr;
      unsigned int flags = IBV_ACCESS_LOCAL_WRITE|IBV_ACCESS_REMOTE_WRITE|IBV_ACCESS_REMOTE_READ;
      
      if (ncclIbRelaxedOrderingEnabled) flags |= IBV_ACCESS_RELAXED_ORDERING;
      
      if (fd != -1) {
        // DMA-BUF 支持
        if (!ncclIbDevs[base->ibDevN].capsProvider.mlx5.dataDirect) {
          NCCLCHECK(wrap_ibv_reg_dmabuf_mr(&mr, base->pd, offset, pages*pageSize, addr, fd, flags));
        } else {
          NCCLCHECK(wrap_mlx5dv_reg_dmabuf_mr(&mr, base->pd, offset, pages*pageSize, addr, fd, flags, MLX5DV_REG_DMABUF_ACCESS_DATA_DIRECT));
        }
      } else {
        // 普通内存注册
        if (ncclIbRelaxedOrderingEnabled) {
          // 使用 IBVERBS_1.8 API - 需要 IBV_ACCESS_RELAXED_ORDERING 支持
          NCCLCHECK(wrap_ibv_reg_mr_iova2(&mr, base->pd, (void*)addr, pages*pageSize, addr, flags));
        } else {
          NCCLCHECK(wrap_ibv_reg_mr(&mr, base->pd, (void*)addr, pages*pageSize, flags));
        }
      }
      
      TRACE(NCCL_INIT|NCCL_NET,"regAddr=0x%lx size=%lld rkey=0x%x lkey=0x%x fd=%d", (unsigned long)addr, (long long)pages*pageSize, mr->rkey, mr->lkey, fd);
      
      if (slot != cache->population) 
        memmove(cache->slots+slot+1, cache->slots+slot, (cache->population-slot)*sizeof(struct ncclIbMr));

      cache->slots[slot].addr = addr;
      cache->slots[slot].pages = pages;
      cache->slots[slot].refs = 1;
      cache->slots[slot].mr = mr;
      cache->population += 1;
      *mhandle = mr;
      return ncclSuccess;
    } else if ((addr >= cache->slots[slot].addr) &&
        ((addr-cache->slots[slot].addr)/pageSize+pages) <= cache->slots[slot].pages) {
      // 找到包含此地址的缓存条目
      cache->slots[slot].refs += 1;
      *mhandle = cache->slots[slot].mr;
      return ncclSuccess;
    }
  }
  return ncclSuccess;
}

// 获取通信设备基础结构的辅助函数
struct ncclIbNetCommDevBase* ncclIbGetNetCommDevBase(ncclIbNetCommBase* base, int devIndex) {
  if (base->isSend) {
    struct ncclIbSendComm* sComm = (struct ncclIbSendComm*) base;
    return &sComm->devs[devIndex].base;
  } else {
    struct ncclIbRecvComm* rComm = (struct ncclIbRecvComm*) base;
    return &rComm->devs[devIndex].base;
  }
}

// DMA-BUF 内存注册（公共接口）
ncclResult_t ncclIbRegMrDmaBuf(void* comm, void* data, size_t size, int type, uint64_t offset, int fd, void** mhandle) {
  ncclResult_t ret = ncclSuccess;
  assert(size > 0);
  struct ncclIbNetCommBase* base = (struct ncclIbNetCommBase*) comm;
  struct ncclIbMrHandle* mhandleWrapper = (struct ncclIbMrHandle*) malloc(sizeof(struct ncclIbMrHandle));
  
  // 为每个设备注册内存
  for (int i = 0; i < base->vProps.ndevs; i++) {
    struct ncclIbNetCommDevBase* devComm = ncclIbGetNetCommDevBase(base, i);
    NCCLCHECKGOTO(ncclIbRegMrDmaBufInternal(devComm, data, size, type, offset, fd, mhandleWrapper->mrs + i), ret, fail);
  }
  *mhandle = (void*) mhandleWrapper;
  
exit:
  return ret;
  
fail:
  free(mhandleWrapper);
  goto exit;
}

// 普通内存注册（无 DMA-BUF）
ncclResult_t ncclIbRegMr(void* comm, void* data, size_t size, int type, void** mhandle) {
  return ncclIbRegMrDmaBuf(comm, data, size, type, 0ULL, -1, mhandle);
}

// 注销内存（内部函数）
ncclResult_t ncclIbDeregMrInternal(ncclIbNetCommDevBase* base, ibv_mr* mhandle) {
  struct ncclIbMrCache* cache = &ncclIbDevs[base->ibDevN].mrCache;
  std::lock_guard<std::mutex> lock(ncclIbDevs[base->ibDevN].mutex);
  
  for (int i=0; i < cache->population; i++) {
    if (mhandle == cache->slots[i].mr) {
      if (0 == --cache->slots[i].refs) {
        // 引用计数为 0，移除并注销
        memmove(&cache->slots[i], &cache->slots[--cache->population], sizeof(struct ncclIbMr));
        if (cache->population == 0) {
          free(cache->slots);
          cache->slots = NULL;
          cache->capacity = 0;
        }
        NCCLCHECK(wrap_ibv_dereg_mr(mhandle));
      }
      return ncclSuccess;
    }
  }
  WARN("NET/IB: could not find mr %p inside cache of %d entries", mhandle, cache->population);
  return ncclInternalError;
}

// 注销内存（公共接口）
ncclResult_t ncclIbDeregMr(void* comm, void* mhandle) {
  if (mhandle == NULL) return ncclSuccess;

  struct ncclIbMrHandle* mhandleWrapper = (struct ncclIbMrHandle*) mhandle;
  struct ncclIbNetCommBase* base = (struct ncclIbNetCommBase*) comm;
  
  for (int i = 0; i < base->vProps.ndevs; i++) {
    struct ncclIbNetCommDevBase* devComm = ncclIbGetNetCommDevBase(base, i);
    NCCLCHECK(ncclIbDeregMrInternal(devComm, mhandleWrapper->mrs[i]));
  }
  free(mhandleWrapper);
  return ncclSuccess;
}

//================================================================================
// 数据传输操作：Isend, Irecv, Test
//================================================================================

// 参数：是否在 QP 上分割数据
NCCL_PARAM(IbSplitDataOnQps, "IB_SPLIT_DATA_ON_QPS", 0);

// 多发送（内部函数）
// 功能：通过多个 QP 发送数据，支持多接收和自适应路由
ncclResult_t ncclIbMultiSend(struct ncclIbSendComm* comm, int slot) {
  struct ncclIbRequest** reqs = comm->fifoReqs[slot];
  volatile struct ncclIbSendFifo* slots = comm->fifo[slot];
  int nreqs = slots[0].nreqs;
  if (nreqs > NCCL_NET_IB_MAX_RECVS) return ncclInternalError;

  uint64_t wr_id = 0ULL;
  
  // 为每个请求准备工作请求（WR）
  for (int r=0; r<nreqs; r++) {
    struct ibv_send_wr* wr = comm->wrs+r;
    memset(wr, 0, sizeof(struct ibv_send_wr));

    struct ibv_sge* sge = comm->sges+r;
    sge->addr=(uintptr_t)reqs[r]->send.data;
    wr->opcode = IBV_WR_RDMA_WRITE;
    wr->send_flags = 0;
    wr->wr.rdma.remote_addr = slots[r].addr;
    wr->next = wr + 1;
    // 将请求 ID 编码到 wr_id 的低 8 位
    wr_id += (reqs[r] - comm->base.reqs) << (r*8);
#ifdef NCCL_ENABLE_NET_PROFILING
    reqs[r]->pInfo[0].nEventHandles = 0;
#endif
  }

  // 将大小作为立即数据写入
  // 在多接收情况下，仅写入 0 或 1 表示是否有数据发送或接收
  uint32_t immData = 0;
  if (nreqs == 1) {
    immData = reqs[0]->send.size;
  } else {
    // 多接收：将大小写入远程 FIFO
    int* sizes = comm->remSizesFifo.elems[slot];
    for (int r=0; r<nreqs; r++) sizes[r] = reqs[r]->send.size;
    comm->remSizesFifo.sge.addr = (uint64_t)sizes;
    comm->remSizesFifo.sge.length = nreqs*sizeof(int);
  }

  // 最后一个 WR：RDMA_WRITE_WITH_IMM（触发接收端完成）
  struct ibv_send_wr* lastWr = comm->wrs+nreqs-1;
  
  // 使用自适应路由时，先发送大部分数据为 RDMA_WRITE，
  // 然后发送 0 字节 RDMA_WRITE_WITH_IMM 触发远程完成
  if (nreqs > 1 || (comm->ar && reqs[0]->send.size > ncclParamIbArThreshold())) {
    lastWr++;
    memset(lastWr, 0, sizeof(struct ibv_send_wr));
    if (nreqs > 1) {
      // 写入远程大小 FIFO
      lastWr->wr.rdma.remote_addr = comm->remSizesFifo.addr + slot*NCCL_NET_IB_MAX_RECVS*sizeof(int);
      lastWr->num_sge = 1;
      lastWr->sg_list = &comm->remSizesFifo.sge;
    }
  }
  
  lastWr->wr_id = wr_id;
  lastWr->opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
  lastWr->imm_data = immData;
  lastWr->next = NULL;
  lastWr->send_flags = IBV_SEND_SIGNALED;

  // 多 QP：确保 IB 写入是 128 字节的倍数，以便 LL 和 LL128 协议正常工作
  const int align = 128;
  int nqps = ncclParamIbSplitDataOnQps() ? comm->base.nqps : comm->base.nDataQps;
  
  for (int i = 0; i < nqps; i++) {
    int qpIndex = comm->base.qpIndex;
    ncclIbQp* qp = comm->base.qps + qpIndex;
    int devIndex = qp->devIndex;
    
    for (int r=0; r<nreqs; r++) {
      // 跟踪此事件以完成
      //ncclIbAddEvent(reqs[r], devIndex, &comm->devs[devIndex].base);

      // 选择正确的 rkey（即使 0 大小发送也需要）
      comm->wrs[r].wr.rdma.rkey = slots[r].rkeys[qp->remDevIdx];

      // 计算此 QP 的块大小（对齐到 128 字节）
      int chunkSize = DIVUP(DIVUP(reqs[r]->send.size, nqps), align) * align;
      int length = std::min(reqs[r]->send.size-reqs[r]->send.offset, chunkSize);
      
      if (length <= 0) {
        comm->wrs[r].sg_list = NULL;
        comm->wrs[r].num_sge = 0;
      } else {
        // 选择正确的 lkey
        comm->sges[r].lkey = reqs[r]->send.lkeys[devIndex];
        comm->sges[r].length = length;
        comm->wrs[r].sg_list = comm->sges+r;
        comm->wrs[r].num_sge = 1;
      }
    }

    if (nreqs > 1) {
      // 确保最后使用正确的 lkey 写入远程大小
      comm->remSizesFifo.sge.lkey = comm->remSizesFifo.mrs[devIndex]->lkey;
      lastWr->wr.rdma.rkey = comm->remSizesFifo.rkeys[devIndex];
    }

    struct ibv_send_wr* bad_wr;
#ifdef NCCL_ENABLE_NET_PROFILING
    // QP 性能分析循环
    for (int r=0; r<nreqs; r++) {
      // 存储此请求的 comm qpIndex
      int nEventHandles = reqs[r]->pInfo[0].nEventHandles;
      assert(nEventHandles < MAX_QPS_PER_REQ);
      reqs[r]->pInfo[0].qpIndex[nEventHandles] = qpIndex;
      
      // 存储性能分析器信息
      int64_t pluginId = NCCL_PROFILER_NET_TYPE_IB | NCCL_PROFILER_NET_IB_VER;
      reqs[r]->pInfo[0].data.type = ncclProfileQp;
      reqs[r]->pInfo[0].data.qp.device = devIndex;
      reqs[r]->pInfo[0].data.qp.wr_id = comm->wrs[r].wr_id;
      reqs[r]->pInfo[0].data.qp.opcode = comm->wrs[r].opcode;
      reqs[r]->pInfo[0].data.qp.qpNum = qp->qp->qp_num;
      reqs[r]->pInfo[0].data.qp.length = comm->sges[r].length;
      void* pHandle = reqs[r]->pInfo[0].pHandle;
      NCCLCHECK(ncclProfilerFunction(&reqs[r]->pInfo[0].qpEventHandles[nEventHandles], ncclProfilerNetEventStart, pHandle, pluginId, &reqs[r]->pInfo[0].data));
      reqs[r]->pInfo[0].nEventHandles++;
    }
#endif
    // 发送工作请求到 QP
    NCCLCHECK(wrap_ibv_post_send(qp->qp, comm->wrs, &bad_wr));

    // 更新偏移量和地址（为下一个 QP 准备）
    for (int r=0; r<nreqs; r++) {
      int chunkSize = DIVUP(DIVUP(reqs[r]->send.size, nqps), align) * align;
      reqs[r]->send.offset += chunkSize;
      comm->sges[r].addr += chunkSize;
      comm->wrs[r].wr.rdma.remote_addr += chunkSize;
    }

    // 选择下一个 qpIndex
    comm->base.qpIndex = (comm->base.qpIndex+1) % comm->base.nqps;
  }

  return ncclSuccess;
}

// 异步发送
// 功能：发起 RDMA 写操作到远程接收缓冲区
ncclResult_t ncclIbIsend(void* sendComm, void* data, size_t size, int tag, void* mhandle, void* phandle, void** request) {
  struct ncclIbSendComm* comm = (struct ncclIbSendComm*)sendComm;
  
  if (comm->base.ready == 0) {
    WARN("NET/IB: ncclIbIsend() called when comm->base.ready == 0");
    *request = NULL;
    return ncclInternalError;
  }
  NCCLCHECK(ncclIbStatsCheckFatalCount(&comm->base.stats,__func__));

  struct ncclIbMrHandle* mhandleWrapper = (struct ncclIbMrHandle*) mhandle;

  // 等待接收端发布相应的接收
  int nreqs = 0;
  volatile struct ncclIbSendFifo* slots;

  int slot = (comm->fifoHead) % MAX_REQUESTS;
  struct ncclIbRequest** reqs = comm->fifoReqs[slot];
  slots = comm->fifo[slot];
  uint64_t idx = comm->fifoHead+1;
  
  // 检查第一个槽位是否就绪
  if (slots[0].idx != idx) { *request = NULL; return ncclSuccess; }
  nreqs = slots[0].nreqs;
  
  // 等待所有数据到达
  for (int r=1; r<nreqs; r++) while(slots[r].idx != idx);
  
  // 内存屏障：确保 nreqsPtr 加载在 tag/rkey/addr 加载之前
  __sync_synchronize(); 
  
  for (int r=0; r<nreqs; r++) {
    // 检查槽位是否空闲或标签匹配
    if (reqs[r] != NULL || slots[r].tag != tag) continue;

    // 调整大小不超过远程缓冲区
    if (size > slots[r].size) size = slots[r].size;
    
    // 合理性检查
    if (slots[r].size < 0 || slots[r].addr == 0 || slots[r].rkeys[0] == 0) {
      char line[SOCKET_NAME_MAXLEN + 1];
      union ncclSocketAddress addr;
      ncclSocketGetAddr(&comm->base.sock, &addr);
      WARN("NET/IB : req %d/%d tag %x peer %s posted incorrect receive info: size %ld addr %lx rkeys[0]=%x",
        r, nreqs, tag, ncclSocketToString(&addr, line), slots[r].size, slots[r].addr, slots[r].rkeys[0]);
      return ncclInternalError;
    }

    // 获取请求
    struct ncclIbRequest* req;
    NCCLCHECK(ncclIbGetRequest(&comm->base, &req));
    req->type = NCCL_NET_IB_REQ_SEND;
    req->sock = &comm->base.sock;
    req->base = &comm->base;
    req->nreqs = nreqs;
    req->send.size = size;
    req->send.data = data;
    req->send.offset = 0;
#ifdef NCCL_ENABLE_NET_PROFILING
    req->pInfo[0].pHandle = phandle;
#endif

    // 填充事件（跟踪待完成的操作）
    int nEvents = ncclParamIbSplitDataOnQps() ? comm->base.nqps : comm->base.nDataQps;
    int qpIndex = comm->base.qpIndex;
    
    // 倒数
    while (nEvents > 0) {
      ncclIbQp* qp = comm->base.qps + qpIndex;
      int devIndex = qp->devIndex;
      ncclIbAddEvent(req, devIndex, &comm->devs[devIndex].base);
      // 跟踪此 RDMA_Write 的有效 lkey
      req->send.lkeys[devIndex] = mhandleWrapper->mrs[devIndex]->lkey;
      nEvents--;
      // 还不更新 comm->base.qpIndex，需要在 ncclIbMultiSend() 中运行相同的 QP 集
      qpIndex = (qpIndex+1)%comm->base.nqps;
    }

    // 存储所有 lkeys
    for (int i = 0; i < comm->base.vProps.ndevs; i++) {
      req->send.lkeys[i] = mhandleWrapper->mrs[i]->lkey;
    }

    *request = reqs[r] = req;

    // 如果是多接收，仅当所有请求都匹配时才发送
    for (int r=0; r<nreqs; r++) {
      if (reqs[r] == NULL) return ncclSuccess;
    }

    TIME_START(0);
    NCCLCHECK(ncclIbMultiSend(comm, slot));

    // 清空槽位，帮助调试和合理性检查
    memset((void*)slots, 0, sizeof(struct ncclIbSendFifo));
    memset(reqs, 0, NCCL_NET_IB_MAX_RECVS*sizeof(struct ncclIbRequest*));
    comm->fifoHead++;
    TIME_STOP(0);
    return ncclSuccess;
  }

  *request = NULL;
  return ncclSuccess;
}

// 发布 FIFO（内部函数）
// 功能：向远程 FIFO 发送接收就绪通知
ncclResult_t ncclIbPostFifo(struct ncclIbRecvComm* comm, int n, void** data, size_t* sizes, int* tags, void** mhandles, struct ncclIbRequest* req) {
  struct ibv_send_wr wr;
  memset(&wr, 0, sizeof(wr));

  int slot = comm->remFifo.fifoTail%MAX_REQUESTS;
  req->recv.sizes = comm->sizesFifo[slot];
  for (int i=0; i<n; i++) req->recv.sizes[i] = 0;
  struct ncclIbSendFifo* localElem = comm->remFifo.elems[slot];

  // 选择下一个 devIndex（本地）和 QP 用于发布此 CTS 消息
  // 由于 QP 通过在 devIndex 间条带化初始化，可以简单地分配为相同值
  ncclIbQp* ctsQp = comm->base.qps + comm->base.devIndex;
  comm->base.devIndex = (comm->base.devIndex + 1) % comm->base.vProps.ndevs;

  // 填充本地 FIFO 元素
  for (int i=0; i<n; i++) {
    localElem[i].addr = (uint64_t)data[i];
    struct ncclIbMrHandle* mhandleWrapper = (struct ncclIbMrHandle*) mhandles[i];

    // 发送所有适用的 rkeys
    for (int j = 0; j < comm->base.vProps.ndevs; j++)
      localElem[i].rkeys[j] = mhandleWrapper->mrs[j]->rkey;

    localElem[i].nreqs = n;
    localElem[i].size = sizes[i];  // 合理性/调试
    localElem[i].tag = tags[i];
    localElem[i].idx = comm->remFifo.fifoTail+1;
  }
  
  wr.wr.rdma.remote_addr = comm->remFifo.addr + slot*NCCL_NET_IB_MAX_RECVS*sizeof(struct ncclIbSendFifo);

  // 查找正确的 fifoRkey
  wr.wr.rdma.rkey = comm->base.remDevs[ctsQp->remDevIdx].fifoRkey;

  // 设置正确的 sge 属性
  comm->devs[ctsQp->devIndex].fifoSge.addr   = (uint64_t)localElem;
  comm->devs[ctsQp->devIndex].fifoSge.length = n*sizeof(struct ncclIbSendFifo);
  wr.sg_list = &comm->devs[ctsQp->devIndex].fifoSge;
  wr.num_sge = 1;

  wr.opcode = IBV_WR_RDMA_WRITE;
  wr.send_flags = comm->remFifo.flags;  // IBV_SEND_INLINE

  // 需要偶尔发布带 IBV_SEND_SIGNALED 标志的请求，
  // 否则发送队列永远不会清空
  //
  // 来自 https://www.rdmamojo.com/2014/06/30/working-unsignaled-completions/
  // "如何使用无信号完成？" / "陷阱和注意事项"
  // 所有已发布的发送请求（信号和无信号）都被视为未完成，
  // 直到从与发送队列关联的完成队列中轮询到它们或在其之后发布的发送请求。
  // 这意味着如果使用配置为无信号完成的队列对，
  // 必须确保偶尔（在发送队列充满未完成的发送请求之前）
  // 发布一个将生成工作完成的发送请求。
  //
  // 不遵循此规则可能导致以下情况：
  //
  //  - 发送队列已满，无法向其发布新的发送请求
  //  - 发送队列无法清空，因为无法再生成工作完成
  //    （原因是无法发布可以生成工作完成的请求，
  //     轮询该请求将清空发送队列）
  //  - 所有已发布的发送请求的状态被视为未知
  //
  // slot == devIndex - 当写入到 fifo 槽位 N，且此 QP 位于设备索引 N 时，
  // 应该发送信号。这样可以确保每个 fifo 发布 QP 都被排空
  if (slot == ctsQp->devIndex) {
    wr.send_flags |= IBV_SEND_SIGNALED;
    wr.wr_id = req - comm->base.reqs;
    ncclIbAddEvent(req, ctsQp->devIndex, &comm->devs[ctsQp->devIndex].base);
  }

  struct ibv_send_wr* bad_wr;
  NCCLCHECK(wrap_ibv_post_send(ctsQp->qp, &wr, &bad_wr));
  comm->remFifo.fifoTail++;

  return ncclSuccess;
}

// 异步接收
// 功能：发布接收工作请求，通知发送端接收缓冲区已就绪
ncclResult_t ncclIbIrecv(void* recvComm, int n, void** data, size_t* sizes, int* tags, void** mhandles, void** phandles, void** request) {
  struct ncclIbRecvComm* comm = (struct ncclIbRecvComm*)recvComm;
  
  if (comm->base.ready == 0) {
    WARN("NET/IB: ncclIbIrecv() called when comm->base.ready == 0");
    *request = NULL;
    return ncclInternalError;
  }
  if (n > NCCL_NET_IB_MAX_RECVS) return ncclInternalError;
  NCCLCHECK(ncclIbStatsCheckFatalCount(&comm->base.stats,__func__));

  // 获取请求
  struct ncclIbRequest* req;
  NCCLCHECK(ncclIbGetRequest(&comm->base, &req));
  req->type = NCCL_NET_IB_REQ_RECV;
  req->sock = &comm->base.sock;
  req->nreqs = n;
#ifdef NCCL_ENABLE_NET_PROFILING
  for (int r = 0; r < n && phandles; r++) req->pInfo[r].nEventHandles = 0;
#endif

  for (int i = 0; i < comm->base.vProps.ndevs; i++) {
    req->devBases[i] = &comm->devs[i].base;
  }

  struct ibv_recv_wr wr;
  memset(&wr, 0, sizeof(wr));
  wr.wr_id = req - comm->base.reqs;
  wr.sg_list = NULL;
  wr.num_sge = 0;

  TIME_START(1);
  // 选择所有 QP 或每个设备一个 QP
  const int nqps = ncclParamIbSplitDataOnQps() ? comm->base.nqps : comm->base.nDataQps;

  // 发布接收工作请求
  struct ibv_recv_wr* bad_wr;
  for (int i = 0; i < nqps; i++) {
    struct ncclIbQp* qp = comm->base.qps + comm->base.qpIndex;
    ncclIbAddEvent(req, qp->devIndex, &comm->devs[qp->devIndex].base);
#ifdef NCCL_ENABLE_NET_PROFILING
    // 为多接收中的每个请求和每个 qp 启动 QP 事件
    for (int r = 0; r < n; r++) {
      int nEventHandles = req->pInfo[r].nEventHandles;
      assert(nEventHandles < MAX_QPS_PER_REQ);
      req->pInfo[r].qpIndex[nEventHandles] = comm->base.qpIndex;
      
      // 存储性能分析器信息
      int64_t pluginId = NCCL_PROFILER_NET_TYPE_IB | NCCL_PROFILER_NET_IB_VER;
      req->pInfo[r].data.type = ncclProfileQp;
      req->pInfo[r].data.qp.device = qp->devIndex;
      req->pInfo[r].data.qp.wr_id = wr.wr_id;
      req->pInfo[r].data.qp.qpNum = qp->qp->qp_num;
      NCCLCHECK(ncclProfilerFunction(&req->pInfo[r].qpEventHandles[nEventHandles], ncclProfilerNetEventStart, phandles[r], pluginId, &req->pInfo[r].data));
      req->pInfo[r].nEventHandles++;
    }
#endif
    NCCLCHECK(wrap_ibv_post_recv(qp->qp, &wr, &bad_wr));
    comm->base.qpIndex = (comm->base.qpIndex+1)%comm->base.nqps;
  }

  TIME_STOP(1);

  // 发布到 FIFO 以通知发送端
  TIME_START(2);
  NCCLCHECK(ncclIbPostFifo(comm, n, data, sizes, tags, mhandles, req));
  TIME_STOP(2);

  *request = req;
  return ncclSuccess;
}

// 异步刷新（GPU Direct RDMA）
// 功能：触发 RDMA 读操作以确保 GPU 写入对网络可见
ncclResult_t ncclIbIflush(void* recvComm, int n, void** data, int* sizes, void** mhandles, void** request) {
  struct ncclIbRecvComm* comm = (struct ncclIbRecvComm*)recvComm;
  int last = -1;
  
  // 查找最后一个非零接收
  for (int i=0; i<n; i++) if (sizes[i]) last = i;
  
  // 如果未启用刷新或没有数据
  if (comm->flushEnabled == 0 || last == -1) return ncclSuccess;

  // 仅使用最后一个非零接收刷新一次
  struct ncclIbRequest* req;
  NCCLCHECK(ncclIbGetRequest(&comm->base, &req));
  req->type = NCCL_NET_IB_REQ_FLUSH;
  req->sock = &comm->base.sock;
  struct ncclIbMrHandle* mhandle = (struct ncclIbMrHandle*) mhandles[last];

  // 我们不知道 recv 在哪个 devIndex 上，所以在所有设备上刷新
  for (int i = 0; i < comm->base.vProps.ndevs; i++) {
    struct ibv_send_wr wr;
    memset(&wr, 0, sizeof(wr));
    wr.wr_id = req - comm->base.reqs;

    wr.wr.rdma.remote_addr = (uint64_t)data[last];
    wr.wr.rdma.rkey = mhandle->mrs[i]->rkey;
    wr.sg_list = &comm->devs[i].gpuFlush.sge;
    wr.num_sge = 1;
    wr.opcode = IBV_WR_RDMA_READ;  // 读操作确保缓存一致性
    wr.send_flags = IBV_SEND_SIGNALED;

    TIME_START(4);
    struct ibv_send_wr* bad_wr;
    NCCLCHECK(wrap_ibv_post_send(comm->devs[i].gpuFlush.qp.qp, &wr, &bad_wr));
    TIME_STOP(4);

    ncclIbAddEvent(req, i, &comm->devs[i].base);
  }

  *request = req;
  return ncclSuccess;
}

// 设备名称宏（用于错误日志）
#define HCA_NAME(req, index) ((req)->devBases[(index)]->pd->context->device->name)

#ifdef NCCL_ENABLE_NET_PROFILING
// 获取请求的 QP 索引
static int getReqQpIndex(struct ncclIbRequest* req, int request, int qpNumber) {
  for (int i = 0; i < MAX_QPS_PER_REQ; i++) {
    int qpIndex = req->pInfo[request].qpIndex[i];
    if (req->base->qps[qpIndex].qp->qp_num == qpNumber) return i;
  }
  return 0;
}
#endif

// 测试请求完成状态
// 功能：轮询完成队列，检查操作是否完成
ncclResult_t ncclIbTest(void* request, int* done, int* sizes) {
  struct ncclIbRequest *r = (struct ncclIbRequest*)request;
  *done = 0;
  
  while (1) {
    // 检查致命错误
    NCCLCHECK(ncclIbStatsCheckFatalCount(&r->base->stats,__func__));
    
    // 检查是否所有事件都已完成
    if (r->events[0] == 0 && r->events[1] == 0 && r->events[2] == 0 && r->events[3] == 0) {
      TRACE(NCCL_NET, "r=%p done", r);
      *done = 1;
      
      // 返回接收的大小
      if (sizes && r->type == NCCL_NET_IB_REQ_RECV) {
        for (int i=0; i<r->nreqs; i++) {
          sizes[i] = r->recv.sizes[i];
#ifdef NCCL_ENABLE_NET_PROFILING
          for (int j = 0; j < r->pInfo[i].nEventHandles; j++) {
            NCCLCHECK(ncclProfilerFunction(&r->pInfo[i].qpEventHandles[j], ncclProfilerNetEventStop, NULL, 0, NULL));
          }
#endif
        }
      }
      
      if (sizes && r->type == NCCL_NET_IB_REQ_SEND) {
        sizes[0] = r->send.size;
#ifdef NCCL_ENABLE_NET_PROFILING
        for (int j = 0; j < r->pInfo[0].nEventHandles; j++) {
          NCCLCHECK(ncclProfilerFunction(&r->pInfo[0].qpEventHandles[j], ncclProfilerNetEventStop, NULL, 0, NULL));
        }
#endif
      }
      
      // 停止此事件的所有剩余 QP 事件
      NCCLCHECK(ncclIbFreeRequest(r));
      return ncclSuccess;
    }

    int totalWrDone = 0;
    int wrDone = 0;
    struct ibv_wc wcs[4];

    // 轮询每个设备的完成队列
    for (int i = 0; i < NCCL_IB_MAX_DEVS_PER_NIC; i++) {
      TIME_START(3);
      
      // 如果期望此设备 CQ 的任何完成
      if (r->events[i]) {
        NCCLCHECK(wrap_ibv_poll_cq(r->devBases[i]->cq, 4, wcs, &wrDone));
        totalWrDone += wrDone;
        if (wrDone == 0) { TIME_CANCEL(3); } else { TIME_STOP(3); }
        if (wrDone == 0) continue;
        
        // 处理每个工作完成
        for (int w=0; w<wrDone; w++) {
          struct ibv_wc *wc = wcs+w;
          
          // 检查错误状态
          if (wc->status != IBV_WC_SUCCESS) {
            union ncclSocketAddress addr;
            ncclSocketGetAddr(r->sock, &addr);
            char localGidString[INET6_ADDRSTRLEN] = "";
            char remoteGidString[INET6_ADDRSTRLEN] = "";
            const char* localGidStr = NULL, *remoteGidStr = NULL;
            
            if (r->devBases[i]->gidInfo.link_layer == IBV_LINK_LAYER_ETHERNET) {
              localGidStr = ibvGetGidStr(&r->devBases[i]->gidInfo.localGid, localGidString, sizeof(localGidString));
              remoteGidStr = ibvGetGidStr(&r->base->remDevs[i].remoteGid, remoteGidString, sizeof(remoteGidString));
            }

            char line[SOCKET_NAME_MAXLEN+1];
            char *hcaName = r->devBases[i]->pd->context->device->name;
            WARN("NET/IB: Got completion from peer %s with status=%d opcode=%d len=%u vendor err %u (%s)%s%s%s%s hca %s",
                ncclSocketToString(&addr, line), wc->status, wc->opcode, wc->byte_len, wc->vendor_err, reqTypeStr[r->type],
                localGidStr ?  " localGid ":"", localGidString, remoteGidStr ? " remoteGids":"", remoteGidString, hcaName);
            return ncclRemoteError;
          }

          union ncclSocketAddress addr;
          ncclSocketGetAddr(r->sock, &addr);
          struct ncclIbRequest* req = r->base->reqs+(wc->wr_id & 0xff);

          #ifdef ENABLE_TRACE
          char line[SOCKET_NAME_MAXLEN+1];
          TRACE(NCCL_NET, "Got completion from peer %s with status=%d opcode=%d len=%u wr_id=%lu r=%p type=%d events={%d,%d,%d,%d}, i=%d",
            ncclSocketToString(&addr, line), wc->status, wc->opcode,wc->byte_len, wc->wr_id, req, req->type, req->events[0], req->events[1], req->events[2], req->events[3], i);
          #endif
          
          if (req && req->type == NCCL_NET_IB_REQ_SEND) {
            // 发送请求完成
            for (int j = 0; j < req->nreqs; j++) {
              struct ncclIbRequest* sendReq = r->base->reqs+((wc->wr_id >> (j*8)) & 0xff);
              if ((sendReq->events[i] <= 0)) {
                WARN("NET/IB: sendReq(%p)->events={%d,%d,%d,%d}, i=%d, j=%d <= 0", sendReq, sendReq->events[0], sendReq->events[1], sendReq->events[2], sendReq->events[3], i, j);
                return ncclInternalError;
              }
              sendReq->events[i]--;
#ifdef NCCL_ENABLE_NET_PROFILING
              // 停止 sendReq 的 QP 事件
              int qpIndex = getReqQpIndex(sendReq, j, wc->qp_num);
              NCCLCHECK(ncclProfilerFunction(&sendReq->pInfo[j].qpEventHandles[qpIndex], ncclProfilerNetEventStop, NULL, 0, NULL));
#endif
            }
          } else {
            // 接收请求完成
            if (req && wc->opcode == IBV_WC_RECV_RDMA_WITH_IMM) {
              if (req->type != NCCL_NET_IB_REQ_RECV) {
                WARN("NET/IB: wc->opcode == IBV_WC_RECV_RDMA_WITH_IMM and req->type=%d", req->type);
                return ncclInternalError;
              }
              // 单接收：从立即数据提取大小
              if (req->nreqs == 1) {
                req->recv.sizes[0] = wc->imm_data;
              }
            }
            req->events[i]--;
#ifdef NCCL_ENABLE_NET_PROFILING
            // 停止 workFifo 的 QP 事件
            for (int j = 0; j < req->nreqs; j++) {
              int qpIndex = getReqQpIndex(req, j, wc->qp_num);
              NCCLCHECK(ncclProfilerFunction(&req->pInfo[j].qpEventHandles[qpIndex], ncclProfilerNetEventStop, NULL, 0, NULL));
            }
#endif
          }
        }
        
        // 一旦在异步线程中报告了 IB 致命事件，
        // 我们要将此错误传播到通信器并防止进一步轮询以减少错误污染
        NCCLCHECK(ncclIbStatsCheckFatalCount(&ncclIbDevs[r->devBases[i]->ibDevN].stats,__func__));
      }
    }

    // 如果在任何设备上都没找到 CQE，返回并稍后再试
    if (totalWrDone == 0) return ncclSuccess;
  }
}

//================================================================================
// 连接关闭
//================================================================================

// 关闭发送通信
ncclResult_t ncclIbCloseSend(void* sendComm) {
  struct ncclIbSendComm* comm = (struct ncclIbSendComm*)sendComm;
  if (comm) {
    NCCLCHECK(ncclSocketClose(&comm->base.sock));

    // 销毁所有 QP
    for (int q = 0; q < comm->base.nqps; q++)
      if (comm->base.qps[q].qp != NULL) 
        NCCLCHECK(wrap_ibv_destroy_qp(comm->base.qps[q].qp));

    // 清理每个设备的资源
    for (int i = 0; i < comm->base.vProps.ndevs; i++) {
      struct ncclIbSendCommDev* commDev = comm->devs + i;
      if (commDev->fifoMr != NULL) 
        NCCLCHECK(wrap_ibv_dereg_mr(commDev->fifoMr));
      if (comm->remSizesFifo.mrs[i] != NULL) 
        NCCLCHECK(wrap_ibv_dereg_mr(comm->remSizesFifo.mrs[i]));
      NCCLCHECK(ncclIbDestroyBase(&commDev->base));
    }
    free(comm);
  }
  TIME_PRINT("IB");
  return ncclSuccess;
}

// 关闭接收通信
ncclResult_t ncclIbCloseRecv(void* recvComm) {
  struct ncclIbRecvComm* comm = (struct ncclIbRecvComm*)recvComm;
  if (comm) {
    NCCLCHECK(ncclSocketClose(&comm->base.sock));

    // 销毁所有 QP
    for (int q = 0; q < comm->base.nqps; q++)
      if (comm->base.qps[q].qp != NULL)
        NCCLCHECK(wrap_ibv_destroy_qp(comm->base.qps[q].qp));

    // 清理每个设备的资源
    for (int i = 0; i < comm->base.vProps.ndevs; i++) {
      struct ncclIbRecvCommDev* commDev = comm->devs + i;
      
      // 清理 GPU Direct RDMA 刷新资源
      if (comm->flushEnabled) {
        if (commDev->gpuFlush.qp.qp != NULL) 
            NCCLCHECK(wrap_ibv_destroy_qp(commDev->gpuFlush.qp.qp));
        if (commDev->gpuFlush.hostMr != NULL) 
            NCCLCHECK(wrap_ibv_dereg_mr(commDev->gpuFlush.hostMr));
      }
      
      if (commDev->fifoMr != NULL) 
        NCCLCHECK(wrap_ibv_dereg_mr(commDev->fifoMr));
      if (commDev->sizesFifoMr != NULL)
        NCCLCHECK(wrap_ibv_dereg_mr(commDev->sizesFifoMr));
      NCCLCHECK(ncclIbDestroyBase(&commDev->base));
    }
    free(comm);
  }
  return ncclSuccess;
}

// 关闭监听通信
ncclResult_t ncclIbCloseListen(void* listenComm) {
  struct ncclIbListenComm* comm = (struct ncclIbListenComm*)listenComm;
  if (comm) {
    NCCLCHECK(ncclSocketClose(&comm->sock));
    free(comm);
  }
  return ncclSuccess;
}

//================================================================================
// 最终化
//================================================================================

// 最终化网络插件
ncclResult_t ncclIbFinalize(void* ctx) {
  netRefCount--;
  return ncclSuccess;
}

//================================================================================
// 网络接口结构
//================================================================================

// NCCL IB 网络接口函数表
// 这是 NCCL 核心调用的主要接口
ncclNet_t ncclNetIb = {
  "IB",  // 网络类型名称
  ncclIbInit,             // 初始化
  ncclIbDevices,          // 获取设备数量
  ncclIbGetProperties,    // 获取设备属性
  ncclIbListen,           // 监听连接
  ncclIbConnect,          // 连接到远程
  ncclIbAccept,           // 接受连接
  ncclIbRegMr,            // 注册内存
  ncclIbRegMrDmaBuf,      // 注册 DMA-BUF 内存
  ncclIbDeregMr,          // 注销内存
  ncclIbIsend,            // 异步发送
  ncclIbIrecv,            // 异步接收
  ncclIbIflush,           // 异步刷新（GDR）
  ncclIbTest,             // 测试完成
  ncclIbCloseSend,        // 关闭发送通信
  ncclIbCloseRecv,        // 关闭接收通信
  ncclIbCloseListen,      // 关闭监听通信
  NULL /* getDeviceMr */,   // 获取设备内存（未使用）
  NULL /* irecvConsumed */, // 接收消费（未使用）
  ncclIbMakeVDevice,      // 创建虚拟设备
  ncclIbFinalize,         // 最终化
  ncclIbSetNetAttr,       // 设置网络属性
};

/*
  预留接口（未来扩展）：
  ncclIbSetProperties,    // 设置属性
  ncclIbRefreshDevices    // 刷新设备列表
*/
