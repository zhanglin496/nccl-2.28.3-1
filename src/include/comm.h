/*************************************************************************
 * Copyright (c) 2015-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// 头文件保护宏：防止重复包含
#ifndef NCCL_COMM_H_
#define NCCL_COMM_H_

//#include "transport.h"  // 传输层头文件（已注释）
#include "p2p.h"           // 点对点通信相关定义
#include "collectives.h"   // 集合通信相关定义
#include "nccl_tuner.h"    // NCCL 调优插件接口
#include "proxy.h"         // 代理相关定义
#include "strongstream.h"  // 强类型 CUDA 流封装
#include "nccl_net.h"      // 网络插件接口
#include "register.h"      // 内存注册相关
#include "graph.h"         // 拓扑图相关
#include "profiler.h"      // 性能分析器接口
#include "allocator.h"     // 内存分配器
#include "dev_runtime.h"   // 设备端运行时
#include "sym_kernels.h"   // 对称内核相关
#include "ce_coll.h"       // Copy Engine 集合通信

// CUDA 9.0 之前版本没有定义 cudaLaunchParams 结构体
#if CUDART_VERSION < 9000
// CUDA 内核启动参数结构体（兼容 CUDA 9.0 之前的版本）
struct cudaLaunchParams {
  void *func;              // 内核函数指针
  dim3 gridDim;            // 网格维度
  dim3 blockDim;           // 线程块维度
  void **args;             // 内核参数数组
  size_t sharedMem;        // 共享内存大小
  cudaStream_t stream;     // CUDA 流
};
#endif

// 缓存行大小：128 字节（用于缓存对齐，避免伪共享）
#define CACHE_LINE_SIZE 128
// 内存对齐大小：4096 字节（页大小）
#define MEM_ALIGN 4096
// CUDA IPC 最小内存：2MB（用于跨进程 CUDA IPC 共享）
#define CUDA_IPC_MIN 2097152UL

// Channels / LL tuning
// 不同协议的线程数阈值，用于决定是否使用 LL/L L128 协议
#define NCCL_LL_THREAD_THRESHOLD 8       // LL 协议最小线程数阈值
#define NCCL_LL128_THREAD_THRESHOLD 8    // LL128 协议最小线程数阈值
#define NCCL_SIMPLE_THREAD_THRESHOLD 64  // Simple 协议最小线程数阈值

// ============================================================================
// ncclSendMem - 发送端内存结构
// ============================================================================
// 用于 GPU 之间通信的发送端共享内存区域
struct ncclSendMem {
  union {
    struct {
      // 发送队列的头部指针（生产者索引）
      uint64_t head;
      // 填充到缓存行边界，避免伪共享（pad1 + head = 128 字节）
      char pad1[CACHE_LINE_SIZE-sizeof(uint64_t)];
      // 指针交换区域（用于 P2P 通信的指针传递）
      void* ptrExchange;
      // 归约操作参数交换（2 个 uint64_t）
      uint64_t redOpArgExchange[2];
      // 填充到缓存行边界
      char pad2[CACHE_LINE_SIZE-sizeof(void*)-2*sizeof(uint64_t)];
      // 偏移 FIFO 队列（每个步骤一个偏移量）
      int offsFifo[NCCL_STEPS];
    };
    // 整体对齐到 MEM_ALIGN（4096 字节）
    char pad3[MEM_ALIGN];
  };
};

// ============================================================================
// ncclRecvMem - 接收端内存结构
// ============================================================================
// 用于 GPU 之间通信的接收端共享内存区域
struct ncclRecvMem {
  union {
    struct {
      // 接收队列的尾部指针（消费者索引）
      uint64_t tail;
      // 填充到缓存行边界
      char pad1[CACHE_LINE_SIZE-sizeof(uint64_t)];
      // 连接 FIFO 队列（每个步骤一个连接信息）
      struct ncclConnFifo connFifo[NCCL_STEPS];
      // 刷新标志（用于 GDRCopy 刷新操作）
      int flush; // For GDRCopy-based flush
    };
    // 整体对齐到 MEM_ALIGN
    char pad4[MEM_ALIGN];
  };
};

// ============================================================================
// 辅助线程状态枚举
// ============================================================================
enum helperThreadState {ThreadStart, ThreadStop};
  // ThreadStart: 启动辅助线程
  // ThreadStop:  停止辅助线程

// IPC 池大小计算
// 2 * NCCL_MAX_LOCAL_RANKS * NCCL_MAX_OPS
// 每个 rank 可以有最多 NCCL_MAX_OPS 个操作，需要双缓冲
#define NCCL_IPC_POOL_SIZE (2*NCCL_MAX_LOCAL_RANKS*NCCL_MAX_OPS)

// ============================================================================
// ncclGraphHelperResources - CUDA Graph 辅助资源
// ============================================================================
// 用于 CUDA Graph 捕获和执行的辅助线程资源
struct ncclGraphHelperResources {
  // 关联的通信器指针
  ncclComm* comm;
  // 线程互斥锁（保护共享资源）
  pthread_mutex_t threadLock;
  // 线程条件变量（用于线程同步）
  pthread_cond_t  threadCond;
  // 辅助线程状态
  enum helperThreadState threadState;
  // IPC 内存基地址池（存储所有 IPC 句柄）
  void* ipcBases[NCCL_IPC_POOL_SIZE];
  // IPC 队列尾部索引（生产者使用）
  int ipcTail;
  // IPC 队列头部索引（消费者使用）
  int ipcHead;
};

// ============================================================================
// ncclUserRedOp - 用户自定义归约操作
// ============================================================================
struct ncclUserRedOp {
  // 下一个空闲条目的索引，-1 表示已分配
  int freeNext; // -1=allocated, otherwise index of next free entry in array
  // 数据类型（如 float, int 等）
  ncclDataType_t datatype;
  // 设备端归约操作的完整描述
  ncclDevRedOpFull opFull;
};

// ============================================================================
// ncclNodeRanks - 节点 Rank 信息
// ============================================================================
// 描述一个节点（物理机器）上的 Rank 分布
struct ncclNodeRanks {
  // 该节点上的本地 Rank 数量
  int localRanks;
  // 本地 Rank 到全局 Rank 的映射表
  // localRankToRank[localRank] = globalRank
  int* localRankToRank;
};

// ============================================================================
// cliqueInfo - MNNVL 集群信息
// ============================================================================
// Multi-Node NVLink 集群（通过 NVLink 桥接器连接的多节点 GPU 组）
struct cliqueInfo {
  // 集群 ID
  int id;
  // 集群大小（节点内的 GPU 数量）
  int size;
  // 集群内所有 Rank 的列表
  int *ranks;
};

// ============================================================================
// ncclDestructor - 析构回调链表节点
// ============================================================================
// 用于通信器销毁时的资源清理回调
struct ncclDestructor {
  // 指向下一个析构器的指针（形成链表）
  struct ncclDestructor* next;
  // 要销毁的对象指针
  void* obj;
  // 析构函数指针
  ncclResult_t(*fn)(struct ncclDestructor* me);
};

// ============================================================================
// ncclCommCallback - 通信器回调链表节点
// ============================================================================
// 通用回调函数结构
struct ncclCommCallback {
  // 指向下一个回调的指针
  struct ncclCommCallback* next;
  // 回调函数指针（接收 comm 和 cb 作为参数）
  ncclResult_t(*fn)(struct ncclComm* comm, struct ncclCommCallback* cb);
};

// ============================================================================
// ncclCommEventCallback - CUDA Event 回调链表节点
// ============================================================================
// 基于 CUDA Event 的异步回调
struct ncclCommEventCallback {
  // 指向下一个回调的指针
  struct ncclCommEventCallback* next;
  // CUDA Event（用于同步）
  cudaEvent_t event;
  // 回调函数指针
  ncclResult_t(*fn)(struct ncclComm* comm, struct ncclCommEventCallback* cb);
};

// ============================================================================
// ncclSharedResources - 共享资源结构
// ============================================================================
// 被 split 通信器共享的资源（父通信器和子通信器共享）
struct ncclSharedResources {
  // 引用计数（跟踪有多少个通信器共享此资源）
  int refCount;

  // 记录是哪个通信器创建了这个 sharedRes
  struct ncclComm* owner; /* comm which creates this shared res. */

  // 对等端通信通道数组（每个通道一个数组，包含所有 rank 的对等端信息）
  struct ncclChannelPeer* peers[MAXCHANNELS];
  // 设备端对等端通道数组
  struct ncclDevChannelPeer* devPeers[MAXCHANNELS];
  /* P2P operation counter, one per channel */
  // 每个 channel 的 P2P 操作计数器
  uint64_t p2pOpCount[MAXCHANNELS];
  /* Collective operation counter */
  // 集合通信操作计数器
  uint64_t collOpCount;

  // tp 是 top parent 的缩写
  // 顶层父通信器的 rank 总数
  int tpNRanks;
  // 顶层父通信器的本地 rank 数量
  int tpNLocalRanks;
  // 顶层父通信器的通道数量
  int tpNChannels;
  // 顶层父通信器的 P2P 通道数量
  int tpP2pNChannels;
  // 顶层父通信器的 P2P 块大小
  int tpP2pChunkSize;
  // 魔数（用于校验共享资源的完整性）
  uint64_t magic;

  // top parent rank to localRank translation table
  // 顶层父 rank 到本地 rank 的转换表
  int* tpRankToLocalRank;

  // Internal streams
  // 内部 CUDA 流（用于 NCCL 内部操作）
  struct ncclStrongStream deviceStream, hostStream;
  // 持久化引用计数
  int persistentRefs;
  // CUDA Event（用于内核启动和临时缓冲区同步）
  cudaEvent_t launchEvent, scratchEvent;

  /* proxy related shared res */
  // 代理相关的共享资源
  struct ncclProxyState* proxyState;
};

// ============================================================================
// ncclChannel - 通信通道
// ============================================================================
// NCCL 通道是并行通信的基本单位，每个通道独立执行一部分通信任务
struct ncclChannel {
  // 对等端通道数组（peers[channel][rank]）
  struct ncclChannelPeer** peers;
  // 设备端对等端通道数组
  struct ncclDevChannelPeer** devPeers;
  /* devPeer pointer array used for host side access */
  // 主机端访问的设备端对等端指针数组
  struct ncclDevChannelPeer** devPeersHostPtr;

  // Ring 算法拓扑结构
  struct ncclRing ring;
  // 设备端 Ring 用户 rank 数组
  int* devRingUserRanks;

  // Tree 算法拓扑结构
  struct ncclTree tree;

  // CollNet Chain 算法拓扑结构
  struct ncclTree collnetChain;
  // CollNet Direct 算法拓扑结构
  struct ncclDirect collnetDirect;

  // NVLS (NVLink SHARP) 算法拓扑结构
  struct ncclNvls nvls;

  // 从 0 开始按序分配
  int id; // index of this channel
  // 工作队列已生产的字节数（work fifo 的后继者）
  uint32_t workFifoProduced; // +1 successor of last used work fifo byte

  /* comm split sharable resources */
  // CollNet 共享资源（可被 split 通信器共享）
  struct ncclChannelPeer* collnetPeers;
  struct ncclDevChannelPeer* collnetDevPeers;
  // NVLS 共享资源（可被 split 通信器共享）
  struct ncclChannelPeer* nvlsPeers;
  struct ncclDevChannelPeer* nvlsDevPeers;
};

// ============================================================================
// ncclWorkBatchList - 工作批次链表节点
// ============================================================================
// 工作批次（work batch）用于组织多个工作项
struct ncclWorkBatchList {
  // 指向下一个批次的指针
  struct ncclWorkBatchList* next;
  // 设备端工作批次结构
  struct ncclDevWorkBatch batch;
};

// ============================================================================
// ncclWorkList - 工作列表节点（16 字节对齐）
// ============================================================================
// 工作项链表，用于排队不同的内核工作
struct alignas(16) ncclWorkList {
  // 指向下一个工作项的指针
  struct ncclWorkList* next;
  // 工作类型（Collective, P2P 等）
  enum ncclDevWorkType workType;
  // 此节点后跟结构体的大小
  int size; // Size of struct following this node
  // ncclDevWorkColl, ncclDevWorkColLReg, ncclDevWorkP2p[]...
};

// ============================================================================
// ncclCollnetHandleList - CollNet 句柄列表
// ============================================================================
// 管理 CollNet 插件的句柄和资源
struct ncclCollnetHandleList {
  // 指向下一个节点的指针
  struct ncclCollnetHandleList *next;
  // CollNet 句柄（插件返回的不透明指针）
  void* collnetHandle;
  // 资源大小
  size_t size;
  // 缓冲区指针
  const void* buffer;
  // 代理连接器
  struct ncclProxyConnector* proxyconn;
};

// ============================================================================
// ncclTaskColl - 集合通信任务
// ============================================================================
// 描述一个集合通信操作的任务（如 AllReduce, Broadcast 等）
struct ncclTaskColl {
  // 指向下一个集合通信任务的指针（形成任务队列）
  struct ncclTaskColl* next;
  // 集合通信函数类型（AllReduce, Broadcast 等）
  ncclFunc_t func;
  // 发送缓冲区指针
  void const* sendbuff;
  // 接收缓冲区指针
  void* recvbuff;
  // 元素数量
  size_t count;
  // 根进程 rank（用于 root 类操作）
  int root;
  // 数据类型
  ncclDataType_t datatype;
  // 主机端归约操作（如 sum, max 等）
  ncclRedOp_t opHost;
  // 设备端归约操作的完整描述
  struct ncclDevRedOpFull opDev;
  // 分块步数和切片步数（用于大消息的分块处理）
  int chunkSteps, sliceSteps;

  // Computed later:
  // 流量字节数（计算得出）
  size_t trafficBytes;
  // 位域：最大通道数（8 位，最大 255）
  int32_t nMaxChannels:8;
  // 位域：每通道 warp 数量（8 位，最大 255）
  int32_t nWarps:8;
  // 位域：算法和协议（各 8 位）
  int32_t algorithm:8, protocol:8;
  // 位域：标志位
  uint32_t isCollnet:1,    // 是否使用 CollNet
           isNvls:1,        // 是否使用 NVLS
           isSymLast:1;     // 是否为最后一个对称内核
  // 位域：设备函数 ID（29 位）
  uint32_t devFuncId:29;
  // 注册缓冲区类型
  int regBufType;
  // planner->ipcMemQueue 中与此集合通信相关的元素数量
  int nCleanupQueueElts;

  // 发送窗口（用于对称内存注册）
  struct ncclDevrWindow* sendWin;
  // 接收窗口
  struct ncclDevrWindow* recvWin;
  // 发送端内存句柄（用于跨进程共享）
  void* sendMhandle;
  // 接收端内存句柄
  void* recvMhandle;
  // 发送端网络句柄数组
  void** sendNetHandles;
  // 接收端网络句柄数组
  void** recvNetHandles;
  // 接收端网络句柄数组（用于特殊接收）
  void** srecvNetHandles;

  // IPC 记录查找的索引
  uintptr_t sendbuffOffset;
  uintptr_t recvbuffOffset;
  // 远程地址数组
  uintptr_t* sendbuffRmtAddrs;
  uintptr_t* recvbuffRmtAddrs;

  // Profiler plugin
  // 性能分析器事件激活掩码
  int eActivationMask;
  // 组 API 事件句柄
  void* groupApiEventHandle;
  // 集合通信 API 事件句柄
  void* collApiEventHandle;
  // 通用事件句柄
  void* eventHandle;
  // 此任务使用的通道数量
  uint8_t nChannels;
};

// ============================================================================
// ncclTaskP2p - 点对点通信任务
// ============================================================================
// 描述一个 P2P 通信任务（Send/Recv）
struct ncclTaskP2p {
  // 指向下一个 P2P 任务的指针
  struct ncclTaskP2p* next;
  // P2P 函数类型（Send/Recv）
  ncclFunc_t func;
  // 集合通信 API（用于某些 P2P 操作）
  ncclFunc_t collAPI;
  // 缓冲区指针
  void* buff;
  // 元素数量
  size_t count;
  // 数据类型
  ncclDataType_t datatype;
  // 根进程 rank
  int root;
  // 字节数
  size_t bytes;

  // Profiler plugin
  // 性能分析器事件激活掩码
  int eActivationMask;
  // 组 API 事件句柄
  void* groupApiEventHandle;
  // P2P API 事件句柄
  void* p2pApiEventHandle;
  // 通用事件句柄
  void* eventHandle;
  // 此任务使用的通道数量
  uint8_t nChannels;
};

// ============================================================================
// ncclKernelPlan - 内核执行计划
// ============================================================================
// 描述一个 CUDA 内核的执行计划（可能包含多个集合通信操作）
struct ncclKernelPlan {
  // A kernel plan is also a callback that reclaims itself. Hence this must
  // be the first member.
  // 内核计划也是一个可以自我回收的回调，因此这必须是第一个成员
  struct ncclCommCallback reclaimer;

  // 关联的通信器
  struct ncclComm* comm;
  // 指向下一个内核计划的指针（形成计划队列）
  struct ncclKernelPlan* next;

  // 标志位
  bool persistent;  // aka captured in a graph（是否被 CUDA Graph 捕获）
  bool isHostCbEnq;  // 是否为主机回调入队
  bool isSymColl;    // 是否为对称集合通信
  bool isCeColl;     // 是否为 Copy Engine 集合通信
  // 工作存储类型
  enum ncclDevWorkStorageType workStorageType;
  // 内核是否已特化
  bool kernelSpecialized;
  // 执行的核函数
  void* kernelFn;

  // 联合体：内核参数（根据类型不同使用不同的参数结构）
  union {
    // 普通内核参数
    struct ncclDevKernelArgs* kernelArgs;
    // 对称内核参数
    void* kernelSymArgs;
    // Copy Engine 内核参数
    struct ncclCeCollArgs* ceCollArgs;
  };
  // 内核参数大小
  size_t kernelArgsSize;
  // 通道掩码（哪些通道存在于此计划中）
  uint64_t channelMask; // bitset of which channels are present
  // 是否有任何通道有非空的代理操作队列
  bool hasProxyOps; // does any channel have a non-empty proxyOpQueue
  // 每个线程块的线程数
  int threadPerBlock;

  // 此计划中的集合通信操作数量
  int collOpCount; // Number of collectives in this plan.
  // 工作批次数数量
  int nWorkBatches; // Number of work batches.
  // FIFO 中所有工作的总大小（字节）
  size_t workBytes; // Sum size of all work (in the fifo) in bytes.
  // 工作队列（有序链表）
  struct ncclIntruQueue<struct ncclWorkList, &ncclWorkList::next> workQueue;
  // 清理队列（有序链表）
  struct ncclIntruQueue<struct ncclCommCallback, &ncclCommCallback::next> cleanupQueue;
  // 持久化工作缓冲区
  void* workBufPersistent;

  // P2P 任务队列（有序链表）
  struct ncclIntruQueue<struct ncclTaskP2p, &ncclTaskP2p::next> p2pTaskQueue;
  // 集合通信任务队列（有序链表）
  struct ncclIntruQueue<struct ncclTaskColl, &ncclTaskColl::next> collTaskQueue;
  // 代理操作队列（有序链表）
  struct ncclIntruQueue<struct ncclProxyOp, &ncclProxyOp::enqNext> proxyOpQueue;

  // Profiler plugin
  void* groupApiEventHandle;
  void* kernelLaunchEventHandle;
  void* groupEventHandle;
};

////////////////////////////////////////////////////////////////////////////////
// ncclTaskCollSorter - 集合通信任务排序器
////////////////////////////////////////////////////////////////////////////////
// Roughly sorts ncclTaskColl's by their size descending. This structure is
// self-referential, meaning that pointers it contains internally may point
// into the structure itself. This means that it is NOT memcpy-moveable:
// 大致按大小降序排序 ncclTaskColl。此结构是自引用的，意味着其内部包含的指针
// 可能指向结构本身。这意味着它不能通过 memcpy 移动：
struct ncclTaskCollSorter {
  // 单位大小的对数（以 2 为底）
  static constexpr int UnitLog2 = 10; // 1K
  // 默认 1KB
  static constexpr size_t UnitSize = 1<<UnitLog2; // 1KB
  // 最大大小的对数
  static constexpr int MaxLog2 = 30; // 1GB
  // 默认 1GB
  static constexpr size_t MaxSize = 1ull<<MaxLog2; // 1GB 大小
  // Number of bins between powers of 2. For 4 bins, the worst case out-of-order
  // relative magnitude is (5/4)-1 = 25%
  // 在 2 的幂次之间的 bin 数量。对于 4 个 bin，最坏情况的乱序相对幅度为 (5/4)-1 = 25%
  static constexpr int BitsPerPow2 = 2;
  // 默认 4
  static constexpr int BinsPerPow2 = 1<<BitsPerPow2;
  // 默认 1+20*4=81
  static constexpr int BinCount = 1 + (MaxLog2-UnitLog2)*BinsPerPow2;

  // 链表头指针
  struct ncclTaskColl* head;
  // 链表尾指针
  struct ncclTaskColl* tail;
  // Least bin such that it and all above are empty.
  // 记录当前最大的非空 bin 编号
  int binEdge;
  // Pointer to the pointer to this bin's head node which is either the
  // previous node's `next` field or `head`.
  // 索引数组（指向每个 bin 的头节点的指针的指针）
  struct ncclTaskColl** bins[BinCount];
};

// ============================================================================
// ncclTaskCollSorterInsert - 插入任务到排序器
// ============================================================================
inline void ncclTaskCollSorterInsert(
    struct ncclTaskCollSorter* me, struct ncclTaskColl* x, size_t size
    ) {
  // 10
  constexpr int UnitLog2 = ncclTaskCollSorter::UnitLog2;
  // 1GB
  constexpr size_t MaxSize = ncclTaskCollSorter::MaxSize;
  // 2
  constexpr int BitsPerPow2 = ncclTaskCollSorter::BitsPerPow2;
  // 81
  constexpr int BinCount = ncclTaskCollSorter::BinCount;
  // 转化为 KB（右移 10 位相当于除以 1024）
  int bin = u32fpEncode(std::min(MaxSize, size)>>UnitLog2, BitsPerPow2);
  // 例：
  // size = 3 MiB → 3 MiB >> 10 = 3072 → u32fpEncode(3072, 2) = 12
  // bin = 81 - 1 - 12 = 68
  // 所以 3 MiB 落在 bin=68（降序索引）
  // size 越大，bin 越大，在数组中的索引越小
  bin = BinCount-1 - bin; // descending bin

  // 如果该 bin 为空
  if (me->bins[bin] == nullptr) {
    if (me->binEdge <= bin) {
      // 更新 bin 边缘
      me->binEdge = bin+1;
      // 设置 bin 指针（指向链表尾部或头部）
      me->bins[bin] = me->tail ? &me->tail->next : &me->head;
      // 更新链表尾部
      me->tail = x;
    } else {
      // Find successor non-empty bin after this one.
      // 查找此 bin 之后的下一个非空 bin
      int succ = bin+1;
      while (me->bins[succ] == nullptr) succ++;
      // What was our successor's head's previous is now our head's previous.
      // 我们后继者的头的前驱现在是我们的头的前驱
      me->bins[bin] = me->bins[succ];
      // The first node we insert is our tail, so that becomes our successor's
      // head's new previous.
      // 我们插入的第一个节点是我们的尾，所以它成为我们后继者的头的新前驱
      me->bins[succ] = &x->next;
    }
  }
  // Push a new head for this bin.
  // x 插入到头部中，这里同一个 bin 没有排序
  x->next = *me->bins[bin];
  // 记录任务 x
  *me->bins[bin] = x;
}

// ============================================================================
// ncclTaskCollSorterEmpty - 检查排序器是否为空
// ============================================================================
inline bool ncclTaskCollSorterEmpty(struct ncclTaskCollSorter* me) {
  return me->head == nullptr;
}

// Reset sorter and return sorted linked list of its coll tasks.
// 重置排序器并返回其集合通信任务的排序链表
inline struct ncclTaskColl* ncclTaskCollSorterDequeueAll(struct ncclTaskCollSorter* me) {
  struct ncclTaskColl* head = me->head;
  // 如果有任务，清空排序器
  if (head != nullptr) memset(me, 0, sizeof(*me));
  return head;
}

////////////////////////////////////////////////////////////////////////////////

// ============================================================================
// ncclCudaStreamList - CUDA 流链表节点
// ============================================================================
struct ncclCudaStreamList {
  // 指向下一个节点的指针
  struct ncclCudaStreamList *next;
  // CUDA 流
  cudaStream_t stream;
};

// ============================================================================
// ncclKernelPlanner - 内核规划器
// ============================================================================
// 负责将任务组织成内核执行计划
struct ncclKernelPlanner {
  ////////////////////////////////////////////////////////////////////////////
  // State for accumulating tasks between ncclGroupStart/End()
  // 在 ncclGroupStart/End() 之间累积任务的状态
  ////////////////////////////////////////////////////////////////////////////

  // 对等端信息（每个 peer 的发送/接收状态）
  struct Peer {
    // 是否已看到发送/接收操作
    bool sendSeen, recvSeen;
    // 发送任务队列（有序链表）
    struct ncclIntruQueue<struct ncclTaskP2p, &ncclTaskP2p::next> sendQueue;
    // 接收任务队列（有序链表）
    struct ncclIntruQueue<struct ncclTaskP2p, &ncclTaskP2p::next> recvQueue;
  };
  // 跟踪和排序 ncclTaskColl
  struct ncclTaskCollSorter collSorter;
  // 对等端数组（大小为 nRanks）
  struct Peer* peers/*[nRanks]*/;
  // 当前排队集合通信操作的任务个数
  int nTasksColl;

  // P2P 任务计数
  int nTasksP2p;
  // P2P 发送/接收任务计数
  int nTasksP2pSend, nTasksP2pRecv;
  // 是否持久化（被 CUDA Graph 捕获）
  bool persistent;
  // The list of user streams aggregated over all tasks present.
  // 所有任务中聚合的用户流列表
  struct ncclCudaStreamList* streams;
  // The most recent user stream. Ignored if streams==nullptr
  // 最新的用户流（如果 streams==nullptr 则忽略）
  cudaStream_t streamRecent;
  // The graph capturing all user streams or invalid if none. Thus we restrict the
  // user that all streams must be captured in the same graph or not captured
  // at all. Technically we could probably relax this, but that would mean
  // collecting a different `ncclTasks` per graph and one for non-graph.
  // 捕获所有用户流的图，如果没有则为无效。因此我们限制用户所有流必须在同一个图中捕获
  // 或者完全不捕获。技术上我们可能可以放宽这个限制，但这意味着为每个图收集不同的 `ncclTasks`
  // 以及一个用于非图的。
  struct ncclCudaGraph capturingGraph;

  ////////////////////////////////////////////////////////////////////////////
  // Lists of tasks to be assembled into plans.
  // 将要组装成计划的任务列表。
  ////////////////////////////////////////////////////////////////////////////

  // 集合通信任务队列（有序链表）
  struct ncclIntruQueue<struct ncclTaskColl, &ncclTaskColl::next> collTaskQueue;
  // Copy Engine 集合通信任务队列（有序链表）
  struct ncclIntruQueue<struct ncclTaskColl, &ncclTaskColl::next> collCeTaskQueue;
  // 对称集合通信任务队列（有序链表）
  struct ncclIntruQueue<struct ncclTaskColl, &ncclTaskColl::next> collSymTaskQueue;
  // 集合通信工作队列（有序链表）
  struct ncclIntruQueue<struct ncclWorkList, &ncclWorkList::next> collWorkQueue;
  // 临时集合通信工作队列（有序链表）
  struct ncclIntruQueue<struct ncclWorkList, &ncclWorkList::next> tmpCollWorkQueue;
  // 集合通信清理队列（有序链表）
  struct ncclIntruQueue<struct ncclCommCallback, &ncclCommCallback::next> collCleanupQueue;

  ////////////////////////////////////////////////////////////////////////////
  // State for building current (Work-In-Progress) plan:
  // 构建当前（进行中）计划的状态：
  ////////////////////////////////////////////////////////////////////////////

  // 工作中的计划（Work-In-Progress Plan）
  struct WipPlan {
    // 通道状态
    struct Channel {
      struct {
        // 此批次引用的工作元数据总大小
        int workBytes; // Sum size of work metadata referenced by this batch.
        // 此批次中的 P2P 工作数量
        int nP2ps; // Number of p2p works in this batch
        // 标记此批次中存在哪些轮次
        int p2pRounds[NCCL_MAX_DEV_WORK_P2P_PER_BATCH]; // which rounds are present in this batch.
      } wipBatch; // work-in-progress batch which will be next tail of workBatchQueue
      // 此通道的 P2P 批次数
      int nWorkBatchesP2p; // number of p2p batches for this channel.
      // 工作批次队列（有序链表）
      struct ncclIntruQueue<struct ncclWorkBatchList, &ncclWorkBatchList::next> workBatchQueue;
      // 代理操作队列（有序链表）
      struct ncclIntruQueue<struct ncclProxyOp, &ncclProxyOp::enqNext> proxyOpQueue;
    } channels[MAXCHANNELS];
  } wipPlan;

  ////////////////////////////////////////////////////////////////////////////
  // State for launching built plans:
  // 启动已构建计划的状态：
  ////////////////////////////////////////////////////////////////////////////

  // List of kernel plans built form tasks.
  // 从任务构建的内核计划队列（有序链表）
  struct ncclIntruQueue<struct ncclKernelPlan, &ncclKernelPlan::next> planQueue;
  // First of the unlaunched kernels in `planQueue`
  // `planQueue` 中第一个未启动的内核计划
  struct ncclKernelPlan* unlaunchedPlansHead;
};

// ============================================================================
// NCCL 魔数
// ============================================================================
#define NCCL_MAGIC 0x0280028002800280 // Nickel atomic number is 28.
// 镍的原子序数是 28，魔数用于校验通信器的完整性

// ============================================================================
// 组任务类型枚举
// ============================================================================
typedef enum ncclGroupTaskType {
  ncclGroupTaskTypeCollective = 0,  // 集合通信任务
  ncclGroupTaskTypeSymRegister = 1, // 对称内存注册任务
  ncclGroupTaskTypeNum = 2,         // 任务类型数量
} ncclGroupTaskType_t;

// 前向声明
struct ncclCommSymTeams;

// ============================================================================
// ncclComm - NCCL 通信器核心结构
// ============================================================================
// 这是 NCCL 的核心数据结构，表示一个通信上下文
struct ncclComm {
  // 起始魔数（用于校验结构完整性，必须是第一个字段）
  uint64_t startMagic;

  // 内存栈：用于管理通信器的内存分配
  struct ncclMemoryStack memPermanent,  // 永久内存（通信器生命周期内一直存在）
                        memScoped;      // 作用域内存（可回收）
  // List of destructors to run when comm is destructed
  // 析构时运行的析构器链表头指针
  struct ncclDestructor* destructorHead;

  // cuda 上下文，commAlloc 中分配
  struct ncclCudaContext* context;

  // 共享资源，commAlloc 中分配
  struct ncclSharedResources* sharedRes;

  /* map to top parent ranks. */
  // 用于映射到顶层父进程的排名和本地排名。
  int* topParentRanks;      // 顶层父 rank 数组（全局 rank -> 顶层父 rank）
  int* topParentLocalRanks; // 顶层父本地 rank 数组（全局 rank -> 顶层父本地 rank）

  // 通信通道，每个通信器最大 64 个
  struct ncclChannel channels[MAXCHANNELS];

  // 数组，存储一个通信组内所有 Rank 的设备信息
  struct ncclPeerInfo* peerInfo;

  // 拓扑系统结构体，用于表示和存储系统的硬件拓扑信息。这个结构体对于优化集合通信操作至关重要，
  // 因为它包含了关于系统中多个 GPU、CPU、网络设备（如网卡）以及它们之间的连接
  // （例如 PCIe 总线、NVLink 等）的详细信息。
  struct ncclTopoSystem* topo;

  // 全局代理连接器
  struct ncclProxyConnector* gproxyConn;
  // 传统注册清理队列
  struct ncclIntruQueue<struct ncclCommCallback, &ncclCommCallback::next> legacyRegCleanupQueue;
  // 标记 peerInfo 信息是否同步完成
  bool peerInfoValid;

  // 插件网络通信
  // ncclNetPluginLoad 中尝试加载 libnccl-net.so 插件，解析 ncclNet_t 上下文
  // 在 libnccl-net.so 中实现 ncclNet_t 结构体，优先使用 libnccl-net.so 的实现
  // 如果没有加载这个插件，则使用 nccl 内置的 ncclNetIb 实现和 ncclNetSocket
  // 所以 ncclNet 不会为空
  // 支持 IB 则直接使用 ncclNetIb
  ncclNet_t* ncclNet;
  // 保存 ncclNet 的 context 上下文
  void* netContext;
  // 网络插件索引
  int netPluginIndex;
  // NCCL 网络版本号
  int ncclNetVer;
  // 网络设备类型
  ncclNetDeviceType netDeviceType;

  // coll 是 collective 的缩写
  // CollNet 是 NVIDIA 提供的一种专门优化集合通信的网络技术
  // 需要 NVIDIA 和 Mellanox switches 的硬件支持
  // 对应 sharp 网络插件 https://github.com/Mellanox/nccl-rdma-sharp-plugins
  // https://github.com/NVIDIA/nccl/issues/1725
  // ncclNetPluginLoad 中尝试加载 collnet 插件，解析 ncclCollNet_t 上下文
  // 在 libnccl-net.so 中实现 ncclCollNet 结构体
  // 如果没有加载这个插件。表示不支持 ncclCollNet
  ncclCollNet_t* ncclCollNet;
  // 保存 ncclCollNet 的 context 上下文
  void* collNetContext;
  // ncclNet_t 更侧重于定义网络通信的基本操作，而 ncclCollNet_t 则用于集合通信操作的插件支持。

  // 记录启动网络信息，指向 struct bootstrapState
  void* bootstrap;

  // Bitmasks for ncclTransportP2pSetup
  // 每一个 bit 对应通道 id 是否有效
  // 比如 011，表示通道 0，1 有效
  uint64_t* connectSend;  // 发送连接位掩码数组
  uint64_t* connectRecv;  // 接收连接位掩码数组

  // 对应每个通信算法，生成一个通信图
  struct ncclTopoGraph graphs[NCCL_NUM_ALGORITHMS];

  // 最大树模式数量
  int maxTreePattern;
  // 每个算法的通道是否已初始化
  bool initAlgoChannels[NCCL_NUM_ALGORITHMS];
  // 是否支持动态连接
  bool runtimeConn; // if dynamic connection is supported
  // 是否使用直接模式
  bool directMode;
  // 是否支持 cuMem
  int cuMemSupport;

  // Magic number for all network communication. Not a security key -- only goal is to detect mismatches.
  // 所有网络通信的魔数。不是安全密钥——仅用于检测不匹配。
  uint64_t magic;

  // 通信器哈希值（用于标识通信器）
  uint64_t commHash;

  // 当前通信器在通信域中的 Rank 编号，从 0 开始编号
  // 因为单个线程可以创建多个通信器
  int rank;    // my rank in the communicator
  // 当前通信组内 gpu 的数量，也就是总的 rank 数量
  int nRanks;  // number of GPUs in communicator

  // 通信器绑定的 cuda gpu 设备索引号
  int cudaDev; // my cuda device index
  // NVML 是 NVIDIA Management Library，
  // 这是 NVML 设备索引，即 GPU 在 NVML 下枚举中的编号（0, 1, 2, ...）。
  int nvmlDev; // my nvml device index

  // gpu 的计算能力，通过调用 ncclCudaCompCap
  // 计算出的一个整数值
  int compCap; // compute capability of the GPU
  // 通信器中所有 GPU 的最小/最大计算能力
  int minCompCap, maxCompCap; // min/max compute capability in the communicator

  // gpu 的 bus id，唯一
  int64_t busId;   // my PCI bus ID in int format
  // cpu 亲和性
  cpu_set_t cpuAffinity; // CPU affinity of the GPU

  // 版本号，表示 cuda 支持哪些能力
  int cudaArch; // matches __CUDA_ARCH__ of device

  // CPU 架构（如 x86/arm/ppc/mixed）
  int cpuArch;   // architecture - As defined in src/include/graph.h, e.g. x86/arm/ppc/mixed
  // CPU 厂商
  int cpuVendor; // vendor - As defined in src/include/graph.h

  // 当前节点编号，可以理解为物理机器的编号
  int node;
  // 总节点编号
  int nNodes;
  // 当前 node 上的局部 localrank 编号，不同 node 会重复
  int localRank;
  // 当前 node 上的 rank 数量
  int localRanks;

  // comm->maxLocalRanks 将包含所有节点中 localRanks 的最大值。
  int maxLocalRanks;
  // comm->minLocalRanks 将包含所有节点中 localRanks 的最小值。
  int minLocalRanks;

  // int 数组
  // rank 得到 Node
  int* rankToNode;
  // 全局 rank 得到 localrank
  int* rankToLocalRank;
  // localrank 得到全局 rank
  int* localRankToRank;

  // localRanks and localRanktoRank for all nodes
  //
  struct ncclNodeRanks* nodeRanks;

  // MNNVL: Multi-Node NVLink
  // 标记是否支持多节点 NVLink
  // 支持跨机器节点的 nvlink
  // 允许跨节点的 GPU 通过 NVLink 进行直接通信
  // MNNVL 通过在节点之间部署 NVLink 桥接器，实现了跨节点的 NVLink 连接。
  // 这意味着不同节点上的 GPU 可以直接通过 NVLink 进行通信，无需经过传统的网络设备。
  int MNNVL; // true when MNNVL is available
  // Our MNNVL clique information
  struct cliqueInfo clique;
  // Our rank within the MNNVL clique
  int cliqueRank;

  // NVL Domain info
  // NVLink 域信息（版本 5）
  ncclNvlDomainInfo_v5_t nvlDomainInfo;

  // 是否检查指针
  bool checkPointers;
  // 是否支持 dma-buf
  bool dmaBufSupport;

  // Counter for tracking CUDA launches (P2P and collectives included)
  // CUDA 启动计数器（包括 P2P 和集合通信）
  uint64_t opCount;
  // Collective operation counter
  // 集合通信操作计数器
  uint64_t collOpCount;

  // Channels for collectives
  int nChannels;        // 连接通道数（实际建立的连接）
  int collChannels;     // 入队通道数（用于集合通信）
  // NVLink SHARP 通道数
  int nvlsChannels;     // 入队通道数（用于 NVLS）
  // all nvls heads stored to check if we can splitShare
  // 所有 NVLS 头部（用于检查是否可以 splitShare）
  int nvlsHeads[MAXCHANNELS];
  // Channels (per peer) for p2p
  int p2pnChannels;        // P2P 总通道数
  int p2pnChannelsPerPeer; // 每个 peer 的 P2P 通道数

  // Should this comm allocate LL buffers for network P2P connections?
  // 此通信器是否应该为网络 P2P 连接分配 LL 缓冲区？
  bool allocP2pNetLLBuffers;

  // Buffer sizes
  // 各协议的缓冲区大小（LL, LL128, Simple）
  int buffSizes[NCCL_NUM_PROTOCOLS];
  // P2P 块大小
  int p2pChunkSize;
  // NVLS 块大小
  int nvlsChunkSize;

  // Tuner values
  // 调优常量
  ncclTunerConstants_t tunerConstants;
  // 每个算法和协议的线程阈值
  ssize_t threadThresholds[NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS];
  // 延迟表（函数 x 算法 x 协议）
  float latencies[NCCL_NUM_FUNCTIONS][NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS];
  // 带宽表（函数 x 算法 x 协议）
  float bandwidths[NCCL_NUM_FUNCTIONS][NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS];
  // 最大线程数（算法 x 协议）
  int maxThreads[NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS];

  /* This attribute can indicate the states of communicators and return code of
   * asynchronous NCCL operations. */
  // 此属性可以指示通信器的状态和异步 NCCL 操作的返回码。
  ncclResult_t asyncResult;

  // Flag to ask NCCL kernels to abort
  // 要求 NCCL 内核中止的标志
  uint32_t* abortFlag;        // 主机端中止标志
  uint32_t* abortFlagDev;     // 设备端中止标志
  int* abortFlagRefCount;     // 中止标志引用计数
  uint32_t* childAbortFlag;   // 子通信器中止标志
  uint32_t* childAbortFlagDev;// 子通信器设备端中止标志
  uint32_t destroyFlag;       // 销毁标志

  // Device side of the communicator (for cudaFree's)
  // 通信器的设备端（用于 cudaFree）
  struct ncclKernelComm* devComm; // actually = &ncclKernelCommAndChannels::comm

  // 工作参数字节数（内核参数的最大大小）
  uint32_t workArgsBytes; // max size of kernel args
  // 工作 FIFO 字节数（workFifoBuf 的大小，2 的幂）
  uint32_t workFifoBytes; // size of workFifoBuf, power of 2
  // 工作 FIFO 缓冲区（主机端）
  void* workFifoBuf;
  // 工作 FIFO 缓冲区（设备端）
  void* workFifoBufDev;
  // 工作 FIFO GDRCopy 句柄
  void* workFifoBufGdrHandle;

  // Monotonic number of bytes (mod 1<<32) sent to fifo.
  // 发送到 fifo 的单调字节数（模 1<<32）
  uint32_t workFifoProduced;
  // 上次记录的生产计数
  uint32_t workFifoProducedLastRecorded;
  // 消费计数
  uint32_t workFifoConsumed;

  // Intra-process sync
  // 节点内同步
  // 进程内通信的领导者（可能是自己）
  struct ncclComm* intraComm0; // leader of intra-process comms (self possible)
  // 进程内下一个通信器，intraComm0 是头
  struct ncclComm* intraNext; // next of intra-process comms, intraComm0 is head

  // 进程内 rank
  int intraRank;
  // 同一进程内的 ranks 总数
  int intraRanks;

  // 进程内屏障阶段
  uint32_t intraBarrierPhase;
  // 填充到 64 字节
  char intraPad1[64 - sizeof(uint64_t)];
  // 进程内屏障计数器（仅 intraComm0 使用）
  uint64_t intraBarrierCounter; // only used if this is intraComm0
  // 填充到 64 字节
  char intraPad2[64 - sizeof(uint64_t)];
  // 进程内屏障门（仅 intraComm0 使用）
  uint64_t intraBarrierGate; // only used if this is intraComm0

  // 代理状态
  struct ncclProxyState* proxyState;
  // 存储代理原子减后的引用计数
  int proxyRefCountOld; /* store proxy post-atomic-sub refcount */
  // Whether this communicator uses collNet
  // 此通信器是否只有一个 rank per node
  bool isOneRPN;
  // CollNet 支持矩阵（sum, prod, max, min）x 数据类型
  uint8_t collNetSupportMatrix[4/*sum,prod,max,min*/][ncclNumTypes];
  // CollNet 头节点数组
  int* collNetHeads;
  // CollNet 头节点数量
  int collNetHeadsNum;
  // CollNet 稠密 rank 到用户 rank 的映射
  int* collNetDenseToUserRank;
  // 用户 rank 到 CollNet 稠密 rank 的映射
  int* collNetUserToDenseRank;
  /* sharable collNet proxy progress resource. */
  // 可共享的 CollNet 代理进度资源
  struct ncclCollNetSharedRes* collNetSharedRes;

  // NVLink SHARP (NVLS) support
  // 是否支持 NVLink SHARP
  int nvlsSupport;
  int nvlsRegSupport;
  /* sharable NVLS resource. */
  // 可共享的 NVLS 资源
  struct ncclNvlsSharedRes* nvlsResources;

  // pools backed by comm->memPermanent
  // 由 comm->memPermanent 支持的内存池
  struct ncclMemoryPool memPool_ncclTaskColl;
  struct ncclMemoryPool memPool_ncclTaskP2p;
  struct ncclMemoryPool memPool_ncclProxyOp;
  struct ncclMemoryPool memPool_ncclKernelPlan;

  // Next comm in this thread's active ncclGroup[Start|End](). Holds "0x1" when
  // this comm is not yet in a group.
  // 此线程的活动 ncclGroup[Start|End]() 中的下一个 comm。
  // 当此 comm 尚未在组中时保存 "0x1"。
  struct ncclComm* groupNext[ncclGroupTaskTypeNum];
  // Subset of those in groupNext list. Holds 0x1 if not needing preconnect.
  // groupNext 列表的子集。如果不需要预连接则保存 0x1。
  struct ncclComm* preconnectNext;
  // 持久化计划列表的数量（捕获此 comm 的持久化计划列表）
  int localPersistentRefs; // number of persistent plan-lists capturing this comm
  // P2P 调度对数组
  struct P2pSchedulePair {
    int sendRank;
    int recvRank;
  } *p2pSchedule;

  // kernel 任务调度队列
  struct ncclKernelPlanner planner;

  // 创建的 cuda 内存池
  cudaMemPool_t memPool;
  // Queue of events and associated callbacks for cleaning up asynchronous work.
  // Using this is preferable to using CUDA host callbacks because host callbacks
  // won't allow the work following the callback to run until the callback completes,
  // which comes at expense to perf.
  // 用于清理异步工作的事件和关联回调队列。
  // 使用此队列优于使用 CUDA 主机回调，因为主机回调不允许回调之后的工作运行，
  // 直到回调完成，这会影响性能。
  struct ncclIntruQueue<struct ncclCommEventCallback, &ncclCommEventCallback::next> eventCallbackQueue;

  // user-created reduction ops
  // 用户创建的归约操作
  int userRedOpCapacity,  // 用户归约操作容量
      userRedOpFreeHead;   // 空闲链表头
  ncclUserRedOp *userRedOps; // 用户归约操作数组

  // Queue of things for the main thread to do
  // 主线程要做的事情的队列
  int reclaimSteps;
  struct ncclIntruQueueMpsc<struct ncclCommCallback, &ncclCommCallback::next> callbackQueue;

  // NCCL 配置
  ncclConfig_t config;
  // initState is to more conveniently reclaim resources when errors happen.
  // initState 用于在发生错误时更方便地回收资源。
  ncclResult_t initState;
  // flag to indicate if ncclCommFinalize() is called
  // 标志：指示是否调用了 ncclCommFinalize()
  bool finalizeCalled;
  // shared structures for finalization
  // 用于终止的共享结构
  int finalizeRankCnt;
  // group job to support multi-thread FT
  // 支持多线程容错的组作业
  struct ncclGroupJob *groupJob;

  // Flag indicating if this communicator shares resources with parent or children
  // 标志：指示此通信器是否与父或子通信器共享资源
  bool shareResources;

  // Tuning plugin
  // 调优插件
  int tunerPluginLoaded;     // 调优插件是否已加载
  ncclTuner_t* tuner;        // 调优插件接口
  // 调优插件上下文
  void *tunerContext;

  // Profiler plugin
  // 性能分析器上下文
  void* profilerContext;
  // 每个函数的序列号
  uint64_t seqNumber[NCCL_NUM_FUNCTIONS];
  // 性能分析器代理
  struct ncclProfilerProxy profiler;

  // CE Collective
  // 复制引擎，CEs 是一种专门用于内存拷贝的硬件单元
  // 对于那些纯数据搬运的集合操作（如 AllGather, AlltoAll），
  // NCCL 可以完全绕过 SM，直接将通信任务编程到复制引擎上去执行。
  // 限制：
  // 1. 依赖对称内存注册：ncclCommWindowRegister
  // 2. 用于芯片内或通过 NVLink 直连的 GPU 间的高速内存传输，
  //    不适用于需要复杂网络协议栈的跨节点 InfiniBand/RoCE 通信。
  // 3. 目前仅支持 AlltoAll, AllGather, Scatter, Gather 等纯数据移动操作。
  //    像 AllReduce 这种需要计算（例如求和）的操作，仍然需要 SM 的参与，无法由 CE 单独完成。
  struct ncclCeColl ceColl;
  // CE 初始化任务队列
  struct ncclIntruQueue<struct ncclCeInitTask, &ncclCeInitTask::next> ceInitTaskQueue;

  // buffer registration cache
  // 缓冲区注册缓存
  struct ncclRegCache regCache;
  // 是否全为 NVLink
  int isAllNvlink;
  // 是否全为直连 P2P
  bool isAllDirectP2p;
  // 对称支持级别
  int symmetricSupport;
  // 是否使用 PXN（Proxy eXchange Network）
  bool useNetPXN;
  // 是否使用 GPUDirect RDMA
  bool useGdr;
  // split 计数
  int splitCount;

  // 对称运行时状态（基于对称内存注册）
  struct ncclDevrState devrState; // The symmetric runtime state
  // 对称内核状态（构建在前者之上）
  struct ncclSymkState symkState; // The symmetric kernels state (built on previous)

  // 结束魔数（必须是最后一个字段）
  uint64_t endMagic;
};

// ============================================================================
// 静态断言：确保魔数在正确位置
// ============================================================================
// 确保 startMagic 是第一个字段
static_assert(offsetof(struct ncclComm, startMagic) == 0, "startMagic must be the first field of ncclComm");
// 确保 endMagic 是最后一个字段
static_assert(offsetof(struct ncclComm, endMagic) == sizeof(struct ncclComm) - sizeof(uint64_t), "endMagic must be the last field of ncclComm");

// ============================================================================
// 内核启动模式枚举
// ============================================================================
enum ncclLaunchMode {
  ncclLaunchModeInvalid=0,  // 无效模式
  ncclLaunchModeParallel,   // 并行启动模式
  ncclLaunchModeGroup       // 组启动模式
};
// 内核启动模式参数（可通过环境变量设置）
extern enum ncclLaunchMode ncclParamLaunchMode;

// ============================================================================
// 内存释放函数声明
// ============================================================================
// 将缓冲区添加到释放队列（延迟释放）
void ncclCommPushFree(struct ncclComm* comm, void* buf);
// 将 CUDA 内存添加到释放队列
void ncclCommPushCudaFree(struct ncclComm* comm, void* buf);
// 将 CUDA 主机内存添加到释放队列
void ncclCommPushCudaHostFree(struct ncclComm* comm, void* buf);
// 将 GDRCopy 句柄添加到释放队列
void ncclCommPushCudaGdrFree(struct ncclComm* comm, void* handle);

// ============================================================================
// ncclCommPollCallbacks - 轮询并执行回调
// ============================================================================
inline ncclResult_t ncclCommPollCallbacks(struct ncclComm* comm, bool waitSome) {
  ncclResult_t result = ncclSuccess;
  // 从 MPSC 队列中出队所有回调（如果 waitSome 为 true，则至少等待一个）
  struct ncclCommCallback* cb = ncclIntruQueueMpscDequeueAll(&comm->callbackQueue, waitSome);
  // 遍历所有回调并执行
  while (cb != nullptr) {
    struct ncclCommCallback* next = cb->next;
    // 执行回调函数（可能会回收 cb 的内存）
    ncclResult_t res1 = cb->fn(comm, cb); // may reclaim memory of cb
    if (res1 != ncclSuccess) result = res1;
    cb = next;
  }
  NCCLCHECK(result);
  return ncclSuccess;
}

// ============================================================================
// ncclCommPollEventCallbacks - 轮询并执行 Event 回调
// ============================================================================
inline ncclResult_t ncclCommPollEventCallbacks(struct ncclComm *comm, bool waitSome) {
  ncclResult_t result = ncclSuccess;
  // 临时切换到宽松捕获模式（允许在图捕获期间查询 event）
  cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed;
  CUDACHECK(cudaThreadExchangeStreamCaptureMode(&mode));
  while (true) {
    // 查看队列头部的回调
    struct ncclCommEventCallback* cb = ncclIntruQueueHead(&comm->eventCallbackQueue);
    if (cb == nullptr) break;
    cudaError_t ok;
    if (waitSome) {
      // 等待 Event 完成
      ok = cudaEventSynchronize(cb->event);
      waitSome = false;
    } else {
      // 查询 Event 状态
      ok = cudaEventQuery(cb->event);
      if (ok == cudaErrorNotReady) break;
    }
    // 从队列中移除
    ncclIntruQueueDequeue(&comm->eventCallbackQueue);
    if (ok == cudaSuccess) {
      // Event 成功，执行回调
      NCCLCHECKGOTO(cb->fn(comm, cb), result, finish);
    } else {
      // Event 失败，返回错误
      CUDACHECKGOTO(ok, result, finish);
    }
  }
finish:
  // 恢复捕获模式
  CUDACHECK(cudaThreadExchangeStreamCaptureMode(&mode));
  return ncclSuccess;
}

// ============================================================================
// ncclCommIntraBarrierIn - 进程内屏障进入
// ============================================================================
inline void ncclCommIntraBarrierIn(struct ncclComm* comm, uint32_t x) {
  // 获取当前屏障阶段
  int phase = comm->intraBarrierPhase;
  if (comm->intraRanks == 1) {
    // 只有一个 rank，直接释放
    // Release everyone (just me).
    comm->intraBarrierGate = (uint64_t(x)<<32) | (phase^1);
  } else {
    // 多个 rank，使用原子操作
    struct ncclComm* comm0 = comm->intraComm0;
    // 原子递增计数器（高 32 位存 x，低 32 位存计数）
    uint64_t count = __atomic_add_fetch(&comm0->intraBarrierCounter, (uint64_t(x)<<32) + 1, __ATOMIC_RELEASE);
    // 如果所有 rank 都到达
    if (uint32_t(count) == uint32_t(comm->intraRanks)) {
      // Reset.
      __atomic_store_n(&comm0->intraBarrierCounter, 0, __ATOMIC_RELAXED);
      // Release everyone.
      __atomic_store_n(&comm0->intraBarrierGate, (count>>32<<32) | (phase^1), __ATOMIC_RELEASE);
    }
  }
}

// returns sum of x values contributed to ncclCommIntraBarrierIn(comm, x)
// 返回贡献给 ncclCommIntraBarrierIn(comm, x) 的所有 x 值的总和
// ============================================================================
// ncclCommIntraBarrierOut - 进程内屏障退出
// ============================================================================
inline uint32_t ncclCommIntraBarrierOut(struct ncclComm* comm) {
  struct ncclComm* comm0 = comm->intraComm0;
  // 切换到下一个阶段
  comm->intraBarrierPhase ^= 1;
  uint32_t phase = comm->intraBarrierPhase;
  // 加载屏障门
  uint64_t gate = __atomic_load_n(&comm0->intraBarrierGate, __ATOMIC_RELAXED);
  // 如果阶段不匹配，等待
  if ((gate & 1) != phase) {
    uint64_t t0 = clockNano();
    do {
      // Spin vigorously for first 5us.
      // 前 5 微秒积极自旋
      if (clockNano()-t0 >= 5*1000) sched_yield();
      gate = __atomic_load_n(&comm0->intraBarrierGate, __ATOMIC_RELAXED);
    } while ((gate & 1) != phase);
  }
  // 获取内存屏障（多个 rank 时需要）
  if (comm->intraRanks != 1) __atomic_thread_fence(__ATOMIC_ACQUIRE);
  // 返回所有 x 的总和
  return gate>>32;
}

// Scrambles the bits of non-builtin values of ncclRedOp_t according to the
// communicator memory address. Used to catch bugs so that integer handles
// associated with this communicator won't collide with handles of other
// communicatrs. This function is its own inverse.
// 根据通信器内存地址打乱 ncclRedOp_t 非内置值的位。
// 用于捕获 bug，使得与此通信器关联的整数句柄不会与其他通信器的句柄冲突。
// 此函数是其自身的逆函数（再次调用可还原）。
// ============================================================================
// ncclUserRedOpMangle - 打乱用户归约操作句柄
// ============================================================================
static inline ncclRedOp_t ncclUserRedOpMangle(ncclComm *comm, ncclRedOp_t op) {
  // Preserve the built-in values.
  // 保留内置值（如 sum, max, min 等）
  if(int(op) < int(ncclNumOps))
    return op;
  // 使用 comm 指针生成哈希
  uint64_t h = reinterpret_cast<uint64_t>(comm);
  h ^= h >> 32;
  // Knuth 的 64 位魔法哈希常数
  h *= 0x9e3779b97f4a7c13u; // Knuth's 64-bit magical hash constant
  h >>= 32; // h is now an excellent 32-bit hash of the comm pointer
           // h 现在是 comm 指针的一个优秀的 32 位哈希
  // 掩码到 ncclMaxRedOp 范围（ncclMaxRedOp 是 2 的幂次减 1）
  h &= int(ncclMaxRedOp); // ncclMaxRedOp is a power of 2 minus 1
  // 异或操作
  int op1 = int(h) ^ int(op);
  // Since builtin values are preserved, we also have to preserve their preimage.
  // 由于保留了内置值，我们还必须保留它们的前像。
  return op1 < int(ncclNumOps) ? op : ncclRedOp_t(op1);
}

// ============================================================================
// 函数声明
// ============================================================================
// 确保通信器已就绪
ncclResult_t ncclCommEnsureReady(ncclComm_t comm);
// 设置异步错误
ncclResult_t ncclCommSetAsyncError(ncclComm_t comm, ncclResult_t nextState);

#endif
