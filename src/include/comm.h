/*************************************************************************
 * Copyright (c) 2015-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_COMM_H_
#define NCCL_COMM_H_

//#include "transport.h"
#include "p2p.h"
#include "collectives.h"
#include "nccl_tuner.h"
#include "proxy.h"
#include "strongstream.h"
#include "nccl_net.h"
#include "register.h"
#include "graph.h"
#include "profiler.h"
#include "allocator.h"
#include "dev_runtime.h"
#include "sym_kernels.h"
#include "ce_coll.h"

#if CUDART_VERSION < 9000
struct cudaLaunchParams {
  void *func;
  dim3 gridDim;
  dim3 blockDim;
  void **args;
  size_t sharedMem;
  cudaStream_t stream;
};
#endif

#define CACHE_LINE_SIZE 128
#define MEM_ALIGN 4096
#define CUDA_IPC_MIN 2097152UL

// Channels / LL tuning
#define NCCL_LL_THREAD_THRESHOLD 8
#define NCCL_LL128_THREAD_THRESHOLD 8
#define NCCL_SIMPLE_THREAD_THRESHOLD 64

struct ncclSendMem {
  union {
    struct {
      uint64_t head;
      char pad1[CACHE_LINE_SIZE-sizeof(uint64_t)];
      void* ptrExchange;
      uint64_t redOpArgExchange[2];
      char pad2[CACHE_LINE_SIZE-sizeof(void*)-2*sizeof(uint64_t)];
      int offsFifo[NCCL_STEPS];
    };
    char pad3[MEM_ALIGN];
  };
};

struct ncclRecvMem {
  union {
    struct {
      uint64_t tail;
      char pad1[CACHE_LINE_SIZE-sizeof(uint64_t)];
      struct ncclConnFifo connFifo[NCCL_STEPS];
      int flush; // For GDRCopy-based flush
    };
    char pad4[MEM_ALIGN];
  };
};

enum helperThreadState {ThreadStart, ThreadStop};

#define NCCL_IPC_POOL_SIZE (2*NCCL_MAX_LOCAL_RANKS*NCCL_MAX_OPS)

struct ncclGraphHelperResources {
  ncclComm* comm;
  pthread_mutex_t threadLock;
  pthread_cond_t  threadCond;
  enum helperThreadState threadState;
  void* ipcBases[NCCL_IPC_POOL_SIZE];
  int ipcTail;
  int ipcHead;
};

struct ncclUserRedOp {
  int freeNext; // -1=allocated, otherwise index of next free entry in array
  ncclDataType_t datatype;
  ncclDevRedOpFull opFull;
};

struct ncclNodeRanks {
  int localRanks;
  int* localRankToRank;
};

struct cliqueInfo {
  int id;
  int size;
  int *ranks;
};

struct ncclDestructor {
  struct ncclDestructor* next;
  void* obj;
  ncclResult_t(*fn)(struct ncclDestructor* me);
};

struct ncclCommCallback {
  struct ncclCommCallback* next;
  ncclResult_t(*fn)(struct ncclComm* comm, struct ncclCommCallback* cb);
};
struct ncclCommEventCallback {
  struct ncclCommEventCallback* next;
  cudaEvent_t event;
  ncclResult_t(*fn)(struct ncclComm* comm, struct ncclCommEventCallback* cb);
};

struct ncclSharedResources {
  int refCount;
  struct ncclComm* owner; /* comm which creates this shared res. */
  struct ncclChannelPeer* peers[MAXCHANNELS];
  struct ncclDevChannelPeer* devPeers[MAXCHANNELS];
  /* P2P operation counter, one per channel */
  uint64_t p2pOpCount[MAXCHANNELS];
  /* Collective operation counter */
  uint64_t collOpCount;
  //tp是top parent 的缩写
  int tpNRanks;
  int tpNLocalRanks;
  int tpNChannels;
  int tpP2pNChannels;
  int tpP2pChunkSize;
  uint64_t magic;

  // top parent rank to localRank translation table
  int* tpRankToLocalRank;
  // Internal streams
  struct ncclStrongStream deviceStream, hostStream;
  int persistentRefs;
  cudaEvent_t launchEvent, scratchEvent;

  /* proxy related shared res */
  struct ncclProxyState* proxyState;
};

struct ncclChannel {
  struct ncclChannelPeer** peers;
  struct ncclDevChannelPeer** devPeers;
  /* devPeer pointer array used for host side access */
  struct ncclDevChannelPeer** devPeersHostPtr;
  struct ncclRing ring;
  int* devRingUserRanks;
  struct ncclTree tree;

  struct ncclTree collnetChain;
  struct ncclDirect collnetDirect;

  struct ncclNvls nvls;

  //从0开始按序分配
  int id; // index of this channel
  uint32_t workFifoProduced; // +1 successor of last used work fifo byte

  /* comm split sharable resources */
  struct ncclChannelPeer* collnetPeers;
  struct ncclDevChannelPeer* collnetDevPeers;
  struct ncclChannelPeer* nvlsPeers;
  struct ncclDevChannelPeer* nvlsDevPeers;
};

struct ncclWorkBatchList {
  struct ncclWorkBatchList* next;
  struct ncclDevWorkBatch batch;
};
struct alignas(16) ncclWorkList {
  struct ncclWorkList* next;
  enum ncclDevWorkType workType;
  int size; // Size of struct following this node
  // ncclDevWorkColl, ncclDevWorkColLReg, ncclDevWorkP2p[]...
};

struct ncclCollnetHandleList {
  struct ncclCollnetHandleList *next;
  void* collnetHandle;
  size_t size;
  const void* buffer;
  struct ncclProxyConnector* proxyconn;
};

struct ncclTaskColl {
  struct ncclTaskColl* next;
  ncclFunc_t func;
  void const* sendbuff;
  void* recvbuff;
  size_t count;
  int root;
  ncclDataType_t datatype;
  ncclRedOp_t opHost;
  struct ncclDevRedOpFull opDev;
  int chunkSteps, sliceSteps;
  // Computed later:
  size_t trafficBytes;
  int32_t nMaxChannels:8;
  int32_t nWarps:8;
  int32_t algorithm:8, protocol:8;
  uint32_t isCollnet:1, isNvls:1, isSymLast:1;
  uint32_t devFuncId:29;
  int regBufType;
  // number of elements in planner->ipcMemQueue associated with this collective
  int nCleanupQueueElts;

  struct ncclDevrWindow* sendWin;
  struct ncclDevrWindow* recvWin;
  void* sendMhandle;
  void* recvMhandle;
  void** sendNetHandles;
  void** recvNetHandles;
  void** srecvNetHandles;
  // index for IPC record lookup
  uintptr_t sendbuffOffset;
  uintptr_t recvbuffOffset;
  uintptr_t* sendbuffRmtAddrs;
  uintptr_t* recvbuffRmtAddrs;

  // Profiler plugin
  int eActivationMask;
  void* groupApiEventHandle;
  void* collApiEventHandle;
  void* eventHandle;
  uint8_t nChannels;
};

struct ncclTaskP2p {
  struct ncclTaskP2p* next;
  ncclFunc_t func;
  ncclFunc_t collAPI;
  void* buff;
  size_t count;
  ncclDataType_t datatype;
  int root;
  size_t bytes;

  // Profiler plugin
  int eActivationMask;
  void* groupApiEventHandle;
  void* p2pApiEventHandle;
  void* eventHandle;
  uint8_t nChannels;
};

struct ncclKernelPlan {
  // A kernel plan is also a callback that reclaims itself. Hence this must
  // be the first member.
  struct ncclCommCallback reclaimer;

  struct ncclComm* comm;
  struct ncclKernelPlan* next;

  bool persistent; // aka captured in a graph
  bool isHostCbEnq;
  bool isSymColl;
  bool isCeColl;
  enum ncclDevWorkStorageType workStorageType;
  bool kernelSpecialized;
  //执行的核函数
  void* kernelFn;
  
  union {
    struct ncclDevKernelArgs* kernelArgs;
    void* kernelSymArgs;
    struct ncclCeCollArgs* ceCollArgs;
  };
  size_t kernelArgsSize;
  uint64_t channelMask; // bitset of which channels are present
  bool hasProxyOps; // does any channel have a non-empty proxyOpQueue
  int threadPerBlock;

  int collOpCount; // Number of collectives in this plan.
  int nWorkBatches; // Number of work batches.
  size_t workBytes; // Sum size of all work (in the fifo) in bytes.
  struct ncclIntruQueue<struct ncclWorkList, &ncclWorkList::next> workQueue;
  struct ncclIntruQueue<struct ncclCommCallback, &ncclCommCallback::next> cleanupQueue;
  void* workBufPersistent;

  struct ncclIntruQueue<struct ncclTaskP2p, &ncclTaskP2p::next> p2pTaskQueue;
  struct ncclIntruQueue<struct ncclTaskColl, &ncclTaskColl::next> collTaskQueue;
  struct ncclIntruQueue<struct ncclProxyOp, &ncclProxyOp::enqNext> proxyOpQueue;

  // Profiler plugin
  void* groupApiEventHandle;
  void* kernelLaunchEventHandle;
  void* groupEventHandle;
};

////////////////////////////////////////////////////////////////////////////////
// Roughly sorts ncclTaskColl's by their size descending. This structure is
// self-referential, meaning that pointers it contains internally may point
// into the structure itself. This means that it is NOT memcpy-moveable:

struct ncclTaskCollSorter {
  static constexpr int UnitLog2 = 10; // 1K
  //默认1KB
  static constexpr size_t UnitSize = 1<<UnitLog2; //1KB
  static constexpr int MaxLog2 = 30; // 1GB
  //默认1GB
  static constexpr size_t MaxSize = 1ull<<MaxLog2; //1GB大小
  // Number of bins between powers of 2. For 4 bins, the worst case out-of-order
  // relative magnitude is (5/4)-1 = 25%
  static constexpr int BitsPerPow2 = 2;
  //默认4
  static constexpr int BinsPerPow2 = 1<<BitsPerPow2;
  //默认1+20*4=81
  static constexpr int BinCount = 1 + (MaxLog2-UnitLog2)*BinsPerPow2;

  struct ncclTaskColl* head;
  struct ncclTaskColl* tail;
  // Least bin such that it and all above are empty.
  //记录当前最大的非空bin编号
  int binEdge;
  // Pointer to the pointer to this bin's head node which is either the
  // previous node's `next` field or `head`.
  //索引数组
  struct ncclTaskColl** bins[BinCount];
};

inline void ncclTaskCollSorterInsert(
    struct ncclTaskCollSorter* me, struct ncclTaskColl* x, size_t size
  ) {
  //10
  constexpr int UnitLog2 = ncclTaskCollSorter::UnitLog2;
  //1GB
  constexpr size_t MaxSize = ncclTaskCollSorter::MaxSize;
  //2
  constexpr int BitsPerPow2 = ncclTaskCollSorter::BitsPerPow2;
  //81
  constexpr int BinCount = ncclTaskCollSorter::BinCount;
  //转化为KB
  int bin = u32fpEncode(std::min(MaxSize, size)>>UnitLog2, BitsPerPow2);
  //例：
//size = 3 MiB → 3 MiB >> 10 = 3072 → u32fpEncode(3072, 2) = 12
//bin = 81 - 1 - 12 = 68
//所以 3 MiB 落在 bin=68（降序索引）
//size越大，bin越大，在数组中的索引越小
  bin = BinCount-1 - bin; // descending bin

  if (me->bins[bin] == nullptr) {
    if (me->binEdge <= bin) {
      me->binEdge = bin+1;
      me->bins[bin] = me->tail ? &me->tail->next : &me->head;
      me->tail = x;
    } else {
      // Find successor non-empty bin after this one.
      int succ = bin+1;
      while (me->bins[succ] == nullptr) succ++;
      // What was our successor's head's previous is now our head's previous.
      me->bins[bin] = me->bins[succ];
      // The first node we insert is our tail, so that becomes our successor's
      // head's new previous.
      me->bins[succ] = &x->next;
    }
  }
  // Push a new head for this bin.
  //x插入到头部中，这里同一个bin没有排序
  x->next = *me->bins[bin];
  //记录任务x
  *me->bins[bin] = x;
}

inline bool ncclTaskCollSorterEmpty(struct ncclTaskCollSorter* me) {
  return me->head == nullptr;
}

// Reset sorter and return sorted linked list of its coll tasks.
inline struct ncclTaskColl* ncclTaskCollSorterDequeueAll(struct ncclTaskCollSorter* me) {
  struct ncclTaskColl* head = me->head;
  if (head != nullptr) memset(me, 0, sizeof(*me));
  return head;
}

////////////////////////////////////////////////////////////////////////////////

struct ncclCudaStreamList {
  struct ncclCudaStreamList *next;
  cudaStream_t stream;
};

struct ncclKernelPlanner {
  //////////////////////////////////////////////////////////////////////////////
  // State for accumulating tasks between ncclGroupStart/End()
  //////////////////////////////////////////////////////////////////////////////

  struct Peer {
    bool sendSeen, recvSeen;
    struct ncclIntruQueue<struct ncclTaskP2p, &ncclTaskP2p::next> sendQueue;
    struct ncclIntruQueue<struct ncclTaskP2p, &ncclTaskP2p::next> recvQueue;
  };
  //跟踪和排序ncclTaskColl
  struct ncclTaskCollSorter collSorter;
  struct Peer* peers/*[nRanks]*/;
  //当前排队集合通信操作的任务个数
  int nTasksColl;
  
  int nTasksP2p;
  
  int nTasksP2pSend, nTasksP2pRecv;
  bool persistent;
  // The list of user streams aggregated over all tasks present.
  struct ncclCudaStreamList* streams;
  // The most recent user stream. Ignored if streams==nullptr
  cudaStream_t streamRecent;
  // The graph capturing all user streams or invalid if none. Thus we restrict the
  // user that all streams must be captured in the same graph or not captured
  // at all. Technically we could probably relax this, but that would mean
  // collecting a different `ncclTasks` per graph and one for non-graph.
  struct ncclCudaGraph capturingGraph;

  //////////////////////////////////////////////////////////////////////////////
  // Lists of tasks to be assembled into plans.
  //////////////////////////////////////////////////////////////////////////////

  struct ncclIntruQueue<struct ncclTaskColl, &ncclTaskColl::next> collTaskQueue;
  struct ncclIntruQueue<struct ncclTaskColl, &ncclTaskColl::next> collCeTaskQueue;
  struct ncclIntruQueue<struct ncclTaskColl, &ncclTaskColl::next> collSymTaskQueue;
  struct ncclIntruQueue<struct ncclWorkList, &ncclWorkList::next> collWorkQueue;
  struct ncclIntruQueue<struct ncclWorkList, &ncclWorkList::next> tmpCollWorkQueue;
  struct ncclIntruQueue<struct ncclCommCallback, &ncclCommCallback::next> collCleanupQueue;

  //////////////////////////////////////////////////////////////////////////////
  // State for building current (Work-In-Progress) plan:
  //////////////////////////////////////////////////////////////////////////////

  struct WipPlan {
    struct Channel {
      struct {
        int workBytes; // Sum size of work metadata referenced by this batch.
        int nP2ps; // Number of p2p works in this batch
        int p2pRounds[NCCL_MAX_DEV_WORK_P2P_PER_BATCH]; // which rounds are present in this batch.
      } wipBatch; // work-in-progress batch which will be next tail of workBatchQueue
      int nWorkBatchesP2p; // number of p2p batches for this channel.
      struct ncclIntruQueue<struct ncclWorkBatchList, &ncclWorkBatchList::next> workBatchQueue;
      struct ncclIntruQueue<struct ncclProxyOp, &ncclProxyOp::enqNext> proxyOpQueue;
    } channels[MAXCHANNELS];
  } wipPlan;

  //////////////////////////////////////////////////////////////////////////////
  // State for launching built plans:
  //////////////////////////////////////////////////////////////////////////////

  // List of kernel plans built form tasks.
  struct ncclIntruQueue<struct ncclKernelPlan, &ncclKernelPlan::next> planQueue;
  // First of the unlaunched kernels in `planQueue`
  struct ncclKernelPlan* unlaunchedPlansHead;
};

#define NCCL_MAGIC 0x0280028002800280 // Nickel atomic number is 28.

typedef enum ncclGroupTaskType {
  ncclGroupTaskTypeCollective = 0,
  ncclGroupTaskTypeSymRegister = 1,
  ncclGroupTaskTypeNum = 2,
} ncclGroupTaskType_t;

struct ncclCommSymTeams;

struct ncclComm {
  uint64_t startMagic;
  struct ncclMemoryStack memPermanent, memScoped;
  // List of destructors to run when comm is destructed
  struct ncclDestructor* destructorHead;

  //cuda上下文
  struct ncclCudaContext* context;
  //共享资源
  struct ncclSharedResources* sharedRes;
  /* map to top parent ranks. */
  //用于映射到顶层父进程的排名和本地排名。
  int* topParentRanks;
  int* topParentLocalRanks;

  //通信通道，每个通信器最大64
  struct ncclChannel channels[MAXCHANNELS];
  
  //数组，存储一个通信组内所有 Rank 的设备信息
  struct ncclPeerInfo* peerInfo;
  //拓扑系统结构体，用于表示和存储系统的硬件拓扑信息。这个结构体对于优化集合通信操作至关重要，
  //因为它包含了关于系统中多个GPU、CPU、网络设备（如网卡）以及它们之间的连接
  //（例如PCIe总线、NVLink等）的详细信息。
  struct ncclTopoSystem* topo;

  struct ncclProxyConnector* gproxyConn;
  struct ncclIntruQueue<struct ncclCommCallback, &ncclCommCallback::next> legacyRegCleanupQueue;
  //标记peerInfo信息是否同步完成
  bool peerInfoValid;

  //插件网络通信
  //ncclNetPluginLoad中尝试加载libnccl-net.so插件, 解析ncclNet_t上下文
  //在libnccl-net.so中实现ncclNet_t结构体，优先使用libnccl-net.so的实现
  //如果没有加载这个插件，则使用nccl内置的ncclNetIb实现和ncclNetSocket
  //所以ncclNet不会为空
  //支持IB则直接使用ncclNetIb
  ncclNet_t* ncclNet;
  //保存ncclNet的context上下文
  void* netContext;
  int netPluginIndex;
  int ncclNetVer;
  ncclNetDeviceType netDeviceType;

  //coll是collective的缩写
  //CollNet 是 NVIDIA 提供的一种专门优化集合通信的网络技术
  //需要NVIDIA和Mellanox switches 的硬件支持
  //对应sharp网络插件https://github.com/Mellanox/nccl-rdma-sharp-plugins
  //https://github.com/NVIDIA/nccl/issues/1725
  //ncclNetPluginLoad中尝试加载collnet插件, 解析ncclCollNet_t上下文
  //在libnccl-net.so中实现ncclCollNet结构体
  //如果没有加载这个插件。表示不支持ncclCollNet
  ncclCollNet_t* ncclCollNet;
  //保存ncclCollNet的context上下文
  void* collNetContext;
  //ncclNet_t更侧重于定义网络通信的基本操作，而ncclCollNet_t则用于集合通信操作的插件支持。

  //记录启动网络信息，指向struct bootstrapState
  void* bootstrap;
 
  // Bitmasks for ncclTransportP2pSetup
  //每一个bit对应通道id是否有效
  //比如011，表示通道0，1有效
  uint64_t* connectSend;
  uint64_t* connectRecv;

  //对应每个通信算法，生成一个通信图
  struct ncclTopoGraph graphs[NCCL_NUM_ALGORITHMS];

  int maxTreePattern;
  bool initAlgoChannels[NCCL_NUM_ALGORITHMS];
  bool runtimeConn; // if dynamic connection is supported
  bool directMode;
  //是否支持cumem
  int cuMemSupport;

  uint64_t magic; // Magic number for all network communication. Not a security key -- only goal is to detect mismatches.

  uint64_t commHash;
  
  //当前通信器在通信域中的 Rank编号,从0开始编号
  //因为单个线程可以创建多个通信器
  int rank;    // my rank in the communicator
  //当前通信组内gpu的数量，也就是总的rank数量
  int nRanks;  // number of GPUs in communicator
  
  //通信器绑定的cuda gpu设备索引号
  int cudaDev; // my cuda device index
  //NVML 是 NVIDIA Management Library，
  //这是 NVML 设备索引，即 GPU 在 NVML 下枚举中的编号（0, 1, 2, ...）。
  int nvmlDev; // my nvml device index
  
  //gpu的计算能力，通过调用ncclCudaCompCap
  //计算出的一个整数值
  int compCap; // compute capability of the GPU
  int minCompCap, maxCompCap; // min/max compute capability in the communicator
  
  //gpu的bus id，唯一
  int64_t busId;   // my PCI bus ID in int format
  //cpu亲和性
  cpu_set_t cpuAffinity; // CPU affinity of the GPU

  //版本号，表示cuda支持哪些能力
  int cudaArch; // matches __CUDA_ARCH__ of device

  int cpuArch;   // architecture - As defined in src/include/graph.h, e.g. x86/arm/ppc/mixed
  int cpuVendor; // vendor - As defined in src/include/graph.h

//当前节点编号，可以理解为物理机器的编号
  int node;
//总节点编号
  int nNodes;
//当前node上的局部localrank编号，不同node会重复
  int localRank;
//当前node上的rank数量
  int localRanks;

  //comm->maxLocalRanks 将包含所有节点中 localRanks 的最大值。
  int maxLocalRanks;
  //comm->minLocalRanks 将包含所有节点中 localRanks 的最小值。
  int minLocalRanks;

  //int数组
  //rank得到Node
  int* rankToNode;
  //全局rank得到localrank
  int* rankToLocalRank;
  //localrank得到全局rank
  int* localRankToRank;
  
  // localRanks and localRanktoRank for all nodes
  //
  struct ncclNodeRanks* nodeRanks;
  
  // MNNVL: Multi-Node NVLink
  //标记是否支持多节点NVLink
  //支持跨机器节点的nvlink
  //允许跨节点的GPU通过NVLink进行直接通信
  //MNNVL通过在节点之间部署NVLink桥接器，实现了跨节点的NVLink连接。
  //这意味着不同节点上的GPU可以直接通过NVLink进行通信，无需经过传统的网络设备。
  int MNNVL; // true when MNNVL is available
  struct cliqueInfo clique; // Our MNNVL clique information
  int cliqueRank; // Our rank within the MNNVL clique

  // NVL Domain info
  ncclNvlDomainInfo_v5_t nvlDomainInfo;

  bool checkPointers;
  //是否支持dma-buf
  bool dmaBufSupport;

  // Counter for tracking CUDA launches (P2P and collectives included)
  uint64_t opCount;
  // Collective operation counter
  uint64_t collOpCount;

  // Channels for collectives
  int nChannels; // connection nChannels
  int collChannels; // enqueue nChannels
  //NVLink SHARP通道数
  int nvlsChannels; // enqueue nChannels
  // all nvls heads stored to check if we can splitShare
  int nvlsHeads[MAXCHANNELS];
  // Channels (per peer) for p2p
  int p2pnChannels;
  int p2pnChannelsPerPeer;

  // Should this comm allocate LL buffers for network P2P connections?
  bool allocP2pNetLLBuffers;

  // Buffer sizes
  int buffSizes[NCCL_NUM_PROTOCOLS];
  int p2pChunkSize;
  int nvlsChunkSize;

  // Tuner values
  ncclTunerConstants_t tunerConstants;
  ssize_t threadThresholds[NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS];
  float latencies[NCCL_NUM_FUNCTIONS][NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS];
  float bandwidths[NCCL_NUM_FUNCTIONS][NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS];
  int maxThreads[NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS];

  /* This attribute can indicate the states of communicators and return code of
   * asynchronous NCCL operations. */
  ncclResult_t asyncResult;

  // Flag to ask NCCL kernels to abort
  uint32_t* abortFlag;
  uint32_t* abortFlagDev;
  int* abortFlagRefCount;
  uint32_t* childAbortFlag;
  uint32_t* childAbortFlagDev;
  uint32_t destroyFlag;

  // Device side of the communicator (for cudaFree's)
  struct ncclKernelComm* devComm; // actually = &ncclKernelCommAndChannels::comm

  uint32_t workArgsBytes; // max size of kernel args
  uint32_t workFifoBytes; // size of workFifoBuf, power of 2
  void* workFifoBuf;
  void* workFifoBufDev;
  void* workFifoBufGdrHandle;

  // Monotonic number of bytes (mod 1<<32) sent to fifo.
  uint32_t workFifoProduced;
  uint32_t workFifoProducedLastRecorded;
  uint32_t workFifoConsumed;

  // Intra-process sync
  //节点内
  struct ncclComm* intraComm0; // leader of intra-process comms (self possible)
  struct ncclComm* intraNext; // next of intra-process comms, intraComm0 is head
  
  int intraRank;
  //同一进程内的ranks总数
  int intraRanks;
  
  uint32_t intraBarrierPhase;
  char intraPad1[64 - sizeof(uint64_t)];
  uint64_t intraBarrierCounter; // only used if this is intraComm0
  char intraPad2[64 - sizeof(uint64_t)];
  uint64_t intraBarrierGate; // only used if this is intraComm0

  struct ncclProxyState* proxyState;
  int proxyRefCountOld; /* store proxy post-atomic-sub refcount */
  // Whether this communicator uses collNet
  bool isOneRPN;
  uint8_t collNetSupportMatrix[4/*sum,prod,max,min*/][ncclNumTypes];
  int* collNetHeads;
  int collNetHeadsNum;
  int* collNetDenseToUserRank;
  int* collNetUserToDenseRank;
  /* sharable collNet proxy progress resource. */
  struct ncclCollNetSharedRes* collNetSharedRes;

  // NVLink SHARP (NVLS) support
  //是否支持NVLink SHARP
  int nvlsSupport;
  int nvlsRegSupport;
  /* sharable NVLS resource. */
  struct ncclNvlsSharedRes* nvlsResources;

  // pools backed by comm->memPermanent
  struct ncclMemoryPool memPool_ncclTaskColl;
  struct ncclMemoryPool memPool_ncclTaskP2p;
  struct ncclMemoryPool memPool_ncclProxyOp;
  struct ncclMemoryPool memPool_ncclKernelPlan;

  // Next comm in this thread's active ncclGroup[Start|End](). Holds "0x1" when
  // this comm is not yet in a group.
  struct ncclComm* groupNext[ncclGroupTaskTypeNum];
  // Subset of those in groupNext list. Holds 0x1 if not needing preconnect.
  struct ncclComm* preconnectNext;
  int localPersistentRefs; // number of persistent plan-lists capturing this comm
  struct P2pSchedulePair { 
    int sendRank;
    int recvRank; 
  } *p2pSchedule;

    //kernel任务调度队列
  struct ncclKernelPlanner planner;

//创建的cuda内存池
  cudaMemPool_t memPool;
  // Queue of events and associated callbacks for cleaning up asynchronous work.
  // Using this is preferable to using CUDA host callbacks because host callbacks
  // won't allow the work following the callback to run until the callback completes,
  // which comes at expense to perf.
  struct ncclIntruQueue<struct ncclCommEventCallback, &ncclCommEventCallback::next> eventCallbackQueue;

  // user-created reduction ops
  int userRedOpCapacity, userRedOpFreeHead;
  ncclUserRedOp *userRedOps;

  // Queue of things for the main thread to do
  int reclaimSteps;
  struct ncclIntruQueueMpsc<struct ncclCommCallback, &ncclCommCallback::next> callbackQueue;

  ncclConfig_t config;
  // initState is to more conveniently reclaim resources when errors happen.
  ncclResult_t initState;
  // flag to indicate if ncclCommFinalize() is called
  bool finalizeCalled;
  // shared structures for finalization
  int finalizeRankCnt;
  // group job to support multi-thread FT
  struct ncclGroupJob *groupJob;

  // Flag indicating if this communicator shares resources with parent or children
  bool shareResources;

  // Tuning plugin
  //调优插件
  int tunerPluginLoaded;
  ncclTuner_t* tuner;
  //调优插件上下文
  void *tunerContext;

  // Profiler plugin
  void* profilerContext;
  uint64_t seqNumber[NCCL_NUM_FUNCTIONS];
  struct ncclProfilerProxy profiler;

  // CE Collective
  //复制引擎 (Copy Engines, CEs)。
  //CE 是一种专门用于内存拷贝的硬件单元
  //对于那些纯数据搬运的集合操作（如 AllGather, AlltoAll），
  //NCCL 可以完全绕过 SM，直接将通信任务编程到复制引擎 (CEs) 上去执行。
  //限制：
  //1. 依赖对称内存注册：ncclCommWindowRegister
  //2. 用于芯片内或通过 NVLink 直连的 GPU 间的高速内存传输，
  //不适用于需要复杂网络协议栈的跨节点 InfiniBand/RoCE 通信。
  //3. 目前仅支持 AlltoAll, AllGather, Scatter, Gather 等纯数据移动操作。
  //像 AllReduce 这种需要计算（例如求和）的操作，仍然需要 SM 的参与，无法由 CE 单独完成。
  struct ncclCeColl ceColl;
  struct ncclIntruQueue<struct ncclCeInitTask, &ncclCeInitTask::next> ceInitTaskQueue;
  
  // buffer registration cache
  struct ncclRegCache regCache;
  int isAllNvlink;
  bool isAllDirectP2p;
  int symmetricSupport;
  bool useNetPXN;
  bool useGdr;
  int splitCount;

  struct ncclDevrState devrState; // The symmetric runtime state
  struct ncclSymkState symkState; // The symmetric kernels state (built on previous)

  uint64_t endMagic;
};

static_assert(offsetof(struct ncclComm, startMagic) == 0, "startMagic must be the first field of ncclComm");
static_assert(offsetof(struct ncclComm, endMagic) == sizeof(struct ncclComm) - sizeof(uint64_t), "endMagic must be the last field of ncclComm");

enum ncclLaunchMode {
  ncclLaunchModeInvalid=0,
  ncclLaunchModeParallel,
  ncclLaunchModeGroup
};
extern enum ncclLaunchMode ncclParamLaunchMode;

void ncclCommPushFree(struct ncclComm* comm, void* buf);
void ncclCommPushCudaFree(struct ncclComm* comm, void* buf);
void ncclCommPushCudaHostFree(struct ncclComm* comm, void* buf);
void ncclCommPushCudaGdrFree(struct ncclComm* comm, void* handle);

inline ncclResult_t ncclCommPollCallbacks(struct ncclComm* comm, bool waitSome) {
  ncclResult_t result = ncclSuccess;
  struct ncclCommCallback* cb = ncclIntruQueueMpscDequeueAll(&comm->callbackQueue, waitSome);
  while (cb != nullptr) {
    struct ncclCommCallback* next = cb->next;
    ncclResult_t res1 = cb->fn(comm, cb); // may reclaim memory of cb
    if (res1 != ncclSuccess) result = res1;
    cb = next;
  }
  NCCLCHECK(result);
  return ncclSuccess;
}

inline ncclResult_t ncclCommPollEventCallbacks(struct ncclComm *comm, bool waitSome) {
  ncclResult_t result = ncclSuccess;
  cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed;
  CUDACHECK(cudaThreadExchangeStreamCaptureMode(&mode));
  while (true) {
    struct ncclCommEventCallback* cb = ncclIntruQueueHead(&comm->eventCallbackQueue);
    if (cb == nullptr) break;
    cudaError_t ok;
    if (waitSome) {
      ok = cudaEventSynchronize(cb->event);
      waitSome = false;
    } else {
      ok = cudaEventQuery(cb->event);
      if (ok == cudaErrorNotReady) break;
    }
    ncclIntruQueueDequeue(&comm->eventCallbackQueue);
    if (ok == cudaSuccess) {
      NCCLCHECKGOTO(cb->fn(comm, cb), result, finish);
    } else {
      CUDACHECKGOTO(ok, result, finish);
    }
  }
finish:
  CUDACHECK(cudaThreadExchangeStreamCaptureMode(&mode));
  return ncclSuccess;
}

inline void ncclCommIntraBarrierIn(struct ncclComm* comm, uint32_t x) {
  int phase = comm->intraBarrierPhase;
  if (comm->intraRanks == 1) {
    // Release everyone (just me).
    comm->intraBarrierGate = (uint64_t(x)<<32) | (phase^1);
  } else {
    struct ncclComm* comm0 = comm->intraComm0;
    uint64_t count = __atomic_add_fetch(&comm0->intraBarrierCounter, (uint64_t(x)<<32) + 1, __ATOMIC_RELEASE);
    if (uint32_t(count) == uint32_t(comm->intraRanks)) {
      // Reset.
      __atomic_store_n(&comm0->intraBarrierCounter, 0, __ATOMIC_RELAXED);
      // Release everyone.
      __atomic_store_n(&comm0->intraBarrierGate, (count>>32<<32) | (phase^1), __ATOMIC_RELEASE);
    }
  }
}

// returns sum of x values contributed to ncclCommIntraBarrierIn(comm, x)
inline uint32_t ncclCommIntraBarrierOut(struct ncclComm* comm) {
  struct ncclComm* comm0 = comm->intraComm0;
  comm->intraBarrierPhase ^= 1;
  uint32_t phase = comm->intraBarrierPhase;
  uint64_t gate = __atomic_load_n(&comm0->intraBarrierGate, __ATOMIC_RELAXED);
  if ((gate & 1) != phase) {
    uint64_t t0 = clockNano();
    do {
      // Spin vigorously for first 5us.
      if (clockNano()-t0 >= 5*1000) sched_yield();
      gate = __atomic_load_n(&comm0->intraBarrierGate, __ATOMIC_RELAXED);
    } while ((gate & 1) != phase);
  }
  if (comm->intraRanks != 1) __atomic_thread_fence(__ATOMIC_ACQUIRE);
  return gate>>32;
}

// Scrambles the bits of non-builtin values of ncclRedOp_t according to the
// communicator memory address. Used to catch bugs so that integer handles
// associated with this communicator won't collide with handles of other
// communicatrs. This function is its own inverse.
static inline ncclRedOp_t ncclUserRedOpMangle(ncclComm *comm, ncclRedOp_t op) {
  // Preserve the built-in values.
  if(int(op) < int(ncclNumOps))
    return op;
  uint64_t h = reinterpret_cast<uint64_t>(comm);
  h ^= h >> 32;
  h *= 0x9e3779b97f4a7c13u; // Knuth's 64-bit magical hash constant
  h >>= 32; // h is now an excellent 32-bit hash of the comm pointer
  h &= int(ncclMaxRedOp); // ncclMaxRedOp is a power of 2 minus 1
  int op1 = int(h) ^ int(op);
  // Since builtin values are preserved, we also have to preserve their preimage.
  return op1 < int(ncclNumOps) ? op : ncclRedOp_t(op1);
}

ncclResult_t ncclCommEnsureReady(ncclComm_t comm);
ncclResult_t ncclCommSetAsyncError(ncclComm_t comm, ncclResult_t nextState);

#endif
