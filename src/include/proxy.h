/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 * 版权所有 (c) 2016-2022, NVIDIA CORPORATION. 保留所有权利。
 *
 * See LICENSE.txt for license information
 * 请参阅 LICENSE.txt 获取许可证信息
 ************************************************************************/

// 防止头文件重复包含的保护宏开始
// 如果未定义 NCCL_PROXY_H_ 宏，则定义它
#ifndef NCCL_PROXY_H_
#define NCCL_PROXY_H_

// 引入设备相关头文件，包含 GPU 设备操作的接口定义
#include "device.h"
// 引入信息头文件，包含系统和设备信息的结构定义
#include "info.h"
// 引入套接字头文件，提供网络套接字通信功能
#include "socket.h"
// 引入 IPC 套接字头文件，提供进程间通信的套接字功能
#include "ipcsocket.h"
// 引入 NCCL 网络头文件，定义网络插件接口
#include "nccl_net.h"
// 引入 POSIX 线程库头文件，提供线程创建、同步等功能
#include <pthread.h>
// 引入共享内存工具头文件，提供共享内存操作功能
#include "shmutils.h"
// 引入 P2P 通信头文件，包含点对点通信的实现
#include "p2p.h"
// 引入集合通信头文件，包含各种集合操作的实现
#include "collectives.h"

// 枚举类型：通信模式（使用 uint8_t 类型，节省内存）
// 定义了 NCCL 中各种通信和数据流动的模式
typedef enum : uint8_t {
  ncclPatternRing,                                      // 环形模式：数据在环形拓扑中单向流动
  ncclPatternRingTwice,                                 // 双环模式：数据在两个方向上流动（优化性能）
  ncclPatternPipelineFrom,                              // 流水线模式（从源）：单向流水线
  ncclPatternPipelineTo,                                // 流水线模式（到目标）：单向流水线
  ncclPatternTreeUp,                                    // 树形上行模式：数据从叶子节点流向根节点（Reduce）
  ncclPatternTreeDown,                                  // 树形下行模式：数据从根节点流向叶子节点（Broadcast）
  ncclPatternTreeUpDown,                                // 树形双向模式：先上行再下行（AllReduce）
  ncclPatternCollnetChain,                              // 集合网络链式模式：通过专用网络进行链式通信
  ncclPatternCollnetDirect,                             // 集合网络直连模式：通过专用网络直接通信
  ncclPatternNvls,                                      // NVLS 模式：使用 NVLink fabric 进行通信
  ncclPatternNvlsTree,                                  // NVLS 树形模式：结合 NVLS 和树形算法
  ncclPatternPatUp,                                     // PAT 上行模式：Path Aware Transport 上行
  ncclPatternPatDown,                                   // PAT 下行模式：Path Aware Transport 下行
  ncclPatternSend,                                      // 发送模式：点对点发送操作
  ncclPatternRecv,                                      // 接收模式：点对点接收操作
  ncclPatternProfiler,                                  // 性能分析器模式：用于性能测试和调优
} ncclPattern_t;

// 枚举类型：代理操作状态
// 定义代理操作在其生命周期中的不同状态
enum ncclProxyOpState {
  ncclProxyOpNone,                                      // 无操作状态：操作未初始化或已完成
  ncclProxyOpReady,                                     // 就绪状态：操作已准备好执行
  ncclProxyOpProgress                                   // 进行中状态：操作正在执行
};

// 前向声明：代理参数结构体
struct ncclProxyArgs;

// 函数指针类型：代理进度推进函数
// 指向负责推进代理操作进度的函数
// 参数：
//   proxyState: 代理状态结构体指针
//   args: 代理参数结构体指针
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
typedef ncclResult_t (*proxyProgressFunc_t)(struct ncclProxyState*, struct ncclProxyArgs*);

// 宏定义：代理操作的最大子操作数量
// 定义每个代理操作可以包含的子操作的最大数量，等于最大通道数
#define NCCL_PROXY_MAX_SUBS MAXCHANNELS

// 静态断言：确保有足够的空间存储最大的工作元素
// 验证两倍的最大每批 P2P 工作元素数不超过最大通道数
// 这是确保代理操作池能够容纳所有必要的子操作
static_assert(2*NCCL_MAX_DEV_WORK_P2P_PER_BATCH <= MAXCHANNELS, "Not enough sub space for max work elements");

// 联合体：代理操作的特定参数
// 用于存储不同通信模式特有的参数，节省内存空间
union ncclProxyOpSpecifics {
  struct {
    size_t sizePerRank;                                 // 每个 rank 的数据大小（collnetDirect 专用）
    int nNodes, node;                                   // 节点总数和当前节点编号
  } collnetDirect;                                      // 集合网络直连模式的参数
};

// 结构体：代理操作
// 描述一个代理通信操作的所有信息
struct ncclProxyOp {
  struct ncclProxyConnection* connection;               // 指向代理连接结构体的指针
  ssize_t nbytes;                                       // 操作的总字节数（有符号大小）
  uint64_t opCount;                                     // 操作计数器，用于标识操作的顺序
  int root;                                             // 根节点的 rank 编号（用于树形等算法）
  int next;                                             // 下一个节点的 rank 编号（用于环形等算法）
  int nsteps;                                           // 操作需要的步数
  size_t chunkSize;                                     // 数据块大小（每次传输的块大小）
  size_t sliceSize;                                     // 数据切片大小（更细粒度的数据划分）
  size_t loopSize;                                      // 循环迭代的数据大小
  size_t loopOffset;                                    // 循环迭代的数据偏移量
  size_t channelSize;                                   // 每个通道处理的数据大小
  uint8_t sliceSteps;                                   // 切片步数（uint8_t 节省空间）
  uint8_t chunkSteps;                                   // 块步数（uint8_t 节省空间）
  uint8_t channelId;                                    // 通道 ID（uint8_t，最多 256 个通道）
  uint8_t /*ncclDataType_t*/ dtype;                     // 数据类型（如 ncclInt32、ncclFloat 等）
  uint8_t /*ncclDevRedOp_t*/ redOp;                     // 归约操作（如 ncclSum、ncclProd 等）
  uint8_t /*ncclFunc_t*/ coll;                          // 集合通信类型（如 AllReduce、Broadcast 等）
  uint8_t /*ncclFunc_t*/ collAPI;                       // 集合通信 API 类型（用户调用的接口）
  uint8_t /*ncclPattern_t*/ pattern;                    // 通信模式（环形、树形等）
  uint8_t protocol;                                      // 通信协议（LL、LL128、Simple 等）
  uint8_t algorithm;                                     // 算法类型（Tree、Ring 等）
  uint8_t reg;                                           // 内存注册标志
  // collnet/p2p/coll buffer reg handles
  // 集合网络/P2P/集合缓冲区的内存注册句柄
  void* sendMhandle;                                    // 发送内存句柄（用于 RDMA 等操作）
  void* recvMhandle;                                    // 接收内存句柄（用于 RDMA 等操作）
  uint8_t* sendbuff;                                    // 发送缓冲区指针
  uint8_t* recvbuff;                                    // 接收缓冲区指针
  int isOneRPN;                                         // 每节点一个进程的标志（One Rank Per Node）
  RingAlgorithm *ringAlgo;                              // 指向环形算法对象的指针
  union ncclProxyOpSpecifics specifics;                 // 特定模式的参数（联合体）
  int nChannels;                                        // 使用的通道数量
  int nPeers;                                           // 对等节点（peer）数量

  // Profiler plugin
  // 性能分析器插件相关字段
  union {
    struct ncclTaskColl* coll;                          // 指向集合通信任务结构体的指针
    struct ncclTaskP2p* p2p;                            // 指向 P2P 任务结构体的指针
  } task;                                                // 任务联合体

  // Profiler work counter increment flag. Set to 'true' if the profiler work counter for this channel needs increment.
  // 性能分析器工作计数器递增标志。如果此通道的性能分析器工作计数器需要递增，则设置为 'true'。
  // Always 'true' for collective operations. Grouped p2p operations are fused into one <send, recv> pair in the GPU kernel,
  // 集合操作始终为 'true'。分组的 P2P 操作在 GPU 内核中融合为一个 <send, recv> 对，
  // meaning the GPU profiler code increments the work counter for the pair rather than the individual p2p. For this
  // 这意味着 GPU 性能分析器代码为该对递增工作计数器，而不是为单个 P2P 操作。因此，
  // reason, the incWorkCounter flag is used to avoid incrementing the work counter twice in the host code. This is done
  // incWorkCounter 标志用于避免在主机代码中两次递增工作计数器。这是通过在入队时仅将
  // by setting incWorkCounter to 'true' only for one of the p2ps in the pair during enqueue.
  // incWorkCounter 设置为 'true' 来实现的，仅对其中一个 P2P 操作设置。
  bool incWorkCounter;                                   // 工作计数器递增标志
  int eActivationMask;                                   // 事件激活掩码
  void* taskEventHandle;                                // 任务事件句柄
  int rank;                                             // 当前 rank 编号
  int peer;                                             // 对端 rank 编号
  pid_t pid;                                            // 进程 ID
  void* profilerContext;                                // 性能分析器上下文指针
  uint64_t workCounter;                                  // 工作计数器

  struct ncclProxyOp *enqNext;                          // 指向下一个入队操作的指针（用于链表）
};

// 前向声明：代理子参数结构体
struct ncclProxySubArgs;

// 结构体：代理事件句柄
// 用于跟踪代理操作中的事件
struct ncclProxyEventHandle {
  void* stepEventHandle;                                // 步骤事件句柄
  struct ncclProxySubArgs* subArgPtr;                   // 指向子参数结构体的指针
};

// 结构体：代理子参数
// 描述代理操作的一个子操作的信息，一个代理操作可能包含多个子操作
struct ncclProxySubArgs {
  struct ncclProxyConnection* connection;               // 指向代理连接结构体的指针
  int reg;                                               // 内存注册标志
  // collnet handles
  // 集合网络句柄
  void* sendMhandle;                                    // 发送内存句柄（用于 RDMA 等操作）
  void* recvMhandle;                                    // 接收内存句柄（用于 RDMA 等操作）
  uint8_t* sendbuff;                                    // 发送缓冲区指针
  uint8_t* recvbuff;                                    // 接收缓冲区指针
  size_t offset;                                        // 数据偏移量
  ssize_t loopSize;                                     // 循环迭代的数据大小（有符号）
  ssize_t loopOffset;                                   // 循环迭代的数据偏移量（有符号）
  int channelId;                                        // 通道 ID
  int nsteps;                                           // 操作需要的步数
  ssize_t nbytes;                                       // 子操作的字节数（有符号）
  ssize_t chunkSize;                                    // 数据块大小（有符号）
  int peer;                                             // 对端 rank 编号
  int isOneRPN;                                         // 每节点一个进程的标志（One Rank Per Node）
  RingAlgorithm *ringAlgo;                              // 指向环形算法对象的指针
  int groupSize; // Number of consecutive sub operations sharing the same recvComm（共享同一接收通信器的连续子操作数量）
  uint64_t base;                                        // 基础地址或计数值
  uint64_t posted;                                      // 已发布的数据量计数
  uint64_t received;                                    // 已接收的数据量计数
  uint64_t flushed;                                      // 已刷新的数据量计数
  uint64_t transmitted;                                 // 已传输的数据量计数
  uint64_t done;                                        // 已完成的数据量计数
  uint64_t end;                                         // 结束位置或计数值
  int regBufferReady;                                   // 注册缓冲区就绪标志
  void* requests[NCCL_STEPS];                           // 每一步的请求指针数组

  // Profiler plugin
  // 性能分析器插件相关字段
  int eActivationMask;                                   // 事件激活掩码
  int rank;                                             // 当前 rank 编号
  pid_t pid;                                            // 进程 ID
  void* profilerContext;                                // 性能分析器上下文指针
  void* taskEventHandle;                                // 任务事件句柄
  void* opEventHandle;                                  // 操作事件句柄
  void* kernelEventHandle;                              // 内核事件句柄
  struct ncclProxyEventHandle pHandles[NCCL_STEPS];     // 每一步的事件句柄数组
  size_t transSize;                                     // 传输的数据大小
  uint64_t workCounter;                                  // 工作计数器

  void* recvRequestsCache[NCCL_STEPS];                  // 接收请求缓存数组
  int recvRequestsSubCount;                             // 接收请求子计数
};

// 结构体：代理参数
// 描述一个代理操作的所有信息，包含多个子操作
struct ncclProxyArgs {
  struct ncclProxySubArgs subs[NCCL_PROXY_MAX_SUBS];   // 子操作数组
  proxyProgressFunc_t progress;                         // 进度推进函数指针
  int nsubs;                                            // 子操作的数量
  int done;                                             // 完成标志
  int onePPN;                                           // 每节点一个进程的标志（One Process Per Node）
  uint64_t opCount;                                     // 操作计数器
  int sliceSteps;                                       // 切片步数
  int chunkSteps;                                       // 块步数
  size_t chunkSize;                                     // 块大小
  size_t totalSendSize;                                 // 总发送大小
  size_t totalRecvSize;                                 // 总接收大小
  size_t sendSizePerRound;                              // 每轮发送大小
  size_t recvSizePerRound;                              // 每轮接收大小
  uint8_t /*ncclDataType_t*/ dtype;                     // 数据类型（如 ncclInt32、ncclFloat 等）
  uint8_t /*ncclDevRedOp_t*/ redOp;                     // 归约操作（如 ncclSum、ncclProd 等）
  uint8_t /*ncclPattern_t*/ pattern;                    // 通信模式（环形、树形等）
  uint8_t /*ncclFunc_t*/ coll;                          // 集合通信类型（如 AllReduce、Broadcast 等）
  uint8_t /*ncclFunc_t*/ collAPI;                       // 集合通信 API 类型（用户调用的接口）
  uint8_t protocol;                                      // 通信协议（LL、LL128、Simple 等）
  uint8_t algorithm;                                     // 算法类型（Tree、Ring 等）
  int state;                                            // 操作状态
  char* sharedBuff[NCCL_STEPS];                         // 每一步的共享缓冲区指针数组
  int sharedSize[NCCL_STEPS];                           // 每一步的共享缓冲区大小数组
  int nChannels;                                        // 使用的通道数量
  int nPeers;                                           // 对等节点（peer）数量

  int idle;                                             // 空闲标志

  // Element linking
  // 元素链接：用于构建代理参数的链表
  struct ncclProxyArgs* next;                           // 指向下一个代理参数的指针
  struct ncclProxyArgs* nextPeer;                       // 指向下一个对等节点的代理参数的指针
  struct ncclProxyArgs** proxyAppendPtr;                // 指向代理追加指针的指针

  union ncclProxyOpSpecifics specifics;                 // 特定模式的参数（联合体）
};
// 宏定义：最大网络设备数量
#define NCCL_MAX_NETDEVS 128

// ProxyOps are used to communicate between main thread and service thread
// ProxyOps 用于主线程和服务线程之间的通信
// Make sure we have enough to store two full rounds of operations on all channels.
// 确保我们有足够的空间来存储所有通道上两轮完整的操作。
// Otherwise we'd be unable to post half of them to free new elements. Each
// 否则我们将无法发布其中一半来释放新元素。每个
// p2p work contains a send and recv proxy op hence the 2x before it.
// P2P 工作包含发送和接收代理操作，因此前面有 2x。
#define MAX_OPS_PER_PEER (2*MAXCHANNELS*2*NCCL_MAX_DEV_WORK_P2P_PER_BATCH)

// 结构体：代理操作池
// 用于管理主线程和服务线程之间的代理操作
struct ncclProxyOpsPool {
  struct ncclProxyOp ops[MAX_OPS_PER_PEER*NCCL_MAX_LOCAL_RANKS]; // 代理操作数组
  volatile int nextOps;                                 // 下一个操作的索引（易变变量，用于线程间通信）
  volatile int nextOpsEnd;                              // 操作结束的索引（易变变量）
  volatile int freeOps[NCCL_MAX_LOCAL_RANKS];           // 每个 rank 的空闲操作索引（易变数组）
  pthread_mutex_t mutex;                               // 互斥锁，保护操作池的并发访问
  pthread_cond_t cond;                                 // 条件变量，用于线程同步
};

// 结构体：代理操作
// 包装代理操作池，提供访问接口
struct ncclProxyOps {
  ncclProxyOpsPool* pool;                              // 指向代理操作池的指针
  ncclShmHandle_t handle;                              // 共享内存句柄
  int count;                                           // 操作数量
  int freeOp;                                          // 空闲操作索引
  int nextOps;                                         // 下一个操作索引
  int nextOpsEnd;                                      // 操作结束索引
};

// 结构体：代理共享 P2P
// 描述 P2P 通信的共享资源
struct ncclProxySharedP2p {
  int refcount;                                        // 引用计数
  int size;                                            // 缓冲区大小
  char* cudaBuff;                                      // CUDA 缓冲区指针
  char* hostBuff;                                      // 主机缓冲区指针
  // CUDA IPC
  // CUDA 进程间通信
  ncclIpcDesc ipcDesc;                                // CUDA IPC 描述符
  struct ncclProxyArgs* proxyAppend[MAXCHANNELS];     // 代理追加指针数组（分离发送和接收）
};

// 结构体：代理对等节点
// 描述与一个对等节点的 P2P 通信资源
struct ncclProxyPeer {
  struct ncclProxySharedP2p send;                      // 发送方向的共享 P2P 资源
  struct ncclProxySharedP2p recv;                      // 接收方向的共享 P2P 资源
};

// 结构体：共享网络通信
// 描述网络通信的共享资源
struct ncclSharedNetComms {
  int activeConnect[MAXCHANNELS];                      // 活跃连接数组（每个通道）
  int activeAccept[MAXCHANNELS];                       // 活跃接受连接数组（每个通道）
  void* sendComm[MAXCHANNELS];                         // 发送通信器数组（每个通道）
  void* recvComm[MAXCHANNELS];                         // 接收通信器数组（每个通道）
  int sendRefCount[MAXCHANNELS];                       // 发送引用计数数组（每个通道）
  int recvRefCount[MAXCHANNELS];                       // 接收引用计数数组（每个通道）
};

// 前向声明：代理池结构体
struct ncclProxyPool;
// 结构体：代理进度状态
// 描述代理进度推进线程的状态
struct ncclProxyProgressState {
  // Used by main threads to send work to progress thread
  // 主线程使用这些字段向进度线程发送工作
  struct ncclProxyOpsPool* opsPool;                    // 指向代理操作池的指针
  ncclShmHandle_t handle;                              // 共享内存句柄
  char opsPoolShmSuffix[6];                            // 操作池共享内存后缀

  pthread_t thread;                                    // 进度线程的 pthread 句柄
  volatile int stop;                                   // 停止标志（易变变量）
  struct ncclProxyPeer** localPeers;                   // 本地对等节点指针数组
  struct ncclSharedNetComms* netComms[NCCL_MAX_NETDEVS]; // 网络通信指针数组
  struct ncclProxyArgs* active;                        // 活跃的代理参数链表头
  struct ncclProxyArgs* pool;                          // 代理参数池
  struct ncclProxyPool* pools;                         // 代理池指针
  int nextOps;                                         // 下一个操作索引
};

// Expected proxy response fifo
// 期望的代理响应 FIFO（先进先出队列）
struct ncclExpectedProxyResponse {
  void*                             opId;              // 操作 ID
  int                               respSize;        // 响应大小
  bool                              done;             // 完成标志
  void*                             respBuff;         // 响应缓冲区指针
  ncclResult_t                      res;              // 结果码
  struct ncclExpectedProxyResponse* next;            // 指向下一个期望响应的指针
};

// 结构体：代理异步操作
// 描述一个异步的代理操作
struct ncclProxyAsyncOp {
  int type;                                            // 操作类型
  struct ncclProxyConnection* connection;             // 指向代理连接的指针
  int reqSize, respSize;                               // 请求大小和响应大小
  char *reqBuff, *respBuff;                            // 请求缓冲区和响应缓冲区指针
  void* opId;                                          // 操作 ID
  ncclProxyAsyncOp* next;                             // 指向下一个异步操作的指针
};

// 结构体：代理本地对等节点
// 描述本地的一个对等节点
struct ncclProxyLocalPeer {
  struct ncclSocket sock;                             // 套接字
  int tpRank;                                          // 线程对（Thread Pair）rank
  int tpLocalRank;                                     // 线程对本地 rank
  ncclProxyAsyncOp* asyncOps;                         // 异步操作链表头
  int asyncOpCounter;                                  // 异步操作计数器
};

// Common response header for all proxyOps
// 所有代理操作的通用响应头
// We pack this into a struct to reduce the number of blocking send and recv calls
// 我们将其打包到一个结构体中以减少阻塞发送和接收调用的次数
struct ncclProxyRpcResponseHeader {
  void* opId;                                          // 操作 ID
  ncclResult_t res;                                   // 结果码
  int respSize;                                       // 响应大小
};

// UDS support
// UDS（Unix Domain Socket）支持
struct ncclIpcHdr {
  int type;                                            // 类型
  int rank;                                            // rank 编号
  int reqSize;                                         // 请求大小
  int respSize;                                        // 响应大小
  void *opId;                                          // 操作 ID
  uint64_t data[16];                                   // 数据数组（128 字节）
};

// 结构体：代理状态
// 描述 NCCL 代理的完整状态信息
struct ncclProxyState {
  int refCount;                                         // 引用计数
  int tpRank;                                           // 线程对（Thread Pair）rank
  int tpnRanks;                                         // 线程对中的 rank 数量
  int tpLocalnRanks;                                   // 线程对本地 rank 数量
  int cudaDev;                                          // CUDA 设备编号
  int p2pnChannels;                                     // P2P 通道数量
  int p2pChunkSize;                                     // P2P 块大小
  int nChannels;                                        // 通道总数
  int buffSizes[NCCL_NUM_PROTOCOLS];                    // 各协议的缓冲区大小数组
  bool allocP2pNetLLBuffers;                            // 是否分配 P2P 网络 LL 缓冲区
  bool dmaBufSupport;                                   // 是否支持 DMA 缓冲区
  ncclNet_t* ncclNet;                                  // 指向网络插件接口的指针
  ncclCollNet_t* ncclCollNet;                          // 指向集合网络插件接口的指针
  uint32_t* abortFlag;                                 // 中止标志指针
  bool directMode;                                     // 直接模式标志
  // Service threads
  // 服务线程
  pthread_t thread;                                    // 主服务线程句柄
  pthread_t threadUDS;                                 // UDS（Unix Domain Socket）服务线程句柄
  struct ncclSocket* listenSock;                       // 监听套接字
  //unix套接字
  // Unix 套接字
  struct ncclIpcSocket ipcSock;                        // IPC 套接字
  int stop;                                             // 停止标志
  CUcontext cudaCtx;                                   // CUDA 上下文
  ncclResult_t asyncResult;                            // 异步操作结果

  // Used by main thread
  // 主线程使用的字段
  union ncclSocketAddress* peerAddresses;              // 对等节点地址数组
  struct ncclSocket* peerSocks;                        // 对等节点套接字数组
  struct ncclProxyOps* proxyOps;                       // 代理操作指针
  void** sharedDevMems;                                // 共享设备内存指针数组
  struct ncclIpcSocket peerIpcSock; // cuMEM API support (UDS) // 对等节点 IPC 套接字（cuMEM API 支持，UDS）
  uint64_t *peerAddressesUDS; // cuMem API support (UDS) // 对等节点 UDS 地址数组（cuMEM API 支持，UDS）

  // Progress thread
  // 进度线程相关字段
  struct ncclProxyProgressState progressState;         // 进度状态结构体

  // Network plugin
  // 网络插件相关字段
  void* netContext;                                    // 网络上下文指针
  ncclNetAttr_t netAttr;                               // 网络属性
  void* collNetContext;                                // 集合网络上下文指针

  // Profiler plugin
  // 性能分析器插件相关字段
  void* profilerContext;                               // 性能分析器上下文指针

  // Queue of expected responses from the proxy
  // 来自代理的期望响应队列
  struct ncclExpectedProxyResponse* expectedResponses; // 期望响应链表头
};

// 枚举：代理连接状态
// 定义代理连接在其生命周期中的各种状态
enum proxyConnectState {
  connUninitialized     = 0,                           // 未初始化状态
  connInitialized       = 1,                           // 已初始化状态
  connSharedInitialized = 2,                           // 共享资源已初始化状态
  connSetupDone         = 3,                           // 设置完成状态
  connConnected         = 4,                           // 已连接状态
  numConnStates         = 5                            // 状态总数（用于边界检查）
};

// 结构体：代理连接
// 描述一个代理连接的所有信息
struct ncclProxyConnection {
  int send, transport, shared;                          // 发送标志、传输类型、共享标志
  int tpLocalRank, sameProcess;                         // 线程对本地 rank、同进程标志
  struct ncclSocket* sock;                             // 套接字指针
  struct ncclTransportComm* tcomm;                      // 传输层通信指针
  struct ncclProxyArgs *proxyAppend;                    // 代理追加参数指针
  struct ncclProxyArgs **proxyAppendPtr;                // 代理追加指针的指针
  void* transportResources;                             // 传输层资源指针
  ncclNetDeviceHandle_t* netDeviceHandle;              // 网络设备句柄指针
  void* mhandles[NCCL_NUM_PROTOCOLS];                   // 各协议的内存句柄数组
  proxyConnectState state;                              // 连接状态
  struct ncclCollNetSharedRes* collNet;                 // 集合网络共享资源指针
  int needsProxyProgress;                              // 需要代理进度推进的标志
};

// 函数指针类型：线程函数
// 指向处理代理参数的线程函数
typedef ncclResult_t (*threadFunc_t)(struct ncclProxyArgs*);

// 枚举：代理模式
// 定义代理操作的三种模式
enum proxyMode {
  proxyRing = 0,                                       // 环形模式
  proxyFrom = 1,                                       // 从源模式（发送）
  proxyTo = 2                                          // 到目标模式（接收）
};

// 函数声明：保存代理操作
// 参数 comm: 通信器指针
// 参数 proxyOp: 代理操作指针
// 参数 justInquire: 输出参数，是否仅查询而不执行
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
ncclResult_t ncclProxySaveOp(struct ncclComm* comm, struct ncclProxyOp* proxyOp, bool *justInquire);

// 函数声明：启动代理
// 启动代理服务线程
// 参数 comm: 通信器指针
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
ncclResult_t ncclProxyStart(struct ncclComm* comm);

// 函数声明：初始化代理
// 初始化代理连接和资源
// 参数 comm: 通信器指针
// 参数 sock: 套接字指针
// 参数 peerAddresses: 对等节点地址数组
// 参数 peerAddressesUDS: 对等节点 UDS 地址数组
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
ncclResult_t ncclProxyInit(struct ncclComm* comm, struct ncclSocket* sock, union ncclSocketAddress* peerAddresses, uint64_t *peerAddressesUDS);

// 函数声明：创建代理
// 创建代理结构并分配资源
// 参数 comm: 通信器指针
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
ncclResult_t ncclProxyCreate(struct ncclComm* comm);

// 函数声明：建立代理连接
// 建立与指定 rank 的代理连接
// 参数 comm: 通信器指针
// 参数 transport: 传输类型
// 参数 send: 发送标志
// 参数 proxyRank: 代理 rank
// 参数 proxyConn: 代理连接器指针
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
ncclResult_t ncclProxyConnect(struct ncclComm* comm, int transport, int send, int proxyRank, struct ncclProxyConnector* proxyConn);

// NB: ncclProxyMsgTypeStr[] in proxy.cc needs to match
// 注意：proxy.cc 中的 ncclProxyMsgTypeStr[] 数组需要与此枚举匹配
// 枚举：代理消息类型
// 定义代理通信中各种消息的类型
enum ncclProxyMsgType {
  ncclProxyMsgInit = 1,                                // 初始化消息
  ncclProxyMsgSharedInit = 2,                          // 共享资源初始化消息
  ncclProxyMsgSetup = 3,                               // 设置消息
  ncclProxyMsgConnect = 4,                             // 连接消息
  ncclProxyMsgStart = 5,                               // 启动消息
  ncclProxyMsgClose = 6,                               // 关闭消息
  ncclProxyMsgAbort = 7,                               // 中止消息
  ncclProxyMsgStop = 8,                                // 停止消息
  ncclProxyMsgGetFd = 9, // cuMem API support (UDS) // 获取文件描述符消息（cuMEM API 支持，UDS）
  ncclProxyMsgQueryFd = 10,                            // 查询文件描述符消息
  ncclProxyMsgRegister = 11,                           // 注册消息
  ncclProxyMsgDeregister = 12                          // 注销消息
};

// This function is called by a client of the proxy that needs to invoke any of the non-progress proxyOp types
// 此函数由需要调用任何非进度代理操作类型的代理客户端调用
// Call this function on the client, supplying a locally unique opId. Then, poll on the return value of
// 在客户端调用此函数，提供本地唯一的 opId。然后，轮询以下函数的返回值：
// ncclPollProxyResponse(), supplying the same opId to confirm the operation has completed
// ncclPollProxyResponse()，提供相同的 opId 以确认操作已完成
// 参数 comm: 通信器指针
// 参数 proxyConn: 代理连接器指针
// 参数 type: 操作类型
// 参数 reqBuff: 请求缓冲区指针
// 参数 reqSize: 请求大小
// 参数 respSize: 响应大小
// 参数 opId: 操作 ID
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
ncclResult_t ncclProxyCallAsync(struct ncclComm* comm, struct ncclProxyConnector* proxyConn, int type, void* reqBuff, int reqSize, int respSize, void* opId);

// This function will internally call ncclProxyCallAsync() and spin until ncclPollProxyResponse() confirms the result is received
// 此函数将在内部调用 ncclProxyCallAsync() 并自旋，直到 ncclPollProxyResponse() 确认接收到结果
// 参数 comm: 通信器指针
// 参数 proxyConn: 代理连接器指针
// 参数 type: 操作类型
// 参数 reqBuff: 请求缓冲区指针
// 参数 reqSize: 请求大小
// 参数 respBuff: 响应缓冲区指针
// 参数 respSize: 响应大小
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
ncclResult_t ncclProxyCallBlocking(struct ncclComm* comm, struct ncclProxyConnector* proxyConn, int type, void* reqBuff, int reqSize, void* respBuff, int respSize);

// 函数声明：轮询代理响应
// 轮询并获取代理操作的响应
// 参数 comm: 通信器指针
// 参数 proxyConn: 代理连接器指针
// 参数 respBuff: 响应缓冲区指针
// 参数 opId: 操作 ID
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
ncclResult_t ncclPollProxyResponse(struct ncclComm* comm, struct ncclProxyConnector* proxyConn, void* respBuff, void* opId);

// UDS support
// UDS（Unix Domain Socket）支持
// 函数声明：阻塞方式获取文件描述符
// 从代理获取文件描述符（用于 cuMEM API）
// 参数 comm: 通信器指针
// 参数 rank: rank 编号
// 参数 handle: 句柄指针
// 参数 convertedFd: 输出参数，转换后的文件描述符
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
ncclResult_t ncclProxyClientGetFdBlocking(struct ncclComm* comm, int rank, void *handle, int* convertedFd);

// 函数声明：阻塞方式查询文件描述符
// 查询远程文件描述符（用于 cuMEM API）
// 参数 comm: 通信器指针
// 参数 proxyConn: 代理连接器指针
// 参数 localFd: 本地文件描述符
// 参数 rmtFd: 输出参数，远程文件描述符
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
ncclResult_t ncclProxyClientQueryFdBlocking(struct ncclComm* comm, struct ncclProxyConnector* proxyConn, int localFd, int* rmtFd);

// 函数声明：停止代理
// 停止代理服务线程
// 参数 comm: 通信器指针
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
ncclResult_t ncclProxyStop(struct ncclComm* comm);

// 函数声明：解除共享内存链接
// 解除并删除代理使用的共享内存
// 参数 comm: 通信器指针
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
ncclResult_t ncclProxyShmUnlink(struct ncclComm* comm);

// 函数声明：销毁代理
// 销毁代理结构并释放所有资源
// 参数 comm: 通信器指针
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
ncclResult_t ncclProxyDestroy(struct ncclComm* comm);

// 头文件保护结束宏
// 与开头的 #ifndef NCCL_PROXY_H_ 配对，防止头文件重复包含
#endif
