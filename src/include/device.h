/*************************************************************************
 * Copyright (c) 2015-2022, NVIDIA CORPORATION. All rights reserved.
 * 版权所有 (c) 2015-2022, NVIDIA CORPORATION. 保留所有权利。
 *
 * See LICENSE.txt for license information
 * 请参阅 LICENSE.txt 获取许可证信息
 ************************************************************************/

// 头文件保护宏开始：防止头文件被重复包含
#ifndef NCCL_DEVICE_H_
#define NCCL_DEVICE_H_

// 包含 NCCL 公共 API 头文件，定义了用户可见的接口和数据类型
#include "nccl.h"
// 包含 NCCL 调优插件头文件，定义了性能调优相关的接口
#include "nccl_tuner.h"
// 包含位操作工具头文件，提供各种位操作辅助函数
#include "bitops.h"
// C++ 标准库算法头文件，提供 std::min, std::max 等算法函数
#include <algorithm>
// 标准整数类型头文件，定义了 int32_t, uint64_t 等固定宽度整数类型
#include <stdint.h>
// 系统数据类型头文件，定义了 size_t 等系统相关类型
#include <sys/types.h>

// 声明外部变量：NCCL 集合通信函数名称字符串数组
// 数组索引对应 ncclFunc_t 枚举值，如 "AllReduce", "Broadcast" 等
extern const char* ncclFuncStr[NCCL_NUM_FUNCTIONS];

// 声明外部变量：NCCL 通信算法名称字符串数组
// 数组索引对应 ncclAlgo_t 枚举值，如 "Ring", "Tree" 等
extern const char* ncclAlgoStr[NCCL_NUM_ALGORITHMS];

// 声明外部变量：NCCL 通信协议名称字符串数组
// 数组索引对应 ncclProto_t 枚举值，如 "LL", "LL128", "Simple" 等
extern const char* ncclProtoStr[NCCL_NUM_PROTOCOLS];

// 定义：每个通道可以排队的最大操作数量
// 这个值限制了单个通道上可以并发执行的操作数目，防止内存溢出
#define NCCL_MAX_OPS 2048

// 定义：流水线步骤数量
// NCCL 使用流水线技术重叠通信和计算，8 步流水线可以隐藏延迟
#define NCCL_STEPS 8

// 条件编译：在设备代码（CUDA 内核）中编译时
#ifdef __CUDA_ARCH__
  // 在设备代码中，NCCL_CUDA_ARCH 被定义为实际的 CUDA 架构版本
  // __CUDA_ARCH__ 是 CUDA 内置宏，如 700(Volta), 800(Ampere), 900(Hopper)
  #define NCCL_CUDA_ARCH __CUDA_ARCH__
#else
  // 在主机代码中，NCCL_CUDA_ARCH 默认为 0
  // 这意味着主机端代码默认不知道设备架构，需要运行时查询
  #define NCCL_CUDA_ARCH 0
#endif

// 条件编译：检测是否定义了特定的 CUDA 架构
#ifdef __CUDA_ARCH_SPECIFIC__
  // 如果已定义特定架构，直接使用该值
  #define NCCL_CUDA_ARCH_SPECIFIC __CUDA_ARCH_SPECIFIC__
#elif defined(__CUDA_ARCH_HAS_FEATURE__)
  // 如果支持特性检测，根据 SM 特性确定架构版本
  // 这允许编译器根据目标 GPU 的特性选择最优实现
  #if __CUDA_ARCH_HAS_FEATURE__(SM90_ALL)
    // Hopper 架构 (SM90) - 计算能力 9.0
    #define NCCL_CUDA_ARCH_SPECIFIC 900
  #elif __CUDA_ARCH_HAS_FEATURE__(SM100_ALL)
    // Blackwell 架构 (SM100) - 计算能力 10.0
    #define NCCL_CUDA_ARCH_SPECIFIC 1000
  #elif __CUDA_ARCH_HAS_FEATURE__(SM101_ALL)
    // Blackwell 架构变体 (SM101) - 计算能力 10.1
    #define NCCL_CUDA_ARCH_SPECIFIC 1010
  #elif __CUDA_ARCH_HAS_FEATURE__(SM120_ALL)
    // 未来架构 (SM120) - 计算能力 12.0
    #define NCCL_CUDA_ARCH_SPECIFIC 1200
  #else
    // 未识别的架构，设置为 0
    #define NCCL_CUDA_ARCH_SPECIFIC 0
  #endif
#else
  // 没有架构信息，设置为 0
  #define NCCL_CUDA_ARCH_SPECIFIC 0
#endif

// 条件编译：检测是否定义了架构家族
#ifdef __CUDA_ARCH_FAMILY_SPECIFIC__
  // 如果已定义，直接使用该值
  #define NCCL_CUDA_ARCH_FAMILY_SPECIFIC __CUDA_ARCH_FAMILY_SPECIFIC__
#else
  // 否则默认为 0
  #define NCCL_CUDA_ARCH_FAMILY_SPECIFIC 0
#endif

// 包含网络设备相关头文件
// 定义了设备端网络通信接口，如 ncclNetDeviceHandle_t 等
#include "net_device.h"

// 设备端归约操作类型枚举
// 这些是 GPU 内核中直接执行的归约操作，比主机端操作更高效
enum ncclDevRedOp_t {
  ncclDevSum,       // 求和操作：对所有元素进行加法归约
  ncclDevProd,      // 求积操作：对所有元素进行乘法归约
  ncclDevMinMax,    // 最小最大操作：同时计算最小值和最大值
  ncclDevPreMulSum, // 预乘求和：先乘以标量再求和
  ncclDevSumPostDiv,// 求和后除：先求和再除以标量
  ncclNumDevRedOps  // 归约操作类型总数（用于数组大小等）
};

// 设备端归约操作的完整描述结构体
struct ncclDevRedOpFull {
  ncclDevRedOp_t op;     // 归约操作类型（枚举值）
  ncclRedOp_t proxyOp;   // 代理操作类型（用于不支持该操作的 fallback）
  bool scalarArgIsPtr;   // 标量参数是否为指针（true=指针，false=立即数）
  uint64_t scalarArg;    // 标量参数值（用于 PreMulSum 的乘数或 SumPostDiv 的除数）
};

// Long Jump (LL) 协议的 FIFO 行联合体
// LL 协议使用双缓冲技术实现低延迟的 GPU 间通信
union ncclLLFifoLine {
  /* 标志必须放在数据之后，否则从网络的不完整接收可能收到标志但没有数据。
   * 注意：这里假设我们要么接收连续的数据块（套接字），
   * 要么数据以 8 字节原子性写入（IB/RDMA）。
   * Flags have to be *after* data, because otherwise, an incomplete receive
   * from the network may receive the flag but not the data.
   * Note this is assuming that either we receive contiguous chunks of data
   * (sockets) or data is written with an atomicity of 8 bytes (IB/RDMA). */
  struct {
    uint32_t data1;      // 第一个数据字（32 位）
    uint32_t flag1;      // 第一个标志位（指示 data1 是否有效）
    uint32_t data2;      // 第二个数据字（32 位）
    uint32_t flag2;      // 第二个标志位（指示 data2 是否有效）
  };
  // 以 64 位整数数组的方式访问同一内存
  // 这种方式便于使用 64 位加载/存储指令，提高访问效率
  uint64_t v[2];         // 2 个 64 位整数，对应上面的 data1+flag1 和 data2+flag2
  // 以 int4 向量类型访问（CUDA 内置向量类型，128 位）
  // 这种方式允许使用 SIMD 指令进行向量化加载/存储
  int4 i4;              // 一个 int4 向量，128 位，与 v[2] 共享内存
};

// 定义：CUDA Warp 的大小（线程数）
// NVIDIA GPU 的基本执行单元，所有 CUDA 架构都是 32
#define WARP_SIZE 32

// 定义：NCCL 支持的最大通道数量
// 每个通道可以独立执行通信操作，实现并行通信以提高带宽利用率
#define MAXCHANNELS 64

// 定义：单个节点内最大的本地 rank 数量
// 这限制了同一台物理机器上可以运行的 GPU 进程数量
#define NCCL_MAX_LOCAL_RANKS 72

// 定义：每个 CUDA 线程块（block）的最大线程数
// NCCL 根据不同的协议和数据大小动态调整线程数，最大不超过 640
#define NCCL_MAX_NTHREADS 640

// 定义：每个线程块的最小线程数
// 设置为 4 个 warp（4 * 32 = 128），确保有足够的并行度
#define NCCL_MIN_NTHREADS (4*WARP_SIZE)

// 定义：SIMPLE 协议的最大线程数
// SIMPLE 协议用于小消息，使用较少的线程以减少启动开销
#define NCCL_SIMPLE_MAX_NTHREADS 512

// 定义：SIMPLE 协议在线程数达到阈值时增加额外的线程组
// 当线程数 >= 96 (3*32) 时，添加额外的组以优化性能
#define NCCL_SIMPLE_EXTRA_GROUP_IF_NTHREADS_GE (3*WARP_SIZE)

// 定义：LL 协议的最大线程数
// LL (Long Jump) 协议用于中等大小的消息，限制为 512 个线程
#define NCCL_LL_MAX_NTHREADS 512

// 定义：LL 协议中每个线程处理的 FIFO 行数
// 每个线程负责管理 8 行 FIFO，实现流水线处理
#define NCCL_LL_LINES_PER_THREAD 8

// 测试模式下的 LL 清理掩码定义
#ifdef TEST_LL_CLEANUP
  // 测试模式：使用较小的掩码值以便观察清理行为
  // 可以设置为 0x100 来禁用清理功能
  #define NCCL_LL_CLEAN_MASK 0x078 // Set to 0x100 to disable cleanup
  // LL 协议标志的最大值（用于取模运算）
  #define NCCL_LL_FLAG_MAX   0x100
  // 对标志值进行取模，确保在有效范围内
  #define NCCL_LL_FLAG(a) ((uint32_t)((a) % NCCL_LL_FLAG_MAX))
#else
  // 生产模式：使用较大的掩码值，优化性能
  // 0x7ffffff8 清除了低 3 位，用于对齐和清理
  #define NCCL_LL_CLEAN_MASK 0x7ffffff8
  // 直接返回标志值，不进行取模（性能优化）
  #define NCCL_LL_FLAG(a) ((uint32_t)(a))
#endif

// 静态断言：确保清理掩码可以被步数整除
// 这保证了每个步骤都能正确对齐到清理边界
// Make sure the clean mask will last for at least NCCL_NSTEPS
static_assert(NCCL_LL_CLEAN_MASK % NCCL_STEPS == 0, "Invalid NCCL_LL_CLEAN_MASK value");

// 定义：LL128 协议的行大小（字节）
// LL128 是 LL 协议的变体，使用 128 字节缓存行对齐
#define NCCL_LL128_LINESIZE 128

// 定义：LL128 协议每行的 uint64_t 元素数量
// 128 字节 / 8 字节 = 16 个元素
#define NCCL_LL128_LINEELEMS (NCCL_LL128_LINESIZE/sizeof(uint64_t))

// 定义：LL128 协议每行的数据元素数量
// 减去 1 是因为最后一个元素用于标志/控制
#define NCCL_LL128_DATAELEMS (NCCL_LL128_LINEELEMS-1)

// 定义：LL128 协议的最大线程数
// 与通用最大线程数相同
#define NCCL_LL128_MAX_NTHREADS 640

// 定义：LL128 协议每个线程处理的元素数量
// 每个线程负责处理 120 个元素，平衡负载和寄存器压力
#define NCCL_LL128_ELEMS_PER_THREAD 120

// 定义：LL128 协议每个线程的共享内存元素数量
// 每个线程在共享内存中分配 8 个元素的空间
#define NCCL_LL128_SHMEM_ELEMS_PER_THREAD 8

// 定义：LL128 协议的共享内存总大小
// 8 元素/线程 * 640 线程 * 8 字节/元素 = 40960 字节
#define NCCL_LL128_SHMEM_SIZE (NCCL_LL128_SHMEM_ELEMS_PER_THREAD*NCCL_LL128_MAX_NTHREADS)

// 定义：P2P 写操作标志
// 表示支持 GPU 直接写入对等节点的内存
#define NCCL_P2P_WRITE 0x01

// 定义：P2P 读操作标志
// 表示支持 GPU 直接读取对等节点的内存
#define NCCL_P2P_READ  0x02

// 定义：直接网卡访问标志
// 表示支持 GPU 直接通过网卡进行 RDMA 操作，无需 CPU 参与
#define NCCL_DIRECT_NIC   0x04

// 定义：NVLS 最小轮询间隔
// NVLS (NVLink SHARP) 的最小轮询时间单位
#define NCCL_NVLS_MIN_POLL 0x80

// CUDA 命名屏障的最大数量
// 命名屏障用于 CUDA 线程块间的同步
// Number of named barriers supported by CUDA
#define NCCL_MAX_GROUPS 16

// 定义：常规缓冲区类型标志
// 表示使用普通的 GPU 内存缓冲区
#define NCCL_REGULAR_BUFFER 0x00

// 定义：IPC 注册缓冲区类型标志
// 表示使用跨进程通信（IPC）的共享内存缓冲区
#define NCCL_IPC_REG_BUFFER 0x01

// 定义：NVLS 注册缓冲区类型标志
// 表示使用 NVLink SHARP 的多播缓冲区
#define NCCL_NVLS_REG_BUFFER 0x02

// 定义：网络注册缓冲区类型标志
// 表示使用网络（IB/RoCE）的注册内存缓冲区
#define NCCL_NET_REG_BUFFER 0x04

// 连接信息结构体
// 存储单个连接方向（发送或接收）的所有必要信息
struct ncclConnInfo {
  // Regular comm mechanism
  // 常规通信机制：缓冲区数组，支持多种协议（LL, LL128, Simple）
  char *buffs[NCCL_NUM_PROTOCOLS]; // Local for recv, remote for send
                                    // 对于接收：本地缓冲区地址
                                    // 对于发送：远程缓冲区地址（通过网络交换）

  // 内存句柄数组，用于跨进程或跨节点内存注册
  // 每种协议对应一个句柄
  void* mhandles[NCCL_NUM_PROTOCOLS];

  // 队列尾部指针（生产者-消费者队列的消费者索引）
  // 接收端：本地 tail，指向下一个可读位置
  // 发送端：远程 tail，告知对端本地已写到哪里
  uint64_t *tail;     // Local for recv, remote for send

  // 队列头部指针（生产者-消费者队列的生产者索引）
  // 发送端：本地 head，指向下一个可写位置
  // 接收端：远程 head，告知对端本地已读到哪里
  uint64_t *head;     // Local for send, remote for recv

  // 通信标志位
  // 包含多种标志：P2P读写、直接网卡访问、NVLS 等（通过按位或组合）
  int flags;          // Direct communication / other flags

  // 缓冲区共享标志
  // 为 1 表示多个连接共享同一块缓冲区内存
  int shared;         // Buffers are shared

  // SIMPLE 协议的步进大小
  // SIMPLE 协议将数据分块，每块大小为 stepSize
  int stepSize;       // Step size for the SIMPLE buffer

  // 指针交换地址（用于直接通信）
  // 在 P2P 直接通信中，用于交换内存指针
  void **ptrExchange; // Pointer exchange for direct communication

  // 归约操作标量参数交换地址（用于直接拉取模式）
  // 在需要预先计算标量参数时使用
  uint64_t* redOpArgExchange; // PreOp scaler exchange for direct pull case

  // 连接 FIFO（用于 GPU 与代理进程通信）
  // 代理模式：CPU 协助 GPU 完成部分通信操作
  struct ncclConnFifo* connFifo; // Used for GPU - Proxy communication

  // 当前步骤计数器
  // 跟踪当前在协议的哪个步骤（0 到 NCCL_STEPS-1）
  uint64_t step;      // Keep where we are

  // LL 协议上次清理的时间戳/计数
  // 用于清理已使用的 FIFO 条目，防止回绕问题
  uint64_t llLastCleaning;

  // 网络设备句柄
  // 底层网络插件（如 IB）的设备句柄，用于 RDMA 操作
  ncclNetDeviceHandle_t netDeviceHandle;
};

// 代理连接器结构体
// 当需要 CPU 协助时（如内存注册），使用代理连接器
struct ncclProxyConnector {
  bool initialized;   // 是否已初始化标志
  int rank;           // 对等节点的全局 rank
  int tpRank;         // 顶层父通信器的 rank（用于 comm split 场景）
  int tpLocalRank;    // 顶层父通信器中的本地 rank
  int sameProcess;    // 是否在同一进程内（0=不同进程，1=同一进程）
  struct ncclProxyConnection* connection;  // 代理连接对象指针
  // 代理进度推进函数指针
  // 从传输层复制过来，用于推进代理操作的执行
  ncclResult_t (*proxyProgress)(struct ncclProxyState* proxyState, struct ncclProxyArgs*); // Copied from transport if necessary
};

// 连接器结构体
// 表示一个方向的通信连接（发送或接收）
struct ncclConnector {
  int connected;      // 连接状态标志（0=未连接，1=已连接）
  int hasSeen;        // 是否已使用过该连接的标志
                      // 用于避免重复建立连接和资源分配
  int p2pOnly;        // 是否仅用于 P2P 操作的标志（0=集合通信，1=仅 P2P）
  struct ncclProxyConnector proxyConn;     // 代理连接器信息
  struct ncclTransportComm* transportComm; // 传输层通信接口指针
  void* transportResources;                // 传输层资源指针（由传输层分配）
  struct ncclConnInfo conn;               // 连接信息（缓冲区、句柄等）
};

// Ring 环形拓扑结构体
// 描述当前节点在环形拓扑中的位置
struct ncclRing {
  // Shortcuts for userRanks[1] and userRanks[n-1]
  // 快捷访问：前驱和后继节点的 rank（避免查找 userRanks 数组）
  int prev;          // 环中前驱节点的 rank（从当前节点看，数据来源）
  int next;          // 环中后继节点的 rank（从当前节点看，数据去向）

  // Maps an internal nccl index to user-specified rank order. This is necessary
  // since we need to know how the user expects data to be ordered across
  // devices. Ordered from current device.
  // 将内部 NCCL 索引映射到用户指定的 rank 顺序。这是必要的，
  // 因为我们需要知道用户期望数据在设备间如何排序。
  // 从当前设备的视角排序。
  int* userRanks;    // rank 映射数组，userRanks[i] = 环上第 i 个位置的用户 rank

  int index;         // 当前 rank 在环中的索引位置（0 到 nRanks-1）
};


// The root of each tree only has one node down (+1 intra-node).
// 每个树的根节点只有一个下行节点（+1 个节点内连接）。
// 定义：树根节点的最大分支度（下行节点数）
#define NCCL_MAX_TREE_ARITY_TOP 2

// Nodes inside the binary tree can have to two nodes down (+1 intra-node).
// 二叉树内的节点可以有两个下行节点（+1 个节点内连接）。
// 定义：树内部节点的最大分支度（包括跨节点和节点内）
#define NCCL_MAX_TREE_ARITY 3

// Tree 树形拓扑结构体
// 描述当前节点在树形拓扑中的位置
struct ncclTree {
  int depth;                              // 当前节点在树中的深度（根节点为 0）
  int up;                                 // 上行节点（父节点）的 rank
  int down[NCCL_MAX_TREE_ARITY];          // 下行节点（子节点）的 rank 数组
};

// 定义：直接网络拓扑的最大分支度
// CollNet Direct 模式支持更多并行连接
#define NCCL_MAX_DIRECT_ARITY 7

// Direct 直接网络拓扑结构体
// 用于 CollNet Direct 算法，节点直接连接到网络交换机
struct ncclDirect {
  int depth;         // 当前节点在树中的深度
  int out;           // 输出连接的 rank（用于 scatter 操作）
  int nHeads;        // 并行 N<->1<->net 操作的数量；up/down 数组的大小
                     // Number of parallel N<->1<->net operations we'll do in parallel; size of up/down
  int headRank;      // 我是哪个 head rank 的索引（0 到 nHeads-1）
                     // 如果不是 head rank（没有本地网卡）则为 -1
                     // Index in 0..nHeads-1 I am the head rank of. -1 if I'm not a head rank (no local NIC)
  int shift;         // scatter/gather 操作的发送/接收偏移量，基本上是 localRank%nHeads
                     // Shuffling of send/recv for scatter/gather operations, basically localRank%nHeads
  // The heads[...] are guaranteed to be in rotated order start with self:
  //   headRank, (headRank+1)%nHeads, (headRank+2)%nHeads, ...
  // heads[...] 保证以轮转顺序排列，从自身开始：
  //   headRank, (headRank+1)%nHeads, (headRank+2)%nHeads, ...
  int heads[NCCL_MAX_DIRECT_ARITY+1];  // head ranks 数组（+1 用于存储 null 终止符）
  int up[NCCL_MAX_DIRECT_ARITY];       // 上行节点 rank 数组（用于 gather）
  int down[NCCL_MAX_DIRECT_ARITY];     // 下行节点 rank 数组（用于 scatter）
};

// 定义：NVLS 拓扑的最大分支度
// NVLink SHARP 支持更多并行连接
#define NCCL_MAX_NVLS_ARITY 32

// 定义：NVLS 树形拓扑的最大分支度
// NVLS 也支持树形拓扑
#define NCCL_MAX_NVLS_TREE_ARITY 3

// NVLS (NVLink SHARP) 拓扑结构体
// 用于 NVLink fabric 的集合通信
struct ncclNvls {
  int out;           // 输出连接的 rank
  int nHeads;        // 并行 N<->1<->net 操作的数量；up/down 数组的大小
                     // Number of parallel N<->1<->net operations we'll do in parallel; size of up/down
  int headRank;      // 我是哪个 head rank 的索引（0 到 nHeads-1）
                     // 如果不是 head rank（没有本地网卡）则为 -1
                     // Index in 0..nHeads-1 I am the head rank of. -1 if I'm not a head rank (no local NIC)
  int up[NCCL_MAX_NVLS_ARITY];         // 上行节点 rank 数组
  int down;         // 下行节点 rank（单个）
  int treeUp;       // NVLS 树的上行节点 rank
  int treeDown[NCCL_MAX_NVLS_TREE_ARITY];  // NVLS 树的下行节点 rank 数组
};

// 根据 CUDA 架构选择最大分支度
#if __CUDA_ARCH__ >= 900
  // Hopper (SM90) 及以上架构使用 NVLS 的大分支度
  #define NCCL_MAX_ARITY NCCL_MAX_NVLS_ARITY
#else
  // 早期架构使用 Direct 模式的分支度
  #define NCCL_MAX_ARITY NCCL_MAX_DIRECT_ARITY
#endif

// 定义：每个 peer 的最大连接数
// 索引 0 用于集合通信，索引 1 用于 P2P 通信
#define NCCL_MAX_CONNS 2

// 通道对等节点结构体
// 存储与某个特定 peer 的所有连接信息
struct ncclChannelPeer {
  struct ncclConnector send[NCCL_MAX_CONNS];  // 发送连接数组
                                                  // send[0]: 集合通信发送连接
                                                  // send[1]: P2P 发送连接
  struct ncclConnector recv[NCCL_MAX_CONNS];  // 接收连接数组
                                                  // recv[0]: 集合通信接收连接
                                                  // recv[1]: P2P 接收连接
  int refCount;        // 引用计数，跟踪有多少地方在使用此 peer
};

// 前向声明：内核通信器结构体
struct ncclKernelComm;

// 设备端 P2P 工作描述结构体（16 字节对齐）
// 描述一个 P2P 操作的所有参数
struct alignas(16) ncclDevWorkP2p {
  void *sendAddr, *recvAddr;      // 发送和接收缓冲区的设备地址
  size_t sendBytes, recvBytes;    // 发送和接收的字节数
  int sendRank, recvRank;         // 发送目标和接收源的 rank

  // 从 part 索引、nP2pChannels 和 channelBase，设备代码可以计算通道负责传输的哪一部分。
  // From the part index, nP2pChannels, and channelBase the device code can
  // calculate which part of the transfer a channel is responsible for.
  uint8_t nP2pChannels;           // P2P 通道总数（始终等于 comm->p2pnChannels）
  uint8_t channelBase;            // 拥有第一部分的通道号

  // 零通道表示该方向没有工作。
  // Zero channels indicates no work in that direction.
  uint8_t nSendChannels, nRecvChannels;  // 该操作使用的发送和接收通道数

  // Chunk size stored in 8 bits via u32fp8Encode/Decode.
  // 块大小通过 u32fp8Encode/Decode 以 8 位存储。
  uint8_t sendChunkSize_u32fp8, recvChunkSize_u32fp8;  // 编码后的块大小

  // 位域：协议和注册标志
  uint8_t sendProtoLL:1,    // 发送是否使用 LL 协议
         recvProtoLL:1;     // 接收是否使用 LL 协议
  uint8_t sendNetReg:1,     // 发送是否使用网络注册内存
         recvNetReg:1;      // 接收是否使用网络注册内存
  uint8_t sendIpcReg:1,     // 发送是否使用 IPC 注册内存
         recvIpcReg:1;      // 接收是否使用 IPC 注册内存
  uint8_t profilerEnabled:1;// 是否启用性能分析器
};

// 计算给定部分索引对应的数据传输子集
// Compute the subset of the data transfer corresponding to the given part index.
inline __host__ __device__ void ncclP2pPartBounds(int nParts, int part, size_t bytes, size_t* partBeg, size_t* partEnd) {
  // 计算每部分的大小：向上取整到 4KB 边界
  // divUp(bytes, nParts) 将数据均匀分配
  // alignUp(..., 4<<10) 对齐到 4KB（4096 字节）以提高内存访问效率
  size_t partBytes = alignUp(divUp(bytes, nParts), 4<<10);

  #if __CUDA_ARCH__
    // 设备代码路径：使用 CUDA 内置的 min 函数
    *partBeg = min((part+0)*partBytes, bytes);
    *partEnd = min((part+1)*partBytes, bytes);
  #else
    // 主机代码路径：使用 C++ 标准库的 std::min
    *partBeg = std::min<size_t>((part+0)*partBytes, bytes);
    *partEnd = std::min<size_t>((part+1)*partBytes, bytes);
  #endif
}

// 在 channel.h 中实现的函数（前向声明）
// implemented in channel.h
inline __host__ uint8_t ncclP2pChannelBaseForRound(struct ncclComm* comm, int p2pRound);

// ncclP2pChannelToPart 和 ncclP2pChannelForPart 互为逆函数。
// 设备代码使用 ncclP2pChannelToPart 来确定"此"通道负责的部分。
// ncclP2pChannelToPart and ncclP2pChannelForPart are inverses. The device code
// uses ncclP2pChannelToPart to determine which part "this" channel is responsible for.

// 主机端函数：根据部分索引计算对应的通道号
// 使用位与运算实现环形映射（nP2pChannels 必须是 2 的幂）
inline __host__ int ncclP2pChannelForPart(int nP2pChannels, int base, int part) {
  return (base + part) & (nP2pChannels-1);
}

// 设备端函数：根据通道号计算对应的部分索引
// 与上面的函数互为逆运算
inline __device__ int ncclP2pChannelToPart(int nP2pChannels, int base, int channel) {
  return (channel - base) & (nP2pChannels-1);
}

// 设备端集合通信工作描述结构体（16 字节对齐）
// 描述一个集合操作（AllReduce, Broadcast 等）的所有参数
struct alignas(16) ncclDevWorkColl {
  // Running on channels [channelLo..channelHi], hi is inclusive.
  //   nChannels == (channelHi - channelLo) + 1
  // 在通道 [channelLo..channelHi] 上运行，hi 是包含的。
  //   nChannels == (channelHi - channelLo) + 1
  uint32_t channelLo:8,  // 使用的最低通道号（8 位，最大 255）
           channelHi:8;  // 使用的最高通道号（8 位，最大 255）
  uint32_t nWarps:8;     // 每个 CTAA 的 warp 数（8 位，最大 255）

  // 位域：各种标志和参数
  uint32_t redOpArgIsPtr:1,  // 归约操作参数是否为指针
           regUsed:1,         // 是否使用了注册缓冲区
           netRegUsed:1,      // 是否使用了网络注册缓冲区
           oneNode:1,         // 是否为单节点模式
           direct:2,          // 直接模式标志（2 位，4 种状态）
           isOneRPN:1;        // 是否每个节点只有一个 rank
  uint32_t profilerEnabled:1;// 是否启用性能分析器
  uint32_t root;             // 根节点 rank（用于 Broadcast, Reduce 等操作）

  // 缓冲区指针
  void* recvbuff;           // 接收缓冲区设备地址
  void* sendbuff;           // 发送缓冲区设备地址

  // 缓冲区偏移量（用于对称内存注册）
  uintptr_t sendbuffOffset; // 发送缓冲区偏移
  uintptr_t recvbuffOffset; // 接收缓冲区偏移

  // 远程地址数组（用于直接内存访问）
  uintptr_t* sendbuffRmtAddrs;  // 发送缓冲区的远程地址数组
  uintptr_t* recvbuffRmtAddrs;  // 接收缓冲区的远程地址数组

  // 联合体：不同调度策略的数据分布
  union {
    // Continuous-byte-distribution scheduling. The lo and hi channels are of
    // different size than the channels in the middle.
    // 连续字节分布调度。低端和高端通道的大小与中间通道不同。
    struct {
      size_t countLo,      // 最低通道处理的字节数
             countMid,     // 中间通道各自处理的字节数
             countHi;      // 最高通道处理的字节数
      // Chunk counts where units are ncclProtoGrainSize(protocol) bytes
      // 块计数，单位是 ncclProtoGrainSize(protocol) 字节
      uint64_t chunkGrainsLo:21,  // 最低通道的块粒度数（21 位，最大 2M-1）
               chunkGrainsMid:21, // 中间通道的块粒度数（21 位）
               chunkGrainsHi:21;  // 最高通道的块粒度数（21 位）
    } cbd;  // CBD: Continuous Byte Distribution（连续字节分布）
    // Collnet scheduling. All channels divide work evenly.
    // Collnet 调度。所有通道均匀分配工作。
    struct {
      size_t count;        // 总大小，不按通道分割
      uint32_t chunkCount; // 块数量
    } collnet;
  };
  uint64_t redOpArg;       // 归约操作参数（用于 PreMulSum 或 SumPostDiv）
};


// 计算协议的粒度大小（常量表达式，编译时计算）
__host__ __device__ constexpr int ncclProtoGrainSize(int proto) {
  return proto == NCCL_PROTO_LL ? 16 :                                      // LL 协议：16 字节粒度
         proto == NCCL_PROTO_LL128 ? WARP_SIZE*NCCL_LL128_SHMEM_ELEMS_PER_THREAD/NCCL_LL128_LINEELEMS*NCCL_LL128_DATAELEMS*sizeof(uint64_t) : // LL128 协议：计算共享内存粒度
         proto == NCCL_PROTO_SIMPLE ? 512 :                                  // SIMPLE 协议：512 字节粒度
         -1;                                                                  // 无效协议
}

// 计算集合通信在连续字节分布模式下，某个通道应该处理的数据部分
// 模板函数：支持不同的整数类型（size_t, int64_t 等）
template<typename Int>
__host__ __device__ inline void ncclCollCbdPart(
    struct ncclDevWorkColl* work,      // 工作描述结构体
    uint32_t channelId,                // 当前通道 ID
    int proto,                         // 通信协议
    int eltSize,                       // 元素大小（字节）
    Int* count,                        // 输出：总元素数
    Int* partOffset,                   // 输出：当前通道的偏移量（元素数）
    Int* partCount,                    // 输出：当前通道的元素数
    Int* chunkCount                    // 输出：当前通道的块数
  ) {
  // 计算每个粒度包含的元素数
  int eltPerGrain = ncclProtoGrainSize(proto)/eltSize;

  // 计算中间通道的数量（可能为 0 或负数）
  int nMidChannels = work->channelHi - work->channelLo - 1;

  // We can assum that nMidChannels<0 implies countMid==0, which let's us assume
  // that countMid*nMidChannels == 0.
  // 我们可以假设 nMidChannels<0 意味着 countMid==0，这让我们假设
  // countMid*nMidChannels == 0。

  // 如果需要计算总元素数
  if (count != nullptr) {
    // 总数 = 最低通道 + 中间通道总和 + 最高通道
    *count = work->cbd.countLo + work->cbd.countMid*nMidChannels + work->cbd.countHi;
  }

  // 根据通道 ID 确定其角色和分配
  if (channelId == work->channelLo) {
    // 最低通道：处理数据的开头部分
    *partOffset = 0;
    *partCount = work->cbd.countLo;
    *chunkCount = work->cbd.chunkGrainsLo*eltPerGrain;
  } else if (channelId == work->channelHi) {
    // 最高通道：处理数据的结尾部分
    *partOffset = work->cbd.countLo + nMidChannels*work->cbd.countMid;
    *partCount = work->cbd.countHi;
    *chunkCount = work->cbd.chunkGrainsHi*eltPerGrain;
  } else {
    // 中间通道：均匀分配中间部分
    int mid = channelId - work->channelLo - 1;
    *partOffset = work->cbd.countLo + mid*work->cbd.countMid;
    *partCount = work->cbd.countMid;
    *chunkCount = work->cbd.chunkGrainsMid*eltPerGrain;
  }
}

// 设备端集合通信注册工作结构体（16 字节对齐）
// 用于需要内存注册的集合通信操作（如直接网络访问）
struct alignas(16) ncclDevWorkCollReg {
  struct ncclDevWorkColl coll;  // 基础集合通信工作描述
  void* dnInputs[NCCL_MAX_DIRECT_ARITY+1];   // 下行输入指针数组（从网络节点接收）
  void* dnOutputs[NCCL_MAX_DIRECT_ARITY+1];  // 下行输出指针数组（发送到网络节点）
  void* upOutputs[NCCL_MAX_DIRECT_ARITY+1];  // 上行输出指针数组（发送到父节点）
};

// 设备端工作类型枚举（uint8_t）
enum ncclDevWorkType: uint8_t {
  ncclDevWorkTypeP2p,      // P2P 工作类型（Send/Recv）
  ncclDevWorkTypeColl,     // 集合通信工作类型（AllReduce, Broadcast 等）
  ncclDevWorkTypeCollReg   // 需要内存注册的集合通信工作类型
};

// 计算设备端工作结构体的大小（常量表达式，编译时计算）
constexpr size_t ncclDevWorkSize(enum ncclDevWorkType type) {
  return type == ncclDevWorkTypeP2p ? sizeof(ncclDevWorkP2p) :           // P2P 工作大小
         type == ncclDevWorkTypeColl ? sizeof(ncclDevWorkColl) :         // 集合通信工作大小
         sizeof(ncclDevWorkCollReg);                                    // 注册集合通信工作大小
}

// 定义：设备端工作批次的字节数限制
#define NCCL_MAX_DEV_WORK_BATCH_BYTES 1024

// 定义：每个批次最多包含的集合通信工作数量
#define NCCL_MAX_DEV_WORK_BATCH_COLLS (NCCL_MAX_DEV_WORK_BATCH_BYTES/sizeof(ncclDevWorkColl))

// 定义：每个批次最多包含的 P2P 工作数量
#define NCCL_MAX_DEV_WORK_P2P_PER_BATCH 8

// 设备端工作批次结构体（16 字节对齐）
// 用于将多个工作项打包成一个批次，提高内核启动效率
struct alignas(16) ncclDevWorkBatch {
  union {
    struct {
      // nextExtends: 下一个批次是否应该合并到此批次
      // nextJump=0: 此通道批次列表的结束
      // nextJump>0: batches[thisIndex+nextJump] 是此列表中的下一个批次
      // nextExtends: should next one be merged into this one.
      // nextJump=0: end of this channel's batch list
      // nextJump>0: batches[thisIndex+nextJump] is next batch in this list
      uint32_t nextJump:14,    // 跳转到下一个批次的偏移量（14 位，最大 16383）
               nextExtends:1;  // 下一个是否扩展此批次（1 位）
      uint32_t workType:2,     // 工作类型（2 位：P2P/Coll/CollReg）
               funcId:15;      // 函数 ID（15 位，最多 32767 个函数）
    };
    // Unioning bitfields with underlying type hints compiler to emit the best
    // SASS LD/ST accesses.
    // 将位域与底层类型联合，提示编译器发出最优的 SASS LD/ST 访问指令。
    uint32_t flags;           // 32 位标志位（与上面的位域共享内存）
  };
  // 此批次工作结构开始的 FIFO 中的滚动偏移量
  uint32_t offsetBase;        // Rolling offset in fifo where this batch's work structs begin

  // 此通道批次子集的相对偏移量集合：
  // 对于 offsetBitset 中的每个位索引 i，在 fifo 偏移量处找到工作：offsetBase + i*sizeof(WorkStructType)
  // Set of relative offsets from offsetBase for this channel's subset of the batch:
  // For each bit index i in offsetMask, find work at fifo offset: offsetBase + i*sizeof(WorkStructType)
  uint64_t offsetBitset;      // 64 位位图，每个位表示一个工作项是否存在
};

// 设备端通道对等节点结构体
// ncclChannelPeer 的精简版本，只保留 ncclConnInfo 而不是完整的 ncclConnector
struct ncclDevChannelPeer {
  // Stripped version of ncclChannelPeer where we only keep the ncclConnInfo
  // instead of the full ncclConnector.
  struct ncclConnInfo send[NCCL_MAX_CONNS];  // 发送连接信息数组（2 个：集合通信和 P2P）
  struct ncclConnInfo recv[NCCL_MAX_CONNS];  // 接收连接信息数组（2 个：集合通信和 P2P）
};

// 设备端通道结构体（16 字节对齐）
// 存储单个通道的所有信息，供设备内核使用
struct alignas(16) ncclDevChannel {
  struct ncclDevChannelPeer** peers;  // 指向对等节点指针数组的指针
                                       // peers[rank] 指向 rank 对应的 ncclDevChannelPeer
  struct ncclRing ring;               // Ring 环形拓扑信息
  struct ncclTree tree;               // Tree 树形拓扑信息
  struct ncclTree collnetChain;       // CollNet Chain 拓扑信息
  struct ncclDirect collnetDirect;    // CollNet Direct 拓扑信息
  struct ncclNvls nvls;               // NVLS 拓扑信息
  uint32_t* workFifoDone;             // 完成计数器位置，设备写入最后处理的工作的 index+1
                                      // Location of done counter, device writes index+1 of last work processed
  uint64_t workCounter;               // 工作计数器
};

// 定义：每个通道的最大性能分析事件数量
#define MAX_PROFILER_EVENTS_PER_CHANNEL 64

// 设备端性能分析器结构体
struct ncclDevProfiler {
  struct {
    uint64_t counter;    // 事件计数器
    uint64_t timestamp;  // 时间戳
  } data[MAX_PROFILER_EVENTS_PER_CHANNEL];  // 事件数据数组
};

// 内核通信器结构体
// 存储设备内核需要的所有通信上下文信息
struct ncclKernelComm {
  int rank;            // 当前 rank 在通信域中的编号（0 到 nRanks-1）
  int nRanks;          // 通信域中的总 rank 数
  int node;            // 当前节点编号（0 到 nNodes-1）
  int nNodes;          // 总节点数
  int buffSizes[NCCL_NUM_PROTOCOLS];  // 每种协议的缓冲区大小
  int p2pChunkSize;    // P2P 通信的块大小
  int isAllNvlink;     // 是否全部使用 NVLink 连接

  int* collNetDenseToUserRank;  // CollNet 稠密 rank 到用户 rank 的映射数组

  // Flag to ask NCCL kernels to abort
  // 要求 NCCL 内核中止的标志
  volatile uint32_t* abortFlag;  // 指向中止标志的指针（volatile 确保跨线程可见）

  // Channels, device side
  // 通道，设备端
  struct ncclDevChannel* channels/*[MAXCHANNELS]*/;  // 指向通道数组的指针
  int* rankToLocalRank;  // rank 到本地 rank 的映射数组

  // Profiler counters
  // 性能分析器计数器
  struct ncclDevProfiler* workStarted/*[MAXCHANNELS]*/;    // 每个通道的工作开始计数器
  struct ncclDevProfiler* workCompleted/*[MAXCHANNELS]*/;  // 每个通道的工作完成计数器
};

// 内核通信器和通道的联合结构体（16 字节对齐）
// 将通信器和所有通道打包在一个连续内存区域，方便传递给内核
struct alignas(16) ncclKernelCommAndChannels {
  struct ncclKernelComm comm;                  // 内核通信器
  struct ncclDevChannel channels[MAXCHANNELS]; // 所有通道的数组（64 个）
};

// 设备端工作存储类型枚举（uint8_t）
enum ncclDevWorkStorageType: uint8_t {
  ncclDevWorkStorageTypeArgs=0,      // 使用内核参数传递工作（小规模）
  ncclDevWorkStorageTypeFifo=1,      // 使用 FIFO 队列传递工作（大规模）
  ncclDevWorkStorageTypePersistent=2  // 使用持久化存储（CUDA Graph）
};

// 设备端内核参数结构体（16 字节对齐）
// 传递给 NCCL 内核的主要参数
struct alignas(16) ncclDevKernelArgs {
  struct ncclKernelComm* comm;      // 指向内核通信器的指针
  uint64_t channelMask;             // 通道掩码，指示哪些通道参与此操作
  enum ncclDevWorkStorageType workStorageType;  // 工作存储类型
  uint32_t workMask;                // 工作掩码（用于某种过滤）
  void* workBuf;                    // 指向工作缓冲区的指针
  // A channel's first batch is at `blockIdx.x`. Use `nextJump` to follow rest of list.
  // 通道的第一个批次位于 `blockIdx.x`。使用 `nextJump` 跟随列表的其余部分。
  // struct ncclDevWorkBatch batches[];  // 批次数组（注释掉，实际内存紧跟在此结构体后）
};

// 计算内核参数的最大大小（常量表达式，编译时计算）
// 默认使用 4KB，可根据架构调整
__host__ __device__ constexpr int ncclMaxKernelArgsSize(/*int cudaDriver, */int cudaArch=NCCL_CUDA_ARCH) {
  //return (cudaArch < 700 || cudaDriver < 12010) ? 4<<10 : (32<<10)-4;
  return 4<<10;  // 返回 4096 字节（4KB）
}

// 内核参数存储模板结构体（16 字节对齐）
// 使用 union 实现 reinterpret_cast 的效果，避免类型别名警告
template<size_t capacity>
struct alignas(16) ncclDevKernelArgsStorage {
  union {
    struct ncclDevKernelArgs args;  // 实际的内核参数结构体
    ulong2 storage[capacity/sizeof(ulong2)];  // 以 ulong2 数组方式访问（CUDA 内置向量类型）
  };
};

// 类型别名：4KB 容量的内核参数存储
typedef ncclDevKernelArgsStorage<(4<<10)> ncclDevKernelArgs4K;
// 32KB 容量的内核参数存储（已注释，可能用于大参数场景）
//typedef ncclDevKernelArgsStorage<(32<<10)-4> ncclDevKernelArgs31K;

// 编译时最小值函数（模板递归实现）
// 单参数版本：返回参数本身
template<typename T>
__host__ __device__ constexpr T min_constexpr(T a) { return a; }

// 多参数版本：递归比较，返回最小值
template<typename T, typename ...Ts>
__host__ __device__ constexpr T min_constexpr(T a, T b, Ts ...c) {
  return min_constexpr<T>((a < b ? a : b), c...);
}

// 编译时最大值函数（模板递归实现）
// 单参数版本：返回参数本身
template<typename T>
__host__ __device__ constexpr T max_constexpr(T a) { return a; }

// 多参数版本：递归比较，返回最大值
template<typename T, typename ...Ts>
__host__ __device__ constexpr T max_constexpr(T a, T b, Ts ...c) {
  return max_constexpr<T>((a > b ? a : b), c...);
}

// 计算给定参数字节大小可以支持的最大通道数（常量表达式）
// 减去 ncclDevKernelArgs 的大小，除以 ncclDevWorkBatch 的大小
constexpr int ncclDevMaxChannelsForArgsBytes(size_t argsBytes) {
  return min_constexpr<size_t>(MAXCHANNELS, (argsBytes - sizeof(struct ncclDevKernelArgs))/sizeof(struct ncclDevWorkBatch));
}

// 计算循环展开因子（常量表达式）
// Calculate the unroll factor given:
// * bytePerPack: 每条指令访问的字节数
// * insns: 最大允许的展开值
// * bytes: 每次迭代期望的飞行字节数 (= unroll*bytePerPack)
__host__ __device__ constexpr int ncclCalcUnroll(int bytePerPack, int insns, int bytes) {
  return min_constexpr(insns, (bytes + bytePerPack-1)/bytePerPack);
}

// 注意：所有展开值逻辑应该依赖于给定的 cudaArch 参数
// 而不是 __CUDA_ARCH__，因为这些都是主机端可执行的，其中
// 架构值严格是仅运行时的。通过默认使用 NCCL_CUDA_ARCH，设备
// 端代码可以省略传递 arch 参数以简洁。
// Note that all unroll value logic should depend on a given cudaArch argument
// and not __CUDA_ARCH__ since these need to be host-side executable where the
// arch value is strictly runtime only. By defaulting to NCCL_CUDA_ARCH, device
// side code can elide passing the arch for brevity.

// 计算集合通信的循环展开因子
__host__ __device__ constexpr int ncclCollUnroll(int cudaArch = NCCL_CUDA_ARCH) {
  // Our collective unroll should move to the same bytes&insns model as NVLS.
  // 我们的集合展开应该移到与 NVLS 相同的 bytes&insns 模型。
  return cudaArch >= 800 ? (cudaArch / 100 == 12 ? 6 : 8) : 4;  // Ada(8.9)为6，其他800+为8，700为4
}

// NVLS 的展开字节数（常量表达式）
__host__ __device__ constexpr int ncclNvlsUnrollBytes(int cudaArch = NCCL_CUDA_ARCH) { return 4*16; }  // 64 字节

// NVLS 的展开指令数（常量表达式）
__host__ __device__ constexpr int ncclNvlsUnrollInsns(int cudaArch = NCCL_CUDA_ARCH) { return 16; }  // 16 条指令

// 计算 NVLS 的循环展开因子
__host__ __device__ constexpr int ncclNvlsUnroll(int bytePerPack, int cudaArch = NCCL_CUDA_ARCH) {
  return ncclCalcUnroll(bytePerPack, ncclNvlsUnrollInsns(cudaArch), ncclNvlsUnrollBytes(cudaArch));
}

// 计算每个 warp 的动态共享内存大小
// The amount of dynamic shmem per warp
__host__ __device__ constexpr int ncclShmemScratchWarpSize(int cudaArch = NCCL_CUDA_ARCH) {
  return (max_constexpr<int>(
      /*LL    */0,                                                         // LL 协议不使用共享内存
      /*LL128 */(NCCL_LL128_SHMEM_ELEMS_PER_THREAD*WARP_SIZE)*sizeof(uint64_t),  // LL128: 8*32*8=2048 字节
      /*SIMPLE*/(ncclCollUnroll(cudaArch)*WARP_SIZE + 1)*16,                 // SIMPLE: (unroll*32+1)*16 字节
      // NVLS needs an extra 16B to read unaligned data.
      // NVLS 需要额外的 16 字节来读取未对齐的数据。
      /*NVLS  */WARP_SIZE*(cudaArch >= 900 ? ncclNvlsUnrollBytes(cudaArch) : 0) + 16
    ) + 15) & -16;  // pad to 16 bytes (向上对齐到 16 字节边界)
}

// 计算每个线程块的动态共享内存大小
// The amount of dynamic shmem per block
__host__ __device__ constexpr int ncclShmemDynamicSize(int cudaArch = NCCL_CUDA_ARCH) {
  return cudaArch < 700 ? 0 : ncclShmemScratchWarpSize(cudaArch)*(NCCL_MAX_NTHREADS/WARP_SIZE);
  // Volta (700) 之前不支持动态共享内存，返回 0
  // 之后：warp 大小 * warp 数量（640/32=20）
}

// 主机端内核函数指针表
// Host-side table of kernel function pointers.
extern int const ncclDevKernelCount;               // 内核函数数量
extern void* const ncclDevKernelList[/*ncclDevKernelCount*/];  // 内核函数指针列表

// 特定函数索引的最特化内核函数表
// Table of most specialized kernel function to run given func index.
extern int const ncclDevFuncRowToId[];             // 函数行到 ID 的映射表
extern void* const ncclDevKernelForFunc[/*funcIndex*/];      // 每个函数的最特化内核
extern bool const ncclDevKernelForFuncIsSpecialized[/*funcIndex*/];  // 内核是否已特化

// 在流上启动单 rank 归约操作
// Launch a one-rank reduction on stream.
ncclResult_t ncclLaunchOneRank(void* dst, void const* src, size_t nElts, struct ncclDevRedOpFull redOp, ncclDataType_t type, cudaStream_t stream);

// 检查 NVLS 是否支持特定的归约操作和数据类型
// `ncclNvlsSupported()` needs to be in sync with "func_valid" in "src/device/generate.py"
inline bool ncclNvlsSupported(int devRedOp, int type) {
  switch (type) {
  // 整数类型支持 Sum 和 MinMax
  case ncclInt32:
  case ncclUint32:
  case ncclInt64:
  case ncclUint64:
  // 半精度浮点支持 Sum 和 MinMax
  case ncclFloat16:
  case ncclBfloat16:
    return devRedOp == ncclDevSum || devRedOp == ncclDevMinMax;
  // 单双精度浮点只支持 Sum
  case ncclFloat:
  case ncclDouble:
    return devRedOp == ncclDevSum;
  default:
    return false;
  }
}

// 计算设备端函数的 ID
// 需要与 "src/device/generate.py" 中的 "all_functions()" 保持同步
// `ncclDevFuncIndex()` needs to be in sync with "all_functions()" in "src/device/generate.py"
inline int ncclDevFuncId(int coll, int devRedOp, int type, int algo, int proto) {
  constexpr int NumTypes = ncclNumTypes;  // 数据类型数量
  int row;
  do {
    // P2P 函数（Send/Recv）的行号为 0
    row = 0; // ncclDevFuncIndex_P2p
    if (coll == ncclFuncSendRecv) break;
    row += 1;

    // AllGather: 支持 4 种算法（Ring, CollNet_Direct, NVLS, PAT）
    // 每种算法有 NCCL_NUM_PROTOCOLS 种协议
    int nAlgos = 4;
    if (coll == ncclFuncAllGather) {
      int algo1 = algo == NCCL_ALGO_RING ? 0 :
                  algo == NCCL_ALGO_COLLNET_DIRECT ? 1 :
                  algo == NCCL_ALGO_NVLS ? 2 :
                /*algo == NCCL_ALGO_PAT*/ 3;
      row += algo1*NCCL_NUM_PROTOCOLS + proto;
      break;
    }
    row += nAlgos*NCCL_NUM_PROTOCOLS;

    // Broadcast: 只支持 1 种算法，但有 NCCL_NUM_PROTOCOLS 种协议
    nAlgos = 1;
    if (coll == ncclFuncBroadcast) {
      row += proto;
      break;
    }
    row += nAlgos*NCCL_NUM_PROTOCOLS;

    // AllReduce: 支持 6 种算法（TREE, RING, COLLNET_DIRECT, COLLNET_CHAIN, NVLS, NVLS_TREE）
    // 每种组合有：ncclNumDevRedOps * NumTypes * nAlgos * NCCL_NUM_PROTOCOLS 种变体
    nAlgos = 6; // TREE RING COLLNET_DIRECT COLLNET_CHAIN NVLS NVLS_TREE
    if (coll == ncclFuncAllReduce) {
      row += ((devRedOp*NumTypes + type)*nAlgos + algo)*NCCL_NUM_PROTOCOLS + proto;
      break;
    }
    row += ncclNumDevRedOps*NumTypes*nAlgos*NCCL_NUM_PROTOCOLS;

    // Reduce: 只支持 1 种算法
    nAlgos = 1;
    if (coll == ncclFuncReduce) {
      row += (devRedOp*NumTypes + type)*NCCL_NUM_PROTOCOLS + proto;
      break;
    }
    row += ncclNumDevRedOps*NumTypes*nAlgos*NCCL_NUM_PROTOCOLS;

    // ReduceScatter: 支持 4 种算法（Ring, CollNet_Direct, NVLS, PAT）
    nAlgos = 4;
    if (coll == ncclFuncReduceScatter) {
      int algo1 = algo == NCCL_ALGO_RING ? 0 :
                  algo == NCCL_ALGO_COLLNET_DIRECT ? 1 :
                  algo == NCCL_ALGO_NVLS ? 2 :
                /*algo == NCCL_ALGO_PAT*/ 3;
      row += ((devRedOp*NumTypes + type)*nAlgos + algo1)*NCCL_NUM_PROTOCOLS + proto;
      break;
    }
    row += ncclNumDevRedOps*NumTypes*nAlgos*NCCL_NUM_PROTOCOLS;
  } while (false);

  // 将行号映射到实际的函数 ID
  return ncclDevFuncRowToId[row];
}

// 获取 P2P 函数的 ID（快捷方式，行号 0）
inline int ncclDevFuncId_P2p() { return ncclDevFuncRowToId[0]; }

// 头文件保护结束宏
#endif
