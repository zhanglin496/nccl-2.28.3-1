/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 * 版权所有 (c) 2016-2022，NVIDIA 公司。保留所有权利。
 *
 * See LICENSE.txt for license information
 * 有关许可信息，请参见 LICENSE.txt 文件
 ************************************************************************/

// 包含通信域头文件，定义 ncclComm 结构体和通信相关接口
#include "comm.h"
// 包含核心功能头文件，提供 NCCL 核心数据结构和宏定义
#include "core.h"
// 包含 Socket 相关的头文件，定义 Socket 操作接口和数据结构
#include "socket.h"
// 包含网络传输头文件，定义网络传输层的接口
#include "net.h"
// 包含参数配置头文件，提供可配置的环境变量和参数定义
#include "param.h"
// 包含 Socket 网络性能分析头文件，定义性能分析相关的接口
#include "profiler/net_socket.h"

// 包含 POSIX 线程库，用于多线程编程
#include <pthread.h>
// 包含标准库，提供内存分配、进程控制等函数
#include <stdlib.h>
// 包含 poll 函数，用于 I/O 多路复用
#include <poll.h>
// 包含系统限制常量，如 PATH_MAX 等
#include <limits.h>
// 包含文件控制函数，如 open、fcntl 等
#include <fcntl.h>
// 包含 C++ 标准库的互斥锁，用于线程同步
#include <mutex>
// 包含 C++ 标准库的条件变量，用于线程间通信
#include <condition_variable>

/* Init functions */
// 初始化相关函数
// 定义静态变量：网络接口数量
// -1 表示尚未初始化，需要通过系统调用发现网络接口
static int ncclNetIfs = -1;
// 定义结构体：Socket 网络设备信息
// 用于存储每个网络接口的地址、名称和 PCI 路径
struct ncclNetSocketDev {
  // addr: Socket 地址（支持 IPv4/IPv6）
  // ncclSocketAddress 是一个联合体，可以存储不同类型的 Socket 地址
  union ncclSocketAddress addr;
  // devName: 网络接口名称（如 "eth0", "ens3" 等）
  // MAX_IF_NAME_SIZE 是接口名称的最大长度
  char devName[MAX_IF_NAME_SIZE];
  // pciPath: PCI 设备路径
  // 指向 /sys/class/net/<devName>/device 的实际路径，用于设备发现和拓扑感知
  char* pciPath;
};
// 定义静态数组：Socket 网络设备列表
// MAX_IFS 是最大网络接口数量，存储所有可用的网络接口信息
static struct ncclNetSocketDev ncclNetSocketDevs[MAX_IFS];

// 定义静态互斥锁：Socket 网络互斥锁
// 用于保护多线程环境下的共享资源访问，如 ncclNetIfs 和 ncclNetSocketDevs
static std::mutex ncclNetSocketMutex;

// 函数功能：获取网络设备的 PCI 路径
// 参数说明：
//   - devName: 网络接口名称（输入）
//   - pciPath: PCI 路径指针的指针（输出）
// 返回值：ncclSuccess 表示成功
// 说明：通过 /sys/class/net/<devName>/device 符号链接获取实际的 PCI 设备路径
static ncclResult_t ncclNetSocketGetPciPath(char* devName, char** pciPath) {
  // 定义缓冲区：设备路径
  // PATH_MAX 是系统定义的最大路径长度常量
  char devicePath[PATH_MAX];
  // 构造 sysfs 中的设备路径字符串
  // /sys/class/net/<devName>/device 是指向 PCI 设备的符号链接
  snprintf(devicePath, PATH_MAX, "/sys/class/net/%s/device", devName);
  // May return NULL if the file doesn't exist.
  // 如果文件不存在，realpath 可能返回 NULL
  // realpath: 解析符号链接，返回规范化的绝对路径
  // 第一个参数：要解析的路径
  // 第二个参数：NULL 表示由函数分配缓冲区
  *pciPath = realpath(devicePath, NULL);
  // 返回成功状态码
  return ncclSuccess;
}

// 定义静态变量：性能分析回调函数指针
// 用于在 Socket 网络操作过程中记录性能数据
static ncclProfilerCallback_t ncclProfilerFunction;

// With ncclNet_v11_t the NCCL core initializes the network plugin per-communicator
// 使用 ncclNet_v11_t 接口时，NCCL 核心为每个通信器（communicator）初始化网络插件
// rather than once for all communicators. However, the internal plugin implementation
// 而不是为所有通信器只初始化一次。但是，内部插件实现
// still assumes the plugin is initialized only once across all communicators. The ref
// 仍然假设插件在所有通信器中只初始化一次。引用
// counter makes sure the plugin internally initializes only once. When per communicator
// 计数器确保插件内部只初始化一次。当支持每个通信器
// context support is added to the plugin the ref counter can be removed.
// 的上下文支持添加到插件时，引用计数器可以被移除。
// 定义静态变量：网络插件引用计数器
// 用于跟踪有多少个通信器正在使用此插件，确保只初始化一次
static int netRefCount;

// 函数功能：初始化 Socket 网络插件
// 参数说明：
//   - ctx: 输出参数，返回插件上下文（本插件不需要）
//   - commId: 通信域 ID
//   - config: 通信配置
//   - logFunction: 日志记录回调函数
//   - profFunction: 性能分析回调函数
// 返回值：ncclSuccess 表示成功
ncclResult_t ncclNetSocketInit(void** ctx, uint64_t commId, ncclNetCommConfig_t* config, ncclDebugLogger_t logFunction, ncclProfilerCallback_t profFunction) {
  // 引用计数自增并检查：如果不是第一次调用，直接返回成功
  // ++netRefCount: 先自增再判断，第一次调用后 netRefCount > 0
  if (netRefCount++) return ncclSuccess;
  // 保存性能分析回调函数指针到全局变量
  ncclProfilerFunction = profFunction;
  // 双重检查锁定模式：检查网络接口是否已初始化
  if (ncclNetIfs == -1) {
    // 加锁保护共享资源
    // std::lock_guard: RAII 风格的锁，构造时加锁，析构时解锁
    std::lock_guard<std::mutex> lock(ncclNetSocketMutex);
    // 再次检查（可能已被其他线程初始化）
    if (ncclNetIfs == -1) {
      // 定义缓冲区：存储网络接口名称数组
      // 每个接口名称最大 MAX_IF_NAME_SIZE 字节
      char names[MAX_IF_NAME_SIZE*MAX_IFS];
      // 定义数组：存储网络接口地址
      union ncclSocketAddress addrs[MAX_IFS];
      // 调用核心函数查找所有可用的网络接口
      // names: 输出参数，存储接口名称
      // addrs: 输出参数，存储接口地址
      // MAX_IF_NAME_SIZE: 每个接口名称的最大长度
      // MAX_IFS: 最大接口数量
      // &ncclNetIfs: 输出参数，实际找到的接口数量
      NCCLCHECK(ncclFindInterfaces(names, addrs, MAX_IF_NAME_SIZE, MAX_IFS, &ncclNetIfs));
      // 检查是否找到可用的网络接口
      if (ncclNetIfs <= 0) {
        // 没有找到任何接口，输出警告日志并返回错误
        WARN("NET/Socket : no interface found");
        return ncclInternalError;
      } else {
        // 定义宏：日志行最大长度
        // 2047 字节，保留一个字节给字符串结束符
        #define MAX_LINE_LEN (2047)
        // 定义缓冲区：日志字符串
        char line[MAX_LINE_LEN+1];
        // 定义缓冲区：地址字符串
        char addrline[SOCKET_NAME_MAXLEN+1];
        // 初始化字符串为空
        line[0] = '\0';
        // 确保地址字符串末尾有结束符
        addrline[SOCKET_NAME_MAXLEN] = '\0';
        // 遍历所有找到的网络接口
        for (int i=0; i<ncclNetIfs; i++) {
          // 复制接口名称到设备结构体
          // names+i*MAX_IF_NAME_SIZE: 定位到第 i 个接口名称的起始位置
          strcpy(ncclNetSocketDevs[i].devName, names+i*MAX_IF_NAME_SIZE);
          // 复制接口地址到设备结构体
          memcpy(&ncclNetSocketDevs[i].addr, addrs+i, sizeof(union ncclSocketAddress));
          // 获取 PCI 路径并保存到设备结构体
          NCCLCHECK(ncclNetSocketGetPciPath(ncclNetSocketDevs[i].devName, &ncclNetSocketDevs[i].pciPath));
          // 将接口信息追加到日志字符串
          // snprintf: 格式化输出字符串，防止缓冲区溢出
          snprintf(line+strlen(line), MAX_LINE_LEN-strlen(line), " [%d]%s:%s", i, names+i*MAX_IF_NAME_SIZE,
              ncclSocketToString(&addrs[i], addrline));
        }
        // 确保日志字符串末尾有结束符
        line[MAX_LINE_LEN] = '\0';
        // 输出信息日志，显示所有使用的网络接口
        INFO(NCCL_INIT|NCCL_NET,"NET/Socket : Using%s", line);
      }
    }
  }
  // 返回成功状态码
  return ncclSuccess;
}

// 函数功能：获取可用的网络设备数量
// 参数说明：
//   - ndev: 输出参数，返回网络设备数量
// 返回值：ncclSuccess 表示成功
ncclResult_t ncclNetSocketDevices(int* ndev) {
  // 将发现到的网络接口数量赋值给输出参数
  *ndev = ncclNetIfs;
  // 返回成功状态码
  return ncclSuccess;
}

// 函数功能：获取网络接口的链路速度
// 参数说明：
//   - devName: 网络接口名称
//   - speed: 输出参数，返回链路速度（单位：Mbps）
// 返回值：ncclSuccess 表示成功
// 说明：从 /sys/class/net/<devName>/speed 文件读取链路速度
static ncclResult_t ncclNetSocketGetSpeed(char* devName, int* speed) {
  // 初始化返回值为成功
  ncclResult_t ret = ncclSuccess;
  // 初始化速度为 0
  *speed = 0;
  // 定义缓冲区：速度文件路径
  char speedPath[PATH_MAX];
  // 构造 sysfs 中的速度文件路径
  // /sys/class/net/<devName>/speed 包含链路速度（单位：Mbps）
  snprintf(speedPath, sizeof(speedPath), "/sys/class/net/%s/speed", devName);
  // 初始化文件描述符为 -1（无效值）
  int fd = -1;
  // 尝试打开速度文件
  // O_RDONLY: 只读模式
  // SYSCHECKSYNC: 同步系统调用检查宏，出错时跳转到 fd 赋值
  SYSCHECKSYNC(open(speedPath, O_RDONLY), "open", fd);
  // 检查文件是否成功打开
  if (fd != -1) {
    // 定义缓冲区：存储读取的速度字符串
    // 8 个空格 + 隐含的结束符
    char speedStr[] = "        ";
    // 声明变量：实际读取的字节数
    int n;
    // Allow this to silently fail
    // 允许静默失败（不打印错误日志）
    // read: 从文件读取数据
    // fd: 文件描述符
    // speedStr: 接收缓冲区
    // sizeof(speedStr)-1: 最大读取字节数（保留一个字节给结束符）
    n = read(fd, speedStr, sizeof(speedStr)-1);
    // 检查是否成功读取数据
    if (n > 0) {
      // 将字符串转换为长整型
      // strtol: 字符串转 long 整数
      // speedStr: 要转换的字符串
      // NULL: 不返回转换结束位置
      // 0: 自动检测基数（0x 前缀为十六进制，0 前缀为八进制，其他为十进制）
      *speed = strtol(speedStr, NULL, 0);
    }
  }
  // 检查速度是否有效（<= 0 表示无效或未获取到）
  if (*speed <= 0) {
    // 输出信息日志，说明无法获取速度，使用默认值
    INFO(NCCL_NET, "Could not get speed from %s. Defaulting to 10 Gbps.", speedPath);
    // 设置默认速度为 10000 Mbps（10 Gbps）
    *speed = 10000;
  }
  // 如果文件已打开，关闭文件描述符
  if (fd != -1) SYSCHECK(close(fd), "close");
  // 返回结果状态码
  return ret;
}

// 函数功能：获取网络设备的属性信息
// 参数说明：
//   - dev: 设备索引
//   - props: 输出参数，返回设备属性结构体
// 返回值：ncclSuccess 表示成功
ncclResult_t ncclNetSocketGetProperties(int dev, ncclNetProperties_t* props) {
  // 设置设备名称：指向设备结构体中的接口名称
  props->name = ncclNetSocketDevs[dev].devName;
  // 设置 PCI 路径：指向设备结构体中的 PCI 路径字符串
  props->pciPath = ncclNetSocketDevs[dev].pciPath;
  // 设置全局唯一标识符：使用设备索引作为 GUID
  props->guid = dev;
  // 设置指针支持类型：只支持主机端指针（HOST）
  // NCCL_PTR_HOST: 表示网络设备只能访问主机内存，不支持 GPU 直接访问
  props->ptrSupport = NCCL_PTR_HOST;
  // 设置注册是否全局：0 表示内存注册不是全局的
  // 不同连接需要单独注册内存
  props->regIsGlobal = 0;
  // 设置是否强制刷新：0 表示不需要强制刷新
  // Socket 传输不需要显式刷新缓存
  props->forceFlush = 0;
  // 获取网络接口的链路速度并保存到属性结构体
  NCCLCHECK(ncclNetSocketGetSpeed(props->name, &props->speed));
  // 设置延迟：0 表示未设置（延迟难以准确测量）
  props->latency = 0; // Not set
  // 设置端口号：0 表示不使用特定端口
  props->port = 0;
  // 设置最大通信连接数：65536 个并发连接
  props->maxComms = 65536;
  // 设置最大接收操作数：每个连接只能有 1 个活跃的接收操作
  props->maxRecvs = 1;
  // 设置网络设备类型：HOST 类型（基于主机的网络设备）
  // 表示网络设备位于主机端，而非 GPU 或专用硬件
  props->netDeviceType    = NCCL_NET_DEVICE_HOST;
  // 设置网络设备版本：无效版本（Socket 没有设备版本概念）
  props->netDeviceVersion = NCCL_NET_DEVICE_INVALID_VERSION;
  // 设置点对点传输的最大字节数：使用 NCCL 定义的最大网络传输大小
  props->maxP2pBytes = NCCL_MAX_NET_SIZE_BYTES;
  // 设置集合操作的最大字节数：使用 CollNet 定义的最大大小
  props->maxCollBytes = MAX_COLLNET_SIZE;
  // 设置多重请求的最大大小：1 表示不支持多重请求
  // Socket 传输一次只能处理一个请求
  props->maxMultiRequestSize = 1;
  // 返回成功状态码
  return ncclSuccess;
}

/* Communication functions */
// 通信相关函数

// 定义宏：每个通信连接使用的最大 Socket 数量
// 64 个 Socket 可以并行传输数据，提高带宽利用率
#define MAX_SOCKETS 64
// 定义宏：每个通信连接使用的最大辅助线程数量
// 16 个线程可以并发处理多个 Socket 的 I/O 操作
#define MAX_THREADS 16
// 定义宏：最大请求数量
// 使用 NCCL 网络层定义的最大请求数
#define MAX_REQUESTS NCCL_NET_MAX_REQUESTS

// 定义参数：Socket 内联数据大小
// 小于此大小的数据会与控制信息一起发送，避免额外的内存拷贝
// 默认值：128 字节（1 << 7 = 128）
NCCL_PARAM(SocketInlineSize, "SOCKET_INLINE", /*128 B=*/1 << 7);
// 定义参数：Socket 最小任务大小
// 将大数据分割成至少此大小的块，分配给不同的 Socket 和线程
// 默认值：64 KiB（1 << 16 = 65536）
NCCL_PARAM(SocketMinTaskSize, "SOCKET_MIN_TASKSIZE", /*64 kiB=*/1 << 16);
// 定义参数：每个线程的 Socket 数量
// -2 表示自动检测（根据网络设备和云环境）
NCCL_PARAM(SocketNsocksPerThread, "NSOCKS_PERTHREAD", -2);
// 定义参数：辅助线程数量
// -2 表示自动检测（根据网络设备和云环境）
NCCL_PARAM(SocketNthreads, "SOCKET_NTHREADS", -2);

// 定义枚举：Socket 通信状态机状态
// 用于跟踪连接建立和数据传输的各个阶段
enum ncclNetSocketCommState {
  ncclNetSocketCommStateStart = 0,      // 初始状态
  ncclNetSocketCommStateConnect = 1,    // 连接状态（主动连接）
  ncclNetSocketCommStateAccept = 3,     // 接受状态（被动接受连接）
  ncclNetSocketCommStateSend = 4,       // 发送状态
  ncclNetSocketCommStateRecv = 5,       // 接收状态
};

// 定义结构体：Socket 通信阶段信息
// 用于非阻塞的连接建立过程，支持跨多次调用恢复状态
struct ncclNetSocketCommStage {
  // state: 当前所处的状态
  enum ncclNetSocketCommState state;
  // iteration: 当前迭代次数（用于多个 Socket 的连接循环）
  uint8_t iteration;
  // sock: 当前正在操作的 Socket 指针
  struct ncclSocket* sock;
  // comm: 通信上下文指针
  struct ncclNetSocketComm* comm;
};

// 定义结构体：Socket 网络连接句柄
// 用于在对等体之间交换连接信息
struct ncclNetSocketHandle {
  // connectAddr: 连接地址（IP + 端口）
  union ncclSocketAddress connectAddr;
  // magic: 魔数（随机数）
  // 用于调试和验证连接的正确性
  uint64_t magic; // random number to help debugging
  // nSocks: 此连接使用的 Socket 数量
  int nSocks;
  // nThreads: 此连接使用的辅助线程数量
  int nThreads;
  // stage: 连接阶段信息（用于非阻塞连接）
  struct ncclNetSocketCommStage stage;
};

// 定义结构体：Socket 网络传输任务
// 代表一个在辅助线程中执行的传输任务
struct ncclNetSocketTask {
  // op: 操作类型（NCCL_SOCKET_SEND 或 NCCL_SOCKET_RECV）
  int op;
  // data: 数据缓冲区指针
  void* data;
  // size: 数据大小（字节数）
  int size;
  // sock: 要使用的 Socket 指针
  struct ncclSocket* sock;
  // offset: 当前已传输的字节偏移量
  int offset;
  // used: 使用标志（0=未使用，1=正在使用）
  int used;
  // result: 操作结果（ncclSuccess 或错误码）
  ncclResult_t result;
};

// 定义结构体：性能分析信息
// 用于存储性能分析句柄
struct ncclProfilerInfo {
  // eHandle: 事件句柄（用于标记事件开始/结束）
  void* eHandle;
  // pHandle: 父句柄（用于关联到上级事件）
  void* pHandle;
};

// 定义结构体：Socket 网络请求
// 代表一个用户发起的网络传输请求，可能包含多个子任务
struct ncclNetSocketRequest {
  // op: 操作类型（NCCL_SOCKET_SEND 或 NCCL_SOCKET_RECV）
  int op;
  // data: 用户数据缓冲区指针
  void* data;
  // size: 用户数据大小
  int size;
  // inlineData: 内联数据缓冲区指针
  // 用于存储小数据的控制信息和数据内容
  void* inlineData;
  // ctrlSock: 控制 Socket 指针
  // 用于传输控制信息和内联数据
  struct ncclSocket* ctrlSock;
  // offset: 当前已传输的字节偏移量
  int offset;
  // used: 使用状态（0=未使用，1=交换大小中，2=传输数据中）
  int used;
  // comm: 所属的通信上下文指针
  struct ncclNetSocketComm* comm;
  // tasks: 子任务数组指针
  // 将大请求分割成多个小任务，分配给不同的 Socket
  struct ncclNetSocketTask* tasks[MAX_SOCKETS];
  // nSubs: 子任务数量
  int nSubs;
  // pInfo: 性能分析信息
  struct ncclProfilerInfo pInfo;
};

// 定义结构体：Socket 任务队列
// 每个辅助线程有自己的任务队列
struct ncclNetSocketTaskQueue {
  // next: 下一个要分配的任务槽位索引
  int next;
  // len: 队列长度（任务槽位总数）
  int len;
  // tasks: 任务数组指针
  struct ncclNetSocketTask* tasks;
};

// 定义结构体：Socket 线程资源
// 每个辅助线程的资源集合
struct ncclNetSocketThreadResources {
  // threadTaskQueue: 线程的任务队列
  struct ncclNetSocketTaskQueue threadTaskQueue;
  // stop: 停止标志（0=运行，1=停止）
  int stop;
  // comm: 所属的通信上下文指针
  struct ncclNetSocketComm* comm;
  // pInfo: 性能分析信息指针
  struct ncclProfilerInfo* pInfo;
  // threadMutex: 线程互斥锁
  // 用于保护任务队列的并发访问
  std::mutex threadMutex;
  // threadCond: 线程条件变量
  // 用于在有新任务时唤醒线程
  std::condition_variable threadCond;
};

// 定义结构体：Socket 监听通信上下文
// 用于监听和接受传入的连接请求
struct ncclNetSocketListenComm {
  // sock: 监听 Socket
  struct ncclSocket sock;
  // stage: 连接建立阶段信息
  struct ncclNetSocketCommStage stage;
  // nSocks: 数据传输使用的 Socket 数量
  int nSocks;
  // nThreads: 辅助线程数量
  int nThreads;
  // dev: 网络设备索引
  int dev;
};

// 定义结构体：Socket 通信上下文
// 代表一个已建立的通信连接
struct ncclNetSocketComm {
  // ctrlSock: 控制 Socket
  // 用于传输控制信息和小数据（内联数据）
  struct ncclSocket ctrlSock;
  // socks: 数据传输 Socket 数组
  // 用于大数据的并行传输
  struct ncclSocket socks[MAX_SOCKETS];
  // dev: 网络设备索引
  int dev;
  // cudaDev: CUDA 设备索引
  // 记录通信关联的 GPU 设备
  int cudaDev;
  // nSocks: 实际使用的 Socket 数量
  int nSocks;
  // nThreads: 实际使用的辅助线程数量
  int nThreads;
  // nextSock: 下一个要分配的 Socket 索引
  // 用于轮询分配任务给不同的 Socket
  int nextSock;
  // inlineData: 内联数据缓冲区
  // 用于存储所有请求的控制信息和小数据
  void* inlineData;
  // requests: 请求数组
  // 管理所有活跃的传输请求
  struct ncclNetSocketRequest requests[MAX_REQUESTS];
  // helperThread: 辅助线程数组
  // 每个线程处理部分 Socket 的 I/O 操作
  pthread_t helperThread[MAX_THREADS];
  // threadResources: 线程资源数组
  // 每个线程有自己的任务队列和同步原语
  struct ncclNetSocketThreadResources threadResources[MAX_THREADS];
};

// 函数功能：持久化 Socket 辅助线程
// 这是一个线程函数，运行在每个辅助线程中，负责处理分配给该线程的 Socket I/O 操作
// 参数说明：
//   - args_: 线程资源指针（通过 pthread_create 传入）
// 返回值：NULL（线程退出时返回）
void* persistentSocketThread(void *args_) {
  // 将参数转换为线程资源指针
  struct ncclNetSocketThreadResources* resource = (struct ncclNetSocketThreadResources*)args_;
  // 获取通信上下文指针
  struct ncclNetSocketComm* comm = resource->comm;
  // 获取当前线程的任务队列指针
  struct ncclNetSocketTaskQueue* myQueue = &resource->threadTaskQueue;
  // 计算每个线程负责的 Socket 数量
  // 总 Socket 数 / 线程数 = 每个线程处理的 Socket 数
  int nSocksPerThread = comm->nSocks / comm->nThreads;
#ifdef NCCL_ENABLE_NET_PROFILING
  // 定义数组：性能分析事件句柄
  // 为每个可能的任务存储一个事件句柄
  void* eHandle[MAX_REQUESTS*MAX_SOCKETS] = { 0 };
#endif
  // 主循环：持续处理任务直到收到停止信号
  while (1) {
    // 标记：是否处于空闲状态（1=空闲，0=忙碌）
    int idle = 1;
    // 标记：当前看到的新任务位置
    // 用于检测是否有新任务到达
    int mark = myQueue->next; // mark newest task seen
    // 遍历任务队列中的所有任务
    // 步长为 nSocksPerThread，因为每个请求被分割到 nSocksPerThread 个槽位
    for (int i=0; i<myQueue->len; i+=nSocksPerThread) {
      // 重复标志：是否需要重复处理（是否有未完成的任务）
      int repeat;
      // 内层循环：重复处理直到所有子任务都完成或阻塞
      do {
        // 初始化重复标志为不重复
        repeat = 0;
        // 遍历当前请求的所有子任务（每个 Socket 一个子任务）
        for (int j=0; j<nSocksPerThread; j++) {
          // 获取任务指针
          struct ncclNetSocketTask* r = myQueue->tasks+i+j;
          // 检查任务是否有效且未完成
          // 条件：任务不为空 && 任务正在使用 && 传输偏移小于总大小
          if (r != NULL && r->used == 1 && r->offset < r->size) {
#ifdef NCCL_ENABLE_NET_PROFILING
            // 检查是否已创建性能分析事件
            if (!eHandle[i+j]) {
              // 定义性能分析数据结构
              ncclProfilerNetSockDescr_v1_t data;
              // 设置事件类型为 Socket
              data.type = ncclProfileSocket;
              // 设置 Socket 文件描述符
              data.sock.fd = r->sock->fd;
              // 设置操作类型（发送/接收）
              data.sock.op = r->op;
              // 设置数据长度
              data.sock.length = r->size;
              // 记录网络事件开始
              // &eHandle[i+j]: 输出事件句柄
              // ncclProfilerNetEventStart: 事件类型（开始）
              // resource->pInfo->pHandle: 父句柄
              // NCCL_PROFILER_NET_TYPE_SOCK | 1: 事件类型标志
              // &data: 事件详情数据
              ncclProfilerFunction(&eHandle[i+j], ncclProfilerNetEventStart, resource->pInfo->pHandle, NCCL_PROFILER_NET_TYPE_SOCK | 1, &data);
            }
#endif
            // 执行 Socket 传输进度
            // 尝试发送或接收更多数据
            // r->op: 操作类型（SEND/RECV）
            // r->sock: Socket 指针
            // r->data: 数据缓冲区
            // r->size: 总大小
            // &r->offset: 输入/输出参数，当前偏移和更新后的偏移
            r->result = ncclSocketProgress(r->op, r->sock, r->data, r->size, &r->offset);
            // 检查传输是否出错
            if (r->result != ncclSuccess) {
#ifdef NCCL_ENABLE_NET_PROFILING
              // 记录网络事件结束（出错情况）
              ncclProfilerFunction(&eHandle[i+j], ncclProfilerNetEventStop, NULL, 0, NULL);
              // 清空事件句柄
              eHandle[i+j] = NULL;
#endif
              // 输出警告日志
              WARN("NET/Socket : socket progress error");
              // 线程退出
              return NULL;
            }
            // 标记为非空闲状态
            idle = 0;
            // 检查任务是否完成（偏移是否达到总大小）
            // 如果未完成，设置重复标志为 1，继续处理
            if (r->offset < r->size) repeat = 1;
#ifdef NCCL_ENABLE_NET_PROFILING
            // 如果任务已完成，停止性能分析事件
            if (repeat == 0) {
              ncclProfilerFunction(&eHandle[i+j], ncclProfilerNetEventStop, NULL, 0, NULL);
              eHandle[i+j] = NULL;
            }
#endif
          }
        }
      } while (repeat); // 如果有未完成的任务，继续循环
    }
    // 如果处于空闲状态，等待新任务到达
    if (idle) {
      // 获取互斥锁
      std::unique_lock<std::mutex> lock(resource->threadMutex);
      // 等待条件变量
      // 条件：有新任务（mark != myQueue->next）或 收到停止信号
      resource->threadCond.wait(lock, [&] { return mark != myQueue->next || resource->stop; });
    }
    // 检查是否收到停止信号
    if (resource->stop) return NULL;
  }
}

// 函数功能：获取 Socket 数量和线程数量的最优配置
// 参数说明：
//   - dev: 网络设备索引
//   - ns: 输出参数，返回 Socket 总数
//   - nt: 输出参数，返回线程数量
// 返回值：ncclSuccess 表示成功
// 说明：根据用户配置或自动检测（云环境）确定最优的 Socket 和线程数量
ncclResult_t ncclNetSocketGetNsockNthread(int dev, int* ns, int* nt) {
  // 初始化返回值为成功
  ncclResult_t ret = ncclSuccess;
  // 获取每个线程的 Socket 数量参数
  int nSocksPerThread = ncclParamSocketNsocksPerThread();
  // 获取线程数量参数
  int nThreads = ncclParamSocketNthreads();
  // 检查线程数量是否超过最大值
  if (nThreads > MAX_THREADS) {
    // 输出警告日志，并将线程数量限制为最大值
    WARN("NET/Socket : NCCL_SOCKET_NTHREADS is greater than the maximum allowed, setting to %d", MAX_THREADS);
    nThreads = MAX_THREADS;
  }
  // 初始化文件描述符为 -1（无效值）
  int fd = -1;
  // 声明变量：Socket 总数
  int nSocks;
  // 检查是否需要自动检测（-2 表示自动检测）
  if (nThreads == -2 || nSocksPerThread == -2) {
    // Auto-detection
    // 自动检测模式
    // 声明并初始化自动检测的线程数和 Socket 数
    // 默认值：只使用主线程，不创建额外的辅助线程
    int autoNt=0, autoNs=1; // By default, we only use the main thread and do not spawn extra threads
    // 定义缓冲区：厂商 PCI 文件路径
    char vendorPath[PATH_MAX];
    // 构造 sysfs 中的 vendor 文件路径
    // /sys/class/net/<dev>/device/vendor 包含 PCI 厂商 ID
    snprintf(vendorPath, PATH_MAX, "/sys/class/net/%s/device/vendor", ncclNetSocketDevs[dev].devName);
    // Coverity is wrong.  NULL second argument to realpath() is OK by POSIX.1-2008.
    // Coverity 静态分析工具的警告是错误的，realpath 的第二个参数为 NULL 是符合 POSIX.1-2008 标准的
    // coverity[alias_transfer:FALSE]
    // 解析符号链接，获取绝对路径
    char* rPath = realpath(vendorPath, NULL);
    // 打开 vendor 文件
    fd = open(rPath, O_RDONLY);
    // 释放 realpath 分配的内存
    free(rPath);
    // 检查文件是否成功打开
    if (fd == -1) {
      // Could not find device vendor. This is handled silently so
      // 无法找到设备厂商文件。这是静默处理的，所以
      // we don't want to print an INFO error.
      // 我们不想打印 INFO 级别的错误
      // 输出跟踪级别的日志
      TRACE(NCCL_NET, "Open of %s failed : %s", vendorPath, strerror(errno));
      // 跳转到结束标签
      goto end;
    }
    // 定义缓冲区：厂商 ID 字符串
    char vendor[7];
    // 初始化为默认值 "0x0000"
    strncpy(vendor, "0x0000", 7);
    // 从文件读取厂商 ID（6 个字符）
    SYSCHECKGOTO(read(fd, vendor, 6), "read", ret, fail);
    // 检查是否为 AWS（Amazon Web Services）
    // 厂商 ID 0x1d0f 是 AWS Elastic Network Adapter
    if (strcmp(vendor, "0x1d0f") == 0) { // AWS
      // AWS 环境的最优配置
      autoNt = 2;  // 2 个线程
      autoNs = 8;  // 每个线程 8 个 Socket
    // 检查是否为 GCP（Google Cloud Platform）
    // 厂商 ID 0x1ae0 是 Google Virtual NIC
    } else if (strcmp(vendor, "0x1ae0") == 0) { // GCP
      // GCP 环境的最优配置
      autoNt = 4;  // 4 个线程
      autoNs = 1;  // 每个线程 1 个 Socket
    }
// 结束标签：自动检测完成后的跳转点
end:
    // 如果线程数量为 -2，使用自动检测的值
    if (nThreads == -2) nThreads = autoNt;
    // 如果每个线程的 Socket 数为 -2，使用自动检测的值
    if (nSocksPerThread == -2) nSocksPerThread = autoNs;
  }
  // 计算 Socket 总数 = 每线程 Socket 数 × 线程数
  nSocks = nSocksPerThread * nThreads;
  // 检查 Socket 总数是否超过最大值
  if (nSocks > MAX_SOCKETS) {
    // 重新计算每个线程的 Socket 数，使总数不超过最大值
    nSocksPerThread = MAX_SOCKETS/nThreads;
    // 输出警告日志
    WARN("NET/Socket : the total number of sockets is greater than the maximum allowed, setting NCCL_NSOCKS_PERTHREAD to %d", nSocksPerThread);
    // 重新计算 Socket 总数
    nSocks = nSocksPerThread * nThreads;
  }
  // 输出 Socket 总数
  *ns = nSocks;
  // 输出线程数量
  *nt = nThreads;
  // 如果使用了 Socket，输出信息日志
  if (nSocks > 0) INFO(NCCL_INIT, "NET/Socket: Using %d threads and %d sockets per thread", nThreads, nSocksPerThread);
// 正常退出标签
exit:
  // 如果文件已打开，关闭文件描述符
  if (fd != -1) close(fd);
  // 返回结果状态码
  return ret;
// 失败标签
fail:
  // 跳转到退出标签
  goto exit;
}

// 函数功能：监听网络连接请求
// 参数说明：
//   - ctx: 插件上下文（未使用）
//   - dev: 网络设备索引
//   - opaqueHandle: 输出参数，返回连接句柄（用于传递给对等体）
//   - listenComm: 输出参数，返回监听通信上下文
// 返回值：ncclSuccess 表示成功
ncclResult_t ncclNetSocketListen(void* ctx, int dev, void* opaqueHandle, void** listenComm) {
  // 检查设备索引是否有效
  if (dev < 0 || dev >= ncclNetIfs) { // data transfer socket is based on specified dev
    // 设备索引无效，输出警告日志
    // 数据传输 Socket 基于指定的设备
    WARN("NET/Socket : ncclNetSocketListen dev=%d ncclNetIfs=%d", dev, ncclNetIfs);
    // 返回内部错误
    return ncclInternalError;
  }
  // 初始化返回值为成功
  ncclResult_t ret = ncclSuccess;
  // 将不透明的句柄指针转换为 Socket 网络句柄指针
  struct ncclNetSocketHandle* handle = (struct ncclNetSocketHandle*) opaqueHandle;
  // 清零句柄内存
  memset(handle, 0, sizeof(struct ncclNetSocketHandle));
  // 编译时断言：确保句柄大小不超过 NCCL 定义的最大值
  // 句柄需要在进程间传递，大小受限制
  static_assert(sizeof(struct ncclNetSocketHandle) <= NCCL_NET_HANDLE_MAXSIZE, "ncclNetSocketHandle size too large");
  // 声明监听通信上下文指针
  struct ncclNetSocketListenComm* comm;
  // 分配并清零监听通信上下文内存
  NCCLCHECK(ncclCalloc(&comm, 1));
  // 设置魔数，用于验证连接的正确性
  handle->magic = NCCL_SOCKET_MAGIC;
  // 初始化监听 Socket
  // &comm->sock: Socket 结构体指针
  // &ncclNetSocketDevs[dev].addr: 绑定地址
  // handle->magic: 魔数
  // ncclSocketTypeNetSocket: Socket 类型
  // NULL: 不使用特定端口（自动分配）
  // 1: 设置为监听模式
  NCCLCHECKGOTO(ncclSocketInit(&comm->sock, &ncclNetSocketDevs[dev].addr, handle->magic, ncclSocketTypeNetSocket, NULL, 1), ret, fail);
  // 开始监听连接请求
  NCCLCHECKGOTO(ncclSocketListen(&comm->sock), ret, fail);
  // 获取监听 Socket 的地址（包含自动分配的端口号）
  NCCLCHECKGOTO(ncclSocketGetAddr(&comm->sock, &handle->connectAddr), ret, fail);
  // 获取最优的 Socket 和线程数量配置
  NCCLCHECKGOTO(ncclNetSocketGetNsockNthread(dev, &comm->nSocks, &comm->nThreads), ret, fail);
  // 将 Socket 数量保存到句柄
  handle->nSocks = comm->nSocks;
  // 将线程数量保存到句柄
  handle->nThreads = comm->nThreads;
  // 保存网络设备索引
  comm->dev = dev;
  // 输出监听通信上下文
  *listenComm = comm;
// 正常退出标签
exit:
  // 返回结果状态码
  return ret;
// 失败标签
fail:
  // 关闭 Socket
  (void)ncclSocketClose(&comm->sock);
  // 释放通信上下文内存
  free(comm);
  // 跳转到退出标签
  goto exit;
}

// 定义宏：Socket 控制信息大小
// 控制信息包含数据大小（int 类型）
#define SOCKET_CTRL_SIZE (sizeof(int))

// 函数功能：连接到远程对等体（主动连接）
// 参数说明：
//   - ctx: 插件上下文（未使用）
//   - dev: 网络设备索引
//   - opaqueHandle: 从 Listen 获取的连接句柄
//   - sendComm: 输出参数，返回发送通信上下文
//   - sendDevComm: 输出参数（未使用）
// 返回值：ncclSuccess 表示成功（可能需要多次调用完成连接）
// 说明：这是一个非阻塞函数，使用状态机支持多次调用完成连接过程
ncclResult_t ncclNetSocketConnect(void* ctx, int dev, void* opaqueHandle, void** sendComm, ncclNetDeviceHandle_t** /*sendDevComm*/) {
  // 检查设备索引是否有效
  if (dev < 0 || dev >= ncclNetIfs) { // data transfer socket is based on specified dev
    // 设备索引无效，返回内部错误
    return ncclInternalError;
  }

  // 声明变量：Socket 就绪标志
  int ready;
  // 将不透明的句柄指针转换为 Socket 网络句柄指针
  struct ncclNetSocketHandle* handle = (struct ncclNetSocketHandle*) opaqueHandle;
  // 获取连接阶段信息指针
  struct ncclNetSocketCommStage* stage = &handle->stage;
  // 获取通信上下文指针（可能为 NULL，表示首次调用）
  struct ncclNetSocketComm* comm = stage->comm;
  // 获取当前迭代次数（用于恢复状态）
  uint8_t i = stage->iteration;
  // 获取当前 Socket 指针
  struct ncclSocket* sock = stage->sock;
  // 初始化输出为 NULL（连接完成前不可用）
  *sendComm = NULL;

  // 状态机：检查当前状态，跳转到对应位置
  // 如果已处于连接状态，跳转到连接检查
  if (stage->state == ncclNetSocketCommStateConnect) goto socket_connect_check;
  // 如果已处于发送状态，跳转到发送阶段
  if (stage->state == ncclNetSocketCommStateSend) goto socket_send;

  // 首次调用：创建通信上下文
  comm = new ncclNetSocketComm();
  // 保存通信上下文指针到阶段信息
  stage->comm = comm;
  // 从句柄复制 Socket 数量
  comm->nSocks = handle->nSocks;
  // 从句柄复制线程数量
  comm->nThreads = handle->nThreads;
  // 保存网络设备索引
  comm->dev = dev;
  // 获取当前 CUDA 设备索引
  CUDACHECK(cudaGetDevice(&comm->cudaDev));
  // 循环：建立所有 Socket 的连接（包括控制 Socket）
  // 从当前迭代位置开始，支持多次调用恢复
  for (; i<comm->nSocks+1; i++) {
    // 确定当前 Socket：
    // 最后一个 (i == nSocks) 是控制 Socket
    // 其他是数据传输 Socket
    sock = (i == comm->nSocks) ? &comm->ctrlSock : comm->socks+i;
    // 初始化 Socket
    NCCLCHECK(ncclSocketInit(sock, &handle->connectAddr, handle->magic, ncclSocketTypeNetSocket, NULL, 1));

    // 保存当前状态到阶段信息（支持非阻塞恢复）
    stage->sock = sock;
    stage->state = ncclNetSocketCommStateConnect;
    stage->iteration = i;
    // 发起连接请求（非阻塞）
    NCCLCHECK(ncclSocketConnect(sock));

// 连接检查标签：检查 Socket 是否连接成功
socket_connect_check:
    // 检查 Socket 是否就绪（连接完成）
    NCCLCHECK(ncclSocketReady(sock, &ready));
    // 如果未就绪，返回成功（需要稍后再次调用）
    if (! ready) return ncclSuccess;
    // 连接完成，更新状态为发送阶段
    stage->state = ncclNetSocketCommStateSend;

// 发送标签：发送 Socket 索引到对等体
socket_send:
    // 声明变量：完成标志
    int done = 0;
    // 发送当前 Socket 索引给对等体
    // 对等体需要知道这个 Socket 在其数组中的位置
    NCCLCHECK(ncclSocketProgress(NCCL_SOCKET_SEND, sock, &i, sizeof(uint8_t), &done));
    // 如果未完成，返回成功（需要稍后再次调用）
    if (done == 0) return ncclSuccess;
  }
  // 所有 Socket 连接完成，分配内联数据缓冲区
  // 每个请求有一个控制信息区和一个内联数据区
  NCCLCHECK(ncclCalloc(&comm->inlineData, MAX_REQUESTS * (SOCKET_CTRL_SIZE + ncclParamSocketInlineSize())));
  // 输出通信上下文
  *sendComm = comm;
  // 返回成功状态码
  return ncclSuccess;
}

// 函数功能：接受传入的连接请求（被动连接）
// 参数说明：
//   - listenComm: 监听通信上下文
//   - recvComm: 输出参数，返回接收通信上下文
//   - recvDevComm: 输出参数（未使用）
// 返回值：ncclSuccess 表示成功（可能需要多次调用完成接受）
// 说明：这是一个非阻塞函数，使用状态机支持多次调用完成接受过程
ncclResult_t ncclNetSocketAccept(void* listenComm, void** recvComm, ncclNetDeviceHandle_t** /*recvDevComm*/) {
  // 将监听通信上下文指针转换为正确的类型
  struct ncclNetSocketListenComm* lComm = (struct ncclNetSocketListenComm*)listenComm;
  // 获取连接阶段信息指针
  struct ncclNetSocketCommStage* stage = &lComm->stage;
  // 获取接收通信上下文指针（可能为 NULL，表示首次调用）
  struct ncclNetSocketComm* rComm = stage->comm;
  // 获取当前迭代次数（用于恢复状态）
  uint8_t i = stage->iteration;
  // 获取当前 Socket 指针
  struct ncclSocket* sock = stage->sock;
  // 声明变量：Socket 就绪标志
  int ready;

  // 初始化输出为 NULL（接受完成前不可用）
  *recvComm = NULL;
  // 状态机：检查当前状态，跳转到对应位置
  // 如果已处于接受状态，跳转到接受检查
  if (stage->state == ncclNetSocketCommStateAccept) goto socket_accept_check;
  // 如果已处于接收状态，跳转到接收阶段
  if (stage->state == ncclNetSocketCommStateRecv) goto socket_recv;

  // 首次调用：创建接收通信上下文
  rComm = new ncclNetSocketComm();
  // 保存通信上下文指针到阶段信息
  stage->comm = rComm;
  // 从监听上下文复制 Socket 数量
  rComm->nSocks = lComm->nSocks;
  // 从监听上下文复制线程数量
  rComm->nThreads = lComm->nThreads;
  // 从监听上下文复制网络设备索引
  rComm->dev = lComm->dev;
  // 获取当前 CUDA 设备索引
  CUDACHECK(cudaGetDevice(&rComm->cudaDev));
  // 循环：接受所有 Socket 的连接（包括控制 Socket）
  // 从当前迭代位置开始，支持多次调用恢复
  for (; i<rComm->nSocks+1; i++) {
    // 声明变量：发送端的 Socket 索引
    // 对等体会发送它的 Socket 索引，我们需要知道这是哪个 Socket
    uint8_t sendSockIdx;

    // 分配并初始化 Socket 结构体
    NCCLCHECK(ncclCalloc(&sock, 1));
    // 初始化 Socket（不连接，用于接受连接）
    NCCLCHECK(ncclSocketInit(sock));
    // 保存当前状态到阶段信息（支持非阻塞恢复）
    stage->sock = sock;
    stage->state = ncclNetSocketCommStateAccept;
    stage->iteration = i;
    // 接受连接请求（非阻塞）
    NCCLCHECK(ncclSocketAccept(sock, &lComm->sock));

// 接受检查标签：检查 Socket 是否接受成功
socket_accept_check:
    // 检查 Socket 是否就绪（连接完成）
    NCCLCHECK(ncclSocketReady(sock, &ready));
    // 如果未就绪，返回成功（需要稍后再次调用）
    if (!ready) return ncclSuccess;

    // 接受完成，更新状态为接收阶段
    stage->state = ncclNetSocketCommStateRecv;
// 接收标签：接收发送端的 Socket 索引
socket_recv:
    // 声明变量：完成标志
    int done = 0;
    // 接收发送端的 Socket 索引
    NCCLCHECK(ncclSocketProgress(NCCL_SOCKET_RECV, sock, &sendSockIdx, sizeof(uint8_t), &done));
    // 如果未完成，返回成功（需要稍后再次调用）
    if (done == 0) return ncclSuccess;

    // 根据发送端的索引，将 Socket 保存到正确位置
    // 如果是控制 Socket（索引 == nSocks）
    if (sendSockIdx == rComm->nSocks)
      // 保存为控制 Socket
      rComm->ctrlSock = *sock;
    else
      // 保存到数据 Socket 数组的对应位置
      rComm->socks[sendSockIdx] = *sock;
    // 释放临时分配的 Socket 结构体（内容已复制到 rComm）
    free(sock);
  }
  // 所有 Socket 接受完成，分配内联数据缓冲区
  NCCLCHECK(ncclCalloc(&rComm->inlineData, MAX_REQUESTS * (SOCKET_CTRL_SIZE + ncclParamSocketInlineSize())));
  // 输出接收通信上下文
  *recvComm = rComm;

  /* reset lComm state */
  /* 重置监听通信上下文状态，准备接受下一个连接 */
  stage->state = ncclNetSocketCommStateStart;
  stage->iteration = 0;
  stage->sock = NULL;
  stage->comm = NULL;
  // 返回成功状态码
  return ncclSuccess;
}

// 函数功能：获取一个可用的请求结构体
// 参数说明：
//   - comm: 通信上下文指针
//   - op: 操作类型（NCCL_SOCKET_SEND 或 NCCL_SOCKET_RECV）
//   - data: 用户数据缓冲区指针
//   - size: 数据大小
//   - req: 输出参数，返回请求结构体指针
// 返回值：ncclSuccess 表示成功
ncclResult_t ncclNetSocketGetRequest(struct ncclNetSocketComm* comm, int op, void* data, int size, struct ncclNetSocketRequest** req) {
  // 遍历所有请求槽位，寻找一个空闲的请求
  for (int i=0; i<MAX_REQUESTS; i++) {
    // 获取当前请求指针
    struct ncclNetSocketRequest* r = comm->requests+i;
    // 检查请求是否未被使用
    if (r->used == 0) {
      // 初始化请求参数
      // 设置操作类型
      r->op = op;
      // 设置数据缓冲区指针
      r->data = data;
      // 设置数据大小
      r->size = size;
      // 设置控制 Socket 指针
      r->ctrlSock = &comm->ctrlSock;
      // 标记请求为正在使用（状态 1：交换控制信息中）
      r->used = 1;
      // 保存通信上下文指针
      r->comm = comm;
      // 初始化子任务数量为 0
      r->nSubs = 0;
      // 计算内联数据缓冲区位置
      // 每个请求的内联缓冲区 = 基址 + 请求索引 * (控制大小 + 内联数据大小)
      r->inlineData = (uint8_t*)comm->inlineData + i * (SOCKET_CTRL_SIZE + ncclParamSocketInlineSize());
      // 输出请求指针
      *req = r;
      // 返回成功状态码
      return ncclSuccess;
    }
  }
  // 没有可用的请求槽位，输出警告日志
  WARN("NET/Socket : unable to allocate requests");
  // 返回内部错误
  return ncclInternalError;
}

// 函数功能：获取一个可用的任务结构体（用于辅助线程）
// 参数说明：
//   - comm: 通信上下文指针
//   - pInfo: 性能分析信息指针
//   - op: 操作类型
//   - data: 数据缓冲区指针
//   - size: 数据大小
//   - req: 输出参数，返回任务结构体指针
// 返回值：ncclSuccess 表示成功
ncclResult_t ncclNetSocketGetTask(struct ncclNetSocketComm* comm, struct ncclProfilerInfo* pInfo, int op, void* data, int size, struct ncclNetSocketTask** req) {
  // 计算线程 ID：使用轮询方式分配给不同线程
  // nextSock % nThreads: 将 Socket 循环分配给线程
  int tid = comm->nextSock % comm->nThreads;
  // 获取线程资源指针
  struct ncclNetSocketThreadResources* res = comm->threadResources+tid;
  // 获取线程的任务队列指针
  struct ncclNetSocketTaskQueue* queue = &res->threadTaskQueue;
  // create helper threads and prepare per-thread task queue
  // 创建辅助线程并准备每线程任务队列
  // 检查任务队列是否已初始化
  if (queue->tasks == NULL) {
    // each request can be divided up to nSocks tasks, and
    // 每个请求可以分割成最多 nSocks 个任务，并且
    // these tasks are distributed to nThreads threads,
    // 这些任务分配给 nThreads 个线程，
    // we need to make sure each thread queue has enough slots for MAX_REQUESTS
    // 我们需要确保每个线程队列有足够的槽位来存储 MAX_REQUESTS 个请求的任务
    // 计算队列长度
    // DIVUP: 向上取整除法
    // 例如：nSocks=8, nThreads=2, MAX_REQUESTS=16 => 16 * 4 = 64
    queue->len = MAX_REQUESTS * DIVUP(comm->nSocks, comm->nThreads);
    // 分配任务数组内存
    NCCLCHECK(ncclCalloc(&queue->tasks, queue->len));
    // 初始化下一个分配位置为 0
    queue->next = 0;
    // 保存通信上下文指针到线程资源
    res->comm = comm;
#ifdef NCCL_ENABLE_NET_PROFILING
    // 保存性能分析信息指针
    res->pInfo = pInfo;
#endif
    // 创建辅助线程
    // 线程函数：persistentSocketThread
    // 线程参数：线程资源指针
    PTHREADCHECK(pthread_create(comm->helperThread+tid, NULL, persistentSocketThread, res), "pthread_create");
    // 设置线程名称（便于调试）
    // 格式：NCCL Sock[S/R]<dev><tid><cudaDev>
    // S=Send, R=Recv
    ncclSetThreadName(comm->helperThread[tid], "NCCL Sock%c%1u%2u%2u", op == NCCL_SOCKET_SEND ? 'S' : 'R', comm->dev, tid, comm->cudaDev);
  }
  // 获取下一个可用任务槽位
  struct ncclNetSocketTask* r = queue->tasks+queue->next;
  // 检查任务是否未被使用
  if (r->used == 0) {
    // 初始化任务参数
    // 设置操作类型
    r->op = op;
    // 设置数据缓冲区指针
    r->data = data;
    // 设置数据大小
    r->size = size;
    // 设置要使用的 Socket（轮询分配）
    r->sock = comm->socks + comm->nextSock;
    // 初始化偏移量为 0
    r->offset = 0;
    // 初始化结果为成功
    r->result = ncclSuccess;
    // 移动到下一个 Socket（循环）
    comm->nextSock = (comm->nextSock + 1) % comm->nSocks;
    // 标记任务为正在使用
    r->used = 1;
    // 输出任务指针
    *req = r;
    // 加锁保护队列操作
    std::lock_guard<std::mutex> lock(res->threadMutex);
    // 移动到下一个任务槽位（循环）
    queue->next = (queue->next+1)%queue->len;
    // 唤醒辅助线程（通知有新任务）
    res->threadCond.notify_one();
    // 返回成功状态码
    return ncclSuccess;
  }
  // 没有可用的任务槽位，输出警告日志
  WARN("NET/Socket : unable to allocate subtasks");
  // 返回内部错误
  return ncclInternalError;
}

// 函数功能：计算内联数据大小
// 参数说明：
//   - dataSize: 数据总大小
// 返回值：如果数据足够小，返回数据大小；否则返回 0
// 说明：如果数据小于等于内联阈值，可以与控制信息一起发送，避免额外的内存拷贝
// if the dataSize is smaller than the inline size, return the inline size; if not, return 0 to avoid the extra copy.
// 如果数据大小小于内联大小，返回内联大小；否则返回 0 以避免额外的拷贝
static int ncclNetSocketInlineSize(int dataSize) { return (dataSize <= ncclParamSocketInlineSize()) ? dataSize : 0; }

// 函数功能：测试请求是否完成
// 参数说明：
//   - request: 请求指针
//   - done: 输出参数，返回是否完成（0=未完成，1=完成）
//   - size: 输出参数，返回实际传输的数据大小（可选）
// 返回值：ncclSuccess 表示成功
// 说明：这是一个非阻塞函数，需要多次调用直到请求完成
ncclResult_t ncclNetSocketTest(void* request, int* done, int* size) {
  // 初始化完成标志为未完成
  *done = 0;
  // 将请求指针转换为正确的类型
  struct ncclNetSocketRequest *r = (struct ncclNetSocketRequest*)request;
  // 检查请求指针是否有效
  if (r == NULL) {
    // 请求指针为空，输出警告日志
    WARN("NET/Socket : test called with NULL request");
    // 返回内部错误
    return ncclInternalError;
  }
  // 状态 1：交换控制信息（数据大小）和内联数据
  if (r->used == 1) { /* try to send/recv size (+ inline data if any) */
    // 尝试发送/接收数据大小（如果有内联数据也一起传输）
    // 声明变量：消息大小
    int msgSize;
    // 消息缓冲区指针（指向内联数据区）
    uint8_t* msg = (uint8_t*)r->inlineData;
    // 根据操作类型分别处理
    if (r->op == NCCL_SOCKET_SEND) {
      // sender side has the right data size, copy size info + inline data to the buffer
      // 发送方知道正确的数据大小，将大小信息和内联数据复制到缓冲区
      // 计算内联数据大小（如果数据足够小）
      int inlineSize = ncclNetSocketInlineSize(r->size);
      // 消息大小 = 内联数据大小 + 控制信息大小
      msgSize = inlineSize + SOCKET_CTRL_SIZE;
      // 复制数据大小到消息缓冲区
      memcpy(msg, &r->size, SOCKET_CTRL_SIZE);
      // 如果有内联数据，复制到消息缓冲区（紧接在控制信息之后）
      if (inlineSize > 0) memcpy(msg + SOCKET_CTRL_SIZE, r->data, inlineSize);
    } else {
      // receiver side doesn't have the right data size, wait for the sender to send it
      // 接收方不知道正确的数据大小，等待发送方发送过来
      // 声明变量：大小信息的偏移量和发送方的数据大小
      int sizeOffset = 0, senderSize = 0;
      // 循环：接收控制信息（数据大小）
      while (sizeOffset < SOCKET_CTRL_SIZE) {
        // 尝试接收更多数据
        NCCLCHECK(ncclSocketProgress(r->op, r->ctrlSock, msg, SOCKET_CTRL_SIZE, &sizeOffset));
        // 如果没有接收到任何数据（Socket 未就绪），返回成功（需要稍后再次调用）
        if (sizeOffset == 0) return ncclSuccess; /* not ready yet*/
      }
      // 从消息缓冲区复制出发送方的数据大小
      memcpy(&senderSize, msg, SOCKET_CTRL_SIZE);
      // 检查发送方的数据大小是否超过接收方的缓冲区大小
      if (senderSize > r->size) {
        // 定义缓冲区：地址字符串
        char line[SOCKET_NAME_MAXLEN + 1];
        // 声明变量：Socket 地址
        union ncclSocketAddress addr;
        // 获取对等体的地址
        NCCLCHECK(ncclSocketGetAddr(r->ctrlSock, &addr));
        // 输出警告日志：消息被截断
        WARN("NET/Socket : peer %s message truncated : receiving %d bytes instead of %d. If you believe your socket network is in a healthy state, "
             "there may be a mismatch in collective sizes or environment settings (e.g. NCCL_PROTO, NCCL_ALGO) between ranks",
             ncclSocketToString(&addr, line), senderSize, r->size);
        // 返回无效使用错误
        return ncclInvalidUsage;
      }
      // copy to the data buffer if we have received some inline data already
      // 如果已经接收到一些内联数据，复制到数据缓冲区
      // 计算已接收的内联数据大小
      int receivedInline = sizeOffset - SOCKET_CTRL_SIZE;
      // 如果有内联数据，复制到用户缓冲区
      if (receivedInline > 0) memcpy(r->data, msg + SOCKET_CTRL_SIZE, receivedInline);
      // from the actual size, extract the remaining inline size to be received and redirect the msg buffer to the user data
      // 从实际大小中，提取剩余需要接收的内联数据大小，并将消息缓冲区重定向到用户数据
      // 更新请求的数据大小为发送方的实际大小
      r->size = senderSize;
      // 计算剩余需要接收的内联数据大小
      msgSize = ncclNetSocketInlineSize(r->size) - receivedInline;
      // 将消息缓冲区指向用户数据区（跳过已接收的部分）
      msg = (uint8_t*)r->data + receivedInline;
    }
    // 声明变量：传输偏移量
    int offset = 0;
    // 循环：传输消息（内联数据或控制信息）
    while (offset < msgSize) {
      // 尝试传输更多数据
      NCCLCHECK(ncclSocketProgress(r->op, r->ctrlSock, msg, msgSize, &offset));
      // 如果没有传输任何数据（Socket 未就绪），返回成功
      if (offset == 0) return ncclSuccess; /* not ready yet*/
    }
    // done exchanging sizes, r->size now contains the actual size
    // 完成大小交换，r->size 现在包含实际的大小
    // 更新状态为 2：传输数据中
    r->used = 2;
    // 设置偏移量为已传输的内联数据大小
    r->offset = ncclNetSocketInlineSize(r->size);
    // 声明变量：数据块偏移量和任务索引
    int chunkOffset = r->offset, i = 0;
    // 检查是否使用数据 Socket（如果配置了多个 Socket）
    if (r->comm->nSocks > 0) {
      // each request can be divided up to nSocks tasks, we use the size left to transfer
      // 每个请求可以分割成最多 nSocks 个任务，我们使用剩余需要传输的大小
      // 计算任务大小：取最小任务大小和平均大小的较大值
      int taskSize = std::max((int)ncclParamSocketMinTaskSize(), DIVUP(r->size - r->offset, r->comm->nSocks));
      // 循环：将剩余数据分割成多个任务
      while (chunkOffset < r->size) {
        // 计算当前数据块大小（取任务大小和剩余大小的较小值）
        int chunkSize = std::min(taskSize, r->size - chunkOffset);
        // 创建子任务
        NCCLCHECK(ncclNetSocketGetTask(r->comm, &r->pInfo, r->op, (char*)(r->data) + chunkOffset, chunkSize, r->tasks + i++));
        // 移动到下一个数据块
        chunkOffset += chunkSize;
      }
    }
    // 保存子任务数量
    r->nSubs = i;
  }

  // 状态 2：传输数据（已完成控制信息交换）
  if (r->used == 2) { // already exchanged size
    // 已经交换大小信息
    // 检查是否有子任务（使用辅助线程）
    if (r->nSubs > 0) {
      // 使用辅助线程处理多个 Socket 的并行传输
      // 声明变量：已完成的子任务数量
      int nCompleted = 0;
      // 遍历所有子任务，检查是否完成
      for (int i=0; i<r->nSubs; i++) {
        // 获取子任务指针
        struct ncclNetSocketTask* sub = r->tasks[i];
        // 检查子任务是否出错
        if (sub->result != ncclSuccess) return sub->result;
        // 检查子任务是否完成（偏移量等于大小）
        if (sub->offset == sub->size) nCompleted++;
      }
      // 检查所有子任务是否都完成
      if (nCompleted == r->nSubs) {
        // 如果提供了 size 指针，返回实际传输的大小
        if (size) *size = r->size;
        // 标记请求为完成
        *done = 1;
        // 释放请求槽位
        r->used = 0;
        // 释放所有子任务槽位
        for (int i=0; i<r->nSubs; i++) {
          struct ncclNetSocketTask* sub = r->tasks[i];
          sub->used = 0;
        }
      }
    } else {
      // progress request using main thread
      // 使用主线程处理请求（没有配置数据 Socket 或数据很小）
#ifdef NCCL_ENABLE_NET_PROFILING
      // 检查是否已创建性能分析事件
      if (!r->pInfo.eHandle) {
        // 定义性能分析数据结构
        ncclProfilerNetSockDescr_v1_t data;
        // 设置事件类型为 Socket
        data.type = ncclProfileSocket;
        // 设置 Socket 文件描述符
        data.sock.fd = r->ctrlSock->fd;
        // 设置操作类型
        data.sock.op = r->op;
        // 设置数据长度
        data.sock.length = r->size;
        // 记录网络事件开始
        ncclProfilerFunction(&r->pInfo.eHandle, ncclProfilerNetEventStart, r->pInfo.pHandle, NCCL_PROFILER_NET_TYPE_SOCK | 1, &data);
      }
#endif
      // 检查是否还有数据需要传输
      if (r->offset < r->size) {
        // 尝试传输更多数据（使用控制 Socket）
        NCCLCHECK(ncclSocketProgress(r->op, r->ctrlSock, r->data, r->size, &r->offset));
      }
      // 检查是否完成所有数据传输
      if (r->offset == r->size) {
        // 如果提供了 size 指针，返回实际传输的大小
        if (size) *size = r->size;
        // 标记请求为完成
        *done = 1;
        // 释放请求槽位
        r->used = 0;
#ifdef NCCL_ENABLE_NET_PROFILING
        // 停止性能分析事件
        ncclProfilerFunction(&r->pInfo.eHandle, ncclProfilerNetEventStop, NULL, 0, NULL);
        // 清空事件句柄
        r->pInfo.eHandle = NULL;
#endif
      }
    }
  }
  // 返回成功状态码
  return ncclSuccess;
}

// 函数功能：注册内存区域
// 参数说明：
//   - comm: 通信上下文（未使用）
//   - data: 内存缓冲区指针
//   - size: 缓冲区大小
//   - type: 内存类型（NCCL_PTR_HOST 或 NCCL_PTR_CUDA）
//   - mhandle: 输出参数，返回内存句柄（未使用）
// 返回值：ncclSuccess 表示成功
// 说明：Socket 传输只支持主机内存，不需要显式注册
ncclResult_t ncclNetSocketRegMr(void* comm, void* data, size_t size, int type, void** mhandle) {
  // Socket 传输只支持主机内存
  // 如果类型不是主机内存，返回内部错误
  return (type != NCCL_PTR_HOST) ? ncclInternalError : ncclSuccess;
}

// 函数功能：注销内存区域
// 参数说明：
//   - comm: 通信上下文（未使用）
//   - mhandle: 内存句柄（未使用）
// 返回值：ncclSuccess 表示成功
// 说明：Socket 传输不需要显式注册，因此也不需要注销
ncclResult_t ncclNetSocketDeregMr(void* comm, void* mhandle) { return ncclSuccess; }

// 函数功能：发起非阻塞发送操作
// 参数说明：
//   - sendComm: 发送通信上下文
//   - data: 要发送的数据缓冲区
//   - size: 数据大小
//   - tag: 标签（未使用）
//   - mhandle: 内存句柄（未使用）
//   - phandle: 性能分析父句柄
//   - request: 输出参数，返回请求句柄
// 返回值：ncclSuccess 表示成功
ncclResult_t ncclNetSocketIsend(void* sendComm, void* data, size_t size, int tag, void* mhandle, void* phandle, void** request) {
  // 将发送通信上下文指针转换为正确的类型
  struct ncclNetSocketComm* comm = (struct ncclNetSocketComm*)sendComm;
  // 获取一个可用的请求结构体并初始化
  NCCLCHECK(ncclNetSocketGetRequest(comm, NCCL_SOCKET_SEND, data, (int) size, (struct ncclNetSocketRequest**)request));
#ifdef NCCL_ENABLE_NET_PROFILING
  // NCCL core profiler callback
  // NCCL 核心性能分析回调
  // 获取请求指针并保存性能分析父句柄
  struct ncclNetSocketRequest* req = *(struct ncclNetSocketRequest **)request;
  req->pInfo.pHandle = phandle;
#endif
  // 返回成功状态码
  return ncclSuccess;
}

// 函数功能：发起非阻塞接收操作
// 参数说明：
//   - recvComm: 接收通信上下文
//   - n: 接收缓冲区数量
//   - data: 接收数据缓冲区数组
//   - sizes: 接收缓冲区大小数组
//   - tags: 标签数组（未使用）
//   - mhandles: 内存句柄数组（未使用）
//   - phandles: 性能分析父句柄数组
//   - request: 输出参数，返回请求句柄
// 返回值：ncclSuccess 表示成功
ncclResult_t ncclNetSocketIrecv(void* recvComm, int n, void** data, size_t* sizes, int* tags, void** mhandles, void** phandles, void** request) {
  // 将接收通信上下文指针转换为正确的类型
  struct ncclNetSocketComm* comm = (struct ncclNetSocketComm*)recvComm;
  // Socket 传输只支持单个缓冲区的接收
  // 检查缓冲区数量是否为 1
  if (n != 1) return ncclInternalError;
  // 获取一个可用的请求结构体并初始化
  NCCLCHECK(ncclNetSocketGetRequest(comm, NCCL_SOCKET_RECV, data[0], (int)sizes[0], (struct ncclNetSocketRequest**)request));
#ifdef NCCL_ENABLE_NET_PROFILING
  // NCCL core profiler callback
  // NCCL 核心性能分析回调
  // 获取请求指针
  struct ncclNetSocketRequest* req = *(struct ncclNetSocketRequest **)request;
  // 如果提供了性能分析句柄数组，保存父句柄
  if (phandles) req->pInfo.pHandle = phandles[0];
#endif
  // 返回成功状态码
  return ncclSuccess;
}

// 函数功能：发起非阻塞刷新操作
// 参数说明：
//   - recvComm: 接收通信上下文
//   - n: 缓冲区数量
//   - data: 数据缓冲区数组
//   - sizes: 数据大小数组
//   - mhandles: 内存句柄数组
//   - request: 输出参数，返回请求句柄
// 返回值：ncclInternalError（不支持）
// 说明：Socket 传输不支持 CUDA 指针，因此不需要刷新操作
ncclResult_t ncclNetSocketIflush(void* recvComm, int n, void** data, int* sizes, void** mhandles, void** request) {
  // We don't support CUDA pointers, so we don't need a flush operation
  // 我们不支持 CUDA 指针，因此不需要刷新操作
  return ncclInternalError;
}

// 函数功能：关闭监听通信上下文
// 参数说明：
//   - opaqueComm: 监听通信上下文指针
// 返回值：ncclSuccess 表示成功
ncclResult_t ncclNetSocketCloseListen(void* opaqueComm) {
  // 将不透明的通信上下文指针转换为正确的类型
  struct ncclNetSocketListenComm* comm = (struct ncclNetSocketListenComm*)opaqueComm;
  // 检查通信上下文是否有效
  if (comm) {
    // 声明变量：Socket 就绪标志
    int ready;
    // 检查监听 Socket 是否就绪
    NCCLCHECK(ncclSocketReady(&comm->sock, &ready));
    // 如果就绪，关闭 Socket
    if (ready) NCCLCHECK(ncclSocketClose(&comm->sock));
    // 释放监听通信上下文内存
    free(comm);
  }
  // 返回成功状态码
  return ncclSuccess;
}

// 函数功能：关闭通信连接并释放资源
// 参数说明：
//   - opaqueComm: 通信上下文指针
// 返回值：ncclSuccess 表示成功
ncclResult_t ncclNetSocketClose(void* opaqueComm) {
  // 将不透明的通信上下文指针转换为正确的类型
  struct ncclNetSocketComm* comm = (struct ncclNetSocketComm*)opaqueComm;
  // 检查通信上下文是否有效
  if (comm) {
    // 遍历所有辅助线程
    for (int i=0; i<comm->nThreads; i++) {
      // 获取线程资源指针
      struct ncclNetSocketThreadResources* res = comm->threadResources+i;
      // 检查线程是否存在
      if (comm->helperThread[i]) {
        // 设置停止标志
        {
          // 加锁保护共享资源
          std::lock_guard<std::mutex> lock(res->threadMutex);
          // 设置停止标志为 1
          res->stop = 1;
          // 唤醒线程（让它检查停止标志）
          res->threadCond.notify_one();
        }
        // 等待线程退出
        PTHREADCHECK(pthread_join(comm->helperThread[i], NULL), "pthread_join");
      }
      // 释放任务队列内存
      free(res->threadTaskQueue.tasks);
    }
    // 声明变量：Socket 就绪标志
    int ready;
    // 检查控制 Socket 是否就绪
    NCCLCHECK(ncclSocketReady(&comm->ctrlSock, &ready));
    // 如果就绪，关闭控制 Socket
    if (ready) NCCLCHECK(ncclSocketClose(&comm->ctrlSock));
    // 遍历所有数据 Socket
    for (int i=0; i<comm->nSocks; i++) {
      // 检查数据 Socket 是否就绪
      NCCLCHECK(ncclSocketReady(&comm->socks[i], &ready));
      // 如果就绪，关闭数据 Socket
      if (ready) NCCLCHECK(ncclSocketClose(&comm->socks[i]));
    }
    // 释放内联数据缓冲区（如果已分配）
    if(comm->inlineData) free(comm->inlineData);
    // 释放通信上下文对象（使用 delete 因为是用 new 创建的）
    delete comm;
  }
  // 返回成功状态码
  return ncclSuccess;
}

// 函数功能：终结 Socket 网络插件
// 参数说明：
//   - ctx: 插件上下文（未使用）
// 返回值：ncclSuccess 表示成功
ncclResult_t ncclNetSocketFinalize(void* ctx) {
  // 减少引用计数
  netRefCount--;
  // 返回成功状态码
  return ncclSuccess;
}

// 定义 Socket 网络插件接口结构体
// 这个结构体定义了 Socket 网络插件的所有函数指针，NCCL 核心通过这些函数调用插件功能
ncclNet_t ncclNetSocket = {
  // 插件名称："Socket"
  "Socket",
  // 初始化函数指针
  ncclNetSocketInit,
  // 获取设备数量函数指针
  ncclNetSocketDevices,
  // 获取设备属性函数指针
  ncclNetSocketGetProperties,
  // 监听连接函数指针
  ncclNetSocketListen,
  // 连接到对等体函数指针
  ncclNetSocketConnect,
  // 接受连接函数指针
  ncclNetSocketAccept,
  // 注册内存函数指针
  ncclNetSocketRegMr,
  // DMA-BUF 支持函数指针（Socket 不支持 DMA-BUF）
  NULL, // No DMA-BUF support
  // 注销内存函数指针
  ncclNetSocketDeregMr,
  // 非阻塞发送函数指针
  ncclNetSocketIsend,
  // 非阻塞接收函数指针
  ncclNetSocketIrecv,
  // 非阻塞刷新函数指针
  ncclNetSocketIflush,
  // 测试请求完成函数指针
  ncclNetSocketTest,
  // 关闭发送连接函数指针
  ncclNetSocketClose,
  // 关闭接收连接函数指针（与发送连接使用相同的函数）
  ncclNetSocketClose,
  // 关闭监听函数指针
  ncclNetSocketCloseListen,
  // 获取设备内存句柄函数指针（Socket 不支持）
  NULL /* getDeviceMr */,
  // 接收消耗函数指针（Socket 不支持）
  NULL /* irecvConsumed */,
  // 合并设备函数指针（Socket 不支持）
  NULL /* mergeDevices */,
  // 终结插件函数指针
  ncclNetSocketFinalize,
  // 设置网络属性函数指针（Socket 不支持）
  NULL /* setNetAttr */,
};
