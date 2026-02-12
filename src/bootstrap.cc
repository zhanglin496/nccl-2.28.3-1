/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

/*
 * bootstrap.cc - NCCL Bootstrap 网络初始化模块
 *
 * Bootstrap 是 NCCL 的初始化阶段，用于在各个进程间建立初始连接，
 * 交换必要的信息（如 rank 地址、拓扑信息等），为后续的 P2P 通信
 * 和集合操作做准备。
 *
 * 主要功能：
 * 1. 网络接口选择和初始化
 * 2. Root 节点创建和连接收集
 * 3. Ring 连接建立（每个 rank 连接到前一个和后一个 rank）
 * 4. AllGather 操作（交换所有 rank 的地址信息）
 * 5. 节点内屏障和广播
 */

#include "nccl.h"
#include "core.h"
#include "utils.h"
#include "bootstrap.h"
#include "net.h"
#include <unistd.h>
#include <sys/types.h>
#include "proxy.h"
#include "param.h"
#include "ras.h"
#include <mutex>

// Bootstrap 操作的标签位定义
// 这些标签用于区分不同类型的 Bootstrap 消息
#define BOOTSTRAP_N_CHECK_ABORT           10000  // 每隔多少次循环检查一次 abort 标志
#define BOOTSTRAP_TAG_CONNECT             (0x1 << 31)  // 连接操作的标签
#define BOOTSTRAP_TAG_ALLGATHER           (0x1 << 30)  // AllGather 操作的标签
#define BOOTSTRAP_TAG_COMMSPLIT           (0x1 << 29)  // CommSplit 操作的标签
#define BOOTSTRAP_TAG_INTRANODE_ALLGATHER (0x1 << 28)  // 节点内 AllGather 的标签

// 计时器索引，用于统计 Bootstrap 各阶段的时间
#define BOOTSTRAP_INIT_TIME_CREATE 0  // 创建监听套接字的时间
#define BOOTSTRAP_INIT_TIME_SEND   1  // 发送信息到 root 的时间
#define BOOTSTRAP_INIT_TIME_RECV   2  // 从 root 接收信息的时间
#define BOOTSTRAP_INIT_TIME_RING   3  // Ring AllGather 的时间
#define BOOTSTRAP_INIT_TIME_TOTAL  4  // 总时间
#define BOOTSTRAP_INIT_TIME_DELAY  5  // 延迟时间（用于错峰连接）
#define BOOTSTRAP_INIT_TIME_N      6  // 计时器数量

// Root 线程的计时器索引
#define BOOTSTRAP_INIT_ROOT_WAIT   0  // 等待所有 rank 连接的时间
#define BOOTSTRAP_INIT_ROOT_SEND   1  // 发送连接信息的时间
#define BOOTSTRAP_INIT_ROOT_RECV   2  // 接收连接信息的时间
#define BOOTSTRAP_INIT_ROOT_N      3  // 计时器数量

// 性能测量宏 - 开始计时
#define BOOTSTRAP_PROF_OPEN(time) \
  do {                            \
    time = clockNano();           \
  } while (0)

// 性能测量宏 - 结束计时并计算耗时
#define BOOTSTRAP_PROF_CLOSE(time) \
  do {                             \
    time = clockNano() - time;     \
  } while (0)

// PID 计算宏，用于周期性地计算 rank 在 root 中的位置
#define BOOTSTRAP_PID(i, n) (((i) + (n)) % (n))

// ============================================================================
// Root-Rank 映射辅助函数
// 这些函数用于管理 rank 到 root 节点的映射关系
// ============================================================================

// 返回与 root 关联的第一个 rank
// 必须满足 root >= 0
// 如果 root >= n_roots，不会假设周期性（即不会循环）
//
// 参数：
//   root: root 节点的索引
//   n_ranks: 总 rank 数量
//   nRoots: root 节点数量
// 返回：属于该 root 的第一个 rank 的全局 rank 号
static int firstRankFromRoot(int root, int n_ranks, int nRoots) {
  // 计算公式：每个 root 分配 n_ranks/nRoots 个 rank
  // 前面的 (n_ranks % nRoots) 个 root 多分配一个 rank
  return root * (n_ranks / nRoots) + std::min(root, n_ranks % nRoots);
}

// 返回 rank 所属的 root ID
// 必须满足 rank >= 0
// 如果 rank >= n_ranks，不会假设周期性
//
// 参数：
//   rank: 全局 rank 号
//   nRanks: 总 rank 数量
//   nRoots: root 节点数量
// 返回：该 rank 所属的 root 节点的索引
static int rootIdFromRank(int rank, int nRanks, int nRoots) {
  int rmr = nRanks % nRoots; // rank mod root - 余数
  int rpr = nRanks / nRoots; // rank per root - 每个 root 的 rank 数
  int D = rmr * (rpr + 1);   // 分界点：前 D 个 rank 属于特殊的 root

  // 如果 rank < D，说明 rank 在前面的"特殊" root 中
  if (rank < D)
    return rank / (rpr + 1);
  else
    // 否则 rank 在后面的"普通" root 中
    return (rank - D) / rpr + rmr;
}

// 返回 root 管理的 rank 数量
// root 会被周期化处理（即可以超过 nRoots）
//
// 参数：
//   root: root 节点索引（会被周期化）
//   nRanks: 总 rank 数量
//   nRoots: root 节点数量
// 返回：该 root 管理的 rank 数量
static int nRankFromRoot(int root, int nRanks, int nRoots) {
  int ir = BOOTSTRAP_PID(root, nRoots);  // 周期化 root 索引
  int rmr = nRanks % nRoots;              // 余数
  int rpr = nRanks / nRoots;              // 每个 root 的基础 rank 数
  // 前面的 root 多分配一个 rank
  return rpr + ((ir < rmr) ? 1 : 0);
}

// 返回给定 rank 在给定 root 中的本地 ID
// root 会被周期化，rank 不会
//
// 参数：
//   rank: 全局 rank 号
//   root: root 节点索引（会被周期化）
//   nRanks: 总 rank 数量
//   nRoots: root 节点数量
// 返回：该 rank 在该 root 中的本地索引
static int localIdFromRoot(int rank, int root, int nRanks, int nRoots) {
  int ir = BOOTSTRAP_PID(root, nRoots);  // 周期化 root 索引
  // 计算本地 ID = rank - 该 root 的第一个 rank
  return rank - firstRankFromRoot(ir, nRanks, nRoots);
}

// 检查给定 rank 是否是给定 root 的第一个 rank
//
// 参数：
//   rank: 全局 rank 号
//   root: root 节点索引
//   nRanks: 总 rank 数量
//   nRoots: root 节点数量
// 返回：如果是返回 1，否则返回 0
static int isFirstFromRoot(int rank, int root, int nRanks, int nRoots) {
  return (rank == firstRankFromRoot(root, nRanks, nRoots));
}

// ============================================================================
// Root 线程参数结构
// ============================================================================
struct bootstrapRootArgs {
  struct ncclSocket* listenSock;  // Root 监听的套接字
  uint64_t magic;                  // 魔术数，用于验证连接
};

// ============================================================================
// 网络初始化相关变量
// ============================================================================
/* Init functions */
static char bootstrapNetIfName[MAX_IF_NAME_SIZE+1];      // 选定的网络接口名称
static union ncclSocketAddress bootstrapNetIfAddr;       // 选定的网络接口地址
static int bootstrapNetInitDone = 0;                     // 初始化完成标志
static std::mutex bootstrapNetMutex;                     // 保护初始化的互斥锁

NCCL_PARAM(BootstrapNetEnable,"OOB_NET_ENABLE", 0);    // 是否启用网络插件（默认关闭）

// ============================================================================
// bootstrapNetInit - 初始化 Bootstrap 网络
// 功能：选择合适的网络接口用于 Bootstrap 通信
// ============================================================================
ncclResult_t bootstrapNetInit() {
    // 找到符合要求的接口，接口名称和地址保存到 bootstrapNetIfName，bootstrapNetIfAddr
  if (bootstrapNetInitDone == 0) {
    std::lock_guard<std::mutex> lock(bootstrapNetMutex);
    if (bootstrapNetInitDone == 0) {
        // NCCL_COMM_ID 环境变量提供了一种替代 ncclGetUniqueId() 函数的方法，
        // 用于在多机环境下初始化 NCCL 通信。它允许用户直接设置一个预定义的、所有进程都知道的标识符，
        // 而不是通过编程方式生成和广播一个唯一的 ID。
        // 这样每个节点上的进程都获取了相同的 ID
        // 环境变量设置的 remoteAddr 地址，比如：192.168.1.100:12345
      const char* env = ncclGetEnv("NCCL_COMM_ID");
      int nIfs = 0;
      if (env) {
        // 用户通过环境变量 NCCL_COMM_ID 指定了地址
        union ncclSocketAddress remoteAddr;
        // 通过环境变量解析监听地址
        if (ncclSocketGetAddrFromString(&remoteAddr, env) != ncclSuccess) {
          WARN("Invalid NCCL_COMM_ID, please use format: <ipv4>:<port> or [<ipv6>]:<port> or <hostname>:<port>");
          return ncclInvalidArgument;
        }
        // 获取系统所有接口的 IP 地址，找到一个和 remoteAddr 在同一个子网的接口的本地 IP 地址
        // 拷贝接口名字和接口地址
        // 所有节点上都设置了相同的 NCCL_COMM_ID，这里实现要求节点的 IP 和 NCCL_COMM_ID 是在同一个子网内的
        // 否则无法通信报错？为啥不能走默认路由三层转发？
        // 接口名称和 IP 地址保存在 bootstrapNetIfName，bootstrapNetIfAddr 中
        NCCLCHECK(ncclFindInterfaceMatchSubnet(bootstrapNetIfName, &bootstrapNetIfAddr, &remoteAddr, MAX_IF_NAME_SIZE,
                                               &nIfs));
        // 如果没找到，报错
        if (nIfs <= 0) {
          WARN("NET/Socket : No usable listening interface found");
          return ncclSystemError;
        }
      } else {
        // 没有设置环境变量，根据设置条件找到一个可用的接口和地址
        NCCLCHECK(ncclFindInterfaces(bootstrapNetIfName, &bootstrapNetIfAddr, MAX_IF_NAME_SIZE, 1, &nIfs));
        if (nIfs <= 0) {
          WARN("Bootstrap : no socket interface found");
          return ncclInvalidUsage;
        }
      }
      char line[SOCKET_NAME_MAXLEN+MAX_IF_NAME_SIZE+2];
      snprintf(line, sizeof(line), " %s:", bootstrapNetIfName);
      ncclSocketToString(&bootstrapNetIfAddr, line+strlen(line));
      INFO(NCCL_BOOTSTRAP, "Bootstrap: Using%s", line);
      // 标记初始化完成
      bootstrapNetInitDone = 1;
    }
  }
  return ncclSuccess;
}

/* Socket Interface Selection type */
enum bootstrapInterface_t { findSubnetIf = -1, dontCareIf = -2 };

// ============================================================================
// checkAbort - 检查 abort 标志
// 在循环中定期检查是否需要中止操作
// ============================================================================
// check abort function
static ncclResult_t checkAbort(volatile uint32_t* flag, int* cntr) {
  // 每隔 BOOTSTRAP_N_CHECK_ABORT 次循环检查一次
  if ((*cntr % BOOTSTRAP_N_CHECK_ABORT) == 0) {
    if (flag && __atomic_load_n(flag, __ATOMIC_ACQUIRE)) {
      TRACE(NCCL_BOOTSTRAP, "bootstrap: abort called");
      return ncclInternalError;
    }
  }
  
  *cntr = (*cntr + 1) % BOOTSTRAP_N_CHECK_ABORT;
  return ncclSuccess;
}

// ============================================================================
// 网络发送/接收辅助函数
// ============================================================================

// 注册内存到网络插件
static ncclResult_t netReg(ncclNet_t* net, void* comm, void* data, int size, void** handle) {
  NCCLCHECK(net->regMr(comm, data, size, NCCL_PTR_HOST, handle));
  return ncclSuccess;
}

// 从网络插件注销内存
static ncclResult_t netDereg(ncclNet_t* net, void* comm, void** handle) {
  NCCLCHECK(net->deregMr(comm, *handle));
  *handle = NULL;
  return ncclSuccess;
}

// 非阻塞发送函数
static ncclResult_t netIsend(ncclNet_t* net, void* sendComm, void* data, int size, void* dataHandle, int tag, void** sendReq,
                             int* done) {
  if (*done) return ncclSuccess;  // 如果已经完成，直接返回
  if (!*sendReq) {
    // 首次调用，发起发送请求
    NCCLCHECK(net->isend(sendComm, data, (size_t)size, tag, dataHandle, NULL, sendReq));
  }
  if (*sendReq) {
    // 测试发送是否完成
    NCCLCHECK(net->test(*sendReq, done, NULL));
    if (*done) {
      *sendReq = NULL;
    }
  }
  return ncclSuccess;
}

// 非阻塞接收函数
static ncclResult_t netIrecv(ncclNet_t* net, void* recvComm, void* data, int size, void* dataHandle, int tag, void** recvReq,
                             int* done) {
  if (*done)
    return ncclSuccess;  // 如果已经完成，直接返回

  if (!*recvReq) {
    // 首次调用，发起接收请求
    size_t size64 = size;
    NCCLCHECK(net->irecv(recvComm, 1, &data, &size64, &tag, &dataHandle, NULL, recvReq));
  }

  if (*recvReq) {
    // 测试接收是否完成
    NCCLCHECK(net->test(*recvReq, done, NULL));
    if (*done) {
      *recvReq = NULL;
    }
  }
  
  return ncclSuccess;
}

// 发送并接收（双向操作）
static ncclResult_t netSendRecv(ncclNet_t* net, void* sendComm, void* sendData, int sendSize, void* sendDataHandle, void* recvComm,
                                void* recvData, int recvSize, void* recvDataHandle, int tag, volatile uint32_t* abortFlag) {
  int abortCounter = 0;
  int doneSend = 0, doneRecv = 0;
  void *sendReq = NULL, *recvReq = NULL;
  do {
    NCCLCHECK(checkAbort(abortFlag, &abortCounter));
    if (!doneRecv) {
      NCCLCHECK(netIrecv(net, recvComm, recvData, recvSize, recvDataHandle, tag, &recvReq, &doneRecv));
    }
    if (!doneSend) {
      NCCLCHECK(netIsend(net, sendComm, sendData, sendSize, sendDataHandle, tag, &sendReq, &doneSend));
    }
  } while (!doneSend || !doneRecv);
  return ncclSuccess;
}

// ============================================================================
// Socket 发送/接收辅助函数（先发送消息大小，再发送消息本身）
// ============================================================================

// Socket 发送：先发送消息大小，再发送消息内容
static ncclResult_t socketSend(struct ncclSocket* sock, void* data, int size) {
  NCCLCHECK(ncclSocketSend(sock, &size, sizeof(int)));
  if (size > 0)
    NCCLCHECK(ncclSocketSend(sock, data, size));
  return ncclSuccess;
}

// Socket 接收：先接收消息大小，再接收消息内容
static ncclResult_t socketRecv(struct ncclSocket* sock, void* data, int size) {
  int recvSize;
  NCCLCHECK(ncclSocketRecv(sock, &recvSize, sizeof(int)));
  if (recvSize > size) {
    WARN("Message truncated : received %d bytes instead of %d", recvSize, size);
    return ncclInternalError;
  }
  int actualSize = std::min(recvSize, size);
  if (actualSize > 0)
    NCCLCHECK(ncclSocketRecv(sock, data, actualSize));
  return ncclSuccess;
}

// Socket 双向操作：同时发送和接收
static ncclResult_t socketSendRecv(struct ncclSocket* sendSock, void* sendData, int sendSize, struct ncclSocket* recvSock,
                                   void* recvData, int recvSize) {
  int senderRecvSize;
  // 交换消息大小
  NCCLCHECK(ncclSocketSendRecv(sendSock, &sendSize, sizeof(int), recvSock, &senderRecvSize, sizeof(int)));
  if (senderRecvSize > recvSize) {
    WARN("Message truncated : received %d bytes instead of %d", senderRecvSize, recvSize);
    return ncclInternalError;
  }
  // 交换消息内容
  NCCLCHECK(ncclSocketSendRecv(sendSock, sendData, sendSize, recvSock, recvData, std::min(recvSize, senderRecvSize)));
  return ncclSuccess;
}

// ============================================================================
// Ring 连接信息结构
// ============================================================================
union ringConnectInfo {
  union ncclSocketAddress addr;   // Socket 地址
  char handle[NCCL_NET_HANDLE_MAXSIZE];  // 网络句柄
};

// ============================================================================
// 扩展信息结构（发送给 root 的信息）
// ============================================================================
struct extInfo {
    // 本进程的 rank 号，当前发起通信或连接的进程的 rank
  int rank;                                  // rank of the process reaching out
  // nranks 表示参与通信的总进程数量
  int nranks;                                // total number of ranks
  // 当前根索引
  int iroot;                                 // current root index
  // 总根节点数
  int nroots;                                // total number of roots
  // 本进程的监听地址信息，用于各个 rank 等待 rank0 来连接
  union ncclSocketAddress listenRootAddress; // address of my listenSocket for the root
  // 等待 ring 连接的监听地址信息，建立环形连接
  union ringConnectInfo connectInfo;
};

// 宏定义：计算网络句柄在数组中的位置
#define NET_HANDLE(h, rank)    ((h) + (rank * NCCL_NET_HANDLE_MAXSIZE))
// 宏定义：获取第 i 个 bootstrap handle
#define BOOTSTRAP_HANDLE(h, i) ((struct ncclBootstrapHandle*)((char*)h + i * NCCL_UNIQUE_ID_BYTES))

#include <sys/resource.h>

// ============================================================================
// setFilesLimit - 设置文件描述符限制
// 增加进程允许打开的最大文件数，以支持大量连接
// ============================================================================
static ncclResult_t setFilesLimit() {
  struct rlimit filesLimit;
  SYSCHECK(getrlimit(RLIMIT_NOFILE, &filesLimit), "getrlimit");
  filesLimit.rlim_cur = filesLimit.rlim_max;  // 设置为最大值
  SYSCHECK(setrlimit(RLIMIT_NOFILE, &filesLimit), "setrlimit");
  return ncclSuccess;
}

// ============================================================================
// rootSend - Root 向 rank 发送 ring 连接信息
// ============================================================================
static ncclResult_t rootSend(union ncclSocketAddress* addr, uint64_t magic, union ringConnectInfo* info) {
  ncclResult_t res = ncclSuccess;
  struct ncclSocket sock;
  NCCLCHECKGOTO(ncclSocketInit(&sock, addr, magic, ncclSocketTypeBootstrap), res, fail);
  NCCLCHECKGOTO(ncclSocketConnect(&sock), res, fail);
  NCCLCHECKGOTO(socketSend(&sock, info, sizeof(union ringConnectInfo)), res, fail);
  NCCLCHECKGOTO(ncclSocketClose(&sock));
  return res;
fail:
  (void)ncclSocketClose(&sock);
  return res;
}

// ============================================================================
// bootstrapRoot - Root 线程的主函数
// Root 线程负责收集所有 rank 的连接信息，并协调建立 ring 连接
// ============================================================================
static void* bootstrapRoot(void* rargs) {
  uint64_t timers[BOOTSTRAP_INIT_ROOT_N] = {0};
  struct bootstrapRootArgs* args = (struct bootstrapRootArgs*)rargs;
  // rank0 监听套接字
  struct ncclSocket* listenSock = args->listenSock;
  uint64_t magic = args->magic;
  ncclResult_t res = ncclSuccess;
  int nranks = 0, c = 0;
  int iroot = 0, nroots = 0, localId = 0;
  int nrecv = 0, n2send = 0;
  // 存储从其他节点接收的扩展信息
  struct extInfo info;
  // 存储 ring 地址信息
  union ringConnectInfo* rankInfo = NULL;
  // 存储监听地址信息，用于 rank0 连接
  union ncclSocketAddress* rankAddressesRoot = NULL; // for initial rank <-> root information exchange
  // 获取零值用于比较
  char zeroHandle[NCCL_NET_HANDLE_MAXSIZE];
  union ncclSocketAddress zeroAddress;
  union ringConnectInfo zeroInfo;

  // 零值结构体（用于初始化检查）
  memset(&zeroAddress, 0, sizeof(union ncclSocketAddress));
  memset(&zeroHandle, 0, NCCL_NET_HANDLE_MAXSIZE);
  memset(&zeroInfo, 0, sizeof(union ringConnectInfo));

  // 设置文件描述限制
  setFilesLimit();

  TRACE(NCCL_BOOTSTRAP, "BEGIN");
  BOOTSTRAP_PROF_OPEN(timers[BOOTSTRAP_INIT_ROOT_WAIT]);

  /* Receive addresses from all ranks */
  // 接收所有 rank 的地址信息
  do {
    struct ncclSocket sock;
    // 创建套接字，接收组内rank同步的地址信息
    NCCLCHECKGOTO(ncclSocketInit(&sock), res, out);
    NCCLCHECKGOTO(ncclSocketAccept(&sock, listenSock), res, out);
    NCCLCHECKGOTO(socketRecv(&sock, &info, sizeof(info)), res, out);
    //关闭连接
    NCCLCHECKGOTO(ncclSocketClose(&sock), res, out);

    // 首次接收，保存初始信息
    if (c == 0) {
      BOOTSTRAP_PROF_CLOSE(timers[BOOTSTRAP_INIT_ROOT_WAIT]);
      BOOTSTRAP_PROF_OPEN(timers[BOOTSTRAP_INIT_ROOT_RECV]);
      // 保存第一次收到的信息用于后续检验同一个组内都是相同的配置
      nranks = info.nranks;
      // 在有多个 root 时，收到的 rank 属于那个 root
      iroot = info.iroot;
      // 有多少个 root
      nroots = info.nroots;

      // if the number of root > 1, we will receive one extra info from the first local_id of the next root
      // 当 nroots==1 时，始终返回 nranks 的值
      // 计算需要发送/接收的节点数（nroots=1 时无需额外处理）
      //该root下有多少个rank
      n2send = nRankFromRoot(iroot, nranks, nroots);
      // 如果 root 大于 1，nrecv需要+1，因为需要存储多root情况下相邻的 rank 地址
      nrecv = n2send + ((nroots > 1) ? 1 : 0);
      // 分配 nrecv 个 ringConnectInfo 和 ncclSocketAddress 用于保存接收到的地址
      // 初始值为 0
      //rankInfo存储ring地址
      NCCLCHECKGOTO(ncclCalloc(&rankInfo, nrecv), res, out);
      //rankAddressesRoot存储socket root地址
      //如果是多root，nrecv需要多分配一个
      NCCLCHECKGOTO(ncclCalloc(&rankAddressesRoot, nrecv), res, out);
    }

    // 同一个组内的信息不相同，报错退出
    if (nranks != info.nranks || nroots != info.nroots || iroot != info.iroot) {
      WARN("Bootstrap Root : mismatch in info from procs, nranks %d vs %d, nroots %d vs %d, iroot %d vs %d", nranks, info.nranks, nroots, info.nroots, iroot, info.iroot);
      goto out;
    }

    // 计算本地 localId 号，用于决定地址存放的位置
    // 如果只有一个 nroots
    // localId 就等于当前收到的 info.rank
    localId = localIdFromRoot(info.rank, iroot, nranks, nroots);

    // 检查是否已经初始化了
    // 如果不为 0，表示已经初始化了，报错退出
    if (memcmp(&zeroAddress, &rankAddressesRoot[localId], sizeof(union ncclSocketAddress)) != 0 ||
        memcmp(&zeroInfo, &rankInfo[localId], sizeof(union ringConnectInfo)) != 0) {
      WARN("Bootstrap Root : rank %d of %d ranks has already checked in", info.rank, nranks);
      goto out;
    }

    // 这里可以看出，连接是按 rank 号升序建立 ring 环
    // 下面 2 段发送代码，在老版本中没有这个逻辑，这是个优化，处理可以立即发送的情况
    // 感觉代码逻辑弄复杂了

    // 如果前一个进程已经 check in，就把新收到的 ring 句柄回环发送给它
    // 如果有多根，local_id = 0 的前一进程不归我管
    // 如果 prev > n2send，就不发送
    // if the previous has already checked in, send the newly received handle, if not save the handle for later
    // if we have more than 1 root, I do not own the previous of local_id = 0
    // if we have prev > n2send, we do not send anything
    int prev = (nroots > 1) ? (localId - 1) : BOOTSTRAP_PID(localId - 1, nrecv);

    // 比如 rank0 先收到了 rank 1 的地址信息，这里 prev 为 0，rankAddressesRoot[prev] 为 0
    if (prev >= 0 && prev < n2send && memcmp(&zeroAddress, &rankAddressesRoot[prev], sizeof(union ncclSocketAddress)) != 0) {
      NCCLCHECKGOTO(rootSend(&rankAddressesRoot[prev], magic, &info.connectInfo), res, out);
    } else {
      // 存储 ring 监听地址
      memcpy(&rankInfo[localId], &info.connectInfo, sizeof(union ringConnectInfo));
    }

    // 如果下一个进程已经 check in，就把新收到的 ring 句柄 回环发送给它
    // 对于 nroots >= 1，我总是拥有下一个连接的信息
    // 如果 local_id 不在 [0, n2send) 范围内，就不回答
    // if the next rank has checked in, send the newly received info, if not save the addr for later
    // for nroots >=1, I will always own the information of the next connection
    // if the local_id id must be [0 ; n2send[ otherwise we do not answer
    int next = BOOTSTRAP_PID(localId + 1, nrecv);

    // 比如 rank0 先收到了 rank 1 的地址信息，这里 next 为 1，rankInfo[next] 为 0
    if (localId >= 0 && localId < n2send && memcmp(&zeroInfo, &rankInfo[next], sizeof(union ringConnectInfo)) != 0) {
      // 发送 ring 地址给指定的 rank
      NCCLCHECKGOTO(rootSend(&info.listenRootAddress, magic, &rankInfo[next]), res, out);
    } else {
      // 存储 rank root 的监听地址
      memcpy(rankAddressesRoot + localId, &info.listenRootAddress, sizeof(union ncclSocketAddress));
    }
    // 处理即时可发送的信息，没发送的在下面代码中补发
    // 上面的 2 个步骤发送的都是通过 rank 的 root 地址发送 ring 监听地址

    // 统计接收到的地址信息
    ++c;
    TRACE(NCCL_BOOTSTRAP, "Received connect from rank %d total %d/%d", info.rank, c, nrecv);
  } while (c < nrecv);

  TRACE(NCCL_BOOTSTRAP, "COLLECTED ALL %d HANDLES", nrecv);
  BOOTSTRAP_PROF_CLOSE(timers[BOOTSTRAP_INIT_ROOT_RECV]);

  // 下面检查是否需要补发信息
  // send the remaining info to the ranks who haven't received anything
  BOOTSTRAP_PROF_OPEN(timers[BOOTSTRAP_INIT_ROOT_SEND]);

  // here we need to send info only to my own local process
  // 假设有 5 个 rank，向 0 发送 1 的 ring 地址，1 发送 2 的 ring 地址，按 rank 大小构成一个升序连接环
  // 0->1->2->3->4->0
  for (int r = 0; r < n2send; ++r) {
    // use nrecv to periodize: if 1 root, we will send the first one to the last one, if >1 roots we will send the additional one we have received
    int next = BOOTSTRAP_PID(r + 1, nrecv);

    // 向尚未接收信息的节点发送 ring 监听地址
    // 为什么上面的 while 循环发生了，还需要检查一下
    // 原因是 while 循环有可能没发送完
    // 已经发送过的，其记录地址的信息为 0，表示上面已经发送了
    // 没发送的，其地址非 0，这里要补发一下
    if (memcmp(&zeroAddress, &rankAddressesRoot[r], sizeof(union ncclSocketAddress)) != 0 &&
        memcmp(&zeroInfo, &rankInfo[next], sizeof(union ringConnectInfo)) != 0) {
      NCCLCHECKGOTO(rootSend(&rankAddressesRoot[r], magic, &rankInfo[next]), res, out);
    }
  }

  BOOTSTRAP_PROF_CLOSE(timers[BOOTSTRAP_INIT_ROOT_SEND]);
  TRACE(NCCL_BOOTSTRAP | NCCL_PROFILE, "Root timings (wait %f, recv %f, send %f)", timers[BOOTSTRAP_INIT_ROOT_WAIT] / 1e9, timers[BOOTSTRAP_INIT_ROOT_RECV] / 1e9, timers[BOOTSTRAP_INIT_ROOT_SEND] / 1e9);

out:
  // root 线程退出，关闭监听套接字
  if (listenSock != NULL) {
    (void)ncclSocketClose(listenSock);
    free(listenSock);
  }
  if (rankInfo)
    free(rankInfo);
  if (rankAddressesRoot)
    free(rankAddressesRoot);
  free(rargs);

  TRACE(NCCL_BOOTSTRAP, "DONE");
  return NULL;
}

// ============================================================================
// bootstrapCreateRoot - 创建 Root 线程并开始监听
// ============================================================================
ncclResult_t bootstrapCreateRoot(struct ncclBootstrapHandle* handle, bool idFromEnv) {
  ncclResult_t ret = ncclSuccess;
  struct ncclSocket* listenSock = NULL;
  struct bootstrapRootArgs* args = NULL;
  pthread_t thread;

  NCCLCHECK(ncclCalloc(&listenSock, 1));
  // 调用 socket 创建一个监听套接字
  NCCLCHECKGOTO(ncclSocketInit(listenSock, &handle->addr, handle->magic, ncclSocketTypeBootstrap, NULL, 0), ret, fail);
  NCCLCHECKGOTO(ncclSocketListen(listenSock), ret, fail);
  // 将 listenSock 的 addr 地址拷回到 handle
  // handle 包含了完整的 IP 地址和端口号
  NCCLCHECKGOTO(ncclSocketGetAddr(listenSock, &handle->addr), ret, fail);

  // 分配 args 结构体
  NCCLCHECKGOTO(ncclCalloc(&args, 1), ret, fail);
  args->listenSock = listenSock;
  args->magic = handle->magic;

  // 创建线程 bootstrapRoot, 把套接字传递给线程
  PTHREADCHECKGOTO(pthread_create(&thread, NULL, bootstrapRoot, (void*)args), "pthread_create", ret, fail);
  ncclSetThreadName(thread, "NCCL BootstrapR");
  PTHREADCHECKGOTO(pthread_detach(thread), "pthread_detach", ret, fail); // will not be pthread_join()'d

exit:
  return ret;
fail:
  if (listenSock) free(listenSock);
  if (args) free(args);
  goto exit;
}

// ============================================================================
// bootstrapGetUniqueId - 获取唯一的 Bootstrap ID
// 这是 NCCL 初始化的第一步，为后续通信建立唯一的标识符
// ============================================================================
ncclResult_t bootstrapGetUniqueId(struct ncclBootstrapHandle* handle) {
  memset(handle, 0, sizeof(ncclBootstrapHandle));

  const char* env = ncclGetEnv("NCCL_COMM_ID");
  if (env) {
    // 如果设置了 NCCL_COMM_ID，这里不会创建 socket
    // 实际创建监听套接字推迟到 ncclCommInitRankDev 函数中由 root 节点创建
    // 为什么不使用全局变量 bootstrapNetIfAddr，原因是 bootstrapNetIfAddr 是本机地址，而不是 rank0 的地址
    // 所以还需要从环境变量再一次获取
    // 推迟的原因是这里还不知道本进程是否是 rank0 进程
    // 而我们期望是在 rank0 上创建监听
    INFO(NCCL_ENV, "NCCL_COMM_ID set by environment to %s", env);
    // 转换成 handle 地址
    if (ncclSocketGetAddrFromString(&handle->addr, env) != ncclSuccess) {
      WARN("Invalid NCCL_COMM_ID, please use format: <ipv4>:<port> or [<ipv6>]:<port> or <hostname>:<port>");
      return ncclInvalidArgument;
    }
    // 使用固定的 magic 值
    handle->magic = NCCL_MAGIC;
  } else {
    // 没有设置 NCCL_COMM_ID，使用之前从系统获取的 bootstrapNetIfAddr
    // 注意，这里 addr 只有 IP 地址，没有端口号
    // bootstrapCreateRoot 中调用 bind 内核会选择一个端口号，再拷贝到 handle 中
    // 获取随机的 magic 值
    NCCLCHECK(getRandomData(&handle->magic, sizeof(handle->magic)));
    memcpy(&handle->addr, &bootstrapNetIfAddr, sizeof(union ncclSocketAddress));

    // 在 bootstrapNetIfAddr 上创建一个监听 socket，等待其他节点来连接
    // root 收集所有的交换信息，将收集到的信息下发给所有节点
    // bootstrapCreateRoot 返回后，包含了 port 信息
    NCCLCHECK(bootstrapCreateRoot(handle, false));
  }

  return ncclSuccess;
}

// ============================================================================
// 非预期连接队列结构
// ============================================================================
struct unexConn {
  int peer;           // 对端 rank
  int tag;            // 消息标签
  struct ncclSocket sock;  // Socket 连接
  struct unexConn* next;      // 链表下一个节点
};

// ============================================================================
// Bootstrap Ring 状态结构
// ============================================================================
struct bootstrapRing_t {
  union {
    // net 插件
    struct {
      void *sendComm, *recvComm;                    // 发送和接收通信上下文
      ncclNetDeviceHandle_t *sendDevHandle, *recvDevHandle;  // 设备句柄
    } net;
    // 传统 tcp 套接字
    struct {
      struct ncclSocket recv;  // 接收 socket
      struct ncclSocket send;  // 发送 socket
    } socket;
  };
};

// ============================================================================
// Bootstrap 监听状态结构
// ============================================================================
struct bootstrapListen_t {
  struct ncclSocket peerSocket; // socket for peers to contact me in P2P
  union {
    struct {
      int dev;                              // 设备 ID
      void* comm;                            // 通信上下文
      char handle[NCCL_NET_HANDLE_MAXSIZE];  // 网络句柄
    } net;
    struct ncclSocket socket; // socket to be used for the ring
  };
};

// ============================================================================
// Bootstrap 状态结构（全局状态）
// ============================================================================
struct bootstrapState {
  // prev 和 next peer rank 的 ring 地址信息保存在这里
  struct bootstrapRing_t ring;
  // 监听信息，用于等待接收 root 发送的 next peer ring 地址
  struct bootstrapListen_t listen;

  // 指向 net 插件
  ncclNet_t* net;
  // 8 字节的 UDS hash 值
  uint64_t* peerProxyAddressesUDS;
  // 保存所有 rank 的地址，指向数组的地址
  //代理线程监听地址
  union ncclSocketAddress* peerProxyAddresses;
  //p2p监听地址
  union ncclSocketAddress* peerP2pAddresses;

  // 非预期连接队列
  struct unexConn* unexpectedConnections;
  // CUDA 设备号
  int cudaDev;
  // 本 rank 号
  int rank;
  // 总 rank 数
  int nranks;
  uint64_t magic;
  volatile uint32_t* abortFlag;
};

//访问ring信息
#define STATE_RING(s, f) (s->ring.f)
//listen信息
#define STATE_LISTEN(s, f) (s->listen.f)

// ============================================================================
// 辅助函数
// ============================================================================

// 创建 TCP 监听套接字
// helper functions
// 创建 tcp 套接字
static ncclResult_t createListenSocket(struct ncclComm* comm, uint64_t magic, struct ncclSocket* socket, union ncclSocketAddress* addr,
                                       ncclSocketType type) {
  NCCLCHECK(ncclSocketInit(socket, &bootstrapNetIfAddr, magic, type, comm->abortFlag));
  //设置为监听状态
  NCCLCHECK(ncclSocketListen(socket));
  //把socket地址返回给addr
  NCCLCHECK(ncclSocketGetAddr(socket, addr));
  return ncclSuccess;
}

// 获取 UDS（Unix Domain Socket）哈希值
static ncclResult_t getUDS(uint64_t* peerUDS) {
  uint64_t randId;
  NCCLCHECK(getRandomData(&randId, sizeof(randId)));
  *peerUDS = getPidHash() + randId;
  return ncclSuccess;
}

#define MAX_OOB_DEVS 16

// 获取网络设备 ID
static ncclResult_t netGetDevice(int rank, struct ncclComm* comm, int* dev) {
  static int devOOB = -1;
  if (devOOB < 0) {
    std::lock_guard<std::mutex> lock(bootstrapNetMutex);
    if (devOOB < 0) {
          //设置了NCCL_OOB_NET_IFNAME环境变量，使用指定接口
      const char* userIfEnv = ncclGetEnv("NCCL_OOB_NET_IFNAME");
      if (userIfEnv && strlen(userIfEnv) > 0) {
        INFO(NCCL_BOOTSTRAP | NCCL_ENV, "NCCL_OOB_NET_IFNAME set to %s", userIfEnv);
        bool searchNot = userIfEnv && userIfEnv[0] == '^';
        if (searchNot) userIfEnv++;
        bool searchExact = userIfEnv && userIfEnv[0] == '=';
        if (searchExact) userIfEnv++;
        struct netIf userIfs[MAX_OOB_DEVS];
        int nUserIfs = parseStringList(userIfEnv, userIfs, MAX_OOB_DEVS);
        // loop over the device and return the first one matching
        int nDev = 0;
        NCCLCHECK(comm->ncclNet->devices(&nDev));
        int devId = 0;
        while (devId < nDev) {
          ncclNetProperties_t props;
          comm->ncclNet->getProperties(devId, &props);
          // check against user specified HCAs/ports
          if (matchIfList(props.name, props.port, userIfs, nUserIfs, searchExact) ^ searchNot) {
            // All plain physical devices have been initialized at this point
            devOOB = devId;
            break;
          }
          devId++;
        }
        if (devOOB == -1) {
          if (!searchNot)
            WARN("no device found matching %s%s, verify NCCL_OOB_NET_IFNAME", searchExact ? "exactly " : "", userIfEnv);
          else
            WARN("no device found after excluding %s%s, verify NCCL_OOB_NET_IFNAME", searchExact ? "exactly " : "", userIfEnv);
          return ncclInvalidArgument;
        }
      } else {
        // default choice is device 0
        devOOB = 0;
      }
      // display info on the chosen device
      ncclNetProperties_t props;
      ncclResult_t res = comm->ncclNet->getProperties(devOOB, &props);
      bool hasProp = res == ncclSuccess;
      INFO(NCCL_BOOTSTRAP, "Bootstrap: Using %s:%d", (hasProp) ? props.name : "N/A", (hasProp) ? props.port : -1);
    }
  }
  *dev = devOOB;
  return ncclSuccess;
}

// ============================================================================
// netRingConnect - 使用网络插件建立 Ring 连接
// ============================================================================
static ncclResult_t netRingConnect(void* ctx, ncclNet_t* net, struct bootstrapListen_t* listen, char peerHandle[NCCL_NET_HANDLE_MAXSIZE],
                                   void** sendComm, ncclNetDeviceHandle_t** sendDevHandle,
                                   void** recvComm, ncclNetDeviceHandle_t** recvDevHandle, volatile uint32_t* abortFlag) {

  int abortCounter = 0;
  do {
    NCCLCHECK(checkAbort(abortFlag, &abortCounter));
    if (!*sendComm)
      NCCLCHECK(net->connect(ctx, listen->net.dev, peerHandle, sendComm, sendDevHandle));
    if (!*recvComm)
      NCCLCHECK(net->accept(listen->net.comm, recvComm, recvDevHandle));
  } while (!*sendComm || !*recvComm);
  return ncclSuccess;
}

// ============================================================================
// socketRingConnect - 使用 Socket 建立 Ring 连接
// ============================================================================
static ncclResult_t socketRingConnect(ncclSocketAddress* addr, struct ncclSocket* sendSocket, struct ncclSocket* listenSock, struct ncclSocket* recvSocket, uint64_t magic, volatile uint32_t* abortFlag) {
  NCCLCHECK(ncclSocketInit(sendSocket, addr, magic, ncclSocketTypeBootstrap, abortFlag));
  NCCLCHECK(ncclSocketConnect(sendSocket));
  NCCLCHECK(ncclSocketInit(recvSocket));
  NCCLCHECK(ncclSocketAccept(recvSocket, listenSock));
  return ncclSuccess;
}

// ============================================================================
// ringAllInfo - 通过 Ring AllGather 同步所有 rank 的地址信息
// ============================================================================
static ncclResult_t ringAllInfo(struct ncclComm* comm, struct bootstrapState* state,
                                union ncclSocketAddress* peerAddresses,
                                union ncclSocketAddress* peerProxy, uint64_t* peerUDS,
                                struct rasRankInit* rasRanks) {
  ncclResult_t res = ncclSuccess;
  int rank = comm->rank;
  //所有rank
  int nRanks = comm->nRanks;

  // 需要同步的单个 rank 信息
  struct bootstrapRingData {
    union ncclSocketAddress peerAddress;
    union ncclSocketAddress peerProxy;
    uint64_t peerUDS;
    struct rasRankInit rasRank;
  }* ringData = NULL;

  // 分配内存
  NCCLCHECK(ncclCalloc(&ringData, nRanks));

  // pack
  // 填充本 rank 的地址信息
  if (peerAddresses)
    memcpy(&(ringData[rank].peerAddress), peerAddresses + rank, sizeof(union ncclSocketAddress));
  if (peerProxy)
    memcpy(&(ringData[rank].peerProxy), peerProxy + rank, sizeof(union ncclSocketAddress));
  if (peerUDS)
    memcpy(&(ringData[rank].peerUDS), peerUDS + rank, sizeof(uint64_t));
  if (rasRanks)
    memcpy(&(ringData[rank].rasRank), rasRanks + rank, sizeof(*rasRanks));

  // allgather
  // 调用 allgather 方法获取所有进程的 peerProxy 和 peerAddress 和 UDS 信息
  NCCLCHECKGOTO(bootstrapAllGather(state, ringData, sizeof(struct bootstrapRingData)), res, exit);

  // unpack
  // 回填信息
  for (int irank = 0; irank < nRanks; ++irank) {
    if (peerAddresss)
      memcpy(peerAddresss + irank, &(ringData[irank].peerAddress), sizeof(union ncclSocketAddress));
    if (peerProxy)
      memcpy(peerProxy + irank, &(ringData[irank].peerProxy), sizeof(union ncclSocketAddress));
    if (peerUDS)
      memcpy(peerUDS + irank, &(ringData[irank].peerUDS), sizeof(uint64_t));
    if (rasRanks)
      memcpy(rasRanks + irank, &(ringData[irank].rasRank), sizeof(*rasRanks));
  }

exit:
  free(ringData);
  return res;
}

// ============================================================================
// sendToRoot - 向 Root 发送连接信息
// ============================================================================
static ncclResult_t sendToRoot(struct ncclBootstrapHandle* handle, struct ncclComm* comm, struct extInfo* info) {
  ncclResult_t ret = ncclSuccess;
  struct ncclSocket sock;
  NCCLCHECK(ncclSocketInit(&sock, &handle->addr, handle->magic, ncclSocketTypeBootstrap, comm->abortFlag));
  NCCLCHECKGOTO(ncclSocketConnect(&sock), ret, fail);
  NCCLCHECKGOTO(socketSend(&sock, info, sizeof(struct extInfo)), ret, fail);
  NCCLCHECKGOTO(ncclSocketClose(&sock));
  return ret;
fail:
  (void)ncclSocketClose(&sock);
  return ret;
}

// ============================================================================
// 环境变量参数定义
// ============================================================================
NCCL_PARAM(StaggerRate, "UID_STAGGER_RATE", 7000);      // 错峰连接速率
NCCL_PARAM(StaggerThreshold, "UID_STAGGER_THRESHOLD", 256); // 错峰阈值
NCCL_PARAM(RasEnable, "RAS_ENABLE", 1);                // 是否启用 RAS

// ============================================================================
// bootstrapInit - Bootstrap 初始化主函数
// 这是每个 rank 调用的初始化函数
// ============================================================================
//
// 这个函数是 NCCL Bootstrap 阶段的核心，负责建立初始化通信连接
//
// 函数调用时机：
// - ncclCommInitRank：创建新的通信器时
// - 每个 rank 都会调用此函数
// - 在集体操作开始之前完成
//
// Bootstrap 阶段的目标：
// 1. 建立基本的点对点连接
// 2. 收集所有 rank 的地址信息
// 3. 初始化 Proxy 服务
// 4. 为后续的拓扑发现和算法选择做准备
//
// 主要功能：
// 1. 创建 bootstrap 状态结构
// 2. 创建监听 Socket（环形连接和 Root 连接）
// 3. 向 Root 节点注册并发送连接信息
// 4. 从 Root 节点获取下一个 rank 的连接信息
// 5. 建立环形连接
// 6. 收集所有 rank 的地址信息（AllGather）
// 7. 初始化 Proxy 服务和 RAS 子系统
//
// 参数说明：
// - nHandles：Root 节点的数量（handles 数组的长度）
// - handles：Root 节点的句柄数组（包含 IP 和端口信息）
//            每个 rank 通过这些 handles 连接到 Root 节点
// - comm：要初始化的通信器
//
// Root 节点的作用：
// - Root 节点是预先启动的辅助进程
// - 负责收集和分发 rank 的连接信息
// - 帮助 ranks 之间建立初始连接
// - 在大规模集群中，可以有多个 Root 节点分担负载
//
// 初始化流程：
// 1. Rank 创建监听 Socket
// 2. Rank 向 Root 发送自己的监听地址
// 3. Root 收集所有 ranks 的地址
// 4. Rank 从 Root 获取环形拓扑中下一个 rank 的地址
// 5. Rank 建立环形连接
// 6. 通过环形连接进行 AllGather，收集所有 ranks 的地址
//
// ============================================================================
ncclResult_t bootstrapInit(int nHandles, void* handles, struct ncclComm* comm) {
  // ============================================================
  // 变量声明
  // ============================================================
  // result：函数返回值，ncclSuccess 表示成功
  ncclResult_t result = ncclSuccess;
  // rank：当前 rank 在通信域中的编号（0 到 nranks-1）
  int rank = comm->rank;
  // nranks：通信域中的总 rank 数量
  int nranks = comm->nRanks;
  // state：Bootstrap 状态结构指针，包含所有 Bootstrap 相关信息
  struct bootstrapState* state;
  // proxySocket：Proxy 服务的 Socket 指针，用于异步操作
  struct ncclSocket* proxySocket;
  // sock：临时 Socket，用于接收 Root 返回的信息
  // listenSockRoot：监听 Root 连接的 Socket
  struct ncclSocket sock, listenSockRoot;
  // info：发送给 Root 的信息结构，包含当前 rank 的地址信息
  struct extInfo info = {0};
  // nextPeer：下一个 rank 的连接信息（从 Root 获取）
  union ringConnectInfo nextPeer;
  // performRasAddRanks：是否执行 RAS AddRanks 操作
  bool performRasAddRanks = true;
  // rasRanks：RAS rank 初始化信息数组
  struct rasRankInit* rasRanks = nullptr;

  // ============================================================
  // 性能计时器数组
  // ============================================================
  // 用于统计 Bootstrap 各阶段的耗时
  // BOOTSTRAP_INIT_TIME_*：TOTAL, CREATE, DELAY, SEND, RECV, RING
  // ============================================================

  uint64_t timers[BOOTSTRAP_INIT_TIME_N] = {0};

  // ============================================================
  // 创建并初始化 bootstrap 状态结构
  // ============================================================
  // bootstrapState 包含：rank 信息、网络连接状态、Peer 地址信息、Socket 句柄
  // comm->magic 和 state->magic：魔数，用于验证连接的合法性
  // 取自第一个 handle 的 magic
  // ============================================================

  // 分配并清零 bootstrapState 结构
  NCCLCHECK(ncclCalloc(&state, 1));
  // 设置当前 rank 号
  state->rank = rank;
  // 设置总 rank 数
  state->nranks = nranks;
  // 设置 CUDA 设备编号
  state->cudaDev = comm->cudaDev;
  // 设置中止标志指针
  state->abortFlag = comm->abortFlag;
  // 设置网络插件接口
  state->net = comm->ncclNet;
  // 记录 bootstrap 地址信息，将 state 保存到 comm 中
  comm->bootstrap = state;
  // 记录 rank0 的地址信息 hash 值
  // 从第一个 handle 中获取 magic，同时设置到 comm 和 state
  comm->magic = state->magic = BOOTSTRAP_HANDLE(handles, 0)->magic; // state and comm magic set to the first magic ID

  // 输出调试信息
  TRACE(NCCL_BOOTSTRAP, "rank %d nranks %d", rank, nranks);

  // ============================================================
  // 开始统计总耗时
  // ============================================================
  // BOOTSTRAP_PROF_OPEN：打开性能计时器，记录起始时间
  // BOOTSTRAP_INIT_TIME_TOTAL：总耗时计时器索引
  // ============================================================

  BOOTSTRAP_PROF_OPEN(timers[BOOTSTRAP_INIT_TIME_TOTAL]);

  // ============================================================
  // 填充发送给 Root 的信息
  // ============================================================
  // info 结构将发送给 Root，包含当前 rank 的地址信息
  // ============================================================

  // fill up the info
  // 总共多少个节点进程，包含所有的
  // 设置总 rank 数
  info.nranks = nranks;
  // 有多少个是 root 节点进程
  // 设置 Root 节点数量
  info.nroots = nHandles;

  // ============================================================
  // 准备环形连接信息
  // ============================================================
  // nextPeer：存储下一个 rank 的连接信息
  // memset：清零内存，确保干净的初始状态
  // ============================================================

  // get the ring connection info
  memset(&nextPeer, 0, sizeof(union ringConnectInfo));
  // 开始统计创建耗时
  BOOTSTRAP_PROF_OPEN(timers[BOOTSTRAP_INIT_TIME_CREATE]);

  // 是否让 bootstrap 也走 IB/RoCE 交换信息，默认关闭
  // 当设置了环境变量 NCCL_OOB_NET_ENABLE=1，走 ncclNet 连接方式，否则走普通 tcp 套接字
  if (ncclParamBootstrapNetEnable()) {
    // Create net interface for other ranks to contact me (all gather)
    NCCLCHECK(netGetDevice(rank, comm, &STATE_LISTEN(state, net.dev)));
    // 如果不支持 IB，这里也可能会走常规 tcp 套接字
    // ncclNetIb
    NCCLCHECK(state->net->listen(comm->netContext, STATE_LISTEN(state, net.dev), STATE_LISTEN(state, net.handle), &STATE_LISTEN(state, net.comm)));
    memcpy(info.connectInfo.handle, STATE_LISTEN(state, net.handle), NCCL_NET_HANDLE_MAXSIZE);
  } else {
    // create socket for ring neighbor to contact me
    // 使用 bootstrapNetIfAddr 创建一个监听套接字，用于等待 ring 邻居连接，建立环形连接
    // ring 地址记录到 connectInfo 中
    NCCLCHECK(createListenSocket(comm, comm->magic, &STATE_LISTEN(state, socket), &info.connectInfo.addr, ncclSocketTypeBootstrap));
  }

  // 告诉当前 rank：你应该归哪一台 bootstrap 根节点管理
  // rootIdFromRank = "rank → 根节点编号"
  // 输入 = 全局 rank + 总节点数 + 根节点数
  // 输出 = 当前 rank 所属的 bootstrap 根 ID
  // 如果 root 节点只有一个，这个 curr_root 始终返回 0
  // 把 rank 平均分配给多个 root，并且保证每个 root 的 rank 号集合都是连续的
  int curr_root = rootIdFromRank(rank, nranks, nHandles);

  // 再创建一次监听套接字，使用相同的 IP 和不同的端口，用于等待 root 连接
  // BOOTSTRAP_HANDLE 获取 root 的 handle
  // Create socket for root to contact me using the root's magic
  NCCLCHECK(createListenSocket(comm, BOOTSTRAP_HANDLE(handles, curr_root)->magic, &listenSockRoot, &info.listenRootAddress, ncclSocketTypeBootstrap));
  BOOTSTRAP_PROF_CLOSE(timers[BOOTSTRAP_INIT_TIME_CREATE]);

  // stagger connection times to avoid an overload of the root
  BOOTSTRAP_PROF_OPEN(timers[BOOTSTRAP_INIT_TIME_DELAY]);

  // 获取 curr_root 下管理多少个 rank
  int nRankRoot = nRankFromRoot(curr_root, nranks, nHandles);

  // 错峰建连速率限制
  // 延迟到 root 的连接，减轻 root 的通信负担
  // 当 rank 数量很大时，所有 rank 同时连接 root 会导致：
  // - Root 端 socket 队列溢出
  // - 连接失败率上升
  // NCCL_UID_STAGGER_THRESHOLD，默认 256
  // 超过阈值 256
  if (nRankRoot > ncclParamStaggerThreshold()) {
    // for socket the message rate in microsec
    // NCCL_UID_STAGGER_RATE，默认 7000 消息/s
    // 微秒转化成秒
    double msg_rate = ncclParamStaggerRate() / 1.0e6;
    // 0.007
    long musec = localIdFromRoot(rank, curr_root, nranks, nHandles) / msg_rate;

    // 具体来说，假设有 10000 个进程，那么：
    // 进程 0 延迟 0 微秒（立即连接）
    // 进程 1 延迟 142.857 微秒 //1/0.007=142.85us
    // 进程 2 延迟 285.714 微秒
    // ...
    // 进程 n 延迟 n * 142.857 微秒
    // 10000/(9999*142.857/1000/1000)=7000/s
    struct timespec tv;
    long c_1e6 = 1e6;
    tv.tv_sec = musec / c_1e6;
    tv.tv_nsec = 1e3 * (musec % c_1e6);
    // 睡眠，延迟连接 root
    TRACE(NCCL_BOOTSTRAP, "rank %d delaying connection to root by %ld microsec", rank, musec);
    (void)nanosleep(&tv, NULL);
  }
  BOOTSTRAP_PROF_CLOSE(timers[BOOTSTRAP_INIT_TIME_DELAY]);

  // send info on my listening socket to root
  BOOTSTRAP_PROF_OPEN(timers[BOOTSTRAP_INIT_TIME_SEND]);

  // send contact info to my own root
  info.rank = rank;
  info.iroot = curr_root;

  // 向 root 发送消息，把本 rank 的 info.listenRootAddress 和 ring 地址信息报告给 handles root
  // handles 包含了所有 root 的 IP 地址和端口信息
  NCCLCHECK(sendToRoot(BOOTSTRAP_HANDLE(handles, curr_root), comm, &info));

  // if needed, send the connection info to the previous root
  // 处理有多个 root 节点的情况
  // rank 是当前 root 集合的第一个 rank，需要同步地址信息给相邻的 root
  if (nHandles > 1 && isFirstFromRoot(rank, curr_root, nranks, nHandles)) {
    // 上一个 rank 号
    int prev_rank = BOOTSTRAP_PID(rank - 1, nranks);
    // 上一个 rank 号在那个 root 节点
    int prev_root = rootIdFromRank(prev_rank, nranks, nHandles);

//这里是+1，
    info.rank = prev_rank + 1; // my rank as seen by the previous root
    info.iroot = prev_root;
    // 同步地址信息给相邻的 root
    NCCLCHECK(sendToRoot(BOOTSTRAP_HANDLE(handles, prev_root), comm, &info));
  }
  BOOTSTRAP_PROF_CLOSE(timers[BOOTSTRAP_INIT_TIME_SEND]);

  // get info on my "next" rank in the bootstrap ring from root
  // 等待 root rank 连接当前 rank，并从 root 节点中拿到 next ring rank 建连地址，存储到 nextPeer
  // 注意这里是从 listenSockRoot 中获取 ring next 的地址信息，并不会新建一个套接字
  BOOTSTRAP_PROF_OPEN(timers[BOOTSTRAP_INIT_TIME_RECV]);

  // ============================================================
  // 从 Root 获取下一个 rank 的连接信息
  // ============================================================
  //
  // 这一步通过监听 Socket 接收 Root 返回的连接信息
  //
  // 工作流程：
  // 1. 初始化临时 Socket（sock）
  // 2. 等待 Root 连接到 listenSockRoot
  // 3. 从 Root 接收下一个 rank 的连接信息
  // 4. 关闭临时 Socket
  // 5. 关闭监听 Socket（已完成使命）
  //
  // 为什么使用临时 Socket？
  // - 只需要接收一次数据
  // - 接收完成后立即关闭
  // - 不需要保持连接
  //
  // ============================================================

  // 初始化临时 Socket
  NCCLCHECK(ncclSocketInit(&sock));
  // 等待 Root 连接
  // Root 会主动连接到 listenSockRoot
  NCCLCHECK(ncclSocketAccept(&sock, &listenSockRoot));
  // 从 Root 接收下一个 rank 的连接信息
  // nextPeer：包含下一个 rank 的监听地址
  NCCLCHECK(socketRecv(&sock, &nextPeer, sizeof(nextPeer)));
  // 关闭临时 Socket
  NCCLCHECK(ncclSocketClose(&sock));
  // 关闭 listenSockRoot 套接字, 释放资源
  // Root 连接已完成，不再需要监听
  NCCLCHECK(ncclSocketClose(&listenSockRoot));
  // 结束统计接收耗时
  BOOTSTRAP_PROF_CLOSE(timers[BOOTSTRAP_INIT_TIME_RECV]);

  // ============================================================
  // 建立环形连接
  // ============================================================
  //
  // 这一步建立环形拓扑中的实际连接
  //
  // 和 nextPeer 建立 ring 网络
  // accept and connect the ring network
  //
  // 连接方向：
  // - 发送连接：当前 rank → next
  // - 接收连接：prev → 当前 rank
  //
  // 两种模式：
  //
  // 1. Bootstrap Net 模式（使用网络插件）：
  //    - 使用 IB/RoCE 等高速网络
  //    - 支持 RDMA，性能更好
  //
  //    netRingConnect 参数：
  //    - comm->netContext：网络上下文
  //    - state->net：网络插件接口
  //    - &state->listen：监听对象
  //    - nextPeer.handle：下一个 rank 的监听句柄
  //    - 返回：sendComm, sendDevHandle, recvComm, recvDevHandle
  //
  // 2. Socket 模式（使用 TCP Socket）：
  //    - 使用传统的 TCP Socket
  //    - 兼容性好，适用于各种网络环境
  //
  //    socketRingConnect 参数：
  //    - &nextPeer.addr：下一个 rank 的 Socket 地址
  //    - &STATE_RING(state, socket.send)：发送 Socket（连接到 next）
  //    - &STATE_LISTEN(state, socket)：监听 Socket（等待 prev 连接）
  //    - &STATE_RING(state, socket.recv)：接收 Socket（prev 的连接）
  //    - comm->magic：魔数，用于验证连接
  //    - state->abortFlag：中止标志
  //
  // 连接建立的对称性：
  // - 当前 rank 连接到 next（作为发送方）
  // - prev 连接到当前 rank（作为接收方）
  // - 所有 rank 同时执行，形成完整的环形连接
  //
  // ============================================================

  if (ncclParamBootstrapNetEnable()) {
    // ============================================================
    // 使用网络插件模式建立环形连接
    // ============================================================
    // 使用 IB 或者其他网络插件连接
    // netRingConnect：建立环形网络连接
    NCCLCHECK(netRingConnect(comm->netContext, state->net, &state->listen, nextPeer.handle,
                             // 输出参数：发送通信对象和设备句柄
                             &STATE_RING(state, net.sendComm), &STATE_RING(state, net.sendDevHandle),
                             // 输出参数：接收通信对象和设备句柄
                             &STATE_RING(state, net.recvComm), &STATE_RING(state, net.recvDevHandle), state->abortFlag));
  } else {
    // ============================================================
    // 使用 Socket 模式建立环形连接
    // ============================================================
    // 否则使用普通的 tcp 套接字连接到下一个 peer 中，目的地址记录在 socket.send 中
    // prev_peer 记录在 socket.recv 中
    // socketRingConnect：建立环形 Socket 连接
    NCCLCHECK(socketRingConnect(&nextPeer.addr, &STATE_RING(state, socket.send), &STATE_LISTEN(state, socket), &STATE_RING(state, socket.recv), comm->magic, state->abortFlag));
  }

  // ============================================================
  // 收集所有 rank 的地址信息（AllGather）
  // ============================================================
  //
  // 这一步为 AllGather 做准备，创建各种监听 Socket
  //
  // AllGather all listen handlers
  // in case of failure, those resources will be free'd when calling bootstrapDestroy, so we can return immediatly
  //
  // 需要创建的 Socket：
  // 1. Proxy Socket：用于 Proxy 服务通信
  // 2. P2P Socket：用于点对点通信
  // 3. Proxy UDS：Unix Domain Socket（本地通信优化）
  //
  // ============================================================

  // ============================================================
  // 创建 Proxy 监听 Socket
  // ============================================================
  //
  // Proxy 是 NCCL 的异步执行引擎
  // Proxy Socket 用于其他 ranks 连接到当前 rank 的 Proxy
  //
  // 分配 nranks 个 peerProxyAddresses
  // peerProxyAddresses[i]：rank i 的 Proxy 监听地址
  NCCLCHECK(ncclCalloc(&state->peerProxyAddresses, nranks));
  // 分配 Proxy Socket 结构
  NCCLCHECK(ncclCalloc(&proxySocket, 1));
  // 创建一个 tcp proxySocket 监听套接字，记录本机监听地址到 peerProxyAddresses
  // ncclSocketTypeProxy：标识这是 Proxy Socket
  // state->peerProxyAddresses + rank：当前 rank 的 Proxy 地址位置
  NCCLCHECKGOTO(createListenSocket(comm, comm->magic, proxySocket, state->peerProxyAddresses + rank, ncclSocketTypeProxy), result, fail);

  // ============================================================
  // 创建 Proxy UDS（Unix Domain Socket）
  // ============================================================
  //
  // UDS（Unix Domain Socket）用于本地进程间通信
  // 比 TCP Socket 更高效，适用于同一节点内的 ranks
  //
  // 生成一个 8 字节的随机 uds 值
  // Uds：Unix Domain Socket
  //
  // peerProxyAddressesUDS：
  // - 存储所有 ranks 的 UDS 地址
  // - 同一节点内的 ranks 使用 UDS 通信
  // - 不同节点的 ranks 仍使用 TCP Socket
  //
  // ============================================================

  // 分配 UDS 地址数组
  NCCLCHECKGOTO(ncclCalloc(&state->peerProxyAddressesUDS, nranks), result, fail);
  // 生成当前 rank 的 UDS 地址
  NCCLCHECKGOTO(getUDS(state->peerProxyAddressesUDS + rank), result, fail);

  // ============================================================
  // 创建 P2P 监听 Socket
  // ============================================================
  //
  // P2P Socket 用于点对点通信
  // 允许任意两个 ranks 之间直接通信
  //
  // peerSocketAddress：
  // - 当前 rank 的 P2P 监听地址
  // - 需要通过 ringAllInfo 分发给所有其他 ranks
  //
  // ============================================================

  // create a socket for others to reach out (P2P)
  union ncclSocketAddress peerSocketAddress;
  // 创建一个 tcp peerSocketAddress 监听套接字，记录到 peerSocketAddress 中
  NCCLCHECKGOTO(createListenSocket(comm, comm->magic, &STATE_LISTEN(state, peerSocket), &peerSocketAddress, ncclSocketTypeBootstrap), result, fail);
  // 分配 nranks 个 peerP2pAddresses 地址
  // peerP2pAddresses[i]：rank i 的 P2P 监听地址
  NCCLCHECKGOTO(ncclCalloc(&state->peerP2pAddresses, nranks), result, fail);
  // 记录本机监听地址到 peerP2pAddresses
  // peerP2pAddresses + rank：当前 rank 的地址位置
  memcpy(state->peerP2pAddresses + rank, &peerSocketAddress, sizeof(union ncclSocketAddress));

  // ============================================================
  // 初始化 RAS（可靠性、可用性、可服务性）
  // ============================================================
  //
  // RAS 是 NCCL 的可靠性增强功能
  //
  // Initialize RAS
  // 可靠性、可用性和可服务性 (RAS) 子系统
  // RAS 子系统，可帮助用户诊断应用崩溃和挂起
  // 可在生产环境中用于在 NCCL 作业执行期间查询其运行状况
  //
  // ncclParamRasEnable()：
  // - 环境变量：NCCL_RAS_ENABLE
  // - 默认值：1（启用）
  // - 可以设置为 0 禁用
  //
  // rasRankInit 结构包含：
  // - addr：网络地址
  // - pid：进程 ID
  // - cudaDev：CUDA 设备编号
  // - nvmlDev：NVML 设备编号
  // - hostHash：主机哈希（用于识别节点）
  // - pidHash：进程哈希（用于识别进程）
  //
  // ============================================================

  if (ncclParamRasEnable() == 1) {
    // ============================================================
    // 分配并填充 RAS 信息
    // ============================================================
    // The RAS thread will take care of freeing the memory allocated below.
    // 分配 Ranks 信息数组
    NCCLCHECK(ncclCalloc(&rasRanks, nranks));
    // 拷贝网络地址
    memcpy(&rasRanks[rank].addr, &bootstrapNetIfAddr, sizeof(rasRanks[rank].addr));
    // 获取进程 ID
    rasRanks[rank].pid = getpid();
    // 设置 CUDA 设备编号
    rasRanks[rank].cudaDev = comm->cudaDev;
    // 设置 NVML 设备编号
    rasRanks[rank].nvmlDev = comm->nvmlDev;
    // 获取主机哈希（用于识别节点）
    rasRanks[rank].hostHash = getHostHash();
    // 获取进程哈希（用于识别进程）
    rasRanks[rank].pidHash = getPidHash();

    // ============================================================
    // 初始化 RAS 子系统
    // ============================================================
    // ncclRasCommInit：初始化 RAS 子系统
    // - 注册监控回调
    // - 启动健康检查线程
    //
    // 如果初始化失败：
    // - 打印警告信息，但继续执行
    // - RAS 不是必需的，失败不会导致 Bootstrap 失败
    // - performRasAddRanks 设为 false，跳过后续的 RAS 操作
    // ============================================================

    if (ncclRasCommInit(comm, rasRanks+rank) != ncclSuccess) {
      INFO(NCCL_INIT|NCCL_RAS, "Continuing in spite of a RAS initialization error");
      // We should still participate in the ringAllInfo below as the peers will be waiting for us.
      // Just make sure that the address is clearly invalid...
      // 清零当前 rank 的 RAS 信息（标记为无效）
      memset(rasRanks+rank, '\0', sizeof(*rasRanks));
      // 设置标志，跳过后续的 RAS AddRanks 操作
      performRasAddRanks = false;
    }
  }

  // ============================================================
  // 执行环形 AllGather
  // ============================================================
  //
  // 这一步通过环形连接收集所有 ranks 的地址信息
  //
  // ringAllInfo 的工作原理：
  // 1. 每个 rank 准备自己的地址信息
  // 2. 第一轮：rank 0 发送给 rank 1，rank 1 发送给 rank 2，...
  // 3. 每个 rank 接收并保存上一个 rank 的地址
  // 4. 第二轮：每个 rank 转发它收到的所有地址
  // 5. 经过 nranks 轮后，每个 rank 都有完整的地址列表
  //
  // 收集的数据：
  // - peerP2pAddresses：所有 ranks 的 P2P 地址
  // - peerProxyAddresses：所有 ranks 的 Proxy 地址
  // - peerProxyAddressesUDS：所有 ranks 的 UDS 地址
  // - rasRanks：所有 ranks 的 RAS 信息（可选）
  //
  // 完成后：
  // - 每个 rank 都可以与任意其他 rank 通信
  // - 不再局限于环形拓扑
  //
  // ============================================================

  // 开始统计环形 AllGather 耗时
  BOOTSTRAP_PROF_OPEN(timers[BOOTSTRAP_INIT_TIME_RING]);

  // 以 Ring 的方式实现 peerP2pAddresses 和 peerProxyAddresses 地址的 AllGather
  // 地址信息保存在 state 中
  // 这样，每个进程上就获得了所有进程的监听地址
  // 也就是说，每个进程都可以和其它任意进程通信
  // ringAllInfo：通过环形连接收集所有 ranks 的地址信息
  NCCLCHECKGOTO(ringAllInfo(comm, state, state->peerP2pAddresses, state->peerProxyAddresses, state->peerProxyAddressesUDS, rasRanks), result, fail);
  // 结束统计环形 AllGather 耗时
  BOOTSTRAP_PROF_CLOSE(timers[BOOTSTRAP_INIT_TIME_RING]);

  // ============================================================
  // 初始化 Proxy 服务
  // ============================================================
  //
  // Proxy 是 NCCL 的异步执行引擎
  //
  // Proxy 的作用：
  // 1. 异步执行网络操作
  // 2. 管理网络连接
  // 3. 处理内存注册和 DMA 操作
  // 4. 提供异步 API 支持
  //
  // ncclProxyInit：
  // - 创建 Proxy 线程
  // - 初始化 Proxy 的通信端点
  // - 注册回调函数
  //
  // proxySocket：
  // - Proxy 的监听 Socket
  // - 其他 ranks 可以通过这个 Socket 连接到 Proxy
  //
  // peerProxyAddresses：
  // - 所有 ranks 的 Proxy 地址
  // - Proxy 使用这些地址与其他 ranks 的 Proxy 通信
  //
  // peerProxyAddressesUDS：
  // - 同一节点内的 ranks 使用 UDS 通信
  // - 不同节点的 ranks 使用 TCP Socket
  //
  // ============================================================

  // Create the service proxy and get the UDS
  // 创建一个 unix 域套接字
  // ncclProxyInit：初始化 Proxy 服务
  NCCLCHECKGOTO(ncclProxyInit(comm, proxySocket, state->peerProxyAddresses, state->peerProxyAddressesUDS), result, fail);

  // ============================================================
  // 完成 RAS 初始化
  // ============================================================
  //
  // 如果 RAS 初始化成功，添加所有 ranks 到 RAS 系统
  //
  // ncclRasAddRanks：
  // - 将所有 ranks 的信息注册到 RAS
  // - RAS 开始监控所有 ranks
  // - 检测异常和故障
  //
  // ============================================================

  // 检查是否启用 RAS 且初始化成功
  if (ncclParamRasEnable() == 1 && performRasAddRanks) {
    // 添加所有 ranks 到 RAS 系统
    // 如果失败，打印警告但继续执行
    if (ncclRasAddRanks(rasRanks, nranks) != ncclSuccess)
      INFO(NCCL_INIT|NCCL_RAS, "Continuing in spite of a RAS initialization error");
  }

  // ============================================================
  // 结束计时并输出性能信息
  // ============================================================
  //
  // 记录总耗时并输出各阶段的耗时
  //
  // 输出的耗时信息：
  // - total：总耗时
  // - create：创建 Socket 的耗时
  // - send：向 Root 发送信息的耗时
  // - recv：从 Root 接收信息的耗时
  // - ring：环形 AllGather 的耗时
  // - delay：错峰延迟的耗时
  //
  // 单位：秒（纳秒 / 1e9）
  //
  // 用途：
  // - 性能分析
  // - 瓶颈诊断
  // - 优化初始化流程
  //
  // ============================================================

  // 结束统计总耗时
  BOOTSTRAP_PROF_CLOSE(timers[BOOTSTRAP_INIT_TIME_TOTAL]);
  // 输出完成信息
  TRACE(NCCL_BOOTSTRAP, "rank %d nranks %d - DONE", rank, nranks);
  // 输出性能计时信息（单位：秒）
  INFO(NCCL_BOOTSTRAP | NCCL_PROFILE, "Bootstrap timings total %f (create %f, send %f, recv %f, ring %f, delay %f)", timers[BOOTSTRAP_INIT_TIME_TOTAL] / 1e9,
       timers[BOOTSTRAP_INIT_TIME_CREATE] / 1e9,   // 创建 Socket 耗时
       timers[BOOTSTRAP_INIT_TIME_SEND] / 1e9,     // 向 Root 发送信息耗时
       timers[BOOTSTRAP_INIT_TIME_RECV] / 1e9,     // 从 Root 接收信息耗时
       timers[BOOTSTRAP_INIT_TIME_RING] / 1e9,     // 环形 AllGather 耗时
       timers[BOOTSTRAP_INIT_TIME_DELAY] / 1e9);    // 错峰延迟耗时

  // ============================================================
  // 正常退出
  // ============================================================
  // 返回成功状态码
  // ============================================================

exit:
  return result;

  // ============================================================
  // 错误处理
  // ============================================================
  //
  // 发生错误时：
  // - 释放 proxySocket（如果已分配）
  // - 跳转到 exit 标签返回错误码
  //
  // 注意：
  // - state 不会在这里释放
  // - comm->bootstrap 仍然指向 state
  // - 如果后续需要清理，会在 comm 销毁时进行
  //
  // ============================================================

fail:
  // 释放 Proxy Socket（如果已分配）
  free(proxySocket);
  // 跳转到 exit 标签
  goto exit;
}

// ============================================================================
// bootstrapSplit - 处理通信器分割的 Bootstrap 初始化
// ============================================================================
//
// 这个函数用于处理 ncclCommSplit 创建的子通信器的 Bootstrap 初始化
//
// 函数调用时机：
// - ncclCommSplit：从父通信器分割出子通信器
// - 子通信器需要建立自己的 Bootstrap 连接
//
// Bootstrap 是什么？
// - NCCL 的引导阶段，负责建立基本的通信连接
// - 在实际集体操作之前完成
// - 建立环形的点对点连接
//
// 通信器分割（Comm Split）：
// - 将一个通信域分割成多个子通信域
// - 例如：按颜色（color）分割，相同 color 的 rank 属于同一个子 comm
// - key 决定了子 comm 中的 rank 编号
//
// 主要功能：
// 1. 创建 bootstrap 状态结构
// 2. 计算子 comm 中的环形拓扑（前驱和后继）
// 3. 建立环形连接（使用父 comm 的现有连接）
// 4. 收集所有 peer 的地址信息
// 5. 初始化 Proxy 服务（如果不共享资源）
//
// 参数说明：
// - magic：魔数，用于验证连接的合法性
// - comm：新创建的子通信器
// - parent：父通信器
// - color：分割颜色，相同 color 的 rank 属于同一个子 comm
// - key：键值，决定在子 comm 中的 rank 编号
// - parentRanks：父 comm rank 到子 comm rank 的映射数组
//
// parentRanks 数组说明：
// - parentRanks[i] = j 表示父 comm 的 rank j 是子 comm 的 rank i
// - 例如：父 comm 有 8 个 rank [0-7]
//         子 comm A 包含父 comm 的 [0, 2, 4, 6]
//         parentRanks = [0, 2, 4, 6]
//         - 子 comm rank 0 是父 comm rank 0
//         - 子 comm rank 1 是父 comm rank 2
//         - 子 comm rank 2 是父 comm rank 4
//         - 子 comm rank 3 是父 comm rank 6
//
// ============================================================================
ncclResult_t bootstrapSplit(uint64_t magic, struct ncclComm* comm, struct ncclComm* parent, int color, int key, int* parentRanks) {
  ncclResult_t ret = ncclSuccess;                    // 返回值，默认成功
  int rank = comm->rank;                              // 当前 rank 在子 comm 中的编号
  int nranks = comm->nRanks;                          // 子 comm 中的总 rank 数
  int prev, next;                                     // 环形拓扑中的前驱和后继 rank
  union ringConnectInfo info;                         // 当前 rank 的连接信息（用于发送给其他 rank）
  union ringConnectInfo nextPeer;                     // 下一个 rank 的连接信息（从下一个 rank 接收）
  struct ncclSocket* proxySocket = NULL;              // Proxy 服务的 socket（如果不共享资源）
  struct bootstrapState* state;                       // Bootstrap 状态结构

  // ============================================================
  // 创建并初始化 bootstrap 状态结构
  // ============================================================
  //
  // bootstrapState 是 Bootstrap 阶段的核心数据结构
  // 包含：
  // - rank 信息
  // - 网络连接状态
  // - Peer 地址信息
  // - Socket 句柄
  //
  // ncclCalloc：分配并清零内存
  // - 分配大小：sizeof(struct bootstrapState)
  // - 内容初始化为 0
  //
  // ============================================================

  NCCLCHECKGOTO(ncclCalloc(&state, 1), ret, fail);
  state->rank = rank;                                  // 当前 rank 号
  state->nranks = nranks;                              // 总 rank 数
  state->cudaDev = comm->cudaDev;                      // CUDA 设备编号
  state->abortFlag = comm->abortFlag;                  // 中止标志（用于错误处理）
  state->net = comm->ncclNet;                          // 网络插件接口
  comm->bootstrap = state;                             // 将 state 保存到 comm 中
  comm->magic = state->magic = magic;                  // 设置魔数（用于连接验证）

  // ============================================================
  // 计算在分割后通信器中的前一个和后一个 rank
  // ============================================================
  //
  // Bootstrap 使用环形拓扑建立连接
  // 每个 rank 连接到前驱（接收数据）和后继（发送数据）
  //
  // 计算公式：
  // - prev = parentRanks[(rank - 1 + nranks) % nranks]
  //   - 在子 comm 的 rank 序列中找到前一个 rank
  //   - 然后通过 parentRanks 找到对应的父 comm rank
  // - next = parentRanks[(rank + 1) % nranks]
  //   - 在子 comm 的 rank 序列中找到后一个 rank
  //   - 然后通过 parentRanks 找到对应的父 comm rank
  //
  // 示例：
  //   父 comm：8 个 rank [0, 1, 2, 3, 4, 5, 6, 7]
  //   子 comm A：包含父 comm 的 [0, 2, 4, 6]
  //   parentRanks = [0, 2, 4, 6]
  //
  //   对于子 comm rank 0（父 comm rank 0）：
  //     - 前一个：(0-1+4)%4 = 3 → parentRanks[3] = 6
  //     - 后一个：(0+1)%4 = 1 → parentRanks[1] = 2
  //     - prev = 6, next = 2
  //
  //   对于子 comm rank 1（父 comm rank 2）：
  //     - 前一个：(1-1+4)%4 = 0 → parentRanks[0] = 0
  //     - 后一个：(1+1)%4 = 2 → parentRanks[2] = 4
  //     - prev = 0, next = 4
  //
  // 环形拓扑图：
  //   0 ← 6 ← 4 ← 2 ← 0 (父 comm rank)
  //   ↓           ↓
  // 子 comm: 0    3
  //           ↑    ↑
  //           1    2
  //
  // 为什么要用父 comm rank？
  // - 因为需要通过父 comm 的连接来交换信息
  // - 父 comm 已经建立了完整的连接
  //
  // ============================================================

  // 计算在分割后通信器中的前一个和后一个 rank
  prev = parentRanks[(rank - 1 + nranks) % nranks];    // 环形前驱（父 comm rank）
  next = parentRanks[(rank + 1) % nranks];            // 环形后继（父 comm rank）

  // ============================================================
  // 创建监听句柄（用于其他 rank 连接到当前 rank）
  // ============================================================
  //
  // 这一步创建一个监听端点，其他 rank 可以通过这个端点连接到当前 rank
  //
  // 两种模式：
  //
  // 1. Bootstrap Net 模式（ncclParamBootstrapNetEnable() == true）：
  //    - 使用网络插件（NCCL Net）提供的监听接口
  //    - 适用于支持 RDMA 的高速网络（如 InfiniBand）
  //    - 优点：可以直接使用网络插件，性能更好
  //
  //    netGetDevice：获取网络设备
  //    - 根据 rank 选择合适的网络设备
  //    - 返回设备编号（net.dev）
  //
  //    state->net->listen：创建监听句柄
  //    - 在指定设备上监听连接
  //    - 返回监听句柄（net.handle）和通信对象（net.comm）
  //
  //    NCCL_NET_HANDLE_MAXSIZE：
  //    - 网络句柄的最大大小
  //    - 将监听句柄拷贝到 info 中，准备发送给其他 rank
  //
  // 2. Socket 模式（ncclParamBootstrapNetEnable() == false）：
  //    - 使用传统的 Socket 连接
  //    - 适用于 TCP/IP 网络
  //    - 优点：兼容性好，适用于各种网络环境
  //
  //    createListenSocket：创建监听 Socket
  //    - 绑定到一个端口
  //    - 返回 Socket 对象和地址（socket.addr）
  //
  // ringConnectInfo 结构：
  // - 包含连接信息（网络句柄或 Socket 地址）
  // - 用于在 rank 之间交换连接信息
  //
  // STATE_LISTEN 宏：
  // - 访问 state 中的监听相关字段
  // - 根据编译时的配置选择网络或 Socket
  //
  // ============================================================

  // create a handle for the others to reach out to me
  if (ncclParamBootstrapNetEnable()) {
    // ============================================================
    // 使用网络插件模式（Bootstrap Net）
    // ============================================================

    NCCLCHECKGOTO(netGetDevice(rank, comm, &STATE_LISTEN(state, net.dev)), ret, fail);
    // 在指定设备上创建监听句柄
    // net.handle：监听句柄，其他 rank 通过这个句柄连接
    // net.comm：通信对象，用于后续的通信操作
    NCCLCHECKGOTO(state->net->listen(comm->netContext, STATE_LISTEN(state, net.dev), STATE_LISTEN(state, net.handle), &STATE_LISTEN(state, net.comm)), ret, fail);
    // 将监听句柄拷贝到 info 中，准备发送给其他 rank
    memcpy(info.handle, STATE_LISTEN(state, net.handle), NCCL_NET_HANDLE_MAXSIZE);
  } else {
    // ============================================================
    // 使用 Socket 模式
    // ============================================================
    // create socket for ring neightbor to contact mee
    // 创建监听 Socket，其他 rank 可以通过这个 Socket 连接
    // socket.addr：Socket 地址（IP + 端口）
    NCCLCHECK(createListenSocket(comm, comm->magic, &STATE_LISTEN(state, socket), &info.addr, ncclSocketTypeBootstrap));
  }

  // ============================================================
  // 创建 P2P 监听 Socket
  // ============================================================
  //
  // 这个 Socket 用于点对点（P2P）通信
  //
  // 与环形 Socket 的区别：
  // - 环形 Socket：仅用于环形拓扑中的前驱和后继
  // - P2P Socket：用于任意两个 rank 之间的直接通信
  //
  // P2P Socket 的用途：
  // - 直接的点对点操作（ncclSend、ncclRecv）
  // - Bootstrap 阶段的信息交换
  // - 不依赖环形拓扑的灵活通信
  //
  // peerSocketAddress：
  // - 当前 rank 的 P2P Socket 地址
  // - 需要分发给所有其他 rank
  //
  // ============================================================

  // create a socket for others to reach out (P2P)
  union ncclSocketAddress peerSocketAddress;           // 当前 rank 的 P2P 地址
  // 创建 P2P 监听 Socket
  NCCLCHECK(createListenSocket(comm, comm->magic, &STATE_LISTEN(state, peerSocket), &peerSocketAddress, ncclSocketTypeBootstrap));

  // ============================================================
  // RAS（可靠性、可用性、可服务性）初始化
  // ============================================================
  //
  // RAS 是 NCCL 的可靠性增强功能
  //
  // RAS 提供的功能：
  // - 错误检测和报告
  // - 故障恢复
  // - 健康监控
  //
  // ncclParamRasEnable()：
  // - 环境变量：NCCL_RAS_ENABLE
  // - 默认为 0（禁用）
  // - 设置为 1 启用 RAS 功能
  //
  // ncclRasCommInit：
  // - 初始化 RAS 子系统
  // - 注册监控回调
  // - 启动健康检查
  //
  // 如果初始化失败：
  // - 打印警告信息，但继续执行
  // - RAS 不是必需的，失败不会导致 Bootstrap 失败
  //
  // ============================================================

  if (ncclParamRasEnable() == 1) {
    // 初始化 RAS 子系统
    if (ncclRasCommInit(comm, nullptr) != ncclSuccess)
      INFO(NCCL_INIT|NCCL_RAS, "Continuing in spite of a RAS initialization error");
  }

  // ============================================================
  // 通过父通信器的连接交换环形连接信息
  // ============================================================
  //
  // 这一步是关键：使用父 comm 的现有连接来交换子 comm 的连接信息
  //
  // 为什么要使用父 comm 的连接？
  // 1. 父 comm 已经建立了完整的通信连接
  // 2. 子 comm 还没有建立自己的连接
  // 3. 复用父 comm 的连接可以快速完成信息交换
  //
  // 交换的信息：
  // - 当前 rank 的监听地址（info）
  // - 下一个 rank 的监听地址（nextPeer）
  //
  // 信息交换流程：
  // 1. 当前 rank → prev：发送自己的连接信息（info）
  //    - 通过 bootstrapSend 发送到父 comm 的 prev rank
  //    - prev 在父 comm 的环形拓扑中是当前 rank 的前驱
  //    - 但在子 comm 中，prev 对应的是环形前驱的父 comm rank
  //
  // 2. next → 当前 rank：接收下一个 rank 的连接信息（nextPeer）
  //    - 通过 bootstrapRecv 从父 comm 的 next rank 接收
  //    - next 在父 comm 的环形拓扑中是当前 rank 的后继
  //    - 但在子 comm 中，next 对应的是环形后继的父 comm rank
  //
  // BOOTSTRAP_TAG_COMMSPLIT：
  // - 通信标签，标识这是 CommSplit 操作
  // - 用于区分不同的 Bootstrap 消息类型
  //
  // 示例（继续之前的例子）：
  //   子 comm A：包含父 comm 的 [0, 2, 4, 6]
  //   parentRanks = [0, 2, 4, 6]
  //
  //   对于子 comm rank 0（父 comm rank 0）：
  //     - prev = 6（父 comm rank）
  //     - next = 2（父 comm rank）
  //     - 发送 info 到父 comm rank 6
  //     - 从父 comm rank 2 接收 nextPeer
  //
  //   注意：prev 和 next 是父 comm 的 rank 编号
  //        因为要通过父 comm 的连接进行通信
  //
  // ============================================================

  // Get addr from next rank using the parent's connections
  // 通过父通信器的连接获取下一个 rank 的地址

  // 向父 comm 的 prev rank 发送当前 rank 的连接信息
  // prev：在子 comm 环形拓扑中，当前 rank 的前驱在父 comm 中的 rank 号
  NCCLCHECKGOTO(bootstrapSend(parent->bootstrap, prev, BOOTSTRAP_TAG_COMMSPLIT, &info, sizeof(union ringConnectInfo)), ret, fail);
  // 从父 comm 的 next rank 接收下一个 rank 的连接信息
  // next：在子 comm 环形拓扑中，当前 rank 的后继在父 comm 中的 rank 号
  // nextPeer：下一个 rank 的连接信息（监听地址）
  NCCLCHECKGOTO(bootstrapRecv(parent->bootstrap, next, BOOTSTRAP_TAG_COMMSPLIT, &nextPeer, sizeof(union ringConnectInfo)), ret, fail);

  // ============================================================
  // 建立环形连接
  // ============================================================
  //
  // 交换完连接信息后，建立实际的环形连接
  //
  // 连接方向：
  // - 发送连接：当前 rank → next
  // - 接收连接：prev → 当前 rank
  //
  // 两种模式（与监听时的模式一致）：
  //
  // 1. Bootstrap Net 模式：
  //    netRingConnect：建立环形网络连接
  //    - 使用网络插件提供的连接接口
  //    - nextPeer.handle：下一个 rank 的监听句柄
  //    - 返回：
  //      - sendComm：发送通信对象（当前 rank → next）
  //      - sendDevHandle：发送设备句柄
  //      - recvComm：接收通信对象（prev → 当前 rank）
  //      - recvDevHandle：接收设备句柄
  //
  // 2. Socket 模式：
  //    socketRingConnect：建立环形 Socket 连接
  //    - 使用传统的 Socket 连接
  //    - nextPeer.addr：下一个 rank 的 Socket 地址
  //    - 返回：
  //      - send：发送 Socket（当前 rank → next）
  //      - recv：接收 Socket（prev → 当前 rank）
  //
  // 连接建立的对称性：
  // - 当前 rank 连接到 next（作为发送方）
  // - prev 连接到当前 rank（作为接收方）
  // - 所有 rank 同时执行，形成完整的环形连接
  //
  // STATE_RING 宏：
  // - 访问 state 中的环形连接相关字段
  // - 根据编译时的配置选择网络或 Socket
  //
  // ============================================================

  if (ncclParamBootstrapNetEnable()) {
    // ============================================================
    // 使用网络插件模式建立环形连接
    // ============================================================

    NCCLCHECKGOTO(netRingConnect(comm->netContext, state->net, &state->listen, nextPeer.handle,
                                 &STATE_RING(state, net.sendComm), &STATE_RING(state, net.sendDevHandle),
                                 &STATE_RING(state, net.recvComm), &STATE_RING(state, net.recvDevHandle), state->abortFlag),
                  ret, fail);
  } else {
    // ============================================================
    // 使用 Socket 模式建立环形连接
    // ============================================================

    NCCLCHECK(socketRingConnect(&nextPeer.addr, &STATE_RING(state, socket.send), &STATE_LISTEN(state, socket), &STATE_RING(state, socket.recv), comm->magic, state->abortFlag));
  }

  // ============================================================
  // 分配并初始化 P2P 地址数组
  // ============================================================
  //
  // peerP2pAddresses 数组：
  // - 存储所有 rank 的 P2P Socket 地址
  // - 大小：nranks 个元素
  // - peerP2pAddresses[i]：rank i 的 P2P 地址
  //
  // 当前步骤：
  // 1. 分配 peerP2pAddresses 数组（nranks 个元素）
  // 2. 将当前 rank 的 P2P 地址（peerSocketAddress）填入数组
  //    - peerP2pAddresses[rank] = peerSocketAddress
  //    - 其他 rank 的地址稍后通过 ringAllInfo 收集
  //
  // P2P 地址的用途：
  // - 任意两个 rank 之间的直接通信
  // - ncclSend/ncclRecv 操作
  // - Bootstrap 阶段的信息交换
  //
  // ============================================================

  NCCLCHECKGOTO(ncclCalloc(&state->peerP2pAddresses, nranks), ret, fail);
  // 将当前 rank 的 P2P 地址填入数组的对应位置
  memcpy(state->peerP2pAddresses + rank, &peerSocketAddress, sizeof(union ncclSocketAddress));

  // ============================================================
  // 根据是否共享资源选择不同的初始化路径
  // ============================================================
  //
  // shareResources 标志：
  // - true：子 comm 与父 comm 共享某些资源（如 Proxy 线程）
  // - false：子 comm 独立创建所有资源
  //
  // 共享资源的优势：
  // - 减少 Proxy 线程数量（多个 comm 共享一个 Proxy）
  // - 减少资源开销
  // - 提高 CPU 和内存利用率
  //
  // 两种路径：
  //
  // 1. 共享资源（parent->shareResources == true）：
  //    - 不创建新的 Proxy 线程
  //    - 复用父 comm 的 Proxy
  //    - 更新 topParentRanks 映射
  //    - 只收集 P2P 地址
  //
  // 2. 不共享资源（parent->shareResources == false）：
  //    - 创建新的 Proxy 线程
  //    - 分配 Proxy 地址数组
  //    - 收集 P2P 和 Proxy 地址
  //    - 初始化 Proxy 服务
  //
  // ============================================================

  if (parent->shareResources) {
    // ============================================================
    // 共享资源模式
    // ============================================================
    //
    // 在这种模式下，子 comm 复用父 comm 的资源
    //

    /* map local rank to top parent local rank. */
    // 更新 topParentRanks 映射
    // topParentRanks[i]：子 comm rank i 在最顶层父 comm 中的 rank 号
    //
    // 计算方式：
    // - parentRanks[i]：子 comm rank i 在父 comm 中的 rank 号
    // - parent->topParentRanks[parentRanks[i]]：父 comm rank 在顶层 comm 中的 rank 号
    //
    // 示例（嵌套分割）：
    //   顶层 comm：8 个 ranks [0, 1, 2, 3, 4, 5, 6, 7]
    //   第一次分割：子 comm A 包含 [0, 2, 4, 6]
    //   第二次分割：从 A 分割出子 comm B 包含 [0, 4]
    //
    //   对于子 comm B 的 rank 0（父 comm A 的 rank 0，顶层 comm 的 rank 0）：
    //     - parentRanks[0] = 0（在父 comm A 中的 rank）
    //     - parent->topParentRanks[0] = 0（在顶层 comm 中的 rank）
    //     - topParentRanks[0] = 0
    //
    //   对于子 comm B 的 rank 1（父 comm A 的 rank 2，顶层 comm 的 rank 4）：
    //     - parentRanks[1] = 2（在父 comm A 中的 rank）
    //     - parent->topParentRanks[2] = 4（在顶层 comm 中的 rank）
    //     - topParentRanks[1] = 4
    //
    for (int i = 0; i < nranks; ++i) {
      comm->topParentRanks[i] = parent->topParentRanks[parentRanks[i]];
    }
    // 收集所有 rank 的 P2P 地址
    // 只需要 P2P 地址，不需要 Proxy 地址（复用父 comm 的）
    NCCLCHECKGOTO(ringAllInfo(comm, state, state->peerP2pAddresses, NULL, NULL, NULL), ret, fail);
  } else {
    // ============================================================
    // 不共享资源模式
    // ============================================================
    //
    // 在这种模式下，子 comm 需要创建自己的资源
    //
    // 需要创建的资源：
    // 1. Proxy 地址数组（peerProxyAddresses）
    // 2. Proxy UDS 地址数组（peerProxyAddressesUDS）
    // 3. Proxy Socket（proxySocket）
    // 4. Proxy 服务线程
    //
    // Proxy 地址：
    // - Proxy 服务的监听地址
    // - 其他 rank 通过这个地址访问 Proxy
    //
    // Proxy UDS 地址：
    // - Unix Domain Socket 地址
    // - 用于本地进程间通信（同一节点内的 rank）
    // - UDS 比 TCP Socket 更高效
    //
    // Proxy Socket：
    // - Proxy 服务的监听 Socket
    // - 其他 rank 可以通过这个 Socket 连接到 Proxy
    //
    // ringAllInfo：
    // - 通过环形连接收集所有 rank 的地址信息
    // - 收集 P2P 地址、Proxy 地址、Proxy UDS 地址
    // - 每个 rank 将自己的地址发送给环形中的下一个 rank
    // - 最终所有 rank 都有完整的地址列表
    //
    // ncclProxyInit：
    // - 初始化 Proxy 服务
    // - 启动 Proxy 线程
    // - Proxy 线程处理网络操作、内存管理等异步任务
    //
    // ============================================================

    // 分配 Proxy 地址数组
    NCCLCHECKGOTO(ncclCalloc(&state->peerProxyAddresses, nranks), ret, fail);
    NCCLCHECKGOTO(ncclCalloc(&state->peerProxyAddressesUDS, nranks), ret, fail);
    // Create the service proxy and get the UDS
    // 分配 Proxy Socket 结构
    NCCLCHECKGOTO(ncclCalloc(&proxySocket, 1), ret, fail);
    // 获取当前 rank 的 UDS 地址
    NCCLCHECKGOTO(getUDS(state->peerProxyAddressesUDS + rank), ret, fail);
    // 创建 Proxy 监听 Socket
    // 其他 rank 通过这个 Socket 连接到当前 rank 的 Proxy
    NCCLCHECKGOTO(createListenSocket(comm, comm->magic, proxySocket, state->peerProxyAddresses + rank, ncclSocketTypeProxy), ret, fail);
    // 收集所有 rank 的地址信息
    // 通过环形连接交换 P2P 地址、Proxy 地址、Proxy UDS 地址
    NCCLCHECKGOTO(ringAllInfo(comm, state, state->peerP2pAddresses, state->peerProxyAddresses, state->peerProxyAddressesUDS, NULL), ret, fail);
    // 初始化 Proxy 服务
    // 启动 Proxy 线程，传入 Proxy Socket 和地址列表
    NCCLCHECKGOTO(ncclProxyInit(comm, proxySocket, state->peerProxyAddresses, state->peerProxyAddressesUDS), ret, fail);
  }

  // ============================================================
  // 调试信息输出
  // ============================================================
  //
  // TRACE：调试级别的日志输出
  // - 包含 comm 的关键信息
  // - 用于调试 Bootstrap Split 流程
  //
  // 输出信息：
  // - comm：子 comm 指针
  // - parent：父 comm 指针
  // - rank：子 comm 中的 rank 号
  // - nranks：子 comm 中的总 rank 数
  // - color：分割颜色
  // - key：键值
  // - prev：环形前驱（父 comm rank）
  // - next：环形后继（父 comm rank）
  //
  // ============================================================

  TRACE(NCCL_BOOTSTRAP, "bootstrapSplit: comm %p parent %p rank %d nranks %d color %d key %d prev %d next %d - DONE", comm, parent, rank, nranks,
        color, key, prev, next);

exit:
  return ret;
fail:
  // ============================================================
  // 错误处理
  // ============================================================
  //
  // 发生错误时：
  // - 释放 proxySocket（如果已分配）
  // - 跳转到 exit 标签返回错误码
  //
  // 注意：
  // - state 不会在这里释放
  // - comm->bootstrap 仍然指向 state
  // - 如果后续需要清理，会在 comm 销毁时进行
  //
  // ============================================================

  free(proxySocket);
  goto exit;
}

// ============================================================================
// Socket 确认信息结构
// ============================================================================
struct socketAckInfo {
  int rank;
  int tag;
};

// ============================================================================
// socketConnect - 建立 Socket 连接并发送确认信息
// ============================================================================
static ncclResult_t socketConnect(void* commState, int peer, int tag, struct ncclSocket* sock) {
  ncclResult_t ret = ncclSuccess;
  struct bootstrapState* state = (struct bootstrapState*)commState;

  struct socketAckInfo ack = (struct socketAckInfo) {
    .rank = state->rank, 
    .tag = tag
  };
    
  NCCLCHECKGOTO(ncclSocketInit(sock, state->peerP2pAddresses + peer, state->magic, ncclSocketTypeBootstrap, state->abortFlag), ret, fail);
  NCCLCHECKGOTO(ncclSocketConnect(sock), ret, fail);
  NCCLCHECKGOTO(socketSend(sock, &ack, sizeof(struct socketAckInfo)), ret, fail);
  
  return ncclSuccess;

fail:
  (void)ncclSocketClose(sock);
  return ret;
}

// ============================================================================
// bootstrapSend - 发送数据到指定 peer peerP2pAddresses
// ============================================================================
ncclResult_t bootstrapSend(void* commState, int peer, int tag, void* data, int size) {
  ncclResult_t ret = ncclSuccess;
  struct ncclSocket sock;
  TRACE(NCCL_BOOTSTRAP, "Sending to peer=%d tag=%d size=%d", peer, tag, size);
  NCCLCHECK(socketConnect(commState, peer, tag, &sock));
  NCCLCHECKGOTO(socketSend(&sock, data, size), ret, fail);
  TRACE(NCCL_BOOTSTRAP, "Sent to peer=%d tag=%d size=%d", peer, tag, size);
  NCCLCHECK(ncclSocketClose(&sock));
  return ret;
fail:
  (void)ncclSocketClose(&sock);
  return ret;
}

// ============================================================================
// 非预期连接队列管理
// 用于处理在预期之外到达的连接
// ============================================================================

// Bootstrap send/receive functions
// 将非预期连接加入队列
static ncclResult_t unexpectedEnqueue(struct bootstrapState* state, int peer, int tag, struct ncclSocket* sock) {
  // New unex
  struct unexConn* unex;
  NCCLCHECK(ncclCalloc(&unex, 1));
  unex->peer = peer;
  unex->tag = tag;
  memcpy(&unex->sock, sock, sizeof(struct ncclSocket));

  // Enqueue
  struct unexConn* list = state->unexpectedConnections;
  if (list == NULL) {
    state->unexpectedConnections = unex;
    return ncclSuccess;
  }

  //加入末尾
  while (list->next) 
    list = list->next;

  list->next = unex;
  return ncclSuccess;
}

// 从队列中取出匹配的连接
static ncclResult_t unexpectedDequeue(struct bootstrapState* state, int peer, int tag, struct ncclSocket* sock, int* found) {
  struct unexConn* elem = state->unexpectedConnections;
  struct unexConn* prev = NULL;
  
  *found = 0;
  while (elem) {
    if (elem->peer == peer && elem->tag == tag) {
      if (prev == NULL) {
        state->unexpectedConnections = elem->next;
      } else {
        prev->next = elem->next;
      }
      memcpy(sock, &elem->sock, sizeof(struct ncclSocket));
      free(elem);
      *found = 1;
      return ncclSuccess;
    }
    prev = elem;
    elem = elem->next;
  }
  return ncclSuccess;
}

// 释放所有非预期连接
static void unexpectedFree(struct bootstrapState* state) {
  struct unexConn* elem = state->unexpectedConnections;
  struct unexConn* prev = NULL;

  while (elem) {
    prev = elem;
    elem = elem->next;
    free(prev);
  }
  return;
}

// We can't know who we'll receive from, so we need to receive everything at once
// 接受来自任意 peer 的连接
static ncclResult_t socketAccept(void* commState, int peer, int tag, struct ncclSocket* sock) {
  ncclResult_t ret = ncclSuccess;
  struct bootstrapState* state = (struct bootstrapState*)commState;

  // Search unexpected connections first
  // 首先在非预期连接队列中查找
  int found;
  NCCLCHECK(unexpectedDequeue(state, peer, tag, sock, &found));
  if (found) 
    return ncclSuccess;

  // Then look for new connections
  // 如果没找到，则接受新连接
  while (1) {
    struct socketAckInfo ack = {0};
    NCCLCHECKGOTO(ncclSocketInit(sock), ret, fail);
    NCCLCHECKGOTO(ncclSocketAccept(sock, &STATE_LISTEN(state, peerSocket)), ret, fail);
    NCCLCHECKGOTO(socketRecv(sock, &ack, sizeof(struct socketAckInfo)), ret, fail);
    //比较peer和tag是否是我们期望的，返回
    if (ack.rank == peer && ack.tag == tag)
        return ncclSuccess;

    // 不是我们期待的连接，加入队列，循环接收
    NCCLCHECKGOTO(unexpectedEnqueue(state, ack.rank, ack.tag, sock), ret, fail);
  }
  
  return ncclSuccess;
fail:
  (void)ncclSocketClose(sock);
  return ret;
}

// ============================================================================
// bootstrapRecv - 从指定 peer 接收数据
// We can't know who we'll receive from, so we need to receive everything at once
// ============================================================================
ncclResult_t bootstrapRecv(void* commState, int peer, int tag, void* data, int size) {
  ncclResult_t ret;
  struct ncclSocket sock;
  NCCLCHECK(socketAccept(commState, peer, tag, &sock));
  TRACE(NCCL_BOOTSTRAP, "Receiving tag=%d peer=%d size=%d", tag, peer, size);
  //接收数据
  NCCLCHECKGOTO(socketRecv(&sock, ((char*)data), size), ret, fail);
  //关闭连接
  NCCLCHECKGOTO(ncclSocketClose(&sock, /*wait=*/true), ret, fail);
  return ret;
  
fail:
  (void)ncclSocketClose(&sock);
  return ret;
}

// ============================================================================
// netRingAllGather - 使用网络插件进行 Ring AllGather
// ============================================================================
static ncclResult_t netRingAllGather(ncclNet_t* net, void* sendComm, void* recvComm, int rank, int nranks, char* data, int size, volatile uint32_t* abortFlag) {
  ncclResult_t res;
  uint64_t tFirst = 0, tRest = 0;
  void* sendDataHandle = NULL;
  void* recvDataHandle = NULL;
  NCCLCHECKGOTO(netReg(net, sendComm, data, nranks * size, &sendDataHandle), res, exit);
  NCCLCHECKGOTO(netReg(net, recvComm, data, nranks * size, &recvDataHandle), res, exit);

  /* Simple ring based AllGather
   * At each step i receive data from (rank-i-1) from prev
   * and send previous step's data from (rank-i) to next
   */
  TRACE(NCCL_BOOTSTRAP, "NetRingAllGather started");
  BOOTSTRAP_PROF_OPEN(tFirst);

  // 总共迭代 nranks - 1 轮，因为每个 rank 都拥有自己 rank 号的本地数据
  for (int i = 0; i < nranks - 1; i++) {
    int tag = i;
    size_t rslice = (rank - i - 1 + nranks) % nranks;  // 接收数据的 slice 索引
    size_t sslice = (rank - i + nranks) % nranks;        // 发送数据的 slice 索引
    void* recv_data = data + rslice * size;
    void* send_data = data + sslice * size;
    NCCLCHECKGOTO(netSendRecv(net, sendComm, send_data, size, sendDataHandle, recvComm, recv_data, size, recvDataHandle, tag, abortFlag), res, exit);
    if (i == 0) {
      BOOTSTRAP_PROF_CLOSE(tFirst);
      BOOTSTRAP_PROF_OPEN(tRest);
    }
  }
  BOOTSTRAP_PROF_CLOSE(tRest);
  TRACE(NCCL_BOOTSTRAP | NCCL_PROFILE, "netRingAllGather first message in %f (%f MB/sec), rest in %f (%f MB/sec)", tFirst / 1e9, (size / 1e6) / (tFirst / 1e9), tRest / 1e9, (nranks - 1) * (size / 1e6) / (tRest / 1e9));

exit:
  // do not fail in case of error, try to deregister as much as possible
  if (sendDataHandle) netDereg(net, sendComm, &sendDataHandle);
  if (recvDataHandle) netDereg(net, recvComm, &recvDataHandle);
  return res;
}

// ============================================================================
// socketRingAllGather - 使用 Socket 进行 Ring AllGather
// ============================================================================
static ncclResult_t socketRingAllGather(struct ncclSocket* sendSock, struct ncclSocket* recvSock, int rank, int nranks, char* data, int size) {
  ncclResult_t res = ncclSuccess;
  uint64_t tFirst = 0, tRest = 0;

  /* Simple ring based AllGather
   * At each step i receive data from (rank-i-1) from prev
   * and send previous step's data from (rank-i) to next
   */
  TRACE(NCCL_BOOTSTRAP, "socketRingAllGather started");
  BOOTSTRAP_PROF_OPEN(tFirst);

  // 如果只有一个 rank，不需要同步数据
  // 总共迭代 nranks - 1 轮，因为每个 rank 都拥有自己 rank 号的本地数据，只需要同步其余数据
  for (int i = 0; i < nranks - 1; i++) {
    // 接收数据时按照 rank 号递减接收，比如有 4 个 rank，rank 0 就按照 rank 号 3->2->1 接收数据
    size_t rslice = (rank - i - 1 + nranks) % nranks;
    // 发送数据时也是按照 rank 号递减发送，比如有 4 个 rank，rank 0 就按照 rank 号 0->3->2 发送数据
    size_t sslice = (rank - i + nranks) % nranks;
    // 计算数据存放位置
    void* recv_data = data + rslice * size;
    void* send_data = data + sslice * size;
    // 调用 socket 函数发送和接收数据
    NCCLCHECKGOTO(socketSendRecv(sendSock, send_data, size, recvSock, recv_data, size), res, exit);
    if (i == 0) {
      BOOTSTRAP_PROF_CLOSE(tFirst);
      BOOTSTRAP_PROF_OPEN(tRest);
    }
  }
  BOOTSTRAP_PROF_CLOSE(tRest);
  TRACE(NCCL_BOOTSTRAP | NCCL_PROFILE, "socketRingAllGather first message in %f (%f MB/sec), rest in %f (%f MB/sec)", tFirst / 1e9, (size / 1e6) / (tFirst / 1e9), tRest / 1e9, (nranks - 1) * (size / 1e6) / (tRest / 1e9));

exit:
  return res;
}

// ============================================================================
// bootstrapAllGather - 在所有 rank 间收集数据
// ============================================================================
ncclResult_t bootstrapAllGather(void* commState, void* allData, int size) {
  ncclResult_t res = ncclSuccess;
  struct bootstrapState* state = (struct bootstrapState*)commState;
  // rank 信息，allgather 根据这个信息才能知道要迭代多少轮
  int rank = state->rank;
  int nranks = state->nranks;

  TRACE(NCCL_BOOTSTRAP, "rank %d nranks %d size %d - AllGather", rank, nranks, size);

  uint64_t time = 0;
  BOOTSTRAP_PROF_OPEN(time);

  if (ncclParamBootstrapNetEnable()) {
    // 使用网络插件（如 IB/RoCE）
    NCCLCHECKGOTO(netRingAllGather(state->net, STATE_RING(state, net.sendComm), STATE_RING(state, net.recvComm), rank, nranks, (char*)allData, size, state->abortFlag), res, exit);
  } else {
    // 使用普通 TCP socket
    NCCLCHECKGOTO(socketRingAllGather(&STATE_RING(state, socket.send), &STATE_RING(state, socket.recv), rank, nranks, (char*)allData, size), res, exit);
  }

exit:
  BOOTSTRAP_PROF_CLOSE(time);
  TRACE(NCCL_BOOTSTRAP | NCCL_PROFILE, "bootstrapAllGather for %d B done in %f sec: %f MB/sec", size, time / 1e9, (nranks * size / 1e6) / (time / 1e9));
  TRACE(NCCL_BOOTSTRAP, "rank %d nranks %d size %d - AllGather DONE", rank, nranks, size);
  return res;
fail:
  goto exit;
}

// ============================================================================
// bootstrapP2PBarrier - 节点内/进程间 P2P 屏障
// ============================================================================
static ncclResult_t bootstrapP2PBarrier(void* commState, int* ranks, int rank, int nranks, int tag) {
  if (nranks == 1)
    return ncclSuccess;

  /* Simple [intra] process barrier
   *
   * Based on the dissemination algorithm by Debra Hensgen, Raphael Finkel, and Udi Manbet,
   * "Two Algorithms for Barrier Synchronization," International Journal of Parallel Programming, 17(1):1-17, 1988
   */
  int data[1] = {0};

  // bootstrap 同步一些信息
  // 这里说明 bootstrap 后面是需要使用到的
  for (int mask = 1; mask < nranks; mask <<= 1) {
    int src = (rank - mask + nranks) % nranks;
    int dst = (rank + mask) % nranks;
    NCCLCHECK(bootstrapSend(commState, ranks ? ranks[dst] : dst, tag, data, sizeof(data)));
    NCCLCHECK(bootstrapRecv(commState, ranks ? ranks[src] : src, tag, data, sizeof(data)));
  }

  return ncclSuccess;
}

// ============================================================================
// bootstrapIntraNodeBarrier - 节点内屏障
// ============================================================================
ncclResult_t bootstrapIntraNodeBarrier(void* commState, int* ranks, int rank, int nranks, int tag) {
  uint64_t time = 0;
  BOOTSTRAP_PROF_OPEN(time);
  NCCLCHECK(bootstrapP2PBarrier(commState, ranks, rank, nranks, tag));
  BOOTSTRAP_PROF_CLOSE(time);
  TRACE(NCCL_BOOTSTRAP | NCCL_PROFILE, "bootstrapIntraNodeBarrier done in %f sec", time / 1e9);
  return ncclSuccess;
}

// ============================================================================
// bootstrapBarrier - 全局屏障
// ============================================================================
ncclResult_t bootstrapBarrier(void* commState, int rank, int nranks, int tag) {
  uint64_t time = 0;
  BOOTSTRAP_PROF_OPEN(time);
  NCCLCHECK(bootstrapP2PBarrier(commState, NULL, rank, nranks, tag));
  BOOTSTRAP_PROF_CLOSE(time);
  TRACE(NCCL_BOOTSTRAP | NCCL_PROFILE, "bootstrapBarrier done in %f sec", time / 1e9);
  return ncclSuccess;
}

// ============================================================================
// bootstrapIntraNodeAllGather - 节点内 AllGather
// ============================================================================
//这里不是用的bootstrap阶段建立的环形连接，而是用了peeraddress临时建连，同步数据后关闭连接
ncclResult_t bootstrapIntraNodeAllGather(void* commState, int* ranks, int rank, int nranks, void* allData, int size) {
  if (nranks == 1) 
    return ncclSuccess;
  TRACE(NCCL_INIT, "rank %d nranks %d size %d - ENTER", rank, nranks, size);

  int prevRank = ranks[(rank - 1 + nranks) % nranks];
  int nextRank = ranks[(rank + 1) % nranks];

  // intraNode bootstrap is done defacto using the socket-based implementation
  struct ncclSocket recvSocket, sendSocket;
  NCCLCHECK(socketConnect(commState, nextRank, BOOTSTRAP_TAG_INTRANODE_ALLGATHER, &sendSocket));
  NCCLCHECK(socketAccept(commState, prevRank, BOOTSTRAP_TAG_INTRANODE_ALLGATHER, &recvSocket));

  NCCLCHECK(socketRingAllGather(&sendSocket, &recvSocket, rank, nranks, (char*)allData, size));

  NCCLCHECK(ncclSocketClose(&sendSocket));
  NCCLCHECK(ncclSocketClose(&recvSocket));

  TRACE(NCCL_INIT, "rank %d nranks %d size %d - DONE", rank, nranks, size);
  return ncclSuccess;
}

// [IntraNode] in-place Broadcast
// ============================================================================
// bootstrapP2PBroadcast - P2P 广播
// ============================================================================
static ncclResult_t bootstrapP2PBroadcast(void* commState, int* ranks, int rank, int nranks, int root, void* bcastData, int size) {
  if (nranks == 1)
    return ncclSuccess;
  
  if (rank == root) {
    // 根节点向所有其他 rank 发送数据
    for (int i = 0; i < nranks; i++) {
      if (i != root) 
        NCCLCHECK(bootstrapSend(commState, ranks ? ranks[i] : i, /*tag=*/ranks ? ranks[i] : i, bcastData, size));
    }
  } else {
    // 非根节点从根节点接收数据
    NCCLCHECK(bootstrapRecv(commState, ranks ? ranks[root] : root, /*tag=*/ranks ? ranks[rank] : rank, bcastData, size));
  }
  return ncclSuccess;
}

// ============================================================================
// bootstrapIntraNodeBroadcast - 节点内广播
// ============================================================================
ncclResult_t bootstrapIntraNodeBroadcast(void* commState, int* ranks, int rank, int nranks, int root, void* bcastData, int size) {
  uint64_t time = 0;
  BOOTSTRAP_PROF_OPEN(time);
  NCCLCHECK(bootstrapP2PBroadcast(commState, ranks, rank, nranks, root, bcastData, size));
  BOOTSTRAP_PROF_CLOSE(time);
  TRACE(NCCL_BOOTSTRAP | NCCL_PROFILE, "bootstrapIntraNodeBroadcast for %d B done in %f sec: %f MB/sec", size, time / 1e9, (nranks * size / 1e6) / (time / 1e9));
  return ncclSuccess;
}

// ============================================================================
// bootstrapBroadcast - 全局广播
// ============================================================================
ncclResult_t bootstrapBroadcast(void* commState, int rank, int nranks, int root, void* bcastData, int size) {
  uint64_t time = 0;
  BOOTSTRAP_PROF_OPEN(time);
  NCCLCHECK(bootstrapP2PBroadcast(commState, NULL, rank, nranks, root, bcastData, size));
  BOOTSTRAP_PROF_CLOSE(time);
  TRACE(NCCL_BOOTSTRAP | NCCL_PROFILE, "bootstrapBroadcast done in %f sec", time / 1e9);
  return ncclSuccess;
}

// ============================================================================
// bootstrapClose - 关闭 Bootstrap 并释放资源
// ============================================================================
ncclResult_t bootstrapClose(void* commState) {
  if (commState == NULL)
    return ncclSuccess;
  struct bootstrapState* state = (struct bootstrapState*)commState;

  // close unexpected and return an error if we are not aborting and still operations in the pipe
  if (state->unexpectedConnections != NULL) {
    unexpectedFree(state);
    if (__atomic_load_n(state->abortFlag, __ATOMIC_ACQUIRE) == 0) {
      WARN("Unexpected connections are not empty");
      return ncclInternalError;
    }
  }

  if (ncclParamBootstrapNetEnable()) {
    // 关闭网络插件连接
    NCCLCHECK(state->net->closeSend(STATE_RING(state, net.sendComm)));
    NCCLCHECK(state->net->closeRecv(STATE_RING(state, net.recvComm)));
    NCCLCHECK(state->net->closeListen(STATE_LISTEN(state, net.comm)));
  } else {
    // 关闭 socket 连接
    NCCLCHECK(ncclSocketClose(&STATE_RING(state, socket.send)));
    NCCLCHECK(ncclSocketClose(&STATE_RING(state, socket.recv)));
    NCCLCHECK(ncclSocketClose(&STATE_LISTEN(state, socket)));
  }

  // close the p2p socket
  NCCLCHECK(ncclSocketClose(&STATE_LISTEN(state, peerSocket)));

  // proxy things are free'd elsewhere
  free(state->peerP2pAddresses);
  free(state);
  return ncclSuccess;
}

// ============================================================================
// bootstrapAbort - 中止 Bootstrap 并释放资源
// ============================================================================
ncclResult_t bootstrapAbort(void* commState) {
  if (commState == NULL)
    return ncclSuccess;
  struct bootstrapState* state = (struct bootstrapState*)commState;
  // when aborting we need to close the proxy here (maybe?)
  free(state->peerProxyAddresses);
  free(state->peerProxyAddressesUDS);
  NCCLCHECK(bootstrapClose(commState));
  return ncclSuccess;
}
