/*************************************************************************
 * Copyright (c) 2015-2017, NVIDIA CORPORATION. All rights reserved.
 * 版权所有 (c) 2015-2017, NVIDIA CORPORATION. 保留所有权利。
 *
 * See LICENSE.txt for license information
 * 请参阅 LICENSE.txt 获取许可证信息
 ************************************************************************/

// 防止头文件重复包含的保护宏开始
// 如果未定义 NCCL_GROUP_H_ 宏，则定义它
#ifndef NCCL_GROUP_H_
#define NCCL_GROUP_H_

// 引入 NCCL 核心头文件，包含基本的类型定义和 API 接口
#include "nccl.h"
// 引入通信器头文件，包含 ncclComm 结构体和相关操作的定义
#include "comm.h"
// 引入分配器头文件，包含内存分配器的实现
#include "allocator.h"
// 引入注册头文件，包含内存注册相关的功能
#include "register.h"

// 函数声明：检查组操作期间的错误
// 在组调用过程中，此函数用于检查和记录错误状态
// 参数 ret: 要检查的错误码
// 返回值：返回传入的错误码
ncclResult_t ncclGroupErrCheck(ncclResult_t ret);

// 函数声明：将通信器加入到当前线程的组中
// 此函数将通信器添加到指定类型的组任务链表中
// 参数 comm: 要加入组的通信器指针
// 参数 type: 任务类型（如集合通信、P2P 通信等）
void ncclGroupCommJoin(struct ncclComm* comm, int type);

// 函数声明：将通信器标记为需要预连接
// 预连接可以在实际通信前提前建立连接，减少延迟
// 参数 comm: 需要预连接的通信器指针
void ncclGroupCommPreconnect(struct ncclComm* comm);

// 函数声明：通信器离开组
// 当通信器完成组操作后调用此函数进行清理
// 参数 comm: 要离开组的通信器指针
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
ncclResult_t ncclGroupCommLeave(struct ncclComm* comm);

// 函数声明：中止组作业
// 此函数用于中止正在执行的非阻塞组作业
// 参数 groupJob: 指向组作业结构体的指针
// 返回值：ncclResult_t 类型，总是返回成功
ncclResult_t ncclGroupJobAbort(struct ncclGroupJob* groupJob);

// 函数声明：完成组作业
// 此函数用于等待非阻塞组作业完成并获取结果
// 参数 groupJob: 指向组作业结构体的指针
// 返回值：ncclResult_t 类型，返回作业的执行结果
ncclResult_t ncclGroupJobComplete(struct ncclGroupJob *groupJob);

// 函数指针类型定义：NCCL 初始化函数类型
// 定义了 NCCL 通信器初始化函数的签名
// 参数：
//   newcomm: 输出参数，返回新创建的通信器指针
//   ndev: 设备数量
//   commId: 通信唯一标识符
//   myrank: 当前进程的 rank 编号
//   cudaDev: CUDA 设备编号
// 返回值：ncclResult_t 类型，表示初始化成功或失败的状态码
typedef ncclResult_t(*ncclInitFunc_t)(ncclComm_t* newcomm, int ndev, ncclUniqueId commId, int myrank, int cudaDev);

// 函数声明：异步初始化 NCCL 通信器
// 此函数在独立线程中执行 NCCL 初始化，避免阻塞主线程
// 参数 func: 初始化函数指针
// 参数 newcomm: 输出参数，返回新创建的通信器指针
// 参数 ndev: 设备数量
// 参数 commId: 通信唯一标识符
// 参数 myrank: 当前进程的 rank 编号
// 参数 cudaDev: CUDA 设备编号
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
ncclResult_t ncclAsyncInit(ncclInitFunc_t func, ncclComm_t* newcomm, int ndev, ncclUniqueId commId, int myrank, int cudaDev);

// 枚举类型：组作业状态
// 定义了异步作业在其生命周期中的各种状态
typedef enum ncclGroupJobState {
  ncclGroupJobRunning = 0,                              // 作业正在运行中
  ncclGroupJobDone    = 1,                             // 作业已完成（线程已结束但尚未 join）
  ncclGroupJobJoined  = 2,                             // 作业已加入（线程已 join，资源已回收）
} ncclGroupJobState_t;

// 结构体：异步作业
// 此结构体表示一个可以异步执行的作业，包含作业的所有相关信息
struct ncclAsyncJob {
  struct ncclAsyncJob* next;                            // 指向下一个作业的指针，用于构建链表（无侵入式队列）
  pthread_t thread;                                     // 执行此作业的 pthread 线程句柄
  ncclResult_t result;                                  // 作业的执行结果
  ncclResult_t(*func)(struct ncclAsyncJob*);            // 作业的执行函数指针
  void(*undo)(struct ncclAsyncJob*);                    // 作业失败时的回滚函数指针（可为 NULL）
  void(*destructor)(void*);                            // 作业的销毁函数指针，用于清理资源（可为 NULL）
  ncclGroupJobState_t state;                            // 作业的当前状态
  uint32_t* abortFlag; /* point to comm abortFlag */    // 指向通信器的主机侧中止标志
  uint32_t* abortFlagDev; /* point to comm abortFlagDev */ // 指向通信器的设备侧中止标志
  uint32_t* childAbortFlag; /* point to child abortFlag */ // 指向子通信器的主机侧中止标志
  uint32_t* childAbortFlagDev; /* point to child abortFlagDev */ // 指向子通信器的设备侧中止标志
  ncclComm_t comm;                                      // 关联的 NCCL 通信器指针
  int destroyFlag;                                      // 销毁标志，表示通信器是否正在被销毁
  bool isThreadMain;                                    // 标识此作业是否在主线程中执行（而非独立线程）
};

// 函数声明：启动异步作业
// 此函数根据当前是否在组调用上下文中，决定是同步执行还是异步执行作业
// 参数：
//   job: 要执行的异步作业结构体指针
//   func: 作业的执行函数指针
//   undo: 作业失败时的回滚函数指针（可为 NULL）
//   destructor: 作业的销毁函数指针，用于清理资源（可为 NULL）
//   comm: NCCL 通信器指针
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
ncclResult_t ncclAsyncLaunch(
  struct ncclAsyncJob* job,                              // 异步作业结构体指针
  ncclResult_t(*func)(struct ncclAsyncJob*),            // 作业执行函数指针
  void(*undo)(struct ncclAsyncJob*),                    // 回滚函数指针
  void(*destructor)(void*), ncclComm_t comm             // 销毁函数指针和通信器
);

// 结构体：组作业
// 此结构体表示一个组操作，包含组内所有通信器和异步作业的信息
struct ncclGroupJob {
  struct ncclAsyncJob base;                             // 基类：异步作业结构体，包含作业的通用信息
  int groupRefCount;                                    // 组引用计数，记录有多少个通信器引用此组作业
  bool nonBlockingInit;                                 // 非阻塞初始化标志，true 表示此组作业以非阻塞模式运行
  bool joined;                                          // 已加入标志，true 表示组作业已完成并等待 join
  struct ncclComm *groupCommHead[ncclGroupTaskTypeNum]; // 各类任务的通信器链表头指针数组
  struct ncclComm *groupCommPreconnectHead;             // 需要预连接的通信器链表头指针
  ncclResult_t groupError;                              // 组操作期间发生的错误码
  bool abortFlag;                                       // 组中止标志，true 表示需要中止组操作
  struct ncclIntruQueue<struct ncclAsyncJob, &ncclAsyncJob::next> asyncJobs; // 异步作业队列
};

// 函数声明：组操作内部开始函数
// 此函数增加组嵌套深度，支持嵌套的组调用
// 返回值：ncclResult_t 类型，总是返回成功
ncclResult_t ncclGroupStartInternal();

// 函数声明：组操作内部结束函数
// 此函数是 ncclGroupEnd 和 ncclGroupSimulateEnd 的内部实现
// 参数 simInfo: 模拟信息指针，用于返回性能模拟数据，默认为 NULL
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
ncclResult_t ncclGroupEndInternal(ncclSimInfo_t* simInfo = NULL);

// 函数声明：完成异步作业
// 此函数等待异步作业线程结束并清理资源
// 参数 job: 指向异步作业结构体的指针
// 返回值：ncclResult_t 类型，返回作业的执行结果
ncclResult_t ncclAsyncJobComplete(struct ncclAsyncJob* job);

////////////////////////////////////////////////////////////////////////////////

// 线程局部变量声明：ncclGroupStart 嵌套调用的深度
// 每次调用 ncclGroupStart 时加 1，调用 ncclGroupEnd 时减 1
// 用于支持嵌套的组调用（组内嵌套组）
extern __thread int ncclGroupDepth; // depth of ncclGroupStart nesting（ncclGroupStart 嵌套深度）

// 线程局部变量声明：组操作期间发生的错误码
// 用于记录组调用过程中发生的第一个错误
extern __thread ncclResult_t ncclGroupError;

// 线程局部变量声明：组操作中各类任务的通信器链表头指针数组
// 每种任务类型（集合通信、P2P 通信等）都有一个对应的链表头
extern __thread struct ncclComm* ncclGroupCommHead[ncclGroupTaskTypeNum];

// 线程局部变量声明：需要预连接的通信器链表头指针
// 指向那些在组操作开始前需要建立连接的通信器
extern __thread struct ncclComm* ncclGroupCommPreconnectHead;

// 线程局部变量声明：组操作的阻塞模式标志
// -1: 默认模式（尚未确定）
//  0: 非阻塞模式
//  1: 阻塞模式
extern __thread int ncclGroupBlocking;

// 内联函数：组操作内部开始
// 此函数增加组嵌套深度，支持嵌套的组调用
// 返回值：ncclResult_t 类型，总是返回成功
inline ncclResult_t ncclGroupStartInternal() {
  ncclGroupDepth++;                                       // 增加组嵌套深度
  return ncclSuccess;                                    // 返回成功状态
}

// 内联函数：检查组操作期间的错误
// 此函数在组调用过程中检查和记录错误状态
// 参数 ret: 要检查的错误码
// 返回值：返回传入的错误码
inline ncclResult_t ncclGroupErrCheck(ncclResult_t ret) {
  if (ncclGroupDepth > 0) {                              // 如果当前在组调用中
    // 如果返回值不是成功也不是进行中，则记录错误
    if (ret != ncclSuccess && ret != ncclInProgress) ncclGroupError = ret;
  }
  return ret;                                             // 返回原始错误码
}

// Add comm to this thread's group
// 将通信器添加到当前线程的组中
//将通信器（ncclComm）加入到当前线程的组操作链表中
inline void ncclGroupCommJoin(struct ncclComm* comm, int type) {
  // 检查通信器是否尚未加入组（groupNext[type] 为特殊值 0x1 表示未加入）
  if (comm->groupNext[type] == reinterpret_cast<struct ncclComm*>(0x1)) {
    // Insert comm into ncclGroupCommHead adjacent to sibling comms. This preserves
    // the users program order yet insures siblings occur consecutively. This
    // is required by doLaunches() in "group.cc".
    // 将通信器插入到 ncclGroupCommHead 中，与其兄弟通信器相邻。这保留了用户程序的顺序，
    // 同时确保兄弟通信器连续出现。这是 group.cc 中 doLaunches() 所要求的。
    struct ncclComm** pp = &ncclGroupCommHead[type];      // 获取链表头指针的地址
    // 查找相同派系的通信器（intraComm0 相同的属于同一派系）
    //比如在同一个进程和多线程情况下有多个comm通信组
    while (*pp != nullptr && comm->intraComm0 != (*pp)->intraComm0)
      pp = &(*pp)->groupNext[type];                       // 移动到下一个通信器

    // didn't find its clique, we need to insert it with ascending order based on commHash
    // 没有找到它的派系，我们需要基于 commHash 按升序插入它
    if (*pp == nullptr) {                                 // 如果未找到相同派系的通信器
      pp = &ncclGroupCommHead[type];                      // 重新从链表头开始
      // 按 commHash 升序查找插入位置
      while (*pp != nullptr && (*pp)->commHash < comm->commHash)
        pp = &(*pp)->groupNext[type];
    }
    comm->groupNext[type] = *pp;                          // 将当前通信器的 next 指向找到的节点
    *pp = comm;                                           // 将前一个节点的 next 指向当前通信器
    // Comms gets a new memory stack scope upon joining. Each task batched for
    // this comm is allocated there.
    // 通信器在加入时获得一个新的内存栈作用域。为此通信器批处理的每个任务都分配在那里。
    ncclMemoryStackPush(&comm->memScoped);                // 推入新的内存栈作用域
    if (type == ncclGroupTaskTypeCollective) {            // 如果是集合通信类型
      // Initialize planner
      // 初始化内核规划器
      ncclKernelPlanner::Peer* tmp = comm->planner.peers; // 临时保存 peers 指针
      memset(&comm->planner, 0, sizeof(comm->planner));   // 清零 planner 结构体
      comm->planner.peers = tmp;                          // 恢复 peers 指针
    }
  }
  ncclGroupBlocking = comm->config.blocking;              // 更新组阻塞模式标志
}

// Add comm to this thread's group needing preconnect
// 将通信器添加到当前线程需要预连接的组中
inline void ncclGroupCommPreconnect(struct ncclComm* comm) {
  // 检查通信器是否尚未加入预连接链表（preconnectNext 为特殊值 0x1 表示未加入）
  if (comm->preconnectNext == reinterpret_cast<struct ncclComm*>(0x1)) {
    // 将通信器插入到预连接链表的头部（头插法）
    comm->preconnectNext = ncclGroupCommPreconnectHead;   // 当前通信器的 next 指向原链表头
    ncclGroupCommPreconnectHead = comm;                   // 更新链表头为当前通信器
  }
}

// Comm has left group
// 通信器离开组
inline ncclResult_t ncclGroupCommLeave(struct ncclComm* comm, int type) {
  comm->groupNext[type] = reinterpret_cast<struct ncclComm*>(0x1); // 重置 groupNext 为特殊值 0x1，表示未加入组
  ncclMemoryStackPop(&comm->memScoped);                   // 弹出内存栈作用域，释放批处理任务的内存
  return ncclSuccess;                                    // 返回成功状态
}

// 头文件保护结束宏
// 与开头的 #ifndef NCCL_GROUP_H_ 配对，防止头文件重复包含
#endif
