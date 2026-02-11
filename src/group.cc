/*************************************************************************
 * Copyright (c) 2015-2022, NVIDIA CORPORATION. All rights reserved.
 * 版权所有 (c) 2015-2022, NVIDIA CORPORATION. 保留所有权利。
 *
 * See LICENSE.txt for license information
 * 请参阅 LICENSE.txt 获取许可证信息
 ************************************************************************/

// 引入组操作相关的头文件，包含组通信的定义和接口
#include "group.h"
// 引入调试相关头文件，提供日志输出和错误处理功能
#include "debug.h"
// 引入入队相关头文件，包含操作入队的实现
#include "enqueue.h"
// 引入传输层头文件，包含各种传输方式的实现
#include "transport.h"
// 引入通道头文件，包含通信通道的定义
#include "channel.h"
// 引入断言头文件，提供运行时断言检查功能
#include <assert.h>
// 引入引导程序头文件，包含通信初始化的引导逻辑
#include "bootstrap.h"
// 引入集合通信引擎头文件，包含 CE（Collective Engine）相关功能
#include "ce_coll.h"
// 引入性能分析器头文件，提供性能分析和统计功能
#include "profiler.h"
// 引入 NVIDIA Tools Extension 头文件，提供性能追踪和可视化功能
#include "nvtx.h"

// 定义组操作中回收内存的最大步数
// 在组操作完成后，可能需要多轮轮询来回收所有资源
// 这个宏定义了最多进行多少轮回收操作
#define GROUP_MAX_RECLAIM_STEPS 10

// 线程局部变量：ncclGroupStart 嵌套调用的深度
// 每次调用 ncclGroupStart 时加 1，调用 ncclGroupEnd 时减 1
// 用于支持嵌套的组调用（组内嵌套组）
__thread int ncclGroupDepth = 0; // depth of ncclGroupStart nesting（ncclGroupStart 嵌套深度）

// 线程局部变量：组操作期间发生的错误码
// 用于记录组调用过程中发生的第一个错误
__thread ncclResult_t ncclGroupError = ncclSuccess;

// 线程局部变量：组操作中各类任务的通信器链表头指针数组
// 每种任务类型（集合通信、P2P 通信等）都有一个对应的链表头
// ncclGroupTaskTypeNum 是任务类型的总数
__thread struct ncclComm* ncclGroupCommHead[ncclGroupTaskTypeNum] = {nullptr};

// 线程局部变量：需要预连接的通信器链表头指针
// 指向那些在组操作开始前需要建立连接的通信器
__thread struct ncclComm* ncclGroupCommPreconnectHead = nullptr;

// 线程局部变量：异步作业队列
// 使用无侵入式队列（Intrusive Queue）管理所有待执行的异步作业
// 队列中的每个作业都包含一个 next 指针用于链表连接
__thread struct ncclIntruQueue<struct ncclAsyncJob, &ncclAsyncJob::next> ncclAsyncJobs;

// 线程局部变量：组操作的阻塞模式标志
// -1: 默认模式（尚未确定）
//  0: 非阻塞模式
//  1: 阻塞模式
__thread int ncclGroupBlocking = -1; /* default mode（默认模式） */

// 函数声明：异步作业的主执行函数
// 这是异步线程的入口点，负责执行异步作业
// 参数 arg: 指向要执行的异步作业结构体
void* ncclAsyncJobMain(void* arg);

// 函数实现：启动异步作业
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
  ) {
  // 初始化返回值为成功状态
  ncclResult_t ret = ncclSuccess;

  // 将通信器的销毁标志传递给作业
  // destroyFlag 表示通信器是否正在被销毁
  job->destroyFlag = comm->destroyFlag;
  //调用ncclGroupStart增加ncclGroupDepth
  // 检查当前是否在组调用上下文中（ncclGroupDepth == 0 表示不在组调用中）
  if (ncclGroupDepth == 0) {
    //不是组调用，直接调用，会阻塞当前调用者直到任务完成
    // 不在组调用中，直接同步执行作业
    ret = func(job);                                      // 调用执行函数
    if (ret != ncclSuccess && undo)                       // 如果执行失败且提供了回滚函数
        undo(job);                                         // 执行回滚操作
    if (destructor)                                       // 如果提供了销毁函数
        destructor(job);                                   // 销毁作业资源
  } else {
    //ncclGroupStart组调用，异步执行任务，不阻塞当前调用者
    // 在组调用中，将作业添加到异步队列中延迟执行
    job->func = func;                                     // 保存执行函数指针
    job->undo = undo;                                     // 保存回滚函数指针
    job->destructor = destructor;                         // 保存销毁函数指针
    job->abortFlag = comm->abortFlag;                     // 保存主机侧中止标志
    job->abortFlagDev = comm->abortFlagDev;               // 保存设备侧中止标志
    job->childAbortFlag = comm->childAbortFlag;           // 保存子通信器主机侧中止标志
    job->childAbortFlagDev = comm->childAbortFlagDev;     // 保存子通信器设备侧中止标志
    job->state = ncclGroupJobRunning;                     // 设置作业状态为运行中
    job->comm = comm;                                     // 保存关联的通信器指针
    /* check if there are blocking and nonblocking comms at the same time in group. */
    /* 检查组中是否同时存在阻塞和非阻塞模式的通信器 */
    // 检查并确保组中所有通信器使用相同的阻塞模式
    if (comm->destroyFlag) {                              // 如果通信器正在被销毁
      ncclGroupBlocking = 1;                              // 强制使用阻塞模式
    } else if (ncclGroupBlocking == -1) {                  // 如果这是第一个遇到的通信器
      /* first met communicator */
      /* 首次遇到通信器，记录其阻塞模式 */
      ncclGroupBlocking = comm->config.blocking;          // 使用该通信器的阻塞模式作为组的模式
    } else if (ncclGroupBlocking != comm->config.blocking) { // 如果新通信器的阻塞模式与组模式不匹配
      WARN("Blocking and nonblocking communicators are not allowed in the same group.");
      // 发出警告：同一组中不允许同时使用阻塞和非阻塞通信器
      ret = ncclInvalidArgument;                          // 返回无效参数错误
    }

    //加入队列
    // 如果没有错误，将作业添加到异步队列中
    if (ret == ncclSuccess) {                             // 检查是否有错误
      ncclIntruQueueEnqueue(&ncclAsyncJobs, job);         // 将作业入队到异步作业队列
    } else {
      // no need to undo, the job hasn't run
      // 不需要回滚，因为作业尚未执行
      if (destructor)                                     // 如果提供了销毁函数
        destructor(job);                                   // 销毁作业资源
    }
  }

  return ret;                                             // 返回操作结果
}

// 函数实现：异步作业的主执行函数
// 此函数在独立的线程中执行，是 pthread_create 的线程入口点
// 参数 arg: 指向要执行的异步作业结构体（通过 pthread_create 传入）
// 返回值: 传入的参数指针
void* ncclAsyncJobMain(void* arg) {
  // 将 void* 类型的参数转换为 ncclAsyncJob 指针
  struct ncclAsyncJob* job = (struct ncclAsyncJob*)arg;
  
  // 调用作业的执行函数并保存结果
  job->result = job->func(job);                           // 执行作业函数，返回结果保存在 job->result 中
  
  if (job->result != ncclSuccess) {                       // 如果执行失败
    INFO(NCCL_INIT,"%s:%d -> %d [Async thread]", __FILE__, __LINE__, job->result);
    // 记录日志：文件名、行号、错误码，标识这是异步线程的错误
  }
  
  // 使用原子操作将作业状态设置为已完成
  // __ATOMIC_RELEASE 确保之前的所有写入操作对其他线程可见
  __atomic_store_n(&job->state, ncclGroupJobDone, __ATOMIC_RELEASE);
  return arg;                                             // 返回传入的参数指针
}

// 函数实现：等待异步作业完成并清理资源
// 此函数会阻塞等待作业线程结束，然后清理作业资源
// 参数 job: 指向要完成的异步作业结构体
// 返回值: 作业的执行结果
ncclResult_t ncclAsyncJobComplete(struct ncclAsyncJob* job) {
  ncclResult_t ret;                                       // 声明返回值变量
  // 等待异步作业线程结束
  // pthread_join 会阻塞直到指定的线程结束
  PTHREADCHECK(pthread_join(job->thread, NULL), "pthread_join");
  // 检查作业是否执行失败
  if (job->result != ncclSuccess) {                       // 如果作业执行结果不是成功
    WARN("ncclAsyncJobComplete: job %p failed, job error %d", job, job->result);
    // 输出警告信息：作业指针地址和错误码
  }
  ret = job->result;                                      // 保存作业的执行结果
  if (job->destructor) job->destructor((void*)job);       // 如果提供了销毁函数，调用它清理资源
  return ret;                                             // 返回作业的执行结果
}

// 定义 NCCL 公共 API：ncclGroupStart
// 这是一个宏，用于导出 ncclGroupStart 函数为公共 API
NCCL_API(ncclResult_t, ncclGroupStart);
// 函数实现：开始一个 NCCL 组操作
// 在组操作中，多个集合通信操作可以被批量提交，以提高效率
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
ncclResult_t ncclGroupStart() {
  ncclResult_t ret = ncclSuccess;                         // 初始化返回值为成功状态
  NCCL_NVTX3_FUNC_RANGE;                                  // NVTX 性能追踪：标记函数范围，用于性能分析工具可视化

  // 调用内部组启动函数，增加组嵌套深度
  NCCLCHECK(ncclGroupStartInternal());                    // 执行组启动的内部逻辑
  TRACE_CALL("ncclGroupStart()");                         // 跟踪调用：记录函数调用信息
  return ret;                                             // 返回操作结果
}

// 定义 NCCL 公共 API：ncclGroupEnd
// 这是一个宏，用于导出 ncclGroupEnd 函数为公共 API
NCCL_API(ncclResult_t, ncclGroupEnd);
// 函数实现：结束一个 NCCL 组操作并提交所有批量操作
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
ncclResult_t ncclGroupEnd() {
  ncclResult_t ret = ncclSuccess;                         // 初始化返回值为成功状态
  NCCL_NVTX3_FUNC_RANGE;                                  // NVTX 性能追踪：标记函数范围
  // 调用内部组结束函数，提交所有批量操作
  NCCLCHECKGOTO(ncclGroupEndInternal(), ret, exit);       // 执行组结束的内部逻辑，出错时跳转到 exit
  TRACE_CALL("ncclGroupEnd()");                           // 跟踪调用：记录函数调用信息
exit:                                                      // 退出标签
  return ret;                                             // 返回操作结果
}

// 定义 NCCL 公共 API：ncclGroupSimulateEnd
// 这是一个宏，用于导出 ncclGroupSimulateEnd 函数为公共 API
// 该函数模拟组操作结束并返回性能模拟信息
NCCL_API(ncclResult_t, ncclGroupSimulateEnd, ncclSimInfo_t* simInfo);
// 函数实现：模拟结束一个 NCCL 组操作，返回性能模拟信息
// 参数 simInfo: 输出参数，返回模拟的性能信息
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
ncclResult_t ncclGroupSimulateEnd(ncclSimInfo_t* simInfo) {
  ncclResult_t ret = ncclSuccess;                         // 初始化返回值为成功状态
  NCCL_NVTX3_FUNC_RANGE;                                  // NVTX 性能追踪：标记函数范围
  // 调用内部组结束函数，传入模拟信息结构体
  NCCLCHECKGOTO(ncclGroupEndInternal(simInfo), ret, exit);// 执行组结束的内部逻辑（带模拟信息），出错时跳转到 exit
  TRACE_CALL("ncclGroupSimulateEnd()");                   // 跟踪调用：记录函数调用信息
exit:                                                      // 退出标签
  return ret;                                             // 返回操作结果
}

// 结构体定义：预连接作业
// 此结构体用于异步执行通信连接的预连接操作
struct ncclPreconnectJob {
  struct ncclAsyncJob base;                              // 基类：异步作业结构体，包含 next 指针用于队列
  struct ncclComm* comm;                                 // 关联的 NCCL 通信器指针
  bool* algoNeedConnect;                                 // 指向算法连接需求数组的指针
};

// 结构体定义：准备任务和集合通信预连接作业
// 此结构体用于异步执行任务准备和集合通信预连接操作
struct ncclPrepareTasksAndCollPreconnectJob {
  struct ncclAsyncJob base;                              // 基类：异步作业结构体
  struct ncclComm* comm;                                 // 关联的 NCCL 通信器指针
  ncclSimInfo_t* simInfo;                                // 指向模拟信息结构体的指针
};

// 函数实现：P2P（点对点）预连接函数
// 此函数在异步线程中执行，用于建立 P2P 通信连接
// 参数 job_: 指向异步作业结构体的指针（实际类型是 ncclPreconnectJob）
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
ncclResult_t ncclP2PPreconnectFunc(struct ncclAsyncJob* job_) {
  // 将基类指针转换为具体的派生类指针
  struct ncclPreconnectJob* job = (struct ncclPreconnectJob*)job_;
  // 获取关联的通信器指针
  struct ncclComm* comm = job->comm;

  // 设置 CUDA 设备为通信器关联的 GPU 设备
  CUDACHECK(cudaSetDevice(comm->cudaDev));               // 设置当前 CUDA 设备
  // 如果不是主线程且设置了 CPU 亲和性，则设置线程的 CPU 亲和性
  // isThreadMain 标识当前线程是否是主线程
  if (!job_->isThreadMain && CPU_COUNT(&comm->cpuAffinity)) // 检查是否非主线程且存在 CPU 亲和性设置
    sched_setaffinity(0, sizeof(cpu_set_t), &comm->cpuAffinity); // 设置当前线程的 CPU 亲和性

  // 执行 P2P 传输层设置，建立点对点连接
  // 参数：通信器、连接信息（NULL 表示自动）、连接模式（1 表示预连接）
  NCCLCHECK(ncclTransportP2pSetup(comm, NULL, 1));       // 设置 P2P 传输层连接
  return ncclSuccess;                                    // 返回成功状态
}

// 静态函数：集合通信预连接
// 此函数根据算法连接需求数组，为各种算法建立必要的连接
// 参数：
//   comm: NCCL 通信器指针
//   algoNeedConnect: 布尔数组，标识哪些算法需要建立连接
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
static ncclResult_t ncclCollPreconnect(struct ncclComm* comm, bool* algoNeedConnect) {
  // 遍历所有集合通信算法
  for (int i = 0; i < NCCL_NUM_ALGORITHMS; ++i) {        // NCCL_NUM_ALGORITHMS 是算法总数
    if (algoNeedConnect[i]) {                            // 如果该算法需要建立连接
      switch (i) {                                       // 根据算法类型进行不同的连接设置
        case NCCL_ALGO_RING: {                            // 环形算法
          NCCLCHECK(ncclTransportRingConnect(comm));     // 建立环形传输连接
          break;                                         // 跳出 switch
        }
        case NCCL_ALGO_TREE: {                            // 树形算法
          NCCLCHECK(ncclTransportTreeConnect(comm));     // 建立树形传输连接
          break;                                         // 跳出 switch
        }
        case NCCL_ALGO_NVLS: {                            // NVLS 算法（NVLink fabric）
          /* If we are using NVLS_TREE algo, we must mark NVLS algo to set up
           * NVLS intra-node buffer */
          /* 如果使用 NVLS_TREE 算法，必须标记 NVLS 算法以设置 NVLS 节点内缓冲区 */
          NCCLCHECK(ncclNvlsBufferSetup(comm));          // 设置 NVLS 缓冲区
          break;                                         // 跳出 switch
        }
        case NCCL_ALGO_NVLS_TREE: {                       // NVLS 树形算法
          NCCLCHECK(ncclNvlsTreeConnect(comm));          // 建立 NVLS 树形连接
          break;                                         // 跳出 switch
        }
        case NCCL_ALGO_COLLNET_CHAIN: {                  // 集合网络链式算法
          NCCLCHECK(ncclCollNetChainBufferSetup(comm));  // 设置集合网络链式缓冲区
          break;                                         // 跳出 switch
        }
        case NCCL_ALGO_COLLNET_DIRECT: {                 // 集合网络直连算法
          NCCLCHECK(ncclCollNetDirectBufferSetup(comm)); // 设置集合网络直连缓冲区
          break;                                         // 跳出 switch
        }
        case NCCL_ALGO_PAT: {                            // PAT 算法（Path Aware Transport）
          NCCLCHECK(ncclTransportPatConnect(comm));      // 建立 PAT 传输连接
          break;                                         // 跳出 switch
        }
        // Yes, it's a dead code.  That's fine...
        // 是的，这是死代码。这没关系...
        // coverity[dead_error_begin]
        // Coverity 静态分析工具的注释，抑制死代码警告
        default: {                                       // 未知算法类型
          NCCLCHECK(ncclInternalError);                  // 返回内部错误
        }
      }
    }
  }
  return ncclSuccess;                                    // 返回成功状态
}

// 函数实现：准备任务和集合通信预连接
// 此函数准备通信任务并在需要时建立集合通信连接
// 参数 job_: 指向异步作业结构体的指针（实际类型是 ncclPrepareTasksAndCollPreconnectJob）
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
ncclResult_t ncclPrepareTasksAndCollPreconnectFunc(struct ncclAsyncJob* job_) {
  // 将基类指针转换为具体的派生类指针
  struct ncclPrepareTasksAndCollPreconnectJob* job = (ncclPrepareTasksAndCollPreconnectJob*)job_;
  // 获取关联的通信器指针
  struct ncclComm* comm = job->comm;
  bool needConnect;                                       // 标识是否需要建立连接
  bool algoNeedConnect[NCCL_NUM_ALGORITHMS];              // 算法连接需求数组
  memset(algoNeedConnect, 0, sizeof(bool)*NCCL_NUM_ALGORITHMS); // 初始化数组为全 0
  CUDACHECK(cudaSetDevice(comm->cudaDev));               // 设置当前 CUDA 设备
  // 如果不是主线程且设置了 CPU 亲和性，则设置线程的 CPU 亲和性
  if (!job_->isThreadMain && CPU_COUNT(&comm->cpuAffinity)) sched_setaffinity(0, sizeof(cpu_set_t), &comm->cpuAffinity);
  // 准备通信任务，确定哪些算法需要建立连接
  NCCLCHECK(ncclPrepareTasks(comm, algoNeedConnect, &needConnect, job->simInfo));
  // 如果支持 CUDA Unified Memory 且需要连接，则执行集合通信预连接
  if (comm->cuMemSupport && needConnect) NCCLCHECK(ncclCollPreconnect(comm, algoNeedConnect));
  return ncclSuccess;                                    // 返回成功状态
}

// 函数实现：集合通信预连接函数
// 此函数在异步线程中执行，根据算法连接需求数组建立集合通信连接
// 参数 job_: 指向异步作业结构体的指针（实际类型是 ncclPreconnectJob）
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
ncclResult_t ncclCollPreconnectFunc(struct ncclAsyncJob* job_) {
  // 将基类指针转换为具体的派生类指针
  struct ncclPreconnectJob* job = (struct ncclPreconnectJob*)job_;
  // 获取关联的通信器指针
  struct ncclComm* comm = job->comm;
  ncclResult_t ret = ncclSuccess;                         // 初始化返回值为成功状态

  // 如果不是主线程，设置 CUDA 设备和 CPU 亲和性
  if (!job_->isThreadMain)                                // 检查是否非主线程
    CUDACHECK(cudaSetDevice(comm->cudaDev));             // 设置当前 CUDA 设备
  if (!job_->isThreadMain && CPU_COUNT(&comm->cpuAffinity)) // 检查是否非主线程且存在 CPU 亲和性设置
    sched_setaffinity(0, sizeof(cpu_set_t), &comm->cpuAffinity); // 设置当前线程的 CPU 亲和性
  // 执行集合通信预连接，出错时跳转到 fail 标签
  NCCLCHECKGOTO(ncclCollPreconnect(comm, job->algoNeedConnect), ret, fail);

exit:                                                      // 正常退出标签
  free(job->algoNeedConnect);                             // 释放算法连接需求数组的内存
  return ret;                                             // 返回操作结果
fail:                                                      // 失败退出标签
  goto exit;                                              // 跳转到 exit 标签进行清理
}

// 结构体定义：组对称作业
// 此结构体用于异步执行对称内存注册相关的操作
struct ncclGroupSymmetricJob {
  struct ncclAsyncJob base;                              // 基类：异步作业结构体
  struct ncclComm* comm;                                 // 关联的 NCCL 通信器指针
};

// 函数实现：组内对称内存注册
// 此函数在组操作中处理对称内存注册、通信器创建等任务
// 参数 job_: 指向异步作业结构体的指针（实际类型是 ncclGroupSymmetricJob）
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
ncclResult_t ncclCommGroupRegisterSymmetric(struct ncclAsyncJob* job_) {
  // 将基类指针转换为具体的派生类指针
  struct ncclGroupSymmetricJob* job = (struct ncclGroupSymmetricJob*)job_;
  // 获取关联的通信器指针
  struct ncclComm* comm = job->comm;
  ncclResult_t ret = ncclSuccess;                         // 初始化返回值为成功状态

  // 设置 CUDA 设备为通信器关联的 GPU 设备
  CUDACHECKGOTO(cudaSetDevice(comm->cudaDev), ret, fail); // 设置当前 CUDA 设备，出错时跳转到 fail

  // 处理所有设备内存注册任务
  while (!ncclIntruQueueEmpty(&comm->devrState.regTaskQueue)) { // 当注册任务队列不为空时循环
    struct ncclDevrRegTask* task = ncclIntruQueueDequeue(&comm->devrState.regTaskQueue); // 从队列中取出任务
    NCCLCHECKGOTO(ncclDevrWindowRegisterInGroup(          // 在组中注册设备内存窗口
      comm, task->userPtr, task->userSize, task->winFlags, task->outWinDev), // 传入通信器、用户指针、大小、标志、输出设备窗口
      ret, fail);                                         // 出错时跳转到 fail
    free(task);                                           // 释放任务结构体内存
  }

  // 处理所有设备通信器创建任务
  while (!ncclIntruQueueEmpty(&comm->devrState.commCreateTaskQueue)) { // 当通信器创建任务队列不为空时循环
    struct ncclDevrCommCreateTask* task = ncclIntruQueueDequeue(&comm->devrState.commCreateTaskQueue); // 从队列中取出任务
    NCCLCHECKGOTO(ncclDevrCommCreateInternal(             // 在设备端创建通信器
      comm, (struct ncclDevCommRequirements const*)task->reqs, task->outDevComm), // 传入通信器、需求、输出设备通信器
      ret, fail);                                         // 出错时跳转到 fail
    freeDevCommRequirements(task->reqs); // free additional task memory for reqs（释放 reqs 占用的额外任务内存）
    free(task);                                           // 释放任务结构体内存
  }

  // 处理所有集合通信引擎初始化任务
  while (!ncclIntruQueueEmpty(&comm->ceInitTaskQueue)) {  // 当 CE 初始化任务队列不为空时循环
    struct ncclCeInitTask* task = ncclIntruQueueDequeue(&comm->ceInitTaskQueue); // 从队列中取出任务
    NCCLCHECKGOTO(ncclCeInit(task->comm), ret, fail);     // 初始化集合通信引擎，出错时跳转到 fail
    free(task);                                           // 释放任务结构体内存
  }

exit:                                                      // 正常退出标签
  return ret;                                             // 返回操作结果
fail:                                                      // 失败退出标签
  goto exit;                                              // 跳转到 exit 标签
}

// 静态函数：执行内核启动
// 此函数负责启动组操作中所有通信器的 CUDA 内核
// 参数 head: 通信器链表的头指针
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
static ncclResult_t doLaunches(struct ncclComm* head) {
  ncclResult_t result = ncclSuccess;                     // 初始化返回值为成功状态
  struct ncclComm* cliqueHead = head;                     // 当前通信器派系的头指针
  struct ncclComm* cliqueNextHead;                        // 下一个通信器派系的头指针
  bool useBarrier = ncclParamLaunchMode == ncclLaunchModeGroup; // 是否使用屏障模式进行启动
  // This outer loop iterates over cliques of comms which are siblings of the
  // same global entity. We calculate a clique as all comms which have the same
  // `intraComm0` value.
  // 外层循环遍历通信器派系，这些派系是同一全局实体的兄弟节点
  // 我们将具有相同 intraComm0 值的所有通信器定义为一个派系
  do {
    struct ncclComm* comm = cliqueHead;                   // 当前通信器指针
    bool capturingYes = false, capturingNo = false;       // CUDA 图捕获状态标志
    do {
      // 检查是否正在进行 CUDA 图捕获，并设置对应标志
      (ncclCudaGraphValid(comm->planner.capturingGraph) ? capturingYes : capturingNo) = true;
      // 设置 CUDA 设备为通信器关联的 GPU 设备
      CUDACHECKGOTO(cudaSetDevice(comm->cudaDev), result, failure);
      // 准备启动操作，包括内核规划等
      NCCLCHECKGOTO(ncclLaunchPrepare(comm), result, failure);
      // 如果使用屏障模式，进入屏障同步
      if (useBarrier)                                      // 检查是否使用屏障
        ncclCommIntraBarrierIn(comm, 1);                  // 进入节点内屏障，传入参数 1 表示可能有更多轮次
      // 移动到下一个通信器
      comm = comm->groupNext[ncclGroupTaskTypeCollective]; // 获取集合通信类型的下一个通信器
    } while (comm != nullptr && comm->intraComm0 == cliqueHead->intraComm0); // 继续循环直到当前派系结束
    cliqueNextHead = comm;                                // 保存下一个派系的头指针

    // 检查是否同时存在捕获和非捕获的通信器
    if (capturingYes && capturingNo) {                    // 如果同时存在两种状态
      // We have entered barriers but are aborting without leaving them. Thus
      // these comms are permanently trashed. We need a good mechanism for
      // tracking and reporting that.
      // 我们已经进入屏障但正在中止而不离开它们。因此这些通信器永久损坏。
      // 我们需要一个良好的机制来跟踪和报告这种情况。
      WARN("Either none or all communicators in a ncclGroup() can be CUDA graph captured.");
      // 发出警告：ncclGroup() 中的通信器要么全部捕获，要么都不捕获
      result = ncclInvalidUsage;                          // 返回无效使用错误
      goto failure;                                       // 跳转到 failure 标签
    }

    // 多轮启动循环：可能需要多轮才能启动所有内核
    while (true) { // Iterate rounds of launches for clique.（迭代派系的启动轮次）
      bool moreRounds = false;                            // 是否需要更多轮次的标志
      comm = cliqueHead;                                  // 重置为派系头
      do { // Iterate clique members.（迭代派系成员）
        struct ncclComm* next = comm->groupNext[ncclGroupTaskTypeCollective]; // 保存下一个通信器指针
        if (useBarrier) {                                 // 如果使用屏障模式
          // Barrier reduction result tells us if this was the final round.
          // 屏障归约结果告诉我们这是否是最后一轮
          moreRounds = 0 != ncclCommIntraBarrierOut(comm); // 退出屏障并检查是否需要更多轮次
        } else {                                          // 不使用屏障模式
          moreRounds |= comm->planner.unlaunchedPlansHead != nullptr; // 检查是否还有未启动的计划
        }
        if (moreRounds) {                                 // 如果需要更多轮次
          // Pop next unlaunched kernel
          // 取出下一个未启动的内核计划
          struct ncclKernelPlan* plan = comm->planner.unlaunchedPlansHead; // 获取未启动计划链表头
          if (plan != nullptr) {                          // 如果存在未启动的计划
            comm->planner.unlaunchedPlansHead = plan->next; // 更新链表头为下一个计划
            CUDACHECKGOTO(cudaSetDevice(comm->cudaDev), result, failure); // 设置 CUDA 设备
            // 启动内核前的准备工作（不包含未捕获的 CUDA 操作）
            NCCLCHECKGOTO(ncclLaunchKernelBefore_NoUncapturedCuda(comm, plan), result, failure);
            if (plan->isCeColl) {                         // 如果是集合通信引擎操作
              NCCLCHECKGOTO(ncclLaunchCeColl(comm, plan), result, failure); // 启动 CE 集合通信内核
            } else {                                      // 普通内核
              NCCLCHECKGOTO(ncclLaunchKernel(comm, plan), result, failure); // 启动普通 NCCL 内核
            }
          }
          // Barrier reduction input indicates if we require further rounds.
          // 屏障归约输入指示我们是否需要更多轮次
          if (useBarrier) ncclCommIntraBarrierIn(comm, comm->planner.unlaunchedPlansHead != nullptr ? 1 : 0); // 进入屏障，如果有更多计划则传入 1
          if (plan != nullptr) {                          // 如果成功启动了计划
            // 启动内核后的清理工作（不包含 CUDA 操作）
            NCCLCHECKGOTO(ncclLaunchKernelAfter_NoCuda(comm, plan), result, failure);
          }
        } else { // Final round.（最后一轮）
          CUDACHECKGOTO(cudaSetDevice(comm->cudaDev), result, failure); // 设置 CUDA 设备
          NCCLCHECKGOTO(ncclLaunchFinish(comm), result, failure); // 完成启动操作
        }
        comm = next;                                      // 移动到下一个通信器
      } while (comm != cliqueNextHead);                   // 继续直到遍历完当前派系
      if (!moreRounds) break;                             // 如果不需要更多轮次，跳出循环
    }
    cliqueHead = cliqueNextHead;                          // 移动到下一个派系
  } while (cliqueHead != nullptr);                        // 继续直到所有派系处理完毕
failure:                                                  // 失败标签
  return result;                                          // 返回操作结果
}

// 静态内联函数：重置本地组作业状态
// 此函数在组操作完成后重置所有线程局部变量，为下一次组操作做准备
static inline void groupLocalResetJobState() {
  ncclGroupError = ncclSuccess;                           // 重置组错误状态为成功
  // 遍历所有任务类型，重置通信器链表头指针
  for (int type = 0; type < ncclGroupTaskTypeNum; ++type) // 遍历所有任务类型
    ncclGroupCommHead[type] = NULL;                       // 将每种类型的通信器链表头置空
  ncclGroupCommPreconnectHead = NULL;                     // 重置预连接通信器链表头为空
  ncclGroupBlocking = -1;                                 // 重置阻塞模式为默认值（未确定）

  ncclIntruQueueConstruct(&ncclAsyncJobs);                // 构造（初始化）异步作业队列
  return;                                                 // 返回
}

// 静态函数：清理组操作资源
// 此函数在组操作失败或完成后清理所有相关资源
// 参数：
//   groupCommHeadPtr: 通信器链表头指针数组
//   asyncJobsPtr: 异步作业队列指针
//   error: 错误码，用于设置通信器的异步错误状态
static void groupCleanup(struct ncclComm** groupCommHeadPtr, struct ncclIntruQueue<struct ncclAsyncJob, &ncclAsyncJob::next>* asyncJobsPtr, ncclResult_t error) {
  struct ncclComm* comm;                                  // 当前通信器指针
  // 遍历所有任务类型
  for (int type = 0; type < ncclGroupTaskTypeNum; ++type) { // 遍历所有任务类型
    comm = groupCommHeadPtr[type];                        // 获取当前类型的通信器链表头
    // reset groupCommHeadPtr[type]
    // 重置 groupCommHeadPtr[type] 为空指针
    groupCommHeadPtr[type] = nullptr;                     // 将通信器链表头置空
    while (comm != nullptr) {                             // 遍历通信器链表
      struct ncclComm* next = comm->groupNext[type];      // 保存下一个通信器指针
      (void)ncclGroupCommLeave(comm, type); // overwrites comm->groupNext（调用组通信器离开函数，会覆盖 comm->groupNext）
      // We don't know if preconnect succeeded or happened at all, so clear
      // the flags that let `taskAppend()` skip over checking if preconnect
      // is needed.
      // 我们不知道预连接是否成功或是否发生了，所以清除那些让 `taskAppend()` 跳过检查预连接是否需要的标志
      if (type == ncclGroupTaskTypeCollective) {          // 如果是集合通信类型
        comm->preconnectNext = reinterpret_cast<struct ncclComm*>(0x1); // 重置预连接下一个指针为特殊值
        for (int i = 0; i < comm->nRanks; i++) {          // 遍历所有 rank
          comm->connectSend[i] = 0UL;                     // 清除发送连接标志
          comm->connectRecv[i] = 0UL;                     // 清除接收连接标志
        }
        // Reclaim abandoned kernel plan memory. Note ncclWork structs were already
        // reclaimed by a `ncclMemoryStackPop(&comm->memScoped)` during `ncclGroupCommLeave()`.
        // 回收废弃的内核计划内存。注意 ncclWork 结构体已经在 `ncclGroupCommLeave()` 期间通过 `ncclMemoryStackPop(&comm->memScoped)` 回收了
        while (!ncclIntruQueueEmpty(&comm->planner.planQueue)) { // 当计划队列不为空时循环
          struct ncclKernelPlan* plan = ncclIntruQueueDequeue(&comm->planner.planQueue); // 从队列中取出计划
          // Persistent plans will be reclaimed via the callbackQueue when the
          // graph drops its UserObject reference.
          // 持久化计划将通过 callbackQueue 在图释放其 UserObject 引用时回收
          if (!plan->persistent) {                        // 如果不是持久化计划
            while (!ncclIntruQueueEmpty(&plan->proxyOpQueue)) { // 回收代理操作队列
              struct ncclProxyOp* pxop = ncclIntruQueueDequeue(&plan->proxyOpQueue); // 取出代理操作
              ncclMemoryPoolFree(&comm->memPool_ncclProxyOp, pxop); // 释放代理操作内存
            }
            ncclMemoryPoolFree(&comm->memPool_ncclKernelPlan, plan); // 释放内核计划内存
          }
        }

        { // Reset comm->planner to empty.（重置 comm->planner 为空）
          ncclKernelPlanner::Peer* tmp = comm->planner.peers; // 临时保存 peers 指针
          memset(&comm->planner, 0, sizeof(comm->planner)); // 清零 planner 结构体
          comm->planner.peers = tmp;                      // 恢复 peers 指针
          if (comm->planner.peers != NULL) memset(comm->planner.peers, 0, comm->nRanks * sizeof(comm->planner.peers[0])); // 清零 peers 数组
        }
      }

      // 如果是非阻塞模式，设置异步错误状态
      if (!comm->config.blocking)                         // 检查是否是非阻塞模式
        (void)ncclCommSetAsyncError(comm, error);         // 设置通信器的异步错误状态
      comm = next;                                        // 移动到下一个通信器
    }
  }

  /* reset everything */
  /* 重置所有内容 */
  // 清理所有异步作业
  while (!ncclIntruQueueEmpty(asyncJobsPtr)) {            // 当异步作业队列不为空时循环
    struct ncclAsyncJob* job = ncclIntruQueueDequeue(asyncJobsPtr); // 从队列中取出作业
    // 如果是非销毁作业且关联非阻塞通信器，设置异步错误状态
    if (!job->destroyFlag && job->comm && !job->comm->config.blocking) // 检查作业标志和通信器模式
      (void) ncclCommSetAsyncError(job->comm, error);     // 设置通信器的异步错误状态
    if (job->undo) job->undo(job);                        // 如果有回滚函数，调用它
    if (job->destructor) job->destructor((void*)job);     // 如果有销毁函数，调用它
  }

  return;                                                 // 返回
}

// 静态函数：启动异步作业
// 此函数负责创建线程执行异步作业，并等待所有作业完成
// 参数：
//   asyncJobsMain: 异步作业队列指针
//   groupAbortFlag: 组中止标志指针
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
static ncclResult_t asyncJobLaunch(struct ncclIntruQueue<struct ncclAsyncJob, &ncclAsyncJob::next> *asyncJobsMain, volatile bool *groupAbortFlag) {
  ncclResult_t ret = ncclSuccess;                         // 初始化返回值为成功状态
  bool jobsDone = false;                                  // 作业是否全部完成的标志
  bool errorJobAbortFlag = false;                         // 是否因错误而需要中止的标志

  // 检查异步作业队列是否为空
  if (!ncclIntruQueueEmpty(asyncJobsMain)) {              // 如果队列不为空
    struct ncclAsyncJob* job = ncclIntruQueueHead(asyncJobsMain); // 获取队列头部的作业
    // 如果只有一个作业，直接在主线程中执行
    if (job->next == nullptr) {                           // 如果只有一个作业（没有 next）
      job->isThreadMain = true;                           // 标记此作业在主线程中执行
      ncclAsyncJobMain(job);                              // 直接调用作业执行函数
      job->state = ncclGroupJobJoined;                    // 设置作业状态为已加入（已完成）
      return job->result;                                 // 返回作业执行结果
    }

    // 为每个作业创建一个线程
    do {
      // 创建线程执行异步作业
      PTHREADCHECKGOTO(pthread_create(&job->thread, nullptr, ncclAsyncJobMain, job), "pthread_create", ret, fail);
      job = job->next;                                    // 移动到下一个作业
    } while (job != nullptr);                             // 继续直到所有作业都创建了线程

    // 等待所有作业完成
    do {
      jobsDone = true;                                    // 假设所有作业已完成
      job = ncclIntruQueueHead(asyncJobsMain);            // 重新从队列头开始
      do {
        // 原子加载作业状态
        ncclGroupJobState_t state = __atomic_load_n(&job->state, __ATOMIC_ACQUIRE); // 使用获取语义加载状态
        if (state == ncclGroupJobRunning) {               // 如果作业仍在运行
          jobsDone = false;                               // 标记为未全部完成
        } else if (state == ncclGroupJobDone) {           // 如果作业已完成
          int err;                                        // 错误码变量
          if ((err = pthread_join(job->thread, nullptr)) != 0) { // 等待线程结束
            WARN("Error waiting for pthread_join: %s", strerror(err)); // 输出警告信息
            ret = ncclSystemError;                        // 设置返回值为系统错误
          }
          job->state = ncclGroupJobJoined;                // 设置作业状态为已加入
          // 如果作业执行失败且当前返回值为成功，更新返回值
          if (job->result != ncclSuccess && ret == ncclSuccess) { // 检查作业结果和当前返回值
            ret = job->result;                            // 更新返回值为作业的错误码
            errorJobAbortFlag = true;                     // 设置错误中止标志
          }
        } else {
          /* safety check */
          /* 安全检查 */
          assert(state == ncclGroupJobJoined);            // 断言状态必须是已加入
        }

        // 检查是否需要中止作业
        if (!job->destroyFlag && (__atomic_load_n(groupAbortFlag, __ATOMIC_ACQUIRE) || errorJobAbortFlag == true)) {
          // 如果作业未销毁且组中止标志或错误中止标志被设置
          __atomic_store_n(job->abortFlag, 1, __ATOMIC_RELEASE); // 设置主机侧中止标志
          __atomic_store_n(job->abortFlagDev, 1, __ATOMIC_RELEASE); // 设置设备侧中止标志
          if (job->childAbortFlag) {                      // 如果有子通信器中止标志
            __atomic_store_n(job->childAbortFlag, 1, __ATOMIC_RELEASE); // 设置子通信器主机侧中止标志
            __atomic_store_n(job->childAbortFlagDev, 1, __ATOMIC_RELEASE); // 设置子通信器设备侧中止标志
          }
        }

        job = job->next;                                  // 移动到下一个作业
      } while (job != nullptr);                           // 继续直到遍历完所有作业
      
      // Let preconnect threads progress.
      // 让预连接线程有机会执行
      if (jobsDone == false) 
        usleep(1);                   // 如果作业未全部完成，休眠 1 微秒
    } while (jobsDone == false);                          // 继续直到所有作业完成

    // 检查是否有错误发生
    if (ret != ncclSuccess) 
        goto fail;                   // 如果有错误，跳转到 fail 标签
  }

exit:                                                      // 正常退出标签
  return ret;                                             // 返回操作结果
fail:                                                      // 失败退出标签
  goto exit;                                              // 跳转到 exit 标签
}

// 定义 NCCL 参数：单进程内存注册启用标志
// 参数名称：SINGLE_PROC_MEM_REG_ENABLE，默认值为 0（禁用）
// 此参数控制是否在单进程模式下启用内存注册功能
NCCL_PARAM(SingleProcMemRegEnable, "SINGLE_PROC_MEM_REG_ENABLE", 0);

// 静态函数：准备任务和集合通信预连接
// 此函数根据参数决定是异步还是同步执行任务准备和预连接
// 参数：
//   comm: NCCL 通信器指针
//   simInfo: 模拟信息指针
//   asyncCollJobs: 异步集合作业队列指针
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
static ncclResult_t ncclPrepareTasksAndCollPreconnect(struct ncclComm* comm, ncclSimInfo_t* simInfo, struct ncclIntruQueue<struct ncclAsyncJob, &ncclAsyncJob::next>* asyncCollJobs) {
  // 检查是否启用单进程内存注册
  if (ncclParamSingleProcMemRegEnable()) {               // 如果启用了单进程内存注册
    struct ncclPrepareTasksAndCollPreconnectJob* job;     // 定义作业结构体指针
    NCCLCHECK(ncclCalloc(&job, 1));                       // 分配并清零作业内存
    job->base.func = ncclPrepareTasksAndCollPreconnectFunc; // 设置执行函数指针
    job->base.undo = nullptr;                             // 设置回滚函数为空
    job->base.destructor = free;                          // 设置销毁函数为 free
    job->base.state = ncclGroupJobRunning;                // 设置作业状态为运行中
    job->base.abortFlag = comm->abortFlag;                // 设置主机侧中止标志
    job->base.abortFlagDev = comm->abortFlagDev;          // 设置设备侧中止标志
    job->comm = comm;                                     // 设置关联的通信器
    job->simInfo = simInfo;                               // 设置模拟信息指针
    ncclIntruQueueEnqueue(asyncCollJobs, &job->base);     // 将作业加入异步队列
  } else {                                                // 未启用单进程内存注册
    bool needConnect = false;                             // 是否需要连接的标志
    bool algoNeedConnect[NCCL_NUM_ALGORITHMS];            // 算法连接需求数组
    memset(algoNeedConnect, 0, sizeof(bool) * NCCL_NUM_ALGORITHMS); // 初始化数组为全 0

    CUDACHECK(cudaSetDevice(comm->cudaDev));             // 设置当前 CUDA 设备
    // 准备通信任务，确定哪些算法需要建立连接
    NCCLCHECK(ncclPrepareTasks(comm, algoNeedConnect, &needConnect, simInfo));

    // 如果支持 CUDA Unified Memory 且需要连接，创建预连接作业
    if (comm->cuMemSupport && needConnect) {              // 检查是否支持 CU Mem 且需要连接
      ncclResult_t ret;                                  // 定义返回值变量
      struct ncclPreconnectJob* job;                     // 定义预连接作业指针
      NCCLCHECK(ncclCalloc(&job, 1));                    // 分配并清零作业内存
      job->base.func = ncclCollPreconnectFunc;            // 设置执行函数指针
      job->base.undo = nullptr;                          // 设置回滚函数为空
      job->base.destructor = free;                       // 设置销毁函数为 free
      job->base.state = ncclGroupJobRunning;             // 设置作业状态为运行中
      job->base.abortFlag = comm->abortFlag;             // 设置主机侧中止标志
      job->base.abortFlagDev = comm->abortFlagDev;       // 设置设备侧中止标志
      job->comm = comm;                                  // 设置关联的通信器
      // 分配算法连接需求数组内存
      if ((ret = ncclCalloc(&job->algoNeedConnect, NCCL_NUM_ALGORITHMS))) { // 分配内存
        free(job);                                       // 如果分配失败，释放作业内存
        NCCLCHECK(ret);                                  // 检查并返回错误
      }
      memcpy(job->algoNeedConnect, algoNeedConnect, sizeof(bool) * NCCL_NUM_ALGORITHMS); // 复制算法连接需求数组
      ncclIntruQueueEnqueue(asyncCollJobs, &job->base);  // 将作业加入异步队列
    }
  }
  return ncclSuccess;                                    // 返回成功状态
}

// 静态函数：启动组操作
// 此函数是组操作的核心执行函数，负责处理所有类型的组操作任务
// 参数：
//   job_: 指向异步作业结构体的指针（实际类型是 ncclGroupJob）
//   simInfo: 模拟信息指针，默认为 NULL
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
static ncclResult_t groupLaunch(struct ncclAsyncJob *job_, ncclSimInfo_t* simInfo = NULL) {
  ncclResult_t ret = ncclSuccess;                         // 初始化返回值为成功状态
  struct ncclGroupJob *gjob = (struct ncclGroupJob*) job_; // 将基类指针转换为组作业指针
  struct ncclComm **groupCommHeadMain = gjob->groupCommHead; // 获取通信器链表头数组
  struct ncclComm *groupCommPreconnectHeadMain = gjob->groupCommPreconnectHead; // 获取预连接通信器链表头
  struct ncclIntruQueue<struct ncclAsyncJob, &ncclAsyncJob::next> *asyncJobsMain = &gjob->asyncJobs; // 获取异步作业队列
  bool *groupAbortFlag = &gjob->abortFlag;                // 获取组中止标志指针

  // 如果不是模拟模式且存在需要预连接的通信器，创建 P2P 预连接作业
  if (!simInfo && groupCommPreconnectHeadMain != nullptr) { // 检查是否非模拟模式且有预连接通信器
    struct ncclComm* comm = groupCommPreconnectHeadMain;  // 获取预连接通信器链表头
    do {
      struct ncclPreconnectJob* job;                      // 定义预连接作业指针
      NCCLCHECKGOTO(ncclCalloc(&job, 1), ret, fail);      // 分配并清零作业内存
      job->base.func = ncclP2PPreconnectFunc;             // 设置执行函数为 P2P 预连接函数
      job->base.undo = nullptr;                           // 设置回滚函数为空
      job->base.destructor = free;                        // 设置销毁函数为 free
      job->base.state = ncclGroupJobRunning;              // 设置作业状态为运行中
      job->base.abortFlag = comm->abortFlag;              // 设置主机侧中止标志
      job->base.abortFlagDev = comm->abortFlagDev;        // 设置设备侧中止标志
      job->comm = comm;                                   // 设置关联的通信器
      ncclIntruQueueEnqueue(asyncJobsMain,  (struct ncclAsyncJob*)job); // 将作业加入异步队列

      struct ncclComm* next = comm->preconnectNext;       // 保存下一个预连接通信器指针
      comm->preconnectNext = reinterpret_cast<struct ncclComm*>(0x1); // 重置预连接下一个指针为特殊值
      comm = next;                                        // 移动到下一个通信器
    } while (comm != nullptr);                            // 继续直到所有预连接通信器处理完毕
  }

  // 启动所有异步作业并等待完成
  NCCLCHECKGOTO(asyncJobLaunch(asyncJobsMain, groupAbortFlag), ret, fail); // 执行异步作业启动

  // only loop through sym alloc and register tasks
  // 只循环处理对称内存分配和注册任务
  for (int type = ncclGroupTaskTypeSymRegister; type <= ncclGroupTaskTypeSymRegister; ++type) { // 只处理对称注册类型
    if (groupCommHeadMain[type]) {                        // 如果存在该类型的通信器
      struct ncclComm* cliqueHead = groupCommHeadMain[type]; // 获取当前派系的头
      struct ncclComm* comm = NULL;                       // 定义当前通信器指针
      struct ncclIntruQueue<struct ncclAsyncJob, &ncclAsyncJob::next> asyncSymJobs; // 定义异步作业队列
      ncclIntruQueueConstruct(&asyncSymJobs);             // 构造队列
      do {
        comm = cliqueHead;                                // 从派系头开始
        do {
          struct ncclGroupSymmetricJob* job;              // 定义对称作业指针
          NCCLCHECKGOTO(ncclCalloc(&job, 1), ret, fail);  // 分配并清零作业内存
          job->base.func = ncclCommGroupRegisterSymmetric; // 设置执行函数为对称注册函数
          job->base.undo = nullptr;                       // 设置回滚函数为空
          job->base.destructor = free;                    // 设置销毁函数为 free
          job->base.state = ncclGroupJobRunning;          // 设置作业状态为运行中
          job->base.abortFlag = comm->abortFlag;          // 设置主机侧中止标志
          job->base.abortFlagDev = comm->abortFlagDev;    // 设置设备侧中止标志
          job->comm = comm;                               // 设置关联的通信器
          ncclIntruQueueEnqueue(&asyncSymJobs, (struct ncclAsyncJob*)job); // 将作业加入异步队列
          comm = comm->groupNext[type];                   // 移动到下一个通信器
        } while (comm != nullptr && comm->intraComm0 == cliqueHead->intraComm0); // 继续直到当前派系结束
        // 启动当前派系的所有对称作业
        NCCLCHECKGOTO(asyncJobLaunch(&asyncSymJobs, groupAbortFlag), ret, fail); // 执行异步作业启动
        // 清理已完成的作业
        while (!ncclIntruQueueEmpty(&asyncSymJobs)) {     // 当队列不为空时循环
          struct ncclAsyncJob* job = ncclIntruQueueDequeue(&asyncSymJobs); // 从队列中取出作业
          if (job->destructor) job->destructor((void*)job); // 如果有销毁函数，调用它
        }
        cliqueHead = comm;                                // 移动到下一个派系
      } while (cliqueHead != nullptr);                    // 继续直到所有派系处理完毕
    }
  }

  /* Connect channels at runtime if cumem is supported */
  /* 如果支持 CUDA Unified Memory，在运行时连接通道 */
  if (groupCommHeadMain[ncclGroupTaskTypeCollective] != nullptr) { // 如果存在集合通信类型的通信器
    struct ncclComm* cliqueHead = groupCommHeadMain[ncclGroupTaskTypeCollective]; // 获取集合通信派系头
    struct ncclComm* comm = NULL;                         // 定义当前通信器指针
    struct ncclIntruQueue<struct ncclAsyncJob, &ncclAsyncJob::next> asyncCollJobs; // 定义异步集合作业队列
    ncclIntruQueueConstruct(&asyncCollJobs);              // 构造队列
    do {
      // We need to preconnect connections for collectives clique by clique to avoid
      // race condition for split shared comms which can connect the same connections
      // at the same time.
      // 我们需要为集合通信逐派系地进行预连接，以避免分裂共享通信器的竞态条件，
      // 这些通信器可能同时连接相同的连接。
      comm = cliqueHead;                                  // 从派系头开始
      do {
        // 准备任务和集合通信预连接
        NCCLCHECKGOTO(ncclPrepareTasksAndCollPreconnect(comm, simInfo, &asyncCollJobs), ret, fail);
        comm = comm->groupNext[ncclGroupTaskTypeCollective]; // 移动到下一个通信器
      } while (comm != nullptr && comm->intraComm0 == cliqueHead->intraComm0); // 继续直到当前派系结束
      // connect（连接）
      // 启动当前派系的所有预连接作业
      NCCLCHECKGOTO(asyncJobLaunch(&asyncCollJobs, groupAbortFlag), ret, fail); // 执行异步作业启动
      // 清理已完成的作业
      while (!ncclIntruQueueEmpty(&asyncCollJobs)) {      // 当队列不为空时循环
        struct ncclAsyncJob* job = ncclIntruQueueDequeue(&asyncCollJobs); // 从队列中取出作业
        if (job->destructor) job->destructor((void*)job); // 如果有销毁函数，调用它
      }
      cliqueHead = comm;                                  // 移动到下一个派系
    } while (cliqueHead != nullptr);                      // 继续直到所有派系处理完毕

    // done with all buffer allocation, start registration and enqueue
    // 完成所有缓冲区分配，开始注册和入队
    comm = groupCommHeadMain[ncclGroupTaskTypeCollective]; // 重新从集合通信链表头开始
    do {
      CUDACHECKGOTO(cudaSetDevice(comm->cudaDev), ret, fail); // 设置 CUDA 设备
      NCCLCHECKGOTO(ncclTasksRegAndEnqueue(comm), ret, fail); // 注册并入队任务
      comm = comm->groupNext[ncclGroupTaskTypeCollective]; // 移动到下一个通信器
    } while (comm);                                       // 继续直到所有通信器处理完毕
  }

  // 如果不是模拟模式且存在集合通信类型的通信器，启动内核
  if ((!simInfo) && (groupCommHeadMain[ncclGroupTaskTypeCollective] != nullptr)) { // 检查是否非模拟模式且有集合通信器
    NCCLCHECKGOTO(doLaunches(groupCommHeadMain[ncclGroupTaskTypeCollective]), ret, fail); // 启动所有内核
  }

  // 清理剩余的异步作业
  while (!ncclIntruQueueEmpty(asyncJobsMain)) {           // 当异步作业队列不为空时循环
    struct ncclAsyncJob* job = ncclIntruQueueDequeue(asyncJobsMain); // 从队列中取出作业
    // 如果是非销毁作业、关联非阻塞通信器且没有集合通信器，设置异步错误状态
    if (!job->destroyFlag && job->comm && !job->comm->config.blocking && groupCommHeadMain[ncclGroupTaskTypeCollective] == nullptr) // 检查条件
      (void) ncclCommSetAsyncError(job->comm, ret);      // 设置通信器的异步错误状态
    if (job->destructor) 
      job->destructor((void*)job);     // 如果有销毁函数，调用它
  }

  // 处理所有类型的通信器，执行资源回收和清理
  for (int type = 0; type < ncclGroupTaskTypeNum; ++type) { // 遍历所有任务类型
    while (groupCommHeadMain[type] != nullptr) {          // 当通信器链表不为空时循环
      struct ncclComm* comm = groupCommHeadMain[type];    // 获取当前通信器
      struct ncclComm* next = comm->groupNext[type];      // 保存下一个通信器指针
      // Poll for callbacks sent to us from other threads. Typically these free
      // resources from to our memory pools and UB
      // 轮询来自其他线程的回调。通常这些会释放我们的内存池和 UB 中的资源
      if (comm->reclaimSteps == GROUP_MAX_RECLAIM_STEPS) { // 如果达到最大回收步数
        NCCLCHECKGOTO(ncclCommPollCallbacks(comm, /*waitSome=*/false), ret, fail); // 轮询回调
        comm->reclaimSteps = 0;                           // 重置回收步数
      } else {                                            // 未达到最大回收步数
        comm->reclaimSteps++;                             // 增加回收步数
      }
      (void)ncclGroupCommLeave(comm, type);               // 调用组通信器离开函数
      if (!comm->config.blocking) {                       // 如果是非阻塞模式
        (void)ncclCommSetAsyncError(comm, ret);           // 设置通信器的异步错误状态
      }
      groupCommHeadMain[type] = next;                     // 移动到下一个通信器
    }
  }

exit:                                                      // 正常退出标签
  return ret;                                             // 返回操作结果
fail:                                                      // 失败退出标签
  groupCleanup(gjob->groupCommHead, &gjob->asyncJobs, ret); // 清理组操作资源
  goto exit;                                              // 跳转到 exit 标签
}

// 静态函数：以非阻塞模式启动组操作
// 此函数是 groupLaunch 的非阻塞包装器
// 参数 job_: 指向异步作业结构体的指针（实际类型是 ncclGroupJob）
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
static ncclResult_t groupLaunchNonBlocking(struct ncclAsyncJob *job_) {
  return groupLaunch(job_ /* estimatedTime = NULL */);   // 调用 groupLaunch，simInfo 参数为 NULL
}

// 函数实现：组操作内部结束函数
// 此函数是 ncclGroupEnd 和 ncclGroupSimulateEnd 的内部实现
// 参数 simInfo: 模拟信息指针，用于返回性能模拟数据
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
ncclResult_t ncclGroupEndInternal(ncclSimInfo_t* simInfo) {
  ncclResult_t ret = ncclSuccess;                         // 初始化返回值为成功状态
  ncclSimInfo_t internalSimInfo = NCCL_SIM_INFO_INITIALIZER; // 初始化内部模拟信息结构
  ncclSimInfo_t* internalSimInfoPtr = NULL;               // 内部模拟信息指针
  size_t realSize = 0;                                    // 实际数据大小
  bool hasCommHead = false;                               // 是否存在通信器链表头的标志
  ncclGroupJob* groupJob = NULL;                          // 组作业指针

  internalSimInfo.magic = 0;                              // 初始化魔数为 0，用于验证

  // 检查是否在组调用中
  if (ncclGroupDepth == 0) {                              // 如果组嵌套深度为 0
    WARN("ncclGroupEnd: not in a group call.");           // 输出警告
    ret = ncclInvalidUsage;                              // 返回无效使用错误
    goto exit;                                            // 跳转到 exit 标签
  }

  // 处理性能分析器的组深度
  if (ncclProfilerApiState.profilerGroupDepth > 0) {      // 如果性能分析器组深度大于 0
    ncclProfilerApiState.profilerGroupDepth--;           // 减少性能分析器组深度
  }
  if (ncclProfilerApiState.profilerGroupDepth == 0) {     // 如果性能分析器组深度降为 0
    NCCLCHECK(ncclProfilerRecordGroupApiEventState(ncclProfilerGroupEndApiStart)); // 记录组结束 API 事件状态
  }

  // 减少组嵌套深度，如果仍大于 0 则直接返回
  if ((--ncclGroupDepth) > 0)                             // 减少组嵌套深度
    goto exit;                                            // 如果仍有嵌套，直接返回

  // 检查组操作期间是否有错误发生
  if ((ret = ncclGroupError) != ncclSuccess)             // 获取组错误状态
    goto fail;                                            // 如果有错误，跳转到 fail 标签

  // 处理模拟信息
  if (simInfo) {                                          // 如果提供了模拟信息结构体
    memcpy((void*)&realSize, (void*)&simInfo->size, sizeof(size_t)); // 获取用户提供的大小
    realSize = realSize > sizeof(ncclSimInfo_t) ? sizeof(ncclSimInfo_t) : realSize; // 限制为结构体实际大小
    memcpy((void*)&internalSimInfo, (void*)simInfo, realSize); // 复制模拟信息
    if (internalSimInfo.magic != 0x74685283) {            // 检查魔数是否正确
      WARN("ncclSimInfo_t argument not initialized via NCCL_SIM_INFO_INITIALIZER"); // 输出警告
      ret = ncclInvalidArgument;                          // 返回无效参数错误
      goto fail;                                          // 跳转到 fail 标签
    }
    internalSimInfoPtr = &internalSimInfo;                // 设置内部模拟信息指针
  }

  // 检查是否存在任何类型的通信器
  for (int type = 0; type < ncclGroupTaskTypeNum; ++type) { // 遍历所有任务类型
    if (ncclGroupCommHead[type]) {                        // 如果存在该类型的通信器
      hasCommHead = true;                                 // 设置标志为 true
      break;                                              // 跳出循环
    }
  }

  // 创建组作业结构
  NCCLCHECKGOTO(ncclCalloc(&groupJob, 1), ret, fail);     // 分配并清零组作业内存
  ncclIntruQueueConstruct(&groupJob->asyncJobs);          // 构造异步作业队列
  groupJob->groupRefCount = 0;                            // 初始化引用计数为 0
  groupJob->nonBlockingInit = false;                      // 初始化非阻塞标志为 false
  memcpy(groupJob->groupCommHead, ncclGroupCommHead, sizeof(ncclGroupCommHead)); // 复制通信器链表头数组
  //taskAppend会设置ncclGroupCommPreconnectHead
  groupJob->groupCommPreconnectHead = ncclGroupCommPreconnectHead; // 复制预连接通信器链表头
  groupJob->groupError = ncclSuccess;                     // 初始化组错误为成功
  groupJob->abortFlag = false;                            // 初始化中止标志为 false
  groupJob->joined = false;                               // 初始化已加入标志为 false
  ncclIntruQueueTransfer(&groupJob->asyncJobs, &ncclAsyncJobs); // 转移异步作业队列

  // 检查是否有任务需要执行
  if (hasCommHead || !ncclIntruQueueEmpty(&groupJob->asyncJobs) || ncclGroupCommPreconnectHead != nullptr) {
    // 如果有通信器、异步作业或预连接通信器
    /* make sure ncclGroupBlocking has been set. */
    /* 确保 ncclGroupBlocking 已经被设置 */
    assert(ncclGroupBlocking == 0 || ncclGroupBlocking == 1); // 断言阻塞模式必须是 0 或 1
    
    if (ncclGroupBlocking == 0) {                         // 如果是非阻塞模式
      /* nonblocking group */
      /* 非阻塞组 */
      // 处理异步作业队列
      if (!ncclIntruQueueEmpty(&groupJob->asyncJobs)) {    // 如果异步作业队列不为空
        ncclAsyncJob* job = ncclIntruQueueHead(&groupJob->asyncJobs); // 获取队列头
        do {
          NCCLCHECKGOTO(ncclCommSetAsyncError(job->comm, ncclInProgress), ret, fail); // 设置异步错误为进行中
          if (job->comm->groupJob == NULL) {               // 如果通信器尚未关联组作业
            job->comm->groupJob = groupJob;               // 关联组作业到通信器
            groupJob->groupRefCount++;                    // 增加组引用计数
          }
          job = job->next;                                // 移动到下一个作业
        } while (job);                                    // 继续直到所有作业处理完毕
      }

      // 处理通信器链表
      for (int type = 0; type < ncclGroupTaskTypeNum; ++type) { // 遍历所有任务类型
        if (ncclGroupCommHead[type]) {                    // 如果存在该类型的通信器
          ncclComm_t comm = ncclGroupCommHead[type];      // 获取通信器链表头
          do {
            NCCLCHECKGOTO(ncclCommSetAsyncError(comm, ncclInProgress), ret, fail); // 设置异步错误为进行中
            /* link group job to communicators. */
            /* 将组作业链接到通信器 */
            if (comm->groupJob == NULL) {                  // 如果通信器尚未关联组作业
              comm->groupJob = groupJob;                  // 关联组作业到通信器
              groupJob->groupRefCount++;                  // 增加组引用计数
            }
            comm = comm->groupNext[type];                 // 移动到下一个通信器
          } while (comm);                                  // 继续直到所有通信器处理完毕
        }
      }

      // 创建线程执行非阻塞组操作
      groupJob->base.func = groupLaunchNonBlocking;       // 设置执行函数
      PTHREADCHECKGOTO(pthread_create(&groupJob->base.thread, NULL, ncclAsyncJobMain, (void*)&groupJob->base), "pthread_create", ret, fail); // 创建线程
      //不等待线程结束，直接返回
      groupJob->nonBlockingInit = true;                   // 设置非阻塞初始化标志为 true
      ret = ncclInProgress;                               // 返回进行中状态
    } else {                                              // 阻塞模式
      /* blocking group */
      /* 阻塞组 */
      int savedDev;                                       // 保存当前 CUDA 设备
      CUDACHECKGOTO(cudaGetDevice(&savedDev), ret, fail); // 获取当前 CUDA 设备
      //内部会等待线程结束
      NCCLCHECKGOTO(groupLaunch(&groupJob->base, internalSimInfoPtr), ret, fail); // 执行组启动
      CUDACHECKGOTO(cudaSetDevice(savedDev), ret, fail); // 恢复 CUDA 设备
      if (simInfo)                                        // 如果提供了模拟信息
        memcpy((void*)simInfo, (void*)internalSimInfoPtr, realSize); // 复制模拟信息到用户缓冲区
      free(groupJob);                                     // 释放组作业内存
    }
  }
  /* Reset the job state for the next group call. */
  /* 为下一次组调用重置作业状态 */
  groupLocalResetJobState();                              // 重置本地组作业状态

exit:                                                      // 正常退出标签
  // Profiler group API start is called inside taskAppend to get graph capture information for the event
  // 性能分析器组 API 开始在 taskAppend 内部调用，以获取事件的图捕获信息
  NCCLCHECK(ncclProfilerStopGroupApiEvent());             // 停止性能分析器组 API 事件
  return ret;                                             // 返回操作结果
fail:                                                      // 失败退出标签
  if (groupJob) {                                         // 如果组作业已创建
    groupCleanup(groupJob->groupCommHead, &groupJob->asyncJobs, ret); // 清理组作业资源
    free(groupJob);                                       // 释放组作业内存
  } else {                                                // 组作业未创建
    groupCleanup(ncclGroupCommHead, &ncclAsyncJobs, ret); // 清理全局资源
  }
  groupLocalResetJobState();                              // 重置本地组作业状态
  goto exit;                                              // 跳转到 exit 标签
}

// 函数实现：完成组作业
// 此函数用于等待非阻塞组作业完成并清理资源
// 参数 groupJob: 指向组作业结构体的指针
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
ncclResult_t ncclGroupJobComplete(struct ncclGroupJob* groupJob) {
  ncclResult_t ret = ncclSuccess;                         // 初始化返回值为成功状态
  // 检查组作业指针是否有效且已初始化为非阻塞模式
  if (groupJob && groupJob->nonBlockingInit) {            // 检查组作业有效性和非阻塞初始化标志
    // 使用原子交换操作将 joined 标志设置为 true，并获取旧值
    // 如果旧值为 false（首次调用），则等待线程完成
    if (!__atomic_exchange_n(&groupJob->joined, true, __ATOMIC_ACQ_REL)) { // 原子交换操作
      ret = ncclAsyncJobComplete(&groupJob->base);        // 等待异步作业完成并获取结果
    }
    // 减少组作业的引用计数，如果计数降为 0，则释放内存
    if (ncclAtomicRefCountDecrement(&groupJob->groupRefCount) == 0) { // 原子减少引用计数
      free(groupJob);                                     // 释放组作业内存
    }
  }
  return ret;                                             // 返回操作结果
}

// 函数实现：中止组作业
// 此函数用于中止正在执行的非阻塞组作业
// 参数 groupJob: 指向组作业结构体的指针
// 返回值：ncclResult_t 类型，总是返回成功
ncclResult_t ncclGroupJobAbort(struct ncclGroupJob* groupJob) {
  // 检查组作业指针是否有效且已初始化为非阻塞模式
  if (groupJob && groupJob->nonBlockingInit) {            // 检查组作业有效性和非阻塞初始化标志
    // 使用原子交换操作将 joined 标志设置为 true，并获取旧值
    // 如果旧值为 false（首次调用），则中止作业
    if (!__atomic_exchange_n(&groupJob->joined, true, __ATOMIC_ACQ_REL)) { // 原子交换操作
      __atomic_store_n(&groupJob->abortFlag, true, __ATOMIC_RELAXED); // 设置中止标志为 true（使用松散内存序）
      ncclAsyncJobComplete(&groupJob->base);              // 等待异步作业完成
    }
    // 减少组作业的引用计数，如果计数降为 0，则释放内存
    if (ncclAtomicRefCountDecrement(&groupJob->groupRefCount) == 0) { // 原子减少引用计数
      free(groupJob);                                     // 释放组作业内存
    }
  }
  return ncclSuccess;                                    // 返回成功状态
}
