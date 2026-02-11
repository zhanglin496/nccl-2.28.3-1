/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 * 版权所有 (c) 2016-2022, NVIDIA CORPORATION. 保留所有权利。
 *
 * See LICENSE.txt for license information
 * 请参阅 LICENSE.txt 获取许可证信息
 ************************************************************************/

// 引入通信器头文件，包含 ncclComm 结构体和相关操作的声明
#include "comm.h"
// 引入信息头文件，包含 NCCL 信息管理相关的定义
#include "info.h"
// 引入集合操作头文件，包含集合通信操作的实现
#include "collectives.h"
// 引入套接字头文件，包含网络套接字操作的封装
#include "socket.h"
// 引入共享内存工具头文件，包含共享内存操作的封装
#include "shmutils.h"
// 定义是否启用计时器，0 表示不启用
#define ENABLE_TIMER 0
// 引入计时器头文件，用于性能测量
#include "timer.h"
// 引入性能分析器头文件，用于性能分析
#include "profiler.h"
// 引入传输层头文件，包含各种传输方式的实现
#include "transport.h"
// 引入 CPU 集合头文件，用于 CPU 亲和性设置
#include "cpuset.h"

// 引入系统调用头文件，用于获取线程 ID 等系统调用
#include <sys/syscall.h>
// 引入断言头文件，用于调试断言
#include <assert.h>
// 引入unistd头文件，提供对 POSIX 操作系统 API 的访问
#include <unistd.h>
// 引入时间头文件，用于时间操作
#include <sys/time.h>
// 引入调度头文件，用于 CPU 亲和性设置
#include <sched.h>
// 引入 C++ 标准库算法头文件
#include <algorithm>

// 定义代理的最大连接数（本地 rank 数加 1）
// NCCL_MAX_LOCAL_RANKS 是单个节点内最大的本地 rank 数量
// 加 1 是为了额外的安全边界或特殊用途
#define NCCL_MAX_PROXY_CONNECTIONS (NCCL_MAX_LOCAL_RANKS+1)

// 定义枚举类型，用于区分接收和发送操作
// proxyRecv=0 表示接收操作
// proxySend=1 表示发送操作
enum { proxyRecv=0, proxySend=1 };
// 函数声明：代理服务 UDS（Unix Domain Socket）线程函数
// 此函数处理通过 Unix 域套接字接收的代理请求
// 参数 _args: 代理状态指针
// 返回值：void* 类型的指针（pthread 线程函数标准返回类型）
void* ncclProxyServiceUDS(void* _args);

// 静态函数：判断是否需要代理
// 此函数根据通信模式和根节点位置判断某个 rank 是否需要代理来处理通信
// 参数 type: 操作类型（proxyRecv 或 proxySend）
// 参数 pattern: 通信模式（环形、树形、管道等）
// 参数 root: 根节点的 rank 编号
// 参数 ring: 环形拓扑结构指针
// 参数 nranks: 总的 rank 数量
// 返回值：bool 类型，true 表示需要代理，false 表示不需要
static bool NeedProxy(int type, int pattern, int root, struct ncclRing* ring, int nranks) {
  // 如果是环形或双环形模式，所有 rank 都需要代理
  if (pattern == ncclPatternRing || pattern == ncclPatternRingTwice) 
    return true;

  /* 在链式模式中，有一个 rank 不需要代理。我们需要找出它是哪一个 */
  /* 我们应该将 root 与重组后的环中的哪个索引进行比较 */
  const int myrank = 0, nextrank = 1, prevrank = nranks-1;
  int index = pattern == ncclPatternPipelineFrom ?
      /*                            无需接收 / 无需发送    如果 root = */
      /* 广播      */ (type == proxyRecv ?   myrank : nextrank ):
      /* 归约      */ (type == proxyRecv ? prevrank :   myrank );
  int rank = ring->userRanks[index];
  // 如果当前 rank 不是根节点，则需要代理
  return (root != rank);
}

// 定义代理参数的分配大小
// 每次分配一个包含 NCCL_MAX_OPS 个元素的代理参数块
#define PROXYARGS_ALLOCATE_SIZE NCCL_MAX_OPS
// 代理池结构体
// 此结构体用于管理代理参数的内存池，以提高内存分配效率
// 通过链表方式连接多个内存池块
struct ncclProxyPool {
  // 指向下一个代理池块的指针（用于链表管理）
  struct ncclProxyPool *next;
  // 代理参数数组，包含 PROXYARGS_ALLOCATE_SIZE 个元素
  // 每个元素是一个 ncclProxyArgs 结构体，存储一个代理操作的参数
  struct ncclProxyArgs elems[PROXYARGS_ALLOCATE_SIZE];
};

// 静态函数：释放预期代理响应队列
// 此函数释放预期响应队列中的所有元素及其关联的内存
// 参数 state: 代理状态指针，包含 expectedResponses 队列
static void expectedProxyResponseFree(struct ncclProxyState* state) {
  // 获取预期响应队列的头元素
  struct ncclExpectedProxyResponse* elem = state->expectedResponses;
  // 前驱元素指针（用于遍历，但此函数中实际未使用）
  struct ncclExpectedProxyResponse* prev = NULL;

  // 遍历整个队列，逐个释放元素
  while (elem) {
    prev = elem;
    elem = elem->next;
    // 释放响应缓冲区内存
    free(prev->respBuff);
    // 释放响应元素结构体内存
    free(prev);
  }
}

// 静态函数：存储代理响应到预期响应队列
// 此函数将代理操作的响应存储到预期响应队列中，等待主线程查询
// 参数 state: 代理状态指针
// 参数 opId: 操作 ID（用于标识响应对应的操作）
// 参数 respBuff: 响应缓冲区指针
// 参数 respSize: 响应大小
// 参数 res: 操作结果状态码
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
static ncclResult_t expectedProxyResponseStore(struct ncclProxyState* state, void* opId, void* respBuff, int respSize, ncclResult_t res) {
  // 获取预期响应队列的头元素
  struct ncclExpectedProxyResponse* elem = state->expectedResponses;
  // 遍历队列查找匹配的 opId
  while (elem) {
    if (elem->opId == opId) {
      // 检查响应大小是否匹配
      if (respSize != elem->respSize) {
        WARN("Mismatched response size for opId=%p", opId);
        return ncclInternalError;
      }

      // 检查操作是否已完成（防止重复存储）
      if (elem->done) {
        WARN("Storing response for already completed opId=%p", opId);
        return ncclInternalError;
      }

      // 如果有响应数据，复制到响应缓冲区
      if (respSize > 0) {
        memcpy(elem->respBuff, respBuff, respSize);
        free(respBuff);
      }
      // 标记操作已完成
      elem->done = true;
      // 保存操作结果
      elem->res  = res;
      return ncclSuccess;
    }
    elem = elem->next;
  }

  // 未找到匹配的 opId
  WARN("Proxy response for opId=%p doesn't match any expected response", opId);
  return ncclInternalError;
}

// 静态函数：将预期响应加入队列
// 此函数创建一个新的预期响应元素并添加到队列末尾
// 参数 state: 代理状态指针
// 参数 opId: 操作 ID
// 参数 respSize: 预期的响应大小
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
static ncclResult_t expectedProxyResponseEnqueue(struct ncclProxyState* state, void* opId, int respSize) {
  // 声明预期响应元素指针
  struct ncclExpectedProxyResponse* ex;
  // 分配并清零内存
  NCCLCHECK(ncclCalloc(&ex, 1));
  // 设置操作 ID
  ex->opId = opId;

  // 预分配响应缓冲区
  ex->respBuff = malloc(respSize);
  ex->respSize = respSize;
  ex->res      = ncclInternalError;
  ex->done     = false;

  // 将元素加入队列
  struct ncclExpectedProxyResponse* list = state->expectedResponses;
  if (list == NULL) {
    // 队列为空，设置为头元素
    state->expectedResponses = ex;
    return ncclSuccess;
  }
  // 遍历到队列末尾
  while (list->next)
    list = list->next;

  // 添加新元素到队列末尾
  list->next = ex;
  return ncclSuccess;
}

// 静态函数：从预期响应队列中取出并删除响应
// 此函数从队列中查找已完成的响应，复制数据后删除该元素
// 参数 state: 代理状态指针
// 参数 opId: 操作 ID
// 参数 respBuff: 输出参数，用于存储响应数据
// 参数 found: 输出参数，指示是否找到匹配的响应
// 返回值：ncclResult_t 类型，表示操作结果状态码
static ncclResult_t expectedProxyResponseDequeue(struct ncclProxyState* state, void* opId, void* respBuff, int* found) {
  // 获取预期响应队列的头元素
  struct ncclExpectedProxyResponse* elem = state->expectedResponses;
  // 前驱元素指针
  struct ncclExpectedProxyResponse* prev = NULL;
  // 初始化 found 标志为 0（未找到）
  *found = 0;
  // 遍历队列查找匹配的 opId
  while (elem) {
    // 检查是否是目标操作且已完成
    if ((elem->opId == opId) && elem->done) {
      // 从队列中移除该元素
      if (prev == NULL) {
        // 元素是队列头
        state->expectedResponses = elem->next;
      } else {
        // 元素在队列中间或末尾
        prev->next = elem->next;
      }
      // 复制响应数据到输出缓冲区
      memcpy(respBuff, elem->respBuff, elem->respSize);
      ncclResult_t res = elem->res;
      // 释放响应缓冲区内存
      free(elem->respBuff);
      // 释放元素结构体内存
      free(elem);
      // 设置 found 标志为 1（已找到）
      *found = 1;
      return res;
    }
    prev = elem;
    elem = elem->next;
  }
  // 未找到匹配的响应，返回成功（这是正常情况）
  return ncclSuccess;
}

// 静态函数：从预期响应队列中移除响应（不复制数据）
// 此函数从队列中查找并删除指定 opId 的响应元素，但不复制响应数据
// 参数 state: 代理状态指针
// 参数 opId: 要移除的操作 ID
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
static ncclResult_t expectedProxyResponseRemove(struct ncclProxyState* state, void* opId) {
  // 获取预期响应队列的头元素
  struct ncclExpectedProxyResponse* elem = state->expectedResponses;
  // 前驱元素指针
  struct ncclExpectedProxyResponse* prev = NULL;
  // 遍历队列查找匹配的 opId
  while (elem) {
    if (elem->opId == opId) {
      // 从队列中移除该元素
      if (prev == NULL) {
        // 元素是队列头
        state->expectedResponses = elem->next;
      } else {
        // 元素在队列中间或末尾
        prev->next = elem->next;
      }
      // 释放响应缓冲区内存
      free(elem->respBuff);
      // 释放元素结构体内存
      free(elem);
      return ncclSuccess;
    }
    prev = elem;
    elem = elem->next;
  }
  // 未找到匹配的 opId，输出警告信息
  WARN("Couldn't find opId=%p", opId);
  return ncclInternalError;
}

// 静态函数：将异步操作加入队列
// 此函数将异步代理操作添加到本地对等节点的异步操作队列末尾
// 参数 peer: 本地对等节点指针
// 参数 op: 异步操作指针
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
static ncclResult_t asyncProxyOpEnqueue(struct ncclProxyLocalPeer* peer, ncclProxyAsyncOp* op) {
  // 获取异步操作队列的头元素
  ncclProxyAsyncOp* list = peer->asyncOps;
  if (list == NULL) {
    // 队列为空，设置为头元素
    peer->asyncOps = op;
    return ncclSuccess;
  }
  // 遍历到队列末尾
  while (list->next) list = list->next;
  // 添加新元素到队列末尾
  list->next = op;
  return ncclSuccess;
}

// 静态函数：从异步操作队列中取出并删除操作
// 此函数从队列中查找并删除指定 opId 的异步操作
// 参数 peer: 本地对等节点指针
// 参数 op: 要删除的异步操作指针
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
static ncclResult_t asyncProxyOpDequeue(struct ncclProxyLocalPeer* peer, ncclProxyAsyncOp* op) {
  // 获取异步操作队列的头元素
  struct ncclProxyAsyncOp* elem = peer->asyncOps;
  // 前驱元素指针
  struct ncclProxyAsyncOp* prev = NULL;
  // 遍历队列查找匹配的 opId
  while (elem) {
    if (elem->opId == op->opId) {
      // 从队列中移除该元素
      if (prev == NULL) {
        // 元素是队列头
        peer->asyncOps = elem->next;
      } else {
        // 元素在队列中间或末尾
        prev->next = elem->next;
      }

      // 释放请求缓冲区内存
      if (elem->reqBuff) {
        free(elem->reqBuff);
      }
      // 释放响应缓冲区内存
      if (elem->respBuff) {
        free(elem->respBuff);
      }
      // 释放元素结构体内存
      free(elem);

      return ncclSuccess;
    }
    prev = elem;
    elem = elem->next;
  }
  // 未找到匹配的操作，输出警告信息
  if (op) {
    WARN("Attempting to dequeue nonexistent async opId=%p", op->opId);
  } else {
    WARN("Attempting to dequeue null operation");
  }
  return ncclInternalError;
}

// 静态函数：从代理参数池中分配一个参数元素
// 此函数从代理参数池中获取一个可用的参数元素，如果池为空则分配新的内存池
// 参数 state: 代理进度状态指针
// 参数 argsptr: 输出参数，返回分配的参数元素指针
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
static ncclResult_t allocateArgs(struct ncclProxyProgressState* state, struct ncclProxyArgs** argsptr) {
  // 声明参数元素指针
  struct ncclProxyArgs* elem;
  // 检查参数池是否为空
  if (state->pool == NULL) {
    // 分配一个新的元素池。确保分配的内存靠近网络线程
    // 这样可以提高缓存局部性和性能
    struct ncclProxyPool* newPool;
    NCCLCHECK(ncclCalloc(&newPool, 1));

    // 获取新池中的元素数组
    struct ncclProxyArgs* newElems = newPool->elems;
    // 将新分配的元素链接成空闲链表
    for (int i=0; i<PROXYARGS_ALLOCATE_SIZE; i++) {
      if (i+1 < PROXYARGS_ALLOCATE_SIZE) newElems[i].next = newElems+i+1;
    }
    // 将所有元素添加到池列表
    state->pool = newElems;
    // 保存池内存块指针，以便后续资源释放
    newPool->next = state->pools;
    state->pools = newPool;
  }
  // 从池中取出一个元素
  elem = state->pool;
  state->pool = state->pool->next;
  // 清空元素的 next 和 nextPeer 指针
  elem->next = elem->nextPeer = NULL;
  // 返回分配的元素
  *argsptr = elem;
  return ncclSuccess;
}

// 调试宏定义：取消注释以启用代理调试模式
//#define DEBUG_PROXY 1
#ifdef DEBUG_PROXY
// 如果启用调试，DEBUG_PROXY_PRINT 映射到 printf
#define DEBUG_PROXY_PRINT printf
#else
// 如果未启用调试，DEBUG_PROXY_PRINT 映射到空操作（不产生任何输出）
#define DEBUG_PROXY_PRINT(...)
#endif

// 宏：获取操作在池中的索引
// 计算操作指针相对于池首元素的位置，如果操作为空则返回 -1
#define OP_INDEX(op) ((op) ? (op)-state->pools->elems : -1)
// 宏：操作已访问标志（用于检测循环链表）
// 使用高位（0x100000）作为标志位，避免与正常状态值冲突
#define OP_SEEN 0x100000

// 函数：获取操作的池索引和操作索引
// 此函数查找操作所在的内存池，并返回池索引和操作在池中的索引
// 参数 op: 代理操作指针
// 参数 state: 代理进度状态指针
// 参数 poolIndex: 输出参数，返回池索引
// 参数 opIndex: 输出参数，返回操作在池中的索引
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
ncclResult_t getOpIndex(struct ncclProxyArgs* op, struct ncclProxyProgressState* state, int* poolIndex, int* opIndex) {
  // 获取池链表的头节点
  struct ncclProxyPool* pool = state->pools;
  int p = 0;
  // 遍历所有池
  while (pool) {
    // 计算操作指针相对于池首元素的偏移量
    uint64_t o = op-pool->elems;
    if (o < PROXYARGS_ALLOCATE_SIZE) {
      // 找到操作所在的池，返回索引
      *opIndex = o;
      *poolIndex = p;
      return ncclSuccess;
    }
    pool = pool->next;
    p++;
  }
  // 未找到操作所在的池
  WARN("Could not find pool of op %p", op);
  return ncclInternalError;
}

// 函数：打印代理操作信息（用于调试）
// 此函数以易读格式打印代理操作的详细信息，包括操作类型和状态
// 参数 op: 代理操作指针
// 参数 poolIndex: 池索引
// 参数 opIndex: 操作在池中的索引
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
ncclResult_t printProxyOp(struct ncclProxyArgs* op, int poolIndex, int opIndex) {
  // 打印池索引、操作索引、操作计数和操作类型
  printf("[%d-%d|%ld| %s", poolIndex, opIndex, op->opCount, op->pattern == ncclPatternSend ? "Send" : op->pattern == ncclPatternRecv ? "Recv" : "Coll");
  // 遍历所有子操作
  for (int s=0; s<op->nsubs; s++) {
    struct ncclProxySubArgs* sub = op->subs+s;
    if (op->state == ncclProxyOpProgress) {
      // 操作正在执行中，计算状态字符
      char status = ' ';
      if (op->pattern == ncclPatternRecv) {
        // 接收操作的状态判断
        if (sub->posted < sub->nsteps && sub->posted < sub->done + NCCL_STEPS) status = 'I'; // 初始化
        else if (sub->received < sub->posted) status = 'R'; // 正在接收
        else if (sub->received < sub->transmitted) status = 'R'; // 正在接收
        else if (sub->transmitted < sub->received) status = 'F'; // 正在刷新
        else if (sub->done < sub->transmitted) status = 'G'; // 等待 GPU
        else status = 'D'; // 完成
      } else if (op->pattern == ncclPatternSend) {
        // 发送操作的状态判断
        if (sub->posted < sub->nsteps && sub->posted < sub->done + NCCL_STEPS) status = 'I'; // 初始化
        else if (sub->transmitted < sub->posted) status = 'G'; // 等待 GPU
        else if (sub->done < sub->transmitted) status = 'S'; // 正在发送
        else status = 'D'; // 完成
      }
      // 打印对等节点、状态字符和通道 ID
      printf(" %d%c/%d", sub->peer, status, sub->channelId);
    } else {
      // 操作未在执行中，只打印对等节点和通道 ID
      printf(" %d/%d", sub->peer, sub->channelId);
    }
  }
  printf("]");
  return ncclSuccess;
}
// 函数：转储代理状态（用于调试）
// 此函数打印代理进度状态的详细信息，包括活动操作和空闲操作
// 参数 state: 代理进度状态指针
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
ncclResult_t dumpProxyState(struct ncclProxyProgressState* state) {
  // 获取活动操作链表的头节点
  struct ncclProxyArgs* op = state->active;
  int poolIndex, opIndex;
  printf("ACTIVE OPS\n");
  // 遍历活动操作链表
  while (op) {
    NCCLCHECK(getOpIndex(op, state, &poolIndex, &opIndex));
    // 检测循环链表
    if (op->state & OP_SEEN) {
      WARN("List loop at element %d-%d", poolIndex, opIndex);
    }
    // 打印操作信息
    NCCLCHECK(printProxyOp(op, poolIndex, opIndex));
    // 标记操作已访问
    op->state |= OP_SEEN;
    printf("\n");
    // 遍历该操作的对等节点链表
    struct ncclProxyArgs* nextOp = op->nextPeer;
    while (nextOp) {
      NCCLCHECK(getOpIndex(nextOp, state, &poolIndex, &opIndex));
      if (nextOp->state & OP_SEEN) {
        WARN("List loop at element %d-%d", poolIndex, opIndex);
      }
      printf("| `-> ");
      NCCLCHECK(printProxyOp(nextOp, poolIndex, opIndex));
      nextOp->state |= OP_SEEN;
      printf("\n");
      // 检查非活动操作是否有 next 指针（异常情况）
      if (nextOp->next) {
        WARN("Inactive op has next set!");
      }
      nextOp = nextOp->nextPeer;
    }
    if (op->nextPeer == NULL) printf("|\n");
    op = op->next;
    printf("v\n");
  }
  printf("[X]\n");

# if 0
  // 此代码块被注释掉（用于调试空闲操作池）
  printf("FREE OPS\n");
  op = state->pool;
  while (op) {
    NCCLCHECK(getOpIndex(op, state, &poolIndex, &opIndex));
    if (op->state & OP_SEEN) {
      WARN("List loop at element %d-%d", poolIndex, opIndex);
    }
    NCCLCHECK(printProxyOp(op, poolIndex, opIndex));
    op->state |= OP_SEEN;
    printf("->");
    op = op->next;
  }
  printf("[X]\n");
#else
  // 遍历空闲操作池，标记所有元素为已访问
  op = state->pool;
  while (op) {
    NCCLCHECK(getOpIndex(op, state, &poolIndex, &opIndex));
    if (op->state & OP_SEEN) {
      WARN("List loop at element %d-%d", poolIndex, opIndex);
    }
    op->state |= OP_SEEN;
    op = op->next;
  }
#endif

  // 遍历所有池，检查是否有元素未被访问
  struct ncclProxyPool* pool = state->pools;
  poolIndex = 0;
  while (pool) {
    struct ncclProxyArgs* elem = pool->elems;
    // 遍历池中的所有元素
    for (int e=0; e<PROXYARGS_ALLOCATE_SIZE; e++, elem++) {
      if ((elem->state & OP_SEEN) == 0) {
        // 元素未被访问，说明不在任何列表中（异常情况）
        printf("Elem %d-%d is not in any list:\n", poolIndex, e);
        NCCLCHECK(printProxyOp(elem, poolIndex, e));
        printf("\n");
      } else {
        // 清除已访问标志，为下次检测做准备
        elem->state -= OP_SEEN;
      }
    }
    pool = pool->next;
    poolIndex++;
  }
  return ncclSuccess;
}

// 静态函数：将代理操作转换为代理参数
// 此函数将代理操作的数据复制到代理参数结构中，用于执行代理操作
// 参数 op: 代理操作指针
// 参数 args: 代理参数指针
// 参数 subIndex: 子操作索引
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
static ncclResult_t ncclProxyOpToArgs(struct ncclProxyOp* op, struct ncclProxyArgs* args, int subIndex) {
  // 获取子参数指针
  struct ncclProxySubArgs* sub = args->subs+subIndex;
  // 检查子操作索引是否越界
  if (subIndex >= NCCL_PROXY_MAX_SUBS) {
    WARN("Proxy append out of bounds");
    return ncclInternalError;
  }
  // 注释掉的 memset：不再需要清零，因为会逐个字段赋值
  //memset(sub, 0, sizeof(struct ncclProxySubArgs));
  // 复制连接相关参数
  sub->connection = op->connection;
  sub->channelId = op->channelId;
  // 复制操作步骤和大小参数
  sub->nsteps = op->nsteps;
  sub->nbytes = op->nbytes;
  sub->chunkSize = op->chunkSize;
  sub->offset = 0;
  // 复制循环参数
  sub->loopSize = op->loopSize;
  sub->loopOffset = op->loopOffset;
  // 复制标志位
  sub->isOneRPN = op->isOneRPN;
  // 复制对等节点信息
  sub->peer = op->peer;
  // 复制内存注册信息
  sub->reg = op->reg;
  sub->sendMhandle = op->sendMhandle;
  sub->recvMhandle = op->recvMhandle;
  // 复制缓冲区指针
  sub->sendbuff = op->sendbuff;
  sub->recvbuff = op->recvbuff;
  // 复制事件句柄
  sub->eActivationMask = op->eActivationMask;
  sub->taskEventHandle = op->taskEventHandle;
  // 复制 rank 和进程 ID
  sub->rank = op->rank;
  sub->pid = op->pid;
  // 复制性能分析器上下文
  sub->profilerContext = op->profilerContext;
  // 复制环形算法指针
  sub->ringAlgo = op->ringAlgo;
  // 复制工作计数器
  sub->workCounter = op->workCounter;
  // 更新子操作数量
  args->nsubs = subIndex+1;
  // 如果不是第一个子操作，需要验证参数一致性
  if (subIndex) {
    // 取最小值以确保一致性
    args->nChannels = std::min(args->nChannels, op->nChannels);
    args->nPeers = std::min(args->nPeers, op->nPeers);
    // 检查关键参数是否匹配
    if ((args->sliceSteps != op->sliceSteps) ||
        (args->chunkSteps != op->chunkSteps) ||
        (args->protocol != op->protocol) ||
        (args->dtype != op->dtype) ||
        (args->redOp != op->redOp) ||
        (args->coll != op->coll)) {
      WARN("Proxy append mismatch");
      return ncclInternalError;
    }
    // 检查操作状态
    if (args->state != ncclProxyOpReady) {
      WARN("Proxy append on running operation");
      return ncclInternalError;
    }
    // 参数验证通过，跳转到退出点
    goto exit;
  }
  // 注释掉的 memset：不再需要清零，因为会逐个字段赋值
  //memset(&args->progress, 0, sizeof(struct ncclProxyArgs)-offsetof(struct ncclProxyArgs, progress));
  // 初始化第一个子操作的参数
  args->done = 0;
  args->opCount = op->opCount;
  args->sliceSteps = op->sliceSteps;
  args->chunkSteps = op->chunkSteps;
  args->chunkSize = op->chunkSize;
  args->dtype = op->dtype;
  args->redOp = op->redOp;
  args->pattern = op->pattern;
  args->protocol = op->protocol;
  args->coll = op->coll;
  args->collAPI = op->collAPI;
  args->algorithm = op->algorithm;
  args->nChannels = op->nChannels;
  args->nPeers = op->nPeers;
  args->specifics = op->specifics;
  // 设置操作状态为就绪
  args->state = ncclProxyOpReady;
  // 设置回调函数指针
  args->progress = op->connection->tcomm->proxyProgress;
  args->proxyAppendPtr = op->connection->proxyAppendPtr;
exit:
  // 如果不是性能分析器模式，启动性能分析事件
  if (args->pattern != ncclPatternProfiler) ncclProfilerStartProxyOpEvent(subIndex, args);
  return ncclSuccess;
}

// 静态函数：将代理操作添加到代理进度状态
// 此函数将代理操作添加到活动列表或作为对等节点链接
// 参数 state: 代理进度状态指针
// 参数 op: 代理操作指针
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
static ncclResult_t ProxyAppend(struct ncclProxyProgressState* state, struct ncclProxyOp* op) {
  // 获取连接指针和共享标志
  struct ncclProxyConnection* connection = op->connection;
  int shared = connection->shared;
  // 获取当前代理参数指针
  struct ncclProxyArgs* args = *connection->proxyAppendPtr;

  if (args) {
    // 已有该对等节点的操作在运行
    if (shared && args->opCount == op->opCount) {
      // 共享操作且操作计数相同，作为子操作添加
      NCCLCHECK(ncclProxyOpToArgs(op, args, args->nsubs));
      DEBUG_PROXY_PRINT("Insert (%d/%5ld/%5ld) as group with %5ld\n", shared, args->opCount, op->opCount, OP_INDEX(args));
    } else {
      // 作为对等节点链接添加
      struct ncclProxyArgs* prevArgs = args;
      NCCLCHECK(allocateArgs(state, &args));
      NCCLCHECK(ncclProxyOpToArgs(op, args, 0));
      prevArgs->nextPeer = args;
      DEBUG_PROXY_PRINT("Insert  %5ld (%d/%5ld/%5ld) as nextPeer of %5ld\n", OP_INDEX(args), shared, prevArgs->opCount, args->opCount, OP_INDEX(prevArgs));
      *(args->proxyAppendPtr) = args;
    }
  } else {
    // 该对等节点没有运行中的操作，添加到列表
    NCCLCHECK(allocateArgs(state, &args));
    NCCLCHECK(ncclProxyOpToArgs(op, args, 0));
    if (state->active == NULL) {
      // 活动列表为空，创建新列表
      DEBUG_PROXY_PRINT("Insert  %5ld (%d/%5ld) as first element\n", OP_INDEX(args), shared, args->opCount);
      state->active = args;
    } else {
      // 将元素添加到列表末尾
      struct ncclProxyArgs* last = state->active;
      while (last->next)
        last = last->next;
      last->next = args;
      DEBUG_PROXY_PRINT("Insert  %5ld (%d/%5ld) as last element\n", OP_INDEX(args), shared, args->opCount);
    }
    *(args->proxyAppendPtr) = args;
  }
  return ncclSuccess;
}

// 函数：发布代理操作到操作池
// 此函数将代理操作发布到共享操作池，并通知代理进度线程
// 参数 pool: 代理操作池指针
// 参数 nextOps: 下一个操作索引
// 参数 nextOpsEnd: 下一个操作结束索引
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
ncclResult_t ncclProxyPost(struct ncclProxyOpsPool* pool, int nextOps, int nextOpsEnd) {
  // 加锁保护共享数据结构
  pthread_mutex_lock(&pool->mutex);
  if (pool->nextOps == -1) {
    // 池为空，设置头操作并通知等待线程
    pool->nextOps = nextOps;
    pthread_cond_signal(&pool->cond);
  } else {
    // 池不为空，将操作链接到现有链表末尾
    pool->ops[pool->nextOpsEnd].next = nextOps;
  }
  // 更新结束索引
  pool->nextOpsEnd = nextOpsEnd;
  // 解锁
  pthread_mutex_unlock(&pool->mutex);
  return ncclSuccess;
}

// 静态函数：将本地操作添加到代理操作队列
// 此函数将代理操作添加到本地代理操作队列，用于后续执行
// 参数 comm: NCCL 通信器指针
// 参数 proxyConn: 代理连接器指针
// 参数 proxyOp: 代理操作指针
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
static ncclResult_t ncclLocalOpAppend(struct ncclComm* comm, struct ncclProxyConnector* proxyConn, struct ncclProxyOp* proxyOp) {
  // 获取顶层父节点的本地 rank
  int tpLocalRank = comm->topParentLocalRanks[comm->localRank];
  // 获取代理操作数组
  struct ncclProxyOps* proxyOps = comm->proxyState->proxyOps;
  if (proxyOps == NULL) return ncclInternalError;
  // 定位到目标本地 rank 的代理操作
  proxyOps += proxyConn->tpLocalRank;
  // 获取操作池
  struct ncclProxyOpsPool* pool = proxyOps->pool;

  TIME_START(0);
  // 获取空闲操作索引
  int opIndex = proxyOps->freeOp;
  struct ncclProxyOp* op;
  if (opIndex != -1) {
    // 本地有空闲操作
    op = pool->ops+opIndex;
    proxyOps->freeOp = op->next;
  } else {
    // 本地无空闲操作，从共享池获取
    int freeOp;
    // 等待有空闲操作可用
    while ((freeOp = pool->freeOps[tpLocalRank]) == -1) sched_yield();
    // 使用原子操作交换空闲操作索引
    int freeOpNew;
    while ((freeOpNew = __sync_val_compare_and_swap(pool->freeOps+tpLocalRank, freeOp, -1)) != freeOp) freeOp = freeOpNew;
    opIndex = freeOp;
    op = pool->ops+opIndex;
    proxyOps->freeOp = op->next;
  }
  // 预取下一个空闲操作到缓存
  if (op->next != -1) __builtin_prefetch(pool->ops+op->next);
  // 复制代理操作数据
  memcpy(op, proxyOp, sizeof(struct ncclProxyOp));
  // 增加环形算法引用计数
  if (proxyOp->ringAlgo) proxyOp->ringAlgo->incRefCount();
  op->next = -1;
  op->connection = proxyConn->connection;
  // 将操作添加到待发布链表
  if (proxyOps->nextOps == -1) {
    // 链表为空，创建新链表
    proxyOps->nextOps = proxyOps->nextOpsEnd = opIndex;
  } else {
    // 链表不为空，添加到末尾
    pool->ops[proxyOps->nextOpsEnd].next = opIndex;
    proxyOps->nextOpsEnd = opIndex;
  }
  // 检查是否达到最大操作数
  if (++proxyOps->count == MAX_OPS_PER_PEER) {
    // 发布已有操作以释放池中的操作
    // 不发布最后的操作，因为可能有更多相同 opCount 的操作到来，
    // 将它们分批发布会破坏代理参数的子操作聚合
    uint64_t lastOpCount = pool->ops[proxyOps->nextOpsEnd].opCount;
    int lastOp = -1;
    int toSend = 0;
    int ops = 0;
    // 遍历链表找到最后一个相同 opCount 的操作
    for (int op= proxyOps->nextOps; op != proxyOps->nextOpsEnd; op=pool->ops[op].next) {
      ops++;
      if (pool->ops[op].opCount != lastOpCount) {
        lastOp = op;
        toSend = ops;
      }
    }
    if (lastOp == -1) {
      WARN("Unable to post incomplete proxy op chain %d..%d (opCount %ld)", proxyOps->nextOps, proxyOps->nextOpsEnd, lastOpCount);
      return ncclInternalError;
    }
    // 在 lastOp 处切断链表
    int nextOps = proxyOps->nextOps;
    proxyOps->nextOps = pool->ops[lastOp].next;
    pool->ops[lastOp].next = -1;
    NCCLCHECK(ncclProxyPost(proxyOps->pool, nextOps, lastOp));
    proxyOps->count -= toSend;
  }
  TIME_STOP(0);
  return ncclSuccess;
}

// 静态函数：增加工作计数器
// 此函数根据操作标志增加或读取工作计数器，用于性能分析和追踪
// 参数 comm: NCCL 通信器指针
// 参数 op: 代理操作指针
static void incWorkCounter(struct ncclComm* comm, struct ncclProxyOp* op) {
  // 如果需要增加计数器，则递增并返回新值；否则只返回当前值
  op->workCounter = (op->incWorkCounter) ? ++comm->profiler.workCounter[op->channelId] : comm->profiler.workCounter[op->channelId];
}

// 静态函数：保存性能分析器代理操作
// 此函数为性能分析器保存代理操作，用于性能数据收集
// 参数 comm: NCCL 通信器指针
// 参数 op: 代理操作指针
// 参数 justInquire: 输出参数，指示是否仅查询而不实际执行
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
static ncclResult_t SaveProxyProfiler(struct ncclComm* comm, struct ncclProxyOp* op, bool* justInquire) {
  // 根据操作类型选择接收或发送代理连接
  struct ncclProxyConnector* proxyConn = (op->coll == ncclFuncRecv) ? &comm->profiler.recvProxyConn[op->channelId] : &comm->profiler.sendProxyConn[op->channelId];
  if (justInquire) {
    // 仅查询模式，设置标志
    *justInquire = true;
    // 如果不是持久化模式，增加工作计数器
    if (!comm->planner.persistent) incWorkCounter(comm, op);
  } else {
    // 设置发送和接收缓冲区指针
    op->sendbuff = (uint8_t *)comm->profiler.workStarted;
    op->recvbuff = (uint8_t *)comm->profiler.workCompleted;
    // 确保在图捕获模式下代理工作计数器递增以与内核工作计数器保持同步
    if (comm->planner.persistent) incWorkCounter(comm, op);
    // 将操作添加到本地队列
    NCCLCHECK(ncclLocalOpAppend(comm, proxyConn, op));
  }
  return ncclSuccess;
}

// 静态函数：保存代理操作
// 此函数将代理操作保存到指定通道的对等节点连接中
// 参数 comm: NCCL 通信器指针
// 参数 channel: 通道指针
// 参数 type: 操作类型（接收或发送）
// 参数 peer: 对等节点索引
// 参数 op: 代理操作指针
// 参数 connIndex: 连接索引
// 参数 justInquire: 输出参数，指示是否仅查询而不实际执行
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
static ncclResult_t SaveProxy(struct ncclComm* comm, struct ncclChannel* channel, int type, int peer, struct ncclProxyOp* op, int connIndex, bool* justInquire) {
  // 检查对等节点索引是否有效
  if (peer < 0) return ncclSuccess;

  // 获取对等节点通信信息
  struct ncclChannelPeer* peerComm = channel->peers[peer];
  // 根据类型获取连接器（接收或发送）
  struct ncclConnector* connector = type == proxyRecv ? peerComm->recv+connIndex : peerComm->send+connIndex;
  // 检查传输层通信器是否存在
  if (connector->transportComm == NULL) {
    WARN("Rank %d has no transport for %s peer %d on channel %d/%d", comm->rank,
        type == proxyRecv ? "recv" : "send", peer, channel->id, connIndex);
    return ncclInternalError;
  }
  // 检查是否支持代理进度
  if (connector->proxyConn.proxyProgress == NULL) return ncclSuccess;

  if (justInquire) *justInquire = true;
  else {
    // 设置对等节点索引并添加操作到本地队列
    op->peer = peer;
    NCCLCHECK(ncclLocalOpAppend(comm, &connector->proxyConn, op));
  }
  return ncclSuccess;
}

// justInquire != nullptr 表示不实际执行任何操作，仅确定此操作是否需要 ncclProxySaveOp
// 函数：保存代理操作
// 此函数根据通信模式将代理操作保存到相应的连接中
// 参数 comm: NCCL 通信器指针
// 参数 op: 代理操作指针
// 参数 justInquire: 输出参数，指示是否仅查询而不实际执行
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
ncclResult_t ncclProxySaveOp(struct ncclComm* comm, struct ncclProxyOp* op, bool* justInquire) {
  // 获取操作对应的通道
  struct ncclChannel* channel = &comm->channels[op->channelId];
  // 初始化查询标志
  if (justInquire) *justInquire = false;
  // 根据通信模式分支处理
  switch (op->pattern) {
  case ncclPatternRing:
  case ncclPatternRingTwice:
  case ncclPatternPipelineFrom:
  case ncclPatternPipelineTo: {
      // 环形和管道模式
      struct ncclRing* ring = &channel->ring;
      // 检查是否需要接收代理
      if (NeedProxy(proxyRecv, op->pattern, op->root, ring, comm->nRanks)) {
        NCCLCHECK(SaveProxy(comm, channel, proxyRecv, ring->prev, op, 0, justInquire));
      }
      // 检查是否需要发送代理
      if (NeedProxy(proxySend, op->pattern, op->root, ring, comm->nRanks)) {
        NCCLCHECK(SaveProxy(comm, channel, proxySend, ring->next, op, 0, justInquire));
      }
    } break;
  case ncclPatternTreeUp:
  case ncclPatternTreeDown:
  case ncclPatternTreeUpDown: {
      // 树形模式
      if (op->pattern != ncclPatternTreeDown) { // 树向上（归约）
        struct ncclTree* tree = &channel->tree;
        // 从所有子节点接收数据
        for (int i=0; i<NCCL_MAX_TREE_ARITY; i++) {
          NCCLCHECK(SaveProxy(comm, channel, proxyRecv, tree->down[i], op, 0, justInquire));
        }
        // 向父节点发送数据
        NCCLCHECK(SaveProxy(comm, channel, proxySend, tree->up, op, 0, justInquire));
      }
      if (op->pattern != ncclPatternTreeUp) { // 树向下（广播）
        struct ncclTree* tree = &channel->tree;
        // 向所有子节点发送数据
        for (int i=0; i< NCCL_MAX_TREE_ARITY; i++) {
          NCCLCHECK(SaveProxy(comm, channel, proxySend, tree->down[i], op, 0, justInquire));
        }
        // 从父节点接收数据
        NCCLCHECK(SaveProxy(comm, channel, proxyRecv, tree->up, op, 0, justInquire));
      }
    } break;
  case ncclPatternCollnetChain: {
      // CollNet 链式模式
      NCCLCHECK(SaveProxy(comm, channel, proxySend, channel->collnetChain.up, op, 1, justInquire));
      NCCLCHECK(SaveProxy(comm, channel, proxyRecv, channel->collnetChain.up, op, 0, justInquire));
    } break;
  case ncclPatternCollnetDirect: {
      // CollNet 直接模式
      NCCLCHECK(SaveProxy(comm, channel, proxySend, channel->collnetDirect.out, op, 1, justInquire));
      NCCLCHECK(SaveProxy(comm, channel, proxyRecv, channel->collnetDirect.out, op, 0, justInquire));
    } break;
  case ncclPatternNvls: {
      // NVLink fabric 模式
      NCCLCHECK(SaveProxy(comm, channel, proxySend, channel->nvls.out, op, 1, justInquire));
      NCCLCHECK(SaveProxy(comm, channel, proxyRecv, channel->nvls.out, op, 0, justInquire));
    } break;
  case ncclPatternNvlsTree: {
      // NVLink 树形模式
      NCCLCHECK(SaveProxy(comm, channel, proxyRecv, channel->nvls.treeDown[1], op, 0, justInquire));
      NCCLCHECK(SaveProxy(comm, channel, proxyRecv, channel->nvls.treeDown[2], op, 0, justInquire));
      NCCLCHECK(SaveProxy(comm, channel, proxySend, channel->nvls.treeUp, op, 0, justInquire));
      NCCLCHECK(SaveProxy(comm, channel, proxySend, channel->nvls.treeDown[1], op, 0, justInquire));
      NCCLCHECK(SaveProxy(comm, channel, proxySend, channel->nvls.treeDown[2], op, 0, justInquire));
      NCCLCHECK(SaveProxy(comm, channel, proxyRecv, channel->nvls.treeUp, op, 0, justInquire));
    } break;
  case ncclPatternPatUp: {
      // PAT 向上模式（Reduce-Scatter）
      // 运行完整算法以计算每个对等节点的步骤数
      ncclResult_t result = ncclSuccess;
      const ssize_t size = op->nbytes/comm->nRanks;
      const int rank = comm->rank, nranks = comm->nRanks;
      int *nstepsSend = NULL, *nstepsRecv = NULL;
      // 创建 Reduce-Scatter 算法实例
      PatRSAlgorithm<char> algo(op->chunkSize, NCCL_STEPS, 16, 0, size, size, op->chunkSize, rank, nranks);
      struct ncclPatStep ps = {0};
      NCCLCHECKGOTO(ncclCalloc(&nstepsSend, log2Up(nranks)), result, exit_pat_up);
      NCCLCHECKGOTO(ncclCalloc(&nstepsRecv, log2Up(nranks)), result, exit_pat_up);

      // 运行算法统计每个维度的步骤数
      do {
        algo.getNextOp(&ps);
        if (ps.flags & PatSkipped) continue;
        if (ps.recvDim != -1 && ps.postRecv) nstepsRecv[ps.recvDim]++;
        if (ps.sendDim != -1 && ps.postSend) nstepsSend[ps.sendDim]++;
      } while (ps.last != 2);
      // 为每个有通信的对等节点保存代理操作
      for (int i=0; i<log2Up(nranks); i++) {
        if (nstepsSend[i]) {
          int sendPeer = (rank + (1<<i)) % nranks;
          op->nsteps = nstepsSend[i];
          NCCLCHECKGOTO(SaveProxy(comm, channel, proxySend, sendPeer, op, 0, justInquire), result, exit_pat_up);
        }
        if (nstepsRecv[i]) {
          int recvPeer = (rank - (1<<i) + nranks) % nranks;
          op->nsteps = nstepsRecv[i];
          NCCLCHECKGOTO(SaveProxy(comm, channel, proxyRecv, recvPeer, op, 0, justInquire), result, exit_pat_up);
        }
      }
    exit_pat_up:
      free(nstepsSend);
      free(nstepsRecv);
      NCCLCHECK(result);
    } break;
  case ncclPatternPatDown: {
      // PAT 向下模式（All-Gather）
      // 运行完整算法以计算每个对等节点的步骤数
      ncclResult_t result = ncclSuccess;
      const ssize_t size = op->nbytes/comm->nRanks;
      const int rank = comm->rank, nranks = comm->nRanks;
      int *nstepsSend = NULL, *nstepsRecv = NULL;
      // 创建 All-Gather 算法实例
      PatAGAlgorithm<char> algo(op->chunkSize, NCCL_STEPS, 16, 0, size, size, op->chunkSize, rank, nranks);
      struct ncclPatStep ps = {0};
      NCCLCHECKGOTO(ncclCalloc(&nstepsSend, log2Up(nranks)), result, exit_pat_down);
      NCCLCHECKGOTO(ncclCalloc(&nstepsRecv, log2Up(nranks)), result, exit_pat_down);

      // 运行算法统计每个维度的步骤数
      do {
        algo.getNextOp(&ps);
        if (ps.flags & PatSkipped) continue;
        if (ps.recvDim != -1 && ps.postRecv) nstepsRecv[ps.recvDim]++;
        if (ps.sendDim != -1 && ps.postSend) nstepsSend[ps.sendDim]++;
      } while (ps.last != 2);
      // 为每个有通信的对等节点保存代理操作
      for (int i=0; i<log2Up(nranks); i++) {
        if (nstepsSend[i]) {
          int sendPeer = (rank - (1<<i) + nranks) % nranks;
          op->nsteps = nstepsSend[i];
          NCCLCHECKGOTO(SaveProxy(comm, channel, proxySend, sendPeer, op, 0, justInquire), result, exit_pat_down);
        }
        if (nstepsRecv[i]) {
          int recvPeer = (rank + (1<<i)) % nranks;
          op->nsteps = nstepsRecv[i];
          NCCLCHECKGOTO(SaveProxy(comm, channel, proxyRecv, recvPeer, op, 0, justInquire), result, exit_pat_down);
        }
      }
    exit_pat_down:
      free(nstepsSend);
      free(nstepsRecv);
      NCCLCHECK(result);
    } break;
  case ncclPatternSend:
  case ncclPatternRecv: {
      // 点对点发送/接收模式
      // 如果当前 rank 是根节点，则不需要代理
      if (op->root == comm->rank) return ncclSuccess;
      // 保存代理操作（根据模式选择发送或接收）
      NCCLCHECK(SaveProxy(comm, channel, op->pattern == ncclPatternSend ? proxySend : proxyRecv, op->root, op, 1, justInquire));
    } break;
  case ncclPatternProfiler: {
      // 性能分析器模式
      // 检查是否需要代理
      if (ncclProfilerNeedsProxy(comm, op)) NCCLCHECK(SaveProxyProfiler(comm, op, justInquire));
      else incWorkCounter(comm, op);
    } break;
  }
  return ncclSuccess;
}

// 静态函数：从活动列表中移除已完成的操作
// 此函数将已完成的操作从活动列表中移除，并将其返回到空闲池
// 参数 state: 代理进度状态指针
// 参数 opPtr: 指向操作指针的指针
// 参数 prevOpPtr: 指向前一个操作指针的指针
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
static ncclResult_t removeOp(struct ncclProxyProgressState* state, struct ncclProxyArgs** opPtr, struct ncclProxyArgs** prevOpPtr) {
  // 获取要释放的操作和下一个操作
  struct ncclProxyArgs* freeOp = *opPtr;
  struct ncclProxyArgs* next = freeOp->next;
  DEBUG_PROXY_PRINT("Remove %ld -> %ld -> %ld\n", OP_INDEX(*prevOpPtr), OP_INDEX(freeOp), OP_INDEX(next));
  // 更新操作指针指向下一个操作
  *opPtr = next;
  if (freeOp->nextPeer) {
    // 有对等节点操作，用 nextPeer 替换当前操作
    struct ncclProxyArgs* nextPeer = freeOp->nextPeer;
    if (*prevOpPtr) {
      (*prevOpPtr)->next = nextPeer;
    } else {
      state->active = nextPeer;
    }
    nextPeer->next = next;
    *(prevOpPtr) = nextPeer;
  } else {
    // 没有对等节点操作，清空代理追加指针
    *(freeOp->proxyAppendPtr) = NULL;
    if (*prevOpPtr) {
      (*prevOpPtr)->next = next;
    } else {
      state->active = next;
    }
  }
  // 将操作返回到空闲池
  freeOp->next = state->pool;
  state->pool = freeOp;
  DEBUG_PROXY_PRINT("Removed %5ld (%5ld) : ", OP_INDEX(freeOp), OP_INDEX(*freeOp->proxyAppendPtr));
#ifdef DEBUG_PROXY
  NCCLCHECK(dumpProxyState(state));
#endif
  return ncclSuccess;
}

// 静态函数：推进代理操作的执行
// 此函数遍历活动操作列表，调用每个操作的进度函数
// 参数 proxyState: 代理状态指针
// 参数 state: 代理进度状态指针
// 参数 opStart: 起始操作指针
// 参数 idle: 输出参数，指示操作是否空闲
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
static ncclResult_t progressOps(struct ncclProxyState* proxyState, struct ncclProxyProgressState* state, struct ncclProxyArgs* opStart, int* idle) {
  struct ncclProxyArgs* prevOp = NULL;
  struct ncclProxyArgs* op = opStart;
  ncclResult_t status = ncclSuccess;
  // 遍历所有活动操作
  while (op) {
    // 检查操作状态
    if (op->state == ncclProxyOpNone)
        return ncclInternalError;
    // 开始计时
    TIME_START(0);
    TIME_START(1);
    // 调用操作的进度函数
    ncclResult_t ret = op->progress(proxyState, op);
    if (op->idle) {
        // 操作空闲，停止空闲计时器
        TIME_STOP(1);
        TIME_CANCEL(0);
    } else {
        // 操作忙碌，取消空闲计时器，停止总计时器
        TIME_CANCEL(1);
        TIME_STOP(0);
    }
    // 更新空闲标志
    *idle &= op->idle;
    // 检查操作是否完成或出错
    if (op->state == ncclProxyOpNone || ret != ncclSuccess) {
      // 跟踪第一个发生的错误
      if (ret != ncclSuccess && status == ncclSuccess)
        status = ret;
      TIME_START(2);
      NCCLCHECK(removeOp(state, &op, &prevOp));
      TIME_STOP(2);
    } else {
      // 操作未完成，移动到下一个操作
      prevOp = op;
      op = op->next;
    }
  }
  return status;
}

// 参数定义：代理追加批量大小
// 控制每次从操作池获取操作的批量大小，默认值为 16
NCCL_PARAM(ProxyAppendBatchSize, "PROXY_APPEND_BATCH_SIZE", 16);

// 静态函数：从共享操作池获取已发布的操作
// 此函数从共享操作池中获取操作并添加到代理进度状态
// 参数 proxyState: 代理状态指针
// 参数 added: 输出参数，返回添加的操作数量
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
static ncclResult_t ncclProxyGetPostedOps(struct ncclProxyState* proxyState, int* added) {
  // 获取代理进度状态和操作池
  struct ncclProxyProgressState* state = &proxyState->progressState;
  if (state->opsPool == NULL) return ncclInternalError;
  struct ncclProxyOpsPool* pool = state->opsPool;

  // 如果有待处理的操作，直接跳到处理部分
  if (state->nextOps != -1) goto process_nextops;

  void* eHandle;
  // 如果有操作需要处理，无需阻塞等待新操作到达或等待锁可用
  // 直接退出，继续处理进度，稍后再回来
  if (state->active != NULL && (pool->nextOps == -1 || pthread_mutex_trylock(&pool->mutex) != 0)) return ncclSuccess;

  if (state->active == NULL) {
    // 没有活动操作，需要等待新操作
    pthread_mutex_lock(&pool->mutex);
    if (pool->nextOps == -1 && !state->stop) {
      // 记录性能分析事件：进入睡眠
      ncclProfilerStartProxyCtrlEvent(proxyState->profilerContext, &eHandle);
      ncclProfilerRecordProxyCtrlEventState(eHandle, 0, ncclProfilerProxyCtrlSleep);
      // 等待条件变量通知
      pthread_cond_wait(&pool->cond, &pool->mutex);
      // 记录性能分析事件：被唤醒
      ncclProfilerRecordProxyCtrlEventState(eHandle, 0, ncclProfilerProxyCtrlWakeup);
      ncclProfilerStopProxyCtrlEvent(eHandle);
    }
  }
  // 获取已发布的操作
  state->nextOps = pool->nextOps;
  pool->nextOps = pool->nextOpsEnd = -1;
  pthread_mutex_unlock(&pool->mutex);

process_nextops:
  // 记录性能分析事件：开始追加操作
  ncclProfilerStartProxyCtrlEvent(proxyState->profilerContext, &eHandle);
  ncclProfilerRecordProxyCtrlEventState(eHandle, 0, ncclProfilerProxyCtrlAppend);
  TIME_START(2);
  // 初始化空闲操作数组
  int freeOp[NCCL_MAX_PROXY_CONNECTIONS];
  int freeOpEnd[NCCL_MAX_PROXY_CONNECTIONS];
  for (int i = 0; i < proxyState->tpLocalnRanks; i++) freeOp[i] = -1;

  // 遍历已发布的操作
  uint64_t lastOpCount = 0;
  int lastPeer = -1;
  int count = 0;
  for (int opIndex = state->nextOps; opIndex != -1;) {
    struct ncclProxyOp* peerOp = pool->ops+opIndex;
    int peer = opIndex / MAX_OPS_PER_PEER;
    // 检查是否需要分批（不同的 opCount 或不同的对等节点）
    if ((lastOpCount && peerOp->opCount != lastOpCount) || ((lastPeer != -1) && peer != lastPeer)) count++;
    // 达到批量大小限制，停止处理
    if (count == ncclParamProxyAppendBatchSize()+1) break;
    lastOpCount = peerOp->opCount;
    lastPeer = peer;
    // 检查连接是否有效
    if (peerOp->connection == NULL) return ncclInternalError;
    // 预取下一个操作到缓存
    if (peerOp->next != -1) __builtin_prefetch(pool->ops+peerOp->next);
    // 将操作添加到代理进度状态
    NCCLCHECK(ProxyAppend(state, peerOp));
    (*added)++;
    int lastOpIndex = opIndex;
    opIndex = peerOp->next;
    // 将操作返回到对等节点池
    if (freeOp[peer] == -1) {
      freeOpEnd[peer] = lastOpIndex;
    } else {
      peerOp->next = freeOp[peer];
    }
    freeOp[peer] = lastOpIndex;
    state->nextOps = opIndex;
  }

  // 将空闲操作返回到共享池
  for (int i = 0; i < proxyState->tpLocalnRanks; i++) {
    if (freeOp[i] == -1) continue;
    int newFree = freeOp[i];
    int oldFree = pool->freeOps[i];
    // Coverity 会被这里的复杂代码结构迷惑。前面的 "for" 循环确保了
    // 只要 freeOp[i] 被初始化（不是 -1），freeOpEnd[i] 就会被初始化。
    // 在当前循环中我们过滤掉未初始化的 freeOp[i]，从而确保 freeOpEnd[i] 也被初始化。
    // coverity[uninit_use:FALSE]
    pool->ops[freeOpEnd[i]].next = oldFree;
    if (oldFree == -1) {
      // 主线程没有消耗的操作，我们可以直接设置
      pool->freeOps[i] = newFree;
    } else {
      // 主线程可能随时回收空闲操作，原子替换 freeOps 值并检查是否成功
      int swap = __sync_val_compare_and_swap(pool->freeOps+i, oldFree, newFree);
      if (swap != oldFree) {
        if (swap != -1) return ncclInternalError;
        // 在我们尝试交换时操作被回收了，现在直接设置值
        pool->ops[freeOpEnd[i]].next = -1;
        pool->freeOps[i] = newFree;
      }
    }
  }
  // 记录性能分析事件：追加操作结束
  ncclProfilerRecordProxyCtrlEventState(eHandle, *added, ncclProfilerProxyCtrlAppendEnd);
  ncclProfilerStopProxyCtrlEvent(eHandle);
  TIME_STOP(2);
  return ncclSuccess;
}

// 引入信号头文件，用于信号处理
#include <signal.h>
// 全局变量：最后一个代理进度状态（用于调试）
static ncclProxyProgressState* ncclLastProxyState;
// 函数：转储代理状态（信号处理函数）
// 此函数由信号处理器调用，用于在发生信号时转储代理状态
// 参数 signal: 触发的信号编号
void ncclDumpProxyState(int signal) {
  dumpProxyState(ncclLastProxyState);
}

// 参数定义：是否创建线程 CUDA 上下文
// 控制是否为代理线程创建独立的 CUDA 上下文，默认值为 0（不创建）
NCCL_PARAM(CreateThreadContext, "CREATE_THREAD_CONTEXT", 0);
// 静态函数：设置代理线程的 CUDA 上下文
// 此函数为代理线程创建或设置 CUDA 上下文
// 参数 proxyState: 代理状态指针
// 返回值：int 类型，1 表示成功设置了上下文，0 表示未设置
static int setProxyThreadContext(struct ncclProxyState* proxyState) {
#if CUDART_VERSION >= 11030
  // 静态变量：是否创建线程上下文（只初始化一次）
  static int createThreadContext = -1;

  if (createThreadContext == -1) {
    // 首次调用，读取参数值
    createThreadContext = ncclParamCreateThreadContext();
    if (createThreadContext) {
      // 检查驱动是否支持必要的 CUDA 函数
      if (CUPFN(cuCtxCreate) == nullptr || CUPFN(cuCtxDestroy) == nullptr || CUPFN(cuCtxSetCurrent) == nullptr) {
        WARN("Unable to create thread context due to old driver, disabling.");
        createThreadContext = 0;
        goto exit;
      }
    }
  }
  if (createThreadContext) {
    if (proxyState->cudaCtx == NULL) {
      // CUDA 上下文不存在，创建新上下文
      if (CUPFN(cuCtxCreate(&proxyState->cudaCtx,
                            NULL, 0, CU_CTX_SCHED_SPIN|CU_CTX_MAP_HOST, proxyState->cudaDev)) != CUDA_SUCCESS) {
        WARN("Failed to create CUDA context on device %d", proxyState->cudaDev);
        createThreadContext = 0;
        goto exit;
      }
    } else {
      // CUDA 上下文已存在，设置为当前上下文
      if (CUPFN(cuCtxSetCurrent(proxyState->cudaCtx)) != CUDA_SUCCESS) {
        WARN("Failed to set CUDA context on device %d", proxyState->cudaDev);
        goto exit;
      }
    }
    return 1;
  }
exit:
#endif
  return 0;
}

// 参数定义：代理转储信号
// 设置为 SIGUSR1 或 SIGUSR2 以在挂起时帮助调试代理状态，默认值为 -1（禁用）
NCCL_PARAM(ProxyDumpSignal, "PROXY_DUMP_SIGNAL", -1);
// 参数定义：进度追加操作频率
// 控制追加代理操作的频率，默认值为 8
NCCL_PARAM(ProgressAppendOpFreq, "PROGRESS_APPENDOP_FREQ", 8);

// 静态变量：代理线程的 CPU 亲和性集合
static cpu_set_t proxyCpuset;
// 静态变量：pthread_once 控制变量，确保初始化只执行一次
static pthread_once_t proxyCpusetOnce = PTHREAD_ONCE_INIT;
// 函数：代理 CPU 集合一次性初始化函数
// 此函数从环境变量读取 NCCL_PROXY_CPUSET 并设置 CPU 亲和性
void proxyCpusetOnceFunc() {
  // 从环境变量获取 CPU 集合配置
  const char* setEnv = ncclGetEnv("NCCL_PROXY_CPUSET");
  if (setEnv) {
    // 解析 CPU 集合字符串
    ncclResult_t res = ncclStrListToCpuset(setEnv, &proxyCpuset);
    if (res != ncclSuccess) {
      INFO(NCCL_ENV, "failed to decode NCCL_PROXY_CPUSET=%s. Ignoring", setEnv);
      goto fail;
    }
    // 输出调试信息
    char msg[1024] = {0};
    cpu_set_t currSet;
    sched_getaffinity(0, sizeof(cpu_set_t), &currSet);
    (void)ncclCpusetToStrList(&currSet, msg, sizeof(msg));
    snprintf(msg + strlen(msg), sizeof(msg) - strlen(msg), " changed to ");
    (void)ncclCpusetToStrList(&proxyCpuset, msg + strlen(msg), sizeof(msg) - strlen(msg));
    INFO(NCCL_ENV, "NCCL_PROXY_CPUSET = %s: %s", setEnv, msg);
    return;
  }
  // 如果到达这里，说明没有环境变量或解析失败
fail:
  CPU_ZERO(&proxyCpuset);
  return;
}

// 函数：代理进度线程主函数
// 此函数是代理进度线程的入口点，负责推进代理操作的执行
// 参数 proxyState_: 代理状态指针（void* 类型以符合 pthread 线程函数签名）
// 返回值：void* 类型的指针（pthread 线程函数标准返回类型）
void* ncclProxyProgress(void *proxyState_) {
  struct ncclProxyState* proxyState = (struct ncclProxyState*)proxyState_;

  // 此线程由 proxyService 创建，因此不需要设置亲和性
  INFO(NCCL_INIT, "[Proxy Progress] Device %d CPU core %d", proxyState->cudaDev, sched_getcpu());

  // 设置 CUDA 上下文
  if (setProxyThreadContext(proxyState)) {
    INFO(NCCL_INIT, "[Proxy Progress] Set CUDA context on device %d", proxyState->cudaDev);
  } else if (cudaSetDevice(proxyState->cudaDev) != cudaSuccess) {
    WARN("[Proxy Progress] Failed to set CUDA device %d", proxyState->cudaDev);
  }

  // 初始化代理进度状态
  struct ncclProxyProgressState* state = &proxyState->progressState;
  state->nextOps = -1;
  // 设置信号处理器用于调试
  const int sig = ncclParamProxyDumpSignal();
  if (sig != -1) signal(sig, ncclDumpProxyState);
  ncclLastProxyState = state;
  // 设置线程名称
  char threadName[NCCL_THREAD_NAMELEN];
  snprintf(threadName, NCCL_THREAD_NAMELEN, "NCCL Progress%2d", proxyState->cudaDev);
  nvtxNameOsThreadA(syscall(SYS_gettid), threadName);

  int lastIdle = 0;
  /* 过于频繁地调用 ncclProxyGetPostedOps() 会导致小消息通信的性能回归。
   * proxyOpAppendCounter 是一个计数器，帮助我们决定是否需要追加代理操作。
   * 每次进度推进后，proxyOpAppendCounter 增加 1 并与环境变量
   * ncclParamProgressAppendOpFreq() 比较。如果相等，我们将追加代理操作。
   * 这会降低调用 ncclProxyGetPostedOps() 的频率并减少性能影响。 */
  int proxyOpAppendCounter = 0;
  // 主循环
  do {
    int idle = 1;
    // 推进所有活动操作
    ncclResult_t ret = progressOps(proxyState, state, state->active, &idle);
    if (ret != ncclSuccess) {
      // 发生错误，保存错误状态并退出
      __atomic_store_n(&proxyState->asyncResult, ret, __ATOMIC_RELEASE);
      INFO(NCCL_ALL,"%s:%d -> %d [Progress Thread]", __FILE__, __LINE__, ret);
      break;
    }
    // 检测空闲状态变化
    if ((lastIdle == 0 && idle == 1) || (lastIdle == 1 && idle == 0)) {
      void* eHandle;
      ncclProfilerStartProxyCtrlEvent(proxyState->profilerContext, &eHandle);
      if (lastIdle == 0 && idle == 1)
        ncclProfilerRecordProxyCtrlEventState(eHandle, 0, ncclProfilerProxyCtrlIdle);
      if (lastIdle == 1 && idle == 0)
        ncclProfilerRecordProxyCtrlEventState(eHandle, 0, ncclProfilerProxyCtrlActive);
      ncclProfilerStopProxyCtrlEvent(eHandle);
    }
    // 根据空闲状态或计数器决定是否获取新操作
    if (idle || !state->active || (++proxyOpAppendCounter == ncclParamProgressAppendOpFreq())) {
      int added = 0;
      proxyOpAppendCounter = 0;
      TIME_START(3);
      ret = ncclProxyGetPostedOps(proxyState, &added);
      if (added) {
        TIME_STOP(3);
      }
      else {
        TIME_CANCEL(3);
      }

      if (ret != ncclSuccess) {
        __atomic_store_n(&proxyState->asyncResult, ret, __ATOMIC_RELEASE);
        INFO(NCCL_ALL,"%s:%d -> %d [Progress Thread]", __FILE__, __LINE__, ret);
      }
      if (added == 0) {
        sched_yield(); // 没有请求被处理，让其他线程运行
      }
    }
    lastIdle = idle;
  // 循环条件：未停止 或 (停止请求但仍有活动操作) 且未收到中止标志
  } while ((state->stop == 0 || (state->stop == 1 && state->active)) && __atomic_load_n(proxyState->abortFlag, __ATOMIC_ACQUIRE) == 0);
  return NULL;
}

// 函数：启动代理操作
// 此函数将所有待发布的代理操作发布到共享操作池
// 参数 comm: NCCL 通信器指针
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
ncclResult_t ncclProxyStart(struct ncclComm* comm) {
  // 获取代理操作数组
  struct ncclProxyOps* proxyOps = comm->proxyState->proxyOps;
  if (proxyOps == NULL) return ncclSuccess;
  TIME_START(1);
  // 遍历所有本地 rank
  for (int r = 0; r < comm->sharedRes->tpNLocalRanks; r++) {
    struct ncclProxyOps* ops = proxyOps + r;
    // 跳过空池或无操作的情况
    if (ops->pool == NULL || ops->nextOps == -1) continue;
    // 发布操作到共享池
    NCCLCHECK(ncclProxyPost(ops->pool, ops->nextOps, ops->nextOpsEnd));
    // 重置操作队列
    ops->nextOps = ops->nextOpsEnd = -1;
    ops->count = 0;
  }
  // 增加操作计数
  comm->opCount++;
  TIME_STOP(1);
  return ncclSuccess;
}

// 静态函数：创建代理进度线程
// 此函数创建代理进度线程并设置线程名称
// 参数 proxyState: 代理状态指针
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
//创建proxy线程
static ncclResult_t ncclProxyProgressCreate(struct ncclProxyState* proxyState) {
  struct ncclProxyProgressState* state = &proxyState->progressState;
  if (!state->thread) {
    // 创建代理进度线程
    PTHREADCHECK(pthread_create(&state->thread, NULL, ncclProxyProgress, proxyState), "pthread_create");
    // 设置线程名称
    ncclSetThreadName(state->thread, "NCCL Progress%2d", proxyState->tpLocalnRanks);
  }
  return ncclSuccess;
}

// 函数：销毁代理进度线程
// 此函数停止代理进度线程并释放相关资源
// 参数 proxyState: 代理状态指针
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
ncclResult_t ncclProxyProgressDestroy(struct ncclProxyState* proxyState) {
  struct ncclProxyProgressState* state = &proxyState->progressState;

  // 请求代理停止并唤醒它
  if (state->opsPool) {
    pthread_mutex_lock(&state->opsPool->mutex);
    state->stop = 1;
    pthread_cond_signal(&state->opsPool->cond);
    pthread_mutex_unlock(&state->opsPool->mutex);
    // 等待线程结束
    PTHREADCHECK(pthread_join(state->thread, NULL), "pthread_join");
  }

  // 释放为代理参数池分配的所有内存
  while (state->pools != NULL) {
    struct ncclProxyPool *next = state->pools->next;
    free(state->pools);
    state->pools = next;
  }

  TIME_PRINT("Proxy");
  return ncclSuccess;
}

// 宏定义：代理连接池大小的幂次方（2^7 = 128）
#define NCCL_PROXY_CONN_POOL_SIZE_POW2 7
// 宏定义：代理连接池大小
#define NCCL_PROXY_CONN_POOL_SIZE (1<<(NCCL_PROXY_CONN_POOL_SIZE_POW2))
// 宏定义：代理连接池掩码（用于取模运算）
#define NCCL_PROXY_CONN_POOL_MASK ((NCCL_PROXY_CONN_POOL_SIZE)-1)
// 结构体：代理连接池
// 此结构体管理代理连接的内存池，使用分层结构以提高内存分配效率
struct ncclProxyConnectionPool {
  // 指向连接池数组的指针（每一层是一个连接数组）
  struct ncclProxyConnection** pools;
  // 层数（已分配的连接池数量）
  int banks;
  // 当前层的偏移量（下一个可用的连接索引）
  int offset;
};

static ncclResult_t ncclProxyNewConnection(struct ncclProxyConnectionPool* pool, int* id) {
  if (pool->offset == NCCL_PROXY_CONN_POOL_SIZE) {
    NCCLCHECK(ncclRealloc(&pool->pools, pool->banks, pool->banks+1));
    NCCLCHECK(ncclCalloc(pool->pools+pool->banks, NCCL_PROXY_CONN_POOL_SIZE));
    pool->banks++;
    pool->offset = 0;
  }
  *id = ((pool->banks-1) << NCCL_PROXY_CONN_POOL_SIZE_POW2) + pool->offset;
  pool->offset++;
  return ncclSuccess;
}

static ncclResult_t ncclProxyGetConnection(struct ncclProxyConnectionPool* pool, int id, struct ncclProxyConnection** conn) {
  int bank = id>>NCCL_PROXY_CONN_POOL_SIZE_POW2;
  int offset = id&NCCL_PROXY_CONN_POOL_MASK;
  if ((pool->pools == NULL) || (bank > pool->banks) || (pool->pools[bank] == NULL)) 
    return ncclInternalError;

  *conn = pool->pools[bank]+offset;
  return ncclSuccess;
}

// 静态函数：释放代理连接
// 此函数根据连接类型调用相应的传输层释放函数
// 参数 connection: 代理连接指针
// 参数 proxyState: 代理状态指针
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
static ncclResult_t proxyFree(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState) {
  if (connection->send) {
    // 发送连接，调用发送传输层的释放函数
    if (ncclTransports[connection->transport]->send.proxyFree) {
      NCCLCHECK(ncclTransports[connection->transport]->send.proxyFree(connection, proxyState));
    }
  } else {
    // 接收连接，调用接收传输层的释放函数
    if (ncclTransports[connection->transport]->recv.proxyFree) {
      NCCLCHECK(ncclTransports[connection->transport]->recv.proxyFree(connection, proxyState));
    }
  }
  return ncclSuccess;
}

// 静态函数：释放所有代理连接
// 此函数遍历连接池并释放所有已初始化的连接
// 参数 pool: 代理连接池指针
// 参数 proxyState: 代理状态指针
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
static ncclResult_t ncclProxyFreeConnections(struct ncclProxyConnectionPool* pool, struct ncclProxyState* proxyState) {
  // 遍历所有层
  for (int b=0; b<pool->banks; b++) {
    // 计算当前层的最大索引
    int max = b == pool->banks-1 ? pool->offset : NCCL_PROXY_CONN_POOL_SIZE;
    // 遍历当前层的所有连接
    for (int i=0; i<max; i++) {
      ncclProxyConnection *connection = pool->pools[b]+i;
      // 只释放已初始化的连接
      if (connection->state != connUninitialized) {
        NCCLCHECK(proxyFree(connection, proxyState));
      }
    }
    // 释放当前层的内存
    free(pool->pools[b]);
  }
  // 释放连接池数组
  free(pool->pools);
  return ncclSuccess;
}

// 引入传输层头文件，包含传输层相关的定义和函数
#include "transport.h"

// 结构体：代理初始化请求
// 此结构体包含代理连接初始化所需的参数
struct ncclProxyInitReq {
  int transport;          // 传输层类型
  int send;               // 是否为发送操作
  int tpLocalRank;        // 顶层父节点本地 rank
  int tpRank;             // 顶层父节点 rank
  int sameProcess;        // 是否在同一进程
};

// 结构体：代理初始化响应
// 此结构体包含代理连接初始化的响应数据
struct ncclProxyInitResp {
  ncclProxyConnection* connection;  // 连接指针
  char devShmPath[6];              // 设备共享内存路径（"XXXXXX" 格式，可能设置也可能不设置）
};

// 函数：连接到代理
// 此函数建立与代理的连接，包括创建套接字、发送初始化请求等
// 参数 comm: NCCL 通信器指针
// 参数 transport: 传输层类型
// 参数 send: 是否为发送操作
// 参数 proxyRank: 代理 rank
// 参数 proxyConn: 代理连接器指针（输出参数）
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
ncclResult_t ncclProxyConnect(struct ncclComm* comm, int transport, int send, int proxyRank, struct ncclProxyConnector* proxyConn) {
  struct ncclSocket* sock;
  int ready;
  struct ncclProxyState* sharedProxyState = comm->proxyState;
  int tpProxyRank = comm->topParentRanks[proxyRank];

//判断是否在同一进程内（单进程或多线程场景）
//例如一个进程管理多个GPU，或者多个线程每个线程管理一个GPU
  proxyConn->sameProcess = ((comm->peerInfo[proxyRank].hostHash == comm->peerInfo[comm->rank].hostHash) &&
                            (comm->peerInfo[proxyRank].pidHash == comm->peerInfo[comm->rank].pidHash)) ? 1 : 0;
  // 每个 local rank 保留一个连接
  proxyConn->connection = NULL;
  proxyConn->tpRank = tpProxyRank;
  proxyConn->rank = proxyRank;
  // 首次连接，分配必要的资源
  if (sharedProxyState->peerSocks == NULL) {
    //分配内存
    NCCLCHECK(ncclCalloc(&sharedProxyState->peerSocks, comm->sharedRes->tpNLocalRanks));
    NCCLCHECK(ncclCalloc(&sharedProxyState->proxyOps, comm->sharedRes->tpNLocalRanks));
    NCCLCHECK(ncclCalloc(&sharedProxyState->sharedDevMems, comm->sharedRes->tpNLocalRanks));
    //描述符fd设置为无效-1
    for (int i = 0; i < comm->sharedRes->tpNLocalRanks; ++i) {
      NCCLCHECK(ncclSocketSetFd(-1, &sharedProxyState->peerSocks[i]));
    }
  }

    //转换为localrank
  proxyConn->tpLocalRank = comm->sharedRes->tpRankToLocalRank[proxyConn->tpRank];
  sock = sharedProxyState->peerSocks + proxyConn->tpLocalRank;
  NCCLCHECK(ncclSocketReady(sock, &ready));
  if (!ready) {
    //和peer建立tcp连接
    NCCLCHECK(ncclSocketInit(sock, sharedProxyState->peerAddresses+proxyConn->tpRank, comm->sharedRes->magic, ncclSocketTypeProxy, comm->abortFlag));
    NCCLCHECK(ncclSocketConnect(sock));
  }

  // 准备初始化请求
  struct ncclProxyInitReq req = {0};
  req.transport = transport;
  req.send = send;
  req.tpLocalRank = comm->topParentLocalRanks[comm->localRank];
  req.tpRank = comm->topParentRanks[comm->rank];
  req.sameProcess = proxyConn->sameProcess;

  struct ncclProxyInitResp resp = {0};
  // This usually sends proxyConn->connection to identify which connection this is.
  // However, this is part of the response and therefore is ignored

  //触发创建ncclProxyProgress线程，发送req数据给proxy
  NCCLCHECK(ncclProxyCallBlocking(comm, proxyConn, ncclProxyMsgInit, &req, sizeof(req), &resp, sizeof(resp)));
  //获取返回的conn指针，这里因为是多线程，共享相同的地址空间
  proxyConn->connection = resp.connection;

  // If we need proxy progress, map progress ops
  //获取当前的传输层通信器
  struct ncclTransportComm* tcomm = send ? &ncclTransports[transport]->send : &ncclTransports[transport]->recv;
    //如果函数指针支持proxy，创建共享内存文件
  if (tcomm->proxyProgress) {
    char poolPath[] = "/dev/shm/nccl-XXXXXX";
    strncpy(poolPath+sizeof("/dev/shm/nccl-")-1, resp.devShmPath, sizeof("XXXXXX")-1);
    struct ncclProxyOps* proxyOps = sharedProxyState->proxyOps + proxyConn->tpLocalRank;
    if (proxyOps->pool == NULL) {
      // 在/dev/shm/目录下创建共享内存文件poolPath
      NCCLCHECK(ncclShmOpen(poolPath, sizeof(poolPath), sizeof(struct ncclProxyOpsPool), (void**)(&proxyOps->pool), NULL, -1, &proxyOps->handle));
      proxyOps->nextOps = proxyOps->nextOpsEnd = proxyOps->freeOp = -1;
    }
  }

  //标记初始化完成
  proxyConn->initialized = true;
  INFO(NCCL_NET|NCCL_PROXY, "Connected to proxy localRank %d -> connection %p", proxyConn->tpLocalRank, proxyConn->connection);
  return ncclSuccess;
}

// UDS support
// 函数实现：通过 Unix Domain Socket (UDS) 发送阻塞式代理调用
// 此函数用于 cuMem API 支持，通过 UDS 与代理进程通信，支持文件描述符传递
// UDS 用于进程间通信，特别是用于传递 CUDA 内存句柄和文件描述符
// 参数 comm: NCCL 通信器指针
// 参数 proxyConn: 代理连接器指针，包含目标代理的连接信息
// 参数 type: 操作类型（如获取文件描述符、查询文件描述符等）
// 参数 reqBuff: 请求数据缓冲区指针
// 参数 reqSize: 请求数据大小
// 参数 respBuff: 响应数据缓冲区指针
// 参数 respSize: 响应数据大小
// 参数 reqFd: 要传递的请求文件描述符指针（可选）
// 参数 respFd: 接收的响应文件描述符指针（可选）
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
ncclResult_t ncclProxyCallBlockingUDS(struct ncclComm* comm, struct ncclProxyConnector* proxyConn, int type, void* reqBuff, int reqSize, void* respBuff, int respSize, int* reqFd, int *respFd) {
  ncclResult_t res = ncclSuccess;                          // 初始化返回值为成功
  struct ncclIpcSocket ipcSock = { 0 };                    // 初始化 IPC socket 结构体（用于 UDS 通信）
  void *opId;                                              // 声明操作 ID 变量（用于标识本次调用）
  NCCLCHECK(getRandomData(&opId, sizeof(opId)));          // 生成随机操作 ID（用于匹配请求和响应）
  int reqFdtmp = -1;                                       // 声明并初始化临时请求文件描述符为 -1

  // 获取当前 rank 在顶级父通信器中的本地 rank 映射
  // topParentLocalRanks 用于在通信器分割时找到原始 rank 编号
  int rank = comm->topParentLocalRanks[comm->localRank];
  // 获取共享代理状态指针（代理状态在父子通信器间共享）
  struct ncclProxyState* sharedProxyState = comm->proxyState;
  // 获取目标代理的 UDS 地址哈希值（用于定位目标进程的 UDS socket）
  uint64_t pidHash = sharedProxyState->peerAddressesUDS[proxyConn->tpRank];

  // 记录调试信息：UDS 代理调用的参数
  INFO(NCCL_PROXY, "ProxyCall UDS comm %p rank %d tpRank %d(%lx) reqSize %d respSize %d respFd %p opId %p",
       comm, rank, proxyConn->tpRank, pidHash, reqSize, respSize, respFd, opId);

  // cuMem: 创建一个 UDS socket 用于接收响应
  // UDS (Unix Domain Socket) 用于同一机器上的进程间通信
  NCCLCHECK(ncclIpcSocketInit(&ipcSock, rank, (uint64_t)opId, comm->abortFlag));

  // 检查是否提供了请求文件描述符
  if (reqFd) {                                             // 如果提供了请求文件描述符
    reqFdtmp = *reqFd;                                     // 使用提供的文件描述符
  } else {                                                  // 如果没有提供文件描述符
    // 为 UDS socket 的另一端提供一个虚拟文件描述符
    // 这个虚拟 fd 用于 UDS 通信协议，即使没有实际需要传递的 fd
    NCCLCHECK(ncclIpcSocketGetFd(&ipcSock, &reqFdtmp));
  }

  // 初始化 IPC 消息头结构体
  ncclIpcHdr hdr;
  memset(&hdr, '\0', sizeof(hdr));                        // 清零消息头
  hdr.type = type;                                        // 设置消息类型（操作类型）
  hdr.rank = rank;                                        // 设置发送者 rank
  hdr.reqSize = reqSize;                                  // 设置请求数据大小
  hdr.respSize = respSize;                                // 设置响应数据大小
  hdr.opId = opId;                                        // 设置操作 ID（用于匹配响应）

  // 确保请求数据大小不超过消息头的数据缓冲区大小
  assert(reqSize <= sizeof(hdr.data));
  // 将请求数据复制到消息头的数据字段中
  memcpy(&hdr.data, reqBuff, reqSize);
  // 通过 UDS 发送消息头和文件描述符，并跳转到 error 标签（如果失败）
  NCCLCHECKGOTO(ncclIpcSocketSendMsg(&ipcSock, &hdr, sizeof(hdr), reqFdtmp, proxyConn->tpRank, pidHash), res, error);
  // 通过 UDS 接收响应数据和文件描述符，并跳转到 error 标签（如果失败）
  NCCLCHECKGOTO(ncclIpcSocketRecvMsg(&ipcSock, respBuff, respSize, respFd), res, error);
  // 关闭 UDS socket，并跳转到 error 标签（如果失败）
  NCCLCHECKGOTO(ncclIpcSocketClose(&ipcSock), res, error);

  // 记录调试信息：UDS 代理调用完成
  INFO(NCCL_PROXY, "ProxyCall UDS comm %p rank %d tpRank %d(%lx) reqSize %d respSize %d respFd %d opId %p - DONE",
       comm, rank, proxyConn->tpRank, pidHash, reqSize, respSize, (respFd ? *respFd : -1), opId);

  return res;                                              // 返回成功状态

error:                                                     // 错误处理标签
  NCCLCHECK(ncclIpcSocketClose(&ipcSock));                // 尝试关闭 UDS socket
  WARN("ncclProxyCallBlockingUDS call to tpRank %d(%lx) failed : %d", proxyConn->tpRank, pidHash, res);
  return res;                                              // 返回错误状态
}

// cuMem API support
// The request/response is sent out-of-band using ncclIpcSocket for this specific command
// 函数实现：获取 cuMem 句柄对应的文件描述符（阻塞式客户端调用）
// 此函数用于 cuMem API 支持，向代理进程请求将 CUDA 内存句柄转换为文件描述符
// 文件描述符可以通过 UDS 在进程间传递，使得其他进程可以访问同一块 GPU 内存
// 参数 comm: NCCL 通信器指针
// 参数 proxyRank: 目标代理的 rank 编号
// 参数 handle: CUDA 内存句柄指针（CUmemGenericAllocationHandle 类型）
// 参数 convertedFd: 输出参数，接收转换后的文件描述符
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
ncclResult_t ncclProxyClientGetFdBlocking(struct ncclComm* comm, int proxyRank, void *handle, int* convertedFd) {
  ncclResult_t ret = ncclSuccess;                          // 初始化返回值为成功

  // Request the allocation of a UDS fd for the handle
  // 请求为该句柄分配一个 UDS 文件描述符
  // 检查代理连接是否已初始化
  if (comm->gproxyConn[proxyRank].initialized == false) {
    // 如果尚未初始化，先建立与目标代理的连接
    // TRANSPORT_P2P 表示使用点对点传输方式
    // 1 表示通道数为 1
    NCCLCHECKGOTO(ncclProxyConnect(comm, TRANSPORT_P2P, 1, proxyRank, &comm->gproxyConn[proxyRank]), ret, error);
  }
  // 通过 UDS 向代理发送获取文件描述符请求
  // ncclProxyMsgGetFd: 消息类型为获取文件描述符
  // handle: CUDA 内存句柄（输入）
  // sizeof(CUmemGenericAllocationHandle): 句柄大小
  // NULL: 不需要请求数据缓冲区
  // 0: 请求数据大小为 0
  // NULL: 不需要响应数据缓冲区
  // 0: 响应数据大小为 0
  // convertedFd: 接收转换后的文件描述符（输出）
  NCCLCHECKGOTO(ncclProxyCallBlockingUDS(comm, &comm->gproxyConn[proxyRank], ncclProxyMsgGetFd, handle, sizeof(CUmemGenericAllocationHandle), NULL, 0, NULL, convertedFd), ret, error);

  // We have now received the converted fd over UDS
  // 已经通过 UDS 接收到转换后的文件描述符
  // 记录调试信息：文件描述符获取成功
  INFO(NCCL_PROXY, "UDS: ClientGetFd handle 0x%lx tpRank %d returned fd %d sameProcess %d", *(uint64_t*)handle, comm->topParentRanks[proxyRank], *convertedFd, comm->gproxyConn[proxyRank].sameProcess);

  return ret;                                              // 返回成功状态

error:                                                     // 错误处理标签
  WARN("ncclProxyClientGetFd call to tpRank %d handle 0x%lx failed : %d", comm->topParentRanks[proxyRank], *(uint64_t*)handle, ret);
  return ret;                                              // 返回错误状态
}

// 函数实现：查询本地文件描述符在远程进程中的对应文件描述符（阻塞式客户端调用）
// 此函数用于 cuMem API 支持，用于内存注册场景
// 当客户端希望将本地内存句柄传递给远程进程时，需要通过代理查询远程进程的文件描述符
// 参数 comm: NCCL 通信器指针
// 参数 proxyConn: 代理连接器指针，包含目标代理的连接信息
// 参数 localFd: 本地文件描述符
// 参数 rmtFd: 输出参数，接收远程进程中的对应文件描述符
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
ncclResult_t ncclProxyClientQueryFdBlocking(struct ncclComm* comm, struct ncclProxyConnector* proxyConn, int localFd, int* rmtFd) {
  ncclResult_t ret = ncclSuccess;                          // 初始化返回值为成功
  // 通过 UDS 向代理发送查询文件描述符请求
  // ncclProxyMsgQueryFd: 消息类型为查询文件描述符
  // NULL: 不需要请求数据缓冲区
  // 0: 请求数据大小为 0
  // (void*)rmtFd: 响应数据缓冲区（接收远程文件描述符）
  // sizeof(int): 响应数据大小（int 类型）
  // &localFd: 要传递的本地文件描述符
  // NULL: 不接收额外的响应文件描述符
  NCCLCHECKGOTO(ncclProxyCallBlockingUDS(comm, proxyConn, ncclProxyMsgQueryFd, NULL, 0, (void*)rmtFd, sizeof(int), &localFd, NULL), ret, fail);
exit:                                                      // 正常退出标签
  // We have now received the converted fd over UDS
  // 已经通过 UDS 接收到远程文件描述符
  // 记录调试信息：文件描述符查询成功
  INFO(NCCL_PROXY, "UDS: ClientQueryFd localFd %d tpRank %d remote fd %d sameProcess %d", localFd, proxyConn->tpRank, *rmtFd, proxyConn->sameProcess);
  return ret;                                              // 返回成功状态
fail:                                                      // 错误处理标签
  WARN("ncclProxyClientQueryFdBlocking call to tpRank %d localFd %d failed : %d", proxyConn->tpRank, localFd, ret);
  goto exit;                                               // 跳转到 exit 标签
}

// 代理消息类型字符串数组（用于调试日志输出）
const char* ncclProxyMsgTypeStr[] = { "Unknown", "Init", "SharedInit", "Setup", "Connect", "Start", "Close", "Abort", "Stop", "GetFd", "QueryFd", "Register", "Deregister" };
// 函数实现：向代理服务发送异步调用请求
// 此函数将请求发送给代理服务线程，但不等待响应
// 异步操作通过 opId 进行标识，后续可以通过 ncclPollProxyResponse 查询结果
// 参数 comm: NCCL 通信器指针
// 参数 proxyConn: 代理连接器指针，包含目标代理的连接信息
// 参数 type: 操作类型（如初始化、连接、注册内存等）
// 参数 reqBuff: 请求数据缓冲区指针
// 参数 reqSize: 请求数据大小
// 参数 respSize: 预期的响应数据大小
// 参数 opId: 操作 ID（用于标识本次异步操作）
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
ncclResult_t ncclProxyCallAsync(struct ncclComm* comm, struct ncclProxyConnector* proxyConn, int type, void* reqBuff, int reqSize, int respSize, void* opId) {
  struct ncclSocket* sock;                               // 声明 socket 指针变量
  ncclResult_t ret = ncclSuccess;                          // 初始化返回值为成功
  struct ncclProxyState* sharedProxyState = comm->proxyState; // 获取共享代理状态指针

  // 检查代理的 peer socket 数组是否已分配
  // peerSocks 在代理初始化时分配，用于与各 peer 通信
  if (sharedProxyState->peerSocks == NULL)
    return ncclInternalError;                            // 如果未分配，返回内部错误

  // 获取目标 peer 的 socket 指针
  // tpLocalRank 是目标 peer 在本地 rank 数组中的索引
  sock = sharedProxyState->peerSocks + proxyConn->tpLocalRank;

  // 发送操作类型（int 大小）
  NCCLCHECKGOTO(ncclSocketSend(sock, &type, sizeof(int)), ret, error);
  // 发送连接指针（用于代理识别操作所属的连接）
  NCCLCHECKGOTO(ncclSocketSend(sock, &proxyConn->connection, sizeof(void*)), ret, error);
  // 发送请求数据大小
  NCCLCHECKGOTO(ncclSocketSend(sock, &reqSize, sizeof(int)), ret, error);
  // 发送响应数据大小
  NCCLCHECKGOTO(ncclSocketSend(sock, &respSize, sizeof(int)), ret, error);
  // 如果有请求数据，发送请求数据
  if (reqSize)
    NCCLCHECKGOTO(ncclSocketSend(sock, reqBuff, reqSize), ret, error);

  // Send opId to proxy
  // 将操作 ID 发送给代理，用于后续匹配响应
  NCCLCHECKGOTO(ncclSocketSend(sock, &opId, sizeof(opId)), ret, error);

  // Add proxyOp to expected response queue
  // 将此操作添加到预期响应队列中，用于后续接收响应
  NCCLCHECK(expectedProxyResponseEnqueue(sharedProxyState, opId, respSize));

  return ncclSuccess;                                      // 返回成功状态
error:                                                     // 错误处理标签
  return ret;                                              // 返回错误状态
}

// 函数实现：轮询代理服务的异步响应
// 此函数用于查询异步操作是否完成，如果完成则获取响应结果
// 它首先检查响应队列，如果没有则尝试从 socket 读取新响应
// 参数 comm: NCCL 通信器指针
// 参数 proxyConn: 代理连接器指针
// 参数 respBuff: 响应数据缓冲区指针（用于接收响应数据）
// 参数 opId: 要查询的操作 ID
// 返回值：ncclResult_t 类型，返回 ncclInProgress 表示操作未完成，其他值表示操作结果
ncclResult_t ncclPollProxyResponse(struct ncclComm* comm, struct ncclProxyConnector* proxyConn, void* respBuff, void* opId) {
  struct ncclProxyState* sharedProxyState = comm->proxyState; // 获取共享代理状态指针
  // Receive the connection pointer from the Proxy
  // 从代理接收连接指针
  // 检查通信器是否处于中止状态（原子加载，获取语义）
  if (__atomic_load_n(comm->abortFlag, __ATOMIC_ACQUIRE)) {
    WARN("Comm %p is in abort state", comm);              // 警告：通信器已中止
    return ncclInternalError;                            // 返回内部错误
  }
  // 检查代理的 peer socket 数组是否已分配
  if (sharedProxyState->peerSocks == NULL)
    return ncclInternalError;                            // 如果未分配，返回内部错误

  // Check response queue
  // 检查响应队列中是否有等待的响应
  int found = 0;                                          // 初始化找到标志为 0
  // 尝试从预期响应队列中取出响应
  ncclResult_t res = expectedProxyResponseDequeue(sharedProxyState, opId, respBuff, &found);
  // 如果没有找到缓存的响应
  if (found == 0) {
    // Attempt to read in a new response header from the proxy thread
    // 尝试从代理线程读取新的响应头
    struct ncclSocket* sock = sharedProxyState->peerSocks + proxyConn->tpLocalRank; // 获取目标 peer 的 socket
    ncclProxyRpcResponseHeader resp = {0};                 // 初始化响应头结构体
    int offset = 0;                                        // 初始化接收偏移量为 0
    // 尝试从 socket 接收响应头（非阻塞）
    if (ncclSuccess != ncclSocketProgress(NCCL_SOCKET_RECV, sock, &resp, sizeof(resp), &offset)) {
      WARN("Socket recv failed while polling for opId=%p", opId);
      return ncclInternalError;                          // Socket 接收失败，返回内部错误
    }

    // 检查是否接收到数据
    if (offset == 0) {
      return ncclInProgress;                             // 没有接收到数据，操作仍在进行中
    // If we've returned a partial response, block to receive the rest of it
    // 如果接收到部分响应，阻塞接收剩余数据
    } else if (offset < sizeof(resp)) {
      // 循环接收直到读取完整的响应头
      while (offset < sizeof(resp))
        NCCLCHECK(ncclSocketProgress(NCCL_SOCKET_RECV, sock, &resp, sizeof(resp), &offset));
    }

    // 记录调试信息：接收到新的响应
    INFO(NCCL_PROXY, "ncclPollProxyResponse Received new opId=%p", resp.opId);

    // If there's a respSize to recv
    // 如果需要接收响应数据
    if (resp.respSize > 0) {
      // 检查响应的操作 ID 是否与期望的操作 ID 匹配
      if (resp.opId != opId) {
        // Unexpected response, need to buffer the socket data
        // 意外的响应，需要缓存 socket 数据
        respBuff = malloc(resp.respSize);                 // 分配内存存储响应数据
      }
      assert(respBuff != NULL);                           // 断言响应缓冲区不为空
      NCCLCHECK(ncclSocketRecv(sock, respBuff, resp.respSize)); // 接收响应数据
    }

    // 检查响应的操作 ID 是否与期望的操作 ID 匹配
    if (resp.opId == opId) {
      INFO(NCCL_PROXY, "resp.opId=%p matches expected opId=%p", resp.opId, opId);
      // 从预期响应队列中移除此操作 ID
      NCCLCHECK(expectedProxyResponseRemove(sharedProxyState, resp.opId));
      return resp.res;                                    // 返回响应结果
    } else {
      INFO(NCCL_PROXY, "Queuing opId=%p respBuff=%p respSize=%d", resp.opId, respBuff, resp.respSize);
      // Store the result and mark response as completed
      // 存储结果并标记响应为已完成
      NCCLCHECK(expectedProxyResponseStore(sharedProxyState, resp.opId, respBuff, resp.respSize, resp.res));
      return ncclInProgress;                             // 返回进行中状态（因为这不是我们等待的操作）
    }
  } else {                                                // 如果找到了缓存的响应
    INFO(NCCL_PROXY, "ncclPollProxyResponse Dequeued cached opId=%p", opId);
  }

  return res;                                              // 返回响应结果
}

// 函数实现：向代理服务发送阻塞式调用请求
// 此函数封装了异步调用和轮询响应的逻辑，提供同步的阻塞式接口
// 它会一直等待直到操作完成或出错
// 参数 comm: NCCL 通信器指针
// 参数 proxyConn: 代理连接器指针，包含目标代理的连接信息
// 参数 type: 操作类型（如初始化、连接、注册内存等）
// 参数 reqBuff: 请求数据缓冲区指针
// 参数 reqSize: 请求数据大小
// 参数 respBuff: 响应数据缓冲区指针（用于接收响应数据）
// 参数 respSize: 预期的响应数据大小
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
ncclResult_t ncclProxyCallBlocking(struct ncclComm* comm, struct ncclProxyConnector* proxyConn, int type, void* reqBuff, int reqSize, void* respBuff, int respSize) {
  // Alloc some memory to act as a handle
  // 分配一些内存作为操作句柄（用于标识本次操作）
  ncclResult_t res = ncclSuccess;                          // 初始化返回值为成功
  void* opId = malloc(1);                                 // 分配 1 字节内存作为操作 ID（只需要唯一地址）

  // 发送异步调用请求，如果失败则跳转到 fail 标签
  NCCLCHECKGOTO(ncclProxyCallAsync(comm, proxyConn, type, reqBuff, reqSize, respSize, opId), res, fail);

  // 循环轮询直到操作完成
  do {
    // 轮询代理服务的响应
    res = ncclPollProxyResponse(comm, proxyConn, respBuff, opId);
  } while (res == ncclInProgress);                        // 如果返回 ncclInProgress，继续轮询

exit:                                                      // 正常退出标签
  free(opId);                                             // 释放操作 ID 内存
  return res;                                              // 返回操作结果
fail:                                                      // 错误处理标签
  goto exit;                                               // 跳转到 exit 标签
}

// 函数实现：初始化代理进度状态
// 此函数创建共享内存操作池，用于管理代理服务的异步操作
// 操作池在多个进程间共享，使用进程间同步原语进行协调
// 参数 proxyState: 代理状态指针
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
static ncclResult_t proxyProgressInit(struct ncclProxyState* proxyState) {
  struct ncclProxyProgressState* state = &proxyState->progressState; // 获取进度状态指针
  // 检查操作池是否已经初始化
  if (state->opsPool == NULL) {
    int size = sizeof(struct ncclProxyOpsPool);            // 计算操作池结构体的大小
    struct ncclProxyOpsPool* pool = NULL;                  // 声明操作池指针

    char shmPath[sizeof("/dev/shm/nccl-XXXXXX")];          // 声明共享内存路径数组
    shmPath[0] = '\0';                                     // 初始化路径字符串为空
    //返回在/dev/shm申请的内存，指针写入到pool中
    // 在共享内存（/dev/shm）中打开/创建操作池
    // shmPath: 输出参数，接收生成的共享内存路径
    // size: 共享内存大小
    // pool: 输出参数，接收映射的内存地址
    // NULL: 不使用特定权限
    // proxyState->tpLocalnRanks: 本地 rank 数量
    // state->handle: 输出参数，接收共享内存句柄
    NCCLCHECK(ncclShmOpen(shmPath, sizeof(shmPath), size, (void**)&pool, NULL, proxyState->tpLocalnRanks, &state->handle));
    // Init pool
    // 初始化操作池
    pool->nextOps = -1;                                    // 初始化下一个操作索引为 -1（表示无操作）

    // 为每个 peer 初始化空闲操作链表
    // 每个 peer 有 MAX_OPS_PER_PEER 个操作槽位
    // 空闲链表将所有空闲操作连接起来，便于快速分配
    for (int r = 0; r < proxyState->tpLocalnRanks; r++) {   // 遍历所有本地 peer
      pool->freeOps[r] = r*MAX_OPS_PER_PEER;               // 设置此 peer 的第一个空闲操作索引
      // 构建空闲操作链表（除最后一个外，每个操作的 next 指向下一个）
      for (int i=0; i<MAX_OPS_PER_PEER-1; i++)
        pool->ops[r*MAX_OPS_PER_PEER+i].next = r*MAX_OPS_PER_PEER+i+1;
      // 最后一个空闲操作的 next 设置为 -1（链表结束）
      pool->ops[(r+1)*MAX_OPS_PER_PEER-1].next = -1;
    }

    //对进程间共享的内存区域初始化mutex
    // Setup mutex/cond to work inter-process
    // 设置互斥锁和条件变量，使其在进程间工作
    pthread_mutexattr_t mutexAttr;                         // 声明互斥锁属性变量
    pthread_mutexattr_init(&mutexAttr);                    // 初始化互斥锁属性
    pthread_mutexattr_setpshared(&mutexAttr, PTHREAD_PROCESS_SHARED); // 设置为进程间共享
    pthread_mutex_init(&pool->mutex, &mutexAttr);          // 使用该属性初始化互斥锁
    pthread_mutexattr_destroy(&mutexAttr);                 // 销毁互斥锁属性
    pthread_condattr_t condAttr;                           // 声明条件变量属性变量
    pthread_condattr_init(&condAttr);                      // 初始化条件变量属性
    pthread_condattr_setpshared(&condAttr, PTHREAD_PROCESS_SHARED); // 设置为进程间共享
    pthread_cond_init(&pool->cond, &condAttr);             // 使用该属性初始化条件变量
    pthread_condattr_destroy(&condAttr);                   // 销毁条件变量属性
    state->opsPool = pool;                                 // 保存操作池指针到状态中

    // 保存共享内存路径后缀（用于后续 unlink 操作）
    // 路径格式为 /dev/shm/nccl-XXXXXX，这里提取 XXXXXX 部分
    memcpy(state->opsPoolShmSuffix, shmPath+sizeof("/dev/shm/nccl-")-1, sizeof("XXXXXX")-1);

    // All ops structures are created, we can start the progress thread
    // 所有操作结构体已创建完成，现在可以启动进度线程
    NCCLCHECK(ncclProxyProgressCreate(proxyState));       // 创建进度线程
  }
  return ncclSuccess;                                      // 返回成功状态
}

// 函数实现：释放代理操作池的共享内存资源
// 此函数关闭共享内存的映射，但不删除共享内存文件
// 共享内存文件需要通过 ncclProxyShmUnlink 显式删除
// 参数 proxyState: 代理状态指针
// 返回值：无（void 函数）
static void proxyOpsFree(struct ncclProxyState* proxyState) {
  struct ncclProxyProgressState* state = &proxyState->progressState; // 获取进度状态指针
  // 关闭共享内存映射（解映射）
  // 如果关闭失败，记录警告日志
  if (ncclShmClose(state->handle) != ncclSuccess) {
    WARN("[Service thread] shm close failed");            // 警告：共享内存关闭失败
  }
}

// 函数实现：删除代理操作池的共享内存文件
// 此函数删除 /dev/shm 下的共享内存文件，释放系统资源
// 应该在所有进程都关闭共享内存映射后调用
// 参数 comm: NCCL 通信器指针
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
ncclResult_t ncclProxyShmUnlink(struct ncclComm* comm) {
  struct ncclProxyProgressState* state = &comm->proxyState->progressState; // 获取进度状态指针
  // 检查操作池是否存在
  if (state->opsPool == NULL)
    return ncclSuccess;                                    // 如果不存在，直接返回成功

  // 删除共享内存文件（从文件系统中移除）
  // 即使某个进程仍映射了该内存，unlink 后新进程无法再访问
  if (ncclShmUnlink(state->handle) != ncclSuccess) {
    WARN("[Service thread] proxy ops shm unlink failed"); // 警告：共享内存 unlink 失败
  }
  return ncclSuccess;                                      // 返回成功状态
}

// 函数实现：初始化代理连接
// 此函数为代理服务创建一个新的连接，并配置连接参数
// 连接可以用于发送或接收操作，支持多种传输类型
// 参数 peer: 本地 peer 指针（包含 socket 和 rank 信息）
// 参数 connectionPool: 连接池指针（用于分配连接）
// 参数 proxyState: 代理状态指针
// 参数 req: 初始化请求指针（包含连接配置参数）
// 参数 resp: 初始化响应指针（用于返回连接信息给调用者）
// 参数 connection: 输出参数，接收创建的连接指针
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
static ncclResult_t proxyConnInit(struct ncclProxyLocalPeer* peer, struct ncclProxyConnectionPool* connectionPool, struct ncclProxyState* proxyState, ncclProxyInitReq* req, ncclProxyInitResp* resp, struct ncclProxyConnection** connection) {
  int id;                                                  // 声明连接 ID 变量
  //分配一个conn
  // 从连接池中分配一个新的连接
  NCCLCHECK(ncclProxyNewConnection(connectionPool, &id));
  //获取conn
  // 从连接池中获取连接指针
  NCCLCHECK(ncclProxyGetConnection(connectionPool, id, connection));

  // 配置连接参数
  (*connection)->sock = &peer->sock;                       // 设置连接使用的 socket
  (*connection)->transport = req->transport;               // 设置传输类型（如 P2P、SHM、Network 等）
  (*connection)->send = req->send;                         // 设置是否为发送连接（true/false）
  (*connection)->tpLocalRank = req->tpLocalRank;           // 设置本地 rank 索引
  (*connection)->sameProcess = req->sameProcess;           // 设置是否为同一进程
  // 更新 peer 信息
  peer->tpLocalRank = req->tpLocalRank;                    // 保存本地 rank 索引
  peer->tpRank = req->tpRank;                              // 保存全局 rank 索引

  // 将连接指针返回给调用者（通过响应结构体）
  resp->connection = *connection;

  // 获取传输层的通信函数指针
  // 根据是发送还是接收，选择对应的传输层接口
  (*connection)->tcomm = (*connection)->send ? &ncclTransports[(*connection)->transport]->send : &ncclTransports[(*connection)->transport]->recv;
  // If we need proxy progress, let's allocate ops and start the thread
  // 如果需要代理进度，分配操作池并启动进度线程
  //检查是否支持proxyProgress，如果支持则创建ncclProxyProgress
  if ((*connection)->tcomm->proxyProgress) {               // 如果传输层需要代理进度
    NCCLCHECK(proxyProgressInit(proxyState));             // 初始化代理进度状态
    struct ncclProxyProgressState* state = &proxyState->progressState; // 获取进度状态
    // 将共享内存路径后缀复制到响应中（用于设备侧访问）
    strncpy(resp->devShmPath, state->opsPoolShmSuffix, sizeof(resp->devShmPath));
  }

  // 记录调试信息：新代理连接已创建
  INFO(NCCL_NET|NCCL_PROXY, "New proxy %s connection %d from local rank %d, transport %d", (*connection)->send ? "send":"recv", id, (*connection)->tpLocalRank, (*connection)->transport);
  // 原子存储连接状态为已初始化（释放语义）
  
  __atomic_store_n(&(*connection)->state, connInitialized, __ATOMIC_RELEASE);
  return ncclSuccess;                                      // 返回成功状态
}

// 函数实现：查询文件描述符（代理服务端）
// 此函数在代理服务端处理查询文件描述符请求
// 它将远程文件描述符通过 UDS 发送回请求者
// 此功能用于 CUDA 11.3+ 版本的 cuMem API 支持
// 参数 proxyState: 代理状态指针
// 参数 rank: 请求者的 rank 编号
// 参数 opId: 操作 ID（用于标识本次操作）
// 参数 rmtFd: 要发送的远程文件描述符
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
static ncclResult_t proxyQueryFd(struct ncclProxyState* proxyState, int rank, void *opId, int rmtFd) {
#if CUDART_VERSION >= 11030                                // 检查 CUDA 运行时版本是否支持 cuMem API（11.3+）
  struct ncclIpcSocket ipcSock = { 0 };                    // 初始化 IPC socket 结构体
  uint64_t hash = (uint64_t) opId;                         // 将操作 ID 转换为哈希值
  ncclResult_t ret = ncclSuccess;                          // 初始化返回值为成功

  // 初始化 UDS socket
  // hash^1 用于生成与客户端对应的 socket 路径（客户端使用 hash，服务端使用 hash^1）
  NCCLCHECKGOTO(ncclIpcSocketInit(&ipcSock, proxyState->tpRank, hash^1, proxyState->abortFlag), ret, exit);
  // 通过 UDS 发送文件描述符
  // &rmtFd: 要发送的数据（文件描述符值）
  // sizeof(int): 数据大小
  // -1: 不发送额外的文件描述符（因为 rmtFd 本身就是要传递的值）
  // rank: 目标 rank
  // hash: 操作哈希值
  NCCLCHECKGOTO(ncclIpcSocketSendMsg(&ipcSock, &rmtFd, sizeof(int), -1, rank, hash), ret, exit);
exit:                                                      // 退出标签
  NCCLCHECK(ncclIpcSocketClose(&ipcSock));                // 关闭 UDS socket
  return ncclSuccess;                                      // 返回成功状态
#else                                                      // 如果 CUDA 版本低于 11.3
  return ncclInternalError;                               // 返回内部错误（不支持此功能）
#endif
}

// cuMem API support
// 函数实现：获取 cuMem 句柄对应的文件描述符（代理服务端）
// 此函数在代理服务端处理获取文件描述符请求
// 它将 CUDA 内存句柄导出为 POSIX 文件描述符，并通过 UDS 发送回请求者
// 此功能用于 CUDA 11.3+ 版本的 cuMem API 支持
// 参数 proxyState: 代理状态指针
// 参数 rank: 请求者的 rank 编号
// 参数 opId: 操作 ID（用于标识本次操作）
// 参数 handle: CUDA 内存分配句柄（CUmemGenericAllocationHandle）
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
static ncclResult_t proxyGetFd(struct ncclProxyState* proxyState, int rank, void *opId, uint64_t handle) {
#if CUDART_VERSION >= 11030                                // 检查 CUDA 运行时版本是否支持 cuMem API（11.3+）
  // cuMem API support
  // cuMem API 支持
  ncclResult_t ret = ncclSuccess;                          // 初始化返回值为成功
  struct ncclIpcSocket ipcSock = { 0 };                    // 初始化 IPC socket 结构体
  uint64_t hash = (uint64_t) opId;                         // 将操作 ID 转换为哈希值
  INFO(NCCL_PROXY, "UDS proxyGetFd received handle 0x%lx peer %d opId %lx", handle, rank, hash);

  // 设置句柄类型为 POSIX 文件描述符
  CUmemAllocationHandleType type = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
  int fd = -1;                                             // 声明并初始化文件描述符为 -1

  // 将 CUDA 内存句柄导出为可共享的文件描述符
  // fd: 输出参数，接收导出的文件描述符
  // handle: CUDA 内存分配句柄
  // type: 导出类型（POSIX 文件描述符）
  // 0: 保留标志（必须为 0）
  CUCHECK(cuMemExportToShareableHandle(&fd, handle, type, 0));
  // Send back the converted fd using UDS
  // 使用 UDS 将转换后的文件描述符发送回客户端
  // hash^1 用于生成与客户端对应的 socket 路径
  NCCLCHECKGOTO(ncclIpcSocketInit(&ipcSock, proxyState->tpRank, hash^1, proxyState->abortFlag), ret, error);
  // 通过 UDS 发送文件描述符
  NCCLCHECKGOTO(ncclIpcSocketSendFd(&ipcSock, fd, rank, hash), ret, error);
error:                                                     // 错误处理标签
  NCCLCHECK(ncclIpcSocketClose(&ipcSock));                // 关闭 UDS socket
  // We can now safely close the exported fd
  // 现在可以安全关闭导出的文件描述符
  // 注意：发送文件描述符会复制它，所以关闭原始 fd 是安全的
  SYSCHECK(close(fd), "close");                           // 关闭文件描述符
  return ret;                                              // 返回操作结果
#else                                                      // 如果 CUDA 版本低于 11.3
  return ncclInternalError;                               // 返回内部错误（不支持此功能）
#endif
}

// 函数实现：推进异步操作的执行
// 此函数在代理服务线程中调用，处理各种类型的异步操作
// 它根据操作类型调用相应的传输层函数，并管理操作的生命周期
// 参数 op: 异步操作指针
// 参数 proxyState: 代理状态指针
// 参数 asyncOpCount: 异步操作计数指针（用于跟踪进行中的操作数量）
// 参数 peer: 本地 peer 指针
// 参数 connectionPool: 连接池指针
// 返回值：ncclResult_t 类型，返回 ncclInProgress 表示操作未完成，ncclSuccess 表示完成
static ncclResult_t proxyProgressAsync(struct ncclProxyAsyncOp* op, struct ncclProxyState* proxyState, int* asyncOpCount, struct ncclProxyLocalPeer* peer, struct ncclProxyConnectionPool* connectionPool) {
  int done = 1;                                            // 初始化完成标志为 1
  ncclResult_t res = ncclInternalError;                    // 初始化结果为内部错误
  // 根据操作类型分发处理
  if (op->type == ncclProxyMsgSetup) {                     // 如果操作类型是 Setup
    TRACE(NCCL_PROXY, "proxyProgressAsync::proxySetup() opId=%p", op->opId);
    // 调用传输层的 proxySetup 函数
    res = op->connection->tcomm->proxySetup(op->connection, proxyState, op->reqBuff, op->reqSize, op->respBuff, op->respSize, &done);
  } else if (op->type == ncclProxyMsgConnect) {            // 如果操作类型是 Connect
    TRACE(NCCL_PROXY, "proxyProgressAsync::proxyConnect() opId=%p op.reqBuff=%p", op->opId, op->reqBuff);
    // 调用传输层的 proxyConnect 函数
    res = op->connection->tcomm->proxyConnect(op->connection, proxyState, op->reqBuff, op->reqSize, op->respBuff, 
        op->respSize, &done);
  } else if (op->type == ncclProxyMsgSharedInit) {         // 如果操作类型是 SharedInit
    int nChannels = (int) *op->reqBuff;                    // 获取通道数量（从请求数据中读取）
    TRACE(NCCL_PROXY, "proxyProgressAsync::ncclProxyMsgSharedInit opId=%p op.reqBuff=%p nChannels=%d", op->opId, op->reqBuff, nChannels);
    // 检查传输层是否支持共享初始化
    if (op->connection->tcomm->proxySharedInit) 
        res = op->connection->tcomm->proxySharedInit(op->connection, proxyState, nChannels);
    // 原子存储连接状态为共享已初始化
    __atomic_store_n(&op->connection->state, connSharedInitialized, __ATOMIC_RELEASE);
  }
  else if (op->type == ncclProxyMsgInit) {                 // 如果操作类型是 Init
    TRACE(NCCL_PROXY, "proxyProgressAsync::ncclProxyMsgInit opId=%p op.reqBuff=%p", op->opId, op->reqBuff);
    // 调用 proxyConnInit 初始化连接
    res = proxyConnInit(peer, connectionPool, proxyState, (ncclProxyInitReq*) op->reqBuff, (ncclProxyInitResp*) op->respBuff, &op->connection);
  } else if (op->type == ncclProxyMsgRegister) {           // 如果操作类型是 Register
    TRACE(NCCL_PROXY, "proxyProgressAsync::ncclProxyMsgRegister opId=%p op.reqBuff=%p, op->reqSize=%d, op->respSize=%d", op->opId, op->reqBuff,
        op->reqSize, op->respSize);
    // 调用传输层的 proxyRegister 函数
    res = op->connection->tcomm->proxyRegister(op->connection, proxyState, op->reqBuff, op->reqSize, op->respBuff, 
    op->respSize, &done);
  } else if (op->type == ncclProxyMsgDeregister) {         // 如果操作类型是 Deregister
    TRACE(NCCL_PROXY, "proxyProgressAsync::ncclProxyMsgDeregister opId=%p op.reqBuff=%p, op->reqSize=%d, op->respSize=%d",
        op->opId, op->reqBuff, op->reqSize, op->respSize);
    // 调用传输层的 proxyDeregister 函数
    res = op->connection->tcomm->proxyDeregister(op->connection, proxyState, op->reqBuff, op->reqSize, &done);
  } else
    return ncclInternalError;                             // 未知的操作类型，返回内部错误

  // 检查操作是否完成
  if (done) {                                              // 如果操作已完成
    INFO(NCCL_PROXY, "proxyProgressAsync opId=%p op.type=%d op.reqBuff=%p op->respSize=%d done", op->opId, op->type, 
        op->reqBuff, op->respSize);
    // 根据操作类型更新连接状态
    if (op->type == ncclProxyMsgSetup)
      __atomic_store_n(&op->connection->state, connSetupDone, __ATOMIC_RELEASE);
    else if (op->type == ncclProxyMsgConnect)
      __atomic_store_n(&op->connection->state, connConnected, __ATOMIC_RELEASE);
    /* if setup or connect is done, we should not return any error at this point since
     * ncclSocketSend might already send the respBuff to the requester. If we still choose
     * to abort and close the connection, it can cause segfault if the requester is using
     * the respBuff. */
    /* 如果 setup 或 connect 操作已完成，此时不应返回任何错误，因为
     * ncclSocketSend 可能已经将 respBuff 发送给请求者。如果我们仍然选择
     * 中止并关闭连接，当请求者正在使用 respBuff 时可能会导致段错误。 */

    // 构造 RPC 响应头
    ncclProxyRpcResponseHeader resp = {op->opId, res, op->respSize};

    // Send the opId for referencing async operation
    // 发送操作 ID 以引用异步操作
    NCCLCHECK(ncclSocketSend(op->connection->sock, &resp, sizeof(resp)));

    // 如果有响应数据，发送响应
    if (op->respSize) {
      // Send the response
      // 发送响应数据
      NCCLCHECK(ncclSocketSend(op->connection->sock, op->respBuff, op->respSize));
    }

    // 从异步操作队列中移除此操作
    asyncProxyOpDequeue(peer, op);
    // 减少异步操作计数
    (*asyncOpCount)--;
    return ncclSuccess;                                    // 返回成功状态

  } else if (__atomic_load_n(proxyState->abortFlag, __ATOMIC_ACQUIRE) != 0) {
    // 如果操作未完成但已设置中止标志，返回内部错误
    return ncclInternalError;
  }

  return ncclInProgress;                                   // 操作仍在进行中，返回进行中状态
}

// 函数实现：代理服务初始化操作
// 此函数在代理服务线程中接收并初始化一个新的异步操作
// 它从 socket 读取操作参数，创建异步操作结构体，并将其加入队列
// 参数 type: 操作类型（如初始化、连接、注册内存等）
// 参数 peer: 本地 peer 指针（包含 socket 和操作队列）
// 参数 connectionPool: 连接池指针
// 参数 proxyState: 代理状态指针
// 参数 asyncOpCount: 异步操作计数指针（用于跟踪进行中的操作数量）
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
static ncclResult_t proxyServiceInitOp(int type, struct ncclProxyLocalPeer* peer, struct ncclProxyConnectionPool* connectionPool, struct ncclProxyState* proxyState, int* asyncOpCount) {
  ncclResult_t ret = ncclSuccess;                          // 初始化返回值为成功
  struct ncclSocket* sock = &peer->sock;                   // 获取 peer 的 socket 指针
  struct ncclProxyAsyncOp* asyncOp;                        // 声明异步操作指针
  NCCLCHECK(ncclCalloc(&asyncOp, 1));                      // 分配并清零异步操作结构体

  asyncOp->type = type;                                    // 设置操作类型
  // 从 socket 接收连接指针（客户端发送的连接句柄）
  NCCLCHECKGOTO(ncclSocketRecv(sock, &asyncOp->connection, sizeof(void*)), ret, fail);

  // 从 socket 接收请求数据大小
  NCCLCHECKGOTO(ncclSocketRecv(sock, &asyncOp->reqSize, sizeof(int)), ret, fail);
  // 从 socket 接收响应数据大小
  NCCLCHECKGOTO(ncclSocketRecv(sock, &asyncOp->respSize, sizeof(int)), ret, fail);
  // 如果有请求数据，分配内存并接收
  if (asyncOp->reqSize) {                                  // 如果请求数据大小大于 0
    NCCLCHECKGOTO(ncclCalloc(&asyncOp->reqBuff, asyncOp->reqSize), ret, fail); // 分配请求缓冲区
    NCCLCHECKGOTO(ncclSocketRecv(sock, asyncOp->reqBuff, asyncOp->reqSize), ret, fail); // 接收请求数据
  }

  // Store opId for completion response
  // 接收并存储操作 ID（用于完成响应的匹配）
  NCCLCHECKGOTO(ncclSocketRecv(sock, &asyncOp->opId, sizeof(asyncOp->opId)), ret, fail);

  // 如果有响应数据，分配响应缓冲区
  if (asyncOp->respSize) NCCLCHECKGOTO(ncclCalloc(&asyncOp->respBuff, asyncOp->respSize), ret, fail);

  // 将异步操作加入 peer 的操作队列
  asyncProxyOpEnqueue(peer, asyncOp);

  // 增加异步操作计数
  (*asyncOpCount)++;
  // 尝试推进异步操作的执行
  NCCLCHECK(proxyProgressAsync(asyncOp, proxyState, asyncOpCount, peer, connectionPool));
exit:                                                      // 正常退出标签
  return ret;                                              // 返回操作结果
fail:                                                      // 错误处理标签
  if (asyncOp->reqBuff) free(asyncOp->reqBuff);            // 释放请求缓冲区
  if (asyncOp->respBuff) free(asyncOp->respBuff);          // 释放响应缓冲区
  free(asyncOp);                                           // 释放异步操作结构体
  goto exit;                                               // 跳转到 exit 标签
}

#include <poll.h>

// 函数实现：检查操作类型是否匹配有效的代理操作
// 此函数用于判断给定的操作类型是否是需要代理服务的异步操作
// 只有返回 true 的操作类型才会被代理服务处理
// 参数 type: 要检查的操作类型
// 返回值：bool 类型，true 表示是有效的代理操作类型，false 表示不是
static bool proxyMatchOpType(int type) {
  switch (type) {
    case ncclProxyMsgInit:                                // 初始化操作
    case ncclProxyMsgSharedInit:                          // 共享初始化操作
    case ncclProxyMsgSetup:                               // 设置操作
    case ncclProxyMsgConnect:                             // 连接操作
    case ncclProxyMsgGetFd:                               // 获取文件描述符操作
    case ncclProxyMsgRegister:                            // 内存注册操作
    case ncclProxyMsgDeregister:                          // 内存注销操作
      return true;                                        // 这些都是有效的代理操作类型
    default:
      return false;                                       // 其他操作类型不需要代理服务
  }
}

enum {
  PROXY_RUNNING = 0,
  PROXY_STOP = 1,
  PROXY_ABORT = 2
};

// 函数实现：NCCL 代理服务线程主函数
// 此函数是代理服务的核心线程，负责处理来自本地通信器的连接和操作请求
// 它管理所有代理连接，轮询事件，并调度异步操作的执行
// 参数 _args: 传入的参数，实际上是 ncclProxyState 结构体指针
// 返回值：void* 类型，线程函数返回 NULL
void* ncclProxyService(void* _args) {
  // 将传入的参数转换为 ncclProxyState 指针
  // 代理状态包含代理服务的所有运行时信息
  struct ncclProxyState* proxyState =  (struct ncclProxyState*) _args;

  // set the thread affinity before setting the cuda context
  // 在设置 CUDA 上下文之前设置线程亲和性（CPU 核心绑定）
  // 这确保代理线程运行在指定的 CPU 核心上，提高性能和可预测性
  pthread_once(&proxyCpusetOnce,proxyCpusetOnceFunc); // 确保只执行一次 CPU 集合初始化
  if (CPU_COUNT(&proxyCpuset))                           // 如果 CPU 集合非空
    sched_setaffinity(0, sizeof(cpu_set_t), &proxyCpuset); // 设置线程的 CPU 亲和性
  // 输出代理服务启动信息，显示设备和 CPU 核心绑定情况
  INFO(NCCL_INIT, "[Proxy Service] Device %d CPU core %d", proxyState->cudaDev, sched_getcpu());

  // 设置代理线程的 CUDA 上下文
  if (setProxyThreadContext(proxyState)) {               // 尝试设置代理线程的 CUDA 上下文
    // 成功创建了 CUDA 上下文
    INFO(NCCL_INIT, "[Proxy Service] Created CUDA context on device %d", proxyState->cudaDev);
  } else if (cudaSetDevice(proxyState->cudaDev) != cudaSuccess) { // 如果创建上下文失败，尝试设置设备
    // 设置 CUDA 设备也失败了
    WARN("[Proxy Service] Failed to set CUDA device %d", proxyState->cudaDev);
  }

  // Prepare poll descriptor
  // 准备轮询描述符，用于监控多个文件描述符的事件
  struct ncclProxyConnectionPool connectionPool;        // 连接池结构，管理所有代理连接
  connectionPool.pools = NULL;                          // 初始化连接池数组为空
  connectionPool.banks = 0;                             // 初始化连接池的银行数为 0
  connectionPool.offset = NCCL_PROXY_CONN_POOL_SIZE;    // 设置偏移量为连接池大小

  // 轮询文件描述符数组：NCCL_MAX_PROXY_CONNECTIONS 个对等连接 + 1 个监听套接字
  struct pollfd pollfds[NCCL_MAX_PROXY_CONNECTIONS+1]; // one extra for listenSock fd
  // 本地对等节点数组，存储所有已连接的对等节点信息
  struct ncclProxyLocalPeer peers[NCCL_MAX_PROXY_CONNECTIONS];
  // 初始化对等节点数组为全零
  memset(&peers, 0, sizeof(struct ncclProxyLocalPeer)*NCCL_MAX_PROXY_CONNECTIONS);
  // 初始化轮询文件描述符数组
  for (int s=0; s<NCCL_MAX_PROXY_CONNECTIONS; s++) {
    pollfds[s].fd = -1;                                 // 设置文件描述符为无效（-1 表示未使用）
    pollfds[s].events = POLLHUP|POLLIN;                  // 监控挂起和可读事件
  }
  // 获取监听套接字的文件描述符（用于接受新连接）
  if (ncclSocketGetFd(proxyState->listenSock, &pollfds[NCCL_MAX_PROXY_CONNECTIONS].fd) != ncclSuccess) {
    WARN("[Proxy Service] Get listenSock fd fails");    // 获取失败，输出警告
    return NULL;                                        // 返回 NULL，退出线程
  };
  // 设置监听套接字的监听事件（只监控可读事件，表示有新连接）
  pollfds[NCCL_MAX_PROXY_CONNECTIONS].events = POLLIN;

  int maxnpeers = 0;                                     // 最大对等节点索引（已使用的最大槽位数）
  int npeers = 0;                                        // 当前活跃的对等节点数量
  int stop = PROXY_RUNNING;                              // 停止标志：初始状态为运行中
  int asyncOpCount = 0;                                  // 当前活跃的异步操作数量
  // 主服务循环：持续处理连接和操作，直到收到停止信号且所有连接关闭
  while (stop == PROXY_RUNNING || npeers > 0) {
    /* Even if local comm aborts, we cannot let proxy thread exit if we still have peer
     * connections. Need to wait until all other related comms call abort and safely exit
     * together, or we could face segmentation fault. */
    /* 即使本地通信器中止，如果我们仍有对等连接，也不能让代理线程退出。
     * 需要等待所有其他相关通信器都调用中止并安全退出，否则可能遇到段错误。 */
    // 原子加载中止标志，检查是否收到中止请求
    if (__atomic_load_n(proxyState->abortFlag, __ATOMIC_ACQUIRE) != 0) stop = PROXY_ABORT;
    /* never let proxy service thread blocks in poll, or it cannot receive abortFlag. */
    /* 永远不要让代理服务线程阻塞在 poll 中，否则它无法接收 abortFlag。 */
    int ret;                                             // poll 函数的返回值
    do {
      // poll all fds including the listenSock
      // 轮询所有文件描述符，包括监听套接字
      // 如果有异步操作在执行，超时时间为 0（非阻塞）；否则超时时间为 500ms
      ret = poll(pollfds, NCCL_MAX_PROXY_CONNECTIONS+1, asyncOpCount ? 0 : 500);
    } while (ret < 0 && errno == EINTR);                // 如果被信号中断，重试
    if (ret < 0) {                                       // poll 出错
      WARN("[Proxy Service] Poll failed: %s", strerror(errno)); // 输出错误信息
      return NULL;                                      // 退出线程
    }
    // 处理监听套接字上的事件（表示有新连接请求）
    if (pollfds[NCCL_MAX_PROXY_CONNECTIONS].revents) {
      // We got an event on the listenSock
      // 监听套接字上有事件，表示有新的连接请求
      int s = 0;                                         // 槽位索引
      // 查找第一个可用的槽位（fd 为 -1 表示未使用）
      while (s < NCCL_MAX_PROXY_CONNECTIONS && pollfds[s].fd >= 0) s++;
      if (s == NCCL_MAX_PROXY_CONNECTIONS) {              // 没有可用槽位
        WARN("[Proxy service] Too many connections (%d max)", NCCL_MAX_PROXY_CONNECTIONS);
        return NULL;                                    // 退出线程
      }
      // 更新最大对等节点索引
      if (maxnpeers < s+1)
        maxnpeers = s+1;
      // 初始化新对等节点的套接字
      if (ncclSocketInit(&peers[s].sock) != ncclSuccess) {
        WARN("[Service thread] Initialize peers[%d].sock fails", s);
        return NULL;
      }
      // 接受新连接
      if (ncclSocketAccept(&peers[s].sock, proxyState->listenSock) != ncclSuccess) {
        WARN("[Service thread] Accept failed %s", strerror(errno)); // 接受失败
      } else {
        // 获取新连接的文件描述符
        if (ncclSocketGetFd(&peers[s].sock, &pollfds[s].fd) != ncclSuccess) {
          WARN("[Service thread] Get peers[%d].sock fd fails", s);
          return NULL;
        }
        npeers++;                                        // 增加对等节点计数
        peers[s].tpLocalRank = -1;                       // 初始化本地 rank 为 -1（未知）
      }
    }

    // 遍历所有对等节点槽位，处理每个对等节点的操作
    for (int s=0; s<maxnpeers; s++) {
      struct ncclProxyLocalPeer* peer = peers+s;         // 获取对等节点指针
      struct ncclSocket* sock = &peer->sock;             // 获取套接字指针
      int closeConn = 0;                                 // 关闭连接标志
      int type = 0;                                      // 操作类型
      ncclResult_t res = ncclSuccess;                    // 操作结果
      if (pollfds[s].fd == -1)
        continue;                 // 跳过无效槽位

      // Progress all ops for this ncclProxyLocalPeer
      // 推进此对等节点的所有异步操作
      // 特殊情况：如果正在中止且启用了 cuMem 且不是直接模式，关闭连接
      if (stop == PROXY_ABORT && ncclCuMemEnable() && ncclCuMemHostEnable() && !proxyState->directMode && __atomic_load_n(&proxyState->stop, __ATOMIC_ACQUIRE)) 
        closeConn = 1;
      
      // 获取此对等节点的异步操作队列头
      ncclProxyAsyncOp* op = peer->asyncOps;
      // 遍历所有异步操作，推进它们的执行
      while (op != nullptr) {
        ncclProxyAsyncOp* opnext = op->next; /* in case op is freed in proxyProgressAsync */ /* 保存下一个操作指针，因为 op 可能在 proxyProgressAsync 中被释放 */
        type = op->type;                                 // 获取操作类型
        // Coverity gets confused here by complex code structure.  Yes, connectionPool.pools gets dereferenced, and
        // while calling proxyProgressAsync() connectionPool.pools is NULL, but that changes before it's dereferenced.
        // coverity[var_deref_model:FALSE]
        // 推进异步操作的执行
        res = proxyProgressAsync(op, proxyState, &asyncOpCount, peer, &connectionPool);
        if (res == ncclSuccess || res == ncclInProgress) { // 操作成功或进行中
          op = opnext;                                   // 移动到下一个操作
        } else {                                          // 操作失败
          // Res is a bad result
          // 结果是错误代码
          closeConn = 1;                                 // 标记需要关闭连接
          WARN("[Service thread] Error encountered progressing operation=%s, res=%d, closing connection", ncclProxyMsgTypeStr[type], res);
          break;                                        // 退出循环
        }
      }

      // Check for additional ops coming in
      // 检查是否有新操作从套接字传入
      if (pollfds[s].revents & POLLIN) {                 // 如果有可读事件
        int closed;                                      // 连接关闭标志
        // 尝试从套接字接收操作类型（非阻塞模式）
        res = ncclSocketTryRecv(sock, &type, sizeof(int), &closed, false /*blocking*/);
        if (res != ncclSuccess && res != ncclInProgress) { // 接收失败
          // 如果不是在中止状态，输出警告
          if (!__atomic_load_n(proxyState->abortFlag, __ATOMIC_RELAXED))
            WARN("[Service thread] Could not receive type from localRank %d, res=%u, closed=%d", peer->tpLocalRank, res, closed);
          closeConn = 1;                                 // 标记需要关闭连接
        } else if (closed) {                             // 连接已关闭
          INFO(NCCL_INIT|NCCL_NET|NCCL_PROXY, "[Service thread] Connection closed by localRank %d", peer->tpLocalRank);
          closeConn = 1;                                 // 标记需要关闭连接
        } else if (res == ncclSuccess) { // We received something from the sock
          // 成功接收到数据
          if (type == ncclProxyMsgStop) {                // 收到停止消息
            stop = PROXY_STOP;                           // 设置停止状态
            closeConn = 1;                               // 关闭连接
          } else if (type == ncclProxyMsgClose) {         // 收到关闭消息
            closeConn = 1;                               // 关闭连接
          } else if (proxyMatchOpType(type)) {            // 如果是有效的操作类型
            // 初始化并开始执行操作
            res = proxyServiceInitOp(type, peers+s, &connectionPool, proxyState, &asyncOpCount);
          } else {                                        // 未知的命令类型
            WARN("[Service thread] Unknown command %d from localRank %d", type, peer->tpLocalRank);
            closeConn = 1;                               // 关闭连接
          }

          // 输出操作接收和启动的信息
          INFO(NCCL_PROXY, "Received and initiated operation=%s res=%d", ncclProxyMsgTypeStr[type], res);
        }
      } else if (pollfds[s].revents & POLLHUP) {         // 如果有挂起事件（对端关闭连接）
        closeConn = 1;                                   // 关闭连接
      }
      // 检查操作结果，如果失败则关闭连接
      if (res != ncclSuccess && res != ncclInProgress) {
        if (!__atomic_load_n(proxyState->abortFlag, __ATOMIC_RELAXED)) // 如果不是在中止状态
          WARN("[Proxy Service %d] Failed to execute operation %s from rank %d, retcode %d", proxyState->tpRank, ncclProxyMsgTypeStr[type], peer->tpRank, res);
        closeConn = 1;                                   // 标记需要关闭连接
      }

      // 如果需要关闭连接，执行清理工作
      if (closeConn) {
        (void)ncclSocketClose(sock);                     // 关闭套接字

        // 如果有正在执行的操作，从队列中移除
        if (op != nullptr) {
          asyncProxyOpDequeue(peer, op);                 // 从异步操作队列中移除
          asyncOpCount--;                                // 减少异步操作计数
        }
        pollfds[s].fd = -1;                             // 标记槽位为无效
        npeers--;                                        // 减少对等节点计数
      }
    }
  }

  // Wait for all operations to complete and stop progress thread before freeing any resource
  // 等待所有操作完成并停止进度线程，然后释放任何资源
  if (ncclProxyProgressDestroy(proxyState) != ncclSuccess) { // 销毁代理进度线程
    WARN("[Proxy Service] proxyDestroy failed");        // 输出警告
  }
  // 关闭所有对等节点的套接字
  for (int s=0; s<maxnpeers; s++) {
    (void)ncclSocketClose(&peers[s].sock);              // 关闭套接字
  }
  // 释放连接池资源
  ncclProxyFreeConnections(&connectionPool, proxyState);
  // 关闭监听套接字
  (void)ncclSocketClose(proxyState->listenSock);
  free(proxyState->listenSock);                         // 释放监听套接字内存
  // 释放代理操作资源
  proxyOpsFree(proxyState);
  return NULL;                                          // 线程退出
}


// Process a request on the UDS socket
// 在 UDS（Unix Domain Socket）套接字上处理请求
// 此函数处理来自 UDS 套接字的请求，主要用于支持 cuMem API 的文件描述符操作
// 参数 proxyState: 代理状态指针，包含代理服务的所有运行时信息
// 参数 reqFd: 请求文件描述符（当前未使用，保留用于未来扩展）
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
static ncclResult_t proxyUDSRecvReq(struct ncclProxyState* proxyState, int reqFd) {
  ncclIpcHdr hdr;                                        // IPC 消息头结构，存储消息类型和数据
  int rmtFd = -1;                                        // 远程文件描述符，初始化为 -1（无效）

  // 从 UDS 套接字接收消息和可能的文件描述符
  // 接收消息头到 hdr，如果消息中带有文件描述符，则接收到 rmtFd
  NCCLCHECK(ncclIpcSocketRecvMsg(&proxyState->ipcSock, &hdr, sizeof(hdr), &rmtFd));

  // 根据消息类型处理不同的请求
  if (hdr.type == ncclProxyMsgGetFd) {                   // 如果是获取文件描述符请求
    // cuMem API support for non-UB case, and rmtFd is not used since UDS proxy thread need to export
    // fd from handle and send it back to the main thread to import the buffer. We just need to close
    // this dummy rmtFd.
    /* cuMem API 支持非 UB（Unified Buffer）情况，rmtFd 未被使用，
     * 因为 UDS 代理线程需要从句柄导出 fd 并将其发送回主线程以导入缓冲区。
     * 我们只需要关闭这个虚拟的 rmtFd。 */
    uint64_t handle = *(uint64_t*)hdr.data;             // 从消息数据中提取句柄（8 字节）
    // 输出获取文件描述符请求的调试信息
    INFO(NCCL_PROXY, "proxyUDSRecvReq::ncclProxyMsgGetFd rank %d opId %p handle=0x%lx", hdr.rank, hdr.opId, handle);
    close(rmtFd);                                        // 关闭虚拟的远程文件描述符
    // 调用 proxyGetFd 处理获取文件描述符请求
    return proxyGetFd(proxyState, hdr.rank, hdr.opId, handle);
  } else if (hdr.type == ncclProxyMsgQueryFd) {         // 如果是查询文件描述符请求
    // remote main thread registers buffer into this rank, it querys rmtFd of this rank through UDS
    // and the rmtFd is returned unchanged back to remote main thread which will use rmtFd to call into
    // proxy service thread for buffer registration.
    /* 远程主线程向此 rank 注册缓冲区，它通过 UDS 查询此 rank 的 rmtFd，
     * rmtFd 原样返回给远程主线程，后者将使用 rmtFd 调用代理服务线程进行缓冲区注册。 */
    // 输出查询文件描述符请求的调试信息
    INFO(NCCL_PROXY, "proxyUDSRecvReq::proxyQueryFd rank %d opId %p rmtFd %d", hdr.rank, hdr.opId, rmtFd);
    // 调用 proxyQueryFd 处理查询文件描述符请求
    return proxyQueryFd(proxyState, hdr.rank, hdr.opId, rmtFd);
  }

  // 如果消息类型未知，返回内部错误
  return ncclInternalError;
}

// UDS fd handle support
// UDS（Unix Domain Socket）文件描述符支持
// 此函数是 UDS 代理服务线程的主函数，专门处理与 cuMem API 相关的文件描述符操作
// 它在独立的线程中运行，与主代理服务线程并行工作
// 参数 _args: 传入的参数，实际上是 ncclProxyState 结构体指针
// 返回值：void* 类型，线程函数返回 NULL
void* ncclProxyServiceUDS(void* _args) {
  // 将传入的参数转换为 ncclProxyState 指针
  // 代理状态包含 UDS 代理服务的所有运行时信息
  struct ncclProxyState* proxyState =  (struct ncclProxyState*) _args;
  // 轮询文件描述符数组，只有一个元素（UDS 套接字）
  struct pollfd pollfds[1];

  // set the thread affinity before setting the cuda context
  // 在设置 CUDA 上下文之前设置线程亲和性（CPU 核心绑定）
  // 这确保 UDS 代理线程运行在指定的 CPU 核心上，提高性能和可预测性
  pthread_once(&proxyCpusetOnce,proxyCpusetOnceFunc); // 确保只执行一次 CPU 集合初始化
  if (CPU_COUNT(&proxyCpuset))                           // 如果 CPU 集合非空
    sched_setaffinity(0, sizeof(cpu_set_t), &proxyCpuset); // 设置线程的 CPU 亲和性
  // 输出 UDS 代理服务启动信息，显示设备和 CPU 核心绑定情况
  INFO(NCCL_INIT, "[Proxy Service UDS] Device %d CPU core %d", proxyState->cudaDev, sched_getcpu());

  // 设置 UDS 代理线程的 CUDA 上下文
  if (setProxyThreadContext(proxyState)) {               // 尝试设置代理线程的 CUDA 上下文
    // 成功设置了 CUDA 上下文
    INFO(NCCL_INIT, "[Proxy Service UDS] Set CUDA context on device %d", proxyState->cudaDev);
  } else if (cudaSetDevice(proxyState->cudaDev) != cudaSuccess) { // 如果设置上下文失败，尝试设置设备
    // 设置 CUDA 设备失败
    WARN("[Proxy Service UDS] Failed to set CUDA device %d", proxyState->cudaDev);
  }

  // 获取 UDS 套接字的文件描述符并设置轮询事件
  if (ncclIpcSocketGetFd(&proxyState->ipcSock, &pollfds[0].fd) != ncclSuccess) {
    WARN("[Proxy Service UDS] Get listenSock fd fails"); // 获取失败，输出警告
    return NULL;                                        // 返回 NULL，退出线程
  };
  // 设置轮询事件：监控可读事件（POLLIN）和挂起事件（POLLHUP）
  pollfds[0].events = POLLIN|POLLHUP;

  // 主服务循环：持续处理 UDS 请求，直到收到停止或中止信号
  while (1) {
    /* never let proxy service thread blocks in poll, or it cannot receive abortFlag. */
    /* 永远不要让代理服务线程阻塞在 poll 中，否则它无法接收 abortFlag。 */
    int ret;                                             // poll 函数的返回值
    do {
      // 轮询 UDS 套接字，超时时间为 500ms
      // 使用超时确保线程可以定期检查 stop 和 abortFlag
      ret = poll(pollfds, 1, 500);
    } while (ret < 0 && errno == EINTR);                // 如果被信号中断，重试
    if (ret < 0) {                                       // poll 出错
      WARN("[Proxy Service UDS] Poll failed: %s", strerror(errno)); // 输出错误信息
      return NULL;                                      // 退出线程
    }

    // Check for stop/abort
    // 检查停止或中止标志
    // 原子加载 stop 和 abortFlag，如果任一标志被设置，退出循环
    if (__atomic_load_n(&proxyState->stop, __ATOMIC_ACQUIRE) || __atomic_load_n(proxyState->abortFlag, __ATOMIC_ACQUIRE)) break;

    // 处理 UDS 套接字上的事件
    if (pollfds[0].revents) {                           // 如果有事件（可读或挂起）
      // A request was seen on the UDS fd
      // 在 UDS 文件描述符上检测到请求
      // 调用 proxyUDSRecvReq 处理 UDS 请求（获取或查询文件描述符）
      proxyUDSRecvReq(proxyState, pollfds[0].fd);
    }
  }

  // 清理资源：关闭 UDS 套接字
  (void)ncclIpcSocketClose(&proxyState->ipcSock);
  // 输出 UDS 代理服务退出的调试信息，显示停止状态和中止标志
  INFO(NCCL_PROXY, "[Proxy Service UDS] exit: stop %d abortFlag %d", proxyState->stop, *proxyState->abortFlag);
  return NULL;                                          // 线程退出
}

// 函数实现：初始化代理服务
// 此函数初始化 NCCL 代理服务的状态结构，为后续创建代理线程做准备
// 它设置代理状态的基本参数，包括套接字、地址和 UDS 支持
// 参数 comm: NCCL 通信器指针
// 参数 sock: 监听套接字指针，用于接受来自本地进程的连接
// 参数 peerAddresses: 对等节点地址数组指针，包含所有对等节点的套接字地址
// 参数 peerAddressesUDS: UDS 对等节点地址数组指针，包含所有对等节点的 UDS 地址
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
ncclResult_t ncclProxyInit(struct ncclComm* comm, struct ncclSocket* sock, union ncclSocketAddress* peerAddresses, uint64_t *peerAddressesUDS) {
  // 断言检查：确保共享资源中的代理状态尚未初始化
  // 这防止重复初始化代理状态
  assert(comm->sharedRes->proxyState == NULL);
  //给proxyState分配内存
  // 为代理状态结构分配内存并初始化为零
  NCCLCHECK(ncclCalloc(&comm->sharedRes->proxyState, 1));
  //指向comm->sharedRes
  // 设置通信器的代理状态指针指向共享资源中的代理状态
  // 这样父通信器和分裂通信器可以共享同一个代理服务
  comm->proxyState = comm->sharedRes->proxyState;
  comm->proxyState->refCount = 1;                        // 初始化引用计数为 1
  //记录proxy 监听套接字
  // 保存监听套接字指针，代理服务线程将使用它接受新连接
  comm->proxyState->listenSock = sock;
  // 保存对等节点地址数组，用于与对等节点建立连接
  comm->proxyState->peerAddresses = peerAddresses;
  // 保存 UDS 对等节点地址数组，用于 UDS 通信
  comm->proxyState->peerAddressesUDS = peerAddressesUDS;
  // 初始化网络属性为初始值
  comm->proxyState->netAttr = NCCL_NET_ATTR_INIT;

  // UDS support
  // UDS（Unix Domain Socket）支持
  //创建一个unix域套接字
  // 初始化 UDS 套接字，用于处理 cuMem API 相关的文件描述符操作
  // 参数：ipcSock-UDS套接字；rank-当前rank；peerAddressesUDS[comm->rank]-当前rank的UDS地址；abortFlag-中止标志
  NCCLCHECK(ncclIpcSocketInit(&comm->proxyState->ipcSock, comm->rank, peerAddressesUDS[comm->rank], comm->abortFlag));
  return ncclSuccess;                                   // 返回成功状态
}

//创建proxy线程。一个gpu一个线程
// 函数实现：创建代理服务线程
// 此函数创建并启动 NCCL 代理服务线程，每个 GPU 设备有一个代理线程
// 代理服务线程在后台处理通信操作，减轻主线程的负担
// 参数 comm: NCCL 通信器指针
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
ncclResult_t ncclProxyCreate(struct ncclComm* comm) {
  /* proxyState is shared among parent comm and split comms. comm->proxyState->thread is
   * pthread_join()'d by commFree() in init.cc when the refCount reduces down to 0. */
  /* 代理状态在父通信器和分裂通信器之间共享。
   * 当引用计数减少到 0 时，comm->proxyState->thread 将在 init.cc 的 commFree() 中被 pthread_join() 等待。 */
   //ncclProxyInit中设置 comm->proxyState
   // 获取代理状态指针（在 ncclProxyInit 中已初始化）
  struct ncclProxyState* proxyState = comm->proxyState;
  // 检查引用计数是否为 1（表示是第一次创建，而不是分裂通信器的重用）
  if (proxyState->refCount == 1) {
    /* we have to make sure all following fields in comm have been initialized. */
    /* 我们必须确保 comm 中的所有以下字段都已经被初始化。
     * 这是必要的，因为代理状态将被多个通信器共享，需要在首次创建时设置所有参数。 */

    // === 通信域信息 ===
    proxyState->tpRank = comm->rank;                   // 保存当前 rank 在拓扑中的位置
    proxyState->tpnRanks = comm->nRanks;                // 保存通信域中的 rank 总数
    proxyState->tpLocalnRanks = comm->localRanks;        // 保存本地节点中的 rank 数量

    // === GPU 设备信息 ===
    proxyState->cudaDev = comm->cudaDev;                // 保存 CUDA 设备号

    // === 中止标志 ===
    proxyState->abortFlag = comm->abortFlag;            // 保存中止标志指针（用于优雅关闭）

    // === P2P 配置 ===
    proxyState->p2pnChannels = comm->p2pnChannels;        // P2P 通道数量
    proxyState->p2pChunkSize = comm->p2pChunkSize;        // P2P 块大小

    // === 通道配置 ===
    proxyState->nChannels = comm->nChannels;              // 通道总数

    // === 内存分配配置 ===
    proxyState->allocP2pNetLLBuffers = comm->allocP2pNetLLBuffers; // P2P 网络缓冲区分配函数
    proxyState->dmaBufSupport = comm->dmaBufSupport;      // DMA 缓冲区支持标志

    // === 网络插件接口 ===
    //记录ncclNet连接方法
    proxyState->ncclNet = comm->ncclNet;                  // NCCL 网络插件接口（用于 IB/RoCE 等网络传输）
    proxyState->ncclCollNet = comm->ncclCollNet;          // CollNet 网络插件接口

    // === 网络上下文 ===
    proxyState->netContext = comm->netContext;            // 网络上下文
    proxyState->collNetContext = comm->collNetContext;    // CollNet 上下文

    // === 性能分析器上下文 ===
    proxyState->profilerContext = comm->profilerContext;  // 性能分析器上下文

    // === 直接模式标志 ===
    proxyState->directMode = comm->directMode;            // 直接模式标志（是否使用直接 GPU 访问）

    // === 缓冲区大小配置 ===
    // 复制各协议的缓冲区大小配置
    memcpy(proxyState->buffSizes, comm->buffSizes, sizeof(comm->buffSizes));

    // === 创建主代理服务线程 ===
    // pthread_create 参数：线程指针、属性（NULL=默认）、线程函数、线程参数
    // 主代理服务线程处理所有 Send/Recv/Connect 等操作
    PTHREADCHECK(pthread_create(&comm->proxyState->thread, NULL, ncclProxyService, comm->proxyState), "pthread_create");
    // 设置线程名称，便于调试和性能分析
    // 格式：NCCL Service <设备号>（例如：NCCL Service 0）
    ncclSetThreadName(comm->proxyState->thread, "NCCL Service %2d", comm->cudaDev);

    // UDS support
    // UDS（Unix Domain Socket）支持
    // 输出 UDS 服务线程创建的调试信息
    INFO(NCCL_PROXY, "UDS: Creating service thread comm %p rank %d", comm, comm->rank);
    // === 创建 UDS 代理服务线程 ===
    // UDS 代理服务线程专门处理 cuMem API 相关的文件描述符操作
    PTHREADCHECK(pthread_create(&comm->proxyState->threadUDS, NULL, ncclProxyServiceUDS, comm->proxyState), "pthread_create");
    // 设置 UDS 线程名称，便于调试和性能分析
    // 格式：NCCL UDS Service <设备号>（例如：NCCL UDS Service 0）
    ncclSetThreadName(comm->proxyState->threadUDS, "NCCL UDS Service %2d", comm->cudaDev);
  }
  return ncclSuccess;                                   // 返回成功状态
}

// 函数实现：停止代理服务
// 此函数是代理服务停止的第一步，它减少引用计数并在引用计数降为 0 时
// 通知代理线程停止。实际的资源释放在 ncclProxyDestroy 中进行
// 参数 comm: NCCL 通信器指针
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
ncclResult_t ncclProxyStop(struct ncclComm* comm) {
  // 检查代理状态是否存在
  if (comm->proxyState) {
    // 获取共享代理状态指针
    // 代理状态在共享资源中，可能被多个通信器共享（父通信器和分裂通信器）
    struct ncclProxyState* sharedProxyState = comm->proxyState;

    // 原子地减少引用计数，并检查是否减少到 0
    // 返回值是减少前的引用计数
    if ((comm->proxyRefCountOld = ncclAtomicRefCountDecrement(&sharedProxyState->refCount)) == 0) {
      // 引用计数降为 0，表示这是最后一个使用代理服务的通信器
      // 需要通知代理线程停止

      // === 发送停止消息给代理服务（优雅关闭） ===
      // 只有在没有中止且对等节点地址存在时才发送停止消息
      if (*comm->abortFlag == 0 && sharedProxyState->peerAddresses) {
        // We need to send a ncclProxyMsgStop message to our own proxy
        /* 我们需要向自己的代理发送 ncclProxyMsgStop 消息
         * 这样可以让代理线程优雅地关闭，而不是强制终止
         * 发送停止消息是代理服务优雅关闭的关键步骤 */
        struct ncclSocket sock;                          // 临时套接字，用于发送停止消息
        int type = ncclProxyMsgStop;                        // 消息类型：停止
        // 初始化套接字，连接到自己的代理服务
        // 参数：sock-套接字；peerAddresses-对等节点地址（自己的地址）；magic-魔术字；socketType-套接字类型；abortFlag-中止标志
        NCCLCHECK(ncclSocketInit(&sock, sharedProxyState->peerAddresses + comm->topParentRanks[comm->rank], comm->sharedRes->magic, ncclSocketTypeProxy, comm->abortFlag));
        // 尝试连接到代理服务
        if (ncclSocketConnect(&sock) == ncclSuccess) {
          // 连接成功，发送停止消息
          (void)ncclSocketSend(&sock, &type, sizeof(int)); // 发送停止消息类型
        }
        // 关闭临时套接字（无论连接是否成功）
        (void)ncclSocketClose(&sock);
      }

      // === 关闭所有本地对等节点的连接 ===
      if (sharedProxyState->peerSocks) {                 // 如果对等节点套接字数组存在
        int tplocalRanks = comm->sharedRes->tpNLocalRanks; // 本地对等节点数量
        // 遍历所有本地对等节点，逐个关闭连接
        for (int i = 0; i < tplocalRanks; i++) {
          int fd;                                            // 文件描述符
          // 获取套接字的文件描述符
          NCCLCHECK(ncclSocketGetFd(sharedProxyState->peerSocks + i, &fd));
          if (fd >= 0) {                                    // 如果文件描述符有效（套接字已连接）
            // === 关闭共享内存池 ===
            if (sharedProxyState->proxyOps[i].pool) {     // 如果有共享内存池
              // 关闭共享内存句柄（用于进程间通信）
              NCCLCHECK(ncclShmClose(sharedProxyState->proxyOps[i].handle));
            }
            // === 关闭 CUDA IPC 内存句柄 ===
            if (sharedProxyState->sharedDevMems[i]) {     // 如果有 CUDA IPC 内存句柄
              // 如果未启用 cuMem API，关闭 IPC 内存句柄
              // cuMem API 有自己的内存管理机制，与 IPC 内存句柄不同
              if (!ncclCuMemEnable()) {
                CUDACHECK(cudaIpcCloseMemHandle(sharedProxyState->sharedDevMems[i]));
              }
            }
            // === 发送关闭消息并关闭套接字 ===
            int type = ncclProxyMsgClose;                   // 消息类型：关闭
            // 向对等节点发送关闭消息
            (void)ncclSocketSend(sharedProxyState->peerSocks + i, &type, sizeof(int));
            // 关闭对等节点套接字
            NCCLCHECK(ncclSocketClose(sharedProxyState->peerSocks + i));
          }
        }
      }
      // Now we notify proxy service and UDS thread to exit.
      /* 现在我们通知代理服务和 UDS 线程退出。
       * 设置 stop 标志为 1，让代理服务线程和 UDS 线程能够检测到停止信号
       * 这使用原子存储确保线程安全 */
      __atomic_store_n(&comm->proxyState->stop, 1, __ATOMIC_RELEASE);
      // 释放内存语义：确保所有之前的写操作对代理线程可见
    }
    // 如果引用计数 > 0，说明还有其他通信器在使用代理服务，不做任何操作
  }
  // 如果代理状态不存在，直接返回成功

  return ncclSuccess;                                   // 返回成功状态
}

// 函数：销毁代理
// 此函数释放代理相关的所有资源
// 参数 comm: NCCL 通信器指针
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
ncclResult_t ncclProxyDestroy(struct ncclComm* comm) {
  // 获取共享代理状态
  struct ncclProxyState* sharedProxyState = comm->sharedRes->proxyState;

  if (sharedProxyState) {
    // 断言引用计数必须为 0
    assert(sharedProxyState->refCount == 0);
    // 释放对等节点地址数组
    free(sharedProxyState->peerAddresses);
    // 释放 UDS 对等节点地址数组
    free(sharedProxyState->peerAddressesUDS);
    // 释放对等节点套接字数组
    free(sharedProxyState->peerSocks);
    // 释放代理操作数组
    free(sharedProxyState->proxyOps);
    // 释放共享设备内存数组
    free(sharedProxyState->sharedDevMems);
    // 释放预期响应队列
    expectedProxyResponseFree(sharedProxyState);
    // 释放代理状态结构体
    free(sharedProxyState);
  }
  return ncclSuccess;
}
