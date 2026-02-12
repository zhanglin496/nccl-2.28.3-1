/*************************************************************************
 * Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
 * 版权所有 (c) 2024，NVIDIA 公司。保留所有权利。
 *
 * See LICENSE.txt for license information
 * 详见 LICENSE.txt 获取许可证信息
 ************************************************************************/
// 包含传输层相关头文件，定义传输层接口和数据结构
#include "transport.h"
// 包含代理相关头文件，用于代理操作和通信
#include "proxy.h"
// 包含性能分析器相关头文件，定义性能分析接口和数据结构
#include "profiler.h"
// 包含设备端相关头文件，定义设备端的数据结构和操作
#include "device.h"

// Profiler 代理连接函数，用于初始化代理连接
// 参数说明：
//   connection: 代理连接指针
//   proxyState: 代理状态指针
//   reqBuff: 请求缓冲区
//   reqSize: 请求大小
//   respBuff: 响应缓冲区
//   respSize: 响应大小
//   done: 完成标志
static ncclResult_t profilerProxyConnect(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState, void* reqBuff, int reqSize, void* respBuff, int respSize, int* done) {
  connection->proxyAppendPtr = &connection->proxyAppend;  // 设置代理追加指针，指向连接的代理追加函数
  connection->shared = 0;  // 设置共享标志为 0（表示不共享）
  return ncclSuccess;  // 返回成功状态
}

// The following ncclProxySubArgs are overloaded by the profiler progress function:
// 以下 ncclProxySubArgs 被 profiler 进度函数重载：
// - base       : is set to the current value of workCounter[channelId]
// - base       : 被设置为 workCounter[channelId] 的当前值
// - posted     : is set to sub->nsteps to indicate that profiler has started event
// - posted     : 被设置为 sub->nsteps，表示 profiler 已启动事件
// - transmitted: is set to sub->nsteps to indicate that profiler has stopped event
// - transmitted: 被设置为 sub->nsteps，表示 profiler 已停止事件

// Profiler 代理进度函数，用于推进 profiler 事件的处理
// 参数说明：
//   proxyState: 代理状态指针
//   args: 代理参数指针，包含子操作信息
static ncclResult_t profilerProxyProgress(struct ncclProxyState* proxyState, struct ncclProxyArgs* args) {
  if (args->state == ncclProxyOpReady) {  // 如果代理状态为 Ready（准备就绪）
    for (int s = 0; s < args->nsubs; s++) {  // 遍历所有子操作
      struct ncclProxySubArgs* sub = args->subs + s;  // 获取当前子操作指针
      sub->base = sub->workCounter;  // 设置 base 为当前工作计数器的值，用于跟踪事件的起始点
      sub->posted = sub->transmitted = 0;  // 初始化 posted 和 transmitted 标志为 0，表示事件尚未启动或完成
    }
    args->state = ncclProxyOpProgress;  // 将代理状态转换为 Progress（进行中）
  }
  if (args->state == ncclProxyOpProgress) {  // 如果代理状态为 Progress（进行中）
    for (int s = 0; s < args->nsubs; s++) {  // 遍历所有子操作
      struct ncclProxySubArgs* sub = args->subs + s;  // 获取当前子操作指针
      struct ncclDevProfiler* workStarted = (struct ncclDevProfiler *)sub->sendbuff;  // 从发送缓冲区获取工作已启动的 profiler 数据结构指针
      struct ncclDevProfiler* workCompleted = (struct ncclDevProfiler *)sub->recvbuff;  // 从接收缓冲区获取工作已完成的 profiler 数据结构指针
      // 检查是否需要启动事件：posted 小于 nsteps 且 base 小于等于工作启动计数器
      if (sub->posted < sub->nsteps && sub->base <= workStarted[sub->channelId].data[sub->base%MAX_PROFILER_EVENTS_PER_CHANNEL].counter) {  // 如果事件未启动且工作计数器达到起始点
        ncclProfilerStartKernelChEvent(args, s, workStarted[sub->channelId].data[sub->base%MAX_PROFILER_EVENTS_PER_CHANNEL].timestamp);  // 启动 kernel 通道事件，记录时间戳
        sub->posted = sub->nsteps;  // 设置 posted 标志为 nsteps，表示事件已启动
        continue; // allow events on every channel to start
        // 继续，允许每个通道上的事件都启动
      }
      // 检查是否需要停止事件：transmitted 小于 nsteps 且 base 小于等于工作完成计数器
      if (sub->transmitted < sub->nsteps && sub->base <= workCompleted[sub->channelId].data[sub->base%MAX_PROFILER_EVENTS_PER_CHANNEL].counter) {  // 如果事件未完成且工作计数器达到完成点
        ncclProfilerStopKernelChEvent(args, s, workCompleted[sub->channelId].data[sub->base%MAX_PROFILER_EVENTS_PER_CHANNEL].timestamp);  // 停止 kernel 通道事件，记录时间戳
        sub->transmitted = sub->nsteps;  // 设置 transmitted 标志为 nsteps，表示事件已完成
        args->done++;  // 增加完成计数
      }
    }
    if (args->done == args->nsubs)  // 如果所有子操作都已完成
        args->state = ncclProxyOpNone;  // 将代理状态设置为 None（无操作）
  }
  return ncclSuccess;  // 返回成功状态
}

// Profiler 传输层结构体，实现了 NCCL 传输层接口
struct ncclTransport profilerTransport = {
  "Prof",  // 传输层名称：Prof（Profiler 的缩写）
  NULL,  // 连接判断函数指针为 NULL（profiler 不需要连接判断）
  { NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL },  // 发送端操作函数指针表，全部为 NULL
  { NULL, NULL, NULL, NULL, NULL, profilerProxyConnect, NULL, profilerProxyProgress, NULL, NULL }  // 接收端操作函数指针表，只设置了代理连接和进度函数
};
