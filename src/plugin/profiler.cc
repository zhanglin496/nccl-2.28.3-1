/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION. All rights reserved.
 * 版权所有 (c) 2022-2024，NVIDIA 公司。保留所有权利。
 *
 * See LICENSE.txt for license information
 * 详见 LICENSE.txt 获取许可证信息
 ************************************************************************/

// 包含参数相关头文件，定义 NCCL 参数和环境变量处理
#include "param.h"
// 包含检查相关头文件，提供各种检查和验证功能
#include "checks.h"
// 包含通信上下文头文件，定义 NCCL 通信器结构
#include "comm.h"
// 包含操作排队相关头文件，用于操作入队管理
#include "enqueue.h"
// 包含工具函数头文件，提供通用工具函数
#include "utils.h"
// 包含代理相关头文件，用于代理操作和通信
#include "proxy.h"
// 包含性能分析器相关头文件，定义性能分析接口和数据结构
#include "profiler.h"
// 包含传输层相关头文件，定义传输层接口
#include "transport.h"
// 包含插件相关头文件，定义插件接口和加载机制
#include "plugin.h"
// 包含互斥锁头文件，提供线程同步机制
#include <mutex>

// 声明外部函数：获取版本 1 的 NCCL Profiler 接口
extern ncclProfiler_t* getNcclProfiler_v1(void* lib);
// 声明外部函数：获取版本 2 的 NCCL Profiler 接口
extern ncclProfiler_t* getNcclProfiler_v2(void* lib);
// 声明外部函数：获取版本 3 的 NCCL Profiler 接口
extern ncclProfiler_t* getNcclProfiler_v3(void* lib);
// 声明外部函数：获取版本 4 的 NCCL Profiler 接口
extern ncclProfiler_t* getNcclProfiler_v4(void* lib);
// 声明外部函数：获取版本 5 的 NCCL Profiler 接口
extern ncclProfiler_t* getNcclProfiler_v5(void* lib);

// 定义性能分析器互斥锁，用于线程安全地访问性能分析器数据
static std::mutex profilerMutex;
// 定义性能分析器插件引用计数，用于跟踪插件使用情况
static int profilerPluginRefCount;
// 定义性能分析器插件库句柄，用于动态链接库管理
static void* profilerPluginLib;
// 定义 NCCL Profiler 接口指针，指向加载的性能分析器
static ncclProfiler_t* ncclProfiler;

// 声明外部线程变量：组深度（用于嵌套集合操作）
extern __thread int ncclGroupDepth;
// 声明线程局部变量：NCCL Profiler API 状态
__thread ncclProfilerApiState_t ncclProfilerApiState;

// 定义最大字符串长度为 256
#define MAX_STR_LEN 256

// 定义性能分析器插件状态枚举
enum {
  profilerPluginLoadFailed = -1,  // 插件加载失败状态
  profilerPluginLoadReady = 0,     // 插件加载完成，准备初始化状态
  profilerPluginLoadSuccess = 1,    // 插件加载成功状态
};
// 定义当前插件状态，初始化为 LoadReady
static int profilerPluginStatus = profilerPluginLoadReady;
// 定义进程 ID，用于标识加载性能分析器的进程
static pid_t pid;

// NCCL 性能分析器插件加载函数
static ncclResult_t ncclProfilerPluginLoad(void) {
  const char* profilerName;  // 性能分析器名称指针
  if (profilerPluginLoadFailed == profilerPluginStatus) {  // 如果插件状态为加载失败
    return ncclSuccess;  // 直接返回成功
  }

  // 使用 RAII 方式加锁，保护插件加载过程
  std::lock_guard<std::mutex> lock(profilerMutex);  // 自动加锁，函数退出时解锁
  if (profilerPluginLoadSuccess == profilerPluginStatus) {  // 如果插件状态为加载成功
    ++profilerPluginRefCount;  // 增加插件引用计数
    goto exit;  // 跳转到退出标签
  }

  // 获取 NCCL_PROFILER_PLUGIN 环境变量
  // 如果设置了性能分析器插件环境变量

  if ((profilerName = ncclGetEnv("NCCL_PROFILER_PLUGIN")) != nullptr) {  // 获取环境变量并检查是否有效
    INFO(NCCL_ENV, "NCCL_PROFILER_PLUGIN set by environment to %s", profilerName);  // 记录日志：环境变量设置了性能分析器插件
    if (strcasecmp(profilerName, "none") == 0)  // 如果环境变量为 "none"（表示不使用任何插件）
      goto fail;  // 跳转到失败处理
    }

    // 尝试打开指定的性能分析器插件库
    profilerPluginLib = ncclOpenProfilerPluginLib(profilerName);  // 打开性能分析器插件动态链接库
    if (profilerPluginLib == nullptr) {  // 如果插件库句柄为空（加载失败）
      profilerPluginLib = ncclGetNetPluginLib(ncclPluginTypeProfiler);  // 尝试获取默认网络插件库作为性能分析器
      if (nullptr == profilerPluginLib) {  // 如果默认插件库也获取失败
        goto fail;  // 跳转到失败处理
      }
      profilerName = nullptr;  // 清空性能分析器名称
    } else if (ncclPluginLibPaths[ncclPluginTypeProfiler]) {  // 如果有预定义的性能分析器插件路径
      profilerName = ncclPluginLibPaths[ncclPluginTypeProfiler];  // 使用预定义的插件路径
    }

  // 按版本从高到低尝试加载性能分析器接口
  // v5 到 v1 依次尝试加载

  ncclProfiler = getNcclProfiler_v5(profilerPluginLib);  // 尝试加载版本 5 的接口
    if (ncclProfiler == nullptr) {  // 如果版本 5 接口不存在
      ncclProfiler = getNcclProfiler_v4(profilerPluginLib);  // 尝试加载版本 4 的接口
      if (ncclProfiler == nullptr) {  // 如果版本 4 接口不存在
        ncclProfiler = getNcclProfiler_v3(profilerPluginLib);  // 尝试加载版本 3 的接口
        if (ncclProfiler == nullptr) {  // 如果版本 3 接口不存在
          ncclProfiler = getNcclProfiler_v2(profilerPluginLib);  // 尝试加载版本 2 的接口
          if (ncclProfiler == nullptr) {  // 如果版本 2 接口不存在
            ncclProfiler = getNcclProfiler_v1(profilerPluginLib);  // 尝试加载版本 1 的接口
          }
        }
      }
    }
    if (ncclProfiler == NULL) {  // 如果所有版本都加载失败
      if (profilerName) INFO(NCCL_INIT, "External profiler plugin %s is unsupported", profilerName);  // 记录日志：外部性能分析器插件不支持
      goto fail;  // 跳转到失败处理
    }
    if (profilerName) INFO(NCCL_INIT, "Successfully loaded external profiler plugin %s", profilerName);  // 记录日志：成功加载外部性能分析器插件

    ++profilerPluginRefCount;  // 增加插件引用计数
    profilerPluginStatus = profilerPluginLoadSuccess;  // 设置插件状态为加载成功

    // Store the pid of process loading the profiler plugin.
    // 存储加载性能分析器插件的进程 ID。

    // This is attached to the proxyOp event descriptor
    // 这附加到代理操作事件描述符

    // so plugin can figure out if parent event
    // 这样插件可以确定是否有父事件

    // is in the same address space or not
    // 是否在同一地址空间

    pid = getpid();  // 获取当前进程 ID

exit:  // 正常退出标签
  return ncclSuccess;  // 返回成功状态

fail:  // 失败处理标签
  if (profilerPluginLib) NCCLCHECK(ncclClosePluginLib(profilerPluginLib, ncclPluginTypeProfiler));  // 如果插件库句柄有效，关闭动态链接库
  profilerPluginLib = nullptr;  // 设置插件库句柄为空
  profilerPluginStatus = profilerPluginLoadFailed;  // 设置插件状态为加载失败
  goto exit;  // 跳转到退出标签
}

// NCCL 性能分析器插件卸载函数
static ncclResult_t ncclProfilerPluginUnload(void) {
  // 使用 RAII 方式加锁，保护插件卸载过程
  std::lock_guard<std::mutex> lock(profilerMutex);  // 自动加锁，函数退出时解锁
  if (0 == (--profilerPluginRefCount)) {  // 如果引用计数减为 0（最后一次引用）
    if (__builtin_expect(ncclProfiler != NULL, 0)) {  // 如果性能分析器接口有效（预期不为空）
      INFO(NCCL_INIT, "PROFILER/Plugin: Closing profiler plugin %s", ncclProfiler->name);  // 记录日志：关闭性能分析器插件
    }
    NCCLCHECK(ncclClosePluginLib(profilerPluginLib, ncclPluginTypeProfiler));  // 关闭性能分析器插件动态链接库
    profilerPluginLib = nullptr;  // 设置插件库句柄为空
    ncclProfiler = nullptr;  // 清空性能分析器接口指针
    profilerPluginStatus = profilerPluginLoadReady;  // 恢复插件状态为准备就绪
  }
  return ncclSuccess;  // 返回成功状态
}

// 定义是否启用计时器的宏（默认为 0，即禁用）
#define ENABLE_TIMER 0

// 如果启用了计时器，包含计时器头文件
#if ENABLE_TIMER
// These counters are used to measure profiler overheads for different part of code
// 这些计数器用于测量代码不同部分的开销

// These counters are only useful/meaningful in controlled test environments where there
// 这些计数器仅在受控测试环境中才有意义/有用

// is only one thread updating each set of counters, i.e., every communicator has its
// 只有一个线程更新每组计数器，即每个通信器有自己的

// own proxy thread and network uses only one thread to make progress (this is true
// 代理线程和网络只使用一个线程进行进度（这是真的）

// for net_ib plugin but might not be true for net_socket plugin).
// 对于 net_ib 插件是这样，但对于 net_socket 插件可能不是这样。

// 定义各部分耗时计数器（64位整数）
static int64_t elapsedCount;  // 总耗时计数
static int64_t initCount;  // 初始化耗时计数
static int64_t finalizeCount;  // 终止耗时计数
static int64_t groupStartCount, groupStopCount;  // 组启动/停止耗时计数
static int64_t taskStartCount, taskStopCount;  // 任务启动/停止耗时计数
static int64_t proxyOpStartCount, proxyOpStopCount;  // 代理操作启动/停止耗时计数
static int64_t proxyStepStartCount, proxyStepStopCount;  // 代理步骤启动/停止耗时计数
static int64_t proxyCtrlStartCount, proxyCtrlStopCount;  // 代理控制启动/停止耗时计数
static int64_t proxyOpRecordCount, proxyStepRecordCount;  // 代理操作记录/步骤记录耗时计数
static int64_t proxyCtrlStartCount, proxyCtrlStopCount;  // 代理控制启动/停止耗时计数
static int64_t proxyOpRecordTs[2], proxyStepRecordTs[2];  // 代理操作/步骤记录时间戳数组（2个元素）
static double proxyCtrlStartTs[2], proxyCtrlStopTs[2];  // 代理控制启动/停止时间戳数组（2个元素）

// 定义开始事件计时的宏（记录开始时间戳）
#define TIME_START_EVENT(event) do { \
  (event ## Count)++; \  // 增加事件计数
  (event ## Ts)[0] = gettime(); \  // 记录开始时间戳到数组的第一个元素
} while(0)  // 循环 0 次（实际上是空循环，用于配合 do-while(0)）

// 定义停止事件计时的宏（计算时间差并累加）
#define TIME_STOP_EVENT(event) do { \
  double val = gettime() - (event ## Ts)[0]; \  // 计算时间差值
  (event ## Ts)[1] += val; \  // 累加时间差到数组的第二个元素
} while(0)  // 循环 0 次（实际上是空循环）

// 定义打印所有事件的宏（输出统计信息）
#define TIME_PRINT_EVENTS(name) do { \
  printf("%s ", name); \  // 打印事件名称
  if (elapsedCount)         printf("[elapsed] %g/%ld = %g ", elapsedTs[1], elapsedCount, elapsedTs[1]/elapsedCount); \  // 打印总耗时
  if (initCount)            printf("[init] %g/%ld = %g ", initTs[1], initCount, initTs[1]/initCount); \  // 打印初始化耗时
  if (finalizeCount)        printf("[finalize] %g/%ld = %g ", finalizeTs[1], finalizeCount, finalizeTs[1]/finalizeCount); \  // 打印终止耗时
  if (groupStartCount)      printf("[groupStart] %g/%ld = %g ", groupStartTs[1], groupStartCount, groupStartTs[1]/groupStartCount); \  // 打印组启动耗时
  if (groupStopCount)       printf("[groupStop] %g/%ld = %g ", groupStopTs[1], groupStopCount, groupStopTs[1]/groupStopCount); \  // 打印组停止耗时
  if (taskStartCount)       printf("[taskStart] %g/%ld = %g ", taskStartTs[1], taskStartCount, taskStartTs[1]/taskStartCount); \  // 打印任务启动耗时
  if (taskStopCount)        printf("[taskStop] %g/%ld = %g ", taskStopTs[1], taskStopCount, taskStopTs[1]/taskStopCount); \  // 打印任务停止耗时
  if (proxyOpStartCount)    printf("[proxyOpStart] %g/%ld = %g ", proxyOpStartTs[1], proxyOpStartCount, proxyOpStartTs[1]/proxyOpStartCount); \  // 打印代理操作启动耗时
  if (proxyOpStopCount)     printf("[proxyOpStop] %g/%ld = %g ", proxyOpStopTs[1], proxyOpStopCount, proxyOpStopTs[1]/proxyOpStopCount); \  // 打印代理操作停止耗时
  if (proxyStepStartCount)   printf("[proxyStepStart] %g/%ld = %g ", proxyStepStartTs[1], proxyStepStartCount, proxyStepStartTs[1]/proxyStepStartCount); \  // 打印代理步骤启动耗时
  if (proxyStepStopCount)    printf("[proxyStepStop] %g/%ld = %g ", proxyStepStopTs[1], proxyStepStopCount, proxyStepStopTs[1]/proxyStepStopCount); \  // 打印代理步骤停止耗时
  if (proxyCtrlStartCount)   printf("[proxyCtrlStart] %g/%ld = %g ", proxyCtrlStartTs[1], proxyCtrlStartCount, proxyCtrlStartTs[1]/proxyCtrlStartCount); \  // 打印代理控制启动耗时
  if (proxyCtrlStopCount)    printf("[proxyCtrlStop] %g/%ld = %g ", proxyCtrlStopTs[1], proxyCtrlStopCount, proxyCtrlStopTs[1]/proxyCtrlStopCount); \  // 打印代理控制停止耗时
  if (proxyOpRecordCount)   printf("[proxyOpRecord] %g/%ld = %g ", proxyOpRecordTs[1], proxyOpRecordCount, proxyOpRecordTs[1]/proxyOpRecordCount); \  // 打印代理操作记录耗时
  if (proxyStepRecordCount)  printf("[proxyStepRecord] %g/%ld = %g", proxyStepRecordTs[1], proxyStepRecordCount, proxyStepRecordTs[1]/proxyStepRecordCount); \  // 打印代理步骤记录耗时
  if (proxyCtrlRecordCount)  printf("[proxyCtrlRecord] %g/%ld = %g", proxyCtrlRecordTs[1], proxyCtrlRecordCount, proxyCtrlRecordTs[1]/proxyCtrlRecordCount); \  // 打印代理控制记录耗时
  printf("\n"); \  // 打印换行符
} while(0)  // 循环 0 次（实际上是空循环，用于配合 do-while(0)）

// 如果禁用计时器，定义空宏以避免计时代码
#else
// 定义空操作的开始事件宏（不执行任何操作）
#define TIME_START_EVENT(event) do {} while(0)
// 定义空操作的停止事件宏（不执行任何操作）
#define TIME_STOP_EVENT(event) do {} while(0)
// 定义空操作的打印事件宏（不执行任何操作）
#define TIME_PRINT_EVENTS(name) do {} while(0)
#endif

// 定义性能分析器事件掩码，由性能分析器设置
int ncclProfilerEventMask;       // Set by profiler
// 由性能分析器设置的事件掩码

// NCCL 性能分析器插件初始化函数
// 参数：comm - NCCL 通信上下文
ncclResult_t ncclProfilerPluginInit(struct ncclComm* comm) {
  TIME_START_EVENT(elapsed);  // 开始总耗时计时
  TIME_START_EVENT(init);  // 开始初始化计时
  ncclProfilerPluginLoad();  // 加载性能分析器插件
  if (__builtin_expect(ncclProfiler != NULL, 0)) {  // 如果性能分析器接口有效（预期不为空）
    int err = ncclProfiler->init(&comm->profilerContext, comm->commHash, &ncclProfilerEventMask, comm->config.commName, comm->nNodes, comm->nRanks, comm->rank, ncclDebugLog);  // 调用性能分析器初始化函数
    if (err) {  // 如果初始化失败
      INFO(NCCL_INIT, "Profiler init failed with error '%d': %s. Continue without profiler.", err, strerror(errno));  // 记录日志：性能分析器初始化失败，继续但不使用性能分析器
    }
  }
  TIME_STOP_EVENT(init);  // 停止初始化计时
  return ncclSuccess;  // 返回成功状态
}

// NCCL 性能分析器插件终止函数
// 参数：comm - NCCL 通信上下文
ncclResult_t ncclProfilerPluginFinalize(struct ncclComm* comm) {
  TIME_START_EVENT(finalize);  // 开始终止计时
  if (__builtin_expect(ncclProfiler != NULL, 0) && comm->profilerContext) {  // 如果性能分析器接口有效且性能分析器上下文有效
    ncclProfiler->finalize(comm->profilerContext);  // 调用性能分析器终止函数
  }
  ncclProfilerPluginUnload();  // 卸载性能分析器插件
  TIME_STOP_EVENT(finalize);  // 停止终止计时
  TIME_STOP_EVENT(elapsed);  // 停止总耗时计时
  TIME_PRINT_EVENTS("Profiler");  // 打印所有性能统计信息
  return ncclSuccess;  // 返回成功状态
}

// 开始组 API 事件函数
// 参数说明：
//   info - NCCL 信息结构体指针
//   isGraphCaptured - 是否为 CUDA Graph 捕获模式
ncclResult_t ncclProfilerStartGroupApiEvent(struct ncclInfo* info, bool isGraphCaptured) {
  ncclProfilerEventDescr_t eDescr = { 0 };  // 初始化事件描述符结构体为 0
  eDescr.type = ncclProfileGroupApi;  // 设置事件类型为组 API
  eDescr.groupApi.graphCaptured = isGraphCaptured;  // 设置是否为 CUDA Graph 捕获模式

  // 原子加载事件掩码，用于控制事件记录
  ncclProfilerApiState.eActivationMask = __atomic_load_n(&ncclProfilerEventMask, __ATOMIC_RELAXED);  // 原子加载事件掩码

  // 计算组 API 掩码，包括组 API、P2P API、集合 API 等
  int groupApiMask = ncclProfileGroupApi | ncclProfileP2pApi | ncclProfileCollApi | ncclProfileKernelLaunch | ncclProfileGroup | ncclProfileColl | ncclProfileP2p | ncclProfileProxyOp | ncclProfileProxyStep | ncclProfileKernelCh | ncclProfileNetPlugin;  // 计算所有相关类型的掩码
  // Only count outermost groups when emitting group API events
  // 仅在最外层组发出组 API 事件时才计数

  if (__builtin_expect(ncclProfiler != NULL, 0) && (ncclProfilerApiState.eActivationMask & groupApiMask)) {  // 如果性能分析器有效且组 API 掩码激活
    if (ncclProfilerApiState.profilerGroupDepth == 0) {  // 如果组深度为 0（表示最外层）
      eDescr.groupApi.groupDepth = ncclGroupDepth;  // 设置组深度
      ncclProfiler->startEvent(info->comm->profilerContext, &ncclProfilerApiState.groupApiEventHandle, &eDescr);  // 启动组 API 事件
      ncclProfilerApiState.profilerGroupDepth = ncclGroupDepth;  // 设置组深度为当前深度
      ncclProfilerApiState.state = ncclProfilerGroupApiStartStateStarted;  // 设置 API 状态为已启动
    }
  }
  return ncclSuccess;  // 返回成功状态
}

// 停止组 API 事件函数
ncclResult_t ncclProfilerStopGroupApiEvent() {
  void* groupApiEventHandle = ncclProfilerApiState.groupApiEventHandle;  // 获取组 API 事件句柄
  if (__builtin_expect(ncclProfiler != NULL, 0) && groupApiEventHandle && ncclProfilerApiState.profilerGroupDepth == 0) {  // 如果性能分析器有效、事件句柄有效且在最外层
    ncclProfiler->stopEvent(groupApiEventHandle);  // 停止组 API 事件
    ncclProfilerApiState.groupApiEventHandle = nullptr;  // 清空组 API 事件句柄
  }
  return ncclSuccess;  // 返回成功状态
}

// 记录组 API 事件状态函数
// 参数：eState - 事件状态
ncclResult_t ncclProfilerRecordGroupApiEventState(ncclProfilerEventState_t eState) {
  void* groupApiEventHandle = ncclProfilerApiState.groupApiEventHandle;  // 获取组 API 事件句柄
  bool shouldRecord = false;  // 是否需要记录标志
  if (eState == ncclProfilerGroupStartApiStop && ncclProfilerApiState.state == ncclProfilerGroupApiStartStateStarted) {  // 如果状态从启动变为停止
    ncclProfilerApiState.state = ncclProfilerGroupApiStartStateStopped;  // 设置状态为已停止
    shouldRecord = true;  // 设置需要记录标志
  } else if (eState == ncclProfilerGroupEndApiStart && ncclProfilerApiState.state == ncclProfilerGroupApiStartStateStopped) {  // 如果状态从结束启动变为停止
    ncclProfilerApiState.state = ncclProfilerGroupApiStartStateReset;  // 设置状态为重置
    shouldRecord = true;  // 设置需要记录标志
  }
  if (__builtin_expect(ncclProfiler != NULL, 0) && groupApiEventHandle && shouldRecord) {  // 如果性能分析器有效、事件句柄有效且需要记录
    ncclProfiler->recordEventState(groupApiEventHandle, eState, NULL);  // 记录事件状态
  }
  return ncclSuccess;  // 返回成功状态
}

// 启动 P2P API 事件函数
// 参数说明：
//   info - NCCL 信息结构体指针
//   isGraphCaptured - 是否为 CUDA Graph 捕获模式
ncclResult_t ncclProfilerStartP2pApiEvent(struct ncclInfo *info, bool isGraphCaptured) {
  ncclProfilerEventDescr_t eDescr = { 0 };  // 初始化事件描述符结构体为 0
  eDescr.type = ncclProfileP2pApi;  // 设置事件类型为 P2P API
  eDescr.parentObj = ncclProfilerApiState.groupApiEventHandle;  // 设置父对象为组 API 事件句柄
  eDescr.p2pApi.func = ncclFuncToString(info->coll);  // 设置集合函数名称
  eDescr.p2pApi.count = info->count;  // 设置操作数量
  eDescr.p2pApi.datatype = ncclDatatypeToString(info->datatype);  // 设置数据类型
  eDescr.p2pApi.stream = (void *) info->stream;  // 设置 CUDA 流
  eDescr.p2pApi.graphCaptured = isGraphCaptured;  // 设置是否为 CUDA Graph 捕获模式
  eDescr.p2pApiMask = ncclProfileP2pApi | ncclProfileP2p | ncclProfileProxyOp | ncclProfileProxyStep | ncclProfileKernelCh | ncclProfileNetPlugin;  // 计算 P2P API 掩码
  if (__builtin_expect(ncclProfiler != NULL, 0) && (ncclProfilerApiState.eActivationMask & eDescr.p2pApiMask)) {  // 如果性能分析器有效且 P2P API 掩码激活
    ncclProfiler->startEvent(info->comm->profilerContext, &ncclProfilerApiState.p2pApiEventHandle, &eDescr);  // 启动 P2P API 事件
  }
  return ncclSuccess;  // 返回成功状态
}

// 停止 P2P API 事件函数
ncclResult_t ncclProfilerStopP2pApiEvent() {
  if (__builtin_expect(ncclProfiler != NULL, 0) && ncclProfilerApiState.p2pApiEventHandle) {  // 如果性能分析器有效且 P2P API 事件句柄有效
    ncclProfiler->stopEvent(ncclProfilerApiState.p2pApiEventHandle);  // 停止 P2P API 事件
    ncclProfilerApiState.p2pApiEventHandle = nullptr;  // 清空 P2P API 事件句柄
  }
  return ncclSuccess;  // 返回成功状态
}

// 启动集合 API 事件函数
// 参数说明：
//   info - NCCL 信息结构体指针
//   isGraphCaptured - 是否为 CUDA Graph 捕获模式
ncclResult_t ncclProfilerStartCollApiEvent(struct ncclInfo *info, bool isGraphCaptured) {
  ncclProfilerEventDescr_t eDescr = { 0 };  // 初始化事件描述符结构体为 0
  eDescr.type = ncclProfileCollApi;  // 设置事件类型为集合 API
  eDescr.parentObj = ncclProfilerApiState.groupApiEventHandle;  // 设置父对象为组 API 事件句柄
  eDescr.collApi.func = ncclFuncToString(info->coll);  // 设置集合函数名称
  eDescr.collApi.count = info->count;  // 设置操作数量
  eDescr.collApi.datatype = ncclDatatypeToString(info->datatype);  // 设置数据类型
  eDescr.collApi.stream = (void *) info->stream;  // 设置 CUDA 流
  eDescr.collApi.root = info->root;  // 设置根进程 rank
  eDescr.collApi.graphCaptured = isGraphCaptured;  // 设置是否为 CUDA Graph 捕获模式
  eDescr.collApiMask = ncclProfileCollApi | ncclProfileColl | ncclProfileProxyOp | ncclProfileProxyStep | ncclProfileKernelCh | ncclProfileNetPlugin;  // 计算集合 API 掩码
  if (__builtin_expect(ncclProfiler != NULL, 0) && (ncclProfilerApiState.eActivationMask & eDescr.collApiMask)) {  // 如果性能分析器有效且集合 API 掩码激活
    ncclProfiler->startEvent(info->comm->profilerContext, &ncclProfilerApiState.collApiEventHandle, &eDescr);  // 启动集合 API 事件
  }
  return ncclSuccess;  // 返回成功状态
}

// 停止集合 API 事件函数
ncclResult_t ncclProfilerStopCollApiEvent() {
  if (__builtin_expect(ncclProfiler != NULL, 0) && ncclProfilerApiState.collApiEventHandle) {  // 如果性能分析器有效且集合 API 事件句柄有效
    ncclProfiler->stopEvent(ncclProfilerApiState.collApiEventHandle);  // 停止集合 API 事件
  }
  return ncclSuccess;  // 返回成功状态
}

// 启动 Kernel Launch 事件函数
// 参数说明：
//   plan - Kernel 计划指针
//   stream - CUDA 流
ncclResult_t ncclProfilerStartKernelLaunchEvent(struct ncclKernelPlan* plan, cudaStream_t stream) {
  ncclProfilerEventDescr_t eDescr = { 0 };  // 初始化事件描述符结构体为 0
  if (__builtin_expect(ncclProfiler != NULL, 0)) {  // 如果性能分析器接口有效
    void* groupApiEventHandle = NULL;  // 清空组 API 事件句柄
    // Check if any collective in the plan has a set event activation mask
    // 检查计划中是否有任何集合操作设置了事件激活掩码

    struct ncclTaskColl* ct = ncclIntruQueueHead(&plan->collTaskQueue);  // 获取集合任务队列头指针
    struct ncclTaskP2p* pt = ncclIntruQueueHead(&plan->p2pTaskQueue);  // 获取 P2P 任务队列头指针
    int eActivationMask_ = 0;  // 初始化有效激活掩码
    while (ct) {  // 遍历所有集合任务
      if (ct->eActivationMask) {  // 如果集合任务有激活掩码
        eActivationMask_ = ct->eActivationMask;  // 更新有效激活掩码
        groupApiEventHandle = ct->groupApiEventHandle;  // 设置组 API 事件句柄
        goto startKernelLaunchEvent;  // 跳转到启动 Kernel Launch 事件
      }
      ct = ct->next;  // 移动到下一个集合任务
    }
    // Check if any pt2pt in plan has a set event activation mask
    // 检查计划中是否有任何 P2P 任务设置了事件激活掩码

    while (pt) {  // 遍历所有 P2P 任务
      if (pt->eActivationMask) {  // 如果 P2P 任务有激活掩码
        eActivationMask_ = pt->eActivationMask;  // 更新有效激活掩码
        groupApiEventHandle = pt->groupApiEventHandle;  // 设置组 API 事件句柄
        goto startKernelLaunchEvent;  // 跳转到启动 Kernel Launch 事件
      }
      pt = pt->next;  // 移动到下一个 P2P 任务
    }

startKernelLaunchEvent:  // Kernel Launch 事件启动标签
  if (eActivationMask_ & ncclProfileKernelLaunch) {  // 如果内核启动激活掩码被设置
    eDescr.type = ncclProfileKernelLaunch;  // 设置事件类型为内核启动
    eDescr.parentObj = groupApiEventHandle;  // 设置父对象为组 API 事件句柄
    eDescr.kernelLaunch.stream = (void *) stream;  // 设置 CUDA 流
    ncclProfiler->startEvent(plan->comm->profilerContext, &plan->kernelLaunchEventHandle, &eDescr);  // 启动内核启动事件
  }
  }
  return ncclSuccess;  // 返回成功状态
}

// 停止 Kernel Launch 事件函数
// 参数：plan - Kernel 计划指针
ncclResult_t ncclProfilerStopKernelLaunchEvent(struct ncclKernelPlan* plan) {
  if (__builtin_expect(ncclProfiler != NULL, 0) && plan->kernelLaunchEventHandle) {  // 如果性能分析器有效且内核启动事件句柄有效
    ncclProfiler->stopEvent(plan->kernelLaunchEventHandle);  // 停止内核启动事件
  }
  return ncclSuccess;  // 返回成功状态
}

// 启动组事件函数
// 参数：plan - Kernel 计划指针
ncclResult_t ncclProfilerStartGroupEvent(struct ncclKernelPlan* plan) {
  TIME_START_EVENT(groupStart);  // 开始组启动计时
  if (__builtin_expect(ncclProfiler != NULL, 0)) {  // 如果性能分析器接口有效
    // Check if any collective in the plan has a set event activation mask
    // 检查计划中是否有任何集合操作设置了事件激活掩码

    struct ncclTaskColl* ct = ncclIntruQueueHead(&plan->collTaskQueue);  // 获取集合任务队列头指针
    struct ncclTaskP2p* pt = ncclIntruQueueHead(&plan->p2pTaskQueue);  // 获取 P2P 任务队列头指针
    int eActivationMask_ = 0;  // 初始化有效激活掩码
    while (ct) {  // 遍历所有集合任务
      if (ct->eActivationMask) {  // 如果集合任务有激活掩码
        eActivationMask_ = ct->eActivationMask;  // 更新有效激活掩码
        goto startGroup;  // 跳转到启动组
      }
      ct = ct->next;  // 移动到下一个集合任务
    }
    // Check if any pt2pt in plan has a set event activation mask
    // 检查计划中是否有任何 P2P 任务设置了事件激活掩码

    while (pt) {  // 遍历所有 P2P 任务
      if (pt->eActivationMask) {  // 如果 P2P 任务有激活掩码
        eActivationMask_ = pt->eActivationMask;  // 更新有效激活掩码
        goto startGroup;  // 跳转到启动组
      }
      pt = pt->next;  // 移动到下一个 P2P 任务
    }

startGroup:  // 启动组标签
  if (eActivationMask_ & (ncclProfileGroup | ncclProfileColl | ncclProfileP2p | ncclProfileProxyOp | ncclProfileProxyStep | ncclProfileKernelCh | ncclProfileNetPlugin)) {  // 如果有效激活掩码包含所有组相关类型
    ncclProfilerEventDescr_t eDescr = { 0 };  // 初始化事件描述符结构体为 0
    eDescr.type = ncclProfileGroup;  // 设置事件类型为组
    eDescr.parentObj = plan->groupEventHandle;  // 设置父对象为组事件句柄
    ncclProfiler->startEvent(plan->comm->profilerContext, &plan->groupEventHandle, &eDescr);  // 启动组事件
  }
  }
  TIME_STOP_EVENT(groupStart);  // 停止组启动计时
  return ncclSuccess;  // 返回成功状态
}

// 停止组事件函数
// 参数：plan - Kernel 计划指针
ncclResult_t ncclProfilerStopGroupEvent(struct ncclKernelPlan* plan) {
  TIME_START_EVENT(groupStop);  // 开始组停止计时
  if (__builtin_expect(ncclProfiler != NULL, 0) && plan->groupEventHandle) {  // 如果性能分析器有效且组事件句柄有效
    ncclProfiler->stopEvent(plan->groupEventHandle);  // 停止组事件
  }
  TIME_STOP_EVENT(groupStop);  // 停止组停止计时
  return ncclSuccess;  // 返回成功状态
}

// 启动任务事件函数
// 参数：plan - Kernel 计划指针
ncclResult_t ncclProfilerStartTaskEvents(struct ncclKernelPlan* plan) {
  TIME_START_EVENT(taskStart);  // 开始任务启动计时
  struct ncclTaskColl* ct = ncclIntruQueueHead(&plan->collTaskQueue);  // 获取集合任务队列头指针
  while (ct) {  // 遍历所有集合任务
    if (__builtin_expect(ncclProfiler != NULL, 0)) {  // 如果性能分析器接口有效
      int enable = ct->eActivationMask & (ncclProfileColl | ncclProfileProxyOp | ncclProfileProxyStep | ncclProfileKernelCh | ncclProfileNetPlugin);  // 计算是否启用此事件
      if (enable) {  // 如果启用
        ncclProfilerEventDescr_t eDescr = { 0 };  // 初始化事件描述符结构体为 0
        eDescr.type = ncclProfileColl;  // 设置事件类型为集合
        eDescr.coll.parentGroup = plan->groupEventHandle;  // 设置父组对象为组事件句柄
        eDescr.parentObj = ct->collApiEventHandle;  // 设置父对象为集合 API 事件句柄
        eDescr.rank = plan->comm->rank;  // 设置 rank
        eDescr.coll.seqNumber = plan->comm->seqNumber[ct->func];  // 设置序列号
        eDescr.coll.func = ncclFuncToString(ct->func);  // 设置集合函数名称
        eDescr.coll.sendBuff = ct->sendbuff;  // 设置发送缓冲区
        eDescr.coll.recvBuff = ct->recvbuff;  // 设置接收缓冲区
        eDescr.coll.count = ct->count;  // 设置操作数量
        eDescr.coll.root = ct->root;  // 设置根进程 rank
        eDescr.coll.datatype = ncclDatatypeToString(ct->datatype);  // 设置数据类型
        eDescr.coll.nChannels = ct->nChannels;  // 设置通道数量
        eDescr.coll.nWarps = ct->nWarps;  // 设置 warp 数量
        eDescr.coll.algo = ncclAlgoToString(ct->algorithm);  // 设置算法名称
        eDescr.coll.proto = ncclProtoToString(ct->protocol);  // 设置协议名称
        eDescr.coll.stream = (void *) info->stream;  // 设置 CUDA 流
        ncclProfiler->startEvent(plan->comm->profilerContext, &ct->eventHandle, &eDescr);  // 启动集合事件
      }
    }
    // comm->seqNumber values are updated even if the plugin is not active, since they are used by RAS as well.
    // comm->seqNumber 值即使插件未激活也会被更新，因为 RAS 也会使用它们。

    // The test for "persistent" is a workaround for graph-captured collectives. In their case this function may not be
    // "persistent" 测试是 graph 捕获集合的变通方法。在这种情况下，此函数可能不会

    // consistently invoked on all ranks, which would lead to mismatched counter values and thus false-positive
    // 在所有 rank 上一致调用，会导致计数器值不匹配，从而产生误报

    // reports from RAS. Instead, we choose not to include graph-captured collectives in our counts. An exception is
    // 来自 RAS 的报告。相反，我们选择不在计数中包含 graph 捕获的集合。例外情况是

    // made if ncclProfileKernelCh profiler events are active, as they result in proxy events always being added, which
    // 如果 ncclProfileKernelCh 性能分析器事件处于活动状态，因为它们会导致代理事件始终被添加，这

    // gives us consistency.
    // 给我们一致性。

    if (!plan->persistent || (__builtin_expect(ncclProfiler != NULL, 0) && (plan->groupEventHandle || ct->collApiEventHandle)) &&  // 如果非持久化计划且存在组或集合 API 事件句柄
                              (ct->eActivationMask & ncclProfileKernelCh)))  // 且集合任务的激活掩码包含内核启动
      __atomic_fetch_add(&plan->comm->seqNumber[ct->func], 1, __ATOMIC_RELAXED);  // 原子递增序列号
      ct = ct->next;  // 移动到下一个集合任务
    }
  }
  TIME_STOP_EVENT(taskStart);  // 停止任务启动计时
  return ncclSuccess;  // 返回成功状态
}

// 停止任务事件函数
// 参数：plan - Kernel 计划指针
ncclResult_t ncclProfilerStopTaskEvents(struct ncclKernelPlan* plan) {
  TIME_START_EVENT(taskStop);  // 开始任务停止计时
  if (__builtin_expect(ncclProfiler != NULL, 0)) {  // 如果性能分析器接口有效
    struct ncclTaskColl* ct = ncclIntruQueueHead(&plan->collTaskQueue);  // 获取集合任务队列头指针
    while (ct) {  // 遍历所有集合任务
      if (ct->eventHandle) ncclProfiler->stopEvent(ct->eventHandle);  // 如果事件句柄有效，停止事件
      ct = ct->next;  // 移动到下一个集合任务
    }
    struct ncclTaskP2p* pt = ncclIntruQueueHead(&plan->p2pTaskQueue);  // 获取 P2P 任务队列头指针
    while (pt) {  // 遍历所有 P2P 任务
      if (pt->eventHandle) ncclProfiler->stopEvent(pt->eventHandle);  // 如果事件句柄有效，停止事件
      pt = pt->next;  // 移动到下一个 P2P 任务
    }
  }
  TIME_STOP_EVENT(taskStop);  // 停止任务停止计时
  return ncclSuccess;  // 返回成功状态
}

// Bellow we set proxy descriptor step number to DIVUP(step, args->sliceSteps).
// 下面我们将代理描述符步号设置为 DIVUP(step, args->sliceSteps)。

// The reason is that for some ncclOp (e.g. AllReduce) one network transfer is
// 原因是对于某些 ncclOp（如 AllReduce），一次网络传输

// made of sliceSteps steps rather than one step. In the profiler we are still
// 包含 sliceSteps 步而不是一步。在性能分析器中，我们仍然

// interested in whole network transfers though, so we account for this when
// 感兴趣的是整个网络传输，因此我们在计算时考虑到这一点

// computing the actual network step number.
// 计算实际的网络步数。

// 启动代理操作事件函数
// 参数说明：
//   s - 子操作索引
//   args - 代理参数指针
ncclResult_t ncclProfilerStartProxyOpEvent(int s, struct ncclProxyArgs* args) {
  TIME_START_EVENT(proxyOpStart);  // 开始代理操作启动计时
  struct ncclProxySubArgs* sub = &args->subs[s];  // 获取子操作指针
  if (__builtin_expect(ncclProfiler != NULL, 0)) {  // 如果性能分析器接口有效
    if (sub->eActivationMask & (ncclProfileProxyOp | ncclProfileProxyStep | ncclProfileNetPlugin)) {  // 如果子操作的激活掩码包含代理操作类型
      ncclProfilerEventDescr_t eDescr = { 0 };  // 初始化事件描述符结构体为 0
      eDescr.type = ncclProfileProxyOp;  // 设置事件类型为代理操作
      eDescr.parentObj = sub->taskEventHandle;  // 设置父对象为任务事件句柄
      eDescr.rank = sub->rank;  // 设置 rank
      eDescr.proxyOp.pid = sub->pid;  // 设置进程 ID
      eDescr.proxyOp.channelId = sub->channelId;  // 设置通道 ID
      eDescr.proxyOp.nSteps = DIVUP(sub->nsteps, args->sliceSteps);  // 设置网络步数（向上取整到 sliceSteps）
      eDescr.proxyOp.chunkSize = args->chunkSize * args->sliceSteps;  // 设置块大小（乘以 sliceSteps）
      eDescr.proxyOp.isSend = args->progress == ncclTransports[TRANSPORT_NET]->send.proxyProgress ? 1 : 0;  // 设置是否为发送操作（根据传输类型判断）
      ncclProfiler->startEvent(sub->profilerContext, &sub->opEventHandle, &eDescr);  // 启动代理操作事件
    }
  }
  TIME_STOP_EVENT(proxyOpStart);  // 停止代理操作启动计时
  return ncclSuccess;  // 返回成功状态
}

// 停止代理操作事件函数
// 参数说明：
//   s - 子操作索引
//   args - 代理参数指针
ncclResult_t ncclProfilerStopProxyOpEvent(int s, struct ncclProxyArgs* args) {
  TIME_START_EVENT(proxyOpStop);  // 开始代理操作停止计时
  struct ncclProxySubArgs* sub = &args->subs[s];  // 获取子操作指针
  if (__builtin_expect(ncclProfiler != NULL, 0) && sub->opEventHandle) {  // 如果性能分析器有效且操作事件句柄有效
    ncclProfiler->stopEvent(sub->opEventHandle);  // 停止操作事件
    sub->opEventHandle = NULL;  // 清空操作事件句柄
  }
  TIME_STOP_EVENT(proxyOpStop);  // 停止代理操作停止计时
  return ncclSuccess;  // 返回成功状态
}

// 启动代理步骤事件函数
// 参数说明：
//   s - 子操作索引
//   args - 代理参数指针
//   stepId - 步骤 ID
ncclResult_t ncclProfilerStartSendProxyStepEvent(int s, struct ncclProxyArgs* args, int stepId) {
  TIME_START_EVENT(proxyStepStart);  // 开始代理步骤启动计时
  struct ncclProxySubArgs* sub = &args->subs[s];  // 获取子操作指针
  int step_ = DIVUP(stepId, args->sliceSteps);  // 计算当前步骤号（向上取整）
  if (__builtin_expect(ncclProfiler != NULL, 0)) {  // 如果性能分析器接口有效
    if (sub->eActivationMask & (ncclProfileProxyStep | ncclProfileNetPlugin)) {  // 如果子操作的激活掩码包含代理步骤类型
      ncclProfilerEventDescr_t eDescr = { 0 };  // 初始化事件描述符结构体为 0
      eDescr.type = ncclProfileProxyStep;  // 设置事件类型为代理步骤
      eDescr.parentObj = sub->opEventHandle;  // 设置父对象为操作事件句柄
      eDescr.rank = sub->rank;  // 设置 rank
      eDescr.proxyStep.step = step_;  // 设置步骤号
      ncclProfiler->startEvent(sub->profilerContext, &sub->pHandles[step_%NCCL_STEPS].stepEventHandle, &eDescr);  // 启动代理步骤事件
    }
  }
  sub->pHandles[step_%NCCL_STEPS].subArgPtr = sub;  // 保存子操作指针到步骤句柄数组
  TIME_STOP_EVENT(proxyStepStart);  // 停止代理步骤启动计时
  return ncclSuccess;  // 返回成功状态
}

// 启动接收代理步骤事件函数
// 参数说明：
//   s - 子操作索引
//   args - 代理参数指针
//   stepId - 步骤 ID
ncclResult_t ncclProfilerStartRecvProxyStepEvent(int s, struct ncclProxyArgs* args, int stepId) {
  TIME_START_EVENT(proxyStepStart);  // 开始代理步骤启动计时
  struct ncclProxySubArgs* sub = &args->subs[s];  // 获取子操作指针
  int step_ = DIVUP(stepId, args->sliceSteps);  // 计算当前步骤号（向上取整）
  if (__builtin_expect(ncclProfiler != NULL, 0)) {  // 如果性能分析器接口有效
    if (sub->eActivationMask & (ncclProfileProxyStep | ncclProfileNetPlugin)) {  // 如果子操作的激活掩码包含代理步骤类型
      ncclProfilerEventDescr_t eDescr = { 0 };  // 初始化事件描述符结构体为 0
      eDescr.type = ncclProfileProxyStep;  // 设置事件类型为代理步骤
      eDescr.parentObj = sub->opEventHandle;  // 设置父对象为操作事件句柄
      eDescr.rank = sub->rank;  // 设置 rank
      eDescr.proxyStep.step = step_;  // 设置步骤号
      ncclProfiler->startEvent(sub->profilerContext, &sub->pHandles[step_%NCCL_STEPS].stepEventHandle, &eDescr);  // 启动代理步骤事件
    }
  }
  sub->pHandles[step_%NCCL_STEPS].subArgPtr = sub;  // 保存子操作指针到步骤句柄数组
  TIME_STOP_EVENT(proxyStepStart);  // 停止代理步骤启动计时
  return ncclSuccess;  // 返回成功状态
}

// 停止代理步骤事件函数
// 参数说明：
//   s - 子操作索引
//   args - 代理参数指针
//   stepId - 步骤 ID
ncclResult_t ncclProfilerStopProxyStepEvent(int s, struct ncclProxyArgs* args, int stepId) {
  TIME_START_EVENT(proxyStepStop);  // 开始代理步骤停止计时
  struct ncclProxySubArgs* sub = &args->subs[s];  // 获取子操作指针
  if (__builtin_expect(ncclProfiler != NULL, 0)) {  // 如果性能分析器接口有效
    int step_ = DIVUP(stepId, args->sliceSteps);  // 计算当前步骤号（向上取整）
    if (sub->pHandles[step_%NCCL_STEPS].stepEventHandle) {  // 如果步骤事件句柄有效
      ncclProfiler->stopEvent(sub->pHandles[step_%NCCL_STEPS].stepEventHandle);  // 停止步骤事件
      sub->pHandles[step_%NCCL_STEPS].stepEventHandle = NULL;  // 清空步骤事件句柄
    }
  }
  TIME_STOP_EVENT(proxyStepStop);  // 停止代理步骤停止计时
  return ncclSuccess;  // 返回成功状态
}

// 启动代理控制事件函数
// 参数说明：
//   profilerContext - 性能分析器上下文
//   eHandle - 输出参数，返回事件句柄
ncclResult_t ncclProfilerStartProxyCtrlEvent(void* profilerContext, void** eHandle) {
  TIME_START_EVENT(proxyCtrlStart);  // 开始代理控制启动计时
  if (__builtin_expect(ncclProfiler != NULL, 0)) {  // 如果性能分析器接口有效
    // for proxy control events we allow profiling mode to change on a per event basis
    // 对于代理控制事件，我们允许在每个事件基础上更改性能分析模式

    int eActivationMaskProxy = __atomic_load_n(&ncclProfilerEventMask, __ATOMIC_RELAXED);  // 原子加载事件掩码
    if (eActivationMaskProxy & ncclProfileProxyCtrl) {  // 如果激活掩码包含代理控制类型
      ncclProfilerEventDescr_t eDescr = { 0 };  // 初始化事件描述符结构体为 0
      eDescr.type = ncclProfileProxyCtrl;  // 设置事件类型为代理控制
      ncclProfiler->startEvent(profilerContext, eHandle, &eDescr);  // 启动代理控制事件
      TIME_STOP_EVENT(proxyCtrlStart);  // 停止代理控制启动计时
      return ncclSuccess;  // 返回成功状态
    }
  }
  *eHandle = NULL;  // 清空事件句柄
  TIME_STOP_EVENT(proxyCtrlStart);  // 停止代理控制启动计时
  return ncclSuccess;  // 返回成功状态
}

// 停止代理控制事件函数
// 参数：eHandle - 事件句柄
ncclResult_t ncclProfilerStopProxyCtrlEvent(void* eHandle) {
  TIME_START_EVENT(proxyCtrlStop);  // 开始代理控制停止计时
  if (__builtin_expect(ncclProfiler != NULL, 0) && eHandle) {  // 如果性能分析器接口有效且事件句柄有效
    ncclProfiler->stopEvent(eHandle);  // 停止事件
  }
  TIME_STOP_EVENT(proxyCtrlStop);  // 停止代理控制停止计时
  return ncclSuccess;  // 返回成功状态
}

// 启动 Kernel 通道事件函数
// 参数说明：
//   args - 代理参数指针
//   s - 子操作索引
//   start - 开始时间戳
ncclResult_t ncclProfilerStartKernelChEvent(struct ncclProxyArgs* args, int s, uint64_t start) {
  if (__builtin_expect(ncclProfiler != NULL, 0)) {  // 如果性能分析器接口有效
    struct ncclProxySubArgs* sub = &args->subs[s];  // 获取子操作指针
    if (sub->eActivationMask & ncclProfileKernelCh) {  // 如果子操作的激活掩码包含内核通道类型
      ncclProfilerEventDescr_t eDescr = { };  // 初始化事件描述符结构体
      eDescr.type = ncclProfileKernelCh;  // 设置事件类型为内核通道
      eDescr.parentObj = sub->taskEventHandle;  // 设置父对象为任务事件句柄
      eDescr.kernelCh.channelId = sub->channelId;  // 设置通道 ID
      eDescr.kernelCh.pTimer = start;  // 设置计时器为开始时间
      ncclProfiler->startEvent(sub->profilerContext, &sub->kernelEventHandle, &eDescr);  // 启动内核通道事件
    }
  }
  return ncclSuccess;  // 返回成功状态
}

// 停止 Kernel 通道事件函数
// 参数说明：
//   args - 代理参数指针
//   s - 子操作索引
//   start - 开始时间戳
//   stop - 停止时间戳
ncclResult_t ncclProfilerStopKernelChEvent(struct ncclProxyArgs* args, int s, uint64_t stop) {
  if (__builtin_expect(ncclProfiler != NULL, 0)) {  // 如果性能分析器接口有效
    struct ncclProxySubArgs* sub = &args->subs[s];  // 获取子操作指针
    if (sub->kernelEventHandle) {  // 如果内核事件句柄有效
      ncclProfilerEventStateArgs_t a = { };  // 初始化事件状态参数结构体
      a.kernelCh.pTimer = stop;  // 设置计时器为停止时间
      ncclProfiler->recordEventState(sub->kernelEventHandle, ncclProfilerKernelChStop, &a);  // 记录内核通道停止事件状态
      ncclProfiler->stopEvent(sub->kernelEventHandle);  // 停止内核通道事件
    }
  }
  return ncclSuccess;  // 返回成功状态
}

// 记录代理操作事件状态函数
// 参数说明：
//   s - 子操作索引
//   args - 代理参数指针
//   eState - 事件状态
ncclResult_t ncclProfilerRecordProxyOpEventState(int s, struct ncclProxyArgs* args, ncclProfilerEventState_t eState) {
  TIME_START_EVENT(proxyOpRecord);  // 开始代理操作记录计时
  struct ncclProxySubArgs* sub = &args->subs[s];  // 获取子操作指针
  if (__builtin_expect(ncclProfiler != NULL, 0) && sub->opEventHandle) {  // 如果性能分析器有效且操作事件句柄有效
    ncclProfilerEventStateArgs_t a = { };  // 初始化事件状态参数结构体
    ncclProfiler->recordEventState(sub->opEventHandle, eState, &a);  // 记录事件状态到操作事件句柄
  }
  TIME_STOP_EVENT(proxyOpRecord);  // 停止代理操作记录计时
  return ncclSuccess;  // 返回成功状态
}

// 记录代理步骤事件状态函数
// 参数说明：
//   s - 子操作索引
//   args - 代理参数指针
//   stepId - 步骤 ID
//   eState - 事件状态
ncclResult_t ncclProfilerRecordProxyStepEventState(int s, struct ncclProxyArgs* args, int stepId, ncclProfilerEventState_t eState) {
  TIME_START_EVENT(proxyStepRecord);  // 开始代理步骤记录计时
  struct ncclProxySubArgs* sub = &args->subs[s];  // 获取子操作指针
  if (__builtin_expect(ncclProfiler != NULL, 0)) {  // 如果性能分析器接口有效
    int step_ = DIVUP(stepId, args->sliceSteps);  // 计算当前步骤号（向上取整）
    if (sub->pHandles[step_%NCCL_STEPS].stepEventHandle) {  // 如果步骤事件句柄有效
      ncclProfilerEventStateArgs_t a = { };  // 初始化事件状态参数结构体
      a.proxyStep.transSize = sub->transSize;  // 设置传输大小
      ncclProfiler->recordEventState(sub->pHandles[step_%NCCL_STEPS].stepEventHandle, eState, &a);  // 记录事件状态到步骤事件句柄
    }
  }
  TIME_STOP_EVENT(proxyStepRecord);  // 停止代理步骤记录计时
  return ncclSuccess;  // 返回成功状态
}

// 记录代理控制事件状态函数
// 参数说明：
//   eHandle - 事件句柄
//   appended - 是否追加
//   eState - 事件状态
ncclResult_t ncclProfilerRecordProxyCtrlEventState(void* eHandle, int appended, ncclProfilerEventState_t eState) {
  TIME_START_EVENT(proxyCtrlRecord);  // 开始代理控制记录计时
  if (__builtin_expect(ncclProfiler != NULL, 0) && eHandle && __atomic_load_n(&ncclProfilerEventMask, __ATOMIC_RELAXED) & ncclProfileProxyCtrl) {  // 如果性能分析器有效、事件句柄有效且代理控制掩码激活
    ncclProfilerEventStateArgs_t args = { };  // 初始化事件状态参数结构体
      args.proxyCtrl.appendedProxyOps = appended;  // 设置追加的代理操作标志
      ncclProfiler->recordEventState(eHandle, eState, &args);  // 记录事件状态
  }
  TIME_STOP_EVENT(proxyCtrlRecord);  // 停止代理控制记录计时
  return ncclSuccess;  // 返回成功状态
}

// 添加 PID 到代理操作函数
// 参数：op - 代理操作指针
ncclResult_t ncclProfilerAddPidToProxyOp(struct ncclProxyOp* op) {
  op->pid = pid;  // 设置进程 ID
  return ncclSuccess;  // 返回成功状态
}

// 定义性能分析器连接互斥锁，用于线程安全地连接性能分析器
static std::mutex proxyProfilerConnectMutex;

// 性能分析器连接函数
// 参数：comm - NCCL 通信上下文
//   op - 代理操作指针
static ncclResult_t proxyProfilerConnect(struct ncclComm* comm, struct ncclProxyOp* op) {
  ncclResult_t ret = ncclSuccess;  // 初始化返回值为成功
  // 使用 RAII 方式加锁，保护连接过程
  std::lock_guard<std::mutex> lock(proxyProfilerConnectMutex);  // 自动加锁，函数退出时解锁
  if (comm->profiler.initialized) goto exit;  // 如果性能分析器已初始化，跳转到退出
  // 遍历所有通道，建立发送和接收连接
  for (int c = 0; c < MAX_CHANNELS; c++) {  // 遍历最大通道数
    NCCLCHECKGOTO(ncclProxyConnect(comm, TRANSPORT_PROFILER, 0, comm->rank, &comm->profiler.sendProxyConn[c]), ret, exit);  // 建立发送代理连接
    NCCLCHECKGOTO(ncclProxyCallBlocking(comm, &comm->profiler.sendProxyConn[c], ncclProxyMsgConnect, NULL, 0, NULL, 0), ret, exit);  // 通过发送连接发送代理消息连接
    NCCLCHECKGOTO(ncclProxyConnect(comm, TRANSPORT_PROFILER, 0, comm->rank, &comm->profiler.recvProxyConn[c]), ret, exit);  // 建立接收代理连接
  }
  comm->profiler.initialized = true;  // 标记性能分析器已初始化

exit:  // 正常退出标签
  return ret;  // 返回结果
}

// 判断是否需要性能分析器的函数
// 参数：comm - NCCL 通信上下文
//   op - 代理操作指针
bool ncclProfilerNeedsProxy(struct ncclComm* comm, struct ncclProxyOp* op) {
  bool enabled = ncclProfilerPluginLoaded() && (op->eActivationMask & ncclProfileKernelCh);  // 检查性能分析器是否已加载且操作掩码包含内核通道
  if (enabled && !comm->profiler.initialized)  // 如果启用但未初始化
    (void)proxyProfilerConnect(comm, op);  // 调用性能分析器连接函数
  return enabled;  // 返回是否启用
}

// 判断性能分析器插件是否已加载的函数
static bool ncclProfilerPluginLoaded(void) {
  return (__builtin_expect(ncclProfiler != NULL, 0));  // 返回性能分析器接口是否有效（预期不为空）
}

// 性能分析器回调函数
// 参数说明：
//   eHandle - 事件句柄
//   type - 事件类型
//   pHandle - 代理句柄
//   pluginId - 插件 ID
//   extData - 扩展数据
ncclResult_t ncclProfilerCallback(void** eHandle, int type, void* pHandle, int64_t pluginId, void* extData) {
  if (__builtin_expect(ncclProfiler != NULL, 0)) {  // 如果性能分析器接口有效
    if (type == ncclProfilerNetEventStart) {  // 如果事件类型为网络插件启动
      // start
      struct ncclProxyEventHandle* p = (struct ncclProxyEventHandle*)pHandle;  // 转换为代理事件句柄指针
      struct ncclProxySubArgs* sub = p->subArgPtr;  // 获取子操作参数指针
      if (sub->eActivationMask & ncclProfileNetPlugin) {  // 如果子操作激活掩码包含网络插件类型
        ncclProfilerEventDescr_t eDescr = { 0 };  // 初始化事件描述符结构体
        eDescr.type = ncclProfileNetPlugin;  // 设置事件类型为网络插件
        eDescr.parentObj = p->stepEventHandle;  // 设置父对象为步骤事件句柄
        eDescr.rank = sub->rank;  // 设置 rank
        eDescr.netPlugin.id = pluginId;  // 设置插件 ID
        eDescr.netPlugin.data = extData;  // 设置扩展数据
        ncclProfiler->startEvent(sub->profilerContext, eHandle, &eDescr);  // 启动网络插件事件
      }
    } else if (type == ncclProfilerNetEventStop) {  // stop
      ncclProfiler->stopEvent(*eHandle);  // 停止事件
    } else if (type == ncclProfilerNetEventUpdate) {  // update
      // update
      ncclProfilerEventStateArgs_t args = { };  // 初始化事件状态参数结构体
      args.netPlugin.data = extData;  // 设置扩展数据
      ncclProfiler->recordEventState(*eHandle, ncclProfilerNetPluginUpdate, &args);  // 记录网络插件更新事件状态
    } else {  // update and stop
      ncclProfilerEventStateArgs_t args = { };  // 初始化事件状态参数结构体
      args.netPlugin.data = extData;  // 设置扩展数据
      ncclProfiler->recordEventState(*eHandle, ncclProfilerNetPluginUpdate, &args);  // 记录网络插件更新事件状态
      ncclProfiler->stopEvent(*eHandle);  // 停止事件
    }
  }
  return ncclSuccess;  // 返回成功状态
}
