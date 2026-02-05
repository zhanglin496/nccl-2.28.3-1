/*************************************************************************
 * Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
 * Copyright (c) 2023, Meta Platforms, Inc. and affiliates.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// 系统标准头文件
#include <errno.h>      // 错误码定义
#include <stdlib.h>     // 标准库函数，如环境变量获取
#include <mutex>        // C++互斥锁，用于线程安全

// NCCL内部头文件
#include "checks.h"     // NCCL错误检查宏
#include "debug.h"      // 调试和日志输出宏
#include "tuner.h"      // Tuner相关结构定义
#include "plugin.h"     // 插件加载和管理接口

//================================================================================
// 外部符号声明：这些函数在不同版本的tuner插件库中定义
// 用于从动态库中获取对应版本的tuner接口结构体
//================================================================================
extern ncclTuner_t* getNcclTuner_v2(void* lib);  // v2版本tuner获取函数
extern ncclTuner_t* getNcclTuner_v3(void* lib);  // v3版本tuner获取函数
extern ncclTuner_t* getNcclTuner_v4(void* lib);  // v4版本tuner获取函数
extern ncclTuner_t* getNcclTuner_v5(void* lib);  // v5版本tuner获取函数

//================================================================================
// 全局状态变量：用于管理tuner插件的生命周期
//================================================================================
static std::mutex tunerPluginMutex;      // 互斥锁，保证多线程环境下插件加载/卸载的线程安全
static int tunerPluginRefCount;          // 引用计数，记录有多少个comm正在使用tuner插件
static void* tunerPluginLib = nullptr;   // tuner插件的动态库句柄
static ncclTuner_t* tunerSymbol = nullptr; // tuner插件接口函数表指针

//================================================================================
// Tuner插件加载状态枚举
//================================================================================
enum {
  tunerPluginLoadFailed  = -1,  // 加载失败状态：曾经尝试加载但失败了，后续不会再尝试
  tunerPluginLoadReady   =  0,  // 就绪状态：尚未尝试加载，可以进行加载尝试
  tunerPluginLoadSuccess =  1,  // 加载成功状态：插件已成功加载并可以使用
};

//================================================================================
// 最大插件加载尝试次数
// 用于限制在遇到暂时性错误时的重试次数
//================================================================================
#define MAX_PLUGIN_LOAD 4

//================================================================================
// 当前插件加载状态
// 初始值为tunerPluginLoadReady（就绪状态）
//================================================================================
static int status = tunerPluginLoadReady;

//================================================================================
// ncclTunerPluginLoad - 加载Tuner插件
//
// 功能说明：
//   1. 为指定的comm加载tuner插件
//   2. 支持通过环境变量NCCL_TUNER_PLUGIN指定tuner插件路径
//   3. 如果未指定外部tuner，会尝试从net插件中获取tuner功能
//   4. 自动适配不同版本的tuner接口（v5→v4→v3→v2降级尝试）
//   5. 使用引用计数管理插件生命周期，多个comm可以共享同一个tuner插件
//
// 参数：
//   comm - NCCL通信器指针，加载成功后会将tuner接口赋值给comm->tuner
//
// 返回值：
//   ncclSuccess - 总是返回成功（即使加载失败也不会报错，只是不使用tuner）
//================================================================================
ncclResult_t ncclTunerPluginLoad(struct ncclComm* comm) {
  const char* tunerName;

  // 默认情况下，如果无法加载插件tuner，将comm->tuner初始化为nullptr
  // 这样后续代码可以通过检查comm->tuner是否为nullptr来判断是否有tuner可用
  comm->tuner = nullptr;

  // 快速路径：如果之前已经尝试过加载并失败了，直接返回成功
  // 这样避免每次创建comm时都重复尝试加载已知失败的插件
  if (tunerPluginLoadFailed == status) {
    return ncclSuccess;
  }

  // 加锁：保护后续的全局状态访问
  // 使用RAII风格的lock_guard，出作用域时自动解锁
  std::lock_guard<std::mutex> lock(tunerPluginMutex);

  // 双重检查：可能在等待锁的过程中，状态已经被其他线程改变了
  // 如果状态已经变成失败，直接退出
  if (tunerPluginLoadFailed == status) {
    goto exit;
  }

  // 快速路径：如果插件已经成功加载过
  // 直接复用已加载的tuner符号，并增加引用计数
  if (tunerPluginLoadSuccess == status) {
    comm->tuner = tunerSymbol;       // 将tuner接口赋值给当前comm
    //应用计数++，处理多线程的情况
    ++tunerPluginRefCount;           // 引用计数增加，表示有一个新的comm在使用此插件
    goto exit;                       // 跳转到退出标签
  }

  //================================================================================
  // 第一次尝试加载tuner插件
  //================================================================================

//指定了环境变量
  // 检查是否设置了NCCL_TUNER_PLUGIN环境变量
  // 用户可以通过此变量指定自定义的tuner插件库路径
  if ((tunerName = ncclGetEnv("NCCL_TUNER_PLUGIN")) != nullptr) {
    INFO(NCCL_ENV|NCCL_TUNING, "NCCL_TUNER_PLUGIN set by environment to %s", tunerName);

    // 如果环境变量设置为"none"，表示用户明确不想使用tuner插件
    // 跳转到fail标签，将状态设置为加载失败
    if (strcasecmp(tunerName, "none") == 0)
      goto fail;
  }

  // 尝试打开tuner插件动态库
  // ncclOpenTunerPluginLib会：
  //   1. 如果tunerName非空，直接加载指定的库
  //   2. 如果tunerName为空，尝试从默认路径加载tuner插件
  tunerPluginLib = ncclOpenTunerPluginLib(tunerName);

  // 如果指定的tuner插件加载失败
  if (nullptr == tunerPluginLib) {
    //把net插件赋值给tunner？？
    // 尝试从net插件库中获取tuner功能
    // 某些net插件（如libnccl-net.so）内部也可能包含tuner功能
    // 这是一个fallback机制，提高tuner的可用性
    tunerPluginLib = ncclGetNetPluginLib(ncclPluginTypeTuner);
    if (nullptr == tunerPluginLib) {
      goto fail;  // net插件也没有tuner功能，加载失败
    }
    tunerName = nullptr;  // 清空tunerName，表示使用的是net插件内置的tuner
  } else if (ncclPluginLibPaths[ncclPluginTypeTuner]) {
    // 如果成功加载了独立的tuner插件库，记录插件路径用于日志
    tunerName = ncclPluginLibPaths[ncclPluginTypeTuner];
  }

//按版本高低解析符号
  //================================================================================
  // 版本适配：从高到低尝试解析不同版本的tuner符号
  // NCCL的tuner接口有多个版本（v2/v3/v4/v5），新版本增加更多功能
  // 这里从最新版本开始尝试，如果失败则降级到旧版本
  //================================================================================

  // 尝试获取v5版本的tuner接口
  tunerSymbol = getNcclTuner_v5(tunerPluginLib);
  if (tunerSymbol == NULL) {
    // v5不存在，尝试v4版本
    tunerSymbol = getNcclTuner_v4(tunerPluginLib);
  }
  if (tunerSymbol == NULL) {
    // v4不存在，尝试v3版本
    tunerSymbol = getNcclTuner_v3(tunerPluginLib);
  }
  if (tunerSymbol == NULL) {
    // v3不存在，尝试v2版本（最低版本）
    tunerSymbol = getNcclTuner_v2(tunerPluginLib);
  }

  // 如果所有版本都无法解析，说明这个库不是有效的tuner插件
  if (tunerSymbol == NULL) {
    if (tunerName) INFO(NCCL_INIT|NCCL_TUNING, "External tuner plugin %s is unsupported", tunerName);
    goto fail;
  }

  // 成功加载并解析tuner插件，输出日志
  if (tunerName) INFO(NCCL_INIT|NCCL_TUNING, "Successfully loaded external tuner plugin %s", tunerName);

//赋值给通信器
  // 将tuner接口指针赋值给当前comm
  comm->tuner = tunerSymbol;
//引用计数++
  ++tunerPluginRefCount;           // 增加引用计数
  //加载成功
  status = tunerPluginLoadSuccess; // 将全局状态设置为成功
  comm->tunerPluginLoaded = 1;     // 标记当前comm已加载tuner插件

exit:
  return ncclSuccess;  // 总是返回成功

fail:
  //================================================================================
  // 加载失败处理路径
  //================================================================================
  // 如果已经打开了动态库（但符号解析失败），需要关闭它
  if (tunerPluginLib)
    NCCLCHECK(ncclClosePluginLib(tunerPluginLib, ncclPluginTypeTuner));

  tunerPluginLib = nullptr;        // 清空库句柄
  status = tunerPluginLoadFailed;  // 将状态设置为失败，后续不会再尝试加载
  goto exit;                       // 跳转到退出
}

//================================================================================
// ncclTunerPluginUnload - 卸载Tuner插件
//
// 功能说明：
//   1. 当comm销毁时调用，减少tuner插件的引用计数
//   2. 当引用计数降为0时，关闭插件动态库并释放资源
//   3. 支持多个comm共享同一个tuner插件的场景
//
// 参数：
//   comm - NCCL通信器指针
//
// 返回值：
//   ncclSuccess - 总是返回成功
//================================================================================
ncclResult_t ncclTunerPluginUnload(struct ncclComm* comm) {
  // 加锁：保护引用计数和插件句柄的访问
  std::lock_guard<std::mutex> lock(tunerPluginMutex);

  // 只有当当前comm确实加载了tuner插件，且引用计数减到0时，才真正卸载插件
  // comm->tunerPluginLoaded = 1 表示这个comm在load时成功加载了插件
  // --tunerPluginRefCount 是先减1，再判断是否为0
  if (comm->tunerPluginLoaded && 0 == (--tunerPluginRefCount)) {
    // 输出卸载日志，记录tuner插件名称
    INFO(NCCL_INIT|NCCL_TUNING, "TUNER/Plugin: Closing tuner: '%s'", tunerSymbol->name);

    // 关闭tuner插件动态库
    NCCLCHECK(ncclClosePluginLib(tunerPluginLib, ncclPluginTypeTuner));

    // 清空全局状态
    tunerPluginLib = nullptr;   // 清空库句柄
    tunerSymbol = nullptr;      // 清空tuner接口指针
    comm->tuner = nullptr;      // 清空comm的tuner指针

    // 将状态重置为就绪，允许未来重新加载插件
    status = tunerPluginLoadReady;

    // 标记当前comm已卸载tuner插件
    comm->tunerPluginLoaded = 0;
  }

  return ncclSuccess;
}
