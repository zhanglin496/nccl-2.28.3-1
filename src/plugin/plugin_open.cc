/*************************************************************************
 * Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

/*
 * plugin_open.cc - NCCL 插件动态库加载模块
 *
 * 本文件负责动态加载和管理 NCCL 的外部插件库，包括：
 * 1. NET 插件 - 网络传输插件（如 libnccl-net.so）
 * 2. TUNER 插件 - 算法调优插件（如 libnccl-tuner.so）
 * 3. PROFILER 插件 - 性能分析插件（如 libnccl-profiler.so）
 *
 * 插件加载策略：
 * - 首先尝试加载用户指定的库
 * - 然后尝试加载默认名称的库（如 libnccl-net.so）
 * - 对于 NET 插件，支持版本后缀（如 libnccl-net-ib.so）
 * - 如果加载失败，NET/PROFILER 会报错，TUNER 会使用内置实现
 */

#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <link.h>     // 用于 dlinfo 获取库路径
#include <dlfcn.h>    // 动态库加载接口

#include "debug.h"
#include "plugin.h"

#define MAX_STR_LEN 255  // 最大字符串长度

// ============================================================================
// 全局变量定义
// ============================================================================

#define NUM_LIBS 3  // 插件类型数量：NET, TUNER, PROFILER

// 用户配置的库名称（如 "libnccl-net.so" 或 "libnccl-net-ib.so"）
static char* libNames[NUM_LIBS];

// 插件库的绝对路径（通过 dlinfo 获取）
char* ncclPluginLibPaths[NUM_LIBS];

// 动态库句柄（dlopen 返回值）
static void *libHandles[NUM_LIBS];

// 插件类型名称（用于日志输出）
static const char *pluginNames[NUM_LIBS] = { "NET", "TUNER", "PROFILER" };

// 插件库默认前缀
static const char *pluginPrefix[NUM_LIBS] = { "libnccl-net", "libnccl-tuner", "libnccl-profiler" };

// 插件加载失败时的回退消息
// "" = 无回退，必须加载
// "Using internal tuner plugin." = 使用内置实现
static const char *pluginFallback[NUM_LIBS] = { "", "Using internal tuner plugin.", "" };

// 插件初始化子系统掩码
static const unsigned long subsys[NUM_LIBS] = {
  NCCL_INIT|NCCL_NET,    // NET 插件：NET 子系统
  NCCL_INIT|NCCL_TUNING,  // TUNER 插件：TUNING 子系统
  NCCL_INIT              // PROFILER 插件：INIT 子系统
};

// ============================================================================
// tryOpenLib - 尝试打开指定的动态库
// ============================================================================
// 参数：
//   name    - 库名称（可以是相对路径、绝对路径或 "STATIC_PLUGIN"）
//   err     - 输出：错误码（0: 成功, ENOENT: 文件不存在）
//   errStr  - 输出：错误信息（来自 dlerror）
//
// 返回：动态库句柄（dlopen 返回值），失败返回 nullptr
//
// 特殊处理：
// - name = "STATIC_PLUGIN" → 视为 nullptr（使用静态链接）
// - 记录 ENOENT 错误码（用于区分"文件不存在"和其他错误）
static void* tryOpenLib(char* name, int* err, char* errStr) {
  *err = 0;

  // 空名称或 nullptr：不加载任何库
  if (nullptr == name || strlen(name) == 0) {
    return nullptr;
  }

  // 检查是否指定了静态插件
  if (strncasecmp(name, "STATIC_PLUGIN", strlen(name)) == 0) {
    name = nullptr;  // 静态插件，返回 nullptr
  }

  // 使用 RTLD_NOW | RTLD_LOCAL 加载库
  // RTLD_NOW: 立即解析所有符号（而不是延迟解析）
  // RTLD_LOCAL: 符号不对外可见（避免符号冲突）
  void *handle = dlopen(name, RTLD_NOW | RTLD_LOCAL);

  if (nullptr == handle) {
    // 加载失败，记录错误信息
    strncpy(errStr, dlerror(), MAX_STR_LEN);
    errStr[MAX_STR_LEN] = '\0';

    // "handle" and "name" won't be NULL at the same time.
    // coverity[var_deref_model]
    // 判断是否是"文件不存在"错误
    if (strstr(errStr, name) && strstr(errStr, "No such file or directory")) {
      *err = ENOENT;
    }
  }
  return handle;
}

// ============================================================================
// appendNameToList - 将库名追加到列表字符串
// ============================================================================
// 参数：
//   nameList - 名称列表缓冲区
//   leftChars - 剩余字符数
//   name      - 要追加的名称
//
// 功能：将 name 追加到 nameList，并更新剩余字符数
static void appendNameToList(char* nameList, int *leftChars, char* name) {
  snprintf(nameList + PATH_MAX - *leftChars, *leftChars, " %s", name);
  *leftChars -= strlen(name) + 1;
}

// ============================================================================
// getLibPath - 获取动态库的绝对路径
// ============================================================================
// 参数：
//   handle - 动态库句柄
//
// 返回：库的绝对路径字符串（需要调用者释放），失败返回 nullptr
//
// 功能：通过 dlinfo 获取动态库的实际路径
static char* getLibPath(void* handle) {
  struct link_map* lm;

  // 查询动态库的链接映射信息
  if (dlinfo(handle, RTLD_DI_LINKMAP, &lm) != 0)
    return nullptr;
  else
    return strdup(lm->l_name);  // 复制路径字符串
}

// ============================================================================
// openPluginLib - 打开指定类型的插件库
// ============================================================================
// 参数：
//   type    - 插件类型（ncclPluginTypeNet/Tuner/Profiler）
//   libName - 用户指定的库名称（可选）
//
// 返回：动态库句柄，失败返回 nullptr
//
// 加载策略（按优先级尝试）：
// 1. 用户指定的完整路径（如 "/path/to/libnccl-net.so"）
// 2. 默认前缀 + .so（如 "libnccl-net.so"）
// 3. 默认前缀 + 用户基础名称 + .so（如 "libnccl-net-ib.so"）
//
// 环境变量：NCCL_NET_PLUGIN_FILE, NCCL_TUNER_PLUGIN_FILE, NCCL_PROFILER_PLUGIN_FILE
static void* openPluginLib(enum ncclPluginType type, const char* libName) {
  int openErr, len = PATH_MAX;
  char libName_[MAX_STR_LEN] = { 0 };     // 处理后的库名称
  char openErrStr[MAX_STR_LEN + 1] = { 0 };  // 错误信息
  char eNoEntNameList[PATH_MAX] = { 0 };    // 所有尝试失败的库列表

//没有指定插件名称，则尝试加载内部自定义前缀插件
  if (libName && strlen(libName)) {
    // 用户指定了库名称
    snprintf(libName_, MAX_STR_LEN, "%s", libName);
  } else {
    //拼接库名称，默认加载
    snprintf(libName_, MAX_STR_LEN, "%s.so", pluginPrefix[type]);
  }

//第一次尝试加载
  libHandles[type] = tryOpenLib(libName_, &openErr, openErrStr);

  //加载成功
  if (libHandles[type]) {
    libNames[type] = strdup(libName_);  // 保存库名
    //获取库的绝对路径
    ncclPluginLibPaths[type] = getLibPath(libHandles[type]);
    return libHandles[type];
  }

  //加载失败
  if (openErr == ENOENT) {
    // 文件不存在，记录到失败列表（稍后统一显示）
    appendNameToList(eNoEntNameList, &len, libName_);
  } else {
    // 其他错误（如依赖缺失、版本不匹配），立即显示错误信息
    INFO(subsys[type], "%s/Plugin: %s: %s", pluginNames[type], libName_, openErrStr);
  }

  //这里代码的目的是运行用户配置提供基础名称，自动构建完整库名，比如libName="ib"
  //则构建为libnccl-net-ib.so
  // libName can't be a relative or absolute path (start with '.' or contain any '/'). It can't be a library name either (start with 'lib' or end with '.so')
  if (libName && strlen(libName) && strchr(libName, '/') == nullptr &&    // 不包含路径分隔符
      (strncmp(libName, "lib", strlen("lib")) ||                       // 不以 "lib" 开头
       strlen(libName) < strlen(".so") ||                                  // 不以 ".so" 结尾
       strncmp(libName + strlen(libName) - strlen(".so"), ".so", strlen(".so")))) {  // 不以 ".so" 结尾
    // 构建完整库名：前缀 + 基础名称 + .so
    snprintf(libName_, MAX_STR_LEN, "%s-%s.so", pluginPrefix[type], libName);

    libHandles[type] = tryOpenLib(libName_, &openErr, openErrStr);
    if (libHandles[type]) {
      libNames[type] = strdup(libName_);
      ncclPluginLibPaths[type] = getLibPath(libHandles[type]);
      return libHandles[type];
    }
    if (openErr == ENOENT) {
      appendNameToList(eNoEntNameList, &len, libName_);
    } else {
      INFO(subsys[type], "%s/Plugin: %s: %s", pluginNames[type], libName_, openErrStr);
    }
  }

  // ========================================================================
  // 所有尝试都失败，显示错误或回退消息
  // ========================================================================
  if (strlen(eNoEntNameList)) {
    // 有尝试的库文件都不存在
    INFO(subsys[type], "%s/Plugin: Could not find:%s%s%s", pluginNames[type], eNoEntNameList,
         (strlen(pluginFallback[type]) > 0 ? ". " : ""), pluginFallback[type]);
  } else if (strlen(pluginFallback[type])) {
    // 有回退方案（TUNER 插件）
    //tunner插件，使用内部调优插件
    INFO(subsys[type], "%s/Plugin: %s", pluginNames[type], pluginFallback[type]);
  }

  return nullptr;  // 所有尝试都失败
}

// ============================================================================
// ncclOpenNetPluginLib - 打开网络插件库
// ============================================================================
// 参数：
//   name - 插件库名称（可选）
//
// 返回：动态库句柄，失败返回 nullptr
//
// 环境变量：NCCL_NET_PLUGIN_FILE
void* ncclOpenNetPluginLib(const char* name) {
  return openPluginLib(ncclPluginTypeNet, name);
}

// ============================================================================
// ncclOpenTunerPluginLib - 打开调优插件库
// ============================================================================
// 参数：
//   name - 插件库名称（可选）
//
// 返回：动态库句柄，失败返回 nullptr
//
// 环境变量：NCCL_TUNER_PLUGIN_FILE
//
// 注意：TUNER 插件有内置实现，加载失败时会回退到内置实现
void* ncclOpenTunerPluginLib(const char* name) {
  return openPluginLib(ncclPluginTypeTuner, name);
}

// ============================================================================
// ncclOpenProfilerPluginLib - 打开性能分析插件库
// ============================================================================
// 参数：
//   name - 插件库名称（可选）
//
// 返回：动态库句柄，失败返回 nullptr
//
// 环境变量：NCCL_PROFILER_PLUGIN_FILE
void* ncclOpenProfilerPluginLib(const char* name) {
  return openPluginLib(ncclPluginTypeProfiler, name);
}

// ============================================================================
// ncclGetNetPluginLib - 获取网络插件句柄
// ============================================================================
// 参数：
//   type - 插件类型
//
// 返回：动态库句柄
//
// 功能：
// - 增加网络插件的引用计数
// - 如果库已加载，重新 dlopen 以增加引用
// - 用于多个 communicator 共享同一个网络插件
void* ncclGetNetPluginLib(enum ncclPluginType type) {
  if (libNames[ncclPluginTypeNet]) {
    // increment the reference counter of the net library
    // 增加网络库的引用计数
    libNames[type] = strdup(libNames[ncclPluginTypeNet]);
    ncclPluginLibPaths[type] = strdup(ncclPluginLibPaths[ncclPluginTypeNet]);

    // 重新打开库以增加引用计数
    libHandles[type] = dlopen(libNames[ncclPluginTypeNet], RTLD_NOW | RTLD_LOCAL);
  }
  return libHandles[type];
}

// ============================================================================
// ncclClosePluginLib - 关闭插件库
// ============================================================================
// 参数：
//   handle - 动态库句柄
//   type   - 插件类型
//
// 返回：ncclSuccess
//
// 功能：
// - 关闭动态库并释放相关资源
// - 清空库名称、路径和句柄
ncclResult_t ncclClosePluginLib(void* handle, enum ncclPluginType type) {
  if (handle && libHandles[type] == handle) {
    dlclose(handle);                       // 关闭动态库
    libHandles[type] = nullptr;            // 清空句柄
    free(ncclPluginLibPaths[type]);         // 释放路径字符串
    ncclPluginLibPaths[type] = nullptr;
    free(libNames[type]);                   // 释放库名称字符串
    libNames[type] = nullptr;
  }
  return ncclSuccess;
}
