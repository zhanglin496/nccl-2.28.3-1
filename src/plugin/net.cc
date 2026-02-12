/*************************************************************************
 * Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved.
 * 版权所有 (c) 2022-2023，NVIDIA 公司。保留所有权利。
 *
 * See LICENSE.txt for license information
 * 详见 LICENSE.txt 获取许可证信息
 ************************************************************************/

// 包含网络相关头文件，定义网络接口和操作
#include "net.h"
// 包含引导程序相关头文件，用于节点内通信和同步
#include "bootstrap.h"
// 包含检查相关头文件，提供各种检查和验证功能
#include "checks.h"
// 包含插件相关头文件，定义插件接口和加载机制
#include "plugin.h"
// 包含 NCCL 网络头文件，定义 NCCL 网络插件接口
#include "nccl_net.h"

// 包含字符串处理头文件，提供字符串操作函数
#include <string.h>
// 包含错误码头文件，定义错误码和错误处理宏
#include <errno.h>
// 包含互斥锁头文件，提供线程同步机制
#include <mutex>
// 注释掉的系统头文件：sys/types.h（系统类型定义）
//#include <sys/types.h>
// 注释掉的系统头文件：sys/stat.h（文件状态信息）
//#include <sys/stat.h>
// 注释掉的系统头文件：unistd.h（UNIX 标准符号常量和类型）
//#include <unistd.h>

// 定义函数指针类型：从网络插件库获取 ncclNet_t 接口
// 参数：void* netPluginLib - 插件库句柄
// 返回：ncclNet_t* - 网络接口指针
typedef ncclNet_t* getNcclNet_t(void* netPluginLib);
// 定义函数指针类型：从网络插件库获取 ncclCollNet_t 接口（集合网络接口）
// 参数：void* netPluginLib - 插件库句柄
// 返回：ncclCollNet_t* - 集合网络接口指针
typedef ncclCollNet_t* getNcclCollNet_t(void* netPluginLib);

// 声明外部函数：获取版本 6 的 ncclNet 接口
extern getNcclNet_t getNcclNet_v6;
// 声明外部函数：获取版本 7 的 ncclNet 接口
extern getNcclNet_t getNcclNet_v7;
// 声明外部函数：获取版本 8 的 ncclNet 接口
extern getNcclNet_t getNcclNet_v8;
// 声明外部函数：获取版本 9 的 ncclNet 接口
extern getNcclNet_t getNcclNet_v9;
// 声明外部函数：获取版本 10 的 ncclNet 接口
extern getNcclNet_t getNcclNet_v10;
// 声明外部函数：获取版本 11 的 ncclNet 接口
extern getNcclNet_t getNcclNet_v11;
// 声明外部函数：获取版本 6 的 ncclCollNet 接口
extern getNcclCollNet_t getNcclCollNet_v6;
// 声明外部函数：获取版本 7 的 ncclCollNet 接口
extern getNcclCollNet_t getNcclCollNet_v7;
// 声明外部函数：获取版本 8 的 ncclCollNet 接口
extern getNcclCollNet_t getNcclCollNet_v8;
// 声明外部函数：获取版本 9 的 ncclCollNet 接口
extern getNcclCollNet_t getNcclCollNet_v9;
// 声明外部函数：获取版本 10 的 ncclCollNet 接口
extern getNcclCollNet_t getNcclCollNet_v10;
// 声明外部函数：获取版本 11 的 ncclCollNet 接口
extern getNcclCollNet_t getNcclCollNet_v11;

// 定义参数：NetPluginRefCount，环境变量名 NET_PLUGIN_REF_COUNT，默认值为 0
// 用于控制插件引用计数的行为
NCCL_PARAM(NetPluginRefCount, "NET_PLUGIN_REF_COUNT", 0);
// 定义 NCCL 网络版本数量为 6（版本 6-11）
#define NCCL_NET_VERSION_COUNT 6
// 定义网络版本号数组，从高到低排序（11, 10, 9, 8, 7, 6）
// 用于按版本顺序尝试加载插件
int ncclNetVersion[NCCL_NET_VERSION_COUNT] = {11, 10, 9, 8, 7, 6};
// 定义获取 ncclNet 接口的函数指针数组，与版本号数组对应
getNcclNet_t* getNcclNet[NCCL_NET_VERSION_COUNT] = {getNcclNet_v11, getNcclNet_v10, getNcclNet_v9, getNcclNet_v8, getNcclNet_v7, getNcclNet_v6};
// 定义获取 ncclCollNet 接口的函数指针数组，与版本号数组对应
getNcclCollNet_t* getNcclCollNet[NCCL_NET_VERSION_COUNT] = {getNcclCollNet_v11, getNcclCollNet_v10, getNcclCollNet_v9, getNcclCollNet_v8, getNcclCollNet_v7, getNcclCollNet_v6};

//2个自定义内部插件
// NCCL 有 2 个内置的内部插件：IB 和 Socket

// 定义内部插件数量为 2
#define NCCL_NET_NUM_INTERNAL_PLUGINS 2

// 定义网络插件状态枚举
typedef enum ncclNetPluginState {
  ncclNetPluginStateDisabled        = -2,       // Plugin library failed to initialize
  // 插件库初始化失败，状态为 -2

  ncclNetPluginStateLoadFailed      = -1,       // Plugin library failed to load
  // 插件库加载失败，状态为 -1

  ncclNetPluginStateLoadReady       = 0,        // Plugin library is ready to be loaded
  // 插件库已准备好加载，状态为 0

  ncclNetPluginStateInitReady       = 1,        // Plugin library is loaded and ready to be initialized
  // 插件库已加载并准备好初始化，状态为 1

  ncclNetPluginStateEnabled         = 2,        // Plugin library is loaded and initialized
  // 插件库已加载并初始化完成，状态为 2
} ncclNetPluginState_t;

// 定义最大字符串长度为 255
#define MAX_STR_LEN 255

// 定义网络插件库结构体
typedef struct netPluginLib {
  char name[MAX_STR_LEN];                       // Name of the plugin library
  // 插件库名称，最大长度为 MAX_STR_LEN

  void* dlHandle;                               // Handle to the plugin library
  // 动态链接库句柄，用于加载和卸载插件库

  ncclNet_t* ncclNet;                           // Pointer to the ncclNet_t structure
  // 指向 ncclNet_t 结构体的指针，包含网络接口函数表

  int ncclNetVer;                               // Version of the nccl net plugin
  // NCCL 网络插件的版本号（6-11）

  ncclCollNet_t* ncclCollNet;                   // Pointer to the ncclCollNet_t structure
  // 指向 ncclCollNet_t 结构体的指针，包含集合网络接口函数表

  ncclNetPluginState_t ncclNetPluginState;      // State of the nccl net plugin
  // NCCL 网络插件的状态

  //CollNet = Collective Network
  // 集合网络（CollNet），支持 SHARP 等集合通信加速技术

  ncclNetPluginState_t ncclCollNetPluginState;  // State of the nccl coll net plugin
  // NCCL 集合网络插件的状态

  int ncclNetPluginRefCount;                    // Reference count for the nccl net plugin
  // NCCL 网络插件的引用计数，用于多线程或多个通信上下文共享插件

  int netPhysDevs;                              // ncclNet - number of physical devices
  // ncclNet 报告的物理设备数量

  int netVirtDevs;                              // ncclNet - number of virtual devices
  // ncclNet 报告的虚拟设备数量

  int collNetPhysDevs;                          // ncclCollNet - number of physical devices
  // ncclCollNet 报告的物理设备数量

  int collNetVirtDevs;                          // ncclCollNet - number of virtual devices
  // ncclCollNet 报告的虚拟设备数量
} netPluginLib_t;

// 插件计数器，记录已加载的插件数量
int pluginCount = 0;
// 网络插件库是否已初始化的标志
bool netPluginLibsInitialized = false;
// 网络插件库数组，最多支持 NCCL_NET_MAX_PLUGINS 个插件
netPluginLib_t netPluginLibs[NCCL_NET_MAX_PLUGINS] = { 0 };
// 网络插件互斥锁，用于线程安全地访问插件数据
static std::mutex netPluginMutex;
// 一次性初始化标志，确保插件库只初始化一次
static std::once_flag initPluginLibsOnceFlag;

// NCCL 网络插件卸载函数
// 参数：pluginLib - 指向要卸载的插件库结构体的指针
static ncclResult_t ncclNetPluginUnload(netPluginLib_t* pluginLib) {
  if ((pluginLib->dlHandle) && ((pluginLib->ncclNetPluginRefCount) == 0)) {  // 如果插件库句柄有效且引用计数为 0
    INFO(NCCL_INIT|NCCL_NET, "Unloading plugin %s", pluginLib->name);  // 记录日志：正在卸载插件
    NCCLCHECK(ncclClosePluginLib(pluginLib->dlHandle, ncclPluginTypeNet));  // 关闭插件动态链接库
    // memset will reset the status to ncllNetPluginStateLoadReady
    // memset 将状态重置为 LoadReady

    memset(pluginLib, 0, sizeof(netPluginLib_t));  // 将整个插件库结构体清零
    // reset the count of devices to UNDEF_DEV_COUNT
    // 将设备计数重置为未定义

    pluginLib->netPhysDevs = pluginLib->netVirtDevs = NCCL_UNDEF_DEV_COUNT;  // 将物理和虚拟设备计数重置为未定义
    pluginLib->collNetPhysDevs = pluginLib->collNetVirtDevs = NCCL_UNDEF_DEV_COUNT;  // 将集合网络的物理和虚拟设备计数重置为未定义
  }
  return ncclSuccess;  // 返回成功状态
}

// NCCL 网络插件加载函数
// 参数：pluginLib - 指向要加载的插件库结构体的指针
static ncclResult_t ncclNetPluginLoad(netPluginLib_t* pluginLib) {
  //dlopen对应的插件库，用于后续解析符号
  // 使用 dlopen 打开对应的插件库，以便后续解析符号

  pluginLib->dlHandle = ncclOpenNetPluginLib(pluginLib->name);  // 打开网络插件动态链接库

//插件不存在，返回
  // 插件不存在，直接返回失败

  if (pluginLib->dlHandle == nullptr)  // 如果插件库句柄为空（加载失败）
    goto fail;  // 跳转到失败处理

  // load ncclNet
  // 从高版本到低版本依次尝试加载 ncclNet

  //从高版本到低版本依次尝试加载ncclnet
  // 按版本从高到低依次尝试加载 ncclNet，优先使用新版本

  for (int i = 0; i < NCCL_NET_VERSION_COUNT; i++) {  // 遍历所有网络版本
    pluginLib->ncclNetVer = ncclNetVersion[i];  // 设置当前尝试的版本号
    //从handle中解析符号
    // 从动态链接库句柄中解析符号

    pluginLib->ncclNet = getNcclNet[i](pluginLib->dlHandle);  // 调用对应的获取函数，解析 ncclNet 符号
    //解析成功
    // 如果解析成功，找到有效的网络接口

    if (pluginLib->ncclNet)  // 如果成功解析到 ncclNet 接口
        break;  // 退出循环，使用当前版本
  }

//符号解析失败，返回
  // if we fail to find a net, exit
  // 如果我们找不到网络接口，退出

  //这里意味着插件必须先实现net，否则也不会加载后面的collnet
  // 这意味着插件必须先实现 ncclNet，否则也不会加载后面的 collnet

  if (pluginLib->ncclNet == nullptr) {  // 如果未能解析到 ncclNet 接口
    INFO(NCCL_INIT|NCCL_NET, "External network plugin %s is unsupported",
         (ncclPluginLibPaths[ncclPluginTypeNet] ? ncclPluginLibPaths[ncclPluginTypeNet] : pluginLib->name));  // 记录日志：外部网络插件不支持
    goto fail;  // 跳转到失败处理
  }

 //更新状态
  // 更新插件状态为 LoadReady

  pluginLib->ncclNetPluginState = ncclNetPluginStateInitReady;  // 设置网络插件状态为 InitReady（已加载，准备初始化）

  // load ncclCollNet
  // 从高版本到低版本依次尝试加载 ncclCollNet

  //从高版本到低版本依次尝试加载ncclcollnet，比如nvidia的collnet sharp插件
  // 按版本从高到低依次尝试加载 ncclCollNet，例如 NVIDIA 的 SHARP 集合网络插件

  for (int i = 0; i < NCCL_NET_VERSION_COUNT; i++) {  // 遍历所有集合网络版本
    pluginLib->ncclCollNet = getNcclCollNet[i](pluginLib->dlHandle);  // 调用对应的获取函数，解析 ncclCollNet 符号
    if (pluginLib->ncclCollNet)  // 如果成功解析到 ncclCollNet 接口
        break;  // 退出循环，使用当前版本
  }

   //collnet加载成功，更新状态
   // 如果 collNet 加载成功，更新状态；否则标记为加载失败

  if (pluginLib->ncclCollNet == nullptr)  // 如果未能解析到 ncclCollNet 接口
    pluginLib->ncclCollNetPluginState = ncclNetPluginStateLoadFailed;  // 设置集合网络插件状态为 LoadFailed
  else  // 如果成功解析到 ncclCollNet
    pluginLib->ncclCollNetPluginState = ncclNetPluginStateInitReady;  // 设置集合网络插件状态为 InitReady

  INFO(NCCL_INIT|NCCL_NET, "Successfully loaded external network plugin %s",
       (ncclPluginLibPaths[ncclPluginTypeNet] ? ncclPluginLibPaths[ncclPluginTypeNet] : pluginLib->name));  // 记录日志：成功加载外部网络插件

exit:  // 正常退出标签
  return ncclSuccess;  // 返回成功状态

fail:  // 失败处理标签
  if (pluginLib->dlHandle) {  // 如果插件库句柄有效
    NCCLCHECK(ncclClosePluginLib(pluginLib->dlHandle, ncclPluginTypeNet));  // 关闭插件动态链接库
  }
  pluginLib->dlHandle = nullptr;  // 设置插件库句柄为空
  //设置状态
  // 设置插件状态为加载失败

  pluginLib->ncclNetPluginState = ncclNetPluginStateLoadFailed;  // 设置网络插件状态为 LoadFailed
  pluginLib->ncclCollNetPluginState = ncclNetPluginStateLoadFailed;  // 设置集合网络插件状态为 LoadFailed
  goto exit;  // 跳转到退出标签
}

// NCCL 网络设备版本检查函数
// 参数说明：
//   comm - NCCL 通信上下文
//   net - 网络接口指针
//   dev - 设备号
ncclResult_t ncclNetCheckDeviceVersion(struct ncclComm* comm, ncclNet_t* net, int dev) {
  ncclNetProperties_t props;  // 网络属性结构体

  NCCLCHECK(net->getProperties(dev, &props));  // 获取网络设备属性
  ncclNetDeviceType type = props.netDeviceType;  // 获取网络设备类型
  if (type) switch (type) {  // 如果设备类型有效，进行版本检查
    case NCCL_NET_DEVICE_UNPACK:  // 如果是 UNPACK 设备类型
      if (props.netDeviceVersion == NCCL_NET_DEVICE_UNPACK_VERSION) {  // 如果设备版本匹配要求的版本
        INFO(NCCL_INIT, "Using NCCL_NET_DEVICE_UNPACK net plugin version %d",
          props.netDeviceVersion);  // 记录日志：使用匹配的 UNPACK 插件版本
        return ncclSuccess;  // 返回成功状态
      } else {  // 如果设备版本不匹配
        WARN("NCCL_DEVICE_UNPACK plugin has incompatible version %d, this NCCL build is compatible with %d, not using it",
          props.netDeviceVersion, NCCL_NET_DEVICE_UNPACK_VERSION);  // 输出警告：UNPACK 插件版本不兼容
        return ncclInternalError;  // 返回内部错误
      }
    default:  // 对于其他设备类型
      WARN("Unknown device code index %d \n", type);  // 输出警告：未知的设备代码
      return ncclInternalError;  // 返回内部错误
  }

  return ncclSuccess;  // 返回成功状态
}

// NCCL 网络插件初始化函数
// 参数说明：
//   comm - NCCL 通信上下文
//   pluginLib - 插件库结构体指针
static ncclResult_t ncclNetPluginInit(struct ncclComm* comm, netPluginLib_t* pluginLib) {
  int ndev;  // 设备数量变量
  if (pluginLib->ncclNetPluginState >= ncclNetPluginStateInitReady && pluginLib->ncclNet) {  // 如果网络插件状态 >= InitReady 且 ncclNet 接口有效
    ncclNetCommConfig_t commConfig = {};  // 通信配置结构体，初始化为零
    // 设置通信流量类别，如果用户未指定则使用未定义值
    commConfig.trafficClass = comm->config.trafficClass == NCCL_CONFIG_UNDEF_INT ? NCCL_NET_TRAFFIC_CLASS_UNDEF : comm->config.trafficClass;  // 设置流量类别

    // 调用init函数，如果失败，则退出
    // 调用插件的初始化函数，如果失败则跳转到失败处理

    if (pluginLib->ncclNet->init(&comm->netContext, comm->commHash, &commConfig, ncclDebugLog, ncclProfilerCallback) != ncclSuccess)  // 初始化网络插件
        goto fail;  // 跳转到失败处理

    //获取设备的数量
    // 获取设备数量

    if (pluginLib->ncclNet->devices(&ndev) != ncclSuccess || ndev <= 0)  // 获取设备数量并检查是否有效
        goto fail;  // 跳转到失败处理
    pluginLib->netPhysDevs = ndev;  // 保存物理设备数量
    pluginLib->netVirtDevs = NCCL_UNDEF_DEV_COUNT;  // 设置虚拟设备数量为未定义
  }

  //初始化插件成功，更新状态
  // 如果网络插件初始化成功，更新状态为 Enabled

  pluginLib->ncclNetPluginState = ncclNetPluginStateEnabled;  // 设置网络插件状态为 Enabled
  INFO(NCCL_INIT|NCCL_NET, "Initialized NET plugin %s", pluginLib->ncclNet->name);  // 记录日志：已初始化网络插件

  if (pluginLib->ncclCollNetPluginState >= ncclNetPluginStateInitReady && pluginLib->ncclCollNet) {  // 如果集合网络插件状态 >= InitReady 且 ncclCollNet 接口有效
    if (pluginLib->ncclCollNet->init(&comm->collNetContext, comm->commHash, ncclDebugLog) != ncclSuccess) {  // 初始化集合网络插件
        pluginLib->ncclCollNetPluginState = ncclNetPluginStateDisabled;  // 如果初始化失败，设置为禁用状态
    } else if (pluginLib->ncclCollNet->devices(&ndev) != ncclSuccess || ndev <= 0) {  // 获取设备数量并检查是否有效
        pluginLib->ncclCollNetPluginState = ncclNetPluginStateDisabled;  // 如果设备数量无效，设置为禁用状态
    } else {  // 如果集合网络初始化成功
      pluginLib->collNetPhysDevs = ndev;  // 保存集合网络物理设备数量
      pluginLib->collNetVirtDevs = NCCL_UNDEF_DEV_COUNT;  // 设置集合网络虚拟设备数量为未定义
      //初始化成功，更新状态
      // 初始化成功，更新状态为 Enabled

      pluginLib->ncclCollNetPluginState = ncclNetPluginStateEnabled;  // 设置集合网络插件状态为 Enabled
    }
  }

exit:  // 正常退出标签
  return ncclSuccess;  // 返回成功状态

fail:  // 失败处理标签
  INFO(NCCL_INIT|NCCL_NET, "Failed to initialize NET plugin %s", pluginLib->ncclNet->name);  // 记录日志：网络插件初始化失败
  pluginLib->ncclNet->finalize(comm->netContext);  // 终止网络插件
  pluginLib->netPhysDevs = pluginLib->netVirtDevs = NCCL_UNDEF_DEV_COUNT;  // 重置物理和虚拟设备计数为未定义
  pluginLib->collNetPhysDevs = pluginLib->collNetVirtDevs = NCCL_UNDEF_DEV_COUNT;  // 重置集合网络物理和虚拟设备计数为未定义
  pluginLib->ncclNetPluginState = ncclNetPluginStateDisabled;  // 设置网络插件状态为禁用
  pluginLib->ncclCollNetPluginState = ncclNetPluginStateDisabled;  // 设置集合网络插件状态为禁用
  goto exit;  // 跳转到退出标签
}

// NCCL 网络插件分配到通信上下文函数
// 参数说明：
//   comm - NCCL 通信上下文
//   pluginIndex - 插件索引
//   isAssigned - 输出参数，返回是否成功分配
static ncclResult_t ncclNetPluginAssignToComm(struct ncclComm* comm, int pluginIndex, bool* isAssigned) {
  const char* netName = comm->config.netName;  // 获取配置的网络名称

  if (netName && strcasecmp(netName, netPluginLibs[pluginIndex].ncclNet->name) != 0)  // 如果指定了网络名称且与插件名称不匹配
    goto fail;  // 跳转到失败处理
  if (ncclSuccess != ncclNetCheckDeviceVersion(comm, netPluginLibs[pluginIndex].ncclNet, 0))  // 检查设备版本是否兼容
    goto fail;  // 跳转到失败处理

  if (netPluginLibs[pluginIndex].ncclNetPluginState >= ncclNetPluginStateEnabled) {  // 如果插件状态为已启用
    comm->ncclNet = netPluginLibs[pluginIndex].ncclNet;  // 将网络接口分配给通信上下文
    comm->ncclNetVer = netPluginLibs[pluginIndex].ncclNetVer;  // 保存网络接口版本
    comm->netPluginIndex = pluginIndex;  // 保存插件索引
    //增加应用计数，用于多线程的情况
    // 增加引用计数，用于多线程场景

    netPluginLibs[pluginIndex].ncclNetPluginRefCount++;  // 增加插件引用计数
    *isAssigned = true;  // 设置分配成功标志
    INFO(NCCL_INIT|NCCL_NET, "Assigned NET plugin %s to comm", netPluginLibs[pluginIndex].ncclNet->name);  // 记录日志：已将网络插件分配给通信上下文

    //ncclCollNet不是必须支持，可以为NULL
    // ncclCollNet 不是必须的，可以为 NULL

    if (netPluginLibs[pluginIndex].ncclCollNetPluginState >= ncclNetPluginStateEnabled) {  // 如果集合网络插件状态为已启用
      comm->ncclCollNet = netPluginLibs[pluginIndex].ncclCollNet;  // 将集合网络接口分配给通信上下文
    }
  }

exit:  // 正常退出标签
  return ncclSuccess;  // 返回成功状态

fail:  // 失败处理标签
  *isAssigned = false;  // 设置分配失败标志
  netPluginLibs[pluginIndex].ncclNetPluginState = ncclNetPluginStateEnabled;  // 恢复网络插件状态为已启用
  netPluginLibs[pluginIndex].ncclCollNetPluginState = ncclNetPluginStateEnabled;  // 恢复集合网络插件状态为已启用
  goto exit;  // 跳转到退出标签
}

// NCCL 网络插件禁用其他外部插件函数
// 参数说明：
//   pluginIndex - 当前启用的插件索引
static ncclResult_t ncclNetPluginDisableOtherExternal(int pluginIndex) {
  // Only if an external plugin is enabled, disable other external plugins
  // 只有在启用外部插件时，才禁用其他外部插件

  if (pluginIndex >= (pluginCount - NCCL_NET_NUM_INTERNAL_PLUGINS))  // 如果不是外部插件（pluginIndex >= 外部插件起始位置）
    return ncclSuccess;  // 直接返回成功

  char names[MAX_STR_LEN*(NCCL_NET_MAX_PLUGINS - NCCL_NET_NUM_INTERNAL_PLUGINS)] = { 0 };  // 字符串缓冲区，用于存储被禁用的插件名称
  for (int i = 0; i < (pluginCount - NCCL_NET_NUM_INTERNAL_PLUGINS); i++) {  // 遍历所有外部插件
    if (i != pluginIndex) {  // 跳过当前启用的插件
      // Append all disabled plugin names to a string
      // 将所有被禁用的插件名称追加到字符串

      snprintf(names+strlen(names), sizeof(names)-strlen(names), (strlen(names) == 0) ? "%s" : ", %s", netPluginLibs[i].name);  // 格式化插件名称列表
      netPluginLibs[i].ncclNetPluginState = ncclNetPluginStateDisabled;  // 设置插件状态为禁用
    }
  }

  if(strlen(names) > 0) {  // 如果有被禁用的插件
    INFO(NCCL_INIT|NCCL_NET, "Disabling external plugins: %s", names);  // 记录日志：禁用外部插件
  }
  return ncclSuccess;  // 返回成功状态
}

// 插件库一次性初始化函数
static void initPluginLibsOnceFunc() {
  char* netPluginName = nullptr;  // 网络插件名称指针
  const char* defaultNetPlugin = "libnccl-net.so";  // 默认网络插件名称
  const char* envNetPlugin = nullptr;  // 环境变量指定的网络插件
  char* envNetPluginList = nullptr;  // 环境变量插件列表
  char* savePtr = nullptr;  // 用于 strtok_r 的保存指针
  int pluginCounter = 0;  // 插件计数器

  memset(netPluginLibs, 0, NCCL_NET_MAX_PLUGINS * sizeof(netPluginLib_t));  // 清零所有插件库结构体
  //如果设置了插件环境变量，尝试加载
  // 如果设置了 NCCL_NET_PLUGIN 环境变量，尝试加载指定的插件

  envNetPlugin = ncclGetEnv("NCCL_NET_PLUGIN");  // 获取 NCCL_NET_PLUGIN 环境变量
  if (envNetPlugin) {  // 如果环境变量存在
    INFO(NCCL_ENV|NCCL_NET, "NCCL_NET_PLUGIN set by environment to %s", envNetPlugin);  // 记录日志：环境变量设置了网络插件
    if (strcasecmp(envNetPlugin, "none") == 0)  // 如果环境变量为 "none"（表示不使用任何插件）
      envNetPlugin = "";  // 设置为空字符串
    envNetPluginList = strdup(envNetPlugin);  // 复制环境变量字符串

    //尝试加载自定义插件，可以逗号分割指定多个插件
    // 比如 NCCL_NET_PLUGIN=lib1.so,lib2.so 等

    //比如NCCL_NET_PLUGIN=1.so,2.so等
    // 例如 NCCL_NET_PLUGIN=plugin1.so,plugin2.so 等格式

    // Iterate over list until the list is empty
    // 遍历列表直到列表为空

    netPluginName = strtok_r(envNetPluginList, ",", &savePtr);  // 使用逗号分隔插件列表
    while(netPluginName) {  // 循环处理每个插件名称
      // We have 2 internal plugins (ib and socket)
      // 我们有 2 个内部插件（IB 和 Socket）

      // So, we can have at most(NCCL_NET_MAX_PLUGINS - (NCCL_NET_NUM_INTERNAL_PLUGINS)) in NCCL_NET_PLUGIN list
      // 所以 NCCL_NET_PLUGIN 列表中最多可以有（NCCL_NET_MAX_PLUGINS - 内部插件数）个外部插件

      if (pluginCounter >= (NCCL_NET_MAX_PLUGINS - (NCCL_NET_NUM_INTERNAL_PLUGINS))) {  // 如果超过最大外部插件数量
        INFO(NCCL_NET|NCCL_ENV,"NCCL_NET_PLUGIN list contains more than %d plugins, ignoring the rest", (NCCL_NET_MAX_PLUGINS - (NCCL_NET_NUM_INTERNAL_PLUGINS + 1)));  // 记录日志：插件数量超过限制，忽略剩余的
        break;  // 退出循环
      }
      // need to leave space for the name + "\n"
      // 需要为名称和换行符预留空间

      if((strlen(netPluginName)+1) <= MAX_STR_LEN) {  // 如果插件名称长度不超过最大长度
        netPluginLibs[pluginCounter].ncclNetPluginState = ncclNetPluginStateLoadReady;  // 设置插件状态为 LoadReady
        netPluginLibs[pluginCounter].ncclNetPluginRefCount = ncclParamNetPluginRefCount();  // 初始化引用计数
        strcpy(netPluginLibs[pluginCounter].name, netPluginName);  // 复制插件名称
        pluginCounter++;  // 增加插件计数器
      } else {  // 如果插件名称过长
        INFO(NCCL_NET|NCCL_ENV,"NCCL_NET_PLUGIN list contains a plugin name %s longer than %d characters, ignoring it.", netPluginName, MAX_STR_LEN);  // 记录日志：插件名称过长，忽略
      }
      netPluginName = strtok_r(nullptr, ",", &savePtr);  // 获取下一个插件名称
    }
    if (envNetPluginList)  // 如果插件列表字符串有效
        free(envNetPluginList);  // 释放插件列表字符串
  } else {  // 如果没有设置环境变量
    // Add default net plugin
    // 添加默认网络插件

    //设置libnccl-net.so为默认的plugin
    // 设置 libnccl-net.so 为默认插件

    netPluginLibs[pluginCounter].ncclNetPluginState = ncclNetPluginStateLoadReady;  // 设置插件状态为 LoadReady
    netPluginLibs[pluginCounter].ncclNetPluginRefCount = ncclParamNetPluginRefCount();  // 初始化引用计数
    strcpy(netPluginLibs[pluginCounter++].name, defaultNetPlugin);  // 复制默认插件名称并增加计数器
  }

  //这里可以看出pluginCounter的值>=3
  // 可以看出 pluginCounter 的值 >= 3（环境变量指定 0-2 个插件，加上默认插件 1 个）

  // Add 2 internal ib and socket plugins
  // 添加 2 个内部插件：IB 和 Socket

  //IB在前，如果支持优先使用IB
  // IB 在前，如果支持优先使用 IB

  netPluginLibs[pluginCounter].ncclNet = &ncclNetIb;  // 设置 IB 网络接口
  netPluginLibs[pluginCounter++].ncclNetPluginState = ncclNetPluginStateInitReady;  // 设置 IB 插件状态为 InitReady 并增加计数器
  netPluginLibs[pluginCounter].ncclNet = &ncclNetSocket;  // 设置 Socket 网络接口
  netPluginLibs[pluginCounter++].ncclNetPluginState = ncclNetPluginStateInitReady;  // 设置 Socket 插件状态为 InitReady 并增加计数器
  pluginCount = pluginCounter;  // 保存插件总数
}

// NCCL 网络初始化函数
// 参数说明：
//   comm - NCCL 通信上下文
ncclResult_t ncclNetInit(struct ncclComm* comm) {
  bool ncclNetPluginInitialized = false;  // 网络插件初始化成功标志
  //加载外部库和2个最重要的内部插件，ncclNetIb和ncclNetSocket
  // 加载外部插件库和 2 个最重要的内部插件：ncclNetIb 和 ncclNetSocket

  std::call_once(initPluginLibsOnceFlag, initPluginLibsOnceFunc);  // 一次性初始化插件库

  std::lock_guard<std::mutex> lock(netPluginMutex);  // 加锁保护插件数据
  //这里按照顺序加载plugin插件，谁先加载成功，优先使用谁。如果自定义插件加载成功，
  // 按顺序加载插件，先加载成功的优先使用。如果自定义插件加载成功，
  //优先使用自定义插件的ncclNet

  //否则使用内部插件，先加载IB插件，再加载socket插件
  // 否则使用内部插件，先加载 IB 插件，再加载 Socket 插件

  //ncclCollNets则表示SHARP插件
  // ncclCollNets 表示 SHARP 插件

  for (int pluginIndex = 0; pluginIndex < pluginCount; pluginIndex++) {  // 遍历所有插件
    //加载自定义插件，这里不包括2个内部插件
    // 加载自定义插件（不包括 2 个内部插件）

    if ((pluginIndex < (pluginCount - NCCL_NET_NUM_INTERNAL_PLUGINS)) && (netPluginLibs[pluginIndex].ncclNetPluginState == ncclNetPluginStateLoadReady)) {  // 如果是外部插件且状态为 LoadReady
      //内部会尝试加载ncclNet和ncclCollNet
      // 内部会尝试加载 ncclNet 和 ncclCollNet

      NCCLCHECK(ncclNetPluginLoad(&netPluginLibs[pluginIndex]));  // 加载插件
    }

    //调用插件的init函数
    // 调用插件的初始化函数

    if (netPluginLibs[pluginIndex].ncclNetPluginState >= ncclNetPluginStateInitReady) {  // 如果插件状态 >= InitReady
      NCCLCHECK(ncclNetPluginInit(comm, &netPluginLibs[pluginIndex]));  // 初始化插件
    }

    //插件已经enable
    // 如果插件已经启用

    //如果外部插件都加载失败，则使用内部的2个插件
    // 如果外部插件都加载失败，则使用内部的 2 个插件

    //分配vft表给comm的ncclNet
    // 分配虚函数表给 comm 的 ncclNet

    if (netPluginLibs[pluginIndex].ncclNetPluginState == ncclNetPluginStateEnabled) {  // 如果插件状态为 Enabled
      bool isAssigned = false;  // 分配标志
      //给通信器指定ncclNet和ncclCollNet
      // 给通信器指定 ncclNet 和 ncclCollNet

      NCCLCHECK(ncclNetPluginAssignToComm(comm, pluginIndex, &isAssigned));  // 尝试将插件分配给通信上下文
      //如果ncclIB先注册成功，就不再使用ncclsocket
      // 如果 ncclIB 先注册成功，就不再使用 ncclSocket

      //分配已经成功，禁用外部其他插件，设置状态为ncclNetPluginStateDisabled
      // 分配成功后，禁用其他外部插件，设置状态为 Disabled

      // If one external plugin is assigned to a comm, then disable all other external plugins
      // 如果一个外部插件分配给通信上下文，则禁用所有其他外部插件

      if (isAssigned) {  // 如果分配成功
        ncclNetPluginDisableOtherExternal(pluginIndex);  // 禁用其他外部插件
        ncclNetPluginInitialized = true;  // 设置插件初始化成功标志
        //插件分配成功，结束循环，不再尝试后续的插件
        // 插件分配成功，结束循环，不再尝试后续的插件

        break;  // 退出循环
      }
    }
  }

  //插件加载成功
  // 如果插件加载成功

  if (ncclNetPluginInitialized)  // 如果有任何插件初始化成功
    return ncclSuccess;  // 返回成功状态

  WARN("Failed to initialize any NET plugin");  // 输出警告：未能初始化任何网络插件
  return ncclInvalidUsage;  // 返回无效用法错误
}

// NCCL 网络终止函数
// 参数说明：
//   comm - NCCL 通信上下文
ncclResult_t ncclNetFinalize(struct ncclComm* comm) {
  int pluginIndex = comm->netPluginIndex;  // 获取插件索引
  std::lock_guard<std::mutex> lock(netPluginMutex);  // 加锁保护插件数据
  NCCLCHECK(comm->ncclNet->finalize(comm->netContext));  // 终止网络插件
  if (comm->collNetContext) NCCLCHECK(comm->ncclCollNet->finalize(comm->collNetContext));  // 如果有集合网络上下文，终止集合网络插件
  netPluginLibs[pluginIndex].ncclNetPluginRefCount--;  // 减少插件引用计数
  for (int i = 0; i < (pluginCount - NCCL_NET_NUM_INTERNAL_PLUGINS); i++) {  // 遍历所有外部插件
    NCCLCHECK(ncclNetPluginUnload(&netPluginLibs[i]));  // 卸载外部插件
  }
  return ncclSuccess;  // 返回成功状态
}

// NCCL 网络获取设备数量函数
// 参数说明：
//   netPluginIndex - 网络插件索引
//   nPhysDevs - 输出参数，返回物理设备数量
//   nVirtDevs - 输出参数，返回虚拟设备数量
ncclResult_t ncclNetGetDevCount(int netPluginIndex, int* nPhysDevs, int* nVirtDevs) {
  if (netPluginLibs[netPluginIndex].ncclNetPluginState != ncclNetPluginStateEnabled ||
     netPluginLibs[netPluginIndex].netPhysDevs == NCCL_UNDEF_DEV_COUNT) goto fail;  // 如果插件未启用或物理设备数量未定义
  // lock not needed as it's called within a lock already in ncclTopoGetSystem
  // 不需要加锁，因为在 ncclTopoGetSystem 中已经加锁

  *nPhysDevs = netPluginLibs[netPluginIndex].netPhysDevs;  // 返回物理设备数量
  *nVirtDevs = netPluginLibs[netPluginIndex].netVirtDevs;  // 返回虚拟设备数量
  return ncclSuccess;  // 返回成功状态

fail:  // 失败处理标签
  WARN("%s: trying to access the number of devices of an uninitialized netPlugin[%d]", __func__, netPluginIndex);  // 输出警告：尝试访问未初始化的插件设备数量
  return ncclInternalError;  // 返回内部错误
}

// NCCL 集合网络获取设备数量函数
// 参数说明：
//   netPluginIndex - 网络插件索引
//   nPhysDevs - 输出参数，返回物理设备数量
//   nVirtDevs - 输出参数，返回虚拟设备数量
ncclResult_t ncclCollNetGetDevCount(int netPluginIndex, int* nPhysDevs, int* nVirtDevs) {
  if (netPluginLibs[netPluginIndex].ncclCollNetPluginState != ncclNetPluginStateEnabled ||
     netPluginLibs[netPluginIndex].collNetPhysDevs == NCCL_UNDEF_DEV_COUNT) goto fail;  // 如果集合网络插件未启用或物理设备数量未定义
  // lock not needed as it's called within a lock already in ncclTopoGetSystem
  // 不需要加锁，因为在 ncclTopoGetSystem 中已经加锁

  *nPhysDevs = netPluginLibs[netPluginIndex].collNetPhysDevs;  // 返回集合网络物理设备数量
  *nVirtDevs = netPluginLibs[netPluginIndex].collNetVirtDevs;  // 返回集合网络虚拟设备数量
  return ncclSuccess;  // 返回成功状态

fail:  // 失败处理标签
  WARN("%s: trying to access the number of devices of an uninitialized netPlugin[%d]", __func__, netPluginIndex);  // 输出警告：尝试访问未初始化的集合网络插件设备数量
  return ncclInternalError;  // 返回内部错误
}

// NCCL 网络设置虚拟设备数量函数
// 参数说明：
//   netPluginIndex - 网络插件索引
//   nVirtDevs - 要设置的虚拟设备数量
ncclResult_t ncclNetSetVirtDevCount(int netPluginIndex, int nVirtDevs) {
  if (netPluginLibs[netPluginIndex].ncclNetPluginState != ncclNetPluginStateEnabled || nVirtDevs < 0) goto fail;  // 如果插件未启用或虚拟设备数量无效
  // lock not needed as it's called within a lock already in ncclTopoGetSystem
  // 不需要加锁，因为在 ncclTopoGetSystem 中已经加锁

  netPluginLibs[netPluginIndex].netVirtDevs = nVirtDevs;  // 设置虚拟设备数量
  return ncclSuccess;  // 返回成功状态

fail:  // 失败处理标签
  WARN("%s: failed to set the number of devices for netPlugin[%d] to %d", __func__, netPluginIndex,nVirtDevs);  // 输出警告：设置虚拟设备数量失败
  return ncclInternalError;  // 返回内部错误
}

// NCCL 集合网络设置虚拟设备数量函数
// 参数说明：
//   netPluginIndex - 网络插件索引
//   nVirtDevs - 要设置的虚拟设备数量
ncclResult_t ncclCollNetSetVirtDevCount(int netPluginIndex, int nVirtDevs) {
  if (netPluginLibs[netPluginIndex].ncclCollNetPluginState != ncclNetPluginStateEnabled || nVirtDevs < 0) goto fail;  // 如果集合网络插件未启用或虚拟设备数量无效
  // lock not needed as it's called within a lock already in ncclTopoGetSystem
  // 不需要加锁，因为在 ncclTopoGetSystem 中已经加锁

  netPluginLibs[netPluginIndex].collNetVirtDevs = nVirtDevs;  // 设置集合网络虚拟设备数量
  return ncclSuccess;  // 返回成功状态

fail:  // 失败处理标签
  WARN("%s: failed to set the number of devices for netPlugin[%d] to %d", __func__, netPluginIndex,nVirtDevs);  // 输出警告：设置集合网络虚拟设备数量失败
  return ncclInternalError;  // 返回内部错误
}

//判读GPU是否支持gdr
// 检测 GPU 是否支持 GPUDirect RDMA (GDR)

ncclResult_t ncclGpuGdrSupport(struct ncclComm* comm, int* gdrSupport) {
  constexpr int GPU_BUF_SIZE = 2*1024*1024;  // 定义 GPU 缓冲区大小常量为 2MB
  //CUDART_VERSION 主要用于"编译期"判断当前代码面向的 CUDA Runtime 版本
  // RT 是 Runtime 的意思

  #if CUDART_VERSION >= 11030  // 如果 CUDA 运行时版本 >= 11.3
  // In CUDA 11.3 and later we can now query the cudaDevAttrGPUDirectRDMA_supported attribute
  // 在 CUDA 11.3 及更高版本中，我们可以查询 cudaDevAttrGPUDirectRDMA_supported 属性

  //高版本CUDA直接可获取相关属性
  // 高版本 CUDA 可以直接获取相关属性

  int driverVersion;  // 驱动版本变量
  CUDACHECK(cudaDriverGetVersion(&driverVersion));  // 获取 CUDA 驱动版本
  if (driverVersion >= 11030) {  // 如果驱动版本 >= 11.3
    int cudaDev, attr = 0;  // CUDA 设备号和属性变量
    CUDACHECK(cudaGetDevice(&cudaDev));  // 获取当前 CUDA 设备
    CUDACHECK(cudaDeviceGetAttribute(&attr, cudaDevAttrGPUDirectRDMA_supported, cudaDev));  // 获取 GPUDirect RDMA 支持属性
    *gdrSupport = attr;  // 返回 GDR 支持状态
    return ncclSuccess;  // 返回成功状态
  }
  #endif  // 结束 CUDA 版本条件编译

  // 单机限制了32张卡
  // 单机限制最多 32 张 GPU 卡

  static int gdrSupportMatrix[32] = {  // 静态数组，记录每个 GPU 的 GDR 支持状态
	  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  // 前 11 个 GPU 初始化为 -1（未检测）
	  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 };  // 后 21 个 GPU 初始化为 -1（未检测）

  //如果cuda版本过低，这里采用其他方式来判断
  // 如果 CUDA 版本过低，这里采用其他方式来判断

  //在GPU中分配显存，并逐个尝试register到对应NIC，建立通信的方式确认是否支持GDR
  // 在 GPU 中分配显存，并逐个尝试注册到对应 NIC，建立通信的方式确认是否支持 GDR

  if (gdrSupportMatrix[comm->cudaDev] == -1) {  // 如果当前 GPU 的 GDR 状态未检测
    int netDevs;  // 网络设备数量
    //获取当前的设备数量
    // 获取当前的网络设备数量

    NCCLCHECK(comm->ncclNet->devices(&netDevs));  // 获取网络设备数量
    gdrSupportMatrix[comm->cudaDev] = 0;  // 初始化当前 GPU 的 GDR 支持为 0

    for (int dev=0; dev<netDevs; dev++) {  // 遍历所有网络设备
      // Find a net device which is GDR-capable
      // 查找支持 GDR 的网络设备

      ncclNetProperties_t props;  // 网络属性结构体
      NCCLCHECK(comm->ncclNet->getProperties(dev, &props));  // 获取网络设备属性
      //对于普通网卡，ncclNet为ncclNetSocket，ptrSupport为NCCL_PTR_HOST
      // 对于普通网卡（ncclNetSocket），ptrSupport 为 NCCL_PTR_HOST

      if ((props.ptrSupport & NCCL_PTR_CUDA) == 0)  // 如果不支持 CUDA 指针（不是 IB 或 RDMA 设备）
        continue;  // 跳过此设备

      // Allocate memory on the GPU and try to register it on the NIC.
      // 在 GPU 上分配内存并尝试将其注册到网卡。

      void *lComm = NULL, *sComm = NULL, *rComm = NULL;  // 发送和接收通信连接指针
      ncclNetHandle_t handle;  // 网络句柄
      char* gpuPtr = NULL;  // GPU 内存指针
      void* mHandle = NULL;  // 内存注册句柄
      ncclResult_t ret;  // 返回值
      ncclDebugNoWarn = NCCL_NET;  // 禁用网络调试日志，避免测试过程产生过多日志
      NCCLCHECKGOTO(comm->ncclNet->listen(comm->netContext, dev, &handle, &lComm), ret, cleanup1);  // 创建监听连接

      bool connected;  // 连接标志
      connected = false;  // 初始化为未连接
      while (!connected) {  // 循环直到建立连接

          // If we're aborting now, skip to cleanup
          // 如果正在中止，跳转到清理

          if (__atomic_load_n(comm->abortFlag, __ATOMIC_ACQUIRE)) {  // 原子加载中止标志
            goto cleanup2;  // 跳转到清理阶段 2
          }

          if (sComm == NULL)  // 如果发送连接未创建
            NCCLCHECKGOTO(comm->ncclNet->connect(comm->netContext, dev, &handle, &sComm, NULL), ret, cleanup2);  // 创建发送连接

          if (rComm == NULL)  // 如果接收连接未创建
            NCCLCHECKGOTO(comm->ncclNet->accept(lComm, &rComm, NULL), ret, cleanup2);  // 接受连接

          connected = (rComm != NULL) && (sComm != NULL);  // 检查是否双方连接都已建立
     }

     //在GPU中分配显存
     // 在 GPU 中分配显存

     NCCLCHECKGOTO(ncclCudaMalloc(&gpuPtr, GPU_BUF_SIZE), ret, cleanup2);  // 分配 GPU 内存

     //调用reg mr，如果成功，表示支持gdr
     // 调用注册内存函数，如果成功，表示支持 GDR

     if (comm->ncclNet->regMr(sComm, gpuPtr, GPU_BUF_SIZE, NCCL_PTR_CUDA, &mHandle) == ncclSuccess) {  // 尝试注册发送端内存
         NCCLCHECK(comm->ncclNet->deregMr(sComm, mHandle));  // 注销发送端内存注册
         NCCLCHECK(comm->ncclNet->regMr(rComm, gpuPtr, GPU_BUF_SIZE, NCCL_PTR_CUDA, &mHandle));  // 尝试注册接收端内存
         NCCLCHECK(comm->ncclNet->deregMr(rComm, mHandle));  // 注销接收端内存注册
         //设置为支持gdr
         // 设置为支持 GDR

         gdrSupportMatrix[comm->cudaDev] = 1;  // 标记当前 GPU 支持 GDR
     }
     ncclDebugNoWarn = 0;  // 恢复网络调试日志设置
     NCCLCHECK(ncclCudaFree(gpuPtr));  // 释放 GPU 内存

cleanup2:  // 清理阶段 2
     if (rComm != NULL)  // 如果接收连接有效
        NCCLCHECK(comm->ncclNet->closeRecv(rComm));  // 关闭接收连接
     if (sComm != NULL)  // 如果发送连接有效
        NCCLCHECK(comm->ncclNet->closeSend(sComm));  // 关闭发送连接

     NCCLCHECK(comm->ncclNet->closeListen(lComm));  // 关闭监听连接

cleanup1:  // 清理阶段 1
      break;  // 退出设备循环
    }
  }
  //返回是否支持gdr
  // 返回是否支持 GDR

  *gdrSupport = gdrSupportMatrix[comm->cudaDev];  // 返回 GDR 支持状态
  return ncclSuccess;  // 返回成功状态
}
