/*************************************************************************
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 * 版权所有 (c) 2025，NVIDIA 公司。保留所有权利。
 *
 * See LICENSE.txt for license information
 * 详见 LICENSE.txt 获取许可证信息
 ************************************************************************/
// 头文件保护宏开始：防止头文件被重复包含
#ifndef NCCL_DEVICE_RUNTIME_H_
// 定义头文件保护宏
#define NCCL_DEVICE_RUNTIME_H_

// 包含 NCCL 公共头文件，定义了 NCCL 的基本数据类型和枚举
#include "nccl.h"
// 包含 NCCL 设备端公共头文件，定义设备端的数据结构和常量
#include "nccl_device.h"
// 包含分配器头文件，定义内存分配相关接口
#include "allocator.h"
// 包含位操作头文件，提供各种位操作辅助函数
#include "bitops.h"
// 包含工具函数头文件，提供各种通用工具函数
#include "utils.h"

// 分隔线：用于区分不同代码区域
////////////////////////////////////////////////////////////////////////////////

// 注释：ncclDevr[_]: runtime implements for symmetric API.
// 含义：NCCL 设备端运行时实现了对称 API（Symmetric API）的运行时支持
// 对称 API：指 NCCL 的本地对称地址（LSA）功能，允许多个 GPU 共享统一的虚拟地址空间

// 结构体：ncclDevrMemory - NCCL 设备端运行时内存结构
// 功能：封装对称内存区域的相关信息
struct ncclDevrMemory {
    // 指向本地注册句柄（用于内存注册和映射）
    void* localRegHandle;
};

// 结构体：ncclDevrWindow - NCCL 设备端运行时窗口结构
// 功能：描述一个对称内存窗口，用于跨 GPU 的数据传输
struct ncclDevrWindow {
    // 指向内存结构体指针
    struct ncclDevrMemory* memory;
    // 用户空间指针（映射到用户空间的内存地址）
    void* userPtr;
    // 内存大小（字节数）
    size_t size;
    // 在大地址空间中的偏移量（相对于大地址空间的起始位置）
    size_t bigOffset; // Offset in big VA space. // 在大地址空间中的偏移
    // 窗口标志位（用于描述窗口属性）
    int winFlags;
    // 本地注册句柄（用于内存注册）
    void* localRegHandle;
    // 视频内存指针（用于 CUDA 可视化相关的内存映射）
    struct ncclWindow_vidmem* vidmem;
};

// 结构体：ncclDevrWindowSorted - 已排序的窗口数组结构
// 功能：管理按优先级排序的窗口列表
struct ncclDevrWindowSorted {
    // 排序后的窗口数量
    int windowCount;
    // 窗口容量（最大可容纳的窗口数）
    int windowCapacity;
    // 窗口数组指针（指向窗口指针数组）
    struct ncclDevrWindow** windows;
};

// 结构体：ncclDevrTeam - LSA（本地对称地址空间）团队结构
// 功能：描述 LSA 团队的配置信息
struct ncclDevrTeam {
    // 团队中的排名数量
    int nranks;
    // 本地排名（当前排名在团队中的索引）
    int localRank;
};

// 结构体：ncclDevrRegTask - 注册任务结构
// 功能：描述内存窗口注册任务
struct ncclDevrRegTask {
    // 指向下一个注册任务（用于链表管理）
    struct ncclDevrRegTask *next;
    // 用户空间指针（要注册的内存地址）
    void* userPtr;
    // 用户空间大小（字节数）
    size_t userSize;
    // 窗口标志位（描述窗口属性）
    int winFlags;
    // 输出设备端窗口指针（设备端可访问的窗口结构）
    ncclWindow_t* outWinDev;
};

// 结构体：ncclDevrCommCreateTask - 通信创建任务结构
// 功能：描述设备通信创建任务
struct ncclDevrCommCreateTask {
    // 指向下一个通信创建任务（用于链表管理）
    struct ncclDevrCommCreateTask *next;
    // 指向通信需求结构体
    struct ncclDevCommRequirements* reqs;
    // 输出设备端通信结构体（设备端可访问的通信结构）
    struct ncclDevComm* outDevComm;
};

// 结构体：ncclDevrState - NCCL 设备端运行时状态结构
// 功能：维护设备端对称内存和通信的完整状态信息
struct ncclDevrState {
    // 注释：Like localRank/localRanks except "lsa" ranks must be consecutive in world
    // 含义：与 localRank/localRanks 类似，但 LSA 排名必须在全局排名中连续
    //       并且所有 LSA 子集具有相同数量的排名
    //       如果任何条件为假，则 lsa 团队只是自身的单例
    // 解释：LSA（Local Symmetric Address）要求参与排名在全局排名中连续排列
    //       且所有 LSA 子集的大小必须相同。如果满足条件，LSA 团队是多排名的；
    //       否则只是单个排名的单例模式
    // LSA 单例：LSA ranks 不连续，只能单个 rank 使用对称内存

    // LSA 自身标志（如果 LSA 团队只是自身的单例，则为 1）
    int lsaSelf;
    // LSA 大小（LSA 团队中的排名数量）
    int lsaSize;
    // LSA 排名列表（每个 LSA 排名对应的全局排名）
    int* lsaRankList;

    // 分配粒度（通过 cuMemGetAllocationGranularity 获取）
    // 用途：内存分配的最小单位，用于对齐以提高性能
    size_t granularity; // cuMemGetAllocationGranularity

    // 内存头指针（指向大地址空间的起始位置）
    struct ncclDevrMemory* memHead;
    // 已排序的窗口列表（按优先级排序的窗口数组）
    struct ncclDevrWindowSorted* winSorted;
    // 窗口容量（已排序窗口数组的容量）
    int winSortedCapacity;
    // 窗口计数（当前窗口数量）
    int winSortedCount;
    // 团队头指针（指向 LSA 团队结构）
    struct ncclDevrTeam* teamHead;
    // 大地址空间大小（我们大逻辑地址空间的大小，例如 128GB）
    // 用途：定义 LSA 团队使用的整个虚拟地址空间范围
    size_t bigSize; // size of our big logical space (128GB?)

    // LSA 平面基地址（所有 LSA rank 的大虚拟地址空间连接后的基地址）
    // 用途：所有 LSA rank 的虚拟地址从这个基地址开始分配
    //       例如：rank 0 从 base + 0*bigSize 开始，rank 1 从 base + 1*bigSize 开始
    void* lsaFlatBase; // base ptr for all lsa ranks big VA's concatenated together: size = lsaRanks*bigSize

    // 阴影池结构（用于管理内存映射的阴影区域）
    struct ncclShadowPool shadows;
    // 窗口表指针（指向设备端通信窗口表）
    struct ncclDevCommWindowTable* windowTable;

    // 注册任务队列（内部队列，用于管理窗口注册任务）
    // 类型：每个元素是 ncclDevrRegTask 结构
    // 用途：链式队列管理，高效处理多个窗口注册请求
    struct ncclIntruQueue<struct ncclDevrRegTask, &ncclDevrRegTask::next> regTaskQueue;

    // 通信创建任务队列（内部队列，用于管理通信建立请求）
    // 类型：每个元素是 ncclDevrCommCreateTask 结构
    struct ncclIntruQueue<struct ncclDevrCommCreateTask, &ncclDevrCommCreateTask::next> commCreateTaskQueue;
};

// 设备端函数声明：NCCL 设备端运行时的核心函数接口

// 功能：NCCL 设备端运行时一次性初始化
// 参数说明：
//   comm: NCCL 通信器指针
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
ncclResult_t ncclDevrInitOnce(struct ncclComm* comm);

// 功能：NCCL 设备端运行时最终清理
// 参数说明：
//   comm: NCCL 通信器指针
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
ncclResult_t ncclDevrFinalize(struct ncclComm* comm);

// 功能：在设备端对称内存中查找窗口
// 参数说明：
//   comm: NCCL 通信器指针
//   userPtr: 用户空间指针（要查找的内存地址）
//   outWin: 输出参数，返回找到的窗口指针
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
// 说明：如果找到窗口，*outWinId >= 0；否则 *outWinId == -1
ncclResult_t ncclDevrFindWindow(struct ncclComm* comm, void const* userPtr, struct ncclDevrWindow** outWin);

// 功能：在组内注册对称内存窗口
// 参数说明：
//   comm: NCCL 通信器指针
//   ptr: 要注册的内存地址
//   size: 内存大小（字节数）
//   winFlags: 窗口标志位（描述窗口属性）
//   outWinDev: 输出参数，返回设备端窗口指针
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
ncclResult_t ncclDevrWindowRegisterInGroup(
    struct ncclComm* comm, void* ptr, size_t size, int winFlags, ncclWindow_t* outWinDev);

// 功能：内部创建设备通信
// 参数说明：
//   comm: NCCL 通信器指针
//   reqs: 通信需求结构体指针
//   outDevComm: 输出参数，返回设备端通信结构体指针
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
ncclResult_t ncclDevrCommCreateInternal(
    struct ncclComm* comm, struct ncclDevCommRequirements const* reqs, struct ncclDevComm* outDevComm);

// 功能：释放设备端通信需求
// 参数说明：
//   reqs: 要释放的通信需求结构体指针
// 返回值：void（无返回值）
void freeDevCommRequirements(struct ncclDevCommRequirements* reqs);

// 功能：获取另一个 LSA rank 的对称内存窗口中的指针
// 参数说明：
//   comm: NCCL 通信器指针
//   winHost: 主机端窗口指针（描述窗口信息）
//   offset: 偏移量（相对于窗口起始位置）
//   lsaRank: LSA 排名（指定要访问哪个 rank 的内存）
//   outPtr: 输出参数，返回计算后的设备端指针
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
ncclResult_t ncclDevrGetLsaRankPtr(struct ncclComm* comm, struct ncclDevrWindow* winHost, size_t offset, int lsaRank, void** outPtr);

// 功能：获取 LSA 团队的多播地址
// 参数说明：
//   comm: NCCL 通信器指针
//   winHost: 主机端窗口指针（描述窗口信息）
//   offset: 偏移量（相对于窗口起始位置）
//   lsaTeam: LSA 团队标识（指定要获取哪个团队的多播地址）
//   outPtr: 输出参数，返回计算后的设备端指针
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
ncclResult_t ncclDevrGetLsaTeamPtrMC(struct ncclComm* comm, struct ncclDevrWindow* winHost, size_t offset, struct ncclTeam lsaTeam, void** outPtr);

// 条件编译结束：结束条件编译指令块
#endif
