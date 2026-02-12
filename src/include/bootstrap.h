/*************************************************************************
 * Copyright (c) 2015-2022, NVIDIA CORPORATION. All rights reserved.
 * 版权所有 (c) 2015-2022，NVIDIA 公司。保留所有权利。
 *
 * See LICENSE.txt for license information
 * 详见 LICENSE.txt 获取许可证信息
 ************************************************************************/
// 头文件保护宏开始：防止头文件被重复包含
#ifndef NCCL_BOOTSTRAP_H_
// 定义头文件保护宏
#define NCCL_BOOTSTRAP_H_

// 包含 NCCL 公共头文件，定义基本数据类型和枚举
#include "nccl.h"
// 包含通信器相关头文件，定义 NCCL 通信器结构和接口
#include "comm.h"

// 结构体：NCCL Bootstrap 句柄
// 功能：封装与对端连接的验证信息
// 用途：在建立连接时验证对方是否合法的 NCCL 对端
struct ncclBootstrapHandle {
    // 魔证对端连接的魔术数字（固定值）
    // 用途：用于确认连接的对端是否为预期的 NCCL 节点
    // 通常是一个约定好的随机数或固定常量，防止意外连接
    uint64_t magic;
    // Socket 地址联合体：可以存储 IPv4 或 IPv6 地址
    union ncclSocketAddress addr;
};

// 编译时断言：检查 Bootstrap 句柄大小是否合适
// 用途：确保句柄大小在预期范围内，防止缓冲区溢出
// 参数说明：Bootstrap handle 的预期大小
// 如果实际大小超过预期大小，编译时报错
static_assert(sizeof(struct ncclBootstrapHandle) <= sizeof(ncclUniqueId), "Bootstrap handle is too large to fit inside NCCL unique ID");
// 编译时断言：解释：NCCL Unique ID 是用于标识 NCCL 唯一 ID 的固定大小
//                 如果 Bootstrap 句柄太大，无法放入 Unique ID 中

// 外部函数声明：网络初始化和获取唯一 ID 的核心函数接口
// 用途：Bootstrap 协议的网络初始化函数，获取通信节点的唯一标识符

// 函数：bootstrapNetInit
// 功能：初始化 Bootstrap 网络连接
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
// 参数说明：无参数
// 说明：初始化网络子系统，准备 Bootstrap 连接
ncclResult_t bootstrapNetInit();

// 函数：bootstrapGetUniqueId
// 功能：获取当前通信节点的唯一标识符
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
// 参数说明：无参数
// 说明：查询并获取本节点的 NCCL Unique ID，用于标识通信节点
ncclResult_t bootstrapGetUniqueId();

// 函数：bootstrapCreateRoot
// 功能：创建 Bootstrap 根节点（主节点）
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
// 参数说明：
//   handle: 输出参数，返回 Bootstrap 句柄
//   idFromEnv: 布尔标志，是否从环境变量读取根节点标识
// 返回值：成功或错误码
ncclResult_t bootstrapCreateRoot(struct ncclBootstrapHandle* handle, bool idFromEnv);

// 函数：bootstrapGetUniqueId
// 功能：获取根节点的唯一标识符
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
// 参数说明：
//   handle: Bootstrap 句柄指针
// 返回值：成功或错误码
// 说明：从句柄中提取并返回节点的唯一 ID
ncclResult_t bootstrapGetUniqueId(struct ncclBootstrapHandle* handle);

// 外部函数声明：初始化和管理 Bootstrap 连接的核心函数接口

// 函数：bootstrapInit
// 功能：初始化 Bootstrap 连接
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
// 参数说明：
//   nHandles: 句柄数量（通信对端数量）
//   handle: Bootstrap 句柄指针（输入/输出参数）
//   comm: 通信器指针（返回通信状态信息）
ncclResult_t bootstrapInit(int nHandles, struct ncclBootstrapHandle* handle, struct ncclComm* comm);

// 函数：bootstrapSplit
// 功能：将通信域划分为多个子域（树形分裂）
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
// 参数说明：
//   magic: 验证魔术数字（用于确认对端）
//   comm: 通信器指针
//   parent: 父通信域（用于树形拓扑）
//   color: 颜色标识（用于拓扑感知）
//   key: 键值（用于一致性哈希）
//   parentRanks: 父域中的排名数组
// 返回值：成功或错误码
ncclResult_t bootstrapSplit(uint64_t magic, struct ncclComm* comm, struct ncclComm* parent, int color, int key, int* parentRanks);

// 函数：bootstrapAllGather
// 功能：根节点收集所有子域的数据（All-Gather 操作）
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
// 参数说明：
//   commState: 通信器状态指针
//   allData: 输出缓冲区指针（存储收集的数据）
//   size: 每个数据块的大小（字节数）
// 返回值：成功或错误码
ncclResult_t bootstrapAllGather(void* commState, void* allData, int size);

// 函数：bootstrapSend
// 功能：发送数据到指定的对端（点对点发送）
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
// 参数说明：
//   commState: 通信器状态指针
//   peer: 目标对端排名
//   tag: 消息标签（用于匹配发送和接收）
//   data: 发送数据的缓冲区指针
//   size: 发送数据的大小（字节数）
// 返回值：成功或错误码
ncclResult_t bootstrapSend(void* commState, int peer, int tag, void* data, int size);

// 函数：bootstrapRecv
// 功能：从指定对端接收数据（点对点接收）
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
// 参数说明：
//   commState: 通信器状态指针
//   peer: 源对端排名
//   协同接收的 tag 标签
//   data: 接收数据的缓冲区指针
//   size: 接收数据的大小（字节数）
// 返回值：成功或错误码
ncclResult_t bootstrapRecv(void* commState, int peer, int tag, void* data, int size);

// 函数：bootstrapBarrier
// 功能：在指定的排名组中执行屏障同步
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
// 参数说明：
//   commState: 通信器状态指针
//   rank: 当前排名
//   nranks: 参与屏障同步的排名总数
//   tag: 屏障标签（用于匹配屏障操作）
// 返回值：成功或错误码
ncclResult_t bootstrapBarrier(void* commState, int rank, int nranks, int tag);

// 函数：bootstrapBroadcast
// 功能：根节点向所有子节点广播数据（一对多广播）
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
// 参数说明：
//   commState: 通信器状态指针
//   rank: 当前排名（根节点为 0）
//   nranks: 参与广播的排名总数
//   root: 根节点标识（谁发送数据）
//   bcastData: 广播数据的缓冲区指针
//   size: 广播数据的大小（字节数）
// 返回值：成功或错误码
ncclResult_t bootstrapBroadcast(void* commState, int rank, int nranks, int root, void* bcastData, int size);

// 函数：bootstrapIntraNodeBarrier
// 功能：在节点内部执行屏障同步（同一节点内的多 GPU 同步）
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
// 参数说明：
//   commState: 通信器状态指针
//   ranks: 参与屏障同步的排名数组
//   rank: 当前排名
//   nranks: 参与屏障同步的排名总数
//   tag: 屏障标签（用于匹配屏障操作）
// 返回值：成功或错误码
ncclResult_t bootstrapIntraNodeBarrier(void* commState, int* ranks, int rank, int nranks, int tag);

// 函数：bootstrapIntraNodeAllGather
// 功能：节点内部收集所有 GPU 的数据（All-Gather 操作）
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
// 参数说明：
//   commState: 通信器状态指针
//   ranks: 参与收集的排名数组
//   rank: 当前排名
//   nranks: 参与收集的排名总数
//   allData: 输出缓冲区指针（存储收集的数据）
//   size: 每个数据块的大小（字节数）
// 返回值：成功或错误码
ncclResult_t bootstrapIntraNodeAllGather(void* commState, int* ranks, int rank, int nranks, void* allData, int size);

// 函数：bootstrapIntraNodeBroadcast
// 功能：节点内部广播数据（一对多广播，限于单节点内）
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
// 参数说明：
//   commState: 通信器状态指针
//   ranks: 参与广播的排名数组
//   rank: 当前排名
//   nranks: 参与广播的排名总数
//   root: 根节点标识（谁发送数据）
//   bcastData: 广播数据的缓冲区指针
//   size: 广播数据的大小（字节数）
// 返回值：成功或错误码
ncclResult_t bootstrapIntraNodeBroadcast(void* commState, int* ranks, int rank, int nranks, int root, void* bcastData, int size);

// 函数：bootstrapClose
// 功能：关闭 Bootstrap 连接，释放相关资源
// 返回值：ncclResult_t 类型，表示操作成功或失败的状态码
// 参数说明：
//   commState: 通信器状态指针
// 返回值：成功或错误码
ncclResult_t bootstrapClose(void* commState);

// 条件编译结束：结束头文件保护宏的条件编译块
#endif
