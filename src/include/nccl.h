/*************************************************************************
 * Copyright (c) 2015-2021, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// 头文件保护宏：防止重复包含
#ifndef NCCL_H_
#define NCCL_H_

// CUDA 运行时库头文件（提供 CUDA 基本功能）
#include <cuda_runtime.h>
// CUDA 半精度浮点数头文件（支持 FP16 数据类型）
#include <cuda_fp16.h>
// CUDA 11.0 及以上版本支持 BFloat16 数据类型
#if CUDART_VERSION >= 11000
#include <cuda_bf16.h>
#endif
// CUDA 11.8 及以上版本支持 FP8 数据类型（仅 C++ 模式）
#if __cplusplus && CUDART_VERSION >= 11080
#include <cuda_fp8.h>
#endif

// NCCL 版本号定义（通过构建系统替换）
#define NCCL_MAJOR ${nccl:Major}  // 主版本号
#define NCCL_MINOR ${nccl:Minor}  // 次版本号
#define NCCL_PATCH ${nccl:Patch}  // 补丁版本号
#define NCCL_SUFFIX "${nccl:Suffix}"  // 版本后缀（如 "-dev"）

// NCCL 版本代码（整数值，用于版本比较）
#define NCCL_VERSION_CODE ${nccl:Version}
// NCCL 版本宏辅助函数（将主、次、补丁版本转换为整数）
// 对于 2.8.x 之前的版本：MAJOR * 1000 + MINOR * 100 + PATCH
// 对于 2.8.x 及之后的版本：MAJOR * 10000 + MINOR * 100 + PATCH
#define NCCL_VERSION(X,Y,Z) (((X) <= 2 && (Y) <= 8) ? (X) * 1000 + (Y) * 100 + (Z) : (X) * 10000 + (Y) * 100 + (Z))

// C++ 编译时启用 C 链接（使 C 代码能在 C++ 中编译通过）
#ifdef __cplusplus
extern "C" {
#endif

// 系统头文件：提供整数类型的极限值（如 INT_MIN）
#include <limits.h>

/* Opaque handle to communicator */
// 通信器不透明句柄（指向 ncclComm 结构体的指针，对外隐藏实现细节）
typedef struct ncclComm* ncclComm_t;
// 内存窗口不透明句柄（指向 ncclWindow_vidmem 结构体的指针）
typedef struct ncclWindow_vidmem* ncclWindow_t;
// 空通信器句柄常量（表示无效的通信器）
#define NCCL_COMM_NULL NULL

// NCCL 唯一 ID 字节数（128 字节）
#define NCCL_UNIQUE_ID_BYTES 128
// NCCL 唯一 ID 结构体（用于通信器初始化时的握手）
// 内部存储 128 字节的 ID 数据，对外隐藏实现细节
typedef struct { char internal[NCCL_UNIQUE_ID_BYTES]; } ncclUniqueId;

/* Error type */
// NCCL 结果类型枚举（定义所有可能的返回值）
typedef enum { ncclSuccess                 =  0,  // 操作成功
               ncclUnhandledCudaError      =  1,  // 未处理的 CUDA 错误
               ncclSystemError             =  2,  // 系统错误（如内存不足）
               ncclInternalError           =  3,  // NCCL 内部错误
               ncclInvalidArgument         =  4,  // 无效参数
               ncclInvalidUsage            =  5,  // 无效使用方式
               ncclRemoteError             =  6,  // 远程节点错误
               ncclInProgress              =  7,  // 操作正在进行中（异步状态）
               ncclNumResults              =  8   // 结果类型总数
            } ncclResult_t;

// NCCL 配置未定义整数值（使用 INT_MIN 表示未设置）
#define NCCL_CONFIG_UNDEF_INT INT_MIN
// NCCL 配置未定义指针值（使用 NULL 表示未设置）
#define NCCL_CONFIG_UNDEF_PTR NULL
// NCCL 分裂无颜色值（表示不参与任何分裂组）
#define NCCL_SPLIT_NOCOLOR -1
// NCCL 未定义浮点数值（使用 -1.0f 表示未设置）
#define NCCL_UNDEF_FLOAT -1.0f

/* Window Registration flags */
// 内存窗口注册标志位
#define NCCL_WIN_DEFAULT 0x00        // 默认模式（普通内存窗口）
#define NCCL_WIN_COLL_SYMMETRIC 0x01  // 对称集合通信模式（支持优化的对称内存注册）

// 内存窗口所需的对齐字节数（4KB 对齐，满足硬件要求）
#define NCCL_WIN_REQUIRED_ALIGNMENT 4096

/* NCCL performance policy */
// NCCL 性能策略（控制 CTA 分配策略）
#define NCCL_CTA_POLICY_DEFAULT 0x00    // 默认策略（自动平衡）
#define NCCL_CTA_POLICY_EFFICIENCY 0x01  // 效率优先策略（最大化 GPU 利用率）
#define NCCL_CTA_POLICY_ZERO 0x02        // 零开销策略（最小化启动开销）

/* ncclCommShrink flags*/
// ncclCommShrink 函数标志位
#define NCCL_SHRINK_DEFAULT 0x00  /* 默认模式：收缩父通信器 */
#define NCCL_SHRINK_ABORT 0x01    /* 中止模式：首先终止正在进行的父操作，然后收缩父通信器 */

/* Communicator configuration. Users can assign value to attributes to specify the
 * behavior of a communicator. */
// 通信器配置结构体（用户可通过设置属性来定制通信器的行为）
typedef struct ncclConfig_v22800 {
  /* attributes that users should never touch. */
  // 用户不应触碰的内部属性（用于版本控制和结构体管理）
  size_t size;               // 结构体大小（用于版本兼容性检查）
  unsigned int magic;        // 魔数（用于验证结构体完整性）
  unsigned int version;      // 版本号（用于版本兼容性检查）

  /* attributes that users are able to customize. */
  // 用户可以自定义的属性
  int blocking;              // 阻塞模式（1=阻塞，0=非阻塞）
  int cgaClusterSize;        // Coordinated Grid Array 集群大小
  int minCTAs;               // 每个 SM 的最小 CTA 数量
  int maxCTAs;               // 每个 SM 的最大 CTA 数量
  const char *netName;       // 网络设备名称（如 "mlx5_0"）
  int splitShare;            // 分裂共享标志（控制通信器分裂时的资源共享）
  int trafficClass;          // 流量类别（用于 QoS 优先级控制）
  const char *commName;      // 通信器名称（用于调试和日志）
  int collnetEnable;         // 是否启用 CollNet（集合网络卸载）
  int CTAPolicy;             // CTA 策略（见 NCCL_CTA_POLICY_* 宏）
  int shrinkShare;           // 收缩共享标志（控制通信器收缩时的资源共享）
  int nvlsCTAs;              // NVLS（NVLink SHARP）的 CTA 数量
  int nChannelsPerNetPeer;   // 每个 peer 网络的通道数量
  int nvlinkCentricSched;    // NVLink 中心调度策略
} ncclConfig_t;

/* Config initializer must be assigned to initialize config structure when it is created.
 * Not initialized config will result in NCCL error. */
// 配置初始化宏（必须在创建配置结构体时赋值）
// 未初始化的配置将导致 NCCL 错误
#define NCCL_CONFIG_INITIALIZER {                                       \
  sizeof(ncclConfig_t), /* size */                                      \
  0xcafebeef,           /* magic */                                     \
  NCCL_VERSION(NCCL_MAJOR, NCCL_MINOR, NCCL_PATCH), /* version */       \
  NCCL_CONFIG_UNDEF_INT,                    /* blocking */              \
  NCCL_CONFIG_UNDEF_INT,                    /* cgaClusterSize */        \
  NCCL_CONFIG_UNDEF_INT,                    /* minCTAs */               \
  NCCL_CONFIG_UNDEF_INT,                    /* maxCTAs */               \
  NCCL_CONFIG_UNDEF_PTR,                    /* netName */               \
  NCCL_CONFIG_UNDEF_INT,                    /* splitShare */            \
  NCCL_CONFIG_UNDEF_INT,                    /* trafficClass */          \
  NCCL_CONFIG_UNDEF_PTR,                    /* commName */              \
  NCCL_CONFIG_UNDEF_INT,                    /* collnetEnable */         \
  NCCL_CONFIG_UNDEF_INT,                    /* CTAPolicy */             \
  NCCL_CONFIG_UNDEF_INT,                    /* shrinkShare */           \
  NCCL_CONFIG_UNDEF_INT,                    /* nvlsCTAs */              \
  NCCL_CONFIG_UNDEF_INT,                    /* nChannelsPerNetPeer */   \
  NCCL_CONFIG_UNDEF_INT,                    /* nvlinkCentricSched */    \
}

/* This struct will be used by ncclGroupSimulateEnd() API to query information about simulation. */
// 模拟信息结构体（由 ncclGroupSimulateEnd() API 使用，用于查询模拟结果）
typedef struct ncclSimInfo_v22200 {
    size_t size;              // 结构体大小（用于版本兼容性检查）
    unsigned int magic;        // 魔数（用于验证结构体完整性）
    unsigned int version;      // 版本号（用于版本兼容性检查）
    float estimatedTime;       // 估计的执行时间（微秒）
} ncclSimInfo_t;

/* NCCL_SIM_INFO_INITIALIZER must be assigned to initialize simInfo structure when it is created.
 * Not initialized simInfo will result in NCCL error. */
// 模拟信息初始化宏（必须在创建结构体时赋值）
// 未初始化的 simInfo 将导致 NCCL 错误
#define NCCL_SIM_INFO_INITIALIZER {                                         \
  sizeof(ncclSimInfo_t),                            /* size */              \
  0x74685283,                                       /* magic */             \
  NCCL_VERSION(NCCL_MAJOR, NCCL_MINOR, NCCL_PATCH), /* version */           \
  NCCL_UNDEF_FLOAT                                  /* estimated time */    \
}

/* NCCL malloc and free function for all types of NCCL optimizations
 * (e.g. user buffer registration). The actual allocated size might
 * be larger than requested due to granularity requirement. */
// NCCL 内存分配和释放函数（用于所有类型的 NCCL 优化，如用户缓冲区注册）
// 实际分配的大小可能大于请求的大小（由于粒度要求）
// 分配内存（主机端）
ncclResult_t  ncclMemAlloc(void** ptr, size_t size);
// 分配内存（支持 Fortran）
ncclResult_t pncclMemAlloc(void** ptr, size_t size);

// 释放内存（主机端）
ncclResult_t  ncclMemFree(void *ptr);
// 释放内存（支持 Fortran）
ncclResult_t pncclMemFree(void *ptr);

/* Return the NCCL_VERSION_CODE of the NCCL library in the supplied integer.
 * This integer is coded with the MAJOR, MINOR and PATCH level of the
 * NCCL library
 */
// 获取 NCCL 库的版本代码（以整数形式返回）
// 该整数编码了 NCCL 库的主版本号、次版本号和补丁级别
// 获取版本（主机端）
ncclResult_t  ncclGetVersion(int *version);
// 获取版本（支持 Fortran）
ncclResult_t pncclGetVersion(int *version);

/* Generates an Id to be used in ncclCommInitRank. ncclGetUniqueId should be
 * called once and the Id should be distributed to all ranks in the
 * communicator before calling ncclCommInitRank. */
// 生成唯一 ID（用于 ncclCommInitRank）
// ncclGetUniqueId 应该只调用一次，然后将 ID 分发给通信器中的所有 rank
// 在调用 ncclCommInitRank 之前必须完成 ID 分发
// 生成唯一 ID（主机端）
ncclResult_t  ncclGetUniqueId(ncclUniqueId* uniqueId);
// 生成唯一 ID（支持 Fortran）
ncclResult_t pncclGetUniqueId(ncclUniqueId* uniqueId);

/* Create a new communicator (multi thread/process version) with a configuration
 * set by users. */
// 创建新通信器（多线程/多进程版本），支持用户自定义配置
ncclResult_t  ncclCommInitRankConfig(ncclComm_t* comm, int nranks, ncclUniqueId commId, int rank, ncclConfig_t* config);
ncclResult_t pncclCommInitRankConfig(ncclComm_t* comm, int nranks, ncclUniqueId commId, int rank, ncclConfig_t* config);

/* Creates a new communicator (multi thread/process version).
 * rank must be between 0 and nranks-1 and unique within a communicator clique.
 * Each rank is associated to a CUDA device, which has to be set before calling
 * ncclCommInitRank.
 * ncclCommInitRank implicitly syncronizes with other ranks, so it must be
 * called by different threads/processes or use ncclGroupStart/ncclGroupEnd. */
// 创建新通信器（多线程/多进程版本）
// rank 必须在 0 到 nranks-1 之间，并且在通信器集群中唯一
// 每个 rank 关联一个 CUDA 设备，必须在调用 ncclCommInitRank 之前设置
// ncclCommInitRank 隐式地与其他 rank 同步，因此必须由不同的线程/进程调用
// 或者使用 ncclGroupStart/ncclGroupEnd
// 初始化通信器（主机端）
ncclResult_t  ncclCommInitRank(ncclComm_t* comm, int nranks, ncclUniqueId commId, int rank);
// 初始化通信器（支持 Fortran）
ncclResult_t pncclCommInitRank(ncclComm_t* comm, int nranks, ncclUniqueId commId, int rank);

/* Creates a clique of communicators (single process version).
 * This is a convenience function to create a single-process communicator clique.
 * Returns an array of ndev newly initialized communicators in comm.
 * comm should be pre-allocated with size at least ndev*sizeof(ncclComm_t).
 * If devlist is NULL, the first ndev CUDA devices are used.
 * Order of devlist defines user-order of processors within the communicator. */
// 创建通信器集群（单进程版本）
// 这是一个便捷函数，用于创建单进程通信器集群
// 在 comm 中返回 ndev 个新初始化的通信器数组
// comm 应预分配至少 ndev*sizeof(ncclComm_t) 的大小
// 如果 devlist 为 NULL，则使用前 ndev 个 CUDA 设备
// devlist 的顺序定义了通信器内处理器的用户顺序
// 初始化所有通信器（主机端）
ncclResult_t  ncclCommInitAll(ncclComm_t* comm, int ndev, const int* devlist);
// 初始化所有通信器（支持 Fortran）
ncclResult_t pncclCommInitAll(ncclComm_t* comm, int ndev, const int* devlist);

/* Finalize a communicator. ncclCommFinalize flushes all issued communications,
 * and marks communicator state as ncclInProgress. The state will change to ncclSuccess
 * when the communicator is globally quiescent and related resources are freed; then,
 * calling ncclCommDestroy can locally free the rest of the resources (e.g. communicator
 * itself) without blocking. */
// 终结通信器。ncclCommFinalize 刷新所有已发出的通信，
// 并将通信器状态标记为 ncclInProgress（进行中）
// 当通信器全局静止且相关资源被释放时，状态将变为 ncclSuccess
// 此时调用 ncclCommDestroy 可以本地释放其余资源（如通信器本身）而无需阻塞
// 终结通信器（主机端）
ncclResult_t  ncclCommFinalize(ncclComm_t comm);
// 终结通信器（支持 Fortran）
ncclResult_t pncclCommFinalize(ncclComm_t comm);

/* Frees local resources associated with communicator object. */
// 释放与通信器对象关联的本地资源
// 销毁通信器（主机端）
ncclResult_t  ncclCommDestroy(ncclComm_t comm);
// 销毁通信器（支持 Fortran）
ncclResult_t pncclCommDestroy(ncclComm_t comm);

/* Frees resources associated with communicator object and aborts any operations
 * that might still be running on the device. */
// 释放与通信器对象关联的资源，并中止设备上可能仍在运行的任何操作
// 中止通信器（主机端）
ncclResult_t  ncclCommAbort(ncclComm_t comm);
// 中止通信器（支持 Fortran）
ncclResult_t pncclCommAbort(ncclComm_t comm);

/* Creates one or more communicators from an existing one.
 * Ranks with the same color will end up in the same communicator.
 * Within the new communicator, key will be used to order ranks.
 * NCCL_SPLIT_NOCOLOR as color will indicate the rank will not be part of any group
 * and will therefore return a NULL communicator.
 * If config is NULL, the new通信器 will inherit the original communicator's configuration*/
// 从现有通信器创建一个或多个新通信器（通信器分裂）
// 具有相同 color 的 rank 将最终位于同一个通信器中
// 在新通信器中，key 将用于对 rank 进行排序
// NCCL_SPLIT_NOCOLOR 作为 color 表示该 rank 不属于任何组，将返回 NULL 通信器
// 如果 config 为 NULL，新通信器将继承原通信器的配置
// 分裂通信器（主机端）
ncclResult_t  ncclCommSplit(ncclComm_t comm, int color, int key, ncclComm_t *newcomm, ncclConfig_t* config);
// 分裂通信器（支持 Fortran）
ncclResult_t pncclCommSplit(ncclComm_t comm, int color, int key, ncclComm_t *newcomm, ncclConfig_t* config);

/* Shrink existing communicator.
 * Ranks in excludeRanksList will be removed form the existing communicator.
 * Within the new communicator, ranks will be re-ordered to fill the gap of removed ones.
 * If config is NULL, the new communicator will inherit the original communicator's configuration
 * The flag enables NCCL to adapt to various states of the parent communicator, see NCCL_SHRINK flags.*/
// 收缩现有通信器（移除指定的 rank）
// excludeRanksList 中的 rank 将从现有通信器中移除
// 在新通信器中，rank 将重新排序以填充被移除 rank 的空缺
// 如果 config 为 NULL，新通信器将继承原通信器的配置
// 标志使 NCCL 能够适应父通信器的各种状态，参见 NCCL_SHRINK 标志
// 收缩通信器（主机端）
ncclResult_t  ncclCommShrink(ncclComm_t comm, int* excludeRanksList, int excludeRanksCount, ncclComm_t* newcomm, ncclConfig_t* config, int shrinkFlags);
// 收缩通信器（支持 Fortran）
ncclResult_t pncclCommShrink(ncclComm_t comm, int* excludeRanksList, int excludeRanksCount, ncclComm_t* newcomm, ncclConfig_t* config, int shrinkFlags);

/* Creates a new communicator (multi thread/process version), similar to ncclCommInitRankConfig.
 * Allows to use more than one ncclUniqueId (up to one per rank), indicated by nId, to accelerate the init operation.
 * The number of ncclUniqueIds and their order must be the same for every rank.
 */
// 创建新通信器（多线程/多进程版本），类似于 ncclCommInitRankConfig
// 允许使用多个 ncclUniqueId（每个 rank 最多一个），由 nId 指示，以加速初始化操作
// ncclUniqueId 的数量和顺序在每个 rank 上必须相同
// 使用可扩展 ID 初始化通信器（主机端）
ncclResult_t ncclCommInitRankScalable(ncclComm_t* newcomm, int nranks, int myrank, int nId, ncclUniqueId* commIds, ncclConfig_t* config);
// 使用可扩展 ID 初始化通信器（支持 Fortran）
ncclResult_t pncclCommInitRankScalable(ncclComm_t* newcomm, int nranks, int myrank, int nId, ncclUniqueId* commIds, ncclConfig_t* config);

/* Returns a string for each error code. */
// 返回每个错误代码对应的字符串描述
// 获取错误字符串（主机端）
const char*  ncclGetErrorString(ncclResult_t result);
// 获取错误字符串（支持 Fortran）
const char* pncclGetErrorString(ncclResult_t result);

/* Returns a human-readable message of the last error that occurred. */
// 返回发生的最后一个错误的可读消息
// 获取最后的错误消息（主机端）
const char*  ncclGetLastError(ncclComm_t comm);
// 获取最后的错误消息（支持 Fortran）
const char* pncclGetLastError(ncclComm_t comm);

/* Reload environment variables that determine logging. */
// 重新加载确定日志记录的环境变量
// 已弃用：ncclResetDebugInit 不作为 NCCL API 的一部分支持，将在未来移除
__attribute__ ((deprecated("ncclResetDebugInit is not supported as part of the NCCL API and will be removed in the future")))
void  ncclResetDebugInit();
// 已弃用：pncclResetDebugInit 不作为 NCCL API 的一部分支持，将在未来移除
__attribute__ ((deprecated("pncclResetDebugInit is not supported as part of the NCCL API and will be removed in the future")))
void pncclResetDebugInit();

/* Checks whether the comm has encountered any asynchronous errors */
// 检查通信器是否遇到任何异步错误
// 检查异步错误（主机端）
ncclResult_t  ncclCommGetAsyncError(ncclComm_t comm, ncclResult_t *asyncError);
// 检查异步错误（支持 Fortran）
ncclResult_t pncclCommGetAsyncError(ncclComm_t comm, ncclResult_t *asyncError);

/* Gets the number of ranks in the communicator clique. */
// 获取通信器集群中的 rank 数量
// 获取 rank 数量（主机端）
ncclResult_t  ncclCommCount(const ncclComm_t comm, int* count);
// 获取 rank 数量（支持 Fortran）
ncclResult_t pncclCommCount(const ncclComm_t comm, int* count);

/* Returns the cuda device number associated with the communicator. */
// 返回与通信器关联的 CUDA 设备号
// 获取 CUDA 设备号（主机端）
ncclResult_t  ncclCommCuDevice(const ncclComm_t comm, int* device);
// 获取 CUDA 设备号（支持 Fortran）
ncclResult_t pncclCommCuDevice(const ncclComm_t comm, int* device);

/* Returns the user-ordered "rank" associated with the communicator. */
// 返回与通信器关联的用户排序的 rank
// 获取用户 rank（主机端）
ncclResult_t  ncclCommUserRank(const ncclComm_t comm, int* rank);
// 获取用户 rank（支持 Fortran）
ncclResult_t pncclCommUserRank(const ncclComm_t comm, int* rank);

/* Register CUDA buffer for zero-copy operation */
// 注册 CUDA 缓冲区以实现零拷贝操作
// 注册缓冲区（主机端）
ncclResult_t  ncclCommRegister(const ncclComm_t comm, void* buff, size_t size, void** handle);
// 注册缓冲区（支持 Fortran）
ncclResult_t pncclCommRegister(const ncclComm_t comm, void* buff, size_t size, void** handle);

/* Deregister CUDA buffer */
// 注销 CUDA 缓冲区
// 注销缓冲区（主机端）
ncclResult_t  ncclCommDeregister(const ncclComm_t comm, void* handle);
// 注销缓冲区（支持 Fortran）
ncclResult_t pncclCommDeregister(const ncclComm_t comm, void* handle);

/* Register memory window  */
// 注册内存窗口（用于对称内存通信）
// 注册内存窗口（主机端）
ncclResult_t  ncclCommWindowRegister(ncclComm_t comm, void* buff, size_t size, ncclWindow_t* win, int winFlags);
// 注册内存窗口（支持 Fortran）
ncclResult_t pncclCommWindowRegister(ncclComm_t comm, void* buff, size_t size, ncclWindow_t* win, int winFlags);

/* Deregister symmetric memory */
// 注销对称内存
// 注销内存窗口（主机端）
ncclResult_t  ncclCommWindowDeregister(ncclComm_t comm, ncclWindow_t win);
// 注销内存窗口（支持 Fortran）
ncclResult_t pncclCommWindowDeregister(ncclComm_t comm, ncclWindow_t win);

/* Reduction operation selector */
// 归约操作选择器枚举
// 占位枚举（用于计算 ncclMaxRedOp）
typedef enum { ncclNumOps_dummy = 5 } ncclRedOp_dummy_t;
// 归约操作类型枚举（定义所有内置的归约操作）
typedef enum { ncclSum        = 0,        // 求和
               ncclProd       = 1,        // 求积
               ncclMax        = 2,        // 求最大值
               ncclMin        = 3,        // 求最小值
               ncclAvg        = 4,        // 求平均值
               /* ncclNumOps: The number of built-in ncclRedOp_t values. Also
                * serves as the least possible value for dynamic ncclRedOp_t's
                * as constructed by ncclRedOpCreate*** functions. */
               // 内置 ncclRedOp_t 值的数量，也是动态 ncclRedOp_t 的最小可能值
               ncclNumOps     = 5,
               /* ncclMaxRedOp: The largest valid value for ncclRedOp_t.
                * It is defined to be the largest signed value (since compilers
                * are permitted to use signed enums) that won't grow
                * sizeof(ncclRedOp_t) when compared to previous NCCL versions to
                * maintain ABI compatibility. */
               // ncclRedOp_t 的最大有效值
               // 定义为最大的有符号值（编译器允许使用有符号枚举），
               // 这样在与以前 NCCL 版本比较时不会增加 sizeof(ncclRedOp_t)
               // 以保持 ABI 兼容性
               ncclMaxRedOp   = 0x7fffffff>>(32-8*sizeof(ncclRedOp_dummy_t))
             } ncclRedOp_t;

/* Data types */
// 数据类型枚举（定义所有支持的数据类型）
typedef enum { ncclInt8       = 0,        // 8 位有符号整数
               ncclChar       = 0,        // 字符型（8 位有符号整数）
               ncclUint8      = 1,        // 8 位无符号整数
               ncclInt32      = 2,        // 32 位有符号整数
               ncclInt        = 2,        // 整数（32 位有符号整数）
               ncclUint32     = 3,        // 32 位无符号整数
               ncclInt64      = 4,        // 64 位有符号整数
               ncclUint64     = 5,        // 64 位无符号整数
               ncclFloat16    = 6,        // 半精度浮点数（FP16）
               ncclHalf       = 6,        // 半精度浮点数（FP16，别名）
               ncclFloat32    = 7,        // 单精度浮点数（FP32）
               ncclFloat      = 7,        // 浮点数（FP32，别名）
               ncclFloat64    = 8,        // 双精度浮点数（FP64）
               ncclDouble     = 8,        // 双精度浮点数（FP64，别名）
               ncclBfloat16   = 9,        // BFloat16（脑浮点 16）
               ncclFloat8e4m3 = 10,       // FP8 E4M3 格式（指数 4 位，尾数 3 位）
               ncclFloat8e5m2 = 11,       // FP8 E5M2 格式（指数 5 位，尾数 2 位）
               ncclNumTypes   = 12        // 数据类型总数
} ncclDataType_t;

/* ncclScalarResidence_t: Location and dereferencing logic for scalar arguments. */
// 标量驻留类型：定义标量参数的位置和解引用逻辑
typedef enum {
  /* ncclScalarDevice: The scalar is in device-visible memory and will be
   * dereferenced while the collective is running. */
  // 标量在设备可见内存中，将在集合操作运行时解引用
  ncclScalarDevice = 0,

  /* ncclScalarHostImmediate: The scalar is in host-visible memory and will be
   * dereferenced before the ncclRedOpCreate***() function returns. */
  // 标量在主机可见内存中，将在 ncclRedOpCreate***() 函数返回前解引用
  ncclScalarHostImmediate = 1
} ncclScalarResidence_t;

/*
 * ncclRedOpCreatePreMulSum
 *
 * Creates a new reduction operator which pre-multiplies input values by a given
 * scalar locally before reducing them with peer values via summation. For use
 * only with collectives launched against *comm* and *datatype*. The
 * *residence* argument indicates how/when the memory pointed to by *scalar*
 * will be dereferenced. Upon return, the newly created operator's handle
 * is stored in *op*.
 */
// 创建预乘求和归约操作符
// 该操作符在本地将输入值乘以给定标量，然后通过求和与 peer 值进行归约
// 仅用于针对 *comm* 和 *datatype* 启动的集合操作
// *residence* 参数指示 *scalar* 指向的内存将如何/何时被解引用
// 返回时，新创建的操作符句柄存储在 *op* 中
// 创建预乘求和操作符（主机端）
ncclResult_t  ncclRedOpCreatePreMulSum(ncclRedOp_t *op, void *scalar, ncclDataType_t datatype, ncclScalarResidence_t residence, ncclComm_t comm);
// 创建预乘求和操作符（支持 Fortran）
ncclResult_t pncclRedOpCreatePreMulSum(ncclRedOp_t *op, void *scalar, ncclDataType_t datatype, ncclScalarResidence_t residence, ncclComm_t comm);

/*
 * ncclRedOpDestroy
 *
 * Destroys the reduction operator *op*. The operator must have been created by
 * ncclRedOpCreatePreMul with the matching communicator *comm*. An operator may be
 * destroyed as soon as the last NCCL function which is given that operator returns.
 */
// 销毁归约操作符
// 销毁由 ncclRedOpCreatePreMul 创建的操作符 *op*
// 操作符必须是由匹配的通信器 *comm* 创建的
// 当给定该操作符的最后一个 NCCL 函数返回后，可以立即销毁操作符
// 销毁归约操作符（主机端）
ncclResult_t ncclRedOpDestroy(ncclRedOp_t op, ncclComm_t comm);
// 销毁归约操作符（支持 Fortran）
ncclResult_t pncclRedOpDestroy(ncclRedOp_t op, ncclComm_t comm);

/*
 * Collective communication operations
 *
 * Collective communication operations must be called separately for each
 * communicator in a communicator clique.
 *
 * They return when operations have been enqueued on the CUDA stream.
 *
 * Since they may perform inter-CPU synchronization, each call has to be done
 * from a different thread or process, or need to use Group Semantics (see
 * below).
 */
/*
 * 集合通信操作
 *
 * 集合通信操作必须对通信器集群中的每个通信器分别调用。
 *
 * 它们在操作已入队到 CUDA 流时返回。
 *
 * 由于它们可能执行 CPU 间同步，每次调用必须由不同的线程或进程完成，
 * 或者需要使用组语义（见下文）。
 */

/*
 * Reduce
 *
 * Reduces data arrays of length count in sendbuff into recvbuff using op
 * operation.
 * recvbuff may be NULL on all calls except for root device.
 * root is the rank (not the CUDA device) where data will reside after the
 * operation is complete.
 *
 * In-place operation will happen if sendbuff == recvbuff.
 */
/*
 * Reduce（归约）
 *
 * 使用 op 操作将长度为 count 的数据数组从 sendbuff 归约到 recvbuff
 * recvbuff 在所有非 root 设备上可以为 NULL
 * root 是操作完成后数据所在的 rank（不是 CUDA 设备号）
 *
 * 如果 sendbuff == recvbuff，则进行就地操作
 */
// 归约操作（主机端）
ncclResult_t  ncclReduce(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
    ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream);
// 归约操作（支持 Fortran）
ncclResult_t pncclReduce(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
    ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream);

/*
 * (deprecated) Broadcast (in-place)
 *
 * Copies count values from root to all other devices.
 * root is the rank (not the CUDA device) where data resides before the
 * operation is started.
 *
 * This operation is implicitely in place.
 */
/*
 * Broadcast（广播）- 已弃用（就地版本）
 *
 * 从 root 复制 count 个值到所有其他设备
 * root 是操作开始前数据所在的 rank（不是 CUDA 设备号）
 *
 * 此操作隐式地为就地操作
 */
// 广播操作（主机端）- 已弃用
ncclResult_t  ncclBcast(void* buff, size_t count, ncclDataType_t datatype, int root,
    ncclComm_t comm, cudaStream_t stream);
// 广播操作（支持 Fortran）- 已弃用
ncclResult_t pncclBcast(void* buff, size_t count, ncclDataType_t datatype, int root,
    ncclComm_t comm, cudaStream_t stream);

/*
 * Broadcast
 *
 * Copies count values from root to all other devices.
 * root is the rank (not the CUDA device) where data resides before the
 * operation is started.
 *
 * In-place operation will happen if sendbuff == recvbuff.
 */
/*
 * Broadcast（广播）
 *
 * 从 root 复制 count 个值到所有其他设备
 * root 是操作开始前数据所在的 rank（不是 CUDA 设备号）
 *
 * 如果 sendbuff == recvbuff，则进行就地操作
 */
// 广播操作（主机端）
ncclResult_t  ncclBroadcast(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, int root,
    ncclComm_t comm, cudaStream_t stream);
// 广播操作（支持 Fortran）
ncclResult_t pncclBroadcast(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, int root,
    ncclComm_t comm, cudaStream_t stream);

/*
 * All-Reduce
 *
 * Reduces data arrays of length count in sendbuff using op operation, and
 * leaves identical copies of result on each recvbuff.
 *
 * In-place operation will happen if sendbuff == recvbuff.
 */
/*
 * All-Reduce（全归约）
 *
 * 使用 op 操作对长度为 count 的数据数组进行归约，
 * 并在每个 recvbuff 上留下相同的结果副本
 *
 * 如果 sendbuff == recvbuff，则进行就地操作
 */
// 全归约操作（主机端）
ncclResult_t  ncclAllReduce(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream);
// 全归约操作（支持 Fortran）
ncclResult_t pncclAllReduce(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream);

/*
 * Reduce-Scatter
 *
 * Reduces data in sendbuff using op operation and leaves reduced result
 * scattered over the devices so that recvbuff on rank i will contain the i-th
 * block of the result.
 * Assumes sendcount is equal to nranks*recvcount, which means that sendbuff
 * should have a size of at least nranks*recvcount elements.
 *
 * In-place operations will happen if recvbuff == sendbuff + rank * recvcount.
 */
/*
 * Reduce-Scatter（归约-分散）
 *
 * 使用 op 操作归约 sendbuff 中的数据，并将归约结果分散到各设备
 * 使得 rank i 上的 recvbuff 包含结果的第 i 块
 * 假设 sendcount 等于 nranks*recvcount，这意味着 sendbuff 的大小至少为 nranks*recvcount 个元素
 *
 * 如果 recvbuff == sendbuff + rank * recvcount，则进行就地操作
 */
// 归约-分散操作（主机端）
ncclResult_t  ncclReduceScatter(const void* sendbuff, void* recvbuff,
    size_t recvcount, ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm,
    cudaStream_t stream);
// 归约-分散操作（支持 Fortran）
ncclResult_t pncclReduceScatter(const void* sendbuff, void* recvbuff,
    size_t recvcount, ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm,
    cudaStream_t stream);

/*
 * All-Gather
 *
 * Each device gathers sendcount values from other GPUs into recvbuff,
 * receiving data from rank i at offset i*sendcount.
 * Assumes recvcount is equal to nranks*sendcount, which means that recvbuff
 * should have a size of at least nranks*sendcount elements.
 *
 * In-place operations will happen if sendbuff == recvbuff + rank * sendcount.
 */
/*
 * All-Gather（全收集）
 *
 * 每个设备从其他 GPU 收集 sendcount 个值到 recvbuff，
 * 从 rank i 接收数据，偏移量为 i*sendcount
 * 假设 recvcount 等于 nranks*sendcount，这意味着 recvbuff 的大小至少为 nranks*sendcount 个元素
 *
 * 如果 sendbuff == recvbuff + rank * sendcount，则进行就地操作
 */
// 全收集操作（主机端）
ncclResult_t  ncclAllGather(const void* sendbuff, void* recvbuff, size_t sendcount,
    ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream);
// 全收集操作（支持 Fortran）
ncclResult_t pncclAllGather(const void* sendbuff, void* recvbuff, size_t sendcount,
    ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream);

/*
 * All-to-All
 *
 * Each device sends count values to all other devices and receives count values
 * from all other devices. Data to send to destination rank j is taken from
 * sendbuff+j*count and data received from source rank i is placed at
 * recvbuff+i*count.
 */
/*
 * All-to-All（全对全）
 *
 * 每个设备向所有其他设备发送 count 个值，并从所有其他设备接收 count 个值
 * 发送到目标 rank j 的数据取自 sendbuff+j*count
 * 从源 rank i 接收的数据放置在 recvbuff+i*count
 */
// 全对全操作（主机端）
ncclResult_t  ncclAlltoAll(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream);
// 全对全操作（支持 Fortran）
ncclResult_t pncclAlltoAll(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream);

/*
 * Gather
 *
 * Each rank sends count elements from sendbuff to the root rank.
 * On the root rank, data from rank i is placed at recvbuff + i*count.
 * On non-root ranks, recvbuff is not used.
 * root is the rank where data will be gathered.
 *
 * In-place operations will happen if sendbuff == recvbuff + root * count.
 */
/*
 * Gather（收集）
 *
 * 每个 rank 从 sendbuff 发送 count 个元素到 root rank
 * 在 root rank 上，来自 rank i 的数据放置在 recvbuff + i*count
 * 在非 root rank 上，不使用 recvbuff
 * root 是数据将聚集到的 rank
 *
 * 如果 sendbuff == recvbuff + root * count，则进行就地操作
 */
// 收集操作（主机端）
ncclResult_t  ncclGather(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, int root, ncclComm_t comm, cudaStream_t stream);
// 收集操作（支持 Fortran）
ncclResult_t  ncclGather(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, int root, ncclComm_t comm, cudaStream_t stream);

/*
 * Scatter
 *
 * On the root rank, count elements from sendbuff+i*count are sent to rank i.
 * On non-root ranks, sendbuff is not used.
 * Each rank receives count elements into recvbuff.
 * root is the rank that will distribute the data.
 *
 * In-place operations will happen if recvbuff == sendbuff + root * count.
 */
/*
 * Scatter（分散）
 *
 * 在 root rank 上，sendbuff+i*count 中的 count 个元素被发送到 rank i
 * 在非 root rank 上，不使用 sendbuff
 * 每个 rank 在 recvbuff 中接收 count 个元素
 * root 是将分发数据的 rank
 *
 * 如果 recvbuff == sendbuff + root * count，则进行就地操作
 */
// 分散操作（主机端）
ncclResult_t  ncclScatter(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, int root, ncclComm_t comm, cudaStream_t stream);
// 分散操作（支持 Fortran）
ncclResult_t  ncclScatter(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, int root, ncclComm_t comm, cudaStream_t stream);

/*
 * Send
 *
 * Send data from sendbuff to rank peer.
 *
 * Rank peer needs to call ncclRecv with the same datatype and the same count from this
 * rank.
 *
 * This operation is blocking for the GPU. If multiple ncclSend and ncclRecv operations
 * need to progress concurrently to complete, they must be fused within a ncclGroupStart/
 * ncclGroupEnd section.
 */
/*
 * Send（发送）
 *
 * 从 sendbuff 发送数据到 rank peer
 *
 * Rank peer 必须调用 ncclRecv，使用相同的数据类型和相同的 count 与此 rank 匹配
 *
 * 此操作对 GPU 是阻塞的。如果多个 ncclSend 和 ncclRecv 操作需要并发完成，
 * 它们必须在 ncclGroupStart/ncclGroupEnd 部分中融合
 */
// 发送操作（支持 Fortran）
ncclResult_t pncclRecv(void* recvbuff, size_t count, ncclDataType_t datatype, int peer,
    ncclComm_t comm, cudaStream_t stream);
// 发送操作（主机端）
ncclResult_t  ncclSend(const void* sendbuff, size_t count, ncclDataType_t datatype, int peer,
    ncclComm_t comm, cudaStream_t stream);

/*
 * Receive
 *
 * Receive data from rank peer into recvbuff.
 *
 * Rank peer needs to call ncclSend with the same datatype and the same count to this
 * rank.
 *
 * This operation is blocking for the GPU. If multiple ncclSend and ncclRecv operations
 * need to progress concurrently to complete, they must be fused within a ncclGroupStart/
 * ncclGroupEnd section.
 */
/*
 * Receive（接收）
 *
 * 从 rank peer 接收数据到 recvbuff
 *
 * Rank peer 必须调用 ncclSend，使用相同的数据类型和相同的 count 与此 rank 匹配
 *
 * 此操作对 GPU 是阻塞的。如果多个 ncclSend 和 ncclRecv 操作需要并发完成，
 * 它们必须在 ncclGroupStart/ncclGroupEnd 部分中融合
 */
// 接收操作（主机端）
ncclResult_t  ncclRecv(void* recvbuff, size_t count, ncclDataType_t datatype, int peer,
    ncclComm_t comm, cudaStream_t stream);
// 接收操作（支持 Fortran）
ncclResult_t pncclRecv(void* recvbuff, size_t count, ncclDataType_t datatype, int peer,
    ncclComm_t comm, cudaStream_t stream);

/*
 * Group semantics
 *
 * When managing multiple GPUs from a single thread, and since NCCL collective
 * calls may perform inter-CPU synchronization, we need to "group" calls for
 * different ranks/devices into a single call.
 *
 * Grouping NCCL calls as being part of the same collective operation is done
 * using ncclGroupStart and ncclGroupEnd. ncclGroupStart will enqueue all
 * collective calls until the ncclGroupEnd call, which will wait for all calls
 * to be complete. Note that for collective communication, ncclGroupEnd only
 * guarantees that the operations are enqueued on the streams, not that
 * the operation is effectively done.
 *
 * Both collective communication and ncclCommInitRank can be used in conjunction
 * of ncclGroupStart/ncclGroupEnd, but not together.
 *
 * Group semantics also allow to fuse multiple operations on the same device
 * to improve performance (for aggregated collective calls), or to permit
 * concurrent progress of multiple send/receive operations.
 */
/*
 * 组语义
 *
 * 当从单个线程管理多个 GPU 时，由于 NCCL 集合调用可能会执行 CPU 间同步，
 * 我们需要将不同 rank/设备的调用"分组"到单个调用中。
 *
 * 使用 ncclGroupStart 和 ncclGroupEnd 将 NCCL 调用分组为同一集合操作的一部分。
 * ncclGroupStart 将入队所有集合调用，直到 ncclGroupEnd 调用，后者将等待所有调用完成。
 * 注意：对于集合通信，ncclGroupEnd 仅保证操作已入队到流中，不保证操作实际完成。
 *
 * 集合通信和 ncclCommInitRank 可以与 ncclGroupStart/ncclGroupEnd 结合使用，
 * 但不能一起使用。
 *
 * 组语义还允许融合同一设备上的多个操作以提高性能（对于聚合的集合调用），
 * 或允许多个发送/接收操作的并发进行。
 */

/*
 * Group Start
 *
 * Start a group call. All calls to NCCL until ncclGroupEnd will be fused into
 * a single NCCL operation. Nothing will be started on the CUDA stream until
 * ncclGroupEnd.
 */
/*
 * 组开始
 *
 * 启动组调用。在 ncclGroupEnd 之前的所有 NCCL 调用将融合为单个 NCCL 操作。
 * 在 ncclGroupEnd 之前，CUDA 流上不会启动任何操作。
 */
// 组开始（主机端）
ncclResult_t  ncclGroupStart();
// 组开始（支持 Fortran）
ncclResult_t pncclGroupStart();

/*
 * Group End
 *
 * End a group call. Start a fused NCCL operation consisting of all calls since
 * ncclGroupStart. Operations on the CUDA stream depending on the NCCL operations
 * need to be called after ncclGroupEnd.
 */
/*
 * 组结束
 *
 * 结束组调用。启动由 ncclGroupStart 之后所有调用组成的融合 NCCL 操作。
 * 依赖于 NCCL 操作的 CUDA 流操作需要在 ncclGroupEnd 之后调用。
 */
// 组结束（主机端）
ncclResult_t  ncclGroupEnd();
// 组结束（支持 Fortran）
ncclResult_t pncclGroupEnd();

/*
 * Group Simulate End
 *
 * Simulate a ncclGroupEnd() call and return NCCL's simulation info in a struct.
 */
/*
 * 组模拟结束
 *
 * 模拟 ncclGroupEnd() 调用，并在结构体中返回 NCCL 的模拟信息。
 */
// 组模拟结束（主机端）
ncclResult_t  ncclGroupSimulateEnd(ncclSimInfo_t* simInfo);
// 组模拟结束（支持 Fortran）
ncclResult_t pncclGroupSimulateEnd(ncclSimInfo_t* simInfo);

// 结束 C 链接（仅 C++ 模式）
#ifdef __cplusplus
} // end extern "C"
#endif

// 结束头文件保护
#endif // end include guard
