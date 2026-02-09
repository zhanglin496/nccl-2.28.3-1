/*************************************************************************
 * Copyright (c) 2015-2023, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// 包含参数检查头文件，因为我们需要访问 comm 结构体成员
#include "argcheck.h" // Need some checks here since we access comm
// 包含集合操作相关的内部定义
#include "collectives.h"
// 包含入队操作相关的函数和结构体
#include "enqueue.h"
// 包含 NCCL 公共 API 定义
#include "nccl.h"
// 包含 NVTX（NVIDIA Tools Extension）性能分析负载模式定义
#include "nvtx_payload_schemas.h"

// 集合操作类型转换为字符串的辅助函数
// 参数 fn: 集合操作类型枚举值
// 返回值: 操作类型的字符串表示，用于调试和日志输出
const char* ncclFuncToString(ncclFunc_t fn) {
  switch (fn) {                                         // 根据集合操作类型判断
  case ncclFuncAllGather: return "AllGather";          // AllGather：所有 rank 收集数据到每个 rank
  case ncclFuncAllReduce: return "AllReduce";          // AllReduce：所有 rank 的数据规约并广播
  case ncclFuncAlltoAll: return "AlltoAll";            // AlltoAll：每个 rank 向所有其他 rank 发送数据
  case ncclFuncBroadcast: return "Broadcast";          // Broadcast：根 rank 广播数据到所有 rank
  case ncclFuncGather: return "Gather";                // Gather：根 rank 从所有 rank 收集数据
  case ncclFuncRecv: return "Recv";                    // Recv：接收点对点消息
  case ncclFuncReduce: return "Reduce";                // Reduce：所有 rank 的数据规约到根 rank
  case ncclFuncReduceScatter: return "ReduceScatter";  // ReduceScatter：先规约再分散到各 rank
  case ncclFuncScatter: return "Scatter";              // Scatter：根 rank 分发数据到所有 rank
  case ncclFuncSendRecv: return "SendRecv";            // SendRecv：发送和接收组合操作
  case ncclFuncSend: return "Send";                    // Send：发送点对点消息
  default: return "Invalid";                           // 无效的操作类型
  }
}

// 设备端规约操作类型转换为字符串的辅助函数
// 参数 op: 设备端规约操作类型枚举值
// 返回值: 规约操作的字符串表示，用于调试和日志输出
const char* ncclDevRedOpToString(ncclDevRedOp_t op) {
  switch (op) {                                         // 根据规约操作类型判断
  case ncclDevSum: return "Sum";                        // Sum：求和
  case ncclDevProd: return "Prod";                      // Prod：求积
  case ncclDevMinMax: return "MinMax";                  // MinMax：最小值或最大值
  case ncclDevPreMulSum: return "PreMulSum";            // PreMulSum：预乘求和（用于求平均值）
  case ncclDevSumPostDiv: return "SumPostDiv";          // SumPostDiv：求和后除（用于求平均值）
  default: return "Unknown";                           // 未知的规约操作
  }
}

// 数据类型转换为字符串的辅助函数
// 参数 type: NCCL 数据类型枚举值
// 返回值: 数据类型的字符串表示，用于调试和日志输出
const char* ncclDatatypeToString(ncclDataType_t type) {
  switch (type) {                                       // 根据数据类型判断
  case ncclInt8: return "ncclInt8";                    // 8 位有符号整数
  case ncclInt32: return "ncclInt32";                  // 32 位有符号整数
  case ncclUint32: return "ncclUint32";                // 32 位无符号整数
  case ncclInt64: return "ncclInt64";                  // 64 位有符号整数
  case ncclUint64: return "ncclUint64";                // 64 位无符号整数
  case ncclFloat16: return "ncclFloat16";              // 16 位浮点数（半精度）
  case ncclFloat32: return "ncclFloat32";              // 32 位浮点数（单精度）
  case ncclFloat64: return "ncclFloat64";              // 64 位浮点数（双精度）
  case ncclBfloat16: return "ncclBfloat16";            // 16 位脑浮点数（BF16）
  case ncclFloat8e4m3: return "ncclFloat8e4m3";        // 8 位浮点数 E4M3 格式
  case ncclFloat8e5m2: return "ncclFloat8e5m2";        // 8 位浮点数 E5M2 格式
  default: return "Unknown";                           // 未知的数据类型
  }
}

// 算法类型转换为字符串的辅助函数
// 参数 algo: NCCL 算法类型枚举值
// 返回值: 算法的字符串表示，用于调试和日志输出
const char* ncclAlgoToString(int algo) {
  switch (algo) {                                       // 根据算法类型判断
  case NCCL_ALGO_TREE: return "TREE";                  // TREE：树形算法（适合小消息）
  case NCCL_ALGO_RING: return "RING";                  // RING：环形算法（适合大消息）
  case NCCL_ALGO_COLLNET_DIRECT: return "COLLNET_DIRECT"; // COLLNET_DIRECT：集合网络直连
  case NCCL_ALGO_COLLNET_CHAIN: return "COLLNET_CHAIN";   // COLLNET_CHAIN：集合网络链式
  case NCCL_ALGO_NVLS: return "NVLS";                  // NVLS：NVLink Switch 算法
  case NCCL_ALGO_NVLS_TREE: return "NVLS_TREE";        // NVLS_TREE：NVLink Switch 树算法
  case NCCL_ALGO_PAT: return "PAT";                    // PAT：分区算法（Partitioned）
  default: return "Unknown";                           // 未知的算法
  }
}

// 协议类型转换为字符串的辅助函数
// 参数 proto: NCCL 协议类型枚举值
// 返回值: 协议的字符串表示，用于调试和日志输出
const char* ncclProtoToString(int proto) {
  switch (proto) {                                      // 根据协议类型判断
  case NCCL_PROTO_LL: return "LL";                      // LL：Long Jump 协议（中等和大消息）
  case NCCL_PROTO_LL128: return "LL128";                // LL128：128 字节对齐的 LL 协议
  case NCCL_PROTO_SIMPLE: return "SIMPLE";              // SIMPLE：简单协议（小消息）
  default: return "Unknown";                           // 未知的协议
  }
}

// AllGather：所有 rank 收集数据到每个 rank
// 每个 rank 提供的数据会被收集，并且所有 rank 都会收到所有数据的副本
// NCCL_API 宏定义用于导出公共 API 函数
NCCL_API(ncclResult_t, ncclAllGather, const void* sendbuff, void* recvbuff, size_t sendcount,
    ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclAllGather(const void* sendbuff, void* recvbuff, size_t sendcount,
    ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream) {
  // Just pass the size of one message and not the total bytes sent/received.
  // 只传递一条消息的大小，而不是发送/接收的总字节数
  // NVTX3 是 NVIDIA 的性能分析工具，用于标记函数调用和参数
  NVTX3_FUNC_WITH_PARAMS(AllGather, NcclNvtxParamsAllGather, // 标记函数名为 AllGather
    NVTX3_PAYLOAD(comm ? comm->commHash : 0, sendcount * ncclTypeSize(datatype))); // 载荷：通信器哈希和数据字节数

  // 初始化操作信息结构体
  struct ncclInfo info = { ncclFuncAllGather, "AllGather", // 操作类型和名称
    sendbuff, recvbuff, sendcount, datatype, ncclSum, 0, comm, stream, /* Args */
    ALLGATHER_CHUNKSTEPS, ALLGATHER_SLICESTEPS };      // AllGather 的块步数和切片步数配置
  return ncclEnqueueCheck(&info);                      // 调用入队检查函数，实际执行操作
}

// AlltoAll：每个 rank 向所有其他 rank 发送数据
// 这是全连接的通信模式，每个 rank 的数据被分成 n 部分，分别发送给 n 个 rank
NCCL_API(ncclResult_t, ncclAlltoAll, const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclComm* comm, cudaStream_t stream);
ncclResult_t ncclAlltoAll(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclComm* comm, cudaStream_t stream) {
  // 标记性能分析事件，记录通信器哈希和数据字节数
  NVTX3_FUNC_WITH_PARAMS(AlltoAll, NcclNvtxParamsAlltoAll, // 标记函数名为 AlltoAll
    NVTX3_PAYLOAD(comm ? comm->commHash : 0, count * ncclTypeSize(datatype))); // 载荷：通信器哈希和数据字节数

  // 初始化操作信息结构体
  struct ncclInfo info = { ncclFuncAlltoAll, "AlltoAll",   // 操作类型和名称
    sendbuff, recvbuff, count, datatype, ncclSum, 0, comm, stream, /* Args */
    ALLTOALL_CHUNKSTEPS, ALLTOALL_SLICESTEPS };         // AlltoAll 的块步数和切片步数配置
  return ncclEnqueueCheck(&info);                      // 调用入队检查函数，实际执行操作
}

//allreduce，大模型最常用集合通信操作
// AllReduce：所有 rank 的数据规约并广播到所有 rank
// 这是最常用的集合通信操作，用于大模型训练中的梯度同步
NCCL_API(ncclResult_t, ncclAllReduce, const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream);
ncclResult_t ncclAllReduce(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream) {
  // 标记性能分析事件，记录通信器哈希、数据字节数和规约操作
  NVTX3_FUNC_WITH_PARAMS(AllReduce, NcclNvtxParamsAllReduce, // 标记函数名为 AllReduce
    NVTX3_PAYLOAD(comm ? comm->commHash : 0, count * ncclTypeSize(datatype), op)); // 载荷：通信器哈希、数据字节数和规约操作

  // 初始化操作信息结构体
  struct ncclInfo info = { ncclFuncAllReduce, "AllReduce",   // 操作类型和名称
    sendbuff, recvbuff, count, datatype, op, 0, comm, stream, /* Args */
    ALLREDUCE_CHUNKSTEPS, ALLREDUCE_SLICESTEPS };        // AllReduce 的块步数和切片步数配置（2，4）
  return ncclEnqueueCheck(&info);                      // 调用入队检查函数，实际执行操作
}

// Broadcast：根 rank 广播数据到所有其他 rank
// 根 rank 的数据会被复制到所有其他 rank 的接收缓冲区中
NCCL_API(ncclResult_t, ncclBroadcast, const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, int root,
    ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclBroadcast(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, int root,
    ncclComm_t comm, cudaStream_t stream) {
  // 标记性能分析事件，记录通信器哈希、数据字节数和根 rank
  NVTX3_FUNC_WITH_PARAMS(Broadcast, NcclNvtxParamsBroadcast, // 标记函数名为 Broadcast
    NVTX3_PAYLOAD(comm ? comm->commHash : 0, count * ncclTypeSize(datatype), root)); // 载荷：通信器哈希、数据字节数和根 rank

  // 初始化操作信息结构体
  struct ncclInfo info = { ncclFuncBroadcast, "Broadcast", // 操作类型和名称
    sendbuff, recvbuff, count, datatype, ncclSum, root, comm, stream, /* Args */
    BROADCAST_CHUNKSTEPS, BROADCAST_SLICESTEPS };      // Broadcast 的块步数和切片步数配置
  return ncclEnqueueCheck(&info);                      // 调用入队检查函数，实际执行操作
}
/* Deprecated original "in place" function, similar to MPI */
/* 已废弃的原生"原地"函数，类似于 MPI */
// Bcast：Broadcast 的原地版本（发送和接收缓冲区相同）
// 已废弃，建议使用 ncclBroadcast 代替
NCCL_API(ncclResult_t, ncclBcast, void* buff, size_t count, ncclDataType_t datatype, int root,
    ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclBcast(void* buff, size_t count, ncclDataType_t datatype, int root,
    ncclComm_t comm, cudaStream_t stream) {
  // 直接调用 ncclBroadcast，发送和接收缓冲区都使用 buff（原地操作）
  return ncclBroadcast(buff, buff, count, datatype, root, comm, stream); // 原地广播：发送和接收是同一个缓冲区
}

// Gather：根 rank 从所有 rank 收集数据
// 所有 rank 的数据会被收集到根 rank 的接收缓冲区中，非根 rank 的接收缓冲区可以忽略
NCCL_API(ncclResult_t, ncclGather, const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, int root,
    ncclComm* comm, cudaStream_t stream);
ncclResult_t ncclGather(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, int root,
    ncclComm* comm, cudaStream_t stream) {
  // 标记性能分析事件，记录通信器哈希、数据字节数和根 rank
  NVTX3_FUNC_WITH_PARAMS(Gather, NcclNvtxParamsGather,
    NVTX3_PAYLOAD(comm ? comm->commHash : 0, count * ncclTypeSize(datatype), root));

  // 初始化操作信息结构体
  struct ncclInfo info = { ncclFuncGather, "Gather",     // 操作类型和名称
    sendbuff, recvbuff, count, datatype, ncclSum, root, comm, stream, /* Args */
    GATHER_CHUNKSTEPS, GATHER_SLICESTEPS };        // Gather 的块步数和切片步数配置
  return ncclEnqueueCheck(&info);                      // 调用入队检查函数，实际执行操作
}

// Reduce：所有 rank 的数据规约到根 rank
// 与 AllReduce 不同，规约结果只存在于根 rank 的接收缓冲区中，其他 rank 不接收结果
NCCL_API(ncclResult_t, ncclReduce, const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclReduce(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream) {
  // 标记性能分析事件，记录通信器哈希、数据字节数、根 rank 和规约操作
  NVTX3_FUNC_WITH_PARAMS(Reduce, NcclNvtxParamsReduce,
    NVTX3_PAYLOAD(comm ? comm->commHash : 0, count * ncclTypeSize(datatype), root, op));

  // 初始化操作信息结构体
  struct ncclInfo info = { ncclFuncReduce, "Reduce",     // 操作类型和名称
    sendbuff, recvbuff, count, datatype, op, root, comm, stream, /* Args */
    REDUCE_CHUNKSTEPS, REDUCE_SLICESTEPS };        // Reduce 的块步数和切片步数配置
  return ncclEnqueueCheck(&info);                      // 调用入队检查函数，实际执行操作
}

// ReduceScatter：先规约再分散到各 rank
// 组合操作：先对所有 rank 的数据进行规约，然后将结果分散到每个 rank
// 这是 AllReduce 的逆操作，常用于分布式训练中的梯度更新
NCCL_API(ncclResult_t, ncclReduceScatter, const void* sendbuff, void* recvbuff, size_t recvcount,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream);
ncclResult_t ncclReduceScatter(const void* sendbuff, void* recvbuff, size_t recvcount,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream) {
  // 标记性能分析事件，记录通信器哈希、接收数据字节数和规约操作
  NVTX3_FUNC_WITH_PARAMS(ReduceScatter, NcclNvtxParamsReduceScatter,
    NVTX3_PAYLOAD(comm ? comm->commHash : 0, recvcount * ncclTypeSize(datatype), op));

  // 初始化操作信息结构体（注意：使用 recvcount 而不是 count）
  struct ncclInfo info = { ncclFuncReduceScatter, "ReduceScatter",  // 操作类型和名称
    sendbuff, recvbuff, recvcount, datatype, op, 0, comm, stream, /* Args */
    REDUCESCATTER_CHUNKSTEPS, REDUCESCATTER_SLICESTEPS };  // ReduceScatter 的块步数和切片步数配置
  return ncclEnqueueCheck(&info);                           // 调用入队检查函数，实际执行操作
}

// Scatter：根 rank 分发数据到所有 rank
// 与 Gather 相反，根 rank 将数据分成 n 份，分别发送给每个 rank 一份
// 常用于将数据从主节点分发到工作节点
NCCL_API(ncclResult_t, ncclScatter, const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, int root, ncclComm* comm, cudaStream_t stream);
ncclResult_t ncclScatter(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, int root, ncclComm* comm, cudaStream_t stream) {
  // 标记性能分析事件，记录通信器哈希、数据字节数和根 rank
  NVTX3_FUNC_WITH_PARAMS(Scatter, NcclNvtxParamsScatter,
    NVTX3_PAYLOAD(comm ? comm->commHash : 0, count * ncclTypeSize(datatype), root));

  // 初始化操作信息结构体
  struct ncclInfo info = { ncclFuncScatter, "Scatter",     // 操作类型和名称
    sendbuff, recvbuff, count, datatype, ncclSum, root, comm, stream, /* Args */
    SCATTER_CHUNKSTEPS, SCATTER_SLICESTEPS };        // Scatter 的块步数和切片步数配置
  return ncclEnqueueCheck(&info);                      // 调用入队检查函数，实际执行操作
}

// Send：点对点发送操作
// 向指定的 peer rank 发送数据，接收方需要调用 ncclRecv
// 这是异步操作，需要配合 stream 使用
NCCL_API(ncclResult_t, ncclSend, const void* sendbuff, size_t count, ncclDataType_t datatype, int peer,
    ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclSend(const void* sendbuff, size_t count, ncclDataType_t datatype, int peer,
    ncclComm_t comm, cudaStream_t stream) {
  // 标记性能分析事件，记录通信器哈希、数据字节数和目标 peer
  NVTX3_FUNC_WITH_PARAMS(Send, NcclNvtxParamsSendRecv,
    NVTX3_PAYLOAD(comm ? comm->commHash : 0, count * ncclTypeSize(datatype), peer));

  // 初始化操作信息结构体（注意：第一个缓冲区为 NULL，第二个是发送缓冲区）
  struct ncclInfo info = { ncclFuncSend, "Send",
    NULL, (void*)sendbuff, count, datatype, ncclSum, peer, comm, stream, /* Args */
    1, 1 };                                             // 点对点操作的块步数和切片步数都是 1
  return ncclEnqueueCheck(&info);                       // 调用入队检查函数，实际执行操作
}

// Recv：点对点接收操作
// 从指定的 peer rank 接收数据，发送方需要调用 ncclSend
// 这是异步操作，需要配合 stream 使用
NCCL_API(ncclResult_t, ncclRecv, void* recvbuff, size_t count, ncclDataType_t datatype, int peer,
    ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclRecv(void* recvbuff, size_t count, ncclDataType_t datatype, int peer,
    ncclComm_t comm, cudaStream_t stream) {
  // 标记性能分析事件，记录通信器哈希、数据字节数和源 peer
  NVTX3_FUNC_WITH_PARAMS(Recv, NcclNvtxParamsSendRecv,
    NVTX3_PAYLOAD(comm ? comm->commHash : 0, count * ncclTypeSize(datatype), peer));

  // 初始化操作信息结构体（注意：第一个缓冲区为 NULL，第二个是接收缓冲区）
  struct ncclInfo info = { ncclFuncRecv, "Recv",
    NULL, recvbuff, count, datatype, ncclSum, peer, comm, stream, /* Args */
    1, 1 };                                             // 点对点操作的块步数和切片步数都是 1
  return ncclEnqueueCheck(&info);                       // 调用入队检查函数，实际执行操作
}
