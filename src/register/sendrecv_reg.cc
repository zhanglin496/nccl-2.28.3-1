/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION. All rights reserved.
 * 版权所有 (c) 2022-2025，NVIDIA 公司。保留所有权利。
 *
 * See LICENSE.txt for license information
 * 有关许可信息，请参见 LICENSE.txt 文件
 ************************************************************************/

// 包含内存注册相关的头文件，定义了内存注册的接口和数据结构
#include "register.h"
// 包含传输层相关的头文件，定义了传输层接口和连接功能
#include "transport.h"

// 函数功能：为 P2P（点对点）网络通信注册缓冲区
// 这个函数用于 send/recv 操作，将用户缓冲区注册到网络设备，实现高效的 DMA 传输
// 参数说明：
//   - comm: 通信上下文指针，包含通信域的所有信息
//   - userbuff: 用户提供的缓冲区地址，需要注册到网络设备
//   - size: 缓冲区的大小（字节数）
//   - conn: 连接器指针，描述与对等体的连接信息
//   - regFlag: 输出参数，指示是否成功注册（0=未注册，1=已注册）
//   - handle: 输出参数，返回注册后的内存句柄，用于后续的网络传输操作
//   - cleanupQueue: 清理队列，用于存储注册后的回调函数，在适当时候释放资源
// 返回值：ncclSuccess 表示成功，其他值表示失败
ncclResult_t ncclRegisterP2pNetBuffer(struct ncclComm* comm, void* userbuff, size_t size, struct ncclConnector* conn, int* regFlag, void** handle, struct ncclIntruQueue<struct ncclCommCallback, &ncclCommCallback::next>* cleanupQueue) {
  // 初始化返回值为成功
  ncclResult_t ret = ncclSuccess;

  // 初始化注册标志为 0（未注册）
  // 调用者可以通过检查这个值来判断是否需要清理资源
  *regFlag = 0;
  // 检查网络设备类型是否为 UNPACK（解包设备）
  // NCCL_NET_DEVICE_UNPACK: 特殊的网络设备类型，不需要内存注册
  // 如果不是 UNPACK 设备，则需要进行内存注册
  if (comm->netDeviceType != NCCL_NET_DEVICE_UNPACK) {
    // 首先尝试图注册（Graph Registration）
    // 图注册适用于持久化规划模式，在 CUDA Graph 捕获期间注册内存
    // 优点：可以预注册所有缓冲区，避免运行时注册开销
    if (comm->planner.persistent && ncclParamGraphRegister()) {
      // 执行网络缓冲区的图注册
      // 参数说明：
      //   - comm: 通信上下文
      //   - userbuff: 用户缓冲区地址
      //   - size: 缓冲区大小
      //   - &conn: 连接器指针的地址（指向单个连接器）
      //   - 1: 连接器数量（这里只有 1 个）
      //   - regFlag: 输出注册状态
      //   - handle: 输出内存句柄
      //   - cleanupQueue: 清理队列
      //   - NULL: 不需要返回清理队列元素数量
      ncclNetGraphRegisterBuffer(comm, userbuff, size, &conn, 1, regFlag, handle, cleanupQueue, NULL);
    }
    // 如果图注册失败（regFlag == 0），尝试本地注册
    // 本地注册在运行时进行，不使用 CUDA Graph
    if (*regFlag == 0 && ncclParamLocalRegister()) {
      // 执行网络缓冲区的本地注册
      // 参数与图注册类似，但不传入 cleanupQueue（本地注册不需要）
      ncclNetLocalRegisterBuffer(comm, userbuff, size, &conn, 1, regFlag, handle);
    }
  }
  // 返回结果状态码
  return ret;
}

// 函数功能：为 P2P IPC（进程间通信）注册缓冲区
// IPC 用于同一节点内不同 GPU 之间的直接内存访问
// 参数说明：
//   - comm: 通信上下文指针
//   - userbuff: 用户提供的缓冲区地址，需要注册到 IPC
//   - size: 缓冲区的大小（字节数）
//   - peerRank: 对等体的 rank 编号
//   - regFlag: 输出参数，指示是否成功注册
//   - regAddr: 输出参数，返回注册后的远程地址，用于直接访问对等 GPU 的内存
//   - cleanupQueue: 清理队列
// 返回值：ncclSuccess 表示成功
ncclResult_t ncclRegisterP2pIpcBuffer(struct ncclComm* comm, void* userbuff, size_t size, int peerRank, int* regFlag, void** regAddr, struct ncclIntruQueue<struct ncclCommCallback, &ncclCommCallback::next>* cleanupQueue) {
  // 初始化返回值为成功
  ncclResult_t ret = ncclSuccess;
  // 声明偏移量变量，用于存储在远程地址数组中的偏移
  uintptr_t offset = 0;
  // 声明远程地址数组指针，用于存储所有对等体的远程地址
  uintptr_t* peerRmtAddrs = NULL;

  // 初始化注册标志为 0（未注册）
  *regFlag = 0;
  // 首先尝试图注册（如果启用了持久化规划和图注册）
  if (comm->planner.persistent && ncclParamGraphRegister()) {
    // 执行 IPC 缓冲区的图注册
    // 参数说明：
    //   - comm: 通信上下文
    //   - userbuff: 用户缓冲区地址
    //   - size: 缓冲区大小
    //   - &peerRank: 对等体 rank 的地址
    //   - 1: 对等体数量（这里只有 1 个）
    //   - NCCL_IPC_SENDRECV: IPC 类型，用于 send/recv 操作
    //   - regFlag: 输出注册状态
    //   - &offset: 输出偏移量
    //   - &peerRmtAddrs: 输出远程地址数组
    //   - reinterpret_cast<void*>(cleanupQueue): 类型转换，将队列指针转换为 void*
    //   - NULL: 不需要返回清理队列元素数量
    ncclIpcGraphRegisterBuffer(comm, userbuff, size, &peerRank, 1, NCCL_IPC_SENDRECV, regFlag, &offset, &peerRmtAddrs, reinterpret_cast<void*>(cleanupQueue), NULL);
  }
  // 如果图注册失败，尝试本地注册
  if (*regFlag == 0 && ncclParamLocalRegister()) {
    // 执行 IPC 缓冲区的本地注册
    // 参数与图注册类似，但不传入 cleanupQueue
    ncclIpcLocalRegisterBuffer(comm, userbuff, size, &peerRank, 1, NCCL_IPC_SENDRECV, regFlag, &offset, &peerRmtAddrs);
  }

  // 如果注册成功，计算并返回注册地址
  if (*regFlag)
    // 注册地址 = 远程地址数组基址 + 偏移量
    // (uintptr_t): 将指针转换为整数类型进行算术运算
    // peerRmtAddrs: 远程地址数组的基址
    // offset: 在该数组中的偏移量
    // (void*): 将计算结果转换回 void* 指针类型
    *regAddr = (void*)((uintptr_t)peerRmtAddrs + offset);
  // 返回结果状态码
  return ret;
}
