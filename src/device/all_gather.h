/*************************************************************************
 * Copyright (c) 2015-2022, NVIDIA CORPORATION. All rights reserved.
 * 版权所有 (c) 2015-2022，NVIDIA 公司。保留所有权利。
 *
 * See LICENSE.txt for license information
 * 详见 LICENSE.txt 获取许可证信息
 ************************************************************************/
// 包含设备端通用头文件，定义设备端数据结构和常量
#include "device.h"
// 包含集合操作头文件，定义各种集合操作的通用接口和宏
#include "collectives.h"
// 包含基本操作头文件，定义通信原语（发送、接收、复制等）
#include "primitives.h"

// 匿名命名空间，避免符号冲突
namespace {
  // runRing 函数模板：实现基于环形拓扑的 AllGather 算法
  // 模板参数：
  //   T: 数据类型（如 float、int 等）
  //   RedOp: 归约操作类型（如 ncclSum、ncclProd 等）
  //   Proto: 协议类型（如 ProtoSimple、ProtoLL、ProtoLL128 等）
  //   isNetOffload: 是否启用网络卸载（默认为 false），用于单进程每节点（onePPN）场景
  template<typename T, typename RedOp, typename Proto, bool isNetOffload = false>
  // __device__ 表示在设备端执行，__forceinline__ 强制内联以优化性能
  __device__ __forceinline__ void runRing(int tid, int nthreads, struct ncclDevWorkColl* work) {
    // 获取环形拓扑结构指针，包含前驱和后继节点信息
    ncclRing *ring = &ncclShmem.channel.ring;
    // 获取环形拓扑中用户排名数组，定义环形顺序
    const int *ringRanks = ring->userRanks;
    // 获取通信域中的进程总数（排名数）
    const int nranks = ncclShmem.comm.nRanks;
    // 定义变量：count（总元素数）、partOffset（分片偏移）、partCount（分片数量）、chunkCount（块数量）
    ssize_t count, partOffset, partCount, chunkCount;
    // 计算当前通道的分片信息，包括偏移、数量和块大小
    // Proto::Id 是协议标识符，sizeof(T) 是元素大小
    ncclCollCbdPart(work, ncclShmem.channelId, Proto::Id, sizeof(T), &count, &partOffset, &partCount, &chunkCount);
    // 定义变量：offset（数据偏移）、dataOffset（数据在缓冲区中的偏移）
    ssize_t offset;
    ssize_t dataOffset;
    // 定义变量：nelem（元素数量）、rankDest（目标排名）、workNthreads（工作线程数）
    int nelem;
    int rankDest;
    int workNthreads;
    // 获取输入缓冲区指针（发送缓冲区）
    T *inputBuf = (T*)work->sendbuff;
    // 获取输出缓冲区指针（接收缓冲区）
    T *outputBuf = (T*)work->recvbuff;

    // 如果启用网络卸载（isNetOffload == true），则仅使用 1 个 warp（32 线程）来驱动 Ring 算法/网络通信
    // 其余 warp 会并行地将源数据复制到目标缓冲区（当 AllGather 非原地操作时）
    if (isNetOffload) {
      // 工作线程数设置为 1 个 warp 的大小（32）
      workNthreads = WARP_SIZE;
      // 块数量设置为网络最大传输大小
      chunkCount = NCCL_MAX_NET_SIZE;
    } else {
      // 否则，所有线程都参与工作
      workNthreads = nthreads;
    }

    // 只有工作线程才执行以下操作
    if (tid < workNthreads) {
      // 创建 Primitives 对象，封装通信原语操作
      // Coverity 报告认为被调用方将 &ring->next 视为数组访问
      // 然而，由于使用了 FanSymmetric<1>，只有第一个元素会被访问，所以这是安全的
      // coverity[callee_ptr_arith:FALSE]
      // 模板参数解释：
      //   T: 数据类型
      //   RedOp: 归约操作
      //   FanSymmetric<1>: 对称扇入扇出，1 个发送方和 1 个接收方
      //   1: 步数
      //   Proto: 协议类型
      //   0: 未使用标志
      //   isNetOffload: 是否网络卸载
      Primitives<T, RedOp, FanSymmetric<1>, 1, Proto, 0, isNetOffload> prims
        // 构造 Primitives 对象
        (tid, workNthreads, &ring->prev, &ring->next, inputBuf, outputBuf, work->redOpArg, 0, 0, 0, work, NULL, isNetOffload ? NCCL_MAX_NET_SIZE : 0);
      // 遍历所有数据块（chunk），逐块处理 AllGather 操作
      for (size_t elemOffset = 0; elemOffset < partCount; elemOffset += chunkCount) {
        /////////////// begin AllGather steps ///////////////
        // AllGather 步骤开始 ///////////////
        // 计算当前块的实际元素数量（取块大小和剩余数据量的较小值）
        nelem = min(chunkCount, partCount - elemOffset);
        // 计算数据在缓冲区中的偏移量（分片偏移 + 元素偏移）
        dataOffset = partOffset + elemOffset;

        // step 0: push data to next GPU
        // 步骤 0: 将数据推送到下一个 GPU
        // 目标排名是环形数组中的第一个排名（当前节点的后继）
        rankDest = ringRanks[0];
        // 计算目标位置偏移：数据偏移 + 目标排名 * 总元素数
        offset = dataOffset + rankDest * count;

        // 如果是原地操作（输入和输出指向同一位置）或者启用网络卸载
        if ((inputBuf + dataOffset == outputBuf + offset) || isNetOffload) { // In place or onePPN
          // 直接发送数据到下一个 GPU
          prims.directSend(dataOffset, offset, nelem);
        } else {
          // 非原地操作，先复制再发送
          prims.directCopySend(dataOffset, offset, nelem);
        }

        // k-2 steps: copy to next GPU
        // k-2 步骤: 复制到下一个 GPU（接收并转发）
        // 循环 nranks-2 次，将数据沿环形传播
        for (int j = 1; j < nranks - 1; ++j) {
          // 目标排名是环形数组中倒数的排名
          rankDest = ringRanks[nranks - j];
          // 计算目标位置偏移
          offset = dataOffset + rankDest * count;
          // 接收数据并立即转发（直接接收复制直接发送）
          prims.directRecvCopyDirectSend(offset, offset, nelem);
        }

        // Make final copy from buffer to dest.
        // 从缓冲区进行最后的复制到目标位置
        // 目标排名是环形数组中的第二个排名
        rankDest = ringRanks[1];
        // 计算最终目标位置偏移
        offset = dataOffset + rankDest * count;

        // Final wait/copy.
        // 最终等待/复制，完成最后一步的接收操作
        prims.directRecv(offset, nelem);
      }
    // 如果是网络卸载模式且当前线程不是工作线程，则执行数据复制
    } else if (inputBuf != outputBuf + ringRanks[0] * count) {
      // 调整输入缓冲区指针到当前分片的起始位置
      inputBuf = inputBuf + partOffset;
      // 调整输出缓冲区指针到当前分片 + 第一个排名的数据起始位置
      outputBuf = outputBuf + partOffset + ringRanks[0] * count;
      // 执行归约复制操作，将输入数据复制到输出缓冲区
      // 模板参数：COLL_UNROLL（展开级别）、RedOp（归约操作）、T（数据类型）
      //   0（未使用）、1（未使用）、1（未使用）、0（未使用）、1（未使用）、1（未使用）
      //   PreOpSrcs=0（无预操作源）
      reduceCopy<COLL_UNROLL, RedOp, T, 0, 1, 1, 0, 1, 1, /*PreOpSrcs=*/0>
        // 函数参数：
        //   tid-workNthreads: 当前线程在工作线程中的 ID
        //   nthreads-workNthreads: 剩余线程数
        //   work->redOpArg: 归约操作参数
        //   &work->redOpArg: 归约操作参数指针
        //   false: 未使用标志
        //   1: 源数量
        //   (void**)&inputBuf: 源缓冲区指针数组
        //   1: 目标数量
        //   (void**)&outputBuf: 目标缓冲区指针数组
        //   partCount: 处理的元素数量
        (tid - workNthreads, nthreads - workNthreads, work->redOpArg, &work->redOpArg, false, 1, (void**)&inputBuf, 1, (void**)&outputBuf, partCount);
    }
    // 在继续下一个工作项之前，我们需要等待所有 warp 完成
    // 否则，如果下一个工作项使用当前工作项的输出缓冲区，可能会发生竞争
    // 使用屏障 14 以避免与 prims 屏障和 __syncthread() 冲突
    if (isNetOffload) barrier_sync(14, nthreads);
  }
}

// RunWorkColl 特化模板：AllGather + RING + SIMPLE 协议
template<typename T, typename RedOp>
struct RunWorkColl<ncclFuncAllGather, T, RedOp, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE> {
  // run 函数：设备端执行的入口函数
  __device__ __forceinline__ void run(int tid, int nthreads, struct ncclDevWorkColl* work) {
    // 判断是否启用网络卸载：单进程每节点（onePPN）且使用了网络注册内存
    bool isNetOffload = work->isOneRPN && work->netRegUsed;
    // 如果启用网络卸载
    if (isNetOffload)
      // 调用 runRing，使用 ProtoSimple<1,1> 协议，启用网络卸载
      runRing<T, RedOp, ProtoSimple<1, 1>, true>(tid, nthreads, work);
    else
      // 否则调用 runRing，使用 ProtoSimple 协议，不启用网络卸载
      // ALLGATHER_CHUNKSTEPS/ALLGATHER_SLICESTEPS: 计算每个 chunk 的 slice 步数
      runRing<T, RedOp, ProtoSimple<ALLGATHER_CHUNKSTEPS/ALLGATHER_SLICESTEPS, ALLGATHER_SLICESTEPS>, false>(tid, nthreads, work);
  }
};

// RunWorkColl 特化模板：AllGather + RING + LL 协议
template<typename T, typename RedOp>
struct RunWorkColl<ncclFuncAllGather, T, RedOp, NCCL_ALGO_RING, NCCL_PROTO_LL> {
  // run 函数：设备端执行的入口函数
  __device__ __forceinline__ void run(int tid, int nthreads, struct ncclDevWorkColl* work) {
    // 调用 runRing，使用 ProtoLL 协议（Long Latency 协议，适合中等和大规模数据）
    runRing<T, RedOp, ProtoLL>(tid, nthreads, work);
  }
};

// RunWorkColl 特化模板：AllGather + RING + LL128 协议
template<typename T, typename RedOp>
struct RunWorkColl<ncclFuncAllGather, T, RedOp, NCCL_ALGO_RING, NCCL_PROTO_LL128> {
  // run 函数：设备端执行的入口函数
  __device__ __forceinline__ void run(int tid, int nthreads, struct ncclDevWorkColl* work) {
    // 调用 runRing，使用 ProtoLL128 协议（128 字节对齐的 Long Latency 协议）
    runRing<T, RedOp, ProtoLL128>(tid, nthreads, work);
  }
};

// RunWorkColl 特化模板：AllGather + PAT 算法 + SIMPLE 协议
// PAT (Pattern) 算法是一种优化的通信模式算法
template<typename T, typename RedOp>
struct RunWorkColl<ncclFuncAllGather, T, RedOp, NCCL_ALGO_PAT, NCCL_PROTO_SIMPLE> {
  // run 函数：设备端执行的入口函数
  __device__ __forceinline__ void run(int tid, int nthreads, struct ncclDevWorkColl* work) {
    // PAT 算法需要 CUDA 架构 >= 6.0（Pascal 及以后）
#if __CUDA_ARCH__ >= 600
    // 定义协议类型为简单协议（1 步，1 slice）
    using Proto = ProtoSimple<1, 1>;
    // 获取进程总数
    const int nranks = ncclShmem.comm.nRanks;
    // 获取当前进程排名
    const int rank = ncclShmem.comm.rank;
    // 定义变量：count（总数）、channelOffset（通道偏移）、channelCount（通道数量）、chunkCount（块数量）
    size_t count, channelOffset, channelCount, chunkCount;
    // 计算当前通道的分片信息
    ncclCollCbdPart(work, ncclShmem.channelId, Proto::Id, sizeof(T), &count, &channelOffset, &channelCount, &chunkCount);

    // 定义工作线程数量常量（PAT 算法专用工作线程数）
    static constexpr int nworkers = NCCL_PAT_NWORKERS;
    // 获取 PAT 共享内存指针，从第 0 个 warp 的 scratch 内存中分配
    struct ncclPatShmem* shmem = (struct ncclPatShmem*)ncclScratchForWarp(0);
    // 轮询计数器，用于自旋等待
    uint64_t pollCount = 0;
    // 同步所有线程，确保在开始使用共享内存之前所有线程都已到达
    __syncthreads(); // Don't start using shared mem until everyone arrives
    // 初始化 PAT 步骤的标志位为 0（清空所有步骤的标志）
    for (int i=tid; i<NCCL_SHMEM_PAT_STEPS; i+=nthreads) shmem->patSteps[i].flags = 0;
    // 线程 0 初始化本地累加器大小为 0
    if (tid == 0) shmem->localAccSize = 0;
    // 线程 nworkers 初始化并行因子为 0
    if (tid == nworkers) shmem->parallelFactor = 0;
    // 同步所有线程，确保初始化完成
    __syncthreads();

    // 如果是算法计算线程（tid == nworkers）
    if (tid == nworkers) { // Algo computation thread
      // 创建 PAT AllGather 算法对象
      // 参数：块大小*元素类型大小、步数、工作线程数/每块大小、通道偏移、通道结束位置、总数、块数量、当前排名、进程总数
      PatAGAlgorithm<T> patAlgo(chunkCount*sizeof(T), NCCL_STEPS, NCCL_PAT_NWORKERS/WARP_SIZE, channelOffset, channelOffset + channelCount, count, chunkCount, rank, nranks);
      // 获取并行因子（并行度）并存储到共享内存
      int parallelFactor = shmem->parallelFactor = patAlgo.getParallelFactor();
      // 初始化步骤计数器
      int step = 0;
      // 无限循环，直到算法完成
      while (1) {
        // 获取当前步骤的 PAT 步骤结构指针（使用环形缓冲区避免覆盖）
        struct ncclPatStep* ps = shmem->patSteps+(step%NCCL_SHMEM_PAT_STEPS);
        // 创建原子引用，用于线程块级别的同步
        cuda::atomic_ref<int, cuda::thread_scope_block> poll(ps->flags);
        // 等待工作线程完成步骤 'step-NCCL_SHMEM_PAT_STEPS'
        // 使用自旋轮询，直到标志位变为 0
        while (poll.load(cuda::memory_order_acquire) != 0) pollCount++; // Wait for workers to be done with step 'step-NCCL_SHMEM_PAT_STEPS'
        // 从算法获取下一个操作并填充到步骤结构中
        patAlgo.getNextOp(ps);
        // 获取是否是最后一步的标志
        int last = ps->last;
        // 步骤计数器递增
        step++;
        // 如果 last == 2，表示算法完成，退出循环
        if (last == 2) break;
      }
    // 如果是工作线程（tid < nworkers）
    } else if (tid < nworkers) { // Worker threads
      // 获取输入缓冲区指针
      T *inputBuf = (T*)work->sendbuff;
      // 获取输出缓冲区指针
      T *outputBuf = (T*)work->recvbuff;
      // 初始化并行因子为 0
      int parallelFactor = 0;
      // 获取并行因子指针（volatile 防止编译器优化掉读取）
      volatile int* pfPtr = &shmem->parallelFactor;
      // 自旋等待，直到算法线程设置并行因子
      while (parallelFactor == 0) parallelFactor = *pfPtr;

      // 计算每组的大小（每组包含多个 warp）
      int groupSize = nworkers/(WARP_SIZE*parallelFactor) * WARP_SIZE;
      // 计算当前线程所属的组编号
      int group = tid / groupSize;
      // 计算总组数
      int nGroups = nworkers / groupSize;
      // 计算当前线程在组内的线程 ID
      int tidInGroup = tid - group*groupSize;
      // 我们不使用 recvPeers/sendPeers，而是传递共享内存结构
      // 创建 Primitives 对象用于执行通信原语
      Primitives<T, RedOp, FanSymmetric<1>, 0, Proto, 0> prims
        // 构造 Primitives 对象
        (tidInGroup, groupSize, (int*)shmem->recvDims, (int*)shmem->sendDims, inputBuf, outputBuf, work->redOpArg, group, 0, 0, nullptr, nullptr, 0, primsModePatAg);

      // 初始化步骤为组编号（实现不同组从不同步骤开始，实现并行）
      int step = group;
      // 无限循环，直到接收到完成信号
      while(1) {
        // 获取当前步骤的 PAT 步骤结构指针
        struct ncclPatStep* ps = shmem->patSteps+(step%NCCL_SHMEM_PAT_STEPS);
        // 创建原子引用，用于同步
        cuda::atomic_ref<int, cuda::thread_scope_block> poll(ps->flags);
        // 等待计算线程设置标志位（非 0 表示就绪）
        while (poll.load(cuda::memory_order_acquire) == 0) pollCount++; // Wait for compute thread
        // 获取是否是最后一步的标志
        int last = ps->last;
        // 执行 PAT 复制操作
        prims.patCopy(ps, shmem);
        // 如果是组内第一个线程，将标志位重置为 0，通知计算线程该步骤已完成
        if (tidInGroup == 0) poll.store(0, cuda::memory_order_release); // Return element to compute thread
        // 如果是最后一步，退出循环
        if (last) break;
        // 移动到下一个步骤（跳过其他组的步骤）
        step += nGroups;
      }
    }
#endif
  }
};

// RunWorkColl 特化模板：AllGather + NVLS 算法 + SIMPLE 协议
// NVLS (NVLink SHARP) 是基于 NVLink 的硬件加速集合通信
template<typename T, typename RedOp>
struct RunWorkColl<ncclFuncAllGather, T, RedOp, NCCL_ALGO_NVLS, NCCL_PROTO_SIMPLE> {
  // Scatterer 结构体模板：用于数据分发操作
  // BcastSendNotRecv: 布尔模板参数，true 表示广播发送模式，false 表示正常接收模式
  template<bool BcastSendNotRecv>
  struct Scatterer {
    // 工作项指针
    struct ncclDevWorkColl* work;
    // 块大小
    ssize_t chunkSize;
    // Rail 网格偏移
    ssize_t railGridOffset;

    // 函数调用运算符重载：执行实际的分发操作
    // 模板参数：
    //   SlicePerChunk: 每个 chunk 的 slice 数
    //   MinSrcs/MaxSrcs: 最小/最大源数量
    //   MinDsts/MaxDsts: 最小/最大目标数量
    //   MultimemSrcs/MultimemDsts: 多内存源/目标数量
    template<int SlicePerChunk, int MinSrcs, int MaxSrcs, int MinDsts, int MaxDsts, int MultimemSrcs, int MultimemDsts>
    __device__ __forceinline__ void operator()(
        int tid, int tn, int slice, int maxSliceSize,
        int nSrcs, void** srcPtrs, int nDsts, void** dstPtrs, int32_t* dstSizes, uint32_t sendDirectFlag, uint32_t recvDirectFlag
      ) {
      // 静态断言：要求每个 chunk 只有 1 个 slice
      static_assert(SlicePerChunk==1, "require: SlicePerChunk==1");
      // 静态断言：要求目标数或源数最多为 1（不能同时有多源多目标）
      static_assert(MaxDsts<=1 || MaxSrcs<=1, "require: MaxDsts<=1 || MaxSrcs<=1");

      // 获取 NVLS 通道结构指针
      struct ncclNvls* nvls = &ncclShmem.channel.nvls;
      // 获取节点数量
      int nNodes = ncclShmem.comm.nNodes;
      // 获取 Rail（链路）数量（NVLS 中的头数量）
      int nRails = nvls->nHeads;
      // 计算当前通道编号（相对于工作项的起始通道）
      int part = ncclShmem.channelId - work->channelLo;
      // 获取输入缓冲区指针（字符指针，便于字节偏移）
      char* inbuf = (char*)work->sendbuff;
      // 获取输出缓冲区指针
      char* outbuf = (char*)work->recvbuff;
      // 获取每个排名的元素数量（字节数）
      ssize_t countPerRank = work->collnet.count;
      // 判断是否为原地操作：输入缓冲区是否等于输出缓冲区中当前排名的数据位置
      bool inPlace = (inbuf == outbuf + ncclShmem.comm.rank * countPerRank);
      // 计算当前 rail 处理的数据范围起始位置（取最小值防止越界）
      ssize_t railAllBeg = min(railGridOffset + part * chunkSize, nNodes * countPerRank);
      // 计算当前 rail 处理的数据范围结束位置
      ssize_t railAllEnd = min(railAllBeg + chunkSize, nNodes * countPerRank);
      // 计算当前 rail 处理的数据大小
      int railAllSize = railAllEnd - railAllBeg;
      // 初始化 rail 索引为 0
      int rail = 0;
      // 初始化源索引为 0
      int src = 0;

      // 根据模式设置 rail 索引
      if (BcastSendNotRecv) {
        // 广播发送模式：使用 headRank（主排名）作为 rail
        rail = nvls->headRank;
      } else {
        // 正常接收模式：如果使用了寄存器，直接返回
        if (work->regUsed) return;
        // 否则从 rail 0 开始
        rail = 0;
      }
      // 如果当前线程是目标线程，设置目标大小
      if (tid < nDsts) dstSizes[tid] = railAllSize;
      // 循环处理所有 rail
      do {
        // 计算当前处理的节点编号
        int node = railAllBeg / countPerRank;
        // 初始化 rail 偏移为 0
        int railAllOffset = 0;
        // 循环处理当前 rail 中的所有数据
        while (railAllOffset < railAllSize) {
          // 计算单个排名的数据起始位置
          ssize_t railOneBeg = node * countPerRank;
          // 计算单个排名的数据结束位置
          ssize_t railOneEnd = railOneBeg + countPerRank;
          // 计算在单个排名数据中的偏移
          ssize_t railOneOffset = (railAllBeg + railAllOffset) - railOneBeg;
          // 计算当前处理的数据增量（取最小值防止越界）
          int delta = min(railAllEnd, railOneEnd) - (railAllBeg + railAllOffset);
          // 从密集排名数组中获取实际的用户排名
          int rank = ncclShmem.comm.collNetDenseToUserRank[node * nRails + rail];
          // 计算用户空间中的数据起始位置
          ssize_t userOneBeg = rank * countPerRank + railOneOffset;
          // 判断输出是否为目标：原地操作且为当前排名，或广播发送模式，或使用寄存器时目标为 0
          int outIsDst = (inPlace && rank == ncclShmem.comm.rank) || BcastSendNotRecv || work->regUsed ? 0 : 1;
          // 如果存在源且存在目标（输出目标或外部目标）
          if (nSrcs != 0 && outIsDst + nDsts != 0) {
            // 执行归约复制操作
            reduceCopy<ncclCollUnroll(), RedOp, T,
              /*MultimemSrcs,MinSrcs,MaxSrcs=*/MultimemSrcs, 1, 1,
              /*MultimemDsts=*/MultimemDsts, 0 + MultimemDsts + MinDsts, 1 + MaxDsts,
              /*PreOpSrcs=*/0>
              // 调用归约复制函数
              (tid, tn, 0, nullptr, false,
                /*nSrcs=*/1, [=]__device__(int s/*==0*/) -> void* {
              // Lambda 函数：返回源指针（rail 偏移位置）
              return (char*)srcPtrs[src] + railAllOffset;
            },
                /*nDsts=*/outIsDst + nDsts, [=]__device__(int d) -> void* {
              // Lambda 函数：返回目标指针
              return d < outIsDst ? outbuf + userOneBeg
                : work->regUsed ? (char*)dstPtrs[d - outIsDst] + userOneBeg
                : (char*)dstPtrs[d - outIsDst] + railAllOffset;
            }, delta); // 数据增量
          }
          // 更新 rail 偏移
          railAllOffset += delta;
          // 移动到下一个节点
          node += 1;
        }
        // 移动到下一个 rail
        rail += 1;
        // 移动到下一个源
        src += 1;
      // 如果不是广播发送模式且还有更多 rail，继续循环
      } while (!BcastSendNotRecv && src < nRails);
    }
  };

  // run 函数：NVLS AllGather 的主入口
  __device__ __forceinline__ void run(int tid, int/*nthreads*/, struct ncclDevWorkColl* work) {
    // 获取 NVLS 通道结构指针
    struct ncclNvls* nvls = &ncclShmem.channel.nvls;
    // 元素数量变量
    int nelem;

    // 计算各阶段的线程数分配
    // 网络发送线程数：单节点时为 0，否则根据是否使用网络寄存器决定（WARP_SIZE 或 6*WARP_SIZE）
    const int nThreadsNetSend = work->oneNode ? 0 : (work->netRegUsed ? WARP_SIZE :  6 * WARP_SIZE);
    // Gather 阶段线程数：使用寄存器时为 nHeads*4 向上取整到 WARP_SIZE，否则为 8*WARP_SIZE
    const int nThreadsGather = work->regUsed ? roundUp(nvls->nHeads << 2, WARP_SIZE) : 8 * WARP_SIZE;
    // Bcast 阶段线程数：剩余的所有线程
    const int nThreadsBcast = NCCL_MAX_NTHREADS - nThreadsNetSend - nThreadsGather;

    // 计算各阶段的线程范围边界
    const int tidEndGather = nThreadsGather;
    const int tidEndNetSend = tidEndGather + nThreadsNetSend;
    const int tidEndBcast = tidEndNetSend + nThreadsBcast;

    // 单节点模式（oneNode == true）
    if (work->oneNode) {
      // 获取当前排名
      const ssize_t rank = ncclShmem.comm.rank;
      // 定义变量：count（总数）、gridOffset（网格偏移）、channelCount（通道数量）、offset（偏移）、chunkCount（块数量）
      size_t count, gridOffset, channelCount, offset, chunkCount;
      // 计算当前通道的分片信息
      ncclCollCbdPart(work, ncclShmem.channelId, NCCL_PROTO_SIMPLE, sizeof(T), &count, &gridOffset, &channelCount, &chunkCount);
      // 如果未使用寄存器
      if (!work->regUsed) {
        // Gather 阶段：tid < tidEndGather 的线程执行
        if (tid < tidEndGather) {
          // Gather
          // 定义协议类型为简单协议（1 步，1 slice，COLL_UNROLL 展开）
          using Proto = ProtoSimple<1, 1, COLL_UNROLL>;
          // 创建 Primitives 对象，用于 Gather 操作
          // FanAsymmetric<NCCL_MAX_NVLS_ARITY, 0>: 非对称扇入扇出，最多 NCCL_MAX_NVLS_ARITY 个输入，0 个输出
          // Direct=0: 不直接操作
          Primitives<T, RedOp, FanAsymmetric<NCCL_MAX_NVLS_ARITY, 0>, /*Direct=*/0, Proto, 0>
            prims(tid, nThreadsGather, nvls->up, NULL, NULL, work->recvbuff,
              work->redOpArg, 0 * Proto::MaxGroupWidth, 1, 1);
          // 遍历所有数据块
          for (size_t elemOffset = 0; elemOffset < channelCount; elemOffset += chunkCount) {
            // 计算数据偏移
            offset = gridOffset + elemOffset;
            // 计算当前块的元素数量
            nelem = min(chunkCount, channelCount - elemOffset);
            // 执行 Gather 操作，从 NVLS 上行链路收集数据
            prims.gather(offset, nvls->nHeads * count, nelem, count, -1, 0);
          }
          // coverity[overrun-call] => Coverity 认为 prims.index 可能大于 1
        // Bcast 阶段：tidEndGather <= tid < tidEndBcast 的线程执行
        } else if (tid < tidEndBcast) {
          // Bcast through NVLS
          // 通过 NVLS 广播
          // 定义协议类型为简单协议（1 步，1 slice，COLL_UNROLL 展开，0 个未使用参数，1 个未使用参数）
          using Proto = ProtoSimple<1, 1, COLL_UNROLL, 0, 1>;
          // 创建 Primitives 对象，用于 Bcast 操作
          // FanAsymmetric<0, 1>: 0 个输入，1 个输出
          Primitives<T, RedOp, FanAsymmetric<0, 1>, /*Direct=*/0, Proto, 0>
            prims(tid - tidEndGather, nThreadsBcast, NULL, &nvls->down, work->sendbuff, NULL,
              work->redOpArg, 3 * Proto::MaxGroupWidth, 0, 0);
          // 遍历所有数据块
          for (size_t elemOffset = 0; elemOffset < channelCount; elemOffset += chunkCount) {
            // 计算数据偏移
            offset = gridOffset + elemOffset;
            // 计算当前块的元素数量
            nelem = min(chunkCount, channelCount - elemOffset);
            // 执行发送操作
            prims.send(offset, nelem);
          }
          // coverity[overrun-call] => Coverity 认为 prims.index 可能大于 1
        }
      // 如果使用了寄存器
      } else {
        // Gather 阶段
        if (tid < tidEndGather) {
          // 定义协议类型
          using Proto = ProtoSimple<1, 1, COLL_UNROLL>;
          // 创建 Primitives 对象
          // FanSymmetric<NCCL_MAX_NVLS_ARITY>: 对称扇入扇出
          Primitives<T, RedOp, FanSymmetric<NCCL_MAX_NVLS_ARITY>, /*Direct=*/0, Proto, 0>
            prims(tid, nThreadsGather, nvls->up, nvls->up, NULL, NULL,
              work->redOpArg, 0 * Proto::MaxGroupWidth, 1, 1);

          /* used as sync */
          /* 用作同步 */
          // 执行 scatter 操作作为同步
          prims.scatter(0, 0, 0, 0, -1, 0);

          // 遍历所有数据块，执行 gather 操作
          for (size_t elemOffset = 0; elemOffset < channelCount; elemOffset += chunkCount) {
            prims.gather(0, 0, 0, 0, -1, 0);
          }
        // Bcast 阶段
        } else if (tid < tidEndBcast) {
          // 定义协议类型
          using Proto = ProtoSimple<1, 1, COLL_UNROLL, 0, 1>;
          // 创建 Primitives 对象
          // FanSymmetric<1>: 对称扇入扇出，1 个对等点
          // Direct=1: 直接操作模式
          Primitives<T, RedOp, FanSymmetric<1>, /*Direct=*/1, Proto, 0>
            prims(tid - tidEndGather, nThreadsBcast, &nvls->down, &nvls->down, work->sendbuff, NULL,
              work->redOpArg, 1 * Proto::MaxGroupWidth, 0, 0, work);
          /* used as sync */
          /* 用作同步 */
          // 执行 recv 操作作为同步
          prims.recv(0, 0);

          // 遍历所有数据块，执行直接发送操作
          for (size_t elemOffset = 0; elemOffset < channelCount; elemOffset += chunkCount) {
            // 计算输入偏移
            ssize_t inpOffset = gridOffset + elemOffset;
            // 计算输出偏移（加上当前排名的数据位置）
            ssize_t outOffset = inpOffset + rank * count;
            // 计算当前块的元素数量
            nelem = min(chunkCount, channelCount - elemOffset);
            // 执行直接发送操作
            prims.directSend(inpOffset, outOffset, nelem);
          }
        }
      }
    // 多节点模式（使用 NVLS + IB SHARP）
    } else {
      // NVLS + IB SHARP
      // 获取节点数量
      int nNodes = ncclShmem.comm.nNodes;
      // 计算当前通道编号
      int part = ncclShmem.channelId - work->channelLo;
      // 获取每个排名的元素数量
      ssize_t countPerRank = work->collnet.count;
      // 获取通道数量
      const int nChannels = work->channelHi - work->channelLo + 1;
      // 获取块数量
      ssize_t chunkCount = work->collnet.chunkCount;
      // Gather 阶段：tid < tidEndGather 的线程执行
      if (tid < tidEndGather) {
        // 定义协议类型
        using Proto = ProtoSimple<1, 1, COLL_UNROLL>;
        // 创建 Primitives 对象
        Primitives<T, RedOp, FanAsymmetric<NCCL_MAX_NVLS_ARITY, 0>, /*Direct=*/1, Proto, 0>
          prims(tid, nThreadsGather, nvls->up, nullptr, nullptr, work->recvbuff,
            /*redOpArg=*/0, 1 * Proto::MaxGroupWidth, 1, 1, work);
        // 遍历所有 rail 网格偏移
        for (ssize_t railGridOffset = 0; railGridOffset < nNodes * countPerRank; railGridOffset += nChannels * chunkCount) {
          // 创建 Scatterer 对象（非广播模式，接收数据）
          Scatterer</*BcastSendNotRecv=*/false> scat;
          scat.work = work;
          scat.chunkSize = chunkCount;
          scat.railGridOffset = railGridOffset;
          // 执行 process 操作（接收）
          prims.template process</*Recv=*/1, /*Send=*/0>(scat);
        }
      // 其余线程处理网络发送和广播
      } else {
        // 如果使用了网络寄存器
        if (work->netRegUsed) {
          // 定义协议类型
          using ProtoSend = ProtoSimple<1, 1, COLL_UNROLL>;
          using ProtoBcast = ProtoSimple<1, 1, COLL_UNROLL, 0, 1>;
          // 计算最大步数
          int maxSteps = (int)divUp(nNodes * countPerRank, nChannels * chunkCount);
          // 初始化当前步数为 -1
          int curSteps = -1;
          // 判断当前线程是否为发布线程（每组的第一个线程）
          int postThread = tid - tidEndGather == 0 ? 1 : 0;
          // 对于 UB（Unified Buffer），我们需要控制发送速度以避免网络拥塞
          // 首先展开 2 步，然后在数据接收后展开剩余步骤
          if (postThread) {
            // 初始化步数为 2 和 maxSteps 的较小值
            curSteps = min(2, maxSteps);
            // 发送对等通知，告知后续线程准备接收数据
            Primitives<T, RedOp, FanAsymmetric<0, 1>, /*Direct=*/1, ProtoSend, 0>::sendPeerNotify(nvls->out, 1, curSteps);
          }
          // 创建 Primitives 对象
          Primitives<T, RedOp, FanSymmetric<1>, /*Direct=*/1, ProtoBcast, 0>
            prims(tid - tidEndGather, nThreadsNetSend + nThreadsBcast, &nvls->out, &nvls->down, nullptr, nullptr,
              /*redOpArg=*/0, 2 * ProtoBcast::MaxGroupWidth, 0, 0, work);
          // 遍历所有 rail 网格偏移
          for (ssize_t railGridOffset = 0; railGridOffset < nNodes * countPerRank; railGridOffset += nChannels * chunkCount) {
            // 创建 Scatterer 对象（广播发送模式）
            Scatterer</*BcastSendNotRecv=*/true> scat;
            scat.work = work;
            scat.chunkSize = chunkCount;
            scat.railGridOffset = railGridOffset;
            // 执行 process 操作（接收和发送）
            prims.template process</*Recv=*/1, /*Send=*/1>(scat);
            // 如果是发布线程且还有未发布的步骤
            if (postThread && curSteps < maxSteps) {
              // 增加步数
              curSteps++;
              // 发送对等通知，发布下一步
              Primitives<T, RedOp, FanAsymmetric<0, 1>, /*Direct=*/1, ProtoSend, 0>::sendPeerNotify(nvls->out, 1, 1);
            }
          }
        // 如果未使用网络寄存器
        } else {
          // 网络发送阶段
          if (tid < tidEndNetSend) {
            // 定义协议类型
            using Proto = ProtoSimple<1, 1, COLL_UNROLL>;
            // 创建 Primitives 对象
            Primitives<T, RedOp, FanAsymmetric<0, 1>, /*Direct=*/0, Proto, 0>
              prims(tid - tidEndGather, nThreadsNetSend, nullptr, &nvls->out, work->sendbuff, nullptr,
                /*redOpArg=*/0, 0 * Proto::MaxGroupWidth, 1, 1);
            // 遍历所有 rail 网格偏移
            for (ssize_t railGridOffset = 0; railGridOffset < nNodes * countPerRank; railGridOffset += nChannels * chunkCount) {
              // 计算当前 rail 处理的数据范围
              ssize_t railAllBeg = railGridOffset + part * chunkCount;
              ssize_t railAllEnd = min(railAllBeg + chunkCount, nNodes * countPerRank);
              // 计算当前节点的数据范围
              ssize_t railOneBeg = ncclShmem.comm.node * countPerRank;
              ssize_t railOneEnd = railOneBeg + countPerRank;
              // 计算实际需要发送的数据范围
              ssize_t beg = max(railAllBeg, railOneBeg);
              ssize_t end = min(railAllEnd, railOneEnd);
              // 执行发送操作
              prims.send(beg - railOneBeg, max(ssize_t(0), end - beg));
            }
          // 广播阶段
          } else {
            // 定义协议类型
            using Proto = ProtoSimple<1, 1, COLL_UNROLL, 0, 1>;
            // 创建 Primitives 对象
            Primitives<T, RedOp, FanSymmetric<1>, /*Direct=*/0, Proto, 0>
              prims(tid - tidEndNetSend, nThreadsBcast, &nvls->out, &nvls->down, nullptr, nullptr,
                /*redOpArg=*/0, 2 * Proto::MaxGroupWidth, 0, 0);
            // 遍历所有 rail 网格偏移
            for (ssize_t railGridOffset = 0; railGridOffset < nNodes * countPerRank; railGridOffset += nChannels * chunkCount) {
              // 创建 Scatterer 对象（广播发送模式）
              Scatterer</*BcastSendNotRecv=*/true> scat;
              scat.work = work;
              scat.chunkSize = chunkCount;
              scat.railGridOffset = railGridOffset;
              // 执行 process 操作（接收和发送）
              prims.template process</*Recv=*/1, /*Send=*/1>(scat);
            }
          }
        }
      }
    }
  }
};

// RunWorkColl 特化模板：AllGather + COLLNET_DIRECT 算法 + SIMPLE 协议
// COLLNET_DIRECT 是直接集合网络算法（使用硬件加速的集体网络）
template<typename T, typename RedOp>
struct RunWorkColl<ncclFuncAllGather, T, RedOp, NCCL_ALGO_COLLNET_DIRECT, NCCL_PROTO_SIMPLE> {
  // Scatterer 结构体模板：用于数据分发操作
  template<bool BcastSendNotRecv>
  struct Scatterer {
    // 工作项指针
    struct ncclDevWorkColl* work;
    // 块大小
    ssize_t chunkSize;
    // Rail 网格偏移
    ssize_t railGridOffset;

    // 函数调用运算符重载：执行实际的分发操作
    template<int SlicePerChunk, int MinSrcs, int MaxSrcs, int MinDsts, int MaxDsts, int MultimemSrcs, int MultimemDsts>
    __device__ __forceinline__ void operator()(
        int tid, int tn, int slice, int maxSliceSize,
        int nSrcs, void** srcPtrs, int nDsts, void** dstPtrs, int32_t* dstSizes, uint32_t sendDirectFlag, uint32_t recvDirectFlag
      ) {
      // 静态断言：要求每个 chunk 只有 1 个 slice
      static_assert(SlicePerChunk==1, "require: SlicePerChunk==1");
      // 静态断言：要求目标数或源数最多为 1
      static_assert(MaxDsts<=1 || MaxSrcs<=1, "require: MaxDsts<=1 || MaxSrcs<=1");

      // 获取直接集合网络通道结构指针
      struct ncclDirect* direct = &ncclShmem.channel.collnetDirect;
      // 获取节点数量
      int nNodes = ncclShmem.comm.nNodes;
      // 获取 Rail（链路）数量
      int nRails = direct->nHeads;
      // 计算当前通道编号
      int part = ncclShmem.channelId - work->channelLo;
      // 获取输入缓冲区指针
      char* inbuf = (char*)work->sendbuff;
      // 获取输出缓冲区指针
      char* outbuf = (char*)work->recvbuff;
      // 获取每个排名的元素数量（字节数，乘以 sizeof(T)）
      ssize_t countPerRank = work->collnet.count*sizeof(T);
      // 判断是否为原地操作
      bool inPlace = (inbuf == outbuf + ncclShmem.comm.rank*countPerRank);

      // 计算当前 rail 处理的数据范围
      ssize_t railAllBeg = min(railGridOffset + part*chunkSize, nNodes*countPerRank);
      ssize_t railAllEnd = min(railAllBeg + chunkSize, nNodes*countPerRank);
      // 计算当前 rail 处理的数据大小
      int railAllSize = railAllEnd - railAllBeg;
      // 如果当前线程是目标线程，设置目标大小
      if (tid < nDsts) dstSizes[tid] = railAllSize;

      // 初始化源索引和 rail 索引
      int src = 0;
      int rail;
      // 根据模式设置 rail 索引
      if (BcastSendNotRecv) {
        // 广播发送模式：使用 headRank
        rail = direct->headRank;
      } else {
        // 正常接收模式：使用 headRank+1，循环到 0
        rail = direct->headRank+1;
        if (rail == nRails) rail = 0;
      }
      // 循环处理所有 rail
      do {
        // 计算当前处理的节点编号
        int node = railAllBeg/countPerRank;
        // 初始化 rail 偏移
        int railAllOffset = 0;
        // 循环处理当前 rail 中的所有数据
        while (railAllOffset < railAllSize) {
          // 计算单个排名的数据范围
          ssize_t railOneBeg = node*countPerRank;
          ssize_t railOneEnd = railOneBeg + countPerRank;
          // 计算在单个排名数据中的偏移
          ssize_t railOneOffset = (railAllBeg+railAllOffset) - railOneBeg;
          // 计算当前处理的数据增量
          int delta = min(railAllEnd, railOneEnd) - (railAllBeg+railAllOffset);
          // 从密集排名数组中获取实际的用户排名
          int rank = ncclShmem.comm.collNetDenseToUserRank[node*nRails + rail];
          // 计算用户空间中的数据起始位置
          ssize_t userOneBeg = rank*countPerRank + railOneOffset;
          // 判断输出是否为目标
          int outIsDst = (inPlace && rank == ncclShmem.comm.rank) ? 0 : 1;
          // 如果存在源且存在目标
          if (nSrcs != 0 && outIsDst+nDsts != 0) {
            // 执行归约复制操作
            reduceCopy<ncclCollUnroll(), RedOp, T,
                     /*MultimemSrcs,MinSrcs,MaxSrcs=*/0,1,1,
                     /*MultimemDsts=*/0, 0+MinDsts, 1+MaxDsts,
                     /*PreOpSrcs=*/0>
            // 调用归约复制函数
            (tid, tn, 0, nullptr, false,
             /*nSrcs=*/1, [=]__device__(int s/*==0*/) -> void* {
               // Lambda 函数：返回源指针
               // 如果使用寄存器且接收支持 P2P 读取，返回用户空间偏移；否则返回 rail 偏移
               return work->regUsed && (recvDirectFlag & NCCL_P2P_READ) ? (char*)srcPtrs[src] + userOneBeg : (char*)srcPtrs[src] + railAllOffset;
             },
             /*nDsts=*/outIsDst+nDsts, [=]__device__(int d) -> void* {
               // Lambda 函数：返回目标指针
               return d < outIsDst ? outbuf + userOneBeg
                                   : work->regUsed && (sendDirectFlag & NCCL_P2P_WRITE) ? (char*)dstPtrs[d-outIsDst] + userOneBeg
                                   : (char*)dstPtrs[d-outIsDst] + railAllOffset;
             },
             delta); // 数据增量
          }
          // 更新 rail 偏移
          railAllOffset += delta;
          // 移动到下一个节点
          node += 1;
        }
        // 移动到下一个源和 rail
        src += 1;
        rail += 1;
        // rail 循环到 0
        if (rail == nRails) rail = 0;
      // 如果不是广播发送模式且还有更多 rail（除了最后一个），继续循环
      } while (!BcastSendNotRecv && src < nRails-1);
    }
  };

  // run 函数：COLLNET_DIRECT AllGather 的主入口
  __device__ __forceinline__ void run(int tid, int/*nthreads*/, struct ncclDevWorkColl* work) {
    // 获取当前通道编号
    const int part = ncclShmem.channelId - work->channelLo;
    // 获取通道数量
    const int nChannels = work->channelHi - work->channelLo + 1;
    // 获取直接集合网络通道结构指针
    struct ncclDirect* direct = &ncclShmem.channel.collnetDirect;
    // 获取节点数量的引用
    int const &nNodes = ncclShmem.comm.nNodes;
    // 获取每个排名的元素数量
    ssize_t countPerRank = work->collnet.count;
    // 获取块大小
    size_t chunkSize = work->collnet.chunkCount;
    // 判断是否有下行连接（是否有本地对等点）
    const int hasDn = (direct->down[0] >= 0) ? 1 : 0;
    // 判断是否为多 Rail 模式（多个链路）
    bool isMultiRail = (direct->nHeads > 1);
    // 初始化各阶段的 warp 数量
    int nWarps1 = 1;
    int nWarps2 = (isMultiRail ? 2 : 1);
    int nWarps3 = (isMultiRail ? 2 : 0);
    // 根据总 warp 数量按比例分配各阶段的 warp 数
    float denom = float(work->nWarps)/float(nWarps1+nWarps2+nWarps3);
    nWarps3 = int(denom*nWarps3);
    nWarps2 = int(denom*nWarps2);
    nWarps1 = work->nWarps - (nWarps2+nWarps3);

    // 定义协议类型为简单协议（1 步，1 slice）
    using Proto = ProtoSimple<1, 1>;

    // 阶段 1：发送到网络
    int tn = nWarps1*WARP_SIZE;
    if (tid < tn) {
      // 如果使用了网络寄存器
      if (work->netRegUsed) {
        // 如果是线程 0
        if (tid == 0) {
          // 如果当前排名有本地对等点（hasDn == true），则不能将所有数据卸载到网络
          // 在这种情况下，步数应该基于 chunkSize 等计算；否则，我们只是
          // 将步数增加 1 以启动集体网络进度
          int steps = hasDn ? (int)divUp(nNodes * countPerRank, nChannels * chunkSize) : 1;
          // 发送对等通知，启动集合网络进度
          Primitives<T, RedOp, FanAsymmetric<0, 1>, /*Direct=*/0, Proto, 0>::sendPeerNotify(direct->out, 1, steps);
        }
        // 同步 warp 内的线程
        __syncwarp();
      // 如果未使用网络寄存器
      } else {
        // Phase 1: send to network
        // 阶段 1：发送到网络
        Primitives<T, RedOp, FanAsymmetric<0, 1>, /*Direct=*/0, Proto, 0>
          prims(tid, tn, nullptr, &direct->out, work->sendbuff, nullptr,
            /*redOpArg=*/0, 0 * Proto::MaxGroupWidth, 1, 1);
        // 遍历所有 rail 网格偏移
        for (ssize_t railGridOffset = 0; railGridOffset < nNodes * countPerRank; railGridOffset += nChannels * chunkSize) {
          // 计算当前 rail 处理的数据范围
          ssize_t railAllBeg = railGridOffset + part * chunkSize;
          ssize_t railAllEnd = min(railAllBeg + chunkSize, nNodes * countPerRank);
          // 计算当前节点的数据范围
          ssize_t railOneBeg = ncclShmem.comm.node * countPerRank;
          ssize_t railOneEnd = railOneBeg + countPerRank;
          // 计算实际需要发送的数据范围
          ssize_t beg = max(railAllBeg, railOneBeg);
          ssize_t end = min(railAllEnd, railOneEnd);
          // 执行发送操作
          prims.send(beg - railOneBeg, max(ssize_t(0), end - beg));
        }
      }
      // 阶段 1 完成，返回
      return;
    }
    // 调整线程 ID（减去阶段 1 的线程数）
    tid -= tn;

    // 阶段 2：从网络接收 -> 存储输出 + 发送到广播
    tn = nWarps2*WARP_SIZE;
    if (tid < tn) {
      // 如果使用了网络寄存器且没有下行连接
      if (work->netRegUsed && !hasDn) {
        // 如果是线程 0
        if (tid == 0) {
          // 接收对等通知
          Primitives<T, RedOp, FanAsymmetric<1, NCCL_MAX_DIRECT_ARITY>, /*Direct=*/0, Proto, 0>::recvPeerNotify(direct->out, 0, 1);
        }
        // 同步 warp 内的线程
        __syncwarp();
      // 否则
      } else {
        // Phase 2: Recv network -> deposit output + send to bcast
        // 阶段 2：从网络接收 -> 存储输出 + 发送到广播
        Primitives<T, RedOp, FanAsymmetric<1, NCCL_MAX_DIRECT_ARITY>, /*Direct=*/1, Proto, 0>
          prims(tid, tn, &direct->out, direct->heads + 1, nullptr, work->recvbuff,
            /*redOpArg=*/0, 1 * Proto::MaxGroupWidth, 0, 0, work);
        // 遍历所有 rail 网格偏移
        for (ssize_t railGridOffset = 0; railGridOffset < nNodes * countPerRank; railGridOffset += nChannels * chunkSize) {
          // 创建 Scatterer 对象（广播发送模式）
          Scatterer</*BcastSendNotRecv=*/true> scat;
          scat.work = work;
          scat.chunkSize = chunkSize;
          scat.railGridOffset = railGridOffset;
          // 执行 process 操作（接收和发送）
          prims.template process</*Recv=*/1, /*Send=*/1>(scat, work->direct, 0);
        }
      }
      // 阶段 2 完成，返回
      return;
    }
    // 调整线程 ID（减去阶段 2 的线程数）
    tid -= tn;

    // 阶段 3：从广播接收 -> 存储输出
    tn = nWarps3*WARP_SIZE;
    if (tid < tn) {
      // Phase 3: Recv bcast -> deposit output
      // 阶段 3：从广播接收 -> 存储输出
      Primitives<T, RedOp, FanAsymmetric<NCCL_MAX_DIRECT_ARITY, 0>, /*Direct=*/1, Proto, 0>
        prims(tid, tn, direct->heads+1, nullptr, nullptr, work->recvbuff,
              /*redOpArg=*/0, 2*Proto::MaxGroupWidth, 0, 0, work);
      // 遍历所有 rail 网格偏移
      for (ssize_t railGridOffset=0; railGridOffset < nNodes*countPerRank; railGridOffset += nChannels*chunkSize) {
        // 创建 Scatterer 对象（非广播模式，接收数据）
        Scatterer</*BcastSendNotRecv=*/false> scat;
        scat.work = work;
        scat.chunkSize = chunkSize;
        scat.railGridOffset = railGridOffset;
        // 执行 process 操作（仅接收）
        prims.template process</*Recv=*/1, /*Send=*/0>(scat, 0, work->direct);
      }
      // 阶段 3 完成，返回
      return;
    }
  }
};
