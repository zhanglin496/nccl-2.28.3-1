/*************************************************************************
 * Copyright (c) 2015-2025, NVIDIA CORPORATION. All rights reserved.
 * 版权所有 (c) 2015-2025，NVIDIA 公司。保留所有权利。
 *
 * See LICENSE.txt for license information
 * 有关许可信息，请参见 LICENSE.txt 文件
 ************************************************************************/

// 头文件保护宏开始
// 防止头文件被多次包含，避免重复定义
#ifndef NCCL_SYMMETRIC_SCHED_H_
#define NCCL_SYMMETRIC_SCHED_H_

// 包含调度器核心头文件，定义了任务调度相关的接口和数据结构
#include "scheduler.h"

// 函数功能：创建对称任务列表
// 对称调度是指将相同函数、操作和数据类型的任务分组，使用相同的内核执行
// 参数说明：
//   - comm: 通信上下文指针，包含通信域的所有信息
//   - task: 任务链表头指针，包含所有待调度的集体通信任务
//   - symTaskQueue: 输出参数，对称任务队列，用于存储可以对称执行的任务
//   - remainTasksHead: 输出参数，剩余任务链表头指针，存储不能对称执行的任务
// 返回值：ncclSuccess 表示成功，其他值表示失败
ncclResult_t ncclMakeSymmetricTaskList(struct ncclComm* comm, struct ncclTaskColl* task, struct ncclIntruQueue<struct ncclTaskColl, &ncclTaskColl::next>* symTaskQueue, struct ncclTaskColl** remainTasksHead) {
  // 初始化返回值为成功
  ncclResult_t ret = ncclSuccess;
  // 声明变量：函数-操作-数据类型组合的数量统计
  // 用于跟踪有多少种不同的函数-操作-数据类型组合
  int fnOpTySymCount = 0;
  // 声明数组：按函数、操作和数据类型分类的任务数组
  // ncclNumFuncs: 集体通信函数的数量（如 AllReduce、Broadcast 等）
  // ncclNumDevRedOps: 设备归约操作的数量（如 sum、prod、max 等）
  // ncclNumTypes: 数据类型的数量（如 int8、float32 等）
  // 数组存储每种组合对应的对称任务链表头
  struct ncclTaskColl* tasksSymByFnOpTy[ncclNumFuncs * ncclNumDevRedOps * ncclNumTypes];
  // 声明数组：存储有效的函数-操作-数据类型组合的索引
  // 用于快速遍历有任务分配的组合
  int fnOpTySymIndices[ncclNumFuncs * ncclNumDevRedOps * ncclNumTypes];
  // 获取内核规划器指针
  // planner: 包含任务调度、通道分配等规划信息
  struct ncclKernelPlanner* planner = &comm->planner;
  // 声明指针：剩余任务链表的尾指针
  // 用于高效地构建剩余任务链表
  struct ncclTaskColl* remainTasksTail = nullptr;

  // 清空任务数组，将所有元素初始化为 nullptr（空指针）
  // memset: 内存填充函数，将指定字节数的内存设置为指定值
  memset(tasksSymByFnOpTy, 0, sizeof(tasksSymByFnOpTy));
  // 初始化剩余任务头指针为 nullptr
  *remainTasksHead = nullptr;
  // 遍历任务链表，将任务分类到对称任务或剩余任务
  while (task != nullptr) {
    // 计算当前任务的函数-操作-数据类型组合索引
    // task->func: 集体通信函数类型（枚举值）
    // task->opDev.op: 设备归约操作类型（枚举值）
    // task->datatype: 数据类型（枚举值）
    // 索引计算公式：func * (DevRedOps * Types) + op * Types + type
    // 这是一个三维索引展开为一维的计算方式
    int index = ((int)task->func*ncclNumDevRedOps + (int)task->opDev.op)*ncclNumTypes + (int)task->datatype;
    // 保存下一个任务的指针，因为后续可能会修改 task->next
    struct ncclTaskColl* next = task->next;
    // 查找发送缓冲区的注册窗口
    // 发送窗口包含内存注册、对齐等信息
    NCCLCHECK(ncclDevrFindWindow(comm, task->sendbuff, &task->sendWin));
    // 查找接收缓冲区的注册窗口
    NCCLCHECK(ncclDevrFindWindow(comm, task->recvbuff, &task->recvWin));
    // 检查是否有可用的对称内核
    // 对称内核要求相同函数、操作、数据类型的任务可以合并执行
    bool symAvailable = ncclSymkAvailable(comm, task->func, task->opDev.op, task->datatype, task->count);

    // 检查任务是否可以对称执行
    // 条件：发送窗口存在 && 接收窗口存在 && 两个窗口都支持对称操作 && 有可用的对称内核
    // NCCL_WIN_COLL_SYMMETRIC: 窗口标志，表示支持对称集体操作
    if (task->sendWin && task->recvWin && (task->sendWin->winFlags & task->recvWin->winFlags & NCCL_WIN_COLL_SYMMETRIC) && symAvailable) {
      // 检查该组合是否是第一次遇到
      if (tasksSymByFnOpTy[index] == nullptr) fnOpTySymIndices[fnOpTySymCount++] = index;
      // 将当前任务添加到对称任务链表的头部
      // 这样做会反转链表顺序，但不影响功能
      task->next = tasksSymByFnOpTy[index];
      // 更新链表头为当前任务
      tasksSymByFnOpTy[index] = task;
      // 减少规划器中的任务计数
      // 这些任务将被合并执行，所以任务数减少
      planner->nTasksColl--;
    // 任务不能对称执行，添加到剩余任务链表
    } else {
      // 检查剩余任务链表是否已有元素
      if (*remainTasksHead) {
        // 链表非空，将任务添加到尾部
        remainTasksTail->next = task;
       // 更新尾指针
       remainTasksTail = task;
      } else {
        // 链表为空，初始化头尾指针
        *remainTasksHead = remainTasksTail = task;
      }
    }
    // 移动到下一个任务
    task = next;
  }
  // 处理完所有任务后，设置尾节点的 next 为 nullptr
  if (remainTasksTail) remainTasksTail->next = nullptr;

  // make sure kernel args space can hold at least a single work
  // 确保内核参数空间至少可以容纳一个工作项
  // comm->workArgsBytes: 内核参数空间的字节数
  // ncclSymkDevWorkArgs::calcArgsSize(MAXCHANNELS, 1): 计算最大通道数和1个工作项的参数大小
  assert(comm->workArgsBytes >= ncclSymkDevWorkArgs::calcArgsSize(MAXCHANNELS, 1));

  // Determine symmetric tasks kernels
  // 确定对称任务应使用的内核
  // 遍历所有有效的函数-操作-数据类型组合
  for (int cursor = 0; cursor < fnOpTySymCount; cursor++) {
    // 获取当前组合的任务链表头
    struct ncclTaskColl* task = tasksSymByFnOpTy[fnOpTySymIndices[cursor]];
    // 遍历该组合的所有任务
    while (task != NULL) {
      // 声明变量：内核 ID（初始化为无效值）
      ncclSymkKernelId kernelId = ncclSymkKernelId_Count;
      // 声明变量：通道数量（初始化为最大值）
      int nChannels = MAXCHANNELS;
      // 声明变量：每块 warp 数量（初始化为 0）
      int nWarps = 0;
      // 声明变量：工作项数量（初始化为 0）
      int nWorks = 0;
      // 声明变量：估算执行时间（初始化为很大的值）
      float estTimeUs = 1.e18;
      // 声明变量：数据总量和最大单个任务数据量
      size_t countTotal = 0, countMax = 0;
      // 保存任务链表头，用于后续遍历
      struct ncclTaskColl* headTask = task;
      // 计算最小单元格大小（以数据元素个数为单位）
      // NCCL_SYM_KERNEL_CELL_SIZE: 对称内核的单元格大小（字节数）
      // ncclTypeSize: 数据类型的大小（字节数）
      // 单元格是对称内核处理数据的基本单位
      size_t cellCount = NCCL_SYM_KERNEL_CELL_SIZE / ncclTypeSize(headTask->datatype);
      // For now we assume higher kernel id means a kernel for larger data size
      // 目前假设较高的内核 ID 表示用于较大数据大小的内核
      // 遍历任务链表，收集统计信息
      while (task != nullptr) {
        // 声明变量：对齐后的任务数据量
        size_t count;
        // 工作项计数加 1
        nWorks++;
        // 将任务数据量向上对齐到单元格边界
        // alignUp: 向上对齐函数
        count = alignUp(task->count, cellCount);
        // 累加总数据量
        countTotal += count;
        // 更新最大单任务数据量
        if (count > countMax) countMax = count;
        // 检查是否超出参数空间或没有更多任务
        // calcArgsSize: 计算给定通道数和工作数的参数大小
        if (ncclSymkDevWorkArgs::calcArgsSize(MAXCHANNELS, nWorks + 1) > comm->workArgsBytes || task->next == nullptr) {
          // 标记为对称链表的最后一个任务
          task->isSymLast = 1;
          // 退出循环
          break;
        }
        // 移动到下一个任务
        task = task->next;
      }
      // 选择最合适的对称内核
      // 根据数据量、工作数等信息选择内核 ID、通道数、warp 数等
      NCCLCHECK(ncclSymkPickKernel(comm, headTask->func, headTask->opDev.op, headTask->datatype,
                                   countTotal, countMax, nWorks,
                                   &estTimeUs, &kernelId, &nChannels, &nWarps));
      // 检查是否成功选择了内核
      if (kernelId == ncclSymkKernelId_Count) {
        // 没有找到合适的对称内核
        // 获取环境变量 NCCL_SYM_KERNEL 的值
        char const* name = ncclGetEnv("NCCL_SYM_KERNEL");
        // 输出警告信息
        WARN("Error: no symmetric kernel available for function %s.%s%s",
             ncclFuncToString(headTask->func), (name ? " NCCL_SYM_KERNEL was set to " : ""), (name ? name: ""));
        // 如果用户设置了 NCCL_SYM_KERNEL，返回无效使用错误；否则返回内部错误
        ret = (name ? ncclInvalidUsage : ncclInternalError);
        // 跳转到失败标签
        goto fail;
      }
      // set all symmetric tasks to the same kernel
      // 将所有对称任务设置为使用相同的内核
      task = headTask;
      // 遍历任务链表
      while (task != nullptr) {
        // 保存下一个任务指针
        struct ncclTaskColl* next = task->next;
        // 保存是否为最后一个任务的标志
        int isSymLast = task->isSymLast;
        // 设置设备函数 ID（内核 ID）
        task->devFuncId = (uint32_t)kernelId;
        // 设置最大通道数
        task->nMaxChannels = nChannels;
        // 设置每块 warp 数
        task->nWarps = nWarps;
        // 将任务加入对称任务队列
        // symTaskQueue: 对称任务队列
        ncclIntruQueueEnqueue(&planner->collSymTaskQueue, task);
        // 移动到下一个任务
        task = next;
        // 如果是最后一个任务，退出循环
        if (isSymLast) break;
      }
    }
  }

// 退出标签，正常返回
exit:
  // 返回结果状态码
  return ret;
// 失败标签
fail:
  // 跳转到退出标签
  goto exit;
}

// 函数功能：对称任务调度器
// 负责将对称任务分配到具体的通道和工作项，生成内核执行参数
// 参数说明：
//   - comm: 通信上下文指针
//   - symTaskQueue: 对称任务队列
//   - plan: 输出参数，内核执行计划，包含内核参数、通道分配等信息
// 返回值：ncclSuccess 表示成功
ncclResult_t ncclSymmetricTaskScheduler(struct ncclComm* comm, struct ncclIntruQueue<struct ncclTaskColl, &ncclTaskColl::next>* symTaskQueue, struct ncclKernelPlan* plan) {
  // 获取任务队列的头节点
  struct ncclTaskColl* headTask = ncclIntruQueueHead(symTaskQueue);
  // 获取设备函数 ID（内核 ID）
  int devFuncId = headTask->devFuncId;
  // 声明任务指针
  struct ncclTaskColl* task = NULL;
  // 声明变量：总数据量（对齐后的字节数）
  ssize_t totalCount = 0;  // aligned bytes
  // 声明变量：原始数据量（用于日志）
  ssize_t logCount = 0;
  // 声明变量：每个通道剩余的单元格数
  ssize_t remainCell = 0;
  // 声明变量：每个通道分配的单元格数
  ssize_t cellPerChannel = 0;
  // 声明变量：工作项总数和当前工作索引
  int workCount = 0, workIndex = 0;
  // 计算最小单元格大小（数据元素个数）
  size_t cellCount = NCCL_SYM_KERNEL_CELL_SIZE / ncclTypeSize(headTask->datatype); // minimal cell size
  // 初始化返回值
  ncclResult_t ret = ncclSuccess;
  // 声明变量：当前通道索引
  int curChannel = 0;
  // 声明变量：当前通道已分配的工作项数
  int curChannelWork = 0;
  // 声明变量：最大通道数
  int nMaxChannels = headTask->nMaxChannels;
  // 声明指针：工作缓冲区指针
  struct ncclSymkDevWork* workBufPtr = NULL;
  // 声明指针：通道工作范围指针
  struct ncclSymkChannelWorkRange* workRangePtr = NULL;
  // 获取函数名称字符串（用于日志）
  const char* funcName = ncclFuncToString(headTask->func);
  // 获取内核名称字符串（用于日志）
  const char* kernelName = ncclSymkKernelIdToString(headTask->devFuncId);
  // 声明指针：内核参数缓冲区
  struct ncclSymkDevWorkArgs* argsBuf = NULL;

  // 标记为对称集体操作
  plan->isSymColl = true;
  // 设置每个线程块的线程数 = warp 数 × WARP_SIZE（32）
  // WARP_SIZE: 通常是 32
  plan->threadPerBlock = headTask->nWarps * WARP_SIZE;
  // 标记没有代理操作
  // 代理操作是指在中间节点进行的额外处理
  plan->hasProxyOps = false;
  // 获取内核函数指针
  // 根据内核 ID、操作类型、数据类型返回对应的内核函数
  plan->kernelFn = ncclSymkGetKernelPtr((ncclSymkKernelId)headTask->devFuncId, headTask->opDev.op, headTask->datatype);
  // 从头开始遍历任务队列
  task = headTask;
  // 统计工作项总数和数据量
  while (task != nullptr && task->devFuncId == devFuncId) {
    // 工作项计数加 1
    workCount++;
    // 累加对齐后的数据量
    totalCount += alignUp(task->count, cellCount);
    // 累加原始数据量（用于日志）
    logCount += task->count;
    // 如果是最后一个任务，退出循环
    if (task->isSymLast == 1) break;
    // 移动到下一个任务
    task = task->next;
  }

  // 计算内核参数缓冲区大小
  plan->kernelArgsSize = ncclSymkDevWorkArgs::calcArgsSize(nMaxChannels, workCount);
  // 分配内核参数缓冲区内存（初始化为 0）
  argsBuf = (struct ncclSymkDevWorkArgs*)calloc(1, plan->kernelArgsSize);

  // 计算每个通道应分配的单元格数
  // DIVUP: 向上整除宏 (a+b-1)/b
  // totalCount / nMaxChannels: 每个通道的字节数
  // 再除以 cellCount: 得到每个通道的单元格数
  remainCell = cellPerChannel = DIVUP(DIVUP(totalCount, nMaxChannels), cellCount);
  // 获取通道工作范围指针
  // 用于记录每个通道的工作项索引范围
  workRangePtr = argsBuf->getWorkRange();
  // 获取工作缓冲区指针
  // 用于存储每个工作项的详细信息
  workBufPtr = argsBuf->getWorks(nMaxChannels);
  // 设置最大通道数
  argsBuf->nMaxChannels = nMaxChannels;

  // 主循环：处理所有对称任务
  while (!ncclIntruQueueEmpty(symTaskQueue)) {
    // 声明并初始化设备工作结构体
    struct ncclSymkDevWork devWork = {};
    // 声明变量：剩余单元格和任务单元格
    size_t cellLeft = 0, taskCell = 0;
    // 声明变量：是否为最后一个任务的标志
    uint8_t isSymLast = 0;

    // 检查队列头部的任务是否属于当前内核
    // 如果不属于，说明调度完成
    if (ncclIntruQueueHead(symTaskQueue)->devFuncId != devFuncId) break; // scheduling is done

    // 从队列中取出一个任务
    task = ncclIntruQueueDequeue(symTaskQueue);
    // 保存是否为最后一个任务的标志
    isSymLast = task->isSymLast;

    // 为该任务创建设备工作结构体
    // 包含通道分配、数据范围等信息
    NCCLCHECKGOTO(ncclSymkMakeDevWork(comm, task, &devWork), ret, fail);

    // 计算任务的单元格数
    cellLeft = taskCell = DIVUP(task->count, cellCount);
    // 将任务分配到各个通道
    for (;curChannel < nMaxChannels;) {
      // 记录当前通道的工作项索引（起始）
      workRangePtr[curChannel].workHi = workIndex;
      // 当前通道还没有分配工作项
      if (curChannelWork == 0) {
        // 设备工作还没有分配通道
        if (devWork.nChannels == 0) {
          // 第一次分配：设置起始通道 ID 和通道数
          devWork.sChannelId = curChannel;
          devWork.nChannels = 1;
        // 检查任务是否可以完全放入剩余空间
        } else if (cellLeft <= remainCell) {
          // the last segment of the task
          // 任务的最后一段，可以完全放入当前通道
          assert(devWork.nChannels > 0);
          // if the remaining cell is less than 1024 bytes, we can fuse the last channel
          // 如果剩余空间很小（<=1024字节）或者没有更多任务，可以合并最后一个通道
          if ((remainCell - cellLeft) * NCCL_SYM_KERNEL_CELL_SIZE <= (1 << 10) || ncclIntruQueueEmpty(symTaskQueue)) devWork.nChannels++;
        } else {
          // middle segment of the task
          // 任务的中间段，需要分配新通道
          devWork.nChannels++;
        }
      // 当前通道已有工作项
      } else {
        // 验证单元格数未被修改
        assert(cellLeft == taskCell);
        // 检查任务是否可以完全放入剩余空间
        if (taskCell <= remainCell) {
          // the first segment of the task is fully scheduled onto the channel
          // 任务的第一个段完全放入当前通道
          devWork.sChannelId = curChannel;
          devWork.nChannels = 1;
        }
      }
      // 根据剩余空间和任务大小的关系处理
      if (cellLeft < remainCell) {
        // 任务小于剩余空间，任务可以完全放入当前通道
        // 设置 fracHi 为最大值（0xFFFF），表示整个段都有效
        workRangePtr[curChannel].fracHi = uint16_t(0x10000UL - 1);
        // 减少剩余单元格数
        remainCell -= cellLeft;
        // 当前通道工作项数加 1
        curChannelWork++;
        // 跳出内层循环，准备处理下一个任务
        break;
      } else if (cellLeft == remainCell) {
        // 任务恰好填满当前通道
        workRangePtr[curChannel].fracHi = uint16_t(0x10000UL - 1);
        // 重置剩余单元格数
        remainCell = cellPerChannel;
        // 移动到下一个通道
        curChannel++;
        // 重置通道工作项计数
        curChannelWork = 0;
        // 跳出内层循环
        break;
      } else {
        // cellLeft > remainCell; the task is partially scheduled onto the channel
        // 任务大于剩余空间，只能部分放入当前通道
        // 减少任务的剩余单元格数
        cellLeft -= remainCell;
        // 计算部分的比例（定点数）
        // 0x10000 = 65536，表示 1.0 的定点数
        workRangePtr[curChannel].fracHi = uint16_t(DIVUP(0x10000L * (taskCell - cellLeft), taskCell) - 1);
        // 重置剩余单元格数
        remainCell = cellPerChannel;
        // 移动到下一个通道
        curChannel++;
        // 重置通道工作项计数
        curChannelWork = 0;
      }
    }
    // 将设备工作结构复制到工作缓冲区
    memcpy(workBufPtr + workIndex, &devWork, sizeof(struct ncclSymkDevWork));
    // 工作索引加 1
    workIndex++;

    // Profiler
    // 保存性能分析事件句柄
    plan->groupApiEventHandle = task->groupApiEventHandle;

    // 释放任务结构体内存
    // 从 NCCL 的内存池中释放
    ncclMemoryPoolFree<struct ncclTaskColl>(&comm->memPool_ncclTaskColl, task);
    // 如果是最后一个任务，退出循环
    if (isSymLast == 1) break;
    // 检查是否超出通道限制
    if (curChannel == nMaxChannels) {
      // 通道空间不足，记录警告并失败
      WARN("ncclSymmetricTaskScheduler ran out of channel space (nMaxChannels=%d, workCount=%d, workIndex=%d)",
           nMaxChannels, workCount, workIndex);
      // 跳转到失败标签
      goto fail;
    }
  }
  // 如果最后一个通道还有部分空间未使用，跳过它
  if (remainCell < cellPerChannel) curChannel++;

  // 复制内核通信状态到参数缓冲区
  // kcomm: 内核通信状态，包含拓扑、连接等信息
  memcpy(&argsBuf->kcomm, &comm->symkState.kcomm, sizeof(comm->symkState.kcomm));
  // 设置工作数据总大小（字节数）
  plan->workBytes = totalCount * ncclTypeSize(headTask->datatype);
  // 设置通道掩码（哪些通道参与计算）
  // 低位 curChannel 位被设置为 1，表示前 curChannel 个通道有效
  plan->channelMask = uint64_t(-1) >> (64 - curChannel);
  // 保存内核参数指针
  plan->kernelSymArgs = (void*)argsBuf;
  // 设置工作存储类型为参数缓冲区类型
  plan->workStorageType = ncclDevWorkStorageTypeArgs;

  // 在 rank 0 上输出调优信息
  if (comm->rank == 0) {
    // 输出日志：函数名、数据量、内核名、通道数、线程数、工作项数
    INFO(NCCL_TUNING, "%s [Symmetric]: %ld Bytes -> Kernel %s nchannels %d nthreads %d nWorks %d", funcName,
         logCount * ncclTypeSize(headTask->datatype), kernelName, curChannel, plan->threadPerBlock, workCount);
  }

// 退出标签
exit:
  // 返回结果状态码
  return ret;
// 失败标签
fail:
  // 跳转到退出标签
  goto exit;
}

// 头文件保护结束
#endif // NCCL_SYMMETRIC_SCHED_H_
