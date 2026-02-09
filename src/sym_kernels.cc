/*************************************************************************
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// 对称内核头文件，包含内核 ID 定义和设备端工作结构
#include "sym_kernels.h"
// 通信器结构，包含拓扑、配置等信息
#include "comm.h"
// 设备端相关定义，包括架构、计算能力等
#include "device.h"
// 传输层接口，网络、共享内存等
#include "transport.h"
// C++ 数学库，用于性能建模计算
#include <cmath>

// 对称内核名称表（用于调试和日志输出）
// 内核命名规则：<集合操作>_<算法组合>_<协议>
// AG = AllGather, RS = ReduceScatter, AR = AllReduce, ST = Store, LD = Load, LL = LongLink
// MC = Multicast（多播），Net = 网络集合通信
constexpr char const* kernelName[] = {
  // Must align with enum ncclSymkKernelId definition in src/include/sym_kernels.h
  // 必须与 src/include/sym_kernels.h 中的 ncclSymkKernelId 枚举定义对齐
  "AllReduce_AGxLL_R",                    // AllReduce：AllGather + LongLink + Reduce（规约）
  "AllReduce_AGxLLMC_R",                  // AllReduce：AllGather + LongLink + Multicast + Reduce
  "AllReduce_RSxLD_AGxST",                // AllReduce：ReduceScatter + Load + AllGather + Store
  "AllReduce_RSxLDMC_AGxSTMC",            // AllReduce：ReduceScatter + Load + Multicast + AllGather + Store + Multicast
  "AllReduce_RSxNet_ARxMC_AGxNet",        // AllReduce：使用网络集合通信
  "AllGather_LL",                         // AllGather：LongLink 协议
  "AllGather_LLMC",                       // AllGather：LongLink + Multicast
  "AllGather_ST",                         // AllGather：Store 协议
  "AllGather_STMC",                       // AllGather：Store + Multicast
  "ReduceScatter_LL",                     // ReduceScatter：LongLink 协议
  "ReduceScatter_LD",                     // ReduceScatter：Load 协议
  "ReduceScatter_LDMC"                    // ReduceScatter：Load + Multicast
};

// Store + Multicast 内核掩码
// 这些内核使用 STMC（Store + Multicast）协议，需要 NVLS 支持
constexpr uint32_t kernelMask_STMC = 1<<ncclSymkKernelId_AllGather_LLMC |
                                     1<<ncclSymkKernelId_AllGather_STMC |
                                     1<<ncclSymkKernelId_AllReduce_AGxLLMC_R |
                                     1<<ncclSymkKernelId_AllReduce_RSxLDMC_AGxSTMC |
                                     1<<ncclSymkKernelId_ReduceScatter_LDMC;

// Load + Multicast 内核掩码
// 这些内核使用 LDMC（Load + Multicast）协议，需要 NVLS 和特定数据类型支持
constexpr uint32_t kernelMask_LDMC = 1<<ncclSymkKernelId_AllReduce_RSxLDMC_AGxSTMC |
                                     1<<ncclSymkKernelId_ReduceScatter_LDMC;

// LongLink（长链接）内核掩码
// 这些内核使用 LL 协议，适合大消息传输
constexpr uint32_t kernelMask_LL = 1<<ncclSymkKernelId_AllReduce_AGxLL_R |
                                   1<<ncclSymkKernelId_AllReduce_AGxLLMC_R |
                                   1<<ncclSymkKernelId_AllGather_LL |
                                   1<<ncclSymkKernelId_AllGather_LLMC |
                                   1<<ncclSymkKernelId_ReduceScatter_LL;

// AllGather 内核掩码
// 包含所有 AllGather 相关的内核变体
constexpr uint32_t kernelMask_AG = 1<<ncclSymkKernelId_AllGather_LL |
                                   1<<ncclSymkKernelId_AllGather_LLMC |
                                   1<<ncclSymkKernelId_AllGather_ST |
                                   1<<ncclSymkKernelId_AllGather_STMC;

// AllReduce 内核掩码
// 包含所有 AllReduce 相关的内核变体
constexpr uint32_t kernelMask_AR = 1<<ncclSymkKernelId_AllReduce_AGxLLMC_R |
                                   1<<ncclSymkKernelId_AllReduce_AGxLL_R |
                                   1<<ncclSymkKernelId_AllReduce_RSxLDMC_AGxSTMC |
                                   1<<ncclSymkKernelId_AllReduce_RSxLD_AGxST;

// ReduceScatter 内核掩码
// 包含所有 ReduceScatter 相关的内核变体
constexpr uint32_t kernelMask_RS = 1<<ncclSymkKernelId_ReduceScatter_LD |
                                   1<<ncclSymkKernelId_ReduceScatter_LDMC |
                                   1<<ncclSymkKernelId_ReduceScatter_LL;

// 根据集合操作类型获取对应的内核掩码
// 参数：coll - 集合操作类型（AllGather、AllReduce、ReduceScatter）
// 返回：对应操作的内核掩码
static uint32_t kernelMask_coll(ncclFunc_t coll) {
  switch (coll) {
  case ncclFuncAllGather: return kernelMask_AG;         // AllGather 操作掩码
  case ncclFuncAllReduce: return kernelMask_AR;         // AllReduce 操作掩码
  case ncclFuncReduceScatter: return kernelMask_RS;     // ReduceScatter 操作掩码
  default: return 0;                                    // 未知操作，返回空掩码
  }
}

// 获取用户指定的内核掩码（通过环境变量 NCCL_SYM_KERNEL）
// 返回：用户选择的内核掩码（如果设置了环境变量）或全部内核掩码
static uint32_t kernelMask_user() {
  static uint32_t cache = -1u;                         // 静态缓存，避免重复解析环境变量
  uint32_t got = __atomic_load_n(&cache, __ATOMIC_RELAXED); // 原子加载缓存值
  if (got == -1u) {                                    // 首次调用，需要解析环境变量
    // TODO: Enhance this to be a pattern match. I like regex's but we also have
    // TODO: 增强这个功能以支持模式匹配。我喜欢正则表达式，但我们也有
    // the parseList() used by NCCL_ALGO/PROTO.
    // parseList() 函数用于 NCCL_ALGO/PROTO 参数解析
    char const* name = ncclGetEnv("NCCL_SYM_KERNEL");   // 获取环境变量
    if (name == nullptr || strcmp(name, "^") == 0) {    // 未设置或设置为 "^"（表示全部）
      static_assert((int)ncclSymkKernelId_Count < 32, "Use more than 32 bits"); // 断言内核数量少于 32 个
      got = (1<<(int)ncclSymkKernelId_Count)-1;         // 返回全部内核的掩码
    } else {                                            // 用户指定了特定内核
      got = 0;
      for (int k=0; k < (int)ncclSymkKernelId_Count; k++) {
        if (strcmp(kernelName[k], name) == 0) {         // 找到匹配的内核名称
          __atomic_store_n(&cache, 1<<k, __ATOMIC_RELAXED); // 原子存储到缓存
          got = 1<<k;                                   // 返回该内核的掩码
          break;
        }
      }
    }
    __atomic_store_n(&cache, got, __ATOMIC_RELAXED);    // 原子存储结果到缓存
  }
  return got;                                          // 返回内核掩码
}

// 参数定义：对称内核的 CTA 数量（ Cooperative Thread Arrays，即线程块数量）
// 环境变量：NCCL_SYM_CTAS，默认值为 0（自动选择）
NCCL_PARAM(SymCTAs, "SYM_CTAS", 0)

// 软最小值函数（Smooth Minimum）
// 实现平滑的 min(x, ceiling) 函数，用于性能建模中的平滑过渡
// 参数：x - 输入值，ceiling - 上限，softness - 平滑度参数
// 返回：平滑的最小值
static double softmin(double x, double ceiling, double softness) {
  // looks like a smooth version of: min(x, ceiling)
  // 公式：ceiling - softness * log1p((exp(ceiling/softness) - 1) * exp(-x/softness))
  // 当 x << ceiling 时返回 x，当 x >> ceiling 时返回 ceiling，中间平滑过渡
  return ceiling - softness*std::log1p((std::exp(ceiling/softness) - 1)*std::exp(-x/softness));
}

// 软正则化函数（Softplus）
// 实现平滑的 max(0, x) 函数，用于性能建模中的非线性激活
// 参数：x - 输入值，softness - 平滑度参数
// 返回：平滑的正值部分
static double softplus(double x, double softness) {
  // looks like a smooth version of: max(0, x)
  double z = x/softness;                                // 归一化输入
  // 当 z 很大时（>100）直接返回 x，避免数值溢出
  // 否则使用公式：softness * log1p(exp(z))
  return 100.0 <= z ? x : softness*std::log1p(std::exp(z));
}

// 性能模型函数
// 根据总线字节数、延迟、SM 数量等参数估算内核执行时间
// 参数：
//   busBytes      - 总线传输字节数
//   baseLat       - 基础延迟（微秒）
//   nSMs          - SM（Streaming Multiprocessor）数量
//   smBw          - 单个 SM 的带宽（GB/s）
//   busMultiplier - 总线倍数（用于多播等情况）
//   peakBw        - 峰值带宽（GB/s）
// 返回：估算的执行时间（微秒）
static double model(double busBytes, double baseLat, int nSMs, double smBw, double busMultiplier, double peakBw) {
  // 计算有效带宽（受峰值带宽限制）
  double bw = softmin(nSMs*smBw*busMultiplier, peakBw, smBw);
  // 计算总时间：基础延迟 + 传输时间
  // busBytes/bw - 1 表示减去 1 字节的基准时间，然后用 softplus 平滑
  return baseLat + softplus(busBytes/bw - 1, 1);
}

// Given the kernel and bytes, return the minimum number of blocks to run on such that
// perf is 99% of running at max blocks, and return the estimate runtime for that
// block count.
// 给定内核和字节数，返回需要运行的线程块最小数量，使得性能达到最大块数量的 99%，
// 并返回该块数量下的估算运行时间。
// 参数：
//   comm   - 通信器
//   k      - 内核 ID
//   nBytes - 字节数
//   timeUs - 输出：估算时间（微秒）
//   nBlocks - 输出：推荐的块数量
static void queryModel(struct ncclComm* comm, ncclSymkKernelId k, size_t nBytes, float* timeUs, int* nBlocks) {
  constexpr double LL_BusFactor = 9; // 2X the bytes, plus some processing, plus no unrolling
  // LL 总线因子：9 倍字节数（2倍数据 + 处理开销 + 无展开优化）

  int nRanks = comm->nRanks;                           // rank 数量
  int nMaxBlocks = ncclSymkMaxBlocks;                  // 最大块数量
  // NVLS 最大块数量：根据架构计算（Ampere < 1000 用 16，Hopper >= 1000 用 32）
  int nMaxBlocksNvls = divUp((comm->cudaArch < 1000 ? 16 : 32), nRanks);
  size_t busBytes;                                     // 总线字节数（发送和接收的最大值）
  double busMultiplier = 1;                            // 总线倍数（用于调整多播等情况）

  // 根据内核类型计算总线字节数和倍数
  switch (k) {
  default:                                             // 未知内核
    busBytes = size_t(1)<<50;                          // 设置一个很大的值，避免被选中
    break;

  // AllReduce 内核
  case ncclSymkKernelId_AllReduce_AGxLL_R:             // AllReduce + AllGather + LL + Reduce
    busBytes = nRanks*nBytes*LL_BusFactor;             // 所有 rank 聚合，使用 LL 因子
    break;
  case ncclSymkKernelId_AllReduce_AGxLLMC_R:           // AllReduce + AllGather + LL + Multicast + Reduce
    busBytes = nRanks*nBytes*LL_BusFactor;
    busMultiplier = 1.1; // To beat non-MC LL（为了击败非多播 LL 版本）
    break;
  case ncclSymkKernelId_AllReduce_RSxLD_AGxST:         // AllReduce + ReduceScatter + Load + AllGather + Store
    busBytes = 2*nBytes*(nRanks-1)/nRanks;             // 2 倍数据 * (rank-1)/rank
    break;
  case ncclSymkKernelId_AllReduce_RSxLDMC_AGxSTMC:     // AllReduce + 多播版本
    busBytes = nBytes/nRanks + nBytes;                  // 本地规约 + 全局聚合
    busMultiplier = nRanks;                             // 多播倍数等于 rank 数
    nMaxBlocks = nMaxBlocksNvls;                        // 限制最大块数量
    break;

  // AllGather 内核
  case ncclSymkKernelId_AllGather_LL:                  // AllGather + LL
    busBytes = nRanks*nBytes*LL_BusFactor;             // 所有 rank 数据聚合
    break;
  case ncclSymkKernelId_AllGather_LLMC:                // AllGather + LL + Multicast
    busBytes = nRanks*nBytes*LL_BusFactor;
    busMultiplier = 1.1; // To beat non-MC LL（为了击败非多播 LL 版本）
    break;
  case ncclSymkKernelId_AllGather_ST:                  // AllGather + Store
    busBytes = (nRanks-1)*nBytes;                      // 接收 (rank-1) 份数据
    break;
  case ncclSymkKernelId_AllGather_STMC:                // AllGather + Store + Multicast
    busBytes = (nRanks-1)*nBytes; // Wrong. Should be nRanks*nBytes but we want to beat non-MC.
    // 注释：这里故意设置较小的值（应该是 nRanks*nBytes），以鼓励使用多播版本
    busMultiplier = 0.55*nRanks;                       // 多播倍数
    nMaxBlocks = nMaxBlocksNvls;                        // 限制最大块数量
    break;

  // ReduceScatter 内核
  case ncclSymkKernelId_ReduceScatter_LL:              // ReduceScatter + LL
    busBytes = nRanks*nBytes*LL_BusFactor;             // 散播到所有 rank
    break;
  case ncclSymkKernelId_ReduceScatter_LD:              // ReduceScatter + Load
    busBytes = (nRanks-1)*nBytes;                      // 发送 (rank-1) 份数据
    break;
  case ncclSymkKernelId_ReduceScatter_LDMC:            // ReduceScatter + Load + Multicast
    busBytes = (nRanks-1)*nBytes; // Wrong. Should be nRanks*nBytes but we want to beat non-MC.
    // 注释：这里故意设置较小的值，以鼓励使用多播版本
    busMultiplier = 0.55*nRanks;                       // 多播倍数
    nMaxBlocks = nMaxBlocksNvls;                        // 限制最大块数量
    break;
  }

  // 限制最大块数量不超过配置值
  nMaxBlocks = std::min<int>(nMaxBlocks, comm->config.maxCTAs);
  int nMinBlocks = comm->config.minCTAs;                // 最小块数量

  // 如果用户指定了 CTA 数量，使用用户指定的值
  int nUserCTAs = std::min<int>(ncclSymkMaxBlocks, ncclParamSymCTAs());
  if (nUserCTAs > 0) nMinBlocks = nMaxBlocks = nUserCTAs;

  // 判断内核类型
  bool isLL = kernelMask_LL>>k & 1;                     // 是否为 LL 内核
  bool isAG = kernelMask_AG>>k & 1;                     // 是否为 AllGather 内核
  bool isAR = kernelMask_AR>>k & 1;                     // 是否为 AllReduce 内核
  constexpr double GBps = (1<<30)/1.e6;                 // GB/s 单位转换因子
  double baseLat, smBw, peakBw;                         // 延迟和带宽参数

  // 根据 GPU 架构设置性能参数
  if (comm->cudaArch < 1000) {                          // Ampere 架构 (sm_80)
    baseLat = isLL ? 4.5 : 7.8;                         // 基础延迟（微秒）：LL 更快
    smBw = isAR ? 65*GBps : 44*GBps;                     // SM 带宽：AllReduce 更高
    peakBw = k == ncclSymkKernelId_AllReduce_RSxLDMC_AGxSTMC ? 480*GBps : 320*GBps; // 峰值带宽
  } else {                                              // Hopper 架构 (sm_90+)
    baseLat = isLL ? (isAG ? 8.5 : 11) : (isAR ? 19.5 : 13.0); // 基础延迟
    smBw = 55*GBps;                                     // SM 带宽
    peakBw = k == ncclSymkKernelId_AllReduce_RSxLDMC_AGxSTMC ? 1000*GBps : 600*GBps; // 峰值带宽
  }

  // 使用最大块数量计算初始时间
  *nBlocks = nMaxBlocks;
  *timeUs = model(busBytes, baseLat, nMaxBlocks, smBw, busMultiplier, peakBw);

  // Use least number of blocks that puts us within a tolerance of peak performance.
  // 使用最少块数量，使得性能在峰值性能的容差范围内（2.5%）
  for (int bn = nMinBlocks; bn < nMaxBlocks; bn++) {
    double time = model(busBytes, baseLat, bn, smBw, busMultiplier, peakBw);
    if (time <= 1.025*(*timeUs)) {                     // 如果时间在峰值性能的 2.5% 范围内
      *nBlocks = bn;                                    // 使用更少的块
      *timeUs = time;
      break;
    }
  }
}

// 对称内核一次性初始化
// 功能：初始化设备端通信器和相关资源
// 参数：comm - NCCL 通信器
// 返回：ncclResult_t - 操作结果状态码
ncclResult_t ncclSymkInitOnce(struct ncclComm* comm) {
  struct ncclSymkState* symk = &comm->symkState;       // 获取对称内核状态
  if (!symk->initialized) {                            // 如果未初始化
    symk->initialized = true;                          // 标记为已初始化
    struct ncclDevCommRequirements reqs = {};          // 设备端通信器需求
    reqs.lsaMultimem = comm->nvlsSupport;              // 是否启用 LSA 多播（取决于 NVLS 支持）
    reqs.lsaBarrierCount = ncclSymkMaxBlocks;          // LSA barrier 数量（等于最大块数）

    // 创建 LLA2A（Load-Load-Accumulate-Accumulate）资源需求
    // 用于设备端原子操作和规约操作
    struct ncclDevResourceRequirements lla2aReq;
    ncclLLA2ACreateRequirement(
      ncclSymkMaxBlocks,                               // 最大块数
      ncclLLA2ACalcSlots(ncclTeamLsa(comm).nRanks*ncclSymkMaxThreads, ncclSymkLLMaxEltSize), // 计算槽位数
      &symk->kcomm.lsaLLA2A,                            // LLA2A 通信器
      &lla2aReq                                        // 资源需求输出
    );
    lla2aReq.next = reqs.resourceRequirementsList;     // 添加到资源需求链表
    reqs.resourceRequirementsList = &lla2aReq;

    // 创建设备端通信器（分配资源窗口、初始化团队等）
    NCCLCHECK(ncclDevrCommCreateInternal(comm, &reqs, &symk->kcomm.devComm));
  }
  return ncclSuccess;                                  // 返回成功
}

// 对称内核清理函数
// 功能：释放对称内核相关的所有资源
// 参数：comm - NCCL 通信器
// 返回：ncclResult_t - 操作结果状态码
ncclResult_t ncclSymkFinalize(struct ncclComm* comm) {
  struct ncclSymkState* symk = &comm->symkState;       // 获取对称内核状态
  if (symk->initialized) {                            // 如果已初始化
    // 销毁设备端通信器（释放资源窗口等）
    NCCLCHECK(ncclDevCommDestroy(comm, &symk->kcomm.devComm));
  }
  return ncclSuccess;                                  // 返回成功
}

// 检查对称内核是否实现了指定的集合操作和数据类型
// 参数：
//   coll - 集合操作类型
//   red  - 规约操作
//   ty   - 数据类型
// 返回：true 如果已实现，false 否则
static bool ncclSymkImplemented(ncclFunc_t coll, int/*ncclDevRedOp_t*/ red, ncclDataType_t ty) {
  bool isFloat;                                         // 是否为浮点类型
  switch (ty) {
  case ncclFloat64:                                    // 双精度浮点
  case ncclFloat32:                                    // 单精度浮点
  case ncclFloat16:                                    // 半精度浮点
  case ncclBfloat16:                                   // 脑浮点（BFloat16）
  case ncclFloat8e4m3:                                 // 8位浮点 E4M3 格式
  case ncclFloat8e5m2:                                 // 8位浮点 E5M2 格式
    isFloat = true;
    break;
  default:
    isFloat = false;                                   // 整数类型
    break;
  }

  // 根据集合操作类型检查支持情况
  switch (coll) {
  case ncclFuncAllGather:                             // AllGather：所有数据类型都支持
    return true;
  case ncclFuncAllReduce:                             // AllReduce/ReduceScatter：仅支持浮点类型的求和规约
  case ncclFuncReduceScatter:
    return red == ncclDevSum && isFloat && ty != ncclFloat64; // 不支持双精度浮点
  default:
    return false;                                      // 其他操作不支持
  }
}

// 计算可用的对称内核掩码
// 功能：根据集合操作、规约操作、数据类型和数据大小，过滤出可用的内核
// 参数：
//   comm   - 通信器
//   coll   - 集合操作类型
//   red    - 规约操作
//   ty     - 数据类型
//   nElts  - 元素数量
// 返回：可用内核的位掩码
static uint32_t ncclSymkMask(struct ncclComm* comm, ncclFunc_t coll, int/*ncclDevRedOp_t*/ red, ncclDataType_t ty, size_t nElts) {
  uint32_t kmask = kernelMask_coll(coll);              // 获取集合操作对应的内核掩码
  kmask &= kernelMask_user();                          // 应用用户指定的内核过滤

  // 检查 NVLS 多播支持
  bool hasSTMC = comm->nvlsSupport;                    // 是否支持 STMC（Store + Multicast）
  bool hasLDMC = false;                                // 是否支持 LDMC（Load + Multicast）
  if (comm->nvlsSupport) {                             // 如果支持 NVLS
    switch (ty) {                                      // 根据数据类型判断多播支持
    case ncclInt32:                                   // 32位整数
    case ncclUint32:
    case ncclInt64:                                   // 64位整数
    case ncclUint64:
    case ncclFloat16:                                 // 半精度浮点
    case ncclBfloat16:                                // BFloat16
      hasLDMC = red == ncclDevSum || red == ncclDevMinMax; // 支持求和和最小/最大值
      break;
    case ncclFloat8e4m3:                             // 8位浮点 E4M3
    case ncclFloat8e5m2:                             // 8位浮点 E5M2
      hasLDMC = red == ncclDevSum || red == ncclDevMinMax;
      hasLDMC &= comm->compCap >= 100;                 // 需要计算能力 >= 100（Hopper）
      break;
    case ncclFloat:                                   // 单精度浮点（C++ 类型）
    case ncclDouble:                                  // 双精度浮点（C++ 类型）
      hasLDMC = red == ncclDevSum;                     // 仅支持求和
      break;
    default: break;
    }
  }
  // 根据多播支持情况过滤内核
  if (!hasSTMC) kmask &= ~kernelMask_STMC;             // 移除 STMC 内核
  if (!hasLDMC) kmask &= ~kernelMask_LDMC;             // 移除 LDMC 内核

  // 根据数据大小过滤内核
  size_t nBytes = nElts*ncclTypeSize(ty);               // 字节数
  size_t nBusBytes = (coll == ncclFuncAllReduce ? 1 : comm->nRanks)*nBytes; // 总线字节数
  // LL kernels use 32-bit ints to track element counts and indices.
  // LL 内核使用 32 位整数来跟踪元素计数和索引
  if (nBusBytes >= (size_t(2)<<30)) kmask &= ~kernelMask_LL; // 如果 >= 2GB，移除 LL 内核
  // Any kernel might use 32-bit int to track unrolled loop chunks (which are going
  // to be at least 32 bytes per chunk)
  // 任何内核都可能使用 32 位整数来跟踪展开的循环块（每块至少 32 字节）
  if (nBusBytes >= 32*(size_t(2)<<30)) kmask = 0;     // 如果 >= 64GB，移除所有内核

  return kmask;                                        // 返回过滤后的内核掩码
}

// 检查对称内核是否可用
// 功能：判断指定的参数组合是否有可用的对称内核
// 参数：
//   comm  - 通信器
//   coll  - 集合操作类型
//   red   - 规约操作
//   ty    - 数据类型
//   nElts - 元素数量
// 返回：true 如果有可用内核，false 否则
bool ncclSymkAvailable(struct ncclComm* comm, ncclFunc_t coll, int/*ncclDevRedOp_t*/ red,
                       ncclDataType_t ty, size_t nElts) {
  if (!ncclSymkImplemented(coll, red, ty))             // 检查是否已实现
    return false;

  // 检查是否有可用的内核（掩码非空）
  return (ncclSymkMask(comm, coll, red, ty, nElts) != 0);
}

// 选择最佳对称内核
// 功能：从可用内核中选择性能最优的内核
// 参数：
//   comm        - 通信器
//   coll        - 集合操作类型
//   red         - 规约操作
//   ty          - 数据类型
//   nEltsTotal  - 总元素数量
//   nEltsMax    - 最大元素数量
//   nWorks      - 工作项数量
//   estTimeUs   - 输出：估算时间（微秒）
//   kernelId    - 输出：选中的内核 ID
//   nBlocks     - 输出：推荐的块数量
//   nWarps      - 输出：推荐的 warp 数量
// 返回：ncclResult_t - 操作结果状态码
ncclResult_t ncclSymkPickKernel(
    struct ncclComm* comm, ncclFunc_t coll, int/*ncclDevRedOp_t*/ red, ncclDataType_t ty,
    size_t nEltsTotal, size_t nEltsMax, int nWorks,
    float* estTimeUs, ncclSymkKernelId* kernelId, int* nBlocks, int* nWarps
  ) {
  uint32_t kmask = ncclSymkMask(comm, coll, red, ty, nEltsMax); // 获取可用内核掩码

  // We currently don't support grouping for LL kernels.
  // 我们目前不支持 LL 内核的分组（grouping）
  if (nWorks > 1)                                    // 如果有多个工作项
    kmask &= ~kernelMask_LL;                         // 移除 LL 内核

  ncclSymkKernelId bestKernel = ncclSymkKernelId_Count; // 最佳内核 ID
  float bestTime = 1.e30f;                           // 最佳时间（初始化为很大值）
  int bestBlocks = 999;                              // 最佳块数量
  size_t nBytes = nEltsTotal*ncclTypeSize(ty);        // 总字节数

  constexpr float smPenalty = .025f;                // SM 惩罚因子：每增加一个 SM 增加 2.5% 时间
  uint32_t kmaskRemain = kmask;                      // 剩余内核掩码
  // 遍历所有可用内核，找到性能最优的
  while (kmaskRemain != 0) {
    ncclSymkKernelId k = (ncclSymkKernelId)popFirstOneBit(&kmaskRemain); // 弹出第一个内核
    float kTime;                                      // 当前内核时间
    int kBlocks;                                      // 当前内核块数
    queryModel(comm, k, nBytes, &kTime, &kBlocks);   // 查询性能模型
    // 比较性能：考虑 SM 惩罚（块数越多，资源占用越大）
    if (kTime*(1.0f + smPenalty*kBlocks) < bestTime*(1.0f + smPenalty*bestBlocks)) {
      bestKernel = k;                                 // 更新最佳内核
      bestTime = kTime;                               // 更新最佳时间
      bestBlocks = kBlocks;                           // 更新最佳块数
    }
  }

  *kernelId = bestKernel;                             // 输出选中的内核
  // 如果用户未指定特定内核，返回估算时间；否则返回 0（强制使用）
  *estTimeUs = kmask==0 || kernelMask_user() == (1<<ncclSymkKernelId_Count)-1 ? bestTime : 0.0f;
  *nBlocks = bestBlocks;                              // 输出推荐的块数
  *nWarps = 16;                                       // 输出推荐的 warp 数（固定为 16）
  return ncclSuccess;
}

// 将内核 ID 转换为字符串
// 参数：kernelId - 内核 ID
// 返回：内核名称字符串，如果 ID 无效则返回 "Unknown"
const char* ncclSymkKernelIdToString(int kernelId) {
  if (kernelId < 0 || kernelId >= ncclSymkKernelId_Count) { // 检查 ID 范围
    return "Unknown";                                 // 无效 ID
  }
  return kernelName[kernelId];                        // 返回内核名称
}

/* this function fills in the devWork except nextWorkOffset */
/* 此函数填充 devWork 结构，但不包括 nextWorkOffset */
// 创建设备端工作结构
// 功能：将集合任务的信息转换为设备端工作结构
// 参数：
//   comm       - 通信器
//   task       - 集合任务
//   outDevWork - 输出：设备端工作结构
// 返回：ncclResult_t - 操作结果状态码
ncclResult_t ncclSymkMakeDevWork(struct ncclComm* comm, struct ncclTaskColl* task, struct ncclSymkDevWork* outDevWork) {
  outDevWork->rootRank = task->root;                  // 根 rank
  outDevWork->redOpArg = task->opDev.scalarArg;      // 规约操作的标量参数（如平均值计算时的除数）
  outDevWork->nElts = task->count;                    // 元素数量
  outDevWork->inputWin = task->sendWin->vidmem;       // 输入窗口的设备端结构
  // 计算输入缓冲区在窗口中的偏移量
  outDevWork->inputOff = (uint8_t*)task->sendbuff - (uint8_t*)task->sendWin->userPtr;
  outDevWork->outputWin = task->recvWin->vidmem;      // 输出窗口的设备端结构
  // 计算输出缓冲区在窗口中的偏移量
  outDevWork->outputOff = (uint8_t*)task->recvbuff - (uint8_t*)task->recvWin->userPtr;
  outDevWork->sChannelId = 0xffff;                    // 源通道 ID（0xFFFF 表示未使用）
  outDevWork->nChannels = 0;                          // 通道数量（对称内核不使用通道）
  return ncclSuccess;
}
