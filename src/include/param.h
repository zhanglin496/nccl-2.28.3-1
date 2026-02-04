/*************************************************************************
 * Copyright (c) 2017-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_PARAM_H_
#define NCCL_PARAM_H_

#include <stdint.h>

const char* userHomeDir();
void setEnvFile(const char* fileName);
void initEnv();
const char *ncclGetEnv(const char *name);

void ncclLoadParam(char const* env, int64_t deftVal, int64_t uninitialized, int64_t* cache);

//通过宏实现函数
#define NCCL_PARAM(name, env, deftVal) \
  int64_t ncclParam##name() { \
    constexpr int64_t uninitialized = INT64_MIN; \
    static_assert(deftVal != uninitialized, "default value cannot be the uninitialized value."); \
    //定义局部静态变量缓存上次获取的值
    static int64_t cache = uninitialized; \
    //检查是否已经获取过 
    if (__builtin_expect(__atomic_load_n(&cache, __ATOMIC_RELAXED) == uninitialized, false)) { \
      ncclLoadParam("NCCL_" env, deftVal, uninitialized, &cache); \
    } \
    return cache; \
  }

#endif
