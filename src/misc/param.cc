/*************************************************************************
 * Copyright (c) 2019-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "param.h"
#include "debug.h"

#include <algorithm>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>
#include <pthread.h>
#include <mutex>
#include <pwd.h>

const char* userHomeDir() {
  struct passwd *pwUser = getpwuid(getuid());
  return pwUser == NULL ? NULL : pwUser->pw_dir;
}

void setEnvFile(const char* fileName) {
  FILE * file = fopen(fileName, "r");
  if (file == NULL) return;

  char *line = NULL;
  char envVar[1024];
  char envValue[1024];
  size_t n = 0;
  ssize_t read;
  while ((read = getline(&line, &n, file)) != -1) {
    if (line[0] == '#') continue;
    if (line[read-1] == '\n') line[read-1] = '\0';
    int s=0; // Env Var Size
    while (line[s] != '\0' && line[s] != '=') s++;
    if (line[s] == '\0') continue;
    strncpy(envVar, line, std::min(1023,s));
    envVar[std::min(1023,s)] = '\0';
    s++;
    strncpy(envValue, line+s, 1023);
    envValue[1023]='\0';
    setenv(envVar, envValue, 0);
    //printf("%s : %s->%s\n", fileName, envVar, envValue);
  }
  if (line) free(line);
  fclose(file);
}

static void initEnvFunc() {
  char confFilePath[1024];
    
  //通过环境变量获取默认配置文件
  const char* userFile = getenv("NCCL_CONF_FILE");

  //根据配置文件设置环境变量
  if (userFile && strlen(userFile) > 0) {
    //shezhi yonghu zhing dingyi wenjian shezhi 
    snprintf(confFilePath, sizeof(confFilePath), "%s", userFile);
    setEnvFile(confFilePath);
  } else {
    const char* userDir = userHomeDir();
    //用户目录下是否存在.nccl.conf文件，如果是，设置环境变量
    //加载～/.nccl.conf
    if (userDir) {
      snprintf(confFilePath, sizeof(confFilePath), "%s/.nccl.conf", userDir);
      setEnvFile(confFilePath);
    }
  }
  
  //检查默认的配置文件
  snprintf(confFilePath, sizeof(confFilePath), "/etc/nccl.conf");
  setEnvFile(confFilePath);
}

// init env once
void initEnv() {
  static std::once_flag once;
  std::call_once(once, initEnvFunc);
}

void ncclLoadParam(char const* env, int64_t deftVal, int64_t uninitialized, int64_t* cache) {
  static std::mutex mutex;
  std::lock_guard<std::mutex> lock(mutex);
  if (__atomic_load_n(cache, __ATOMIC_RELAXED) == uninitialized) {
    const char* str = ncclGetEnv(env);
    int64_t value = deftVal;
  //若没有设置环境变量，则返回默认值
    if (str && strlen(str) > 0) {
      errno = 0;
      value = strtoll(str, nullptr, 0);
      if (errno) {
        value = deftVal;
        INFO(NCCL_ALL,"Invalid value %s for %s, using default %lld.", str, env, (long long)deftVal);
      } else {
        INFO(NCCL_ENV,"%s set by environment to %lld.", env, (long long)value);
      }
    }
    __atomic_store_n(cache, value, __ATOMIC_RELAXED);
  }
}

const char* ncclGetEnv(const char* name) {
  initEnv();
  return getenv(name);
}
