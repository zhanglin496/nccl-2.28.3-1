/*
 * Copyright (c) 2016-2023, NVIDIA CORPORATION. All rights reserved.
 * 版权所有 (c) 2016-2023，NVIDIA 公司。保留所有权利。
 *
 * See COPYRIGHT for license information
 * 有关许可信息，请参见 COPYRIGHT 文件
 */

// 包含 NCCL IPC 套接字头文件，定义了套接字操作相关的数据结构和函数声明
#include "ipcsocket.h"
// 包含 NCCL 工具函数头文件，提供日志、宏定义等辅助功能
#include "utils.h"
// 标准库头文件，提供内存分配、进程控制等函数
#include <stdlib.h>
// 字符串处理头文件，提供 strlen、strcpy、strncpy 等字符串操作函数
#include <string.h>
// 错误码头文件，定义了 errno 变量和各种错误码常量（如 EAGAIN、EINTR 等）
#include <errno.h>

// Enable Linux abstract socket naming
// 启用 Linux 抽象套接字命名机制
// 抽象套接字是 Linux 特有的特性，不需要在文件系统中创建套接字文件
// 套接字名称以 null 字节开头，存储在内核的命名空间中，进程退出时自动清理
#define USE_ABSTRACT_SOCKET

// 定义 IPC 套接字的名称格式字符串模板
// %d: 会被替换为 rank（进程的 rank 编号）
// %lx: 会被替换为 hash（64位哈希值，用于唯一标识）
// 例如: /tmp/nccl-socket-0-a1b2c3d4
#define NCCL_IPC_SOCKNAME_STR "/tmp/nccl-socket-%d-%lx"

/*
 * Create a Unix Domain Socket
 * 创建一个 Unix 域套接字
 * Unix 域套接字用于同一主机上的进程间通信（IPC）
 */
// 函数功能：初始化并创建一个 Unix 域套接字
// 参数说明：
//   - handle: 指向 ncclIpcSocket 结构体的指针，用于返回套接字句柄
//   - rank: 当前进程的 rank 编号，用于生成唯一的套接字名称
//   - hash: 64位哈希值，用于生成唯一的套接字名称
//   - abortFlag: 原子标志指针，用于非阻塞模式下的中止检测
// 返回值：ncclSuccess 表示成功，其他值表示失败
ncclResult_t ncclIpcSocketInit(ncclIpcSocket *handle, int rank, uint64_t hash, volatile uint32_t* abortFlag) {
  // 声明文件描述符变量，初始化为 -1（表示无效的文件描述符）
  // 在 Unix/Linux 中，-1 通常用于表示文件描述符未初始化或已关闭
  int fd = -1;
  // 声明 Unix 域套接字地址结构体
  // struct sockaddr_un 用于存储 Unix 域套接字的地址信息
  // 包含地址族（sun_family）和套接字路径（sun_path）
  struct sockaddr_un cliaddr;
  // 声明临时字符数组，用于存储生成的套接字名称
  // NCCL_IPC_SOCKNAME_LEN 是预定义的最大长度常量
  // 初始化为空字符串，确保内存清零
  char temp[NCCL_IPC_SOCKNAME_LEN] = "";

  // 参数有效性检查：验证句柄指针是否为 NULL
  // 如果传入的 handle 指针为空，说明调用者传递了无效参数
  if (handle == NULL) {
    // 返回内部错误码，表示函数调用使用了无效参数
    return ncclInternalError;
  }

  // 初始化句柄结构体中的文件描述符为 -1
  // -1 表示文件描述符尚未分配或无效
  // 这是一种防御性编程，确保结构体处于已知状态
  handle->fd = -1;
  // 将套接字名称的第一个字符设置为空字符 '\0'
  // 这实际上是清空字符串，确保 socketName 字段初始化为空字符串
  handle->socketName[0] = '\0';
  // 创建 Unix 域套接字
  // AF_UNIX: 表示使用 Unix 域地址族（本地进程间通信）
  // SOCK_DGRAM: 使用数据报套接字（无连接、不可靠的消息传递）
  // 0: 协议类型，0 表示自动选择默认协议
  // 返回值：成功返回文件描述符（非负整数），失败返回 -1
  if ((fd = socket(AF_UNIX, SOCK_DGRAM, 0)) < 0) {
    // 记录警告日志，输出套接字创建失败的错误信息
    // WARN 是 NCCL 的日志宏，用于输出警告级别的日志
    // strerror(errno): 将错误码转换为可读的错误描述字符串
    WARN("UDS: Socket creation error : %s (%d)", strerror(errno), errno);
    // 返回系统错误码，表示操作系统级别的错误（如创建套接字失败）
    return ncclSystemError;
  }

  // 清空套接字地址结构体的内存
  // bzero 是 BSD 函数，将指定字节的内存设置为零
  // &cliaddr: 要清零的内存地址
  // sizeof(cliaddr): 要清零的字节数
  // 这确保了结构体中的所有字段都被初始化为零
  bzero(&cliaddr, sizeof(cliaddr));
  // 设置地址族为 AF_UNIX（Unix 域）
  // sun_family 字段必须设置为 AF_UNIX，表示这是一个 Unix 域套接字地址
  cliaddr.sun_family = AF_UNIX;

  // Create unique name for the socket.
  // 为套接字创建唯一的名称
  // 使用格式化字符串生成唯一的套接字路径名
  // snprintf: 格式化输出到字符串，防止缓冲区溢出
  // temp: 目标缓冲区，存储生成的套接字名称
  // NCCL_IPC_SOCKNAME_LEN: 缓冲区最大长度，防止溢出
  // NCCL_IPC_SOCKNAME_STR: 格式化字符串模板
  // rank: 进程的 rank 编号
  // hash: 64位哈希值
  // 返回值: 实际写入的字符数（不包括结尾的 '\0'）
  int len = snprintf(temp, NCCL_IPC_SOCKNAME_LEN, NCCL_IPC_SOCKNAME_STR, rank, hash);
  // 检查生成的套接字名称长度是否超过系统限制
  // sizeof(cliaddr.sun_path) - 1: sun_path 数组的最大可用长度（-1 是为了保留空间给 '\0'）
  // 如果名称太长，无法放入 sun_path 数组，则返回错误
  if (len > (sizeof(cliaddr.sun_path) - 1)) {
    // 记录警告日志：套接字名称过长，无法绑定
    WARN("UDS: Cannot bind provided name to socket. Name too large");
    // 关闭已创建的套接字文件描述符，释放系统资源
    // close() 是系统调用，用于关闭文件描述符
    close(fd);
    // 返回内部错误，表示参数或配置错误
    return ncclInternalError;
  }
#ifndef USE_ABSTRACT_SOCKET
  // 如果不使用抽象套接字，则删除可能存在的旧套接字文件
  // unlink() 系统调用删除文件系统中的文件
  // 在绑定之前先删除同名文件，避免 bind() 因文件已存在而失败
  // 这是传统 Unix 域套接字的使用方式，需要在文件系统中创建套接字文件
  unlink(temp);
#endif

  // 记录跟踪日志：正在创建套接字
  // TRACE 是 NCCL 的调试跟踪宏，用于输出详细的调试信息
  // NCCL_INIT: 日志类别，表示这是初始化相关的日志
  TRACE(NCCL_INIT, "UDS: Creating socket %s", temp);

  // 将生成的套接字名称复制到地址结构的 sun_path 字段
  // strncpy: 字符串复制函数，复制指定长度的字符
  // cliaddr.sun_path: 目标地址
  // temp: 源字符串（生成的套接字名称）
  // len: 要复制的字符数
  strncpy(cliaddr.sun_path, temp, len);
#ifdef USE_ABSTRACT_SOCKET
  // Linux 抽象套接字技巧
  // 将 sun_path 的第一个字符设置为 '\0'（null 字符）
  // 这告诉 Linux 内核这是一个抽象套接字
  // 抽象套接字不会在文件系统中创建文件，而是存储在内核的命名空间中
  // 优点：1) 不需要文件系统权限；2) 进程退出时自动清理；3) 避免文件名冲突
  cliaddr.sun_path[0] = '\0'; // Linux abstract socket trick
  // Linux 抽象套接字技巧：使用 null 字节开头的套接字名称
#endif
  // 将套接字绑定到指定的地址
  // bind(): 系统调用，将套接字与本地地址关联
  // fd: 套接字文件描述符
  // (struct sockaddr *)&cliaddr: 地址结构的通用指针
  // sizeof(cliaddr): 地址结构的大小
  // 返回值：成功返回 0，失败返回 -1
  if (bind(fd, (struct sockaddr *)&cliaddr, sizeof(cliaddr)) < 0) {
    // 记录警告日志：绑定套接字失败，输出错误信息
    WARN("UDS: Binding to socket %s failed : %s (%d)", temp, strerror(errno), errno);
    // 关闭套接字文件描述符，释放资源
    close(fd);
    // 返回系统错误码
    return ncclSystemError;
  }

  // 将文件描述符保存到句柄结构体中
  // 这样后续的操作可以使用这个文件描述符
  handle->fd = fd;
  // 将套接字名称复制到句柄结构体中保存
  // strcpy: 标准字符串复制函数
  // 保存名称用于后续的日志输出和资源清理
  strcpy(handle->socketName, temp);

  // 保存中止标志指针到句柄结构体
  // abortFlag 是一个原子变量指针，用于在非阻塞模式下检测是否需要中止操作
  // 其他线程或信号处理器可以设置这个标志来中止正在进行的套接字操作
  handle->abortFlag = abortFlag;
  // Mark socket as non-blocking
  // 将套接字标记为非阻塞模式
  // 只有在提供了 abortFlag 时才设置为非阻塞模式
  // 非阻塞模式允许 recv/send 操作立即返回，而不是等待数据就绪
  if (handle->abortFlag) {
    // 声明变量用于存储文件描述符的标志位
    int flags;
    // 获取文件描述符的当前标志位
    // fcntl(): 文件控制函数，用于操作文件描述符的属性
    // F_GETFL: 命令码，表示获取文件状态标志
    // "fcntl": 用于错误日志的描述字符串
    // SYSCHECK: NCCL 宏，检查系统调用返回值，失败时输出错误并返回
    SYSCHECK(flags = fcntl(fd, F_GETFL), "fcntl");
    // 设置文件描述符为非阻塞模式
    // F_SETFL: 命令码，表示设置文件状态标志
    // flags | O_NONBLOCK: 在原有标志基础上添加 O_NONBLOCK 标志
    // O_NONBLOCK: 非阻塞标志，使 I/O 操作立即返回
    SYSCHECK(fcntl(fd, F_SETFL, flags | O_NONBLOCK), "fcntl");
  }

  // 返回成功状态码
  // ncclSuccess 表示函数执行成功
  return ncclSuccess;
}

// 函数功能：获取套接字的文件描述符
// 这是一个简单的访问器函数，允许外部获取套接字的文件描述符
// 参数说明：
//   - handle: 指向 ncclIpcSocket 结构体的指针
//   - fd: 指向整数的指针，用于返回文件描述符
// 返回值：ncclSuccess 表示成功，ncclInvalidArgument 表示参数无效
ncclResult_t ncclIpcSocketGetFd(struct ncclIpcSocket* handle, int* fd) {
  // 参数有效性检查：验证句柄指针是否为 NULL
  if (handle == NULL) {
    // 记录警告日志：传递了 NULL 套接字句柄
    WARN("ncclSocketGetFd: pass NULL socket");
    // 返回无效参数错误码
    return ncclInvalidArgument;
  }
  // 如果 fd 指针不为 NULL，则将文件描述符值写入到 fd 指向的内存
  // 使用 if (fd) 检查指针是否为 NULL 是一种防御性编程
  if (fd) *fd = handle->fd;
  // 返回成功状态码
  return ncclSuccess;
}

// 函数功能：关闭 Unix 域套接字并释放相关资源
// 这个函数负责关闭套接字文件描述符，并在非抽象套接字模式下删除套接字文件
// 参数说明：
//   - handle: 指向 ncclIpcSocket 结构体的指针
// 返回值：ncclSuccess 表示成功
ncclResult_t ncclIpcSocketClose(ncclIpcSocket *handle) {
  // 参数有效性检查：验证句柄指针是否为 NULL
  if (handle == NULL) {
    // 返回内部错误码
    return ncclInternalError;
  }
  // 检查文件描述符是否有效
  // 有效的文件描述符应该是正整数
  // <= 0 表示文件描述符无效或未初始化，无需关闭
  if (handle->fd <= 0) {
    // 文件描述符无效，直接返回成功
    // 这避免了关闭无效的文件描述符（可能关闭了其他重要的 fd，如 0=stdin, 1=stdout, 2=stderr）
    return ncclSuccess;
  }
#ifndef USE_ABSTRACT_SOCKET
  // 在非抽象套接字模式下，需要删除文件系统中的套接字文件
  // 检查套接字名称是否非空（第一个字符不是 '\0'）
  if (handle->socketName[0] != '\0') {
    // 删除套接字文件
    // unlink() 删除文件系统中的文件链接
    // 当所有进程都关闭了这个文件的文件描述符后，文件才会被真正删除
    handle->socketName: 要删除的套接字文件路径
    unlink(handle->socketName);
  }
#endif
  // 关闭套接字文件描述符
  // close() 系统调用关闭文件描述符，释放系统资源
  // 关闭后，该文件描述符可以被重用
  close(handle->fd);

  // 返回成功状态码
  return ncclSuccess;
}

// 函数功能：从 Unix 域套接字接收消息，可选接收文件描述符
// Unix 域套接字的一个独特特性是可以通过 sendmsg/recvmsg 传递文件描述符
// 这允许进程间共享打开的文件、管道、套接字等资源
// 参数说明：
//   - handle: 指向 ncclIpcSocket 结构体的指针
//   - hdr: 指向接收数据缓冲区的指针，如果为 NULL 则使用临时缓冲区
//   - hdrLen: 接收缓冲区的长度（字节数）
//   - recvFd: 指向整数的指针，用于接收传递过来的文件描述符，如果为 NULL 则不接收
// 返回值：ncclSuccess 表示成功
ncclResult_t ncclIpcSocketRecvMsg(ncclIpcSocket *handle, void *hdr, int hdrLen, int *recvFd) {
  // 声明并初始化消息头结构体
  // struct msghdr 用于 sendmsg/recvmsg 系统调用，包含消息的所有信息
  // 初始化为全零：{0, 0, 0, 0, 0, 0, 0} 分别对应 msg_name, msg_namelen, msg_iov, msg_iovlen,
  //                                             msg_control, msg_controllen, msg_flags
  struct msghdr msg = {0, 0, 0, 0, 0, 0, 0};
  // 声明 iovec 数组，用于分散/聚集 I/O
  // struct iovec 表示一个内存缓冲区：iov_base 指向缓冲区地址，iov_len 表示长度
  // 这里只需要一个缓冲区，所以数组大小为 1
  struct iovec iov[1];

  // Union to guarantee alignment requirements for control array
  // 使用联合体（union）来保证控制数组的对齐要求
  // 联合体的所有成员共享同一块内存，大小等于最大成员的大小
  // 这样可以确保控制缓冲区有正确的对齐，满足 cmsghdr 的对齐要求
  union {
    // cmsghdr 结构体，用于描述辅助数据（如文件描述符）
    struct cmsghdr cm;
    // 控制缓冲区，使用 CMSG_SPACE 计算所需空间
    // CMSG_SPACE(sizeof(int)) 计算存储一个整数（文件描述符）所需的辅助数据空间
    // 包括：cmsghdr 结构体、数据、以及对齐填充
    char control[CMSG_SPACE(sizeof(int))];
  } control_un;

  // 声明辅助消息头指针，用于遍历辅助数据
  struct cmsghdr *cmptr;
  // 声明虚拟缓冲区，当 hdr 为 NULL 时使用
  // 即使不需要接收数据，recvmsg 也需要一个有效的缓冲区
  char dummy_buffer[1];
  // 声明接收返回值变量
  int ret;

  // 设置消息的控制数据字段
  // msg.msg_control: 指向辅助数据的指针（辅助数据用于传递文件描述符、凭证等）
  msg.msg_control = control_un.control;
  // msg.msg_controllen: 辅助数据缓冲区的长度
  msg.msg_controllen = sizeof(control_un.control);

  // 根据 hdr 参数设置 I/O 向量
  // 如果 hdr 为 NULL，表示调用者不关心接收的数据内容
  if (hdr == NULL) {
    // 使用虚拟缓冲区接收数据
    // iov_base: 指向缓冲区的指针
    iov[0].iov_base = (void *)dummy_buffer;
    // iov_len: 缓冲区的大小
    iov[0].iov_len = sizeof(dummy_buffer);
  } else {
    // 使用调用者提供的缓冲区接收数据
    iov[0].iov_base = hdr;
    // 使用调用者指定的缓冲区长度
    iov[0].iov_len = hdrLen;
  }

  // 将 I/O 向量数组关联到消息结构体
  // msg.msg_iov: 指向 iovec 数组的指针
  msg.msg_iov = iov;
  // msg.msg_iovlen: iovec 数组的元素个数
  msg.msg_iovlen = 1;

  // 循环接收消息，直到成功或发生错误
  // recvmsg(): 系统调用，从套接字接收消息
  // handle->fd: 套接字文件描述符
  // &msg: 指向消息结构的指针
  // 0: 标志位，0 表示默认行为
  // 返回值：成功返回接收的字节数，失败返回 -1
  while ((ret = recvmsg(handle->fd, &msg, 0)) <= 0) {
    // 检查错误类型
    // EAGAIN: 资源暂时不可用（非阻塞模式下没有数据）
    // EWOULDBLOCK: 操作会阻塞（通常与 EAGAIN 相同）
    // EINTR: 系统调用被信号中断
    // 这些是"正常"的错误，应该重试
    if (errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR) {
      // 其他错误是严重错误，记录警告并返回
      WARN("UDS: Receiving data over socket failed : %d", errno);
      return ncclSystemError;
    }
    // 检查中止标志
    // 如果设置了 abortFlag 并且标志位为 1，则中止操作
    // __atomic_load_n: 原子加载函数，防止竞态条件
    // __ATOMIC_ACQUIRE: 内存顺序，确保后续操作能看到之前的写入
    if (handle->abortFlag && __atomic_load_n(handle->abortFlag, __ATOMIC_ACQUIRE)) return ncclInternalError;
  }

  // 如果调用者要求接收文件描述符（recvFd 不为 NULL）
  if (recvFd != NULL) {
    // 从辅助数据中获取第一个 cmsghdr 结构体
    // CMSG_FIRSTHDR: 宏，返回指向第一个辅助消息头的指针
    // 检查：1) 辅助消息存在；2) 辅助消息的长度正确
    if (((cmptr = CMSG_FIRSTHDR(&msg)) != NULL) && (cmptr->cmsg_len == CMSG_LEN(sizeof(int)))) {
      // 验证辅助消息的级别和类型
      // cmsg_level 必须是 SOL_SOCKET（套接字级别）
      // cmsg_type 必须是 SCM_RIGHTS（传递文件描述符）
      if ((cmptr->cmsg_level != SOL_SOCKET) || (cmptr->cmsg_type != SCM_RIGHTS)) {
        // 级别或类型不匹配，记录警告
        WARN("UDS: Receiving data over socket failed");
      // 返回系统错误码
      return ncclSystemError;
      }

      // 从辅助数据中提取文件描述符
      // CMSG_DATA: 宏，返回指向辅助数据载荷的指针
      // memmove: 内存移动函数，用于复制指定字节数
      // recvFd: 目标地址
      // CMSG_DATA(cmptr): 源地址（文件描述符在辅助数据中的位置）
      // sizeof(*recvFd): 要复制的字节数（整数的大小）
      memmove(recvFd, CMSG_DATA(cmptr), sizeof(*recvFd));
    } else {
      // 辅助消息不存在或长度不正确，记录警告
      WARN("UDS: Receiving data over socket %s failed", handle->socketName);
      // 返回系统错误码
      return ncclSystemError;
    }
    // 记录跟踪日志：成功接收到文件描述符
    // NCCL_INIT|NCCL_P2P: 日志类别，初始化或 P2P 通信
    TRACE(NCCL_INIT|NCCL_P2P, "UDS: Got recvFd %d from socket %s", *recvFd, handle->socketName);
  }

  // 返回成功状态码
  return ncclSuccess;
}

// 函数功能：仅接收文件描述符（不接收数据内容）
// 这是一个便利函数，封装了 ncclIpcSocketRecvMsg，专门用于接收文件描述符
// 参数说明：
//   - handle: 指向 ncclIpcSocket 结构体的指针
//   - recvFd: 指向整数的指针，用于接收传递过来的文件描述符
// 返回值：ncclSuccess 表示成功
ncclResult_t ncclIpcSocketRecvFd(ncclIpcSocket *handle, int *recvFd) {
  // 调用通用接收函数，传入 NULL 作为 hdr，0 作为 hdrLen
  // 这样使用虚拟缓冲区接收数据，但实际数据被丢弃
  // 只接收文件描述符
  return ncclIpcSocketRecvMsg(handle, NULL, 0, recvFd);
}

// 函数功能：通过 Unix 域套接字发送消息，可选发送文件描述符
// 参数说明：
//   - handle: 指向 ncclIpcSocket 结构体的指针
//   - hdr: 指向要发送的数据缓冲区的指针，如果为 NULL 则发送空数据
//   - hdrLen: 要发送的数据长度（字节数）
//   - sendFd: 要发送的文件描述符，如果为 -1 则不发送文件描述符
//   - rank: 目标进程的 rank 编号
//   - hash: 目标进程的哈希值
// 返回值：ncclSuccess 表示成功
ncclResult_t ncclIpcSocketSendMsg(ncclIpcSocket *handle, void *hdr, int hdrLen, const int sendFd, int rank, uint64_t hash) {
  // 声明并初始化消息头结构体
  // struct msghdr 用于 sendmsg/recvmsg 系统调用，包含消息的所有信息
  struct msghdr msg = {0, 0, 0, 0, 0, 0, 0};
  // 声明 iovec 数组，用于数据发送
  struct iovec iov[1];
  // 声明临时字符数组，用于存储生成的目标套接字名称
  char temp[NCCL_IPC_SOCKNAME_LEN];

  // 声明联合体，用于控制数据的对齐
  union {
    struct cmsghdr cm;
    char control[CMSG_SPACE(sizeof(int))];
  } control_un;

  // 声明辅助消息头指针
  struct cmsghdr *cmptr;
  // 声明虚拟缓冲区，初始化为空字符
  // 当 hdr 为 NULL 时使用
  char dummy_buffer[1] = {'\0'};
  // 声明目标套接字地址结构
  struct sockaddr_un cliaddr;

  // Construct client address to send this shareable handle to
  // 构造客户端地址，用于将可共享的句柄（文件描述符）发送给目标
  // 清空目标地址结构体的内存
  bzero(&cliaddr, sizeof(cliaddr));
  // 设置地址族为 AF_UNIX
  cliaddr.sun_family = AF_UNIX;

  // 根据目标 rank 和 hash 生成目标套接字名称
  int len = snprintf(temp, NCCL_IPC_SOCKNAME_LEN, NCCL_IPC_SOCKNAME_STR, rank, hash);
  // 检查生成的名称长度是否超过限制
  if (len > (sizeof(cliaddr.sun_path) - 1)) {
    // 记录警告：名称过长
    WARN("UDS: Cannot connect to provided name for socket. Name too large");
    return ncclInternalError;
  }
  // 将生成的名称复制到地址结构的 sun_path 字段
  // (void) 前缀告诉编译器忽略 strncpy 的返回值（未使用）
  (void) strncpy(cliaddr.sun_path, temp, len);

#ifdef USE_ABSTRACT_SOCKET
  // 使用抽象套接字技巧
  // 将 sun_path 的第一个字符设置为 '\0'（null 字节）
  cliaddr.sun_path[0] = '\0'; // Linux abstract socket trick
#endif

  // 记录跟踪日志：正在发送消息
  TRACE(NCCL_INIT, "UDS: Sending hdr %p len %d fd %d to UDS socket %s", hdr, hdrLen, sendFd, temp);

  // 如果需要发送文件描述符（sendFd != -1）
  if (sendFd != -1) {
    // 清空控制数据联合体的内存
    memset(&control_un, '\0', sizeof(control_un));
    // 设置消息的控制数据字段
    msg.msg_control = control_un.control;
    msg.msg_controllen = sizeof(control_un.control);

    // 获取第一个辅助消息头的位置
    cmptr = CMSG_FIRSTHDR(&msg);
    // 设置辅助消息的长度
    cmptr->cmsg_len = CMSG_LEN(sizeof(int));
    // 设置辅助消息的级别为 SOL_SOCKET（套接字级别）
    cmptr->cmsg_level = SOL_SOCKET;
    // 设置辅助消息的类型为 SCM_RIGHTS（发送文件描述符）
    cmptr->cmsg_type = SCM_RIGHTS;
    // 将文件描述符复制到辅助数据中
    memmove(CMSG_DATA(cmptr), &sendFd, sizeof(sendFd));
  }

  // 设置消息的目标地址
  msg.msg_name = (void *)&cliaddr;
  // 设置目标地址的长度
  msg.msg_namelen = sizeof(struct sockaddr_un);

  // 根据 hdr 参数设置 I/O 向量
  if (hdr == NULL) {
    // hdr 为 NULL，使用虚拟缓冲区
    iov[0].iov_base = (void *)dummy_buffer;
    iov[0].iov_len = sizeof(dummy_buffer);
  } else {
    // hdr 不为 NULL，使用调用者提供的缓冲区
    iov[0].iov_base = hdr;
    iov[0].iov_len = hdrLen;
  }
  // 将 I/O 向量数组关联到消息结构体
  msg.msg_iov = iov;
  // 设置 I/O 向量数组的元素个数
  msg.msg_iovlen = 1;
  // 清空消息标志位
  msg.msg_flags = 0;

  // 声明发送结果的变量
  ssize_t sendResult;
  // 循环发送消息，直到成功或发生错误
  // sendmsg(): 系统调用，发送消息到套接字
  while ((sendResult = sendmsg(handle->fd, &msg, 0)) < 0) {
    // 检查错误类型
    if (errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR) {
      // 严重错误，记录警告并返回
      WARN("UDS: Sending data over socket %s failed : %s (%d)", temp, strerror(errno), errno);
      return ncclSystemError;
    }
    // 检查中止标志
    // 如果设置了 abortFlag 并且标志位为 1，则中止操作
    if (handle->abortFlag && __atomic_load_n(handle->abortFlag, __ATOMIC_ACQUIRE))
        return ncclInternalError;
  }

  // 返回成功状态码
  return ncclSuccess;
}

// 函数功能：仅发送文件描述符（不发送数据内容）
// 这是一个便利函数，封装了 ncclIpcSocketSendMsg，专门用于发送文件描述符
// 参数说明：
//   - handle: 指向 ncclIpcSocket 结构体的指针
//   - sendFd: 要发送的文件描述符
//   - rank: 目标进程的 rank 编号
//   - hash: 目标进程的哈希值
// 返回值：ncclSuccess 表示成功
ncclResult_t ncclIpcSocketSendFd(ncclIpcSocket *handle, const int sendFd, int rank, uint64_t hash) {
  // 调用通用发送函数，传入 NULL 作为 hdr，0 作为 hdrLen
  // 这样发送空数据，只发送文件描述符
  return ncclIpcSocketSendMsg(handle, NULL, 0, sendFd, rank, hash);
}
