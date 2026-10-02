// The few POSIX differences on Windows (MinGW-w64), so the rest of the code reads the same everywhere.
// Paths are UTF-8 everywhere: on Windows the exe's manifest sets the process code page to UTF-8, so the narrow C
// runtime calls (fopen, stat, open, rename...) take UTF-8; Win32 calls go through util::widen().
#pragma once

#include <sys/stat.h>
#include <fcntl.h>
#include <cstdint>
#include <ctime>

#ifdef _WIN32
#include <io.h>
#include <process.h>
#ifndef O_CLOEXEC
#define O_CLOEXEC (O_BINARY | O_NOINHERIT)  // every open() here means binary, not inherited
#endif
#define ST_MTIME(st) ((int64_t)(st).st_mtime)
#define ST_ATIME(st) ((int64_t)(st).st_atime)
inline struct tm* localtime_r(const time_t* t, struct tm* out) { return localtime_s(out, t) == 0 ? out : nullptr; }
#else
#include <unistd.h>
#define ST_MTIME(st) ((int64_t)(st).st_mtim.tv_sec)
#define ST_ATIME(st) ((int64_t)(st).st_atim.tv_sec)
#endif
