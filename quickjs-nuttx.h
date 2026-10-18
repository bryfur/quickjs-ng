/*
 * quickjs-nuttx.h: what quickjs-libc.c takes from a Unix that NuttX does
 * not have.  No processes to start (no fork(), no popen()), no shared
 * libraries (no dlopen()), no getrusage(): the JavaScript functions that
 * need them fail with an error.
 */
#ifndef QUICKJS_NUTTX_H
#define QUICKJS_NUTTX_H

#include <errno.h>
#include <stdio.h>
#include <stdint.h>

static inline FILE *qjs_nuttx_popen(const char *cmd, const char *mode)
{
    (void)cmd; (void)mode;
    errno = ENOSYS;
    return NULL;
}
static inline int qjs_nuttx_pclose(FILE *f)
{
    return fclose(f);
}
#define popen  qjs_nuttx_popen
#define pclose qjs_nuttx_pclose

#define RTLD_NOW     0
#define RTLD_LOCAL   0
#define RTLD_DEFAULT ((void *)0)
static inline void *dlopen(const char *f, int flags)
{
    (void)f; (void)flags;
    return NULL;
}
static inline void *dlsym(void *h, const char *s)
{
    (void)h; (void)s;
    return NULL;
}
static inline const char *dlerror(void)
{
    return "shared library modules are not supported";
}
static inline int dlclose(void *h)
{
    (void)h;
    return 0;
}

#define RUSAGE_SELF 0
struct rusage {
    struct timeval ru_utime, ru_stime;
};
static inline int getrusage(int who, struct rusage *ru)
{
    (void)who; (void)ru;
    errno = ENOSYS;
    return -1;
}

#endif
