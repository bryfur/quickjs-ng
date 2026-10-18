/*
 * QuickJS-ng: the baseline JIT's types and what JS_CallInternal() calls
 * (quickjs-jit.c).  #included by quickjs.c when QJS_JIT is defined.
 */

enum {
    JIT_STATE_NONE,
    JIT_STATE_DONE,
    JIT_STATE_FAILED,
};

enum {
    JIT_RET  = 0,       /* returned: f->ret, f->sp */
    JIT_EXC  = 1,       /* exception: f->sp, f->pc (after the instruction) */
    JIT_BAIL = 2,       /* the interpreter goes on at f->pc with f->sp */
};

typedef struct JSJitFrame {
    JSContext *ctx;             /* 0 */
    JSStackFrame *sf;           /* 4 */
    JSValue *var_buf;           /* 8 */
    JSValue *arg_buf;           /* 12 */
    JSValue *sp;                /* 16 */
    const uint8_t *pc;          /* 20 */
    JSVarRef **var_refs;        /* 24 */
    JSFunctionBytecode *b;      /* 28 */
    JSValue ret;                /* 32 */
    JSValue this_obj;           /* 40 */
    JSValue new_target;         /* 48 */
    JSValue *stack_buf;         /* 56 */
    JSContext *caller_ctx;      /* 64 */
    int argc;                   /* 68 */
    JSValue *argv;              /* 72 */
} JSJitFrame;

#define F_SP   16
#define F_PC   20
#define F_RET  32

typedef struct JSJitEntry {
    uint32_t bc;                /* bytecode offset */
    uint32_t nat;               /* code offset */
} JSJitEntry;

typedef struct JSJitCode {
    uint8_t *code;
    uint32_t size;
    struct JitIC *ics;          /* the property caches */
    uint8_t direct;             /* never hands back to the interpreter (no
                                   bail, no catch): jit_call_direct() */
    uint32_t nentries;
    JSJitEntry entries[];
} JSJitCode;

typedef int JSJitEnterFunc(JSJitFrame *f, uintptr_t entry);

static int qjs_jit_stats;               /* QJS_JIT_STATS=1: counts at exit */
static uint32_t qjs_jit_nop[256];       /* jit_op() calls an opcode */
static uint32_t qjs_jit_nbail[256];     /* bails at an opcode */
static uint32_t qjs_jit_nenter, qjs_jit_nfunc, qjs_jit_ndirect;

struct JSFunctionBytecode;
static int qjs_jit_compile(JSContext *ctx, struct JSFunctionBytecode *b);
static bool qjs_jit_want(struct JSFunctionBytecode *b, int weight);
static uintptr_t qjs_jit_entry(struct JSFunctionBytecode *b, const uint8_t *pc);
static void qjs_jit_free(JSRuntime *rt, struct JSFunctionBytecode *b);
