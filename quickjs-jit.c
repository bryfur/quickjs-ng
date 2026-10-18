/*
 * QuickJS-ng: a baseline JIT, bytecode to Thumb-2 (ARMv7-M, ARMv8-M
 * Mainline: the FPC's RP2350).  #included at the end of quickjs.c when
 * QJS_JIT is defined: it uses the engine's internals.
 *
 * The compiled code works on the interpreter's own frame: the arguments,
 * the local variables and the operand stack stay where JS_CallInternal()
 * put them, in memory, between any two instructions.  So:
 *
 *  - an instruction the JIT does not compile hands the frame back to the
 *    interpreter at that instruction (JIT_BAIL), which goes on from there;
 *  - the interpreter enters the compiled code at the start of the
 *    function and at a backward jump's target (a loop's head), at the
 *    instruction it has reached (qjs_jit_enter());
 *  - an exception leaves the compiled code (JIT_EXC) with the frame as the
 *    interpreter would have it, and the interpreter's own exception path
 *    unwinds it (catch, finally).
 *
 * Simple instructions are compiled inline (constants, locals, arguments,
 * closure variables, stack shuffles, int arithmetic and comparisons fused
 * with the jump that follows, jumps, return); everything else, and the
 * slow paths of the inline ones, calls jit_op(), which runs that one
 * instruction in C as the interpreter would.
 *
 * Registers: r4 the operand stack pointer (JSValue *), r5 var_buf, r6
 * arg_buf, r7 ctx, r8 the JSJitFrame, r10 the closure's var_refs; r0-r3,
 * ip and lr scratch.  JSValue is 64-bit (NaN boxing): the payload at +0,
 * the tag at +4.
 *
 * QJS_JIT environment variable: 0 off, N compile a function at its Nth
 * call (or after N*16 backward jumps in one), unset 8.  QJS_JIT_LOG=1:
 * a line a compiled function on stderr.
 */

#include "jit_thumb.h"



static int qjs_jit_threshold = -1;      /* -1: not read yet; 0: off */
static int qjs_jit_log;

static const char *const jit_op_names[] = {
#define FMT(f)
#define DEF(id, size, n_pop, n_push, f) #id,
#define def(id, size, n_pop, n_push, f)
#include "quickjs-opcode.h"
#undef DEF
#undef def
#undef FMT
};

static void qjs_jit_print_stats(void)
{
    int i;
    fprintf(stderr, "qjs jit: %u functions compiled, %u entries\n",
            qjs_jit_nfunc, qjs_jit_nenter);
    for (i = 0; i < 256; i++) {
        if (qjs_jit_nop[i] || qjs_jit_nbail[i])
            fprintf(stderr, "qjs jit: %-22s jit_op %9u  bail %9u\n",
                    i < OP_COUNT ? jit_op_names[i] : "?",
                    qjs_jit_nop[i], qjs_jit_nbail[i]);
    }
}

static void qjs_jit_config(void)
{
    const char *s = getenv("QJS_JIT");
    qjs_jit_threshold = s ? atoi(s) : 8;
    s = getenv("QJS_JIT_LOG");
    qjs_jit_log = s && *s == '1';
    s = getenv("QJS_JIT_STATS");
    qjs_jit_stats = s && *s == '1';
    if (qjs_jit_stats)
        atexit(qjs_jit_print_stats);
}

/* ---------------------------------------------------------------------
 * Executable memory
 * --------------------------------------------------------------------- */

#if defined(__NuttX__)
void *up_textheap_memalign(const char *sectname, size_t align, size_t size);
void up_textheap_free(void *p);

static void *jit_code_alloc(size_t size)
{
    /* the SRAM if the loader's allocator has room for it, else the PSRAM */
    return up_textheap_memalign(".time_critical.qjsjit", 8, size);
}

static void jit_code_free(void *p, size_t size)
{
    (void)size;
    up_textheap_free(p);
}

static void jit_code_sync(void *p, size_t size)
{
    (void)p; (void)size;
    __asm__ __volatile__("dsb\n\tisb" ::: "memory");
}
#else
#include <sys/mman.h>

static void *jit_code_alloc(size_t size)
{
    void *p = mmap(NULL, size, PROT_READ | PROT_WRITE | PROT_EXEC,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    return p == MAP_FAILED ? NULL : p;
}

static void jit_code_free(void *p, size_t size)
{
    munmap(p, size);
}

static void jit_code_sync(void *p, size_t size)
{
    __builtin___clear_cache((char *)p, (char *)p + size);
}
#endif

void *jit_realloc(void *p, uint32_t size)
{
    if (size == 0) {
        free(p);
        return NULL;
    }
    return realloc(p, size);
}

static void qjs_jit_free(JSRuntime *rt, JSFunctionBytecode *b)
{
    JSJitCode *jc = b->jit;
    if (jc) {
        jit_code_free(jc->code, jc->size);
        js_free_rt(rt, jc->ics);
        js_free_rt(rt, jc);
        b->jit = NULL;
    }
}

/* ---------------------------------------------------------------------
 * What the compiled code calls
 * --------------------------------------------------------------------- */

static void jit_free_value(JSContext *ctx, JSValue v)
{
    js_free_value_rt(ctx->rt, v);
}

/* JS_ToBoolFree() of a popped value (cannot throw) */
static int jit_to_bool(JSContext *ctx, JSValue v)
{
    return JS_ToBoolFree(ctx, v);
}

static int jit_poll(JSContext *ctx)
{
    return __js_poll_interrupts(ctx);
}

/* Whether jit_op() runs this opcode */
static bool jit_op_has(int op)
{
    switch (op) {
    case OP_push_const: case OP_push_const8: case OP_fclosure:
    case OP_fclosure8: case OP_push_atom_value: case OP_push_empty_string:
    case OP_push_bigint_i32:
    case OP_object: case OP_push_this: case OP_special_object:
    case OP_nip1: case OP_dup1: case OP_dup2: case OP_dup3:
    case OP_insert2: case OP_insert3: case OP_insert4:
    case OP_perm3: case OP_perm4: case OP_perm5: case OP_swap2:
    case OP_rot3l: case OP_rot3r: case OP_rot4l: case OP_rot5l:
    case OP_get_length: case OP_get_field: case OP_get_field2:
    case OP_put_field: case OP_get_array_el: case OP_get_array_el2:
    case OP_put_array_el:
    case OP_get_var: case OP_get_var_undef: case OP_put_var:
    case OP_put_var_init:
    case OP_define_field: case OP_set_name: case OP_array_from:
    case OP_call: case OP_call_method: case OP_call_constructor:
    case OP_call0: case OP_call1: case OP_call2: case OP_call3:
    case OP_tail_call: case OP_tail_call_method:
    case OP_apply:
    case OP_get_loc_check: case OP_put_loc_check: case OP_put_loc_check_init:
    case OP_get_var_ref_check: case OP_put_var_ref_check:
    case OP_put_var_ref_check_init: case OP_set_loc_uninitialized:
    case OP_close_loc:
    case OP_add: case OP_sub: case OP_mul: case OP_div: case OP_mod:
    case OP_pow: case OP_add_loc:
    case OP_neg: case OP_plus: case OP_inc: case OP_dec:
    case OP_post_inc: case OP_post_dec: case OP_inc_loc: case OP_dec_loc:
    case OP_not: case OP_lnot:
    case OP_shl: case OP_sar: case OP_shr: case OP_and: case OP_or:
    case OP_xor:
    case OP_lt: case OP_lte: case OP_gt: case OP_gte: case OP_eq:
    case OP_neq: case OP_strict_eq: case OP_strict_neq:
    case OP_in: case OP_instanceof: case OP_typeof:
    case OP_to_propkey: case OP_to_propkey2:
    case OP_is_undefined_or_null: case OP_is_undefined: case OP_is_null:
    case OP_typeof_is_undefined: case OP_typeof_is_function:
    case OP_catch: case OP_nip_catch: case OP_throw:
        return true;
    default:
        return false;
    }
}

/* One instruction (op_pc) run as the interpreter runs it, on the frame:
   the operand stack's new top, or NULL after an exception (f->sp and
   f->pc set) */
static JSValue *jit_op(JSJitFrame *f, JSValue *sp, const uint8_t *op_pc)
{
    JSContext *ctx = f->ctx;
    JSContext *caller_ctx = f->caller_ctx;
    JSRuntime *rt = ctx->rt;
    JSStackFrame *sf = f->sf;
    JSFunctionBytecode *b = f->b;
    JSValue *var_buf = f->var_buf;
    JSValue *arg_buf = f->arg_buf;
    JSVarRef **var_refs = f->var_refs;
    JSValue *stack_buf = f->stack_buf;
    const uint8_t *pc = op_pc + 1;
    int opcode = *op_pc;
    int call_argc, i;
    JSValue *call_argv, ret_val;
    JSValueConst this_obj = f->this_obj;
    JSValueConst new_target = f->new_target;
    int argc = f->argc;
    JSValue *argv = f->argv;
    JSObject *p = JS_VALUE_GET_OBJ(sf->cur_func);

    (void)arg_buf;
    sf->cur_pc = (uint8_t *)op_pc + short_opcode_info(opcode).size;
    if (unlikely(qjs_jit_stats))
        qjs_jit_nop[opcode]++;

    switch (opcode) {
    case OP_push_const:
        *sp++ = js_dup(b->cpool[get_u32(pc)]);
        break;
    case OP_push_const8:
        *sp++ = js_dup(b->cpool[*pc]);
        break;
    case OP_push_bigint_i32:
        *sp++ = __JS_NewShortBigInt(ctx, (int)get_u32(pc));
        break;
    case OP_fclosure:
    case OP_fclosure8:
        {
            int idx = opcode == OP_fclosure ? (int)get_u32(pc) : *pc;
            *sp++ = js_closure(ctx, js_dup(b->cpool[idx]), var_refs, sf);
            if (unlikely(JS_IsException(sp[-1])))
                goto exception;
        }
        break;
    case OP_push_atom_value:
        *sp++ = JS_AtomToValue(ctx, get_u32(pc));
        break;
    case OP_push_empty_string:
        *sp++ = js_empty_string(rt);
        break;
    case OP_object:
        *sp++ = JS_NewObject(ctx);
        if (unlikely(JS_IsException(sp[-1])))
            goto exception;
        break;
    case OP_push_this:
        {
            JSValue val;
            if (!b->is_strict_mode) {
                uint32_t tag = JS_VALUE_GET_TAG(this_obj);
                if (likely(tag == JS_TAG_OBJECT))
                    goto normal_this;
                if (tag == JS_TAG_NULL || tag == JS_TAG_UNDEFINED) {
                    val = js_dup(ctx->global_obj);
                } else {
                    val = JS_ToObject(ctx, this_obj);
                    if (JS_IsException(val))
                        goto exception;
                }
            } else {
            normal_this:
                val = js_dup(this_obj);
            }
            *sp++ = val;
        }
        break;
    case OP_special_object:
        switch (*pc) {
        case OP_SPECIAL_OBJECT_ARGUMENTS:
            *sp++ = js_build_arguments(ctx, argc, vc(argv));
            if (unlikely(JS_IsException(sp[-1])))
                goto exception;
            break;
        case OP_SPECIAL_OBJECT_MAPPED_ARGUMENTS:
            *sp++ = js_build_mapped_arguments(ctx, argc, vc(argv),
                                              sf, min_int(argc, b->arg_count));
            if (unlikely(JS_IsException(sp[-1])))
                goto exception;
            break;
        case OP_SPECIAL_OBJECT_THIS_FUNC:
            *sp++ = js_dup(sf->cur_func);
            break;
        case OP_SPECIAL_OBJECT_NEW_TARGET:
            *sp++ = js_dup(new_target);
            break;
        case OP_SPECIAL_OBJECT_HOME_OBJECT:
            {
                JSObject *p1 = p->u.func.home_object;
                if (unlikely(!p1))
                    *sp++ = JS_UNDEFINED;
                else
                    *sp++ = js_dup(JS_MKPTR(JS_TAG_OBJECT, p1));
            }
            break;
        case OP_SPECIAL_OBJECT_VAR_OBJECT:
            *sp++ = JS_NewObjectProto(ctx, JS_NULL);
            if (unlikely(JS_IsException(sp[-1])))
                goto exception;
            break;
        case OP_SPECIAL_OBJECT_IMPORT_META:
            *sp++ = js_import_meta(ctx);
            if (unlikely(JS_IsException(sp[-1])))
                goto exception;
            break;
        case OP_SPECIAL_OBJECT_NULL_PROTO:
            *sp++ = JS_NewObjectProtoClass(ctx, JS_NULL, JS_CLASS_OBJECT);
            if (unlikely(JS_IsException(sp[-1])))
                goto exception;
            break;
        default:
            abort();
        }
        break;

    /* the stack */
    case OP_nip1: /* a b c -> b c */
        JS_FreeValue(ctx, sp[-3]);
        sp[-3] = sp[-2];
        sp[-2] = sp[-1];
        sp--;
        break;
    case OP_dup1: /* a b -> a a b */
        sp[0] = sp[-1];
        sp[-1] = js_dup(sp[-2]);
        sp++;
        break;
    case OP_dup2: /* a b -> a b a b */
        sp[0] = js_dup(sp[-2]);
        sp[1] = js_dup(sp[-1]);
        sp += 2;
        break;
    case OP_dup3: /* a b c -> a b c a b c */
        sp[0] = js_dup(sp[-3]);
        sp[1] = js_dup(sp[-2]);
        sp[2] = js_dup(sp[-1]);
        sp += 3;
        break;
    case OP_insert2: /* obj a -> a obj a (dup_x1) */
        sp[0] = sp[-1];
        sp[-1] = sp[-2];
        sp[-2] = js_dup(sp[0]);
        sp++;
        break;
    case OP_insert3: /* obj prop a -> a obj prop a (dup_x2) */
        sp[0] = sp[-1];
        sp[-1] = sp[-2];
        sp[-2] = sp[-3];
        sp[-3] = js_dup(sp[0]);
        sp++;
        break;
    case OP_insert4: /* this obj prop a -> a this obj prop a */
        sp[0] = sp[-1];
        sp[-1] = sp[-2];
        sp[-2] = sp[-3];
        sp[-3] = sp[-4];
        sp[-4] = js_dup(sp[0]);
        sp++;
        break;
    case OP_perm3: /* obj a b -> a obj b (213) */
        {
            JSValue tmp = sp[-2];
            sp[-2] = sp[-3];
            sp[-3] = tmp;
        }
        break;
    case OP_rot3l: /* x a b -> a b x (231) */
        {
            JSValue tmp = sp[-3];
            sp[-3] = sp[-2];
            sp[-2] = sp[-1];
            sp[-1] = tmp;
        }
        break;
    case OP_rot4l: /* x a b c -> a b c x */
        {
            JSValue tmp = sp[-4];
            sp[-4] = sp[-3];
            sp[-3] = sp[-2];
            sp[-2] = sp[-1];
            sp[-1] = tmp;
        }
        break;
    case OP_rot5l: /* x a b c d -> a b c d x */
        {
            JSValue tmp = sp[-5];
            sp[-5] = sp[-4];
            sp[-4] = sp[-3];
            sp[-3] = sp[-2];
            sp[-2] = sp[-1];
            sp[-1] = tmp;
        }
        break;
    case OP_rot3r: /* a b x -> x a b (312) */
        {
            JSValue tmp = sp[-1];
            sp[-1] = sp[-2];
            sp[-2] = sp[-3];
            sp[-3] = tmp;
        }
        break;
    case OP_perm4: /* obj prop a b -> a obj prop b */
        {
            JSValue tmp = sp[-2];
            sp[-2] = sp[-3];
            sp[-3] = sp[-4];
            sp[-4] = tmp;
        }
        break;
    case OP_perm5: /* this obj prop a b -> a this obj prop b */
        {
            JSValue tmp = sp[-2];
            sp[-2] = sp[-3];
            sp[-3] = sp[-4];
            sp[-4] = sp[-5];
            sp[-5] = tmp;
        }
        break;
    case OP_swap2: /* a b c d -> c d a b */
        {
            JSValue tmp1 = sp[-4], tmp2 = sp[-3];
            sp[-4] = sp[-2];
            sp[-3] = sp[-1];
            sp[-2] = tmp1;
            sp[-1] = tmp2;
        }
        break;

    /* properties */
    case OP_get_length:
    case OP_get_field:
    case OP_get_field2:
        {
            JSValue val, obj;
            JSAtom atom = opcode == OP_get_length ? JS_ATOM_length : get_u32(pc);
            JSObject *p1;
            JSProperty *pr;
            JSShapeProperty *prs;

            obj = sp[-1];
            if (likely(JS_VALUE_GET_TAG(obj) == JS_TAG_OBJECT)) {
                p1 = JS_VALUE_GET_OBJ(obj);
                for(;;) {
                    prs = find_own_property(&pr, p1, atom);
                    if (prs) {
                        if (unlikely(prs->flags & JS_PROP_TMASK))
                            goto get_field_slow_path;
                        val = js_dup(pr->u.value);
                        break;
                    }
                    if (unlikely(p1->is_exotic)) {
                        obj = JS_MKPTR(JS_TAG_OBJECT, p1);
                        goto get_field_slow_path;
                    }
                    p1 = p1->shape->proto;
                    if (!p1) {
                        val = JS_UNDEFINED;
                        break;
                    }
                }
            } else {
            get_field_slow_path:
                val = JS_GetPropertyInternal(ctx, obj, atom, sp[-1], false);
                if (unlikely(JS_IsException(val)))
                    goto exception;
            }
            if (opcode == OP_get_field2) {
                *sp++ = val;
            } else {
                JS_FreeValue(ctx, sp[-1]);
                sp[-1] = val;
            }
        }
        break;
    case OP_put_field:
        {
            int ret;
            JSValue obj;
            JSAtom atom = get_u32(pc);
            JSObject *p1;
            JSProperty *pr;
            JSShapeProperty *prs;

            obj = sp[-2];
            if (likely(JS_VALUE_GET_TAG(obj) == JS_TAG_OBJECT)) {
                p1 = JS_VALUE_GET_OBJ(obj);
                prs = find_own_property(&pr, p1, atom);
                if (!prs)
                    goto put_field_slow_path;
                if (likely((prs->flags & (JS_PROP_TMASK | JS_PROP_WRITABLE |
                                          JS_PROP_LENGTH)) == JS_PROP_WRITABLE)) {
                    set_value(ctx, &pr->u.value, sp[-1]);
                } else {
                    goto put_field_slow_path;
                }
                JS_FreeValue(ctx, obj);
                sp -= 2;
            } else {
            put_field_slow_path:
                ret = JS_SetPropertyInternal2(ctx, obj, atom, sp[-1], obj,
                                              JS_PROP_THROW_STRICT);
                JS_FreeValue(ctx, obj);
                sp -= 2;
                if (unlikely(ret < 0))
                    goto exception;
            }
        }
        break;
    case OP_get_array_el:
    case OP_get_array_el2:
        {
            JSValue val;
            if (likely(JS_VALUE_GET_TAG(sp[-2]) == JS_TAG_OBJECT &&
                       JS_VALUE_GET_TAG(sp[-1]) == JS_TAG_INT)) {
                JSObject *p1 = JS_VALUE_GET_OBJ(sp[-2]);
                uint32_t idx = JS_VALUE_GET_INT(sp[-1]);
                if (likely(p1->class_id == JS_CLASS_ARRAY &&
                           idx < p1->u.array.count)) {
                    val = js_dup(p1->u.array.u.values[idx]);
                    goto got_el;
                }
                if (js_get_fast_array_element(ctx, p1, idx, &val))
                    goto got_el;
            }
            if (opcode == OP_get_array_el2) {
                val = JS_GetPropertyValue(ctx, sp[-2], sp[-1]);
                sp[-1] = val;
                if (unlikely(JS_IsException(val)))
                    goto exception;
                break;
            }
            val = JS_GetPropertyValue(ctx, sp[-2], sp[-1]);
            JS_FreeValue(ctx, sp[-2]);
            sp[-2] = val;
            sp--;
            if (unlikely(JS_IsException(val)))
                goto exception;
            break;
        got_el:
            if (opcode == OP_get_array_el2) {
                sp[-1] = val;
            } else {
                JS_FreeValue(ctx, sp[-2]);
                sp[-2] = val;
                sp--;
            }
        }
        break;
    case OP_put_array_el:
        {
            int ret;
            JSValue val = sp[-1];
            uint32_t idx;
            JSObject *p1;

            if (likely(JS_VALUE_GET_TAG(sp[-2]) == JS_TAG_INT)) {
                idx = JS_VALUE_GET_INT(sp[-2]);
                if (likely(JS_VALUE_GET_TAG(sp[-3]) == JS_TAG_OBJECT)) {
                    p1 = JS_VALUE_GET_OBJ(sp[-3]);
                    if (likely(p1->class_id == JS_CLASS_ARRAY &&
                               idx < (uint32_t)p1->u.array.count)) {
                        set_value(ctx, &p1->u.array.u.values[idx], val);
                        JS_FreeValue(ctx, sp[-3]);
                        sp -= 3;
                        break;
                    }
                    if (likely(p1->class_id == JS_CLASS_ARRAY &&
                               idx == (uint32_t)p1->u.array.count &&
                               p1->fast_array &&
                               p1->extensible &&
                               p1->shape->proto == JS_VALUE_GET_OBJ(ctx->class_proto[JS_CLASS_ARRAY]) &&
                               ctx->std_array_prototype)) {
                        uint32_t array_len;
                        if (likely(JS_VALUE_GET_TAG(p1->prop[0].u.value) == JS_TAG_INT)) {
                            uint32_t new_len = idx + 1;
                            array_len = JS_VALUE_GET_INT(p1->prop[0].u.value);
                            if (likely(new_len <= p1->u.array.u1.size)) {
                                p1->u.array.u.values[idx] = val;
                                p1->u.array.count = new_len;
                                if (new_len > array_len)
                                    p1->prop[0].u.value = js_int32(new_len);
                                JS_FreeValue(ctx, sp[-3]);
                                sp -= 3;
                                break;
                            }
                        }
                    }
                }
            }
            ret = JS_SetPropertyValue(ctx, sp[-3], sp[-2], sp[-1], JS_PROP_THROW_STRICT);
            JS_FreeValue(ctx, sp[-3]);
            sp -= 3;
            if (unlikely(ret < 0))
                goto exception;
        }
        break;
    case OP_get_var:
    case OP_get_var_undef:
        {
            JSValue val = JS_GetGlobalVar(ctx, get_u32(pc), opcode - OP_get_var_undef);
            if (unlikely(JS_IsException(val)))
                goto exception;
            *sp++ = val;
        }
        break;
    case OP_put_var:
    case OP_put_var_init:
        {
            int ret = JS_SetGlobalVar(ctx, get_u32(pc), sp[-1], opcode - OP_put_var);
            sp--;
            if (unlikely(ret < 0))
                goto exception;
        }
        break;
    case OP_define_field:
        {
            int ret = JS_DefinePropertyValue(ctx, sp[-2], get_u32(pc), sp[-1],
                                             JS_PROP_C_W_E | JS_PROP_THROW);
            sp--;
            if (unlikely(ret < 0))
                goto exception;
        }
        break;
    case OP_set_name:
        if (unlikely(JS_DefineObjectName(ctx, sp[-1], get_u32(pc),
                                         JS_PROP_CONFIGURABLE) < 0))
            goto exception;
        break;
    case OP_array_from:
        call_argc = get_u16(pc);
        call_argv = sp - call_argc;
        ret_val = JS_NewArrayFrom(ctx, call_argc, call_argv);
        sp -= call_argc;
        if (unlikely(JS_IsException(ret_val)))
            goto exception;
        *sp++ = ret_val;
        break;

    /* calls */
    case OP_call0: case OP_call1: case OP_call2: case OP_call3:
    case OP_call:
    case OP_tail_call:    /* (the compiled code returns the result) */
        call_argc = opcode == OP_call || opcode == OP_tail_call ?
                    get_u16(pc) : opcode - OP_call0;
        call_argv = sp - call_argc;
        ret_val = JS_CallInternal(ctx, call_argv[-1], JS_UNDEFINED,
                                  JS_UNDEFINED, call_argc, vc(call_argv), 0);
        if (unlikely(JS_IsException(ret_val)))
            goto exception;
        for(i = -1; i < call_argc; i++)
            JS_FreeValue(ctx, call_argv[i]);
        sp -= call_argc + 1;
        *sp++ = ret_val;
        break;
    case OP_call_method:
    case OP_tail_call_method:
        call_argc = get_u16(pc);
        call_argv = sp - call_argc;
        ret_val = JS_CallInternal(ctx, call_argv[-1], call_argv[-2],
                                  JS_UNDEFINED, call_argc, vc(call_argv), 0);
        if (unlikely(JS_IsException(ret_val)))
            goto exception;
        for(i = -2; i < call_argc; i++)
            JS_FreeValue(ctx, call_argv[i]);
        sp -= call_argc + 2;
        *sp++ = ret_val;
        break;
    case OP_call_constructor:
        call_argc = get_u16(pc);
        call_argv = sp - call_argc;
        ret_val = JS_CallConstructorInternal(ctx, call_argv[-2], call_argv[-1],
                                             call_argc, vc(call_argv), 0);
        if (unlikely(JS_IsException(ret_val)))
            goto exception;
        for(i = -2; i < call_argc; i++)
            JS_FreeValue(ctx, call_argv[i]);
        sp -= call_argc + 2;
        *sp++ = ret_val;
        break;
    case OP_apply:
        {
            int magic = get_u16(pc);
            ret_val = js_function_apply(ctx, sp[-3], 2, vc(&sp[-2]), magic);
            if (unlikely(JS_IsException(ret_val)))
                goto exception;
            JS_FreeValue(ctx, sp[-3]);
            JS_FreeValue(ctx, sp[-2]);
            JS_FreeValue(ctx, sp[-1]);
            sp -= 3;
            *sp++ = ret_val;
        }
        break;

    /* checked variables */
    case OP_get_loc_check:
        {
            int idx = get_u16(pc);
            if (unlikely(JS_IsUninitialized(var_buf[idx]))) {
                JS_ThrowReferenceErrorUninitialized2(caller_ctx, b, idx, false);
                goto exception;
            }
            *sp++ = js_dup(var_buf[idx]);
        }
        break;
    case OP_put_loc_check:
        {
            int idx = get_u16(pc);
            if (unlikely(JS_IsUninitialized(var_buf[idx]))) {
                JS_ThrowReferenceErrorUninitialized2(caller_ctx, b, idx, false);
                goto exception;
            }
            set_value(ctx, &var_buf[idx], sp[-1]);
            sp--;
        }
        break;
    case OP_put_loc_check_init:
        {
            int idx = get_u16(pc);
            if (unlikely(!JS_IsUninitialized(var_buf[idx]))) {
                JS_ThrowReferenceError(caller_ctx, "'this' can be initialized only once");
                goto exception;
            }
            set_value(ctx, &var_buf[idx], sp[-1]);
            sp--;
        }
        break;
    case OP_get_var_ref_check:
        {
            int idx = get_u16(pc);
            JSValue val = *var_refs[idx]->pvalue;
            if (unlikely(JS_IsUninitialized(val))) {
                JS_ThrowReferenceErrorUninitialized2(ctx, b, idx, true);
                goto exception;
            }
            *sp++ = js_dup(val);
        }
        break;
    case OP_put_var_ref_check:
        {
            int idx = get_u16(pc);
            if (unlikely(JS_IsUninitialized(*var_refs[idx]->pvalue))) {
                JS_ThrowReferenceErrorUninitialized2(ctx, b, idx, true);
                goto exception;
            }
            set_value(ctx, var_refs[idx]->pvalue, sp[-1]);
            sp--;
        }
        break;
    case OP_put_var_ref_check_init:
        {
            int idx = get_u16(pc);
            if (unlikely(!JS_IsUninitialized(*var_refs[idx]->pvalue))) {
                JS_ThrowReferenceErrorUninitialized2(ctx, b, idx, true);
                goto exception;
            }
            set_value(ctx, var_refs[idx]->pvalue, sp[-1]);
            sp--;
        }
        break;
    case OP_set_loc_uninitialized:
        set_value(ctx, &var_buf[get_u16(pc)], JS_UNINITIALIZED);
        break;
    case OP_close_loc:
        close_lexical_var(ctx, b, sf, get_u16(pc));
        break;

    /* arithmetic, as the interpreter has it (the inline int paths failed) */
    case OP_add:
        {
            JSValue op1 = sp[-2], op2 = sp[-1];
            if (likely(JS_VALUE_IS_BOTH_INT(op1, op2))) {
                int64_t r = (int64_t)JS_VALUE_GET_INT(op1) + JS_VALUE_GET_INT(op2);
                if (unlikely(r < INT32_MIN || r > INT32_MAX))
                    sp[-2] = js_float64(r);
                else
                    sp[-2] = js_int32(r);
                sp--;
            } else if (JS_VALUE_IS_BOTH_FLOAT(op1, op2)) {
                sp[-2] = js_float64(JS_VALUE_GET_FLOAT64(op1) +
                                    JS_VALUE_GET_FLOAT64(op2));
                sp--;
            } else if (JS_TAG_IS_FLOAT64(JS_VALUE_GET_TAG(op1)) ||
                       JS_TAG_IS_FLOAT64(JS_VALUE_GET_TAG(op2))) {
                double d1, d2;
                if (!js_arith_to_float64(op1, &d1) ||
                    !js_arith_to_float64(op2, &d2))
                    goto add_slow_case;
                sp[-2] = js_float64(d1 + d2);
                sp--;
            } else {
            add_slow_case:
                if (js_add_slow(ctx, sp))
                    goto exception;
                sp--;
            }
        }
        break;
    case OP_sub:
    case OP_mul:
    case OP_div:
        {
            JSValue op1 = sp[-2], op2 = sp[-1];
            double d1, d2, d;
            if (likely(JS_VALUE_IS_BOTH_INT(op1, op2))) {
                int32_t v1 = JS_VALUE_GET_INT(op1), v2 = JS_VALUE_GET_INT(op2);
                if (opcode == OP_sub) {
                    int64_t r = (int64_t)v1 - v2;
                    sp[-2] = (int)r != r ? __JS_NewFloat64((double)r) : js_int32(r);
                } else if (opcode == OP_mul) {
                    int64_t r = (int64_t)v1 * v2;
                    if (unlikely((int)r != r))
                        sp[-2] = js_float64((double)r);
                    else if (unlikely(r == 0 && (v1 | v2) < 0))
                        sp[-2] = js_float64(-0.0);
                    else
                        sp[-2] = js_int32(r);
                } else {
                    sp[-2] = js_number((double)v1 / (double)v2);
                }
                sp--;
                break;
            }
            if (JS_VALUE_IS_BOTH_FLOAT(op1, op2)) {
                d1 = JS_VALUE_GET_FLOAT64(op1);
                d2 = JS_VALUE_GET_FLOAT64(op2);
            } else if (JS_TAG_IS_FLOAT64(JS_VALUE_GET_TAG(op1)) ||
                       JS_TAG_IS_FLOAT64(JS_VALUE_GET_TAG(op2))) {
                if (!js_arith_to_float64(op1, &d1) ||
                    !js_arith_to_float64(op2, &d2))
                    goto binary_arith_slow;
            } else {
                goto binary_arith_slow;
            }
            if (opcode == OP_sub) {
                d = d1 - d2;
                sp[-2] = js_float64(d);
            } else if (opcode == OP_mul) {
                d = d1 * d2;
                sp[-2] = js_float64(d);
            } else {
                sp[-2] = js_number(d1 / d2);
            }
            sp--;
        }
        break;
    case OP_mod:
        {
            JSValue op1 = sp[-2], op2 = sp[-1];
            if (likely(JS_VALUE_IS_BOTH_INT(op1, op2))) {
                int v1 = JS_VALUE_GET_INT(op1), v2 = JS_VALUE_GET_INT(op2);
                if (unlikely(v1 < 0 || v2 <= 0))
                    goto binary_arith_slow;
                sp[-2] = js_int32(v1 % v2);
                sp--;
                break;
            }
        }
        /* fall through */
    case OP_pow:
    binary_arith_slow:
        if (js_binary_arith_slow(ctx, sp, opcode))
            goto exception;
        sp--;
        break;
    case OP_add_loc:
        {
            JSValue *pv = &var_buf[*pc];
            if (likely(JS_VALUE_IS_BOTH_INT(*pv, sp[-1]))) {
                int64_t r = (int64_t)JS_VALUE_GET_INT(*pv) + JS_VALUE_GET_INT(sp[-1]);
                if (unlikely((int)r != r))
                    *pv = __JS_NewFloat64((double)r);
                else
                    *pv = js_int32(r);
                sp--;
            } else if (JS_VALUE_GET_TAG(*pv) == JS_TAG_STRING) {
                JSValue op1 = sp[-1];
                sp--;
                op1 = JS_ToPrimitiveFree(ctx, op1, HINT_NONE);
                if (JS_IsException(op1))
                    goto exception;
                op1 = JS_ConcatString(ctx, js_dup(*pv), op1);
                if (JS_IsException(op1))
                    goto exception;
                set_value(ctx, pv, op1);
            } else {
                JSValue ops[2];
                ops[0] = js_dup(*pv);
                ops[1] = sp[-1];
                sp--;
                if (js_add_slow(ctx, ops + 2))
                    goto exception;
                set_value(ctx, pv, ops[0]);
            }
        }
        break;
    case OP_plus:
        {
            uint32_t tag = JS_VALUE_GET_TAG(sp[-1]);
            if (!(tag == JS_TAG_INT || JS_TAG_IS_FLOAT64(tag))) {
                if (js_unary_arith_slow(ctx, sp, opcode))
                    goto exception;
            }
        }
        break;
    case OP_neg:
        {
            JSValue op1 = sp[-1];
            uint32_t tag = JS_VALUE_GET_TAG(op1);
            if (tag == JS_TAG_INT) {
                int val = JS_VALUE_GET_INT(op1);
                if (unlikely(val == 0))
                    sp[-1] = js_float64(-0.0);
                else if (unlikely(val == INT32_MIN))
                    sp[-1] = js_float64(-(double)val);
                else
                    sp[-1] = js_int32(-val);
            } else if (JS_TAG_IS_FLOAT64(tag)) {
                sp[-1] = js_float64(-JS_VALUE_GET_FLOAT64(op1));
            } else {
                if (js_unary_arith_slow(ctx, sp, opcode))
                    goto exception;
            }
        }
        break;
    case OP_inc:
    case OP_dec:
        {
            JSValue op1 = sp[-1];
            if (JS_VALUE_GET_TAG(op1) == JS_TAG_INT &&
                JS_VALUE_GET_INT(op1) != (opcode == OP_inc ? INT32_MAX : INT32_MIN)) {
                sp[-1] = js_int32(JS_VALUE_GET_INT(op1) + (opcode == OP_inc ? 1 : -1));
            } else {
                if (js_unary_arith_slow(ctx, sp, opcode))
                    goto exception;
            }
        }
        break;
    case OP_post_inc:
    case OP_post_dec:
        {
            JSValue op1 = sp[-1];
            if (JS_VALUE_GET_TAG(op1) == JS_TAG_INT &&
                JS_VALUE_GET_INT(op1) != (opcode == OP_post_inc ? INT32_MAX : INT32_MIN)) {
                sp[0] = js_int32(JS_VALUE_GET_INT(op1) + (opcode == OP_post_inc ? 1 : -1));
            } else {
                if (js_post_inc_slow(ctx, sp, opcode))
                    goto exception;
            }
            sp++;
        }
        break;
    case OP_inc_loc:
    case OP_dec_loc:
        {
            int idx = *pc;
            JSValue op1 = var_buf[idx];
            int inc = opcode == OP_inc_loc;
            if (JS_VALUE_GET_TAG(op1) == JS_TAG_INT &&
                JS_VALUE_GET_INT(op1) != (inc ? INT32_MAX : INT32_MIN)) {
                var_buf[idx] = js_int32(JS_VALUE_GET_INT(op1) + (inc ? 1 : -1));
            } else {
                op1 = js_dup(op1);
                if (js_unary_arith_slow(ctx, &op1 + 1, inc ? OP_inc : OP_dec))
                    goto exception;
                set_value(ctx, &var_buf[idx], op1);
            }
        }
        break;
    case OP_not:
        if (JS_VALUE_GET_TAG(sp[-1]) == JS_TAG_INT) {
            sp[-1] = js_int32(~JS_VALUE_GET_INT(sp[-1]));
        } else {
            if (js_not_slow(ctx, sp))
                goto exception;
        }
        break;
    case OP_lnot:
        {
            int res;
            JSValue op1 = sp[-1];
            if ((uint32_t)JS_VALUE_GET_TAG(op1) <= JS_TAG_UNDEFINED)
                res = JS_VALUE_GET_INT(op1) != 0;
            else
                res = JS_ToBoolFree(ctx, op1);
            sp[-1] = js_bool(!res);
        }
        break;
    case OP_shl:
    case OP_sar:
    case OP_and:
    case OP_or:
    case OP_xor:
        {
            JSValue op1 = sp[-2], op2 = sp[-1];
            if (likely(JS_VALUE_IS_BOTH_INT(op1, op2))) {
                int32_t v1 = JS_VALUE_GET_INT(op1), v2 = JS_VALUE_GET_INT(op2), r;
                switch (opcode) {
                case OP_shl: r = (int32_t)((uint32_t)v1 << (v2 & 0x1f)); break;
                case OP_sar: r = v1 >> (v2 & 0x1f); break;
                case OP_and: r = v1 & v2; break;
                case OP_or:  r = v1 | v2; break;
                default:     r = v1 ^ v2; break;
                }
                sp[-2] = js_int32(r);
                sp--;
            } else {
                if (js_binary_logic_slow(ctx, sp, opcode))
                    goto exception;
                sp--;
            }
        }
        break;
    case OP_shr:
        {
            JSValue op1 = sp[-2], op2 = sp[-1];
            if (likely(JS_VALUE_IS_BOTH_INT(op1, op2))) {
                sp[-2] = js_uint32((uint32_t)JS_VALUE_GET_INT(op1) >>
                                   (JS_VALUE_GET_INT(op2) & 0x1f));
                sp--;
            } else {
                if (js_shr_slow(ctx, sp))
                    goto exception;
                sp--;
            }
        }
        break;
    case OP_lt: case OP_lte: case OP_gt: case OP_gte:
    case OP_eq: case OP_neq: case OP_strict_eq: case OP_strict_neq:
        {
            JSValue op1 = sp[-2], op2 = sp[-1];
            int r;
            if (likely(JS_VALUE_IS_BOTH_INT(op1, op2))) {
                int32_t v1 = JS_VALUE_GET_INT(op1), v2 = JS_VALUE_GET_INT(op2);
                switch (opcode) {
                case OP_lt:  r = v1 < v2; break;
                case OP_lte: r = v1 <= v2; break;
                case OP_gt:  r = v1 > v2; break;
                case OP_gte: r = v1 >= v2; break;
                case OP_eq: case OP_strict_eq: r = v1 == v2; break;
                default: r = v1 != v2; break;
                }
                sp[-2] = js_bool(r);
                sp--;
                break;
            }
            switch (opcode) {
            case OP_lt: case OP_lte: case OP_gt: case OP_gte:
                r = js_relational_slow(ctx, sp, opcode);
                break;
            case OP_eq: case OP_neq:
                r = js_eq_slow(ctx, sp, opcode == OP_neq);
                break;
            default:
                r = js_strict_eq_slow(ctx, sp, opcode == OP_strict_neq);
                break;
            }
            if (r)
                goto exception;
            sp--;
        }
        break;
    case OP_in:
        if (js_operator_in(ctx, sp))
            goto exception;
        sp--;
        break;
    case OP_instanceof:
        if (js_operator_instanceof(ctx, sp))
            goto exception;
        sp--;
        break;
    case OP_typeof:
        {
            JSValue op1 = sp[-1];
            JSAtom atom = js_operator_typeof(ctx, op1);
            JS_FreeValue(ctx, op1);
            sp[-1] = JS_AtomToString(ctx, atom);
        }
        break;
    case OP_to_propkey:
        switch (JS_VALUE_GET_TAG(sp[-1])) {
        case JS_TAG_INT:
        case JS_TAG_STRING:
        case JS_TAG_SYMBOL:
            break;
        default:
            ret_val = JS_ToPropertyKey(ctx, sp[-1]);
            if (JS_IsException(ret_val))
                goto exception;
            JS_FreeValue(ctx, sp[-1]);
            sp[-1] = ret_val;
            break;
        }
        break;
    case OP_to_propkey2:
        /* must be tested first to avoid the exception */
        if (unlikely(JS_IsUndefined(sp[-2]) || JS_IsNull(sp[-2]))) {
            JS_ThrowTypeError(ctx, "value has no property");
            goto exception;
        }
        switch (JS_VALUE_GET_TAG(sp[-1])) {
        case JS_TAG_INT:
        case JS_TAG_STRING:
        case JS_TAG_SYMBOL:
            break;
        default:
            ret_val = JS_ToPropertyKey(ctx, sp[-1]);
            if (JS_IsException(ret_val))
                goto exception;
            JS_FreeValue(ctx, sp[-1]);
            sp[-1] = ret_val;
            break;
        }
        break;
    case OP_is_undefined_or_null:
    case OP_is_undefined:
    case OP_is_null:
    case OP_typeof_is_undefined:
    case OP_typeof_is_function:
        {
            JSValue v = sp[-1];
            uint32_t tag = JS_VALUE_GET_TAG(v);
            bool r;
            switch (opcode) {
            case OP_is_undefined_or_null:
                r = tag == JS_TAG_UNDEFINED || tag == JS_TAG_NULL;
                break;
            case OP_is_undefined:
                r = tag == JS_TAG_UNDEFINED;
                break;
            case OP_is_null:
                r = tag == JS_TAG_NULL;
                break;
            case OP_typeof_is_undefined:
                r = js_operator_typeof(ctx, v) == JS_ATOM_undefined;
                break;
            default:
                r = js_operator_typeof(ctx, v) == JS_ATOM_function;
                break;
            }
            JS_FreeValue(ctx, v);
            sp[-1] = js_bool(r);
        }
        break;

    /* exceptions */
    case OP_catch:
        sp[0] = JS_NewCatchOffset(ctx, pc + (int32_t)get_u32(pc) - b->byte_code_buf);
        sp++;
        break;
    case OP_nip_catch:
        {
            JSValue rv = *--sp;
            while (sp > stack_buf &&
                   JS_VALUE_GET_TAG(sp[-1]) != JS_TAG_CATCH_OFFSET) {
                JS_FreeValue(ctx, *--sp);
            }
            if (unlikely(sp == stack_buf)) {
                JS_ThrowInternalError(ctx, "nip_catch");
                JS_FreeValue(ctx, rv);
                goto exception;
            }
            sp[-1] = rv;
        }
        break;
    case OP_throw:
        JS_Throw(ctx, *--sp);
        goto exception;

    default:
        abort();
    }
    return sp;

 exception:
    f->sp = sp;
    f->pc = op_pc + short_opcode_info(opcode).size;
    return NULL;
}

/* ---------------------------------------------------------------------
 * Property caches: a get_field, get_field2, get_length, put_field, get_var
 * or put_var site's guesses
 *
 *  - own: the slot (idx) the property had in the last object's shape:
 *    right whenever that slot of the object's shape has the atom with the
 *    flags wanted, whatever the shape (no shape identity needed);
 *  - proto (get only): the receiver's shape (hashed: hash and pointer),
 *    its prototype (holder) and the slot there: the receiver has no own
 *    property of that name while its shape is that one;
 *  - globals: the global object's slot, while the lexical globals' object
 *    has the shape and property count it had (it only grows).
 * --------------------------------------------------------------------- */

typedef struct JitIC {
    uint32_t idx;               /* own: the slot guessed */
    JSShape *rshape;            /* proto: the receiver's shape */
    uint32_t rhash;             /* ... its hash */
    JSObject *holder;           /* ... its prototype */
    JSShape *hshape;            /* ... the prototype's shape */
    uint32_t hidx;              /* ... the slot there */
} JitIC;

static inline __attribute__((always_inline)) JSProperty *ic_slot(JSObject *p, uint32_t i, JSAtom atom,
                                        int mask, int want)
{
    JSShape *sh = p->shape;
    if (likely(i < (uint32_t)sh->prop_count)) {
        JSShapeProperty *prs = get_shape_prop(sh) + i;
        if (likely(prs->atom == atom && (prs->flags & mask) == want))
            return &p->prop[i];
    }
    return NULL;
}

/* get_field, get_field2, get_length */
static JSValue *jit_get_field(JSJitFrame *f, JSValue *sp, const uint8_t *op_pc,
                              JitIC *ic)
{
    JSContext *ctx = f->ctx;
    int opcode = *op_pc;
    JSAtom atom = opcode == OP_get_length ? JS_ATOM_length : get_u32(op_pc + 1);
    JSValue obj = sp[-1], val;
    JSObject *p, *p1;
    JSProperty *pr;
    JSShapeProperty *prs;

    if (likely(JS_VALUE_GET_TAG(obj) == JS_TAG_OBJECT)) {
        p = JS_VALUE_GET_OBJ(obj);
        pr = ic_slot(p, ic->idx, atom, JS_PROP_TMASK, 0);
        if (pr) {
            val = js_dup(pr->u.value);
            goto done;
        }
        if (p->shape == ic->rshape && p->shape->is_hashed &&
            p->shape->hash == ic->rhash && p->shape->proto == ic->holder &&
            ic->holder->shape == ic->hshape) {
            pr = ic_slot(ic->holder, ic->hidx, atom, JS_PROP_TMASK, 0);
            if (pr) {
                val = js_dup(pr->u.value);
                goto done;
            }
        }
        /* the interpreter's way, the guesses updated */
        p1 = p;
        for (;;) {
            prs = find_own_property(&pr, p1, atom);
            if (prs) {
                if (unlikely(prs->flags & JS_PROP_TMASK))
                    goto slow;
                val = js_dup(pr->u.value);
                if (p1 == p) {
                    ic->idx = prs - get_shape_prop(p->shape);
                } else if (p1 == p->shape->proto && p->shape->is_hashed &&
                           !p->is_exotic) {
                    ic->rshape = p->shape;
                    ic->rhash = p->shape->hash;
                    ic->holder = p1;
                    ic->hshape = p1->shape;
                    ic->hidx = prs - get_shape_prop(p1->shape);
                }
                goto done;
            }
            if (unlikely(p1->is_exotic)) {
                obj = JS_MKPTR(JS_TAG_OBJECT, p1);
                goto slow;
            }
            p1 = p1->shape->proto;
            if (!p1) {
                val = JS_UNDEFINED;
                goto done;
            }
        }
    }
 slow:
    f->sf->cur_pc = (uint8_t *)op_pc + short_opcode_info(opcode).size;
    val = JS_GetPropertyInternal(ctx, obj, atom, sp[-1], false);
    if (unlikely(JS_IsException(val))) {
        f->sp = sp;
        f->pc = op_pc + short_opcode_info(opcode).size;
        return NULL;
    }
 done:
    if (opcode == OP_get_field2) {
        *sp++ = val;
    } else {
        JS_FreeValue(ctx, sp[-1]);
        sp[-1] = val;
    }
    return sp;
}

static JSValue *jit_put_field(JSJitFrame *f, JSValue *sp, const uint8_t *op_pc,
                              JitIC *ic)
{
    JSContext *ctx = f->ctx;
    JSAtom atom = get_u32(op_pc + 1);
    JSValue obj = sp[-2];
    JSObject *p;
    JSProperty *pr;
    JSShapeProperty *prs;
    int ret;

    if (likely(JS_VALUE_GET_TAG(obj) == JS_TAG_OBJECT)) {
        p = JS_VALUE_GET_OBJ(obj);
        pr = ic_slot(p, ic->idx, atom,
                     JS_PROP_TMASK | JS_PROP_WRITABLE | JS_PROP_LENGTH,
                     JS_PROP_WRITABLE);
        if (!pr) {
            prs = find_own_property(&pr, p, atom);
            if (prs && (prs->flags & (JS_PROP_TMASK | JS_PROP_WRITABLE |
                                      JS_PROP_LENGTH)) == JS_PROP_WRITABLE)
                ic->idx = prs - get_shape_prop(p->shape);
            else
                pr = NULL;
        }
        if (pr) {
            set_value(ctx, &pr->u.value, sp[-1]);
            JS_FreeValue(ctx, obj);
            return sp - 2;
        }
    }
    f->sf->cur_pc = (uint8_t *)op_pc + 5;
    ret = JS_SetPropertyInternal2(ctx, obj, atom, sp[-1], obj,
                                  JS_PROP_THROW_STRICT);
    JS_FreeValue(ctx, obj);
    sp -= 2;
    if (unlikely(ret < 0)) {
        f->sp = sp;
        f->pc = op_pc + 5;
        return NULL;
    }
    return sp;
}

/* get_var, get_var_undef: global variables (JS_GetGlobalVar()) */
static JSValue *jit_get_var(JSJitFrame *f, JSValue *sp, const uint8_t *op_pc,
                            JitIC *ic)
{
    JSContext *ctx = f->ctx;
    JSAtom atom = get_u32(op_pc + 1);
    JSObject *gvo = JS_VALUE_GET_OBJ(ctx->global_var_obj);
    JSObject *go = JS_VALUE_GET_OBJ(ctx->global_obj);
    JSProperty *pr;
    JSShapeProperty *prs;
    JSValue val;

    /* a lexical global (let, const, class at the top level) */
    pr = ic_slot(gvo, ic->idx, atom, 0, 0);
    if (pr && !JS_IsUninitialized(pr->u.value)) {
        *sp++ = js_dup(pr->u.value);
        return sp;
    }
    /* a property of the global object, none of that name in gvo */
    if (gvo->shape == ic->rshape && (uint32_t)gvo->shape->prop_count == ic->rhash) {
        pr = ic_slot(go, ic->hidx, atom, JS_PROP_TMASK, 0);
        if (pr) {
            *sp++ = js_dup(pr->u.value);
            return sp;
        }
    }
    prs = find_own_property(&pr, gvo, atom);
    if (prs) {
        ic->idx = prs - get_shape_prop(gvo->shape);
    } else {
        prs = find_own_property(&pr, go, atom);
        if (prs && (prs->flags & JS_PROP_TMASK) == 0) {
            ic->rshape = gvo->shape;
            ic->rhash = gvo->shape->prop_count;
            ic->hidx = prs - get_shape_prop(go->shape);
        }
    }
    f->sf->cur_pc = (uint8_t *)op_pc + 5;
    val = JS_GetGlobalVar(ctx, atom, *op_pc - OP_get_var_undef);
    if (unlikely(JS_IsException(val))) {
        f->sp = sp;
        f->pc = op_pc + 5;
        return NULL;
    }
    *sp++ = val;
    return sp;
}

/* ---------------------------------------------------------------------
 * The compiler
 * --------------------------------------------------------------------- */

#define R_SPV   4               /* the operand stack pointer */
#define R_VAR   5
#define R_ARG   6
#define R_CTX   7
#define R_F     8
#define R_VREF  10

/* A branch to patch: at code offset pos, to the code of bytecode offset bc
   (kind 0) or to stub n (kind 1); cond CC_AL: B.W */
typedef struct {
    uint32_t pos;
    uint32_t target;
    uint8_t kind;
    uint8_t cond;
} JitFix;

/* A cold path, emitted after the function's code */
enum {
    STUB_OP,        /* jit_op(f, sp, pc), then back (NULL: exception exit) */
    STUB_FREE,      /* jit_free_value(ctx, r2:r3), then back */
    STUB_BAIL,      /* the interpreter goes on at pc */
    STUB_BOOL,      /* jit_to_bool(ctx, r2:r3): taken (target) or back */
    STUB_POLL,      /* the interrupt counter ran out: poll, then target */
};

typedef struct {
    uint8_t kind;
    uint8_t cond;               /* STUB_BOOL: when the result is non-zero
                                   (CC_NE) or zero (CC_EQ) the jump is taken */
    uint32_t pc;                /* bytecode offset (STUB_OP, BAIL, POLL: of
                                   the instruction; BOOL: unused) */
    uint32_t back;              /* code offset to go back to */
    uint32_t target;            /* bytecode offset of the jump's target */
} JitStub;

typedef struct {
    JSContext *ctx;
    JSFunctionBytecode *b;
    jbuf_t cb;
    uint32_t *natoff;           /* code offset of each bytecode offset */
    uint8_t *is_target;
    JitFix *fix;
    uint32_t nfix, cfix;
    JitStub *stub;
    uint32_t nstub, cstub;
    struct JitIC *ics;          /* the property caches (one a site) */
    uint32_t nics;
    uint32_t exc_exit;          /* code offset of the exception exit */
    uint32_t ret_exit;          /* ... of the return exit (f->ret set) */
    bool fail;
} JitC;

static void jc_fix(JitC *c, uint32_t target, int kind, int cond)
{
    if (c->nfix == c->cfix) {
        c->cfix = c->cfix ? c->cfix * 2 : 64;
        c->fix = realloc(c->fix, c->cfix * sizeof(*c->fix));
        if (!c->fix) {
            c->fail = true;
            c->nfix = c->cfix = 0;
            return;
        }
    }
    c->fix[c->nfix++] = (JitFix){ c->cb.len, target, kind, cond };
    e32(&c->cb, 0, 0);
}

static uint32_t jc_stub(JitC *c, int kind, int cond, uint32_t pc,
                        uint32_t target)
{
    if (c->nstub == c->cstub) {
        c->cstub = c->cstub ? c->cstub * 2 : 64;
        c->stub = realloc(c->stub, c->cstub * sizeof(*c->stub));
        if (!c->stub) {
            c->fail = true;
            c->nstub = c->cstub = 0;
            return 0;
        }
    }
    c->stub[c->nstub] = (JitStub){ kind, cond, pc, 0, target };
    return c->nstub++;
}

/* A branch (cond) to a new stub that comes back here */
static void jc_cold(JitC *c, int cond, int kind, uint32_t pc)
{
    uint32_t n = jc_stub(c, kind, cond, pc, 0);
    jc_fix(c, n, 1, cond);
    if (!c->fail)
        c->stub[n].back = c->cb.len;
}

static void t_ldrd(jbuf_t *b, int rt, int rt2, int rn, int off, int mode)
{
    /* mode 0: [rn, #off]; 1: [rn, #off]!; 2: [rn], #off */
    int p = mode != 2, w = mode != 0, u = off >= 0;
    int a = u ? off : -off;
    e32(b, 0xe850 | p << 8 | u << 7 | w << 5 | rn, rt << 12 | rt2 << 8 | (a >> 2));
}

static void t_strd(jbuf_t *b, int rt, int rt2, int rn, int off, int mode)
{
    int p = mode != 2, w = mode != 0, u = off >= 0;
    int a = u ? off : -off;
    e32(b, 0xe840 | p << 8 | u << 7 | w << 5 | rn, rt << 12 | rt2 << 8 | (a >> 2));
}

/* ldrd/strd [rn, #off] for any off (ip used out of range) */
static void jc_ldrd(JitC *c, int rt, int rt2, int rn, int off)
{
    if (off >= -1020 && off <= 1020) {
        t_ldrd(&c->cb, rt, rt2, rn, off, 0);
    } else {
        t_movi(&c->cb, R_IP, (uint32_t)off);
        t_dp_reg(&c->cb, DP_ADD, 0, R_IP, rn, R_IP, SH_LSL, 0);
        t_ldrd(&c->cb, rt, rt2, R_IP, 0, 0);
    }
}

static void jc_strd(JitC *c, int rt, int rt2, int rn, int off)
{
    if (off >= -1020 && off <= 1020) {
        t_strd(&c->cb, rt, rt2, rn, off, 0);
    } else {
        t_movi(&c->cb, R_IP, (uint32_t)off);
        t_dp_reg(&c->cb, DP_ADD, 0, R_IP, rn, R_IP, SH_LSL, 0);
        t_strd(&c->cb, rt, rt2, R_IP, 0, 0);
    }
}

static void jc_call(JitC *c, void *fn)
{
    t_movw(&c->cb, R_IP, (uint32_t)(uintptr_t)fn & 0xffff);
    t_movt(&c->cb, R_IP, (uint32_t)(uintptr_t)fn >> 16);
    t_blx(&c->cb, R_IP);
}

/* The reference count of a value's object: in the allocator's block
   header, before the object (js_rc()) */
#define JIT_RC_OFF ((int)offsetof(JSMallocBlockHeader, ref_count) - \
                    (int)offsetof(JSMallocBlockHeader, user_data))

/* r0:r1 (a value) referenced once more */
static void jc_dup01(JitC *c)
{
    t_cmni_enc(&c->cb, 1, t_modimm(9));             /* cmn r1, #9 */
    t_it(&c->cb, CC_HS, "ttt");
    t_ldst(&c->cb, LS_LDR, 2, 0, JIT_RC_OFF);       /* ldr r2, [r0, #rc] */
    e16(&c->cb, 0x3201);                            /* add r2, #1 */
    t_ldst(&c->cb, LS_STR, 2, 0, JIT_RC_OFF);
}

/* r2:r3 (a value) referenced once less (freed at 0) */
static void jc_free23(JitC *c)
{
    t_cmni_enc(&c->cb, 3, t_modimm(9));             /* cmn r3, #9 */
    e16(&c->cb, 0xd307);                            /* bcc: over the next 16 bytes */
    t_ldst(&c->cb, LS_LDR, R_IP, 2, JIT_RC_OFF);    /* ldr.w ip, [r2, #rc] */
    t_dp_imm(&c->cb, DP_SUB, 1, R_IP, R_IP, t_modimm(1)); /* subs ip, #1 */
    t_ldst(&c->cb, LS_STR, R_IP, 2, JIT_RC_OFF);    /* str.w ip, [r2, #rc] */
    jc_cold(c, CC_LE, STUB_FREE, 0);
}

static void jc_push01(JitC *c)
{
    t_strd(&c->cb, 0, 1, R_SPV, 8, 2);              /* strd r0, r1, [r4], #8 */
}

static void jc_pop01(JitC *c)
{
    t_ldrd(&c->cb, 0, 1, R_SPV, -8, 1);             /* ldrd r0, r1, [r4, #-8]! */
}

static void jc_pop23(JitC *c)
{
    t_ldrd(&c->cb, 2, 3, R_SPV, -8, 1);
}

static void jc_push_imm(JitC *c, int tag, uint32_t v)
{
    t_movi(&c->cb, 0, v);
    t_movi(&c->cb, 1, (uint32_t)tag);
    jc_push01(c);
}

/* get: base[idx] pushed (base r5 var_buf, r6 arg_buf) */
static void jc_get(JitC *c, int base, int idx)
{
    jc_ldrd(c, 0, 1, base, idx * 8);
    jc_dup01(c);
    jc_push01(c);
}

/* put: popped into base[idx] (dup: the top stays, referenced again) */
static void jc_put(JitC *c, int base, int idx, bool keep)
{
    if (keep) {
        t_ldrd(&c->cb, 0, 1, R_SPV, -8, 0);
        jc_dup01(c);
    } else {
        jc_pop01(c);
    }
    if (idx * 8 > 1020) {
        t_movi(&c->cb, R_LR, (uint32_t)idx * 8);
        t_dp_reg(&c->cb, DP_ADD, 0, R_LR, base, R_LR, SH_LSL, 0);
        base = R_LR;
        idx = 0;
    }
    t_ldrd(&c->cb, 2, 3, base, idx * 8, 0);
    t_strd(&c->cb, 0, 1, base, idx * 8, 0);
    jc_free23(c);
}

/* the address of var_refs[idx]->pvalue into lr */
static void jc_vref(JitC *c, int idx)
{
    t_ldst(&c->cb, LS_LDR, R_LR, R_VREF, idx * 4);
    t_ldst(&c->cb, LS_LDR, R_LR, R_LR, offsetof(JSVarRef, pvalue));
}

/* the instruction at bytecode offset pc run by jit_op() */
static void jc_op(JitC *c, uint32_t pc)
{
    t_mov(&c->cb, 0, R_F);
    t_mov(&c->cb, 1, R_SPV);
    t_movi(&c->cb, 2, (uint32_t)(uintptr_t)(c->b->byte_code_buf + pc));
    jc_call(c, (void *)jit_op);
    e16(&c->cb, 0x0004);                            /* movs r4, r0 */
    jc_fix(c, c->exc_exit, 2, CC_EQ);
}

/* the instruction at pc run by fn(f, sp, pc, its property cache) */
static void jc_op_ic(JitC *c, uint32_t pc, void *fn)
{
    struct JitIC *ic = &c->ics[c->nics++];
    t_mov(&c->cb, 0, R_F);
    t_mov(&c->cb, 1, R_SPV);
    t_movi(&c->cb, 2, (uint32_t)(uintptr_t)(c->b->byte_code_buf + pc));
    t_movi(&c->cb, 3, (uint32_t)(uintptr_t)ic);
    jc_call(c, fn);
    e16(&c->cb, 0x0004);                            /* movs r4, r0 */
    jc_fix(c, c->exc_exit, 2, CC_EQ);
}

/* a jump to bytecode offset target (cond); backward: the interrupt
   counter first */
static void jc_jump(JitC *c, int cond, uint32_t pc, uint32_t target)
{
    if (target <= pc) {
        uint32_t n = jc_stub(c, STUB_POLL, 0, pc, target);
        jc_fix(c, n, 1, cond);
    } else {
        jc_fix(c, target, 0, cond);
    }
}

/* Whether the instruction at pc is an int comparison the next if_false or
   if_true can take as its condition: the condition code (signed, of
   "cmp a, b") or -1 */
static int jc_cmp_cc(int op)
{
    switch (op) {
    case OP_lt: return CC_LT;
    case OP_lte: return CC_LE;
    case OP_gt: return CC_GT;
    case OP_gte: return CC_GE;
    case OP_eq: case OP_strict_eq: return CC_EQ;
    case OP_neq: case OP_strict_neq: return CC_NE;
    default: return -1;
    }
}

static int jc_body(JitC *c)
{
    JSFunctionBytecode *b = c->b;
    const uint8_t *bc = b->byte_code_buf;
    uint32_t pc, len = b->byte_code_len;

    /* the jumps' targets */
    for (pc = 0; pc < len; ) {
        int op = bc[pc];
        const JSOpCode *oi = &short_opcode_info(op);
        int32_t t = -1;
        switch (op) {
        case OP_if_false: case OP_if_true: case OP_goto: case OP_catch:
        case OP_gosub:
            t = pc + 1 + (int32_t)get_u32(bc + pc + 1);
            break;
        case OP_if_false8: case OP_if_true8: case OP_goto8:
            t = pc + 1 + (int8_t)bc[pc + 1];
            break;
        case OP_goto16:
            t = pc + 1 + (int16_t)get_u16(bc + pc + 1);
            break;
        case OP_with_get_var: case OP_with_put_var: case OP_with_delete_var:
        case OP_with_make_ref: case OP_with_get_ref: case OP_with_get_ref_undef:
            t = pc + 5 + (int32_t)get_u32(bc + pc + 5);
            break;
        }
        if (t >= 0 && (uint32_t)t < len)
            c->is_target[t] = 1;
        pc += oi->size;
    }
    c->is_target[0] = 1;

    for (pc = 0; pc < len && !c->fail && !c->cb.oom; ) {
        int op = bc[pc];
        int size = short_opcode_info(op).size;
        const uint8_t *a = bc + pc + 1;
        int next = pc + size < len ? bc[pc + size] : -1;
        bool next_free = pc + size < len && !c->is_target[pc + size];

        c->natoff[pc] = c->cb.len;
        switch (op) {
        /* constants */
        case OP_push_i32: jc_push_imm(c, JS_TAG_INT, get_u32(a)); break;
        case OP_push_i8:  jc_push_imm(c, JS_TAG_INT, (int8_t)a[0]); break;
        case OP_push_i16: jc_push_imm(c, JS_TAG_INT, (int16_t)get_u16(a)); break;
        case OP_push_minus1: case OP_push_0: case OP_push_1: case OP_push_2:
        case OP_push_3: case OP_push_4: case OP_push_5: case OP_push_6:
        case OP_push_7:
            jc_push_imm(c, JS_TAG_INT, op - OP_push_0);
            break;
        case OP_push_true:  jc_push_imm(c, JS_TAG_BOOL, 1); break;
        case OP_push_false: jc_push_imm(c, JS_TAG_BOOL, 0); break;
        case OP_null:       jc_push_imm(c, JS_TAG_NULL, 0); break;
        case OP_undefined:  jc_push_imm(c, JS_TAG_UNDEFINED, 0); break;

        /* locals, arguments, closure variables */
        case OP_get_loc:  jc_get(c, R_VAR, get_u16(a)); break;
        case OP_get_loc8: jc_get(c, R_VAR, a[0]); break;
        case OP_get_loc0: case OP_get_loc1: case OP_get_loc2: case OP_get_loc3:
            jc_get(c, R_VAR, op - OP_get_loc0);
            break;
        case OP_get_loc0_loc1:
            jc_get(c, R_VAR, 0);
            jc_get(c, R_VAR, 1);
            break;
        case OP_put_loc:  jc_put(c, R_VAR, get_u16(a), false); break;
        case OP_put_loc8: jc_put(c, R_VAR, a[0], false); break;
        case OP_put_loc0: case OP_put_loc1: case OP_put_loc2: case OP_put_loc3:
            jc_put(c, R_VAR, op - OP_put_loc0, false);
            break;
        case OP_set_loc:  jc_put(c, R_VAR, get_u16(a), true); break;
        case OP_set_loc8: jc_put(c, R_VAR, a[0], true); break;
        case OP_set_loc0: case OP_set_loc1: case OP_set_loc2: case OP_set_loc3:
            jc_put(c, R_VAR, op - OP_set_loc0, true);
            break;
        case OP_get_arg:  jc_get(c, R_ARG, get_u16(a)); break;
        case OP_get_arg0: case OP_get_arg1: case OP_get_arg2: case OP_get_arg3:
            jc_get(c, R_ARG, op - OP_get_arg0);
            break;
        case OP_put_arg:  jc_put(c, R_ARG, get_u16(a), false); break;
        case OP_put_arg0: case OP_put_arg1: case OP_put_arg2: case OP_put_arg3:
            jc_put(c, R_ARG, op - OP_put_arg0, false);
            break;
        case OP_set_arg:  jc_put(c, R_ARG, get_u16(a), true); break;
        case OP_set_arg0: case OP_set_arg1: case OP_set_arg2: case OP_set_arg3:
            jc_put(c, R_ARG, op - OP_set_arg0, true);
            break;
        case OP_get_var_ref:
        case OP_get_var_ref0: case OP_get_var_ref1: case OP_get_var_ref2:
        case OP_get_var_ref3:
            jc_vref(c, op == OP_get_var_ref ? get_u16(a) : op - OP_get_var_ref0);
            t_ldrd(&c->cb, 0, 1, R_LR, 0, 0);
            jc_dup01(c);
            jc_push01(c);
            break;
        case OP_put_var_ref: case OP_set_var_ref:
        case OP_put_var_ref0: case OP_put_var_ref1: case OP_put_var_ref2:
        case OP_put_var_ref3:
        case OP_set_var_ref0: case OP_set_var_ref1: case OP_set_var_ref2:
        case OP_set_var_ref3:
            {
                bool keep = op == OP_set_var_ref ||
                            (op >= OP_set_var_ref0 && op <= OP_set_var_ref3);
                int idx = op == OP_put_var_ref || op == OP_set_var_ref ? get_u16(a) :
                          keep ? op - OP_set_var_ref0 : op - OP_put_var_ref0;
                if (keep) {
                    t_ldrd(&c->cb, 0, 1, R_SPV, -8, 0);
                    jc_dup01(c);
                } else {
                    jc_pop01(c);
                }
                jc_vref(c, idx);
                t_ldrd(&c->cb, 2, 3, R_LR, 0, 0);
                t_strd(&c->cb, 0, 1, R_LR, 0, 0);
                jc_free23(c);
            }
            break;

        /* the stack */
        case OP_drop:
            jc_pop23(c);
            jc_free23(c);
            break;
        case OP_dup:
            t_ldrd(&c->cb, 0, 1, R_SPV, -8, 0);
            jc_dup01(c);
            jc_push01(c);
            break;
        case OP_nip:
            jc_pop01(c);
            t_ldrd(&c->cb, 2, 3, R_SPV, -8, 0);
            t_strd(&c->cb, 0, 1, R_SPV, -8, 0);
            jc_free23(c);
            break;
        case OP_swap:
            t_ldrd(&c->cb, 0, 1, R_SPV, -16, 0);
            t_ldrd(&c->cb, 2, 3, R_SPV, -8, 0);
            t_strd(&c->cb, 2, 3, R_SPV, -16, 0);
            t_strd(&c->cb, 0, 1, R_SPV, -8, 0);
            break;

        /* int arithmetic inline, the rest in jit_op() */
        case OP_add:
        case OP_sub:
            t_ldrd(&c->cb, 0, 1, R_SPV, -16, 0);
            t_ldrd(&c->cb, 2, 3, R_SPV, -8, 0);
            t_dp_reg(&c->cb, DP_ORR, 1, R_IP, 1, 3, SH_LSL, 0);  /* orrs ip, r1, r3 */
            jc_cold(c, CC_NE, STUB_OP, pc);
            {
                /* the stub comes back after the operation: emit it so */
                uint32_t n = c->nstub - 1;
                t_dp_reg(&c->cb, op == OP_add ? DP_ADD : DP_SUB, 1, 0, 0, 2,
                         SH_LSL, 0);                         /* adds/subs r0, r0, r2 */
                jc_fix(c, n, 1, CC_VS);
                t_ldst(&c->cb, LS_STR, 0, R_SPV, -16);
                t_addi(&c->cb, R_SPV, R_SPV, -8);
                c->stub[n].back = c->cb.len;
            }
            break;
        case OP_mul:
            t_ldrd(&c->cb, 0, 1, R_SPV, -16, 0);
            t_ldrd(&c->cb, 2, 3, R_SPV, -8, 0);
            t_dp_reg(&c->cb, DP_ORR, 1, R_IP, 1, 3, SH_LSL, 0);
            jc_cold(c, CC_NE, STUB_OP, pc);
            {
                uint32_t n = c->nstub - 1;
                e32(&c->cb, 0xfb80 | 0, R_LR << 12 | R_IP << 8 | 2);  /* smull lr, ip, r0, r2 */
                t_dp_reg(&c->cb, DP_SUB, 1, 15, R_IP, R_LR, SH_ASR, 31); /* cmp ip, lr, asr #31 */
                jc_fix(c, n, 1, CC_NE);
                t_cmpi_enc(&c->cb, R_LR, 0);
                jc_fix(c, n, 1, CC_EQ);                 /* 0: maybe -0, the slow way */
                t_ldst(&c->cb, LS_STR, R_LR, R_SPV, -16);
                t_addi(&c->cb, R_SPV, R_SPV, -8);
                c->stub[n].back = c->cb.len;
            }
            break;
        case OP_and: case OP_or: case OP_xor:
            t_ldrd(&c->cb, 0, 1, R_SPV, -16, 0);
            t_ldrd(&c->cb, 2, 3, R_SPV, -8, 0);
            t_dp_reg(&c->cb, DP_ORR, 1, R_IP, 1, 3, SH_LSL, 0);
            jc_cold(c, CC_NE, STUB_OP, pc);
            {
                uint32_t n = c->nstub - 1;
                t_dp_reg(&c->cb, op == OP_and ? DP_AND : op == OP_or ? DP_ORR : DP_EOR,
                         0, 0, 0, 2, SH_LSL, 0);
                t_ldst(&c->cb, LS_STR, 0, R_SPV, -16);
                t_addi(&c->cb, R_SPV, R_SPV, -8);
                c->stub[n].back = c->cb.len;
            }
            break;
        case OP_inc_loc:
        case OP_dec_loc:
            {
                int off = a[0] * 8;
                jc_ldrd(c, 0, 1, R_VAR, off);
                t_cmpi_enc(&c->cb, 1, 0);
                jc_cold(c, CC_NE, STUB_OP, pc);
                uint32_t n = c->nstub - 1;
                t_dp_imm(&c->cb, op == OP_inc_loc ? DP_ADD : DP_SUB, 1, 0, 0,
                         t_modimm(1));                    /* adds/subs r0, #1 */
                jc_fix(c, n, 1, CC_VS);
                t_ldst(&c->cb, LS_STR, 0, R_VAR, off);
                c->stub[n].back = c->cb.len;
            }
            break;
        case OP_inc:
        case OP_dec:
            t_ldrd(&c->cb, 0, 1, R_SPV, -8, 0);
            t_cmpi_enc(&c->cb, 1, 0);
            jc_cold(c, CC_NE, STUB_OP, pc);
            {
                uint32_t n = c->nstub - 1;
                t_dp_imm(&c->cb, op == OP_inc ? DP_ADD : DP_SUB, 1, 0, 0, t_modimm(1));
                jc_fix(c, n, 1, CC_VS);
                t_ldst(&c->cb, LS_STR, 0, R_SPV, -8);
                c->stub[n].back = c->cb.len;
            }
            break;
        case OP_lt: case OP_lte: case OP_gt: case OP_gte:
        case OP_eq: case OP_neq: case OP_strict_eq: case OP_strict_neq:
            {
                int cc = jc_cmp_cc(op);
                bool fuse = next_free && (next == OP_if_false || next == OP_if_true ||
                                          next == OP_if_false8 || next == OP_if_true8);
                t_ldrd(&c->cb, 0, 1, R_SPV, -16, 0);
                t_ldrd(&c->cb, 2, 3, R_SPV, -8, 0);
                t_dp_reg(&c->cb, DP_ORR, 1, R_IP, 1, 3, SH_LSL, 0);
                if (!fuse) {
                    jc_cold(c, CC_NE, STUB_OP, pc);
                    uint32_t n = c->nstub - 1;
                    t_cmp(&c->cb, 0, 2);
                    t_it(&c->cb, cc, "te");               /* ite cc */
                    e16(&c->cb, 0x2001);                /* mov r0, #1 */
                    e16(&c->cb, 0x2000);                /* mov r0, #0 */
                    e16(&c->cb, 0x2101);                /* movs r1, #1 (JS_TAG_BOOL) */
                    t_strd(&c->cb, 0, 1, R_SPV, -16, 0);
                    t_addi(&c->cb, R_SPV, R_SPV, -8);
                    c->stub[n].back = c->cb.len;
                } else {
                    /* compare and jump: the slow way runs the comparison in
                       jit_op() and goes on with the jump's own code */
                    uint32_t npc = pc + size;
                    const uint8_t *na = bc + npc + 1;
                    int32_t target = next == OP_if_false || next == OP_if_true ?
                        (int32_t)(npc + 1 + (int32_t)get_u32(na)) :
                        (int32_t)(npc + 1 + (int8_t)na[0]);
                    bool on_true = next == OP_if_true || next == OP_if_true8;
                    uint32_t n = jc_stub(c, STUB_OP, CC_NE, pc, 0);
                    jc_fix(c, n, 1, CC_NE);
                    t_cmp(&c->cb, 0, 2);
                    t_addi(&c->cb, R_SPV, R_SPV, -16);
                    jc_jump(c, on_true ? cc : CC_INV(cc), npc, target);
                    jc_fix(c, npc + short_opcode_info(next).size, 0, CC_AL);
                    /* the slow comparison comes back to the jump */
                    c->stub[n].back = 0xffffffff;
                    c->stub[n].target = npc;
                }
            }
            break;

        /* jumps */
        case OP_goto:
            jc_jump(c, CC_AL, pc, pc + 1 + (int32_t)get_u32(a));
            break;
        case OP_goto16:
            jc_jump(c, CC_AL, pc, pc + 1 + (int16_t)get_u16(a));
            break;
        case OP_goto8:
            jc_jump(c, CC_AL, pc, pc + 1 + (int8_t)a[0]);
            break;
        case OP_if_false: case OP_if_true: case OP_if_false8: case OP_if_true8:
            {
                bool on_true = op == OP_if_true || op == OP_if_true8;
                uint32_t target = op == OP_if_false || op == OP_if_true ?
                    pc + 1 + (int32_t)get_u32(a) : pc + 1 + (int8_t)a[0];
                uint32_t n;
                jc_pop23(c);
                t_cmpi_enc(&c->cb, 3, JS_TAG_UNDEFINED);
                n = jc_stub(c, STUB_BOOL, on_true ? CC_NE : CC_EQ, pc, target);
                jc_fix(c, n, 1, CC_HI);
                t_cmpi_enc(&c->cb, 2, 0);
                jc_jump(c, on_true ? CC_NE : CC_EQ, pc, target);
                if (!c->fail)
                    c->stub[n].back = c->cb.len;
            }
            break;

        /* properties, globals: through their caches */
        case OP_get_field: case OP_get_field2: case OP_get_length:
            jc_op_ic(c, pc, (void *)jit_get_field);
            break;
        case OP_put_field:
            jc_op_ic(c, pc, (void *)jit_put_field);
            break;
        case OP_get_var: case OP_get_var_undef:
            jc_op_ic(c, pc, (void *)jit_get_var);
            break;
        case OP_push_this:
            /* (only at a function's start: the frame's this) */
            t_ldrd(&c->cb, 0, 1, R_F, offsetof(JSJitFrame, this_obj), 0);
            if (!b->is_strict_mode) {
                uint32_t n;
                t_cmni_enc(&c->cb, 1, t_modimm(1));     /* cmn r1, #1: an object */
                jc_cold(c, CC_NE, STUB_OP, pc);
                n = c->nstub - 1;
                jc_dup01(c);
                jc_push01(c);
                c->stub[n].back = c->cb.len;            /* (after the push) */
            } else {
                jc_dup01(c);
                jc_push01(c);
            }
            break;

        /* returns */
        case OP_return:
            jc_pop01(c);
            t_strd(&c->cb, 0, 1, R_F, F_RET, 0);
            jc_fix(c, c->ret_exit, 2, CC_AL);
            break;
        case OP_return_undef:
            t_movi(&c->cb, 0, 0);
            t_movi(&c->cb, 1, JS_TAG_UNDEFINED);
            t_strd(&c->cb, 0, 1, R_F, F_RET, 0);
            jc_fix(c, c->ret_exit, 2, CC_AL);
            break;

        case OP_tail_call:
        case OP_tail_call_method:
            jc_op(c, pc);
            jc_pop01(c);
            t_strd(&c->cb, 0, 1, R_F, F_RET, 0);
            jc_fix(c, c->ret_exit, 2, CC_AL);
            break;

        default:
            if (jit_op_has(op)) {
                jc_op(c, pc);
            } else {
                uint32_t n = jc_stub(c, STUB_BAIL, 0, pc, 0);
                jc_fix(c, n, 1, CC_AL);
            }
            break;
        }
        pc += size;
    }
    return c->fail || c->cb.oom ? -1 : 0;
}

/* The code: prologue, body, exits, stubs; branches patched */
static JSJitCode *jc_compile(JSContext *ctx, JSFunctionBytecode *b)
{
    JitC c;
    JSJitCode *jc = NULL;
    uint32_t i, n, pos_body, len = b->byte_code_len;
    uint32_t *stub_off = NULL;
    uint8_t *mem;

    memset(&c, 0, sizeof(c));
    c.ctx = ctx;
    c.b = b;
    c.natoff = calloc(len + 1, sizeof(uint32_t));
    c.is_target = calloc(len + 1, 1);
    if (!c.natoff || !c.is_target)
        goto done;

    /* prologue: (JSJitFrame *f, entry) */
    if (getenv("QJS_JIT_TRAP"))
        e16(&c.cb, 0xbe00);                         /* bkpt: a debugger stops */
    t_push(&c.cb, 0x4df0);                          /* push {r4-r8, r10, r11, lr} */
    t_mov(&c.cb, R_F, 0);
    t_ldst(&c.cb, LS_LDR, R_SPV, R_F, F_SP);
    t_ldst(&c.cb, LS_LDR, R_VAR, R_F, offsetof(JSJitFrame, var_buf));
    t_ldst(&c.cb, LS_LDR, R_ARG, R_F, offsetof(JSJitFrame, arg_buf));
    t_ldst(&c.cb, LS_LDR, R_CTX, R_F, offsetof(JSJitFrame, ctx));
    t_ldst(&c.cb, LS_LDR, R_VREF, R_F, offsetof(JSJitFrame, var_refs));
    t_bx(&c.cb, 1);

    /* exits (their offsets: fixups of kind 2 target these code offsets) */
    c.ret_exit = c.cb.len;
    t_ldst(&c.cb, LS_STR, R_SPV, R_F, F_SP);
    t_movi(&c.cb, 0, JIT_RET);
    t_pop(&c.cb, 0x8df0);                           /* pop {r4-r8, r10, r11, pc} */
    c.exc_exit = c.cb.len;
    t_movi(&c.cb, 0, JIT_EXC);
    t_pop(&c.cb, 0x8df0);

    pos_body = c.cb.len;
    (void)pos_body;
    {
        uint32_t pc, n_ic = 0;
        for (pc = 0; pc < len; pc += short_opcode_info(b->byte_code_buf[pc]).size) {
            switch (b->byte_code_buf[pc]) {
            case OP_get_field: case OP_get_field2: case OP_get_length:
            case OP_put_field: case OP_get_var: case OP_get_var_undef:
                n_ic++;
            }
        }
        c.ics = js_mallocz(ctx, (n_ic + 1) * sizeof(*c.ics));
        if (!c.ics)
            goto done;
    }
    if (jc_body(&c) < 0)
        goto done;

    /* stubs */
    for (i = 0; i < c.nstub && !c.fail; i++) {
        JitStub st, *s = &st;
        if ((i & 63) == 0) {
            uint32_t *p = realloc(stub_off, (i + 64) * sizeof(uint32_t));
            if (!p)
                goto done;
            stub_off = p;
        }
        stub_off[i] = c.cb.len;
        st = c.stub[i];         /* (a copy: emitting may grow c.stub) */
        switch (s->kind) {
        case STUB_OP:
            jc_op(&c, s->pc);
            if (s->back == 0xffffffff)
                jc_fix(&c, s->target, 0, CC_AL);    /* to the jump's code */
            else
                jc_fix(&c, s->back, 2, CC_AL);
            break;
        case STUB_FREE:
            t_mov(&c.cb, 0, R_CTX);
            jc_call(&c, (void *)jit_free_value);
            jc_fix(&c, s->back, 2, CC_AL);
            break;
        case STUB_BAIL:
            t_ldst(&c.cb, LS_STR, R_SPV, R_F, F_SP);
            t_movi(&c.cb, 0, (uint32_t)(uintptr_t)(b->byte_code_buf + s->pc));
            t_ldst(&c.cb, LS_STR, 0, R_F, F_PC);
            t_movi(&c.cb, 0, JIT_BAIL);
            t_pop(&c.cb, 0x8df0);
            break;
        case STUB_BOOL:
            t_mov(&c.cb, 0, R_CTX);
            jc_call(&c, (void *)jit_to_bool);
            t_cmpi_enc(&c.cb, 0, 0);
            jc_jump(&c, s->cond, s->pc, s->target);
            jc_fix(&c, s->back, 2, CC_AL);
            break;
        case STUB_POLL:
            /* --ctx->interrupt_counter <= 0: poll */
            t_ldst(&c.cb, LS_LDR, 0, R_CTX, offsetof(JSContext, interrupt_counter));
            t_dp_imm(&c.cb, DP_SUB, 1, 0, 0, t_modimm(1));
            t_ldst(&c.cb, LS_STR, 0, R_CTX, offsetof(JSContext, interrupt_counter));
            jc_fix(&c, s->target, 0, CC_GT);
            t_mov(&c.cb, 0, R_CTX);
            jc_call(&c, (void *)jit_poll);
            t_cmpi_enc(&c.cb, 0, 0);
            jc_fix(&c, s->target, 0, CC_EQ);
            /* interrupted: the exception exit, after the jump */
            t_ldst(&c.cb, LS_STR, R_SPV, R_F, F_SP);
            t_movi(&c.cb, 0, (uint32_t)(uintptr_t)(b->byte_code_buf + s->pc +
                                       short_opcode_info(b->byte_code_buf[s->pc]).size));
            t_ldst(&c.cb, LS_STR, 0, R_F, F_PC);
            jc_fix(&c, c.exc_exit, 2, CC_AL);
            break;
        }
        /* new stubs made while emitting stubs (none expected: jc_jump of
           a backward target makes a POLL stub, emitted in this loop) */
    }
    if (c.fail || c.cb.oom)
        goto done;

    /* patch */
    for (i = 0; i < c.nfix; i++) {
        JitFix *fx = &c.fix[i];
        uint32_t to;
        if (fx->kind == 0)
            to = c.natoff[fx->target];
        else if (fx->kind == 1)
            to = stub_off[fx->target];
        else
            to = fx->target;
        if (fx->cond == CC_AL)
            t_b_at(&c.cb, fx->pos, (int32_t)(to - fx->pos), false);
        else
            t_bcc_at(&c.cb, fx->pos, fx->cond, (int32_t)(to - fx->pos));
    }

    /* the entry points: the start and every jump's target */
    for (n = 0, i = 0; i < len; i++)
        n += c.is_target[i];
    jc = js_malloc(ctx, sizeof(JSJitCode) + n * sizeof(JSJitEntry));
    if (!jc)
        goto done;
    mem = jit_code_alloc(c.cb.len);
    if (!mem) {
        js_free(ctx, jc);
        jc = NULL;
        goto done;
    }
    memcpy(mem, c.cb.buf, c.cb.len);
    jit_code_sync(mem, c.cb.len);
    jc->code = mem;
    jc->size = c.cb.len;
    jc->ics = c.ics;
    c.ics = NULL;
    jc->nentries = n;
    for (n = 0, i = 0; i < len; i++) {
        if (c.is_target[i]) {
            jc->entries[n].bc = i;
            jc->entries[n].nat = c.natoff[i];
            n++;
        }
    }
    if (getenv("QJS_JIT_DUMP")) {
        char buf[64], path[128];
        FILE *df;
        snprintf(path, sizeof(path), "%s/%s.bin", getenv("QJS_JIT_DUMP"),
                 JS_AtomGetStr(ctx, buf, sizeof(buf), b->func_name));
        df = fopen(path, "wb");
        if (df) {
            fwrite(mem, 1, jc->size, df);
            fclose(df);
        }
    }
    if (qjs_jit_log) {
        char buf[64];
        fprintf(stderr, "qjs jit: %s: %d bytes of bytecode, %u of code, %u entries\n",
                JS_AtomGetStr(ctx, buf, sizeof(buf), b->func_name),
                b->byte_code_len, jc->size, jc->nentries);
    }
 done:
    free(stub_off);
    free(c.natoff);
    free(c.is_target);
    free(c.fix);
    free(c.stub);
    free(c.cb.buf);
    js_free(ctx, c.ics);
    return jc;
}

/* Compile b if it may be: 0, or -1 (it stays interpreted) */
static int qjs_jit_compile(JSContext *ctx, JSFunctionBytecode *b)
{
    if (b->func_kind != JS_FUNC_NORMAL) {
        b->jit_state = JIT_STATE_FAILED;
        return -1;
    }
    b->jit = jc_compile(ctx, b);
    qjs_jit_nfunc += b->jit != NULL;
    b->jit_state = b->jit ? JIT_STATE_DONE : JIT_STATE_FAILED;
    return b->jit ? 0 : -1;
}

/* Whether b should be compiled now: counted at each call (calls) or
   backward jump */
static bool qjs_jit_want(JSFunctionBytecode *b, int weight)
{
    if (unlikely(qjs_jit_threshold < 0))
        qjs_jit_config();
    if (qjs_jit_threshold == 0)
        return false;
    b->jit_hits += weight;
    return b->jit_hits >= (uint32_t)qjs_jit_threshold * 16;
}

/* The compiled code's entry for bytecode at pc, or 0 */
static uintptr_t qjs_jit_entry(JSFunctionBytecode *b, const uint8_t *pc)
{
    JSJitCode *jc = b->jit;
    uint32_t bc = pc - b->byte_code_buf;
    uint32_t lo = 0, hi = jc->nentries;
    while (lo < hi) {
        uint32_t mid = (lo + hi) / 2;
        if (jc->entries[mid].bc < bc)
            lo = mid + 1;
        else
            hi = mid;
    }
    if (lo < jc->nentries && jc->entries[lo].bc == bc)
        return (uintptr_t)jc->code + jc->entries[lo].nat + 1;
    return 0;
}
