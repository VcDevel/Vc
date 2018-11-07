#ifndef _GLIBCXX_EXPERIMENTAL_SIMD_DEBUG_H_
#define _GLIBCXX_EXPERIMENTAL_SIMD_DEBUG_H_

#pragma GCC system_header

#if defined _GLIBCXX_SIMD_DEBUG && !defined _GLIBCXX_SIMD_ENABLE_DEBUG
#define _GLIBCXX_SIMD_ENABLE_DEBUG 1
#endif

#ifdef _GLIBCXX_SIMD_ENABLE_DEBUG
#include <iostream>
#include <iomanip>
#include <sstream>
#endif  // _GLIBCXX_SIMD_ENABLE_DEBUG

_GLIBCXX_SIMD_BEGIN_NAMESPACE
enum class __area : unsigned {
    __disabled = 0,
    __enabled = 1,
    __ = __enabled,

#ifdef _GLIBCXX_SIMD_DEBUG

#define _Sine       0x0000000000000001ull
#define _Cosine     0x0000000000000002ull
#define _SIMD_TUPLE 0x0000000000000004ull
#define _Simd_view  0x0000000000000008ull
#define _Logarithm  0x0000000000000010ull
#define _Frexp      0x0000000000000020ull

    __Sine       = ((_GLIBCXX_SIMD_DEBUG) &       _Sine) ? __enabled : __disabled,
    __Cosine     = ((_GLIBCXX_SIMD_DEBUG) &     _Cosine) ? __enabled : __disabled,
    __SIMD_TUPLE = ((_GLIBCXX_SIMD_DEBUG) & _SIMD_TUPLE) ? __enabled : __disabled,
    __Simd_view  = ((_GLIBCXX_SIMD_DEBUG) & _Simd_view ) ? __enabled : __disabled,
    __Logarithm  = ((_GLIBCXX_SIMD_DEBUG) & _Logarithm ) ? __enabled : __disabled,
    __Frexp      = ((_GLIBCXX_SIMD_DEBUG) &     _Frexp ) ? __enabled : __disabled,
#undef _Sine
#undef _Cosine
#undef _SIMD_TUPLE
#undef simd_view
#undef _Logarithm
#undef _Frexp

#undef _GLIBCXX_SIMD_DEBUG

#else // _GLIBCXX_SIMD_DEBUG
    __Sine = __disabled,
    __Cosine = __disabled,
    __SIMD_TUPLE = __disabled,
    __Simd_view  = __disabled,
    __Logarithm  = __disabled,
    __Frexp = __disabled,
#endif // _GLIBCXX_SIMD_DEBUG
};

#define _GLIBCXX_SIMD_DEBUG(_Area)                                                       \
    std::experimental::__debug_stream<std::experimental::__area::_##_Area>(              \
        __PRETTY_FUNCTION__, __FILE__, __LINE__, std::experimental::__debug_instr_ptr())

#ifdef _GLIBCXX_SIMD_ENABLE_DEBUG
#define _GLIBCXX_SIMD_PRETTY_PRINT(var_) std::setw(16), #var_ " = ", (var_)

#define _GLIBCXX_SIMD_CONCAT_IMPL(a_, b_, c_) a_##b_##c_
#define _GLIBCXX_SIMD_CONCAT(a_, b_, c_) _GLIBCXX_SIMD_CONCAT_IMPL(a_, b_, c_)

#define _GLIBCXX_SIMD_DEBUG_DEFERRED(_Area, ...)                                         \
    const auto &_GLIBCXX_SIMD_CONCAT(_GLIBCXX_SIMD_deferred_, __LINE__, _) =             \
        __defer([&]() { _GLIBCXX_SIMD_DEBUG(_Area)                                       \
                        (__VA_ARGS__); });
#else   // _GLIBCXX_SIMD_ENABLE_DEBUG
#define _GLIBCXX_SIMD_PRETTY_PRINT(var_) (var_)

#define _GLIBCXX_SIMD_DEBUG_DEFERRED(_Area, ...)
#endif  // _GLIBCXX_SIMD_ENABLE_DEBUG

_GLIBCXX_SIMD_ALWAYS_INLINE void *__debug_instr_ptr()
{
    void *ip = nullptr;
#if defined _GLIBCXX_SIMD_ENABLE_DEBUG
#ifdef __x86_64__
    asm volatile("lea 0(%%rip),%0" : "=r"(ip));
#elif defined __i386__
    asm volatile("1: movl $1b,%0" : "=r"(ip));
#elif defined __arm__
    asm volatile("mov %0,pc" : "=r"(ip));
#endif
#endif  //__GNUC__
    return ip;
}

template <__area> class __debug_stream;

#ifdef _GLIBCXX_SIMD_ENABLE_DEBUG
template <> class __debug_stream<__area::__enabled>
{
    std::stringstream __buffer;
    int __color = 31;

public:
    __debug_stream(const char *__func, const char *__file, int __line, void *__instr_ptr)
    {
        __buffer << "\033[1;40;" << __color << "mDEBUG: " << __file << ':' << __line
                 << " @ " << __instr_ptr << "\n       " << __func;
    }

    ~__debug_stream()
    {
        __buffer << "\033[0m\n";
        std::cout << __buffer.str() << std::flush;
    }

    template <class... _Ts> __debug_stream &operator()(const _Ts &... args)
    {
        __color = __color > 37 ? 30 : __color + 1;
        __buffer << "\n\033[1;40;" << __color << "m       ";
        //__buffer << "\n        ";
#if 0 // __cpp_fold_expressions
        __buffer << ... << std::forward<_Ts>(args);
#else
        [](const std::initializer_list<int> &) {}({(__print(args, int()), 0)...});
#endif
        return *this;
    }

private:
    template <class T, class = decltype(__buffer << std::declval<const T &>())>
    void __print(const T &x, int)
    {
        __buffer << x;
    }

    static char hexChar(char x) { return x + (x > 9 ? 87 : 48); }
    template <class T> void __print(const T &x, float)
    {
        using Bytes = char[sizeof(T)];
        auto &&bytes = reinterpret_cast<const Bytes &>(x);
        int __i = -1;
        for (const unsigned char b : bytes) {
            if (++__i && (__i & 0x3) == 0) {
                __buffer.put('\'');
            }
            __buffer.put(hexChar(b >> 4));
            __buffer.put(hexChar(b & 0xf));
        }
    }
};
#endif  // _GLIBCXX_SIMD_ENABLE_DEBUGGING

template <> class __debug_stream<__area::__disabled>
{
public:
    __debug_stream(const char *, const char *, int, void *) {}
    template <class... _Ts> const __debug_stream &operator()(_Ts &&...) const { return *this; }
};

template <class _F> class __defer_raii
{
public:
    // construct the object from the given callable
    template <class _FF> __defer_raii(_FF &&f) : __cleanup_function(std::forward<_FF>(f))
    {
    }

    // when the object goes out of scope call the cleanup function
    ~__defer_raii() { __cleanup_function(); }

private:
    _F __cleanup_function;
};

template <typename _F> __defer_raii<_F> __defer(_F &&f) { return {std::forward<_F>(f)}; }

_GLIBCXX_SIMD_END_NAMESPACE

#endif  // _GLIBCXX_EXPERIMENTAL_SIMD_DEBUG_H_
